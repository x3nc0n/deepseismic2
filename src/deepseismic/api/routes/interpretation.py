"""Fault detection interpretation routes.

Endpoints
---------
POST /api/interpretation/fault-detection          queue a UNet inference run
GET  /api/interpretation/{run_id}/status          job status polling
GET  /api/interpretation/{run_id}/results         completed run results + metadata
GET  /api/interpretation/{run_id}/overlay/{il}    fault probability overlay for an inline

Storage layout
--------------
    results/interpretation/{run_id}/fault_prob.zarr   — probability volume
    results/interpretation/{run_id}/fault_mask.zarr   — binary mask
    catalog/interpretation/{run_id}/status.json       — run manifest
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from deepseismic.api.dependencies import StorageClientDep, is_mock_mode
from deepseismic.api.schemas import (
    FaultOverlay,
    InterpretationRequest,
    InterpretationResult,
    InterpretationStatus,
    JobStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interpretation", tags=["interpretation"])

# In-memory run registry — single-process PoC only
_interp_jobs: dict[str, dict[str, Any]] = {}


def _resolve_run_id(run_id: str, storage: Any) -> str:
    """Resolve a possibly-abbreviated ``run_id`` to a full run identifier.

    The UI displays runs by their 8-char prefix (``run_id[:8]``), so users and
    the chat agent only ever see/quote the prefix.  Looking that prefix up
    against the full-UUID registry/catalog would 404.  This resolves a prefix
    (or full id) to the single matching full run id by checking, in order:

    1. an exact in-memory job key,
    2. an exact persisted ``catalog`` manifest,
    3. a unique prefix match across in-memory jobs **and** catalog manifests.

    Returns the input unchanged if it already resolves exactly.  Raises 404 if
    nothing matches and 409 if a prefix is ambiguous (>1 match).
    """
    # 1. Exact in-memory hit.
    if run_id in _interp_jobs:
        return run_id

    # 2. Exact persisted manifest.
    try:
        storage.download_blob("catalog", f"interpretation/{run_id}/status.json")
        return run_id
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 — fall through to prefix search
        pass

    # 3. Unique prefix match across in-memory jobs + catalog manifests.
    matches: set[str] = {rid for rid in _interp_jobs if rid.startswith(run_id)}
    try:
        for name in storage.list_blobs("catalog", "interpretation/"):
            # name == "interpretation/{full_run_id}/status.json"
            parts = name.split("/")
            if len(parts) >= 2 and parts[1].startswith(run_id):
                matches.add(parts[1])
    except Exception:  # noqa: BLE001 — catalog listing is best-effort
        pass

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run id '{run_id}' is ambiguous ({len(matches)} matches). "
                "Provide more characters of the run id."
            ),
        )
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_RUN_ID = "mock-run-00000000"
_MOCK_SURVEY_ID = "volve-st10010"
_MOCK_COMPLETED_AT = datetime(2026, 6, 1, 14, 30, 0, tzinfo=UTC)


def _mock_status(run_id: str) -> InterpretationStatus:
    if run_id == _MOCK_RUN_ID or run_id in _interp_jobs:
        return InterpretationStatus(
            run_id=run_id,
            survey_id=_interp_jobs.get(run_id, {}).get("survey_id", _MOCK_SURVEY_ID),
            status=JobStatus.complete,
            created_at=_MOCK_COMPLETED_AT,
            updated_at=_MOCK_COMPLETED_AT,
            progress_pct=100.0,
            message="Fault detection complete (mock)",
        )
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


def _mock_result(run_id: str) -> InterpretationResult:
    if run_id != _MOCK_RUN_ID and run_id not in _interp_jobs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return InterpretationResult(
        run_id=run_id,
        survey_id=_interp_jobs.get(run_id, {}).get("survey_id", _MOCK_SURVEY_ID),
        status=JobStatus.complete,
        prob_zarr_path=f"results/interpretation/{run_id}/fault_prob.zarr",
        mask_zarr_path=f"results/interpretation/{run_id}/fault_mask.zarr",
        fault_voxel_fraction=0.0412,
        completed_at=_MOCK_COMPLETED_AT,
        download_url=None,
    )


def _mock_overlay(run_id: str, inline_number: int) -> FaultOverlay:
    rng = np.random.default_rng(seed=inline_number + 999)
    n_xl, n_s = 50, 100
    xl_start = 2064
    crossline_coords = list(range(xl_start, xl_start + n_xl))
    twtt_ms = [float(i * 4.0) for i in range(n_s)]

    # Low-probability background with isolated fault zones
    fault_prob = []
    fault_mask = []
    for xl in range(n_xl):
        fault_zone = float(np.sin(xl * 0.2) > 0.7)
        probs = np.clip(rng.standard_normal(n_s) * 0.05 + fault_zone * 0.6, 0, 1)
        fault_prob.append(probs.tolist())
        fault_mask.append((probs >= 0.5).astype(int).tolist())

    return FaultOverlay(
        run_id=run_id,
        inline_number=inline_number,
        crossline_coords=crossline_coords,
        twtt_ms=twtt_ms,
        fault_probability=fault_prob,
        fault_mask=fault_mask,
    )


# ---------------------------------------------------------------------------
# Background inference task
# ---------------------------------------------------------------------------


def _run_fault_detection(run_id: str, req: InterpretationRequest, storage: Any) -> None:
    """Download checkpoint + seismic Zarr, run UNet inference, upload results."""
    job = _interp_jobs[run_id]
    job["status"] = "running"
    job["updated_at"] = datetime.now(UTC).isoformat()

    try:
        from deepseismic.models.inference import VolumeInference

        # Download model checkpoint
        ckpt_bytes = storage.download_blob("features", req.checkpoint_blob)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt_path = tmp / "checkpoint.pt"
            ckpt_path.write_bytes(ckpt_bytes)

            # Open seismic Zarr directly from blob storage
            store = storage.open_zarr_store(
                "staged", f"surveys/{req.survey_id}/amplitude.zarr"
            )
            root = zarr.open_group(store, mode="r")
            seismic: zarr.Array = root["amplitude"]
            inline_arr = np.asarray(root["inline"][:])
            crossline_arr = np.asarray(root["crossline"][:])
            twtt_arr = np.asarray(root["twtt_ms"][:])

            # Clamp the patch to the volume so small surveys degrade
            # gracefully instead of erroring (#21): a survey with fewer
            # inlines than patch_size[0] (e.g. 50 < 64) cannot host a 64-thick
            # patch.  Clamping per-axis keeps inference valid on any size.
            patch = tuple(
                int(min(p, d)) for p, d in zip(req.patch_size, seismic.shape, strict=False)
            )
            if patch != tuple(req.patch_size):
                logger.info(
                    "Clamped patch_size %s -> %s to fit volume shape %s (run %s)",
                    tuple(req.patch_size), patch, tuple(seismic.shape), run_id,
                )

            # Bound to a subvolume around the requested inline (issue #19):
            # the full ST10010 cube needs >8 GiB of accumulators, but the
            # viewer only renders one inline, so a +/-window slab is enough
            # and fits the baseline web container.
            il0, il1 = 0, seismic.shape[0]
            if req.inline_center is not None:
                center_idx = int(
                    np.clip(
                        np.searchsorted(inline_arr, req.inline_center),
                        0, len(inline_arr) - 1,
                    )
                )
                min_thick = patch[0]
                half = max(int(req.inline_window), min_thick // 2 + 1)
                il0 = max(0, center_idx - half)
                il1 = min(seismic.shape[0], center_idx + half + 1)
                # Guarantee the slab is at least one patch thick.
                if il1 - il0 < min_thick:
                    il0 = max(0, min(il0, seismic.shape[0] - min_thick))
                    il1 = min(seismic.shape[0], il0 + min_thick)
                seismic_input: np.ndarray = np.asarray(
                    seismic[il0:il1, :, :], dtype=np.float32
                )
            else:
                seismic_input = seismic  # full-cube zarr (lazy)

            sub_inline = inline_arr[il0:il1]

            # Local paths for output volumes
            prob_path = tmp / f"{run_id}_prob.zarr"
            mask_path = tmp / f"{run_id}_mask.zarr"

            engine = VolumeInference.from_checkpoint(
                ckpt_path,
                patch_size=patch,
                overlap=req.overlap,
                batch_size=req.batch_size,
                threshold=req.threshold,
            )
            prob_vol, mask_vol = engine.run(
                seismic_input,
                prob_output=prob_path,
                mask_output=mask_path,
                overwrite=True,
            )

            # Upload results to blob storage
            storage.upload_zarr_store(
                zarr.storage.LocalStore(str(prob_path)),
                "results",
                f"interpretation/{run_id}/fault_prob.zarr",
            )
            storage.upload_zarr_store(
                zarr.storage.LocalStore(str(mask_path)),
                "results",
                f"interpretation/{run_id}/fault_mask.zarr",
            )

            fault_fraction = float(mask_vol.sum()) / max(mask_vol.size, 1)

        # Write run manifest to catalog — store the inline/crossline/twtt
        # coordinates of the (sub)volume so the overlay endpoint can map an
        # absolute inline to the correct local index.
        manifest = {
            "run_id": run_id,
            "survey_id": req.survey_id,
            "status": "complete",
            "prob_zarr_path": f"results/interpretation/{run_id}/fault_prob.zarr",
            "mask_zarr_path": f"results/interpretation/{run_id}/fault_mask.zarr",
            "fault_voxel_fraction": fault_fraction,
            "inline_coords": [int(i) for i in sub_inline],
            "crossline_coords": [int(x) for x in crossline_arr],
            "twtt_ms": [float(t) for t in twtt_arr],
            "completed_at": datetime.now(UTC).isoformat(),
        }
        storage.upload_blob(
            "catalog",
            f"interpretation/{run_id}/status.json",
            json.dumps(manifest).encode(),
        )

        job["status"] = "complete"
        job["fault_voxel_fraction"] = fault_fraction
        job["updated_at"] = datetime.now(UTC).isoformat()
        logger.info("Fault detection complete: run_id=%s fraction=%.4f", run_id, fault_fraction)

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["updated_at"] = datetime.now(UTC).isoformat()
        logger.exception("Fault detection failed: run_id=%s", run_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/fault-detection",
    response_model=InterpretationStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_fault_detection(
    req: InterpretationRequest,
    background_tasks: BackgroundTasks,
    storage: StorageClientDep,
) -> InterpretationStatus:
    """Queue a UNet fault detection run on a survey.  Returns immediately."""
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    _interp_jobs[run_id] = {
        "run_id": run_id,
        "survey_id": req.survey_id,
        "status": "pending",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "error": None,
        "fault_voxel_fraction": None,
    }

    if is_mock_mode():
        _interp_jobs[run_id]["status"] = "complete"
        _interp_jobs[run_id]["fault_voxel_fraction"] = 0.0412
        logger.info("Mock fault detection for survey=%s run_id=%s", req.survey_id, run_id)
    else:
        background_tasks.add_task(_run_fault_detection, run_id, req, storage)
        logger.info("Fault detection queued: run_id=%s survey=%s", run_id, req.survey_id)

    return InterpretationStatus(
        run_id=run_id,
        survey_id=req.survey_id,
        status=JobStatus(_interp_jobs[run_id]["status"]),
        created_at=now,
        updated_at=now,
        message="Fault detection job queued",
    )


@router.get("/{run_id}/status", response_model=InterpretationStatus)
def get_status(run_id: str, storage: StorageClientDep) -> InterpretationStatus:
    """Poll the status of a fault detection run."""
    if is_mock_mode():
        return _mock_status(run_id)

    run_id = _resolve_run_id(run_id, storage)

    if run_id in _interp_jobs:
        job = _interp_jobs[run_id]
        return InterpretationStatus(
            run_id=run_id,
            survey_id=job["survey_id"],
            status=JobStatus(job["status"]),
            created_at=datetime.fromisoformat(job["created_at"]),
            updated_at=datetime.fromisoformat(job["updated_at"]),
            error=job.get("error"),
        )

    # Load from catalog if the job survived a restart (resolve guarantees it exists)
    try:
        raw = storage.download_blob("catalog", f"interpretation/{run_id}/status.json")
        manifest = json.loads(raw)
        return InterpretationStatus(
            run_id=run_id,
            survey_id=manifest.get("survey_id", ""),
            status=JobStatus(manifest.get("status", "complete")),
            created_at=manifest.get("completed_at", datetime.now(UTC).isoformat()),
            updated_at=manifest.get("completed_at", datetime.now(UTC).isoformat()),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{run_id}/results", response_model=InterpretationResult)
def get_results(run_id: str, storage: StorageClientDep) -> InterpretationResult:
    """Return fault probability volume metadata and download URL for a completed run."""
    if is_mock_mode():
        return _mock_result(run_id)

    run_id = _resolve_run_id(run_id, storage)

    # Check in-memory first
    if run_id in _interp_jobs:
        job = _interp_jobs[run_id]
        if job["status"] != "complete":
            raise HTTPException(
                status_code=409,
                detail=f"Run '{run_id}' is not complete (status={job['status']})",
            )
        return InterpretationResult(
            run_id=run_id,
            survey_id=job["survey_id"],
            status=JobStatus.complete,
            prob_zarr_path=f"results/interpretation/{run_id}/fault_prob.zarr",
            mask_zarr_path=f"results/interpretation/{run_id}/fault_mask.zarr",
            fault_voxel_fraction=float(job.get("fault_voxel_fraction") or 0.0),
            completed_at=datetime.fromisoformat(job["updated_at"]),
        )

    try:
        raw = storage.download_blob("catalog", f"interpretation/{run_id}/status.json")
        manifest = json.loads(raw)
        if manifest.get("status") != "complete":
            raise HTTPException(
                status_code=409,
                detail=f"Run '{run_id}' is not complete",
            )
        return InterpretationResult(
            run_id=run_id,
            survey_id=manifest.get("survey_id", ""),
            status=JobStatus.complete,
            prob_zarr_path=manifest.get("prob_zarr_path", ""),
            mask_zarr_path=manifest.get("mask_zarr_path", ""),
            fault_voxel_fraction=manifest.get("fault_voxel_fraction", 0.0),
            completed_at=manifest.get("completed_at", datetime.now(UTC).isoformat()),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found") from None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{run_id}/overlay/{inline_number}", response_model=FaultOverlay)
def get_overlay(
    run_id: str, inline_number: int, storage: StorageClientDep
) -> FaultOverlay:
    """Return fault probability + binary mask for a single **absolute** inline.

    ``inline_number`` is the absolute survey inline (e.g. 9961-10361 for
    ST10010).  It is mapped to the result volume's local index via the
    coordinate arrays recorded in the run manifest, so bounded subvolume runs
    (issue #19) resolve correctly.
    """
    if is_mock_mode():
        return _mock_overlay(run_id, inline_number)

    run_id = _resolve_run_id(run_id, storage)

    # Load the manifest to recover the (sub)volume coordinate mapping.
    try:
        raw = storage.download_blob("catalog", f"interpretation/{run_id}/status.json")
        manifest = json.loads(raw)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found") from None

    inline_coords = manifest.get("inline_coords") or []
    crossline_coords = manifest.get("crossline_coords") or []
    twtt_ms = manifest.get("twtt_ms") or []

    try:
        prob_store = storage.open_zarr_store(
            "results", f"interpretation/{run_id}/fault_prob.zarr"
        )
        mask_store = storage.open_zarr_store(
            "results", f"interpretation/{run_id}/fault_mask.zarr"
        )
        prob_root = zarr.open_group(prob_store, mode="r")
        mask_root = zarr.open_group(mask_store, mode="r")

        prob_arr: zarr.Array = prob_root["fault_probability"]
        mask_arr: zarr.Array = mask_root["fault_mask"]

        # Map absolute inline -> local volume index via the manifest coords.
        if inline_coords:
            idx_matches = np.where(np.asarray(inline_coords) == inline_number)[0]
            if len(idx_matches) == 0:
                lo, hi = inline_coords[0], inline_coords[-1]
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Inline {inline_number} is outside this run's window "
                        f"[{lo}, {hi}]. Re-run fault detection centered on it."
                    ),
                )
            il_idx = int(idx_matches[0])
        else:
            # Legacy/full-cube manifest without coords — treat as positional.
            if inline_number < 0 or inline_number >= prob_arr.shape[0]:
                raise HTTPException(
                    status_code=404,
                    detail=f"Inline index {inline_number} out of bounds",
                )
            il_idx = inline_number

        prob_slice = np.asarray(prob_arr[il_idx, :, :]).tolist()  # (n_xl, n_s)
        mask_slice = np.asarray(mask_arr[il_idx, :, :]).tolist()

        n_xl, n_s = prob_arr.shape[1], prob_arr.shape[2]
        if not crossline_coords:
            crossline_coords = list(range(n_xl))
        if not twtt_ms:
            twtt_ms = [float(i * 4.0) for i in range(n_s)]

        return FaultOverlay(
            run_id=run_id,
            inline_number=inline_number,
            crossline_coords=[int(x) for x in crossline_coords],
            twtt_ms=[float(t) for t in twtt_ms],
            fault_probability=prob_slice,
            fault_mask=mask_slice,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to read overlay for run %s inline %d", run_id, inline_number)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
