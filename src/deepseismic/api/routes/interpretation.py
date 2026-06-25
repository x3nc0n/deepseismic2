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

            # Local paths for output volumes
            prob_path = tmp / f"{run_id}_prob.zarr"
            mask_path = tmp / f"{run_id}_mask.zarr"

            engine = VolumeInference.from_checkpoint(
                ckpt_path,
                patch_size=req.patch_size,
                overlap=req.overlap,
                batch_size=req.batch_size,
                threshold=req.threshold,
            )
            prob_vol, mask_vol = engine.run(
                seismic,
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

        # Write run manifest to catalog
        manifest = {
            "run_id": run_id,
            "survey_id": req.survey_id,
            "status": "complete",
            "prob_zarr_path": f"results/interpretation/{run_id}/fault_prob.zarr",
            "mask_zarr_path": f"results/interpretation/{run_id}/fault_mask.zarr",
            "fault_voxel_fraction": fault_fraction,
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

    if is_mock_mode():
        return _mock_status(run_id)

    # Attempt to load from catalog if the job survived a restart
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
    """Return fault probability and binary mask for a single inline section."""
    if is_mock_mode():
        return _mock_overlay(run_id, inline_number)

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

        # Resolve inline index — assume inline axis matches the source survey
        if inline_number < 0 or inline_number >= prob_arr.shape[0]:
            raise HTTPException(
                status_code=404,
                detail=f"Inline index {inline_number} out of bounds",
            )

        prob_slice = np.asarray(prob_arr[inline_number, :, :]).tolist()  # (n_xl, n_s)
        mask_slice = np.asarray(mask_arr[inline_number, :, :]).tolist()

        n_xl, n_s = prob_arr.shape[1], prob_arr.shape[2]
        crossline_coords = list(range(n_xl))
        twtt_ms = [float(i * 4.0) for i in range(n_s)]

        return FaultOverlay(
            run_id=run_id,
            inline_number=inline_number,
            crossline_coords=crossline_coords,
            twtt_ms=twtt_ms,
            fault_probability=prob_slice,
            fault_mask=mask_slice,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to read overlay for run %s inline %d", run_id, inline_number)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
