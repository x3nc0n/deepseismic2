"""Survey catalog and inline/crossline slice routes.

Endpoints
---------
GET  /api/surveys                           list available surveys
GET  /api/surveys/{survey_id}               survey metadata (JSON sidecar)
POST /api/surveys/ingest                    queue SEG-Y → Zarr ingest job
GET  /api/surveys/{survey_id}/inline/{n}    inline section amplitude data
GET  /api/surveys/{survey_id}/crossline/{n} crossline section amplitude data

Storage layout (catalog container)
-----------------------------------
    surveys/{survey_id}/metadata.json    — IngestMetadata JSON sidecar
    (amplitude Zarr lives in staged/surveys/{survey_id}/amplitude.zarr)
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
    CrosslineSlice,
    IngestRequest,
    IngestResponse,
    InlineSlice,
    SurveyListItem,
    SurveyMetadata,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/surveys", tags=["surveys"])

# In-memory ingest job registry — single-process PoC only
_ingest_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Mock data — Volve ST10010 geometry
# ---------------------------------------------------------------------------

_MOCK_SURVEY_ID = "volve-st10010"

_MOCK_GEOMETRY: dict[str, Any] = {
    "inline_min": 9985,
    "inline_max": 10369,
    "inline_step": 1,
    "crossline_min": 2064,
    "crossline_max": 2536,
    "crossline_step": 1,
    "sample_rate_ms": 4.0,
    "n_samples": 1001,
    "n_inlines": 385,
    "n_crosslines": 473,
    "datum_ms": 0.0,
}

_MOCK_STATS: dict[str, float] = {
    "min": -4821.3,
    "max": 4956.1,
    "mean": -0.021,
    "std": 412.7,
    "p01": -1240.5,
    "p99": 1238.9,
    "nonzero_fraction": 0.9943,
}

_MOCK_INGESTED_AT = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def _mock_survey_list() -> list[SurveyListItem]:
    return [
        SurveyListItem(
            survey_id=_MOCK_SURVEY_ID,
            source_file="ST10010_PSDM_TIME.segy",
            ingested_at=_MOCK_INGESTED_AT,
            n_inlines=_MOCK_GEOMETRY["n_inlines"],
            n_crosslines=_MOCK_GEOMETRY["n_crosslines"],
        )
    ]


def _mock_survey_metadata(survey_id: str) -> SurveyMetadata:
    if survey_id != _MOCK_SURVEY_ID:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found")
    return SurveyMetadata(
        survey_id=_MOCK_SURVEY_ID,
        source_file="ST10010_PSDM_TIME.segy",
        ingested_at=_MOCK_INGESTED_AT,
        sample_mode=True,
        n_inlines_loaded=50,
        geometry=_MOCK_GEOMETRY,
        amplitude_stats=_MOCK_STATS,
        zarr_path="staged/surveys/volve-st10010/amplitude.zarr",
    )


def _mock_inline_slice(survey_id: str, number: int) -> InlineSlice:
    """Synthetic inline section with seismic-like layered texture."""
    rng = np.random.default_rng(seed=number)
    n_xl, n_s = 50, 100  # compact for API performance

    xl_start = _MOCK_GEOMETRY["crossline_min"]
    xl_step = max(1, _MOCK_GEOMETRY["n_crosslines"] // n_xl)
    crossline_coords = list(range(xl_start, xl_start + n_xl * xl_step, xl_step))[:n_xl]
    twtt_ms = [float(i * 4.0) for i in range(n_s)]

    depth = np.linspace(0, 1, n_s)
    template = np.sin(depth * 20 * np.pi) * 500 + np.sin(depth * 7 * np.pi) * 200
    amplitude = [
        (template + rng.standard_normal(n_s) * 80 + np.sin(xl * 0.3) * 100).tolist()
        for xl in range(n_xl)
    ]

    return InlineSlice(
        survey_id=survey_id,
        inline_number=number,
        crossline_coords=crossline_coords,
        twtt_ms=twtt_ms,
        amplitude=amplitude,
    )


def _mock_crossline_slice(survey_id: str, number: int) -> CrosslineSlice:
    """Synthetic crossline section with seismic-like layered texture."""
    rng = np.random.default_rng(seed=number + 100_000)
    n_il, n_s = 50, 100

    il_start = _MOCK_GEOMETRY["inline_min"]
    il_step = max(1, _MOCK_GEOMETRY["n_inlines"] // n_il)
    inline_coords = list(range(il_start, il_start + n_il * il_step, il_step))[:n_il]
    twtt_ms = [float(i * 4.0) for i in range(n_s)]

    depth = np.linspace(0, 1, n_s)
    template = np.sin(depth * 20 * np.pi) * 500 + np.sin(depth * 7 * np.pi) * 200
    amplitude = [
        (template + rng.standard_normal(n_s) * 80 + np.cos(il * 0.3) * 100).tolist()
        for il in range(n_il)
    ]

    return CrosslineSlice(
        survey_id=survey_id,
        crossline_number=number,
        inline_coords=inline_coords,
        twtt_ms=twtt_ms,
        amplitude=amplitude,
    )


# ---------------------------------------------------------------------------
# Background ingest task
# ---------------------------------------------------------------------------


def _run_ingest(run_id: str, req: IngestRequest, storage: Any) -> None:
    """Download SEG-Y from blob, convert to Zarr, upload results to catalog."""
    _ingest_jobs[run_id]["status"] = "running"
    try:
        from deepseismic.ingest.segy_loader import SEGYLoader

        raw_bytes = storage.download_blob("raw", req.blob_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / f"{req.survey_id}.zarr"
            with SEGYLoader(
                raw_bytes,
                sample_mode=req.sample_mode,
                sample_n_inlines=req.sample_n_inlines,
            ) as ldr:
                _, meta = ldr.to_zarr(zarr_path, overwrite=True, survey_id=req.survey_id)

            # Upload Zarr amplitude store to staged container
            local_store = zarr.storage.LocalStore(str(zarr_path))
            storage.upload_zarr_store(
                local_store,
                "staged",
                f"surveys/{req.survey_id}/amplitude.zarr",
            )

            # Write metadata sidecar to catalog container
            storage.upload_blob(
                "catalog",
                f"surveys/{req.survey_id}/metadata.json",
                meta.to_json().encode(),
            )

        _ingest_jobs[run_id]["status"] = "complete"
        logger.info("Ingest complete: run_id=%s survey=%s", run_id, req.survey_id)

    except Exception as exc:
        _ingest_jobs[run_id]["status"] = "failed"
        _ingest_jobs[run_id]["error"] = str(exc)
        logger.exception("Ingest failed: run_id=%s", run_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=list[SurveyListItem])
def list_surveys(storage: StorageClientDep) -> list[SurveyListItem]:
    """List all ingested surveys in the catalog."""
    if is_mock_mode():
        return _mock_survey_list()

    try:
        blobs = storage.list_blobs("catalog", prefix="surveys/")
    except Exception as exc:
        logger.error("Could not list surveys from storage: %s", exc)
        raise HTTPException(status_code=503, detail=f"Storage unavailable: {exc}") from exc

    seen: set[str] = set()
    items: list[SurveyListItem] = []

    for blob_name in blobs:
        # Expected path: surveys/{survey_id}/metadata.json
        parts = blob_name.split("/")
        if len(parts) < 3 or parts[-1] != "metadata.json":
            continue
        survey_id = parts[1]
        if survey_id in seen:
            continue
        seen.add(survey_id)

        try:
            raw = storage.download_blob("catalog", blob_name)
            meta = json.loads(raw)
            geom = meta.get("geometry", {})
            items.append(
                SurveyListItem(
                    survey_id=survey_id,
                    source_file=meta.get("source_file", ""),
                    ingested_at=meta.get("ingested_at", datetime.now(UTC).isoformat()),
                    n_inlines=geom.get("n_inlines", 0),
                    n_crosslines=geom.get("n_crosslines", 0),
                )
            )
        except Exception as exc:
            logger.warning("Could not parse survey metadata for %s: %s", survey_id, exc)

    return items


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_survey(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    storage: StorageClientDep,
) -> IngestResponse:
    """Queue a SEG-Y → Zarr ingest job.  Returns immediately with a run_id."""
    run_id = str(uuid.uuid4())
    _ingest_jobs[run_id] = {
        "run_id": run_id,
        "survey_id": req.survey_id,
        "status": "pending",
        "error": None,
    }

    if is_mock_mode():
        _ingest_jobs[run_id]["status"] = "complete"
        logger.info("Mock ingest for survey=%s run_id=%s", req.survey_id, run_id)
    else:
        background_tasks.add_task(_run_ingest, run_id, req, storage)
        logger.info("Ingest queued: run_id=%s survey=%s", run_id, req.survey_id)

    return IngestResponse(
        run_id=run_id,
        survey_id=req.survey_id,
        status=_ingest_jobs[run_id]["status"],
    )


@router.get("/{survey_id}", response_model=SurveyMetadata)
def get_survey(survey_id: str, storage: StorageClientDep) -> SurveyMetadata:
    """Return the JSON sidecar metadata for a survey."""
    if is_mock_mode():
        return _mock_survey_metadata(survey_id)

    blob_path = f"surveys/{survey_id}/metadata.json"
    try:
        raw = storage.download_blob("catalog", blob_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found") from None

    try:
        meta = json.loads(raw)
        geom = meta.get("geometry", {})
        stats = meta.get("amplitude_stats", {})
        return SurveyMetadata(
            survey_id=survey_id,
            source_file=meta.get("source_file", ""),
            ingested_at=meta.get("ingested_at", datetime.now(UTC).isoformat()),
            sample_mode=meta.get("sample_mode", False),
            n_inlines_loaded=meta.get("n_inlines_loaded", geom.get("n_inlines", 0)),
            geometry=geom,
            amplitude_stats=stats,
            zarr_path=meta.get("zarr_path", ""),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Malformed metadata: {exc}") from exc


@router.get("/{survey_id}/inline/{number}", response_model=InlineSlice)
def get_inline(survey_id: str, number: int, storage: StorageClientDep) -> InlineSlice:
    """Return an inline section as amplitude arrays (crossline × time)."""
    if is_mock_mode():
        return _mock_inline_slice(survey_id, number)

    try:
        store = storage.open_zarr_store("staged", f"surveys/{survey_id}/amplitude.zarr")
        root = zarr.open_group(store, mode="r")
        amp_arr: zarr.Array = root["amplitude"]
        il_arr: zarr.Array = root["inline"]
        xl_arr: zarr.Array = root["crossline"]
        tt_arr: zarr.Array = root["twtt_ms"]

        inlines = il_arr[:]
        il_indices = np.where(inlines == number)[0]
        if len(il_indices) == 0:
            raise HTTPException(status_code=404, detail=f"Inline {number} not in survey")

        il_idx = int(il_indices[0])
        slice_data: np.ndarray = np.asarray(amp_arr[il_idx, :, :])  # (n_xl, n_s)

        return InlineSlice(
            survey_id=survey_id,
            inline_number=number,
            crossline_coords=xl_arr[:].tolist(),
            twtt_ms=tt_arr[:].tolist(),
            amplitude=slice_data.tolist(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to read inline %d for survey %s", number, survey_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{survey_id}/crossline/{number}", response_model=CrosslineSlice)
def get_crossline(
    survey_id: str, number: int, storage: StorageClientDep
) -> CrosslineSlice:
    """Return a crossline section as amplitude arrays (inline × time)."""
    if is_mock_mode():
        return _mock_crossline_slice(survey_id, number)

    try:
        store = storage.open_zarr_store("staged", f"surveys/{survey_id}/amplitude.zarr")
        root = zarr.open_group(store, mode="r")
        amp_arr: zarr.Array = root["amplitude"]
        il_arr: zarr.Array = root["inline"]
        xl_arr: zarr.Array = root["crossline"]
        tt_arr: zarr.Array = root["twtt_ms"]

        crosslines = xl_arr[:]
        xl_indices = np.where(crosslines == number)[0]
        if len(xl_indices) == 0:
            raise HTTPException(status_code=404, detail=f"Crossline {number} not in survey")

        xl_idx = int(xl_indices[0])
        slice_data: np.ndarray = np.asarray(amp_arr[:, xl_idx, :])  # (n_il, n_s)

        return CrosslineSlice(
            survey_id=survey_id,
            crossline_number=number,
            inline_coords=il_arr[:].tolist(),
            twtt_ms=tt_arr[:].tolist(),
            amplitude=slice_data.tolist(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to read crossline %d for survey %s", number, survey_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
