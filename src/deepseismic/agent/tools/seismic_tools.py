"""Seismic survey tools for the DeepSeismic Analyst agent.

Exposes four tools covering survey discovery, inline inspection, fault detection
job submission, and interpretation status polling:

- ``query_survey_metadata``    — discover loaded surveys, formats, and coverage
- ``get_inline_section``       — retrieve amplitude metadata for a seismic inline
- ``run_fault_detection``      — submit a UNet fault-detection inference job
- ``get_interpretation_status``— poll an interpretation or preprocessing run

Each function has:
* A docstring used verbatim as the tool description shown to the LLM.
* A ``MOCK_LLM=true`` path that returns canned data for offline iteration.
* A live path that calls the FastAPI backend via ``httpx``.

``SEISMIC_TOOL_DEFINITIONS`` holds JSON-schema tool definitions for Foundry
registration; ``SEISMIC_TOOL_HANDLERS`` maps tool names to callables for
local dispatch in the agent loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from deepseismic.agent.tools._api_client import APIError
from deepseismic.agent.tools._api_client import get as _api_get
from deepseismic.agent.tools._api_client import get_list as _api_get_list
from deepseismic.agent.tools._api_client import post as _api_post

logger = logging.getLogger(__name__)

MOCK_MODE: bool = os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------

_MOCK_SURVEY_METADATA: dict[str, Any] = {
    "surveys": [
        {
            "id": "volve-survey-a",
            "name": "Volve 3D Survey A (PoC subset)",
            "formats": ["segy", "zarr"],
            "inline_range": [1000, 1200],
            "crossline_range": [950, 1100],
            "sample_interval_ms": 4,
            "depth_twt_range_ms": [0, 4000],
            "status": "loaded",
            "manifest_path": "az://deepseismic-raw/volve/volve-survey-a/manifest.json",
        }
    ],
    "count": 1,
    "note": "Mock response — no live data queried.",
}

_MOCK_INLINE_SECTION: dict[str, Any] = {
    "survey_id": "volve-survey-a",
    "inline": 1050,
    "crossline_range": [950, 1100],
    "sample_interval_ms": 4,
    "num_samples": 1000,
    "amplitude_stats": {
        "min": -18432,
        "max": 21504,
        "mean": 127,
        "std": 4812,
    },
    "anomaly_zones": [
        {
            "crossline_start": 980,
            "crossline_end": 1040,
            "twt_ms_start": 3480,
            "twt_ms_end": 3560,
            "note": (
                "Elevated amplitude; possible fluid contact effect "
                "at Hugin Fm level (~3 510 ms TWT)."
            ),
        }
    ],
    "zarr_chunk_path": (
        "az://deepseismic-processed/volve/volve-survey-a/zarr/il_1050.zarr"
    ),
    "note": "Mock response — amplitude statistics are illustrative.",
}

_MOCK_FAULT_DETECTION: dict[str, Any] = {
    "job_id": "fault-job-volve-01",
    "survey_id": "volve-survey-a",
    "status": "queued",
    "model": "unet-fault-v1",
    "estimated_minutes": 8,
    "note": (
        "Mock response — fault detection job queued. "
        "Use get_interpretation_status to poll job completion."
    ),
}

_MOCK_INTERP_STATUS: dict[str, Any] = {
    "run_id": "run-volve-unet-01",
    "survey_id": "volve-survey-a",
    "type": "inference",
    "model": "unet-baseline-v1",
    "status": "completed",
    "started_at": "2026-06-09T18:00:00Z",
    "completed_at": "2026-06-09T18:14:22Z",
    "duration_minutes": 14.4,
    "qc_artifacts": {
        "slices_generated": 12,
        "overlay_path": "az://deepseismic-results/volve/run-volve-unet-01/qc/",
    },
    "result_id": "res-volve-unet-01",
    "caveats": [
        "This describes model output, not confirmed geological truth.",
        "Analyst review is required before sign-off.",
    ],
    "note": "Mock response — status reflects an illustrative completed run.",
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def query_survey_metadata(
    survey_name: str | None = None,
    format: str | None = None,  # noqa: A002
) -> dict[str, Any]:
    """Search loaded seismic surveys and return metadata.

    Returns survey names, formats (SEG-Y, Zarr), inline and crossline ranges,
    sample intervals, and storage paths. Use this first to confirm which datasets
    are available before requesting analysis or QC steps.

    Args:
        survey_name: Optional partial name to filter surveys (case-insensitive).
        format: Optional format filter — one of ``"segy"``, ``"zarr"``, or ``"any"``.

    Returns:
        dict with ``surveys`` list and ``count``. Each survey entry contains ``id``,
        ``name``, ``formats``, ``inline_range``, ``crossline_range``, ``status``,
        and ``manifest_path``.
    """
    if MOCK_MODE:
        surveys = _MOCK_SURVEY_METADATA["surveys"]
        if survey_name:
            surveys = [
                s for s in surveys if survey_name.lower() in s["name"].lower()
            ]
        if format and format != "any":
            surveys = [s for s in surveys if format in s["formats"]]
        return {**_MOCK_SURVEY_METADATA, "surveys": surveys, "count": len(surveys)}
    try:
        items = _api_get_list("/api/surveys")
        if survey_name:
            items = [
                s for s in items
                if survey_name.lower() in (
                    s.get("survey_id", "") + s.get("source_file", "")
                ).lower()
            ]
        return {"surveys": items, "count": len(items)}
    except APIError as exc:
        return {"error": str(exc), "available": False}


def get_inline_section(survey_id: str, inline_number: int) -> dict[str, Any]:
    """Retrieve metadata and amplitude statistics for a single seismic inline.

    Returns the inline number, crossline range, sample interval, amplitude
    statistics (min, max, mean, std), and any anomaly zones flagged by the
    preprocessing pipeline. Use this to inspect specific seismic sections during
    QC or interpretation. Does not return the raw array — only summary metadata.

    Args:
        survey_id: Dataset identifier (e.g., ``"volve-survey-a"``).
        inline_number: Inline number to retrieve.

    Returns:
        dict with ``inline``, ``amplitude_stats``, ``anomaly_zones``, and
        ``zarr_chunk_path`` reference for downstream array access.
    """
    if MOCK_MODE:
        return {**_MOCK_INLINE_SECTION, "inline": inline_number, "survey_id": survey_id}
    try:
        return _api_get(f"/api/surveys/{survey_id}/inline/{inline_number}")
    except APIError as exc:
        return {"error": str(exc), "available": False}


def run_fault_detection(
    survey_id: str,
    model_version: str = "unet-fault-v1",
    inline_range: list[int] | None = None,
) -> dict[str, Any]:
    """Submit a fault-detection inference job for a seismic survey.

    Queues a UNet-based fault detection pass on the specified survey and optional
    inline range. Returns a job ID for polling with ``get_interpretation_status``.

    Fault detection is a deterministic model pass — results describe model output,
    not confirmed geological faults. Analyst review is required before interpretation.

    Args:
        survey_id: Dataset identifier to run fault detection on.
        model_version: Model version tag (default: ``"unet-fault-v1"``).
        inline_range: Optional ``[start, end]`` inline range to limit scope.

    Returns:
        dict with ``job_id``, ``status`` (``"queued"``), ``model``, and
        ``estimated_minutes``. Pass ``job_id`` to ``get_interpretation_status``
        to poll completion.
    """
    if MOCK_MODE:
        return {**_MOCK_FAULT_DETECTION, "survey_id": survey_id, "model": model_version}

    payload: dict[str, Any] = {"survey_id": survey_id}
    # Map model_version to a checkpoint blob path when non-default
    if model_version and model_version != "unet-fault-v1":
        payload["checkpoint_blob"] = f"checkpoints/{model_version}.pt"
    if inline_range:
        payload["inline_range"] = inline_range
    try:
        return _api_post("/api/interpretation/fault-detection", payload)
    except APIError as exc:
        return {"error": str(exc)}


def get_interpretation_status(run_id: str) -> dict[str, Any]:
    """Retrieve the current status of an interpretation or preprocessing run.

    Returns run state (``queued`` | ``running`` | ``completed`` | ``failed``),
    timing, QC artifact paths, result ID, and any caveats. Use this to confirm
    a run completed successfully before requesting result summaries or handoff
    notes.

    Args:
        run_id: Run identifier returned by a prior submission or listed via
                ``query_survey_metadata`` / ``searchDatasets``.

    Returns:
        dict with ``status``, ``started_at``, ``completed_at``,
        ``duration_minutes``, ``qc_artifacts``, ``result_id``, and ``caveats``.
    """
    if MOCK_MODE:
        return {**_MOCK_INTERP_STATUS, "run_id": run_id}
    try:
        return _api_get(f"/api/interpretation/{run_id}/status")
    except APIError as exc:
        return {"error": str(exc), "available": False}


# ---------------------------------------------------------------------------
# Foundry tool definition registry
# ---------------------------------------------------------------------------

SEISMIC_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_survey_metadata",
        "description": query_survey_metadata.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "survey_name": {
                    "type": "string",
                    "description": "Optional partial survey name to filter by.",
                },
                "format": {
                    "type": "string",
                    "enum": ["segy", "zarr", "any"],
                    "description": "Optional format filter.",
                },
            },
        },
    },
    {
        "name": "get_inline_section",
        "description": get_inline_section.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "survey_id": {
                    "type": "string",
                    "description": "Dataset identifier (e.g. 'volve-survey-a').",
                },
                "inline_number": {
                    "type": "integer",
                    "description": "Inline number to retrieve.",
                },
            },
            "required": ["survey_id", "inline_number"],
        },
    },
    {
        "name": "run_fault_detection",
        "description": run_fault_detection.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "survey_id": {
                    "type": "string",
                    "description": "Dataset identifier.",
                },
                "model_version": {
                    "type": "string",
                    "description": "Model version tag (default: 'unet-fault-v1').",
                },
                "inline_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional [start, end] inline range.",
                },
            },
            "required": ["survey_id"],
        },
    },
    {
        "name": "get_interpretation_status",
        "description": get_interpretation_status.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run identifier to poll.",
                },
            },
            "required": ["run_id"],
        },
    },
]

SEISMIC_TOOL_HANDLERS: dict[str, Any] = {
    "query_survey_metadata": query_survey_metadata,
    "get_inline_section": get_inline_section,
    "run_fault_detection": run_fault_detection,
    "get_interpretation_status": get_interpretation_status,
}
