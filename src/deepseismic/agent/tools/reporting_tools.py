"""Reporting tools for the DeepSeismic Analyst agent.

Exposes three tools for generating analyst deliverables from run results:

- ``generate_summary``       — produce a structured QC and results summary
- ``export_interpretation``  — package interpretation results for downstream handoff
- ``create_qc_report``       — compile a QC artifact report from run outputs

Each function has:
* A docstring used verbatim as the tool description shown to the LLM.
* A ``MOCK_LLM=true`` path returning canned report content for offline iteration.
* A live path that calls the FastAPI backend via ``httpx``.

``REPORTING_TOOL_DEFINITIONS`` holds JSON-schema definitions for Foundry
registration; ``REPORTING_TOOL_HANDLERS`` maps tool names to callables.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from deepseismic.agent.tools._api_client import APIError
from deepseismic.agent.tools._api_client import get as _api_get
from deepseismic.agent.tools._api_client import post as _api_post  # noqa: F401

logger = logging.getLogger(__name__)

MOCK_MODE: bool = os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------

_MOCK_SUMMARY: dict[str, Any] = {
    "result_id": "res-volve-unet-01",
    "run_id": "run-volve-unet-01",
    "dataset_id": "volve-survey-a",
    "model_version": "unet-baseline-v1",
    "status": "completed",
    "generated_at": "2026-06-09T18:20:00Z",
    "summary": (
        "Inference completed successfully on the Volve PoC subset. "
        "The UNet baseline model processed all 201 inlines in the target window "
        "(IL 1000–1200, XL 950–1100) with no missing traces reported."
    ),
    "key_findings": [
        "Prediction mask written to results storage at az://deepseismic-results/volve/run-volve-unet-01/",
        "QC overlays generated for 12 sampled inline slices.",
        "Candidate fault corridor identified: IL 1050–1120, XL 980–1040.",
        "Amplitude anomaly at Hugin Fm level (~3 510 ms TWT); "
        "correlates with well picks in 15/9-F-1 B and 15/9-F-4.",
        "No validation errors reported by the backend.",
    ],
    "qc_stats": {
        "slices_reviewed": 12,
        "warnings": 1,
        "warning_detail": "Low signal-to-noise on IL 1180–1200 (edge of survey).",
        "coverage_pct": 100,
    },
    "caveats": [
        "This describes model output, not confirmed geological truth.",
        "The fault corridor is a candidate — structural interpretation requires expert review.",
        "Edge inlines (IL 1180–1200) show reduced SNR; treat results there with caution.",
        "Analyst sign-off is required before this result is used in operational decisions.",
    ],
    "note": "Mock response — illustrative result summary.",
}

_MOCK_EXPORT: dict[str, Any] = {
    "export_id": "export-volve-unet-01",
    "result_id": "res-volve-unet-01",
    "format": "json",
    "package_path": (
        "az://deepseismic-results/volve/exports/export-volve-unet-01.json"
    ),
    "included_artifacts": [
        "result_summary.json",
        "qc_overlays/ (12 PNG files)",
        "prediction_mask.zarr (reference — large, not inline)",
        "analyst_note.md",
    ],
    "status": "ready",
    "expires_at": "2026-07-09T00:00:00Z",
    "note": "Mock response — export package details are illustrative.",
}

_MOCK_QC_REPORT: dict[str, Any] = {
    "report_id": "qc-report-volve-unet-01",
    "run_id": "run-volve-unet-01",
    "dataset_id": "volve-survey-a",
    "generated_at": "2026-06-09T18:18:00Z",
    "sections": [
        {
            "title": "Preprocessing QC",
            "status": "pass",
            "details": "All 201 inlines ingested without geometry errors. "
                       "Sample interval confirmed at 4 ms. No missing traces detected.",
        },
        {
            "title": "Inference QC",
            "status": "pass_with_warnings",
            "details": "UNet inference completed. 12 QC slice overlays generated. "
                       "Warning: edge inlines IL 1180–1200 show lower SNR; "
                       "model confidence is reduced in that zone.",
        },
        {
            "title": "Result Validation",
            "status": "pass",
            "details": "Prediction mask dimensions match input volume. "
                       "Hugin Fm correlation verified against well tops in "
                       "15/9-F-1 B and 15/9-F-4 (within 15 ms TWT tolerance).",
        },
        {
            "title": "Analyst Readiness",
            "status": "ready_for_review",
            "details": "QC overlays are accessible at "
                       "az://deepseismic-results/volve/run-volve-unet-01/qc/. "
                       "Analyst sign-off is required before operational use.",
        },
    ],
    "overall_status": "pass_with_warnings",
    "recommended_action": (
        "Review QC overlay slices, paying particular attention to IL 1180–1200. "
        "If edge-zone results are not required, the main interpretation window "
        "(IL 1000–1180) can proceed to analyst sign-off."
    ),
    "caveats": [
        "QC pass does not imply geological correctness.",
        "Model output requires qualified geoscientist review before use.",
    ],
    "note": "Mock response — QC report content is illustrative.",
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def generate_summary(
    result_id: str,
    include_caveats: bool = True,
) -> dict[str, Any]:
    """Generate a structured QC and results summary for a completed inference run.

    Produces a concise, analyst-readable summary of a completed UNet inference
    result, including key findings, QC statistics, and mandatory caveats. The
    summary is grounded in run metadata from the backend — it does not invent
    findings. If the result is not complete or not found, returns a descriptive
    error rather than a speculative summary.

    Args:
        result_id: Result identifier (e.g., ``"res-volve-unet-01"``).
        include_caveats: Whether to include the caveats list in the output
                         (default: ``True``). Always include for analyst handoffs.

    Returns:
        dict with ``result_id``, ``run_id``, ``dataset_id``, ``model_version``,
        ``status``, ``summary`` text, ``key_findings`` list, ``qc_stats``,
        and ``caveats`` (when ``include_caveats=True``).
    """
    if MOCK_MODE:
        result = dict(_MOCK_SUMMARY)
        result["result_id"] = result_id
        if not include_caveats:
            result.pop("caveats", None)
        return result
    # Treat result_id as run_id — GET /api/interpretation/{run_id}/results
    run_id = result_id
    try:
        results = _api_get(f"/api/interpretation/{run_id}/results")
        fault_fraction = results.get("fault_voxel_fraction", 0.0)
        key_findings: list[str] = [
            f"Fault probability volume: {results.get('prob_zarr_path', 'N/A')}",
            f"Fault mask volume: {results.get('mask_zarr_path', 'N/A')}",
            f"Fault voxel fraction: {fault_fraction:.4f}",
        ]
        if results.get("download_url"):
            key_findings.append(f"Download URL: {results['download_url']}")
        out: dict[str, Any] = {
            "result_id": result_id,
            "run_id": results.get("run_id", run_id),
            "dataset_id": results.get("survey_id"),
            "status": results.get("status"),
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": (
                f"Fault detection run {run_id} completed on survey "
                f"{results.get('survey_id', 'unknown')}. "
                f"Fault voxel fraction: {fault_fraction:.4f}. "
                f"Completed at: {results.get('completed_at', 'N/A')}."
            ),
            "key_findings": key_findings,
            "qc_stats": {
                "fault_voxel_fraction": fault_fraction,
                "prob_zarr_path": results.get("prob_zarr_path"),
                "mask_zarr_path": results.get("mask_zarr_path"),
                "completed_at": results.get("completed_at"),
            },
        }
        if include_caveats:
            out["caveats"] = [
                "This describes model output, not confirmed geological truth.",
                "Analyst review is required before sign-off.",
            ]
        return out
    except APIError as exc:
        return {"error": str(exc), "available": False}


def export_interpretation(
    result_id: str,
    export_format: str = "json",
    include_qc_overlays: bool = True,
) -> dict[str, Any]:
    """Package an interpretation result for downstream analyst handoff.

    Assembles a portable export package containing the result summary,
    QC overlays, prediction mask reference, and an analyst note stub.
    Returns a storage path and list of included artifacts.

    Use this when an analyst is ready to share findings with a subsurface team
    or archive the result. The export is read-only — it does not modify the
    source result or approve it for operational use.

    Args:
        result_id: Result identifier to export.
        export_format: Output format — ``"json"`` (default) or ``"zip"``.
        include_qc_overlays: Whether to include PNG QC overlays in the package
                              (default: ``True``).

    Returns:
        dict with ``export_id``, ``package_path`` (storage URI), ``included_artifacts``
        list, ``status``, and ``expires_at`` for the temporary download link.
    """
    if MOCK_MODE:
        result = dict(_MOCK_EXPORT)
        result["result_id"] = result_id
        result["format"] = export_format
        if not include_qc_overlays:
            result["included_artifacts"] = [
                a for a in result["included_artifacts"] if "qc" not in a.lower()
            ]
        return result
    # Aggregate status + results into an export package — no dedicated export endpoint exists yet.
    run_id = result_id
    try:
        status_data = _api_get(f"/api/interpretation/{run_id}/status")
        results = _api_get(f"/api/interpretation/{run_id}/results")
        artifacts: list[str] = [
            f"fault_prob.zarr — {results.get('prob_zarr_path', 'N/A')}",
            f"fault_mask.zarr — {results.get('mask_zarr_path', 'N/A')}",
            f"run_status.json (status={status_data.get('status', 'unknown')})",
        ]
        if include_qc_overlays:
            artifacts.append(
                f"overlay/{run_id}/{{inline}} — "
                f"GET /api/interpretation/{run_id}/overlay/{{inline}}"
            )
        if results.get("download_url"):
            artifacts.append(f"download_url: {results['download_url']}")
        return {
            "export_id": f"export-{run_id}",
            "result_id": result_id,
            "run_id": run_id,
            "format": export_format,
            "survey_id": results.get("survey_id"),
            "status": results.get("status"),
            "fault_voxel_fraction": results.get("fault_voxel_fraction"),
            "included_artifacts": artifacts,
            "package_path": results.get("prob_zarr_path", f"results/interpretation/{run_id}/"),
            "completed_at": results.get("completed_at"),
        }
    except APIError as exc:
        return {"error": str(exc), "available": False}


def create_qc_report(
    run_id: str,
    include_recommendations: bool = True,
) -> dict[str, Any]:
    """Compile a structured QC artifact report from inference run outputs.

    Generates a multi-section QC report covering preprocessing quality,
    inference output validation, result validation against well control,
    and analyst readiness status. Each section has a pass / fail /
    pass-with-warnings status and detailed notes.

    The report is grounded in backend QC metadata — it surfaces what the
    pipeline actually measured, not inferred geological meaning.

    Args:
        run_id: Run identifier (e.g., ``"run-volve-unet-01"``).
        include_recommendations: Whether to include recommended next actions
                                  (default: ``True``).

    Returns:
        dict with ``report_id``, ``sections`` list, ``overall_status``,
        ``recommended_action`` (when ``include_recommendations=True``),
        and ``caveats``.
    """
    if MOCK_MODE:
        result = dict(_MOCK_QC_REPORT)
        result["run_id"] = run_id
        result["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not include_recommendations:
            result.pop("recommended_action", None)
        return result
    try:
        status_data = _api_get(f"/api/interpretation/{run_id}/status")
        job_status = status_data.get("status", "unknown")
        is_complete = job_status == "complete"
        has_error = bool(status_data.get("error"))

        sections: list[dict[str, Any]] = [
            {
                "title": "Run Status",
                "status": "pass" if is_complete and not has_error else (
                    "fail" if has_error else "pending"
                ),
                "details": (
                    f"Run {run_id} — status: {job_status}. "
                    f"Survey: {status_data.get('survey_id', 'unknown')}. "
                    f"Progress: {status_data.get('progress_pct', 0):.0f}%."
                    + (f" Error: {status_data['error']}." if has_error else "")
                    + (f" {status_data['message']}." if status_data.get("message") else "")
                ),
            }
        ]

        if is_complete:
            try:
                results = _api_get(f"/api/interpretation/{run_id}/results")
                sections.append({
                    "title": "Inference Results",
                    "status": "pass",
                    "details": (
                        f"Fault probability volume: {results.get('prob_zarr_path', 'N/A')}. "
                        f"Fault mask volume: {results.get('mask_zarr_path', 'N/A')}. "
                        f"Fault voxel fraction: {results.get('fault_voxel_fraction', 0.0):.4f}. "
                        f"Completed at: {results.get('completed_at', 'N/A')}."
                    ),
                })
            except APIError as exc:
                sections.append({
                    "title": "Inference Results",
                    "status": "fail",
                    "details": f"Could not retrieve result metadata: {exc}",
                })

        overall = (
            "pass" if is_complete and not has_error
            else "fail" if has_error
            else "pending"
        )
        out: dict[str, Any] = {
            "report_id": f"qc-report-{run_id}",
            "run_id": run_id,
            "dataset_id": status_data.get("survey_id"),
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sections": sections,
            "overall_status": overall,
            "caveats": [
                "QC pass does not imply geological correctness.",
                "Model output requires qualified geoscientist review before use.",
            ],
        }
        if include_recommendations:
            if is_complete:
                out["recommended_action"] = (
                    "Review fault probability overlays for each inline of interest "
                    f"via GET /api/interpretation/{run_id}/overlay/{{inline}}. "
                    "Compare against well formation tops before analyst sign-off."
                )
            else:
                out["recommended_action"] = (
                    f"Wait for run to complete — current status: {job_status}."
                )
        return out
    except APIError as exc:
        return {"error": str(exc), "available": False}


# ---------------------------------------------------------------------------
# Foundry tool definition registry
# ---------------------------------------------------------------------------

REPORTING_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "generate_summary",
        "description": generate_summary.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "Result identifier (e.g. 'res-volve-unet-01').",
                },
                "include_caveats": {
                    "type": "boolean",
                    "description": "Include caveats in output (default: true).",
                    "default": True,
                },
            },
            "required": ["result_id"],
        },
    },
    {
        "name": "export_interpretation",
        "description": export_interpretation.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "Result identifier to export.",
                },
                "export_format": {
                    "type": "string",
                    "enum": ["json", "zip"],
                    "description": "Output format (default: 'json').",
                    "default": "json",
                },
                "include_qc_overlays": {
                    "type": "boolean",
                    "description": "Include PNG QC overlays in the package.",
                    "default": True,
                },
            },
            "required": ["result_id"],
        },
    },
    {
        "name": "create_qc_report",
        "description": create_qc_report.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run identifier (e.g. 'run-volve-unet-01').",
                },
                "include_recommendations": {
                    "type": "boolean",
                    "description": "Include recommended next actions.",
                    "default": True,
                },
            },
            "required": ["run_id"],
        },
    },
]

REPORTING_TOOL_HANDLERS: dict[str, Any] = {
    "generate_summary": generate_summary,
    "export_interpretation": export_interpretation,
    "create_qc_report": create_qc_report,
}
