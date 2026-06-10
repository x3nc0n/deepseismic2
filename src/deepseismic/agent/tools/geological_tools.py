"""Geological context tools for the DeepSeismic Analyst agent.

Exposes four tools covering well metadata, formation tops, cross-well correlation,
and regional geological context:

- ``get_well_data``         — retrieve well headers, trajectory, and log availability
- ``get_formation_tops``    — look up formation top picks from well control
- ``correlate_wells``       — compare formation tops across multiple wells
- ``get_regional_context``  — retrieve regional geological and tectonic context

Each function has:
* A docstring used verbatim as the tool description shown to the LLM.
* A ``MOCK_LLM=true`` path returning Volve reference data for offline iteration.
* A live path that calls the FastAPI backend via ``httpx``.

``GEOLOGICAL_TOOL_DEFINITIONS`` holds JSON-schema definitions for Foundry
registration; ``GEOLOGICAL_TOOL_HANDLERS`` maps tool names to callables.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MOCK_MODE: bool = os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")
BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000")
_HTTP_TIMEOUT: float = 15.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{BACKEND_URL}{path}"
    try:
        resp = httpx.get(url, params=params, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.RequestError as exc:
        logger.warning("Backend unreachable at %s: %s", url, exc)
        return {"error": str(exc), "available": False}
    except httpx.HTTPStatusError as exc:
        return {"error": f"HTTP {exc.response.status_code}", "available": False}


# ---------------------------------------------------------------------------
# Mock payloads — Volve field reference data
# ---------------------------------------------------------------------------

_MOCK_WELLS: list[dict[str, Any]] = [
    {
        "id": "15/9-F-1B",
        "name": "15/9-F-1 B",
        "type": "producer",
        "status": "abandoned",
        "spud_date": "1998-12-01",
        "td_m_tvdss": 3850,
        "surface_latitude": 58.4372,
        "surface_longitude": 1.9003,
        "log_curves": ["GR", "RHOB", "NPHI", "RT", "DTCO"],
        "linked_survey": "volve-survey-a",
    },
    {
        "id": "15/9-F-4",
        "name": "15/9-F-4",
        "type": "producer",
        "status": "abandoned",
        "spud_date": "1999-03-15",
        "td_m_tvdss": 3831,
        "surface_latitude": 58.4351,
        "surface_longitude": 1.8994,
        "log_curves": ["GR", "RHOB", "NPHI", "RT"],
        "linked_survey": "volve-survey-a",
    },
    {
        "id": "15/9-F-11",
        "name": "15/9-F-11",
        "type": "injector",
        "status": "abandoned",
        "spud_date": "2000-05-10",
        "td_m_tvdss": 3740,
        "surface_latitude": 58.4339,
        "surface_longitude": 1.9021,
        "log_curves": ["GR", "RHOB", "NPHI"],
        "linked_survey": "volve-survey-a",
    },
    {
        "id": "15/9-F-15D",
        "name": "15/9-F-15 D",
        "type": "producer",
        "status": "abandoned",
        "spud_date": "2001-02-20",
        "td_m_tvdss": 3892,
        "surface_latitude": 58.4391,
        "surface_longitude": 1.9012,
        "log_curves": ["GR", "RHOB", "NPHI", "RT", "DTCO", "DTSM"],
        "linked_survey": "volve-survey-a",
    },
]

# Formation tops for well 15/9-F-1 B (representative of the Volve field)
_MOCK_FORMATION_TOPS_1B: list[dict[str, Any]] = [
    {
        "formation": "Utsira",
        "top_m_tvdss": 820,
        "top_m_md": 835,
        "lithology": "sandstone",
        "age": "Neogene",
    },
    {
        "formation": "Shetland",
        "top_m_tvdss": 1420,
        "top_m_md": 1450,
        "lithology": "chalk / shale",
        "age": "Cretaceous",
    },
    {
        "formation": "Draupne",
        "top_m_tvdss": 3395,
        "top_m_md": 3445,
        "lithology": "shale (seal)",
        "age": "Late Jurassic",
    },
    {
        "formation": "Hugin",
        "top_m_tvdss": 3512,
        "top_m_md": 3564,
        "lithology": "sandstone (reservoir)",
        "age": "Late Jurassic",
    },
    {
        "formation": "Skagerrak",
        "top_m_tvdss": 3680,
        "top_m_md": 3740,
        "lithology": "interbedded sandstone / shale",
        "age": "Triassic",
    },
    {
        "formation": "Smith Bank",
        "top_m_tvdss": 3800,
        "top_m_md": 3852,
        "lithology": "shale",
        "age": "Triassic",
    },
]

# Hugin Fm correlation table across Volve wells
_MOCK_HUGIN_CORRELATION: dict[str, Any] = {
    "formation": "Hugin",
    "wells": [
        {
            "well_id": "15/9-F-1B",
            "well_name": "15/9-F-1 B",
            "top_m_tvdss": 3512,
            "status": "confirmed",
        },
        {
            "well_id": "15/9-F-4",
            "well_name": "15/9-F-4",
            "top_m_tvdss": 3498,
            "status": "confirmed",
        },
        {
            "well_id": "15/9-F-11",
            "well_name": "15/9-F-11",
            "top_m_tvdss": 3471,
            "status": "confirmed",
        },
        {
            "well_id": "15/9-F-15D",
            "well_name": "15/9-F-15 D",
            "top_m_tvdss": 3535,
            "status": "confirmed",
        },
    ],
    "depth_range_m_tvdss": [3471, 3535],
    "depth_variation_m": 64,
    "structural_trend": (
        "Depth increases eastward; shallowest at 15/9-F-11 (injector, west flank). "
        "Consistent with a westward-dipping anticlinal closure over the basement high."
    ),
    "seismic_reflector_tie": (
        "Estimated two-way time at Hugin top: ~3 490–3 510 ms TWT across the mapped "
        "Volve area. Tie is based on checkshot velocities from 15/9-F-1 B and 15/9-F-4."
    ),
    "note": "Mock response — cross-well correlation from reference well tops.",
}

_MOCK_REGIONAL_CONTEXT: dict[str, Any] = {
    "field": "Volve",
    "block": "15/9",
    "basin": "Central Graben, Norwegian North Sea",
    "country": "Norway",
    "operator": "Equinor (formerly Statoil)",
    "production_period": "2008–2016",
    "coordinates_center": {"latitude": 58.44, "longitude": 1.90},
    "tectonic_setting": (
        "Graben-related half-graben on the eastern margin of the Utsira High. "
        "Basement faults trend NNW–SSE. Late Jurassic extensional tectonics created "
        "the rotated fault-block trap hosting the Hugin Fm reservoir."
    ),
    "primary_reservoir": {
        "name": "Hugin Formation",
        "age": "Late Jurassic (Oxfordian)",
        "lithology": "Fine- to medium-grained marine / fluvio-deltaic sandstone",
        "porosity_range_pct": [18, 28],
        "permeability_range_md": [10, 500],
        "net_pay_range_m": [10, 40],
    },
    "primary_seal": {
        "name": "Draupne Formation",
        "age": "Late Jurassic (Kimmeridgian)",
        "lithology": "Organic-rich marine shale",
        "note": (
            "Regionally extensive seal across the Central Graben. "
            "Also the main Jurassic source rock for North Sea petroleum systems."
        ),
    },
    "structural_trap": (
        "Faulted anticline over a basement high; four-way dip closure with the main "
        "bounding fault on the western flank."
    ),
    "discovered": 1994,
    "recoverable_reserves_mm_boe": 186,
    "data_license": (
        "Equinor Volve Data Village — open access research and development dataset."
    ),
    "note": "Mock response — Volve field public reference context.",
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_well_data(
    well_id: str | None = None,
    survey_id: str | None = None,
    well_type: str | None = None,
) -> dict[str, Any]:
    """Retrieve well metadata including trajectory, TD, log availability, and survey linkage.

    Returns well names, types (producer / injector / observer), total depth in TVDSS,
    surface coordinates, available log curve mnemonics, and linked survey identifiers.
    Use this to understand what well control exists near the seismic survey area before
    tying reflectors to formation boundaries.

    Args:
        well_id: Optional well identifier or partial name to filter by.
        survey_id: Optional survey ID to return only wells linked to that survey.
        well_type: Optional type filter — one of ``"producer"``, ``"injector"``,
                   or ``"observer"``.

    Returns:
        dict with ``wells`` list and ``count``. Each entry contains ``id``, ``name``,
        ``type``, ``status``, ``td_m_tvdss``, ``surface_latitude``,
        ``surface_longitude``, ``log_curves``, and ``linked_survey``.
    """
    if MOCK_MODE:
        wells = list(_MOCK_WELLS)
        if well_id:
            wells = [
                w
                for w in wells
                if well_id.lower() in w["id"].lower()
                or well_id.lower() in w["name"].lower()
            ]
        if survey_id:
            wells = [w for w in wells if w.get("linked_survey") == survey_id]
        if well_type:
            wells = [w for w in wells if w["type"] == well_type]
        return {
            "wells": wells,
            "count": len(wells),
            "note": "Mock response — Volve field reference wells.",
        }
    params = {"well_id": well_id, "survey_id": survey_id, "type": well_type}
    return _get("/api/wells", params=params)


def get_formation_tops(well_id: str, formation: str | None = None) -> dict[str, Any]:
    """Retrieve formation top picks for a named well.

    Returns a stratigraphic column with formation names, top depths in both TVDSS and MD,
    lithology descriptions, and geological ages. Primary reservoir and seal formations
    are flagged explicitly. Essential for tying seismic reflectors to known formation
    boundaries and for validating UNet interpretation output.

    Args:
        well_id: Well identifier (e.g., ``"15/9-F-1B"`` or ``"15/9-F-1 B"``).
        formation: Optional formation name to retrieve a single top
                   (e.g., ``"Hugin"``).

    Returns:
        dict with ``formation_tops`` list, ``primary_reservoir``, and
        ``primary_seal``. Each top contains ``formation``, ``top_m_tvdss``,
        ``top_m_md``, ``lithology``, and ``age``.
    """
    if MOCK_MODE:
        tops = list(_MOCK_FORMATION_TOPS_1B)
        if formation:
            tops = [t for t in tops if formation.lower() in t["formation"].lower()]
        return {
            "well_id": well_id,
            "well_name": "15/9-F-1 B (reference — mock mode)",
            "formation_tops": tops,
            "primary_reservoir": "Hugin",
            "primary_seal": "Draupne",
            "note": "Mock response — Volve field reference stratigraphy.",
        }
    return _get(f"/api/wells/{well_id}/formation-tops", params={"formation": formation})


def correlate_wells(well_ids: list[str], formation: str = "Hugin") -> dict[str, Any]:
    """Cross-correlate formation tops across multiple wells to map structural trends.

    Compares formation top depths (TVDSS) across the named wells for the specified
    formation, quantifies depth variation, and characterises the structural trend.
    Provides an estimated seismic reflector TWT tie to anchor interpretation to
    survey geometry.

    Args:
        well_ids: List of well identifiers to correlate
                  (e.g., ``["15/9-F-1B", "15/9-F-4"]``).
        formation: Formation to correlate across wells (default: ``"Hugin"``).

    Returns:
        dict with per-well tops, ``depth_range_m_tvdss``, ``depth_variation_m``,
        ``structural_trend`` description, and ``seismic_reflector_tie`` in TWT.
    """
    if MOCK_MODE:
        corr = dict(_MOCK_HUGIN_CORRELATION)
        corr["formation"] = formation
        if well_ids:
            corr["wells"] = [
                w for w in corr["wells"] if w["well_id"] in well_ids
            ]
        return corr

    try:
        resp = httpx.post(
            f"{BACKEND_URL}/api/wells/correlate",
            json={"well_ids": well_ids, "formation": formation},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        return {"error": str(exc)}


def get_regional_context(field_name: str = "Volve") -> dict[str, Any]:
    """Retrieve regional geological and tectonic context for the survey area.

    Returns basin setting, tectonic history, primary reservoir and seal descriptions,
    structural trap characterisation, and field production history. Use this to ground
    geological interpretations in the regional framework and to answer questions about
    the broader depositional or structural context.

    All content is reference material describing the geological setting — not model
    output. Analyst judgment is still required when applying this context to specific
    seismic observations.

    Args:
        field_name: Field or area name (default: ``"Volve"``).

    Returns:
        dict with ``basin``, ``tectonic_setting``, ``primary_reservoir``,
        ``primary_seal``, ``structural_trap``, ``production_period``, and
        ``data_license``.
    """
    if MOCK_MODE:
        return _MOCK_REGIONAL_CONTEXT
    return _get("/api/knowledge/regional-context", params={"field": field_name})


# ---------------------------------------------------------------------------
# Foundry tool definition registry
# ---------------------------------------------------------------------------

GEOLOGICAL_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_well_data",
        "description": get_well_data.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "well_id": {
                    "type": "string",
                    "description": "Well identifier or partial name to filter by.",
                },
                "survey_id": {
                    "type": "string",
                    "description": "Survey ID to return only linked wells.",
                },
                "well_type": {
                    "type": "string",
                    "enum": ["producer", "injector", "observer"],
                    "description": "Optional well type filter.",
                },
            },
        },
    },
    {
        "name": "get_formation_tops",
        "description": get_formation_tops.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "well_id": {
                    "type": "string",
                    "description": "Well identifier.",
                },
                "formation": {
                    "type": "string",
                    "description": "Optional formation name to retrieve a single top.",
                },
            },
            "required": ["well_id"],
        },
    },
    {
        "name": "correlate_wells",
        "description": correlate_wells.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "well_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Well identifiers to correlate.",
                },
                "formation": {
                    "type": "string",
                    "description": "Formation to correlate (default: 'Hugin').",
                },
            },
            "required": ["well_ids"],
        },
    },
    {
        "name": "get_regional_context",
        "description": get_regional_context.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "description": "Field or area name (default: 'Volve').",
                },
            },
        },
    },
]

GEOLOGICAL_TOOL_HANDLERS: dict[str, Any] = {
    "get_well_data": get_well_data,
    "get_formation_tops": get_formation_tops,
    "correlate_wells": correlate_wells,
    "get_regional_context": get_regional_context,
}
