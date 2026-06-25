"""Well catalog routes.

Endpoints
---------
GET  /api/wells                   list wells in catalog
GET  /api/wells/{well_id}         well metadata (formation tops, log inventory)
GET  /api/wells/{well_id}/logs    log curves as JSON arrays

Storage layout (catalog container)
-----------------------------------
    wells/{well_id}/metadata.json   — WellMetadata JSON
    wells/{well_id}/logs.json       — WellLog JSON
"""

from __future__ import annotations

import json
import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from deepseismic.api.dependencies import StorageClientDep, is_mock_mode
from deepseismic.api.schemas import (
    FormationTop,
    WellListItem,
    WellLog,
    WellLogCurve,
    WellMetadata,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wells", tags=["wells"])


# ---------------------------------------------------------------------------
# Mock data — Volve field wells
# ---------------------------------------------------------------------------

_MOCK_WELLS: list[dict] = [
    {
        "well_id": "15-9-F-11B",
        "name": "15/9-F-11B",
        "uwi": "NO 15/9-F-11 B",
        "latitude": 58.4346,
        "longitude": 1.8745,
        "kb_elevation_m": 27.5,
        "total_depth_m": 3878.0,
        "spud_date": "2004-08-12",
        "formation_tops": [
            {"formation": "Utsira", "depth_m": 843.0, "tvdss_m": 815.5},
            {"formation": "Skade", "depth_m": 1025.0, "tvdss_m": 997.5},
            {"formation": "Frigg", "depth_m": 1212.0, "tvdss_m": 1184.5},
            {"formation": "Sleipner", "depth_m": 1430.0, "tvdss_m": 1402.5},
            {"formation": "Hugin", "depth_m": 2290.0, "tvdss_m": 2262.5},
            {"formation": "Ness", "depth_m": 2620.0, "tvdss_m": 2592.5},
            {"formation": "Rannoch", "depth_m": 2750.0, "tvdss_m": 2722.5},
            {"formation": "Etive", "depth_m": 2810.0, "tvdss_m": 2782.5},
            {"formation": "Oseberg", "depth_m": 2920.0, "tvdss_m": 2892.5},
            {"formation": "Dunlin", "depth_m": 3150.0, "tvdss_m": 3122.5},
            {"formation": "Drake", "depth_m": 3280.0, "tvdss_m": 3252.5},
        ],
        "logs_available": ["GR", "DT", "RHOB", "NPHI", "RT"],
    },
    {
        "well_id": "15-9-F-1C",
        "name": "15/9-F-1C",
        "uwi": "NO 15/9-F-1 C",
        "latitude": 58.4291,
        "longitude": 1.8831,
        "kb_elevation_m": 27.5,
        "total_depth_m": 3762.0,
        "spud_date": "2005-06-03",
        "formation_tops": [
            {"formation": "Utsira", "depth_m": 839.0, "tvdss_m": 811.5},
            {"formation": "Hugin", "depth_m": 2285.0, "tvdss_m": 2257.5},
            {"formation": "Ness", "depth_m": 2615.0, "tvdss_m": 2587.5},
            {"formation": "Rannoch", "depth_m": 2744.0, "tvdss_m": 2716.5},
        ],
        "logs_available": ["GR", "DT", "RHOB"],
    },
]

_MOCK_WELLS_BY_ID = {w["well_id"]: w for w in _MOCK_WELLS}


def _mock_well_list() -> list[WellListItem]:
    return [
        WellListItem(
            well_id=w["well_id"],
            name=w["name"],
            latitude=w["latitude"],
            longitude=w["longitude"],
            kb_elevation_m=w["kb_elevation_m"],
            total_depth_m=w["total_depth_m"],
        )
        for w in _MOCK_WELLS
    ]


def _mock_well_metadata(well_id: str) -> WellMetadata:
    w = _MOCK_WELLS_BY_ID.get(well_id)
    if w is None:
        raise HTTPException(status_code=404, detail=f"Well '{well_id}' not found")
    return WellMetadata(
        well_id=w["well_id"],
        name=w["name"],
        uwi=w.get("uwi"),
        latitude=w.get("latitude"),
        longitude=w.get("longitude"),
        kb_elevation_m=w.get("kb_elevation_m"),
        total_depth_m=w.get("total_depth_m"),
        spud_date=w.get("spud_date"),
        formation_tops=[FormationTop(**t) for t in w.get("formation_tops", [])],
        logs_available=w.get("logs_available", []),
    )


def _mock_well_logs(well_id: str) -> WellLog:
    """Synthetic log curves for the requested well (Volve-like character)."""
    w = _MOCK_WELLS_BY_ID.get(well_id)
    if w is None:
        raise HTTPException(status_code=404, detail=f"Well '{well_id}' not found")

    kb = float(w.get("kb_elevation_m") or 27.5)
    td = float(w.get("total_depth_m") or 3800.0)
    rng = np.random.default_rng(seed=hash(well_id) % (2**32))

    # Depth samples every 0.1524 m (0.5 ft) from 800 m to TD
    depth = np.arange(800.0, td + 0.1524, 0.1524)
    n = len(depth)

    # Gamma ray: shale baseline ~90 API, sand beds ~30 API
    gr_base = 90.0 - 60.0 * (depth > 2250) * (depth < 2900)
    gr = np.clip(gr_base + rng.standard_normal(n) * 12.0, 0, 200)

    # Sonic: shale ~90 μs/ft, reservoir ~60 μs/ft
    dt_base = 90.0 - 30.0 * (depth > 2250) * (depth < 2900)
    dt = np.clip(dt_base + rng.standard_normal(n) * 4.0, 40, 140)

    # Bulk density: shale ~2.4 g/cc, reservoir ~2.2 g/cc
    rhob_base = 2.42 - 0.25 * (depth > 2250) * (depth < 2900)
    rhob = np.clip(rhob_base + rng.standard_normal(n) * 0.03, 1.7, 2.85)

    # Replace values above KB with None to mimic casing intervals
    kb_mask = depth < (kb + 50)
    gr_out: list[float | None] = [None if kb_mask[i] else float(gr[i]) for i in range(n)]
    dt_out: list[float | None] = [None if kb_mask[i] else float(dt[i]) for i in range(n)]
    rhob_out: list[float | None] = [None if kb_mask[i] else float(rhob[i]) for i in range(n)]

    return WellLog(
        well_id=well_id,
        depth_unit="m",
        depth_values=depth.tolist(),
        curves=[
            WellLogCurve(
                mnemonic="GR",
                unit="API",
                description="Gamma Ray",
                values=gr_out,
            ),
            WellLogCurve(
                mnemonic="DT",
                unit="us/ft",
                description="Sonic Travel Time",
                values=dt_out,
            ),
            WellLogCurve(
                mnemonic="RHOB",
                unit="g/cc",
                description="Bulk Density",
                values=rhob_out,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=list[WellListItem])
def list_wells(storage: StorageClientDep) -> list[WellListItem]:
    """List all wells in the catalog."""
    if is_mock_mode():
        return _mock_well_list()

    try:
        blobs = storage.list_blobs("catalog", prefix="wells/")
    except Exception as exc:
        logger.error("Could not list wells from storage: %s", exc)
        raise HTTPException(status_code=503, detail=f"Storage unavailable: {exc}") from exc

    seen: set[str] = set()
    items: list[WellListItem] = []

    for blob_name in blobs:
        # Expected path: wells/{well_id}/metadata.json
        parts = blob_name.split("/")
        if len(parts) < 3 or parts[-1] != "metadata.json":
            continue
        well_id = parts[1]
        if well_id in seen:
            continue
        seen.add(well_id)

        try:
            raw = storage.download_blob("catalog", blob_name)
            meta = json.loads(raw)
            items.append(
                WellListItem(
                    well_id=well_id,
                    name=meta.get("name", well_id),
                    latitude=meta.get("latitude"),
                    longitude=meta.get("longitude"),
                    kb_elevation_m=meta.get("kb_elevation_m"),
                    total_depth_m=meta.get("total_depth_m"),
                )
            )
        except Exception as exc:
            logger.warning("Could not parse well metadata for %s: %s", well_id, exc)

    return items


@router.get("/{well_id}", response_model=WellMetadata)
def get_well(well_id: str, storage: StorageClientDep) -> WellMetadata:
    """Return well metadata including formation tops and available log inventory."""
    if is_mock_mode():
        return _mock_well_metadata(well_id)

    blob_path = f"wells/{well_id}/metadata.json"
    try:
        raw = storage.download_blob("catalog", blob_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Well '{well_id}' not found") from None

    try:
        meta = json.loads(raw)
        return WellMetadata(
            well_id=well_id,
            name=meta.get("name", well_id),
            uwi=meta.get("uwi"),
            latitude=meta.get("latitude"),
            longitude=meta.get("longitude"),
            kb_elevation_m=meta.get("kb_elevation_m"),
            total_depth_m=meta.get("total_depth_m"),
            spud_date=meta.get("spud_date"),
            formation_tops=[FormationTop(**t) for t in meta.get("formation_tops", [])],
            logs_available=meta.get("logs_available", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Malformed well metadata: {exc}") from exc


@router.get("/{well_id}/logs", response_model=WellLog)
def get_well_logs(well_id: str, storage: StorageClientDep) -> WellLog:
    """Return all log curves for a well as JSON arrays (depth + named curves)."""
    if is_mock_mode():
        return _mock_well_logs(well_id)

    blob_path = f"wells/{well_id}/logs.json"
    try:
        raw = storage.download_blob("catalog", blob_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Logs for well '{well_id}' not found"
        ) from None

    try:
        data = json.loads(raw)
        return WellLog(
            well_id=well_id,
            depth_unit=data.get("depth_unit", "m"),
            depth_values=data.get("depth_values", []),
            curves=[WellLogCurve(**c) for c in data.get("curves", [])],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Malformed log data: {exc}") from exc
