"""Browse ADLS storage — project/file picker for the UI.

Lists containers and blob prefixes so users can navigate the data lake
structure and select different surveys or projects to work with.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from deepseismic.api.dependencies import StorageClientDep, is_mock_mode
from deepseismic.storage.blob_client import CONTAINERS

router = APIRouter(prefix="/browse", tags=["browse"])


class BrowseItem(BaseModel):
    """A navigable item in the data lake."""

    name: str
    path: str
    type: str  # "folder" or "file"
    size: int | None = None


class BrowseResponse(BaseModel):
    """Response for a browse request."""

    container: str
    prefix: str
    items: list[BrowseItem]


# Mock data for demo mode
_MOCK_TREE: dict[str, dict[str, list[BrowseItem]]] = {
    "raw": {
        "": [BrowseItem(name="volve", path="volve/", type="folder")],
        "volve/": [
            BrowseItem(name="seismic", path="volve/seismic/", type="folder"),
            BrowseItem(
                name="interpretations", path="volve/interpretations/", type="folder"
            ),
            BrowseItem(name="wells", path="volve/wells/", type="folder"),
        ],
        "volve/seismic/": [
            BrowseItem(
                name="sample_volume.segy",
                path="volve/seismic/sample_volume.segy",
                type="file",
                size=44_800_000,
            ),
        ],
        "volve/interpretations/": [
            BrowseItem(
                name="fault_sticks_hugin.csv",
                path="volve/interpretations/fault_sticks_hugin.csv",
                type="file",
                size=245_000,
            ),
            BrowseItem(
                name="fault_sticks_alpha.csv",
                path="volve/interpretations/fault_sticks_alpha.csv",
                type="file",
                size=189_000,
            ),
            BrowseItem(
                name="fault_sticks_beta.csv",
                path="volve/interpretations/fault_sticks_beta.csv",
                type="file",
                size=156_000,
            ),
        ],
    },
    "staged": {
        "": [BrowseItem(name="volve", path="volve/", type="folder")],
        "volve/": [BrowseItem(name="zarr", path="volve/zarr/", type="folder")],
        "volve/zarr/": [
            BrowseItem(
                name="seismic_volume.zarr",
                path="volve/zarr/seismic_volume.zarr/",
                type="folder",
                size=44_800_000,
            ),
            BrowseItem(
                name="fault_prob.zarr",
                path="volve/zarr/fault_prob.zarr/",
                type="folder",
                size=22_400_000,
            ),
        ],
    },
    "results": {
        "": [BrowseItem(name="volve", path="volve/", type="folder")],
        "volve/": [
            BrowseItem(
                name="fault_detection_run_001.json",
                path="volve/fault_detection_run_001.json",
                type="file",
                size=4_200,
            ),
        ],
    },
}


@router.get("/containers", response_model=list[str])
def list_containers():
    """Return the list of standard ADLS containers."""
    return list(CONTAINERS)


@router.get("/{container}", response_model=BrowseResponse)
def browse_container(
    container: str,
    prefix: str = Query("", description="Blob prefix to browse (use trailing / for folders)"),
    storage: StorageClientDep = None,
):
    """Browse a container's contents at a given prefix level.

    Returns folders (common prefixes) and files at the specified level.
    Uses virtual folder hierarchy — no actual directory objects needed.
    """
    if container not in CONTAINERS:
        raise HTTPException(404, f"Container '{container}' not recognized. Valid: {CONTAINERS}")

    if is_mock_mode():
        mock_container = _MOCK_TREE.get(container, {})
        items = mock_container.get(prefix, [])
        return BrowseResponse(container=container, prefix=prefix, items=items)

    # Real storage: hierarchical listing via delimiter — returns only the
    # immediate level (virtual folders + files), NOT every blob in the
    # container.  A flat list_blobs over a large container (e.g. the multi-TB
    # `raw` lake) iterates hundreds of thousands of blobs and times out, which
    # is what made the UI browser appear empty.
    try:
        cc = storage._container(container)
        items: list[BrowseItem] = []

        for blob in cc.walk_blobs(name_starts_with=prefix or None, delimiter="/"):
            name = blob.name
            relative = name[len(prefix):] if prefix else name
            if name.endswith("/"):
                # Virtual folder (BlobPrefix) — name ends with the delimiter.
                folder_name = relative.rstrip("/").split("/")[-1]
                items.append(BrowseItem(
                    name=folder_name,
                    path=name,
                    type="folder",
                ))
            else:
                items.append(BrowseItem(
                    name=relative,
                    path=name,
                    type="file",
                    size=getattr(blob, "size", None),
                ))

        return BrowseResponse(container=container, prefix=prefix, items=items)

    except Exception as exc:
        raise HTTPException(500, f"Storage browse failed: {exc}") from exc
