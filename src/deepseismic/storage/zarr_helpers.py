"""Zarr store resolution helpers — local filesystem or Azure ADLS.

Used by training and evaluation scripts to open a Zarr root group from either
a local path (development) or an Azure Blob Storage container (in-VNet cloud).

Backend selection
-----------------
``storage_backend="local"``  (default)
    Opens ``zarr.open_group(str(local_path), mode="r")`` directly.

``storage_backend="azure"``
    Constructs a :class:`~deepseismic.storage.blob_client.StorageClient`,
    calls ``open_zarr_store(container, prefix)`` to get an
    :class:`~deepseismic.storage.blob_client.ABSZarrV3Store`, then opens the
    group from that store.  Requires ``STORAGE_CONNECTION_STRING`` or
    ``AZURE_STORAGE_ACCOUNT`` to be set in the environment.

ADLS path convention (infra issue #11)
---------------------------------------
``staged/surveys/{survey_id}/amplitude.zarr``
``staged/surveys/{survey_id}/fault_label.zarr``

Example::

    from deepseismic.storage.zarr_helpers import open_zarr_root

    # Local dev (default)
    root = open_zarr_root("data/volve/staged/synthetic.zarr")

    # Azure (in-VNet job)
    root = open_zarr_root(
        None,
        backend="azure",
        az_container="staged",
        az_prefix="surveys/volve-st10010/amplitude.zarr",
    )
    amplitude = root["amplitude"]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import zarr

logger = logging.getLogger(__name__)


def open_zarr_root(
    local_path: str | Path | None,
    *,
    backend: str = "local",
    az_container: str = "",
    az_prefix: str = "",
) -> zarr.Group:
    """Open a Zarr group from either a local store or Azure Blob Storage.

    Parameters
    ----------
    local_path:
        Local filesystem path to the Zarr store directory.
        Used when ``backend="local"``.  Ignored when ``backend="azure"``.
    backend:
        ``"local"`` (default) or ``"azure"``.
    az_container:
        Azure Blob container name (e.g. ``"staged"``).
        Required when ``backend="azure"``.
    az_prefix:
        Blob key prefix for the Zarr store
        (e.g. ``"surveys/volve-st10010/amplitude.zarr"``).
        Required when ``backend="azure"``.

    Returns
    -------
    zarr.Group
        Opened root group (read-only).

    Raises
    ------
    ValueError
        If ``backend="azure"`` but ``az_container`` or ``az_prefix`` are empty.
    FileNotFoundError
        If ``backend="local"`` and the path does not exist.
    OSError
        If ``backend="azure"`` and storage credentials are not configured.
    """
    if backend == "azure":
        if not az_container or not az_prefix:
            raise ValueError(
                "az_container and az_prefix must be non-empty when backend='azure'. "
                f"Got container={az_container!r}, prefix={az_prefix!r}"
            )
        from deepseismic.storage.blob_client import StorageClient

        client = StorageClient()
        store = client.open_zarr_store(az_container, az_prefix)
        logger.info("Opening zarr from ADLS: %s/%s", az_container, az_prefix)
        return zarr.open_group(store, mode="r")  # type: ignore[arg-type]

    # Local filesystem
    if local_path is None:
        raise ValueError("local_path must be provided when backend='local'")
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Zarr store not found at {p}. "
            "Run the ingest pipeline first, or use --storage-backend azure for ADLS."
        )
    logger.info("Opening zarr from local filesystem: %s", p)
    return zarr.open_group(str(p), mode="r")


def resolve_zarr_array(
    local_path: str | Path | None,
    array_name: str,
    *,
    backend: str = "local",
    az_container: str = "",
    az_prefix: str = "",
) -> Any:
    """Open a named array from a Zarr group (local or ADLS).

    Convenience wrapper around :func:`open_zarr_root` that also extracts a
    named array from the opened group.

    Parameters
    ----------
    local_path, backend, az_container, az_prefix:
        See :func:`open_zarr_root`.
    array_name:
        Name of the array within the zarr group (e.g. ``"amplitude"``).

    Returns
    -------
    zarr.Array
    """
    root = open_zarr_root(
        local_path, backend=backend, az_container=az_container, az_prefix=az_prefix
    )
    if array_name not in root:
        raise KeyError(
            f"Array '{array_name}' not found in zarr store. "
            f"Available: {list(root.keys())}"
        )
    return root[array_name]
