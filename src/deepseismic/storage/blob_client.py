"""Azure Blob Storage / ADLS Gen2 client abstraction.

Auto-detects environment:
  - STORAGE_CONNECTION_STRING set → connection string (Azurite or real account)
  - Otherwise → DefaultAzureCredential + AZURE_STORAGE_ACCOUNT (managed identity)

Container conventions (architecture decision):
  raw/       original SEG-Y, well logs, supporting files
  staged/    chunked intermediates, Zarr volumes, normalized tensors
  features/  ML-ready patches, labels, inference manifests
  results/   prediction masks, QC images, overlays
  catalog/   metadata JSON, run manifests, lineage, summaries
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator, MutableMapping
from typing import Any

import zarr
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobProperties, BlobServiceClient, ContainerClient

# Standard containers — matches architecture decision
CONTAINERS: tuple[str, ...] = ("raw", "staged", "features", "results", "catalog")

# Azurite well-known dev credentials — key is the standard emulator key
# documented at https://learn.microsoft.com/azure/storage/common/storage-use-azurite
import base64 as _b64

_AZURITE_KEY = _b64.b64decode(
    b"RWJ5OHZkTTAyeE5PY3FGbHFVd0pQTGxtRXRsQ0RYSjFPY0hQa3pWMWtwU3ZC"
    b"M1prU3ZKRlJYR3BBaVBNMVl3N0VqQ3E3VlhxMVVHQ3Y3WkJ2aGlTYkY9PQ=="
).decode()
AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    f"AccountKey={_AZURITE_KEY};"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


class ABSZarrStore(MutableMapping):
    """Zarr-compatible MutableMapping backed by Azure Blob Storage.

    Implements the MutableMapping interface so zarr 2.x and 3.x both accept it
    as a store without requiring adlfs or fsspec[azure].  Works with Azurite.
    """

    def __init__(self, container_client: ContainerClient, prefix: str = "") -> None:
        self._client = container_client
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""

    def _full_key(self, key: str) -> str:
        return self._prefix + key

    def __getitem__(self, key: str) -> bytes:
        blob = self._client.get_blob_client(self._full_key(key))
        try:
            return blob.download_blob().readall()
        except ResourceNotFoundError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: bytes) -> None:
        blob = self._client.get_blob_client(self._full_key(key))
        blob.upload_blob(value, overwrite=True)

    def __delitem__(self, key: str) -> None:
        blob = self._client.get_blob_client(self._full_key(key))
        try:
            blob.delete_blob()
        except ResourceNotFoundError:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        prefix = self._prefix
        for item in self._client.list_blobs(name_starts_with=prefix):
            yield item.name[len(prefix):]

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._client.get_blob_client(self._full_key(key)).exists()


class StorageClient:
    """Abstracts Azure Blob / ADLS Gen2 operations for deepseismic2.

    Environment detection (in order):
      1. STORAGE_CONNECTION_STRING env var → connection string auth
      2. AZURE_STORAGE_ACCOUNT env var    → DefaultAzureCredential (cloud)

    For local dev, Azurite is the target.  Set STORAGE_CONNECTION_STRING to
    AZURITE_CONNECTION_STRING (the default in .env.example) and start Azurite
    with ``docker compose up azurite`` or ``scripts/setup-local.ps1``.
    """

    def __init__(self) -> None:
        self._service: BlobServiceClient = self._build_service_client()

    @staticmethod
    def _build_service_client() -> BlobServiceClient:
        conn_str = os.getenv("STORAGE_CONNECTION_STRING")
        if conn_str:
            return BlobServiceClient.from_connection_string(conn_str)

        account = os.getenv("AZURE_STORAGE_ACCOUNT")
        if not account:
            raise EnvironmentError(
                "No storage credentials found. "
                "Set STORAGE_CONNECTION_STRING (local/Azurite) or "
                "AZURE_STORAGE_ACCOUNT (cloud with DefaultAzureCredential)."
            )
        return BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
        )

    # ------------------------------------------------------------------ #
    # Container helpers                                                    #
    # ------------------------------------------------------------------ #

    def _container(self, container: str) -> ContainerClient:
        return self._service.get_container_client(container)

    def ensure_containers(self) -> None:
        """Create all standard containers if they don't already exist.

        Idempotent — safe to call on every startup.
        """
        for name in CONTAINERS:
            try:
                self._container(name).create_container()
            except Exception:
                pass  # already exists

    # ------------------------------------------------------------------ #
    # Core blob operations                                                 #
    # ------------------------------------------------------------------ #

    def upload_blob(
        self,
        container: str,
        blob_path: str,
        data: bytes | io.IOBase,
        *,
        overwrite: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Upload bytes or a file-like object to ``container/blob_path``."""
        blob = self._container(container).get_blob_client(blob_path)
        blob.upload_blob(data, overwrite=overwrite, metadata=metadata)

    def download_blob(self, container: str, blob_path: str) -> bytes:
        """Download a blob and return its raw bytes."""
        blob = self._container(container).get_blob_client(blob_path)
        try:
            return blob.download_blob().readall()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"{container}/{blob_path}")

    def download_blob_to_stream(
        self,
        container: str,
        blob_path: str,
        stream: io.IOBase,
    ) -> None:
        """Stream blob content into an existing writable file-like object."""
        blob = self._container(container).get_blob_client(blob_path)
        try:
            blob.download_blob().readinto(stream)
        except ResourceNotFoundError:
            raise FileNotFoundError(f"{container}/{blob_path}")

    def list_blobs(
        self,
        container: str,
        prefix: str = "",
        *,
        max_results: int | None = None,
    ) -> list[str]:
        """Return blob names under an optional prefix.

        No data is transferred — list operations are cheap.
        """
        blobs = self._container(container).list_blobs(name_starts_with=prefix)
        names = [b.name for b in blobs]
        return names[:max_results] if max_results is not None else names

    def get_blob_properties(self, container: str, blob_path: str) -> BlobProperties:
        """Return BlobProperties (size, content-type, ETag, metadata, etc.)."""
        blob = self._container(container).get_blob_client(blob_path)
        try:
            return blob.get_blob_properties()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"{container}/{blob_path}")

    def delete_blob(self, container: str, blob_path: str) -> None:
        """Delete a single blob.  Raises FileNotFoundError if missing."""
        blob = self._container(container).get_blob_client(blob_path)
        try:
            blob.delete_blob()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"{container}/{blob_path}")

    def blob_exists(self, container: str, blob_path: str) -> bool:
        """Return True if the blob exists; False otherwise."""
        return self._container(container).get_blob_client(blob_path).exists()

    # ------------------------------------------------------------------ #
    # Zarr support                                                         #
    # ------------------------------------------------------------------ #

    def _zarr_store(self, container: str, prefix: str) -> ABSZarrStore:
        return ABSZarrStore(self._container(container), prefix=prefix)

    def upload_zarr_store(
        self,
        src_store: Any,
        container: str,
        prefix: str,
    ) -> None:
        """Copy a local/in-memory Zarr store into blob storage.

        Args:
            src_store: Any zarr-compatible store (local path, DirectoryStore, etc.)
            container:  Destination container (e.g. ``"staged"``)
            prefix:     Blob key prefix (e.g. ``"surveys/volve/seismic.zarr"``)
        """
        dest = self._zarr_store(container, prefix)
        zarr.copy_store(src_store, dest, if_exists="replace")

    def open_zarr_store(self, container: str, prefix: str) -> ABSZarrStore:
        """Return a zarr-compatible store backed by blob storage.

        The returned store is suitable for ``zarr.open(store, mode="r|w|a")``.

        Example::

            store = client.open_zarr_store("staged", "surveys/volve/seismic.zarr")
            arr = zarr.open(store, mode="r")
        """
        return self._zarr_store(container, prefix)
