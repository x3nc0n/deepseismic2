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

import asyncio
import base64 as _b64
import io
import os
from collections.abc import AsyncIterator, Iterable, Iterator, MutableMapping
from typing import Any

import zarr
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobProperties, BlobServiceClient, ContainerClient
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.buffer.cpu import Buffer
from zarr.core.buffer.core import BufferPrototype

# Standard containers — matches architecture decision
CONTAINERS: tuple[str, ...] = ("raw", "staged", "features", "results", "catalog")

# Azurite well-known dev credentials — key is the standard emulator key
# documented at https://learn.microsoft.com/azure/storage/common/storage-use-azurite

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
    """MutableMapping-based ABS store kept for backward compatibility.

    Used internally by ``upload_zarr_store`` and any code that needs a
    plain dict-like interface.  For opening Zarr groups with zarr v3,
    use :class:`ABSZarrV3Store` (returned by :meth:`StorageClient.open_zarr_store`).
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
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: bytes) -> None:
        blob = self._client.get_blob_client(self._full_key(key))
        blob.upload_blob(value, overwrite=True)

    def __delitem__(self, key: str) -> None:
        blob = self._client.get_blob_client(self._full_key(key))
        try:
            blob.delete_blob()
        except ResourceNotFoundError:
            raise KeyError(key) from None

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


def _apply_byte_range(data: bytes, byte_range: ByteRequest | None) -> bytes:
    """Slice raw bytes according to a zarr v3 ByteRequest."""
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        if byte_range.suffix == 0:
            return b""
        return data[-byte_range.suffix :]
    return data


class ABSZarrV3Store(Store):
    """Zarr v3 ``Store`` subclass backed by Azure Blob Storage / Azurite.

    This is the correct store to pass to ``zarr.open_group()`` / ``zarr.open_array()``
    with zarr ≥ 3.0.  Azure Blob operations are synchronous; they are dispatched via
    ``asyncio.to_thread`` so the zarr async runtime is not blocked.

    Args:
        container_client: An ``azure.storage.blob.ContainerClient`` instance.
        prefix: Blob key prefix (e.g. ``"volve/synthetic.zarr"``).  A trailing
            ``/`` is added automatically if missing.
        read_only: When ``True`` write/delete operations raise ``ValueError``.
    """

    supports_writes: bool = True
    supports_deletes: bool = True
    supports_listing: bool = True

    def __init__(
        self,
        container_client: ContainerClient,
        prefix: str = "",
        *,
        read_only: bool = False,
    ) -> None:
        super().__init__(read_only=read_only)
        self._client = container_client
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        # Mark as open immediately — Azure connection is stateless.
        self._is_open = True

    def _full_key(self, key: str) -> str:
        return self._prefix + key

    # ------------------------------------------------------------------ #
    # Equality / repr                                                      #
    # ------------------------------------------------------------------ #

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._prefix == other._prefix
            and self._client.container_name == other._client.container_name
        )

    def __str__(self) -> str:
        return f"abs://{self._client.container_name}/{self._prefix}"

    def with_read_only(self, read_only: bool = False) -> ABSZarrV3Store:
        return type(self)(self._client, prefix=self._prefix.rstrip("/"), read_only=read_only)

    # ------------------------------------------------------------------ #
    # Open / close (no-ops for stateless HTTP connection)                  #
    # ------------------------------------------------------------------ #

    async def _open(self) -> None:
        self._is_open = True

    def close(self) -> None:  # type: ignore[override]
        self._is_open = False

    # ------------------------------------------------------------------ #
    # Read methods                                                         #
    # ------------------------------------------------------------------ #

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Download a blob and return as a zarr Buffer, or None if not found."""
        blob_client = self._client.get_blob_client(self._full_key(key))
        try:
            raw: bytes = await asyncio.to_thread(lambda: blob_client.download_blob().readall())
        except ResourceNotFoundError:
            return None
        raw = _apply_byte_range(raw, byte_range)
        return prototype.buffer.from_bytes(raw)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Retrieve multiple (possibly partial) values concurrently."""
        tasks = [self.get(key, prototype, br) for key, br in key_ranges]
        return list(await asyncio.gather(*tasks))

    async def exists(self, key: str) -> bool:
        blob_client = self._client.get_blob_client(self._full_key(key))
        return await asyncio.to_thread(blob_client.exists)

    # ------------------------------------------------------------------ #
    # Write / delete methods                                               #
    # ------------------------------------------------------------------ #

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        self._check_writable()
        if byte_range is not None:
            raise NotImplementedError("ABSZarrV3Store does not support partial writes")
        blob_client = self._client.get_blob_client(self._full_key(key))
        data = value.to_bytes()
        await asyncio.to_thread(lambda: blob_client.upload_blob(data, overwrite=True))

    async def delete(self, key: str) -> None:
        self._check_writable()
        blob_client = self._client.get_blob_client(self._full_key(key))
        try:
            await asyncio.to_thread(blob_client.delete_blob)
        except ResourceNotFoundError:
            pass  # zarr v3 Store.delete() is a no-op for missing keys

    # ------------------------------------------------------------------ #
    # Listing methods                                                      #
    # ------------------------------------------------------------------ #

    async def list(self) -> AsyncIterator[str]:
        blobs: list[str] = await asyncio.to_thread(
            lambda: [b.name for b in self._client.list_blobs(name_starts_with=self._prefix)]
        )
        for name in blobs:
            yield name[len(self._prefix):]

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        full = self._prefix + prefix
        blobs: list[str] = await asyncio.to_thread(
            lambda: [b.name for b in self._client.list_blobs(name_starts_with=full)]
        )
        for name in blobs:
            yield name[len(self._prefix):]

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        full = self._prefix + (prefix + "/" if prefix else "")
        blobs: list[str] = await asyncio.to_thread(
            lambda: [b.name for b in self._client.list_blobs(name_starts_with=full)]
        )
        seen: set[str] = set()
        for name in blobs:
            relative = name[len(full):]
            top = relative.split("/")[0]
            if top and top not in seen:
                seen.add(top)
                yield top


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

        account = os.getenv("AZURE_STORAGE_ACCOUNT") or os.getenv("STORAGE_ACCOUNT_NAME")
        if not account:
            raise OSError(
                "No storage credentials found. "
                "Set STORAGE_CONNECTION_STRING (local/Azurite) or "
                "AZURE_STORAGE_ACCOUNT / STORAGE_ACCOUNT_NAME (cloud with DefaultAzureCredential)."
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
            raise FileNotFoundError(f"{container}/{blob_path}") from None

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
            raise FileNotFoundError(f"{container}/{blob_path}") from None

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
            raise FileNotFoundError(f"{container}/{blob_path}") from None

    def delete_blob(self, container: str, blob_path: str) -> None:
        """Delete a single blob.  Raises FileNotFoundError if missing."""
        blob = self._container(container).get_blob_client(blob_path)
        try:
            blob.delete_blob()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"{container}/{blob_path}") from None

    def blob_exists(self, container: str, blob_path: str) -> bool:
        """Return True if the blob exists; False otherwise."""
        return self._container(container).get_blob_client(blob_path).exists()

    # ------------------------------------------------------------------ #
    # Zarr support                                                         #
    # ------------------------------------------------------------------ #

    def _abs_zarr_v3_store(
        self, container: str, prefix: str, *, read_only: bool = False
    ) -> ABSZarrV3Store:
        return ABSZarrV3Store(self._container(container), prefix=prefix, read_only=read_only)

    def upload_zarr_store(
        self,
        src_path: Any,
        container: str,
        prefix: str,
    ) -> None:
        """Upload a local Zarr store directory into blob storage.

        Walks every file under *src_path* (a local directory or
        :class:`zarr.storage.LocalStore`) and uploads each one as a blob,
        preserving the relative key structure required by zarr v3.

        Args:
            src_path: Local filesystem path (``str`` or ``Path``) to a zarr store
                directory, or a :class:`zarr.storage.LocalStore` instance.
            container:  Destination container (e.g. ``"staged"``)
            prefix:     Blob key prefix (e.g. ``"volve/synthetic.zarr"``)
        """
        from pathlib import Path

        if isinstance(src_path, zarr.storage.LocalStore):
            root_dir = Path(src_path.root)
        else:
            root_dir = Path(src_path)

        dest_container = self._container(container)
        pfx = prefix.rstrip("/") + "/" if prefix else ""

        for blob_file in root_dir.rglob("*"):
            if blob_file.is_file():
                rel = blob_file.relative_to(root_dir).as_posix()
                blob_key = pfx + rel
                with open(blob_file, "rb") as fh:
                    dest_container.get_blob_client(blob_key).upload_blob(fh, overwrite=True)

    def open_zarr_store(self, container: str, prefix: str) -> ABSZarrV3Store:
        """Return a zarr v3-compatible Store backed by blob storage.

        The returned :class:`ABSZarrV3Store` is a proper ``zarr.abc.store.Store``
        subclass and can be passed directly to ``zarr.open_group(store, mode="r")``.

        Example::

            store = client.open_zarr_store("staged", "volve/synthetic.zarr")
            root = zarr.open_group(store, mode="r")
            amp = root["amplitude"][0, :, :]
        """
        return self._abs_zarr_v3_store(container, prefix, read_only=False)
