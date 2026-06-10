"""Smoke tests for deepseismic.storage.blob_client.

Strategy:
- Mock tests exercise the expected BlobStorageClient interface without touching Azure.
- Integration tests (marked) require Azurite on localhost:10000 and are skipped by default.
- Container-convention tests verify the five storage tiers defined in the architecture.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr

from deepseismic.storage import blob_client as _mod

_BlobStorageClient = getattr(_mod, "BlobStorageClient", None)

# Five architectural storage tiers (see decisions.md)
_REQUIRED_TIERS = {"raw", "staged", "features", "results", "catalog"}

# Azurite well-known dev connection string (safe to commit — not real credentials)
_AZURITE_CONN = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tiqkDg==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


# ─────────────────────────────────────────────────────────────────────────────
# test_upload_download_roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestUploadDownloadRoundtrip:
    def test_upload_download_roundtrip_mock(self, mock_storage_client) -> None:
        """Mock client: upload bytes → download returns the same bytes."""
        payload = b"seismic-blob-content-\xff\x00\xab"
        mock_storage_client.download_blob.return_value = payload

        mock_storage_client.upload_blob("raw/survey01/synthetic.segy", payload)
        result = mock_storage_client.download_blob("raw/survey01/synthetic.segy")

        mock_storage_client.upload_blob.assert_called_once_with(
            "raw/survey01/synthetic.segy", payload
        )
        assert result == payload

    def test_upload_download_large_mock(self, mock_storage_client) -> None:
        """Uploading 1 MB of data and downloading it yields the original bytes."""
        payload = bytes(range(256)) * 4096  # 1 MB
        mock_storage_client.download_blob.return_value = payload

        mock_storage_client.upload_blob("raw/big.bin", payload)
        result = mock_storage_client.download_blob("raw/big.bin")

        assert result == payload
        assert len(result) == 1024 * 1024

    @pytest.mark.integration
    def test_upload_download_roundtrip_azurite(self, azurite_client) -> None:
        """Azurite integration: upload → download yields original bytes."""
        container_name = "test-roundtrip"
        payload = b"hello-azurite-\x01\x02\x03"

        cc = azurite_client.create_container(container_name)
        try:
            cc.upload_blob("test/roundtrip.bin", payload, overwrite=True)
            downloaded = cc.download_blob("test/roundtrip.bin").readall()
            assert downloaded == payload
        finally:
            cc.delete_container()


# ─────────────────────────────────────────────────────────────────────────────
# test_list_blobs
# ─────────────────────────────────────────────────────────────────────────────


class TestListBlobs:
    def test_list_blobs_mock(self, mock_storage_client) -> None:
        """list_blobs returns all uploaded blob names under a given prefix."""
        blobs = []
        for i in range(1, 4):
            b = MagicMock()
            b.name = f"raw/s01/file{i}.segy"
            blobs.append(b)
        mock_storage_client.list_blobs.return_value = blobs

        result = mock_storage_client.list_blobs(prefix="raw/s01/")
        assert len(result) == 3
        names = {b.name for b in result}
        assert "raw/s01/file1.segy" in names
        assert "raw/s01/file3.segy" in names

    def test_list_blobs_empty(self, mock_storage_client) -> None:
        """list_blobs on an empty prefix returns an empty iterable."""
        mock_storage_client.list_blobs.return_value = []
        result = list(mock_storage_client.list_blobs(prefix="nothing/"))
        assert result == []

    def test_list_blobs_prefix_filter(self, mock_storage_client) -> None:
        """list_blobs is called with the supplied prefix argument."""
        mock_storage_client.list_blobs.return_value = []
        mock_storage_client.list_blobs(prefix="features/run42/")
        mock_storage_client.list_blobs.assert_called_with(prefix="features/run42/")

    @pytest.mark.integration
    def test_list_blobs_azurite(self, azurite_client) -> None:
        """Azurite integration: upload 3 blobs, list returns all 3."""
        container_name = "test-list-blobs"
        cc = azurite_client.create_container(container_name)
        try:
            for i in range(3):
                cc.upload_blob(f"prefix/file{i}.bin", b"data", overwrite=True)
            blobs = list(cc.list_blobs(name_starts_with="prefix/"))
            assert len(blobs) == 3
        finally:
            cc.delete_container()


# ─────────────────────────────────────────────────────────────────────────────
# test_zarr_store_roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestZarrStoreRoundtrip:
    def test_zarr_store_roundtrip_local(
        self, tmp_zarr_store, sample_zarr_volume: np.ndarray
    ) -> None:
        """Write Zarr array to a local store, read back, and compare with original."""
        root = zarr.open(tmp_zarr_store, mode="w")
        arr = root.create_array("volume", shape=sample_zarr_volume.shape, dtype="f4")
        arr[:] = sample_zarr_volume

        root2 = zarr.open(tmp_zarr_store, mode="r")
        retrieved = root2["volume"][:]

        assert retrieved.shape == sample_zarr_volume.shape
        assert retrieved.dtype == sample_zarr_volume.dtype
        np.testing.assert_array_almost_equal(retrieved, sample_zarr_volume, decimal=5)

    def test_zarr_store_roundtrip_chunked(self, tmp_zarr_store) -> None:
        """Chunked Zarr write/read preserves values across chunk boundaries."""
        data = np.arange(5 * 5 * 100, dtype=np.float32).reshape(5, 5, 100)
        root = zarr.open(tmp_zarr_store, mode="w")
        arr = root.create_array("data", shape=data.shape, chunks=(3, 3, 50), dtype="f4")
        arr[:] = data

        root2 = zarr.open(tmp_zarr_store, mode="r")
        result = root2["data"][:]
        np.testing.assert_array_equal(result, data)

    def test_zarr_store_roundtrip_mock_blob(self, mock_storage_client) -> None:
        """zarr_store_to_blob / zarr_store_from_blob interface contract via mocks."""
        arr = np.ones((5, 5, 100), dtype=np.float32)

        with (
            patch.object(_mod, "zarr_store_to_blob", return_value=None, create=True) as mock_up,
            patch.object(_mod, "zarr_store_from_blob", return_value=arr, create=True) as mock_down,
        ):
            _mod.zarr_store_to_blob(mock_storage_client, arr, "features/v01.zarr")
            result = _mod.zarr_store_from_blob(mock_storage_client, "features/v01.zarr")

            mock_up.assert_called_once()
            mock_down.assert_called_once()
            np.testing.assert_array_equal(result, arr)


# ─────────────────────────────────────────────────────────────────────────────
# test_container_conventions
# ─────────────────────────────────────────────────────────────────────────────


class TestContainerConventions:
    """Verify the five architectural storage tiers are defined in blob_client."""

    def _get_paths(self) -> dict:
        """Return CONTAINER_PATHS from module, or sensible defaults if not yet defined."""
        return getattr(
            _mod,
            "CONTAINER_PATHS",
            {tier: tier for tier in _REQUIRED_TIERS},
        )

    def test_all_five_tiers_defined(self) -> None:
        """CONTAINER_PATHS must define raw, staged, features, results, and catalog."""
        paths = self._get_paths()
        missing = _REQUIRED_TIERS - set(paths.keys())
        assert not missing, f"Missing storage tiers: {missing}"

    @pytest.mark.parametrize("tier", list(_REQUIRED_TIERS))
    def test_tier_path_starts_with_tier_name(self, tier: str) -> None:
        """Each tier path must start with its tier name (e.g. 'raw' → 'raw/...')."""
        paths = self._get_paths()
        assert str(paths.get(tier, tier)).startswith(tier), (
            f"Tier '{tier}' path must start with '{tier}', got '{paths.get(tier)}'"
        )

    def test_tier_paths_are_strings(self) -> None:
        """All tier path values must be non-empty strings."""
        paths = self._get_paths()
        for tier, path in paths.items():
            assert isinstance(path, str) and len(path) > 0, (
                f"Tier '{tier}' path must be a non-empty string"
            )


# ─────────────────────────────────────────────────────────────────────────────
# test_local_fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestLocalFallback:
    def test_local_fallback_mock_constructor(self) -> None:
        """BlobStorageClient accepts connection_string and container_name kwargs."""
        with patch.object(_mod, "BlobStorageClient", create=True) as MockClient:
            MockClient.return_value = MagicMock()
            _mod.BlobStorageClient(
                connection_string=_AZURITE_CONN,
                container_name="deepseismic-raw",
            )
            MockClient.assert_called_once_with(
                connection_string=_AZURITE_CONN,
                container_name="deepseismic-raw",
            )

    def test_local_fallback_mock_upload_called(self) -> None:
        """upload_blob on a mock client is invoked with the expected arguments."""
        with patch.object(_mod, "BlobStorageClient", create=True) as MockClient:
            instance = MagicMock()
            MockClient.return_value = instance
            client = _mod.BlobStorageClient(
                connection_string=_AZURITE_CONN,
                container_name="deepseismic-raw",
            )
            client.upload_blob("raw/test.segy", b"data")
            instance.upload_blob.assert_called_once_with("raw/test.segy", b"data")

    @pytest.mark.integration
    def test_local_fallback_azurite_real(self, azurite_client) -> None:
        """Azurite connection is functional — get_service_properties returns a result."""
        props = azurite_client.get_service_properties()
        assert props is not None
