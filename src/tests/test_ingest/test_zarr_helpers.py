"""Integration tests for zarr_helpers.open_zarr_root and segy_to_zarr.

Coverage
--------
TestOpenZarrRootLocal     — local dispatch: success, FileNotFoundError, ValueError.
TestResolveZarrArray      — resolve_zarr_array wraps open_zarr_root correctly.
TestSegyToZarr            — segy_to_zarr produces a valid, well-shaped zarr store from
                            the synthetic SEG-Y (format proxy — tests code path, not
                            geophysical values). Validates arrays present, dtype, shape,
                            NaN/Inf absence, survey_id sidecar, and sample_mode limiting.
TestOpenZarrRootAzureMock — Azure branch dispatches to StorageClient with dict-backed
                            MockContainerClient (no real Azure / no Azurite required).

All tests are CI-safe (local filesystem only). No @pytest.mark.integration needed here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import zarr
import zarr.storage
from azure.core.exceptions import ResourceNotFoundError

from deepseismic.ingest.segy_loader import segy_to_zarr
from deepseismic.storage.blob_client import ABSZarrV3Store
from deepseismic.storage.zarr_helpers import open_zarr_root, resolve_zarr_array

# ---------------------------------------------------------------------------
# Dict-backed mock ContainerClient (mirrors test_data_readers pattern)
# ---------------------------------------------------------------------------


class _MockDownloader:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _MockBlobClient:
    def __init__(self, store: dict[str, bytes], key: str) -> None:
        self._store = store
        self._key = key

    def download_blob(self) -> _MockDownloader:
        if self._key not in self._store:
            raise ResourceNotFoundError(f"Blob not found: {self._key}")
        return _MockDownloader(self._store[self._key])

    def upload_blob(self, data: bytes, *, overwrite: bool = True) -> None:
        self._store[self._key] = data if isinstance(data, bytes) else data.read()

    def exists(self) -> bool:
        return self._key in self._store


class _MockContainerClient:
    container_name = "staged"

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def get_blob_client(self, key: str) -> _MockBlobClient:
        return _MockBlobClient(self._blobs, key)

    def list_blobs(self, *, name_starts_with: str = "") -> list[Any]:
        from unittest.mock import MagicMock

        result = []
        for name in self._blobs:
            if name.startswith(name_starts_with):
                m = MagicMock()
                m.name = name
                result.append(m)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# TestOpenZarrRootLocal
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenZarrRootLocal:
    def test_open_local_success(self, tmp_path: Path) -> None:
        """open_zarr_root returns a zarr.Group for a valid local path."""
        store_path = tmp_path / "test.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store, mode="w")
        root.create_array("data", data=np.array([1, 2, 3], dtype=np.float32))

        result = open_zarr_root(store_path)
        assert isinstance(result, zarr.Group)

    def test_open_local_content_readable(self, tmp_path: Path) -> None:
        """open_zarr_root returns a group whose arrays are readable."""
        store_path = tmp_path / "content.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store, mode="w")
        arr_in = np.arange(10, dtype=np.float32)
        root.create_array("values", data=arr_in)

        result = open_zarr_root(store_path)
        np.testing.assert_array_equal(result["values"][:], arr_in)

    def test_open_local_accepts_string_path(self, tmp_path: Path) -> None:
        """open_zarr_root accepts a string path as well as a Path object."""
        store_path = tmp_path / "str.zarr"
        zarr.open_group(zarr.storage.LocalStore(str(store_path)), mode="w")

        result = open_zarr_root(str(store_path))
        assert isinstance(result, zarr.Group)

    def test_open_local_missing_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """open_zarr_root raises FileNotFoundError for a nonexistent path."""
        missing = tmp_path / "does_not_exist.zarr"
        with pytest.raises(FileNotFoundError, match="does_not_exist.zarr"):
            open_zarr_root(missing)

    def test_open_local_none_path_raises_value_error(self) -> None:
        """open_zarr_root(None, backend='local') raises ValueError."""
        with pytest.raises(ValueError, match="local_path must be provided"):
            open_zarr_root(None, backend="local")

    def test_open_azure_empty_container_raises(self) -> None:
        """open_zarr_root raises ValueError when az_container is empty."""
        with pytest.raises(ValueError, match="az_container"):
            open_zarr_root(None, backend="azure", az_container="", az_prefix="some/prefix")

    def test_open_azure_empty_prefix_raises(self) -> None:
        """open_zarr_root raises ValueError when az_prefix is empty."""
        with pytest.raises(ValueError, match="az_prefix"):
            open_zarr_root(None, backend="azure", az_container="staged", az_prefix="")

    def test_open_azure_both_empty_raises(self) -> None:
        """open_zarr_root raises ValueError when both az_container and az_prefix are empty."""
        with pytest.raises(ValueError):
            open_zarr_root(None, backend="azure", az_container="", az_prefix="")


# ─────────────────────────────────────────────────────────────────────────────
# TestResolveZarrArray
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveZarrArray:
    def test_resolve_array_success(self, tmp_path: Path) -> None:
        """resolve_zarr_array returns the named array from a local store."""
        store_path = tmp_path / "resolve.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store, mode="w")
        expected = np.arange(20, dtype=np.float32)
        root.create_array("amplitude", data=expected)

        arr = resolve_zarr_array(store_path, "amplitude")
        np.testing.assert_array_equal(arr[:], expected)

    def test_resolve_array_missing_raises_key_error(self, tmp_path: Path) -> None:
        """resolve_zarr_array raises KeyError when the array name is absent."""
        store_path = tmp_path / "missing.zarr"
        zarr.open_group(zarr.storage.LocalStore(str(store_path)), mode="w")

        with pytest.raises(KeyError, match="not found"):
            resolve_zarr_array(store_path, "nonexistent")


# ─────────────────────────────────────────────────────────────────────────────
# TestSegyToZarr — real segyio + zarr pipeline, synthetic SEG-Y as format proxy
# ─────────────────────────────────────────────────────────────────────────────


class TestSegyToZarr:
    """Validate segy_to_zarr end-to-end: code path correctness, not geophysics.

    The synthetic SEG-Y fixture (5 × 5 × 100, seed=0) is a format proxy.
    Assertions are structural (dtype, shape, NaN/Inf, required arrays, sidecar
    fields) — NOT tied to Volve survey geometry or specific amplitude values.
    """

    def test_segy_to_zarr_produces_amplitude_array(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """segy_to_zarr must write an 'amplitude' array in the output zarr store."""
        dest = tmp_path / "amp.zarr"
        segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        root = open_zarr_root(dest)
        assert "amplitude" in root, f"'amplitude' not in zarr keys: {list(root.keys())}"

    def test_segy_to_zarr_coordinate_arrays_present(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """segy_to_zarr must write inline, crossline, and twtt_ms coordinate arrays."""
        dest = tmp_path / "coords.zarr"
        segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        root = open_zarr_root(dest)
        for name in ("inline", "crossline", "twtt_ms"):
            assert name in root, f"Coordinate array '{name}' missing from zarr store"

    def test_segy_to_zarr_amplitude_dtype_float32(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """Amplitude array must be float32 (SEG-Y IBM float maps to float32)."""
        dest = tmp_path / "dtype.zarr"
        segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        root = open_zarr_root(dest)
        arr = root["amplitude"]
        assert arr.dtype == np.dtype("float32"), (
            f"Expected float32 amplitude, got {arr.dtype}"
        )

    def test_segy_to_zarr_shape_matches_geometry(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """Amplitude shape must match (n_inlines, n_crosslines, n_samples) from SEG-Y headers."""
        dest = tmp_path / "shape.zarr"
        meta = segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        root = open_zarr_root(dest)
        arr = root["amplitude"]
        geom = meta.geometry
        assert arr.shape == (geom["n_inlines"], geom["n_crosslines"], geom["n_samples"]), (
            f"Expected shape {(geom['n_inlines'], geom['n_crosslines'], geom['n_samples'])}, "
            f"got {arr.shape}"
        )

    def test_segy_to_zarr_no_nans_in_amplitude(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """Amplitude array must contain no NaN values after ingest."""
        dest = tmp_path / "nonan.zarr"
        segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        data = np.asarray(open_zarr_root(dest)["amplitude"])
        assert not np.any(np.isnan(data)), "NaN found in amplitude after segy_to_zarr"

    def test_segy_to_zarr_no_infs_in_amplitude(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """Amplitude array must contain no Inf values after ingest."""
        dest = tmp_path / "noinf.zarr"
        segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        data = np.asarray(open_zarr_root(dest)["amplitude"])
        assert not np.any(np.isinf(data)), "Inf found in amplitude after segy_to_zarr"

    def test_segy_to_zarr_survey_id_in_sidecar(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """Sidecar JSON must embed the survey_id passed to segy_to_zarr."""
        dest = tmp_path / "meta.zarr"
        survey_id = "synthetic-test-survey"
        segy_to_zarr(sample_segy_path, dest, survey_id=survey_id, overwrite=True)

        sidecar = dest.with_suffix(".json")
        assert sidecar.exists(), "JSON sidecar must be written alongside the zarr store"
        meta = json.loads(sidecar.read_text())
        assert meta.get("survey_id") == survey_id

    def test_segy_to_zarr_sidecar_geometry_positive_dims(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """Sidecar geometry must report positive n_inlines, n_crosslines, n_samples."""
        dest = tmp_path / "geom.zarr"
        meta = segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        geom = meta.geometry
        assert geom["n_inlines"] > 0, "n_inlines must be > 0"
        assert geom["n_crosslines"] > 0, "n_crosslines must be > 0"
        assert geom["n_samples"] > 0, "n_samples must be > 0"

    def test_segy_to_zarr_sample_mode_limits_inlines(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """sample_mode=True with sample_n_inlines=2 must produce only 2 inlines."""
        dest = tmp_path / "sample.zarr"
        segy_to_zarr(
            sample_segy_path, dest,
            survey_id="test",
            sample_mode=True,
            sample_n_inlines=2,
            overwrite=True,
        )

        root = open_zarr_root(dest)
        n_inlines = root["amplitude"].shape[0]
        assert n_inlines == 2, (
            f"sample_mode=True, sample_n_inlines=2 must yield 2 inlines, got {n_inlines}"
        )

    def test_segy_to_zarr_inline_coord_matches_segy_headers(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """Inline coordinate array must match inline numbers from SEG-Y headers."""
        import segyio

        dest = tmp_path / "ilines.zarr"
        segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        root = open_zarr_root(dest)
        zarr_inlines = set(root["inline"][:].tolist())

        with segyio.open(str(sample_segy_path), ignore_geometry=False) as f:
            segy_inlines = set(int(il) for il in f.ilines)

        assert zarr_inlines == segy_inlines, (
            f"Inline coords in zarr {zarr_inlines} != SEG-Y inlines {segy_inlines}"
        )

    def test_segy_to_zarr_amplitude_stats_in_metadata(
        self, sample_segy_path: Path, tmp_path: Path
    ) -> None:
        """IngestMetadata amplitude_stats must contain required stat keys."""
        dest = tmp_path / "stats.zarr"
        meta = segy_to_zarr(sample_segy_path, dest, survey_id="test", overwrite=True)

        required_keys = {"min", "max", "mean", "std", "p01", "p99", "nonzero_fraction"}
        missing = required_keys - set(meta.amplitude_stats)
        assert not missing, f"amplitude_stats missing keys: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# TestOpenZarrRootAzureMock — Azure branch with dict-backed mock client
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenZarrRootAzureMock:
    """Azure dispatch via patched StorageClient — no real Azure / Azurite.

    StorageClient is imported inside open_zarr_root() via a local import, so we
    patch 'deepseismic.storage.blob_client.StorageClient' (the source) rather than
    a module-level attribute in zarr_helpers.
    """

    def _write_zarr_to_mock(
        self, container: _MockContainerClient, prefix: str, data: np.ndarray
    ) -> None:
        """Write a tiny zarr group into the mock container via ABSZarrV3Store."""

        async def _write() -> None:
            store = ABSZarrV3Store(container, prefix=prefix)
            await store._open()
            root = zarr.open_group(store, mode="w")
            root.create_array("amplitude", data=data)

        asyncio.run(_write())

    def test_open_azure_mock_returns_group(self) -> None:
        """open_zarr_root(backend='azure') returns a zarr.Group via the mock client."""
        container = _MockContainerClient()
        data = np.ones((3, 3, 10), dtype=np.float32)
        self._write_zarr_to_mock(container, "test/amp.zarr", data)

        class _MockStorageClient:
            def open_zarr_store(self, az_container: str, az_prefix: str) -> ABSZarrV3Store:
                return ABSZarrV3Store(container, prefix=az_prefix)

        with patch(
            "deepseismic.storage.blob_client.StorageClient",
            return_value=_MockStorageClient(),
        ):
            result = open_zarr_root(
                None,
                backend="azure",
                az_container="staged",
                az_prefix="test/amp.zarr",
            )

        assert isinstance(result, zarr.Group)

    def test_open_azure_mock_reads_correct_data(self) -> None:
        """Data written to mock container is readable via open_zarr_root(backend='azure')."""
        container = _MockContainerClient()
        expected = np.arange(30, dtype=np.float32).reshape(3, 2, 5)
        self._write_zarr_to_mock(container, "surveys/s1/amplitude.zarr", expected)

        class _MockStorageClient:
            def open_zarr_store(self, az_container: str, az_prefix: str) -> ABSZarrV3Store:
                return ABSZarrV3Store(container, prefix=az_prefix)

        with patch(
            "deepseismic.storage.blob_client.StorageClient",
            return_value=_MockStorageClient(),
        ):
            root = open_zarr_root(
                None,
                backend="azure",
                az_container="staged",
                az_prefix="surveys/s1/amplitude.zarr",
            )

        actual = np.asarray(root["amplitude"])
        np.testing.assert_array_equal(actual, expected)
