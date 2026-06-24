"""CI-safe tests for ABSZarrV3Store and _data_readers backend resolver.

Coverage
--------
1. TestApplyByteRange           — _apply_byte_range() for all four ByteRequest variants + None.
2. TestABSZarrV3StoreRoundTrip  — dict-backed mock ContainerClient: write→read allclose,
                                   missing-key returns None, with_read_only raises on write,
                                   mode="r" open on read-only store reads correctly.
3. TestBackendResolverDefaults  — _azure_sources() env-var contract; default values.
4. TestBackendResolverLocalMissing — local backend: fault_prob absent → get_fault_prob_slice
                                     returns None (graceful degradation, no real files needed).
5. TestBackendResolverLocalPresent — local backend with real on-disk zarr (skipif missing).
6. TestBackendResolverAzure     — azure backend: mock _storage_client; get_volume_coords/
                                   get_amplitude_slice read from dict-backed store; resolver
                                   picks correct container/prefix per env vars.
7. TestAzureFaultSticksAzure    — azure fault sticks: canonical coordinate mapping
                                   abs_il=1001+idx, abs_xl=1900+idx, twt_ms=z*4.0; TWT≥800 ms
                                   guard; missing sticks returns {} gracefully.
8. TestAzuriteIntegration       — @pytest.mark.integration — real Azurite round-trip via
                                   upload_zarr_store + open_zarr_store. Excluded from CI.

No Streamlit imports.  All CI-blocking tests require zero network/Azurite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr
import zarr.storage
from azure.core.exceptions import ResourceNotFoundError
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.buffer.cpu import Buffer
from zarr.core.buffer import default_buffer_prototype

from deepseismic.storage.blob_client import ABSZarrV3Store, _apply_byte_range
from deepseismic.ui import _data_readers

# ---------------------------------------------------------------------------
# Paths to real staged data (used only in TestBackendResolverLocalPresent)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[3]
_ZARR_AMP  = _REPO_ROOT / "data/volve/staged/synthetic.zarr"
_ZARR_PROB = _REPO_ROOT / "data/volve/staged/fault_prob.zarr"


# ---------------------------------------------------------------------------
# Dict-backed mock ContainerClient — mirrors Dallas's proof-of-concept pattern.
#
# Each _MockBlobClient proxies reads/writes through the shared _store dict so
# multiple ABSZarrV3Store instances over the same mock see consistent data.
# ResourceNotFoundError is raised (not FileNotFoundError) so ABSZarrV3Store.get()
# catches it and returns None, matching real Azure SDK behaviour.
# ---------------------------------------------------------------------------


class _MockDownloader:
    """Minimal stand-in for azure.storage.blob.StorageStreamDownloader."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _MockBlobClient:
    """Thin proxy into a shared dict for one blob key."""

    def __init__(self, store: dict[str, bytes], key: str) -> None:
        self._store = store
        self._key = key

    def download_blob(self) -> _MockDownloader:
        if self._key not in self._store:
            raise ResourceNotFoundError(self._key)
        return _MockDownloader(self._store[self._key])

    def upload_blob(self, data: bytes, overwrite: bool = True) -> None:
        self._store[self._key] = bytes(data)

    def delete_blob(self) -> None:
        if self._key not in self._store:
            raise ResourceNotFoundError(self._key)
        del self._store[self._key]

    def exists(self) -> bool:
        return self._key in self._store


class _MockContainerClient:
    """In-memory ContainerClient backed by a plain dict.

    Provides the subset of ContainerClient API consumed by ABSZarrV3Store:
    - get_blob_client(blob_name) → _MockBlobClient
    - list_blobs(name_starts_with="") → list of BlobProperties-like objects
    - container_name attribute
    """

    def __init__(self, initial: dict[str, bytes] | None = None) -> None:
        self._store: dict[str, bytes] = dict(initial or {})
        self.container_name = "mock-container"

    def get_blob_client(self, blob_name: str) -> _MockBlobClient:
        return _MockBlobClient(self._store, blob_name)

    def list_blobs(self, name_starts_with: str = "") -> list[Any]:
        blobs = []
        for key in sorted(self._store):
            if key.startswith(name_starts_with):
                prop = MagicMock()
                prop.name = key
                blobs.append(prop)
        return blobs


class _MockStorageClient:
    """Minimal StorageClient stand-in for _data_readers azure-path tests.

    open_zarr_store  — returns ABSZarrV3Store over a shared _MockContainerClient.
    list_blobs       — returns blob names from a separate sticks dict.
    download_blob    — downloads bytes from the sticks dict.
    """

    def __init__(
        self,
        mock_container: _MockContainerClient,
        sticks_blobs: dict[str, bytes] | None = None,
    ) -> None:
        self._mock_container = mock_container
        self._sticks_blobs: dict[str, bytes] = sticks_blobs or {}

    def open_zarr_store(self, container: str, prefix: str) -> ABSZarrV3Store:
        return ABSZarrV3Store(self._mock_container, prefix=prefix)

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        return [name for name in sorted(self._sticks_blobs) if name.startswith(prefix)]

    def download_blob(self, container: str, blob_name: str) -> bytes:
        if blob_name not in self._sticks_blobs:
            raise FileNotFoundError(blob_name)
        return self._sticks_blobs[blob_name]


# ---------------------------------------------------------------------------
# Helper: write a tiny amplitude zarr INTO a mock container
# Returns (populated_mock_container, amp_data) so caller can seed assertions.
# ---------------------------------------------------------------------------


def _seed_amp_zarr(
    mock_container: _MockContainerClient,
    prefix: str,
    n_il: int = 10,
    n_xl: int = 20,
    n_s: int = 50,
) -> np.ndarray:
    """Write a minimal amplitude zarr group into *mock_container* under *prefix*.

    Arrays written: ``inline``, ``crossline``, ``twtt_ms``, ``amplitude``.
    Returns the amplitude data so tests can assert allclose.
    """
    store = ABSZarrV3Store(mock_container, prefix=prefix)
    root = zarr.open_group(store, mode="w")

    inline_arr  = np.arange(1001, 1001 + n_il, dtype=np.float32)
    xl_arr      = np.arange(1900, 1900 + n_xl, dtype=np.float32)
    twtt_arr    = np.arange(0, n_s * 4, 4, dtype=np.float32)
    amp_data    = np.random.default_rng(42).standard_normal((n_il, n_xl, n_s)).astype(np.float32)

    root.create_array("inline",    data=inline_arr)
    root.create_array("crossline", data=xl_arr)
    root.create_array("twtt_ms",   data=twtt_arr)
    root.create_array("amplitude", data=amp_data)
    return amp_data


# ============================================================================
# 1. _apply_byte_range
# ============================================================================


class TestApplyByteRange:
    """Unit-level coverage for _apply_byte_range — all ByteRequest variants + None."""

    _DATA = b"0123456789"

    def test_none_returns_full_bytes(self) -> None:
        assert _apply_byte_range(self._DATA, None) == self._DATA

    def test_range_byte_request(self) -> None:
        # RangeByteRequest(start, end) → data[start:end]
        result = _apply_byte_range(self._DATA, RangeByteRequest(2, 5))
        assert result == b"234"

    def test_offset_byte_request(self) -> None:
        # OffsetByteRequest(offset) → data[offset:]
        result = _apply_byte_range(self._DATA, OffsetByteRequest(3))
        assert result == b"3456789"

    def test_suffix_byte_request(self) -> None:
        # SuffixByteRequest(suffix) → data[-suffix:]
        result = _apply_byte_range(self._DATA, SuffixByteRequest(4))
        assert result == b"6789"


# ============================================================================
# 2. ABSZarrV3Store round-trip (CI-safe, dict-backed mock container)
# ============================================================================


class TestABSZarrV3StoreRoundTrip:
    """ABSZarrV3Store end-to-end: write through store, read back, assert correctness.

    No Azurite.  No network.  Uses dict-backed _MockContainerClient.
    """

    @pytest.fixture
    def mock_container(self) -> _MockContainerClient:
        return _MockContainerClient()

    def test_write_then_read_allclose(self, mock_container: _MockContainerClient) -> None:
        """Write a 3-D float32 array via ABSZarrV3Store; read back; assert allclose."""
        prefix = "test/vol.zarr"
        rng = np.random.default_rng(0)
        expected = rng.standard_normal((5, 10, 20)).astype(np.float32)

        # Write path
        store_w = ABSZarrV3Store(mock_container, prefix=prefix)
        root_w = zarr.open_group(store_w, mode="w")
        root_w.create_array("amplitude", data=expected)

        # Read path — new store instance, same dict
        store_r = ABSZarrV3Store(mock_container, prefix=prefix)
        root_r = zarr.open_group(store_r, mode="r")
        result = np.asarray(root_r["amplitude"][:])

        assert result.shape == expected.shape
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_missing_key_returns_none(self, mock_container: _MockContainerClient) -> None:
        """get() for a key absent from the store returns None (not raises)."""
        store = ABSZarrV3Store(mock_container, prefix="empty/prefix")
        result = asyncio.run(
            store.get("zarr.json", default_buffer_prototype())
        )
        assert result is None, "Expected None for missing key, got data"

    def test_with_read_only_returns_read_only_instance(
        self, mock_container: _MockContainerClient
    ) -> None:
        """with_read_only(True) returns a new store with read_only == True."""
        store = ABSZarrV3Store(mock_container, prefix="test/")
        ro = store.with_read_only(True)
        assert ro.read_only is True, "with_read_only(True) must set read_only=True"

    def test_with_read_only_set_raises(self, mock_container: _MockContainerClient) -> None:
        """set() on a read-only store raises — write is rejected."""
        store = ABSZarrV3Store(mock_container, prefix="test/")
        ro = store.with_read_only(True)
        buf = Buffer.from_bytes(b"should-not-write")
        with pytest.raises(ValueError):
            asyncio.run(ro.set("any_key", buf))

    def test_mode_r_open_on_populated_store_reads_correctly(
        self, mock_container: _MockContainerClient
    ) -> None:
        """zarr.open_group(with_read_only(True), mode='r') reads data allclose."""
        prefix = "test/ro.zarr"
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)

        store_w = ABSZarrV3Store(mock_container, prefix=prefix)
        root_w = zarr.open_group(store_w, mode="w")
        root_w.create_array("data", data=expected)

        ro_store = store_w.with_read_only(True)
        root_r = zarr.open_group(ro_store, mode="r")
        result = np.asarray(root_r["data"][:])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_multiple_arrays_preserved(self, mock_container: _MockContainerClient) -> None:
        """Multiple arrays in one group all round-trip correctly."""
        prefix = "multi/test.zarr"
        rng = np.random.default_rng(7)
        a1 = rng.standard_normal((4, 5)).astype(np.float32)
        a2 = rng.integers(0, 2, size=(4, 5)).astype(np.uint8)

        store_w = ABSZarrV3Store(mock_container, prefix=prefix)
        root_w = zarr.open_group(store_w, mode="w")
        root_w.create_array("float_arr", data=a1)
        root_w.create_array("uint8_arr", data=a2)

        store_r = ABSZarrV3Store(mock_container, prefix=prefix)
        root_r = zarr.open_group(store_r, mode="r")

        np.testing.assert_allclose(np.asarray(root_r["float_arr"][:]), a1, rtol=1e-5)
        np.testing.assert_array_equal(np.asarray(root_r["uint8_arr"][:]), a2)


# ============================================================================
# 3. Backend resolver defaults
# ============================================================================


class TestBackendResolverDefaults:
    """Env-var contract: _azure_sources() defaults match the architecture decision."""

    def test_default_backend_is_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEISMIC_DATA_BACKEND", raising=False)
        assert _data_readers._backend() == "local"

    def test_azure_sources_amp_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "DEEPSEISMIC_AMP_CONTAINER",
            "DEEPSEISMIC_AMP_PREFIX",
            "DEEPSEISMIC_FAULT_PROB_CONTAINER",
            "DEEPSEISMIC_FAULT_PROB_PREFIX",
            "DEEPSEISMIC_STICKS_CONTAINER",
            "DEEPSEISMIC_STICKS_PREFIX",
        ):
            monkeypatch.delenv(var, raising=False)
        src = _data_readers._azure_sources()
        assert src.amp_container == "staged"
        assert src.amp_prefix == "volve/synthetic.zarr"
        assert src.prob_container == "results"
        assert src.prob_prefix == "volve/fault_prob.zarr"
        assert src.sticks_container == "raw"
        assert src.sticks_prefix == "volve/interpretations/fault_sticks"

    def test_azure_sources_honours_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEISMIC_AMP_CONTAINER", "custom-staged")
        monkeypatch.setenv("DEEPSEISMIC_AMP_PREFIX", "acme/amp.zarr")
        monkeypatch.setenv("DEEPSEISMIC_STICKS_CONTAINER", "custom-raw")
        src = _data_readers._azure_sources()
        assert src.amp_container == "custom-staged"
        assert src.amp_prefix == "acme/amp.zarr"
        assert src.sticks_container == "custom-raw"


# ============================================================================
# 4. Local backend — missing artifacts → graceful None
# ============================================================================


class TestBackendResolverLocalMissing:
    """Local backend with missing zarr files: graceful degradation, no real data needed."""

    def test_fault_prob_absent_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the local fault_prob zarr is absent, get_fault_prob_slice returns None."""
        monkeypatch.setenv("DEEPSEISMIC_DATA_BACKEND", "local")
        monkeypatch.setenv("DEEPSEISMIC_DATA_DIR", str(tmp_path / "nonexistent_volve"))
        result = _data_readers.get_fault_prob_slice(1050)
        assert result is None, (
            "get_fault_prob_slice must return None when fault_prob.zarr is absent, not raise"
        )


# ============================================================================
# 5. Local backend — real on-disk data (skipif when artifacts absent)
# ============================================================================


@pytest.mark.skipif(
    not _ZARR_AMP.exists(),
    reason="data/volve/staged/synthetic.zarr absent — run scripts/bake_demo_faults.py",
)
class TestBackendResolverLocalPresent:
    """Local backend reads against the real baked zarr (skipped in CI where gitignored)."""

    def test_get_volume_coords_returns_three_arrays(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEISMIC_DATA_BACKEND", "local")
        monkeypatch.delenv("DEEPSEISMIC_DATA_DIR", raising=False)
        inline_arr, xl_arr, twtt_arr = _data_readers.get_volume_coords()
        assert inline_arr.ndim == 1
        assert xl_arr.ndim == 1
        assert twtt_arr.ndim == 1
        assert inline_arr[0] == pytest.approx(1001.0, abs=0.5)

    def test_get_amplitude_slice_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEISMIC_DATA_BACKEND", "local")
        monkeypatch.delenv("DEEPSEISMIC_DATA_DIR", raising=False)
        arr = _data_readers.get_amplitude_slice(1050)
        assert arr.shape == (200, 500), f"Expected (200, 500), got {arr.shape}"
        assert arr.dtype == np.float32

    @pytest.mark.skipif(
        not _ZARR_PROB.exists(),
        reason="data/volve/staged/fault_prob.zarr absent — run scripts/bake_demo_faults.py",
    )
    def test_get_fault_prob_slice_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEISMIC_DATA_BACKEND", "local")
        monkeypatch.delenv("DEEPSEISMIC_DATA_DIR", raising=False)
        arr = _data_readers.get_fault_prob_slice(1050)
        assert arr is not None
        assert arr.shape == (200, 500)
        assert 0.0 <= float(arr.min()) <= float(arr.max()) <= 1.0 + 1e-6


# ============================================================================
# 6. Azure backend resolver — dict-backed mock storage client
# ============================================================================


class TestBackendResolverAzure:
    """Azure backend: monkeypatched _storage_client + dict-backed ABSZarrV3Store."""

    _PREFIX = "volve/synthetic.zarr"
    _CONTAINER = "staged"

    @pytest.fixture
    def amp_mock_container(self) -> _MockContainerClient:
        container = _MockContainerClient()
        _seed_amp_zarr(container, self._PREFIX, n_il=10, n_xl=20, n_s=50)
        return container

    @pytest.fixture
    def mock_client(
        self, amp_mock_container: _MockContainerClient
    ) -> _MockStorageClient:
        return _MockStorageClient(amp_mock_container)

    @pytest.fixture(autouse=True)
    def azure_env(self, monkeypatch: pytest.MonkeyPatch, mock_client: _MockStorageClient) -> None:
        monkeypatch.setenv("DEEPSEISMIC_DATA_BACKEND", "azure")
        monkeypatch.delenv("DEEPSEISMIC_AMP_CONTAINER", raising=False)
        monkeypatch.delenv("DEEPSEISMIC_AMP_PREFIX", raising=False)
        monkeypatch.setattr(_data_readers, "_storage_client", lambda: mock_client)

    def test_get_volume_coords_reads_inline_array(self) -> None:
        """get_volume_coords() with azure backend returns the mock inline array."""
        inline_arr, xl_arr, twtt_arr = _data_readers.get_volume_coords()
        assert inline_arr[0] == pytest.approx(1001.0, abs=0.5)
        assert inline_arr[-1] == pytest.approx(1010.0, abs=0.5)
        assert len(xl_arr) == 20
        assert len(twtt_arr) == 50

    def test_get_amplitude_slice_reads_data(self) -> None:
        """get_amplitude_slice() with azure backend returns (n_xl, n_s) float32 slice."""
        arr = _data_readers.get_amplitude_slice(1001)
        assert arr.shape == (20, 50), f"Expected (20, 50), got {arr.shape}"
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr)), "Amplitude data must be finite"

    def test_resolver_uses_amp_container_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolver passes the correct container/prefix from env vars to open_zarr_store."""
        monkeypatch.setenv("DEEPSEISMIC_AMP_CONTAINER", "custom-staged")
        monkeypatch.setenv("DEEPSEISMIC_AMP_PREFIX", "volve/synthetic.zarr")
        src = _data_readers._azure_sources()
        assert src.amp_container == "custom-staged"
        assert src.amp_prefix == "volve/synthetic.zarr"

    def test_fault_prob_absent_azure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Azure backend: if fault_prob zarr is missing, get_fault_prob_slice returns None."""
        # Default prob_container="results" has no data in our mock → returns None
        result = _data_readers.get_fault_prob_slice(1001)
        assert result is None, (
            "Azure backend: get_fault_prob_slice must return None when store is empty"
        )


# ============================================================================
# 7. Azure fault sticks — canonical coordinate mapping on azure path
# ============================================================================


class TestAzureFaultSticksAzure:
    """Azure fault sticks path: canonical mapping guards run on azure backend too.

    Regression guard: z column is sample index (×4 ms), not raw ms.
    Any regression re-introducing raw-z-as-ms (202–307 ms) fails loudly on ≥800 ms guard.
    """

    _STICKS_PREFIX = "volve/interpretations/fault_sticks"
    _STICKS_CONTAINER = "raw"

    @pytest.fixture
    def sticks_blobs(self) -> dict[str, bytes]:
        """Two synthetic .dat blobs — same content as the CI coordinate-mapping fixture."""
        prefix = self._STICKS_PREFIX
        return {
            f"{prefix}/fault_main_normal.dat": b"45 84 202\n70 104 215\n95 124 227\n",
            f"{prefix}/fault_antithetic.dat": b"0 0 300\n5 5 303\n10 10 307\n",
        }

    @pytest.fixture
    def mock_client(self, sticks_blobs: dict[str, bytes]) -> _MockStorageClient:
        return _MockStorageClient(_MockContainerClient(), sticks_blobs=sticks_blobs)

    @pytest.fixture(autouse=True)
    def azure_env(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: _MockStorageClient
    ) -> None:
        monkeypatch.setenv("DEEPSEISMIC_DATA_BACKEND", "azure")
        for var in ("DEEPSEISMIC_STICKS_CONTAINER", "DEEPSEISMIC_STICKS_PREFIX"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(_data_readers, "_storage_client", lambda: mock_client)

    def test_both_faults_loaded(self) -> None:
        sticks = _data_readers.load_fault_sticks()
        assert "fault_main_normal" in sticks, "fault_main_normal.dat not parsed"
        assert "fault_antithetic" in sticks, "fault_antithetic.dat not parsed"

    def test_twt_not_raw_z_column(self) -> None:
        """TWT values must be ≥800 ms; raw z_sample values (202–307) would be <<800."""
        sticks = _data_readers.load_fault_sticks()
        for name, arr in sticks.items():
            twt = arr[:, 2]
            assert float(twt.min()) >= 800.0, (
                f"{name}: TWT min {twt.min():.1f} ms < 800 ms — "
                "raw z_ms bug: z column must be multiplied by 4.0 (sample index, not true ms)"
            )

    def test_main_fault_first_row_exact_mapping(self) -> None:
        """Pin: dat(45, 84, 202) → abs_il=1046, abs_xl=1984, twt=808.0 ms."""
        sticks = _data_readers.load_fault_sticks()
        row = sticks["fault_main_normal"][0]
        assert row[0] == pytest.approx(1046.0, abs=0.1), f"abs_inline: {row[0]}"
        assert row[1] == pytest.approx(1984.0, abs=0.1), f"abs_crossline: {row[1]}"
        assert row[2] == pytest.approx(808.0, abs=0.1), f"twt_ms: {row[2]}"

    def test_antithetic_fault_twt_band(self) -> None:
        """Antithetic fault TWT: 1200–1228 ms (z_samp 300–307 × 4.0)."""
        sticks = _data_readers.load_fault_sticks()
        twt = sticks["fault_antithetic"][:, 2]
        assert float(twt.min()) == pytest.approx(1200.0, abs=0.5)
        assert float(twt.max()) == pytest.approx(1228.0, abs=0.5)

    def test_missing_sticks_returns_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Azure backend: unreachable sticks prefix returns {} (not raises)."""
        empty_client = _MockStorageClient(_MockContainerClient(), sticks_blobs={})
        monkeypatch.setattr(_data_readers, "_storage_client", lambda: empty_client)
        result = _data_readers.load_fault_sticks()
        assert result == {}, "Empty sticks prefix must return {} not raise"


# ============================================================================
# 8. Azurite integration — real upload_zarr_store + open_zarr_store round-trip
#    Excluded from CI: pytest -m "not integration"
# ============================================================================


@pytest.mark.integration
class TestAzuriteIntegration:
    """Real Azurite round-trip: upload_zarr_store → open_zarr_store → allclose.

    Requires Azurite on localhost:10000.  Automatically skipped when unavailable.
    Run locally with: pytest src/tests/test_viewer/test_data_readers.py -m integration
    """

    _CONTAINER = "staged"
    _PREFIX = "test/integration.zarr"

    @pytest.fixture
    def storage_client(self):
        """StorageClient pointed at local Azurite. Skipped if Azurite not running."""
        import os

        from deepseismic.storage.blob_client import AZURITE_CONNECTION_STRING, StorageClient

        os.environ["STORAGE_CONNECTION_STRING"] = AZURITE_CONNECTION_STRING
        try:
            client = StorageClient()
            # Probe connectivity via list_blobs; raises if Azurite is down
            client.list_blobs(self._CONTAINER, max_results=1)
        except Exception:
            pytest.skip("Azurite not running — skipping integration test")
        client.ensure_containers()
        return client

    def test_upload_then_open_zarr_allclose(
        self, storage_client: Any, tmp_path: Path
    ) -> None:
        """Write a tiny zarr to disk, upload to Azurite, read back via open_zarr_store."""
        rng = np.random.default_rng(99)
        original = rng.standard_normal((4, 8, 16)).astype(np.float32)

        # Write to local store
        local_store = zarr.storage.LocalStore(str(tmp_path / "local.zarr"))
        root_local = zarr.open_group(local_store, mode="w")
        root_local.create_array("amplitude", data=original)

        # Upload to Azurite
        storage_client.upload_zarr_store(local_store, self._CONTAINER, self._PREFIX)

        # Read back via ABSZarrV3Store
        remote_store = storage_client.open_zarr_store(self._CONTAINER, self._PREFIX)
        root_remote = zarr.open_group(remote_store, mode="r")
        result = np.asarray(root_remote["amplitude"][:])

        assert result.shape == original.shape
        np.testing.assert_allclose(result, original, rtol=1e-5)
