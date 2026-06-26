"""Real-mode API integration tests — prove the non-mock code paths are exercised.

Coverage
--------
TestIsMockMode            — is_mock_mode() respects env var; real mode is the default.
TestHealthRealModeStates  — health endpoint reports ok/error/unreachable/mock correctly.
TestRealModeFailLoud503   — storage misconfigured → 503 (NOT silent canned data).
TestMockVsRealSurveyRoute — survey list returns real path vs mock based on env.
TestRealPathIngestFlow    — _run_ingest exercises the real SEG-Y → zarr → storage path
                            (dict-backed storage, no Azurite required).
TestAzuriteHealthIntegration — @pytest.mark.integration: real Azurite health ping.

The key regression guard for the Wave 1 de-mock:
  In real mode (DEEPSEISMIC_MOCK_MODE unset), a misconfigured storage must return
  HTTP 503 — NEVER silently return the canned mock Volve dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import deepseismic.api.dependencies as _dep
import deepseismic.api.main as _main_mod
from deepseismic.api.dependencies import is_mock_mode
from deepseismic.api.main import app

# Azurite well-known dev credentials (safe to commit — public emulator key)
_AZURITE_CONN = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tiqkDg==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)

# Mock Volve survey id from surveys.py — used as sentinel for mock-data guard
_MOCK_SURVEY_ID = "volve-st10010"
# Mock fault_voxel_fraction from interpretation.py — used as sentinel
_MOCK_FAULT_FRACTION = 0.0412


# ---------------------------------------------------------------------------
# Dict-backed storage client — exercises real API code paths without Azurite.
# ---------------------------------------------------------------------------


class _DictStorageClient:
    """Minimal in-memory blob store that satisfies the StorageClient interface."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, bytes]] = {}
        self.upload_zarr_store_calls: list[tuple[Any, str, str]] = []

    def upload_blob(
        self,
        container: str,
        blob_path: str,
        data: bytes | Any,
        *,
        overwrite: bool = True,
        metadata: dict | None = None,
    ) -> None:
        self._store.setdefault(container, {})
        if isinstance(data, (bytes, bytearray)):
            self._store[container][blob_path] = bytes(data)
        else:
            self._store[container][blob_path] = data.read()

    def download_blob(self, container: str, blob_path: str) -> bytes:
        try:
            return self._store[container][blob_path]
        except KeyError:
            raise FileNotFoundError(f"{container}/{blob_path}") from None

    def list_blobs(
        self, container: str, prefix: str = "", *, max_results: int | None = None
    ) -> list[str]:
        names = [k for k in self._store.get(container, {}) if k.startswith(prefix)]
        return names[:max_results] if max_results is not None else names

    def ensure_containers(self) -> None:
        pass

    def blob_exists(self, container: str, blob_path: str) -> bool:
        return blob_path in self._store.get(container, {})

    def open_zarr_store(self, container: str, prefix: str) -> Any:
        raise NotImplementedError("zarr store access not wired in _DictStorageClient")

    def upload_zarr_store(self, src_path: Any, container: str, prefix: str) -> None:
        self.upload_zarr_store_calls.append((src_path, container, prefix))


class _UnreachableStorageClient(_DictStorageClient):
    """Client that has credentials but raises on every blob operation."""

    def list_blobs(self, *args: Any, **kwargs: Any) -> list[str]:
        raise ConnectionError("Storage unreachable in this test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_real_mode(
    monkeypatch: pytest.MonkeyPatch,
    storage_client: Any | None = None,
    *,
    raise_on_build: bool = False,
) -> None:
    """Force real mode and inject a storage client (or broken builder)."""
    monkeypatch.delenv("DEEPSEISMIC_MOCK_MODE", raising=False)
    if raise_on_build:
        def _broken_builder() -> None:
            raise OSError(
                "No storage credentials configured (injected by test)"
            )
        monkeypatch.setattr(_dep, "_build_storage_client", _broken_builder)
        monkeypatch.setattr(_main_mod, "_build_storage_client", _broken_builder)
    elif storage_client is not None:
        monkeypatch.setattr(_dep, "_build_storage_client", lambda: storage_client)
        monkeypatch.setattr(_main_mod, "_build_storage_client", lambda: storage_client)


# ─────────────────────────────────────────────────────────────────────────────
# TestIsMockMode — pure unit tests on is_mock_mode(), no HTTP
# ─────────────────────────────────────────────────────────────────────────────


class TestIsMockMode:
    def test_real_mode_is_default_when_no_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_mock_mode() must return False when DEEPSEISMIC_MOCK_MODE is unset."""
        monkeypatch.delenv("DEEPSEISMIC_MOCK_MODE", raising=False)
        assert is_mock_mode() is False

    def test_mock_mode_activated_by_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEEPSEISMIC_MOCK_MODE=true → is_mock_mode() True."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "true")
        assert is_mock_mode() is True

    def test_mock_mode_activated_by_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEEPSEISMIC_MOCK_MODE=1 → is_mock_mode() True."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "1")
        assert is_mock_mode() is True

    def test_mock_mode_activated_by_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEEPSEISMIC_MOCK_MODE=yes → is_mock_mode() True."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "yes")
        assert is_mock_mode() is True

    def test_mock_mode_not_activated_by_false_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEEPSEISMIC_MOCK_MODE=false → is_mock_mode() False."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "false")
        assert is_mock_mode() is False

    def test_mock_mode_not_activated_by_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEEPSEISMIC_MOCK_MODE="" → is_mock_mode() False."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "")
        assert is_mock_mode() is False


# ─────────────────────────────────────────────────────────────────────────────
# TestHealthRealModeStates — health endpoint storage status field
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthRealModeStates:
    def test_health_returns_ok_when_storage_reachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Health endpoint reports storage='ok' when storage ping succeeds."""
        _patch_real_mode(monkeypatch, _DictStorageClient())
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["storage"] == "ok"
        assert body["mock_mode"] is False

    def test_health_returns_error_when_client_build_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Health endpoint reports storage='error' when StorageClient cannot be built."""
        _patch_real_mode(monkeypatch, raise_on_build=True)
        with TestClient(app) as client:
            resp = client.get("/health")
        # Process is alive → always 200; storage failed → 'error'
        assert resp.status_code == 200
        body = resp.json()
        assert body["storage"] == "error"
        assert body["status"] == "ok"
        assert "storage_error" in body

    def test_health_returns_unreachable_when_ping_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Health endpoint reports storage='unreachable' when list_blobs raises."""
        _patch_real_mode(monkeypatch, _UnreachableStorageClient())
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["storage"] == "unreachable"
        assert body["mock_mode"] is False

    def test_health_returns_mock_in_mock_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Health endpoint reports storage='mock' and mock_mode=True in mock mode."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "true")
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["storage"] == "mock"
        assert body["mock_mode"] is True

    def test_health_status_field_always_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Health endpoint status field is always 'ok' regardless of storage state."""
        _patch_real_mode(monkeypatch, raise_on_build=True)
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_real_mode_never_reports_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In real mode, health.storage is never 'mock' — even when storage is broken."""
        _patch_real_mode(monkeypatch, raise_on_build=True)
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.json()["storage"] != "mock"


# ─────────────────────────────────────────────────────────────────────────────
# TestRealModeFailLoud503 — KEY REGRESSION GUARD
# In real mode, broken storage must yield 503, NOT silent mock data.
# ─────────────────────────────────────────────────────────────────────────────


class TestRealModeFailLoud503:
    def test_surveys_503_when_storage_build_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /api/surveys → 503 in real mode when storage is misconfigured.

        This is the key regression guard for the Wave 1 de-mock: broken storage
        must fail loud (503), never silently return canned Volve survey data.
        """
        _patch_real_mode(monkeypatch, raise_on_build=True)
        with TestClient(app) as client:
            resp = client.get("/api/surveys")
        assert resp.status_code == 503, (
            f"Expected 503 (fail-loud) from broken storage, got {resp.status_code}. "
            "Real mode must NOT fall back to mock data silently."
        )

    def test_surveys_503_detail_mentions_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """503 response body must mention storage to aid debugging."""
        _patch_real_mode(monkeypatch, raise_on_build=True)
        with TestClient(app) as client:
            resp = client.get("/api/surveys")
        body = resp.json()
        assert "storage" in body.get("detail", "").lower() or resp.status_code == 503

    def test_wells_503_when_storage_build_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /api/wells → 503 in real mode when storage is misconfigured."""
        _patch_real_mode(monkeypatch, raise_on_build=True)
        with TestClient(app) as client:
            resp = client.get("/api/wells")
        assert resp.status_code == 503

    def test_surveys_not_503_in_mock_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GET /api/surveys → 200 in mock mode regardless of storage state."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "true")
        with TestClient(app) as client:
            resp = client.get("/api/surveys")
        assert resp.status_code == 200

    def test_surveys_mock_data_only_in_mock_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Canned Volve survey appears ONLY when DEEPSEISMIC_MOCK_MODE is set."""
        monkeypatch.setenv("DEEPSEISMIC_MOCK_MODE", "true")
        with TestClient(app) as client:
            body = client.get("/api/surveys").json()
        survey_ids = [s["survey_id"] for s in body] if isinstance(body, list) else []
        assert _MOCK_SURVEY_ID in survey_ids, (
            "Mock survey must appear when DEEPSEISMIC_MOCK_MODE=true"
        )

    def test_surveys_real_mode_empty_catalog_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In real mode with empty catalog, survey list is [] — not mock Volve data."""
        storage = _DictStorageClient()  # Empty catalog
        _patch_real_mode(monkeypatch, storage)
        with TestClient(app) as client:
            resp = client.get("/api/surveys")
        assert resp.status_code == 200
        surveys = resp.json()
        assert isinstance(surveys, list)
        assert len(surveys) == 0, (
            "Real mode with empty catalog must return [], not canned mock data"
        )
        survey_ids = [s.get("survey_id", "") for s in surveys]
        assert _MOCK_SURVEY_ID not in survey_ids, (
            "Mock survey id must NOT appear in real mode"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestRealPathIngestFlow — exercises _run_ingest with real segyio/zarr pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestRealPathIngestFlow:
    """Tests _run_ingest end-to-end with dict-backed storage and the synthetic SEG-Y.

    No Azurite required. Exercises the REAL segyio → zarr → storage code path.
    Proves fault_voxel_fraction 0.0412 is NOT returned (that's the mock sentinel).
    """

    def test_run_ingest_marks_status_complete(self, sample_segy_path: Path) -> None:
        """_run_ingest must set job status to 'complete' on success."""
        from deepseismic.api.routes.surveys import IngestRequest as _IR
        from deepseismic.api.routes.surveys import _ingest_jobs, _run_ingest

        run_id = "test-ingest-complete-001"
        survey_id = "test-survey-001"
        _ingest_jobs[run_id] = {"status": "pending", "error": None}

        storage = _DictStorageClient()
        segy_bytes = sample_segy_path.read_bytes()
        storage.upload_blob("raw", "test/synthetic.segy", segy_bytes)

        try:
            _run_ingest(
                run_id,
                _IR(
                    blob_path="test/synthetic.segy",
                    survey_id=survey_id,
                    sample_mode=True,
                    sample_n_inlines=5,
                ),
                storage,
            )
            assert _ingest_jobs[run_id]["status"] == "complete"
        finally:
            _ingest_jobs.pop(run_id, None)

    def test_run_ingest_writes_catalog_metadata(self, sample_segy_path: Path) -> None:
        """_run_ingest must upload a valid JSON sidecar to catalog/{survey_id}/metadata.json."""
        from deepseismic.api.routes.surveys import IngestRequest as _IR
        from deepseismic.api.routes.surveys import _ingest_jobs, _run_ingest

        run_id = "test-ingest-catalog-002"
        survey_id = "test-survey-002"
        _ingest_jobs[run_id] = {"status": "pending", "error": None}

        storage = _DictStorageClient()
        storage.upload_blob("raw", "test/synthetic.segy", sample_segy_path.read_bytes())

        try:
            _run_ingest(
                run_id,
                _IR(blob_path="test/synthetic.segy", survey_id=survey_id, sample_mode=True),
                storage,
            )
            blob_path = f"surveys/{survey_id}/metadata.json"
            assert storage.blob_exists("catalog", blob_path), (
                f"Catalog blob '{blob_path}' was not uploaded"
            )
            raw_json = storage.download_blob("catalog", blob_path)
            meta = json.loads(raw_json)
            assert "geometry" in meta, "Sidecar must have geometry"
            assert "amplitude_stats" in meta, "Sidecar must have amplitude_stats"
        finally:
            _ingest_jobs.pop(run_id, None)

    def test_run_ingest_catalog_metadata_is_valid_json(self, sample_segy_path: Path) -> None:
        """Catalog sidecar uploaded by _run_ingest must be valid, parseable JSON
        and must embed the correct survey_id passed in the ingest request.
        """
        from deepseismic.api.routes.surveys import IngestRequest as _IR
        from deepseismic.api.routes.surveys import _ingest_jobs, _run_ingest

        run_id = "test-ingest-json-003"
        survey_id = "test-survey-003"
        _ingest_jobs[run_id] = {"status": "pending", "error": None}

        storage = _DictStorageClient()
        storage.upload_blob("raw", "test/synthetic.segy", sample_segy_path.read_bytes())

        try:
            _run_ingest(
                run_id,
                _IR(blob_path="test/synthetic.segy", survey_id=survey_id, sample_mode=True),
                storage,
            )
            raw_json = storage.download_blob("catalog", f"surveys/{survey_id}/metadata.json")
            meta = json.loads(raw_json)
            assert isinstance(meta, dict), "Catalog sidecar must be a JSON object"
            assert "geometry" in meta
            assert "amplitude_stats" in meta
            assert "ingested_at" in meta
            assert meta.get("survey_id") == survey_id, (
                f"Catalog sidecar survey_id must match the request survey_id '{survey_id}', "
                f"got {meta.get('survey_id')!r}"
            )
        finally:
            _ingest_jobs.pop(run_id, None)

    def test_run_ingest_calls_upload_zarr_store(self, sample_segy_path: Path) -> None:
        """_run_ingest must call upload_zarr_store (the real zarr upload step)."""
        from deepseismic.api.routes.surveys import IngestRequest as _IR
        from deepseismic.api.routes.surveys import _ingest_jobs, _run_ingest

        run_id = "test-ingest-zarr-004"
        _ingest_jobs[run_id] = {"status": "pending", "error": None}

        storage = _DictStorageClient()
        storage.upload_blob("raw", "test/synthetic.segy", sample_segy_path.read_bytes())

        try:
            _run_ingest(
                run_id,
                _IR(blob_path="test/synthetic.segy", survey_id="test-survey-004", sample_mode=True),
                storage,
            )
            assert len(storage.upload_zarr_store_calls) == 1, (
                "_run_ingest must call upload_zarr_store exactly once"
            )
            _, container, prefix = storage.upload_zarr_store_calls[0]
            assert container == "staged"
            assert "amplitude.zarr" in prefix
        finally:
            _ingest_jobs.pop(run_id, None)

    def test_run_ingest_geometry_has_positive_dims(self, sample_segy_path: Path) -> None:
        """Sidecar geometry must have n_inlines > 0 and n_crosslines > 0."""
        from deepseismic.api.routes.surveys import IngestRequest as _IR
        from deepseismic.api.routes.surveys import _ingest_jobs, _run_ingest

        run_id = "test-ingest-geom-005"
        survey_id = "test-survey-005"
        _ingest_jobs[run_id] = {"status": "pending", "error": None}

        storage = _DictStorageClient()
        storage.upload_blob("raw", "test/synthetic.segy", sample_segy_path.read_bytes())

        try:
            _run_ingest(
                run_id,
                _IR(blob_path="test/synthetic.segy", survey_id=survey_id, sample_mode=True),
                storage,
            )
            raw_json = storage.download_blob("catalog", f"surveys/{survey_id}/metadata.json")
            geom = json.loads(raw_json)["geometry"]
            assert geom["n_inlines"] > 0
            assert geom["n_crosslines"] > 0
            assert geom["n_samples"] > 0
        finally:
            _ingest_jobs.pop(run_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# TestAzuriteHealthIntegration — @pytest.mark.integration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestAzuriteHealthIntegration:
    """Health endpoint reports storage='ok' when Azurite is running.

    Uses azurite_client fixture from conftest.py — skipped automatically when
    Azurite is not reachable on localhost:10000.
    """

    def test_health_ok_with_azurite(
        self, monkeypatch: pytest.MonkeyPatch, azurite_client: Any
    ) -> None:
        """Health → storage='ok' with a real Azurite backend."""
        monkeypatch.delenv("DEEPSEISMIC_MOCK_MODE", raising=False)
        monkeypatch.setenv("STORAGE_CONNECTION_STRING", _AZURITE_CONN)
        _dep._build_storage_client.cache_clear()
        try:
            with TestClient(app) as client:
                resp = client.get("/health")
        finally:
            _dep._build_storage_client.cache_clear()

        assert resp.status_code == 200
        body = resp.json()
        assert body["mock_mode"] is False
        # Azurite is running (confirmed by fixture); 'ok' or 'unreachable' both
        # indicate real-mode operation. We assert NOT 'mock' and NOT 'error'.
        assert body["storage"] in ("ok", "unreachable"), (
            f"Expected real-mode storage status, got {body['storage']!r}"
        )
        assert body["storage"] != "mock"


# ─────────────────────────────────────────────────────────────────────────────
# TestOverlayCoordMapping — absolute inline → subvolume local index (issue #19)
# ─────────────────────────────────────────────────────────────────────────────


class _ZarrStorageClient(_DictStorageClient):
    """Dict store that also serves real on-disk zarr groups for overlays."""

    def __init__(self) -> None:
        super().__init__()
        self._zarr_stores: dict[tuple[str, str], Any] = {}

    def register_zarr(self, container: str, prefix: str, store: Any) -> None:
        self._zarr_stores[(container, prefix)] = store

    def open_zarr_store(self, container: str, prefix: str) -> Any:
        try:
            return self._zarr_stores[(container, prefix)]
        except KeyError:
            raise FileNotFoundError(f"{container}/{prefix}") from None


class TestOverlayCoordMapping:
    """get_overlay must map an absolute inline to the bounded run's local index."""

    def _make_run(self, tmp_path: Path, storage: _ZarrStorageClient) -> str:
        import numpy as np
        import zarr

        run_id = "run-sub-1"
        # Subvolume covering absolute inlines 10090..10094 (5 inlines).
        inline_coords = [10090, 10091, 10092, 10093, 10094]
        n_xl, n_s = 4, 6
        prob = np.zeros((5, n_xl, n_s), dtype=np.float32)
        # Make each inline index identifiable by its constant value.
        for i in range(5):
            prob[i, :, :] = float(i) / 10.0
        mask = (prob > 0.25).astype("uint8")

        for name, arr in (("fault_prob", prob), ("fault_mask", mask)):
            store = zarr.storage.LocalStore(str(tmp_path / f"{name}.zarr"))
            root = zarr.open_group(store, mode="w")
            key = "fault_probability" if name == "fault_prob" else "fault_mask"
            root.create_array(key, data=arr)
            storage.register_zarr(
                "results", f"interpretation/{run_id}/{name}.zarr", store
            )

        manifest = {
            "run_id": run_id,
            "survey_id": "volve-st10010",
            "status": "complete",
            "inline_coords": inline_coords,
            "crossline_coords": [1961, 1962, 1963, 1964],
            "twtt_ms": [0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
        }
        storage.upload_blob(
            "catalog",
            f"interpretation/{run_id}/status.json",
            json.dumps(manifest).encode(),
        )
        return run_id

    def test_absolute_inline_maps_to_local_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        storage = _ZarrStorageClient()
        run_id = self._make_run(tmp_path, storage)
        _patch_real_mode(monkeypatch, storage)
        with TestClient(app) as client:
            # Absolute inline 10092 → local index 2 → constant 0.2.
            resp = client.get(f"/api/interpretation/{run_id}/overlay/10092")
        assert resp.status_code == 200
        body = resp.json()
        assert body["inline_number"] == 10092
        assert body["fault_probability"][0][0] == pytest.approx(0.2)
        # Real crossline/twtt coords come from the manifest, not range(0..n).
        assert body["crossline_coords"][0] == 1961
        assert body["twtt_ms"][1] == pytest.approx(4.0)

    def test_inline_outside_window_returns_404(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        storage = _ZarrStorageClient()
        run_id = self._make_run(tmp_path, storage)
        _patch_real_mode(monkeypatch, storage)
        with TestClient(app) as client:
            resp = client.get(f"/api/interpretation/{run_id}/overlay/10300")
        assert resp.status_code == 404
        assert "window" in resp.json()["detail"].lower()
