"""Focused tests for issue #26 — _resolve_run_id prefix resolution.

Scenarios covered
-----------------
(a) Prefix resolves to full id via index.json when list_blobs returns empty / raises.
(b) Ambiguous prefix (>1 match in index) → 409.
(c) Unknown prefix → 404.
(d) Pending manifest is written at submit time so a freshly-submitted run
    resolves by full id immediately (without completing inference).
(e) Full UUID resolves via exact manifest download (step 2) — baseline.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import deepseismic.api.dependencies as _dep
import deepseismic.api.main as _main_mod
from deepseismic.api.main import app
from deepseismic.api.routes.interpretation import (
    _CATALOG_INDEX_BLOB,
    _interp_jobs,
    _resolve_run_id,
)


# ---------------------------------------------------------------------------
# Minimal in-memory storage that supports exact download + configurable list
# ---------------------------------------------------------------------------


class _IndexStorageClient:
    """Storage client that serves blobs from a dict and makes list_blobs controllable."""

    def __init__(self, *, list_blobs_raises: bool = False, list_blobs_returns: list[str] | None = None) -> None:
        self._store: dict[str, dict[str, bytes]] = {}
        self._list_blobs_raises = list_blobs_raises
        self._list_blobs_returns = list_blobs_returns  # overrides normal listing when set

    def upload_blob(
        self,
        container: str,
        blob_path: str,
        data: bytes | Any,
        *,
        overwrite: bool = True,
        metadata: dict | None = None,
    ) -> None:
        self._store.setdefault(container, {})[blob_path] = (
            bytes(data) if isinstance(data, (bytes, bytearray)) else data.read()
        )

    def download_blob(self, container: str, blob_path: str) -> bytes:
        try:
            return self._store[container][blob_path]
        except KeyError:
            raise FileNotFoundError(f"{container}/{blob_path}") from None

    def list_blobs(
        self, container: str, prefix: str = "", *, max_results: int | None = None
    ) -> list[str]:
        if self._list_blobs_raises:
            raise OSError("list_blobs not supported on HNS container (simulated)")
        if self._list_blobs_returns is not None:
            return self._list_blobs_returns
        names = [k for k in self._store.get(container, {}) if k.startswith(prefix)]
        return names[:max_results] if max_results is not None else names

    def ensure_containers(self) -> None:
        pass

    def blob_exists(self, container: str, blob_path: str) -> bool:
        return blob_path in self._store.get(container, {})


def _make_storage_with_index(*run_ids: str, list_blobs_raises: bool = False) -> _IndexStorageClient:
    """Return a storage client pre-populated with an index.json for *run_ids*."""
    client = _IndexStorageClient(list_blobs_raises=list_blobs_raises)
    client.upload_blob(
        "catalog",
        _CATALOG_INDEX_BLOB,
        json.dumps(list(run_ids)).encode(),
    )
    return client


def _patch_real_mode(monkeypatch: pytest.MonkeyPatch, storage: Any) -> None:
    monkeypatch.delenv("DEEPSEISMIC_MOCK_MODE", raising=False)
    monkeypatch.setattr(_dep, "_build_storage_client", lambda: storage)
    monkeypatch.setattr(_main_mod, "_build_storage_client", lambda: storage)


# ---------------------------------------------------------------------------
# (a) Prefix resolves via index.json when list_blobs returns empty or raises
# ---------------------------------------------------------------------------


class TestPrefixResolvesViaIndex:
    def test_prefix_resolves_when_list_blobs_returns_empty(self) -> None:
        """_resolve_run_id resolves prefix via index.json when list_blobs returns []."""
        full_id = str(uuid.uuid4())
        prefix = full_id[:8]
        storage = _make_storage_with_index(full_id, list_blobs_raises=False)
        # Override list_blobs to return empty (simulates HNS listing nothing)
        storage._list_blobs_returns = []

        result = _resolve_run_id(prefix, storage)
        assert result == full_id

    def test_prefix_resolves_when_list_blobs_raises(self) -> None:
        """_resolve_run_id resolves prefix via index.json even when list_blobs raises."""
        full_id = str(uuid.uuid4())
        prefix = full_id[:8]
        storage = _make_storage_with_index(full_id, list_blobs_raises=True)

        result = _resolve_run_id(prefix, storage)
        assert result == full_id

    def test_full_uuid_resolves_via_exact_manifest_download(self) -> None:
        """Step 2: full UUID resolves via exact catalog manifest download."""
        full_id = str(uuid.uuid4())
        storage = _IndexStorageClient(list_blobs_raises=True)
        manifest = {"run_id": full_id, "status": "pending", "survey_id": "s1"}
        storage.upload_blob(
            "catalog",
            f"interpretation/{full_id}/status.json",
            json.dumps(manifest).encode(),
        )

        result = _resolve_run_id(full_id, storage)
        assert result == full_id


# ---------------------------------------------------------------------------
# (b) Ambiguous prefix → 409
# ---------------------------------------------------------------------------


class TestAmbiguousPrefix:
    def test_ambiguous_prefix_raises_409(self) -> None:
        """Two run ids sharing a prefix → HTTPException 409."""
        from fastapi import HTTPException

        id1 = "aaaabbbb-0000-0000-0000-000000000001"
        id2 = "aaaabbbb-0000-0000-0000-000000000002"
        storage = _make_storage_with_index(id1, id2, list_blobs_raises=True)

        with pytest.raises(HTTPException) as exc_info:
            _resolve_run_id("aaaabbbb", storage)

        assert exc_info.value.status_code == 409
        assert "ambiguous" in exc_info.value.detail.lower()

    def test_ambiguous_prefix_via_list_blobs_fallback(self) -> None:
        """Ambiguous matches from list_blobs fallback also yield 409."""
        from fastapi import HTTPException

        id1 = "ccccdddd-1111-0000-0000-000000000001"
        id2 = "ccccdddd-1111-0000-0000-000000000002"
        storage = _IndexStorageClient(list_blobs_raises=False)
        # No index — rely on list_blobs fallback
        storage.upload_blob("catalog", f"interpretation/{id1}/status.json", b"{}")
        storage.upload_blob("catalog", f"interpretation/{id2}/status.json", b"{}")

        with pytest.raises(HTTPException) as exc_info:
            _resolve_run_id("ccccdddd", storage)

        assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# (c) Unknown prefix → 404
# ---------------------------------------------------------------------------


class TestUnknownPrefix:
    def test_unknown_prefix_raises_404(self) -> None:
        """Prefix with no match → HTTPException 404."""
        from fastapi import HTTPException

        storage = _IndexStorageClient(list_blobs_raises=True)  # empty + no list
        with pytest.raises(HTTPException) as exc_info:
            _resolve_run_id("deadbeef", storage)

        assert exc_info.value.status_code == 404

    def test_unknown_prefix_with_empty_index_raises_404(self) -> None:
        """Empty index + list_blobs raises → 404."""
        from fastapi import HTTPException

        storage = _make_storage_with_index(list_blobs_raises=True)  # index exists but empty
        with pytest.raises(HTTPException) as exc_info:
            _resolve_run_id("00000000", storage)

        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# (d) Pending manifest written at submit time — freshly submitted run resolves
# ---------------------------------------------------------------------------


class TestPendingManifestAtSubmit:
    def test_pending_manifest_written_before_background_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Submitting a run writes status.json=pending BEFORE the background task fires."""
        storage = _IndexStorageClient()
        _patch_real_mode(monkeypatch, storage)

        # Intercept background_tasks to confirm no task has run yet when we check
        from deepseismic.api.routes import interpretation as interp_mod

        submitted_run_id: list[str] = []

        original_add_task = None

        def _fake_add_task(fn: Any, *args: Any, **kwargs: Any) -> None:
            # Record the run_id but DON'T actually run the task
            submitted_run_id.append(args[0])

        with TestClient(app) as client:
            # Monkeypatch BackgroundTasks.add_task on the request level via the router
            # We use the TestClient but intercept the background task
            import deepseismic.api.routes.interpretation as _i_mod
            original_run = _i_mod._run_fault_detection

            called = []

            def _noop_run(*a: Any, **kw: Any) -> None:
                called.append(True)

            monkeypatch.setattr(_i_mod, "_run_fault_detection", _noop_run)

            resp = client.post(
                "/api/interpretation/fault-detection",
                json={
                    "survey_id": "volve-st10010",
                    "checkpoint_blob": "checkpoints/test.pt",
                    "patch_size": [16, 16, 16],
                },
            )

        assert resp.status_code == 202, resp.text
        run_id = resp.json()["run_id"]

        # Pending manifest must exist in storage
        manifest_bytes = storage.download_blob(
            "catalog", f"interpretation/{run_id}/status.json"
        )
        manifest = json.loads(manifest_bytes)
        assert manifest["run_id"] == run_id
        assert manifest["status"] == "pending"
        assert manifest["survey_id"] == "volve-st10010"

    def test_pending_manifest_enables_full_id_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After submit, the full run id resolves via step-2 exact manifest download."""
        storage = _IndexStorageClient()
        _patch_real_mode(monkeypatch, storage)

        import deepseismic.api.routes.interpretation as _i_mod

        monkeypatch.setattr(_i_mod, "_run_fault_detection", lambda *a, **kw: None)

        with TestClient(app) as client:
            resp = client.post(
                "/api/interpretation/fault-detection",
                json={
                    "survey_id": "volve-st10010",
                    "checkpoint_blob": "checkpoints/test.pt",
                    "patch_size": [16, 16, 16],
                },
            )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        # Full id must resolve immediately (step 2 — exact manifest)
        resolved = _resolve_run_id(run_id, storage)
        assert resolved == run_id

    def test_catalog_index_updated_at_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Submitting a run appends the run_id to catalog/interpretation/index.json."""
        storage = _IndexStorageClient()
        _patch_real_mode(monkeypatch, storage)

        import deepseismic.api.routes.interpretation as _i_mod

        monkeypatch.setattr(_i_mod, "_run_fault_detection", lambda *a, **kw: None)

        with TestClient(app) as client:
            resp = client.post(
                "/api/interpretation/fault-detection",
                json={
                    "survey_id": "volve-st10010",
                    "checkpoint_blob": "checkpoints/test.pt",
                    "patch_size": [16, 16, 16],
                },
            )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        index_bytes = storage.download_blob("catalog", _CATALOG_INDEX_BLOB)
        index: list[str] = json.loads(index_bytes)
        assert run_id in index

    def test_prefix_resolves_via_index_after_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After submit, the 8-char prefix resolves to the full run id via index.json."""
        storage = _IndexStorageClient(list_blobs_raises=True)
        _patch_real_mode(monkeypatch, storage)

        import deepseismic.api.routes.interpretation as _i_mod

        monkeypatch.setattr(_i_mod, "_run_fault_detection", lambda *a, **kw: None)

        with TestClient(app) as client:
            resp = client.post(
                "/api/interpretation/fault-detection",
                json={
                    "survey_id": "volve-st10010",
                    "checkpoint_blob": "checkpoints/test.pt",
                    "patch_size": [16, 16, 16],
                },
            )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        prefix = run_id[:8]

        # Remove from in-memory registry to force catalog-only resolution
        _interp_jobs.pop(run_id, None)

        resolved = _resolve_run_id(prefix, storage)
        assert resolved == run_id


# ---------------------------------------------------------------------------
# (e) list_blobs failure is now logged, not silently swallowed
# ---------------------------------------------------------------------------


class TestListBlobsFailureLogged:
    def test_list_blobs_failure_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """list_blobs failure in step 3b must emit a WARNING — not be silently swallowed."""
        import logging

        full_id = str(uuid.uuid4())
        # No index, list_blobs raises — should log warning then 404
        storage = _IndexStorageClient(list_blobs_raises=True)
        from fastapi import HTTPException

        with caplog.at_level(logging.WARNING, logger="deepseismic.api.routes.interpretation"):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_run_id(full_id[:8], storage)

        assert exc_info.value.status_code == 404
        # At least one WARNING mentioning list_blobs should have been emitted
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("list_blobs" in m for m in warning_msgs), (
            f"Expected a WARNING mentioning 'list_blobs'; got: {warning_msgs}"
        )
