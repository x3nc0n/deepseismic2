"""Shared pytest fixtures for the deepseismic2 test suite.

Fixtures are grouped by concern:
  - SEG-Y synthetic file generation (segyio)
  - Zarr volume / fault-label arrays
  - Temporary Zarr store
  - Storage client mocks and Azurite integration
  - LLM / agent mock responses
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr
import zarr.storage

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_N_INLINES = 5
_N_CROSSLINES = 5
_N_SAMPLES = 100

# Azurite well-known dev credentials (safe to commit)
_AZURITE_CONN = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tiqkDg==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


# ─────────────────────────────────────────────────────────────────────────────
# SEG-Y
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_segy_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny synthetic SEG-Y: 5 inlines × 5 crosslines × 100 samples @ 2 ms."""
    import segyio

    inlines = list(range(1, _N_INLINES + 1))
    crosslines = list(range(1, _N_CROSSLINES + 1))

    tmp_dir = tmp_path_factory.mktemp("segy")
    segy_path = tmp_dir / "synthetic.segy"

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = segyio.SegySampleFormat.IBM_FLOAT_4_BYTE
    spec.samples = np.arange(_N_SAMPLES, dtype=np.float32)
    spec.ilines = inlines
    spec.xlines = crosslines

    rng = np.random.default_rng(seed=0)

    with segyio.create(str(segy_path), spec) as f:
        f.bin.update(
            tsort=segyio.TraceSortingFormat.INLINE_SORTING,
            hdt=2000,  # 2 ms sample interval
            dto=2000,
        )
        for il_idx, il in enumerate(inlines):
            for xl_idx, xl in enumerate(crosslines):
                tr_idx = il_idx * len(crosslines) + xl_idx
                f.header[tr_idx] = {
                    segyio.TraceField.INLINE_3D: il,
                    segyio.TraceField.CROSSLINE_3D: xl,
                    segyio.TraceField.CDP_X: il * 100,
                    segyio.TraceField.CDP_Y: xl * 100,
                }
                f.trace[tr_idx] = rng.standard_normal(_N_SAMPLES).astype(np.float32)

    return segy_path


# ─────────────────────────────────────────────────────────────────────────────
# Volume / label arrays
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_zarr_volume() -> np.ndarray:
    """Small seismic volume array (5 × 5 × 100) as float32."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((_N_INLINES, _N_CROSSLINES, _N_SAMPLES)).astype(np.float32)


@pytest.fixture(scope="session")
def sample_fault_labels() -> np.ndarray:
    """Binary fault mask matching sample_zarr_volume dimensions (5 × 5 × 100)."""
    rng = np.random.default_rng(seed=7)
    return (rng.random((_N_INLINES, _N_CROSSLINES, _N_SAMPLES)) > 0.85).astype(np.uint8)


@pytest.fixture
def tmp_zarr_store(tmp_path: Path) -> zarr.storage.LocalStore:
    """Function-scoped temporary Zarr LocalStore (zarr v3 compatible)."""
    return zarr.storage.LocalStore(str(tmp_path / "test.zarr"))


# ─────────────────────────────────────────────────────────────────────────────
# Storage mocks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_storage_client() -> MagicMock:
    """Fully mocked storage client — no Azure calls whatsoever."""
    client = MagicMock(name="BlobStorageClient")
    client.upload_blob.return_value = MagicMock(etag="mock-etag-abc123")
    client.download_blob.return_value = b"fake-blob-content"
    client.list_blobs.return_value = []
    client.container_name = "mock-container"
    return client


@pytest.fixture
def azurite_client():
    """Real Azurite-backed BlobServiceClient.

    Requires Azurite to be running on localhost:10000. Skipped automatically
    when Azurite is not reachable — mark tests using this fixture with
    ``@pytest.mark.integration``.
    """
    pytest.importorskip("azure.storage.blob")
    from azure.storage.blob import BlobServiceClient

    conn_str = os.environ.get("AZURITE_CONNECTION_STRING", _AZURITE_CONN)
    try:
        service = BlobServiceClient.from_connection_string(conn_str, retry_total=0)
        list(service.list_containers(max_results=1))
        return service
    except Exception:
        pytest.skip("Azurite is not running — skipping integration test")


# ─────────────────────────────────────────────────────────────────────────────
# LLM / agent mocks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_response() -> dict:
    """Canned OpenAI-compatible chat-completion payload — no live API call."""
    return {
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "model": "gpt-4o-mock",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "Based on the seismic amplitude anomaly at CDP 150-200, "
                        "I interpret a potential fault trending NW-SE. "
                        "Confidence: moderate. Recommend well-log correlation."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 35, "total_tokens": 77},
    }
