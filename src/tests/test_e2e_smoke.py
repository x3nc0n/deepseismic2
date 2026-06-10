"""End-to-end smoke test: exercises the full pipeline in mock mode.

Verifies: sample data generation → ingest → patches → model forward pass →
API endpoints → agent tool calls. No external services needed.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def sample_segy(tmp_path_factory):
    """Generate a small synthetic SEG-Y for the smoke test."""
    import segyio

    tmp = tmp_path_factory.mktemp("smoke")
    segy_path = tmp / "smoke_volume.segy"

    n_il, n_xl, n_s = 20, 30, 100
    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 1
    spec.samples = np.arange(n_s) * 4.0
    spec.ilines = np.arange(1, n_il + 1)
    spec.xlines = np.arange(1, n_xl + 1)

    rng = np.random.default_rng(99)
    with segyio.create(str(segy_path), spec) as f:
        for il_idx in range(n_il):
            for xl_idx in range(n_xl):
                trace_idx = il_idx * n_xl + xl_idx
                f.trace[trace_idx] = rng.normal(0, 0.1, n_s).astype(np.float32)
                f.header[trace_idx].update({
                    segyio.TraceField.INLINE_3D: il_idx + 1,
                    segyio.TraceField.CROSSLINE_3D: xl_idx + 1,
                })

    return segy_path, tmp


class TestEndToEndPipeline:
    """Smoke test: full pipeline from raw data to agent response."""

    def test_01_ingest_segy_to_zarr(self, sample_segy):
        """SEG-Y loads and converts to Zarr."""
        from deepseismic.ingest.segy_loader import segy_to_zarr

        segy_path, tmp = sample_segy
        zarr_path = tmp / "zarr_out"

        metadata = segy_to_zarr(str(segy_path), str(zarr_path))

        assert metadata.n_inlines_loaded == 20
        assert metadata.geometry["n_crosslines"] == 30
        assert metadata.geometry["n_samples"] == 100
        assert metadata.amplitude_stats["nonzero_fraction"] > 0.9
        assert zarr_path.exists()

    def test_02_patch_extraction(self, sample_segy):
        """Patches can be extracted from the Zarr volume."""
        import zarr

        from deepseismic.ingest.segy_loader import segy_to_zarr
        from deepseismic.preprocessing.patches import PatchDataset

        segy_path, tmp = sample_segy
        zarr_path = tmp / "zarr_patches"
        segy_to_zarr(str(segy_path), str(zarr_path))

        # Open zarr and extract patches
        store = zarr.open(str(zarr_path), mode="r")
        if hasattr(store, "amplitude"):
            volume = np.array(store["amplitude"])
        else:
            volume = np.array(store)

        # Create patch dataset with small patches
        patch_size = (8, 8, 32)
        if all(
            volume.shape[i] >= patch_size[i] for i in range(3)
        ):
            dataset = PatchDataset(
                volume=volume,
                patch_size=patch_size,
                stride=patch_size,
            )
            assert len(dataset) > 0
            patch = dataset[0]
            assert patch.shape == patch_size or patch["data"].shape == patch_size

    def test_03_model_forward_pass(self):
        """UNet does a forward pass on a random patch."""
        import torch

        from deepseismic.models.unet import build_model

        model = build_model(init_features=16, depth=3)
        model.eval()

        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            y = model(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_04_api_health(self):
        """FastAPI app responds to health check."""
        from starlette.testclient import TestClient

        from deepseismic.api.main import app

        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_05_api_surveys_list(self):
        """API lists surveys in mock mode."""
        import os

        os.environ["DEEPSEISMIC_MOCK_MODE"] = "true"

        from starlette.testclient import TestClient

        from deepseismic.api.main import app

        client = TestClient(app)

        resp = client.get("/api/surveys")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (list, dict))

    def test_06_agent_mock_response(self):
        """Agent responds in mock LLM mode."""
        import os

        import deepseismic.agent.agent as agent_module

        os.environ["MOCK_LLM"] = "true"
        # Force mock mode (module-level constant cached at import)
        agent_module.MOCK_MODE = True

        from deepseismic.agent.agent import DeepSeismicAgent

        agent = DeepSeismicAgent()
        # chat() is a generator — collect all chunks
        response = "".join(agent.chat("What surveys are available?"))

        assert response is not None
        assert len(response) > 0

    def test_07_full_chain_mock(self):
        """Full chain: user question → agent → API → response."""
        import os

        import deepseismic.agent.agent as agent_module

        os.environ["DEEPSEISMIC_MOCK_MODE"] = "true"
        os.environ["MOCK_LLM"] = "true"
        agent_module.MOCK_MODE = True

        from deepseismic.agent.agent import DeepSeismicAgent

        agent = DeepSeismicAgent()

        # Simulate a typical user interaction
        questions = [
            "What is the Volve field?",
            "Show me available surveys",
            "What faults have been identified?",
        ]

        for q in questions:
            # chat() is a generator — collect all chunks
            response = "".join(agent.chat(q))
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 10  # Non-trivial response
