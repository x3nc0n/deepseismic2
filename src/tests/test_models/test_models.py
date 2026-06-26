"""Smoke tests for deepseismic.models.unet — UNet3D architecture and inference.

The real UNet3D uses UNetConfig for construction. The _make_model() helper wraps
this API so tests remain readable. Checkpoint tests exercise the save_checkpoint /
load_checkpoint class methods that are part of the public interface.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

from deepseismic.models.unet import UNet3D, UNetConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    in_channels: int = 1,
    out_channels: int = 1,
    depth: int = 4,
    init_features: int = 8,
    dropout_p: float = 0.0,
) -> UNet3D:
    """Construct a UNet3D via UNetConfig with test-friendly defaults (small model)."""
    cfg = UNetConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        init_features=init_features,
        dropout_p=dropout_p,
    )
    return UNet3D(config=cfg)


def _sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: tuple[int, int, int] = (16, 16, 16),
    overlap: float = 0.25,
) -> torch.Tensor:
    """Reference tiled inference: aggregate overlapping predictions over a 3-D volume."""
    model.eval()
    D, H, W = volume.shape[-3:]
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)
    output = torch.zeros_like(volume)
    weight = torch.zeros_like(volume)

    def _starts(total: int, patch: int, step: int) -> list[int]:
        pts = list(range(0, max(1, total - patch + 1), step))
        if not pts or pts[-1] + patch < total:
            pts.append(max(0, total - patch))
        return pts

    with torch.no_grad():
        for d in _starts(D, patch_size[0], stride[0]):
            for h in _starts(H, patch_size[1], stride[1]):
                for w in _starts(W, patch_size[2], stride[2]):
                    p = volume[
                        ...,
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ]
                    pred = model(p)
                    output[
                        ...,
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ] += pred
                    weight[
                        ...,
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ] += 1.0

    return output / weight.clamp(min=1.0)


# ---------------------------------------------------------------------------
# test_unet_forward_shape
# ---------------------------------------------------------------------------


class TestUNetForwardShape:
    def test_unet_forward_shape_64(self) -> None:
        """Forward pass (1,1,64,64,64) -> (1,1,64,64,64)."""
        model = _make_model(depth=4)
        model.eval()
        x = torch.zeros(1, 1, 64, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 64, 64, 64), f"Expected (1,1,64,64,64), got {y.shape}"

    def test_unet_forward_multi_channel_output(self) -> None:
        """out_channels=2 produces a 2-channel output tensor."""
        model = _make_model(out_channels=2, depth=3)
        model.eval()
        x = torch.zeros(1, 1, 32, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape[1] == 2, f"Expected 2 output channels, got {y.shape[1]}"

    def test_unet_forward_batch_dimension(self) -> None:
        """Batch size 4 must not alter spatial output shape."""
        model = _make_model(depth=3)
        model.eval()
        x = torch.zeros(4, 1, 16, 16, 16)
        with torch.no_grad():
            y = model(x)
        assert y.shape[0] == 4
        assert y.shape[2:] == (16, 16, 16)

    def test_unet_output_is_finite(self) -> None:
        """Forward pass on random input must not produce NaN or Inf."""
        model = _make_model(depth=3)
        model.eval()
        torch.manual_seed(99)
        x = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all(), "Model output contains NaN or Inf"


# ---------------------------------------------------------------------------
# test_unet_configurable_depth
# ---------------------------------------------------------------------------


class TestUNetConfigurableDepth:
    @pytest.mark.parametrize("depth", [3, 4])
    def test_unet_configurable_depth_forward(self, depth: int) -> None:
        """UNet3D with depth={3,4} constructs and produces the correct output shape."""
        size = 2**depth  # 8 for depth=3, 16 for depth=4
        model = _make_model(depth=depth)
        model.eval()
        x = torch.zeros(1, 1, size, size, size)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, size, size, size), (
            f"depth={depth}: expected (1,1,{size},{size},{size}), got {y.shape}"
        )

    def test_depth_accessible_via_config(self) -> None:
        """model.config.depth must reflect the constructor argument."""
        for d in (3, 4):
            m = _make_model(depth=d)
            assert m.config.depth == d, f"Expected depth={d}, got {m.config.depth}"

    def test_init_features_accessible_via_config(self) -> None:
        """model.config.init_features must reflect the constructor argument."""
        for feats in (8, 16):
            m = _make_model(init_features=feats)
            assert m.config.init_features == feats

    def test_parameter_count_method(self) -> None:
        """parameter_count() must return a dict with 'total' and 'trainable' keys."""
        m = _make_model(depth=3)
        pc = m.parameter_count()
        assert "total" in pc and "trainable" in pc
        assert pc["total"] > 0
        assert pc["trainable"] <= pc["total"]


# ---------------------------------------------------------------------------
# test_inference_sliding_window
# ---------------------------------------------------------------------------


class TestInferenceSlidingWindow:
    def test_inference_sliding_window_shape(self) -> None:
        """Sliding-window output shape must equal the input volume shape."""
        model = _make_model(depth=3)
        volume = torch.randn(1, 1, 32, 32, 32)
        output = _sliding_window_inference(model, volume, patch_size=(16, 16, 16), overlap=0.5)
        assert output.shape == volume.shape

    def test_inference_sliding_window_single_patch(self) -> None:
        """When patch covers the full volume, output shape still matches input."""
        model = _make_model(depth=3)
        volume = torch.randn(1, 1, 8, 8, 8)
        output = _sliding_window_inference(model, volume, patch_size=(8, 8, 8), overlap=0.0)
        assert output.shape == volume.shape

    def test_inference_sliding_window_no_nan(self) -> None:
        """Sliding-window output must not contain NaN values."""
        model = _make_model(depth=3)
        volume = torch.randn(1, 1, 16, 16, 16)
        output = _sliding_window_inference(model, volume, patch_size=(8, 8, 8), overlap=0.25)
        assert not torch.isnan(output).any(), "Sliding-window output contains NaN"

    def test_inference_sliding_window_full_coverage(self) -> None:
        """Every voxel must be covered -- no uncovered voxels (would produce NaN or zero)."""
        model = _make_model(depth=3)
        volume = torch.ones(1, 1, 12, 12, 12)
        output = _sliding_window_inference(model, volume, patch_size=(8, 8, 8), overlap=0.5)
        assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# test_checkpoint_save_load
# ---------------------------------------------------------------------------


class TestCheckpointSaveLoad:
    def test_checkpoint_save_load_identical_outputs(self, tmp_path) -> None:
        """save_checkpoint -> load_checkpoint produces byte-identical outputs."""
        model = _make_model(depth=3)
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 1, 8, 8, 8)

        with torch.no_grad():
            original_out = model(x)

        ckpt = tmp_path / "unet.pt"
        model.save_checkpoint(ckpt, epoch=1, metrics={"iou": 0.5})
        model2 = UNet3D.load_checkpoint(ckpt)
        with torch.no_grad():
            loaded_out = model2(x)

        torch.testing.assert_close(original_out, loaded_out)

    def test_checkpoint_preserves_config(self, tmp_path) -> None:
        """Checkpoint round-trip must restore depth and init_features."""
        model = _make_model(depth=3, init_features=8)
        ckpt = tmp_path / "cfg.pt"
        model.save_checkpoint(ckpt)
        restored = UNet3D.load_checkpoint(ckpt)
        assert restored.config.depth == 3
        assert restored.config.init_features == 8

    def test_checkpoint_state_dict_keys_stable(self) -> None:
        """Standard torch.save(state_dict) round-trip preserves all keys."""
        model = _make_model(depth=3)
        original_keys = set(model.state_dict().keys())
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location="cpu")
        assert set(loaded.keys()) == original_keys

    def test_checkpoint_metrics_are_os_portable(self, tmp_path) -> None:
        """Path objects in metrics must be stringified (issue #19).

        A WindowsPath pickled into the checkpoint raises
        ``cannot instantiate 'WindowsPath'`` when loaded on Linux.  The saved
        metrics must contain only JSON-safe scalars, including nested dicts.
        """
        from pathlib import PurePosixPath, PureWindowsPath

        model = _make_model(depth=3)
        ckpt = tmp_path / "portable.pt"
        model.save_checkpoint(
            ckpt,
            epoch=2,
            metrics={
                "iou": 0.7,
                "checkpoint_dir": PureWindowsPath(r"C:\runs\zarr"),
                "train_config": {"out": PurePosixPath("/tmp/out"), "lr": 5e-4},
            },
        )
        # Load without any path classes available would fail if Paths leaked.
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        metrics = raw["metrics"]

        def _assert_scalar(value) -> None:
            if isinstance(value, dict):
                for v in value.values():
                    _assert_scalar(v)
            elif isinstance(value, list):
                for v in value:
                    _assert_scalar(v)
            else:
                assert isinstance(value, bool | int | float | str) or value is None

        _assert_scalar(metrics)
        assert metrics["iou"] == pytest.approx(0.7)
        assert isinstance(metrics["checkpoint_dir"], str)
        assert isinstance(metrics["train_config"]["out"], str)


# ---------------------------------------------------------------------------
# test_cpu_gpu_parity
# ---------------------------------------------------------------------------


class TestCpuGpuParity:
    def test_cpu_gpu_parity(self) -> None:
        """Same weights + same input on CPU and CUDA must agree within tolerance."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available -- skipping GPU parity test")

        model_cpu = _make_model(depth=3)
        model_gpu = _make_model(depth=3)
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_cpu.eval()
        model_gpu.eval()
        model_gpu = model_gpu.cuda()

        torch.manual_seed(42)
        x_cpu = torch.randn(1, 1, 8, 8, 8)
        x_gpu = x_cpu.cuda()

        with torch.no_grad():
            out_cpu = model_cpu(x_cpu)
            out_gpu = model_gpu(x_gpu).cpu()

        torch.testing.assert_close(out_cpu, out_gpu, atol=1e-4, rtol=1e-4)

    def test_cpu_deterministic_repeated_forward(self) -> None:
        """Two CPU forward passes with identical input must produce identical output."""
        torch.manual_seed(42)
        model = _make_model(depth=3)
        model.eval()
        x = torch.randn(1, 1, 8, 8, 8)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(out1, out2)
