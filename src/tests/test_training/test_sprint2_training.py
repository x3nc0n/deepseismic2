"""Sprint 2 S2-02/S2-05/S2-08 tests: training plumbing.

Coverage:
- TrainConfig defaults: seed=42, data_mode="synthetic" (S2-05)
- _accum_tp_fp_fn: confusion counts from logit tensors (S2-08)
- _epoch_metrics: IoU/Dice/Precision/Recall from raw count accumulators (S2-08)
  — tested with exact hand-computed expected values to guard formula correctness
- Seed determinism: same seed=42 yields identical first DataLoader batch (S2-05)

All tests are CPU-only and data-free (no zarr, no GPU, no disk I/O for unit tests).
Integration-tagged tests use small in-memory numpy data only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.utils.data
import zarr

from deepseismic.training.train import (
    VAL_THRESHOLD_GRID,
    NumpyPatchDataset,
    TrainConfig,
    _accum_tp_fp_fn,
    _build_zarr_loaders,
    _epoch_metrics,
    _select_best_checkpoint,
    _should_cache_volume,
    _sweep_probs_metrics,
    _upload_checkpoints,
)

# ---------------------------------------------------------------------------
# TrainConfig defaults (S2-05)
# ---------------------------------------------------------------------------


class TestTrainConfig:
    """Verify TrainConfig default values set by Sprint 2."""

    def test_default_seed_is_42(self):
        """Reproducibility seed must default to 42 (S2-05)."""
        cfg = TrainConfig()
        assert cfg.seed == 42

    def test_default_data_mode_is_synthetic(self):
        """Default data_mode must be 'synthetic' (backward compat, S2-02)."""
        cfg = TrainConfig()
        assert cfg.data_mode == "synthetic"

    def test_zarr_data_mode_is_accepted(self):
        """Setting data_mode='zarr' must not raise (S2-02)."""
        cfg = TrainConfig(data_mode="zarr")
        assert cfg.data_mode == "zarr"

    def test_cache_volume_defaults_to_azure_only(self):
        """Auto cache policy must be on for azure and off for local."""
        assert _should_cache_volume(TrainConfig(storage_backend="azure")) is True
        assert _should_cache_volume(TrainConfig(storage_backend="local")) is False

    def test_cache_volume_explicit_flags_win(self):
        """Explicit cache_volume True/False overrides the backend default."""
        assert _should_cache_volume(
            TrainConfig(storage_backend="local", cache_volume=True)
        ) is True
        assert _should_cache_volume(
            TrainConfig(storage_backend="azure", cache_volume=False)
        ) is False

    def test_zarr_loader_caches_arrays_when_enabled(self, monkeypatch):
        """Cached zarr loaders should hand numpy arrays to PatchDataset."""
        seismic = zarr.array(
            np.ones((20, 4, 4), dtype=np.float32),
            chunks=(5, 2, 2),
        )
        labels_np = np.zeros((20, 4, 4), dtype=np.uint8)
        labels_np[0:2, 0:2, 0:2] = 1
        labels = zarr.array(labels_np, chunks=(5, 2, 2))

        def _fake_open_zarr_root(path, *, backend, az_container=None, az_prefix=None):
            path_str = str(path)
            if path_str == "labels":
                return {"fault_mask": labels}
            if path_str == "seismic":
                return {"amplitude": seismic}
            raise AssertionError(f"unexpected zarr path: {path}")

        monkeypatch.setattr(
            "deepseismic.storage.zarr_helpers.open_zarr_root",
            _fake_open_zarr_root,
        )

        cfg = TrainConfig(
            data_mode="zarr",
            seismic_zarr=Path("seismic"),
            label_zarr=Path("labels"),
            patch_size=(2, 2, 2),
            stride=(2, 2, 2),
            cache_volume=True,
        )
        train_loader, val_loader = _build_zarr_loaders(cfg)

        assert isinstance(train_loader.dataset._seismic, np.ndarray)
        assert isinstance(train_loader.dataset._labels, np.ndarray)
        assert isinstance(val_loader.dataset._seismic, np.ndarray)
        assert isinstance(val_loader.dataset._labels, np.ndarray)


# ---------------------------------------------------------------------------
# Checkpoint blob upload
# ---------------------------------------------------------------------------


class TestCheckpointUpload:
    """Verify final checkpoint uploads use StorageClient without real Azure."""

    @staticmethod
    def _write_artifacts(checkpoint_dir):
        for filename in ("best.pt", "latest.pt", "run_config.json"):
            (checkpoint_dir / filename).write_bytes(filename.encode())

    def test_uploads_existing_checkpoint_artifacts(self, tmp_path, monkeypatch):
        """A configured prefix uploads best/latest/config to the staged container."""
        self._write_artifacts(tmp_path)
        calls = []

        class _MockStorageClient:
            def upload_blob(self, container, blob_path, data, *, overwrite=True, metadata=None):
                calls.append((container, blob_path, data.read(), overwrite, metadata))

        monkeypatch.setattr(
            "deepseismic.storage.blob_client.StorageClient",
            _MockStorageClient,
        )

        cfg = TrainConfig(
            checkpoint_dir=tmp_path,
            checkpoint_upload_prefix="models/f3-demo/run-001/",
        )
        _upload_checkpoints(cfg)

        assert calls == [
            ("staged", "models/f3-demo/run-001/best.pt", b"best.pt", True, None),
            ("staged", "models/f3-demo/run-001/latest.pt", b"latest.pt", True, None),
            (
                "staged",
                "models/f3-demo/run-001/run_config.json",
                b"run_config.json",
                True,
                None,
            ),
        ]

    def test_no_prefix_does_not_build_storage_client(self, tmp_path, monkeypatch):
        """No upload prefix leaves local checkpoint behavior unchanged."""
        self._write_artifacts(tmp_path)

        def _raise_if_called():
            raise AssertionError("StorageClient should not be constructed")

        monkeypatch.setattr(
            "deepseismic.storage.blob_client.StorageClient",
            _raise_if_called,
        )

        cfg = TrainConfig(checkpoint_dir=tmp_path, checkpoint_upload_prefix=None)
        _upload_checkpoints(cfg)

    def test_upload_failure_raises(self, tmp_path, monkeypatch):
        """Upload errors must fail the training process loudly."""
        self._write_artifacts(tmp_path)

        class _FailingStorageClient:
            def upload_blob(self, container, blob_path, data, *, overwrite=True, metadata=None):
                raise OSError("network unavailable")

        monkeypatch.setattr(
            "deepseismic.storage.blob_client.StorageClient",
            _FailingStorageClient,
        )

        cfg = TrainConfig(
            checkpoint_dir=tmp_path,
            checkpoint_upload_prefix="models/f3-demo/run-001",
        )
        with pytest.raises(RuntimeError, match="Failed to upload checkpoint artifact"):
            _upload_checkpoints(cfg)


# ---------------------------------------------------------------------------
# _accum_tp_fp_fn (S2-08)
# ---------------------------------------------------------------------------


class TestAccumTpFpFn:
    """Tests for _accum_tp_fp_fn: confusion counts from logit tensors.

    The function takes raw LOGITS (not probabilities) and applies sigmoid
    before thresholding.  Large positive logit (≥10) → binary=1; large
    negative logit (≤-10) → binary=0.
    """

    def test_all_correct_positives(self):
        """All predictions correct and positive → TP=N, FP=0, FN=0."""
        n = 10
        preds = torch.full((n,), 10.0)   # sigmoid ≈ 1 → binary=1
        targets = torch.ones(n)
        tp, fp, fn = _accum_tp_fp_fn(preds, targets)
        assert tp == pytest.approx(n)
        assert fp == pytest.approx(0.0)
        assert fn == pytest.approx(0.0)

    def test_all_false_negatives(self):
        """Predicting all 0 when truth is all 1 → TP=0, FP=0, FN=N."""
        n = 10
        preds = torch.full((n,), -10.0)  # sigmoid ≈ 0 → binary=0
        targets = torch.ones(n)
        tp, fp, fn = _accum_tp_fp_fn(preds, targets)
        assert tp == pytest.approx(0.0)
        assert fp == pytest.approx(0.0)
        assert fn == pytest.approx(n)

    def test_all_false_positives(self):
        """Predicting all 1 when truth is all 0 → TP=0, FP=N, FN=0."""
        n = 10
        preds = torch.full((n,), 10.0)   # binary=1
        targets = torch.zeros(n)
        tp, fp, fn = _accum_tp_fp_fn(preds, targets)
        assert tp == pytest.approx(0.0)
        assert fp == pytest.approx(n)
        assert fn == pytest.approx(0.0)

    def test_mixed_known_counts(self):
        """4 TPs + 3 FNs + 2 FPs — hand-constructed tensor."""
        preds = torch.tensor([
            10., 10., 10., 10.,    # 4 → binary=1, target=1 → TP
            -10., -10., -10.,      # 3 → binary=0, target=1 → FN
            10., 10.,              # 2 → binary=1, target=0 → FP
        ])
        targets = torch.tensor([
            1., 1., 1., 1.,
            1., 1., 1.,
            0., 0.,
        ])
        tp, fp, fn = _accum_tp_fp_fn(preds, targets)
        assert tp == pytest.approx(4.0)
        assert fp == pytest.approx(2.0)
        assert fn == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _epoch_metrics (S2-08)
# ---------------------------------------------------------------------------


class TestEpochMetrics:
    """Tests for _epoch_metrics: IoU/Dice/Precision/Recall from count accumulators.

    These tests verify the exact formula:
        IoU       = TP / (TP + FP + FN + ε)
        Dice      = 2*TP / (2*TP + FP + FN + ε)
        Precision = TP / (TP + FP + ε)
        Recall    = TP / (TP + FN + ε)

    Epoch-level accumulation (not per-batch) is the correct method for sparse
    labels; the formula must not change to per-batch averaging.
    """

    def test_perfect_prediction(self):
        """TP=10, FP=0, FN=0 → all metrics ≈ 1.0."""
        m = _epoch_metrics(tp=10.0, fp=0.0, fn=0.0)
        assert m["iou"] == pytest.approx(1.0, abs=1e-5)
        assert m["dice"] == pytest.approx(1.0, abs=1e-5)
        assert m["precision"] == pytest.approx(1.0, abs=1e-5)
        assert m["recall"] == pytest.approx(1.0, abs=1e-5)

    def test_known_iou_and_dice(self):
        """TP=5, FP=2, FN=3 → IoU=0.5, Dice=10/15 (exact formula verification)."""
        # IoU  = 5 / (5 + 2 + 3) = 5/10 = 0.5
        # Dice = 10 / (10 + 2 + 3) = 10/15 ≈ 0.6667
        m = _epoch_metrics(tp=5.0, fp=2.0, fn=3.0)
        assert m["iou"] == pytest.approx(0.5, abs=1e-4)
        assert m["dice"] == pytest.approx(10.0 / 15.0, abs=1e-4)

    def test_known_precision_and_recall(self):
        """TP=5, FP=2, FN=3 → Precision=5/7, Recall=5/8."""
        m = _epoch_metrics(tp=5.0, fp=2.0, fn=3.0)
        assert m["precision"] == pytest.approx(5.0 / 7.0, abs=1e-4)
        assert m["recall"] == pytest.approx(5.0 / 8.0, abs=1e-4)

    def test_zero_counts_no_division_error(self):
        """All-zero counts must not raise and must return 0.0 for all metrics."""
        m = _epoch_metrics(tp=0.0, fp=0.0, fn=0.0)
        assert m["iou"] == pytest.approx(0.0, abs=1e-5)
        assert m["dice"] == pytest.approx(0.0, abs=1e-5)
        assert m["precision"] == pytest.approx(0.0, abs=1e-5)
        assert m["recall"] == pytest.approx(0.0, abs=1e-5)

    def test_all_metrics_bounded_0_to_1(self):
        """IoU and Dice must always lie in [0, 1] for non-negative inputs."""
        cases = [(1, 1, 1), (10, 5, 0), (0, 10, 10), (100, 0, 1), (0, 0, 5)]
        for tp, fp, fn in cases:
            m = _epoch_metrics(float(tp), float(fp), float(fn))
            assert 0.0 <= m["iou"] <= 1.0 + 1e-8, f"IoU out of range for {tp},{fp},{fn}"
            assert 0.0 <= m["dice"] <= 1.0 + 1e-8, f"Dice out of range for {tp},{fp},{fn}"


# ---------------------------------------------------------------------------
# Validation threshold sweep / best-checkpoint selection
# ---------------------------------------------------------------------------


class TestValidationThresholdSweep:
    """Tests for swept validation metrics used by train.validate."""

    def test_threshold_sweep_finds_lower_threshold_signal(self):
        """A model with probabilities below 0.5 can still score true positives."""
        probs = torch.tensor([0.40, 0.35, 0.01, 0.02])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        metrics = _sweep_probs_metrics(probs, targets)

        assert metrics["iou"] > 0.0
        assert metrics["best_threshold"] < 0.5
        assert metrics["best_threshold"] in VAL_THRESHOLD_GRID

    def test_grid_average_precision_is_one_for_separable_probs(self):
        """Perfectly separated positives/negatives should have AP near 1."""
        probs = torch.tensor([0.90, 0.80, 0.10, 0.05])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        metrics = _sweep_probs_metrics(probs, targets)

        assert metrics["ap"] == pytest.approx(1.0, abs=1e-5)

    def test_grid_average_precision_degenerate_case_is_finite(self):
        """No positive predictions across the grid should produce AP=0, not NaN."""
        probs = torch.zeros(4)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        metrics = _sweep_probs_metrics(probs, targets)

        assert metrics["ap"] == pytest.approx(0.0, abs=1e-5)
        assert np.isfinite(metrics["ap"])


class TestBestCheckpointSelection:
    """Tests for robust best.pt selection when swept IoU remains zero."""

    def test_fallback_saves_first_best_by_loss_when_iou_is_zero(self):
        """The first zero-IoU epoch still writes best.pt using the loss fallback."""
        should_save, selected_by, best_iou, best_loss, best_saved = _select_best_checkpoint(
            {"iou": 0.0, "loss": 0.75},
            best_val_iou=0.0,
            best_val_loss=float("inf"),
            best_saved=False,
        )

        assert should_save is True
        assert selected_by == "loss"
        assert best_iou == pytest.approx(0.0)
        assert best_loss == pytest.approx(0.75)
        assert best_saved is True

    def test_iou_improvement_takes_primary_precedence(self):
        """Any strict IoU improvement writes best.pt and marks the reason as IoU."""
        should_save, selected_by, best_iou, best_loss, best_saved = _select_best_checkpoint(
            {"iou": 0.10, "loss": 0.90},
            best_val_iou=0.0,
            best_val_loss=1.00,
            best_saved=True,
        )

        assert should_save is True
        assert selected_by == "iou"
        assert best_iou == pytest.approx(0.10)
        assert best_loss == pytest.approx(0.90)
        assert best_saved is True

    def test_fallback_updates_best_on_subsequent_loss_improvement_when_iou_zero(self):
        """After epoch 1 saves via loss-fallback, epoch 2 with lower loss must ALSO update.

        Regression guard for issue #37: the original `not best_saved` condition only
        captured the first epoch (essentially a 'first-epoch' fallback, not a
        'best-by-loss' fallback).  With best_saved=True and IoU stuck at 0, a later
        epoch that achieves lower validation loss must still overwrite best.pt.
        """
        # Simulate epoch 2: best_saved=True (epoch 1 saved via loss), IoU still 0
        should_save, selected_by, best_iou, best_loss, best_saved = _select_best_checkpoint(
            {"iou": 0.0, "loss": 0.55},  # better loss than epoch-1's 0.75
            best_val_iou=0.0,
            best_val_loss=0.75,           # epoch 1 set this
            best_saved=True,
        )

        assert should_save is True, "epoch-2 with lower loss must overwrite best.pt"
        assert selected_by == "loss"
        assert best_loss == pytest.approx(0.55)
        assert best_saved is True

    def test_fallback_does_not_overwrite_better_iou_checkpoint_with_lower_loss(self):
        """Loss fallback must NOT fire when IoU has regressed below the saved best.

        If epoch 5 had IoU=0.3 (saved as best.pt) and epoch 6 has IoU=0.2 with
        a better loss, the IoU-regression guard must prevent overwriting.
        """
        should_save, selected_by, best_iou, best_loss, best_saved = _select_best_checkpoint(
            {"iou": 0.2, "loss": 0.30},  # worse IoU, better loss
            best_val_iou=0.3,             # epoch 5 was better
            best_val_loss=0.50,
            best_saved=True,
        )

        assert should_save is False, "must not overwrite a better-IoU checkpoint with loss alone"
        assert selected_by is None


# ---------------------------------------------------------------------------
# Seed determinism (S2-05)
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    """Verify seed=42 produces identical first DataLoader batch across runs (S2-05).

    Shuffle order is determined by torch.Generator at iterator-creation time.
    We use explicit Generator objects (not the global RNG) to make the test
    hermetic and independent of whatever random state surrounds it.
    """

    @pytest.fixture(scope="class")
    def vol_and_mask(self):
        rng = np.random.default_rng(99)
        shape = (32, 32, 32)
        vol = rng.standard_normal(shape).astype(np.float32)
        mask = (rng.random(shape) > 0.9).astype(np.uint8)
        return vol, mask

    @staticmethod
    def _first_batch(
        vol: np.ndarray,
        mask: np.ndarray,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the first batch from a shuffled DataLoader seeded at *seed*."""
        ds = NumpyPatchDataset(vol, mask, patch_size=(8, 8, 8), stride=(4, 4, 4))
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, shuffle=True, generator=gen, num_workers=0
        )
        return next(iter(loader))

    def test_same_seed_produces_identical_first_batch(self, vol_and_mask):
        """Two DataLoaders with the same seed must yield identical first batches."""
        vol, mask = vol_and_mask
        xa, ya = self._first_batch(vol, mask, seed=42)
        xb, yb = self._first_batch(vol, mask, seed=42)
        torch.testing.assert_close(xa, xb)
        torch.testing.assert_close(ya, yb)

    def test_different_seeds_produce_different_batches(self, vol_and_mask):
        """Different seeds should (with overwhelming probability) yield different batches."""
        vol, mask = vol_and_mask
        xa, _ = self._first_batch(vol, mask, seed=42)
        xb, _ = self._first_batch(vol, mask, seed=123)
        assert not torch.allclose(xa, xb), (
            "Seeds 42 and 123 unexpectedly produced identical first batches"
        )
