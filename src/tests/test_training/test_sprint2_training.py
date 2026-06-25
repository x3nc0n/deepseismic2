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

import numpy as np
import pytest
import torch
import torch.utils.data

from deepseismic.training.train import (
    NumpyPatchDataset,
    TrainConfig,
    _accum_tp_fp_fn,
    _epoch_metrics,
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

    def test_all_metrics_bounded_0_to_1(self):
        """IoU and Dice must always lie in [0, 1] for non-negative inputs."""
        cases = [(1, 1, 1), (10, 5, 0), (0, 10, 10), (100, 0, 1), (0, 0, 5)]
        for tp, fp, fn in cases:
            m = _epoch_metrics(float(tp), float(fp), float(fn))
            assert 0.0 <= m["iou"] <= 1.0 + 1e-8, f"IoU out of range for {tp},{fp},{fn}"
            assert 0.0 <= m["dice"] <= 1.0 + 1e-8, f"Dice out of range for {tp},{fp},{fn}"


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
