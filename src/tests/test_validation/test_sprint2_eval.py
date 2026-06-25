"""Sprint 2 S2-03 tests: evaluation metrics computation and JSON output schema.

Coverage:
- compute_binary_metrics: exact IoU/Dice values from hand-constructed pred/gt tensors
- evaluate_model: correct field schema (ValidationMetrics), edge cases
- run_evaluation: JSON output has all expected keys and serialisable values
  (integration-marked — uses tiny zarr stores + minimal UNet checkpoint)

All unit tests are synthetic / CPU-only with no disk I/O beyond tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.storage

from deepseismic.validation import (
    ValidationMetrics,
    compute_binary_metrics,
    evaluate_model,
)

# ---------------------------------------------------------------------------
# Expected JSON output keys from run_evaluation
# ---------------------------------------------------------------------------

_EVAL_JSON_REQUIRED_KEYS = frozenset({
    "iou",
    "dice",
    "precision",
    "recall",
    "f1",
    "tolerant_precision_3",
    "tolerant_recall_3",
    "tolerant_precision_5",
    "tolerant_recall_5",
    "mean_distance_to_true_fault",
    "n_true_fault_voxels",
    "n_predicted_fault_voxels",
    "volume_shape",
    "eval_region",
    "checkpoint",
    "threshold",
})


# ---------------------------------------------------------------------------
# compute_binary_metrics — exact formula verification
# ---------------------------------------------------------------------------


class TestComputeBinaryMetricsExact:
    """Verify compute_binary_metrics returns exact values from known TP/FP/FN.

    The test constructs pred/gt arrays with a completely known confusion
    matrix, then asserts exact metric values from the formulas:
        IoU       = TP / (TP + FP + FN)
        Dice      = 2*TP / (2*TP + FP + FN)
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
    """

    def test_perfect_overlap_gives_all_ones(self):
        """Identical pred and gt → IoU=1, Dice=1, Precision=1, Recall=1."""
        shape = (8, 8, 8)
        gt = np.zeros(shape, dtype=np.float32)
        gt[2:6, 2:6, :] = 1.0
        pred = gt.copy()
        m = compute_binary_metrics(pred, gt)
        assert m["iou"] == pytest.approx(1.0)
        assert m["dice"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_zero_overlap_gives_all_zeros(self):
        """No overlap → IoU=0, Dice=0, Precision=0, Recall=0."""
        shape = (10, 10, 10)
        gt = np.zeros(shape, dtype=np.float32)
        gt[0, :, :] = 1.0
        pred = np.zeros(shape, dtype=np.float32)
        pred[9, :, :] = 1.0   # opposite face — no overlap
        m = compute_binary_metrics(pred, gt)
        assert m["iou"] == pytest.approx(0.0)
        assert m["dice"] == pytest.approx(0.0)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_known_confusion_matrix_exact_values(self):
        """Hand-constructed confusion: TP=4, FP=2, FN=3 → exact metric values.

        Build a tiny 3D volume where we know exactly which voxels are:
        - TP: in both pred and gt
        - FP: in pred only
        - FN: in gt only
        """
        shape = (9, 1, 1)
        gt = np.zeros(shape, dtype=np.float32)
        gt[:7, 0, 0] = 1.0    # 7 true-positive + false-negative positions
        pred = np.zeros(shape, dtype=np.float32)
        pred[:6, 0, 0] = 1.0  # first 6 predicted positive
        pred[7:9, 0, 0] = 1.0 # 2 false positives

        # Confusion: TP=6, FP=2, FN=1
        # IoU = 6 / (6+2+1) = 6/9
        # Dice = 12 / (12+2+1) = 12/15
        m = compute_binary_metrics(pred, gt)
        assert m["iou"] == pytest.approx(6.0 / 9.0, abs=1e-5)
        assert m["dice"] == pytest.approx(12.0 / 15.0, abs=1e-5)
        assert m["precision"] == pytest.approx(6.0 / 8.0, abs=1e-5)  # 6/(6+2)
        assert m["recall"] == pytest.approx(6.0 / 7.0, abs=1e-5)     # 6/(6+1)


# ---------------------------------------------------------------------------
# evaluate_model — schema and edge cases
# ---------------------------------------------------------------------------


class TestEvaluateModelSchema:
    """Verify evaluate_model returns a correctly structured ValidationMetrics."""

    @pytest.fixture
    def tiny_pred_gt(self):
        shape = (16, 16, 16)
        rng = np.random.default_rng(7)
        gt = np.zeros(shape, dtype=np.float32)
        gt[6:10, :, :] = 1.0
        pred = np.zeros(shape, dtype=np.float32)
        pred[6:10, :, :] = 0.9
        pred += rng.uniform(0, 0.1, shape).astype(np.float32)
        return pred, gt

    def test_returns_validation_metrics_instance(self, tiny_pred_gt):
        pred, gt = tiny_pred_gt
        m = evaluate_model(pred, gt)
        assert isinstance(m, ValidationMetrics)

    def test_all_required_fields_present(self, tiny_pred_gt):
        pred, gt = tiny_pred_gt
        m = evaluate_model(pred, gt)
        # Every field from the evaluate.py output dict must exist on the object
        for field in ("iou", "dice", "precision", "recall", "f1",
                      "tolerant_precision_3", "tolerant_recall_3",
                      "tolerant_precision_5", "tolerant_recall_5",
                      "mean_distance_to_true_fault",
                      "n_true_fault_voxels", "n_predicted_fault_voxels",
                      "volume_shape"):
            assert hasattr(m, field), f"ValidationMetrics missing field: {field}"

    def test_volume_shape_matches_input(self, tiny_pred_gt):
        pred, gt = tiny_pred_gt
        m = evaluate_model(pred, gt)
        assert m.volume_shape == pred.shape

    def test_metrics_in_valid_range(self, tiny_pred_gt):
        pred, gt = tiny_pred_gt
        m = evaluate_model(pred, gt)
        for attr in ("iou", "dice", "precision", "recall", "f1"):
            val = getattr(m, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} outside [0,1]"

    def test_summary_string_contains_required_labels(self, tiny_pred_gt):
        pred, gt = tiny_pred_gt
        m = evaluate_model(pred, gt)
        summary = m.summary()
        for token in ("VALIDATION RESULTS", "IoU", "Dice", "Precision", "Recall"):
            assert token in summary, f"Summary missing '{token}'"


# ---------------------------------------------------------------------------
# run_evaluation JSON schema (S2-03 integration test)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunEvaluationJSONSchema:
    """Verify that run_evaluation produces a JSON file with the expected schema.

    Builds a tiny UNet3D checkpoint and synthetic zarr stores to exercise
    the full evaluation pipeline end-to-end without real Volve data.

    Marked integration because it involves UNet3D inference on CPU
    (~5 s for a 4³ volume) and disk I/O.
    """

    @pytest.fixture(scope="class")
    def tiny_eval_artifacts(self, tmp_path_factory):
        """Create minimal checkpoint + seismic/label zarr for a 4-inline eval region."""
        from deepseismic.models.unet import build_model

        tmp = tmp_path_factory.mktemp("eval")

        # --- Seismic zarr (amplitude array, shape 8×8×8) ---
        rng = np.random.default_rng(11)
        seismic_data = rng.standard_normal((8, 8, 8)).astype(np.float32)
        seismic_path = tmp / "seismic.zarr"
        s_store = zarr.storage.LocalStore(str(seismic_path))
        s_root = zarr.open_group(s_store, mode="w")
        s_root.create_array("amplitude", data=seismic_data, chunks=(4, 4, 4))

        # --- Label zarr (fault_mask array, same shape) ---
        label_data = np.zeros((8, 8, 8), dtype=np.uint8)
        label_data[3:5, :, :] = 1
        label_path = tmp / "label.zarr"
        l_store = zarr.storage.LocalStore(str(label_path))
        l_root = zarr.open_group(l_store, mode="w")
        l_root.create_array("fault_mask", data=label_data, chunks=(4, 4, 4))

        # --- Tiny UNet3D checkpoint ---
        model = build_model(init_features=4, depth=2, dropout_p=0.0)
        ckpt_path = tmp / "test_ckpt.pt"
        model.save_checkpoint(str(ckpt_path), epoch=1, metrics={"iou": 0.0})

        output_path = tmp / "eval_out.json"
        return {
            "checkpoint": ckpt_path,
            "seismic_zarr": seismic_path,
            "label_zarr": label_path,
            "output": output_path,
        }

    def test_json_output_has_all_required_keys(self, tiny_eval_artifacts):
        """run_evaluation must produce a JSON file with every required key."""
        from scripts.evaluate import run_evaluation  # noqa: PLC0415

        arts = tiny_eval_artifacts
        metrics = run_evaluation(
            checkpoint_path=arts["checkpoint"],
            seismic_zarr=arts["seismic_zarr"],
            label_zarr=arts["label_zarr"],
            output_path=arts["output"],
            patch_size=4,
            overlap=0.0,
            il_start=0,
            il_end=8,
        )
        assert _EVAL_JSON_REQUIRED_KEYS.issubset(set(metrics.keys())), (
            f"Missing keys: {_EVAL_JSON_REQUIRED_KEYS - set(metrics.keys())}"
        )

    def test_json_file_is_valid_and_serialisable(self, tiny_eval_artifacts):
        """The output JSON file must be readable and contain only JSON-safe values."""
        arts = tiny_eval_artifacts
        json_path: Path = arts["output"]
        assert json_path.exists(), "JSON output file was not created"
        with open(json_path) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        # All numeric metrics must be JSON-finite or None
        for key in ("iou", "dice", "precision", "recall", "f1"):
            val = data[key]
            assert isinstance(val, (int, float)), f"{key}={val!r} is not numeric"
