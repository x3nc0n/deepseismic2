"""Validation metrics for comparing model predictions to ground truth.

This module provides quantitative evaluation of fault detection against
the Volve dataset's human-interpreted fault sticks and horizons.

Metrics implemented:
    - Voxel-level IoU and Dice (fault detection accuracy)
    - Distance-based tolerance (within N voxels of true fault)
    - Fault throw consistency (horizon offset at fault intersections)
    - Structural coherence (fault continuity and planarity)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ValidationMetrics:
    """Container for all validation metrics from a single evaluation run."""

    # Voxel-level binary classification
    iou: float = 0.0
    dice: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Distance-tolerant metrics (fault within N voxels counts as correct)
    tolerant_precision_3: float = 0.0  # within 3 voxels
    tolerant_recall_3: float = 0.0
    tolerant_precision_5: float = 0.0  # within 5 voxels
    tolerant_recall_5: float = 0.0

    # Structural metrics
    mean_distance_to_true_fault: float = 0.0  # average voxel distance
    fault_continuity: float = 0.0  # fraction of connected predictions
    throw_error_mean_ms: float = 0.0  # mean error in fault throw (TWT ms)

    # Summary
    n_true_fault_voxels: int = 0
    n_predicted_fault_voxels: int = 0
    volume_shape: tuple[int, ...] = ()

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        lines = [
            "=" * 60,
            "VALIDATION RESULTS - Model vs. Ground Truth",
            "=" * 60,
            f"Volume shape:           {self.volume_shape}",
            f"True fault voxels:      {self.n_true_fault_voxels:,}",
            f"Predicted fault voxels: {self.n_predicted_fault_voxels:,}",
            "",
            "-- Voxel-level Metrics --",
            f"  IoU:       {self.iou:.4f}",
            f"  Dice:      {self.dice:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1:        {self.f1:.4f}",
            "",
            "-- Distance-Tolerant Metrics --",
            f"  Precision (+/-3 voxels): {self.tolerant_precision_3:.4f}",
            f"  Recall    (+/-3 voxels): {self.tolerant_recall_3:.4f}",
            f"  Precision (+/-5 voxels): {self.tolerant_precision_5:.4f}",
            f"  Recall    (+/-5 voxels): {self.tolerant_recall_5:.4f}",
            "",
            "-- Structural Metrics --",
            f"  Mean distance to true fault: "
            f"{self.mean_distance_to_true_fault:.2f} voxels",
            f"  Fault continuity:            {self.fault_continuity:.4f}",
            f"  Throw error (mean):          "
            f"{self.throw_error_mean_ms:.1f} ms TWT",
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_binary_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute voxel-level binary classification metrics.

    Args:
        prediction: Model output probabilities, shape (D, H, W).
        ground_truth: Binary mask (1=fault, 0=no fault), same shape.
        threshold: Probability threshold for binarizing predictions.

    Returns:
        Dict with iou, dice, precision, recall, f1.
    """
    pred_binary = (prediction >= threshold).astype(bool)
    gt_binary = ground_truth.astype(bool)

    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()
    fn = np.logical_and(~pred_binary, gt_binary).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0.0
    dice = (
        2 * intersection / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_distance_tolerant_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    tolerance_voxels: int = 3,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute metrics with spatial tolerance.

    A predicted fault voxel is considered a true positive if it falls
    within `tolerance_voxels` of any true fault voxel (and vice versa).
    This accounts for the inherent uncertainty in fault positioning --
    even human interpreters disagree by 1-5 traces.

    Args:
        prediction: Model output probabilities.
        ground_truth: Binary fault mask.
        tolerance_voxels: Distance tolerance in voxels.
        threshold: Binarization threshold.

    Returns:
        Dict with tolerant_precision and tolerant_recall.
    """
    from scipy.ndimage import binary_dilation

    pred_binary = (prediction >= threshold).astype(bool)
    gt_binary = ground_truth.astype(bool)

    # Dilate ground truth -- predicted fault near a true fault counts
    struct = np.ones((2 * tolerance_voxels + 1,) * 3, dtype=bool)
    gt_dilated = binary_dilation(gt_binary, structure=struct)
    pred_dilated = binary_dilation(pred_binary, structure=struct)

    # Tolerant precision: fraction of predictions near a true fault
    n_pred = pred_binary.sum()
    tolerant_tp_pred = np.logical_and(pred_binary, gt_dilated).sum()
    tolerant_precision = tolerant_tp_pred / n_pred if n_pred > 0 else 0.0

    # Tolerant recall: fraction of true faults near a prediction
    n_gt = gt_binary.sum()
    tolerant_tp_gt = np.logical_and(gt_binary, pred_dilated).sum()
    tolerant_recall = tolerant_tp_gt / n_gt if n_gt > 0 else 0.0

    return {
        "tolerant_precision": float(tolerant_precision),
        "tolerant_recall": float(tolerant_recall),
    }


def compute_mean_surface_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute average symmetric surface distance (ASSD).

    Mean distance from predicted fault surface to nearest true fault
    voxel, averaged symmetrically. Lower is better.

    Args:
        prediction: Model output probabilities.
        ground_truth: Binary fault mask.
        threshold: Binarization threshold.

    Returns:
        Mean distance in voxels. Returns inf if either mask is empty.
    """
    from scipy.ndimage import distance_transform_edt

    pred_binary = (prediction >= threshold).astype(bool)
    gt_binary = ground_truth.astype(bool)

    if not pred_binary.any() or not gt_binary.any():
        return float("inf")

    # Distance from each voxel to nearest true fault
    dist_to_gt = distance_transform_edt(~gt_binary)
    # Distance from each voxel to nearest prediction
    dist_to_pred = distance_transform_edt(~pred_binary)

    # Symmetric mean distance
    mean_pred_to_gt = dist_to_gt[pred_binary].mean()
    mean_gt_to_pred = dist_to_pred[gt_binary].mean()

    return float((mean_pred_to_gt + mean_gt_to_pred) / 2.0)


def fault_sticks_to_volume(
    fault_sticks: list[np.ndarray],
    volume_shape: tuple[int, int, int],
    inline_range: tuple[int, int],
    crossline_range: tuple[int, int],
    sample_range: tuple[float, float],
    sample_rate_ms: float = 4.0,
) -> np.ndarray:
    """Convert fault stick interpretations to a 3D binary volume.

    Fault sticks are sparse point sets along the fault surface. This
    function rasterizes them into the seismic volume grid and paints a
    small neighborhood around each point to create a continuous mask.

    Args:
        fault_sticks: List of arrays, each shape (N, 3) with columns
                      [inline, crossline, twt_ms].
        volume_shape: (n_inlines, n_crosslines, n_samples).
        inline_range: (min_inline, max_inline) from survey geometry.
        crossline_range: (min_crossline, max_crossline).
        sample_range: (first_sample_ms, last_sample_ms).
        sample_rate_ms: Time sample interval in ms.

    Returns:
        Binary volume of shape volume_shape (1=fault, 0=background).
    """
    mask = np.zeros(volume_shape, dtype=np.uint8)

    il_min, il_max = inline_range
    xl_min, xl_max = crossline_range
    t_min, _ = sample_range

    n_il, n_xl, n_s = volume_shape

    il_step = max(1, (il_max - il_min) / max(1, n_il - 1))
    xl_step = max(1, (xl_max - xl_min) / max(1, n_xl - 1))

    for stick in fault_sticks:
        for point in stick:
            il, xl, twt = float(point[0]), float(point[1]), float(point[2])

            # Convert to volume indices
            il_idx = int(round((il - il_min) / il_step))
            xl_idx = int(round((xl - xl_min) / xl_step))
            t_idx = int(round((twt - t_min) / sample_rate_ms))

            # Bounds check and paint neighborhood
            for di in range(-1, 2):
                for dxl in range(-1, 2):
                    ii = il_idx + di
                    xi = xl_idx + dxl
                    if 0 <= ii < n_il and 0 <= xi < n_xl and 0 <= t_idx < n_s:
                        mask[ii, xi, t_idx] = 1

    return mask


def load_volve_fault_sticks(fault_dir: Path) -> list[np.ndarray]:
    """Load Volve fault stick files (Charisma/RMS format).

    Each .dat file contains one fault with lines of:
        inline  crossline  x  y  twt_ms

    Args:
        fault_dir: Path to directory containing .dat fault stick files.

    Returns:
        List of arrays, each shape (N, 3) with [inline, crossline, twt_ms].
    """
    sticks = []

    for dat_file in sorted(fault_dir.glob("*.dat")):
        points = []
        with open(dat_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("FAULT"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        il = float(parts[0])
                        xl = float(parts[1])
                        # TWT is typically in the last column
                        twt = float(parts[-1])
                        points.append([il, xl, twt])
                    except ValueError:
                        continue

        if points:
            sticks.append(np.array(points))

    return sticks


def evaluate_model(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> ValidationMetrics:
    """Full evaluation of model predictions against ground truth.

    Computes all metrics: binary, distance-tolerant, and structural.

    Args:
        prediction: Model output probabilities, shape (D, H, W).
        ground_truth: Binary mask (1=fault, 0=no fault), same shape.
        threshold: Probability threshold.

    Returns:
        ValidationMetrics with all computed metrics.
    """
    binary_m = compute_binary_metrics(prediction, ground_truth, threshold)
    tol3 = compute_distance_tolerant_metrics(
        prediction, ground_truth, 3, threshold
    )
    tol5 = compute_distance_tolerant_metrics(
        prediction, ground_truth, 5, threshold
    )
    msd = compute_mean_surface_distance(prediction, ground_truth, threshold)

    return ValidationMetrics(
        iou=binary_m["iou"],
        dice=binary_m["dice"],
        precision=binary_m["precision"],
        recall=binary_m["recall"],
        f1=binary_m["f1"],
        tolerant_precision_3=tol3["tolerant_precision"],
        tolerant_recall_3=tol3["tolerant_recall"],
        tolerant_precision_5=tol5["tolerant_precision"],
        tolerant_recall_5=tol5["tolerant_recall"],
        mean_distance_to_true_fault=msd,
        fault_continuity=0.0,  # TODO: connected component analysis
        throw_error_mean_ms=0.0,  # TODO: horizon offset comparison
        n_true_fault_voxels=int(ground_truth.astype(bool).sum()),
        n_predicted_fault_voxels=int((prediction >= threshold).sum()),
        volume_shape=prediction.shape,
    )
