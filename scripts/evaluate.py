"""Evaluation script for trained fault-detection checkpoints (S2-03).

Loads a trained UNet3D checkpoint, runs full-volume sliding-window inference
on a held-out region of the Volve Zarr data, and computes validation metrics
against the fault_label.zarr ground truth.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt

    # Custom region (default: val+test split, il 70–100)
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --il-start 70 --il-end 100 \\
        --output output/eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import zarr

from deepseismic.models.inference import VolumeInference
from deepseismic.validation import evaluate_model

logger = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: Path,
    seismic_zarr: Path,
    label_zarr: Path,
    output_path: Path,
    patch_size: int = 32,
    overlap: float = 0.25,
    threshold: float = 0.5,
    device: str = "cpu",
    il_start: int = 64,
    il_end: int = 100,
) -> dict:
    """Run full-volume inference on a held-out region and compute metrics.

    Parameters
    ----------
    checkpoint_path:
        Path to a UNet3D .pt checkpoint.
    seismic_zarr:
        Zarr store containing 'amplitude' array, shape (n_il, n_xl, n_s).
    label_zarr:
        Zarr store containing 'fault_mask' array, same shape.
    output_path:
        JSON output path for metrics.
    patch_size:
        Cube side length for sliding-window inference.
    overlap:
        Patch overlap fraction [0, 1).
    threshold:
        Probability threshold for binary mask.
    device:
        'cpu' or 'cuda'.
    il_start, il_end:
        Inline range for the evaluation region (0-based indices).
        Default (70–100) covers the val+test split (70/15/15 inline split).

    Returns
    -------
    dict with all computed metric values.
    """
    # --- Load data ---------------------------------------------------------
    logger.info("Opening seismic zarr: %s", seismic_zarr)
    seismic_root = zarr.open_group(str(seismic_zarr), mode="r")
    amplitude = seismic_root["amplitude"]

    logger.info("Opening label zarr: %s", label_zarr)
    label_root = zarr.open_group(str(label_zarr), mode="r")
    fault_mask_arr = label_root["fault_mask"]

    n_il = amplitude.shape[0]
    il_start = max(0, il_start)
    il_end = min(n_il, il_end)
    logger.info(
        "Evaluating on inline range [%d, %d) — %d inlines",
        il_start, il_end, il_end - il_start,
    )

    # Load region into RAM (val+test region is small: ~30 inlines × 200 × 500)
    seismic_region = np.asarray(
        amplitude[il_start:il_end, :, :], dtype=np.float32
    )
    label_region = np.asarray(
        fault_mask_arr[il_start:il_end, :, :], dtype=np.float32
    )

    n_true = int(label_region.sum())
    n_total = label_region.size
    logger.info(
        "Ground truth: %d fault voxels / %d total (%.4f %%)",
        n_true, n_total, 100.0 * n_true / n_total,
    )

    # --- Inference ---------------------------------------------------------
    ps = (patch_size, patch_size, patch_size)
    engine = VolumeInference.from_checkpoint(
        checkpoint_path,
        device=device,
        patch_size=ps,
        overlap=overlap,
        threshold=threshold,
    )

    logger.info("Running sliding-window inference...")
    prob_volume, binary_mask = engine.run(seismic_region)

    n_pred = int(binary_mask.sum())
    logger.info("Predicted fault voxels: %d", n_pred)

    # --- Metrics -----------------------------------------------------------
    logger.info("Computing validation metrics...")
    vm = evaluate_model(prob_volume, label_region, threshold=threshold)

    # Build metrics dict for JSON output
    metrics = {
        "iou": vm.iou,
        "dice": vm.dice,
        "precision": vm.precision,
        "recall": vm.recall,
        "f1": vm.f1,
        "tolerant_precision_3": vm.tolerant_precision_3,
        "tolerant_recall_3": vm.tolerant_recall_3,
        "tolerant_precision_5": vm.tolerant_precision_5,
        "tolerant_recall_5": vm.tolerant_recall_5,
        "mean_distance_to_true_fault": (
            vm.mean_distance_to_true_fault
            if np.isfinite(vm.mean_distance_to_true_fault)
            else None
        ),
        "n_true_fault_voxels": vm.n_true_fault_voxels,
        "n_predicted_fault_voxels": vm.n_predicted_fault_voxels,
        "volume_shape": list(vm.volume_shape),
        "eval_region": {"il_start": il_start, "il_end": il_end},
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
    }

    # --- Print report ------------------------------------------------------
    print("\n" + vm.summary())

    # --- Write JSON --------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics written -> {output_path}")

    return metrics


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained UNet3D checkpoint on fault_label.zarr",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint (e.g. checkpoints/best.pt)",
    )
    parser.add_argument(
        "--seismic-zarr", default="data/volve/staged/synthetic.zarr",
        help="Seismic amplitude Zarr store (default: data/volve/staged/synthetic.zarr)",
    )
    parser.add_argument(
        "--label-zarr", default="data/volve/staged/fault_label.zarr",
        help="Fault label Zarr store (default: data/volve/staged/fault_label.zarr)",
    )
    parser.add_argument(
        "--output", default="output/eval_metrics.json",
        help="Output path for metrics JSON (default: output/eval_metrics.json)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=32,
        help="Sliding-window patch cube size (default: 32)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.25,
        help="Patch overlap fraction 0–1 (default: 0.25)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for binary mask (default: 0.5)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--il-start", type=int, default=64,
        help="Inline start (0-based) for evaluation region (default: 64 = val split start)",
    )
    parser.add_argument(
        "--il-end", type=int, default=100,
        help="Inline end (exclusive, 0-based) for evaluation region (default: 100)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_evaluation(
        checkpoint_path=Path(args.checkpoint),
        seismic_zarr=Path(args.seismic_zarr),
        label_zarr=Path(args.label_zarr),
        output_path=Path(args.output),
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        device=args.device,
        il_start=args.il_start,
        il_end=args.il_end,
    )


if __name__ == "__main__":
    main()
