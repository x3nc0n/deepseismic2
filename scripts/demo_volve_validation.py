"""End-to-end demo with Volve data: ingest, train, predict, validate.

This script demonstrates the full pipeline using either synthetic data
(generated locally) or real Volve data (if available from Equinor).

Usage:
    # With synthetic data (always available):
    python scripts/demo_volve_validation.py

    # With real Volve SEGY (after downloading):
    python scripts/demo_volve_validation.py --segy data/volve/seismic/ST10010.segy

    # Skip training (use existing checkpoint):
    python scripts/demo_volve_validation.py --checkpoint checkpoints/latest.pt

The script:
    1. Loads seismic data (synthetic or real SEGY)
    2. Loads ground truth fault interpretations
    3. Trains (or loads) the UNet3D model
    4. Runs inference on the full volume
    5. Compares predictions to ground truth
    6. Prints a validation report with metrics
    7. Saves a visual comparison as PNG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_or_load_data(
    segy_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load seismic volume and ground truth fault mask.

    If segy_path is provided, loads real data + fault sticks.
    Otherwise generates synthetic data with known faults.

    Returns:
        (seismic_volume, fault_mask, metadata_dict)
    """
    if segy_path and segy_path.exists():
        logger.info("Loading real SEGY: %s", segy_path)
        return _load_real_volve(segy_path)
    else:
        logger.info("Generating synthetic demo data...")
        return _generate_synthetic()


def _generate_synthetic() -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate synthetic seismic with known fault geometry."""
    rng = np.random.default_rng(42)
    n_il, n_xl, n_s = 64, 128, 128

    # Background layered reflectivity
    seismic = np.zeros((n_il, n_xl, n_s), dtype=np.float32)
    for layer_depth in rng.integers(10, n_s - 10, size=8):
        amplitude = rng.uniform(-0.5, 0.5)
        seismic[:, :, layer_depth - 2 : layer_depth + 2] += amplitude

    # Ground truth fault mask
    fault_mask = np.zeros((n_il, n_xl, n_s), dtype=np.float32)

    # Main normal fault (dipping ~60 degrees)
    for il in range(n_il):
        fault_xl = int(n_xl * 0.4 + il * 0.3)
        for s in range(n_s):
            xl_at_depth = fault_xl + int(s * 0.15)
            if 0 <= xl_at_depth < n_xl:
                fault_mask[il, xl_at_depth, s] = 1.0
                # Apply throw to seismic
                if xl_at_depth + 1 < n_xl:
                    throw = 3
                    seismic[il, xl_at_depth + 1 :, s] = np.roll(
                        seismic[il, xl_at_depth + 1 :, s], throw
                    )

    # Antithetic fault
    for il in range(n_il // 3, 2 * n_il // 3):
        fault_xl = int(n_xl * 0.7 - (il - n_il // 3) * 0.2)
        for s in range(n_s // 4, 3 * n_s // 4):
            xl_at_depth = fault_xl - int((s - n_s // 4) * 0.1)
            if 0 <= xl_at_depth < n_xl:
                fault_mask[il, xl_at_depth, s] = 1.0

    # Add noise
    seismic += rng.normal(0, 0.05, seismic.shape).astype(np.float32)

    metadata = {
        "source": "synthetic",
        "shape": seismic.shape,
        "n_faults": 2,
        "fault_types": ["main_normal", "antithetic"],
        "inline_range": (1, n_il),
        "crossline_range": (1, n_xl),
        "sample_rate_ms": 4.0,
    }

    return seismic, fault_mask, metadata


def _load_real_volve(segy_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load real Volve SEGY and fault stick interpretations."""
    import segyio

    from deepseismic.validation import fault_sticks_to_volume, load_volve_fault_sticks

    logger.info("Reading SEGY file...")
    with segyio.open(str(segy_path), "r", strict=False) as f:
        n_il = len(f.ilines)
        n_xl = len(f.xlines)
        n_s = len(f.samples)
        seismic = segyio.tools.cube(f).astype(np.float32)
        il_min, il_max = int(f.ilines[0]), int(f.ilines[-1])
        xl_min, xl_max = int(f.xlines[0]), int(f.xlines[-1])
        sample_rate = float(f.samples[1] - f.samples[0])
        t_min = float(f.samples[0])
        t_max = float(f.samples[-1])

    logger.info("Seismic shape: %s", seismic.shape)

    # Load fault sticks
    fault_dir = Path("data/volve/interpretations/fault_sticks")
    if fault_dir.exists():
        sticks = load_volve_fault_sticks(fault_dir)
        logger.info("Loaded %d fault sticks", len(sticks))
        fault_mask = fault_sticks_to_volume(
            sticks,
            volume_shape=(n_il, n_xl, n_s),
            inline_range=(il_min, il_max),
            crossline_range=(xl_min, xl_max),
            sample_range=(t_min, t_max),
            sample_rate_ms=sample_rate,
        ).astype(np.float32)
    else:
        logger.warning("No fault sticks found at %s", fault_dir)
        fault_mask = np.zeros_like(seismic)

    metadata = {
        "source": str(segy_path),
        "shape": seismic.shape,
        "inline_range": (il_min, il_max),
        "crossline_range": (xl_min, xl_max),
        "sample_rate_ms": sample_rate,
        "n_fault_sticks": len(sticks) if fault_dir.exists() else 0,
    }

    return seismic, fault_mask, metadata


def run_inference(
    seismic: np.ndarray,
    checkpoint_path: Path | None = None,
    patch_size: tuple[int, int, int] = (32, 32, 32),
    stride: tuple[int, int, int] = (16, 16, 16),
) -> np.ndarray:
    """Run UNet3D inference on the seismic volume.

    Uses sliding window with overlap and averages predictions.

    Args:
        seismic: Input volume, shape (D, H, W).
        checkpoint_path: Path to trained model checkpoint. If None,
                         uses untrained model (for pipeline testing).
        patch_size: Inference patch dimensions.
        stride: Step between patches.

    Returns:
        Prediction volume (probabilities), same shape as input.
    """
    import torch

    from deepseismic.models.unet import UNet3D, build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load or create model
    if checkpoint_path and checkpoint_path.exists():
        logger.info("Loading checkpoint: %s", checkpoint_path)
        model = UNet3D()
        model.load_checkpoint(str(checkpoint_path))
    else:
        logger.info("No checkpoint — using untrained model (pipeline test)")
        model = build_model(init_features=16, depth=3)

    model = model.to(device)
    model.eval()

    # Sliding window inference
    d, h, w = seismic.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    prediction = np.zeros_like(seismic, dtype=np.float32)
    counts = np.zeros_like(seismic, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, max(1, d - pd + 1), sd):
            for j in range(0, max(1, h - ph + 1), sh):
                for k in range(0, max(1, w - pw + 1), sw):
                    # Clamp to volume bounds
                    ie = min(i + pd, d)
                    je = min(j + ph, h)
                    ke = min(k + pw, w)
                    i_s = ie - pd
                    j_s = je - ph
                    k_s = ke - pw

                    patch = seismic[i_s:ie, j_s:je, k_s:ke]
                    x = (
                        torch.from_numpy(patch)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    y = torch.sigmoid(model(x))
                    pred_patch = y.squeeze().cpu().numpy()

                    prediction[i_s:ie, j_s:je, k_s:ke] += pred_patch
                    counts[i_s:ie, j_s:je, k_s:ke] += 1.0

    # Average overlapping predictions
    counts = np.maximum(counts, 1.0)
    prediction /= counts

    return prediction


def save_comparison_image(
    seismic: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    inline_idx: int | None = None,
) -> None:
    """Save a side-by-side comparison of GT vs prediction.

    Args:
        seismic: Seismic volume.
        ground_truth: Binary fault mask.
        prediction: Model predictions (probabilities).
        output_path: Where to save the PNG.
        inline_idx: Which inline to display. Defaults to middle.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if inline_idx is None:
        inline_idx = seismic.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1e293b")

    # Seismic
    axes[0].imshow(
        seismic[inline_idx].T,
        aspect="auto",
        cmap="seismic",
        vmin=-np.percentile(np.abs(seismic), 95),
        vmax=np.percentile(np.abs(seismic), 95),
    )
    axes[0].set_title("Seismic (Input)", color="white")

    # Ground truth overlay
    axes[1].imshow(
        seismic[inline_idx].T, aspect="auto", cmap="gray_r", alpha=0.5
    )
    axes[1].imshow(
        ground_truth[inline_idx].T,
        aspect="auto",
        cmap="Reds",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground Truth Faults", color="white")

    # Prediction overlay
    axes[2].imshow(
        seismic[inline_idx].T, aspect="auto", cmap="gray_r", alpha=0.5
    )
    axes[2].imshow(
        prediction[inline_idx].T,
        aspect="auto",
        cmap="hot",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Model Predictions", color="white")

    for ax in axes:
        ax.set_xlabel("Crossline", color="#94a3b8")
        ax.set_ylabel("Time (samples)", color="#94a3b8")
        ax.tick_params(colors="#64748b")

    fig.suptitle(
        f"Validation Comparison — Inline {inline_idx}",
        color="white",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved comparison image: %s", output_path)


def main():
    """Run the full validation demo."""
    parser = argparse.ArgumentParser(
        description="Volve validation demo: ingest → train → predict → evaluate"
    )
    parser.add_argument(
        "--segy",
        type=Path,
        default=None,
        help="Path to real Volve SEGY file (uses synthetic if not provided)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
        help="Model checkpoint to use for inference",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=0,
        help="Train for N epochs before inference (0=skip training)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/validation"),
        help="Directory for output images and reports",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for binarization",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  DeepSeismic2 — Volve Validation Demo")
    print("=" * 60)
    print()

    # Step 1: Load data
    seismic, fault_mask, metadata = generate_or_load_data(args.segy)
    logger.info(
        "Data loaded: shape=%s, source=%s",
        metadata["shape"],
        metadata["source"],
    )
    logger.info(
        "Ground truth: %d fault voxels (%.2f%% of volume)",
        fault_mask.sum(),
        fault_mask.mean() * 100,
    )

    # Step 2: Train if requested
    if args.train_epochs > 0:
        logger.info("Training for %d epochs...", args.train_epochs)
        from deepseismic.training.train import TrainConfig, train

        config = TrainConfig(
            epochs=args.train_epochs,
            batch_size=4,
            checkpoint_dir=args.checkpoint.parent,
        )
        args.checkpoint = train(config)
        logger.info("Training complete. Checkpoint: %s", args.checkpoint)

    # Step 3: Run inference
    logger.info("Running inference...")
    checkpoint = args.checkpoint if args.checkpoint.exists() else None
    prediction = run_inference(seismic, checkpoint)
    logger.info(
        "Inference complete. Prediction range: [%.3f, %.3f]",
        prediction.min(),
        prediction.max(),
    )

    # Step 4: Evaluate
    from deepseismic.validation import evaluate_model

    logger.info("Computing validation metrics...")
    metrics = evaluate_model(prediction, fault_mask, threshold=args.threshold)

    # Step 5: Report
    print()
    print(metrics.summary())
    print()

    # Step 6: Save comparison image
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_image(
        seismic,
        fault_mask,
        prediction,
        args.output_dir / "validation_comparison.png",
    )

    # Save metrics as JSON
    import json

    metrics_dict = {
        "iou": metrics.iou,
        "dice": metrics.dice,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "tolerant_precision_3": metrics.tolerant_precision_3,
        "tolerant_recall_3": metrics.tolerant_recall_3,
        "tolerant_precision_5": metrics.tolerant_precision_5,
        "tolerant_recall_5": metrics.tolerant_recall_5,
        "mean_surface_distance": metrics.mean_distance_to_true_fault,
        "volume_shape": list(metrics.volume_shape),
        "metadata": metadata,
    }
    report_path = args.output_dir / "validation_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    logger.info("Saved metrics report: %s", report_path)

    print()
    print("Demo complete! Outputs saved to:", args.output_dir)
    print()

    # Return exit code based on quality
    if metrics.iou > 0.3:
        print("PASS: Model exceeds minimum IoU threshold (0.3)")
        return 0
    elif checkpoint is None:
        print("INFO: No trained checkpoint — metrics reflect untrained model")
        return 0
    else:
        print("WARN: Model below minimum IoU threshold (0.3)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
