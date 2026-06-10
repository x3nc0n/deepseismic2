"""Training pipeline for the 3D UNet fault detection model.

Orchestrates: label generation -> patch extraction -> model training -> checkpoint.
Supports both real Volve data and synthetic sample data for local validation.

Usage:
    # Train on synthetic sample (local dev, no GPU needed)
    python -m deepseismic.training.train --epochs 5

    # Train on real Volve data (GPU recommended)
    python -m deepseismic.training.train --epochs 50 --device cuda

    # Resume from checkpoint
    python -m deepseismic.training.train --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patch_size: tuple[int, int, int] = (32, 32, 32)
    stride: tuple[int, int, int] = (16, 16, 16)
    init_features: int = 16
    depth: int = 3
    dropout_p: float = 0.1
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 5
    pos_weight: float = 10.0  # Handle class imbalance (faults are sparse)


def generate_synthetic_training_data(
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic volume + fault mask for training validation.

    Creates a small volume with known fault geometry so we can verify
    the model learns _something_ even without real data.
    """
    rng = np.random.default_rng(42)
    shape = (96, 128, 128)  # Enough for meaningful train/val split

    # Layered reflectivity
    volume = np.zeros(shape, dtype=np.float32)
    for layer_z in [20, 40, 60, 80, 100, 115]:
        volume[:, :, layer_z] = rng.uniform(-0.2, 0.2)

    # Convolve each trace with a Ricker wavelet
    def _ricker(points: int, a: float) -> np.ndarray:
        """Ricker wavelet (Mexican hat)."""
        t = np.arange(points) - (points - 1) / 2
        norm = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
        tsq = (t / a) ** 2
        return norm * (1 - tsq) * np.exp(-tsq / 2)

    wavelet = _ricker(32, 4)
    for il in range(shape[0]):
        for xl in range(shape[1]):
            volume[il, xl, :] = np.convolve(
                volume[il, xl, :], wavelet, mode="same"
            )

    # Add noise
    volume += rng.normal(0, 0.01, shape).astype(np.float32)

    # Create fault mask - a planar fault cutting through the volume
    mask = np.zeros(shape, dtype=np.uint8)
    for il in range(shape[0]):
        # Fault at crossline ~60, dipping in inline direction
        fault_xl = int(60 + il * 0.3)
        if 0 <= fault_xl < shape[1]:
            mask[il, fault_xl, 20:110] = 1
            # Dilate slightly
            if fault_xl + 1 < shape[1]:
                mask[il, fault_xl + 1, 20:110] = 1

    # Apply fault throw to the volume (shift one side)
    for il in range(shape[0]):
        fault_xl = int(60 + il * 0.3)
        if fault_xl < shape[1]:
            volume[il, fault_xl + 2:, :] = np.roll(
                volume[il, fault_xl + 2:, :], shift=5, axis=1
            )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "volume.npy", volume)
    np.save(output_dir / "mask.npy", mask)

    logger.info(
        "Synthetic data: volume=%s, mask=%s, fault_fraction=%.4f",
        volume.shape,
        mask.shape,
        mask.sum() / mask.size,
    )
    return volume, mask


class NumpyPatchDataset(torch.utils.data.Dataset):
    """Simple patch dataset from numpy arrays (for synthetic training)."""

    def __init__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        patch_size: tuple[int, int, int] = (32, 32, 32),
        stride: tuple[int, int, int] = (16, 16, 16),
        normalize: bool = True,
    ):
        self.volume = volume
        self.mask = mask
        self.patch_size = patch_size
        self.normalize = normalize

        # Build patch index
        self.patches: list[tuple[int, int, int]] = []
        for i in range(
            0, volume.shape[0] - patch_size[0] + 1, stride[0]
        ):
            for j in range(
                0, volume.shape[1] - patch_size[1] + 1, stride[1]
            ):
                for k in range(
                    0, volume.shape[2] - patch_size[2] + 1, stride[2]
                ):
                    self.patches.append((i, j, k))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i, j, k = self.patches[idx]
        ps = self.patch_size

        seismic = self.volume[
            i:i + ps[0], j:j + ps[1], k:k + ps[2]
        ].copy()
        label = self.mask[
            i:i + ps[0], j:j + ps[1], k:k + ps[2]
        ].copy()

        if self.normalize:
            std = seismic.std()
            if std > 1e-8:
                seismic = (seismic - seismic.mean()) / std

        # Add channel dimension: (1, D, H, W)
        seismic_t = torch.from_numpy(seismic).unsqueeze(0)
        label_t = torch.from_numpy(
            label.astype(np.float32)
        ).unsqueeze(0)

        return seismic_t, label_t


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute IoU and Dice for binary segmentation."""
    with torch.no_grad():
        probs = torch.sigmoid(preds)
        binary = (probs > threshold).float()

        intersection = (binary * targets).sum()
        union = binary.sum() + targets.sum() - intersection

        iou = (intersection / (union + 1e-8)).item()
        dice = (
            2 * intersection / (binary.sum() + targets.sum() + 1e-8)
        ).item()
        precision = (intersection / (binary.sum() + 1e-8)).item()
        recall = (intersection / (targets.sum() + 1e-8)).item()

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_metrics: dict[str, float] = {"iou": 0, "dice": 0}
    n_batches = 0

    for seismic, labels in loader:
        seismic = seismic.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(seismic)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metrics = compute_metrics(logits, labels)
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {
        k: v / max(n_batches, 1) for k, v in total_metrics.items()
    }
    return {"loss": avg_loss, **avg_metrics}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_metrics: dict[str, float] = {"iou": 0, "dice": 0}
    n_batches = 0

    for seismic, labels in loader:
        seismic = seismic.to(device)
        labels = labels.to(device)

        logits = model(seismic)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        metrics = compute_metrics(logits, labels)
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {
        k: v / max(n_batches, 1) for k, v in total_metrics.items()
    }
    return {"loss": avg_loss, **avg_metrics}


def train(config: TrainConfig) -> Path:
    """Main training loop. Returns path to best checkpoint."""
    device = torch.device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training config: %s", config)
    logger.info("Device: %s", device)

    # Generate or load data
    data_dir = Path("data/training")
    vol_path = data_dir / "volume.npy"
    mask_path = data_dir / "mask.npy"

    if vol_path.exists() and mask_path.exists():
        logger.info("Loading existing training data from %s", data_dir)
        volume = np.load(vol_path)
        mask = np.load(mask_path)
    else:
        logger.info("Generating synthetic training data...")
        volume, mask = generate_synthetic_training_data(data_dir)

    # Split: 70% train, 15% val, 15% test (along inline axis)
    n_il = volume.shape[0]
    train_end = int(n_il * 0.7)
    val_end = int(n_il * 0.85)

    train_vol, train_mask = volume[:train_end], mask[:train_end]
    val_vol, val_mask = volume[train_end:val_end], mask[train_end:val_end]

    logger.info(
        "Split: train=%d, val=%d, test=%d inlines",
        train_end, val_end - train_end, n_il - val_end,
    )

    train_ds = NumpyPatchDataset(
        train_vol, train_mask, config.patch_size, config.stride
    )
    val_ds = NumpyPatchDataset(
        val_vol, val_mask, config.patch_size, config.stride
    )

    logger.info(
        "Train patches: %d, Val patches: %d", len(train_ds), len(val_ds)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Build model
    from deepseismic.models.unet import build_model

    model = build_model(
        init_features=config.init_features,
        depth=config.depth,
        dropout_p=config.dropout_p,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{param_count:,}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Weighted BCE for class imbalance
    pos_weight = torch.tensor([config.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_val_iou = 0.0
    best_checkpoint = config.checkpoint_dir / "best.pt"

    print(
        f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} "
        f"{'Train IoU':>10} {'Val IoU':>8} {'Val Dice':>9} "
        f"{'LR':>10}"
    )
    print("-" * 72)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>5d} {train_metrics['loss']:>11.4f} "
            f"{val_metrics['loss']:>9.4f} "
            f"{train_metrics['iou']:>10.4f} "
            f"{val_metrics['iou']:>8.4f} "
            f"{val_metrics['dice']:>9.4f} "
            f"{lr:>10.2e}"
        )

        # Save best
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            model.save_checkpoint(
                str(best_checkpoint),
                epoch=epoch,
                metrics=val_metrics,
            )

        # Periodic save
        if epoch % config.save_every == 0:
            ckpt = config.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            model.save_checkpoint(
                str(ckpt), epoch=epoch, metrics=val_metrics
            )

    # Save final
    final_ckpt = config.checkpoint_dir / "latest.pt"
    model.save_checkpoint(
        str(final_ckpt), epoch=config.epochs, metrics=val_metrics
    )

    print(f"\nTraining complete. Best val IoU: {best_val_iou:.4f}")
    print(f"   Best checkpoint: {best_checkpoint}")
    print(f"   Latest checkpoint: {final_ckpt}")

    return best_checkpoint


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train 3D UNet for seismic fault detection",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--features", type=int, default=16,
        help="Initial UNet features (default: 16, use 32+ for GPU)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        init_features=args.features,
    )

    train(config)


if __name__ == "__main__":
    main()
