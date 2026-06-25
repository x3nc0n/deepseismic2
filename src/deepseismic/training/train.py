"""Training pipeline for the 3D UNet fault detection model.

Orchestrates: label generation -> patch extraction -> model training -> checkpoint.
Supports both real Volve data (zarr mode) and synthetic sample data.

Usage:
    # Train on synthetic sample (local dev, no GPU needed)
    python -m deepseismic.training.train --epochs 5

    # Train on real Volve fault_label.zarr (PoC)
    python -m deepseismic.training.train --data-mode zarr --epochs 10

    # Resume from checkpoint
    python -m deepseismic.training.train --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
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

    # Core training
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
    # Imbalance: synthetic default; zarr default overridden in train()
    pos_weight: float = 10.0

    # S2-02: data mode
    data_mode: str = "synthetic"  # "synthetic" | "zarr"
    seismic_zarr: Path = Path("data/volve/staged/synthetic.zarr")
    label_zarr: Path = Path("data/volve/staged/fault_label.zarr")
    # Zarr-mode patch filtering: keep only patches with ≥ this fault fraction
    # 0.00003 ≈ 1 fault voxel out of 32³=32768 — ensures every patch has fault signal
    min_fault_fraction: float = 0.00003

    # S2-05: reproducibility seed
    seed: int = 42


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


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss — robust to class imbalance without requiring pos_weight tuning."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def _accum_tp_fp_fn(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return (TP, FP, FN) counts for epoch-level metric accumulation (S2-08)."""
    with torch.no_grad():
        binary = (torch.sigmoid(preds) > threshold).float()
        tp = (binary * targets).sum().item()
        fp = (binary * (1.0 - targets)).sum().item()
        fn = ((1.0 - binary) * targets).sum().item()
    return tp, fp, fn


def _epoch_metrics(tp: float, fp: float, fn: float) -> dict[str, float]:
    """Compute IoU and Dice from epoch-level TP/FP/FN accumulators (S2-08).

    Accumulating raw counts across batches then computing the ratio once is
    the correct method for sparse labels — per-batch averaging biases the
    result toward 0 when most batches contain zero positive labels.
    """
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_dice: bool = False,
) -> dict[str, float]:
    """Run one training epoch with epoch-level metric accumulation (S2-08)."""
    model.train()
    total_loss = 0.0
    total_tp = total_fp = total_fn = 0.0
    n_batches = 0

    for seismic, labels in loader:
        seismic = seismic.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(seismic)
        loss = criterion(logits, labels)
        if use_dice:
            loss = 0.5 * loss + 0.5 * _dice_loss(logits, labels)
        loss.backward()
        optimizer.step()

        tp, fp, fn = _accum_tp_fp_fn(logits, labels)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"loss": avg_loss, **_epoch_metrics(total_tp, total_fp, total_fn)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_dice: bool = False,
) -> dict[str, float]:
    """Run validation with epoch-level metric accumulation (S2-08)."""
    model.eval()
    total_loss = 0.0
    total_tp = total_fp = total_fn = 0.0
    n_batches = 0

    for seismic, labels in loader:
        seismic = seismic.to(device)
        labels = labels.to(device)

        logits = model(seismic)
        loss = criterion(logits, labels)
        if use_dice:
            loss = 0.5 * loss + 0.5 * _dice_loss(logits, labels)

        tp, fp, fn = _accum_tp_fp_fn(logits, labels)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"loss": avg_loss, **_epoch_metrics(total_tp, total_fp, total_fn)}


def train(config: TrainConfig) -> Path:
    """Main training loop. Returns path to best checkpoint."""
    # S2-05: Reproducibility — seed everything before any data or model ops
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training config: %s", config)
    logger.info("Device: %s", device)

    # S2-05: Persist resolved config to disk immediately
    config_dict = dataclasses.asdict(config)
    run_config_path = config.checkpoint_dir / "run_config.json"
    with open(run_config_path, "w") as fh:
        json.dump(config_dict, fh, indent=2, default=str)
    logger.info("Run config saved → %s", run_config_path)

    # S2-02: Branch on data mode
    use_dice = False  # combined BCE+Dice flag
    if config.data_mode == "zarr":
        train_loader, val_loader = _build_zarr_loaders(config)
        use_dice = True  # always use combined loss for real sparse labels
    else:
        train_loader, val_loader = _build_synthetic_loaders(config)

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

    pos_weight = torch.tensor([config.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_val_iou = 0.0
    best_checkpoint = config.checkpoint_dir / "best.pt"

    # Val metrics from last epoch (used for final checkpoint save below loop)
    val_metrics: dict[str, float] = {}

    print(
        f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} "
        f"{'Train IoU':>10} {'Val IoU':>8} {'Val Dice':>9} "
        f"{'LR':>10}"
    )
    print("-" * 72)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, use_dice=use_dice
        )
        val_metrics = validate(model, val_loader, criterion, device, use_dice=use_dice)
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

        # Save best checkpoint with REAL metrics (S2-08) + config+seed (S2-05)
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            ckpt_payload = {
                **val_metrics,
                "seed": config.seed,
                "train_config": config_dict,
            }
            model.save_checkpoint(
                str(best_checkpoint),
                epoch=epoch,
                metrics=ckpt_payload,
            )

        # Periodic save
        if epoch % config.save_every == 0:
            ckpt = config.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            model.save_checkpoint(
                str(ckpt), epoch=epoch, metrics=val_metrics
            )

    # Save final checkpoint
    final_ckpt = config.checkpoint_dir / "latest.pt"
    final_payload = {
        **val_metrics,
        "seed": config.seed,
        "train_config": config_dict,
    }
    model.save_checkpoint(
        str(final_ckpt), epoch=config.epochs, metrics=final_payload
    )

    print(f"\nTraining complete. Best val IoU: {best_val_iou:.4f}")
    print(f"   Best checkpoint: {best_checkpoint}")
    print(f"   Latest checkpoint: {final_ckpt}")
    print(f"   Run config: {run_config_path}")

    return best_checkpoint


# ---------------------------------------------------------------------------
# Data loader builders
# ---------------------------------------------------------------------------


def _build_zarr_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from Zarr amplitude + fault_label volumes.

    Class-imbalance strategy for 0.08% fault fraction (S2-02):
    - All patches included (min_fault_fraction=0) so the model sees background.
    - Fault-containing patches are oversampled 50× via WeightedRandomSampler
      so every training batch contains at least one fault patch in expectation.
    - BCEWithLogitsLoss(pos_weight=200) + soft Dice loss (combined 50/50):
        * BCE pos_weight=200 (≤ capped neg/pos ratio ~1255) penalises missed faults.
        * Dice loss adds precision pressure — penalises predicting fault everywhere.
    This combination avoids the all-positive collapse from BCE-only high pos_weight
    while still driving recall on the sparse fault class.
    """
    from torch.utils.data import WeightedRandomSampler

    from deepseismic.preprocessing.patches import PatchConfig, PatchDataset, Split

    patch_cfg = PatchConfig(
        patch_size=config.patch_size,
        stride=config.stride,
        min_fault_fraction=0.0,  # include ALL patches (negative + positive)
    )

    train_ds = PatchDataset(
        config.seismic_zarr,
        config.label_zarr,
        config=patch_cfg,
        split=Split.TRAIN,
    )
    val_ds = PatchDataset(
        config.seismic_zarr,
        config.label_zarr,
        config=patch_cfg,
        split=Split.VAL,
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "No training patches found. Check zarr paths and volume size."
        )

    # Build fault-aware sampling weights: scan label zarr for each train patch.
    # Fault patches get 50× weight → ~1-2 fault patches per batch of 4 on average.
    import zarr as _zarr
    _label_root = _zarr.open_group(str(config.label_zarr), mode="r")
    _fault_arr = _label_root["fault_mask"]
    ps = config.patch_size

    sample_weights: list[float] = []
    n_fault_patches = 0
    for p in train_ds._patches:
        lbl_sum = int(
            _fault_arr[
                p.il_start : p.il_start + ps[0],
                p.xl_start : p.xl_start + ps[1],
                p.s_start  : p.s_start  + ps[2],
            ].sum()
        )
        if lbl_sum > 0:
            sample_weights.append(50.0)
            n_fault_patches += 1
        else:
            sample_weights.append(1.0)

    logger.info(
        "Zarr loaders — train=%d patches (%d with fault), val=%d patches",
        len(train_ds), n_fault_patches, len(val_ds),
    )
    logger.info(
        "WeightedRandomSampler: fault patches weight=50, background weight=1"
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=min(200, len(sample_weights)),  # 50 batches/epoch on CPU
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def _build_synthetic_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from synthetic numpy data (backward compat)."""
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

    n_il = volume.shape[0]
    train_end = int(n_il * 0.7)
    val_end = int(n_il * 0.85)

    logger.info(
        "Split: train=%d, val=%d, test=%d inlines",
        train_end, val_end - train_end, n_il - val_end,
    )

    train_ds = NumpyPatchDataset(
        volume[:train_end], mask[:train_end], config.patch_size, config.stride
    )
    val_ds = NumpyPatchDataset(
        volume[train_end:val_end], mask[train_end:val_end],
        config.patch_size, config.stride,
    )

    logger.info("Train patches: %d, Val patches: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


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
    # S2-02: data mode
    parser.add_argument(
        "--data-mode", choices=["synthetic", "zarr"], default="synthetic",
        help="Data source: synthetic (default) or zarr (real Volve data)",
    )
    parser.add_argument(
        "--seismic-zarr", default="data/volve/staged/synthetic.zarr",
        help="Path to seismic amplitude Zarr store (zarr mode only)",
    )
    parser.add_argument(
        "--label-zarr", default="data/volve/staged/fault_label.zarr",
        help="Path to fault label Zarr store (zarr mode only)",
    )
    parser.add_argument(
        "--pos-weight", type=float, default=None,
        help="BCE pos_weight for class imbalance. "
             "Defaults: synthetic=10, zarr=200 (capped neg/pos ratio)",
    )
    # S2-05: seed
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory for checkpoints and run_config.json",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Default pos_weight depends on data mode
    # zarr: neg/pos ≈ 1255, capped at 200 to avoid numeric instability
    if args.pos_weight is not None:
        pos_weight = args.pos_weight
    elif args.data_mode == "zarr":
        pos_weight = 200.0
    else:
        pos_weight = 10.0

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        init_features=args.features,
        data_mode=args.data_mode,
        seismic_zarr=Path(args.seismic_zarr),
        label_zarr=Path(args.label_zarr),
        pos_weight=pos_weight,
        seed=args.seed,
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    train(config)


if __name__ == "__main__":
    main()
