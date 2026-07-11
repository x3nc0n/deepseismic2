"""3D patch extraction for seismic deep learning.

Provides :class:`PatchDataset`, a PyTorch ``Dataset`` that yields
``(seismic_patch, label_patch)`` tensors from a pair of Zarr arrays —
the amplitude volume and the binary fault-mask volume.

Design decisions
----------------
- **Spatial splits, not random splits.**  Train / val / test boundaries are
  set along the inline axis so that spatially adjacent patches never straddle
  a split boundary.  This prevents data leakage caused by overlapping patches
  between splits.
- **Per-patch normalisation.**  Each seismic patch is independently normalised
  to zero-mean unit-variance.  This accommodates amplitude variation across
  the survey without needing a precomputed global statistics file.
- **Lazy reads.**  Patches are read on demand in ``__getitem__`` from a Zarr
  array or a pre-cached in-memory ``np.ndarray``.
- **Unlabelled-patch filtering.**  ``PatchConfig.min_fault_fraction`` lets
  you drop patches with fewer than a threshold fraction of fault voxels,
  which helps balance training on sparse fault labels.

Usage
-----
    from deepseismic.preprocessing.patches import PatchConfig, PatchDataset, Split

    config = PatchConfig(patch_size=(64, 64, 64), stride=(32, 32, 32))
    train_ds = PatchDataset(
        "data/staged/ST10010.zarr",
        "data/staged/fault_mask.zarr",
        config=config,
        split=Split.TRAIN,
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)

    for seismic_t, label_t in loader:
        # seismic_t : (B, 1, 64, 64, 64) float32
        # label_t   : (B, 1, 64, 64, 64) float32  (0/1 binary)
        ...
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_DEFAULT_SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Split(StrEnum):
    """Dataset subset identifier."""

    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"


@dataclass
class PatchConfig:
    """Configuration for 3D patch extraction.

    Parameters
    ----------
    patch_size:
        Spatial extent of each patch ``(n_inline, n_crossline, n_sample)``.
    stride:
        Step between patch origins.  Defaults to ``patch_size`` (no overlap).
        Use smaller strides (e.g., half patch_size) for denser sampling during
        inference.
    split_fractions:
        Relative sizes of the train / val / test splits along the inline axis.
        Must sum to 1.0.
    min_fault_fraction:
        Fraction of labelled (fault=1) voxels required to include a patch in
        the index.  ``0.0`` keeps all patches; ``0.001`` drops patches with
        fewer than 0.1 % fault voxels.  Note: filtering is applied at
        **index-build time**, so it may be slow on very large volumes.
    normalize:
        If True, apply per-patch zero-mean / unit-variance normalisation.
    """

    patch_size:           tuple[int, int, int]             = (64, 64, 64)
    stride:               tuple[int, int, int] | None      = None
    split_fractions:      tuple[float, float, float]       = _DEFAULT_SPLIT_FRACTIONS
    min_fault_fraction:   float                            = 0.0
    normalize:            bool                             = True

    def __post_init__(self) -> None:
        if self.stride is None:
            self.stride = self.patch_size
        total = sum(self.split_fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split_fractions must sum to 1.0, got {total:.6f}"
            )


# ---------------------------------------------------------------------------
# Patch index
# ---------------------------------------------------------------------------


@dataclass
class _PatchIndex:
    """Location descriptor for one patch (0-based voxel offsets)."""

    il_start: int
    xl_start: int
    s_start:  int
    split:    Split


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PatchDataset(Dataset):
    """PyTorch Dataset over 3D seismic patches with spatial train/val/test splits.

    Parameters
    ----------
    seismic_zarr:
        Path to the seismic amplitude Zarr store (contains dataset ``amplitude``)
        *or* an already-open ``zarr.Array`` / ``np.ndarray``.
    label_zarr:
        Path to the fault mask Zarr store (contains dataset ``fault_mask``)
        *or* an already-open ``zarr.Array`` / ``np.ndarray``.  Pass ``None`` to return
        all-zero labels (useful for unlabelled inference datasets).
    config:
        :class:`PatchConfig`.  Defaults to 64³ patches, full-stride.
    split:
        Which subset to expose (``TRAIN`` / ``VAL`` / ``TEST``).
    transform:
        Optional callable applied to the seismic ``torch.Tensor`` after
        patch extraction and normalisation.
    label_transform:
        Optional callable applied to the label ``torch.Tensor``.
    """

    def __init__(
        self,
        seismic_zarr: str | Path | zarr.Array | np.ndarray,
        label_zarr:   str | Path | zarr.Array | np.ndarray | None,
        config:        PatchConfig | None     = None,
        split:         Split                  = Split.TRAIN,
        transform:     Callable | None        = None,
        label_transform: Callable | None      = None,
    ) -> None:
        self.config          = config or PatchConfig()
        self.split           = split
        self.transform       = transform
        self.label_transform = label_transform

        self._seismic = _open_zarr_array(seismic_zarr, "amplitude")
        self._labels  = (
            None
            if label_zarr is None
            else _open_zarr_array(label_zarr, "fault_mask")
        )

        vol_shape: tuple[int, int, int] = self._seismic.shape[:3]  # type: ignore[assignment]
        ps = self.config.patch_size
        st = self.config.stride  # guaranteed non-None after __post_init__

        # Validate patch vs volume sizes
        for dim, (vs, p, _s) in enumerate(zip(vol_shape, ps, st, strict=False)):  # type: ignore[arg-type]
            if p > vs:
                raise ValueError(
                    f"patch_size[{dim}]={p} exceeds volume dim {dim} size {vs}."
                )

        all_patches = self._build_index(vol_shape)
        self._patches = [p for p in all_patches if p.split == split]

        logger.info(
            "PatchDataset [%s]: %d patches  (volume %s, patch %s, stride %s)",
            split.value, len(self._patches), vol_shape, ps, st,
        )

    # --- Dataset protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._patches[idx]
        ps_il, ps_xl, ps_s = self.config.patch_size

        seismic_np: np.ndarray = self._seismic[
            p.il_start : p.il_start + ps_il,
            p.xl_start : p.xl_start + ps_xl,
            p.s_start  : p.s_start  + ps_s,
        ].astype(np.float32)

        if self._labels is not None:
            label_np: np.ndarray = self._labels[
                p.il_start : p.il_start + ps_il,
                p.xl_start : p.xl_start + ps_xl,
                p.s_start  : p.s_start  + ps_s,
            ].astype(np.float32)
        else:
            label_np = np.zeros(self.config.patch_size, dtype=np.float32)

        if self.config.normalize:
            seismic_np = _normalize_patch(seismic_np)

        # Add channel dimension: (1, IL, XL, S)
        seismic_t = torch.from_numpy(seismic_np).unsqueeze(0)
        label_t   = torch.from_numpy(label_np).unsqueeze(0)

        if self.transform:
            seismic_t = self.transform(seismic_t)
        if self.label_transform:
            label_t = self.label_transform(label_t)

        return seismic_t, label_t

    # --- index construction ------------------------------------------------

    def _build_index(self, vol_shape: tuple[int, int, int]) -> list[_PatchIndex]:
        n_il, n_xl, n_s = vol_shape
        ps_il, ps_xl, ps_s = self.config.patch_size
        st_il, st_xl, st_s = self.config.stride  # type: ignore[misc]

        frac_train, frac_val, _ = self.config.split_fractions
        il_train_end = int(n_il * frac_train)
        il_val_end   = il_train_end + int(n_il * frac_val)

        indices: list[_PatchIndex] = []

        for il in range(0, n_il - ps_il + 1, st_il):
            il_centre = il + ps_il // 2
            if il_centre < il_train_end:
                spl = Split.TRAIN
            elif il_centre < il_val_end:
                spl = Split.VAL
            else:
                spl = Split.TEST

            for xl in range(0, n_xl - ps_xl + 1, st_xl):
                for s in range(0, n_s - ps_s + 1, st_s):

                    # Fault-fraction filter (optional; can be slow on large volumes)
                    if self.config.min_fault_fraction > 0.0 and self._labels is not None:
                        lbl_patch = self._labels[
                            il : il + ps_il,
                            xl : xl + ps_xl,
                            s  : s  + ps_s,
                        ]
                        if float(lbl_patch.mean()) < self.config.min_fault_fraction:
                            continue

                    indices.append(_PatchIndex(il, xl, s, spl))

        return indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_zarr_array(
    src: str | Path | zarr.Array | np.ndarray,
    dataset_name: str,
) -> zarr.Array | np.ndarray:
    """Open a named dataset from a Zarr store or pass through an existing Array."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, zarr.Array):
        return src
    root = zarr.open_group(str(src), mode="r")
    if dataset_name not in root:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in Zarr store '{src}'.  "
            f"Available: {list(root.keys())}"
        )
    return root[dataset_name]


def _normalize_patch(patch: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean unit-variance normalisation, robust to near-flat patches."""
    mean = float(patch.mean())
    std  = float(patch.std())
    return (patch - mean) / (std + eps)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_dataloaders(
    seismic_zarr:  str | Path,
    label_zarr:    str | Path | None,
    config:        PatchConfig | None = None,
    batch_size:    int               = 4,
    num_workers:   int               = 0,
    **loader_kwargs,
) -> dict[str, DataLoader]:
    """Build DataLoaders for all three splits in a single call.

    Parameters
    ----------
    seismic_zarr:
        Path to the seismic amplitude Zarr store.
    label_zarr:
        Path to the fault mask Zarr store.  Pass ``None`` to yield zero labels
        (unlabelled use-case).
    config:
        :class:`PatchConfig`.  Defaults to ``PatchConfig()``.
    batch_size:
        Mini-batch size.
    num_workers:
        DataLoader worker count.  ``0`` runs in the main process — recommended
        when reading from Zarr (avoids file-handle duplication issues).
    **loader_kwargs:
        Forwarded to :class:`torch.utils.data.DataLoader`.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    cfg = config or PatchConfig()
    loaders: dict[str, DataLoader] = {}

    for spl in Split:
        ds = PatchDataset(seismic_zarr, label_zarr, config=cfg, split=spl)
        loaders[spl.value] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(spl == Split.TRAIN),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **loader_kwargs,
        )

    return loaders
