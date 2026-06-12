"""Full-volume inference for the deepseismic2 UNet.

Takes a trained :class:`~deepseismic.models.unet.UNet3D`, slides a 3D window
over a Zarr amplitude volume, blends overlapping predictions with a Gaussian
kernel, and writes the probability volume plus a thresholded binary mask.

Design decisions
----------------
- **Gaussian overlap-blending** reduces tiling artifacts that arise with
  hard-edge averaging: each prediction is weighted by a 3D Gaussian centred
  on the patch, so central predictions receive full weight and boundary
  predictions taper to near-zero.
- **Batch inference** groups multiple patches into one forward pass to
  utilise GPU compute efficiently.
- **Lazy Zarr reads** avoid loading the full volume into RAM.
- **CPU and CUDA** are both supported; ``device=None`` auto-detects CUDA.

Progress is reported via the standard ``logging`` module at ``INFO`` level
every 5 % of patches.

Usage
-----
    from deepseismic.models.inference import run_inference

    prob_vol, binary_mask = run_inference(
        seismic_zarr     = "data/staged/ST10010.zarr",
        checkpoint_path  = "checkpoints/best.pt",
        prob_output      = "results/fault_prob.zarr",
        mask_output      = "results/fault_mask.zarr",
        patch_size       = (64, 64, 64),
        overlap          = 0.25,
        batch_size       = 4,
        threshold        = 0.5,
        overwrite        = True,
    )
    print(f"Fault voxels: {binary_mask.sum():,} / {binary_mask.size:,}")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import zarr

from deepseismic.models.unet import UNet3D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian blending kernel
# ---------------------------------------------------------------------------


def _gaussian_kernel_3d(
    size: tuple[int, int, int],
    sigma: float | None = None,
) -> np.ndarray:
    """Generate a 3D Gaussian weighting kernel normalised to a peak of 1.0.

    Parameters
    ----------
    size:
        ``(D, H, W)`` kernel dimensions — must match the patch size.
    sigma:
        Gaussian sigma.  Defaults to ``min(size) / 4`` so the weight drops
        to ≈ e⁻² at the patch corners.
    """
    D, H, W = size
    if sigma is None:
        sigma = min(D, H, W) / 4.0

    kernel = np.ones(size, dtype=np.float32)
    for dim, n in enumerate(size):
        coords = np.arange(n) - (n - 1) / 2.0
        gauss  = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        shape  = [1, 1, 1]
        shape[dim] = n
        kernel *= gauss.reshape(shape)

    kernel /= kernel.max()
    return kernel


# ---------------------------------------------------------------------------
# Sliding-window position generator
# ---------------------------------------------------------------------------


def _sliding_windows(
    volume_shape: tuple[int, int, int],
    patch_size:   tuple[int, int, int],
    overlap:      float,
) -> Iterator[tuple[slice, slice, slice]]:
    """Yield ``(il_slice, xl_slice, s_slice)`` tuples for a sliding window.

    Parameters
    ----------
    volume_shape:
        ``(n_il, n_xl, n_s)``
    patch_size:
        ``(p_il, p_xl, p_s)``
    overlap:
        Fraction in ``[0, 1)`` of patch overlap between adjacent windows.
        ``0.0`` = no overlap; ``0.5`` = 50 % overlap.

    Raises
    ------
    ValueError
        If ``overlap`` is outside ``[0, 1)``.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap!r}")

    n_il, n_xl, n_s = volume_shape
    p_il, p_xl, p_s = patch_size
    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    st_il, st_xl, st_s = stride

    def _positions(n: int, p: int, s: int) -> list[int]:
        """Patch start positions ensuring the last patch reaches the end."""
        starts = list(range(0, n - p + 1, s))
        if not starts or starts[-1] + p < n:
            starts.append(n - p)
        return starts

    for il in _positions(n_il, p_il, st_il):
        for xl in _positions(n_xl, p_xl, st_xl):
            for s in _positions(n_s, p_s, st_s):
                yield (
                    slice(il, il + p_il),
                    slice(xl, xl + p_xl),
                    slice(s,  s  + p_s),
                )


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class VolumeInference:
    """Sliding-window inference over a 3D seismic volume.

    Parameters
    ----------
    model:
        Trained :class:`~deepseismic.models.unet.UNet3D` instance.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detects CUDA).
    patch_size:
        ``(D, H, W)`` sliding window dimensions.  Should match training patch
        size to avoid domain mismatch.
    overlap:
        Fraction of patch overlap ``[0, 0.9]``.  Higher values give smoother
        boundaries at the cost of more forward passes.
    batch_size:
        Number of patches per GPU forward pass.
    threshold:
        Probability cutoff for the binary mask output.
    """

    def __init__(
        self,
        model:      UNet3D,
        device:     str | torch.device | None = None,
        patch_size: tuple[int, int, int]      = (64, 64, 64),
        overlap:    float                     = 0.25,
        batch_size: int                       = 4,
        threshold:  float                     = 0.5,
    ) -> None:
        self.patch_size = patch_size
        self.overlap    = overlap
        self.batch_size = batch_size
        self.threshold  = threshold

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.model = model.to(self.device)
        self.model.eval()

        self._kernel = _gaussian_kernel_3d(patch_size)

        logger.info(
            "VolumeInference ready — device=%s  patch=%s  overlap=%.2f  threshold=%.2f",
            self.device, patch_size, overlap, threshold,
        )

    # --- main entry point --------------------------------------------------

    def run(
        self,
        seismic:           zarr.Array | np.ndarray,
        prob_output:       str | Path | None = None,
        mask_output:       str | Path | None = None,
        *,
        overwrite:         bool              = False,
        normalize_patches: bool              = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide the model over *seismic* and return probability + binary mask.

        Parameters
        ----------
        seismic:
            Amplitude array of shape ``(n_il, n_xl, n_s)``.  Accepts a
            ``zarr.Array`` (read lazily) or a pre-loaded ``np.ndarray``.
        prob_output:
            Optional Zarr directory path for the probability volume.
        mask_output:
            Optional Zarr directory path for the binary fault mask.
        overwrite:
            Overwrite existing Zarr stores.
        normalize_patches:
            Apply per-patch zero-mean unit-variance normalisation before
            inference.  Should be ``True`` if the model was trained with
            normalisation enabled.

        Returns
        -------
        prob_volume : np.ndarray
            Float32 ``(n_il, n_xl, n_s)`` fault probabilities.
        binary_mask : np.ndarray
            Uint8 ``(n_il, n_xl, n_s)`` thresholded fault mask.
        """
        vol_shape: tuple[int, int, int] = seismic.shape[:3]  # type: ignore[assignment]
        n_il, n_xl, n_s = vol_shape

        for dim, (vs, ps) in enumerate(zip(vol_shape, self.patch_size, strict=False)):
            if ps > vs:
                raise ValueError(
                    f"patch_size[{dim}]={ps} > volume dim {dim} size {vs}.  "
                    "Reduce patch_size or use a larger volume."
                )

        prob_acc   = np.zeros(vol_shape, dtype=np.float64)
        weight_acc = np.zeros(vol_shape, dtype=np.float64)

        all_windows = list(_sliding_windows(vol_shape, self.patch_size, self.overlap))
        n_patches   = len(all_windows)
        log_step    = max(1, n_patches // 20)   # log at every 5 %

        logger.info("Starting inference — %d patches total", n_patches)

        processed = 0
        for batch_start in range(0, n_patches, self.batch_size):
            batch_wins = all_windows[batch_start : batch_start + self.batch_size]

            # Build float32 batch tensor
            patches_np: list[np.ndarray] = []
            for sl_il, sl_xl, sl_s in batch_wins:
                patch = np.asarray(seismic[sl_il, sl_xl, sl_s], dtype=np.float32)
                if normalize_patches:
                    mean = float(patch.mean())
                    std  = float(patch.std())
                    patch = (patch - mean) / (std + 1e-8)
                patches_np.append(patch)

            # (B, 1, D, H, W)
            batch_t = torch.from_numpy(
                np.stack(patches_np)[:, np.newaxis]
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_t)            # (B, 1, D, H, W)
                probs  = torch.sigmoid(logits).cpu().numpy()  # (B, 1, D, H, W)

            # Weighted accumulation
            for (sl_il, sl_xl, sl_s), prob_patch in zip(batch_wins, probs[:, 0], strict=False):
                prob_acc[sl_il, sl_xl, sl_s]   += prob_patch * self._kernel
                weight_acc[sl_il, sl_xl, sl_s] += self._kernel

            processed += len(batch_wins)
            if processed % log_step == 0 or processed == n_patches:
                pct = 100.0 * processed / n_patches
                logger.info("  %d / %d patches  (%.1f%%)", processed, n_patches, pct)

        # Normalise by accumulated Gaussian weights
        valid = weight_acc > 0
        prob_acc[valid] /= weight_acc[valid]

        prob_volume: np.ndarray = prob_acc.astype(np.float32)
        binary_mask: np.ndarray = (prob_volume >= self.threshold).astype(np.uint8)

        logger.info(
            "Inference complete — fault voxels: %d / %d  (%.4f %%)",
            int(binary_mask.sum()), binary_mask.size,
            100.0 * binary_mask.sum() / binary_mask.size,
        )

        if prob_output is not None:
            _write_zarr_volume(
                prob_volume, Path(prob_output),
                dataset_name="fault_probability",
                dtype=np.float32,
                overwrite=overwrite,
            )
        if mask_output is not None:
            _write_zarr_volume(
                binary_mask, Path(mask_output),
                dataset_name="fault_mask",
                dtype=np.uint8,
                overwrite=overwrite,
            )

        return prob_volume, binary_mask

    # --- factory -----------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device:          str | None = None,
        **inference_kwargs,
    ) -> VolumeInference:
        """Construct a :class:`VolumeInference` by loading a model checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``.pt`` file produced by
            :meth:`~deepseismic.models.unet.UNet3D.save_checkpoint`.
        device:
            ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
        **inference_kwargs:
            Forwarded to :class:`VolumeInference`.
        """
        model = UNet3D.load_checkpoint(
            checkpoint_path, map_location=device or "cpu"
        )
        return cls(model=model, device=device, **inference_kwargs)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_inference(
    seismic_zarr:    str | Path,
    checkpoint_path: str | Path,
    prob_output:     str | Path,
    mask_output:     str | Path,
    *,
    patch_size:  tuple[int, int, int] = (64, 64, 64),
    overlap:     float                = 0.25,
    batch_size:  int                  = 4,
    threshold:   float                = 0.5,
    device:      str | None           = None,
    overwrite:   bool                 = False,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end inference: Zarr amplitude → probability volume + binary mask.

    Parameters
    ----------
    seismic_zarr:
        Path to the seismic amplitude Zarr store (must contain ``amplitude``
        dataset).
    checkpoint_path:
        Path to a :class:`~deepseismic.models.unet.UNet3D` checkpoint.
    prob_output:
        Output path for the fault probability Zarr store.
    mask_output:
        Output path for the binary fault mask Zarr store.
    patch_size:
        Sliding window dimensions ``(D, H, W)``.
    overlap:
        Patch overlap fraction.
    batch_size:
        Patches per GPU forward pass.
    threshold:
        Binary mask probability cutoff.
    device:
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    overwrite:
        Overwrite existing output stores.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(prob_volume, binary_mask)``.
    """
    root    = zarr.open_group(str(seismic_zarr), mode="r")
    seismic: zarr.Array = root["amplitude"]

    engine = VolumeInference.from_checkpoint(
        checkpoint_path,
        device=device,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        threshold=threshold,
    )

    return engine.run(
        seismic,
        prob_output=prob_output,
        mask_output=mask_output,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_zarr_volume(
    array:        np.ndarray,
    path:         Path,
    dataset_name: str,
    dtype:        np.dtype | type,
    chunks:       tuple[int, int, int] = (64, 64, 128),
    overwrite:    bool                 = False,
) -> zarr.Array:
    store = zarr.DirectoryStore(str(path))
    root  = zarr.open_group(store, mode="w" if overwrite else "w-")
    z = root.create_dataset(
        dataset_name,
        data=array.astype(dtype),
        chunks=chunks,
        compressor=zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.SHUFFLE),
        overwrite=overwrite,
    )
    logger.info("Wrote %s → %s  shape=%s", dataset_name, path, z.shape)
    return z
