"""3D UNet for binary seismic fault segmentation.

Architecture overview
---------------------
Follows Çiçek et al. "3D U-Net" (2016) with three practical modifications:

1. **Configurable depth and initial feature count** — controlled via
   :class:`UNetConfig` so the model can be downsized for local dev or scaled
   up for GPU training on the full ST10010 volume.
2. **Batch normalisation + optional Dropout3d** in every DoubleConv block.
3. **ConvTranspose3d upsampling** (not trilinear) for learned, data-adaptive
   upsampling that tends to reduce checkerboard artifacts on seismic data.

Input / output
--------------
- Input:  ``(B, 1, D, H, W)`` float32 seismic amplitude patch.
- Output: ``(B, 1, D, H, W)`` raw logit.  Apply ``torch.sigmoid`` to get
  fault probability; use ``BCEWithLogitsLoss`` during training.

Inference on a full volume is handled by :mod:`deepseismic.models.inference`,
which runs a sliding window with Gaussian overlap-blending.

References
----------
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
  Segmentation", MICCAI 2015.
- Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from
  Sparse Annotation", MICCAI 2016.

Usage
-----
    from deepseismic.models.unet import UNet3D, UNetConfig, build_model

    model = build_model(init_features=32, depth=4, dropout_p=0.1)
    x = torch.randn(2, 1, 64, 64, 64)   # batch of 2 patches
    logits = model(x)                    # (2, 1, 64, 64, 64)
    probs  = torch.sigmoid(logits)

    # Save / load checkpoint
    model.save_checkpoint("checkpoints/epoch_10.pt", epoch=10, metrics={"iou": 0.72})
    model2 = UNet3D.load_checkpoint("checkpoints/epoch_10.pt")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DoubleConv3d(nn.Module):
    """Two consecutive (3×3×3 Conv → BN → ReLU) blocks.

    An optional :class:`~torch.nn.Dropout3d` is inserted between the two
    conv layers when ``dropout_p > 0``.

    Parameters
    ----------
    in_ch:
        Input channel count.
    out_ch:
        Output channel count (used for both convolutions).
    dropout_p:
        Dropout3d probability.  ``0.0`` = no dropout.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout3d(p=dropout_p))
        layers += [
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Contracting step: :class:`DoubleConv3d` + :class:`~torch.nn.MaxPool3d`.

    Returns both the pooled feature map (passed deeper) and the pre-pool
    skip tensor (concatenated in the matching decoder block).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv = DoubleConv3d(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(pooled, skip)``."""
        skip = self.conv(x)
        out  = self.pool(skip)
        return out, skip


class DecoderBlock(nn.Module):
    """Expanding step: :class:`~torch.nn.ConvTranspose3d` + skip concat + DoubleConv3d.

    Parameters
    ----------
    in_ch:
        Channel count coming from the deeper level.
    out_ch:
        Channel count after upsampling (= the skip channel count).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        # After concat with skip: out_ch (upsampled) + out_ch (skip) = 2 * out_ch
        self.conv = DoubleConv3d(out_ch * 2, out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _pad_to_match(x, skip)   # handles off-by-one from odd input dims
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet configuration
# ---------------------------------------------------------------------------


@dataclass
class UNetConfig:
    """Hyper-parameters for :class:`UNet3D`.

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for single-stack seismic amplitude).
    out_channels:
        Number of output channels (1 for binary fault / no-fault).
    init_features:
        Feature maps in the first encoder block.  Doubles at each depth level.
    depth:
        Number of encoder/decoder level pairs (not counting the bottleneck).
        Depth 4 → encoder has feature counts [32, 64, 128, 256] with default
        ``init_features=32``; bottleneck has 512.
    dropout_p:
        Dropout3d probability inside :class:`DoubleConv3d` blocks.
    bilinear_up:
        Reserved for future use (trilinear upsampling variant).
    """

    in_channels:   int   = 1
    out_channels:  int   = 1
    init_features: int   = 32
    depth:         int   = 4
    dropout_p:     float = 0.1
    bilinear_up:   bool  = False


# ---------------------------------------------------------------------------
# UNet3D
# ---------------------------------------------------------------------------


class UNet3D(nn.Module):
    """3D UNet for binary seismic fault segmentation.

    See module docstring for architecture details.

    Parameters
    ----------
    config:
        :class:`UNetConfig`.  Defaults to the PoC baseline (depth=4,
        init_features=32, dropout=0.1).

    Examples
    --------
    >>> model = UNet3D()
    >>> x = torch.randn(1, 1, 64, 64, 64)
    >>> logits = model(x)           # (1, 1, 64, 64, 64)
    >>> probs = torch.sigmoid(logits)
    """

    def __init__(self, config: UNetConfig | None = None) -> None:
        super().__init__()
        cfg = config or UNetConfig()
        self.config = cfg

        feat = cfg.init_features
        dp   = cfg.dropout_p

        # Encoder ------------------------------------------------------------
        self.encoders = nn.ModuleList()
        in_ch = cfg.in_channels
        enc_channels: list[int] = []

        for _ in range(cfg.depth):
            self.encoders.append(EncoderBlock(in_ch, feat, dropout_p=dp))
            enc_channels.append(feat)
            in_ch  = feat
            feat  *= 2

        # Bottleneck ---------------------------------------------------------
        self.bridge = DoubleConv3d(in_ch, feat, dropout_p=dp)

        # Decoder ------------------------------------------------------------
        self.decoders = nn.ModuleList()
        dec_in = feat
        for skip_ch in reversed(enc_channels):
            self.decoders.append(DecoderBlock(dec_in, skip_ch, dropout_p=dp))
            dec_in = skip_ch

        # Output head --------------------------------------------------------
        self.head = nn.Conv3d(dec_in, cfg.out_channels, kernel_size=1)

        self._init_weights()

    # --- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor ``(B, 1, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Logit tensor ``(B, 1, D, H, W)``.
            Apply ``torch.sigmoid`` to obtain fault probabilities.
        """
        skips: list[torch.Tensor] = []

        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bridge(x)

        for dec, skip in zip(self.decoders, reversed(skips), strict=False):
            x = dec(x, skip)

        return self.head(x)

    # --- weight init -------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # --- diagnostics -------------------------------------------------------

    def parameter_count(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    # --- checkpoint utilities ----------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int | None = None,
        optimizer_state: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Save model state to a ``.pt`` checkpoint file.

        Parameters
        ----------
        path:
            Output file path.  Parent directories are created if needed.
        epoch:
            Training epoch number (stored for resumption).
        optimizer_state:
            Optimizer ``state_dict``, if full training resumption is needed.
        metrics:
            Scalar metrics to embed in the checkpoint (e.g. ``{"iou": 0.72}``).
        """
        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "config": {
                "in_channels":   self.config.in_channels,
                "out_channels":  self.config.out_channels,
                "init_features": self.config.init_features,
                "depth":         self.config.depth,
                "dropout_p":     self.config.dropout_p,
            },
        }
        if epoch is not None:
            payload["epoch"] = epoch
        if optimizer_state is not None:
            payload["optimizer_state_dict"] = optimizer_state
        if metrics is not None:
            payload["metrics"] = metrics

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved checkpoint → %s  (epoch=%s)", path, epoch)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> UNet3D:
        """Load a model from a checkpoint file.

        Parameters
        ----------
        path:
            Path to a ``.pt`` file produced by :meth:`save_checkpoint`.
        map_location:
            Torch device string or device object.  Defaults to ``"cpu"`` so
            checkpoints trained on GPU can be loaded on CPU for inference.

        Returns
        -------
        UNet3D
            Model with weights loaded, set to ``eval()`` mode.
        """
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
        cfg  = UNetConfig(**ckpt.get("config", {}))
        model = cls(config=cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        epoch   = ckpt.get("epoch", "?")
        metrics = ckpt.get("metrics", {})
        logger.info("Loaded checkpoint %s  epoch=%s  metrics=%s", path, epoch, metrics)
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_match(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pad *src* to match *target*'s spatial dimensions.

    :class:`~torch.nn.ConvTranspose3d` can produce tensors that are 1 voxel
    smaller than the skip tensor when the input spatial size is odd.
    ``F.pad`` is applied in the ``(W, H, D)`` order expected by PyTorch.
    """
    dd = target.size(2) - src.size(2)
    dh = target.size(3) - src.size(3)
    dw = target.size(4) - src.size(4)

    if dd or dh or dw:
        src = nn.functional.pad(
            src,
            [
                dw // 2, dw - dw // 2,
                dh // 2, dh - dh // 2,
                dd // 2, dd - dd // 2,
            ],
        )
    return src


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_model(
    init_features: int   = 32,
    depth:         int   = 4,
    dropout_p:     float = 0.1,
    **kwargs,
) -> UNet3D:
    """Construct a :class:`UNet3D` with keyword hyper-parameter overrides.

    Parameters
    ----------
    init_features:
        Feature count for the first encoder block.
    depth:
        Number of encoder/decoder level pairs.
    dropout_p:
        Dropout3d probability.
    **kwargs:
        Additional :class:`UNetConfig` fields.
    """
    cfg = UNetConfig(init_features=init_features, depth=depth, dropout_p=dropout_p, **kwargs)
    return UNet3D(config=cfg)

