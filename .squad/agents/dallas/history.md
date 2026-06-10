# Dallas — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, PyTorch, Azure, LLM APIs
- **Goal:** Modernize seismic interpretation — replace legacy monolithic apps with cloud-native + AI
- **Data:** Equinor Volve dataset (3D seismic, well logs, production data)
- **Reference:** microsoft/seismic-deeplearning (UNet, SEResNet, HRNet for facies classification)
- **Key formats:** SEG-Y, numpy arrays (3D volumes), facies labels
- **User:** jospaid

## Learnings

### 2026-06-09 — Ingest pipeline + UNet implementation

#### Key file paths
| File | Role |
|------|------|
| `src/deepseismic/ingest/segy_loader.py` | SEG-Y → xarray → Zarr + JSON sidecar |
| `src/deepseismic/ingest/label_generator.py` | Fault-stick parser + rasteriser → binary mask Zarr |
| `src/deepseismic/preprocessing/patches.py` | 3D patch extraction, spatial splits, PyTorch Dataset |
| `src/deepseismic/models/unet.py` | 3D UNet (configurable depth/features, checkpointing) |
| `src/deepseismic/models/inference.py` | Sliding-window inference with Gaussian overlap-blending |

#### Architecture decisions

- **SEGYLoader as context manager** — segyio requires a file path, so byte/stream inputs are materialised to a platform temp directory and cleaned up in `__exit__`. The SHA-256 fingerprint uses a "quick" mode (first + last 4 MB) to avoid blocking on the full ST10010 ~1 GB file.

- **Zarr chunks (64, 64, 128)** — inline × crossline × sample. The asymmetry on the sample axis (128 vs 64) reflects that seismic data has far more samples (~1000) than spatial bins in any one tile and that the sample axis is queried contiguously during both training (patches) and interpretation (horizon extraction).

- **Spatial train/val/test splits (not random)** — split boundary is on the inline axis at 70/15/15 %. Random splits across a volume with spatial correlation and overlapping patches cause data leakage; spatial splits don't. This is the dominant consideration for seismic ML work.

- **Petrel fault-stick format** — Volve interpretations from Petrel are exported as whitespace-delimited `FaultName X Y Z` rows. The parser handles both 4-column (name + XYZ) and 3-column (XYZ continuation) lines, plus the `FAULT FaultName` section header style. OpendTect style is also supported.

- **Dilation voxels = 1 default** — fault sticks are typically spaced 25–100 m apart horizontally; a 1-voxel dilation (3×3×3 cube per point) keeps the mask conservative. Increase to 2–3 for thick-paint training where label precision is low.

- **UNet depth=4, init_features=32** — produces 32→64→128→256 encoder channels with a 512-channel bottleneck: ~19 M parameters at standard config. Comfortable in 8 GB VRAM with 64³ patches at batch size 4. Depth and feature count are configurable via `UNetConfig`.

- **BCEWithLogitsLoss during training** — the model outputs raw logits; sigmoid is applied only at inference time. This is numerically stabler than sigmoid + BCELoss.

- **Gaussian overlap-blending** — Gaussian kernel (sigma = min(patch_size)/4) gives each patch a soft weighting so boundary predictions taper smoothly. This is the standard approach in medical image segmentation and transfers well to seismic volumes.

- **`zarr.Blosc(lz4)` for float32 amplitude, `zarr.Blosc(zstd+bitshuffle)` for uint8 masks** — LZ4 is faster for decompression of floating-point data; zstd+bitshuffle achieves much better ratios on binary/near-binary uint8 data.

- **`segyio.tools.dt(f) / 1_000`** — converts microseconds to milliseconds. This is the correct way to get sample rate from segyio; the BinHeader `dt` field is in μs.

#### Patterns to reuse
- `PatchConfig.min_fault_fraction` filter — apply during training to oversample fault-rich patches; set to 0 during inference.
- `VolumeInference.from_checkpoint()` — preferred entry point for inference scripts; avoids caller needing to instantiate UNet manually.
- `segy_to_zarr()` / `run_inference()` — convenience one-call functions for pipeline scripts and notebooks.


## Scribe Cross-Agent Update — 2026-06-10T04:30-05:00
Sprint 1 coordination complete. All agents delivered successfully.
- 5 agents synchronized
- 7 decision documents archived
- Full team context available in decisions.md
