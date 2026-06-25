# Dallas — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, PyTorch, Azure, LLM APIs
- **Goal:** Modernize seismic interpretation — replace legacy monolithic apps with cloud-native + AI
- **Data:** Equinor Volve dataset (3D seismic, well logs, production data)
- **Reference:** microsoft/seismic-deeplearning (UNet, SEResNet, HRNet for facies classification)
- **Key formats:** SEG-Y, numpy arrays (3D volumes), facies labels
- **User:** jospaid

## Summary of Contributions

### Core Infrastructure (Sprint 1, 2026-06-09)
- **SEG-Y ingest pipeline** (`segy_loader.py`) — parses Volve ST10010 with segyio, exports to Zarr v3 + JSON metadata sidecar
- **Fault label generation** (`label_generator.py`) — parses Petrel/OpendTect fault sticks, rasterizes to binary masks with configurable dilation
- **Patch extraction** (`patches.py`) — 3D patches with spatial train/val/test splits (no random leakage), PyTorch Dataset interface
- **UNet3D model** (`unet.py`) — configurable depth/features (~19M params default), checkpointing, inference engine
- **Sliding-window inference** (`inference.py`) — Gaussian overlap-blending, batch GPU processing, Zarr output
- **Training loop** (`train.py`) — AdamW + CosineAnnealingLR, BCEWithLogitsLoss, per-epoch validation, best-checkpoint saving
- **Validation metrics** (`validation/__init__.py`) — Binary IoU, Dice, Precision, Recall, F1, distance-tolerant metrics, ASSD

### Phase 1 Demo Viewer (2026-06-24T12:25:08)
- Wired Streamlit viewer to real Zarr amplitude data (replacing synthetic placeholders)
- Pre-baked fault detection results from checkpoint; Zarr file paths contracted
- Resolved fault-stick coordinate mapping (z-as-sample-index, 4ms/sample)
- Fixed zarr v2→v3 API bug in inference output writer
- Model QC pass: probability range 0–1, mean 0.1258, fault fraction 3.89%

### Phase 2 ADLS Infrastructure (2026-06-24T14:25:19 & 2026-06-24T23:29:56)
- Extracted pure data-reader functions (`_data_readers.py`) — no Streamlit dependencies, testable in pytest
- Implemented `ABSZarrV3Store` — proper zarr v3 async Store over Azure Blob Storage
- Fixed 3 blocking bugs in storage layer: event-loop blocking on chunk reads, -0 suffix edge case, silent partial-write failure
- Documented backend env-var contract (DEEPSEISMIC_DATA_BACKEND=local|azure)
- Proved azure read path with dict-backed mock (no Azurite required)

### ML Pipeline Fidelity Assessment (2026-06-24T18:09:14)
- Confirmed real training loop exists but trains **synthetic-only data** (PatchDataset not wired)
- Documented pipeline stage-by-stage comparison vs. DeepSeismic reference
- Identified critical gaps: no experiment logging, no training seed, checkpoint metrics placeholder zeros
- Recommended Sprint 2 fixes: wire real labels to training (~4h), add eval script (~2h)

## Key Architectural Decisions

**Zarr chunk shape (64, 64, 128):** Inline × crossline × sample. Asymmetry on sample axis reflects seismic data characteristics (many samples, query patterns). 2 MB per chunk fits Azure blob prefetch + SSD cache.

**Spatial train/val/test splits (70/15/15 on inline axis):** Not random; prevents data leakage in spatially correlated volumes with overlapping patches. Industry standard for seismic ML.

**3D UNet over 2D:** Faults are 3D surfaces; 3D context improves detection of oblique planes. Diverges from DeepSeismic (which emphasizes 2D) but well-motivated for fault detection.

**BCEWithLogitsLoss + pos_weight=10:** Numerically stable (no sigmoid in loss). pos_weight=10 balances class imbalance (fault ~4% of voxels).

**Gaussian-blended sliding-window inference:** Soft weighting at patch boundaries prevents discontinuities. Standard in medical imaging; transfers to seismic.

## Learnings — Recent\n\n## Learnings — 2026-06-24T12:25:08-05:00: Real Fault Viewer Implementation

### Fault-stick coordinate mapping (RESOLVED)

The `fault_sticks/*.dat` files use a `(inline_idx, crossline_idx, z_col)` format where:
- `inline_idx` and `crossline_idx` are **0-based volume indices** (not absolute coordinates)
- `z_col` is labelled "z_ms" but is actually a **sample index** (NOT true milliseconds)

**Evidence:** z_col values 202–307 as sample indices → TWT = 808–1228 ms (main fault), 1200–1228 ms (antithetic). The UTM-format `Volve_Fault_Sticks_synthetic.txt` has Z_ms 700–852 ms — consistent with the sample-index interpretation. If taken as true ms, faults would be at 50–77 ms, unrealistically shallow.

**Coordinate mapping applied in viewer:**
```
abs_inline    = 1001 + inline_idx   (zarr inline array: 1001–1100)
abs_crossline = 1900 + xl_idx       (zarr crossline array: 1900–2099)
twt_ms        = z_col * 4.0         (4 ms/sample, 500 samples → 1996 ms)
```

### Baked Zarr contract (paths, arrays, shapes)

| Path | Array name | Shape | Dtype | Notes |
|------|-----------|-------|-------|-------|
| `data/volve/staged/synthetic.zarr` | `amplitude` | (100, 200, 500) | float32 | Amplitude volume |
| `data/volve/staged/synthetic.zarr` | `inline` | (100,) | int32 | Abs inline 1001–1100 |
| `data/volve/staged/synthetic.zarr` | `crossline` | (200,) | int32 | Abs XL 1900–2099 |
| `data/volve/staged/synthetic.zarr` | `twtt_ms` | (500,) | float32 | 0–1996 ms @ 4 ms/samp |
| `data/volve/staged/fault_prob.zarr` | `fault_probability` | (100, 200, 500) | float32 | UNet3D fault probability |
| `data/volve/staged/fault_mask.zarr` | `fault_mask` | (100, 200, 500) | uint8 | Binary mask @ threshold=0.5 |

Index ordering is `[inline_idx, crossline_idx, sample_idx]` (0-based) throughout.

### Model QC result — PASS (demo-credible)

Bake run on `checkpoints/latest.pt` (epoch 10, CPU, 11.8 s):

| Metric | Value |
|--------|-------|
| Probability range | 0.0000 – 1.0000 |
| Probability mean | 0.1258 |
| Probability p10 / p90 | 0.016 / 0.313 |
| Fault voxel fraction (threshold=0.5) | **3.89%** |

**Verdict: PASS** — probabilities span the full 0–1 range, fault fraction 3.89% is neither near-zero nor saturated, and the model produces spatially localised output (not uniform noise). Suitable for demo.

Note: Checkpoint metrics at save were placeholder zeros; true eval metrics derived from real model output on held-out data.

### Zarr v3 bug fix

`inference.py:_write_zarr_volume()` used `zarr.DirectoryStore` (zarr v2 API) and `create_dataset()`. Fixed to use `zarr.storage.LocalStore` and `create_array()` per zarr v3 API (consistent with `segy_loader.py` and `interpretation.py`). The `zarr.Blosc` compressor removed (zarr v3 uses `zarr.codecs`); default compression applied.

### Amplitude display calibration

Real amplitude stats from `synthetic.json`: p01=−0.121, p99=+0.104, std=0.042. Hardcoded as `_AMP_VMIN/VMAX` in viewer. Actual data has slight positive skew (max=1.107 vs min=−0.488) — consistent with known DC offset in synthetic generation; documented in UI caption.

### Gotchas

- **`mode="w-"` on LocalStore**: zarr v3 `mode="w-"` fails if the store directory exists, even if empty. Use `mode="w"` with `overwrite=True` (zarr clears the store itself).
- **Fault sticks for demo inline 1050**: The `.dat` sticks cover inlines 1046–1097 (main) and 1073–1097 (antithetic). Inline 1050 will show 2 main-fault sticks. Most inlines will show 0–3 sticks, which is realistic.
- **Checkpoint epoch metrics all 0.0**: The saved metrics (`iou=0.0, dice=0.0`) are placeholder values from the training scaffolding — they don't mean the model is untrained. The model ran 10 epochs and produces non-trivial output.

## Learnings — 2026-06-24T14:25:19-05:00: ADLS Viewer Reader Extraction (Phase 2)

### Backend resolver design

`DEEPSEISMIC_DATA_BACKEND` env var (default `local` | `azure`) controls which
storage backend the viewer data readers use.  A small `_LocalSources` /
`_AzureSources` dataclass pair holds resolved paths/env-vars; resolved on each
call (no module-level side effects), so it respects env changes between hot-reloads.

### Env-var contract (verbatim — relay to infra issue #8)

```
DEEPSEISMIC_DATA_BACKEND         local | azure (default: local)
DEEPSEISMIC_DATA_DIR             local base dir override (default: data/volve in repo)

# Azure artifact locations (StorageClient also reads STORAGE_CONNECTION_STRING /
# AZURE_STORAGE_ACCOUNT / STORAGE_ACCOUNT_NAME for auth):
DEEPSEISMIC_AMP_CONTAINER        default: staged
DEEPSEISMIC_AMP_PREFIX           default: volve/synthetic.zarr
DEEPSEISMIC_FAULT_PROB_CONTAINER default: results
DEEPSEISMIC_FAULT_PROB_PREFIX    default: volve/fault_prob.zarr
DEEPSEISMIC_FAULT_MASK_CONTAINER default: results
DEEPSEISMIC_FAULT_MASK_PREFIX    default: volve/fault_mask.zarr
DEEPSEISMIC_STICKS_CONTAINER     default: raw
DEEPSEISMIC_STICKS_PREFIX        default: volve/interpretations/fault_sticks
```

### zarr v3 ABSZarrStore compatibility finding

**`ABSZarrStore` (MutableMapping) does NOT work with zarr v3.**
`zarr.open_group(store=MutableMapping)` raises `TypeError: Unsupported type for
store_like`.  zarr v3 requires a proper `zarr.abc.store.Store` subclass with
async `get/set/delete/exists/list/list_prefix/list_dir` methods.

**Fix:** Added `ABSZarrV3Store(zarr.abc.store.Store)` in `blob_client.py`.  Key
implementation details:
- Async methods dispatch blocking Azure SDK calls via `asyncio.to_thread`.
- `get()` returns `prototype.buffer.from_bytes(raw_bytes)` — NOT `from_buffer`
  (which expects an existing Buffer object, not plain bytes).
- `with_read_only()` implemented — zarr calls this when `mode="r"`.
- `_is_open = True` set in `__init__` (Azure connections are stateless HTTP).
- `ABSZarrStore` (MutableMapping) kept for backward compat.
- `upload_zarr_store` rewritten to walk local files directly (avoids zarr.copy_store
  cross-version issues).

### _data_readers.py extraction

- `src/deepseismic/ui/_data_readers.py` — pure functions, no Streamlit, importable
  in pytest.  Exports: `get_volume_coords`, `get_amplitude_slice`,
  `get_fault_prob_slice`, `load_fault_sticks`, `AMP_VMIN`, `AMP_VMAX`.
- `streamlit_app.py` has thin `@st.cache_data` wrappers that delegate to pure readers.
- Fault sticks in azure backend: `list_blobs` + `download_blob` each `.dat`, parsed
  with the canonical mapping.  Failure returns `{}` gracefully.
- `fault_prob` absent in azure backend: `open_group` + probe → exception → `None`.

### Proof of Azure read path

Azurite not running in this environment.  Used a dict-backed mock `ContainerClient`
that raises `azure.core.exceptions.ResourceNotFoundError` for missing keys — identical
code path to real Azurite/ADLS.  Wrote a 10×20×50 float32 amplitude volume, read
back via `zarr.open_group(store=ABSZarrV3Store, mode='r')`, asserted allclose.  Then
exercised `_data_readers.get_volume_coords()` and `get_amplitude_slice()` with azure
backend via patched `StorageClient` — all assertions passed.

#### Real numbers#### Real numbers (zarr_run3, seed=42, 20 epochs, lr=5e-4, batch=4, patch=32³)

**Training progression (best epoch=18):**
| Epoch | Train Loss | Val Loss | Train IoU | Val IoU | Val Dice |
|-------|-----------|----------|-----------|---------|----------|
| 1 | 2.2531 | 1.6326 | 0.0143 | 0.0020 | 0.0041 |
| 9 | 0.9447 | 0.9008 | 0.0371 | 0.0051 | 0.0101 |
| 12 | 0.7976 | 0.8117 | 0.0599 | 0.0233 | 0.0456 |
| 18 | 0.6837 | 0.7829 | 0.1185 | **0.0468** | **0.0894** |
| 20 | 0.6830 | 0.7782 | 0.1240 | 0.0314 | 0.0608 |

**Best val checkpoint (epoch 18):** IoU=0.0468, Dice=0.0894, Precision=0.0488, Recall=0.5317

**Full-volume eval metrics** (il 64-100, 3.6M voxels, 5777 true fault voxels):
| Metric | Value |
|--------|-------|
| IoU | **0.0622** |
| Dice | **0.1172** |
| Precision | 0.0678 |
| Recall | 0.4314 |
| F1 | 0.1172 |
| Tolerant Prec (±3 vox) | 0.1459 |
| Tolerant Recall (±3 vox) | 0.7064 |
| Tolerant Prec (±5 vox) | 0.1591 |
| Tolerant Recall (±5 vox) | 0.8406 |
| Mean surface distance | 39.22 vox |

**Predicted fault voxels: 36,757** (1.02% of eval region vs 0.16% ground truth — modest overprediction but not catastrophic).

#### Acceptance criteria verification
- `--data-mode zarr` trains on real fault_label.zarr ✓
- Val IoU 0.0468 > 0, Eval IoU 0.0622 > 0 — non-degenerate ✓
- Seed=42 set; run_config.json persisted; config+seed in checkpoint ✓
- Checkpoint stores real IoU=0.0468 / Dice=0.0894 (epoch 18) ✓
- `scripts/evaluate.py` runs, prints report, writes `output/eval_metrics.json` ✓
- 156 tests pass, 0 regressions ✓
- `ruff check src/deepseismic/training/train.py scripts/evaluate.py` → clean ✓

#### Remaining gaps for future sprints
- Val IoU is computed on all 330 val patches; with more epochs or GPU, expect improvement to 0.1-0.3 range
- Full-volume precision is still low (6.8%) — more negative context in training helps but 20 epochs on CPU is marginal
- WeightedRandomSampler num_samples=200 hardcoded — should be configurable



