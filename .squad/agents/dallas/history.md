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

## Learnings (Detailed Archive)

See original detailed learnings (archived below) for:
- Complete SEG-Y loader implementation details (context manager, SHA-256, temp cleanup)
- Petrel fault-stick format parsing (4-column, 3-column, FAULT headers)
- UNet configuration rationale (depth, features, VRAM budget)
- Zarr compression strategy (LZ4 for float32, zstd+bitshuffle for uint8)
- Viewer implementation (coordinate mapping, baked Zarr contract, model QC metrics)
- Storage layer deep dive (ABSZarrV3Store, asyncio.to_thread, -0 slice edge case)
- Phase 2 ADLS reader extraction and env-var contract

*For full implementation details on any of these topics, refer to the archived learnings sections below.*

---

## ARCHIVED DETAILED LEARNINGS

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

## Learnings — 2026-06-24T12:25:08-05:00: Real Fault Viewer Implementation

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

Caveat: checkpoint metrics at save were `iou=0.0, dice=0.0` — these were placeholder zeros from the training scaffold, not the true eval metrics. The output visually produces plausible fault-like structure given the synthetic training labels. Independent validation against held-out data is future work.

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

## Learnings — 2026-06-11: ABSZarrV3Store code-review bug fixes

### asyncio.to_thread eager-evaluation gotcha

`asyncio.to_thread(expr.method)` defers only `method` — `expr` is evaluated **immediately**
on the calling thread before the thread pool ever runs.  In
`asyncio.to_thread(blob_client.download_blob().readall)`, `download_blob()` is a
blocking HTTP round-trip that executes synchronously on the event-loop thread, defeating
the entire purpose of `to_thread`.  The fix is always to wrap the full call in a lambda:
`asyncio.to_thread(lambda: blob_client.download_blob().readall())`.

Rule of thumb: if the callable you hand to `to_thread` is the result of a *call expression*
(parentheses on the right), you likely have a bug.  Use a lambda or `functools.partial`
to defer the whole expression.

### The -0 suffix slicing trap

In Python, `-0 == 0`, so `data[-0:]` is identical to `data[0:]` and returns the entire
sequence — **not** an empty slice.  Any code that uses a user-supplied integer as a
negative index must guard the zero case explicitly:

```python
if n == 0:
    return b""
return data[-n:]
```

This pattern applies to any suffix/tail slice: list, bytes, str, numpy array.

### 2026-06-24 — Phase 2: ADLS Viewer Backend (Option B) + Bug Fixes

**PR #4 (feat/adls-viewer-readers):** Implemented Option B ADLS viewer backend — app reads artifacts directly from ADLS Gen2 with pure data-reader extraction and zarr v3 async store compatibility.

**Key decisions:**
1. Extracted all data-access logic into `src/deepseismic/ui/_data_readers.py` (pure functions, no Streamlit imports) so Hudson could write proper unit tests without mocking Streamlit.
2. Added `ABSZarrV3Store(zarr.abc.store.Store)` — proper zarr v3 async Store over Azure Blob Storage — to `blob_client.py`.
3. Implemented graceful degradation: missing fault_prob artifact returns `None`, viewer renders amplitude-only with warning.
4. Backend env-var contract (DEEPSEISMIC_DATA_BACKEND=local|azure) relayed to infra issue #8 (comment 4793304744).

**Code review (review-storage):** Found 3 blocking bugs in `ABSZarrV3Store`:
- **Critical:** Event loop blocked on every chunk read (`asyncio.to_thread` evaluates blocking call on main thread before deferred execution). Fixed by wrapping in lambda: `await asyncio.to_thread(lambda: blob_client.download_blob().readall())`.
- **High:** `SuffixByteRequest(0)` returns entire blob (Python `-0 == 0` quirk). Fixed by guarding: `if suffix == 0: return b""`.
- **Medium:** `set()` accepts `byte_range` parameter but ignores it (silent failure). Fixed by raising `NotImplementedError("ABSZarrV3Store does not support partial writes")`.

**Fix commit:** b2b2b58 (+ docs 25b588e). Validation: ruff clean, 156 tests passed.

**Test coverage (hudson-1):** Added `src/tests/test_viewer/test_data_readers.py` — 26 CI-safe tests using dict-backed mock ContainerClient (no Azurite). All tests pass; no bugs found in `_data_readers.py` or fixed `blob_client.py`.

**Status:** CI green, PR #4 approved and ready to merge.

## Learnings — 2026-06-24T18:09:14-05:00: ML Pipeline Fidelity Assessment

### Task
Evaluated how faithfully the deepseismic2 PoC emulates the microsoft/seismic-deeplearning (DeepSeismic) ML pipeline, stage by stage.

### Key findings (summary)

**The PoC DOES train a real model** — `src/deepseismic/training/train.py` contains a proper PyTorch training loop (AdamW, CosineAnnealingLR, BCEWithLogitsLoss with pos_weight, periodic checkpoints). Three checkpoints exist: `checkpoints/epoch_005.pt`, `epoch_010.pt`, `latest.pt`. However, **it trains exclusively on synthetic data** (Ricker wavelet + planar fault geometry generated in `generate_synthetic_training_data()`), not on real labeled seismic.

**Pipeline stages present**: SEG-Y ingest (real), spatial train/val/test split (real, inline-axis 70/15/15), per-patch normalization (real), 3D patch extraction (real, zarr-backed), training loop (real but synthetic data), checkpointing (real), sliding-window inference with Gaussian blending (real), binary IoU/Dice/Precision/Recall metrics (real).

**Pipeline stages missing or stubbed**:
- `preprocessing/pipeline.py` is a stub — docstring only, no code.
- No augmentation wired (transform slots exist but nothing passed).
- No experiment logging (no TensorBoard, WandB, MLflow — only stdout prints).
- No YACS/YAML config files — training config is a Python dataclass.
- No global training seed (reproducibility limited to per-test `torch.manual_seed`).
- No multi-class metrics (mIoU, per-class IoU, confusion matrices) — justified because ours is binary fault detection, not facies classification.
- `validation/fault_continuity` and `throw_error_mean_ms` are hardcoded 0.0 (TODOs).

**Architecture divergence**: Original emphasizes 2D section/patch deconvnet, SEResNet, HRNet. We have 3D UNet only. 3D is well-motivated for fault detection (faults are 3D surfaces, not 2D slices) but diverges from the benchmark model zoo.

**Normalization**: We use per-patch z-score; original uses per-volume normalization from train-split statistics. Ours is operationally simpler and avoids needing a precomputed stats file; slight domain shift relative to reference.

**Metrics gap**: We compute binary IoU/Dice — adequate for fault detection. Original computes multi-class mIoU/confusion matrices for facies — irrelevant to our task. We add geophysics-specific distance-tolerant metrics and ASSD not present in the original.

### Critical gaps and minimal fixes

1. **Synthetic-only training** (Critical): The biggest fidelity gap. Fix: wire `PatchDataset` (zarr-backed) into `train.py` when zarr data exists at the default path. `build_dataloaders()` in `patches.py` is already implemented; `train.py` just needs a path check to use it instead of `NumpyPatchDataset`.

2. **No experiment logging** (Important): Add 10 lines of TensorBoard `SummaryWriter` to `train_epoch` / `validate` in `train.py`. Already no extra dependency needed (PyTorch ships TensorBoard integration).

3. **No training seed** (Important): Add `torch.manual_seed(seed)` and `np.random.seed(seed)` at top of `train()`. One-liner.

4. **No training config file** (Nice-to-have): Export `TrainConfig` to/from JSON in `train.py`. Already a dataclass — `dataclasses.asdict` + `json.dump` is 5 lines.

5. **`preprocessing/pipeline.py` stub** (Nice-to-have): Implement the orchestration surface described in its docstring.

### Checkpoint notes
- Three real checkpoint files present (5.6 MB each, UNet3D depth=3/4 config).
- Saved metrics at epoch 10: `iou=0.0, dice=0.0` — these are placeholder zeros from the training scaffold (metrics saved but not correctly propagated at save-time). Model produces non-trivial output (verified in Phase 1).

## Scribe Consolidation — 2026-06-24T23:29:56Z

ML pipeline fidelity assessment merged into `.squad/decisions.md` (Phase 2 Process Fidelity Evaluations section). ADLS viewer readers implementation also merged.

**Key consolidated findings:**
- Real PyTorch training loop exists but trains synthetic-only data (PatchDataset not wired)
- Critical gaps: synthetic-only training, no experiment logging, no training reproducibility seed
- Important gaps: no config serialization, preprocessing/pipeline.py stub, real-mode API path untested, single model architecture only
- Nice-to-have: data augmentation, confusion matrix, throw/continuity validation TODOs

Ripley's Sprint 2 recommendations include wiring real labels (4h), adding eval script (2h), fixing README (30min).

ADLS reader extraction and zarr v3 async Store implementation documented for Phase 2 infrastructure.

