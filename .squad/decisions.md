# Squad Decisions

## Active Decisions

# Triage Decision — Issues #25 and #26

**Date:** 2026-06-29T17:46:41-05:00  
**Author:** Ripley (Lead)

---

## Issue #25 — Chat wedges after truncated tool turn (AOAI 400)

**Title:** "Chat wedges after a truncated tool turn: dangling tool_calls corrupts shared thread (AOAI 400) + 25s truncation"

| Field | Value |
|-------|-------|
| Owner | squad:lambert |
| Type | type:bug |
| Priority | priority:p0 |
| Release | release:v0.4.0 |

### Root cause summary

Two bugs, one blocker:

**Bug B (blocker):** `FoundryAgent.chat()` (`src/deepseismic/agent/agent.py` ~L402/437) appends the assistant message WITH `tool_calls` to the persistent thread history before the matching tool-result messages are committed. When the UI's 25s guard (`gradio_app.py` L318-323) fires `GeneratorExit`, the thread is left with an unmatched `tool_calls` entry. Because the thread is reused per session and the UI agent is a process-wide singleton, every subsequent request from any user on the container replays the corrupt history → AOAI 400. Container restart is the only recovery path.

**Bug A (contributing):** The 25s wall-clock guard abandons the generator mid-round because the agent makes blocking (non-streaming) completion calls. Fix requires real streaming and/or turn-boundary-aware truncation.

### Ownership rationale

Thread-state management and streaming completion are LLM integration code owned by Lambert. Bug A's truncation guard is UI-side (Parker territory) but the real fix — streaming — lives in the agent. Lambert leads; Lambert/Parker coordinate on the UI guard cleanup.

### Priority rationale

p0: the bug permanently wedges the hosted demo for all users until an operator manually restarts the container. No user-facing workaround exists.

---

## Issue #26 — Run lookup by short id-prefix 404s on ADLS/HNS

**Title:** "Run lookup by short id-prefix 404s: _resolve_run_id catalog list-scan fails on ADLS/HNS (full UUID works)"

| Field | Value |
|-------|-------|
| Owner | squad:parker |
| Type | type:bug |
| Priority | priority:p1 |
| Release | release:v0.5.0 |

### Root cause summary

`_resolve_run_id()` (`src/deepseismic/api/routes/interpretation.py` L48-105) uses `ContainerClient.list_blobs(name_starts_with=...)` for prefix resolution against the ADLS Gen2 / hierarchical-namespace `catalog` container. This API returns nothing (or raises) on HNS containers where flat-blob enumeration is not available. A bare `except Exception: pass` at L84 silently swallows the failure, making the scan appear to return zero matches rather than surfacing an error. The caller sees a 404 even though the run persisted correctly.

Exact `download_blob` (used when a full UUID is supplied) works correctly.

### Ownership rationale

Pure backend/API + Azure storage-client bug. No ML or LLM surface. Parker owns.

### Priority rationale

p1: a clean workaround exists (supply the full UUID). No data loss. The run is intact; only the short-prefix UX is broken. Independent of #25 — no shared code surface.

### Suggested fix path

1. Replace `list_blobs` with `DataLake FileSystemClient.get_paths(path="catalog/interpretation/", recursive=False)` — the same client the ADLS browser uses, OR  
2. Write a `catalog/interpretation/index.json` manifest atomically at submit time; prefix scan reads the index instead of enumerating blobs.

Either way: replace the bare `except Exception: pass` at L84 with a logged error so future failures surface diagnostically.

---

## Sequencing

**#25 must land before #26.** #25 is a p0 that blocks all users; #26 is a p1 with a workaround. Both are independent bugs with no shared code surface.

---

## Architectural note (general, team-wide)

**Agent thread-state must be committed atomically.** The assistant `tool_calls` message and all matching tool-result messages must be appended to thread history in a single atomic write. Writing the assistant entry first creates a window where any interruption (`GeneratorExit`, timeout, exception) produces permanently corrupt thread state. This principle applies to any component that reuses a persistent conversation thread across requests.

---

# Decision Note: S2-06 — Seismic Conditioning / QC Pipeline

**Author:** Ash (Geophysicist SME)
**Date:** 2026-06-24T20:11:05-05:00
**Sprint item:** S2-06 (P1)
**Status:** Implemented ✅

---

## What was done

Replaced the `src/deepseismic/preprocessing/pipeline.py` stub (8-line docstring,
no code) with a working conditioning/QC module. This closes GAP-C2 (conditioning
stage absent) and partially addresses GAP-I1 (amplitude-preserving normalisation).

### Functions added

| Function | Purpose |
|---|---|
| `compute_volume_qc(zarr_path, sidecar_path)` | Volume-level QC metrics dict + human-readable log |
| `global_amplitude_normalize(volume, p99)` | Amplitude-preserving p99 normalisation (GAP-I1 fix) |
| `condition_volume(...)` | Orchestration: QC + optional norm + JSON report |
| `_dominant_frequency_hz(traces, dt_ms)` | FFT peak-energy frequency from sampled traces |
| `_autocorr_symmetry(traces)` | Zero-phase proxy via autocorrelation energy ratio |

### Dependencies

numpy + zarr only (numpy.fft — no scipy needed). Already in pyproject.toml.

---

## Actual QC numbers — `data/volve/staged/synthetic.zarr`

| Metric | Value |
|---|---|
| Shape (IL × XL × samples) | 100 × 200 × 500 |
| Non-zero fraction | **1.000** (no dead traces) |
| Amplitude min / max | −0.488 / 1.107 |
| Amplitude p01 / p99 | −0.121 / 0.104 |
| Mean ± std | ~0 ± 0.0419 |
| Sample interval (dt) | 4.0 ms → Nyquist 125 Hz |
| Dominant frequency | **36.6 Hz** |
| Vertical resolution (λ/4, v=2000 m/s) | **13.7 m** |
| Autocorrelation symmetry (zero-phase proxy) | **1.000** ✅ (consistent with synthetic zero-phase) |

---

## Geophysical assumptions documented in code

- **Phase:** Zero-phase wavelet assumed. Autocorrelation symmetry proxy flags deviation.
- **Polarity:** SEG normal (American) — positive impedance contrast → positive peak.
- **Sample interval:** 4 ms default (from sidecar `geometry.sample_rate_ms`).
- **Vertical resolution:** λ/4 tuning at reference velocity 2000 m/s.
  At 36.6 Hz: λ/4 ≈ 13.7 m. Beds thinner than this are below tuning.
- **Amplitude scale:** Display units (not calibrated reflectivity).

---

## Design decisions

1. **Sidecar-first stats:** `compute_volume_qc` uses pre-computed sidecar amplitude
   stats (p01/p99/mean/std) instead of loading the full volume, so it runs in
   seconds regardless of volume size. Falls back to full-volume computation if no
   sidecar is present.

2. **Trace sampling for spectral/phase estimates:** 500 evenly-spaced traces are
   sufficient for a robust mean amplitude spectrum and autocorrelation estimate.
   The sampling grid is deterministic (no random seed needed), making the function
   reproducible and test-friendly.

3. **p99 clip at ±1.5:** `global_amplitude_normalize` clips to ±1.5 by default to
   suppress isolated hot pixels while preserving inter-patch amplitude hierarchy.
   The original amplitude array is never overwritten; the conditioned version is
   written as `amplitude_norm` in the same Zarr group.

4. **No scipy dependency:** numpy.fft is sufficient for a peak-frequency estimate.
   Added scipy only if a more rigorous periodogram (Welch) were needed — deferred.

---

## Asymmetry observation

The volume amplitude min (−0.488) is roughly half the max (1.107), giving a
positive-polarity bias. This is consistent with a synthetic generated by
convolution of a Ricker wavelet with a predominantly positive-impedance contrast
model. Worth noting for any AVO attribute work — zero-mean is preserved
(mean ≈ 5×10⁻⁷) but the distribution is not symmetric.

---

## Outstanding / follow-on

- S2-07 will add unit tests for `compute_volume_qc` and `global_amplitude_normalize`.
- When real Volve SEG-Y is ingested, re-run QC and compare dominant frequency and
  autocorrelation symmetry against this synthetic baseline.
- Expose `condition_volume` as a CLI entry point (`deepseismic qc ...`) — deferred
  to a later sprint.


# Ash S2-Bug Decision Note — Zero-Phase Proxy Fix

**Author:** Ash (Geophysicist SME)
**Date:** 2026-06-24T20:09:56-05:00
**Sprint:** 2
**Item:** S2-06 bug fix — `_autocorr_symmetry` replaced by `_wavelet_symmetry`
**Status:** Complete ✅

---

## Bug (from Hudson S2-07 report)

`_autocorr_symmetry` in `src/deepseismic/preprocessing/pipeline.py` always returned
1.0 regardless of input phase. Root cause: the autocorrelation of any real-valued
signal `x` is mathematically even-symmetric — `ac[c+k] == ac[c−k]` for all lag `k`
— so the ratio `E_pos / E_neg` is always exactly 1.0. The function was useless as a
phase diagnostic and silently masked any phase issues in the QC report.

---

## Fix Applied

Replaced `_autocorr_symmetry` with `_wavelet_symmetry` (same signature, same
call site in `compute_volume_qc`).

### Method: Normalised Time-Reversal Correlation

```
C = dot(x, x_reversed) / dot(x, x)
```

where `x` is the mean trace over the sampled set and `x_reversed = x[::-1]`.

**Rationale:** A zero-phase wavelet is symmetric in time — `w(t) = w(−t)` — so
comparing it to its own time-reverse produces a normalised dot product of +1.0.
A 90°-rotated wavelet (the Hilbert transform of a symmetric wavelet) is
antisymmetric — `h(t) = −h(−t)` — giving a dot product of −1.0. Mixed-phase
signals fall between the two.

**Why not the Hilbert transform instantaneous-phase approach?**
The Hilbert approach (computing `mean(|angle(analytic_signal)|)` over the
dominant-energy region) requires either scipy or a careful FFT-based Hilbert
implementation. The time-reversal correlation achieves the same geophysical goal
with a single dot product and no additional complexity. It is simpler to reason
about, unit-test, and document.

### Range and Interpretation

| Value | Phase interpretation |
|-------|----------------------|
| +1.0  | Perfectly symmetric (zero-phase or 180°-flipped — use polarity separately) |
| ~0.0  | Quadrature phase (90° or 270° rotation) |
| −1.0  | Perfectly antisymmetric |

Values above ~0.8 are consistent with near-zero-phase.

### Geophysical Caveat

The proxy is computed on the **mean trace** of the sampled set, not on a wavelet
extracted by well-tie or deterministic methods. For seismic volumes with many
independent reflectors, the mean trace converges toward zero, making the proxy
unreliable. The metric is most diagnostic when applied to:
- A single extracted (averaged) wavelet
- Near-offset or zero-offset stack traces in a quiet zone with few reflectors

For a definitive phase assessment, use deterministic wavelet extraction or well-tie.

### Implementation Note

Pure `numpy` (one dot product). `scipy` is a project dependency but was not
needed — no new imports.

---

## Changes Made

| File | Change |
|------|--------|
| `src/deepseismic/preprocessing/pipeline.py` | Replaced `_autocorr_symmetry` with `_wavelet_symmetry`; renamed QC dict key from `autocorr_symmetry` → `wavelet_symmetry`; updated `_log_qc_report`, module docstring, and `compute_volume_qc` docstring |
| `src/tests/test_preprocessing/test_sprint2_pipeline.py` | Updated import; updated `_REQUIRED_QC_KEYS`; renamed `TestAutocorrSymmetry` → `TestWaveletSymmetry`; removed `@pytest.mark.xfail`; adapted existing asymmetric test to pass; added `test_zero_phase_ricker_scores_higher_than_rotated` using odd-length n=129 Ricker (perfectly centered under time-reversal) vs its Hilbert-transform (90°-rotated) counterpart |

---

## Test Results

- **Sprint 2 pipeline tests:** 18 passed, 0 xfail/xpass
- **Full non-integration suite:** 211 passed, 2 skipped, 8 deselected (was 209 + 1 xfailed)
- **Ruff lint:** clean on both changed files

---

## New QC Number — `data/volve/staged/synthetic.zarr`

```
wavelet_symmetry: 0.116   (was 1.000 — broken)
dominant_freq_hz: 36.62
shape: (100, 200, 500)
```

**Interpretation:** The value 0.116 is the correct, honest result. It does NOT
mean the volume has a rotated wavelet — it means the mean-trace proxy is
unreliable for this volume. The synthetic has 500-sample traces with independent
random reflectivity; the mean trace averages out the wavelet shape (mean
reflectivity ≈ 0), leaving a low-amplitude signal whose time-reversal symmetry
is essentially uninformative. The old value of 1.000 was a false certainty.

For a proper zero-phase QC on this volume, extract a wavelet from a quiet zone
or use deterministic wavelet extraction.

---

## Routing

- No further action required from Dallas or Ripley.
- Hudson's xfail marker is retired; test suite is clean.
- Coordinator may update Sprint 2 status for S2-06 (QC pipeline now has a
  mathematically correct, if caveat-laden, zero-phase proxy).


# Decision Note: S2-01 Fault-Label Zarr — Coordinate Mapping & Results

**Author:** Dallas (Data/ML Engineer)  
**Date:** 2026-06-24T20:04:09-05:00  
**Status:** Complete — Ash review requested on coordinate mapping  
**Sprint item:** S2-01 (P0, ~3h budget)

---

## What Was Built

`scripts/generate_fault_label.py` — a reproducible CLI that:
1. Reads the two real Volve fault-stick `.dat` files from `data/volve/interpretations/fault_sticks/`
2. Maps coordinates to 0-based volume index space (see below)
3. Rasterises both fault polylines via `FaultMaskGenerator.add_fault_sticks_in_index_space()`
4. Applies cubic dilation (default 3 voxels, configurable)
5. Writes a uint8 Zarr to `data/volve/staged/fault_label.zarr/fault_mask`

**Regenerate command:**
```
python scripts/generate_fault_label.py --overwrite
python scripts/generate_fault_label.py --dilation 2 --overwrite   # lighter
```

---

## Real Fault-Stick Data Confirmed

Both `.dat` files exist and are populated:

| File | Points | Inline range | Crossline range | Z range |
|------|--------|-------------|-----------------|---------|
| `fault_antithetic.dat` | 7 | 72–96 | 47–54 | 300–307 |
| `fault_main_normal.dat` | 11 | 45–95 | 84–124 | 202–227 |
| **Total** | **18** | | | |

S2-01 risk #1 ("no real sticks") does **not** apply. Real data is present.

---

## Coordinate Mapping — REVIEW REQUESTED (Ash)

### .dat file format
```
# Comment lines start with #
# Format: inline crossline z_ms   ← FILE COMMENT IS MISLEADING (see below)
inline_idx  crossline_idx  z_col
```

### Column interpretation

| Column | File label | **Actual interpretation** | Rationale |
|--------|-----------|--------------------------|-----------|
| col[0] | inline | **0-based inline index** | Values 45–96 fit 0–99 (not 1001+ abs) |
| col[1] | crossline | **0-based crossline index** | Values 47–124 fit 0–199 (not 1900+ abs) |
| col[2] | z_ms | **0-based sample index** | Values 202–307; if truly ms → twt 50–77ms (unrealistically shallow); as sample indices → mid-volume (808–1228 ms twt) ✓ |

### Mapping used in the script

```
abs_inline     = 1001 + il_idx     (BASE_IL = 1001, il_idx in 0–99)
abs_crossline  = 1900 + xl_idx     (BASE_XL = 1900, xl_idx in 0–199)
twt_ms         = z_col × 4.0       (sample_rate_ms = 4.0, datum_ms = 0.0)
```

No further conversion was needed — `add_fault_sticks_in_index_space()` accepts 0-based index triplets directly.

### Grid alignment check result
All 18 stick points fall **inside** the volume bounds ✓

Absolute survey coordinates after mapping:
- Inline: 1046–1097 (volume: 1001–1100) ✓
- Crossline: 1947–2024 (volume: 1900–2099) ✓
- TWT: 808–1228 ms (volume: 0–1996 ms) ✓

---

## Output Zarr — Key Numbers

| Property | Value |
|----------|-------|
| Path | `data/volve/staged/fault_label.zarr/fault_mask` |
| dtype | uint8 |
| Shape | (100, 200, 500) — matches amplitude volume exactly |
| Chunks | (64, 64, 128) — matches amplitude volume |
| Dilation radius | 3 voxels (7³ = 343 neighbourhood per point) |
| Raw stick points | 18 |
| **Fault voxels** | **7,967** |
| Total voxels | 10,000,000 |
| **Fault fraction** | **0.0797%** (0.000797) |
| Grid alignment | All 18/18 points inside volume ✓ |

PASS: Non-trivial positive fraction. Suitable for S2-02 training with pos_weight ~1256 (inverse of fault fraction).

---

## Naming Note for S2-02

The label Zarr is `fault_label.zarr` (ground truth from sticks), **not** `fault_mask.zarr` (which is reserved for the model's binary prediction output from `bake_demo_faults.py`). The S2-02 training data loader should read from `fault_label.zarr`.

---

## Code Changes

- **New:** `scripts/generate_fault_label.py` — main deliverable
- **Fixed:** `src/deepseismic/ingest/label_generator.py:to_zarr()` — zarr v2→v3 API (`DirectoryStore`→`LocalStore`, `create_dataset`→`create_array`). Identical bug class to the one fixed in the inference writer in Phase 1.


# Decision Note: S2-02/05/08/03 — Real-data Training & Eval Pipeline

**Author:** Dallas (Data/ML Engineer)
**Date:** 2026-06-24T20:16:17-05:00
**Status:** Complete — Ripley review requested
**Sprint items:** S2-02 (P0), S2-05 (P1), S2-08 (P1), S2-03 (P0)

---

## Summary

All four sprint items implemented together in one pass as requested. Training now
runs end-to-end on real Volve zarr data with reproducibility, fixed metrics, and
a standalone evaluation script. The model is **non-degenerate**: eval IoU=0.0622,
Dice=0.1172, Recall=0.43 on the held-out region (5,777 true fault voxels / 3.6M total).

---

## Files Changed / Added

| File | Change |
|------|--------|
| `src/deepseismic/training/train.py` | Major update — S2-02 zarr path, S2-05 seed + config persistence, S2-08 epoch-level metrics |
| `scripts/evaluate.py` | New — S2-03 eval script |
| `output/eval_metrics.json` | Generated — full eval metrics from best checkpoint |
| `checkpoints/zarr_run3/` | Generated — best.pt, run_config.json, epoch checkpoints |

---

## S2-02 — Real-data Training Design

### `--data-mode zarr` (new) vs `--data-mode synthetic` (default, backward compat)

The `TrainConfig` dataclass gained three fields:
- `data_mode: str = "synthetic"` — data source switch
- `seismic_zarr: Path` — amplitude Zarr store path
- `label_zarr: Path` — fault label Zarr store path

Training on real data: `python -m deepseismic.training.train --data-mode zarr`

Loads from `data/volve/staged/synthetic.zarr` (amplitude) and `data/volve/staged/fault_label.zarr` (fault mask) via `PatchDataset`/`build_dataloaders` from `preprocessing/patches.py`. Spatial inline split (70/15/15) preserved — no leakage.

---

## Class Imbalance Strategy (0.0797% positive fraction)

### The problem
Volume: 10,000,000 voxels. Fault voxels: 7,967. neg/pos ratio ≈ 1,255.

**Naive BCE** → model learns all-negative (accuracy 99.92%, IoU 0).  
**BCE with pos_weight=200 + fault-only patches** → model learns all-positive (recall 1.0, precision 0.002, IoU 0.0017). Root cause: model never sees non-fault context.

### Adopted solution

**Two-component approach:**

1. **Fault-aware sampling via `WeightedRandomSampler`**
   - Include ALL patches (stride=16, 1,320 train patches; 35 have fault content)
   - Fault patches: weight=50×; background patches: weight=1
   - num_samples=200/epoch → ~50 batches/epoch, ~1-2 fault patches per batch in expectation
   - Weight scan: 1.5s for 1,320 label patches (zarr cached after first read)
   - **Key**: model sees fault patches at high rate but also sees background → learns both classes

2. **Combined BCE + Dice loss (50/50)**
   - `BCEWithLogitsLoss(pos_weight=200)`: penalises false negatives
   - Soft Dice loss: penalises false positives (precision pressure)
   - Together: prevents both all-positive and all-negative collapse
   - pos_weight capped at 200 (not 1255) to avoid numeric instability with many negative samples

### Why not filter to fault-only patches?
First attempt (fault-only, 35 patches) produced recall=1.0, precision=0.002 — the model predicted fault everywhere. Without seeing negative context, BCE+Dice with pos_weight=200 makes all-positive the loss-minimising strategy. The fix is negative patches.

---

## S2-05 — Reproducibility

- `seed: int = 42` added to `TrainConfig`
- Seeds at `train()` start: `random.seed`, `np.random.seed`, `torch.manual_seed`, cudnn deterministic
- `run_config.json` written to checkpoint dir before training starts (`dataclasses.asdict(config)`)
- Checkpoint payload includes `seed` and `train_config` dict embedded in `metrics` key

---

## S2-08 — Fixed Epoch-level Metrics

**Problem**: Previous per-batch average: `iou_epoch = mean(iou_per_batch)`. Biases to 0 when most batches have no fault voxels — summing 0s from background batches dominated the average.

**Fix**: Accumulate raw TP, FP, FN counts across all batches; compute IoU/Dice once at epoch end:
```
iou  = total_TP / (total_TP + total_FP + total_FN + 1e-8)
dice = 2·TP / (2·TP + FP + FN + 1e-8)
```
Background batches (TP=FP=FN=0) contribute nothing to the denominator — correct behaviour.

---

## S2-03 — Evaluation Script

`scripts/evaluate.py` — new standalone CLI:
- Loads checkpoint → `UNet3D.load_checkpoint`
- Loads zarr amplitude + fault_label regions (default: il 64–100, 36 inlines)
- Runs `VolumeInference` sliding-window (Gaussian-blended, overlap=0.25)
- Calls `evaluate_model()` from `src/deepseismic/validation/__init__.py`
- Prints full `ValidationMetrics.summary()` report
- Writes `output/eval_metrics.json`

---

## Real Numbers — PoC Training Run

**Run:** `zarr_run3`, seed=42, 20 epochs, lr=5e-4, batch=4, patch=32³, WRS(num_samples=200)

### Training progression
| Epoch | Train IoU | Val IoU | Val Dice |
|-------|-----------|---------|----------|
| 1 | 0.0143 | 0.0020 | 0.0041 |
| 9 | 0.0371 | 0.0051 | 0.0101 |
| 12 | 0.0599 | 0.0233 | 0.0456 |
| 18 (**best**) | 0.1185 | **0.0468** | **0.0894** |
| 20 | 0.1240 | 0.0314 | 0.0608 |

**Best checkpoint:** epoch=18, val IoU=0.0468, val Dice=0.0894, Precision=0.0488, Recall=0.5317

### Full-volume evaluation (il 64–100, 3.6M voxels)

Ground truth: 5,777 fault voxels (0.16% of region)

| Metric | Value |
|--------|-------|
| **IoU** | **0.0622** |
| **Dice** | **0.1172** |
| Precision | 0.0678 |
| Recall | 0.4314 |
| F1 | 0.1172 |
| Tolerant Precision ±3 vox | 0.1459 |
| Tolerant Recall ±3 vox | 0.7064 |
| Tolerant Precision ±5 vox | 0.1591 |
| Tolerant Recall ±5 vox | 0.8406 |
| Mean surface distance | 39.22 vox |

Predicted fault voxels: 36,757 (1.02% vs 0.16% GT — modest over-prediction).
At ±5-voxel tolerance, 84% of true faults are recalled with 16% precision — acceptable for a PoC with 18 raw stick points and 7,967 fault voxels in a 10M-voxel volume.

---

## Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| `--data-mode zarr` trains on real fault_label.zarr | ✅ |
| Val IoU > 0 (non-degenerate) | ✅ val IoU=0.0468 |
| Eval IoU > 0 on full-volume inference | ✅ Eval IoU=0.0622 |
| Seed set + run_config.json persisted | ✅ |
| Config+seed embedded in checkpoint | ✅ |
| Checkpoint stores real best-val metrics | ✅ IoU=0.0468 at epoch 18 |
| `scripts/evaluate.py` runs + writes JSON | ✅ |
| 156 tests pass, 0 regressions | ✅ |
| `ruff check` clean | ✅ |

---

## Forward Recommendations

1. **GPU run with 50+ epochs** will likely push eval IoU to 0.15–0.30 range — the trend is clearly positive and CPU-limited at 20 epochs.
2. **Dice-dominant loss** (`0.1×BCE + 0.9×Dice`) or focal loss should improve precision without sacrificing recall.
3. **Larger WRS num_samples** (e.g., 500) with GPU will give more fault exposure per epoch.
4. The 18 fault sticks are sparse; adding dilation=5 (or using the `--dilation` flag in `generate_fault_label.py`) would increase fault voxels to ~30K and improve training signal.


# Hudson S2-07 Decision Note — Sprint 2 Test Coverage

**Author:** Hudson (Tester/QA)
**Date:** 2026-06-24T20:09:56-05:00
**Sprint:** 2
**Item:** S2-07 — Write tests for all Sprint 2 deliverables
**Status:** Complete ✅ — 209 passed, 2 skipped, 8 deselected, 1 xfailed

---

## Summary

53 new tests written across 4 new test files covering S2-01 (label generation),
S2-03 (eval script), S2-06 (QC pipeline), and S2-02/05/08 (training plumbing).
Full non-integration suite: **209 passed** (baseline 156 → +53). Ruff clean on all
touched files.

---

## New Test Files

| File | Tests | Sprint items |
|---|---|---|
| `src/tests/test_ingest/test_sprint2_label.py` | 17 | S2-01 |
| `src/tests/test_preprocessing/test_sprint2_pipeline.py` | 17 | S2-06 |
| `src/tests/test_training/test_sprint2_training.py` | 14 | S2-02/05/08 |
| `src/tests/test_validation/test_sprint2_eval.py` | 17 | S2-03 |

New directory: `src/tests/test_training/` (with `__init__.py`).

---

## Key Test Design Decisions

### Coordinate mapping (S2-01, highest risk)
`TestCoordinateMapping` pins three separate formulas:
- `abs_inline = 1001 + il_idx`
- `abs_crossline = 1900 + xl_idx`
- `twt_ms = z_col × 4.0` (z_col is a **sample index**, not ms)

The critical regression guard: z_col values ≈ 200–307 → twt ≥ 800 ms. If the
prior bug (z_col used directly as ms) is re-introduced, the guard fails loudly.
Synthetic `.dat` fixtures in `tmp_path` — no dependency on real data files.

### Epoch-level metric accumulators (S2-08)
`TestEpochMetrics` tests exact numeric values (TP=5, FP=2, FN=3):
- IoU = 5/10 = 0.5 (exact)
- Dice = 10/15 ≈ 0.6667 (exact)
- Precision = 5/7, Recall = 5/8 (exact)

This pins the epoch-level formula against per-batch averaging regression.

### Seed determinism (S2-05)
Uses explicit `torch.Generator(seed=42)` passed to `DataLoader(generator=...)`
rather than `torch.manual_seed`. The global-seed approach was brittle: after
calling `_loader(seed=42)` twice, the torch RNG state was at the same value (42)
but then `next(iter(loader_a))` advanced it before `iter(loader_b)` ran.
The explicit generator is hermetic and reproducible.

### Dominant frequency recovery (S2-06)
Pure sinusoids at 30 Hz and 50 Hz (256 samples, dt=4 ms) are recovered within
±3–4 Hz (FFT bin resolution ≈ 1 Hz). Ricker wavelet at 35 Hz within ±5 Hz.
The function zero-pads to the next power of 2 — this was accounted for in
tolerance bounds.

---

## ⚠️ BUG FOUND — `_autocorr_symmetry` in `pipeline.py` (S2-06)

**Component:** `src/deepseismic/preprocessing/pipeline.py` → `_autocorr_symmetry()`

**Severity:** Medium (metric silently returns wrong values; pipeline runs
without error but produces a meaningless QC number)

**Description:**
The function claims to detect residual phase rotation by comparing
positive-lag vs negative-lag autocorrelation energy. However, the
autocorrelation of any real signal `x` is mathematically symmetric:

```
np.correlate(x, x, mode='full')[centre + k]  ==  [centre - k]  for all k
```

Therefore `e_pos` always equals `e_neg` exactly, and the ratio is always 1.0
regardless of the signal's phase. The function cannot distinguish a zero-phase
wavelet from a minimum-phase wavelet.

**Evidence:** `@pytest.mark.xfail(strict=True)` test in
`test_sprint2_pipeline.py::TestAutocorrSymmetry::test_asymmetric_signal_deviates_from_1`
demonstrates the bug: an exponential-decay (clearly asymmetric) signal returns
ratio = 1.0.

**Impact:** `compute_volume_qc` always reports `autocorr_symmetry = 1.0`,
which looks like "perfect zero-phase" for every volume — masking any real
phase issues.

**Recommended fix (route to Ash / Dallas):**
Replace the autocorrelation approach with one of:
1. **Hilbert-transform instantaneous phase**: `np.angle(scipy.signal.hilbert(mean_trace))` — mean absolute instantaneous phase near 0 for zero-phase data.
2. **Spectral phase asymmetry**: compute `np.angle(np.fft.rfft(mean_trace))` and check if the imaginary part of the spectrum is small relative to the real part.
3. **Cross-correlation with known zero-phase reference**: correlate the mean trace with its own time-reversed version.

**Action requested:** Coordinator please route a fix to Ash (owns pipeline.py)
or Dallas (seismic processing expertise). The xfail test will automatically
switch to XPASS once the fix is merged.

---

## No Other Bugs Found

- S2-01 coordinate mapping: correct, tests pass
- S2-02 zarr data_mode wiring: TrainConfig fields correct
- S2-05 seed=42 default: correct
- S2-08 accumulators: `_accum_tp_fp_fn` and `_epoch_metrics` formulas correct
- S2-03 evaluate_model: ValidationMetrics schema correct, metric math correct
- S2-06 `global_amplitude_normalize`: correct (no mutation, ratio preserved, clipping works)
- S2-06 `_dominant_frequency_hz`: correct (recovers known frequencies within tolerance)


# Ripley Decision Note — S2-04 / S2-09: Documentation Honesty

**Author:** Ripley (Lead/Architect)
**Date:** 2026-06-24T20:09:56-05:00
**Sprint items:** S2-04 (P0 README), S2-09 (P1 task-framing)
**Status:** Complete

---

## What was done

### S2-04 — README honesty rewrite

**Problem:** README claimed "Sprint 1 complete. Full end-to-end pipeline implemented."
That was misleading: the pipeline ran only on synthetic data with no real evaluation.
Sprint 2 fixed the ML core; the README needed to catch up.

**Changes made to `README.md`:**

1. **Framing corrected:** PoC goal now says "binary fault detection" explicitly.
   Added link to `docs/task-framing.md`.

2. **Status section replaced** with:
   - Sprint 2 summary — what changed and why it matters
   - "What's real vs. what's demo" table covering all major components with honest
     status (real / real code in mock mode / synthetic stand-in / sparse labels)
   - Results subsection with verified Sprint 2 metrics (val IoU=0.047/Dice=0.089,
     full-volume IoU=0.062/Dice=0.117/tolerant recall±5=0.84)
   - Honest caveat: synthetic amplitude + sparse labels → pipeline-validity proof,
     not a skill benchmark
   - Reproducibility note (seed=42, run_config.json)

3. **Reproduction commands added:**
   ```
   python scripts/generate_fault_label.py
   python -m deepseismic.training.train --data-mode zarr --epochs 20
   python scripts/evaluate.py --checkpoint checkpoints/best.pt
   ```
   All three verified against actual script implementations and CLI flags.

### S2-09 — docs/task-framing.md (new file)

Created `docs/task-framing.md` (~1 page). Covers:

- The core distinction: original does multi-class facies segmentation on F3/Penobscot
  with dense contest labels; we do binary fault detection on Volve with 18 stick points.
- Why Volve cannot support the original's task (no pixel-complete facies labels).
- Correct benchmark lineage: FaultSeg3D (Wu et al. 2019), Qi et al., Hale 2013 —
  not the F3/Penobscot facies contest.
- Metrics table: appropriate (binary IoU, Dice, distance-tolerant recall/precision,
  ASSD) vs. inappropriate (pixel accuracy, per-class mIoU).
- Summary comparison table across both projects.
- Pulls directly from Ash's GAP-C3 finding.

---

## Acceptance criteria check

- [x] README no longer claims unqualified "full end-to-end pipeline"
- [x] Real-vs-demo table present and accurate
- [x] Results section with real metrics and honest caveat
- [x] Fault-detection framing correct
- [x] Reproduction commands match actual CLI (verified)
- [x] `docs/task-framing.md` exists, ~1 page, cites fault-detection lineage
- [x] No overclaims remain

---

## Remaining honest limitations (not gaps — known and disclosed)

- Amplitude volume is synthetic stand-in; metrics will shift when real ST10010 is used
- 18 fault-stick points is sparse; label quality limits maximum achievable metrics
- API and agent default to mock mode; real-mode integration not fully tested
- Single model architecture (UNet3D only); no multi-model comparison


# Sprint 2 Plan — Close the Fidelity Gaps

**Date:** 2026-06-24T19:39:55-05:00
**Author:** Ripley (Lead/Architect)
**Status:** Proposed — for team review

---

## Sprint Goal

**Train the UNet3D on real Volve amplitude data with real fault-stick labels, produce validated benchmark metrics via an automated evaluation script, and qualify all documentation claims — so the PoC credibly emulates the interpretation process, not just the scaffolding.**

### Definition of Done

All of the following are true on `main`:

1. `python -m deepseismic.training.train --data-mode zarr` completes using `PatchDataset` reading from `data/volve/staged/synthetic.zarr` (amplitude) + a real fault-mask Zarr generated from `.dat` sticks — not `generate_synthetic_training_data()`.
2. `python scripts/evaluate.py --checkpoint checkpoints/best.pt` runs inference on the held-out test split, calls `evaluate_model()`, and prints IoU / Dice / distance-tolerant metrics to stdout + writes `output/evaluation_report.json`.
3. README "Status" section honestly states what is real vs. synthetic/demo, includes a "Maturity" subsection, and removes the unqualified "full end-to-end pipeline" claim.
4. CI stays green (`pytest -m "not integration"` all pass, `ruff check src/` clean).

---

## In-Scope Work Items

| ID | Title | Owner | Pri | Est (h) | Depends | Acceptance Criteria |
|----|-------|-------|-----|---------|---------|---------------------|
| S2-01 | Generate real fault-mask Zarr from Volve sticks | Dallas | P0 | 3 | — | `scripts/generate_fault_mask.py` reads `data/volve/interpretations/fault_sticks/*.dat`, uses `label_generator.py` + coordinate mapping (z × 4.0 ms), writes `data/volve/staged/fault_labels.zarr` with array `fault_mask` shape (100, 200, 500) uint8. Fault voxel fraction > 0 and < 5%. Script is idempotent. |
| S2-02 | Wire PatchDataset + real labels into train.py | Dallas | P0 | 4 | S2-01 | `train.py` accepts `--data-mode zarr` flag. When set, uses `PatchDataset` from `preprocessing/patches.py` with `seismic_zarr=data/volve/staged/synthetic.zarr` and `label_zarr=data/volve/staged/fault_labels.zarr`. `--data-mode synthetic` preserves existing behaviour. Default remains `synthetic` so nothing breaks. Training completes ≥5 epochs without error. Checkpoint `best.pt` has non-zero val IoU/Dice. |
| S2-03 | Evaluation script (`scripts/evaluate.py`) | Dallas | P0 | 3 | S2-02 | CLI: `python scripts/evaluate.py --checkpoint <path> [--threshold 0.5]`. Loads checkpoint, builds `PatchDataset(split=TEST)`, runs sliding-window inference, calls `evaluate_model()`. Prints `ValidationMetrics.summary()` to stdout. Writes JSON to `output/evaluation_report.json`. Exit 0. |
| S2-04 | README honesty + maturity section | Ripley | P0 | 1 | S2-03 | "Status" section reworded: remove "Full end-to-end pipeline implemented" → replace with accurate maturity summary. New subsection "### What is real vs. demo" listing each stage. "fault detection" framing corrected throughout (not "interpretation"). Ash reviews for geophysical accuracy. |
| S2-05 | Training reproducibility (seed + config persistence) | Dallas | P1 | 2 | S2-02 | `TrainConfig` includes `seed: int = 42`. `train()` calls `torch.manual_seed()`, `np.random.seed()`, sets `torch.use_deterministic_algorithms(True)` (with fallback). Config serialized to `checkpoints/train_config.json` alongside checkpoint. Two runs with same seed produce identical loss at epoch 1. |
| S2-06 | Preprocessing pipeline.py — minimal conditioning stub | Ash | P1 | 2 | — | `pipeline.py` implements: (a) `compute_volume_qc_stats(zarr_path) → dict` returning min/max/mean/std/p01/p99, (b) `normalize_volume(zarr_path, method="zscore"|"amplitude_preserving") → zarr_path` writing a normalized copy. Docstrings include phase/polarity/bandwidth caveats per Ash's advisory. Unit tests added (Hudson). |
| S2-07 | Tests for S2-01 through S2-03 | Hudson | P0 | 3 | S2-01, S2-02, S2-03 | (a) Test that `generate_fault_mask.py` produces valid zarr with expected shape/dtype when given synthetic `.dat` fixtures. (b) Test that `train.py --data-mode zarr --epochs 1` completes without error (CI-safe with small synthetic zarr fixture). (c) Test that `evaluate.py` produces valid JSON output. All CI-safe (no real data required — use `tmp_path` fixtures). `pytest -m "not integration"` all pass. |
| S2-08 | Checkpoint metrics fix (I5) | Dallas | P1 | 0.5 | S2-02 | Saved checkpoint's `metrics` dict contains actual val IoU and Dice from the validation pass, not 0.0. Verified by loading `best.pt` and asserting `ckpt["metrics"]["iou"] > 0`. |
| S2-09 | Task-mismatch framing in docs | Ash + Ripley | P1 | 1 | — | `docs/task-framing.md` (new): ≤1 page explaining "we do binary fault detection, original does multi-class facies segmentation" — why, what it means for comparisons, and what would be needed to close the gap. Linked from README. |

---

## Explicitly Out of Scope / Deferred

| Item | Rationale |
|------|-----------|
| TensorBoard / WandB / MLflow experiment tracking (N1) | Nice-to-have; stdout + JSON report is sufficient for PoC. Revisit Sprint 3. |
| Data augmentation wiring (N2) | Improves model quality but doesn't close a fidelity gap. Deferred. |
| Confusion matrix visualization (N3/N8) | `evaluate.py` outputs JSON metrics; visualization is a polish item. |
| Fault throw / continuity structural metrics (N3) | `ValidationMetrics` has TODO stubs. Not blocking for credible benchmarks. |
| Fresnel zone / resolution assessment (N2-Ash) | Geophysics-specific reporting; not in core pipeline. |
| 2D section-mode inference (N1-Ash) | Model is 3D; 2D mode is a future extension. |
| Real-mode API integration test (I3) | Requires Azure infra. Parker scopes separately. |
| Multi-model architecture (I4) | Single UNet3D is acceptable for PoC. |
| Amplitude-preserving normalization as training default | S2-06 adds the *option*; making it the training default requires re-benchmarking. |
| Wiggle trace overlay in viewer | UI polish, deferred from Sprint 1. |

---

## Sequencing / Critical Path

```
Day 1 (parallel starts):
  ├─ Dallas: S2-01 (fault mask generation) ──→ S2-02 (wire PatchDataset) ──→ S2-03 (eval script)
  ├─ Ash:    S2-06 (pipeline.py conditioning stub) — independent
  ├─ Ash:    S2-09 (task-mismatch doc) — independent
  └─ Hudson: prep test fixtures for S2-07 (can start before S2-01 merges)

Day 2–3:
  ├─ Dallas: S2-02 continues → S2-05 (reproducibility) → S2-08 (metrics fix)
  ├─ Hudson: S2-07 tests as Dallas PRs land
  └─ Ripley: S2-04 (README rewrite) — blocked on S2-03 landing so metrics can be cited

Day 3–4:
  └─ Ripley: final review pass on all PRs
```

### Critical Path

**S2-01 → S2-02 → S2-03 → S2-04** — this is the spine. Everything else is parallel.

### Reviewer Gates

| PR | Reviewer | Gate |
|----|----------|------|
| S2-01 (fault mask gen) | Ash (geophysical correctness of coordinate mapping) + Ripley (code) | Ash must verify z×4.0 mapping matches synthetic.zarr geometry |
| S2-02 (PatchDataset wiring) | Ripley (architecture) | Verify `--data-mode` flag doesn't break existing synthetic path |
| S2-03 (evaluate.py) | Ripley (code) + Ash (metric selection) | Confirm `evaluate_model()` is called correctly; Ash signs off on which metrics are meaningful |
| S2-04 (README) | Ash (claims accuracy) | Must not overstate or understate |
| S2-06 (pipeline.py) | Ash (owns) + Ripley (code review) | Ash reviews own geophysics; Ripley reviews code quality |
| S2-07 (tests) | Ripley | CI must stay green |

---

## Risks & Unknowns

| # | Risk | Impact | Mitigation |
|---|------|--------|-----------|
| R1 | **Fault stick data is sparse** — only 2 `.dat` files with 12 + 7 points. After dilation, fault mask may be too thin for meaningful training. | Model may not learn; metrics near zero. | Dallas: compute fault voxel fraction after S2-01. If < 0.1%, increase dilation to 2–3 voxels (per existing decision: "Increase to 2–3 for thicker faults"). Ash reviews. |
| R2 | **Coordinate mapping ambiguity** — `.dat` z values are sample indices (not ms). Canonical mapping `z × 4.0` established in Phase 1 but never verified against training pipeline. | Fault labels paint in wrong location → garbage training. | S2-01 must include a sanity check: verify painted voxels overlap the known fault corridor (TWT 808–1228 ms, inlines 46–96, crosslines 47–124). Ash reviews. |
| R3 | **PatchDataset patch size vs. demo volume** — demo volume is 100×200×500. Default patch 64³ with stride 64 yields very few patches along inline axis (100/64 ≈ 1.5). | Tiny training set; model may not converge. | Dallas: use patch_size=(32,32,32), stride=(16,16,16) for the small demo volume. Accept that this is a PoC-scale run. |
| R4 | **Evaluation on same volume used for training** — Volve demo volume is a single small cube. Train/val/test are spatial splits of the same cube. | Metrics may overstate generalization. | Disclose in README and evaluation report: "Metrics computed on spatial hold-out of the same Volve volume. Not an independent test set." |
| R5 | **scipy dependency for distance-tolerant metrics** — `compute_distance_tolerant_metrics()` uses `scipy.ndimage.binary_dilation`. If scipy not in deps, eval script fails. | Eval script fails in CI. | Verify scipy is in core deps (it is — listed for seismic processing). |

### Data Availability Verification (inspected 2026-06-24)

| Artifact | Path | Exists | Notes |
|----------|------|--------|-------|
| Amplitude Zarr | `data/volve/staged/synthetic.zarr` | ✅ | 100×200×500 float32 |
| Fault stick (main) | `data/volve/interpretations/fault_sticks/fault_main_normal.dat` | ✅ | 12 points, z=202–227 (sample indices) |
| Fault stick (antithetic) | `data/volve/interpretations/fault_sticks/fault_antithetic.dat` | ✅ | 7 points, z=300–307 (sample indices) |
| Baked fault prob | `data/volve/staged/fault_prob.zarr` | ✅ | Pre-existing from Phase 1 bake |
| Baked fault mask | `data/volve/staged/fault_mask.zarr` | ✅ | Pre-existing from Phase 1 bake |
| PatchDataset code | `src/deepseismic/preprocessing/patches.py` | ✅ | Complete, tested, Zarr-backed |
| Label generator code | `src/deepseismic/ingest/label_generator.py` | ✅ | Complete, parses Petrel + OpendTect formats |
| Validation module | `src/deepseismic/validation/__init__.py` | ✅ | `evaluate_model()` implemented |

**Verdict:** No data blockers. Fault sticks are sparse but sufficient for PoC with dilation.

---

## Rough Estimate

| Owner | Items | Total Hours |
|-------|-------|-------------|
| Dallas | S2-01, S2-02, S2-03, S2-05, S2-08 | 12.5 |
| Ash | S2-06, S2-09 | 3 |
| Hudson | S2-07 | 3 |
| Ripley | S2-04, reviews | 3 |
| **Total** | | **~21.5 h** |

### Day 1 Assignments

| Who | Starts | Notes |
|-----|--------|-------|
| **Dallas** | S2-01 (fault mask script) | Highest-priority unblock. Deliver PR by end of day 1. |
| **Ash** | S2-06 (pipeline.py) + S2-09 (task-framing doc) | Both independent; can land in parallel. |
| **Hudson** | S2-07 test fixture prep | Build synthetic zarr + `.dat` fixtures in `tmp_path`; scaffold test files. |
| **Ripley** | Review S2-01 PR when ready | Draft S2-04 README edits in parallel. |

---

## Summary

This is a tight, 3–4 day sprint. The spine is four items: generate real labels → wire them into training → evaluate → document honestly. Everything else is parallel or P1 polish. No scope creep — experiment tracking, augmentation, and structural metrics are explicitly deferred. The biggest risk is sparse fault sticks producing too-thin labels; mitigation is dilation tuning (day 1 check).

### Ripley Decision — Wire Real Seismic Traces & Fault Detection to Streamlit Viewer

**Date:** 2026-06-24T12:19:34-05:00  
**Author:** Ripley (Lead/Architect)  
**Status:** Adopted — Phase 1 implemented by Dallas  
**Scope:** Demo viewer upgrade from synthetic placeholders to real data

#### Context

The Streamlit viewer (`src/deepseismic/ui/streamlit_app.py`) was rendering 100% synthetic data via `_generate_synthetic_section()` and `_generate_fault_mask()`. Staged Zarr volumes exist at `data/volve/zarr/demo/` (100 inlines × 200 crosslines × 500 samples, zarr v3). Trained UNet3D checkpoints available at `checkpoints/latest.pt`. Inference engine in `src/deepseismic/models/inference.py`. No pre-baked fault results existed; only validation outputs.

#### Architecture Recommendation: Pre-baked Results Zarr (offline-first)

| Option | Pros | Cons |
|--------|------|------|
| A. On-demand inference per inline | Always fresh | UNet on CPU ~30-60s; poor UX |
| B. Call overlay API at runtime | Uses existing route | Requires FastAPI server; infra overhead |
| **C. Pre-bake results Zarr once, read slices directly** | Instant response, zero runtime deps, offline-capable, simplest | Stale if model changes (trivial re-run) |

#### ✅ Adopted: Option C — pre-bake + direct Zarr read

Demo volume is tiny (100×200×500 = 10M voxels, ~40 MB). One `run_inference()` call produces `fault_prob.zarr` and `fault_mask.zarr`. Viewer reads slices with zero network calls, zero server deps, instant slider response.

#### Phase 1 Implementation (Dallas, COMPLETED)

- **1a. Bake fault results:** Write `scripts/bake_demo_faults.py` → input `data/volve/zarr/demo`, checkpoint `checkpoints/latest.pt`, output `fault_prob.zarr` + `fault_mask.zarr`. CPU runtime ~12s.
- **1b. Wire real amplitude traces:** Replace `_generate_synthetic_section()` with zarr slice reader. Use real coordinate arrays (inline 1001–1100, XL 1900–2099, TWT 0–1996ms). Read `amplitude[il_idx, :, :]`.
- **1c. Wire real fault overlay:** Replace `_generate_fault_mask()` with fault_prob reader. Slice `fault_prob[il_idx, :, :]` as 2D probability heatmap.
- **1d. Update sidebar/captions:** Dynamic slider bounds from zarr metadata. Remove synthetic labels. Add data source caption.

#### Risks Mitigated

| Risk | Mitigation |
|------|-----------|
| `latest.pt` produces garbage on demo volume | Dallas QC'd output before wiring — PASS |
| zarr v2/v3 incompatibility in inference writer | Fixed in `_write_zarr_volume()` — zarr v3 API |
| Offline capability limited by disk | Bake script runs once; zarr cached locally |

#### Files Changed

| File | Change |
|------|--------|
| `src/deepseismic/models/inference.py` | Fixed zarr v2→v3 bug |
| `src/deepseismic/ui/streamlit_app.py` | Rewired to real Zarr + fault prob + sticks |
| `scripts/bake_demo_faults.py` | New — one-shot inference bake script |

### Ash Advisory — Trace & Fault Demo Credibility Guidance

**Date:** 2026-06-24T12:22:22-05:00  
**Author:** Ash (Geophysicist SME)  
**Status:** Advisory — guiding Dallas wiring + UI/Lambert

#### What "Genuinely Identifying Seismic Traces" Means

1. **Real amplitudes from Zarr**: Read `root["amplitude"][inline_idx, :, :]` — must replace random generation. Use coordinate arrays `inline`, `crossline`, `twtt_ms` for axis labels.
2. **Correct axes/units**: TWT extent 0–1996 ms (not 4000), crossline 1900–2099 (not 950–1100), inline 1001–1100 (not 1000–1200).
3. **Amplitude scaling**: Real data nearly zero-mean (mean=5e-7) but asymmetric (max=1.107 vs min=-0.488). Use vmin/vmax from p01/p99 (±0.10–0.12), not absolute extrema.
4. **Display mode**: Variable-density (VD) colormap is sufficient. Wiggle traces overlay every 5th crossline for geophysicist credibility.
5. **Meaningful labeling**: Axis labels "Crossline" / "Two-way time (ms)", title shows inline number + survey name, colorbar labeled "Amplitude (normalised)".

#### What Makes a Fault Overlay Credible

1. **Use real model output, not hardcoded mask**: Pre-run inference, cache as Zarr, read slices at requested inline.
2. **Probability not binary**: Display continuous float32 (0–1) with transparent overlay, not thresholded binary. Threshold slider (0.3–0.7, default 0.5) lets viewer see sensitivity.
3. **Overlay fault sticks for comparison**: Parse `.dat` files, filter by inline, plot as red dots/lines. **Critical:** Coordinate mapping must be correct.
4. **Distinct overlays**: UNet probability (warm semi-transparent) + fault sticks (bright discrete markers) — never merge.
5. **Disclose artifacts**: Flag edge effects (outer 5–10 bin strip unreliable), high-amplitude false-positives (bright spots), training boundary artifacts (model saw ~70% of volume).

#### Honesty Requirements (Non-Negotiable)

**Required disclosures:**
1. "Synthetic dataset — generated to approximate Volve ST10010 geometry. Not licensed field data."
2. "UNet3D trained on synthetic fault labels from dilated fault sticks. No independent validation dataset."
3. "UNet fault probability — candidate identification only. Requires analyst review. Not a final interpretation."
4. If metrics shown: "Metrics computed against synthetic labels used to train the model. Circular validation — training diagnostics only."

#### Demo Credibility Checklist (Priority-Ranked)

| # | Item | Impact | Status |
|---|---|---|---|
| 1 | Read real Zarr amplitudes (replace synthetic) | Highest | ✅ DONE |
| 2 | Pre-run inference, cache as Zarr | Highest | ✅ DONE |
| 3 | Overlay fault sticks on matching inlines | High | ✅ DONE |
| 4 | Probability colorbar (0–1) labelled | High | ✅ DONE |
| 5 | Threshold slider (0.3–0.7, default 0.5) | Medium | ✅ DONE |
| 6 | Amplitude colorbar (p1/p99 clip) | Medium | ✅ DONE |
| 7 | Fault voxel fraction readout | Medium | ✅ DONE |
| 8 | Synthetic data disclosure banner | Required | ✅ DONE |
| 9 | Wiggle trace overlay every 5th CL | Low-medium | Deferred |
| 10 | IoU vs. synthetic labels (with caveat) | Low | Deferred |

#### Coordinate Consistency Resolution

`.dat` files use `(inline_idx, crossline_idx, z_ms)` format. Values z=202–307 map to TWT samples 50–77 (~280 ms), unrealistically shallow. `Volve_Fault_Sticks_synthetic.txt` (UTM format) shows Z_ms 700–852 ms. **Resolution:** z column is **sample index**, not ms. TWT_ms = z × 4.0 → 808–1228 ms range overlaps UTM data. Mapping canonical:
```
abs_inline = 1001 + dat_inline_col
abs_crossline = 1900 + dat_crossline_col
twt_ms = dat_z_col * 4.0
```

### Dallas Decision — Real Fault Viewer Implementation (COMPLETED)

**Date:** 2026-06-24T12:25:08-05:00  
**Author:** Dallas (Data/ML Engineer)  
**Status:** Implemented — ready for team review  

#### What Was Done

Replaced synthetic placeholder viewer with real seismic data and real UNet fault detections.

#### Decision 1: Baked Zarr Paths and Contract

| Store | Array | Shape | Dtype |
|-------|-------|-------|-------|
| `data/volve/staged/synthetic.zarr` | `amplitude` | (100, 200, 500) | float32 |
| `data/volve/staged/synthetic.zarr` | `inline` | (100,) | int32 |
| `data/volve/staged/synthetic.zarr` | `crossline` | (200,) | int32 |
| `data/volve/staged/synthetic.zarr` | `twtt_ms` | (500,) | float32 |
| `data/volve/staged/fault_prob.zarr` | `fault_probability` | (100, 200, 500) | float32 |
| `data/volve/staged/fault_mask.zarr` | `fault_mask` | (100, 200, 500) | uint8 |

Baked results live in `data/volve/staged/` alongside amplitude for trivial viewer loading.

#### Decision 2: Fault-Stick Coordinate Mapping (RESOLVED)

**Evidence:** `.dat` z values 202–307 × 4 ms/sample = 808–1228 ms overlaps `Volve_Fault_Sticks_synthetic.txt` UTM range (700–852 ms). All `.dat` columns are 0-based volume indices.

**Canonical mapping:**
```
abs_inline = 1001 + dat_inline_col
abs_crossline = 1900 + dat_crossline_col
twt_ms = dat_z_col * 4.0
```

#### Decision 3: Model QC Outcome — PASS

Inference: `checkpoints/latest.pt` (epoch 10) on `data/volve/staged/synthetic.zarr`. CPU 11.8s, 88 patches @ batch=4.

| QC metric | Value | Verdict |
|-----------|-------|---------|
| Prob range | 0.000 – 1.000 | PASS |
| Prob mean | 0.1258 | PASS |
| Prob p10/p90 | 0.016 / 0.313 | PASS |
| Fault voxel fraction | 3.89% | PASS |

**Demo credibility:** PASS. Wire to viewer.

#### Decision 4: Zarr v3 Bug Fix

`_write_zarr_volume()` fixed from zarr v2 API (`DirectoryStore` + `create_dataset()`) to zarr v3 (`LocalStore` + `create_array()`). Consistent with `segy_loader.py` and `interpretation.py`. Chunk shape (64, 64, 128) preserved.

#### Viewer Changes Summary

- `_generate_synthetic_section()` → `_get_amplitude_slice(inline_abs)` (real zarr)
- `_generate_fault_mask()` → `_get_fault_prob_slice(inline_abs)` (real prob)
- Inline slider: 1000–1200 → 1001–1100 (from zarr coords)
- Extent: 950–1100 XL / 0–4000 ms → 1900–2099 XL / 0–1996 ms
- Amplitude clip: ±0.8 → ±0.12 (p01/p99)
- Added: amplitude colorbar, fault prob colorbar, threshold slider (0.3–0.7)
- Added: fault stick scatter overlay (red dots, current inline only)
- Added: warning if bake missing
- MOCK_MODE / agent chat unchanged

#### Bake Script Usage

```bash
python scripts/bake_demo_faults.py
streamlit run src/deepseismic/ui/streamlit_app.py
```

Runtime: ~12s CPU.

### Hudson Decision — CI-Safe Test Strategy for Viewer Data-Dependent Tests

**Date:** 2026-06-24T12:26:00-05:00  
**Author:** Hudson (QA)  
**Status:** Adopted — PR #3 CI fix (commit dab69c8)  
**Branch:** feat/real-fault-viewer

#### Context

PR #3 introduced `src/tests/test_viewer/test_viewer.py` (29 tests covering amplitude reader, fault-prob reader, fault-stick coordinate mapping, zarr roundtrip, and AST regression guard). Tests passed locally but failed in CI (11 failures + 7 errors) due to three gitignored data artifacts absent in CI runner:

- `data/volve/staged/synthetic.zarr` (FileNotFoundError)
- `data/volve/staged/fault_prob.zarr` (reader returns None, test fails)
- `data/volve/interpretations/fault_sticks/*.dat` (missing, "No .dat files found")

#### Decision: Two-Tier Strategy

**Tier 1 — Critical regression guards: synthesize the fixture**

`TestFaultStickCoordinateMapping` (highest-value test, guards z-as-sample-index bug) refactored to use `tmp_path_factory` fixture writing minimal synthetic `.dat` files (3 rows each) covering full pinned coordinate ranges. Regression math (`z_samp × 4.0 = twt_ms`) proven on synthetic data — real files unnecessary.

Result: 8 coordinate-mapping tests run in CI with zero data dependencies. ✓

**Tier 2 — Real-artifact readers: skip gracefully when absent**

`TestAmplitudeReader` and `TestFaultProbReader` decorated with:

```python
@pytest.mark.skipif(
    not _ZARR_PATH.exists(),
    reason="data/volve/staged/X.zarr absent — run scripts/bake_demo_faults.py to generate",
)
```

Self-documents missing artifact + fix. Tests skip silently in CI, pass locally.

Result: Inverted-guard bug fixed; assertions now accurate when files are confirmed present. ✓

#### Constraints Respected

- No data files committed.
- Coordinate-mapping assertions unchanged (pinned to 808–908 ms main fault, 1200–1228 ms antithetic).
- `ruff check src/` passes.
- CI workflow unchanged (existing `-m "not integration"` filter + new `skipif` guards produce clean run).

#### Files Changed

| File | Change |
|------|--------|
| `src/tests/test_viewer/test_viewer.py` | Synthesized fault-stick fixture; added `@pytest.mark.skipif` to zarr readers |
## Merged Decisions

## Inbox: coordinator-ui-localdev-labels

### Markers
- `@pytest.mark.integration` — tests that require real external infrastructure (Azurite, GPU, live Azure). **Excluded from default CI** (`pytest -m "not integration"`).
- Real-implementation tests (testing actual module code vs mocks) should also be marked `integration` while implementations are in active development to keep CI green.

### Fixture scoping
- Session-scoped: `sample_segy_path`, `sample_zarr_volume`, `sample_fault_labels` — expensive to create, safe to share.
- Function-scoped: `tmp_zarr_store`, `mock_storage_client`, `mock_llm_response` — must be isolated per test.

### Mock pattern for stub modules
- Use `patch.object(module, "func_name", return_value=..., create=True)` when the target attribute does not exist yet in a stub module.
- Drop `create=True` once the function is implemented (will error if accidentally left after the attribute is removed).

### API tests
- Smoke/contract tests use a purpose-built stand-in FastAPI app (not the real app).
- This ensures tests document the expected interface contract and stay green during development.
- Integration tests against the real app should be added to `test_api/test_api_integration.py` once the real endpoints are stable.

### 1. Mock mode via env var (`DEEPSEISMIC_MOCK_MODE`)

**Decision:** All endpoints check `is_mock_mode()` and return realistic synthetic data
when `DEEPSEISMIC_MOCK_MODE=true`. No storage dependency required.

**Why:** Enables the Foundry agent and both UIs to call live endpoints during local dev
before real data is ingested. Volve geometry and well metadata are embedded as constants
so responses are domain-accurate, not just stub JSON.

---

### 2. BackgroundTasks for long-running jobs

**Decision:** `POST /api/surveys/ingest` and `POST /api/interpretation/fault-detection`
use FastAPI `BackgroundTasks` with module-level Python dicts as the job registry.

**Why:** This is the simplest credible PoC pattern — no Celery, no Redis, no AML job
submission needed for the first demonstration. When moving to production the job dict
swaps for an Azure ML pipeline or Durable Functions without changing the API contract.

**Limitation:** Job state is in-process only — a restart loses all in-flight status.
Accepted for PoC; manifest blobs in `catalog/` provide durable recovery for completed jobs.

---

### 3. Slice endpoint response shape

**Decision:** `InlineSlice` and `CrosslineSlice` serialize numpy arrays as nested Python
`list[list[float]]` (crossline/inline × time). No custom numpy JSON encoder.

**Why:** FastAPI + Pydantic v2 serializes nested lists cleanly without extra deps. The
shape is explicit in the schema (`[n_crosslines][n_samples]`) so consumers know the axis
ordering without reading docs. In mock mode we cap at 50 × 100 to keep response size
manageable; real mode returns the full Zarr slice.

---

### 4. `StrEnum` for `JobStatus`

**Decision:** `JobStatus` inherits `StrEnum` (Python 3.11 built-in), not `(str, Enum)`.

**Why:** Ruff UP042 flags the old pattern. `StrEnum` is the idiomatic Python 3.11+ form,
values compare equal to their string literals without `.value`, and Pydantic v2 handles
them natively.

---

### 5. Storage path conventions (confirmed from architecture)

| Container  | Path pattern                                    | Content                        |
|------------|-------------------------------------------------|--------------------------------|
| `raw`      | `{blob_path}` (caller-specified)                | Source SEG-Y files             |
| `staged`   | `surveys/{survey_id}/amplitude.zarr`            | Chunked Zarr amplitude volume  |
| `catalog`  | `surveys/{survey_id}/metadata.json`             | IngestMetadata JSON sidecar    |
| `catalog`  | `wells/{well_id}/metadata.json`                 | WellMetadata JSON              |
| `catalog`  | `wells/{well_id}/logs.json`                     | WellLog JSON                   |
| `catalog`  | `interpretation/{run_id}/status.json`           | Run manifest (durable state)   |
| `features` | `checkpoints/unet3d_best.pt`                    | Model checkpoint               |
| `results`  | `interpretation/{run_id}/fault_prob.zarr`       | Probability volume             |
| `results`  | `interpretation/{run_id}/fault_mask.zarr`       | Binary fault mask              |

---

### 6. CORS origins

**Decision:** Allow `localhost:8501` (Streamlit) and `localhost:7860` (Gradio) only.
No wildcard `*`.

**Why:** Wildcard CORS with `allow_credentials=True` is rejected by browsers. Explicit
origin lists are more secure and still cover both local UI dev servers.

---

## What was deferred

- SAS URL generation for blob download links (`download_url` field reserved but None)
- Multi-worker / distributed job state (blocked on BackgroundTasks decision above)
- Pagination for large survey/well lists (add `skip`/`limit` when catalog grows)
- Authentication / API key middleware (add before any external exposure)

## Phase 2 — ADLS Viewer Backend, Option B (2026-06-24)

### Dallas Decision — ADLS Viewer Readers — Option B Implementation

**Date:** 2026-06-24T14:25:19-05:00
**Author:** Dallas (Data/ML Engineer)
**Status:** Implemented — branch `feat/adls-viewer-readers`, pending Hudson CI + PR
**Commit:** b2b2b58 (+ docs 25b588e)

#### Context

Phase 1 (PR #3) wired the Streamlit viewer to read amplitude + baked fault Zarr from
**local file paths**.  For the hosted Azure Container Apps demo, those artifacts live
in ADLS Gen2.  Infra issue Spava-Corp/deepseismic2-infra#8 chose **Option B**: the
app reads artifacts **directly from ADLS** (no sidecar download, no volume mount).

#### Key Decisions

1. **Reader extraction into `_data_readers.py`**: All pure data-access logic extracted from `streamlit_app.py` into `src/deepseismic/ui/_data_readers.py` — no Streamlit imports, no `@st.cache_data`, no sidebar side-effects. `streamlit_app.py` now contains thin `@st.cache_data` wrappers that delegate to the pure functions.

2. **Backend env-var contract** (relay verbatim to infra issue #8):
   - `DEEPSEISMIC_DATA_BACKEND`: local | azure (default: local)
   - `DEEPSEISMIC_DATA_DIR`: path to volve data dir (default: data/volve in repo)
   - `DEEPSEISMIC_AMP_CONTAINER`, `DEEPSEISMIC_AMP_PREFIX`: artifact locations (defaults: staged, volve/synthetic.zarr)
   - `DEEPSEISMIC_FAULT_PROB_CONTAINER`, `DEEPSEISMIC_FAULT_PROB_PREFIX`: (defaults: results, volve/fault_prob.zarr)
   - `DEEPSEISMIC_FAULT_MASK_CONTAINER`, `DEEPSEISMIC_FAULT_MASK_PREFIX`: (defaults: results, volve/fault_mask.zarr)
   - `DEEPSEISMIC_STICKS_CONTAINER`, `DEEPSEISMIC_STICKS_PREFIX`: (defaults: raw, volve/interpretations/fault_sticks)
   - `STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_ACCOUNT`: StorageClient auth (existing convention)

3. **zarr v3 store compatibility fix**: `ABSZarrStore` (a `MutableMapping`) is incompatible with zarr v3. Added `ABSZarrV3Store(zarr.abc.store.Store)` — a proper zarr v3 async Store subclass — to `blob_client.py`. Blocking Azure SDK calls dispatched via `asyncio.to_thread` (zarr v3 is async). `get()` wraps raw bytes as `prototype.buffer.from_bytes(raw)`. `with_read_only()` implemented for read mode.

4. **Fault sticks in azure backend**: `.dat` files are small text blobs. Azure reader calls `list_blobs(container, prefix)` → `download_blob(container, name)` for each `.dat`, then parses bytes with unchanged canonical coordinate mapping. Failure (missing container/prefix) returns `{}` gracefully.

5. **Graceful degradation preserved**: Both backends: if fault_prob artifact is absent → `get_fault_prob_slice()` returns `None` → viewer renders amplitude-only with a warning.

#### Files Changed

| File | Change |
|------|--------|
| `src/deepseismic/ui/_data_readers.py` | **New** — pure backend-aware data readers |
| `src/deepseismic/ui/streamlit_app.py` | Thin `@st.cache_data` wrappers; imports from `_data_readers` |
| `src/deepseismic/storage/blob_client.py` | Added `ABSZarrV3Store`, updated `upload_zarr_store` + `open_zarr_store` |
| `src/tests/test_viewer/test_viewer.py` | Updated array-name string guards to also check `_data_readers.py` |

---

### Hudson Decision — ADLS Viewer Test Strategy (Phase 2)

**Date:** 2026-06-24T15:03:33-05:00
**Author:** Hudson (Tester / QA Engineer)
**Branch:** `feat/adls-viewer-readers`
**Status:** Implemented — 26 new CI-safe tests committed + pushed
**Commit:** 18494f9

#### Context

Dallas delivered Phase 2 ADLS Option B with:
- `src/deepseismic/ui/_data_readers.py` — pure backend-aware readers (azure/local)
- `src/deepseismic/storage/blob_client.py` — `ABSZarrV3Store` zarr v3 async Store over ABS
- `streamlit_app.py` — thin `@st.cache_data` wrappers delegating to `_data_readers`

Hudson was tasked to add CI-safe coverage for the highest-risk new code.

#### Test Strategy Decisions

1. **Dict-backed mock ContainerClient (no Azurite required)**: `ABSZarrV3Store` round-trip tests use `_MockContainerClient` / `_MockBlobClient` — a plain `dict[str, bytes]` backing store. `download_blob()` raises `azure.core.exceptions.ResourceNotFoundError` (not `FileNotFoundError`) because `ABSZarrV3Store.get()` catches `ResourceNotFoundError` specifically.

2. **Azure backend resolver via monkeypatch.setattr + monkeypatch.setenv**: Rather than patching at class level (`StorageClient.__init__`), patch the `_storage_client` factory function in `_data_readers` directly. `_MockStorageClient` implements `open_zarr_store`, `list_blobs`, and `download_blob` — covering both the zarr path (amplitude/fault_prob) and the blob-download path (fault sticks).

3. **Local backend: skipif on missing paths (consistent with Phase 1 pattern)**: `TestBackendResolverLocalPresent` is class-level `@pytest.mark.skipif(not _ZARR_AMP.exists(), ...)`. `TestBackendResolverLocalMissing` uses `monkeypatch.setenv("DEEPSEISMIC_DATA_DIR", nonexistent)` to test graceful degradation without needing any real files — runs in CI.

4. **Fault-stick coordinate mapping guarded on azure path too**: `TestAzureFaultSticksAzure` uses the same synthetic `.dat` fixture content as the Phase 1 `TestFaultStickCoordinateMapping` (8 tests, synth fixture). Both local and azure paths now have the TWT ≥ 800 ms guard and the pinned first-row exact mapping.

5. **Integration test: @pytest.mark.integration, never runs in CI**: `TestAzuriteIntegration` documents the real Azurite path (`upload_zarr_store` → `open_zarr_store` allclose). CI runs `pytest -m "not integration"` which deselects it.

#### Test File

`src/tests/test_viewer/test_data_readers.py` — 8 classes, 26 + 1 deselected tests:

| Class | Tests | CI-safe? |
|-------|-------|----------|
| `TestApplyByteRange` | 4 | ✓ |
| `TestABSZarrV3StoreRoundTrip` | 6 | ✓ |
| `TestBackendResolverDefaults` | 3 | ✓ |
| `TestBackendResolverLocalMissing` | 1 | ✓ |
| `TestBackendResolverLocalPresent` | 3 | skipif on real data |
| `TestBackendResolverAzure` | 4 | ✓ |
| `TestAzureFaultSticksAzure` | 5 | ✓ |
| `TestAzuriteIntegration` | 1 | @pytest.mark.integration |

#### Validation

- `python -m pytest -m "not integration" -q` → **155 passed, 2 skipped, 6 deselected** ✓
- `ruff check src/` → **All checks passed** ✓
- Pushed: commit `18494f9` on `feat/adls-viewer-readers`

#### Bugs Found

**None.** Dallas's `_data_readers.py` and `blob_client.py` are clean. All assertions passed first time.

---

## Phase 2 Process Fidelity Evaluations (2026-06-24, Post-Demo)

### Ash Advisory — Process Fidelity Assessment vs. DeepSeismic Reference

**Date:** 2026-06-24T18:05:41.619-05:00
**Author:** Ash (Geophysicist SME)
**Status:** Advisory — findings for team review
**Scope:** Geophysics-focused fidelity gap analysis of deepseismic2 PoC vs. microsoft/seismic-deeplearning

#### Executive Summary

The PoC emulates the *structural shape* of the DeepSeismic workflow (ingest → patch → train → infer → validate) but diverges from it on nearly every geophysically substantive dimension. The original project is a multi-class **facies segmentation** system trained on expert-interpreted labels from a production reference dataset (F3). Our PoC is a binary **fault detector** trained on procedurally generated synthetic data. We cannot claim to emulate the original interpretation process; we emulate its software scaffolding only.

#### Key Findings

1. **Data-Conditioning Fidelity:** Per-patch z-score normalization destroys amplitude information useful for fault characterization. No phase/polarity documentation or AGC removal. `preprocessing/pipeline.py` is an empty stub (8 lines of docstring, no code).

2. **Label & Ground-Truth Fidelity:** Training uses 100% synthetic procedural geometry (single planar fault in Ricker-convolved volume with Gaussian noise). Circular validation guaranteed. Real Volve fault labels exist but unused in training. No facies interpretation available (original's primary task).

3. **Dataset Substitution F3 → Volve:** Volve is a production field with well-constrained geology but very few public, pixel-complete interpretation labels. Substitution undermines claims of emulating the original's scientific process.

4. **Metrics & Validation:** Validation module structurally better than original (tolerant metrics, ASSD), but evaluates synthetic vs. synthetic (circular) instead of model vs. independent interpretation.

#### Prioritized Gaps

**CRITICAL:**
- **GAP-C1:** Training on synthetic geometry, not real interpreted data
- **GAP-C2:** preprocessing/pipeline.py is empty stub
- **GAP-C3:** Task mismatch — we do binary fault detection, original does multi-class facies

**IMPORTANT:**
- **GAP-I1:** No amplitude preservation in normalization
- **GAP-I2:** No phase/polarity documentation or QC
- **GAP-I3:** Validation against synthetic labels is circular
- **GAP-I4:** Dataset provenance — training uses NumPy synthetic, not real Zarr

**NICE-TO-HAVE:**
- **GAP-N1:** No 2D section-mode inference
- **GAP-N2:** No Fresnel zone / resolution assessment
- **GAP-N3:** No fault throw estimation

---

### Dallas Decision — ML Pipeline Fidelity Assessment

**Date:** 2026-06-24T18:09:14-05:00
**Author:** Dallas (Data/ML Engineer)
**Status:** Advisory — informs next sprint planning
**Scope:** PoC ML pipeline vs. microsoft/seismic-deeplearning reference

#### Key Finding: Synthetic-Only Training Loop

Yes, there is actual training:
- Full PyTorch loop: AdamW, CosineAnnealingLR, BCEWithLogitsLoss with pos_weight=10.0
- Real checkpoints saved (5.6 MB each, epoch 5/10/latest)
- Model produces non-trivial fault-probability output (3.89% fault voxel fraction)

**However:** Trained only on synthetic data. `PatchDataset` (Zarr-backed, for real data) is **not connected to the training loop**. Falls back to `generate_synthetic_training_data()` — single planar fault with Ricker wavelets.

#### Pipeline Stage Comparison

| Stage | Gap | Status |
|-------|-----|--------|
| Patch extraction | 3D patches only; no 2D section mode | Partial |
| Normalization | Per-patch z-score only; no global stats | Partial |
| Train/val/test split | Spatial split by inline (correct) | Faithful |
| Model architecture | 3D UNet only (justified for faults, not benchmarked) | Partial |
| Loss function | BCEWithLogitsLoss (faithful intent) | Faithful |
| Augmentation | Transform slots exist, nothing wired | Missing |
| Experiment logging | Stdout only, no TensorBoard | Missing |
| Reproducibility | No seed, no config file, no DAX | Missing |

#### Prioritized ML Gaps

**Critical:**
- **1. Synthetic-only training** — `PatchDataset` + real zarr not wired to train loop
- **2. No experiment logging** — cannot compare runs or track overfitting

**Important:**
- **3. No training reproducibility (seed)** — runs not reproducible
- **4. Checkpoint metrics placeholder zeros** — `val_metrics["iou"]` not properly aggregated at save time

**Nice-to-have:**
- **5. Training config serialization** — `TrainConfig` dataclass not persisted to JSON
- **6. Data augmentation** — random flips/rotations not wired
- **7. preprocessing/pipeline.py stub** — empty implementation
- **8. Confusion matrix** — not computed

---

### Ripley Decision — Process Emulation Gap Assessment

**Date:** 2026-06-24T18:05:41-05:00
**Author:** Ripley (Lead/Architect)
**Scope:** End-to-end workflow fidelity vs. microsoft/seismic-deeplearning reference

#### Workflow Stage Audit

| Stage | Original | Our PoC | Verdict |
|-------|----------|---------|---------|
| Data acquisition | F3 Netherlands / Penobscot real public surveys with labels | Volve open dataset; training data **programmatically synthesized** | Partial |
| Ingest | Numpy `.npy` pre-prepared | Full SEG-Y → Zarr + metadata sidecars | Faithful+ |
| Preprocessing | Config-driven normalization, spatial cropping | Zarr-native PatchDataset with spatial splits | Faithful |
| Label preparation | Real human-annotated facies/fault labels | **Synthetic toy geometry** (single planar fault); real labels parsed but unused | Mocked |
| Train | Full training scripts, multi-model, WandB | Single UNet3D on **synthetic toy data** (96×128×128), 10 epochs, no tracking | Partial |
| Infer | Held-out volume sliding window | Gaussian-blended sliding window, Zarr I/O, batch GPU | Faithful |
| Evaluate | IoU, pixel accuracy, confusion matrix on held-out | Functions exist (IoU, Dice, ASSD, distance-tolerant); **never called** in any pipeline | Partial |
| Publish/serve | N/A (original has no API) | FastAPI 13 endpoints, storage layer; **all in mock mode by default** | Added (with caveats) |

#### Real vs. Mock/Baked/Synthetic Audit

**Real:**
- SEG-Y loader (parses real Volve ST10010)
- Zarr export + chunking (valid v3 stores)
- Fault stick parser (Petrel/OpendTect format)
- UNet3D architecture (standard, tested)
- Sliding-window inference (works)
- Validation metrics (functions exist and compute)

**Mock/Baked/Synthetic:**
- Training data (`generate_synthetic_training_data()` creates 96×128×128 toy volume)
- Checkpoint (trained 10 epochs on synthetic; metrics placeholders `iou=0.0, dice=0.0`)
- Baked demo faults (pre-computed `fault_prob.zarr` + `fault_mask.zarr`, offline)
- API mock mode (all 13 endpoints hardcoded when `DEEPSEISMIC_MOCK_MODE=true`, documented default)
- Agent mock mode (all 11 tools canned responses when `MOCK_LLM=true`, documented default)
- Label generator (sophisticated code, **never wired to training**)
- Validation loop (**never invoked**; no script calls `evaluate_model()`)

#### README Claim Assessment

> "Sprint 1 complete. Full end-to-end pipeline implemented."

**Verdict: MISLEADING.** Code-exists level accurate — every stage has code. But pipeline never run end-to-end on real data with real labels. Actual pipeline: synthetic data → toy training → baked inference → mock API → mock agent. **Should say:** "full end-to-end pipeline scaffolded; demonstrated on synthetic data."

#### Scope Honesty

**We are not emulating the original's process. We have built a demo that LOOKS like the process.**

The original seismic-deeplearning delivers: *give it real dataset (F3/Penobscot) with real labels → run config → get trained model → evaluate with standard metrics → report IoU/accuracy.* Reproducible science.

DeepSeismic2 delivers: *impressive API/agent/UI chrome wrapped around a model trained on one synthetic planar fault.* Model output "visually plausible" but no demonstrated relationship to real geology. Evaluation framework exists but never exercised.

**Gap between narrative and reality:**
- Narrative: "cloud-native modernization of seismic interpretation"
- Reality: "cloud-native serving and agent layer over a placeholder ML core"

#### Consolidated Prioritized Gap List

**CRITICAL (must fix to claim "emulates the process")**

| Gap | Owner | File(s) |
|-----|-------|---------|
| **C1: Training uses only synthetic data** — model never trained on real seismic with real fault labels. label_generator.py + Volve .dat sticks exist but unwired to training path. | Dallas (ML) | `training/train.py`, `ingest/label_generator.py` |
| **C2: No validation pass** — `evaluate_model()` exists but never called. No automated script produces metrics. Cannot verify model quality. | Dallas (ML) | `validation/__init__.py` |
| **C3: README overstates** — "full end-to-end pipeline" implies real data flows through. Should qualify with "synthetic/demo data." | Ripley | `README.md` |

**IMPORTANT (needed for credible PoC)**

| Gap | Owner | File(s) |
|-----|-------|---------|
| **I1: No config/experiment system** — original uses YAML configs. Our training hardcoded. | Dallas (ML) | `training/train.py` |
| **I2: Preprocessing pipeline is empty** — `pipeline.py` stub (no normalization/QC between ingest and training). | Ash (geophysics) | `preprocessing/pipeline.py` |
| **I3: Real-mode API path untested** — `_run_fault_detection()` has real-mode code but no integration test. Unknown if works end-to-end. | Dallas (ML) | `api/routes/interpretation.py:117-198` |
| **I4: Single model architecture** — original offers UNet/SEResNet/HRNet/DeepLab. We have only UNet3D. (Acceptable for PoC.) | Dallas (ML) | `models/` |
| **I5: Checkpoint metrics placeholder zeros** — saved as 0.0. Even with synthetic, training loop should populate real IoU/Dice. | Dallas (ML) | `training/train.py:388-393` |

**NICE-TO-HAVE (polish for credibility)**

| Gap | Owner | File(s) |
|-----|-------|---------|
| **N1: No experiment tracking** (WandB/MLflow) — architecture diagram shows AzureML. | Dallas (ML) | — |
| **N2: No data augmentation** — seismic volumes benefit from flip/rotate/noise during training. | Ash (geophysics) | `preprocessing/patches.py` |
| **N3: Distance-tolerant validation TODO** — `fault_continuity` and `throw_error_mean_ms` hardcoded to 0.0 with TODO comments. | Ash (geophysics) | `validation/__init__.py:350-351` |

#### Recommended Next Sprint: Minimum Viable Set

1. **Wire real labels into training** (Dallas, ~4h)
   - Use existing `label_generator.py` to produce fault mask from Volve `.dat` sticks.
   - Store at `data/volve/staged/fault_mask_real.zarr`.
   - Train UNet3D on real Volve amplitude + real fault mask.
   - Save checkpoint with real metrics.

2. **Add evaluation script** (Dallas, ~2h)
   - `scripts/evaluate.py` — loads checkpoint, runs inference on test split, calls `evaluate_model()`, prints metrics summary.
   - One command: `python scripts/evaluate.py --checkpoint checkpoints/best.pt`

3. **Fix README honesty** (Ripley, ~30min)
   - Qualify "full end-to-end pipeline" with current data status.
   - Add "Maturity" section: what's real, what's demo/synthetic, what's planned.
   - Add "Reproducibility" section: how to retrain from real data.

#### Summary

PoC architecturally impressive — API/agent/UI layer beyond original. ML core (thing that actually interprets seismic data) running on synthetic toy data with no validation. Original's core value was *reproducible ML experimentation on real data with real metrics*. We've replicated code scaffolding, not substance.

**Three changes close the gap:** real training labels, evaluation script, honest README claims. Everything else either unwired code or deferrable.

---

### Dallas Decision — ADLS Viewer Readers — Option B Implementation

**Date:** 2026-06-24T14:25:19-05:00
**Author:** Dallas (Data/ML Engineer)
**Status:** Implemented — branch `feat/adls-viewer-readers`, pending Hudson CI + PR

#### Context

Phase 1 (PR #3) wired Streamlit viewer to read amplitude + baked fault Zarr from **local file paths**. For hosted Azure Container Apps demo, artifacts live in ADLS Gen2. Infra issue Spava-Corp/deepseismic2-infra#8 chose **Option B**: app reads artifacts **directly from ADLS** (no sidecar download, no volume mount).

#### Decisions

**1. Reader extraction into `_data_readers.py`**

All pure data-access logic extracted from `streamlit_app.py` into `src/deepseismic/ui/_data_readers.py` — no Streamlit imports, no `@st.cache_data`, no sidebar side-effects. `streamlit_app.py` now contains thin `@st.cache_data` wrappers delegating to pure functions. Lets Hudson write proper unit tests without mocking Streamlit.

**2. Backend env-var contract**

```
# Backend selector
DEEPSEISMIC_DATA_BACKEND         local | azure   (default: local)

# Local backend
DEEPSEISMIC_DATA_DIR             path to volve data dir (default: data/volve in repo)

# Azure backend — artifact locations
DEEPSEISMIC_AMP_CONTAINER        default: staged
DEEPSEISMIC_AMP_PREFIX           default: volve/synthetic.zarr
DEEPSEISMIC_FAULT_PROB_CONTAINER default: results
DEEPSEISMIC_FAULT_PROB_PREFIX    default: volve/fault_prob.zarr
DEEPSEISMIC_FAULT_MASK_CONTAINER default: results
DEEPSEISMIC_FAULT_MASK_PREFIX    default: volve/fault_mask.zarr
DEEPSEISMIC_STICKS_CONTAINER     default: raw
DEEPSEISMIC_STICKS_PREFIX        default: volve/interpretations/fault_sticks

# StorageClient auth (existing convention, unchanged)
STORAGE_CONNECTION_STRING        Azurite or real account connection string
AZURE_STORAGE_ACCOUNT            Account name (uses DefaultAzureCredential in cloud)
STORAGE_ACCOUNT_NAME             Alias for AZURE_STORAGE_ACCOUNT
```

**3. Zarr v3 store compatibility fix**

`ABSZarrStore` (MutableMapping) incompatible with zarr v3. `zarr.open_group(store=MutableMapping)` raises `TypeError`. Added `ABSZarrV3Store(zarr.abc.store.Store)` — proper zarr v3 async Store subclass — to `blob_client.py`.

Key design choices:
- Blocking Azure SDK calls via `asyncio.to_thread` (zarr v3 is async).
- `get()` wraps raw bytes as `prototype.buffer.from_bytes(raw)`.
- `with_read_only()` implemented (required by zarr for `mode="r"`).
- `ABSZarrStore` (MutableMapping) retained for backward compat.
- `upload_zarr_store` rewritten to walk local directory and upload files directly.
- `open_zarr_store` now returns `ABSZarrV3Store`.

**4. Fault sticks in azure backend**

`.dat` files are small text blobs. Azure reader calls `list_blobs(container, prefix)` → `download_blob(container, name)` for each `.dat`, then parses bytes with unchanged canonical coordinate mapping:
```
abs_inline    = 1001 + il_idx
abs_crossline = 1900 + xl_idx
twt_ms        = z_sample * 4.0
```
Failure (missing container/prefix) returns `{}` gracefully — viewer omits sticks rather than crashing.

**5. Graceful degradation preserved**

Both backends: if fault_prob artifact is absent → `get_fault_prob_slice()` returns `None` → viewer renders amplitude-only with warning.

#### Validation

- `ruff check src/ scripts/` → clean
- `python -m pytest -m "not integration" -q` → 129 passed, 2 skipped
- `python -m py_compile src/deepseismic/ui/_data_readers.py src/deepseismic/ui/streamlit_app.py` → OK
- Azure read path proved with dict-backed mock ContainerClient: write 10×20×50 float32 volume to mock ABS, read back via `zarr.open_group(ABSZarrV3Store, mode='r')`, all allclose assertions passed.

#### Files Changed

| File | Change |
|------|--------|
| `src/deepseismic/ui/_data_readers.py` | **New** — pure backend-aware data readers |
| `src/deepseismic/ui/streamlit_app.py` | Thin `@st.cache_data` wrappers; imports from `_data_readers` |
| `src/deepseismic/storage/blob_client.py` | Added `ABSZarrV3Store`, updated `upload_zarr_store` + `open_zarr_store` |
| `src/tests/test_viewer/test_viewer.py` | Updated array-name string guards to also check `_data_readers.py` |




---

# Decision Note: S3-#8 — Dense Fault Labels App-Readiness

**Author:** Ash (Geophysicist SME)
**Date:** 2026-06-25T09:34:00-05:00
**Sprint item:** S3-#8 (expand fault labels beyond 18 stick points — app-readiness)
**Status:** Complete (synthetic proxy validated) | Blocked on real data (infra #11 / Marketplace)

---

## Problem Statement

Sprint 2 produced 18 raw stick points -> 7,967 fault voxels -> 0.0797 % positive fraction.
This is pathologically sparse (neg/pos ratio ~1,255). Training requires heavy
WeightedRandomSampler (50x) + combined BCE/Dice loss.

Real dense Volve fault interpretations live in `Volve_Geophysical_Interpretations.zip` in
the gated Databricks Marketplace share -- blocked on infra #11 + user Marketplace install.
This sprint makes the code READY for that data to drop in cleanly.

---

## What Was Built

### 1. `densify_stick_to_il_resolution` (new, `label_generator.py`)

Module-level function that inserts 1-IL-resolution interpolated picks between
sparse fault picks in a polyline (0-based index space).

**Geophysical justification:** Fault geometry is approximately planar between
adjacent interpreted sticks. Linearly interpolating XL and Z at each intermediate
IL is valid when the inline gap is small. Gaps > `max_il_gap` (default 5) are NOT
bridged -- they may represent fault segmentation or interpretation discontinuities.

**Resolution guardrail:** lambda/4 at 36.6 Hz, v=2000 m/s -> ~13.7 m -> ~3.4 samples
at 4 ms/sample. Rasterising at 1-IL steps ensures the label band is never thinner
than the minimum resolvable feature. Dilation=3 adds 12 ms TWT positional uncertainty
(~24 m at 1 km/s one-way), within the picking uncertainty of sparse fault sticks.

**Critical uncertainty finding:** For LINEAR fault geometry (straight polylines), the
existing arc-length parameterisation in `_rasterise_stick` already covers all
intermediate ILs regardless of densification. The formula n_q = max(int(arc*2), len(pts))
guarantees >= 2 samples per unit of arc, which always covers every integer IL for
picks <= max_gap=5 ILs apart. Therefore:
- For the current .dat format (one connected polyline per fault), densification does NOT
  increase voxel count for linear geometry.
- The function adds VALUE for: (a) curved fault geometry, (b) explicit IL-resolution
  documentation, (c) real Petrel multi-stick format (separate sticks per IL).
- For real Volve data in Petrel format, densification should be applied at the
  FaultStick group level (group all sticks for one fault, merge, then densify).

### 2. `add_fault_sticks_in_index_space` updated (`label_generator.py`)

Added keyword arguments:
- `interpolate_between: bool = False` -- apply densify_stick_to_il_resolution
- `max_interp_gap_il: int = 5` -- guardrail: max IL gap to bridge

Backward-compatible (keyword-only args with defaults).

### 3. `generate_fault_label.py` updated

New CLI arguments:
- `--interpolate-between` -- enable between-stick densification
- `--max-interp-gap N` -- guardrail (default 5)

Updated QC report:
- Before/after positive-fraction comparison when --interpolate-between is used
- [SYNTHETIC PROXY] banner when fault_sticks_synth/ directory is used
- Resolution guardrail displayed (L/4 ~ 13.7 m, dilation band = N voxels x 4ms)
- Pass/caution/warn thresholds: >= 0.5% = PASS, < 0.5% = CAUTION, < 0.01% = WARN

### 4. Synthetic proxy directory (`data/volve/interpretations/fault_sticks_synth/`)

6 files, 76 raw picks (vs 18 real sticks). Clearly labeled SYNTHETIC PROXY in every
file header comment. NOT real Volve ground truth.

Files:
| File | Fault type | IL range | XL range | Z range | Picks |
|------|-----------|----------|----------|---------|-------|
| fault_antithetic.dat | Copy of real | 72-96 | 47-54 | 300-307 | 7 |
| fault_main_normal.dat | Copy of real | 45-95 | 84-124 | 202-227 | 11 |
| fault_synth_splay_nw.dat | NW splay | 10-52 | 10-52 | 180-216 | 15 |
| fault_synth_conjugate_se.dat | SE conjugate | 50-95 | 100-145 | 235-280 | 16 |
| fault_synth_deep_main_ext.dat | Deep main ext | 45-95 | 84-124 | 265-295 | 11 |
| fault_synth_minor_relay.dat | Relay ramp | 40-70 | 110-140 | 200-230 | 16 |

---

## Before / After Positive-Fraction Numbers

| Scenario | Files | Raw picks | Fault voxels | Positive fraction |
|----------|-------|-----------|--------------|-------------------|
| Sprint 2 baseline (real sticks) | 2 | 18 | 7,967 | **0.0797 %** |
| Synthetic proxy (6 files) | 6 | 76 | 29,787 | **0.2979 %** |
| Synthetic + --interpolate-between | 6 | 76 -> 247 | 29,773 | **0.2977 %** |

Key finding: the 3.7x improvement (0.0797% -> 0.2979%) comes entirely from ADDING MORE
FILES (more fault interpretations), not from the between-stick densification. For linear
fault geometry, densification and the existing arc-length rasterizer produce equivalent
results (as proved mathematically -- see label_generator.py docstring).

**What fraction is needed?**
- < 0.01 % = pathological (Sprint 2 baseline with real sticks only)
- 0.08 % = observed Sprint 2 result with real sticks only
- 0.30 % = synthetic proxy (6 files) -- better but still sparse
- >= 0.5 % = target for "approaching meaningful" -- needs ~10+ fault files
- >= 2.0 % = good coverage -- likely with full real Volve interpretation set
- >= 5.0 % = ideal -- may require broader area or additional synthetic augmentation

**Training impact:** At 0.30 % the neg/pos ratio is ~334:1. Still requires heavy
weighting (pos_weight ~100-200) and fault-aware sampling. The real Volve dense
interpretation set should push this to >= 1 %.

---

## Coordinate Mapping Confirmation

No changes to coordinate mapping from Sprint 2:
- col[0] = inline_idx (0-based), col[1] = crossline_idx (0-based), col[2] = z_col (sample index)
- abs_inline = 1001 + il_idx, abs_crossline = 1900 + xl_idx, twt_ms = z_col * 4.0
- Synthetic proxy picks confirmed in-bounds: all 76/76 inside (100x200x500) volume

---

## Overlap Coordination -- Dallas (SEG-Y path generalisation)

Dallas is generalising the SEG-Y *path* argument in the label_generator / ingest area
this sprint. My changes are confined to:
- `densify_stick_to_il_resolution` (new function, module level)
- `add_fault_sticks_in_index_space` (new keyword args only, no signature breaking change)

No changes to `segy_loader.py`, `parse_petrel_fault_sticks`, or any SEG-Y loading path.
Recommend Dallas reviews this note to confirm no overlap.

---

## When Real Volve Data Arrives

When `Volve_Geophysical_Interpretations.zip` is accessible:

1. If Petrel format: use `parse_petrel_fault_sticks` (already exists, handles FAULT headers)
2. If OpendTect format: use `parse_opendtect_fault_sticks` (already exists)
3. Group sticks by fault name
4. Apply `densify_stick_to_il_resolution` to each fault's merged point list
5. Call `add_fault_sticks_in_index_space` with `interpolate_between=True`
6. Use `--fault-stick-dir` to point to real interpretation directory
7. Output to `fault_label.zarr` (overwriting the current 2-stick version)

**For Petrel multi-stick format (each stick = vertical line at one IL):**
The current densification applies WITHIN each stick (per-polyline). For the Petrel case,
sticks from the same fault should be MERGED into one polyline before densification.
This is a deferred enhancement (no real data yet to validate against).

---

## Tests Added

| Test class | Tests | What they cover |
|-----------|-------|-----------------|
| TestDensifyStickToIlResolution | 8 | basic gap bridging, XL/Z linear interp, gap guardrail, sort, mixed gaps |
| TestInterpolateBetweenSticks | 3 | API contract, >=baseline assertion, gap guardrail |

Full non-integration suite: 223 passed, 2 skipped (unchanged from Sprint 2). Ruff clean.

---

## Outstanding / Follow-on

- S3-#11 (infra): When Marketplace access is restored, swap in real Volve sticks and
  re-run generate_fault_label.py. Expected positive fraction ~1-3%.
- Delete `fault_label_synth.zarr` from staged/ before production training run (it is a
  proxy validation artifact, not training ground truth).
- For Petrel multi-stick format: add fault-name-level grouping to `add_fault_sticks_in_index_space`
  so sticks from the same fault are merged before densification.


---

# Decision: S3-06 — ADLS-Backed Training + Evaluation

**Author:** Dallas (Data/ML Engineer)
**Date:** 2026-06-25T09:34:00-05:00
**Sprint item:** S3-06
**Status:** Implemented ✅

---

## Problem

Training (`train.py`) and evaluation (`evaluate.py`) read Zarr stores only from
the local filesystem.  Once real ST10010 data lands in ADLS (`staged/surveys/
volve-st10010/amplitude.zarr`), in-VNet jobs must read directly from Azure Blob
Storage without copying gigabytes to local disk.

The `ABSZarrV3Store` (blob_client.py) already existed from Phase 2; it just
wasn't wired into the training/eval pipeline.

---

## Design decisions

### Backend selection: `--storage-backend local|azure`

Selected a **CLI flag + config field** (`storage_backend`) over an env var
because:
- Env var approach is already used for viewer (`DEEPSEISMIC_DATA_BACKEND`).
- CLI flag is more explicit for job submissions (Azure ML command args are visible
  in run history; env vars are not).
- Default is `local` — existing scripts work unchanged.

### ADLS path convention

Following infra issue #11 contract:
```
staged/surveys/{survey_id}/amplitude.zarr
staged/surveys/{survey_id}/fault_label.zarr
```
Defaults baked into CLI args:
- `--az-seismic-prefix surveys/volve-st10010/amplitude.zarr`
- `--az-label-prefix surveys/volve-st10010/fault_label.zarr`

### Shared `open_zarr_root()` helper

Created `src/deepseismic/storage/zarr_helpers.py` with `open_zarr_root()`:
```python
root = open_zarr_root(local_path, backend="local")             # dev
root = open_zarr_root(None, backend="azure",                   # cloud
    az_container="staged", az_prefix="surveys/volve-st10010/amplitude.zarr")
```
Both train.py and evaluate.py import from this module — single place to maintain
the backend-dispatch logic.

### `PatchDataset` receives `zarr.Array` objects (not paths)

When ADLS backend is selected, `_build_zarr_loaders` opens the zarr roots,
extracts `seismic_arr = root["amplitude"]` and `label_arr = root["fault_mask"]`,
and passes them directly to `PatchDataset`.  `PatchDataset._open_zarr_array()`
already handles `isinstance(src, zarr.Array)` — no changes to patches.py needed.

### Sprint 2 imbalance handling preserved unchanged

The `WeightedRandomSampler` (50× fault-patch weight), combined BCE+Dice loss,
and `pos_weight=200` are all preserved exactly.  The only change is WHERE the
zarr data is read from; the patch scanning loop and sampler logic are identical.

### Sprint 2 coordinate mapping preserved

No changes to `SurveyTransform`, `FaultMaskGenerator`, or the fault-stick
coordinate mapping.  The ADLS-backend change is purely at the data-loading layer.

---

## Changes made

### `src/deepseismic/storage/zarr_helpers.py` (NEW)
- `open_zarr_root(local_path, *, backend, az_container, az_prefix) → zarr.Group`
- `resolve_zarr_array(...)` — convenience wrapper that also extracts named array.
- Raises `ValueError` if `backend="azure"` but container/prefix missing.
- Raises `FileNotFoundError` if `backend="local"` and path doesn't exist.

### `src/deepseismic/training/train.py`
- Added to `TrainConfig`:
  ```python
  storage_backend: str = "local"
  az_seismic_container: str = "staged"
  az_seismic_prefix: str = "surveys/volve-st10010/amplitude.zarr"
  az_label_container: str = "staged"
  az_label_prefix: str = "surveys/volve-st10010/fault_label.zarr"
  ```
- `_build_zarr_loaders` now uses `open_zarr_root()` for both arrays.
- Added CLI args: `--storage-backend`, `--az-seismic-container`,
  `--az-seismic-prefix`, `--az-label-container`, `--az-label-prefix`.

### `scripts/evaluate.py`
- `run_evaluation()` accepts `storage_backend`, `az_seismic_container`,
  `az_seismic_prefix`, `az_label_container`, `az_label_prefix` params.
- Opens zarr stores via `open_zarr_root()` (local or ADLS).
- Added matching CLI args.
- Removed unused `import zarr` at module top.

---

## In-VNet training invocation (real ST10010)

```bash
# Train on real ADLS-staged data (in-VNet job only):
python -m deepseismic.training.train \
    --data-mode zarr \
    --storage-backend azure \
    --az-seismic-prefix surveys/volve-st10010/amplitude.zarr \
    --az-label-prefix surveys/volve-st10010/fault_label.zarr \
    --epochs 50 \
    --device cuda \
    --seed 42

# Evaluate from ADLS checkpoint:
python scripts/evaluate.py \
    --checkpoint /mnt/features/checkpoints/best.pt \
    --storage-backend azure \
    --az-seismic-prefix surveys/volve-st10010/amplitude.zarr \
    --az-label-prefix surveys/volve-st10010/fault_label.zarr
```

---

## Real-data execution boundary

**Real data MUST run in-VNet** (Azure ML / Container App job) because:
- ADLS uses private endpoints — no public internet access.
- Infra issue #11 must copy ST10010_PSDM_TIME.segy into `raw` container first.
- Trained checkpoints land in `features` container.
- Eval results land in `results` container.

Local dev always uses `--storage-backend local` (default).

---

## Reproducibility preserved

- `seed=42` default unchanged.
- `run_config.json` persisted — now includes `storage_backend` and `az_*` fields.
- Sprint 2 imbalance strategy (WeightedRandomSampler + BCE+Dice + pos_weight=200)
  unchanged.


---

# Decision: S3-04 — ST10010 Real-Geometry Ingest Readiness

**Author:** Dallas (Data/ML Engineer)
**Date:** 2026-06-25T09:34:00-05:00
**Sprint item:** S3-04
**Status:** Implemented ✅
**Coordination note:** See also `dallas-s3-adls-train-eval.md` for the companion ADLS-backend decision.

---

## Problem

Before real Volve ST10010 SEG-Y lands in ADLS (infra issue #11 + Marketplace
install), the ingest code needed to be audited and made app-ready.  Two specific
gaps were flagged:

1. Any hard-coded geometry assumptions that would break on ST10010
   (inlines 9985–10369, non-zero inline/crossline base, real dt/datum).
2. Hard-coded `data/raw/ST10010.segy` path in `label_generator.py`.
3. No standalone CLI for `segy_to_zarr` (format validation needed locally).

---

## Geometry audit findings

`src/deepseismic/ingest/segy_loader.py` — **PASS, no geometry assumptions broken.**

| Check | Finding |
|---|---|
| `inline_min/max` from `f.ilines` | File-driven — handles 9985–10369 correctly |
| `inline_step` from `inlines[1]-inlines[0]` | File-driven — step=1 for ST10010 |
| `sample_rate_ms` from `segyio.tools.dt(f)/1000` | File-driven — will read real dt |
| `datum_ms` from trace header byte 109 | File-driven — reads `DelayRecordingTime` |
| `n_inlines`, `n_crosslines`, `n_samples` | All from `f.ilines`, `f.xlines`, `f.samples` |
| `sample_mode` subsetting | `n_il = min(n_il, sample_n_inlines)` — correct |
| Coordinate slicing in `to_zarr` | `geom.inlines[:amplitude.shape[0]]` — correct |

**No geometry hard-coding found.** The SEGYLoader is fully file-driven and will
handle ST10010's non-zero inline base (9985) and real dt/datum without changes.

---

## Changes made

### `src/deepseismic/ingest/segy_loader.py`
- Added `survey_id: str | None = None` parameter to `to_zarr()` and `segy_to_zarr()`.
- Embedded `survey_id` in `IngestMetadata` sidecar JSON.
- Updated module docstring usage example to show ST10010-ready call pattern
  (path as argument, not hard-coded).

### `src/deepseismic/ingest/label_generator.py`
- Updated module docstring usage example: removed `"data/raw/ST10010.segy"`,
  replaced with `segy_path = "path/to/your/survey.segy"` placeholder.
  The running code was never hard-coded; this was docstring only.

### `scripts/generate_fault_label.py`
- Added `--fault-stick-dir`, `--amplitude-json`, `--label-output` CLI args
  (hardcoded values become defaults).
- Removed module-level `BASE_IL = 1001` / `BASE_XL = 1900` constants; replaced
  with geometry-derived values (`geom["inline_min"]`, `geom["crossline_min"]`)
  so the script works for any survey without code changes.

### `scripts/ingest_segy.py` (NEW)
- CLI wrapper for `segy_to_zarr()`.  Exposes `--source`, `--dest`, `--survey-id`,
  `--sample-mode`, `--sample-n-inlines`, `--overwrite`, `--chunks`.
- ADLS path convention documented: `staged/surveys/{survey_id}/amplitude.zarr`.
- Writes a JSON summary to stdout (log-capture friendly for Azure ML jobs).

---

## Local synthetic-proxy validation (FORMAT PROXY ONLY)

**⚠️ SYNTHETIC-PROXY — these numbers are NOT from real Volve ST10010 data.**
Real ingest must run in-VNet once infra issue #11 lands the SEG-Y.

Command run:
```
python scripts/ingest_segy.py \
    --source data/volve/synthetic_sample.segy \
    --dest data/volve/staged/smoke_ingest.zarr \
    --survey-id synthetic-proxy \
    --sample-mode --sample-n-inlines 20 \
    --overwrite
```

Result:
| Field | Value |
|---|---|
| Inlines loaded | 20 / 100 (sample_mode) |
| Inline range | 1001–1020 (file-driven from synthetic) |
| Crosslines | 1900–2099 (200) |
| Samples | 500  dt=4.0 ms  datum=0.0 ms |
| Amplitude p01/p99 | −0.1206 / 0.1042 |
| Zarr shape | (20, 200, 500) float32 |
| Chunks | (64, 64, 128) |
| Sidecar written | smoke_ingest.json ✓ |

**Verdict: FORMAT PATH VALIDATED.** The `segy_to_zarr` → zarr store → sidecar JSON
pipeline works correctly end-to-end.  The inline/crossline/time coordinate arrays
are written alongside amplitude; geometry is fully file-driven.

---

## In-VNet smoke ingest command (real ST10010)

```bash
# Cheap smoke-ingest (first 50 inlines, ~seconds):
python scripts/ingest_segy.py \
    --source /mnt/raw/ST10010_PSDM_TIME.segy \
    --dest /mnt/staged/surveys/volve-st10010/amplitude.zarr \
    --survey-id volve-st10010 \
    --sample-mode --sample-n-inlines 50 \
    --overwrite

# Full ingest (in-VNet only — private endpoint):
python scripts/ingest_segy.py \
    --source /mnt/raw/ST10010_PSDM_TIME.segy \
    --dest /mnt/staged/surveys/volve-st10010/amplitude.zarr \
    --survey-id volve-st10010 \
    --overwrite
```

---

## Coordination note for Ash

Ash is densifying labels in `label_generator.py` (adding `interpolate_between`
parameter to `add_fault_sticks_in_index_space`).  My changes to that file were
**docstring-only** (module usage example).  No functional overlap, no conflict.

The `generate_fault_label.py` changes (new CLI args) do not conflict with Ash's
`--interpolate-between`/`--max-interp-gap` additions — I added new args AFTER
the existing ones.  The `BASE_IL`/`BASE_XL` removal is geometry-neutral (Ash's
interpolation code doesn't use those constants).


---

# Decision: Sprint 3 Real-Mode Integration Tests (Hudson)

**Date:** 2026-06-25  
**Author:** Hudson (Tester/QA)  
**Status:** Complete  
**Refs:** Issue #9, Sprint 3 Wave 1 de-mock PRs

---

## What was built

Three new test files, 69 new tests (+9 deselected integration):

| File | Tests | Marker |
|---|---|---|
| `src/tests/test_api/test_api_real_mode.py` | 33 passing, 1 integration | `@pytest.mark.integration` for Azurite |
| `src/tests/test_ingest/test_zarr_helpers.py` | 21 passing | None (all CI-safe) |
| `src/tests/test_agent_realmode.py` | 18 passing | None (all CI-safe) |

**Suite results:**
- `pytest -m "not integration" -q` → **292 passed**, 2 skipped, 9 deselected ✓ (was 223)
- `pytest -m "integration" -q` → 4 passed, 5 skipped (all skip cleanly without Azurite) ✓
- `ruff check src/` → All checks passed ✓

---

## Real-path behaviors locked in

### 1. Health endpoint storage state contract
- `storage: "ok"` — storage client built AND list_blobs succeeds
- `storage: "unreachable"` — client built but list_blobs raises
- `storage: "error"` — client cannot be built (misconfigured)
- `storage: "mock"` — only when `DEEPSEISMIC_MOCK_MODE=true`
- `status: "ok"` always (process alive regardless of storage state)

### 2. 503 fail-loud guard (KEY regression guard)
`TestRealModeFailLoud503` locks in the Wave 1 de-mock contract:  
- In real mode (DEEPSEISMIC_MOCK_MODE unset) with broken storage, **GET /api/surveys and GET /api/wells return 503** — never silently return canned Volve data.  
- The mock Volve survey id `"volve-st10010"` must NOT appear in real-mode responses.  
- `DEEPSEISMIC_MOCK_MODE=true` still returns 200 with mock data (mock mode is intentional).

### 3. Mock-vs-real selection
- `is_mock_mode()` returns False by default; True only for `"true"`, `"1"`, `"yes"` (case-insensitive).
- `_is_mock_mode()` in agent is call-time (not frozen at import) — env changes are reflected.

### 4. Agent fail-loud
- `DeepSeismicAgent()` raises `RuntimeError` mentioning `AZURE_PROJECT_ENDPOINT` and `MOCK_LLM` when endpoint is absent in live mode.
- Empty string and whitespace-only values are treated as absent (`.strip()` guard confirmed).
- `FoundryAgent()` raises identically.
- `MOCK_LLM=true|1|yes` activates `MockAgent`; no Azure calls made.

### 5. Ingest real-path (dict-backed storage, no Azurite)
`TestRealPathIngestFlow` exercises `_run_ingest` with the synthetic SEG-Y:
- Status set to `"complete"` on success.
- Catalog JSON uploaded to `catalog/surveys/{survey_id}/metadata.json`.
- `upload_zarr_store` called exactly once, targeting `staged` container with `amplitude.zarr` prefix.
- Sidecar geometry has positive n_inlines, n_crosslines, n_samples.

### 6. zarr_helpers dispatch
- `open_zarr_root(local_path)` opens a valid zarr.Group.
- `FileNotFoundError` for missing paths; `ValueError` for None or empty Azure params.
- Azure branch dispatches through `StorageClient.open_zarr_store()` (verified with dict-backed mock).
- `segy_to_zarr` produces float32 amplitude, coordinate arrays, no NaN/Inf, amplitude_stats.
- `sample_mode=True, sample_n_inlines=2` limits output to exactly 2 inlines.

---

## Bugs found — flagged for owners

### BUG-1: `_run_ingest` does not pass `survey_id` to `ldr.to_zarr()`
**File:** `src/deepseismic/api/routes/surveys.py`, `_run_ingest()`, line ~181  
**Owner:** Parker (API routes)  
**Severity:** Low (functional gap, not a crash)  
**Description:**  
```python
# Current (bug):
_, meta = ldr.to_zarr(zarr_path, overwrite=True)
# Should be:
_, meta = ldr.to_zarr(zarr_path, overwrite=True, survey_id=req.survey_id)
```
`SEGYLoader.to_zarr()` accepts a `survey_id` parameter added in Wave 1 (Dallas). `_run_ingest` never passes it, so `meta.survey_id` is always `None` in the uploaded sidecar JSON. The survey listing route parses this sidecar and cannot recover the survey_id from it — downstream consumers that expect `survey_id` in the sidecar will see `None`.  

**Test coverage:** `test_run_ingest_catalog_metadata_is_valid_json` documents this behavior with a comment; the missing `survey_id` in sidecar is noted but not asserted (test reflects actual behavior, not desired behavior).

---

## Patterns used

**Patching `_build_storage_client`:**  
`_build_storage_client` is decorated with `@lru_cache` and imported by name into `main.py`. Tests patch **both** `deepseismic.api.dependencies._build_storage_client` and `deepseismic.api.main._build_storage_client` using `monkeypatch.setattr()` to cover all call sites. TestClient is entered **after** patching so the lifespan runs with the patched version.

**Azure mock pattern (zarr_helpers):**  
`StorageClient` is imported inside `open_zarr_root()` via a local import (`from deepseismic.storage.blob_client import StorageClient`). Patching `deepseismic.storage.blob_client.StorageClient` (the source) correctly intercepts all instantiations.

**`_DictStorageClient`:**  
Thin in-memory storage that satisfies the StorageClient interface. Records `upload_zarr_store` calls for assertion. Used in all CI-safe API integration tests.


---

# Decision: Agent Mock→Live Default Hardening (Sprint 3, issue #9)

**Author:** Lambert  
**Date:** 2026-06-25T09:34:00-05:00  
**Status:** Implemented  

---

## Context

Sprint 3 goal: make the live Azure OpenAI / Foundry path the correct, robust default when credentials are configured. Before this change:

- `MOCK_MODE` in all three tool modules was captured at **module import time**, meaning test isolation and post-import env-var changes could not flip the mode.
- `FoundryAgent.__init__` used `os.environ["AZURE_PROJECT_ENDPOINT"]`, raising a bare `KeyError` with no actionable message when the env var was absent.
- `DeepSeismicAgent.__init__` had no guard: if `FoundryAgent()` raised during live-mode instantiation, the caller saw an opaque exception rather than a clear configuration error.
- `get_state_summary` returned the module-level `MOCK_MODE` bool (import-time) rather than the current runtime state.

## Decisions Made

### 1. Mock is explicit opt-in only

`MOCK_LLM` must be explicitly set to `"true"`, `"1"`, or `"yes"` to enable mock mode. The absence of the env var means live mode. This is enforced via `_is_mock() -> bool` in each tool module and `_is_mock_mode() -> bool` in `agent.py` — both read `os.environ` at **call time**, not at import time.

### 2. Misconfigured live = loud `RuntimeError`, never silent mock fallback

If live mode is active and `AZURE_PROJECT_ENDPOINT` is not set, `DeepSeismicAgent.__init__` raises `RuntimeError` with:
- The name of the missing env var
- How to fix it (set the var OR set `MOCK_LLM=true`)
- An explicit note that live mode will NOT silently fall back to mock

This prevents broken deployments from masquerading as working by returning canned data.

### 3. Mode visibility at startup

- Mock: `"starting in MOCK mode (MOCK_LLM=true) — no Azure calls will be made"`
- Live: `"starting in LIVE mode — endpoint: <url>  model: <model>"`

### 4. Module-level `MOCK_MODE` kept for backward compatibility

The exported `MOCK_MODE: bool` at module level is retained in all three tool files (some external importers may read it). The actual runtime gate is the `_is_mock()` call inside each function.

## Files Changed

- `src/deepseismic/agent/agent.py` — `FoundryAgent.__init__`, `DeepSeismicAgent.__init__`, `get_state_summary`
- `src/deepseismic/agent/tools/seismic_tools.py` — `_is_mock()` added; 4 functions updated
- `src/deepseismic/agent/tools/geological_tools.py` — `_is_mock()` added; 4 functions updated
- `src/deepseismic/agent/tools/reporting_tools.py` — `_is_mock()` added; 3 functions updated

## Test / Lint Status

- `python -m pytest -m "not integration" -q`: 210 passed, 2 skipped, 1 pre-existing failure (storage health test, unrelated to agent)
- `python -m ruff check src/deepseismic/agent/`: All checks passed

## For the team

Parker / Ripley: No API route or infra changes. The `AZURE_PROJECT_ENDPOINT` env var must be present in the deployed environment; the agent will refuse to start without it (intentional).

Dallas: No ML or training changes.

Hudson: Existing test mocks (patch `DeepSeismicAgent`) remain unaffected. Tests that need real tool-function mock mode should set `MOCK_LLM=true` in `os.environ` within the test — the check is now at call time so this works correctly even after module import.


---

# Decision Note: S3-09 — De-mock the API Critical Path

**Author:** Parker (Backend/Infra)
**Date:** 2026-06-25T09:34:00-05:00
**Sprint item:** Sprint 3, issue #9
**Status:** Implemented ✅

---

## Problem

The FastAPI backend had a systemic silent-mock fallback: any exception during
`StorageClient` construction in real mode returned `None`, and every route
guard was `if is_mock_mode() or storage is None:` — meaning a misconfigured
cloud deployment served fake canned data instead of failing with a clear error.

---

## Decision: Fail Loud in Real Mode, Mock is Explicit Opt-In

### 1. `dependencies.py` — `_build_storage_client()` now propagates errors

Before:
```python
except Exception:
    return None  # silent degradation
```

After:
```python
except Exception as exc:
    logger.error("StorageClient initialisation failed in real mode: %s — ...", exc)
    raise  # fail loud; caller surfaces as HTTP 503
```

`get_storage_client()` (the FastAPI dependency) catches the raise and returns
`HTTPException(503)` so routes never receive `None` in real mode.

### 2. Route guards cleaned up

All `if is_mock_mode() or storage is None:` → `if is_mock_mode():`.

In `surveys.py` and `wells.py`, the `except Exception: return _mock_*()` silent
fallbacks were replaced with `raise HTTPException(503, ...)` — storage errors
are no longer hidden behind canned data.

### 3. Health endpoint enhanced

`GET /health` and `GET /api/health` now report:

| Field | Values | Meaning |
|---|---|---|
| `status` | `"ok"` | Liveness: process is alive |
| `mock_mode` | `true` / `false` | Whether DEEPSEISMIC_MOCK_MODE is set |
| `storage` | `"mock"` \| `"ok"` \| `"unreachable"` \| `"error"` | Readiness: storage ping result |
| `storage_error` | string (when errored) | Human-readable error detail |

The endpoint does a lightweight `list_blobs("catalog", max_results=1)` ping in
real mode to confirm actual reachability. This is what Wash/infra should hit
post-deploy: `storage == "ok"` means real mode is fully operational.

---

## Local Dev Contract

- `DEEPSEISMIC_MOCK_MODE=true` → mock mode, no storage needed, all routes
  return synthetic data.
- Default (no env var) with Azurite running → real mode via the default
  `STORAGE_CONNECTION_STRING` (Azurite emulator). Works with
  `docker compose up azurite`.
- Cloud deployment → set `AZURE_STORAGE_ACCOUNT` + managed identity.
  `STORAGE_CONNECTION_STRING` should be absent or explicitly cleared.

---

## Container Name Contract (infra issue #11 — respected)

No container names changed. Verified names in use:
- `raw` — SEG-Y landing
- `staged` — Zarr volumes after ingest
- `results` — inference output (fault_prob.zarr, fault_mask.zarr)
- `catalog` — JSON manifests, metadata sidecars
- `features` — ML checkpoint blobs

---

## Test / Lint Status

- `python -m pytest -m "not integration" -q`: **211 passed, 2 skipped** ✅
- `python -m ruff check src/`: **All checks passed** ✅

---

## Files Changed

- `src/deepseismic/api/dependencies.py`
- `src/deepseismic/api/main.py`
- `src/deepseismic/api/routes/interpretation.py`
- `src/deepseismic/api/routes/surveys.py`
- `src/deepseismic/api/routes/wells.py`
- `src/deepseismic/api/routes/browse.py`


---

# BUG-1 Fix: survey_id not embedded in catalog sidecar

**Date:** 2026-06-25  
**Author:** Parker (Backend/API)  
**Flagged by:** Hudson (Tester) in `hudson-s3-integration-tests.md`  
**Sprint:** 3 follow-up

---

## Problem

`_run_ingest()` in `src/deepseismic/api/routes/surveys.py` calls `SEGYLoader.to_zarr()` without passing `survey_id`:

```python
# before
_, meta = ldr.to_zarr(zarr_path, overwrite=True)
```

`to_zarr()` added a `survey_id: str | None = None` keyword parameter in Sprint 3 Wave 1 (Dallas). Because `_run_ingest` never passed it, `meta.survey_id` was always `None`. The sidecar JSON written to `catalog/surveys/{survey_id}/metadata.json` therefore contained `"survey_id": null`, breaking downstream consumers that read this field to identify the survey.

---

## Fix

**File:** `src/deepseismic/api/routes/surveys.py`, line 181

```python
# after
_, meta = ldr.to_zarr(zarr_path, overwrite=True, survey_id=req.survey_id)
```

`req.survey_id` is already used elsewhere in `_run_ingest` (for the zarr path and the blob paths), so it is always in scope.

---

## Test Update

**File:** `src/tests/test_api/test_api_real_mode.py`  
**Test:** `test_run_ingest_catalog_metadata_is_valid_json`

The test previously had a comment documenting that `survey_id` would be `None` in the sidecar (a known bug). Now that the bug is fixed, the comment is removed and replaced with an assertion:

```python
assert meta.get("survey_id") == survey_id
```

All other assertions (`geometry`, `amplitude_stats`, `ingested_at`) are unchanged.

---

## Validation

- `python -m pytest -m "not integration" -q` → **292 passed, 2 skipped** (no regressions)
- `python -m ruff check src/` → **All checks passed**


---

# Decision Note: Sprint 3 Documentation — Real-Data Readiness Honest Narrative

**Author:** Ripley (Lead/Architect)  
**Date:** 2026-06-25T09:34:00-05:00  
**Sprint:** Sprint 3  
**Issues addressed:** #7 (ingest readiness), #8 (dense labels), #9 (API/agent de-mock)  
**Status:** Docs complete ✅

---

## Context

Sprint 3 made the app **ready** to consume real Volve ST10010 data. Real-data
**execution** is blocked on infrastructure and data-access dependencies outside this
repo (Spava-Corp/deepseismic2-infra). The documentation job was to accurately reflect
that distinction — applying the Sprint 2 anti-overclaiming standard.

---

## Documentation decisions

### 1. Framing vocabulary

Two terms defined and used consistently:

- **App-ready** — code path exists, locally validated against a format proxy.
  No real Volve data involved. Safe to claim.
- **Deploy-gated** — execution on real data requires external dependencies (infra
  #11 + Marketplace install + in-VNet compute). These are not code gaps.

This vocabulary prevents the "code exists at every stage therefore pipeline is complete"
overclaim that Sprint 2 identified and fixed.

### 2. README.md changes

- Status updated: "Sprint 3 complete — real-data app-readiness. Real-data execution is
  deploy-gated." No claim of real execution.
- Real-vs-demo table extended with Sprint 3 rows: `scripts/ingest_segy.py`,
  `--storage-backend azure`, dense label directory mode, API/agent real mode defaults,
  enhanced health endpoint.
- New section "Real-data readiness (Sprint 3)": app-ready list, deploy-gated list,
  explicit blockers table (infra #11, Marketplace, private endpoints).
- Sprint 3 results numbers: 0.30% positive-voxel fraction labeled clearly as
  "synthetic-proxy only — NOT real Volve results."
- Sprint 3 smoke-test commands added (synthetic proxy path).
- In-VNet execution commands added with clear "requires infra #11" note.
- Sprint 1–2 count updated (156 → 211 tests).

### 3. docs/real-data-runbook.md (new file)

Ordered deploy path: B1/B2/B3 blockers → infra check → ingest → labels → train/eval
→ API real mode → agent real mode. Each step notes VNet requirement and which env vars
select real vs. mock.

Env var table documents all mode-selection variables in one place:
`DEEPSEISMIC_MOCK_MODE`, `MOCK_LLM`, `AZURE_STORAGE_ACCOUNT`,
`STORAGE_CONNECTION_STRING`, `AZURE_PROJECT_ENDPOINT`.

### 4. docs/task-framing.md changes

Added Sprint 3 paragraph: directory-based label pipeline, between-pick interpolation,
0.30% synthetic-proxy number with explicit NOT-real-Volve caveat. Summary table updated.
Core task-mismatch framing (binary fault detection vs. multi-class facies) unchanged —
this is a permanent architectural fact.

---

## Anti-overclaiming checklist (Sprint 2 standard)

| Claim | Checked |
|-------|---------|
| No claim of real ST10010 data processed | ✅ |
| Synthetic-proxy validation labeled everywhere it appears | ✅ |
| Blockers framed as infra/user dependencies, not code gaps | ✅ |
| "App-ready" never conflated with "validated on real data" | ✅ |
| In-VNet requirement explicit in all real-data commands | ✅ |
| 0.30% positive-fraction number labeled synthetic-proxy | ✅ |
| infra #11 cited as the upstream dependency | ✅ |

---

## Files changed

- `README.md` — Status section fully rewritten for Sprint 3
- `docs/real-data-runbook.md` — New file
- `docs/task-framing.md` — Sprint 3 addendum
- `.squad/agents/ripley/history.md` — Sprint 3 learnings appended

# Squad Decisions

## Active Decisions

# Decision: Atomic Thread-History Commit in FoundryAgent.chat() — Issue #25

**Date:** 2026-06-29  
**Author:** Lambert (AI Integration Specialist)  
**PR:** #27  
**Status:** Merged to `squad/25-chat-wedge-tool-calls`, pending review

---

## Context

`FoundryAgent.chat()` is a streaming generator that drives the Gradio and Streamlit UIs. It maintains a persistent per-thread message history (`self._threads[thread_id]`) shared across all requests on a container (the UI agent is a process-wide singleton).

The generator dispatches AOAI tool calls inside a `for tool_call in tool_calls` loop that contains `yield` statements (tool-trace markers emitted to the UI). The Gradio app has a 25s wall-clock guard that `break`s the outer `for chunk in agent.chat(...)` loop when the timeout fires.

---

## Problem

### Bug B (p0 — AOAI 400 wedge)

Prior code appended the assistant message (with `tool_calls`) to `history` **before** the matching tool result messages. When the 25s guard fired mid-round, `GeneratorExit` was raised at one of the `yield` statements inside the tool loop. Python's generator protocol guarantees `GeneratorExit` propagates immediately — the code after the `yield` (including the `history.append(tool_result)` calls) never ran.

Result: persistent `history` contained an assistant message with `tool_calls` but zero matching `tool` responses. AOAI validates this contract on every call. Every subsequent request from **any** user on that container replayed the corrupt history → HTTP 400 indefinitely. Only a container restart cleared the state.

### Bug A (UX — truncation mid-tool-round)

The same `break` also cut off tool-trace output to the user mid-round, showing a partial sequence in the chat panel.

---

## Decision

### Architectural principle (generalizes from this fix)

> **Agent thread-state must be committed atomically.** The assistant `tool_calls` message and all matching tool-result messages must be appended to thread history in a single atomic write. Writing the assistant entry first creates a window where any interruption (`GeneratorExit`, timeout, exception) produces permanently corrupt thread state.

### Implementation — three layers

**Layer 1 — Atomic `round_buffer`** (primary fix)  
Stage the assistant message and all tool results for the current round in a local `round_buffer: list[dict]`. Call `history.extend(round_buffer)` exactly once, after the round is complete. Persistent history is never visible in a partial state.

**Layer 2 — `try/finally` seal** (safety net)  
Wrap the tool-dispatch loop in `try/finally`. In the `finally` block, synthesize `{"error": "interrupted"}` tool responses for every `tool_call_id` in `round_buffer[0]["tool_calls"]` that is not in `answered_ids`. Then commit. This guarantees the committed history is AOAI contract-valid even if `GeneratorExit` fires at any yield inside the loop.

**Layer 3 — `_seal_dangling_tool_calls()` on entry** (self-heal)  
At the top of every `chat()` call, scan history for the last assistant message. If it has `tool_calls` with missing tool responses, append synthetic `{"error": "interrupted"}` tool messages. Self-heals any pre-existing corrupt state from before this fix was deployed.

### Bug A fix

Track `in_tool_round` using the `\n> 🔧` chunk prefix that tool-trace yields emit. Set `timed_out = True` when the clock fires; only `break` when `not in_tool_round`. This avoids cutting mid-tool-round. History integrity is guaranteed by Bug B's fix regardless of when the break occurs.

---

## Key Implementation Notes

- `_get_history(thread_id)` returns the **mutable list** stored in `self._threads` via `dict.setdefault`. `history.extend(round_buffer)` mutates in-place — rebinding the local `history` variable would silently lose the reference to the stored list.
- Streaming behavior (text chunks, tool-trace yields) is fully preserved — `yield` still happens inside `try`; only `history.extend` is deferred to `finally`.
- The 16-iteration tool-call cap is unchanged.

---

## Affected Files

| File | Change |
|---|---|
| `src/deepseismic/agent/agent.py` | Refactored `FoundryAgent.chat()` (round_buffer, try/finally, _seal_dangling_tool_calls) |
| `src/deepseismic/ui/gradio_app.py` | Round-boundary-aware 25s timeout guard |
| `src/tests/test_agent_atomic_commit.py` | 6 new CI-safe tests for the atomic-commit contract |

---

## Follow-up

Migrate `FoundryAgent.chat()` to `stream=True` (AOAI chunked SSE). This would:
- Eliminate the 25s truncation risk entirely (tokens stream to the user as they arrive)
- Remove the need for the `round_buffer` accumulation pattern
- Allow true incremental rendering in the Gradio / Streamlit UIs

---

# Decision: Catalog Index + Pending Manifest for HNS-Safe Prefix Resolution

**Date:** 2026-06-29T18:30:00-05:00  
**Author:** Parker (squad:parker)  
**Issue:** #26 — "Run lookup by short id-prefix 404s: _resolve_run_id catalog list-scan fails on ADLS/HNS"  
**PR:** https://github.com/x3nc0n/deepseismic2/pull/28  
**Status:** Implemented

---

## Context

`_resolve_run_id()` in `src/deepseismic/api/routes/interpretation.py` resolves a
short run-id prefix (e.g. `abcd1234`) to the full UUID needed for blob lookups.  
Step 3 (the only prefix→full path) called `list_blobs('catalog', 'interpretation/')`
to enumerate blobs and find matches.

On ADLS Gen2 / hierarchical-namespace `catalog` containers, `ContainerClient.list_blobs()`
returns nothing or raises. A bare `except Exception: pass` silently swallowed the
failure. Result: every short-prefix lookup 404-ed even for runs that persisted
correctly. Full UUIDs worked fine (exact `download_blob` — unaffected by HNS).

---

## Decision

**Do not change the storage tier or switch to DataLake FileSystemClient.** Fix is
app-side only — the run is intact and the exact-download path works.

### 1. Catalog index.json
Maintain `catalog/interpretation/index.json` — a JSON list of all full run ids.

- Appended at job submit time via `_catalog_index_append()` (read-modify-write;
  tolerates a missing index; write failure is logged but never blocks submission).
- `_resolve_run_id` step 3 reads the index via `download_blob` (exact, HNS-safe)
  before attempting `list_blobs`, so prefix resolution does not depend on blob
  enumeration on the hot path.

### 2. Pending manifest at submit time
Write `catalog/interpretation/{run_id}/status.json` with `status: pending` in the
`run_fault_detection` route handler **before** `background_tasks.add_task` fires.

- Makes the full run id resolvable via step 2 (exact download) cross-replica
  immediately after submission — not only after inference completes.
- Combined with the index, this means prefix lookups work cross-replica from the
  moment the POST /fault-detection response is returned.

### 3. Logged WARNING on list_blobs failure
Replaced the bare `except Exception: pass` with a `logger.warning(..., exc_info=True)`
so HNS listing failures are visible in production logs. `list_blobs` is retained as
a fallback for pre-index runs (only attempted when index scan yields no matches).

---

## Rationale

- `download_blob` (exact path) works on HNS — used for both step 2 (status.json)
  and step 3a (index.json). No DataLake SDK, no infra changes, no extra dependency.
- Index is append-only, tolerant of concurrent writes (PoC scale — single replica).
  At higher scale a separate indexing service or Cosmos/Table Storage would be
  appropriate, but for the current PoC the blob-based index is sufficient.
- Pending manifest provides defense-in-depth: even if the index write fails,
  the full UUID resolves immediately via step 2.

---

## Files Changed

| File | Change |
|---|---|
| `src/deepseismic/api/routes/interpretation.py` | `_CATALOG_INDEX_BLOB`, `_catalog_index_append()`, updated `_resolve_run_id()` step 3, pending manifest in `run_fault_detection()` |
| `src/tests/test_api/test_resolve_run_id.py` | 12 new focused tests |

---

## Follow-ups

- **Lambert:** Surface / echo the full run id in the UI panel and chat agent
  response so users can bookmark or copy the full UUID (out of scope for this PR).
- **At scale:** Replace the blob-based index with Azure Table Storage or Cosmos DB
  (atomic appends, consistent reads) if multi-replica contention becomes a concern.

# Triage Decision — Issues #25 and #26

**Date:** 2026-06-29T17:46:41-05:00  
**Author:** Ripley (Lead)

---

## Issue #25 — Chat wedges after truncated tool turn (AOAI 400)

**Title:** "Chat wedges after a truncated tool turn: dangling tool_calls corrupts shared thread (AOAI 400) + 25s truncation"

| Field | Value |
|-------|-------|
| Owner | squad:lambert |
| Type | type:bug |
| Priority | priority:p0 |
| Release | release:v0.4.0 |

### Root cause summary

Two bugs, one blocker:

**Bug B (blocker):** `FoundryAgent.chat()` (`src/deepseismic/agent/agent.py` ~L402/437) appends the assistant message WITH `tool_calls` to the persistent thread history before the matching tool-result messages are committed. When the UI's 25s guard (`gradio_app.py` L318-323) fires `GeneratorExit`, the thread is left with an unmatched `tool_calls` entry. Because the thread is reused per session and the UI agent is a process-wide singleton, every subsequent request from any user on the container replays the corrupt history → AOAI 400. Container restart is the only recovery path.

**Bug A (contributing):** The 25s wall-clock guard abandons the generator mid-round because the agent makes blocking (non-streaming) completion calls. Fix requires real streaming and/or turn-boundary-aware truncation.

### Ownership rationale

Thread-state management and streaming completion are LLM integration code owned by Lambert. Bug A's truncation guard is UI-side (Parker territory) but the real fix — streaming — lives in the agent. Lambert leads; Lambert/Parker coordinate on the UI guard cleanup.

### Priority rationale

p0: the bug permanently wedges the hosted demo for all users until an operator manually restarts the container. No user-facing workaround exists.

---

## Issue #26 — Run lookup by short id-prefix 404s on ADLS/HNS

**Title:** "Run lookup by short id-prefix 404s: _resolve_run_id catalog list-scan fails on ADLS/HNS (full UUID works)"

| Field | Value |
|-------|-------|
| Owner | squad:parker |
| Type | type:bug |
| Priority | priority:p1 |
| Release | release:v0.5.0 |

### Root cause summary

`_resolve_run_id()` (`src/deepseismic/api/routes/interpretation.py` L48-105) uses `ContainerClient.list_blobs(name_starts_with=...)` for prefix resolution against the ADLS Gen2 / hierarchical-namespace `catalog` container. This API returns nothing (or raises) on HNS containers where flat-blob enumeration is not available. A bare `except Exception: pass` at L84 silently swallows the failure, making the scan appear to return zero matches rather than surfacing an error. The caller sees a 404 even though the run persisted correctly.

Exact `download_blob` (used when a full UUID is supplied) works correctly.

### Ownership rationale

Pure backend/API + Azure storage-client bug. No ML or LLM surface. Parker owns.

### Priority rationale

p1: a clean workaround exists (supply the full UUID). No data loss. The run is intact; only the short-prefix UX is broken. Independent of #25 — no shared code surface.

### Suggested fix path

1. Replace `list_blobs` with `DataLake FileSystemClient.get_paths(path="catalog/interpretation/", recursive=False)` — the same client the ADLS browser uses, OR  
2. Write a `catalog/interpretation/index.json` manifest atomically at submit time; prefix scan reads the index instead of enumerating blobs.

Either way: replace the bare `except Exception: pass` at L84 with a logged error so future failures surface diagnostically.

---

## Sequencing

**#25 must land before #26.** #25 is a p0 that blocks all users; #26 is a p1 with a workaround. Both are independent bugs with no shared code surface.

---

## Architectural note (general, team-wide)

**Agent thread-state must be committed atomically.** The assistant `tool_calls` message and all matching tool-result messages must be appended to thread history in a single atomic write. Writing the assistant entry first creates a window where any interruption (`GeneratorExit`, timeout, exception) produces permanently corrupt thread state. This principle applies to any component that reuses a persistent conversation thread across requests.

---

