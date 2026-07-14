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
- **SEG-Y ingest pipeline** (segy_loader.py) — parses Volve ST10010 with segyio, exports to Zarr v3 + JSON metadata sidecar
- **Fault label generation** (label_generator.py) — parses Petrel/OpendTect fault sticks, rasterizes to binary masks with configurable dilation
- **Patch extraction** (patches.py) — 3D patches with spatial train/val/test splits (no random leakage), PyTorch Dataset interface
- **UNet3D model** (unet.py) — configurable depth/features (~19M params default), checkpointing, inference engine
- **Sliding-window inference** (inference.py) — Gaussian overlap-blending, batch GPU processing, Zarr output
- **Training loop** (train.py) — AdamW + CosineAnnealingLR, BCEWithLogitsLoss, per-epoch validation, best-checkpoint saving
- **Validation metrics** (validation/__init__.py) — Binary IoU, Dice, Precision, Recall, F1, distance-tolerant metrics, ASSD

## Key Architectural Decisions

**Zarr chunk shape (64, 64, 128):** Inline × crossline × sample. Asymmetry on sample axis reflects seismic data characteristics (many samples, query patterns). 2 MB per chunk fits Azure blob prefetch + SSD cache.

**Spatial train/val/test splits (70/15/15 on inline axis):** Not random; prevents data leakage in spatially correlated volumes with overlapping patches. Industry standard for seismic ML.

**3D UNet over 2D:** Faults are 3D surfaces; 3D context improves detection of oblique planes. Diverges from DeepSeismic (which emphasizes 2D) but well-motivated for fault detection.

**BCEWithLogitsLoss + pos_weight=10:** Numerically stable (no sigmoid in loss). pos_weight=10 balances class imbalance (fault ~4% of voxels).

**Gaussian-blended sliding-window inference:** Soft weighting at patch boundaries prevents discontinuities. Standard in medical imaging; transfers to seismic.

**See history-archive.md for detailed learnings from Sprints 1–2.**

## Sprint 3 — De-Mock + Real-Data Readiness (2026-06-25)

Released v0.4.0 with API/agent de-mock and real-data readiness. Integrated with production data pipelines. All integration tests passing (292/296).

**Completed:**
- De-mock: fail-loud 503 handling, AZURE_PROJECT_ENDPOINT validation
- Real data: ST10010 geometry, survey_id integration
- Dense labels: densify + interpolation (0.30% synthetic)
- Integration tests: 69 new (292 total)
- Docs: README, real-data-runbook, task-framing

**Outcomes:** 292 passed / 2 skipped (unit), 4 passed / 5 skipped (integration), ruff clean, v0.4.0 released.

## Learnings — v0.7.2 Fix Verification (2026-07-13)

### Issue #37 Root Cause and v0.7.2 Fix
- **Root cause:** `validate()` used a hardcoded 0.5 threshold on the ~97%-background F3 distribution → IoU=0 every epoch. `best.pt` saved only on IoU improvement → nothing ever saved in 50 epochs.
- **v0.7.2 ships:** `src/deepseismic/training/train.py` — `validate()` now accumulates tp/fp/fn at 19 thresholds (0.05–0.95) via `_accum_tp_fp_fn_from_probs`, then calls `_sweep_threshold_metrics` to compute IoU@best-threshold + AP.
- **Key functions:** `VAL_THRESHOLD_GRID` (19-point linspace), `_accum_tp_fp_fn_from_probs`, `_sweep_threshold_metrics`, `_average_precision_from_curve`, `_select_best_checkpoint`, `_sweep_probs_metrics`.
- **`best_threshold` / `best_val_iou` / `best_val_ap` persisted** to both `best.pt` checkpoint payload and `run_config.json` (end of `train()`).
- **Leakage gate:** Structural only — no explicit survey_id assert in training code; F3-only guarantee relies on configuring correct zarr paths. Volve defaults exist in `TrainConfig` but are overridden by CLI args for F3 runs.

### Bug Found and Fixed (commit 1d184c6)
- **Bug:** `_select_best_checkpoint` fallback condition `not best_saved and val_metrics["loss"] < best_val_loss` only fired once (epoch 1). In the all-zero-IoU case (50 epochs), best.pt = epoch-1 checkpoint, NOT the best-loss checkpoint.
- **Fix:** Changed to `val_metrics["iou"] >= best_val_iou and val_metrics["loss"] < best_val_loss`. This fires on every epoch where IoU hasn't regressed AND loss improved, giving true best-by-loss tracking while protecting higher-IoU checkpoints from being overwritten.
- **Tests added:** `test_fallback_updates_best_on_subsequent_loss_improvement_when_iou_zero` and `test_fallback_does_not_overwrite_better_iou_checkpoint_with_lower_loss`.

### De-risk Result (synthetic)
- Synthetic sparse-positive (3% faults, probs 0.35 on faults vs 0.02 on background): IoU@0.5 = 0.0000, IoU@best-thr (0.05) = 1.0000. Confirms the sweep fix resolves the zero-IoU symptom.

### Test Coverage (post-fix)
- 389 passed / 2 skipped, ruff clean. All 5 spec items covered by tests; 2 new tests close the loss-fallback gap.

### Key File Paths
- Training: `src/deepseismic/training/train.py` — `validate()`, `_sweep_threshold_metrics`, `_select_best_checkpoint`
- Tests: `src/tests/test_training/test_sprint2_training.py` — `TestValidationThresholdSweep`, `TestBestCheckpointSelection`

