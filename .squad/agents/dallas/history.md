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

