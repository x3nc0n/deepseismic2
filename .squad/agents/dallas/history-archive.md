# Dallas — History Archive

## Early Work Summary (Sprints 1–2, archived 2026-06-25)

### Core Infrastructure (Sprint 1)
- SEG-Y ingest pipeline (segy_loader.py) with Zarr v3 + JSON metadata
- Fault label generation (label_generator.py) with configurable dilation
- 3D patch extraction with spatial train/val/test splits
- UNet3D model with ~19M params, checkpointing, inference
- Sliding-window inference with Gaussian blending
- Training loop with AdamW + CosineAnnealingLR
- Binary IoU, Dice, Precision, Recall, F1, distance-tolerant metrics

### Phase 1 Demo Viewer (2026-06-24)
- Streamlit viewer wired to real Zarr amplitude data
- Pre-baked fault detection from checkpoint
- Fault-stick coordinate mapping resolved (z=sample-index × 4ms)
- zarr v2→v3 API bug fixed
- Model QC: probabilities 0–1, fault fraction 3.89%

### Phase 2 ADLS Infrastructure (2026-06-24)
- Pure data-reader functions (_data_readers.py, no Streamlit)
- ABSZarrV3Store with proper async zarr v3 Store API
- Azure Blob Storage integration with env-var contract
- Env-var contract documented (DEEPSEISMIC_DATA_BACKEND, containers, prefixes)
- Dict-backed mock proof of azure read path

### ML Pipeline Fidelity Assessment (2026-06-24)
- Real training loop wired to synthetic data only
- Stage-by-stage pipeline comparison vs. DeepSeismic reference
- Critical gaps identified: no experiment logging, no training seed
- Recommended fixes: wire real labels (~4h), add eval script (~2h)

### Sprint 2 Real Training (2026-06-24)
- Real training on synthetic fault labels (seed=42, 20 epochs, lr=5e-4)
- Best epoch=18: Val IoU=0.0468, Dice=0.0894
- Full-volume eval: IoU=0.0622, Dice=0.1172
- Tolerant metrics (±3 vox): Precision=0.1459, Recall=0.7064
- Checkpoint stores real IoU/Dice metrics (no more placeholders)
- run_config.json persisted with seed

### Sprint 2 Post-Training (2026-06-25)
- S3-04 Ingest Readiness: ST10010 geometry audit passed
- segy_loader.py already file-driven (no hard-coded geometry)
- generate_fault_label.py: BASE_IL/BASE_XL removed, geometry-derived
- ADLS backend wiring: open_zarr_root() dual-path, pass zarr.Array objects
- Synthetic-proxy smoke ingest validated format path

**Key Learnings:**
- zarr v3 ABSZarrStore requires proper async Store subclass, not MutableMapping
- z_col in fault sticks is sample index, not true milliseconds
- Spatial train/val/test splits prevent data leakage in seismic volumes
- Gaussian-blended sliding-window inference prevents discontinuities
- Config flags for in-VNet jobs documented (--storage-backend azure)

