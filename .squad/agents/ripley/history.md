# Ripley — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, PyTorch, Azure, LLM APIs (M365 Copilot, GitHub Copilot, Copilot Studio/Foundry)
- **Goal:** Modernize seismic interpretation from legacy monolithic apps (Dell Isilon, Azure Premium Files, Azure NetApp Files) to affordable cloud-native + AI patterns
- **Data:** Equinor Volve dataset (https://www.equinor.com/energy/volve-data-sharing)
- **Reference:** microsoft/seismic-deeplearning repo (UNet, SEResNet, HRNet segmentation on seismic data)
- **User:** jospaid

## Learnings

### Sprint 1 → Sprint 2: FastAPI backend implementation (2026-06-09)

**What was built:**
- `api/main.py` — full app factory with lifespan, CORS (Streamlit 8501, Gradio 7860), health check
- `api/schemas.py` — complete Pydantic v2 models: SurveyMetadata, IngestRequest/Response,
  InlineSlice, CrosslineSlice, InterpretationRequest/Status/Result, FaultOverlay,
  WellMetadata, WellLog, FormationTop, ErrorResponse
- `api/dependencies.py` — `get_storage_client`, `get_settings_dep`, `is_mock_mode()`,
  `StorageClientDep` / `SettingsDep` annotated type aliases
- `api/routes/surveys.py` — 5 endpoints: list, metadata, ingest (BackgroundTasks),
  inline slice, crossline slice; full mock data with Volve ST10010 geometry
- `api/routes/interpretation.py` — 4 endpoints: fault-detection (BackgroundTasks),
  status, results, overlay; mock data with realistic fault probability patterns
- `api/routes/wells.py` — 3 endpoints: list, metadata, logs; mock data for
  Volve wells 15/9-F-11B and 15/9-F-1C with formation tops and GR/DT/RHOB log curves

**Design decisions made:**
- `DEEPSEISMIC_MOCK_MODE=true` env var gates all mock behaviour — enables Foundry agent
  tools to call real endpoints without real storage
- BackgroundTasks (single-process) is the job runner for PoC; module-level dicts store
  run state (survives in-process but not across restarts)
- Inline/crossline slices cap at 50 traces × 100 samples in mock mode (manageable JSON)
- Real mode reads Zarr directly from blob storage via `open_zarr_store`; uploads via
  `upload_zarr_store` after local temp write
- `StrEnum` used for `JobStatus` (Python 3.11+ UP042 compliance)
- All `raise HTTPException` inside `except` blocks use `from None` / `from exc` (B904)
- Ingest response echoes actual job status so mock callers see "complete" immediately

**API contract (base URL: http://localhost:8000):**
```
GET  /health
GET  /api/surveys
GET  /api/surveys/{survey_id}
POST /api/surveys/ingest
GET  /api/surveys/{survey_id}/inline/{number}
GET  /api/surveys/{survey_id}/crossline/{number}
POST /api/interpretation/fault-detection
GET  /api/interpretation/{run_id}/status
GET  /api/interpretation/{run_id}/results
GET  /api/interpretation/{run_id}/overlay/{inline_number}
GET  /api/wells
GET  /api/wells/{well_id}
GET  /api/wells/{well_id}/logs
```

**Ruff passed clean on all six API files.**

## Scribe Cross-Agent Update — 2026-06-10T04:30-05:00
Sprint 1 coordination complete. All agents delivered successfully.
- 5 agents synchronized
- 7 decision documents archived
- Full team context available in decisions.md

## Learnings — 2026-06-24T18:05:41-05:00: Process Emulation Gap Assessment

### Key findings

1. **The ML core is hollow.** Training runs only on programmatically-generated synthetic data (one planar fault in a 96×128×128 toy volume). The model has never seen real geology. `checkpoints/latest.pt` has placeholder metrics (IoU=0.0, Dice=0.0).

2. **Label pipeline exists but is unwired.** `label_generator.py` correctly parses Volve fault sticks and rasterises them. But the training module (`train.py`) never calls it — it generates its own synthetic data instead.

3. **Validation code exists but is never exercised.** `validation/__init__.py` implements IoU, Dice, ASSD, distance-tolerant metrics. No script or pipeline ever invokes `evaluate_model()`.

4. **Default demo is 100% mock.** README tells users to set `DEEPSEISMIC_MOCK_MODE=true` and `MOCK_LLM=true`. Both API and agent return canned data. The Streamlit viewer shows pre-baked inference results.

5. **README "full end-to-end pipeline" claim is misleading.** Code exists at every stage, but the pipeline has never run end-to-end on real data with real labels and real evaluation.

6. **What IS real and good:** SEG-Y ingest, Zarr conversion, UNet3D architecture, sliding-window inference engine, patch extraction with spatial splits, API contract design, agent tool wiring. The serving/consumption layer exceeds the original.

### Critical gaps (3)
- C1: Training on synthetic-only data
- C2: No validation pass
- C3: README overstates maturity

### Minimum fix set for Sprint 2
- Wire real Volve fault labels into training path
- Add `scripts/evaluate.py` that calls `evaluate_model()`
- Qualify README claims with honest maturity section

### Decision
Full gap list written to `.squad/decisions/inbox/ripley-process-emulation-gaps.md`.

## Scribe Consolidation — 2026-06-24T23:29:56Z

Process emulation gap assessment merged into `.squad/decisions.md` (Phase 2 Process Fidelity Evaluations section). Consolidated findings from Ash (geophysics gaps), Dallas (ML pipeline gaps), and Ripley (architecture audit).

**Consolidated verdict:** README "full end-to-end pipeline" claim is MISLEADING. Actual pipeline: synthetic → toy training → baked inference → mock API → mock agent. Should state "pipeline scaffolded; demonstrated on synthetic data."

**Critical gaps (3):** C1 (synthetic-only training), C2 (no validation pass), C3 (README overstates).

**Important gaps (5):** I1 (no config system), I2 (preprocessing stub), I3 (real-mode API untested), I4 (single model), I5 (placeholder metrics).

**Sprint 2 minimum viable:** Wire real labels (~4h Dallas), eval script (~2h Dallas), README fix (~30min Ripley). Stretch: YAML config (~3h Dallas), fill pipeline.py (~2h Ash).

**What IS real and good:** SEG-Y ingest, Zarr export, UNet3D, sliding-window inference, patch extraction with spatial splits, API contract design, agent tool wiring. Serving/consumption layer exceeds original.

## Learnings — 2026-06-24T20:09:56-05:00: Sprint 2 Documentation Honesty (S2-04 / S2-09)

### What was written

**S2-04 — README rewrite:**
- Replaced "Sprint 1 complete. Full end-to-end pipeline implemented" with honest
  Sprint 2 status and a "What's real vs. what's demo" table.
- Added real Sprint 2 metrics: val IoU=0.047/Dice=0.089 (epoch 18); full-volume eval
  IoU=0.062/Dice=0.117/tolerant recall±5=0.84. Caveat states these are pipeline-validity
  numbers, not a skill benchmark (sparse labels + synthetic amplitude stand-in).
- Fixed PoC goal framing to say "binary fault detection" explicitly.
- Added Reproduction commands section with verified CLI (generate_fault_label.py,
  `--data-mode zarr`, scripts/evaluate.py). All flags verified against actual source.

**S2-09 — docs/task-framing.md:**
- New ~1-page doc explaining task difference: original does multi-class facies
  segmentation on F3/Penobscot with dense contest labels; we do binary fault detection
  on Volve with sparse fault sticks.
- Cites correct fault-detection lineage: Wu et al. 2019 (FaultSeg3D), Qi et al.,
  Hale 2013.
- Documents appropriate metrics (binary IoU/Dice/distance-tolerant recall) vs.
  original's per-class mIoU.
- Pulls directly from Ash's GAP-C3 finding in decisions.md — not re-derived.

### Key lesson

The gap between "code exists at every stage" and "pipeline has run on real data with
real metrics" is the hardest kind of overclaim to catch. Sprint 2 closed that gap
at PoC scale; the documentation now draws the line correctly between what is real,
what is demo-grade, and what is an honest limitation.



**Sprint goal:** Train UNet3D on real Volve data with real fault-stick labels, produce validated benchmark metrics, and qualify all documentation claims.

**9 work items scoped (P0: 4, P1: 5). ~21.5 hours total across Dallas, Ash, Hudson, Ripley.**

**Critical path:** S2-01 (fault mask zarr from sticks) → S2-02 (wire PatchDataset into train.py) → S2-03 (evaluate.py script) → S2-04 (README honesty).

**Key risks identified:**
- R1: Sparse fault sticks (19 points total) → may need dilation=2-3
- R2: Coordinate mapping (z × 4.0) must be verified against training pipeline
- R3: Small demo volume (100×200×500) → use smaller patches (32³, stride 16)
- R4: Train/test from same volume — must disclose in docs

**Data verified:** All required artifacts exist (amplitude zarr, fault sticks, PatchDataset code, label_generator, validation module).

**10 items explicitly deferred:** TensorBoard, augmentation, confusion matrix, fault throw metrics, Fresnel assessment, 2D inference, real-mode API test, multi-model, amplitude-preserving default, wiggle traces.

Full plan: `.squad/decisions/inbox/ripley-sprint2-plan.md`

