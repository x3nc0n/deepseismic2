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
