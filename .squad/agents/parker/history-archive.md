# Parker — History Archive

Older sessions and detailed learning narratives from earlier sprint phases. Retained for reference and team knowledge preservation.

## Archived Sessions
### 2026-06-25 — v0.4.0 Release (Sprint 3 — De-Mock + Real-Data Readiness)
Released v0.4.0 with API/agent de-mock and real-data readiness. Integrated with production data pipelines. All integration tests passing (292/296).
- De-mock: fail-loud 503 handling, AZURE_PROJECT_ENDPOINT validation
- Real data: ST10010 geometry, survey_id integration
- Dense labels: densify + interpolation (0.30% synthetic)
- Integration tests: 69 new (292 total)
- Docs: README, real-data-runbook, task-framing

### 2026-06-10 — Local Dev Environment + Storage Abstraction Layer (Phase 1)
Built full StorageClient + ABSZarrStore, pydantic-settings config singleton, Azurite docker-compose service, multi-stage Dockerfile, one-shot Windows setup script. Auto-detects connection string vs. DefaultAzureCredential. ABSZarrStore is a MutableMapping wrapping ContainerClient (works with zarr 2.x/3.x without adlfs/fsspec[azure]). Zero Azure spend during local dev via Azurite.

### 2026-06-09 — Scribe Cross-Agent Update
Sprint 1 coordination complete. All agents delivered successfully. 5 agents synchronized, 7 decision documents archived, full team context available in decisions.md.

### 2026-06-24 — Scribe Cross-Agent Update (Phase 1 → Phase 2 Transition)
Phase 1 (Real Fault Viewer) complete. Infra deployment follow-up required. Deploy updated Streamlit UI + FastAPI backend to hosted demo. Baked fault probability Zarr volumes must be available at deployment time. No infrastructure scaling needed — data/code update only.

## Archived Learnings

### 2026-06-25 — Sprint 3 Issue #9: De-Mock API Critical Path
src/deepseismic/api/dependencies.py: removed silent xcept Exception: return None. In real mode, if StorageClient() raises (e.g., no env credentials), the exception propagates with clear log message.

get_storage_client() FastAPI dependency catches exception and surfaces as HTTP 503. Routes in real mode never receive None storage — they either get a client or get 503 before handler runs.

health() fully rewritten: status = liveness (always "ok"), storage field = readiness ("mock" | "ok" | "unreachable" | "error"). Does lightweight list_blobs("catalog", max_results=1) ping in real mode for reachability check. storage_error field added for errors.

All route mock guards changed: if is_mock_mode() or storage is None: → if is_mock_mode():. Silent fallbacks removed: xcept Exception: return _mock_*() → aise HTTPException(503, ...).

### 2026-06-25 — Sprint 3 BUG-1: Survey ID Missing from Catalog Sidecar
src/deepseismic/api/routes/surveys.py line 181: ldr.to_zarr(zarr_path, overwrite=True) was missing the survey_id keyword argument. Fixed to: ldr.to_zarr(zarr_path, overwrite=True, survey_id=req.survey_id). Without this, meta.survey_id was always None in the uploaded catalog/surveys/{survey_id}/metadata.json sidecar.

Test updated to assert meta["survey_id"] == survey_id. Key fact: SEGYLoader.to_zarr() accepts survey_id: str | None = None — always pass survey_id=req.survey_id when calling from _run_ingest.

### 2026-07-09 — F3 External Data Sourcing Decision
Cross-survey training run blocked until F3 data is externally sourced. Investigation confirmed: real F3 data NOT in repo (only synthetic proxy). Must ingest from public OpendTect F3 Demo (dGB Earth Sciences / TerraNubis, CC BY-SA). Use parse_opendtect_fault_sticks parser (not Petrel). Leakage gate: F3 = training input only; Volve = scoring/evaluation only (issue #24). T4 GPU workload profile provisioned (Spava-Corp/deepseismic2-infra#23). Data staging must complete before T4 training run. Decision: .squad/decisions.md — F3 Ingest Contract (approved/in-progress).
