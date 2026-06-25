# Hudson — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, pytest, numpy testing utilities
- **Goal:** Ensure data pipelines, models, and AI integrations work correctly
- **Key concerns:** Floating point precision in seismic data, SEG-Y parsing edge cases, model input shape validation
- **User:** jospaid

## Learnings

### 2026-06-25 — Sprint 3 Issue #9: Real-Mode Integration Tests

**Session:** Added integration tests for Wave 1 de-mock (Sprint 3, issue #9). +69 tests across 3 new files.

**New test files:**
1. test_api_real_mode.py (33 CI-safe + 1 integration) — real-mode API behavior.
2. test_zarr_helpers.py (21 CI-safe) — zarr_helpers dispatch + segy_to_zarr.
3. test_agent_realmode.py (18 CI-safe) — agent fail-loud + mock-vs-real selection.

**Real-path behaviors locked in:**

- **503 fail-loud guard (KEY regression):** In real mode (DEEPSEISMIC_MOCK_MODE unset), broken storage returns 503 on /api/surveys and /api/wells. The canned volve-st10010 survey never appears in real-mode responses.
- **Health endpoint state contract:** storage: "ok"|"unreachable"|"error"|"mock" all tested. "mock" only when DEEPSEISMIC_MOCK_MODE=true.
- **Agent fail-loud:** DeepSeismicAgent() raises RuntimeError mentioning AZURE_PROJECT_ENDPOINT and MOCK_LLM; FoundryAgent() same. Empty/whitespace endpoint treated as absent.
- **is_mock_mode() selection:** Default False; "true", "1", "yes" activate; "false", "", unset deactivate.
- **_is_mock_mode() (agent):** Call-time evaluation — env changes reflected without restart.
- **Ingest real-path:** _run_ingest with dict-backed storage + synthetic SEG-Y → status "complete", catalog JSON uploaded, upload_zarr_store called once.
- **zarr_helpers dispatch:** Local path success/failure, None path ValueError, empty Azure params ValueError.
- **segy_to_zarr:** float32 amplitude, inline/crossline/twtt_ms arrays, no NaN/Inf, positive dims, sample_mode limits inlines.

**Patching patterns:**
- _build_storage_client is @lru_cache'd and imported by name into main.py. Must patch BOTH deepseismic.api.dependencies._build_storage_client AND deepseismic.api.main._build_storage_client.
- _DictStorageClient — tiny in-memory store for CI-safe API tests; records upload_zarr_store calls.
- asyncio.run() for sync-wrapping async zarr store operations in tests.

**BUG FOUND — _run_ingest missing survey_id:**
src/deepseismic/api/routes/surveys.py _run_ingest() calls ldr.to_zarr(zarr_path, overwrite=True) without passing survey_id=req.survey_id. Wave 1 (Dallas) added this parameter, but the route never passes it — meta.survey_id is always None.

**Suite results (2026-06-25):**
- pytest -m "not integration" -q → **292 passed**, 2 skipped, 9 deselected ✓ (baseline was 223)
- pytest -m "integration" -q → 4 passed, 5 skipped ✓
- ruff check src/ → All checks passed ✓

**See history-archive.md for Sprint 1–2 test coverage details.**

## Sprint 3 — De-Mock + Real-Data Readiness (2026-06-25)

Released v0.4.0 with API/agent de-mock and real-data readiness. Integrated with production data pipelines. All integration tests passing (292/296).

**Completed:**
- De-mock: fail-loud 503 handling, AZURE_PROJECT_ENDPOINT validation
- Real data: ST10010 geometry, survey_id integration
- Dense labels: densify + interpolation (0.30% synthetic)
- Integration tests: 69 new (292 total)
- Docs: README, real-data-runbook, task-framing

**Outcomes:** 292 passed / 2 skipped (unit), 4 passed / 5 skipped (integration), ruff clean, v0.4.0 released.

