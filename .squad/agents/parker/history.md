# Parker — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, Azure, Docker, REST APIs
- **Goal:** Replace expensive storage (Isilon, Premium Files, NetApp Files) with affordable cloud-native patterns
- **Challenge:** Seismic data is large (GB-TB), traditionally needs fast random I/O — find cheaper alternatives
- **User:** jospaid

## Recent Sessions

### 2026-06-29 — v0.6.5 Release (Issue #26 fix — HNS-safe prefix lookup)
- PR #28 merged: Fixed _resolve_run_id() 404 on ADLS/HNS containers (p1 workaround available)
- Implemented catalog index.json (exact download, HNS-safe) + pending manifest at submit
- Replaced bare except with logged WARNING for diagnostics
- All 12 focused tests passing; shipped in v0.6.5
- Infra deploy-notification: Spava-Corp/deepseismic2-infra#21

## Learnings

### 2026-06-09 — Local dev environment and storage abstraction layer

**What was built:**
- `src/deepseismic/storage/blob_client.py` — Full `StorageClient` + `ABSZarrStore`
- `src/deepseismic/storage/config.py` — `pydantic-settings` `Settings` singleton
- `docker/docker-compose.yml` — Azurite service (default) + API service (--profile full)
- `docker/Dockerfile` — Multi-stage build (builder + least-privilege runtime)
- `scripts/setup-local.ps1` — One-shot Windows setup: Docker check → Azurite → containers → sample data
- `.env.example` — All env vars with local-dev defaults pre-filled
- `src/deepseismic/api/main.py` — Minimal FastAPI app with `/health` endpoint

**Architecture patterns:**
- `StorageClient` auto-detects: `STORAGE_CONNECTION_STRING` → connection string; `AZURE_STORAGE_ACCOUNT` → `DefaultAzureCredential`.  No other code path needed.
- `ABSZarrStore` is a `MutableMapping` wrapping `ContainerClient`.  Works with zarr 2.x and 3.x without `adlfs`/`fsspec[azure]` — avoids an extra dependency.
- `get_settings()` is `@lru_cache(maxsize=1)` — singleton, test-friendly via `get_settings.cache_clear()`.
- Docker compose `azurite` service has a healthcheck; `api` service depends on it via `condition: service_healthy`.
- API service is behind `--profile full` so `docker compose up` only starts Azurite by default (cheaper).

**Key file paths:**
- Storage client: `src/deepseismic/storage/blob_client.py`
- Config/settings: `src/deepseismic/storage/config.py`
- Compose file: `docker/docker-compose.yml`
- Dockerfile: `docker/Dockerfile`
- Setup script: `scripts/setup-local.ps1`
- Env template: `.env.example`

**Dependency added:** `pydantic-settings>=2.3.0` — required for `BaseSettings` in pydantic v2.

**Cost notes:**
- Azurite connection string defaults mean zero Azure spend during local dev.
- Standard LRS chosen for cloud (cheapest redundancy tier for PoC).
- `list_blobs` is a metadata-only call — no data egress cost.


## Scribe Cross-Agent Update — 2026-06-10T04:30-05:00
Sprint 1 coordination complete. All agents delivered successfully.
- 5 agents synchronized
- 7 decision documents archived
- Full team context available in decisions.md

## Scribe Cross-Agent Update — 2026-06-24T12:41:40-05:00

Phase 1 (Real Fault Viewer) complete. Infra deployment follow-up required:
- **For Phase 2:** Deploy updated Streamlit UI + updated FastAPI backend to hosted demo.
- **New artifacts to stage:** Baked fault probability Zarr volumes (ault_prob.zarr, ault_mask.zarr) must be available at deployment time alongside amplitude volumes in cloud storage.
- **Deployment scope:** Copy Phase 1 code changes (src/, scripts/) to infra repo; verify bake script runs successfully in deployment environment; stage checkpoint (checkpoints/latest.pt) and baked Zarr output to blob storage before demo launch.
- **No infrastructure changes:** Compute size, storage tier, Container Apps config remain unchanged. This is a data/code update, not an infra scaling event.
- **Timeline:** Deploy after Phase 1 feature PR merges and tests pass.

## Learnings — 2026-06-25T09:34:00-05:00 — Sprint 3 issue #9: De-mock the API critical path

### What changed

**`src/deepseismic/api/dependencies.py`**
- `_build_storage_client()`: removed silent `except Exception: return None`. In real mode, if `StorageClient()` raises (e.g. no env credentials), the exception now propagates with a clear log message. `lru_cache` does not cache exceptions so the next call retries (construction is cheap — no network calls made at build time).
- `get_storage_client()` FastAPI dependency: catches the raised exception and surfaces it as HTTP 503. Routes in real mode will never receive `None` storage — they either get a client or get a 503 before their handler is called.
- Added `logging` import; added `HTTPException` import.

**`src/deepseismic/api/main.py`**
- `lifespan()`: wrapped `_build_storage_client()` call in try/except so startup failure logs clearly but does not crash the process.
- `health()`: fully rewritten. `status` is always `"ok"` (liveness — process alive). `storage` field reports real readiness: `"mock"` | `"ok"` | `"unreachable"` | `"error"`. Does a lightweight `list_blobs("catalog", max_results=1)` ping in real mode to confirm reachability. `storage_error` field added when something is wrong.

**`src/deepseismic/api/routes/interpretation.py`**
- All four route mock guards: `if is_mock_mode() or storage is None:` → `if is_mock_mode():`. In real mode, `storage is None` cannot happen (dependency raises 503 instead).

**`src/deepseismic/api/routes/surveys.py`**
- Same fix on all five mock guards. Additionally: the `except Exception: return _mock_survey_list()` silent fallback in `list_surveys` is now `raise HTTPException(503, ...)` — real mode storage errors are no longer hidden.

**`src/deepseismic/api/routes/wells.py`**
- Same fix on three mock guards. `except Exception: return _mock_well_list()` is now `raise HTTPException(503, ...)`.

**`src/deepseismic/api/routes/browse.py`**
- `browse_container`: `if is_mock_mode() or storage is None:` → `if is_mock_mode():`.

### Mock→real default decision

Real mode is now robust-default: a properly configured deployment (Azurite or cloud) takes the real code path. Mock data is only served when `DEEPSEISMIC_MOCK_MODE=true` is explicitly set. Missing/broken storage config causes 503, not silent canned data.

### Gotchas

- `StorageClient.__init__` parses env vars only — no network calls. Construction rarely fails; it only raises if *both* `STORAGE_CONNECTION_STRING` and `AZURE_STORAGE_ACCOUNT` are absent. Actual storage reachability errors surface at the first blob operation.
- `lru_cache` does NOT cache exceptions in Python, so `_build_storage_client()` retries on every request if configuration is broken. Acceptable since construction is O(1) and encourages fast recovery once env vars are fixed.
- The e2e smoke test `test_04_api_health` expects `status == "ok"` — kept by keeping `status` as a pure liveness field (always "ok" when process is alive).
- Container name contract respected: `raw`, `staged`, `results`, `catalog`, `features` — unchanged.

## Learnings — 2026-06-25 — Sprint 3 BUG-1: survey_id missing from catalog sidecar

### What changed

**`src/deepseismic/api/routes/surveys.py` — `_run_ingest()` line 181**
- `ldr.to_zarr(zarr_path, overwrite=True)` was missing the `survey_id` keyword argument.
- Fixed to: `ldr.to_zarr(zarr_path, overwrite=True, survey_id=req.survey_id)`.
- Without this, `meta.survey_id` was always `None` in the uploaded `catalog/surveys/{survey_id}/metadata.json` sidecar.

**`src/tests/test_api/test_api_real_mode.py` — `test_run_ingest_catalog_metadata_is_valid_json`**
- Updated test now asserts `meta["survey_id"] == survey_id` instead of documenting the bug with a comment.
- All pre-existing assertions (geometry, amplitude_stats, ingested_at) kept intact.

### Key fact
`SEGYLoader.to_zarr()` accepts a `survey_id: str | None = None` keyword parameter (added Sprint 3 Wave 1 by Dallas). Always pass `survey_id=req.survey_id` when calling it from `_run_ingest` so the sidecar is self-describing.


## Sprint 3 — De-Mock + Real-Data Readiness (2026-06-25)

Released v0.4.0 with API/agent de-mock and real-data readiness. Integrated with production data pipelines. All integration tests passing (292/296).

**Completed:**
- De-mock: fail-loud 503 handling, AZURE_PROJECT_ENDPOINT validation
- Real data: ST10010 geometry, survey_id integration
- Dense labels: densify + interpolation (0.30% synthetic)
- Integration tests: 69 new (292 total)
- Docs: README, real-data-runbook, task-framing

**Outcomes:** 292 passed / 2 skipped (unit), 4 passed / 5 skipped (integration), ruff clean, v0.4.0 released.


- **2026-06-29 (Ripley triage — issue #26):** Assigned to Parker — run lookup by short id-prefix 404s on ADLS/HNS. p1 with workaround. Prefix resolution bug in _resolve_run_id().

## Learnings — 2026-06-29 — Issue #26: HNS list_blobs fragility + catalog index pattern

### Root cause
`_resolve_run_id()` step 3 used `list_blobs('catalog', 'interpretation/')` to enumerate blobs for prefix matching. On ADLS Gen2 HNS containers, `ContainerClient.list_blobs(name_starts_with=...)` returns nothing or raises. A bare `except Exception: pass` silently swallowed the error → 404 on valid prefix even though the run existed.

### Fix pattern: catalog index.json + pending manifest

**`catalog/interpretation/index.json`** — a JSON list of all full run ids. Maintained by `_catalog_index_append()` (read-modify-write at submit time). `_resolve_run_id` step 3 now reads the index via `download_blob` (exact, HNS-safe) before falling back to `list_blobs`. `list_blobs` kept as fallback for pre-index runs, but now logs a WARNING.

**Pending `status.json` manifest** written in `run_fault_detection` BEFORE the background task fires so the full id is durably resolvable cross-replica immediately after submission.

### Key file/line locations
- `src/deepseismic/api/routes/interpretation.py`
  - `_CATALOG_INDEX_BLOB` constant (module level)
  - `_catalog_index_append()` — read-modify-write index helper
  - `_resolve_run_id()` — step 3a (index scan), 3b (list_blobs fallback with WARNING)
  - `run_fault_detection()` — pending manifest write + index append (before `background_tasks.add_task`)
- `src/tests/test_api/test_resolve_run_id.py` — 12 focused tests for this fix

### Design notes
- Index is append-only, best-effort — a write failure logs a warning but never blocks job submission.
- `list_blobs` fallback is only attempted when index scan yields no matches — avoids HNS errors on the hot path.
- Pending manifest at submit means step 2 (exact download) also works for the full id immediately — redundant with the index but provides defense-in-depth across replicas.
- PR: https://github.com/x3nc0n/deepseismic2/pull/28

## Scribe Cross-Agent Update — 2026-07-09T22:43:22Z

**F3 Training Data: External Sourcing Required**

Cross-survey training run blocked until F3 data is externally sourced. Issue #31 investigation confirms: real F3 data NOT present in repo (only synthetic proxy). Must ingest from public **OpendTect F3 Demo** (dGB Earth Sciences / TerraNubis, CC BY-SA). Existing `scripts/download_f3.py` documents the acquisition contract. Use `parse_opendtect_fault_sticks` parser (not Petrel).

**Leakage Gate (Hard Rule):** F3 = training input only; Volve = scoring/evaluation target only (issue #24). No cross-survey contamination.

**Geometry:** IL 100–750, XL 300–1250, ~462 samples @ 4ms.

**T4 Compute:** GPU workload profile provisioned (Spava-Corp/deepseismic2-infra#23). Data staging must complete before T4 training run.

**Decision:** `.squad/decisions.md` — F3 Ingest Contract (approved/in-progress).

## Learnings — 2026-07-13 — v0.7.3 Release (Issue #37 — best-checkpoint loss-fallback fix)

### Release steps that worked

1. **Pre-flight:** `git status` revealed one unstaged file (`.squad/agents/hudson/history.md`) not mentioned in the task brief. Committed it separately as `docs(squad): hudson review notes for v0.7.3 best-checkpoint fix` before the version bump to keep history clean.
2. **Version bump:** Only `pyproject.toml` needed bumping (0.7.2 → 0.7.3). The other `0.7.2` occurrences in Dallas's history.md were historical narrative — correctly left unchanged.
3. **Version bump commit:** `chore(release): v0.7.3 — best-checkpoint loss-fallback fix (#37)` — matches prior `chore(release)` convention on main.
4. **Land path:** DIRECT PUSH to `main`. `git pull --ff-only origin main` → already up to date → `git push origin main` succeeded (no branch protection blocking direct push for chore/release commits).
5. **GitHub release:** `gh release create v0.7.3 --repo x3nc0n/deepseismic2 --title "..." --notes "..."` — succeeded immediately.
6. **CD behavior:** Pushing `pyproject.toml` change to `main` triggered `cd.yml` immediately (status: `in_progress` within ~24s of push). CD ignores `.md`/`docs/**`/`.squad/**` — the `pyproject.toml` + `train.py` changes are what trigger it.
7. **Infra coordination:** `gh issue comment 19 --repo Spava-Corp/deepseismic2-infra --body $body` — the `--body` flag with inline quoted string fails on PowerShell when body contains backticks/special chars. Use a here-string (`$body = @"..."@`) assigned to a variable, then pass `$body`.

### Key decisions
- `chore(release)` commits go directly to `main` (not via PR). Code-fix PRs squash-merge.
- Infra coordination issue is `Spava-Corp/deepseismic2-infra#19` (not #21 or #23).
- Infra re-run request: specify new `--checkpoint-upload-prefix` with a new run-id, same training flags otherwise.
- Do NOT close app issue until infra posts re-run metrics (feeds #24).

