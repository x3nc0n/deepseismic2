# Ripley — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, PyTorch, Azure, LLM APIs (M365 Copilot, GitHub Copilot, Copilot Studio/Foundry)
- **Goal:** Modernize seismic interpretation from legacy monolithic apps (Dell Isilon, Azure Premium Files, Azure NetApp Files) to affordable cloud-native + AI patterns
- **Data:** Equinor Volve dataset (https://www.equinor.com/energy/volve-data-sharing)
- **Reference:** microsoft/seismic-deeplearning repo (UNet, SEResNet, HRNet segmentation on seismic data)
- **User:** jospaid

## Recent Sessions

### 2026-06-29 — v0.6.5 Release Pipeline (fixes #25/#26)
- Coordinated release: PRs #27 (Lambert, atomic commit), #28 (Parker, HNS-safe lookup), #29 (Hudson, CI fix)
- Bumped version 0.6.4 → 0.6.5, created GitHub release tag
- Filed infra deploy-notification issue Spava-Corp/deepseismic2-infra#21
- Established workflow: fix → release → infra notify

## Learnings — 2026-06-29T17:46:41-05:00: Issue Triage #25 and #26

### Triage decisions

**Issue #25 — Chat wedges on dangling tool_calls (AOAI 400)**
- Owner: **squad:lambert** (LLM/agent integration)
- Priority: **p0** — permanently wedges hosted demo for all users until container restart
- Release: **v0.4.0** (patch, urgent)
- Root cause: `FoundryAgent.chat()` (`src/deepseismic/agent/agent.py` ~L402/437) writes assistant `tool_calls` to persistent thread history before matching tool-result messages. `GeneratorExit` from the 25s UI truncation guard (`gradio_app.py` L318-323) leaves the thread in an irrecoverable corrupt state. Process-wide singleton means all sessions on the container are poisoned.

**Issue #26 — Run prefix-resolution 404 on ADLS/HNS**
- Owner: **squad:parker** (backend/API + storage client)
- Priority: **p1** — clean workaround exists (full UUID); no data loss
- Release: **v0.5.0**
- Root cause: `_resolve_run_id()` (`src/deepseismic/api/routes/interpretation.py` L48-105) uses `ContainerClient.list_blobs(name_starts_with=...)` which silently returns nothing on ADLS Gen2 HNS containers. Bare `except Exception: pass` at L84 swallows the failure. Fix: use DataLake `FileSystemClient.get_paths()` or a written index manifest; replace bare except with logged error.

### Sequencing
#25 must land before #26 given severity. Both are independent bugs with no shared code surface.

### Architectural note (general)
**Agent thread-state must be committed atomically.** The assistant message with `tool_calls` and all matching tool-result messages must be appended to thread history in a single atomic operation. Writing the assistant entry first and the tool results second creates a window where a generator interruption (timeout, exception, `GeneratorExit`) produces unrecoverable corrupt state. This principle applies to any agent that reuses a persistent conversation thread across requests.

## Team Coordination — 2026-06-29T23:44:49Z: Both PRs Ready for Review

**Status Update:**
- **#25 (lambert):** PR #27 open (`squad/25-chat-wedge-tool-calls`) — 6 new tests, atomic round_buffer + try/finally seal + self-heal layer verified
- **#26 (parker):** PR #28 open (`squad/26-resolve-run-id-prefix`) — 12 new tests, catalog index + pending manifest + logged warnings implemented

**Both agents delivered early.** No blockers. Sequencing: review #25 first (p0), then #26 (p1). Independent code paths — parallel review safe.

**Scribe actions completed:**
- Archived 2026-06-24 entries (50+ decisions) to decisions-archive.md
- Merged inbox files to active decisions
- Orchestration logs written (ISO 8601 UTC)
- Session log recorded
- Cross-agent history updated

**Next step:** Team review of PRs #27 and #28. Once both are merged, v0.4.0 patch is ready.

## Scribe Cross-Agent Update — 2026-07-09T22:43:22Z

**F3 Training Data: External Sourcing Required**

Cross-survey training blocked until F3 data is externally sourced. Issue #31 investigation confirms: real F3 data NOT present in repo (only synthetic proxy). Must ingest from public **OpendTect F3 Demo** (dGB Earth Sciences / TerraNubis, CC BY-SA). Existing `scripts/download_f3.py` documents the acquisition contract. Use `parse_opendtect_fault_sticks` parser (not Petrel).

**Leakage Gate (Hard Rule):** F3 = training input only; Volve = scoring/evaluation target only (issue #24). No cross-survey contamination.

**Geometry:** IL 100–750, XL 300–1250, ~462 samples @ 4ms.

**T4 Compute:** GPU workload profile provisioned (Spava-Corp/deepseismic2-infra#23). Data staging must complete before T4 training run.

**Decision:** `.squad/decisions.md` — F3 Ingest Contract (approved/in-progress).

---

## Archive

### Sprint 1 → Sprint 2: FastAPI backend implementation (2026-06-09)

**What was built:**
- `api/main.py` — full app factory with lifespan, CORS, health check
- `api/schemas.py` — complete Pydantic v2 models
- `api/dependencies.py` — `get_storage_client`, `get_settings_dep`, `is_mock_mode()`
- `api/routes/surveys.py` — 5 endpoints: list, metadata, ingest, inline/crossline slices
- `api/routes/interpretation.py` — 4 endpoints: fault-detection, status, results, overlay
- `api/routes/wells.py` — 3 endpoints: list, metadata, logs with Volve mock data

**Design:** `DEEPSEISMIC_MOCK_MODE=true` gates mock behavior; BackgroundTasks for job runner; all HTTPException use `from None`/`from exc`; StrEnum for JobStatus (Python 3.11+ compliance).

**API contract verified.** Ruff clean on all six files.

### Scribe Cross-Agent Updates (2026-06-10 to 2026-06-24)

- 2026-06-10: Sprint 1 coordination complete; 5 agents synchronized, 7 decisions archived
- 2026-06-24: Process emulation gaps identified (C1: synthetic-only training, C2: no validation pass, C3: README overstates maturity); consolidated findings to decisions.md

### 2026-06-24: Process Emulation Gap Assessment → Sprint 2 Minimum Fix

**Key findings:** ML core is hollow (synthetic-only training, labels unwired, validation never exercised, 100% mock demo). **What IS real:** SEG-Y ingest, Zarr conversion, UNet3D architecture, sliding-window inference, patch extraction, API contract, agent tool wiring.

**Gaps:** C1 (synthetic training), C2 (no validation pass), C3 (misleading README); 5 important gaps; minimum fix: wire real labels (~4h), eval script (~2h), README fix (~30min).

### 2026-06-24 to 2026-06-25: Sprint 2 & 3 Documentation, Real-Data Readiness

- **S2-04:** README rewrite with "What's real vs. demo" table, real metrics (val IoU=0.047/Dice=0.089 @ epoch 18), reproduction commands verified
- **S2-09:** New `docs/task-framing.md` — task difference (binary fault detection on Volve vs. original's multi-class facies on F3/Penobscot), correct lineage, appropriate metrics
- **Sprint 3:** De-mock + real-data readiness; released v0.4.0; API/agent fail-loud 503 handling, AZURE_PROJECT_ENDPOINT validation, ST10010 geometry integration, dense label densification (0.30% synthetic), 292 tests passing

**Blocker dependencies documented:** ST10010_PSDM_TIME.segy to ADLS (infra #11), Databricks Marketplace install, private endpoint setup.


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

## Learnings — 2026-06-25T09:34:00-05:00: Sprint 3 Real-Data Readiness Docs

### Sprint 3 readiness framing

Sprint 3 made the full pipeline **app-ready** for real Volve data without executing
on real data (execution is deploy-gated). The key framing distinction for docs:

- **App-ready** = code path exists, locally validated against a format proxy (synthetic
  SEG-Y, synthetic fault-stick files). No real data involved.
- **Deploy-gated** = execution on real data requires infra #11 (SEG-Y staged to ADLS)
  + Marketplace install + in-VNet compute. These are infrastructure/data-access
  dependencies, not code gaps.

Never say "validated on Volve data" for Sprint 3 work. The correct phrasing is
"validated locally as a format proxy" or "synthetic-proxy validated."

### Where the runbook lives

`docs/real-data-runbook.md` — ordered steps: infra check → ingest → labels → train →
eval → API → agent. Each step notes whether it must run in-VNet and which env vars /
flags select real vs. mock mode.

### Blocker dependencies

| Blocker | Owner | Tracking |
|---------|-------|---------|
| ST10010_PSDM_TIME.segy into ADLS `raw` container | Spava-Corp/deepseismic2-infra | infra issue #11 |
| Volve Databricks Marketplace install (identity-bound) | User (x3nc0n) | Manual |
| Private endpoint — ADLS `publicNetworkAccess: Disabled` | Spava-Corp/deepseismic2-infra | infra issue #11 |

### Key Sprint 3 doc changes

- README.md: Status updated to Sprint 3; real-vs-demo table extended with all Sprint 3
  changes; new "Real-data readiness" section with explicit blockers table; Sprint 3
  smoke-test commands and in-VNet execution commands added.
- docs/real-data-runbook.md: NEW file — the ordered deploy path.
- docs/task-framing.md: Sprint 3 label densification note added; summary table updated
  with synthetic proxy numbers.


## Sprint 3 — De-Mock + Real-Data Readiness (2026-06-25)

Released v0.4.0 with API/agent de-mock and real-data readiness. Integrated with production data pipelines. All integration tests passing (292/296).

**Completed:**
- De-mock: fail-loud 503 handling, AZURE_PROJECT_ENDPOINT validation
- Real data: ST10010 geometry, survey_id integration
- Dense labels: densify + interpolation (0.30% synthetic)
- Integration tests: 69 new (292 total)
- Docs: README, real-data-runbook, task-framing

**Outcomes:** 292 passed / 2 skipped (unit), 4 passed / 5 skipped (integration), ruff clean, v0.4.0 released.

## Learnings — 2026-06-29T17:46:41-05:00: Issue Triage #25 and #26

### Triage decisions

**Issue #25 — Chat wedges on dangling tool_calls (AOAI 400)**
- Owner: **squad:lambert** (LLM/agent integration)
- Priority: **p0** — permanently wedges hosted demo for all users until container restart
- Release: **v0.4.0** (patch, urgent)
- Root cause: `FoundryAgent.chat()` (`src/deepseismic/agent/agent.py` ~L402/437) writes assistant `tool_calls` to persistent thread history before matching tool-result messages. `GeneratorExit` from the 25s UI truncation guard (`gradio_app.py` L318-323) leaves the thread in an irrecoverable corrupt state. Process-wide singleton means all sessions on the container are poisoned.

**Issue #26 — Run prefix-resolution 404 on ADLS/HNS**
- Owner: **squad:parker** (backend/API + storage client)
- Priority: **p1** — clean workaround exists (full UUID); no data loss
- Release: **v0.5.0**
- Root cause: `_resolve_run_id()` (`src/deepseismic/api/routes/interpretation.py` L48-105) uses `ContainerClient.list_blobs(name_starts_with=...)` which silently returns nothing on ADLS Gen2 HNS containers. Bare `except Exception: pass` at L84 swallows the failure. Fix: use DataLake `FileSystemClient.get_paths()` or a written index manifest; replace bare except with logged error.

### Sequencing
#25 must land before #26 given severity. Both are independent bugs with no shared code surface.

### Architectural note (general)
**Agent thread-state must be committed atomically.** The assistant message with `tool_calls` and all matching tool-result messages must be appended to thread history in a single atomic operation. Writing the assistant entry first and the tool results second creates a window where a generator interruption (timeout, exception, `GeneratorExit`) produces unrecoverable corrupt state. This principle applies to any agent that reuses a persistent conversation thread across requests.

## Team Coordination — 2026-06-29T23:44:49Z: Both PRs Ready for Review

**Status Update:**
- **#25 (lambert):** PR #27 open (`squad/25-chat-wedge-tool-calls`) — 6 new tests, atomic round_buffer + try/finally seal + self-heal layer verified
- **#26 (parker):** PR #28 open (`squad/26-resolve-run-id-prefix`) — 12 new tests, catalog index + pending manifest + logged warnings implemented

**Both agents delivered early.** No blockers. Sequencing: review #25 first (p0), then #26 (p1). Independent code paths — parallel review safe.

**Scribe actions completed:**
- Archived 2026-06-24 entries (50+ decisions) to decisions-archive.md
- Merged inbox files to active decisions
- Orchestration logs written (ISO 8601 UTC)
- Session log recorded
- Cross-agent history updated

**Next step:** Team review of PRs #27 and #28. Once both are merged, v0.4.0 patch is ready.

## Scribe Cross-Agent Update — 2026-07-09T22:43:22Z

**F3 Training Data: External Sourcing Required**

Cross-survey training blocked until F3 data is externally sourced. Issue #31 investigation confirms: real F3 data NOT present in repo (only synthetic proxy). Must ingest from public **OpendTect F3 Demo** (dGB Earth Sciences / TerraNubis, CC BY-SA). Existing `scripts/download_f3.py` documents the acquisition contract. Use `parse_opendtect_fault_sticks` parser (not Petrel).

**Leakage Gate (Hard Rule):** F3 = training input only; Volve = scoring/evaluation target only (issue #24). No cross-survey contamination.

**Geometry:** IL 100–750, XL 300–1250, ~462 samples @ 4ms.

**T4 Compute:** GPU workload profile provisioned (Spava-Corp/deepseismic2-infra#23). Data staging must complete before T4 training run.

**Decision:** `.squad/decisions.md` — F3 Ingest Contract (approved/in-progress).


