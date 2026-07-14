# Parker — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, Azure, Docker, REST APIs
- **Goal:** Replace expensive storage (Isilon, Premium Files, NetApp Files) with affordable cloud-native patterns
- **Challenge:** Seismic data is large (GB-TB), traditionally needs fast random I/O — find cheaper alternatives
- **User:** jospaid

## Recent Sessions

### 2026-07-14 — v0.7.3 + v0.8.0 + v0.8.1 Triple Release (ML Fix, UI Redesign, Emergency Patch)

**v0.7.3 (ML Fix):**
- Bumped pyproject.toml 0.7.2 → 0.7.3; committed version bump to main
- Tagged GitHub release; CD built + pushed ghcr.io images (latest + sha)
- Notified infra (deepseismic2-infra#19) of v0.7.3 availability, requested warm T4 rebuild + 50-epoch F3 re-run
- Issue #37 left OPEN pending re-run results

**v0.8.0 (UI Redesign):**
- Merged PR #40 (Lambert's Impeccable UI redesign: Barlow typography, amber/stone/teal palette, improved copy)
- CD validated (391 tests pass, ruff clean)

**v0.8.1 (Emergency Patch — Gradio 6 Container Boot Crisis):**
- Post-release smoke testing: deployed UI container would not boot
- Root cause: pyproject.toml had gradio>=4.40.0 (no ceiling); docker/Dockerfile.gradio had explicit pip install gradio that won version race → resolved gradio 6.17.3 which removed gr.Chatbot(type=) and relocated Blocks(theme=, css=)
- Decision: Pin <6, do NOT migrate gradio 6 now (too large for emergency patch)
- Changed pyproject.toml [ui] to gradio>=4.44.0,<6; removed explicit gradio install from Dockerfile
- Verified: gradio 5.50.0 resolves cleanly; UI imports, 391 tests pass, ruff clean
- Created .squad/skills/dockerfile-dep-pinning/SKILL.md documenting the Dockerfile unpinned-install foot-gun
- Gradio 6 migration opened as separate feature-branch task

**CD Status:** All three releases validated (ghcr.io images built + pushed, 391 tests, ruff clean)

### 2026-06-29 — v0.6.5 Release (Issue #26 fix — HNS-safe prefix lookup)
- PR #28 merged: Fixed _resolve_run_id() 404 on ADLS/HNS containers (p1 workaround available)
- Implemented catalog index.json (exact download, HNS-safe) + pending manifest at submit
- Replaced bare except with logged WARNING for diagnostics
- All 12 focused tests passing; shipped in v0.6.5
- Infra deploy-notification: Spava-Corp/deepseismic2-infra#21

## Learnings

## Learnings

### Release Convention & Deployment Pattern (Consolidated)
- chore(release) commits go directly to main (not via PR). Code-fix PRs squash-merge.
- Version bump only touches pyproject.toml. Historical narrative in agent histories is NOT updated during release.
- GitHub release: gh release create {tag} --repo x3nc0n/deepseismic2 --title "..." --notes "..."
- CD behavior: Pushing pyproject.toml change to main triggers cd.yml (status in-progress within ~24s). CD ignores .md/docs/**/.squad/**.
- Infra coordination: Use here-strings (@"..."@) to avoid PowerShell escaping issues with backticks/special chars.
- Pre-flight: git status before version bump to catch unstaged files not mentioned in task brief.

### Gradio Dependency Pinning Foot-Gun (Critical Pattern)
**Problem:** pyproject.toml [ui] pinned gradio>=4.40.0 (no upper bound). docker/Dockerfile.gradio had bare pip install gradio that overrode pyproject constraint and pulled gradio 6.17.3 at build time.

**Gradio 6 breaking changes:**
- gr.Chatbot(type="messages") kwarg removed → TypeError at module import (fatal)
- gr.Blocks(theme=..., css=...) args removed → silently ignored on gradio 6 (design lost)

**Fix:** gradio>=4.44.0,<6 (4.44 is baseline for 	ype="messages"; <6 ceiling enforces API compatibility).

**Principle:** Dockerfile explicit pip install alongside a .[extra] install wins the version race when there is no lock file. Never duplicate package installs in Dockerfile; let pyproject.toml constraints govern. Documented in .squad/skills/dockerfile-dep-pinning/SKILL.md.

### HNS Prefix Resolution Pattern (Issue #26 Fix)
Root cause: list_blobs(name_starts_with=...) returns nothing or raises on ADLS Gen2 HNS containers. Silent xcept swallowed errors → 404 on valid prefixes.

Fix pattern:
1. catalog/interpretation/index.json — JSON list of all full run ids, maintained by _catalog_index_append() (read-modify-write at submit time)
2. _resolve_run_id() step 3: Read index via download_blob (exact, HNS-safe) before falling back to list_blobs (now logs WARNING)
3. Pending status.json manifest written in un_fault_detection BEFORE background task fires → full id is durably resolvable cross-replica immediately after submission

Key files: src/deepseismic/api/routes/interpretation.py (_CATALOG_INDEX_BLOB constant, _resolve_run_id, _catalog_index_append, run_fault_detection), src/tests/test_api/test_resolve_run_id.py (12 focused tests). PR #28.

### API De-Mock Pattern (Sprint 3 — Fail-Loud Readiness)
Mock mode is now opt-in (DEEPSEISMIC_MOCK_MODE=true); real mode is robust default.

**Changes:**
- StorageClient.__init__ exceptions now propagate (no silent except) with clear logging
- get_storage_client() FastAPI dependency catches exception and surfaces as HTTP 503
- health() endpoint: status = liveness (always "ok" when process alive); storage field = real readiness ("mock" | "ok" | "unreachable" | "error")
- All route mock guards: if is_mock_mode() or storage is None: → if is_mock_mode(): (storage cannot be None in real mode)
- Silent fallbacks removed: xcept Exception: return _mock_*() → aise HTTPException(503, ...)

**Key facts:**
- StorageClient construction is O(1), never network calls → lru_cache does NOT cache exceptions, retries on next call
- Catalog index, pending manifest, survey_id in sidecar all follow this pattern: fail-loud, explicit, traceable

### F3 Ingest & Data Leakage Gate (Cross-Survey Boundary)
- F3 data is training input ONLY; Volve data is scoring/evaluation target only (issue #24 hard rule)
- NO cross-survey contamination allowed
- F3 geometry: IL 100–750, XL 300–1250, ~462 samples @ 4ms
- T4 compute GPU workload profile provisioned (Spava-Corp/deepseismic2-infra#23)
- Decision: .squad/decisions.md — F3 Ingest Contract (approved/in-progress)
