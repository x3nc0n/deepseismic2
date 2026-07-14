# Lambert — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, Azure AI, M365 Copilot, GitHub Copilot, Copilot Studio, Microsoft Foundry
- **Goal:** LLM-enable seismic workflows — make interpretation accessible, automate reporting, build NL interfaces
- **Opportunity areas:** Geological report generation, well log interpretation assist, seismic attribute explanation, interactive Q&A over survey data
- **User:** jospaid

## Recent Sessions

### 2026-07-14 — v0.8.0 UI Redesign (Impeccable Theming + Geoscience Visual Identity)
- Delivered PR #40 (merged): Impeccable-guided Gradio UI overhaul
- Typography: Barlow Condensed (headings, uppercase) + Barlow (body) + Fira Code (mono) — replaced Inter defaults
- Color palette: Amber (primary, earthy/geological) + Stone (neutral, warm gray) + Teal (secondary, depth) — replaced blue/slate
- Copy refinement: "Agent Conversation" → "Analyst Chat", "Seismic Inline Viewer" → "Inline Section", improved affordance labels
- CSS surface: Custom _CUSTOM_CSS injected via gr.Blocks(css=...), stable elem_id/elem_classes for future audits
- Impeccable toolkit untracked (gitignored), skill doc kept for future design passes
- Issue #39 (storage browser reposition) intentionally deferred; tracked separately
- v0.8.0 shipped with new visual identity; decision merged to squad decisions.md

## Recent Sessions (older)

### 2026-06-29 — v0.6.5 Release (Issue #25 fix — atomic commit)
- PR #27 merged: Fixed FoundryAgent.chat() thread-state corruption (p0 blocker)
- Implemented atomic round_buffer + try/finally commit + _seal_dangling_tool_calls() self-heal
- All 6 contract tests passing; shipped in v0.6.5
- Infra deploy-notification: Spava-Corp/deepseismic2-infra#21

## Learnings

- **2026-06-25 (Sprint 3 — issue #9 — de-mock the Foundry agent):**
  - **What changed:** Hardened mock→live default across `agent.py` and all three tool modules.
  - **Key file paths:** `src/deepseismic/agent/agent.py`, `src/deepseismic/agent/tools/{seismic,geological,reporting}_tools.py`
  - **Tools — call-time mock check:** Each tool module had `MOCK_MODE: bool = os.environ.get("MOCK_LLM", ...)` captured at **import time**. Added `_is_mock() -> bool` helper (checked at call time) and replaced every `if MOCK_MODE:` guard in tool functions with `if _is_mock():`. Module-level `MOCK_MODE` kept for backward-compat imports.
  - **FoundryAgent — clear missing-credential error:** `os.environ["AZURE_PROJECT_ENDPOINT"]` raised a bare `KeyError` when missing. Changed to `os.environ.get(...)` + explicit `RuntimeError` naming the env var and offering the `MOCK_LLM=true` escape hatch.
  - **DeepSeismicAgent — fail-loud, no silent fallback:** When not in mock mode and `AZURE_PROJECT_ENDPOINT` is missing, `__init__` raises `RuntimeError` immediately. Live mode never silently degrades to mock.
  - **Mode visibility:** Mock path logs `"starting in MOCK mode (MOCK_LLM=true) — no Azure calls will be made"`. Live path logs endpoint URL and model name.
  - **`get_state_summary`:** Changed `MOCK_MODE` (module-level import-time) to `_is_mock_mode()` (call-time).
  - **Design decision:** Mock = explicit opt-in only. Misconfigured live = loud `RuntimeError`. No silent fallbacks anywhere.
  - **Tests:** 210 passed, 2 skipped, 1 pre-existing failure (`test_04_api_health` — storage credentials absent locally, unrelated to agent). Ruff clean.

- **2026-06-10:** Foundry-first decision locked. SharePoint removed. Azure AI Search for grounding.

- **2026-06-09 (overnight sprint):** Wired agent tool modules to real FastAPI endpoints. Key decisions:
  - **`_api_client.py` shared client:** Created `src/deepseismic/agent/tools/_api_client.py` with `get()`, `get_list()`, `post()` helpers. Reads base URL from `DEEPSEISMIC_API_URL` → `BACKEND_URL` → `http://localhost:8000` at call time. Retries up to 2× on `httpx.RequestError` and HTTP 503 with linear back-off. Raises `APIError` so all callers can return `{"error": ..., "available": False}` uniformly. Moved `httpx` from `ui` optional dep to core `pyproject.toml` dependencies since agent tools need it.
  - **seismic_tools.py:** `query_survey_metadata` → `GET /api/surveys` (client-side filter by name); `get_inline_section` → `GET /api/surveys/{id}/inline/{n}`; `run_fault_detection` → `POST /api/interpretation/fault-detection` (maps `model_version` to `checkpoint_blob` path); `get_interpretation_status` → `GET /api/interpretation/{run_id}/status`.
  - **geological_tools.py:** `get_well_data` → `GET /api/wells/{id}` (single) or `GET /api/wells` (list); `get_formation_tops` → `GET /api/wells/{id}` and extract `formation_tops`; `correlate_wells` → per-well `GET /api/wells/{id}` + `GET /api/wells/{id}/logs`, compute depth stats client-side; `get_regional_context` → stays as embedded knowledge-base (no live endpoint exists or needed).
  - **reporting_tools.py:** No dedicated summary/export/QC endpoints exist — all three tools compose their responses from `GET /api/interpretation/{run_id}/status` and `GET /api/interpretation/{run_id}/results`. `generate_summary` formats `InterpretationResult` into an analyst-readable summary dict; `export_interpretation` aggregates status+results into an artifact manifest; `create_qc_report` builds sections from status+results with pass/fail/pending states.
  - **Mock fallback:** All tools check `MOCK_MODE` at call time (not import time) so `MOCK_LLM=true` works even after the module has been imported. When the API is unreachable in live mode, `APIError` is caught and the tool returns `{"error": ..., "available": False}` so the agent can degrade gracefully.
  - **Tests:** 79 pass, 5 skipped (infra-dependent). All ruff checks clean.

- **2026-06-09:** Implemented full Foundry agent and three demo UIs. Key patterns:
  - **Agent structure:** `DeepSeismicAgent` façade delegates to `MockAgent` (MOCK_LLM=true) or `FoundryAgent` (live). Both expose the same `chat()` streaming generator so all UIs are mode-agnostic.
  - **Tool registry:** Three tool modules (`seismic_tools`, `geological_tools`, `reporting_tools`) each export `*_TOOL_DEFINITIONS` (JSON schema for Foundry registration) and `*_TOOL_HANDLERS` (callable dict for local dispatch). Agent collects all definitions at boot and dispatches tool calls through `_dispatch_tool_call()`.
  - **Mock responses:** Every tool function checks `MOCK_LLM` env var at call time (not import time) so mocking works even when the agent is imported in live mode. Canned Volve reference data is embedded in each tool module.
  - **Knowledge files:** Three `.md` files in `src/deepseismic/agent/knowledge/` — `volve_field_overview.md`, `interpretation_workflow.md`, `seismic_basics.md`. Ready to index into Azure AI Search with heading-level chunking.
  - **UI surfaces:** Terminal chat (`ui/chat.py`) with readline fallback for Windows, Streamlit app with two-panel layout, Gradio app with chatbot + seismic viewer. All three import `DeepSeismicAgent` directly.
  - **Synthetic seismic:** Both Streamlit and Gradio apps generate a plausible bandlimited synthetic inline section using scipy convolution + synthetic reflectors so the seismic viewer looks credible to a geologist without live data.
  - **Key file paths:** `src/deepseismic/agent/agent.py`, `src/deepseismic/agent/tools/{seismic,geological,reporting}_tools.py`, `src/deepseismic/agent/knowledge/*.md`, `src/deepseismic/ui/{chat,streamlit_app,gradio_app}.py`.
  - **pyproject.toml:** `ui` optional-dependency group added: `streamlit>=1.38.0`, `gradio>=4.40.0`. Install with `pip install -e ".[ui]"`.
  - **Session state:** `SessionState` dataclass tracks `thread_id`, `dataset_id`, `run_id`, `result_id`, `persona`, `step_history`, and `tool_call_log`. Foundry thread ID is set on `DeepSeismicAgent.__init__()`.


## Scribe Cross-Agent Update — 2026-06-10T04:30-05:00
Sprint 1 coordination complete. All agents delivered successfully.
- 5 agents synchronized
- 7 decision documents archived
- Full team context available in decisions.md

## Scribe Cross-Agent Update — 2026-06-24T12:41:40-05:00

Phase 1 (Real Fault Viewer) complete. Phase 2 planning note:
- **For Phase 2:** Wire a new "detect_faults" agent tool. Current viewer pre-bakes fault results; Phase 2 will add on-demand detection (button in UI, agent tool for Foundry). Viewer no longer fakes seismic data or fault detection — both now real/live.
- **Data source:** Real Zarr amplitude from data/volve/staged/synthetic.zarr, real UNet fault probability from data/volve/staged/fault_prob.zarr (baked once by Dallas).
- **Coordinate mapping:** Fault stick overlay now uses correct mapping (z_ms column = sample index × 4.0). All coordinate transforms verified against UTM reference data.
- **Agent-tool feasibility:** Phase 2 detect_faults tool will call POST /api/interpretation/fault-detection with survey_id + optional checkpoint_blob parameter, return probability volume → agent streams inline slices to UI on user request.

## Sprint 3 — De-Mock + Real-Data Readiness (2026-06-25)

Released v0.4.0 with API/agent de-mock and real-data readiness. Integrated with production data pipelines. All integration tests passing (292/296).

**Completed:**
- De-mock: fail-loud 503 handling, AZURE_PROJECT_ENDPOINT validation
- Real data: ST10010 geometry, survey_id integration
- Dense labels: densify + interpolation (0.30% synthetic)
- Integration tests: 69 new (292 total)
- Docs: README, real-data-runbook, task-framing

**Outcomes:** 292 passed / 2 skipped (unit), 4 passed / 5 skipped (integration), ruff clean, v0.4.0 released.


- **2026-06-29 (Ripley triage — issue #25):** Assigned to Lambert — chat wedges after truncated tool turn (AOAI 400). p0 blocker for hosted demo. Thread-state atomicity required in FoundryAgent.chat().

## Learnings — 2026-07-13 (Impeccable design pass, branch squad/ui-impeccable-design)

- **What Impeccable is:** A design-guidance toolkit for AI coding agents (https://impeccable.style). Installs via `npx impeccable install` — it placed a full skill set under `.github/skills/impeccable/` (SKILL.md, 20+ `/command` reference docs, detector scripts). CLI version **3.2.0** installed successfully on this project. Commands (`/typeset`, `/colorize`, `/layout`, `/audit`, etc.) are designed to run inside an AI harness (like Copilot CLI), not as standalone CLI tools against arbitrary files. The deterministic detector (`detect.mjs --json <file>`) targets HTML/CSS/JS source — returned 0 findings on `gradio_app.py` (it's Python, not CSS/HTML), so all principles were applied manually by reading the reference docs.
- **Levers that work on a Gradio UI:** (a) the Gradio theme object (`primary_hue`, `neutral_hue`, `font`, `font_mono`, `radius_size`, `.set()` token overrides); (b) custom CSS injected via `gr.Blocks(css=...)`; (c) `elem_id` / `elem_classes` on components for stable CSS targeting; (d) copy/labels/empty-states; (e) layout composition (rows/columns/scale/min_width). Cannot hand-author arbitrary DOM — Gradio generates its own.
- **Type/color choices:**
  - **Heading/label font:** `Barlow Condensed` (500/600/700) — industrial precision, geological survey-report feel, not on Impeccable reflex-reject list. Loaded via `@import` in the CSS block.
  - **Body font:** `Barlow` (Google Font via theme) — humanist sans, readable at tool density.
  - **Mono font:** `Fira Code` (via theme) — replaces JetBrains Mono; ligature-enabled, characterful.
  - **Primary hue:** `amber` — earthy, geological (evokes core samples and sediment cross-sections). Replaces generic `blue`.
  - **Neutral hue:** `stone` — warm gray, coheres with amber. Replaces cold `slate`.
  - **Secondary hue:** `teal` — subsurface/depth. Replaces `slate`.
  - **Rationale:** Amber+Stone+Teal is intentionally domain-specific (earthy/subsurface) and avoids the purple-blue SaaS gradient Impeccable flags as AI-slop aesthetic.
- **Key file paths:** `src/deepseismic/ui/gradio_app.py` (all changes — theme, CSS constant `_CUSTOM_CSS`, `elem_id`/`elem_classes` on components, updated copy). No separate CSS file needed; CSS string kept inline.
- **Impeccable installation note:** Added `.github/skills/impeccable/` and `.github/hooks/impeccable.json`. These are dev tooling files; the PR body notes we should keep them (helps future design passes).
- **Tests:** 391 passed, 2 skipped. Ruff clean. Branched from `origin/main` (not local `main`) to keep Dallas's unpushed commits out of the PR diff.

- **2026-06-29 (issue #25 — atomic thread-history commit, PR #27):**
  - **Root cause (Bug B):** `FoundryAgent.chat()` (`src/deepseismic/agent/agent.py` ~L402) appended the assistant `tool_calls` message to the persistent `history` list before the matching `tool` result messages were appended. Generator `yield`s inside the tool-dispatch loop meant that the UI's 25s `break` sent `GeneratorExit` mid-round, leaving a dangling `tool_calls` entry with no tool response — every subsequent AOAI call on that process got HTTP 400. Container restart was the only recovery path.
  - **Root cause (Bug A):** `gradio_app.py` 25s guard could break mid-tool-round, cutting off tool-trace output to the user.
  - **Fix (Bug B — belt-and-suspenders):**
    1. **Atomic `round_buffer`:** Stage assistant msg + all tool results in a local list; `history.extend(round_buffer)` only after the round is complete (`src/deepseismic/agent/agent.py`, `FoundryAgent.chat()`).
    2. **`try/finally` seal:** Synthesize `{"error": "interrupted"}` tool responses for any unanswered `tool_call_id`s before committing — handles `GeneratorExit` at any yield point.
    3. **`_seal_dangling_tool_calls()` on entry:** Self-heals any pre-existing corrupt thread at the start of every `chat()` call.
  - **Fix (Bug A):** Track `in_tool_round` via the `\n> 🔧` chunk prefix; defer `break` until `not in_tool_round`.
  - **Key file paths:** `src/deepseismic/agent/agent.py` (lines ~360-447), `src/deepseismic/ui/gradio_app.py` (lines ~316-335), `src/tests/test_agent_atomic_commit.py` (new, 6 tests).
  - **Important implementation note:** `_get_history()` returns the mutable list stored in `self._threads` (via `setdefault`). `history.extend(round_buffer)` mutates the list in-place — do NOT rebind the local variable.
  - **Tests:** 6 new focused tests in `test_agent_atomic_commit.py`; all 48 targeted agent/chat/thread tests pass; ruff clean; py_compile clean.
  - **Follow-up:** Migrate `FoundryAgent.chat()` to `stream=True` (chunked SSE) to eliminate the truncation risk entirely and remove the round-buffer workaround.

