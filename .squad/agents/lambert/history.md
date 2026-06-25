# Lambert — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, Azure AI, M365 Copilot, GitHub Copilot, Copilot Studio, Microsoft Foundry
- **Goal:** LLM-enable seismic workflows — make interpretation accessible, automate reporting, build NL interfaces
- **Opportunity areas:** Geological report generation, well log interpretation assist, seismic attribute explanation, interactive Q&A over survey data
- **User:** jospaid

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

