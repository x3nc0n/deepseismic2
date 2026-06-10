# Lambert — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, Azure AI, M365 Copilot, GitHub Copilot, Copilot Studio, Microsoft Foundry
- **Goal:** LLM-enable seismic workflows — make interpretation accessible, automate reporting, build NL interfaces
- **Opportunity areas:** Geological report generation, well log interpretation assist, seismic attribute explanation, interactive Q&A over survey data
- **User:** jospaid

## Learnings

- **2026-06-10:** Foundry-first decision locked. SharePoint removed. Azure AI Search for grounding.

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
