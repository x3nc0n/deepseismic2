"""DeepSeismic Analyst agent — Azure AI Foundry Agent Service.

Bootstraps the DeepSeismic Analyst agent using the ``azure-ai-projects`` SDK.
The agent acts as an AI-native analyst assistant grounded by:

* **Azure AI Search** over indexed markdown knowledge (methods, glossary, runbooks)
* **FastAPI tool calls** for live dataset, run, QC, and result data

Supports local mock mode via ``MOCK_LLM=true`` for offline iteration without
Azure credentials or live backend services.

Usage
-----
Run a single conversation turn::

    from deepseismic.agent.agent import DeepSeismicAgent

    agent = DeepSeismicAgent()
    for chunk in agent.chat("What data is loaded for the Volve survey?"):
        print(chunk, end="", flush=True)

Run a multi-step end-to-end workflow::

    for chunk in agent.chat("Analyze the latest Volve run end-to-end."):
        print(chunk, end="", flush=True)

Environment variables
---------------------
``MOCK_LLM``
    Set to ``"true"`` to bypass Azure calls and return canned responses.
``AZURE_PROJECT_ENDPOINT``
    Azure AI Foundry project endpoint URL.
``AZURE_OPENAI_MODEL``
    Model deployment name (default: ``"gpt-4o"``).
``BACKEND_URL``
    FastAPI backend base URL (default: ``"http://localhost:8000"``).
"""

from __future__ import annotations

import json
import logging
import os
from uuid import uuid4
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

def _is_mock_mode() -> bool:
    """Check mock mode at call time (not module import time)."""
    return os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")


MOCK_MODE: bool = _is_mock_mode()
BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the DeepSeismic Analyst, an AI assistant helping petroleum geoscientists
interpret 3D seismic data from the Volve field proof-of-concept.

## Core Role
Guide analysts through a four-step interpretation workflow:
1. **Ingest**    — verify data availability and preprocessing status
2. **Interpret** — summarize model outputs, flag anomalies, cite evidence
3. **Validate**  — surface QC artifacts and highlight confidence limits
4. **Report**    — generate analyst handoff notes with structured findings

## Grounding Rules
- Ground every factual claim in tool output or indexed documentation.
- Never claim subsurface truth that is not supported by evidence.
- Distinguish clearly: observed evidence | interpretation guidance | caveats | next steps.
- When uncertainty exists, state it explicitly.

## Domain Perspectives
You answer from three expert viewpoints — declare your perspective on substantive questions:

**Ash / Geophysics**
  Focus: data quality, amplitude reliability, signal behavior, acquisition and processing caveats.
  Triggers: amplitude, waveform, QC, reliability, noise, signal, acquisition.

**Kane / Geology**
  Focus: facies meaning, depositional interpretation, structural context, lithology.
  Triggers: facies, depositional, lithology, structure, formation, stratigraphy.

**Brett / Geoengineering**
  Focus: reservoir development implications, production impact, operational uncertainty.
  Triggers: production, completion, reservoir, operations, development, well.

If a question is ambiguous, state your default perspective and offer the other two views.

## Response Structure
For substantive answers, use this structure:
1. **Observed Evidence** — facts from tools or indexed documentation
2. **Interpretation**    — what the evidence likely means in analyst language
3. **Caveats**          — what remains uncertain or requires expert review
4. **Recommended Next Step** — what the analyst should do next

## Safety Boundaries
- LLMs assist; they do not replace deterministic seismic interpretation.
- Do not assert a fault, reservoir, or geological feature exists without tool confirmation.
- Recommend expert review before any operational or development decision.
- When tool data is missing, say so clearly — do not speculate from partial metadata.
"""

# Discipline-specific additions appended when a persona is active
PERSONA_SUPPLEMENTS: dict[str, str] = {
    "geophysics": (
        "\n## Active Perspective: Ash / Geophysics\n"
        "Prioritize signal quality evidence, QC artifact review, and amplitude "
        "reliability assessment. Surface acquisition or processing caveats before "
        "drawing geological conclusions."
    ),
    "geology": (
        "\n## Active Perspective: Kane / Geology\n"
        "Prioritize facies classification, depositional interpretation, and structural "
        "context. Reference indexed methodology and model cards. Distinguish model "
        "labels from confirmed subsurface interpretation."
    ),
    "geoengineering": (
        "\n## Active Perspective: Brett / Geoengineering\n"
        "Prioritize reservoir-development and production-impact framing. Translate "
        "seismic results into engineering-relevant risk language. Specify what "
        "additional well, petrophysical, or reservoir evidence is needed before action."
    ),
}

# ---------------------------------------------------------------------------
# Mock responses (offline / local dev)
# ---------------------------------------------------------------------------

MOCK_RESPONSES: dict[str, str] = {
    "default": (
        "**[MOCK MODE — DeepSeismic Analyst]**\n\n"
        "**Observed Evidence (Ash / Geophysics perspective):**\n"
        "- Dataset `volve-survey-a` is loaded; Zarr derivative and SEG-Y source intact.\n"
        "- Preprocessing run `run-volve-preproc-01` completed without errors.\n"
        "- Inference run `run-volve-unet-01` completed; 12 QC slices generated.\n"
        "- Amplitude anomaly detected: IL 1050–1120, XL 980–1040, ~3 510 m TVDSS.\n\n"
        "**Interpretation (Kane / Geology cross-check):**\n"
        "- Anomaly depth correlates with the Hugin Formation top in offset wells "
        "15/9-F-1 B and 15/9-F-4 (~3 512 m TVDSS).\n"
        "- Facies probability suggests a candidate sandstone body; structural dip is "
        "consistent with a westward-dipping drape over the basement high.\n\n"
        "**Caveats:**\n"
        "- This is a mock response — no live data was queried.\n"
        "- The UNet baseline requires analyst sign-off before subsurface interpretation.\n"
        "- Fluid contact inference is speculative without well log integration.\n\n"
        "**Recommended Next Step:**\n"
        "Run `/status` to confirm live run state, then ask me to generate an "
        "analyst handoff note once you have reviewed the QC slices."
    ),
    "status": (
        "**[MOCK MODE — Run Status]**\n\n"
        "| Component | ID | Status | Updated |\n"
        "|---|---|---|---|\n"
        "| Dataset | `volve-survey-a` | ✅ loaded | 2026-06-09 |\n"
        "| Preprocessing | `run-volve-preproc-01` | ✅ completed | 2026-06-09 |\n"
        "| Inference | `run-volve-unet-01` | ✅ completed | 2026-06-09 |\n"
        "| QC Artifacts | `res-volve-unet-01` | ✅ 12 slices | 2026-06-09 |\n\n"
        "Everything is nominal. The result is ready for analyst review."
    ),
    "wells": (
        "**[MOCK MODE — Well Data]**\n\n"
        "Volve field wells in scope:\n\n"
        "| Well | Type | TD (m TVDSS) | Hugin Fm Top |\n"
        "|---|---|---|---|\n"
        "| 15/9-F-1 B | Producer | 3 850 | 3 512 m TVDSS |\n"
        "| 15/9-F-4 | Producer | 3 831 | 3 498 m TVDSS |\n"
        "| 15/9-F-11 | Injector | 3 740 | 3 471 m TVDSS |\n"
        "| 15/9-F-15 D | Producer | 3 892 | 3 535 m TVDSS |\n\n"
        "Well 15/9-F-1 B provides the primary formation-top control for "
        "the current interpretation window."
    ),
    "interpret": (
        "**[MOCK MODE — End-to-End Workflow Analysis]**\n\n"
        "**Step 1 — Data inventory:** `volve-survey-a` loaded; Zarr + SEG-Y present.\n"
        "**Step 2 — QC review:** preprocessing and inference both completed; "
        "12 QC slices available.\n"
        "**Step 3 — Result summary:** UNet detected a candidate fault corridor "
        "(IL 1050–1120) and amplitude anomaly at Hugin Fm level (~3 510 ms TWT).\n\n"
        "**Step 4 — Analyst handoff note:**\n\n"
        "> *Volve Subset Analysis — 2026-06-09*\n"
        "> Inference complete. Candidate fault and amplitude anomaly identified in "
        "> the southeastern quadrant. Hugin Fm correlation ties well 15/9-F-1 B at "
        "> 3 512 m TVDSS. Recommend geologist review of QC slices before sign-off. "
        "> No production decisions should be made on seismic output alone.\n\n"
        "**Caveats:** Mock data only. Expert review required before any action.\n\n"
        "**Recommended Next Step:** Load live data and re-run with `/interpret`."
    ),
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Working memory for a single agent conversation thread."""

    thread_id: str | None = None
    dataset_id: str | None = None
    run_id: str | None = None
    result_id: str | None = None
    persona: str | None = None  # geophysics | geology | geoengineering
    step_history: list[str] = field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool registry helpers
# ---------------------------------------------------------------------------

def _load_tool_definitions() -> list[dict[str, Any]]:
    """Collect JSON schema tool definitions from all tool modules."""
    from deepseismic.agent.tools.geological_tools import GEOLOGICAL_TOOL_DEFINITIONS
    from deepseismic.agent.tools.reporting_tools import REPORTING_TOOL_DEFINITIONS
    from deepseismic.agent.tools.seismic_tools import SEISMIC_TOOL_DEFINITIONS

    return SEISMIC_TOOL_DEFINITIONS + GEOLOGICAL_TOOL_DEFINITIONS + REPORTING_TOOL_DEFINITIONS


def _dispatch_tool_call(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Route a tool call to its handler and return the result dict."""
    from deepseismic.agent.tools.geological_tools import GEOLOGICAL_TOOL_HANDLERS
    from deepseismic.agent.tools.reporting_tools import REPORTING_TOOL_HANDLERS
    from deepseismic.agent.tools.seismic_tools import SEISMIC_TOOL_HANDLERS

    all_handlers: dict[str, Any] = {
        **SEISMIC_TOOL_HANDLERS,
        **GEOLOGICAL_TOOL_HANDLERS,
        **REPORTING_TOOL_HANDLERS,
    }
    handler = all_handlers.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return handler(**arguments)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tool call failed: %s(%s)", tool_name, arguments)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Mock agent (local dev, no Azure)
# ---------------------------------------------------------------------------

class MockAgent:
    """Returns canned responses for local development without Azure credentials."""

    def chat(
        self,
        message: str,
        state: SessionState | None = None,  # noqa: ARG002
    ) -> Generator[str, None, None]:
        """Yield a mock response based on keyword matching, word by word."""
        lower = message.lower()
        if any(k in lower for k in ("status", "run", "preproc", "inference", "job")):
            key = "status"
        elif any(k in lower for k in ("well", "formation", "tops", "borehole", "15/9")):
            key = "wells"
        elif any(k in lower for k in ("analyze", "end-to-end", "workflow", "full analysis")):
            key = "interpret"
        else:
            key = "default"

        response = MOCK_RESPONSES[key]
        # Emit word-by-word to simulate streaming
        for word in response.split(" "):
            yield word + " "


# ---------------------------------------------------------------------------
# Live Foundry agent
# ---------------------------------------------------------------------------

class FoundryAgent:
    """Azure OpenAI chat client backed by function calling with local tool dispatch."""

    def __init__(self, persona: str | None = None) -> None:
        from azure.identity import DefaultAzureCredential
        from openai import AzureOpenAI

        endpoint = os.environ["AZURE_PROJECT_ENDPOINT"]
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")

        self._openai_client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=token.token,
            api_version="2024-12-01-preview",
        )
        self._credential = credential
        self._persona = persona
        self._model = os.environ.get("AZURE_OPENAI_MODEL", "chat")
        self._tools = self._build_tools()
        self._threads: dict[str, list[dict[str, Any]]] = {}

    def _build_instructions(self) -> str:
        instructions = SYSTEM_PROMPT
        if self._persona and self._persona in PERSONA_SUPPLEMENTS:
            instructions += PERSONA_SUPPLEMENTS[self._persona]
        return instructions

    def _build_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI chat-completions format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "parameters": tool_def["parameters"],
                },
            }
            for tool_def in _load_tool_definitions()
        ]

    def _create_history(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self._build_instructions()}]

    def _get_history(self, thread_id: str) -> list[dict[str, Any]]:
        return self._threads.setdefault(thread_id, self._create_history())

    @staticmethod
    def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
        return {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments or "{}",
            },
        }

    @staticmethod
    def _yield_text_chunks(text: str) -> Generator[str, None, None]:
        for line in text.splitlines(keepends=True):
            yield line

    def create_thread(self) -> str:
        """Create a new in-memory conversation thread and return its ID."""
        thread_id = uuid4().hex
        self._threads[thread_id] = self._create_history()
        return thread_id

    def chat(
        self,
        message: str,
        thread_id: str | None = None,
        state: SessionState | None = None,
    ) -> Generator[str, None, None]:
        """Send a user message, execute tool calls locally, and stream text chunks."""
        if thread_id is None:
            thread_id = state.thread_id if state and state.thread_id else self.create_thread()

        if state is not None and state.thread_id is None:
            state.thread_id = thread_id

        history = self._get_history(thread_id)
        history.append({"role": "user", "content": message})

        for _ in range(16):
            response = self._openai_client.chat.completions.create(
                model=self._model,
                messages=history,
                tools=self._tools,
            )
            choice = response.choices[0]
            assistant_message = choice.message
            assistant_content = assistant_message.content or ""
            tool_calls = assistant_message.tool_calls or []

            serialized_tool_calls = [
                self._serialize_tool_call(tool_call)
                for tool_call in tool_calls
            ]
            history.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    **({"tool_calls": serialized_tool_calls} if serialized_tool_calls else {}),
                }
            )

            if assistant_content:
                yield from self._yield_text_chunks(assistant_content)

            if not tool_calls:
                return

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise TypeError("Tool arguments must decode to a JSON object")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Invalid tool arguments for %s", tool_name)
                    yield f"\n> 🔧 `{tool_name}(<invalid arguments>)`\n"
                    args = {}
                    result = {"error": f"Invalid tool arguments: {exc}"}
                else:
                    args_display = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    yield f"\n> 🔧 `{tool_name}({args_display})`\n"
                    result = _dispatch_tool_call(tool_name, args)

                if state is not None:
                    state.tool_call_log.append(
                        {"tool": tool_name, "args": args, "result": result}
                    )

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        logger.error("Model exceeded tool-call limit for thread %s", thread_id)
        yield "\n⚠️ Assistant stopped after too many tool calls.\n"


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------

class DeepSeismicAgent:
    """Public entry point for the DeepSeismic Analyst.

    Automatically selects mock or live mode based on the ``MOCK_LLM`` environment
    variable. Both modes expose the same ``chat()`` streaming interface.

    Attributes
    ----------
    persona:
        Active domain perspective: ``"geophysics"``, ``"geology"``, or
        ``"geoengineering"``. ``None`` means the agent selects based on context.
    state:
        Current session working memory (thread ID, dataset, run, result IDs).

    Example
    -------
    ::

        from deepseismic.agent.agent import DeepSeismicAgent

        agent = DeepSeismicAgent()
        for chunk in agent.chat("Is the Volve preprocessing run complete?"):
            print(chunk, end="", flush=True)
    """

    def __init__(self, persona: str | None = None) -> None:
        self.persona = persona
        self.state = SessionState(persona=persona)

        if _is_mock_mode():
            logger.info("DeepSeismicAgent: starting in MOCK mode (MOCK_LLM=true)")
            self._impl: MockAgent | FoundryAgent = MockAgent()
        else:
            logger.info("DeepSeismicAgent: connecting to Azure AI Foundry")
            self._impl = FoundryAgent(persona=persona)
            self.state.thread_id = self._impl.create_thread()

    @property
    def is_mock(self) -> bool:
        """True when the agent is running in local mock mode."""
        return _is_mock_mode()

    def chat(self, message: str) -> Generator[str, None, None]:
        """Send a message and yield response text as streaming chunks.

        Tool calls are surfaced as abbreviated inline markers (``> 🔧 tool_name(...)``).
        Callers can accumulate chunks for a full response or render them token-by-token.

        Args:
            message: Natural language message from the analyst.

        Yields:
            Text chunks of the assistant response.
        """
        if isinstance(self._impl, MockAgent):
            yield from self._impl.chat(message, self.state)
        else:
            yield from self._impl.chat(message, self.state.thread_id, self.state)

    def set_persona(self, persona: str) -> None:
        """Switch the active domain perspective.

        Args:
            persona: One of ``"geophysics"``, ``"geology"``, ``"geoengineering"``.

        Raises:
            ValueError: If the persona name is not recognised.
        """
        valid = {"geophysics", "geology", "geoengineering"}
        if persona not in valid:
            raise ValueError(f"persona must be one of {valid}; got {persona!r}")
        self.persona = persona
        self.state.persona = persona

    def get_state_summary(self) -> dict[str, Any]:
        """Return a compact snapshot of current session state suitable for UI display."""
        return {
            "thread_id": self.state.thread_id,
            "dataset_id": self.state.dataset_id,
            "run_id": self.state.run_id,
            "result_id": self.state.result_id,
            "persona": self.state.persona,
            "steps_completed": len(self.state.step_history),
            "tool_calls": len(self.state.tool_call_log),
            "mock_mode": MOCK_MODE,
        }
