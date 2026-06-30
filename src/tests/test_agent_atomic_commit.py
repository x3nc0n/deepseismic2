"""Tests for FoundryAgent.chat() atomic thread-history commit (issue #25).

Scenario under test:
- FoundryAgent receives an assistant message with tool_calls.
- The consumer closes the generator mid-round (simulating the UI 25s guard).
- BEFORE the fix: history contained a dangling assistant tool_calls with no matching
  tool response → every subsequent AOAI call returned 400.
- AFTER the fix: history must always be contract-valid:
    every assistant message that carries tool_calls has matching tool responses,
    OR the round was not committed at all.

All tests are CI-safe — no Azure credentials or live API calls.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_foundry_agent() -> Any:
    """Instantiate FoundryAgent with all Azure/OpenAI dependencies mocked.

    FoundryAgent lazily imports azure.identity and openai inside __init__, so
    we must patch at the source-module level rather than deepseismic.agent.agent.
    """
    with (
        patch.dict(os.environ, {"AZURE_PROJECT_ENDPOINT": "https://mock.endpoint/"}),
        patch("azure.identity.DefaultAzureCredential"),
        patch("azure.identity.get_bearer_token_provider"),
        patch("openai.AzureOpenAI"),
    ):
        from deepseismic.agent.agent import FoundryAgent

        agent = FoundryAgent()

    # Replace the openai client with a fresh MagicMock so tests control completions.
    agent._openai_client = MagicMock()
    return agent


def _make_tool_call_mock(
    tool_call_id: str = "tc-001",
    tool_name: str = "query_survey_metadata",
    arguments: str = '{"survey_name": "Volve"}',
) -> MagicMock:
    """Build a mock tool_call object matching the OpenAI SDK shape."""
    tc = MagicMock()
    tc.id = tool_call_id
    tc.type = "function"
    tc.function.name = tool_name
    tc.function.arguments = arguments
    return tc


def _make_completion_with_tool_calls(
    tool_calls: list[MagicMock],
    assistant_text: str = "Let me look that up.",
) -> MagicMock:
    """Build a mock chat completion that includes tool_calls."""
    message = MagicMock()
    message.content = assistant_text
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_completion_text_only(text: str = "Here is your answer.") -> MagicMock:
    """Build a mock chat completion with no tool calls (terminal turn)."""
    message = MagicMock()
    message.content = text
    message.tool_calls = []

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _assert_no_dangling_tool_calls(history: list[dict[str, Any]]) -> None:
    """Assert that every assistant tool_calls entry has matching tool responses."""
    for i, msg in enumerate(history):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc_ids = {tc["id"] for tc in msg["tool_calls"]}
            answered = {
                m["tool_call_id"]
                for m in history[i + 1 :]
                if m.get("role") == "tool" and "tool_call_id" in m
            }
            assert tc_ids == answered, (
                f"Dangling tool_calls at history[{i}]: "
                f"expected responses for {tc_ids}, found {answered}.\n"
                f"Full history: {json.dumps(history, indent=2, default=str)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestAtomicToolCallCommit
# ─────────────────────────────────────────────────────────────────────────────


class TestAtomicToolCallCommit:
    """FoundryAgent.chat() must never leave dangling tool_calls in thread history."""

    def test_close_after_text_chunk_no_dangling(self) -> None:
        """Closing the generator after the assistant text chunk must not corrupt history."""
        agent = _make_foundry_agent()
        thread_id = agent.create_thread()

        tc = _make_tool_call_mock()
        agent._openai_client.chat.completions.create.return_value = (
            _make_completion_with_tool_calls([tc])
        )

        gen = agent.chat("What surveys are available?", thread_id=thread_id)
        chunk = next(gen)  # assistant text chunk: "Let me look that up."
        assert chunk  # ensure we got something

        gen.close()  # simulate UI 25s guard breaking the loop

        history = agent._get_history(thread_id)
        _assert_no_dangling_tool_calls(history)

    def test_close_after_tool_trace_no_dangling(self) -> None:
        """Closing the generator after the first tool-trace yield must not corrupt history."""
        agent = _make_foundry_agent()
        thread_id = agent.create_thread()

        tc = _make_tool_call_mock()
        agent._openai_client.chat.completions.create.return_value = (
            _make_completion_with_tool_calls([tc])
        )

        gen = agent.chat("Analyse the Volve survey.", thread_id=thread_id)
        for _chunk in gen:
            if _chunk.startswith("\n> 🔧"):
                break  # bail immediately after the tool-trace yield

        gen.close()

        history = agent._get_history(thread_id)
        _assert_no_dangling_tool_calls(history)

    def test_multiple_tool_calls_close_mid_round_no_dangling(self) -> None:
        """Close mid-round with multiple tool calls: all must be sealed in history."""
        agent = _make_foundry_agent()
        thread_id = agent.create_thread()

        tc1 = _make_tool_call_mock("tc-1", "query_survey_metadata", '{"survey_name":"Volve"}')
        tc2 = _make_tool_call_mock("tc-2", "get_well_data", '{"well_id":"15/9-F-11"}')
        agent._openai_client.chat.completions.create.return_value = (
            _make_completion_with_tool_calls([tc1, tc2])
        )

        gen = agent.chat("Multi-tool question.", thread_id=thread_id)
        # Advance past the assistant text and the first tool trace, then close
        chunks_seen = 0
        for _chunk in gen:
            chunks_seen += 1
            if chunks_seen >= 2:
                break

        gen.close()

        history = agent._get_history(thread_id)
        _assert_no_dangling_tool_calls(history)

    def test_normal_completion_history_valid(self) -> None:
        """When the generator runs to completion, history is valid (regression guard)."""
        agent = _make_foundry_agent()
        thread_id = agent.create_thread()

        tc = _make_tool_call_mock()
        # First call: tool round; second call: text-only terminal answer
        agent._openai_client.chat.completions.create.side_effect = [
            _make_completion_with_tool_calls([tc]),
            _make_completion_text_only("Here is your answer."),
        ]

        with patch(
            "deepseismic.agent.agent._dispatch_tool_call", return_value={"data": "mock"}
        ):
            chunks = list(agent.chat("Full round trip.", thread_id=thread_id))

        assert chunks  # got some output
        history = agent._get_history(thread_id)
        _assert_no_dangling_tool_calls(history)

    def test_seal_dangling_on_entry_heals_corrupt_history(self) -> None:
        """_seal_dangling_tool_calls removes pre-existing corrupt state before next request."""
        agent = _make_foundry_agent()
        thread_id = agent.create_thread()

        # Manually plant a corrupt history entry (dangling tool_calls, no tool response)
        history = agent._get_history(thread_id)
        history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "orphan-1",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                }
            ],
        })
        # Confirm it's dangling before the fix runs
        with pytest.raises(AssertionError, match="Dangling"):
            _assert_no_dangling_tool_calls(history)

        # Trigger seal via chat() entry path
        agent._openai_client.chat.completions.create.return_value = (
            _make_completion_text_only("Fixed.")
        )
        chunks = list(agent.chat("Follow-up after crash.", thread_id=thread_id))

        assert chunks
        history = agent._get_history(thread_id)
        _assert_no_dangling_tool_calls(history)

    def test_tool_call_round_committed_only_after_all_results(self) -> None:
        """The assistant+tool round must be committed atomically.

        At the tool-trace yield (inside the try block), the round buffer has NOT yet
        been flushed to persistent history.  After gen.close() the finally block runs
        and commits the sealed round, leaving history contract-valid.
        """
        agent = _make_foundry_agent()
        thread_id = agent.create_thread()

        tc = _make_tool_call_mock()
        agent._openai_client.chat.completions.create.return_value = (
            _make_completion_with_tool_calls([tc], assistant_text="")
        )

        history = agent._get_history(thread_id)
        gen = agent.chat("Check survey.", thread_id=thread_id)

        # Consume until the tool-trace chunk; at that yield the round buffer is in
        # memory but NOT flushed to history yet (no assistant tool_calls visible).
        for _chunk in gen:
            if _chunk.startswith("\n> 🔧"):
                assistant_with_tools = [
                    m for m in history
                    if m.get("role") == "assistant" and m.get("tool_calls")
                ]
                assert not assistant_with_tools, (
                    "Round buffer committed to history before all tool results ready; "
                    f"found: {assistant_with_tools}"
                )
                break

        len_before_close = len(history)
        gen.close()

        # After close, finally block must have committed the full sealed round.
        assert len(history) > len_before_close, "Round must be committed after generator close"
        _assert_no_dangling_tool_calls(history)
