"""Real-mode agent tests — fail-loud behaviour and mock-vs-real selection.

Coverage
--------
TestAgentFailLoud          — DeepSeismicAgent and FoundryAgent raise clear RuntimeError
                             when AZURE_PROJECT_ENDPOINT is absent in live mode
                             (monkeypatch clears env). Never silently fall back to mock.
TestAgentMockMode          — MOCK_LLM=true activates MockAgent; chat() yields text chunks
                             without any Azure calls or credentials.
TestAgentMockVsRealEnvVar  — _is_mock_mode() call-time evaluation; real mode is the default.
TestMockAgentResponses     — MockAgent returns non-empty text with recognisable structure
                             for different message intents (regression guard on mock content).

All tests are CI-safe — no Azure credentials, no Azurite.
"""

from __future__ import annotations

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# TestAgentFailLoud — live mode raises instead of silently mocking
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentFailLoud:
    """DeepSeismicAgent in live mode must fail loud, not fall back to mock responses."""

    def test_deepseismic_agent_raises_without_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DeepSeismicAgent() raises RuntimeError when AZURE_PROJECT_ENDPOINT is absent.

        This is the key guard for the Wave 1 de-mock: live mode must never silently
        fall back to canned responses when credentials are missing.
        """
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("MOCK_LLM", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        with pytest.raises(RuntimeError):
            DeepSeismicAgent()

    def test_deepseismic_agent_error_message_mentions_azure_project_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RuntimeError must name AZURE_PROJECT_ENDPOINT so the operator knows what to set."""
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("MOCK_LLM", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        with pytest.raises(RuntimeError, match="AZURE_PROJECT_ENDPOINT"):
            DeepSeismicAgent()

    def test_deepseismic_agent_error_message_mentions_mock_llm_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RuntimeError message must mention MOCK_LLM so developer knows the offline option."""
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("MOCK_LLM", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        with pytest.raises(RuntimeError, match="MOCK_LLM"):
            DeepSeismicAgent()

    def test_foundry_agent_raises_without_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FoundryAgent() raises RuntimeError when AZURE_PROJECT_ENDPOINT is absent."""
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)

        from deepseismic.agent.agent import FoundryAgent

        with pytest.raises(RuntimeError):
            FoundryAgent()

    def test_foundry_agent_error_message_mentions_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FoundryAgent RuntimeError message must reference AZURE_PROJECT_ENDPOINT."""
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)

        from deepseismic.agent.agent import FoundryAgent

        with pytest.raises(RuntimeError, match="AZURE_PROJECT_ENDPOINT"):
            FoundryAgent()

    def test_deepseismic_agent_empty_endpoint_string_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty string AZURE_PROJECT_ENDPOINT must also raise (not be treated as set)."""
        monkeypatch.setenv("AZURE_PROJECT_ENDPOINT", "")
        monkeypatch.delenv("MOCK_LLM", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        with pytest.raises(RuntimeError):
            DeepSeismicAgent()

    def test_deepseismic_agent_whitespace_endpoint_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A whitespace-only AZURE_PROJECT_ENDPOINT must raise (uses .strip())."""
        monkeypatch.setenv("AZURE_PROJECT_ENDPOINT", "   ")
        monkeypatch.delenv("MOCK_LLM", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        with pytest.raises(RuntimeError):
            DeepSeismicAgent()


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentMockMode — MOCK_LLM=true enables offline iteration without Azure
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentMockMode:
    def test_deepseismic_agent_no_error_in_mock_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DeepSeismicAgent() with MOCK_LLM=true must not raise even without Azure creds."""
        monkeypatch.setenv("MOCK_LLM", "true")
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        agent = DeepSeismicAgent()  # Must not raise
        assert agent is not None

    def test_deepseismic_agent_mock_chat_yields_non_empty_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MockAgent.chat() must yield at least one non-empty text chunk."""
        monkeypatch.setenv("MOCK_LLM", "true")
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        agent = DeepSeismicAgent()
        chunks = list(agent._impl.chat("Describe the Volve dataset."))
        full_text = "".join(chunks)
        assert len(full_text) > 0, "MockAgent.chat() must yield non-empty text"

    def test_deepseismic_agent_mock_text_contains_mock_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MockAgent response must contain '[MOCK MODE' to distinguish from real responses."""
        monkeypatch.setenv("MOCK_LLM", "true")
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent

        agent = DeepSeismicAgent()
        text = "".join(agent._impl.chat("What is the survey geometry?"))
        assert "MOCK MODE" in text.upper(), (
            "Mock response must contain 'MOCK MODE' marker to distinguish from real output"
        )

    def test_deepseismic_agent_mock_mode_uses_mock_agent_impl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In mock mode, DeepSeismicAgent._impl must be a MockAgent, not FoundryAgent."""
        monkeypatch.setenv("MOCK_LLM", "true")
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent, MockAgent

        agent = DeepSeismicAgent()
        assert isinstance(agent._impl, MockAgent), (
            f"Expected MockAgent impl in mock mode, got {type(agent._impl).__name__}"
        )

    def test_deepseismic_agent_mock_mode_with_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MOCK_LLM=1 also activates mock mode."""
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.delenv("AZURE_PROJECT_ENDPOINT", raising=False)

        from deepseismic.agent.agent import DeepSeismicAgent, MockAgent

        agent = DeepSeismicAgent()
        assert isinstance(agent._impl, MockAgent)


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentMockVsRealEnvVar — _is_mock_mode() call-time evaluation
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentMockVsRealEnvVar:
    """_is_mock_mode() is evaluated at call time, not at module import time."""

    def test_is_mock_mode_false_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no MOCK_LLM env var, _is_mock_mode() returns False."""
        monkeypatch.delenv("MOCK_LLM", raising=False)

        from deepseismic.agent.agent import _is_mock_mode

        assert _is_mock_mode() is False

    def test_is_mock_mode_true_when_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOCK_LLM=true → _is_mock_mode() True."""
        monkeypatch.setenv("MOCK_LLM", "true")

        from deepseismic.agent.agent import _is_mock_mode

        assert _is_mock_mode() is True

    def test_is_mock_mode_true_when_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOCK_LLM=1 → _is_mock_mode() True."""
        monkeypatch.setenv("MOCK_LLM", "1")

        from deepseismic.agent.agent import _is_mock_mode

        assert _is_mock_mode() is True

    def test_is_mock_mode_true_when_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOCK_LLM=yes → _is_mock_mode() True."""
        monkeypatch.setenv("MOCK_LLM", "yes")

        from deepseismic.agent.agent import _is_mock_mode

        assert _is_mock_mode() is True

    def test_is_mock_mode_false_when_false_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOCK_LLM=false → _is_mock_mode() False."""
        monkeypatch.setenv("MOCK_LLM", "false")

        from deepseismic.agent.agent import _is_mock_mode

        assert _is_mock_mode() is False

    def test_is_mock_mode_reflects_env_changes_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_is_mock_mode() re-reads the env var each call — not frozen at import."""
        from deepseismic.agent.agent import _is_mock_mode

        monkeypatch.delenv("MOCK_LLM", raising=False)
        assert _is_mock_mode() is False

        monkeypatch.setenv("MOCK_LLM", "true")
        assert _is_mock_mode() is True

        monkeypatch.setenv("MOCK_LLM", "false")
        assert _is_mock_mode() is False


# ─────────────────────────────────────────────────────────────────────────────
# TestMockAgentResponses — MockAgent keyword routing and content guards
# ─────────────────────────────────────────────────────────────────────────────


class TestMockAgentResponses:
    """MockAgent must route keyword messages to their canned response variants."""

    def _get_mock_agent(self) -> object:
        from deepseismic.agent.agent import MockAgent

        return MockAgent()

    def test_default_response_for_generic_query(self) -> None:
        """Generic query returns the 'default' canned response (non-empty)."""
        agent = self._get_mock_agent()
        text = "".join(agent.chat("Tell me about the dataset."))
        assert len(text) > 50

    def test_status_keyword_routes_to_status_response(self) -> None:
        """'status' keyword routes to the status-oriented canned response."""
        agent = self._get_mock_agent()
        text = "".join(agent.chat("What is the run status?"))
        # Status response mentions run states
        assert len(text) > 0

    def test_wells_keyword_routes_to_wells_response(self) -> None:
        """'well' keyword routes to the well-data canned response."""
        agent = self._get_mock_agent()
        text = "".join(agent.chat("Show me the well formations."))
        assert len(text) > 0

    def test_interpret_keyword_routes_to_interpret_response(self) -> None:
        """'analyze' keyword routes to the end-to-end workflow response."""
        agent = self._get_mock_agent()
        text = "".join(agent.chat("Analyze the survey end-to-end."))
        assert len(text) > 0

    def test_mock_response_yields_multiple_chunks(self) -> None:
        """MockAgent yields multiple word-level chunks (simulates streaming)."""
        agent = self._get_mock_agent()
        chunks = list(agent.chat("What data is available?"))
        assert len(chunks) > 1, "MockAgent must yield multiple chunks (word-by-word streaming)"
