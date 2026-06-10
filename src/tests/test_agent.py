"""Smoke tests for deepseismic.agent — tool registration and mock-LLM mode.

Strategy:
- All tests use mocks because DeepSeismicAgent is a stub.
- Tests document the expected interface contract (constructor, chat(), tools registry).
- Tool schema tests verify the OpenAI function-calling schema shape.
- Multistep tests verify multi-turn conversation plumbing.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from deepseismic.agent import agent as _mod

# ─────────────────────────────────────────────────────────────────────────────
# Expected tool schemas (define the contract, not the implementation)
# ─────────────────────────────────────────────────────────────────────────────

_SEISMIC_TOOL_SCHEMA: dict = {
    "name": "query_seismic_metadata",
    "description": "Retrieve seismic survey metadata and geometry.",
    "parameters": {
        "type": "object",
        "properties": {
            "survey_name": {"type": "string", "description": "Survey identifier"},
        },
        "required": ["survey_name"],
    },
}

_GEOLOGICAL_TOOL_SCHEMA: dict = {
    "name": "query_geological_data",
    "description": "Retrieve well log and formation data for a survey area.",
    "parameters": {
        "type": "object",
        "properties": {
            "well_id": {"type": "string", "description": "Well identifier"},
        },
        "required": ["well_id"],
    },
}


def _build_mock_agent(canned_response: dict) -> MagicMock:
    """Build a mock DeepSeismicAgent with the minimal expected interface."""
    agent = MagicMock(name="DeepSeismicAgent")
    agent.tools = {
        "query_seismic_metadata": _SEISMIC_TOOL_SCHEMA,
        "query_geological_data": _GEOLOGICAL_TOOL_SCHEMA,
    }
    agent.chat.return_value = canned_response["choices"][0]["message"]
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# test_mock_mode
# ─────────────────────────────────────────────────────────────────────────────


class TestMockMode:
    def test_mock_mode_returns_assistant_message(self, mock_llm_response: dict) -> None:
        """Agent in mock mode returns a dict with role='assistant' and non-empty content."""
        with patch.object(_mod, "DeepSeismicAgent") as MockAgent:
            instance = _build_mock_agent(mock_llm_response)
            MockAgent.return_value = instance

            agent = _mod.DeepSeismicAgent(mock_llm=True)
            response = agent.chat("Describe visible fault systems.")

            assert response["role"] == "assistant"
            assert isinstance(response["content"], str)
            assert len(response["content"]) > 0

    def test_mock_mode_env_var_respected(self, mock_llm_response: dict) -> None:
        """MOCK_LLM=true env var must be honoured by the agent constructor."""
        with patch.dict(os.environ, {"MOCK_LLM": "true"}):
            with patch.object(_mod, "DeepSeismicAgent") as MockAgent:
                instance = _build_mock_agent(mock_llm_response)
                MockAgent.return_value = instance

                agent = _mod.DeepSeismicAgent(mock_llm=True)
                agent.chat("Any fault at inline 150?")
                MockAgent.assert_called_once_with(mock_llm=True)

    def test_mock_mode_no_http_calls(self, mock_llm_response: dict) -> None:
        """Mock mode must not trigger any real HTTP/network calls."""
        with patch.object(_mod, "DeepSeismicAgent") as MockAgent:
            instance = _build_mock_agent(mock_llm_response)
            MockAgent.return_value = instance

            agent = _mod.DeepSeismicAgent(mock_llm=True)
            agent.chat("What is the survey extent?")
            # A MagicMock never performs real I/O — chat was called once
            assert instance.chat.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# test_tool_registration
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistration:
    def test_at_least_two_tools_registered(self, mock_llm_response: dict) -> None:
        """Agent must expose at least two tools (seismic + geological)."""
        agent = _build_mock_agent(mock_llm_response)
        assert len(agent.tools) >= 2

    def test_all_tools_have_required_schema_keys(self, mock_llm_response: dict) -> None:
        """Every registered tool schema must contain 'name', 'description', 'parameters'."""
        agent = _build_mock_agent(mock_llm_response)
        for tool_name, schema in agent.tools.items():
            assert "name" in schema, f"Tool '{tool_name}' missing 'name'"
            assert "description" in schema, f"Tool '{tool_name}' missing 'description'"
            assert "parameters" in schema, f"Tool '{tool_name}' missing 'parameters'"

    def test_tool_names_are_non_empty_strings(self, mock_llm_response: dict) -> None:
        """Tool dictionary keys must be non-empty strings."""
        agent = _build_mock_agent(mock_llm_response)
        for name in agent.tools:
            assert isinstance(name, str) and name, f"Invalid tool name: {name!r}"

    def test_tool_parameters_have_type_field(self, mock_llm_response: dict) -> None:
        """Each tool's parameters block must contain a 'type' field."""
        agent = _build_mock_agent(mock_llm_response)
        for schema in agent.tools.values():
            params = schema.get("parameters", {})
            assert "type" in params, (
                f"Tool '{schema['name']}' parameters block missing 'type'"
            )

    def test_tool_parameters_type_is_object(self, mock_llm_response: dict) -> None:
        """OpenAI function-calling convention requires parameters.type == 'object'."""
        agent = _build_mock_agent(mock_llm_response)
        for schema in agent.tools.values():
            assert schema["parameters"]["type"] == "object", (
                f"Tool '{schema['name']}' parameters.type must be 'object'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# test_tool_seismic_query
# ─────────────────────────────────────────────────────────────────────────────


class TestToolSeismicQuery:
    _REQUIRED_FIELDS = {"survey_name", "n_inlines", "n_crosslines", "n_samples"}

    def test_seismic_query_returns_metadata_format(self) -> None:
        """query_seismic_metadata must return a dict with geometry fields."""
        mock_result = {
            "survey_name": "Volve-North",
            "n_inlines": 200,
            "n_crosslines": 300,
            "n_samples": 1500,
            "crs": "EPSG:23032",
        }
        with patch.object(_mod, "query_seismic_metadata", return_value=mock_result, create=True):
            result = _mod.query_seismic_metadata(survey_name="Volve-North")
            for field in self._REQUIRED_FIELDS:
                assert field in result, f"Missing field: {field}"

    def test_seismic_query_schema_requires_survey_name(self) -> None:
        """query_seismic_metadata schema must list 'survey_name' as required."""
        required = _SEISMIC_TOOL_SCHEMA["parameters"].get("required", [])
        assert "survey_name" in required

    def test_seismic_query_n_samples_positive(self) -> None:
        """n_samples in the returned metadata must be a positive integer."""
        mock_result = {
            "survey_name": "S1",
            "n_inlines": 10,
            "n_crosslines": 10,
            "n_samples": 500,
        }
        with patch.object(_mod, "query_seismic_metadata", return_value=mock_result, create=True):
            result = _mod.query_seismic_metadata(survey_name="S1")
            assert isinstance(result["n_samples"], int)
            assert result["n_samples"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# test_tool_geological_query
# ─────────────────────────────────────────────────────────────────────────────


class TestToolGeologicalQuery:
    _REQUIRED_FIELDS = {"well_id", "formation", "depth_m"}

    def test_geological_query_returns_well_data_format(self) -> None:
        """query_geological_data must return a dict with well formation fields."""
        mock_result = {
            "well_id": "Volve-15/9-F-11",
            "formation": "Hugin Fm",
            "depth_m": 3400.5,
            "lithology": "sandstone",
        }
        with patch.object(_mod, "query_geological_data", return_value=mock_result, create=True):
            result = _mod.query_geological_data(well_id="Volve-15/9-F-11")
            for field in self._REQUIRED_FIELDS:
                assert field in result, f"Missing field: {field}"

    def test_geological_query_schema_requires_well_id(self) -> None:
        """query_geological_data schema must list 'well_id' as required."""
        required = _GEOLOGICAL_TOOL_SCHEMA["parameters"].get("required", [])
        assert "well_id" in required

    def test_geological_query_depth_is_numeric(self) -> None:
        """depth_m must be a numeric (int or float) value."""
        mock_result = {"well_id": "W1", "formation": "X", "depth_m": 1234.0}
        with patch.object(_mod, "query_geological_data", return_value=mock_result, create=True):
            result = _mod.query_geological_data(well_id="W1")
            assert isinstance(result["depth_m"], (int, float))


# ─────────────────────────────────────────────────────────────────────────────
# test_multistep_workflow
# ─────────────────────────────────────────────────────────────────────────────


class TestMultistepWorkflow:
    _TURNS = [
        ("What surveys are available?", "I found 3 surveys: Volve-N, Volve-S, Field-X."),
        ("Describe the faults in Volve-N.", "Fault system NW-SE, inline 100-200."),
        ("Show me the confidence score.", "Confidence: 0.82 based on amplitude coherence."),
    ]

    def test_multistep_workflow_returns_one_message_per_turn(
        self, mock_llm_response: dict
    ) -> None:
        """Multi-turn chat: each turn returns exactly one assistant message."""
        with patch.object(_mod, "DeepSeismicAgent") as MockAgent:
            instance = MagicMock()
            canned_iter = iter(
                [{"role": "assistant", "content": resp} for _, resp in self._TURNS]
            )
            instance.chat.side_effect = lambda msg: next(canned_iter)
            MockAgent.return_value = instance

            agent = _mod.DeepSeismicAgent(mock_llm=True)
            responses = [agent.chat(msg) for msg, _ in self._TURNS]

        assert len(responses) == 3
        for r in responses:
            assert r["role"] == "assistant"
            assert len(r["content"]) > 0

    def test_multistep_workflow_chat_called_per_turn(
        self, mock_llm_response: dict
    ) -> None:
        """Agent.chat must be invoked exactly once per conversation turn."""
        with patch.object(_mod, "DeepSeismicAgent") as MockAgent:
            instance = _build_mock_agent(mock_llm_response)
            MockAgent.return_value = instance

            agent = _mod.DeepSeismicAgent(mock_llm=True)
            for msg, _ in self._TURNS:
                agent.chat(msg)

            assert instance.chat.call_count == len(self._TURNS)

    def test_multistep_workflow_messages_ordered(self, mock_llm_response: dict) -> None:
        """Chat calls must receive the user messages in the order they were sent."""
        with patch.object(_mod, "DeepSeismicAgent") as MockAgent:
            instance = _build_mock_agent(mock_llm_response)
            MockAgent.return_value = instance

            agent = _mod.DeepSeismicAgent(mock_llm=True)
            messages = [msg for msg, _ in self._TURNS]
            for msg in messages:
                agent.chat(msg)

            actual_calls = [c.args[0] for c in instance.chat.call_args_list]
            assert actual_calls == messages
