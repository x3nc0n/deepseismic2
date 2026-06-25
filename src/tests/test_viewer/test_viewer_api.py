"""CI-safe unit tests for the gradio-free viewer API client (``_viewer_api``).

These tests mock ``requests`` so no network or gradio import is required.
Coverage:
1. SurveyGeometry.inline_to_index / inline_choices_bounds math.
2. list_surveys / get_survey_geometry happy path + fail-loud (ViewerAPIError).
3. fetch_inline fail-loud on 404 / 500 / empty payload.
4. start_fault_detection returns run_id; raises without one.
5. fetch_overlay fail-loud on missing data.

Design contract (issue #17): real-data helpers must RAISE ViewerAPIError on any
failure, never silently return None and fall back to a synthetic placeholder.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from deepseismic.ui import _viewer_api as vapi
from deepseismic.ui._viewer_api import SurveyGeometry, ViewerAPIError

BASE = "http://test-api:8000"


def _resp(status_code: int, json_body: Any = None, text: str = "") -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body
    r.text = text
    return r


# ---------------------------------------------------------------------------
# SurveyGeometry math
# ---------------------------------------------------------------------------


class TestSurveyGeometry:
    def test_inline_choices_bounds(self) -> None:
        g = SurveyGeometry("s", inline_min=9961, inline_max=10361, inline_step=1)
        assert g.inline_choices_bounds() == (9961, 10361, 1)

    def test_step_zero_coerced_to_one(self) -> None:
        g = SurveyGeometry("s", inline_min=0, inline_max=10, inline_step=0)
        assert g.inline_choices_bounds() == (0, 10, 1)

    def test_inline_to_index_basic(self) -> None:
        g = SurveyGeometry("s", inline_min=9961, inline_max=10361, inline_step=1)
        assert g.inline_to_index(9961) == 0
        assert g.inline_to_index(9971) == 10
        assert g.inline_to_index(10361) == 400

    def test_inline_to_index_clamped(self) -> None:
        g = SurveyGeometry("s", inline_min=100, inline_max=200, inline_step=1)
        assert g.inline_to_index(50) == 0       # below range
        assert g.inline_to_index(9999) == 100   # above range -> n

    def test_inline_to_index_with_step(self) -> None:
        g = SurveyGeometry("s", inline_min=1000, inline_max=1100, inline_step=5)
        assert g.inline_to_index(1000) == 0
        assert g.inline_to_index(1025) == 5


# ---------------------------------------------------------------------------
# list_surveys
# ---------------------------------------------------------------------------


class TestListSurveys:
    def test_happy_path_list_payload(self) -> None:
        body = [{"survey_id": "a"}, {"survey_id": "b"}]
        with patch("requests.get", return_value=_resp(200, body)):
            assert vapi.list_surveys(BASE) == ["a", "b"]

    def test_happy_path_wrapped_payload(self) -> None:
        body = {"surveys": [{"survey_id": "x"}]}
        with patch("requests.get", return_value=_resp(200, body)):
            assert vapi.list_surveys(BASE) == ["x"]

    def test_http_error_raises(self) -> None:
        with patch("requests.get", return_value=_resp(503, text="down")):
            with pytest.raises(ViewerAPIError):
                vapi.list_surveys(BASE)

    def test_transport_error_raises(self) -> None:
        with patch("requests.get", side_effect=OSError("boom")):
            with pytest.raises(ViewerAPIError):
                vapi.list_surveys(BASE)


# ---------------------------------------------------------------------------
# get_survey_geometry
# ---------------------------------------------------------------------------


class TestGetSurveyGeometry:
    def test_happy_path(self) -> None:
        body = {"geometry": {"inline_min": 9961, "inline_max": 10361, "inline_step": 1}}
        with patch("requests.get", return_value=_resp(200, body)):
            g = vapi.get_survey_geometry("volve-st10010", BASE)
        assert g == SurveyGeometry("volve-st10010", 9961, 10361, 1)

    def test_404_raises(self) -> None:
        with patch("requests.get", return_value=_resp(404)):
            with pytest.raises(ViewerAPIError, match="not found"):
                vapi.get_survey_geometry("nope", BASE)

    def test_missing_geometry_raises(self) -> None:
        with patch("requests.get", return_value=_resp(200, {"geometry": {}})):
            with pytest.raises(ViewerAPIError, match="inline geometry"):
                vapi.get_survey_geometry("s", BASE)


# ---------------------------------------------------------------------------
# fetch_inline — fail-loud
# ---------------------------------------------------------------------------


class TestFetchInline:
    def test_happy_path(self) -> None:
        body = {"amplitude": [[1.0, 2.0]], "crossline_coords": [0], "twtt_ms": [0.0, 4.0]}
        with patch("requests.get", return_value=_resp(200, body)):
            assert vapi.fetch_inline("s", 9961, BASE)["amplitude"] == [[1.0, 2.0]]

    def test_404_raises_not_none(self) -> None:
        with patch("requests.get", return_value=_resp(404)):
            with pytest.raises(ViewerAPIError, match="not found"):
                vapi.fetch_inline("s", 9961, BASE)

    def test_500_raises(self) -> None:
        with patch("requests.get", return_value=_resp(500, text="kaboom")):
            with pytest.raises(ViewerAPIError):
                vapi.fetch_inline("s", 9961, BASE)

    def test_empty_amplitude_raises(self) -> None:
        with patch("requests.get", return_value=_resp(200, {"amplitude": []})):
            with pytest.raises(ViewerAPIError, match="no amplitude"):
                vapi.fetch_inline("s", 9961, BASE)


# ---------------------------------------------------------------------------
# start_fault_detection + poll + overlay
# ---------------------------------------------------------------------------


class TestInference:
    def test_start_returns_run_id(self) -> None:
        with patch("requests.post", return_value=_resp(202, {"run_id": "abc123"})):
            assert vapi.start_fault_detection("s", base_url=BASE) == "abc123"

    def test_start_without_run_id_raises(self) -> None:
        with patch("requests.post", return_value=_resp(202, {})):
            with pytest.raises(ViewerAPIError, match="run_id"):
                vapi.start_fault_detection("s", base_url=BASE)

    def test_start_http_error_raises(self) -> None:
        with patch("requests.post", return_value=_resp(503, text="x")):
            with pytest.raises(ViewerAPIError):
                vapi.start_fault_detection("s", base_url=BASE)

    def test_poll_status_returns_dict(self) -> None:
        with patch("requests.get", return_value=_resp(200, {"status": "complete"})):
            assert vapi.poll_status("abc", BASE)["status"] == "complete"

    def test_fetch_overlay_happy(self) -> None:
        body = {"fault_probability": [[0.1, 0.2]], "fault_mask": [[0, 0]]}
        with patch("requests.get", return_value=_resp(200, body)):
            assert vapi.fetch_overlay("abc", 0, BASE)["fault_probability"] == [[0.1, 0.2]]

    def test_fetch_overlay_missing_data_raises(self) -> None:
        with patch("requests.get", return_value=_resp(200, {"fault_probability": None})):
            with pytest.raises(ViewerAPIError):
                vapi.fetch_overlay("abc", 0, BASE)
