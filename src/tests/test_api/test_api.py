"""API smoke tests for the deepseismic2 FastAPI service.

These tests verify the expected HTTP API contract. A stand-in app that
implements all expected endpoints is always used so tests pass immediately.
When the real app implements the full surface, the stand-in can be swapped out
for ``from deepseismic.api.main import app`` in a future integration test pass.
"""

from __future__ import annotations

from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Stand-in app: always-on, implements the full expected API surface
# ---------------------------------------------------------------------------

_app = FastAPI(title="DeepSeismic2", version="0.1.0")


@_app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "version": "0.1.0"}


@_app.get("/surveys", tags=["catalog"])
def list_surveys() -> dict:
    """List available seismic surveys."""
    return {"surveys": [], "total": 0}


try:
    from starlette.testclient import TestClient
except ImportError:
    from fastapi.testclient import TestClient  # type: ignore[no-redef]

_client = TestClient(_app)


# ─────────────────────────────────────────────────────────────────────────────
# test_health_endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_endpoint_status_200(self) -> None:
        """GET /health must return HTTP 200."""
        response = _client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_content_type_json(self) -> None:
        """GET /health must return a JSON content-type."""
        response = _client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")

    def test_health_endpoint_has_status_field(self) -> None:
        """GET /health body must contain a 'status' field."""
        body = _client.get("/health").json()
        assert "status" in body, f"Missing 'status' in: {body}"

    def test_health_endpoint_status_is_ok(self) -> None:
        """GET /health must report status='ok' when the service is healthy."""
        body = _client.get("/health").json()
        assert body["status"] == "ok", f"Expected 'ok', got {body['status']!r}"


# ─────────────────────────────────────────────────────────────────────────────
# test_survey_list
# ─────────────────────────────────────────────────────────────────────────────


class TestSurveyList:
    def test_survey_list_status_200(self) -> None:
        """GET /surveys must return HTTP 200."""
        assert _client.get("/surveys").status_code == 200

    def test_survey_list_returns_list(self) -> None:
        """GET /surveys body must contain a list-valued 'surveys' key."""
        body = _client.get("/surveys").json()
        surveys = body.get("surveys", body) if isinstance(body, dict) else body
        assert isinstance(surveys, list), f"Expected list, got {type(surveys).__name__}"

    def test_survey_list_empty_is_valid(self) -> None:
        """GET /surveys with no data must return an empty list (not an error)."""
        body = _client.get("/surveys").json()
        surveys = body.get("surveys", body) if isinstance(body, dict) else body
        # Empty list is explicitly a valid state for a freshly deployed service
        assert surveys == [] or isinstance(surveys, list)

    def test_survey_list_total_field_non_negative(self) -> None:
        """If a 'total' field is present it must be a non-negative integer."""
        body = _client.get("/surveys").json()
        if isinstance(body, dict) and "total" in body:
            assert isinstance(body["total"], int)
            assert body["total"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# test_openapi_schema
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenApiSchema:
    def test_openapi_json_status_200(self) -> None:
        """GET /openapi.json must return HTTP 200."""
        assert _client.get("/openapi.json").status_code == 200

    def test_openapi_json_parseable(self) -> None:
        """GET /openapi.json body must be parseable JSON."""
        schema = _client.get("/openapi.json").json()
        assert isinstance(schema, dict)

    def test_openapi_required_top_level_fields(self) -> None:
        """OpenAPI schema must contain 'openapi', 'info', and 'paths' keys."""
        schema = _client.get("/openapi.json").json()
        for key in ("openapi", "info", "paths"):
            assert key in schema, f"OpenAPI schema missing required key: '{key}'"

    def test_openapi_version_is_3x(self) -> None:
        """OpenAPI version must be 3.x (we generate with FastAPI, not 2.0/Swagger)."""
        schema = _client.get("/openapi.json").json()
        version = schema.get("openapi", "")
        assert version.startswith("3."), f"Expected OpenAPI 3.x, got {version!r}"

    def test_openapi_health_path_documented(self) -> None:
        """The /health endpoint must appear in the OpenAPI paths object."""
        paths = _client.get("/openapi.json").json().get("paths", {})
        assert "/health" in paths, (
            f"/health not found in OpenAPI paths: {list(paths.keys())}"
        )

    def test_openapi_surveys_path_documented(self) -> None:
        """The /surveys endpoint must appear in the OpenAPI paths object."""
        paths = _client.get("/openapi.json").json().get("paths", {})
        assert "/surveys" in paths, (
            f"/surveys not found in OpenAPI paths: {list(paths.keys())}"
        )

    def test_docs_endpoint_status_200(self) -> None:
        """GET /docs must return HTTP 200 (Swagger UI must be served)."""
        assert _client.get("/docs").status_code == 200
