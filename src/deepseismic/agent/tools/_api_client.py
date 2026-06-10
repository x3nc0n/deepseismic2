"""Shared HTTP client for DeepSeismic agent tool modules.

Reads base URL from the environment (first match wins):

    DEEPSEISMIC_API_URL  — canonical env var
    BACKEND_URL          — legacy fallback
    (default)            — http://localhost:8000

Additional tunables::

    DEEPSEISMIC_API_TIMEOUT   seconds (float, default 20)

Retry policy:  up to 2 retries on ``httpx.RequestError`` and HTTP 503,
with a linear back-off of 1 s, 2 s between attempts.

Raises ``APIError`` on all failure paths so callers can return a uniform
``{"error": ..., "available": False}`` dict without scattering try/except
throughout every tool module.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT: float = float(os.environ.get("DEEPSEISMIC_API_TIMEOUT", "20"))
_MAX_RETRIES: int = 2
_RETRY_DELAY_S: float = 1.0


class APIError(Exception):
    """Raised when the backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _base_url() -> str:
    """Resolve the API base URL at call time so test overrides take effect."""
    return (
        os.environ.get("DEEPSEISMIC_API_URL")
        or os.environ.get("BACKEND_URL")
        or "http://localhost:8000"
    )


def get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET ``{base_url}{path}``, retry on 503 / network error.

    Args:
        path:   URL path including leading slash (e.g. ``"/api/surveys"``).
        params: Optional query parameters; ``None``-valued keys are dropped.

    Returns:
        Parsed JSON response body as a dict.

    Raises:
        APIError: On HTTP error status or when the backend is unreachable.
    """
    url = f"{_base_url()}{path}"
    clean_params = {k: v for k, v in (params or {}).items() if v is not None} or None
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        if attempt:
            time.sleep(_RETRY_DELAY_S * attempt)
        try:
            resp = httpx.get(url, params=clean_params, timeout=_TIMEOUT)
            if resp.status_code == 503 and attempt < _MAX_RETRIES:
                logger.debug("503 on GET %s — retry %d/%d", url, attempt + 1, _MAX_RETRIES)
                last_exc = APIError("503 Service Unavailable", 503)
                continue
            resp.raise_for_status()
            data = resp.json()
            # Normalise: list responses are always wrapped by callers
            return data if isinstance(data, dict) else {"_list": data}
        except httpx.RequestError as exc:
            logger.debug("GET %s attempt %d failed: %s", url, attempt + 1, exc)
            last_exc = exc
            if attempt < _MAX_RETRIES:
                continue
        except httpx.HTTPStatusError as exc:
            raise APIError(
                f"HTTP {exc.response.status_code} from GET {url}: {exc.response.text[:200]}",
                status_code=exc.response.status_code,
            ) from exc

    raise APIError(f"Backend unreachable at {url}: {last_exc}") from last_exc


def get_list(path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """GET ``{base_url}{path}`` where the response is a JSON array.

    Args:
        path:   URL path including leading slash.
        params: Optional query parameters; ``None``-valued keys are dropped.

    Returns:
        Parsed JSON response body as a list of dicts.

    Raises:
        APIError: On HTTP error status or when the backend is unreachable.
    """
    url = f"{_base_url()}{path}"
    clean_params = {k: v for k, v in (params or {}).items() if v is not None} or None
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        if attempt:
            time.sleep(_RETRY_DELAY_S * attempt)
        try:
            resp = httpx.get(url, params=clean_params, timeout=_TIMEOUT)
            if resp.status_code == 503 and attempt < _MAX_RETRIES:
                logger.debug("503 on GET %s — retry %d/%d", url, attempt + 1, _MAX_RETRIES)
                last_exc = APIError("503 Service Unavailable", 503)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else [data]
        except httpx.RequestError as exc:
            logger.debug("GET %s attempt %d failed: %s", url, attempt + 1, exc)
            last_exc = exc
            if attempt < _MAX_RETRIES:
                continue
        except httpx.HTTPStatusError as exc:
            raise APIError(
                f"HTTP {exc.response.status_code} from GET {url}: {exc.response.text[:200]}",
                status_code=exc.response.status_code,
            ) from exc

    raise APIError(f"Backend unreachable at {url}: {last_exc}") from last_exc


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST ``{base_url}{path}`` with a JSON body, retry on 503 / network error.

    Args:
        path:    URL path including leading slash (e.g. ``"/api/interpretation/fault-detection"``).
        payload: Request body serialised as JSON.

    Returns:
        Parsed JSON response body as a dict.

    Raises:
        APIError: On HTTP error status or when the backend is unreachable.
    """
    url = f"{_base_url()}{path}"
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        if attempt:
            time.sleep(_RETRY_DELAY_S * attempt)
        try:
            resp = httpx.post(url, json=payload, timeout=_TIMEOUT)
            if resp.status_code == 503 and attempt < _MAX_RETRIES:
                logger.debug("503 on POST %s — retry %d/%d", url, attempt + 1, _MAX_RETRIES)
                last_exc = APIError("503 Service Unavailable", 503)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as exc:
            logger.debug("POST %s attempt %d failed: %s", url, attempt + 1, exc)
            last_exc = exc
            if attempt < _MAX_RETRIES:
                continue
        except httpx.HTTPStatusError as exc:
            raise APIError(
                f"HTTP {exc.response.status_code} from POST {url}: {exc.response.text[:200]}",
                status_code=exc.response.status_code,
            ) from exc

    raise APIError(f"Backend unreachable at {url}: {last_exc}") from last_exc
