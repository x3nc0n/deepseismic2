"""Pure (gradio-free) API-client helpers for the DeepSeismic web viewer.

This module has **no gradio import** so it can be unit-tested without the heavy
UI stack.  ``gradio_app.py`` composes these helpers into the interactive UI.

Design rule — **fail loud** (see issue #17): real-data calls raise
:class:`ViewerAPIError` with a human-readable message on any failure instead of
silently returning ``None``.  The UI surfaces that message; it must never quietly
fall back to a synthetic placeholder on the real-data path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class ViewerAPIError(RuntimeError):
    """Raised when an API call fails — carries a display-ready message."""


def api_base_url() -> str:
    """Resolve the backend API base URL from the environment."""
    return (
        os.environ.get("API_BASE_URL")
        or os.environ.get("DEEPSEISMIC_API_URL")
        or os.environ.get("BACKEND_URL")
        or "http://localhost:8000"
    )


@dataclass(frozen=True)
class SurveyGeometry:
    """Inline-axis geometry needed to drive the viewer controls."""

    survey_id: str
    inline_min: int
    inline_max: int
    inline_step: int

    def inline_choices_bounds(self) -> tuple[int, int, int]:
        """Return ``(min, max, step)`` suitable for a slider, step >= 1."""
        step = self.inline_step if self.inline_step and self.inline_step > 0 else 1
        return self.inline_min, self.inline_max, step

    def inline_to_index(self, inline_abs: int) -> int:
        """Convert an **absolute** inline number to a 0-based volume index.

        The interpretation ``/overlay/{i}`` endpoint indexes the result volume
        positionally, so we map absolute inline -> index via this geometry.
        Result is clamped to ``[0, n_inlines - 1]``.
        """
        step = self.inline_step if self.inline_step and self.inline_step > 0 else 1
        n = (self.inline_max - self.inline_min) // step
        idx = (int(inline_abs) - self.inline_min) // step
        return max(0, min(n, idx))


# ---------------------------------------------------------------------------
# Survey catalog
# ---------------------------------------------------------------------------


def list_surveys(base_url: str | None = None, *, timeout: float = 10.0) -> list[str]:
    """Return survey ids from ``GET /api/surveys``.

    Raises :class:`ViewerAPIError` on transport/HTTP failure so the UI can show
    why the picker is empty instead of silently offering nothing.
    """
    import requests

    base = base_url or api_base_url()
    try:
        resp = requests.get(f"{base}/api/surveys", timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — re-raised as a typed UI error
        raise ViewerAPIError(f"Could not reach API at {base}: {exc}") from exc
    if resp.status_code != 200:
        raise ViewerAPIError(
            f"GET /api/surveys returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json()
    items = data if isinstance(data, list) else data.get("surveys", data.get("items", []))
    return [it["survey_id"] for it in items if isinstance(it, dict) and it.get("survey_id")]


def get_survey_geometry(
    survey_id: str, base_url: str | None = None, *, timeout: float = 10.0
) -> SurveyGeometry:
    """Return :class:`SurveyGeometry` from ``GET /api/surveys/{id}``.

    Raises :class:`ViewerAPIError` on failure or malformed geometry.
    """
    import requests

    base = base_url or api_base_url()
    try:
        resp = requests.get(f"{base}/api/surveys/{survey_id}", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise ViewerAPIError(f"Could not reach API at {base}: {exc}") from exc
    if resp.status_code == 404:
        raise ViewerAPIError(f"Survey '{survey_id}' not found")
    if resp.status_code != 200:
        raise ViewerAPIError(
            f"GET /api/surveys/{survey_id} returned HTTP {resp.status_code}"
        )
    geom = (resp.json() or {}).get("geometry") or {}
    try:
        return SurveyGeometry(
            survey_id=survey_id,
            inline_min=int(geom["inline_min"]),
            inline_max=int(geom["inline_max"]),
            inline_step=int(geom.get("inline_step", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ViewerAPIError(
            f"Survey '{survey_id}' metadata is missing inline geometry"
        ) from exc


# ---------------------------------------------------------------------------
# Amplitude inline slice
# ---------------------------------------------------------------------------


def fetch_inline(
    survey_id: str, inline: int, base_url: str | None = None, *, timeout: float = 30.0
) -> dict[str, Any]:
    """Return the amplitude inline payload from the API.

    Raises :class:`ViewerAPIError` on any failure (transport, 404, 5xx).  Never
    returns ``None`` — the caller must surface the error, not hide it.
    """
    import requests

    base = base_url or api_base_url()
    try:
        resp = requests.get(
            f"{base}/api/surveys/{survey_id}/inline/{inline}", timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        raise ViewerAPIError(f"Could not reach API at {base}: {exc}") from exc
    if resp.status_code == 404:
        raise ViewerAPIError(
            f"Inline {inline} not found for survey '{survey_id}' (HTTP 404)"
        )
    if resp.status_code != 200:
        raise ViewerAPIError(
            f"Inline fetch failed: HTTP {resp.status_code}: {resp.text[:200]}"
        )
    payload = resp.json()
    if not payload or not payload.get("amplitude"):
        raise ViewerAPIError(f"Inline {inline} returned no amplitude data")
    return payload


# ---------------------------------------------------------------------------
# Storage browser
# ---------------------------------------------------------------------------


def browse(
    container: str,
    prefix: str = "",
    base_url: str | None = None,
    *,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """List one level of a storage container via ``GET /api/browse/{container}``.

    Returns the ``items`` list (folders + files at *prefix*).  Raises
    :class:`ViewerAPIError` on failure so the UI surfaces *why* a listing is
    empty rather than silently showing nothing.
    """
    import requests

    base = base_url or api_base_url()
    try:
        resp = requests.get(
            f"{base}/api/browse/{container}",
            params={"prefix": prefix},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise ViewerAPIError(f"Could not reach API at {base}: {exc}") from exc
    if resp.status_code != 200:
        raise ViewerAPIError(
            f"Browse '{container}/{prefix}' failed: "
            f"HTTP {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json() or {}
    return data.get("items", [])



def start_fault_detection(
    survey_id: str,
    checkpoint_blob: str = "checkpoints/unet3d_best.pt",
    base_url: str | None = None,
    *,
    threshold: float = 0.5,
    inline_center: int | None = None,
    inline_window: int = 32,
    timeout: float = 30.0,
) -> str:
    """POST ``/api/interpretation/fault-detection`` and return the ``run_id``.

    When ``inline_center`` is given, the API bounds inference to a +/-
    ``inline_window`` inline slab (issue #19) so it fits the web container.
    """
    import requests

    base = base_url or api_base_url()
    payload: dict[str, Any] = {
        "survey_id": survey_id,
        "checkpoint_blob": checkpoint_blob,
        "threshold": threshold,
        "inline_window": inline_window,
    }
    if inline_center is not None:
        payload["inline_center"] = inline_center
    try:
        resp = requests.post(
            f"{base}/api/interpretation/fault-detection", json=payload, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        raise ViewerAPIError(f"Could not reach API at {base}: {exc}") from exc
    if resp.status_code not in (200, 202):
        raise ViewerAPIError(
            f"Fault detection request failed: HTTP {resp.status_code}: {resp.text[:200]}"
        )
    run_id = (resp.json() or {}).get("run_id")
    if not run_id:
        raise ViewerAPIError("Fault detection response did not include a run_id")
    return run_id


def poll_status(
    run_id: str, base_url: str | None = None, *, timeout: float = 15.0
) -> dict[str, Any]:
    """GET ``/api/interpretation/{run_id}/status``.  Returns the JSON dict."""
    import requests

    base = base_url or api_base_url()
    try:
        resp = requests.get(
            f"{base}/api/interpretation/{run_id}/status", timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        raise ViewerAPIError(f"Could not reach API at {base}: {exc}") from exc
    if resp.status_code != 200:
        raise ViewerAPIError(
            f"Status poll failed: HTTP {resp.status_code}: {resp.text[:200]}"
        )
    return resp.json() or {}


def fetch_overlay(
    run_id: str,
    inline_number: int,
    base_url: str | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """GET ``/api/interpretation/{run_id}/overlay/{inline_number}``.

    ``inline_number`` is the **absolute** survey inline (e.g. 9961-10361).
    The API maps it to the result volume's local index via the run manifest,
    so bounded subvolume runs resolve correctly.
    """
    import requests

    base = base_url or api_base_url()
    try:
        resp = requests.get(
            f"{base}/api/interpretation/{run_id}/overlay/{inline_number}",
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise ViewerAPIError(f"Could not reach API at {base}: {exc}") from exc
    if resp.status_code != 200:
        raise ViewerAPIError(
            f"Overlay fetch failed: HTTP {resp.status_code}: {resp.text[:200]}"
        )
    payload = resp.json()
    if not payload or payload.get("fault_probability") is None:
        raise ViewerAPIError("Overlay returned no fault_probability data")
    return payload
