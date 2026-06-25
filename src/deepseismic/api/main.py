"""FastAPI application factory for deepseismic2.

Start the API server::

    uvicorn deepseismic.api.main:app --reload

In mock mode (no real storage required)::

    DEEPSEISMIC_MOCK_MODE=true uvicorn deepseismic.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from deepseismic.api.dependencies import _build_storage_client, is_mock_mode
from deepseismic.api.routes import browse, interpretation, surveys, wells

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: initialise shared resources on startup, log on shutdown."""
    if is_mock_mode():
        logger.info(
            "deepseismic2 API starting in MOCK MODE — no real storage required"
        )
    else:
        try:
            client = _build_storage_client()
            if client is not None:
                try:
                    client.ensure_containers()
                    logger.info("Storage containers verified / created")
                except Exception as exc:
                    logger.warning("Could not verify storage containers: %s", exc)
        except Exception as exc:
            logger.error(
                "Storage client failed to initialise — API will return 503 on "
                "storage-dependent endpoints until the configuration is fixed: %s",
                exc,
            )

    yield

    logger.info("deepseismic2 API shutting down")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title="deepseismic2",
        description=(
            "Cloud-native seismic interpretation PoC — "
            "storage, ML pipeline, and Foundry agent integration layer."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for Streamlit (8501) and Gradio (7860) local dev servers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://localhost:7860",
            "http://127.0.0.1:8501",
            "http://127.0.0.1:7860",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(surveys.router, prefix="/api")
    app.include_router(interpretation.router, prefix="/api")
    app.include_router(wells.router, prefix="/api")
    app.include_router(browse.router, prefix="/api")

    return app


app = create_app()


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["ops"])
@app.get("/api/health", tags=["ops"], include_in_schema=False)
def health() -> dict[str, Any]:
    """Liveness + readiness probe.

    ``status`` is always ``"ok"`` when the process is alive (liveness).
    ``storage`` reports real storage reachability (readiness):
      - ``"mock"``        — explicit mock mode, no real storage needed
      - ``"ok"``          — real mode, storage pinged successfully
      - ``"unreachable"`` — real mode, client built but storage not responding
      - ``"error"``       — real mode, client could not be built (config error)

    Used by Docker HEALTHCHECK, load balancers, and post-deploy infra checks.
    """
    mock = is_mock_mode()
    if mock:
        return {"status": "ok", "mock_mode": True, "storage": "mock"}

    # Try to obtain the storage client (raises if misconfigured)
    try:
        client = _build_storage_client()
    except Exception as exc:
        return {
            "status": "ok",  # process is alive
            "mock_mode": False,
            "storage": "error",
            "storage_error": str(exc),
        }

    # Lightweight reachability check — list up to 1 blob from catalog
    try:
        client.list_blobs("catalog", max_results=1)
        storage_status = "ok"
        storage_error = None
    except Exception as exc:
        storage_status = "unreachable"
        storage_error = str(exc)

    result: dict[str, Any] = {
        "status": "ok",
        "mock_mode": False,
        "storage": storage_status,
    }
    if storage_error:
        result["storage_error"] = storage_error
    return result
