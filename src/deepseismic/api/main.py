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

from deepseismic.api.dependencies import _build_storage_client, is_mock_mode
from deepseismic.api.routes import interpretation, surveys, wells

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: initialise shared resources on startup, log on shutdown."""
    if is_mock_mode():
        logger.info(
            "deepseismic2 API starting in MOCK MODE — no real storage required"
        )
    else:
        client = _build_storage_client()
        if client is not None:
            try:
                client.ensure_containers()
                logger.info("Storage containers verified / created")
            except Exception as exc:
                logger.warning("Could not verify storage containers: %s", exc)
        else:
            logger.warning("Storage client unavailable — running in degraded mode")

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

    return app


app = create_app()


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Liveness probe used by Docker HEALTHCHECK and load balancers."""
    mock = is_mock_mode()
    storage_up = not mock and _build_storage_client() is not None
    return {
        "status": "ok",
        "mock_mode": mock,
        "storage": "mock" if mock else ("ok" if storage_up else "unavailable"),
    }
