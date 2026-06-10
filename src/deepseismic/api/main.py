"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="deepseismic2",
    description="Cloud-native seismic interpretation PoC",
    version="0.1.0",
)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by Docker HEALTHCHECK and load balancers."""
    return {"status": "ok"}
