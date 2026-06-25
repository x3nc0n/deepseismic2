"""FastAPI dependency injectors for deepseismic2.

Import and annotate route parameters with Depends() to inject the storage
client and settings without repeating construction logic in every handler.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException

from deepseismic.storage.blob_client import StorageClient
from deepseismic.storage.config import Settings, get_settings

logger = logging.getLogger(__name__)


def is_mock_mode() -> bool:
    """Return True when DEEPSEISMIC_MOCK_MODE is set to a truthy value."""
    return os.getenv("DEEPSEISMIC_MOCK_MODE", "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def _build_storage_client() -> StorageClient | None:
    """Build one StorageClient per process.

    Returns None only in explicit mock mode.  In real mode, raises on any
    configuration error so misconfigured deployments fail loud instead of
    silently falling back to mock behaviour.
    """
    if is_mock_mode():
        return None
    try:
        return StorageClient()
    except Exception as exc:
        logger.error(
            "StorageClient initialisation failed in real mode: %s — "
            "set STORAGE_CONNECTION_STRING (local/Azurite) or "
            "AZURE_STORAGE_ACCOUNT (cloud). "
            "Use DEEPSEISMIC_MOCK_MODE=true for offline dev without storage.",
            exc,
        )
        raise


def get_storage_client() -> StorageClient | None:
    """FastAPI dependency: the process-level StorageClient, or None in mock mode.

    Raises HTTP 503 in real mode when the storage client cannot be built, so
    callers never receive a None storage in real mode — they get a proper error.
    """
    if is_mock_mode():
        return None
    try:
        return _build_storage_client()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Storage unavailable: {exc}. "
                "Check STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT."
            ),
        ) from exc


def get_settings_dep() -> Settings:
    """FastAPI dependency: cached Settings singleton."""
    return get_settings()


# Annotated type aliases for use in route signatures
StorageClientDep = Annotated[StorageClient | None, Depends(get_storage_client)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
