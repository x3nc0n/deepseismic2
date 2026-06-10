"""FastAPI dependency injectors for deepseismic2.

Import and annotate route parameters with Depends() to inject the storage
client and settings without repeating construction logic in every handler.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from deepseismic.storage.blob_client import StorageClient
from deepseismic.storage.config import Settings, get_settings


def is_mock_mode() -> bool:
    """Return True when DEEPSEISMIC_MOCK_MODE is set to a truthy value."""
    return os.getenv("DEEPSEISMIC_MOCK_MODE", "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def _build_storage_client() -> StorageClient | None:
    """Build one StorageClient per process. Returns None in mock mode or on error."""
    if is_mock_mode():
        return None
    try:
        return StorageClient()
    except Exception:
        return None


def get_storage_client() -> StorageClient | None:
    """FastAPI dependency: the process-level StorageClient, or None in mock mode."""
    return _build_storage_client()


def get_settings_dep() -> Settings:
    """FastAPI dependency: cached Settings singleton."""
    return get_settings()


# Annotated type aliases for use in route signatures
StorageClientDep = Annotated[StorageClient | None, Depends(get_storage_client)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
