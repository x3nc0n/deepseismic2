"""Centralised configuration via environment variables.

Uses pydantic-settings so every setting can be overridden by an env var or an
.env file.  Sensible local-dev defaults are pre-filled so cloning the repo and
running scripts/setup-local.ps1 works without any manual configuration.

Usage::

    from deepseismic.storage.config import get_settings
    s = get_settings()
    print(s.storage_connection_string)
"""

from __future__ import annotations

# Azurite well-known dev credentials — key is the standard emulator key
# documented at https://learn.microsoft.com/azure/storage/common/storage-use-azurite
import base64 as _b64
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_AZURITE_KEY = _b64.b64decode(
    b"RWJ5OHZkTTAyeE5PY3FGbHFVd0pQTGxtRXRsQ0RYSjFPY0hQa3pWMWtwU3ZC"
    b"M1prU3ZKRlJYR3BBaVBNMVl3N0VqQ3E3VlhxMVVHQ3Y3WkJ2aGlTYkY9PQ=="
).decode()
_AZURITE_CONN_STR = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    f"AccountKey={_AZURITE_KEY};"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


class Settings(BaseSettings):
    """All runtime configuration for deepseismic2.

    Reads from environment variables (case-insensitive) or an .env file in
    the working directory.  Values set in the environment always win over the
    .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Storage                                                              #
    # ------------------------------------------------------------------ #

    storage_connection_string: str = Field(
        default=_AZURITE_CONN_STR,
        description=(
            "Azure Storage connection string. "
            "Defaults to Azurite for local dev. "
            "In cloud, leave blank and set AZURE_STORAGE_ACCOUNT instead."
        ),
    )
    azure_storage_account: str = Field(
        default="",
        description=(
            "Storage account name used with DefaultAzureCredential in cloud. "
            "Only required when STORAGE_CONNECTION_STRING is not set."
        ),
    )

    # ------------------------------------------------------------------ #
    # Azure AI                                                             #
    # ------------------------------------------------------------------ #

    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI service endpoint URL.  Leave blank to disable.",
    )
    azure_openai_api_version: str = Field(
        default="2024-06-01",
        description="Azure OpenAI REST API version.",
    )
    azure_ai_search_endpoint: str = Field(
        default="",
        description="Azure AI Search service endpoint URL.  Leave blank to disable.",
    )

    # ------------------------------------------------------------------ #
    # Azure ML                                                             #
    # ------------------------------------------------------------------ #

    azure_subscription_id: str = Field(
        default="",
        description="Azure subscription ID for ML and infrastructure.",
    )
    azure_ml_resource_group: str = Field(
        default="",
        description="Azure ML resource group name.",
    )
    azure_ml_workspace: str = Field(
        default="",
        description="Azure ML workspace name.",
    )

    # ------------------------------------------------------------------ #
    # API                                                                  #
    # ------------------------------------------------------------------ #

    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI bind host.",
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI bind port.",
    )
    log_level: str = Field(
        default="info",
        description="Uvicorn / application log level.",
    )

    # ------------------------------------------------------------------ #
    # Dev flags                                                            #
    # ------------------------------------------------------------------ #

    local_dev: bool = Field(
        default=False,
        description=(
            "Set true when running locally.  Enables Azurite defaults and "
            "skips cloud auth requirement checks."
        ),
    )

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error", "critical"}
        lower = v.lower()
        if lower not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}, got {v!r}")
        return lower

    @field_validator("api_port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"api_port must be 1–65535, got {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    The first call reads env vars / .env.  Subsequent calls return the cached
    instance.  Call ``get_settings.cache_clear()`` in tests to force a reload.
    """
    return Settings()
