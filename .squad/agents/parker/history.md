# Parker — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, Azure, Docker, REST APIs
- **Goal:** Replace expensive storage (Isilon, Premium Files, NetApp Files) with affordable cloud-native patterns
- **Challenge:** Seismic data is large (GB-TB), traditionally needs fast random I/O — find cheaper alternatives
- **User:** jospaid

## Learnings

### 2026-06-09 — Local dev environment and storage abstraction layer

**What was built:**
- `src/deepseismic/storage/blob_client.py` — Full `StorageClient` + `ABSZarrStore`
- `src/deepseismic/storage/config.py` — `pydantic-settings` `Settings` singleton
- `docker/docker-compose.yml` — Azurite service (default) + API service (--profile full)
- `docker/Dockerfile` — Multi-stage build (builder + least-privilege runtime)
- `scripts/setup-local.ps1` — One-shot Windows setup: Docker check → Azurite → containers → sample data
- `.env.example` — All env vars with local-dev defaults pre-filled
- `src/deepseismic/api/main.py` — Minimal FastAPI app with `/health` endpoint

**Architecture patterns:**
- `StorageClient` auto-detects: `STORAGE_CONNECTION_STRING` → connection string; `AZURE_STORAGE_ACCOUNT` → `DefaultAzureCredential`.  No other code path needed.
- `ABSZarrStore` is a `MutableMapping` wrapping `ContainerClient`.  Works with zarr 2.x and 3.x without `adlfs`/`fsspec[azure]` — avoids an extra dependency.
- `get_settings()` is `@lru_cache(maxsize=1)` — singleton, test-friendly via `get_settings.cache_clear()`.
- Docker compose `azurite` service has a healthcheck; `api` service depends on it via `condition: service_healthy`.
- API service is behind `--profile full` so `docker compose up` only starts Azurite by default (cheaper).

**Key file paths:**
- Storage client: `src/deepseismic/storage/blob_client.py`
- Config/settings: `src/deepseismic/storage/config.py`
- Compose file: `docker/docker-compose.yml`
- Dockerfile: `docker/Dockerfile`
- Setup script: `scripts/setup-local.ps1`
- Env template: `.env.example`

**Dependency added:** `pydantic-settings>=2.3.0` — required for `BaseSettings` in pydantic v2.

**Cost notes:**
- Azurite connection string defaults mean zero Azure spend during local dev.
- Standard LRS chosen for cloud (cheapest redundancy tier for PoC).
- `list_blobs` is a metadata-only call — no data egress cost.


## Scribe Cross-Agent Update — 2026-06-10T04:30-05:00
Sprint 1 coordination complete. All agents delivered successfully.
- 5 agents synchronized
- 7 decision documents archived
- Full team context available in decisions.md
