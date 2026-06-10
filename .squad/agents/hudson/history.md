# Hudson — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, pytest, numpy testing utilities
- **Goal:** Ensure data pipelines, models, and AI integrations work correctly
- **Key concerns:** Floating point precision in seismic data, SEG-Y parsing edge cases, model input shape validation
- **User:** jospaid

## Learnings

### 2026-06-09 — Test Harness Setup

**Session:** Wrote full smoke test suite for all PoC modules.

**Fixture design (`src/tests/conftest.py`):**
- `sample_segy_path` is session-scoped (slow to create) using `tmp_path_factory`; use `segyio.SegySampleFormat.IBM_FLOAT_4_BYTE` not `.FLOAT`.
- `tmp_zarr_store` is function-scoped `zarr.storage.LocalStore` (zarr v3 changed from `DirectoryStore`).
- `azurite_client` uses try/except + `pytest.skip()` for graceful fallback — do NOT use pytest.mark.skipif at fixture level.
- `mock_storage_client` and `mock_llm_response` are function-scoped MagicMocks.

**Key zarr v3 API changes:**
- `zarr.DirectoryStore` is gone → use `zarr.storage.LocalStore(path_str)`
- `root.create_dataset(name, data=arr)` is gone → use `root.create_array(name, shape=..., dtype=...)` then `arr[:] = data`

**UNet3D API:**
- Constructor takes `config: UNetConfig | None`, not positional `in_channels/out_channels/depth`.
- Use `_make_model(depth=3, init_features=8)` helper that wraps `UNetConfig` — keeps tests readable.
- `model.save_checkpoint(path)` / `UNet3D.load_checkpoint(path)` are the canonical checkpoint methods.
- `model.parameter_count()` returns `{"total": N, "trainable": N}`.

**Mock pattern for stub modules:**
- `patch.object(_mod, "func_name", return_value=..., create=True)` — use `create=True` when the attribute doesn't exist in the stub yet; omit `create=True` once the function is implemented.
- Real-implementation tests that exercise modules still in dev: mark with `@pytest.mark.integration` to exclude from default CI.

**Bug caught by tests:**
- `segy_loader.load_segy()` fails on synthetic SEG-Y with `f.gather[il_no]` — `numpy.intc` not subscriptable as gather key. Flagged via `@pytest.mark.integration` test.

**Test file locations:**
- `src/tests/test_ingest/test_ingest.py` — ingest pipeline (5 test classes)
- `src/tests/test_models/test_models.py` — UNet3D shape/checkpoint/parity (5 test classes)
- `src/tests/test_storage.py` — blob client + zarr roundtrip (5 test classes)
- `src/tests/test_agent.py` — agent tool registration + mock LLM (5 test classes)
- `src/tests/test_api/test_api.py` — HTTP API contract (3 test classes, stand-in app)
- `.github/workflows/ci.yml` — ruff + pytest (no integration) + coverage artifact upload

**CI status:** 77 passed, 2 skipped (CUDA, extract_metadata stub), 5 deselected (integration) — **green**.
