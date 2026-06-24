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

### 2026-06-24 — Phase 1 Real-Data Viewer Tests

**Session:** Added `src/tests/test_viewer/test_viewer.py` to cover Dallas's real-data viewer implementation (29 new tests; suite: 131 passed, 5 skipped).

**What was tested:**

1. **Amplitude reader** (7 tests) — verified `_get_amplitude_slice` logic against real `data/volve/staged/synthetic.zarr` (100×200×500 float32). Pinned: shape (200, 500), index 0 = inline 1001, index 99 = inline 1100, out-of-range clamping (both directions), no NaN/Inf.

2. **Fault prob reader** (5 tests) — verified `_get_fault_prob_slice` logic against `data/volve/staged/fault_prob.zarr`. Pinned: shape (200, 500), all values ∈ [0, 1], no NaN/Inf. **Missing-bake path** confirmed to return `None` not raise.

3. **Fault-stick coordinate mapping** (8 tests — highest value) — regression guard for z_ms = sample index bug. Pinned:
   - Main fault: TWT 808–908 ms (z_samples 202–227 × 4.0 ms)
   - Antithetic fault: TWT 1200–1228 ms (z_samples 300–307 × 4.0 ms)
   - First row exact mapping: dat(45, 84, 202) → abs_il=1046, abs_xl=1984, twt=808.0 ms
   - Inline 0-based → absolute: +1001; Crossline 0-based → absolute: +1900
   - Any regression re-introducing raw-z-as-ms (values 202–307 ms) will fail loudly on `>=800 ms` guard.

4. **`_write_zarr_volume` zarr v3 roundtrip** (5 tests) — float32 and uint8, shape/dtype/value exact match. `overwrite=False` raises `FileExistsError` (zarr v3 builtins.FileExistsError on mode="w-"). `overwrite=True` replaces data cleanly. Custom chunk shape preserved in metadata.

5. **Viewer module regression guard** (4 tests) — AST-level: `streamlit_app.py` parses without SyntaxError; all 5 key functions present; array names `amplitude` and `fault_probability` confirmed in source.

**Coupling / testability finding (flagged for Dallas):**
- `_get_amplitude_slice`, `_get_fault_prob_slice`, and `_load_fault_sticks` are decorated with `@st.cache_data` and `streamlit_app.py` runs top-level Streamlit calls + sidebar rendering at import time. The module is un-importable in pytest without a comprehensive Streamlit mock (which fights the module's own side-effects). Tests replicate the reader logic directly. **Recommend Dallas extract reader logic into `deepseismic/ui/_data_readers.py`** (no Streamlit dependency) so they can be tested in isolation.

**Marker decision:** Tests use real local zarr files (not Azurite/Azure/GPU). Per project convention, `@pytest.mark.integration` is reserved for infrastructure-dependent tests. These tests are **unmarked** and run in standard CI.

**zarr v3 note:** `zarr.open_group(store, mode="w-")` raises `builtins.FileExistsError` (not a zarr-specific type) when the store already contains data.

**Suite delta:** 102→131 passed (+29), 5 skipped unchanged, ruff clean.

## Scribe Cross-Agent Update — 2026-06-10T04:30-05:00
Sprint 1 coordination complete. All agents delivered successfully.
- 5 agents synchronized
- 7 decision documents archived
- Full team context available in decisions.md
