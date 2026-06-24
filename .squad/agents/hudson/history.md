# Hudson — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, pytest, numpy testing utilities
- **Goal:** Ensure data pipelines, models, and AI integrations work correctly
- **Key concerns:** Floating point precision in seismic data, SEG-Y parsing edge cases, model input shape validation
- **User:** jospaid

## Learnings

### 2026-06-24 — CI-Safe Viewer Tests (PR #3 Fix)

**Session:** Fixed 11 CI failures + 7 errors caused by gitignored data artifacts absent on CI runner.

**CI-vs-local data-availability gotcha:**
Tests that read from `data/volve/staged/` or `data/volve/interpretations/fault_sticks/` pass locally because the files exist on disk but fail in CI because they are gitignored (large binaries). The test suite must never hard-depend on gitignored artifacts for tests that run in standard CI mode.

**Fixture-synthesis pattern for the coordinate guard:**
`TestFaultStickCoordinateMapping` — the highest-value regression guard (z-as-sample-index bug) — was refactored to use a synthesized `.dat` fixture written to a `tmp_path_factory` temp dir. Three rows per fault file are sufficient to cover the full pinned coordinate ranges (min/max inline, crossline, z_samp). The regression math (`z_samp × 4.0 == twt_ms`) is fully proven on synthetic data. Pattern: parameterize the helper with `sticks_dir: Path = _STICKS_DIR` and inject `synth_sticks_dir` from the test fixture.

**skipif-on-missing-path pattern:**
For tests that genuinely need real baked zarr volumes (`TestAmplitudeReader`, `TestFaultProbReader`), use class-level `@pytest.mark.skipif(not _PATH.exists(), reason="...")`. This: (a) is self-documenting about why CI skips, (b) still runs locally where data exists, (c) fixes the inverted-guard bug where tests asserted `arr is not None` (saying "file exists but reader returned None") when in CI the file simply didn't exist — with skipif, those classes only execute when the file is confirmed present.

**No filename mismatch found:**
`bake_demo_faults.py`, `streamlit_app.py`, and `test_viewer.py` all consistently reference `fault_prob.zarr` / `fault_probability` array. The task brief's mention of `synthetic_fault_prob.zarr` was a red herring — not present anywhere in the codebase.

**Verification results (2026-06-24):**
- `pytest src/tests/test_viewer/ -m "not integration" -v`: 29 passed, 0 failed, 0 errors ✓
- `pytest -m "not integration" -q`: 129 passed, 2 skipped, 5 deselected, 0 failures ✓
- `ruff check src/`: All checks passed ✓
- Pushed: `dab69c8` on `feat/real-fault-viewer`

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

### 2026-06-24 — PR #3 CI Fix and Merge

**Session:** Hudson fixed 11 CI failures + 7 errors in PR #3 viewer tests. Coordinator merged feat/real-fault-viewer to main.

**CI remediation:** Two-tier strategy — (1) Synthesized minimal `.dat` fixture for critical coordinate-mapping regression guard; 8 tests now run in CI with zero data dependencies. (2) Added `@pytest.mark.skipif(path_missing)` to real-artifact reader tests; skip silently in CI, pass locally. Fixed inverted-guard bug where tests asserted file-exists but reader-returned-None when file simply wasn't present.

**Validation:** All 29 viewer tests pass locally; 8 coordinate guards pass in CI; 21 reader tests skip gracefully in CI when artifacts absent. `ruff check src/` clean.

**Merge:** Commit 776400a (squash). PR #3 ready for Phase 2 development.

## Scribe Cross-Agent Update — 2026-06-10T04:30-05:00
Sprint 1 coordination complete. All agents delivered successfully.
- 5 agents synchronized
- 7 decision documents archived
- Full team context available in decisions.md
