# Hudson — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, pytest, numpy testing utilities
- **Goal:** Ensure data pipelines, models, and AI integrations work correctly
- **Key concerns:** Floating point precision in seismic data, SEG-Y parsing edge cases, model input shape validation
- **User:** jospaid

## Learnings

### 2026-06-24 — Sprint 2 S2-07 Test Coverage

**Session:** Implemented Sprint 2 item S2-07 — comprehensive test suite for all Sprint 2 deliverables.

**New test files (53 new tests across 4 files + 1 new test_training/ directory):**

1. `src/tests/test_ingest/test_sprint2_label.py` (17 tests) — S2-01 label generation:
   - `TestCoordinateMapping` (7 tests): pins abs_inline=1001+il_idx, abs_xl=1900+xl_idx, twt_ms=z_col×4.0; critical regression guard that z_col≈200 maps to ≥800 ms (not raw 202 ms); `load_volve_fault_sticks` parses synthetic .dat fixtures correctly.
   - `TestFaultMaskGenerator` (4 tests): initial zeros, binary values, positive fraction > 0, dilation monotone.
   - `TestLabelZarrOutput` (3 tests): dtype=uint8, shape matches volume_shape, values roundtrip exactly via zarr v3 LocalStore.

2. `src/tests/test_preprocessing/test_sprint2_pipeline.py` (17 tests) — S2-06 QC pipeline:
   - `TestComputeVolumeQC` (5 tests): all required keys present, shape/dtype correct, stat ordering sane (min≤p01≤p99≤max), sidecar_stats_used=False without sidecar.
   - `TestDominantFrequency` (3 tests): 30 Hz and 50 Hz sinusoids recovered within ±3/4 Hz; Ricker 35 Hz within ±5 Hz.
   - `TestAutocorrSymmetry` (2 tests): symmetric Gaussian → ratio ≈ 1.0; asymmetric → `xfail` (see BUG below).
   - `TestGlobalAmplitudeNormalize` (7 tests): no mutation, ratio preserved, p99 scales to 1.0, ValueError for p99≤0, clip bounds enforced, clip=False allows extremes.

3. `src/tests/test_training/test_sprint2_training.py` (14 tests) — S2-02/05/08:
   - `TestTrainConfig` (3 tests): seed=42 default, data_mode="synthetic" default, zarr mode accepted.
   - `TestAccumTpFpFn` (4 tests): exact confusion counts from logit tensors (TP=4,FP=2,FN=3 hand-crafted).
   - `TestEpochMetrics` (5 tests): exact IoU/Dice/Precision/Recall from TP=5,FP=2,FN=3 (formulas pinned against per-batch averaging regression).
   - `TestSeedDeterminism` (2 tests): same seed=42 via explicit `torch.Generator` → identical first batches; different seeds → different batches.

4. `src/tests/test_validation/test_sprint2_eval.py` (17 tests) — S2-03:
   - `TestComputeBinaryMetricsExact` (3 tests): exact IoU/Dice/Precision/Recall from hand-constructed confusion matrices.
   - `TestEvaluateModelSchema` (5 tests): ValidationMetrics instance, all required fields, volume_shape, metrics in [0,1], summary string tokens.
   - `TestRunEvaluationJSONSchema` (2 tests, `@pytest.mark.integration`): tiny zarr+checkpoint end-to-end JSON schema and serialisability.

**Seed determinism fix:** Use explicit `torch.Generator` passed to `DataLoader(generator=...)` rather than `torch.manual_seed` on the global RNG; global-seed approach was fragile because RNG state differed between loader creation and first iteration.

**BUG FOUND — `_autocorr_symmetry` (S2-06):**
`pipeline._autocorr_symmetry` is mathematically broken. The autocorrelation of any real signal satisfies `ac[centre+k] = ac[centre-k]` exactly, making `e_pos = e_neg` always — the function returns 1.0 regardless of phase. Documented with `@pytest.mark.xfail(strict=True)` and decision note `hudson-s2-07-tests.md`. Recommended fix: Hilbert-transform instantaneous phase, or spectral asymmetry proxy.

**Suite results (2026-06-24):**
- Full non-integration: `pytest -m "not integration" -q` → **209 passed**, 2 skipped, 8 deselected, 1 xfailed ✓
- Baseline was 156 passed → net +53
- `ruff check` on all 4 new files → All checks passed ✓
- Pre-existing `generate_fault_label.py` ruff issues (I001, E501) not introduced here.

### 2026-06-24 — ABSZarrV3Store + Azure/Local Resolver Tests (feat/adls-viewer-readers)

**Session:** Added `src/tests/test_viewer/test_data_readers.py` — 26 new CI-safe tests covering Dallas's ADLS Phase 2 work (ABSZarrV3Store, _data_readers backend resolver).

**Dict-backed mock ContainerClient pattern for testing ABSZarrV3Store without Azurite:**
Build `_MockContainerClient` with a plain `dict[str, bytes]` as backing store. `get_blob_client(key)` returns a `_MockBlobClient` that proxies reads/writes through the shared dict. `download_blob()` raises `azure.core.exceptions.ResourceNotFoundError` for missing keys (not `FileNotFoundError`) so `ABSZarrV3Store.get()` catches it and returns `None`, exactly matching real Azure SDK behaviour. `asyncio.to_thread(blob_client.download_blob().readall)` calls `download_blob()` synchronously before threading — the mock `_Downloader` object returned by `download_blob()` has a plain `readall()` method. This pattern proves zarr v3 async store correctness with zero network dependency.

**azure/local resolver testing via monkeypatch.setenv:**
Use `monkeypatch.setenv("DEEPSEISMIC_DATA_BACKEND", "azure")` to activate the azure path, then `monkeypatch.setattr(_data_readers, "_storage_client", lambda: mock_client)` to inject a `_MockStorageClient` instance whose `open_zarr_store(container, prefix)` returns `ABSZarrV3Store(shared_mock_container, prefix=prefix)`. Multiple calls to `_storage_client()` in the same request (e.g. `get_amplitude_slice` calling `get_volume_coords` internally) all receive the same mock instance and thus read consistent data from the shared dict. For fault sticks, the mock also implements `list_blobs(container, prefix)` and `download_blob(container, blob_name)` to serve `.dat` file bytes.

**Suite results (2026-06-24):**
- New: 26 tests pass, 1 deselected (`@pytest.mark.integration`)
- Full: `pytest -m "not integration" -q` → 155 passed, 2 skipped, 6 deselected ✓
- `ruff check src/` → All checks passed ✓
- Pushed: `18494f9` on `feat/adls-viewer-readers`

**No bugs found** in Dallas's `_data_readers.py` or `blob_client.py`.

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

### 2026-06-24 — Phase 2: ADLS Viewer Backend Test Coverage (PR #4)

**Session:** Hudson added comprehensive CI-safe test coverage for Dallas's Phase 2 ADLS viewer backend work (ABSZarrV3Store, backend resolver).

**Test file:** `src/tests/test_viewer/test_data_readers.py` — 26 new tests covering:
- `ABSZarrV3Store` round-trip (dict-backed mock ContainerClient, no Azurite/Azure SDK calls)
- Backend resolver (local vs. azure via DEEPSEISMIC_DATA_BACKEND env-var)
- Fault-stick coordinate mapping on both backends (regression guard: z_sample × 4.0 = twt_ms)
- Graceful degradation (missing containers/blobs return empty dicts, not errors)

**Dict-backed mock pattern:** Build `_MockContainerClient` with plain `dict[str, bytes]` backing store. `download_blob()` raises `azure.core.exceptions.ResourceNotFoundError` (not `FileNotFoundError`) so real zarr v3 store error-handling works identically. `asyncio.to_thread(blob_client.download_blob().readall)` pattern correctly deferred via shared mock instance — multiple calls in same request (e.g., `get_amplitude_slice` → `get_volume_coords` internally) read consistent data.

**CI-safe:** No Azurite, no real Azure calls, no gitignored data files. 1 integration test deselected (`@pytest.mark.integration`); 26 unit tests pass in CI.

**Result:** All tests pass; no bugs found in Dallas's code. (Three bugs in `ABSZarrV3Store` identified and fixed by review-storage + Dallas in parallel.)

**Commit:** 18494f9 on feat/adls-viewer-readers. Validation: `pytest -m "not integration" -q` → 155 passed, 2 skipped, 6 deselected ✓; `ruff check src/` → All checks passed ✓




