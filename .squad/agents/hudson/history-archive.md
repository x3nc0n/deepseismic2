# Hudson — History Archive

## Test Coverage Development (Sprints 1–2, archived 2026-06-25)

### Sprint 2 S2-07 Test Coverage (2026-06-24)
- test_sprint2_label.py (17 tests): Coordinate mapping, fault mask generation, Zarr output
- test_sprint2_pipeline.py (17 tests): Volume QC, dominant frequency, amplitude normalization
- test_sprint2_training.py (14 tests): Train config, metric accumulation, epoch metrics, seed determinism
- test_sprint2_eval.py (17 tests): Binary metrics, validation schema, evaluation JSON
- **Seed determinism fix:** Use explicit torch.Generator passed to DataLoader
- **BUG FOUND:** _autocorr_symmetry is mathematically broken (always returns 1.0)
- Suite: 209 passed (+53 from baseline), 2 skipped, 1 xfailed

### Phase 1 Real-Data Viewer Tests (2026-06-24)
- test_viewer.py (29 tests): Amplitude reader, fault prob reader, fault-stick mapping, zarr roundtrip
- **Fixture-synthesis pattern:** Synthesized .dat fixture for coordinate-mapping regression guard
- **skipif-on-missing-path pattern:** Skip reader tests in CI when gitignored data absent
- **CI fix for PR #3:** 11 CI failures + 7 errors resolved; 8 coordinate guards pass in CI
- Suite: 131 passed (+29), 0 failures

### Phase 2 ADLS Viewer Backend Tests (2026-06-24)
- test_data_readers.py (26 tests): ABSZarrV3Store roundtrip, backend resolver, fault sticks
- **Dict-backed mock pattern:** _MockContainerClient with plain dict backing store
- **azure/local resolver testing:** monkeypatch.setenv + _MockStorageClient injection
- **No bugs found** in Dallas's _data_readers.py or blob_client.py
- Suite: 155 passed, 2 skipped, 6 deselected

## Early Learnings

- Tests that read gitignored artifacts fail in CI; use synthesized fixtures or @pytest.mark.skipif
- zarr v3 mode="w-" raises builtins.FileExistsError when store exists; use mode="w" + overwrite=True
- asyncio.to_thread(blob_client.download_blob().readall) correctly defers blocking calls in async stores
- _build_storage_client is @lru_cache'd and must be patched in both dependencies and main imports

