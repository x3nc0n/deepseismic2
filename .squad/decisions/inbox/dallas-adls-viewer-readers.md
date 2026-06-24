# Decision: ADLS Viewer Readers — Option B Implementation

**Date:** 2026-06-24T14:25:19-05:00
**Author:** Dallas (Data/ML Engineer)
**Status:** Implemented — branch `feat/adls-viewer-readers`, pending Hudson CI + PR

## Context

Phase 1 (PR #3) wired the Streamlit viewer to read amplitude + baked fault Zarr from
**local file paths**.  For the hosted Azure Container Apps demo, those artifacts live
in ADLS Gen2.  Infra issue Spava-Corp/deepseismic2-infra#8 chose **Option B**: the
app reads artifacts **directly from ADLS** (no sidecar download, no volume mount).

## Decisions

### 1. Reader extraction into `_data_readers.py`

All pure data-access logic extracted from `streamlit_app.py` into
`src/deepseismic/ui/_data_readers.py` — no Streamlit imports, no `@st.cache_data`,
no sidebar side-effects.  `streamlit_app.py` now contains thin `@st.cache_data`
wrappers that delegate to the pure functions.  This lets Hudson write proper unit
tests without mocking Streamlit.

### 2. Backend env-var contract (relay verbatim to infra issue #8)

```
# Backend selector
DEEPSEISMIC_DATA_BACKEND         local | azure   (default: local)

# Local backend — optional base dir override
DEEPSEISMIC_DATA_DIR             path to volve data dir (default: data/volve in repo)

# Azure backend — artifact locations
DEEPSEISMIC_AMP_CONTAINER        default: staged
DEEPSEISMIC_AMP_PREFIX           default: volve/synthetic.zarr
DEEPSEISMIC_FAULT_PROB_CONTAINER default: results
DEEPSEISMIC_FAULT_PROB_PREFIX    default: volve/fault_prob.zarr
DEEPSEISMIC_FAULT_MASK_CONTAINER default: results
DEEPSEISMIC_FAULT_MASK_PREFIX    default: volve/fault_mask.zarr
DEEPSEISMIC_STICKS_CONTAINER     default: raw
DEEPSEISMIC_STICKS_PREFIX        default: volve/interpretations/fault_sticks

# StorageClient auth (existing convention, unchanged)
STORAGE_CONNECTION_STRING        Azurite or real account connection string
AZURE_STORAGE_ACCOUNT            Account name (uses DefaultAzureCredential in cloud)
STORAGE_ACCOUNT_NAME             Alias for AZURE_STORAGE_ACCOUNT
```

Container defaults follow the architecture decision:
- `staged` for amplitude (chunked seismic volume)
- `results` for fault prob/mask (model inference outputs)
- `raw` for fault sticks (original interpretation supporting files)

### 3. zarr v3 store compatibility fix

`ABSZarrStore` (a `MutableMapping`) is **incompatible with zarr v3**.
`zarr.open_group(store=MutableMapping)` raises `TypeError: Unsupported type for
store_like`.  Fixed by adding `ABSZarrV3Store(zarr.abc.store.Store)` — a proper
zarr v3 async Store subclass — to `blob_client.py`.

Key design choices:
- Blocking Azure SDK calls dispatched via `asyncio.to_thread` (zarr v3 is async).
- `get()` wraps raw bytes as `prototype.buffer.from_bytes(raw)`.
- `with_read_only()` implemented (required by zarr for `mode="r"`).
- `ABSZarrStore` (MutableMapping) retained for backward compat.
- `upload_zarr_store` rewritten to walk local directory and upload files directly
  (avoids zarr.copy_store cross-version dependency).
- `open_zarr_store` now returns `ABSZarrV3Store`.

### 4. Fault sticks in azure backend

`.dat` files are small text blobs.  The azure reader calls
`list_blobs(container, prefix)` → `download_blob(container, name)` for each `.dat`,
then parses bytes with the **unchanged canonical coordinate mapping**:
```
abs_inline    = 1001 + il_idx
abs_crossline = 1900 + xl_idx
twt_ms        = z_sample * 4.0
```
Failure (missing container/prefix) returns `{}` gracefully — viewer omits sticks
rather than crashing.

### 5. Graceful degradation preserved

Both backends: if fault_prob artifact is absent → `get_fault_prob_slice()` returns
`None` → viewer renders amplitude-only with a warning.

## Validation

- `ruff check src/ scripts/` → clean
- `python -m pytest -m "not integration" -q` → 129 passed, 2 skipped
- `python -m py_compile src/deepseismic/ui/_data_readers.py src/deepseismic/ui/streamlit_app.py` → OK
- Azure read path proved with dict-backed mock ContainerClient (Azurite not running
  in this environment): write 10×20×50 float32 volume to mock ABS, read back via
  `zarr.open_group(ABSZarrV3Store, mode='r')`, all allclose assertions passed.
  `_data_readers.get_volume_coords()` and `get_amplitude_slice()` with azure backend
  both passed.

## Files Changed

| File | Change |
|------|--------|
| `src/deepseismic/ui/_data_readers.py` | **New** — pure backend-aware data readers |
| `src/deepseismic/ui/streamlit_app.py` | Thin `@st.cache_data` wrappers; imports from `_data_readers` |
| `src/deepseismic/storage/blob_client.py` | Added `ABSZarrV3Store`, updated `upload_zarr_store` + `open_zarr_store` |
| `src/tests/test_viewer/test_viewer.py` | Updated array-name string guards to also check `_data_readers.py` |

## Notes for Hudson

- `_data_readers.py` is now directly importable in pytest without any Streamlit mock.
- All four pure reader functions (`get_volume_coords`, `get_amplitude_slice`,
  `get_fault_prob_slice`, `load_fault_sticks`) are testable by setting
  `DEEPSEISMIC_DATA_BACKEND=azure` and patching `StorageClient`.
- Add azure-backend tests in `src/tests/test_viewer/` or a new
  `test_data_readers.py` targeting `_data_readers` directly.
