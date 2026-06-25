

### Ripley Decision — DeepSeismic2 PoC Architecture

**Date:** 2026-06-09T21:56:47-05:00

**Decision:** Adopt a **cloud-native, object-storage-first PoC architecture** rather than a lift-and-shift filesystem design.

**Key Calls:**
1. **Storage:** Azure Blob Storage / ADLS Gen2 is the system of record for raw, staged, features, results, and catalog data.
2. **Raw format:** Keep SEG-Y as source truth; create cloud-friendly derived artifacts in Zarr plus JSON/Parquet metadata.
3. **Compute split:** CPU jobs handle ingest/preprocessing; GPU jobs handle model inference.
4. **GPU platform:** Prefer Azure Machine Learning managed compute for PoC speed.
5. **Backend:** Use a thin FastAPI service for metadata, run status, and results lookup.
6. **Model posture:** Use UNet as the first credible baseline; defer broader model bake-offs.
7. **LLM posture:** LLMs assist with summarization, metadata Q&A, and workflow guidance; CNNs remain responsible for seismic interpretation outputs.

**Why:** This gives the strongest modernization story with the least PoC risk:
- avoids coupling the design to expensive premium storage
- isolates GPU spend to short-lived inference windows
- keeps deterministic seismic processing in the ML stack
- uses LLMs where they improve analyst productivity without overselling them

**Scope Boundary:** For this PoC, do **not** build a full production platform, multi-user interpretation app, or full retraining system. Prove the architecture with a constrained Volve subset and one end-to-end workflow.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction

## Phase 1 — Real Fault Viewer Decisions (2026-06-24)


### 2026-06-09T22:50:47-05:00: User decisions — UI, local dev, labels

**By:** Joe Spaid (via Copilot)

**What:**
1. **Local dev enabled:** Developers can iterate without Azure. Use Azurite for storage emulation, mock LLM responses, and a small Volve data sample for fast iteration.
2. **Demo UI:** Primary interface is pure chat (Foundry agent). Additionally, provide basic Streamlit AND Gradio interfaces for showing visual screens to the geologist SME (John Spaid).
3. **ML label strategy:** Generate training labels from existing Volve fault interpretations first. Fall back to semi-supervised labeling if existing interpretations are insufficient.

**Why:** User request — final decisions before overnight build sprint.


## Inbox: dallas-ingest-unet-impl

# Dallas Decision — Ingest + UNet PoC Implementation

**Date:** 2026-06-09T22:50:47-05:00
**Author:** Dallas (Data/ML Engineer)
**Status:** Proposal — for team review

---

## Decision: Spatial train/val/test splits along the inline axis

**Context:**
When extracting overlapping 3D patches from a seismic volume, adjacent patches share
voxels. A random split would place voxels from the same physical location in both
train and test, causing data leakage and overstated evaluation metrics.

**Decision:**
Splits are defined by inline-axis ranges (70 / 15 / 15 %) and assigned by the
**centre** of each patch. Patches entirely within the train inline range go to
train, etc. No patch straddles a split boundary.

**Implication:**
- Evaluation metrics on val/test reflect true spatial generalisation.
- The model must generalise across ~30 % of the inline range it never saw during training.
- This is the correct approach for any spatially correlated gridded volume.

---

## Decision: Zarr chunk shape (64, 64, 128) as default

**Context:**
The seismic volume (ST10010 PSDM time, ~1 GB post-stack) has typical dimensions on
the order of ~1000 inlines × ~1000 crosslines × ~1000 samples.

**Decision:**
Default Zarr chunk shape is `(64, 64, 128)` — asymmetric on the sample axis.

**Rationale:**
- Training patches are 64³; a 64×64 tile in inline/crossline maps exactly to one
  chunk per crossline slice, minimising unnecessary chunk reads.
- The sample axis uses 128 because seismic is queried in contiguous runs along
  twtt during interpretation (horizon extraction, amplitude extraction).
- 64 × 64 × 128 × 4 bytes = 2 MB per chunk — reasonable for Azure Blob Storage
  sequential prefetch and local SSD cache.

---

## Decision: Fault-stick dilation default = 1 voxel (3×3×3 kernel)

**Context:**
Volve fault sticks from Petrel are spaced at roughly 25–100 m intervals along
the fault plane. At 12.5 m bins and 4 ms sample rate the inter-stick gap can span
several voxels, leaving unlabelled gaps between consecutive stick points.

**Decision:**
Default `dilation_voxels=1` (paints a 3×3×3 neighbourhood around each interpolated
point). The stick is also densely interpolated along its arc length at 2 samples/voxel
before painting, so gaps between stick points are filled.

**Implication:**
- Conservative label inflation — each fault voxel expands by one voxel in all
  directions. This is appropriate for training but users should document the
  dilation value in their label metadata.
- Increase to 2–3 for thicker faults or when stick spacing is coarse.

---

## No decision required on model architecture

The UNet (depth=4, init_features=32, ~19 M params) is the agreed baseline per the
team architecture decision (Ripley, 2026-06-09). No new model decision needed.


## Inbox: hudson-test-infrastructure-conventions

# Hudson Decision — Test Infrastructure Conventions

**Date:** 2026-06-09T22:50:47-05:00
**Author:** Hudson (QA)
**Status:** Proposed

## Decision

Adopt the following conventions for the deepseismic2 test suite:


### Zarr v3
- Project uses zarr 3.x. `zarr.DirectoryStore` is removed; use `zarr.storage.LocalStore(path_str)`.
- Array creation: `root.create_array(name, shape=..., dtype=...)` then `arr[:] = data`.

## Why
These conventions prevent CI from going red during parallel development while still ensuring bugs are caught when implementations are complete. The `integration` marker boundary is the key mechanism.


## Inbox: lambert-dual-ui-streamlit-gradio

# Lambert Decision — Dual UI Strategy (Streamlit + Gradio)

**Date:** 2026-06-09T22:50:47-05:00  
**Author:** Lambert  
**Scope:** Demo UI surfaces

---

## Decision

Implement **both** Streamlit and Gradio as distinct demo surfaces, with the
terminal chat as the primary development interface. Do not consolidate to one
framework.

## Rationale

1. **Different audiences:** Streamlit's two-panel layout (chat + matplotlib seismic
   viewer) suits geoscientist demos with a richer spatial context. Gradio's chatbot
   + image component is faster to stand up and more shareable via a public link.

2. **Shared agent:** Both apps import `DeepSeismicAgent` directly with the same
   streaming interface. Neither UI owns any agent logic — they are pure rendering
   surfaces. The dual-UI cost is low.

3. **Synthetic seismic as placeholder:** Both apps render a bandlimited synthetic
   inline section (scipy convolution + Ricker wavelet + reflectors) with a fault
   probability overlay. This makes the viewer credible to a geologist during demo
   without requiring live Zarr data. The swap to real data is a single function
   replacement in each app.

4. **Terminal chat for development:** The readline-based chat UI (`ui/chat.py`) is
   the lowest-friction interface for iterating on agent prompt and tool behaviour.
   Supports `/status`, `/interpret`, `/wells`, `/persona`, and `/state` shortcuts.

## Consequences

- `streamlit>=1.38.0` and `gradio>=4.40.0` added as `[ui]` optional dependencies
  in `pyproject.toml`.
- Both apps require `scipy` (already in core dependencies via seismic processing).
- `MOCK_LLM=true` works identically in all three surfaces — no live Azure required
  for UI development.

## Open questions

- Should the Streamlit app support side-by-side inline and crossline display?
  Deferred — single-inline viewer is sufficient for the PoC demo.
- Should QC overlay images from real runs replace the synthetic visualization?
  Yes — wire `get_inline_section` + blob Zarr read when live data is available.


## Inbox: parker-infra-scaffold

# Parker Infra Scaffold

- **Date:** 2026-06-09T22:41:18-05:00
- **Requested by:** jospaid
- **Decision:** Create a dedicated infrastructure repo at `Spava-Corp/deepseismic2-infra` for Azure Bicep and GitHub Actions CI/CD, separate from the application repo.
- **Why:** Keeps Azure credentials and deployment workflows isolated, makes teardown simpler, and preserves a clean two-repo split between product code and platform automation.
- **Cost posture:** Default everything to cheapest workable PoC tiers: Standard_LRS ADLS Gen2, ACR Basic, AI Search Basic with Free as an optional downgrade, Azure Container Apps consumption, and AML compute set to min 0 / max 1.
- **Operational posture:** Include one-click deploy, one-click destroy, and local helper scripts so the team can stand resources up fast and tear them back down when idle.


## Inbox: parker-zarr-store-abs-mappings

# Parker Decision — Zarr store uses custom MutableMapping, not adlfs

**Date:** 2026-06-09T22:50:47-05:00
**Author:** Parker
**Status:** Active

## Decision

`ABSZarrStore` in `src/deepseismic/storage/blob_client.py` is a hand-rolled
`MutableMapping` that wraps `azure-storage-blob`'s `ContainerClient`, rather
than using `adlfs` (Azure Data Lake Storage fsspec driver) or `fsspec[azure]`.

## Rationale

- `azure-storage-blob` is already a required dependency.
- `adlfs` would add one more package that pulls in `fsspec`, `aiohttp`, and
  async machinery we don't need for the PoC's synchronous batch jobs.
- A `MutableMapping` is zarr's documented store interface — zarr 2.x and 3.x
  both accept it, so this approach is forward-compatible.
- Works with Azurite locally without any extra configuration.

## Trade-offs

- No async I/O — each read/write is synchronous.  Acceptable for PoC-scale
  batch jobs; revisit if we need concurrent multi-part uploads.
- `zarr.copy_store` handles large array uploads correctly (key-by-key copy),
  but it is not parallelised.  Consider `adlfs` + `zarr.open(..., mode="w")`
  if upload performance becomes a bottleneck.

## Affected files

- `src/deepseismic/storage/blob_client.py` — `ABSZarrStore`, `upload_zarr_store`, `open_zarr_store`
- `pyproject.toml` — no new dependency added for zarr/fsspec


## Inbox: ripley-api-contract

# Decision: API Contract Design for deepseismic2

**Author:** Ripley (Lead/Architect)
**Date:** 2026-06-09T23:27:49-05:00
**Status:** Adopted

---

## Context

Sprint 1 delivered storage, ingest, ML model, and Foundry agent. Sprint 2 required the
FastAPI integration seam that connects all three layers so the Foundry agent tools can
call real HTTP endpoints.

---

## Decisions


### Dallas Decision — ABSZarrV3Store Code-Review Bug Fixes

**Date:** 2026-06-11
**Author:** Dallas
**Branch:** feat/adls-viewer-readers
**Commit:** b2b2b58

#### Context

A code review of `ABSZarrV3Store` in `src/deepseismic/storage/blob_client.py` identified three bugs of varying severity. All three were fixed in commit b2b2b58.

#### Bug 1 (CRITICAL) — Event loop blocked on every chunk read

**Location:** `ABSZarrV3Store.get()`

**Root cause:** `asyncio.to_thread(blob_client.download_blob().readall)` evaluates `blob_client.download_blob()` — a blocking HTTP call — on the event-loop thread *before* `to_thread` dispatches anything. Only the already-returned `.readall` callable is deferred, not the network round-trip. This serialises all concurrent zarr chunk fetches behind blocking I/O on the main thread.

**Fix:**
```python
# Before (buggy)
raw: bytes = await asyncio.to_thread(blob_client.download_blob().readall)

# After (correct)
raw: bytes = await asyncio.to_thread(lambda: blob_client.download_blob().readall())
```

**Impact:** Without this fix, multi-chunk parallel reads (`get_partial_values`) offer zero concurrency benefit — all downloads queue behind each other on the event loop.

#### Bug 2 (HIGH) — `SuffixByteRequest(0)` returns the entire blob

**Location:** `_apply_byte_range()`, `SuffixByteRequest` branch

**Root cause:** Python `-0 == 0`, so `data[-0:]` equals `data[0:]` — the full byte sequence. A suffix of zero should logically return no bytes, but the unguarded slice silently returned all bytes, corrupting callers that request an empty suffix tail.

**Fix:**
```python
# Before (buggy)
if isinstance(byte_range, SuffixByteRequest):
    return data[-byte_range.suffix :]

# After (correct)
if isinstance(byte_range, SuffixByteRequest):
    if byte_range.suffix == 0:
        return b""
    return data[-byte_range.suffix :]
```

**Regression test added:** `TestApplyByteRange.test_suffix_zero_returns_empty` in `src/tests/test_viewer/test_data_readers.py`.

#### Bug 3 (MEDIUM) — `set()` silently ignores `byte_range` (partial write)

**Location:** `ABSZarrV3Store.set()`

**Root cause:** The method signature accepted `byte_range: tuple[int, int] | None = None` but never used it. A caller requesting a partial write would receive a full-blob overwrite with no error. `ABSZarrV3Store` does not advertise `supports_partial_writes = True`, so partial-write calls should be rejected loudly.

**Fix:**
```python
async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
    self._check_writable()
    if byte_range is not None:
        raise NotImplementedError("ABSZarrV3Store does not support partial writes")
    ...
```

#### Validation

- `ruff check src/` — clean (0 errors)
- `pytest -m "not integration" -q` — **156 passed, 2 skipped, 6 deselected** (up from 155)

---

