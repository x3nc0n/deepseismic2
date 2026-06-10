# Squad Decisions

## Active Decisions

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
