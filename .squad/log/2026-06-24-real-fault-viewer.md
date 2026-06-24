# Session Log — Real Fault Viewer Phase 1

**Date:** 2026-06-24  
**Session:** "Make the demo really identify seismic traces and faults"  
**Duration:** Overnight autonomous sprint  
**Requested by:** jospaid

---

## Spawn (Agent Team)

- **Ripley** (Lead, opus) — produced phased plan: pre-bake fault inference + direct Zarr read. Wrote decision paper.
- **Ash** (Geophysics SME, sonnet) — produced credibility guidance (real axes, probability overlay, fault-stick overlay, honesty captions). Resolved coordinate ambiguity.
- **Dallas** (Data/ML, sonnet) — **IMPLEMENTED Phase 1**: fixed zarr v3 LocalStore bug; added bake script; rewired viewer to real amplitude + baked fault probability with correct axes; implemented fault-stick overlay with resolved coordinate mapping. All QC checks passing.
- **Hudson** (Tester, sonnet) — **IN PROGRESS**: adding integration tests for readers and coordinate mapping regression guard.

---

## Outcome Summary

### Phase 1 Complete ✅

Transitioned viewer from 100% synthetic placeholders to **real seismic data** and **real UNet fault detection**:

- ✅ Fixed zarr v3 bug in inference writer (`DirectoryStore` → `LocalStore`)
- ✅ Created `scripts/bake_demo_faults.py` (11.8s CPU on 10M voxels; QC PASS)
- ✅ Rewired `streamlit_app.py` to real Zarr amplitude + baked fault probability
- ✅ Implemented correct display axes (inline 1001–1100, XL 1900–2099, TWT 0–1996ms)
- ✅ Added threshold slider (0.3–0.7), colorbars, fault-stick overlay
- ✅ Resolved fault-stick coordinate ambiguity (z_ms column = sample index × 4.0)
- ✅ Added Ash's honesty captions and credibility checklist items
- ✅ Tests: 102 passed / 5 skipped, ruff clean

### Bake QC Results

| Metric | Value | Status |
|--------|-------|--------|
| Probability range | 0.000–1.000 | ✅ PASS |
| Mean probability | 0.1258 | ✅ PASS |
| Fault voxel fraction | 3.89% | ✅ PASS |
| Model credibility | Spatially selective | ✅ PASS |

### Phase 2 Stretch (Deferred)

- Crossline view
- On-demand fault detection button
- Wiggle trace overlay
- Agent integration

---

## Decisions Merged

Four decision documents merged into `.squad/decisions.md`:
1. `ripley-real-fault-viewer-plan.md` — architecture & phased plan
2. `ash-trace-fault-demo-credibility.md` — domain credibility requirements & checklist
3. `dallas-real-fault-viewer-impl.md` — implementation details, QC results, coordinate mapping
4. `lambert-api-wiring.md` — agent tool endpoints (re-verified with Phase 1)

---

## Next Steps

- Hudson completes integration tests (coordinate regression guard)
- Feature PR: Phase 1 implementation (src/, scripts/, tests)
- Infra follow-up (Parker): deploy updated UI/API + baked fault zarr to hosted demo

---

*Scribe session.*
