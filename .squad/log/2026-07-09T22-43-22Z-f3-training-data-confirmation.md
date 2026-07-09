# Session Log: F3 Training Data Confirmation

**Date:** 2026-07-09T22:43:22Z  
**Incident/Topic:** F3 data readiness for T4 training run  
**Issue:** x3nc0n/deepseismic2#31  
**Infra Issue:** Spava-Corp/deepseismic2-infra#23

---

## Status

✓ **Confirmed:** Real F3 data is not present in application repo. Must be externally sourced from public OpendTect F3 Demo before T4 training can proceed.

---

## Summary

Dallas (Data/ML Engineer) investigated the repo structure in response to infrastructure team's three questions about F3 cube location/format, fault-interpretation parser, and survey geometry. Finding: only a synthetic proxy (`data/f3/`) exists; real data must come from the public OpendTect F3 Demo (dGB Earth Sciences / TerraNubis, CC BY-SA).

**Decision:** Use existing `scripts/download_f3.py` to ingest F3 Demo data, apply `parse_opendtect_fault_sticks` parser (not Petrel parser), stage at `staged/surveys/f3-demo/` in Azure ADLS.

**Leakage gate confirmed:** F3 for training only; Volve for evaluation only (hard rule per issue #24).

---

## Next Actions

1. Review and approve F3 ingest contract (decisions.md)
2. Infra team: Provision T4 data staging before training job submission
3. Validate real F3 geometry (IL 100–750, XL 300–1250, ~462 samples @ 4ms)
