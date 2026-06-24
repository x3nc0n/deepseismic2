# Ash — History

## Project Context

- **Project:** deepseismic2 — Petroleum seismic data analysis PoC
- **Stack:** Python, PyTorch, segyio, Azure, LLM APIs
- **Goal:** Modernize seismic interpretation; make it affordable and AI-accessible
- **Data:** Equinor Volve dataset — North Sea, Jurassic-Cretaceous section, marine 3D seismic
- **Role:** Domain SME for seismic acquisition, processing, and quantitative interpretation
- **User:** jospaid

## Learnings

- **2026-06-10:** ST10010 PSDM recommended as primary PoC volume.

- **2026-06-24:** Process fidelity audit completed. Key findings: (1) Task mismatch — original DeepSeismic does multi-class facies segmentation; we do binary fault detection only. (2) Labels are synthetic procedural geometry, not expert interpretation — circular validation. (3) Training is on synthetic NumPy arrays, not real Volve SEG-Y. (4) preprocessing/pipeline.py is a stub — conditioning stage is unimplemented. (5) No amplitude preservation, no phase/polarity QC, no bandwidth documentation. (6) No multi-class IoU, no well-tie, no per-class IoU matching original benchmarks. Full report in .squad/decisions/inbox/ash-process-fidelity.md.

## Scribe Consolidation — 2026-06-24T23:29:56Z

Ash's process fidelity assessment merged into `.squad/decisions.md` (Phase 2 Process Fidelity Evaluations section). Three critical gaps documented:
- GAP-C1: Training on synthetic geometry, not real interpreted data
- GAP-C2: preprocessing/pipeline.py empty stub
- GAP-C3: Task mismatch (binary fault vs multi-class facies)

Orchestration log written to `.squad/orchestration-log/2026-06-24-232956Z-ash.md`. Ripley recommends wiring real labels into training as part of Sprint 2 minimum viable set.

