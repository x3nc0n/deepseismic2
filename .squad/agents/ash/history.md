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

- **2026-06-24 (S2-06):** Implemented `src/deepseismic/preprocessing/pipeline.py` — the conditioning/QC stage that was previously a stub (GAP-C2). Functions implemented:
  - `compute_volume_qc(zarr_path, sidecar_path)` — returns nonzero fraction, full amplitude stats (min/max/mean/std/p01/p99), dominant frequency via FFT of sampled traces, λ/4 vertical resolution estimate, and autocorrelation symmetry as a zero-phase proxy.
  - `global_amplitude_normalize(volume, p99)` — amplitude-preserving normalisation by volume p99 (from sidecar), with optional ±1.5 clip. Fixes GAP-I1: per-patch z-score in patches.py destroys lateral amplitude gradients; this function preserves them.
  - `condition_volume(...)` — thin orchestration entry tying QC + optional normalisation + JSON report write.
  - Geophysical assumptions documented in module docstring: zero-phase, SEG normal polarity, dt=4 ms default, λ/4 resolution.
  - **Actual QC numbers on data/volve/staged/synthetic.zarr (100×200×500, float32):**
    - Non-zero fraction: 1.0 (no dead traces)
    - Amplitude min/max: −0.488 / 1.107 (asymmetric — synthetic volume has positive polarity bias)
    - p01/p99: −0.121 / 0.104
    - Mean ± std: ~0 ± 0.0419
    - dt: 4 ms → Nyquist 125 Hz
    - Dominant frequency: **36.6 Hz**
    - Vertical resolution (λ/4 at 2000 m/s): **13.7 m**
    - Autocorrelation symmetry: **1.000** (exactly zero-phase, as expected for a synthetic)
- **2026-06-24 (S2-bug zerophase fix):** Fixed broken `_autocorr_symmetry` proxy (identified by Hudson in S2-07 testing). Root cause: the autocorrelation of any real signal is mathematically symmetric (ac[c+k] == ac[c−k] for all k), so the old ratio was always 1.0 regardless of phase. Replaced with `_wavelet_symmetry` — the normalised time-reversal correlation dot(x, x[::-1]) / dot(x, x) on the mean trace.
  - **Range:** +1.0 = symmetric (zero-phase), ~0.0 = quadrature (90°-rotated), −1.0 = antisymmetric.
  - **Implementation:** Pure numpy, no scipy required.
  - **Tests:** Removed `@pytest.mark.xfail(strict=True)`. All 3 tests now genuinely pass: symmetric Gaussian → 1.0, causal exponential deviates from 1.0, zero-phase Ricker (n=129, odd, perfectly centered) scores >0.5 above its Hilbert-transform (90°-rotated) counterpart.
  - **Suite:** 211 passed, 2 skipped (up from 209+1xfail). Lint clean.
  - **New QC number on data/volve/staged/synthetic.zarr:** `wavelet_symmetry = 0.116` (was 1.000 broken). The low value is geophysically expected: the mean of 500 traces with independent random reflectivity averages out the wavelet shape, making the mean-trace proxy unreliable for this volume. This is correctly documented as a caveat — the metric is most informative on a single extracted wavelet, not an averaged multi-trace record. The honest answer replaces the false "perfect" score.
  - Renamed metric key in QC dict from `autocorr_symmetry` → `wavelet_symmetry`. Decision note: `.squad/decisions/inbox/ash-s2-bug-zerophase-fix.md`.

