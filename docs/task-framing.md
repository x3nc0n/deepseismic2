# Task Framing: Binary Fault Detection vs. Multi-Class Facies Segmentation

**Author:** Ripley (Lead/Architect)
**Date:** 2026-06-24T20:09:56-05:00

---

## The distinction that matters

The reference project — [microsoft/seismic-deeplearning](https://github.com/microsoft/seismic-deeplearning)
— solves **multi-class lithofacies segmentation** on the F3 Netherlands and Penobscot
datasets. It assigns every voxel (or section pixel) to one of several depositional
facies classes using expert-interpreted contest labels. Evaluation uses pixel-level
accuracy and mean IoU over classes. That is the baseline it establishes.

DeepSeismic2 solves **binary fault detection** on the Equinor Volve dataset. It
assigns every voxel a probability of belonging to a fault plane. Evaluation uses
binary IoU, Dice/F1, and distance-tolerant recall/precision. These are different
scientific problems with different label sources, different output spaces, and
different appropriate benchmarks.

Stating that DeepSeismic2 "modernizes" or "extends" the original without this
qualification is misleading. We have not replicated the original's task — we have
built a different task on different data using similar architectural scaffolding.

---

## Why the original uses F3/Penobscot, not Volve

The F3 and Penobscot datasets come with **dense, pixel-complete interpretation labels**
produced for public ML contests. Every inline/crossline section has a ground-truth
facies label per sample. That makes them suitable for supervised segmentation training
and for apples-to-apples benchmarking against other methods on the same splits.

Volve is a **production field dataset** with well-constrained geology, but it has
almost no publicly available pixel-complete interpretation labels. What it does have
is **fault sticks** — Petrel-format line segments that mark where an interpreter
identified a fault trace on a section. These are sparse: the two `.dat` files in the
Volve release contain 18 fault-stick points in total. That is appropriate for a binary
fault detector (given sufficient dilation and a realistic positive-fraction strategy),
but it cannot support multi-class facies segmentation at all.

**Do not compare DeepSeismic2 metrics against F3/Penobscot facies benchmarks.** The
datasets, tasks, and label densities are incompatible.

---

## The correct comparison lineage for fault detection

Binary fault detection from 3D seismic is an active research area with its own
literature and baselines:

- **Wu et al. (2019) — FaultSeg3D** ([paper](https://doi.org/10.1190/geo2018-0819.1)):
  Synthetic training data (65 volumes) with binary fault masks, 3D UNet architecture,
  evaluated on real North Sea data. Establishes the synthetic-training → real-test
  paradigm for fault detection. Our architecture and training approach are most
  comparable to this work.

- **Qi et al. (2019, 2020) — Seismic fault detection via structure-oriented
  filtering and image processing**: Earlier CNN-based fault attribute work that
  informs the broader fault-detection literature.

- **Hale (2013) — Methods to compute fault images, extract fault surfaces, and
  estimate fault throws from 3D seismic images**: Classic deterministic baseline;
  useful as a lower bound for comparison.

When reporting results or positioning DeepSeismic2, cite these works — not the
F3/Penobscot facies benchmarks.

---

## Appropriate metrics for fault detection

| Metric | Appropriate for us | Notes |
|--------|-------------------|-------|
| Binary IoU (Jaccard) | ✅ | Primary metric |
| Dice / F1 | ✅ | Equivalent to F1; common in segmentation literature |
| Distance-tolerant recall (±N vox) | ✅ | Critical for fault detection: a prediction 3–5 voxels from a true fault plane is geophysically meaningful and should not count as a miss |
| Distance-tolerant precision (±N vox) | ✅ | Symmetric: a prediction near a true fault is not a false positive |
| Average Symmetric Surface Distance (ASSD) | ✅ | Summarizes spatial displacement of predicted fault surface |
| Pixel accuracy | ❌ misleading | Dominated by the ~99.9% negative background; always looks good |
| Per-class mIoU (multi-class) | ❌ N/A | Only meaningful for multi-class segmentation |

The `validation/__init__.py` module computes all the appropriate metrics above.
`scripts/evaluate.py` reports them and writes `output/eval_metrics.json`.

---

## Ash's geophysics assessment (incorporated)

Ash's process fidelity audit (2026-06-24, `.squad/decisions.md`) identified this task
mismatch as **GAP-C3**: "we do binary fault detection; the original does multi-class
facies." That finding is the direct source of this document. Key quote:

> "The original project is a multi-class **facies segmentation** system trained on
> expert-interpreted labels from a production reference dataset (F3). Our PoC is a
> binary **fault detector** trained on procedurally generated synthetic data. We cannot
> claim to emulate the original interpretation process; we emulate its software
> scaffolding only."

Sprint 2 addressed the synthetic-data problem: we now train on real Volve fault-stick
labels. The task-mismatch framing documented here remains a permanent architectural
fact — not a gap to close, but a context every reader must have.

Sprint 3 made the label pipeline **directory-based** (any number of `.dat` files) and
added optional between-pick interpolation (`--interpolate-between`). Validated with a
synthetic proxy set (6 files, 76 picks → 0.30% positive-voxel fraction). Real Volve
interpretation data is blocked on the Databricks Marketplace install (infra #11). The
0.30% proxy number is **not** a real Volve result — it is a format-and-code-path
validation proxy only. Expected positive fraction with full real interpretation: ≥ 1–3%.

---

## Summary

| Dimension | microsoft/seismic-deeplearning | DeepSeismic2 |
|-----------|-------------------------------|--------------|
| Task | Multi-class facies segmentation | Binary fault detection |
| Dataset | F3 Netherlands, Penobscot | Equinor Volve (ST10010 geometry) |
| Labels | Dense pixel-complete contest annotations | Sparse fault sticks (18 real picks, 2 files; Sprint 3 synthetic proxy: 76 picks, 6 files → 0.30% positive fraction) |
| Output | Per-voxel class assignment (N classes) | Per-voxel fault probability (0–1) |
| Primary metrics | mIoU, pixel accuracy per class | Binary IoU, Dice, distance-tolerant recall |
| Benchmark lineage | F3/Penobscot ML contest baselines | FaultSeg3D (Wu et al. 2019) and related |
| Architecture | UNet, SEResNet, HRNet, DeepLab | UNet3D (single model, PoC scope) |
