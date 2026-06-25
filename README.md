# DeepSeismic2

[![CI](https://github.com/x3nc0n/deepseismic2/actions/workflows/ci.yml/badge.svg)](https://github.com/x3nc0n/deepseismic2/actions/workflows/ci.yml)
[![CD](https://github.com/x3nc0n/deepseismic2/actions/workflows/cd.yml/badge.svg)](https://github.com/x3nc0n/deepseismic2/actions/workflows/cd.yml)
[![API Image](https://img.shields.io/badge/GHCR-deepseismic2--api-blue?logo=github)](https://ghcr.io/x3nc0n/deepseismic2-api)
[![UI Image](https://img.shields.io/badge/GHCR-deepseismic2--ui-blue?logo=github)](https://ghcr.io/x3nc0n/deepseismic2-ui)

DeepSeismic2 is a proof-of-concept for modernizing seismic interpretation workflows with cloud-native data handling, ML inference, and AI-assisted analyst experiences. The goal is to prove that a lightweight Azure-first stack can replace expensive, workstation-bound legacy tooling for a constrained exploration workflow built around the Equinor Volve dataset.

## Why this project exists

Traditional seismic interpretation workflows are often tied to costly desktop software, file-share-centric storage, and highly specialized manual processes. That makes it hard to scale experimentation, expose results through APIs, or give non-specialists access to insights.

DeepSeismic2 addresses that gap by combining:
- object-storage-first seismic data management with SEG-Y as source truth and Zarr for cloud-friendly derivatives
- deterministic preprocessing and UNet-based interpretation workflows in Python
- a thin FastAPI backend for dataset, run, and results access
- a Foundry agent experience grounded by Azure AI Search knowledge assets

## PoC goals

- Ingest a constrained subset of the Equinor Volve dataset
- Convert seismic data into analysis-friendly formats for downstream processing
- Run baseline binary fault detection using a 3D UNet model trained on real fault-stick labels
- Expose metadata, run status, and result discovery through an API
- Enable an AI-native analyst workflow with domain grounding for geophysics, geology, and geoengineering perspectives

## Architecture snapshot

The current architecture follows the team's agreed direction:
- **Storage:** ADLS Gen2 / Azure Blob Storage as the system of record
- **Compute:** CPU for ingest and preprocessing, GPU for inference
- **ML platform:** Azure Machine Learning managed compute for PoC speed
- **API layer:** FastAPI for metadata, run status, and result lookup
- **Agent layer:** Foundry agent with Azure AI Search grounding
- **Knowledge scope:** Seismic glossary, workflow methodology, and Volve project context

## Repository layout

- `src/deepseismic/ingest/` - SEG-Y loading and format conversion
- `src/deepseismic/preprocessing/` - conditioning, patching, and quality control
- `src/deepseismic/models/` - model definitions and inference entry points
- `src/deepseismic/api/` - FastAPI service and route contracts
- `src/deepseismic/agent/` - Foundry agent scaffolding, tools, and grounding content
- `src/deepseismic/storage/` - Azure storage abstraction layer
- `src/tests/` - test package structure for ingest, preprocessing, models, and API
- `notebooks/` - exploration and demo notebooks
- `infra/` - infrastructure notes and future IaC assets
- `docker/` - container build context
- `scripts/` - utility scripts such as Volve download helpers

## Quick start

### Prerequisites

- Python 3.11+
- Docker (for Azurite local storage emulator)

### Setup

```powershell
# Clone and install
git clone https://github.com/x3nc0n/deepseismic2.git
cd deepseismic2
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,ui]"

# Copy environment template (pre-filled for local dev)
cp .env.example .env

# Start local storage (Azurite) + create containers + upload sample data
.\scripts\setup-local.ps1
```

### Run the demo

```powershell
# Start the API server (mock mode — no real data needed)
$env:DEEPSEISMIC_MOCK_MODE = "true"
$env:MOCK_LLM = "true"
uvicorn deepseismic.api.main:app --reload --port 8000

# In another terminal — pick your UI:
# Option 1: Terminal chat
python -m deepseismic.ui.chat

# Option 2: Streamlit (visual demo for geologists)
streamlit run src/deepseismic/ui/streamlit_app.py

# Option 3: Gradio
python src/deepseismic/ui/gradio_app.py
```

### Run tests

```powershell
python -m pytest src/tests/ -q          # unit tests (no infra needed)
python -m pytest src/tests/ -m integration  # requires Azurite running
python -m ruff check src/               # linting
```

## Status

**Sprint 2 complete — real-data training and evaluation implemented.**

The core ML loop is now real: UNet3D is trained on genuine Volve fault-stick labels
rasterized from field interpretation data, and evaluation runs against a held-out
volume region with non-degenerate metrics. This is **binary fault detection** on
Volve — a different task from the reference microsoft/seismic-deeplearning project
(which does multi-class facies segmentation on F3/Penobscot). See
[docs/task-framing.md](docs/task-framing.md) for the full rationale.

### What's real vs. what's demo

| Component | Status | Notes |
|-----------|--------|-------|
| SEG-Y → Zarr ingest | ✅ Real | Parses actual Volve ST10010 geometry |
| Fault label generation | ✅ Real | 18 fault-stick points from two Volve `.dat` files, rasterized at dilation=3 |
| UNet3D training on real labels | ✅ Real | `--data-mode zarr`, seed=42, `run_config.json` persisted |
| Evaluation with real metrics | ✅ Real | `scripts/evaluate.py` → `output/eval_metrics.json` |
| QC / signal conditioning stage | ✅ Real | Dominant freq, λ/4 resolution, zero-phase check, amplitude-preserving normalisation |
| Sliding-window inference engine | ✅ Real | Gaussian overlap-blending, CPU and CUDA |
| FastAPI backend (13 endpoints) | ✅ Real code | **Defaults to mock mode** (`DEEPSEISMIC_MOCK_MODE=true`); real-mode integration path not fully tested |
| Foundry agent (11 tools) | ✅ Real code | **Defaults to mock mode** (`MOCK_LLM=true`); real API calls work when `false` |
| Amplitude volume | ⚠️ Synthetic stand-in | Ricker-wavelet synthetic approximating Volve ST10010 geometry — **not** actual ST10010 amplitudes |
| Fault labels | ⚠️ Sparse | 18 points → 0.08% positive voxel fraction; adequate for pipeline validation, not a full interpretation |
| Demo UIs | ℹ️ Pre-baked | Streamlit / Gradio visualise pre-baked inference results from `fault_prob.zarr` |

### Results (Sprint 2, seed=42)

Trained on the synthetic amplitude stand-in with real Volve fault-stick labels.
20 epochs, CPU, 32³ patches, WeightedRandomSampler + combined BCE/Dice loss.

| Metric | Validation (patches, epoch 18) | Full-volume held-out (il 64–100) |
|--------|-------------------------------|----------------------------------|
| IoU | 0.047 | 0.062 |
| Dice / F1 | 0.089 | 0.117 |
| Tolerant recall (±5 vox) | — | 0.84 |
| Precision | 0.049 | 0.068 |

**Honest caveat:** These numbers show the pipeline runs end-to-end on real labels and
produces non-degenerate output. They are **not** a geophysical skill benchmark. Two
factors limit their interpretive weight: (1) the amplitude volume is a synthetic
stand-in, not actual Volve seismic; (2) only 18 fault-stick points were available,
making the label set extremely sparse. Results will differ substantially when run
against the full Volve ST10010 post-stack volume with a complete interpretation.
Reproducibility: seed=42, run config persisted at `checkpoints/run_config.json`.

### Reproduce the real pipeline

```powershell
# 1. Generate fault labels from the real Volve fault sticks
python scripts/generate_fault_label.py

# 2. Train on real labels
#    Requires: data/volve/staged/synthetic.zarr + data/volve/staged/fault_label.zarr
python -m deepseismic.training.train --data-mode zarr --epochs 20

# 3. Evaluate on the held-out volume region
python scripts/evaluate.py --checkpoint checkpoints/best.pt
#    Writes metrics to output/eval_metrics.json
```

### Also implemented (Sprint 1)

- ✅ SEG-Y ingest → Zarr conversion with metadata sidecars
- ✅ FastAPI backend (13 endpoints, mock mode supported)
- ✅ Foundry agent with 11 tools wired to real API
- ✅ Three demo UIs: terminal chat, Streamlit, Gradio
- ✅ 156 tests passing + CI workflow
- ✅ Local dev environment (Azurite, Docker, zero-config)
