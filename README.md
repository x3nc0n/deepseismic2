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

**Sprint 3 complete — real-data app-readiness. Real-data execution is deploy-gated.**

The app is now **ready** to consume real Volve ST10010 data end-to-end: ingest,
label generation, training, evaluation, and the API + agent all have real-data
execution paths wired and locally validated. **Execution on real data is blocked
by infrastructure and data-access dependencies** described in the blockers section
below — not by code gaps. See [docs/real-data-runbook.md](docs/real-data-runbook.md)
for the ordered deploy path.

This is **binary fault detection** on Volve — a different task from the reference
microsoft/seismic-deeplearning project (multi-class facies segmentation on
F3/Penobscot). See [docs/task-framing.md](docs/task-framing.md) for the full rationale.

### What's real vs. what's demo (Sprint 3)

| Component | Status | Notes |
|-----------|--------|-------|
| SEG-Y → Zarr ingest geometry | ✅ App-ready | File-driven; handles real ST10010 inlines 9985–10369. New `scripts/ingest_segy.py` CLI. Real execution deferred — blocked on infra #11 |
| Fault label generation | ✅ App-ready | Directory-based (any `.dat` files); optional between-pick interpolation (`--interpolate-between`). Validated with synthetic proxy (76 picks → 0.30% positive fraction). Real data deferred — blocked on Marketplace install |
| UNet3D training (ADLS backend) | ✅ App-ready | `--storage-backend azure` reads staged Zarr via `ABSZarrV3Store`. Real execution deferred — in-VNet only |
| Evaluation (ADLS backend) | ✅ App-ready | `scripts/evaluate.py --storage-backend azure`. Real execution deferred — in-VNet only |
| FastAPI backend (real mode default) | ✅ Real default | Real mode is now the default when storage is configured. Mock requires explicit opt-in (`DEEPSEISMIC_MOCK_MODE=true`). Misconfigured real mode fails loud (HTTP 503) — no silent mock fallback |
| Foundry agent (real mode default) | ✅ Real default | Live Azure OpenAI is the default when `AZURE_PROJECT_ENDPOINT` is set. Mock requires explicit opt-in (`MOCK_LLM=true`). Misconfiguration raises `RuntimeError` |
| Health endpoint | ✅ Enhanced | Reports `storage: ok\|unreachable\|error\|mock` — confirms real storage reachability post-deploy |
| QC / signal conditioning stage | ✅ Real | Dominant freq, λ/4 resolution, zero-phase check, amplitude-preserving normalisation |
| Sliding-window inference engine | ✅ Real | Gaussian overlap-blending, CPU and CUDA |
| Amplitude volume | ⚠️ Synthetic stand-in | Ricker-wavelet synthetic approximating Volve ST10010 geometry — **not** actual ST10010 amplitudes |
| Fault labels | ⚠️ Synthetic proxy (Sprint 3) | 76 picks across 6 synthetic files → 0.30% positive fraction. **NOT real Volve ground truth.** Sprint 2 real sticks: 18 picks → 0.08% |
| Demo UIs | ℹ️ Pre-baked | Streamlit / Gradio visualise pre-baked inference results from `fault_prob.zarr` |

### Real-data readiness (Sprint 3)

**App-ready (code complete, locally validated as format proxy):**
- SEG-Y ingest geometry handles ST10010 natively — validated against synthetic SEG-Y as format proxy only
- Fault label generation accepts a directory of `.dat` files; optional between-pick interpolation
- Training and evaluation read Zarr from ADLS via `--storage-backend azure`
- API real mode is now the default; health endpoint confirms storage reachability

**Deploy-gated (execution requires infra + data access):**
- Real ingest and training must run **in-VNet** — ADLS uses private endpoints (`publicNetworkAccess: Disabled`)
- Suitable execution environments: Azure ML managed compute, Container App jobs

**Explicit blockers (not code gaps):**

| Blocker | Owner | Tracking |
|---------|-------|---------|
| ST10010 SEG-Y copy job into `raw` ADLS container | Spava-Corp/deepseismic2-infra | infra issue #11 |
| Equinor Volve Databricks Marketplace listing install (identity-bound) | User action | Manual step |
| Private-endpoint networking — all real ingest/train/eval must run in-VNet | Spava-Corp/deepseismic2-infra | infra issue #11 |

See [docs/real-data-runbook.md](docs/real-data-runbook.md) for the ordered sequence of steps
once blockers are resolved.

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

Sprint 3 synthetic-proxy validation used 76 picks (6 synthetic `.dat` files) → 0.30%
positive-voxel fraction. **⚠️ These are synthetic-proxy numbers only — NOT real Volve
results.**

### Reproduce the real pipeline (Sprint 2 baseline, local)

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

### Sprint 3 — local smoke-test commands (synthetic proxy, NOT real Volve data)

```powershell
# SEG-Y ingest smoke-test (format proxy only — synthetic SEG-Y, not ST10010)
# ⚠️  Numbers are NOT from real Volve data.
python scripts/ingest_segy.py `
    --source data/volve/synthetic_sample.segy `
    --dest data/volve/staged/smoke_ingest.zarr `
    --survey-id synthetic-proxy `
    --sample-mode --overwrite

# Dense fault labels — synthetic proxy (6 synthetic .dat files)
# ⚠️  NOT real Volve ground truth.
python scripts/generate_fault_label.py `
    --fault-stick-dir data/volve/interpretations/fault_sticks_synth `
    --interpolate-between

# Real-mode API (requires Azurite or real storage configured)
# Default is now real mode — set mock only if you explicitly want synthetic responses
$env:DEEPSEISMIC_MOCK_MODE = "true"   # explicit mock opt-in
uvicorn deepseismic.api.main:app --reload --port 8000

# Check health / storage reachability
curl http://localhost:8000/health
# Returns: {"status":"ok","mock_mode":true,"storage":"mock"}
# Real mode returns storage: "ok" | "unreachable" | "error"
```

### In-VNet execution (real ST10010 — requires infra #11 + Marketplace)

See [docs/real-data-runbook.md](docs/real-data-runbook.md) for the full ordered sequence.

```bash
# Full ingest from ADLS raw container (in-VNet job only)
python scripts/ingest_segy.py \
    --source /mnt/raw/ST10010_PSDM_TIME.segy \
    --dest /mnt/staged/surveys/volve-st10010/amplitude.zarr \
    --survey-id volve-st10010 --overwrite

# Train on ADLS-staged data (in-VNet only — private endpoint)
python -m deepseismic.training.train \
    --data-mode zarr --storage-backend azure \
    --az-seismic-prefix surveys/volve-st10010/amplitude.zarr \
    --az-label-prefix surveys/volve-st10010/fault_label.zarr \
    --epochs 50 --device cuda --seed 42

# Evaluate against ADLS data
python scripts/evaluate.py \
    --checkpoint /mnt/features/checkpoints/best.pt \
    --storage-backend azure \
    --az-seismic-prefix surveys/volve-st10010/amplitude.zarr \
    --az-label-prefix surveys/volve-st10010/fault_label.zarr
```

### Also implemented (Sprints 1–2)

- ✅ SEG-Y ingest → Zarr conversion with metadata sidecars
- ✅ FastAPI backend (13 endpoints, mock mode supported)
- ✅ Foundry agent with 11 tools wired to real API
- ✅ Three demo UIs: terminal chat, Streamlit, Gradio
- ✅ 211 tests passing + CI workflow
- ✅ Local dev environment (Azurite, Docker, zero-config)
