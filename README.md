# DeepSeismic2

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
- Run a baseline interpretation workflow using a UNet-style model
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

**Sprint 1 complete.** Full end-to-end pipeline implemented:
- ✅ SEG-Y ingest → Zarr conversion with metadata sidecars
- ✅ Fault label generation from existing interpretations
- ✅ 3D UNet model with sliding-window inference
- ✅ FastAPI backend (13 endpoints, mock mode supported)
- ✅ Foundry agent with 11 tools wired to real API
- ✅ Three demo UIs: terminal chat, Streamlit, Gradio
- ✅ 79 tests passing + CI workflow
- ✅ Local dev environment (Azurite, Docker, zero-config)
