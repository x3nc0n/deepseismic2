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

1. Install Python 3.11.
2. Create and activate a virtual environment.
3. Install the project in editable mode with development dependencies.
4. Populate environment variables for Azure resources.
5. Add a Volve subset to local development storage or wire the project to ADLS Gen2.
6. Start implementing ingest, preprocessing, model, API, and agent modules.

Example placeholder setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Team perspective

This PoC is intentionally simple: prove one end-to-end path that connects seismic data ingest, preprocessing, model inference, API access, and AI-assisted interpretation. Domain grounding is organized around three SME perspectives: geophysics, geology, and geoengineering.

## Status

This repository is currently scaffolded for implementation. Most Python modules are placeholders with documented responsibilities and planned interfaces so the team can start filling in working code without revisiting the top-level structure.
