# DeepSeismic2 PoC Architecture Proposal

**Author:** Ripley  
**Requested by:** jospaid  
**Date:** 2026-06-09T21:56:47-05:00

## Problem Statement

Legacy seismic interpretation stacks usually couple large SEG-Y volumes, preprocessing, model execution, and user workflows to premium shared storage and long-lived monolithic applications. That works for large enterprises, but it drives cost into the storage layer, slows experimentation, and makes AI integration awkward.

This PoC should prove a simpler story:

- large seismic data can live on affordable object storage
- preprocessing and model inference can scale independently
- classical seismic CNNs still do the heavy numerical work
- LLMs add value around orchestration, summarization, metadata, and analyst productivity

## What Good Looks Like for the PoC

- Run a small but credible Volve dataset workflow end to end
- Ingest SEG-Y and related metadata without requiring premium NAS
- Produce a segmentation or interpretation artifact using an existing PyTorch model pattern
- Add an LLM-assisted analyst experience without pretending the LLM replaces seismic models
- Keep the implementation small enough for one week per person

---

## 1) Modernization Approaches Compared

| Approach | Summary | Pros | Cons | Cost Implications |
|---|---|---|---|---|
| **A. Lift-and-shift with cheaper managed file storage** | Keep legacy-style shared filesystem workflow, but replace premium NAS with lower-cost cloud file storage where possible. | Lowest migration risk; familiar operating model; minimal code rewrite. | Still storage-centric; still chatty I/O; weak elasticity; LLM integration remains bolted on. | Better than Premium Files/NetApp, but still pays for shared filesystem semantics and over-provisioned capacity. Savings are limited. |
| **B. Cloud-native object storage + decoupled batch compute** | Store raw and derived seismic assets in object storage; use containerized preprocessing and GPU inference jobs. | Best PoC balance; cheap storage; scale CPU and GPU separately; clear modernization story; easy to add APIs and AI assistants. | Requires data packaging changes; some tooling must adapt from POSIX assumptions; job orchestration needed. | Lowest practical cost for PoC. Blob storage is cheap, CPU jobs run on demand, GPU cost is isolated to inference windows. |
| **C. Event-driven/serverless pipeline** | Trigger preprocessing and downstream steps from object events, queues, and functions. | Operationally elegant; good for automation; small always-on footprint. | Poor fit for heavy seismic transforms and long GPU jobs; debugging is harder; more moving parts than needed for PoC. | Cheap for control plane, but still needs external GPU compute. Complexity cost is higher than cash savings for this phase. |
| **D. Fully AI-native analyst copilot** | Put an LLM-first interface in front of the workflow, with retrieval, planning, and natural-language interaction as the primary user experience. | Strong demo value; compelling modernization narrative; good fit for Copilot/Foundry tooling. | Dangerous if used as core interpretation engine; hallucination risk; still needs deterministic seismic processing underneath. | LLM token spend is manageable for a PoC, but this does not remove the need for storage and model compute. AI-only positioning would oversell capability. |

### Recommendation

**Pick Approach B for the PoC**, then layer a thin slice of Approach D on top.

That gives the cleanest story: cheap storage, decoupled compute, proven CNN-based interpretation, and targeted LLM assistance where it actually helps.

---

## 2) Recommended Architecture

### Core Design

Use **Azure Blob Storage / ADLS Gen2 as the system of record**, **containerized Python preprocessing on CPU**, and **on-demand GPU inference for seismic models**. Expose results and metadata through a lightweight backend API. Add an LLM assistant for workflow guidance, metadata Q&A, and interpretation summarization.

### Storage Tier

**Recommendation:** Azure Data Lake Storage Gen2 on standard object storage.

#### Storage layout

- `raw/` — original SEG-Y, well logs, supporting Volve files
- `staged/` — chunked intermediate volumes, extracted headers, normalized tensors
- `features/` — ML-ready patches, labels, training/inference manifests
- `results/` — prediction volumes, masks, QC images, overlays
- `catalog/` — metadata JSON, run manifests, lineage, prompts, summaries

#### Why this works

- object storage is much cheaper than premium shared filesystems
- seismic volumes are large, sequential, and batch-friendly
- derived data can be chunked into cloud-friendly formats for downstream jobs

#### Format decisions

- keep **SEG-Y** as the raw source format for fidelity and interoperability
- use **Zarr** for chunked intermediate arrays where random access matters
- store metadata and manifests as **JSON / Parquet**

### Compute Tier

**CPU path**

- preprocessing
- SEG-Y header extraction
- volume slicing / patch generation
- QC image generation
- metadata indexing

**Recommended service:** Azure Container Apps Jobs or Azure Batch for PoC CPU workloads.

**GPU path**

- batch inference for segmentation models
- optional fine-tuning only if time remains

**Recommended service:** Azure Machine Learning managed compute or Azure Batch GPU pools.

For this PoC, **Azure ML managed compute is the better choice** because it shortens setup time for PyTorch containers, experiment tracking, and GPU job execution.

### Data Flow

1. **Ingest**
   - Upload Volve SEG-Y and related files to `raw/`
   - Register dataset manifest and metadata in `catalog/`

2. **Preprocess**
   - Read SEG-Y with `segyio`
   - Extract geometry, headers, and basic stats
   - Convert selected seismic windows or cubes into chunked Zarr
   - Generate patches or slices for model input

3. **Model inference**
   - Run UNet-style segmentation inference in PyTorch on GPU
   - Write prediction masks and probability outputs to `results/`

4. **Post-process / interpretation packaging**
   - Create QC plots, inline/xline previews, horizon/facies summaries
   - Publish derived metadata and preview artifacts for API access

5. **Analyst experience**
   - Backend API serves dataset inventory, run status, result references, and summaries
   - LLM assistant answers questions against metadata, run outputs, and curated domain notes

### LLM Integration Points

#### Where CNNs do the heavy lifting

- seismic facies / segmentation inference
- numeric transforms
- volume patching and tensor processing
- deterministic QC metrics

Do **not** ask an LLM to classify seismic voxels directly. That is the CNN’s job.

#### Where LLMs add real value

- dataset and run summarization
- natural-language querying of metadata, wells, surveys, and outputs
- operator assistance: “what data is loaded?”, “what model ran?”, “what changed between runs?”
- interpretation explanation: convert model outputs and metadata into analyst-readable summaries
- Copilot workflow integration for notebooks, scripts, and operational runbooks

#### Good PoC LLM features

- “Explain this run” summary from job metadata and QC outputs
- “What does this volume contain?” grounded answer from dataset catalog
- “Generate a next-step checklist” after inference completes
- lightweight retrieval over project docs, manifests, and model cards

### What the Volve Dataset PoC Demonstrates

- ingesting real petroleum seismic data into object storage instead of premium NAS
- preprocessing SEG-Y into cloud-friendly derived formats
- running a modern PyTorch segmentation workflow on demand
- producing interpretation-ready artifacts without a monolithic application
- adding a Copilot-style assistant that helps analysts understand the workflow and outputs

---

## 3) Component Breakdown by Team Member

Scope target: roughly **one week per person**. Keep each slice demoable on its own.

### Dallas — Data / ML Engineer

**Deliverables**

- Python ingestion + preprocessing pipeline for a constrained Volve sample
- SEG-Y to Zarr or tensor conversion flow
- baseline inference notebook/script using UNet-style model
- sample output masks and QC visualizations

**Definition of done**

- one command or notebook path that goes from selected Volve input to prediction artifacts
- model inputs/outputs stored in agreed container paths
- documented assumptions on cube size, patching, and label availability

### Parker — Backend / Infra

**Deliverables**

- Azure storage layout and container conventions
- containerized job definitions for CPU preprocessing and GPU inference
- minimal backend API for dataset catalog, job status, and results lookup
- cost-conscious deployment notes for PoC environments

**Definition of done**

- storage containers created and documented
- one CPU job and one GPU job runnable on Azure
- API returns dataset metadata and result locations

### Lambert — AI Integration

**Deliverables**

- LLM-assisted analyst workflow using Azure OpenAI / Foundry
- retrieval layer over run manifests, metadata, model cards, and generated summaries
- prompt templates for run explanation and interpretation summary
- Copilot-friendly examples for GitHub Copilot / M365 Copilot grounding

**Definition of done**

- assistant can answer grounded questions about the loaded dataset and latest run
- assistant can generate a concise interpretation summary from model outputs + metadata
- prompt and grounding boundaries are explicit

### Hudson — Tester

**Deliverables**

- validation checklist for ingest, preprocessing, inference, API, and LLM outputs
- smoke tests for pipeline contracts and file presence
- QC rubric for “credible PoC result” review
- demo script that proves the modernization story end to end

**Definition of done**

- test checklist covers happy path and obvious failure cases
- deterministic validation for manifests, output locations, and response shapes
- demo run can be repeated from documented steps

---

## 4) Tech Stack Decisions

### Language and runtime

- **Python:** 3.11
  - mature PyTorch support
  - broad library compatibility
  - modern typing and performance without chasing the newest runtime

### Key Python libraries

- **PyTorch** — model inference and optional fine-tuning
- **TorchIO** or basic custom tensor transforms — patch/slice handling if useful
- **segyio** — SEG-Y reading
- **xarray + zarr** — chunked array handling
- **numpy, scipy, pandas** — numeric and tabular processing
- **dask** — optional parallel preprocessing if needed
- **fastapi** — thin backend API
- **pydantic** — API contracts and metadata models
- **matplotlib / plotly** — QC and preview plots

### Azure services

- **Azure Blob Storage / ADLS Gen2** — raw and derived seismic data
- **Azure Machine Learning** — GPU inference jobs and experiment execution
- **Azure Container Apps Jobs** or **Azure Batch** — CPU preprocessing jobs
- **Azure Container Registry** — container images
- **Azure Functions** — optional lightweight orchestration hooks only if needed
- **Application Insights** — logs and telemetry for the API/jobs

### LLM stack

- **Azure OpenAI** as primary hosted LLM API
- **Microsoft Foundry** for agent experimentation and orchestration patterns
- **GitHub Copilot** for developer productivity on notebooks, scripts, and API work
- **M365 Copilot / Copilot Studio** only as a demo-side integration layer, not as the core runtime

### Model choices

- Start with **UNet** as the first segmentation baseline
- Defer HRNet / SEResNet comparisons unless the base path works
- Optimize for a credible end-to-end story, not benchmark breadth

---

## What Changes

- storage becomes object-first instead of filesystem-first
- workloads become job-based instead of monolithic
- model execution is isolated from ingestion and API concerns
- LLM capability is added as a grounded assistant, not as the numerical engine

## What Stays the Same

- seismic interpretation still depends on deterministic domain models
- SEG-Y remains the raw source of truth
- PyTorch remains the right baseline for CNN inference
- domain experts still validate interpretation quality

---

## Risks and Mitigations

### Risk 1: SEG-Y handling is too slow if read naively from object storage

**Likelihood:** Medium  
**Impact:** High

**Mitigation:** Pre-stage only the needed Volve subset, convert once to chunked Zarr, and reuse derived artifacts for repeat runs.

### Risk 2: GPU setup burns too much of the PoC week

**Likelihood:** Medium  
**Impact:** High

**Mitigation:** Use Azure ML managed environments and start with inference only, not training.

### Risk 3: LLM outputs drift into unsupported geoscience claims

**Likelihood:** Medium  
**Impact:** Medium

**Mitigation:** Ground responses on manifests, QC summaries, and model metadata. Keep prompts scoped to explanation and summarization.

### Risk 4: Volve dataset is too large for a tight demo loop

**Likelihood:** High  
**Impact:** Medium

**Mitigation:** Explicitly define a reduced survey subset for the PoC and treat full-volume scaling as follow-on work.

---

## Scope

### In scope for this PoC

- one real dataset path using a constrained Volve subset
- object storage based ingest and derived data layout
- one CPU preprocessing path
- one GPU inference path
- one thin API
- one grounded LLM assistant workflow

### Deferred

- full training pipeline at production scale
- multi-user interpretation application
- real-time streaming interpretation
- automated reservoir decision-making
- large-scale benchmark comparison across many models

---

## Final Recommendation

Build the PoC as a **cloud-native, object-storage-first seismic workflow** with:

- **ADLS Gen2 / Blob** for raw and derived data
- **Python 3.11 + PyTorch + segyio + Zarr**
- **CPU preprocessing jobs**
- **Azure ML GPU inference jobs**
- **FastAPI** for metadata and results access
- **Azure OpenAI / Foundry** for grounded analyst assistance

This is the simplest architecture that proves the modernization story without pretending the PoC is a full platform.
