# M365 Copilot Agent Design for DeepSeismic2

**Author:** Lambert  
**Requested by:** jospaid  
**Date:** 2026-06-09T22:13:24-05:00

## Executive Summary

DeepSeismic2 can evolve from the current **Approach B + thin grounded chat** into a credible **AI-native analyst experience** by publishing a domain-specific agent into Microsoft 365 Copilot. The right shape is **not** an LLM that interprets geology by itself. The right shape is a **grounded petroleum geoscience copilot** that:

- answers questions over curated project knowledge
- calls deterministic backend APIs for live dataset, run, and result data
- returns structured summaries, QC views, and links to authoritative artifacts
- stays within the existing object-storage-first and FastAPI-centered architecture

The recommended implementation path is:

1. keep **Approach B** as the system backbone  
2. add a **published M365 Copilot declarative agent** as the analyst-facing enterprise UX  
3. expose the FastAPI backend as a secure **OpenAPI plugin/action surface**  
4. ground the agent on curated docs in **SharePoint / OneDrive** plus dynamic API calls  
5. optionally add a **Foundry agent** later for more advanced orchestration

---

## 1) The M365 Copilot Agent Approach

### What is a declarative agent?

A **declarative agent** in Microsoft 365 Copilot is a purpose-built Copilot experience configured through:

- **instructions** — what the agent is for, what it should and should not do
- **knowledge sources** — what enterprise content it can ground on
- **actions/plugins** — what external APIs or tools it can call
- **app metadata** — its identity, icon, name, and distribution settings

Conceptually, it is a **specialized Copilot front end** that runs on the Microsoft 365 Copilot orchestration stack instead of a custom chat shell. It inherits the Microsoft 365 Copilot experience, governance model, and user context, but its behavior is narrowed to a specific business scenario.

For DeepSeismic2, that means the agent would behave like a **seismic analyst assistant**, not a generic enterprise chatbot.

### Why this matters for DeepSeismic2

This is a good fit when the team wants:

- discovery inside the existing M365 user experience
- enterprise-grade sharing and governance
- grounding on SharePoint / OneDrive project content
- simple to moderate action-taking against backend APIs
- fast demo value for analysts, PMs, and leadership

This is a weaker fit when the team needs:

- complex multi-step orchestration loops
- long-running workflows
- large result sets
- rich custom UI beyond Adaptive Cards
- full control over model routing and reasoning strategy

### Key architectural constraints to respect

Current Microsoft guidance makes declarative agents powerful but bounded:

- grounding and tool calling are effectively **sequential**
- they are best for **single-step retrieval + action** patterns
- they are not ideal for iterative planning loops
- practical limits apply around **grounding volume, plugin response size, tokens, and timeout**

For the seismic use case, that means:

- use the agent for **query, summarize, route, explain, and present**
- keep heavy work in **FastAPI + Azure ML + storage**
- pre-aggregate results before returning them to Copilot

### Publishing options

There are three relevant publishing/governance paths.

| Path | Best use | Notes for DeepSeismic2 |
|---|---|---|
| **Copilot Studio** | Low-code agent authoring, actions, knowledge, ALM | Best primary authoring tool if the team wants a published M365 agent quickly. |
| **Teams Admin Center** | Org catalog upload and custom app distribution | Good for app-package distribution and controlled internal rollout. |
| **Microsoft 365 Admin Center** | Agent governance, approval, assignment, deployment, lifecycle management | Best control plane for enterprise rollout, user assignment, and compliance review. |

### Recommended publishing path for this project

Use:

1. **Copilot Studio** to build and test the seismic analyst agent  
2. **Microsoft 365 Admin Center** to govern and deploy it to the right pilot users  
3. **Teams Admin Center** only if the app package needs classic organizational catalog handling or Teams-specific rollout controls

### What capabilities does a published M365 Copilot agent get?

#### 1. Microsoft 365 grounding

The agent can ground on Microsoft 365 content sources such as:

- SharePoint sites
- OneDrive files
- supported enterprise connectors
- uploaded files

This is ideal for:

- project documentation
- model cards
- runbooks
- interpretation reports
- SOPs and data dictionaries

#### 2. User-context-aware enterprise access

The experience runs in the signed-in Microsoft 365 context, so the agent naturally fits:

- tenant governance
- enterprise access controls
- familiar Copilot entry points
- app discovery through the Agent Store

#### 3. Custom actions

The agent can call external APIs through plugins/actions. For DeepSeismic2, this is the critical bridge to:

- dataset inventory
- run status
- result retrieval
- QC artifact lookup
- interpretation summaries

#### 4. Rich responses

The agent can return:

- grounded natural-language answers
- citations to enterprise content
- **Adaptive Cards** for structured data presentation
- action buttons that deep-link into reports, dashboards, storage viewers, or internal apps

#### 5. Governance and compliance inheritance

The agent benefits from Microsoft 365 governance features, including:

- tenant-level admin visibility
- approval and deployment controls
- app catalog / agent store distribution
- existing Microsoft 365 security and compliance posture

---

## 2) Seismic Analyst Agent Design

### Agent persona

**Name:** DeepSeismic Analyst  
**Audience:** Petroleum geoscientists, interpretation leads, subsurface data managers, technical PMs  
**Role:** A grounded assistant that helps users discover datasets, understand runs, inspect results, summarize findings, and generate analyst-ready narrative from authoritative seismic artifacts.

### Scope

The agent should:

- explain what seismic datasets exist and what they contain
- answer questions about run manifests, model versions, and result availability
- summarize interpretation outputs produced by deterministic pipelines
- help users locate QC imagery, overlays, and reports
- compare recent runs at a metadata and result-summary level
- generate human-readable summaries for handoff and review

The agent should **not**:

- claim independent geological certainty
- invent faults, horizons, or facies not present in authoritative outputs
- replace model inference or expert interpretation
- act as the system of record for seismic data

### Core operating principle

**The agent speaks in analyst language, but grounds in machine-verifiable artifacts.**

That means every substantive answer should come from one or more of:

- curated project docs
- run manifests
- dataset metadata
- interpretation result summaries
- QC artifacts generated by the backend

### Example conversation starters

- “What seismic datasets are currently loaded for the Volve subset?”
- “Show me the latest UNet segmentation run and whether it completed successfully.”
- “Summarize the most recent interpretation results for inline 1420.”
- “What changed between the last two segmentation runs?”
- “Find the QC artifacts for the latest inference on the North Sea sample.”
- “Generate a short analyst handoff note for the latest results.”
- “What wells intersect this survey and what metadata do we have for them?”
- “Which runs used model version `unet-baseline-v1`?”

### Example prompts

#### Metadata and catalog

- “List the available surveys and their spatial coverage.”
- “Which dataset has the most recent preprocessing manifest?”
- “Do we have both SEG-Y and Zarr for this survey?”

#### Operational status

- “Did yesterday’s GPU inference finish?”
- “What failed in the latest preprocessing run?”
- “Which runs are still in progress?”

#### Interpretation support

- “Summarize the facies segmentation output for the latest run.”
- “Give me a non-technical summary I can send to stakeholders.”
- “What caveats should I mention when using this result?”

#### QC and evidence

- “Show the QC previews for run `run-volve-unet-01`.”
- “Do the latest masks cover the full inline range?”
- “Open the comparison view for the last two runs.”

### Knowledge sources

The agent should use a **hybrid grounding model**:

| Knowledge source | Type | Why it matters | Recommended host |
|---|---|---|---|
| Architecture and project docs | Unstructured | Explains system design and constraints | SharePoint document library |
| Model cards and runbooks | Unstructured | Provides safe usage guidance and caveats | SharePoint |
| Analyst reports and summaries | Unstructured | Gives domain narrative and reusable wording | SharePoint / OneDrive |
| Dataset catalog | Structured | Survey-level metadata, storage state, lineage | FastAPI action |
| Run manifests | Structured | Inference status, model version, timestamps, outputs | FastAPI action |
| Well metadata | Structured | Links surveys to wells and context | FastAPI action |
| Interpretation result summaries | Structured + generated | Analyst-facing explanation of deterministic results | FastAPI action |
| QC artifact index | Structured | Locates images, overlays, and previews | FastAPI action |

### Knowledge design recommendation

Split knowledge into two layers:

#### Layer A — Microsoft 365 grounding

Use SharePoint / OneDrive for relatively stable knowledge:

- architecture docs
- pipeline runbooks
- model cards
- glossary
- data dictionary
- interpretation methodology notes
- prior generated analyst reports

#### Layer B — live backend actions

Use API actions for dynamic operational data:

- latest runs
- run health
- dataset readiness
- result file locations
- QC previews
- comparison summaries

This split is important because dynamic operational data is a poor fit for document-grounding alone.

### Custom actions the agent should expose

| Action | Purpose | Example backend route |
|---|---|---|
| `searchDatasets` | Search seismic catalog by survey, basin, status, format | `GET /api/datasets` |
| `getDatasetDetail` | Return authoritative dataset metadata | `GET /api/datasets/{dataset_id}` |
| `listRuns` | List preprocessing / inference runs with filters | `GET /api/runs` |
| `getRunStatus` | Return run state, duration, failure reason, outputs | `GET /api/runs/{run_id}` |
| `compareRuns` | Compare two runs on model version, inputs, outputs, QC counts | `GET /api/runs/compare` |
| `getResultSummary` | Summarize interpretation outputs for a run or region | `GET /api/results/{result_id}/summary` |
| `getQcArtifacts` | Retrieve preview images and overlay references | `GET /api/results/{result_id}/qc` |
| `generateAnalystNote` | Produce a controlled narrative from structured result facts | `POST /api/results/{result_id}/analyst-note` |
| `getWellContext` | Return wells intersecting a survey or subvolume | `GET /api/wells` |
| `openArtifactLink` | Return signed or application links to reports and previews | `GET /api/artifacts/{artifact_id}` |

### How the agent integrates with the Approach B FastAPI backend

The agent should not reach into ADLS directly. It should call the **same thin FastAPI service** already planned in Approach B.

That preserves the existing architecture:

- **ADLS/Blob** stays the system of record
- **Azure ML** stays the inference engine
- **FastAPI** remains the metadata/results abstraction layer
- **M365 Copilot agent** becomes a governed enterprise UX layer

Recommended interaction flow:

1. user asks a question in M365 Copilot  
2. agent checks its instructions and knowledge  
3. if dynamic data is required, it calls a FastAPI action  
4. FastAPI queries catalog metadata, manifests, result indices, or QC assets  
5. FastAPI returns compact structured JSON  
6. agent responds with grounded narrative plus citations / Adaptive Card

---

## 3) Implementation Architecture

### High-level architecture

```text
Petroleum geoscientist
        |
        v
Microsoft 365 Copilot
        |
        v
DeepSeismic Analyst declarative agent
   |                     |
   |                     +--> SharePoint / OneDrive knowledge
   |
   +--> OpenAPI plugin / custom actions
                 |
                 v
            FastAPI backend
           /       |       \
          v        v        v
      catalog   run state  result/QC index
          \        |        /
                 ADLS + Azure ML outputs
```

### Copilot Studio declarative agent manifest structure

Exact packaging can vary by authoring surface, but the logical structure should include:

- **agent identity**
  - name
  - description
  - icons
- **instructions**
  - purpose
  - boundaries
  - response style
  - safety rules
- **conversation starters**
- **knowledge sources**
  - SharePoint libraries
  - OneDrive folders
- **actions/plugins**
  - OpenAPI-backed FastAPI actions
- **response UX**
  - Adaptive Card templates

### Suggested declarative definition shape

```yaml
name: DeepSeismic Analyst
description: Grounded assistant for seismic dataset discovery, run inspection, QC review, and interpretation summaries.
instructions: |
  You assist petroleum geoscientists working with seismic datasets and model outputs.
  Use project knowledge for architecture, methods, and runbooks.
  Use actions for live run, catalog, well, and result data.
  Never infer geology that is not supported by returned evidence.
  Present uncertainty and caveats explicitly.
conversation_starters:
  - What seismic datasets are available for the Volve subset?
  - Show me the latest UNet inference run.
  - Summarize the most recent interpretation outputs.
  - Find QC artifacts for the latest completed run.
knowledge_sources:
  - sharepoint://DeepSeismic2/ProjectDocs
  - sharepoint://DeepSeismic2/Runbooks
  - onedrive://DeepSeismic2/Reports
actions:
  - seismicCatalogPlugin
  - seismicRunPlugin
  - seismicResultsPlugin
response_rules:
  - cite sources when grounded on documents
  - prefer compact tables and cards for lists
  - ask the backend for dynamic data instead of guessing
  - never present model output as confirmed geology
```

### Recommended instruction set

The instruction layer should explicitly encode:

#### Role

- assistant to geoscientists and interpretation stakeholders
- translator between technical artifacts and human analyst questions

#### Boundaries

- do not claim subsurface truth from LLM reasoning alone
- only summarize deterministic model outputs and curated knowledge
- if evidence is incomplete, say so

#### Tool-use policy

- use knowledge for stable documentation
- use actions for fresh operational facts
- prefer result summaries over raw large payloads

#### Response policy

- concise first answer
- cite evidence where available
- highlight caveats, lineage, and confidence limits

### API plugin definition

The FastAPI backend should publish a **clean OpenAPI 3.x spec** with:

- stable operation IDs
- compact schemas
- bounded list sizes
- filter parameters for survey, run, model, date, and status
- response payloads designed for LLM consumption

### Suggested FastAPI API surface

```yaml
openapi: 3.1.0
info:
  title: DeepSeismic Analyst API
  version: 1.0.0
servers:
  - url: https://deepseismic-api.contoso.com
paths:
  /api/datasets:
    get:
      operationId: searchDatasets
      summary: Search seismic datasets
  /api/datasets/{dataset_id}:
    get:
      operationId: getDatasetDetail
      summary: Get dataset metadata
  /api/runs:
    get:
      operationId: listRuns
      summary: List preprocessing and inference runs
  /api/runs/{run_id}:
    get:
      operationId: getRunStatus
      summary: Get run status and outputs
  /api/runs/compare:
    get:
      operationId: compareRuns
      summary: Compare two runs
  /api/results/{result_id}/summary:
    get:
      operationId: getResultSummary
      summary: Get summarized interpretation result
  /api/results/{result_id}/qc:
    get:
      operationId: getQcArtifacts
      summary: Get QC previews and overlays
  /api/results/{result_id}/analyst-note:
    post:
      operationId: generateAnalystNote
      summary: Generate a controlled analyst note from result facts
```

### API design guidance for Copilot compatibility

Design the backend specifically for agent use:

- return **small, opinionated summaries**, not raw cubes
- precompute key result facts
- limit list responses
- include stable IDs and deep links
- include user-safe display text
- include explicit caveats in result summaries

Example result summary payload:

```json
{
  "resultId": "res-volve-unet-01",
  "runId": "run-volve-unet-01",
  "datasetId": "volve-survey-a",
  "modelVersion": "unet-baseline-v1",
  "status": "completed",
  "summary": "Segmentation completed successfully. Output generated for the targeted inline range. QC previews available.",
  "keyFindings": [
    "Prediction mask written to authoritative results storage",
    "QC overlays generated for 12 sampled slices",
    "No backend validation errors reported"
  ],
  "caveats": [
    "This summary describes model output, not confirmed geological truth",
    "Analyst review is required before interpretation sign-off"
  ],
  "links": [
    {
      "label": "Open QC preview",
      "url": "https://deepseismic.contoso.com/qc/res-volve-unet-01"
    }
  ]
}
```

### Authentication model

#### Recommended choice: Microsoft Entra ID SSO to FastAPI

For the M365 Copilot agent path, the best enterprise model is:

- user signs into Microsoft 365
- agent action authenticates with **Microsoft Entra ID SSO**
- FastAPI validates the user token
- FastAPI enforces app roles / group membership / project authorization
- FastAPI uses its own managed identity server-side to reach storage or other Azure services

This gives the cleanest separation:

- **user identity** for authorization
- **backend managed identity** for resource access

#### Why this is better than API keys

- avoids secret sprawl
- matches enterprise governance
- fits user-scoped data access
- supports future per-project authorization

#### Backend auth responsibilities

FastAPI should:

- validate Entra-issued JWTs
- map users to project or survey access
- redact storage paths where needed
- generate short-lived links or deep links instead of exposing raw storage internals

### Knowledge and grounding configuration

#### Recommended document layout

Create a dedicated SharePoint site or library for the agent:

- `Project Docs/`
- `Runbooks/`
- `Model Cards/`
- `Glossary/`
- `Interpretation Reports/`
- `Demo Narratives/`

#### What belongs in SharePoint vs the API

| Put in SharePoint / OneDrive | Put behind FastAPI action |
|---|---|
| architecture docs | dataset catalog |
| runbooks | live run status |
| model cards | latest results |
| glossary | QC artifact references |
| published reports | run comparisons |
| curated interpretation notes | well intersections and metadata |

### Adaptive Card responses

Adaptive Cards are the right rich response mechanism for:

- dataset summaries
- run status summaries
- QC preview selections
- interpretation result snapshots
- quick navigation actions

### Recommended seismic cards

#### 1. Dataset summary card

Fields:

- survey name
- basin / area
- SEG-Y available?
- Zarr available?
- last processed date
- linked wells

#### 2. Run status card

Fields:

- run ID
- dataset
- model version
- status
- submitted time
- completed time
- outputs produced
- failure reason if present

#### 3. Result summary card

Fields:

- result ID
- run ID
- summary text
- key findings
- caveats
- buttons for QC preview and report

#### 4. QC card

Fields:

- preview thumbnails
- slice identifiers
- overlay availability
- open-in-viewer action

### Example card shape

```json
{
  "type": "AdaptiveCard",
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.5",
  "body": [
    { "type": "TextBlock", "size": "Medium", "weight": "Bolder", "text": "Latest Run Status" },
    { "type": "FactSet", "facts": [
      { "title": "Run", "value": "run-volve-unet-01" },
      { "title": "Dataset", "value": "volve-survey-a" },
      { "title": "Model", "value": "unet-baseline-v1" },
      { "title": "Status", "value": "Completed" }
    ]},
    { "type": "TextBlock", "wrap": true, "text": "QC overlays were generated and are ready for analyst review." }
  ],
  "actions": [
    { "type": "Action.OpenUrl", "title": "Open QC Preview", "url": "https://deepseismic.contoso.com/qc/res-volve-unet-01" }
  ]
}
```

---

## 4) The Lift Assessment

### What changes from the thin slice to a full published M365 agent?

The thin slice already assumed:

- object-storage-first data architecture
- FastAPI backend for metadata and results
- grounded LLM assistance

The full M365 agent adds four major workstreams:

1. **enterprise packaging and publishing**
2. **action design and OpenAPI hardening**
3. **knowledge curation for M365 grounding**
4. **governed UX design for analyst workflows**

### What the team already has

Based on the current architecture direction, the team already has the right conceptual backbone:

- ADLS/Blob as source of truth
- Azure ML for GPU inference
- FastAPI as the integration seam
- Azure OpenAI / Foundry as the LLM posture
- clear safety boundary: LLM assists, CNNs interpret

### What is new work

New work required for a true published M365 agent:

- turning FastAPI into a Copilot-friendly action surface
- setting up Entra SSO for agent-to-backend calls
- curating SharePoint-hosted knowledge
- authoring agent instructions and conversation design
- building Adaptive Card responses
- admin packaging, approval, deployment, and pilot rollout

### Effort estimate

#### A. Backend API work — **5 to 8 person-days**

Includes:

- finalize resource model for datasets, runs, results, wells, artifacts
- expose clean OpenAPI 3.x definitions
- add bounded response models for Copilot-friendly summaries
- implement comparison and analyst-note endpoints
- add auth, authorization, and telemetry

#### B. Agent configuration and UX — **3 to 5 person-days**

Includes:

- Copilot Studio agent setup
- instructions and safety boundaries
- conversation starters
- action wiring
- Adaptive Card templates
- prompt tuning for grounded behavior

#### C. Knowledge curation — **3 to 5 person-days**

Includes:

- identify source docs
- move / clean content into SharePoint or OneDrive structure
- create glossary, model card, and runbook material where missing
- remove stale or duplicate content that would confuse grounding

#### D. Testing, admin publishing, and pilot rollout — **3 to 5 person-days**

Includes:

- test prompts and tool calls
- permission validation
- Responsible AI review
- pilot assignment in Microsoft 365 Admin Center
- packaging and deployment checks across M365 surfaces

### Total estimated lift

| Scope | Estimate |
|---|---|
| **Credible internal pilot** | **14 to 23 person-days** |
| **Polished demo-grade published experience** | **20 to 30 person-days** |

### Recommended sequencing

#### Phase 1 — Backend readiness

- implement stable FastAPI routes
- shape responses for agent consumption
- add auth and telemetry

#### Phase 2 — Knowledge readiness

- curate SharePoint / OneDrive sources
- create glossary, runbook, and model-card material

#### Phase 3 — Agent build

- configure Copilot Studio declarative agent
- connect knowledge and actions
- build cards and test prompts

#### Phase 4 — Admin rollout

- package and publish
- assign pilot users
- validate M365 surfaces

### Dependencies and prerequisites

#### Microsoft-side prerequisites

- Microsoft 365 Copilot licensing or metering path for target users
- Copilot Studio licensing for makers
- tenant admin support
- access to Microsoft 365 Admin Center and, if needed, Teams Admin Center

#### Engineering prerequisites

- public HTTPS FastAPI endpoint reachable by Microsoft 365 services
- Entra app registrations and SSO configuration
- stable OpenAPI specification
- storage-safe artifact link strategy
- telemetry and error handling

#### Content prerequisites

- curated SharePoint site or document library
- model card and workflow documentation
- controlled set of reports for grounding

### Primary risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Agent overreaches beyond evidence | Dangerous in geology | Strong instructions, explicit caveats, API-grounded summaries |
| Backend returns too much data | Copilot limits and latency | Pre-aggregate, paginate server-side, return compact summaries |
| Knowledge corpus is noisy | Bad grounding quality | Curate SharePoint content aggressively |
| Auth complexity delays rollout | Common enterprise blocker | Choose Entra SSO early and implement first |
| Users expect full interpretation app behavior | Copilot is not a seismic workstation | Position it as analyst copilot, not primary interpretation UI |

---

## 5) Alternative: Microsoft Foundry Agent

### How this differs in Foundry

A **Foundry agent** would shift the center of gravity from Microsoft 365 UX to Azure AI runtime control.

Compared with a declarative M365 agent, Foundry gives the team more control over:

- model choice
- orchestration logic
- tool chains
- memory strategy
- observability
- custom code
- multi-step workflows

It is better suited if the team wants:

- richer orchestration than declarative agents support
- custom reasoning pipelines
- larger non-M365 integration footprint
- deeper engineering control
- a future analyst workbench outside M365

### Foundry implementation shape

In a Foundry-first design:

- the agent runs in Foundry Agent Service
- FastAPI is still the main backend for seismic metadata and results
- the agent calls FastAPI through OpenAPI tools or custom functions
- optional memory, workflow logic, or additional tools are layered in Foundry
- the agent can later be surfaced into M365 if desired

### When to choose Copilot Studio vs Foundry vs both

| Choice | Best when | DeepSeismic2 fit |
|---|---|---|
| **Copilot Studio / declarative M365 agent** | Fastest route to a published M365 analyst experience | Best for the immediate ask |
| **Foundry agent** | Need more orchestration, tool chains, model control, or non-M365 UX | Best if analyst workflows become more complex |
| **Both** | Need enterprise M365 reach plus deeper Azure-side orchestration | Best medium-term target |

### Recommended platform choice for this project

#### Near term

Use **Copilot Studio + FastAPI** for the first published analyst agent.

Why:

- fastest path to an M365-native analyst experience
- leverages Microsoft 365 grounding and governance immediately
- fits the current thin-slice goal
- keeps the backend aligned with Approach B

#### Medium term

Add a **Foundry agent** if the team needs:

- multi-step analyst workflows
- richer planning and tool orchestration
- more advanced evaluation and observability
- a non-M365 channel or dedicated analyst application

### Can they share the same backend?

Yes. They should.

The best long-term architecture is a **shared backend contract**:

- **FastAPI** remains the stable domain API
- **Copilot Studio / M365 agent** consumes it as OpenAPI actions
- **Foundry agent** consumes the same API as OpenAPI tools or custom functions

That gives the team:

- one source of operational truth
- one authorization layer
- reusable schemas
- channel flexibility without backend duplication

---

## Recommended Decision

Build the first full analyst experience as a **published Microsoft 365 Copilot declarative agent backed by the FastAPI service from Approach B**.

Specifically:

1. keep storage, compute, and deterministic interpretation in the existing object-storage-first architecture  
2. use **SharePoint / OneDrive** for curated static grounding  
3. use **FastAPI OpenAPI actions** for live dataset, run, and result data  
4. use **Entra SSO** for secure backend access  
5. use **Adaptive Cards** for dataset, run, and QC summaries  
6. treat **Foundry** as the next-step platform for advanced orchestration, not as a replacement for the backend

This produces a credible AI-native analyst experience without violating the core project safety principle: **LLMs assist interpretation workflows; they do not replace seismic science or deterministic model outputs.**
