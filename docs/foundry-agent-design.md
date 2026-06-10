# Foundry Agent Design for DeepSeismic2

**Author:** Lambert  
**Requested by:** jospaid  
**Date:** 2026-06-09T22:28:44-05:00

## Executive Summary

DeepSeismic2 should now take a **Foundry-first** path for the analyst AI experience.

The right PoC is not an LLM pretending to interpret subsurface truth on its own. The right PoC is an **AI-native analyst assistant** built on **Azure AI Foundry Agent Service**, grounded by:

- **cheap object storage** for seismic assets and derived artifacts
- **on-demand CPU and GPU compute** for deterministic preprocessing and inference
- **FastAPI tools** for live dataset, run, QC, and results access
- **Azure AI Search over markdown knowledge** for methods, model cards, glossary, and workflow guidance

This gives the team the full modernization story:

1. keep seismic data in low-cost object storage  
2. run preprocessing and inference only when needed  
3. expose authoritative metadata and outputs through a thin API  
4. add a multi-step analyst assistant that can inspect, summarize, compare, and recommend without inventing geology

The primary user experience should be a **Foundry agent with a code-first definition checked into the repo**. M365 Copilot can remain an optional later surfacing layer rather than the initial runtime center of gravity.

---

## 1) Foundry Agent Architecture

### Why Foundry is now the primary fit

Foundry is the better primary platform for this PoC because the user wants:

- **multi-step workflows**
- **full model choice per task**
- **code-first, version-controlled agent definition**
- **stronger control of tool calling and memory**
- **a non-M365-first analyst experience**

This aligns better with Azure AI Foundry Agent Service than with a declarative M365 Copilot agent.

### Core architecture

```text
Analyst UI (web chat / Streamlit / Gradio / notebook / CLI)
                         |
                         v
             Azure AI Foundry Agent Service
                 |         |          |
                 |         |          +--> Azure AI Search (markdown knowledge)
                 |         |
                 |         +--> Foundry tools / functions
                 |                    |
                 |                    v
                 |               FastAPI backend
                 |              /      |       \
                 v             v       v        v
         thread memory   dataset catalog  run/QC data  result summaries
                                                \
                                                 v
                                     ADLS/Blob + Azure ML outputs
```

### Foundry Agent Service configuration

The agent should be defined in Python and committed to the repo so the full behavior is reviewable and repeatable.

Recommended configuration responsibilities:

- create or connect to the Foundry project
- define the agent instructions
- register tool/function definitions
- choose default model and per-tool/task routing rules
- configure evaluation hooks and tracing
- manage thread/session lifecycle

### Recommended code-first definition shape

Suggested repo-owned configuration:

```python
agent_config = {
    "name": "DeepSeismic Analyst",
    "description": "Grounded assistant for seismic workflow inspection, QC review, run comparison, and analyst summaries.",
    "instructions": """
    You assist petroleum analysts working with deterministic seismic workflows.
    Use Azure AI Search for project knowledge such as methods, model cards, glossary, and runbooks.
    Use FastAPI-backed tools for live operational facts.
    Never claim subsurface truth that is not supported by tool results or indexed documentation.
    Separate observed evidence, interpretation guidance, and recommended next steps.
    When uncertainty exists, say so explicitly.
    """,
    "default_model": "gpt-4.1",
    "tools": [
        "searchDatasets",
        "getRunStatus",
        "getQcArtifacts",
        "getResultSummary",
        "compareRuns",
        "generateAnalystNote",
        "getWellContext",
        "searchKnowledge"
    ]
}
```

Exact SDK calls may change over time, but the design intent should stay stable: **agent identity, instructions, model policy, tools, memory, and evaluation all live in code**.

### Tool/function definitions that call FastAPI

The Foundry agent should not read blob storage directly. It should call a thin FastAPI service that returns compact, authoritative summaries.

Recommended tool surface:

| Tool | Purpose | Backend route |
|---|---|---|
| `searchDatasets` | Find loaded datasets, formats, readiness, and coverage | `GET /api/datasets` |
| `getDatasetDetail` | Retrieve authoritative dataset metadata | `GET /api/datasets/{dataset_id}` |
| `listRuns` | List preprocess/inference runs | `GET /api/runs` |
| `getRunStatus` | Return run state, timing, outputs, and failure reason | `GET /api/runs/{run_id}` |
| `compareRuns` | Compare run inputs, model version, outputs, and QC counts | `GET /api/runs/compare` |
| `getQcArtifacts` | Return preview images, overlays, and QC references | `GET /api/results/{result_id}/qc` |
| `getResultSummary` | Return summarized deterministic output facts | `GET /api/results/{result_id}/summary` |
| `generateAnalystNote` | Produce a controlled handoff note from structured facts | `POST /api/results/{result_id}/analyst-note` |
| `getWellContext` | Link wells and contextual metadata to a dataset or region | `GET /api/wells` |
| `searchKnowledge` | Query Azure AI Search over markdown corpus | `POST /api/knowledge/search` |

### Tool design rules

All tool responses should be:

- compact
- structured
- bounded in size
- explicit about caveats
- safe for LLM consumption
- linked to authoritative IDs and deep links

Example result summary contract:

```json
{
  "resultId": "res-volve-unet-01",
  "runId": "run-volve-unet-01",
  "datasetId": "volve-survey-a",
  "modelVersion": "unet-baseline-v1",
  "status": "completed",
  "summary": "Inference completed successfully for the targeted Volve subset. QC previews and overlays are available for analyst review.",
  "keyFindings": [
    "Prediction mask written to results storage",
    "QC overlays generated for 12 sampled slices",
    "No validation errors reported by the backend"
  ],
  "caveats": [
    "This describes model output, not confirmed geological truth",
    "Analyst review is required before sign-off"
  ]
}
```

### Model selection strategy

Foundry-first is attractive because model selection can be explicit rather than hidden.

Recommended task-to-model policy:

| Task type | Preferred model behavior | Example model policy |
|---|---|---|
| Tool planning, short orchestration, extraction | fast, low-cost, reliable function calling | smaller general model |
| Multi-step reasoning across QC + results + recommendations | higher reasoning quality | stronger general model |
| Final analyst note / executive summary | stronger writing + instruction following | stronger generation model |
| Search query reformulation / retrieval rewrite | fast and cheap | smaller general model |
| Embeddings for AI Search | embedding model | Azure OpenAI embeddings |

Suggested PoC policy:

- use a **cost-efficient default model** for ordinary turns
- escalate to a **stronger reasoning model** only when the task requires comparison, synthesis, or recommendation
- keep **embeddings separate** from generation models
- preserve routing in code so the team can change model choice per task without redesigning the agent

Practical examples:

- “What data is loaded?” → fast default model + `searchDatasets`
- “Compare the last two Volve runs and explain whether the anomaly looks trustworthy” → stronger reasoning model + `compareRuns` + `getQcArtifacts` + `getResultSummary`
- “Write a handoff note for the subsurface team” → stronger writing model + `generateAnalystNote`

### Session and memory management

The PoC expects conversations that span QC, comparison, summarization, and recommendations.

Recommended memory approach:

#### Working memory in the Foundry thread

Store short-lived context such as:

- current dataset
- selected run IDs
- latest result ID
- chosen domain perspective
- previous step outputs

#### Durable conversation state outside the model

Persist structured session state in the app layer, keyed by thread/session ID:

- user or analyst ID
- dataset/run selections
- saved handoff notes
- user-approved next actions
- evaluation metadata

#### Summary compression

For longer threads:

- periodically summarize prior turns into a compact state block
- keep authoritative IDs and caveats
- do not rely on unconstrained conversational memory alone

This prevents drift while preserving multi-step continuity.

### Safety boundary for memory

Memory must preserve workflow state, not invent interpretation state.

Allowed memory examples:

- “current dataset = Volve subset”
- “latest completed run = run-volve-unet-01”
- “user asked for geology interpretation perspective”

Not allowed as durable truth:

- “the anomaly is definitely a reservoir feature”
- “the model proved a fault exists”

### Evaluation and observability setup

Foundry-first should include evaluation from the start.

Track:

- groundedness
- citation/use-of-evidence quality
- tool selection correctness
- task completion rate
- hallucination rate
- latency
- token cost
- failure modes by tool and prompt

Recommended observability stack:

- Azure AI Foundry evaluation workflows for prompt/task evaluation
- Application Insights for API, tool, and UI telemetry
- OpenTelemetry tracing across UI → agent → FastAPI → downstream calls
- persisted test set of analyst questions and expected evidence sources

Recommended eval scenarios:

1. dataset inventory lookup  
2. latest run status retrieval  
3. QC summary generation  
4. run-to-run comparison  
5. geology explanation grounded in docs  
6. geophysics caveat generation  
7. next-step recommendation with evidence and uncertainty

Success criteria:

- answer references tool/data evidence when making factual claims
- answer separates observation from recommendation
- answer never claims geology certainty without evidence
- wrong or missing tool data leads to explicit uncertainty, not invention

---

## 2) Knowledge and Grounding Without SharePoint

### Grounding posture

The user has explicitly chosen **no SharePoint dependency** for the initial experience.

Grounding should therefore use:

1. **Azure AI Search over markdown content** for stable knowledge  
2. **FastAPI tools** for live operational facts

This is a better fit for a code-first repo and blob-backed document pipeline.

### What belongs in the Azure AI Search index

Index markdown and related text content such as:

- architecture docs
- model cards
- glossary
- methodology notes
- inference caveats
- runbooks
- analyst interpretation guidance
- generated summaries approved for reuse

Recommended source locations:

- `docs/`
- `knowledge/` or similar future repo folder
- exported markdown from blob storage
- curated generated notes promoted into the knowledge corpus

### How to index content from repo or blob

Two credible PoC options:

#### Option A — Repo-driven indexing

- markdown lives in git
- CI job extracts and chunks changed `.md` files
- embeddings are generated during indexing
- documents are pushed into Azure AI Search

Best when the team wants version-controlled grounding content.

#### Option B — Blob-driven indexing

- approved markdown is copied to blob storage
- Azure AI Search indexer or custom ingestion job reads blob content
- the same chunking and embedding pipeline is applied

Best when operations want content outside the app container or repo.

### Recommended PoC choice

Use **repo-authored markdown as the primary source**, with optional blob mirroring later.

Why:

- simplest code-first workflow
- easier review and pull-request governance
- aligns with the user’s version-control requirement
- avoids introducing SharePoint administration into the initial path

### Suggested search index schema

| Field | Purpose |
|---|---|
| `id` | stable chunk ID |
| `title` | document title |
| `path` | repo or blob source path |
| `section` | heading / chunk label |
| `content` | chunk text |
| `tags` | domain tags such as geology, geophysics, geoengineering, methodology |
| `documentType` | model-card, glossary, runbook, architecture, report |
| `sourceCommit` | git provenance for repo content |
| `updatedAt` | freshness |
| `contentVector` | embeddings |

### Chunking guidance

Keep chunks organized by semantic section, not arbitrary token cuts.

Good chunk boundaries:

- heading-level sections
- model caveats
- glossary entries
- methodology steps
- report summary sections

Also store:

- source path
- heading
- domain tags
- revision metadata

That makes citations and discipline routing much cleaner.

### RAG pattern

Recommended retrieval pattern per knowledge-heavy turn:

1. agent classifies whether the question needs static knowledge, live data, or both  
2. if static knowledge is needed, call `searchKnowledge` against Azure AI Search  
3. retrieve top relevant chunks with citations and metadata  
4. if live data is also needed, call FastAPI tools  
5. inject both retrieved knowledge and live tool outputs into the final response generation step  
6. require the answer to distinguish:
   - documented knowledge
   - live operational facts
   - recommendations or caveats

### Dynamic grounding via FastAPI tools

Static RAG is not enough for this use case.

Questions like these require live tools:

- “What completed today?”
- “Which Volve run failed?”
- “What changed between the last two runs?”
- “What QC artifacts exist for this result?”

That is why the grounding design must be **hybrid**:

- **AI Search** for domain and workflow knowledge
- **FastAPI** for current truth

### Response policy for grounded output

Every substantive answer should follow this pattern:

1. **Observed evidence** — from tools or indexed docs  
2. **Interpretation help** — what that evidence likely means in analyst language  
3. **Caveats** — what remains uncertain or requires expert review  
4. **Recommended next step** — what the analyst should do next

This structure is especially important because hallucination in geology is dangerous.

---

## 3) Multi-Step Workflow Design

### Target workflow: “Analyze the latest Volve run end-to-end”

The PoC should demonstrate a real analyst conversation, not a single-turn Q&A.

### Workflow steps

#### Step 1 — Check what data is loaded

Use:

- `searchDatasets`

Goal:

- confirm the relevant Volve dataset exists
- identify whether SEG-Y, Zarr, manifests, and result artifacts are present
- determine the latest relevant dataset/run identifiers

#### Step 2 — QC the preprocessing

Use:

- `getRunStatus`
- `getQcArtifacts`

Goal:

- confirm preprocessing completed
- surface warnings, failures, missing artifacts, or geometry issues
- identify whether the result is suitable for downstream interpretation review

#### Step 3 — Summarize inference results

Use:

- `getResultSummary`

Goal:

- explain what the deterministic model produced
- highlight any caveats, missing coverage, or validation warnings
- translate technical result metadata into analyst language

#### Step 4 — Generate analyst handoff note

Use:

- `generateAnalystNote`

Goal:

- produce a concise note for downstream human review
- preserve evidence, caveats, and next actions
- avoid overclaiming interpretation confidence

#### Step 5 — Recommend next steps

Use:

- prior tool outputs
- optional `searchKnowledge` for methodology/runbook guidance

Goal:

- recommend what the analyst should do next
- distinguish operational next steps from geoscience review next steps

### How the agent plans and executes

The Foundry agent should follow an explicit workflow policy:

1. identify the target dataset/run if not already known  
2. gather live status and QC facts before summarizing results  
3. only generate narrative after authoritative facts are collected  
4. recommend next steps only after surfacing caveats and gaps

The agent should be allowed to perform these tool calls automatically for low-risk read operations.

### Example execution trace in human terms

User: “Analyze the latest Volve run end-to-end.”

Agent behavior:

1. finds the latest Volve dataset and relevant run  
2. checks whether the run completed and whether QC artifacts exist  
3. retrieves the result summary  
4. generates a handoff note grounded in those facts  
5. returns:
   - a concise summary
   - observed issues or caveats
   - recommended next actions by discipline

### Error handling

The workflow must degrade safely.

#### Missing dataset

Response:

- say the target dataset was not found
- offer the closest matching dataset or ask the user to choose

#### Run incomplete or failed

Response:

- do not generate a confident interpretation summary
- instead summarize the operational issue and recommended remediation

#### QC artifacts missing

Response:

- state that confidence is limited because QC evidence is incomplete
- recommend regenerating or locating QC outputs before analyst sign-off

#### Result summary unavailable

Response:

- say the backend did not return result facts
- avoid speculating from partial metadata alone

### User confirmation points

Because the PoC is mostly read-heavy, most steps can auto-execute.

Ask for user confirmation when:

- switching from one dataset/run to another ambiguous candidate
- generating a final handoff note intended for sharing
- storing or publishing a recommended action list
- escalating from summary to any write-back workflow in the future

### Workflow output shape

A strong end-to-end response should contain:

- current dataset/run
- preprocessing and QC status
- result summary
- caveats and confidence limits
- analyst handoff note
- next-step recommendations grouped by discipline

---

## 4) Domain Expert Personas Inside the Agent

### Why personas matter

The same seismic output can prompt different questions:

- **Ash / geophysics:** signal quality, amplitude reliability, acquisition/processing caveats
- **Kane / geology:** facies meaning, depositional interpretation, structural context
- **Brett / geoengineering:** production or operational implications, uncertainty for development decisions

The Foundry agent should not pretend these are the same question.

### Recommended persona architecture

Use one shared agent identity with **discipline-aware response modes**.

Recommended design:

- common root safety and grounding instructions
- discipline-specific sub-prompts
- optional discipline-specific retrieval filters
- optional discipline-specific tool wrappers or post-processors

This keeps a unified user experience while still respecting domain differences.

### Shared root instruction

Every discipline mode should inherit:

- LLMs do not replace deterministic seismic interpretation
- tool and document evidence outrank prior conversational guesses
- recommendations must include caveats
- when discipline evidence is weak, say so

### Discipline-specific behavior

#### Geophysics mode — Ash

Focus on:

- data quality
- signal behavior
- amplitude reliability
- acquisition/processing caveats
- whether the anomaly is trustworthy enough for interpretation

Example user question:

> Is this amplitude anomaly reliable?

Expected response posture:

- inspect QC artifacts and run caveats first
- discuss possible reliability limits
- avoid claiming geological meaning before signal trust is established

#### Geology mode — Kane

Focus on:

- facies meaning
- depositional interpretation
- structural context
- lithologic possibilities
- interpretation caveats given model limits

Example user question:

> What does this facies classification mean geologically?

Expected response posture:

- use retrieved methodology/model-card/glossary material
- explain plausible geological meaning in constrained language
- distinguish model label from confirmed subsurface interpretation

#### Geoengineering mode — Brett

Focus on:

- operational significance
- reservoir-development implications
- impact on planning or production decisions
- what additional evidence is needed before action

Example user question:

> What’s the production impact?

Expected response posture:

- translate results into engineering-relevant risk and next-step language
- avoid claiming direct production outcomes from seismic output alone
- recommend what additional reservoir, well, or petrophysical evidence is needed

### Could the agent route to specialized sub-prompts or tools?

Yes, and it should.

Recommended routing patterns:

1. **Prompt routing** — choose a discipline-specific instruction block  
2. **Retrieval routing** — filter Azure AI Search by tags such as `geology`, `geophysics`, `geoengineering`  
3. **Tool routing** — prefer QC tools for geophysics questions, glossary/method docs for geology questions, and result-plus-context summaries for geoengineering questions

### Suggested routing heuristic

| User signal | Default perspective |
|---|---|
| amplitude, waveform, QC, reliability, noise | geophysics |
| facies, depositional, lithology, structure | geology |
| production, completion, reservoir impact, operations | geoengineering |

If ambiguous:

- answer briefly with the default best guess
- explicitly offer alternate discipline views

Example:

> I can explain this from a geophysics, geology, or geoengineering perspective. Here is the geology-first view; I can also give the signal-quality or production-impact view.

---

## 5) Implementation Plan

### Python packages

Recommended packages for the Foundry-first path:

- `azure-ai-projects`
- `azure-ai-inference`
- `azure-identity`
- `azure-search-documents`
- `fastapi`
- `uvicorn`
- `pydantic`
- `httpx`
- `opentelemetry-api`
- `opentelemetry-sdk`
- `azure-monitor-opentelemetry`
- `python-dotenv`

Likely existing domain/runtime packages elsewhere in the system:

- `segyio`
- `xarray`
- `zarr`
- `numpy`
- `pandas`

Optional PoC UI packages:

- `streamlit` or `gradio`

### Suggested project structure

```text
src/
  deepseismic/
    api/
      main.py
      routes/
        datasets.py
        runs.py
        results.py
        wells.py
        knowledge.py
    agent/
      config.py
      foundry_client.py
      agent_definition.py
      session_store.py
      model_routing.py
      tools/
        datasets.py
        runs.py
        results.py
        knowledge.py
      prompts/
        system.md
        personas/
          ash-geophysics.md
          kane-geology.md
          brett-geoengineering.md
      evals/
        datasets.jsonl
        workflows.jsonl
        groundedness.py
    ui/
      streamlit_app.py
tests/
  agent/
  api/
docs/
  foundry-agent-design.md
```

### Local development flow

Recommended local loop:

1. run FastAPI locally  
2. run or mock Azure AI Search queries  
3. start the lightweight demo UI  
4. execute scripted eval prompts against the Foundry agent  
5. inspect traces, grounding quality, and tool outputs

### How to test locally before deploying

Test layers separately:

#### 1. Backend contract tests

Validate:

- route availability
- schema shape
- bounded payloads
- caveats present where required

#### 2. Search relevance tests

Validate:

- glossary questions retrieve glossary chunks
- model caveat questions retrieve model-card chunks
- discipline tags narrow retrieval appropriately

#### 3. Agent workflow tests

Validate:

- latest run workflow completes end-to-end
- missing QC artifact path fails safely
- ambiguous dataset selection triggers clarification
- responses separate evidence from recommendation

#### 4. Persona tests

Validate:

- the same prompt can be answered from geophysics, geology, and geoengineering perspectives with clearly different emphasis

### Deployment to Azure

Two credible PoC deployment shapes:

#### Option A — Container Apps

Use for:

- FastAPI backend
- lightweight custom UI
- supporting ingestion/indexing jobs

Why it fits:

- fast PoC deployment
- container-native
- cost-conscious for small workloads

#### Option B — App Service

Use for:

- simpler always-on web/API hosting if the team wants a more familiar web deployment path

Why it fits:

- straightforward app hosting
- simpler operational posture for small apps

### Recommended deployment split

- **Foundry Agent Service** for the agent runtime
- **Container Apps** for FastAPI and optional demo UI
- **Azure AI Search** for markdown grounding
- **ADLS/Blob + Azure ML** for data and deterministic inference
- **Application Insights** for telemetry

### Optional later surfacing into M365 Copilot

Foundry-first does not block later M365 surfacing.

Later options:

- expose the same FastAPI surface to an M365 Copilot agent
- reuse the same markdown corpus, possibly mirrored into a Microsoft-friendly source if needed later
- add Teams as a channel while keeping Foundry as the orchestration brain

This is the right long-term posture: **one backend contract, multiple user surfaces**.

---

## 6) Revised Lift Assessment

### What is simpler than the previous Copilot Studio path

Foundry-first simplifies:

- no initial SharePoint dependency
- no need to package the first experience as an M365 app
- no initial Copilot Studio authoring overhead
- no initial M365 admin rollout dependency
- easier code review and version control of the agent definition
- better control over model routing and memory

### What is harder than the previous Copilot Studio path

Foundry-first is harder in areas such as:

- the team owns more of the UX shell
- memory/session handling is now an app concern
- more engineering responsibility for evals and orchestration quality
- richer agent behavior means more testing is needed

### Updated effort estimate

#### A. FastAPI and tool contract hardening — **4 to 7 person-days**

Includes:

- finalize dataset/run/result/QC endpoints
- shape responses for agent consumption
- add auth, telemetry, and caveats

#### B. Azure AI Search grounding pipeline — **2 to 4 person-days**

Includes:

- content selection
- markdown chunking
- indexing
- retrieval tuning

#### C. Foundry agent implementation — **4 to 7 person-days**

Includes:

- agent definition in code
- tool wiring
- model routing
- session handling
- persona prompts

#### D. Demo UI and end-to-end testing — **2 to 5 person-days**

Includes:

- lightweight web UI or notebook integration
- workflow testing
- groundedness evaluation
- demo polish

### Total estimated lift

| Scope | Estimate |
|---|---|
| **Credible Foundry-first internal PoC** | **12 to 23 person-days** |
| **Polished demo-grade PoC** | **18 to 28 person-days** |

### Net assessment vs Copilot Studio-first

For this exact ask, Foundry-first is the better fit.

Why:

- multi-step workflows are a first-class requirement now
- model control is explicitly desired
- repo-owned agent definition matters
- M365 surfacing is no longer the primary goal

### Dependencies and prerequisites

#### Azure prerequisites

- Azure AI Foundry project and agent runtime access
- Azure AI Search service
- Azure OpenAI or equivalent Foundry-connected model access
- Application Insights
- ADLS/Blob and Azure ML resources from the core architecture

#### Engineering prerequisites

- stable FastAPI contract
- managed identity / Entra auth plan
- markdown corpus ready for indexing
- a minimal session store for workflow continuity

#### Content prerequisites

- glossary content
- model cards
- methodology notes
- runbooks
- analyst-safe caveat wording

---

## 7) UX Options

### Where can the user interact with the agent?

Foundry-first supports several credible PoC surfaces.

#### 1. Lightweight web chat

Best default PoC choice.

Why:

- easiest demo surface
- easiest to control layout and evidence display
- independent of M365 rollout dependencies

#### 2. Streamlit UI

Very strong PoC option.

Why:

- fast to build
- easy to show chat + evidence + QC links
- good for notebook-adjacent technical users

Possible panels:

- conversation
- current dataset/run context
- retrieved evidence
- recommended next steps

#### 3. Gradio UI

Also credible for a quick demo.

Why:

- lightweight
- easy to create a polished interaction fast
- useful if the team wants a more obviously AI-demo interface

#### 4. Notebook integration

Good for technical analysts and ML engineers.

Why:

- natural fit for exploratory workflows
- useful for side-by-side code, charts, and agent summaries

#### 5. CLI integration

Good for engineering and pipeline operators.

Why:

- efficient for scripted workflows
- useful for validation and smoke-test flows

### What about Teams or M365 later?

Still possible, but later.

Recommended sequence:

1. prove the workflow in Foundry + custom UI  
2. stabilize tools, prompts, and evals  
3. optionally surface into Teams or M365 Copilot once the analyst experience is credible

### Are Adaptive Cards still possible?

Yes, if the experience is later surfaced into Teams or M365 Copilot.

For the initial Foundry-first PoC, equivalent structured UI components in Streamlit/Gradio are enough. Adaptive Cards become relevant when the team decides to publish into Microsoft surfaces.

### Recommended PoC UX

Build a **lightweight Streamlit web experience** first.

Why:

- fastest route to a “really nice PoC”
- easy to show the multi-step workflow in one screen
- easy to display citations, QC links, run summaries, and persona-based recommendations
- keeps the team focused on the actual analyst experience rather than M365 packaging

Suggested demo layout:

- left: chat and conversation history
- right top: current dataset/run context
- right middle: evidence and citations
- right bottom: next-step recommendations by Ash / Kane / Brett perspective

---

## Recommended Decision

Adopt **Azure AI Foundry Agent Service as the primary analyst agent platform** for the DeepSeismic2 PoC.

Specifically:

1. keep **ADLS/Blob + Azure ML + FastAPI** as the deterministic backbone  
2. use **Azure AI Search over markdown** for stable grounding  
3. use **FastAPI tools** for live operational truth  
4. implement **multi-step workflows** in Foundry with controlled memory and tool use  
5. encode **Ash, Kane, and Brett** as discipline-aware perspectives inside the agent  
6. deliver the first demo in a **lightweight web UI**, preferably Streamlit  
7. treat **M365 Copilot / Teams** as an optional later surfacing layer

This path best matches the new requirements while preserving the core safety boundary:

**LLMs assist seismic workflows through grounded reasoning, summarization, and recommendation; deterministic seismic models and human experts remain responsible for interpretation truth.**
