# DeepSeismic2 — Cost Tracking

## AI Production Costs (Building the Product)

Tracks the AI/LLM costs incurred during development — model calls, agent spawns, token usage.

### Session Log

| Date | Session | Agents Spawned | Models Used | Est. Token Cost | Outcomes |
|------|---------|---------------|-------------|-----------------|----------|
| 2026-06-09 | Architecture kickoff | Ripley (sonnet), Lambert (sonnet), Scribe (haiku) | claude-sonnet-4.6, claude-haiku-4.5 | ~$0.85 | Architecture proposal, M365 Copilot agent design, team setup |
| 2026-06-09 | SME onboarding | — (coordinator inline) | claude-opus-4.6 | ~$0.40 | Created 3 domain SME charters (Ash, Kane, Brett) |
| 2026-06-09 | Data assessment | Ash (sonnet), Kane (sonnet), Scribe (haiku) | claude-sonnet-4.6, claude-haiku-4.5 | ~$0.70 | Volve geophysics + geology assessments |
| 2026-06-09 | Scaffold & design | Ripley (sonnet), Lambert (sonnet), Parker (sonnet), Scribe (haiku) | claude-sonnet-4.6, claude-haiku-4.5 | ~$1.20 | Repo scaffold, Foundry design, infra repo, email draft |
| 2026-06-09 | Sprint 1 build | Dallas (sonnet), Parker (sonnet), Lambert (sonnet), Hudson (sonnet), Scribe (haiku) | claude-sonnet-4.6, claude-haiku-4.5 | ~$4.50 | Full ingest pipeline, UNet, agent+3 UIs, storage client, 79 tests, CI |
| 2026-06-10 | Data acquisition | Dallas (sonnet), Scribe (haiku) | claude-sonnet-4.6, claude-haiku-4.5 | ~$0.90 | Volve data guide, download scripts, Databricks export, real notebook |
| 2026-06-10 | Azure deploy & live app | Opus (coordinator), general-purpose agents | claude-opus-4.6, claude-sonnet-4.6 | ~$6.50 | 8 infra fixes, architecture diagrams, ACR builds, data upload, 2 Container Apps live, OpenAI wired |
| 2026-06-24 | Process-fidelity evaluation | Ash, Dallas (sonnet), Ripley (opus), Scribe (haiku) | claude-opus-4.8 (coord), claude-sonnet-4.6, claude-opus-4.6, claude-haiku-4.5 | ~$3.00 | Gap analysis vs microsoft/seismic-deeplearning: synthetic-only training, circular validation, mocked critical path identified |
| 2026-06-24 | Sprint 2 build (real loop) | Dallas×2, Ash×2, Hudson, Ripley (sonnet/opus), Scribe (haiku) | claude-opus-4.8 (coord), claude-sonnet-4.6, claude-haiku-4.5 | ~$6.50 | Real-label training wired, real metrics (eval IoU 0.062, tol recall±5 0.84), QC stage, 53 new tests (211 total), README honesty, zero-phase bug fixed, issues #6-#12 |
| 2026-06-25 | Sprint 3 (de-mock + real-data readiness) | Parker×2, Lambert, Dallas, Ash, Hudson, Ripley (sonnet), Scribe (haiku) | claude-opus-4.8 (coord), claude-sonnet-4.6, claude-haiku-4.5 | ~$5.50 | De-mocked API+agent (fail-loud 503/RuntimeError), ST10010-ready ingest, ADLS train/eval, densified labels (0.30% synthetic proxy), 69 new tests (292 total) + BUG-1 fix, real-data runbook, v0.4.0 release; P1 #9 done, #7/#8 app-ready (deploy-gated) |
| 2026-06-25 | Real ST10010 ingest (first real data) | Dallas (sonnet, interrupted by reboot), coordinator recovery | claude-opus-4.8 (coord), claude-sonnet-4.6 | ~$1.50 | Ingested REAL Volve ST10010 full-stack PSDM → staged/surveys/volve-st10010/amplitude.zarr (IL 9961-10361, XL 1961-2680, 850×4ms, verified readable); resumed/repaired interrupted upload; #7 ingest-half done (training/eval blocked on real labels) |

### Cumulative Totals

| Metric | Value |
|--------|-------|
| **Total sessions** | 11 |
| **Total agent spawns** | 41+ (Dallas×8, Parker×5, Lambert×3, Ripley×6, Hudson×3, Ash×6, Kane, Scribe×7, general-purpose×3) |
| **Estimated total AI cost** | ~$31.55 |
| **Artifacts produced** | 2 design docs, 8 charters, full code scaffold, ingest pipeline, UNet, Foundry agent, 3 UIs, storage client, 292 tests, CI, infra repo, data acquisition guide, download scripts, 2 architecture diagrams, 2 Dockerfiles, live Azure deployment, process-fidelity gap analysis, real-label training pipeline + evaluation, QC/conditioning stage, task-framing doc, de-mocked fail-loud API+agent, ADLS-backed train/eval, ST10010-ready ingest CLI, real-data runbook, Sprint 3 backlog (#6-#12) |

### Cost Notes

- Opus used for coordinator (high-reasoning orchestration)
- Sonnet used for code-producing agents (quality tier)
- Haiku used for Scribe (mechanical file ops)
- Estimates based on approximate token counts × published API pricing

---

## Azure Runtime Cost Estimates (Running the Solution)

Estimated monthly costs for the PoC environment based on Ripley's architecture proposal.

### Storage Tier

| Service | Configuration | Est. Monthly Cost | Notes |
|---------|--------------|-------------------|-------|
| ADLS Gen2 / Blob (Hot) | ~50 GB (Volve subset) | ~$1.00 | Raw SEG-Y + derived Zarr |
| ADLS Gen2 / Blob (Cool) | ~200 GB (full Volve archive) | ~$2.00 | Infrequent access archive |
| **Storage subtotal** | | **~$3.00** | vs. ~$200+/mo for Premium Files or ~$500+/mo for NetApp Files equivalent |

### Compute Tier

| Service | Configuration | Est. Monthly Cost | Notes |
|---------|--------------|-------------------|-------|
| Azure Container Apps Jobs (CPU) | Consumption plan, ~10 runs/month | ~$5.00 | Preprocessing, SEG-Y parsing |
| Azure ML Compute (GPU) | Standard_NC6s_v3, ~5 hrs/month | ~$15.00 | UNet inference (pay-per-use) |
| Azure Container Apps (API) | 1 vCPU, 2 GB RAM, low traffic | ~$10.00 | FastAPI backend |
| **Compute subtotal** | | **~$30.00** | |

### AI/LLM Tier

| Service | Configuration | Est. Monthly Cost | Notes |
|---------|--------------|-------------------|-------|
| Azure OpenAI (GPT-4o) | ~100K tokens/day analyst queries | ~$15.00 | M365 Copilot agent backend |
| Azure OpenAI (embeddings) | RAG indexing + queries | ~$2.00 | Document grounding |
| M365 Copilot licensing | Per-user (enterprise) | $30.00/user | Required for published agent access |
| **AI subtotal** | | **~$47.00** + licensing | |

### Supporting Services

| Service | Configuration | Est. Monthly Cost | Notes |
|---------|--------------|-------------------|-------|
| Azure Container Registry | Basic tier | ~$5.00 | Container images |
| Application Insights | Basic telemetry | ~$2.00 | Logs and monitoring |
| Azure Key Vault | Standard | ~$0.50 | Secrets management |
| **Supporting subtotal** | | **~$7.50** | |

### Actual Azure Spend (rg-deepseismic2-dev)

| Period | Total | Top Services |
|--------|-------|-------------|
| 2026-06-10 (1 day MTD) | $1.87 | AI Search $1.62, Defender $0.13, ACR $0.11 |

**Projected monthly (based on actual):** ~$55–60/month (AI Search Basic @ $49/mo is dominant cost)

### Total Estimated Monthly Cost (PoC)

| Category | Monthly |
|----------|---------|
| Storage | $3.00 |
| Compute | $30.00 |
| AI/LLM | $47.00 + licensing |
| AI Search (Basic) | $49.00 |
| Supporting | $7.50 |
| **Total (excl. M365 licensing)** | **~$136.50/month** |
| **Total (incl. 1 M365 Copilot user)** | **~$166.50/month** |

### Cost Comparison vs. Legacy

| Approach | Est. Monthly | Notes |
|----------|-------------|-------|
| **This PoC (cloud-native)** | ~$88–118 | Object storage + on-demand compute + AI |
| **Premium Files (1 TB)** | ~$200+ | Just storage, no compute or AI |
| **Azure NetApp Files (1 TB)** | ~$500+ | Just storage, no compute or AI |
| **Dell Isilon (on-prem)** | $$$$ | Hardware + maintenance + power + cooling |

**Key insight:** The entire PoC (storage + compute + AI) costs less than the storage tier alone in the legacy approach.

---

## Update History

| Date | What Changed |
|------|-------------|
| 2026-06-09 | Initial estimates based on architecture proposal |
| 2026-06-09 | Added Sprint 1 build costs; updated totals; added AI Search to runtime |
| 2026-06-10 | Added data acquisition session; updated cumulative totals |
| 2026-06-10 | Added Azure deploy session ($6.50); actual Azure spend ($1.87 day 1); revised AI Search to $49/mo (actual Basic SKU pricing); live endpoints confirmed |
| 2026-06-25 | Added Sprint 3 session ($5.50): de-mock + real-data readiness, v0.4.0 release; updated cumulative totals (10 sessions, ~$30.05) |
| 2026-06-25 | Added real ST10010 ingest session ($1.50): first real Volve data ingested to staged; cumulative 11 sessions, ~$31.55 |
