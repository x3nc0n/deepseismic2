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

### Cumulative Totals

| Metric | Value |
|--------|-------|
| **Total sessions** | 6 |
| **Total agent spawns** | 17 (Dallas×2, Parker×3, Lambert×2, Ripley×2, Hudson, Ash, Kane, Scribe×4) |
| **Estimated total AI cost** | ~$8.55 |
| **Artifacts produced** | 2 design docs, 8 charters, full code scaffold, ingest pipeline, UNet, Foundry agent, 3 UIs, storage client, 79 tests, CI, infra repo, data acquisition guide, download scripts |

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

### Total Estimated Monthly Cost (PoC)

| Category | Monthly |
|----------|---------|
| Storage | $3.00 |
| Compute | $30.00 |
| AI/LLM | $47.00 + licensing |
| AI Search (Basic) | $70.00 |
| Supporting | $7.50 |
| **Total (excl. M365 licensing)** | **~$157.50/month** |
| **Total (incl. 1 M365 Copilot user)** | **~$187.50/month** |

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
