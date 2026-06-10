# Decisions Log

**Archive:** None yet.  
**Last updated:** 2026-06-10T03:47:00Z

---

## 2026-06-09: Batch 2 — Architecture, Data, and Agent Platform

### ARCHITECTURE: Foundry-First Agent Design [Lambert]

**Date:** 2026-06-09T22:28:44-05:00

1. **Foundry Agent Service** is the primary analyst agent platform.
   - Multi-step workflows, explicit model control, agent definition in code.
   - M365 surfacing is optional, not primary delivery.

2. **Object-storage-first architecture** remains the runtime backbone.
   - ADLS/Blob = system of record; Azure ML runs deterministic inference; FastAPI orchestrates.
   - Keeps deterministic seismic processing separate from LLM behavior.

3. **Azure AI Search over markdown** replaces SharePoint-first grounding.
   - Removes SharePoint as prerequisite; fits code-first, version-controlled workflow.

4. **FastAPI tools** ground all dynamic answers.
   - Run status, QC outputs, result summaries via tool calls, not retrieval alone.

5. **Discipline-aware perspectives** inside one shared agent.
   - Geophysics (Ash), Geology (Kane), Geoengineering (Brett) as modes.

6. **Lightweight web UI** (Streamlit) for first demo.
   - Fastest path to polished PoC; multi-step state on one screen.

7. **M365 Copilot** as a later surfacing option.
   - Preserve ability without optimizing initial workflow around it.

### DATA: Volve Geophysics Assessment [Ash]

**Date:** 2026-06-09

**Primary PoC Seismic:**  
- **Survey:** ST10010 (richest Volve package).
- **Demo volume:** `ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` (~1 GB).
  - Time-domain, best quality/realism/speed balance.
- **Extended set:** Add NEAR_T, MID_T, FAR_T, MIG_VEL (~4 GB total).

**Strongest PoC Use Cases:**
1. Automated seismic QC and conditioning.
2. Horizon seed suggestion / auto-tracking.
3. Fault probability generation.
4. Attribute extraction + interpretation summary.
5. Well-seismic tie assistance.
6. Velocity / time-depth QC.

**Not first-demo-safe:**  
- Full AVO / quantitative amplitude interpretation (pre-stack capability exists but amplitude reliability needs expert calibration).

**Why Volve:** Realistic linked geophysical tasks, not just a cube. Includes horizons, velocity, logs, well calibration data for validation.

**Expert-guided boundaries:**  
- AI can standardize: SEG-Y ingest, metadata, QC, candidate picks, candidate faults, tie setup, explanation.
- AI should NOT autonomously sign off: final interpretation, quantitative amplitude claims, velocity model acceptance, structural risk conclusions.

**Regional knowledge:** Tacit North Sea interpretation knowledge (normal faulting style, key reflectors, amplitude reliability, time-depth uncertainty) is the real barrier; LLM can lower it by explaining context and guiding task sequence.

### DATA: Volve Geology Assessment [Kane]

**Date:** 2026-06-09T22:31:20-05:00

**Subset:** Geology-first Volve centered on tied Hugin wells.
- Wells: `15/9-19A`, `15/9-19BT2`, `15/9-19SR`.
- Cropped 3D subvolume spanning main Volve rotated fault block and adjacent fault zone.
- Smallest subset telling a real structural/stratigraphic story; supports well tie, structural mapping, correlation before ML facies work.

**Workflow Sequence:**
1. Well-to-seismic tie (first, calibration step, easier to validate).
2. Structural interpretation.
3. Stratigraphic correlation.
4. Facies classification (last).

**Rationale:** Matches how a geologist actually interprets a field; reduces risk of overselling unconstrained ML output; strongest modernization story with least geological risk.

**Regional Grounding:** Ground the analyst assistant with explicit North Sea / South Viking Graben / Hugin context. Depositional expectations strongly affect facies, thickness, and correlations. LLM helps junior or out-of-basin interpreters most here.

**Recommendation:** Proceed with geology-first Volve workflow around tied wells, structural framework, and stratigraphic correlation. Best AI story: "AI makes standard interpretation workflow more repeatable, better grounded, and easier to transfer across teams and basins," not "AI replaces the geologist."

### INFRASTRUCTURE: Two-Repo Split and Azure Tenant

**Date:** 2026-06-09T22:31:20-05:00

**Azure Resources Tenant:** `ef4ecf0b-a160-444b-a405-ce3bf1f98752` (MCAPS).

**Two-Repo Structure:**
- `x3nc0n/deepseismic2` — project code, ML pipelines, agent, API (public).
- `Spava-Corp/deepseismic2-infra` — infrastructure CI/CD GitHub Actions (org-private).

**Volve Data:** Copied into ADLS for direct evaluation.

**ML Model Choice:** Deferred until Ash and Kane complete Volve data assessment.

**Rationale:** Separation of concerns. Infra automation (secrets, provisioning, deployment) in private org repo; project itself publishable. MCAPS subscription provides PoC resources.

**Infra Scaffold Details [Parker]:**
- **Repository:** `Spava-Corp/deepseismic2-infra`.
- **Tooling:** Azure Bicep and GitHub Actions CI/CD.
- **Cost Posture:** Default to cheapest workable PoC: Standard_LRS ADLS Gen2, ACR Basic, AI Search Basic (Free optional), Azure Container Apps consumption, AML compute min 0 / max 1.
- **Operations:** One-click deploy, one-click destroy, local helper scripts. Stand up fast, tear down when idle.

### REPOSITORY: Initial Scaffold [Ripley]

**Date:** 2026-06-09T22:31:20-05:00

**Structure:**
- Single Python package under `src/`.
- Thin FastAPI service.
- Foundry agent area.
- Azure storage abstractions.
- Placeholder modules by responsibility.

**Packaging:**
- Python 3.11, `pyproject.toml`-based build, editable installs.
- Basic lint, type, and test configuration.

**Data:**
- Keep `data/` local and ignored.
- SEG-Y as source truth.
- Knowledge markdown under agent package for Azure AI Search indexing.

**Delivery:** Local branch normalized to `main` before GitHub publish.

---

## Decision Deduplication Summary

- **Duplicate Found:** `lambert-m365-agent-design.md` vs. `lambert-foundry-first.md`
  - SUPERSEDED: M365 Copilot (lambert-m365-agent-design.md) by Foundry-first decision.
  - KEPT: Foundry-first decision; M365 now a later surfacing option.
- **All Others:** Unique decisions from distinct agents and contexts.

---

## Status

- Inbox merge: Complete (7 files → 1 decisions.md).
- Deduplication: M365 path superseded; Foundry-first locked in.
- Ready for orchestration log, session log, and cross-agent history updates.
