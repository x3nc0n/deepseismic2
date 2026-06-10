# Work Routing

How to decide who handles what.

## Routing Table

| Work Type | Route To | Examples |
|-----------|----------|----------|
| Architecture & design | Ripley | System design, component boundaries, API contracts, trade-offs |
| ML models & data pipelines | Dallas | SEG-Y parsing, model training, data preprocessing, evaluation |
| Infrastructure & storage | Parker | Azure setup, storage architecture, cost optimization, containers |
| LLM/AI integration | Lambert | Copilot agents, RAG pipelines, NL interfaces, prompt engineering |
| Code review | Ripley | Review PRs, check quality, architectural alignment |
| Testing & validation | Hudson | Write tests, data validation, model evaluation, edge cases |
| Seismic acquisition & processing | Ash | Data QC, processing advice, AVO, resolution analysis, wave physics |
| Geology & interpretation | Kane | Stratigraphy, facies, depositional models, well-seismic ties, Volve context |
| Reservoir & production | Brett | Petrophysics, volumetrics, production analysis, simulation, economics |
| Scope & priorities | Ripley | What to build next, trade-offs, decisions |
| Session logging | Scribe | Automatic — never needs routing |

## Issue Routing

| Label | Action | Who |
|-------|--------|-----|
| `squad` | Triage: analyze issue, assign `squad:{member}` label | Ripley |
| `squad:ripley` | Architecture/design work | Ripley |
| `squad:dallas` | ML/data pipeline work | Dallas |
| `squad:parker` | Infrastructure/storage work | Parker |
| `squad:lambert` | AI/LLM integration work | Lambert |
| `squad:hudson` | Testing work | Hudson |

### How Issue Assignment Works

1. When a GitHub issue gets the `squad` label, **Ripley** triages it — analyzing content, assigning the right `squad:{member}` label, and commenting with triage notes.
2. When a `squad:{member}` label is applied, that member picks up the issue in their next session.
3. Members can reassign by removing their label and adding another member's label.
4. The `squad` label is the "inbox" — untriaged issues waiting for Lead review.

## Rules

1. **Eager by default** — spawn all agents who could usefully start work, including anticipatory downstream work.
2. **Scribe always runs** after substantial work, always as `mode: "background"`. Never blocks.
3. **Quick facts → coordinator answers directly.** Don't spawn an agent for "what port does the server run on?"
4. **When two agents could handle it**, pick the one whose domain is the primary concern.
5. **"Team, ..." → fan-out.** Spawn all relevant agents in parallel as `mode: "background"`.
6. **Anticipate downstream work.** If a feature is being built, spawn the tester to write test cases from requirements simultaneously.
7. **Issue-labeled work** — when a `squad:{member}` label is applied to an issue, route to that member. Ripley handles all `squad` (base label) triage.
