# Real-Data Deploy-Path Runbook

**Author:** Ripley (Lead/Architect)  
**Date:** 2026-06-25T09:34:00-05:00  
**Status:** App-ready. Execution gated on infra #11 + Marketplace install.

---

## Prerequisites and blockers

Before any step below can run against real Volve data, all three items must be resolved:

| # | Blocker | Owner | Tracking |
|---|---------|-------|---------|
| B1 | ST10010_PSDM_TIME.segy copy job into ADLS `raw` container | Spava-Corp/deepseismic2-infra | infra issue #11 |
| B2 | Equinor Volve Databricks Marketplace listing install (identity-bound, user action) | User (x3nc0n) | Manual step |
| B3 | ADLS `publicNetworkAccess: Disabled` — all ingest/train/eval must run inside the VNet | Spava-Corp/deepseismic2-infra | infra issue #11 |

Steps 1–4 below **must run inside the VNet** (Azure ML managed compute or Container App
job). Steps 5–6 (API + agent) can run locally once ADLS credentials are available, but
the storage endpoint must be reachable.

---

## Environment variables

| Variable | Real mode value | Mock mode value | Effect |
|----------|----------------|----------------|--------|
| `AZURE_STORAGE_ACCOUNT` | your ADLS account name | (absent) | Selects managed-identity auth to ADLS |
| `STORAGE_CONNECTION_STRING` | (absent or cleared) | `DefaultEndpointsProtocol=http;...` (Azurite) | Connection string auth — takes precedence over managed identity if set |
| `DEEPSEISMIC_MOCK_MODE` | (absent or `false`) | `true` | API: real mode is default; set `true` for explicit mock |
| `MOCK_LLM` | (absent or `false`) | `true` | Agent: live mode is default; set `true` for explicit mock |
| `AZURE_PROJECT_ENDPOINT` | your Foundry/Azure AI project endpoint URL | (absent → agent raises RuntimeError) | Required for live agent mode |

---

## Step 1 — Confirm infra #11 complete (in-VNet)

Check that the raw SEG-Y has landed:

```bash
az storage blob exists \
  --account-name $AZURE_STORAGE_ACCOUNT \
  --container-name raw \
  --name ST10010_PSDM_TIME.segy \
  --auth-mode login
```

Expected: `"exists": true`. If false, infra #11 is not complete — stop here.

---

## Step 2 — SEG-Y → Zarr ingest (in-VNet, Azure ML or Container App job)

```bash
# Cheap smoke-ingest first (50 inlines, ~seconds) to verify format:
python scripts/ingest_segy.py \
    --source /mnt/raw/ST10010_PSDM_TIME.segy \
    --dest /mnt/staged/surveys/volve-st10010/amplitude_sample.zarr \
    --survey-id volve-st10010 \
    --sample-mode --sample-n-inlines 50 \
    --overwrite

# Full ingest (385 inlines, larger job — size TBD):
python scripts/ingest_segy.py \
    --source /mnt/raw/ST10010_PSDM_TIME.segy \
    --dest /mnt/staged/surveys/volve-st10010/amplitude.zarr \
    --survey-id volve-st10010 \
    --overwrite
```

ADLS path written: `staged/surveys/volve-st10010/amplitude.zarr`  
Sidecar JSON written: `staged/surveys/volve-st10010/amplitude.json`

Expected geometry: inlines 9985–10369, step=1. Verify in the sidecar JSON and console output.

---

## Step 3 — Generate fault labels (in-VNet or locally with Marketplace access)

Once `Volve_Geophysical_Interpretations.zip` is accessible after B2 (Marketplace install):

```bash
# Extract fault-stick .dat files to a local directory, then:
python scripts/generate_fault_label.py \
    --fault-stick-dir /path/to/real/fault_sticks \
    --amplitude-json /mnt/staged/surveys/volve-st10010/amplitude.json \
    --label-output /mnt/staged/surveys/volve-st10010/fault_label.zarr \
    --interpolate-between \
    --max-interp-gap 5

# QC output: check positive-fraction in the printed report.
# Target: >= 0.5% (CAUTION), ideally >= 2.0% (good coverage).
# If < 0.01%: something is wrong — check coordinate mapping.
```

> **Note on Petrel multi-stick format:** Real Volve sticks are likely one stick per
> inline. If so, group sticks by fault name and merge into a single polyline before
> passing to `add_fault_sticks_in_index_space`. This grouping step is a deferred
> enhancement; document what the raw files look like when they arrive.

---

## Step 4 — Train and evaluate (in-VNet only — private endpoint)

```bash
# Train on ADLS-staged data:
python -m deepseismic.training.train \
    --data-mode zarr \
    --storage-backend azure \
    --az-seismic-prefix surveys/volve-st10010/amplitude.zarr \
    --az-label-prefix surveys/volve-st10010/fault_label.zarr \
    --epochs 50 \
    --device cuda \
    --seed 42
# Checkpoint written to: features/ container → checkpoints/best.pt

# Evaluate:
python scripts/evaluate.py \
    --checkpoint /mnt/features/checkpoints/best.pt \
    --storage-backend azure \
    --az-seismic-prefix surveys/volve-st10010/amplitude.zarr \
    --az-label-prefix surveys/volve-st10010/fault_label.zarr
# Metrics written to: output/eval_metrics.json
```

`--storage-backend local` (the default) reads from local disk — use this for local dev
with `data/volve/staged/`.

---

## Step 5 — API in real mode

Set environment (in-VNet or any host with VNet reachability to ADLS):

```bash
export AZURE_STORAGE_ACCOUNT=<your_adls_account>
# STORAGE_CONNECTION_STRING must be absent or empty — managed identity takes over
unset STORAGE_CONNECTION_STRING
# DEEPSEISMIC_MOCK_MODE must be absent or false (default)
unset DEEPSEISMIC_MOCK_MODE

uvicorn deepseismic.api.main:app --port 8000
```

Verify real mode:

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","mock_mode":false,"storage":"ok"}
# "storage":"unreachable" → ADLS private endpoint not reachable from this host
# "storage":"error" → credentials issue (check AZURE_STORAGE_ACCOUNT + managed identity)
```

In mock mode (`DEEPSEISMIC_MOCK_MODE=true`), health returns `"storage":"mock"` — useful
for local UI development and CI without real storage.

**Misconfiguration contract:** If `DEEPSEISMIC_MOCK_MODE` is absent and storage is
misconfigured, the API returns HTTP 503 on every route. It does **not** silently fall
back to canned data.

---

## Step 6 — Foundry agent in real mode

```bash
export AZURE_PROJECT_ENDPOINT=https://<your-project>.services.ai.azure.com/api/projects/<id>
# MOCK_LLM must be absent or false (default)
unset MOCK_LLM

python -m deepseismic.ui.chat
# Startup log: "starting in LIVE mode — endpoint: <url>  model: <model>"
```

**Misconfiguration contract:** If `AZURE_PROJECT_ENDPOINT` is absent in live mode, the
agent raises `RuntimeError` at startup with an actionable message — it does **not**
silently serve canned responses.

---

## Local dev / demo mode (no real data needed)

```powershell
# Explicit mock opt-in — safe for local dev, CI, demos
$env:DEEPSEISMIC_MOCK_MODE = "true"
$env:MOCK_LLM = "true"
uvicorn deepseismic.api.main:app --reload --port 8000

# Demo UI
streamlit run src/deepseismic/ui/streamlit_app.py
```

---

## Cross-repo dependencies

Data staging and infrastructure are owned by **Spava-Corp/deepseismic2-infra**
(tracked as infra issue #11). This repository owns only the application code.
Frame any staging or networking blockers as infra dependencies, not code gaps.
