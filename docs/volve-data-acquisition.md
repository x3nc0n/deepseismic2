# Volve Data Acquisition Guide

**Project:** deepseismic2 PoC  
**Updated:** 2026-06-10

## Quick Start (no download required)

```bash
python scripts/download_volve.py --sample
jupyter notebook notebooks/01_data_exploration.ipynb
```

Generates a synthetic ~45 MB SEG-Y with realistic geometry and opens the exploration notebook. No credentials, no waiting.

---

## Overview

The Equinor Volve dataset is a comprehensive open-data release from the decommissioned Volve oil field in the Norwegian North Sea (Viking Graben, block 15/9). It contains ~40,000 files including seismic volumes, well logs, production data, and geological interpretations — published under **CC BY 4.0**.

For this PoC, we need a targeted subset focused on fault detection and seismic interpretation.

---

---

## 1. Dataset Catalog

### 1.1 Primary seismic — ST10010

ST10010 is the richest seismic package in the Volve release: final Kirchhoff PSDM in both time and depth domains, angle stacks, and velocity cubes. The PoC targets **time-domain post-stack** (easiest for interpretation storytelling).

| Component | Filename | Size | Priority |
|---|---|---|---|
| **Full stack (time)** | `ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | ~0.98 GB | **Required** |
| FAR angle stack | `ST10010ZC11_PZ_PSDM_KIRCH_FAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | ~0.98 GB | Optional |
| NEAR angle stack | `ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | ~0.85 GB | Optional |
| MID angle stack | `ST10010ZC11_PZ_PSDM_KIRCH_MID_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | ~0.98 GB | Optional |
| Migration velocity | `ST10010ZC11_PZ_PSDM_MIG_VEL.segy` | ~0.03 GB | Optional |

**Survey specs:**
- Kirchhoff PSDM, final post-stack, time domain; block 15/9, South Viking Graben
- Sample interval: 4 ms; bin: 12.5 m × 12.5 m
- Minimal PoC footprint: ~1.2 GB (full stack + velocity)
- AVO subset (near + mid + far + velocity): ~3.9 GB

### 1.2 Well logs

| Well | File | Format | Description |
|---|---|---|---|
| 15/9-19A | `15_9-19A.las` | LAS 2.0 | Primary producer — complete log suite (GR, RHOB, NPHI, DT, RT) |
| 15/9-19BT2 | `15_9-19BT2.las` | LAS 2.0 | Oil producer |
| 15/9-19SR | `15_9-19SR.las` | LAS 2.0 | Side-track injector |

Checkshot and DLIS-format logs also available from the portal.

### 1.3 Fault interpretations

| File | Contents |
|---|---|
| `Volve_Fault_Sticks.txt` | Petrel fault-stick export → input to `label_generator.py` |
| `Volve_Horizons.txt` | Mapped horizons: Base Cretaceous, Hugin Top, Hugin Base, Draupne Top |

---

## 2. Path A — Databricks Marketplace (Recommended)

Joe's workspace: `https://dbc-63d65b56-08e4.cloud.databricks.com`  
Provider: Equinor (ID: 006d1a44-e5d8-4d20-8bd7-e9a311106007)

### 2.1 Prerequisites

- Databricks workspace access with Unity Catalog enabled
- Volve listing installed from Marketplace (already done per task context)
- Personal Access Token (PAT): **User Settings → Developer → Access Tokens → Generate new token**
- For local use: set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars

### 2.2 Discover the Volve catalog (in a Databricks notebook)

Open any notebook in the workspace and run:

```python
# Cell 1 — see all available catalogs (find the Volve one)
display(spark.sql("SHOW CATALOGS"))
```

```python
# Cell 2 — explore schemas (adjust catalog name to match output above)
VOLVE_CATALOG = "volve_field_data"   # ← update to actual catalog name

display(spark.sql(f"SHOW SCHEMAS IN {VOLVE_CATALOG}"))
```

```python
# Cell 3 — list tables in the seismic schema
display(spark.sql(f"SHOW TABLES IN {VOLVE_CATALOG}.seismic"))
```

```python
# Cell 4 — survey inventory / metadata
df = spark.sql(f"""
    SELECT survey_name, n_inlines, n_crosslines, sample_rate_ms,
           inline_min, inline_max, crossline_min, crossline_max
    FROM {VOLVE_CATALOG}.seismic.survey_inventory
    ORDER BY survey_name
""")
display(df)
```

```python
# Cell 5 — find our target files
display(spark.sql(f"""
    SELECT file_name, file_size_bytes / 1e9 AS size_gb, component, survey
    FROM {VOLVE_CATALOG}.seismic.file_registry
    WHERE survey LIKE '%ST10010%'
    ORDER BY component
"""))
```

### 2.3 Export SEG-Y to local pipeline

**Option A — copy to DBFS, then pull with Databricks CLI:**

```python
# In a Databricks notebook
SRC = "/mnt/volve/seismic/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy"
DST = "dbfs:/tmp/deepseismic2/ST10010_FULL.segy"
dbutils.fs.cp(SRC, DST)
print(f"Staged to DBFS: {DST}")
```

```bash
# On your local machine (requires Databricks CLI: pip install databricks-cli)
databricks configure --token   # enter host + PAT once

databricks fs cp \
  dbfs:/tmp/deepseismic2/ST10010_FULL.segy \
  data/volve/raw/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy
```

**Option B — REST API + download script (no CLI install required):**

```bash
export DATABRICKS_HOST="https://dbc-63d65b56-08e4.cloud.databricks.com"
export DATABRICKS_TOKEN="<your-pat>"

# Discover available datasets
python scripts/databricks_export.py --discover

# Export a seismic table to local Zarr
python scripts/databricks_export.py \
  --export-zarr volve_field_data.seismic.st10010_full_stack \
  --dest data/volve/staged/ST10010_full.zarr

# Get a signed download URL for the raw SEG-Y
python scripts/databricks_export.py \
  --get-url volve_field_data.seismic.file_registry \
  --component full_stack
```

### 2.4 Get the underlying SEG-Y storage URL

If the Delta table wraps an External Location pointing at the original SEG-Y blob, you can retrieve the signed URL and use it with `download_volve.py`:

```python
# In a Databricks notebook — retrieve signed storage URL
row = spark.sql("""
    SELECT file_path FROM volve_field_data.seismic.file_registry
    WHERE component = 'full_stack'
""").first()

print(row.file_path)
# → abfss://volve@<storage>.dfs.core.windows.net/...
# or https://volveflex.blob.core.windows.net/...?sig=...

# Copy the root URL base (everything before the filename) then:
# python scripts/download_volve.py --seismic --base-url "<root_url>"
```

---

## 3. Path B — Direct Equinor Download

### 3.1 Accept the data-sharing agreement

1. Go to **https://www.equinor.com/energy/volve-data-sharing**
2. Click **"Get access to Volve data"**
3. Accept the **CC BY 4.0** license
4. Receive a confirmation email containing a **storage root URL** (Azure Blob SAS URL)

The URL looks like:
```
https://volveflex.blob.core.windows.net/volve-field-data/?sv=2021-06-08&sig=...
```

### 3.2 Download with `download_volve.py`

```bash
# Minimal PoC subset (~1 GB)
python scripts/download_volve.py \
  --seismic \
  --components full_stack velocity \
  --base-url "https://volveflex.blob.core.windows.net/volve-field-data/?sv=..." \
  --dest data/volve

# Full AVO subset (~4 GB)
python scripts/download_volve.py \
  --seismic \
  --base-url "https://..." \
  --dest data/volve

# Wells + interpretations
python scripts/download_volve.py \
  --wells --interpretations \
  --base-url "https://..." \
  --dest data/volve

# Verify what we have
python scripts/download_volve.py --verify --dest data/volve
```

### 3.3 Manual download (curl / browser)

```bash
BASE="<SAS_URL_ROOT>/Seismic.ST10010/Stack"
FNAME="ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy"

curl -L -o "data/volve/raw/${FNAME}" "${BASE}/${FNAME}"
```

---

## 4. Local Pipeline Integration

### 4.1 Expected directory layout

```
data/
  volve/
    synthetic_sample.segy          ← generated by --sample mode
    raw/
      ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy
      ST10010ZC11_PZ_PSDM_KIRCH_FAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy
      ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy
      ST10010ZC11_PZ_PSDM_KIRCH_MID_T.MIG_FIN.POST_STACK.3D.JS-017536.segy
      ST10010ZC11_PZ_PSDM_MIG_VEL.segy
    wells/
      15_9-19A.las
      15_9-19BT2.las
      15_9-19SR.las
    interpretations/
      Volve_Fault_Sticks.txt
      Volve_Horizons.txt
    staged/
      ST10010_full.zarr/           ← output of segy_to_zarr()
      ST10010_full.json            ← metadata sidecar
      fault_mask.zarr/             ← output of FaultMaskGenerator
```

### 4.2 Validate a downloaded SEG-Y

```python
from deepseismic.ingest.segy_loader import load_segy

ds, geom = load_segy(
    "data/volve/raw/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy",
    sample_mode=True,       # load first 50 inlines only for quick QC
    sample_n_inlines=50,
)
print(f"Inlines:    {geom.inline_min}–{geom.inline_max}  n={geom.n_inlines}")
print(f"Crosslines: {geom.crossline_min}–{geom.crossline_max}  n={geom.n_crosslines}")
print(f"Samples:    {geom.n_samples}  ({geom.sample_rate_ms} ms/sample)")
print(f"TWT range:  {geom.times_ms[0]:.0f}–{geom.times_ms[-1]:.0f} ms")
```

### 4.3 Full ingest pipeline

```python
from deepseismic.ingest.segy_loader import segy_to_zarr

# Full file — ~10 min on a modern laptop
meta = segy_to_zarr(
    "data/volve/raw/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy",
    "data/volve/staged/ST10010_full.zarr",
    overwrite=True,
)
print(f"Shape:       {meta.n_inlines_loaded} × {meta.geometry['n_crosslines']} × {meta.geometry['n_samples']}")
print(f"p99 amplitude: {meta.amplitude_stats['p99']:.4f}")
print(f"Sidecar:     data/volve/staged/ST10010_full.json")
```

### 4.4 Generate fault labels

```python
from deepseismic.ingest.label_generator import (
    parse_petrel_fault_sticks, FaultMaskGenerator, SurveyTransform
)
from deepseismic.ingest.segy_loader import load_segy

_, geom = load_segy(
    "data/volve/raw/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy",
    sample_mode=True,
)

sticks = parse_petrel_fault_sticks("data/volve/interpretations/Volve_Fault_Sticks.txt")

# Three-point survey transform (calibrate from trace headers or survey report)
transform = SurveyTransform.from_three_points(
    tie_points=[
        (456123.0, 6470456.0, 1001, 1900),   # (X, Y, inline, crossline)
        (456623.0, 6470456.0, 1001, 1940),
        (456123.0, 6470956.0, 1041, 1900),
    ],
    sample_rate_ms=geom.sample_rate_ms,
    datum_ms=geom.datum_ms,
)

gen = FaultMaskGenerator(
    volume_shape=(geom.n_inlines, geom.n_crosslines, geom.n_samples),
    inline_range=(geom.inline_min, geom.inline_max, geom.inline_step),
    crossline_range=(geom.crossline_min, geom.crossline_max, geom.crossline_step),
    sample_rate_ms=geom.sample_rate_ms,
    datum_ms=geom.datum_ms,
    dilation_voxels=2,
)
gen.add_fault_sticks(sticks, transform)
gen.to_zarr("data/volve/staged/fault_mask.zarr", overwrite=True)
print(f"Fault voxels: {gen.mask.sum()}")
```

### 4.5 Upload to Azurite (local dev blob storage)

```bash
# Start Azurite
docker compose -f docker/docker-compose.yml up -d azurite

# Copy Zarr store into container (simple azcopy approach)
azcopy copy "data/volve/staged/ST10010_full.zarr" \
  "http://127.0.0.1:10000/devstoreaccount1/seismic/ST10010_full.zarr?<SAS>" \
  --recursive
```

---

## 5. Synthetic Test Data (`--sample` mode)

The `--sample` flag in `download_volve.py` generates a fully synthetic 3-D SEG-Y with:

- Survey geometry matching real ST10010 (IL 1001–1100, XL 1900–2099, 500 samples at 4 ms)
- Two Ricker wavelet reflectors at ~700 ms and ~1000 ms TWT simulating the Hugin interval
- Gaussian noise at realistic S/N ratios
- Correct SEG-Y trace headers (INLINE_3D, CROSSLINE_3D, DELAY_RECORDING_TIME)
- Size ≈ 45 MB; reproducible (fixed random seed)

```bash
# Default ~45 MB
python scripts/download_volve.py --sample

# Larger for more realistic memory testing (~200 MB)
python scripts/download_volve.py --sample \
  --sample-inlines 200 \
  --sample-crosslines 400 \
  --sample-n-samples 500
```

The synthetic file is 100% compatible with the full ingest pipeline.

---

## 6. Licensing and Attribution

The Volve dataset is published by Equinor under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

| | |
|---|---|
| ✅ Commercial use | Allowed |
| ✅ Modification | Allowed |
| ✅ Distribution | Allowed |
| 📝 Attribution | **Required** |

**Required credit in any publication, model card, or demo:**

> *Seismic, well, and interpretation data from the Volve field (Norwegian block 15/9), made available by Equinor ASA and Volve license partners under CC BY 4.0 (https://www.equinor.com/energy/volve-data-sharing).*

Full license: https://creativecommons.org/licenses/by/4.0/legalcode  
Data portal: https://www.equinor.com/energy/volve-data-sharing

---

## 7. Troubleshooting

| Issue | Fix |
|-------|-----|
| Databricks "no tables found" | Check Unity Catalog permissions; admin may need to grant SELECT on the catalog |
| Download 403 / Forbidden | SAS token expired — re-register on the portal to get a fresh URL |
| `segyio` can't read file | Try `ignore_geometry=True` in `SEGYLoader`; check IBM vs IEEE float format |
| Checksum mismatch | Re-download; confirm `--verify` output matches expected GB sizes |
| File too large for local disk | Use `sample_mode=True` or process directly in Databricks using `databricks_export.py` |
| Well log parse error | LAS version mismatch — open in a text editor to check the `~V` section |

