# Seismic Interpretation Workflow

**Tags:** methodology, workflow, geophysics, geology, geoengineering, qc  
**Document type:** runbook  
**Updated:** 2026-06-09

---

## Overview

The DeepSeismic2 interpretation workflow is a constrained, end-to-end sequence
designed to take a Volve seismic dataset from raw SEG-Y through to a validated,
analyst-reviewed inference result. The workflow is intentionally narrow in scope —
it proves one credible path rather than attempting a full production interpretation
platform.

LLMs assist at the summary, QC review, and handoff stages. Deterministic seismic
processing and UNet model inference remain responsible for the actual interpretation
outputs. Analysts provide the expert judgment required before any result is used for
operational or development decisions.

---

## Workflow Steps

### Step 1 — Ingest and Data Verification

**Goal:** Confirm that the raw SEG-Y data is intact, properly catalogued, and
has been converted to cloud-friendly derived formats.

**Actions:**
1. Ingest the Volve subset SEG-Y file into Azure Blob / ADLS Gen2.
2. Extract trace headers and geometry metadata into a JSON manifest.
3. Convert to Zarr for efficient cloud access.
4. Validate inline and crossline ranges, sample interval, and trace count.
5. Confirm the manifest is written to the catalog.

**Quality checks:**
- Inline range covers the target interpretation window (IL 1000–1200, XL 950–1100).
- Sample interval confirmed at 4 ms.
- No missing or duplicate traces.
- Geometry consistent with expected Volve survey parameters.

**Common pitfalls:**
- Incorrect byte-swap settings in SEG-Y ingest — always verify amplitude polarity.
- Geometry gaps at survey edges — flag edge inlines separately in QC.
- Zarr chunk size misaligned with access pattern — verify chunk dimensions.

**Agent tool:** `query_survey_metadata`

---

### Step 2 — Preprocessing QC

**Goal:** Confirm that preprocessing conditioning steps completed without errors
and that the data is suitable for model inference.

**Actions:**
1. Check preprocessing run status (completed / failed / partial).
2. Review any preprocessing warnings — geometry issues, outlier traces, NaN counts.
3. Confirm output format and storage path are correct.

**Quality checks:**
- Run status is ``completed`` with no failure reason.
- Warning count is within acceptable bounds (< 5% of trace count).
- Output stored at the expected Zarr path.

**Common pitfalls:**
- Partial completion — preprocessing stopped on a subset of inlines. Check
  error logs; re-run from the failed inline.
- Amplitude scaling differences between pre- and post-processing — verify
  normalization parameters are consistent with training data distribution.

**Agent tools:** `get_interpretation_status`, `create_qc_report`

---

### Step 3 — Model Inference

**Goal:** Run the UNet baseline inference model on the preprocessed volume
and confirm that results are stored and accessible.

**Actions:**
1. Submit a fault-detection or facies inference job via the API.
2. Monitor job status until ``completed``.
3. Verify that the prediction mask is written to results storage.
4. Confirm that QC overlay slices were generated.

**Quality checks:**
- Prediction mask dimensions match the input volume.
- QC slices cover a representative sample of the inference window.
- No inference failures or NaN outputs in the mask.

**Common pitfalls:**
- GPU timeout on large volumes — use inline range limits during PoC testing.
- Model applied outside its training distribution — check that Volve data
  characteristics fall within the expected input range.
- QC slices not generated — check that the QC slice step ran after inference.

**Agent tools:** `run_fault_detection`, `get_interpretation_status`

---

### Step 4 — Result Summary and QC Review

**Goal:** Translate model output into analyst-readable findings, highlight
caveats, and flag zones requiring closer attention.

**Actions:**
1. Retrieve the result summary from the backend.
2. Review QC artifact overlays for visual plausibility.
3. Cross-check amplitude anomaly depths against well formation tops.
4. Flag any edge-zone results with reduced confidence.

**Quality checks:**
- Key findings reference specific inline / crossline ranges.
- Caveats are explicit about model confidence limits.
- Well-tie depth verification within 15–20 ms TWT tolerance.
- Edge-zone results marked with a lower confidence flag.

**Common pitfalls:**
- Treating model probability as geological certainty — always include caveats.
- Over-interpreting edge-zone results where SNR is low.
- Missing well-tie verification — always check Hugin Fm top depth correlation.

**Agent tools:** `generate_summary`, `get_inline_section`, `get_formation_tops`,
`correlate_wells`

---

### Step 5 — Analyst Handoff

**Goal:** Produce a structured handoff note for downstream review, archiving
the evidence, findings, caveats, and recommended next steps.

**Actions:**
1. Generate a structured summary including all key findings.
2. Package the result for export (summary + QC overlays + mask reference).
3. Record recommended next steps, separated by discipline.
4. Obtain analyst sign-off before sharing with the subsurface team.

**Handoff note structure:**
- Dataset and run identifiers
- Preprocessing and inference status
- Key findings with spatial references
- Caveats and confidence limits
- Recommended next steps by discipline (geophysics / geology / geoengineering)

**Agent tools:** `generate_summary`, `export_interpretation`, `create_qc_report`

---

## Discipline-Specific Review Guidance

### Geophysics review (Ash perspective)

Focus on:
- Amplitude reliability — is the anomaly above the noise floor?
- Processing caveats — any migration artefacts near the fault zone?
- Signal-to-noise in the edge inlines
- Whether QC slice overlays show coherent structural patterns

### Geology review (Kane perspective)

Focus on:
- Does the anomaly depth tie to the expected Hugin Fm top in nearby wells?
- Is the spatial pattern consistent with the mapped structural setting?
- Could the anomaly be a processing artefact rather than a geological feature?
- Are facies classifications plausible given the known depositional environment?

### Geoengineering review (Brett perspective)

Focus on:
- Does the structural interpretation support the existing field development model?
- Are there implications for well placement or injection strategy?
- What additional data (PVT, well test, petrophysical) is needed before
  any operational decision?

---

## Error Handling and Safe Degradation

| Situation | Agent response |
|---|---|
| Dataset not found | State the dataset was not found; offer the closest match |
| Run incomplete or failed | Summarise the operational issue; recommend remediation |
| QC artifacts missing | State confidence is limited; recommend regenerating QC outputs |
| Result summary unavailable | Say the backend returned no facts; do not speculate |
| Well-tie discrepancy > 20 ms | Flag as requiring investigation before sign-off |

---

## Definition of Done

A workflow cycle is complete when:

1. ✅ Dataset loaded and verified
2. ✅ Preprocessing completed without critical errors
3. ✅ Inference completed and mask stored
4. ✅ QC slices reviewed by a qualified analyst
5. ✅ Handoff note produced with explicit caveats
6. ✅ Analyst has reviewed and approved the result

No result should pass to operational use without step 6 being completed by
a qualified petroleum geoscientist.
