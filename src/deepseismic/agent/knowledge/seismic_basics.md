# Seismic Basics — Quick Reference

**Tags:** geophysics, methodology, glossary, seismic-fundamentals  
**Document type:** glossary  
**Updated:** 2026-06-09

This document provides a quick-reference glossary of seismic concepts relevant
to the DeepSeismic2 proof of concept. It is intended to ground the analyst agent
when answering questions about seismic data characteristics, acquisition, and
interpretation concepts.

---

## Data Formats

### SEG-Y
The industry-standard seismic data exchange format. SEG-Y files store seismic
traces as binary amplitude samples with associated headers containing geometry
(inline, crossline, CDP X/Y) and acquisition metadata. SEG-Y is the source-of-truth
input for the DeepSeismic2 workflow.

**PoC usage:** Raw Volve SEG-Y files are ingested into Azure Blob storage unchanged.
Derived formats (Zarr) are generated for efficient cloud access.

### Zarr
A cloud-native, chunked, compressed array format. Zarr divides seismic volumes
into regular spatial chunks that can be read in parallel without downloading
the entire volume. Used in DeepSeismic2 as the derived format for ML training
and inference input.

### Inline / Crossline
The two horizontal indexing axes of a 3D seismic survey. Inlines and crosslines
define the survey grid. In the Volve PoC subset, inlines run approximately
north–south (IL 1000–1200) and crosslines run approximately east–west
(XL 950–1100).

---

## Seismic Acquisition and Signal Concepts

### Two-Way Time (TWT)
The time (in milliseconds) for a seismic wave to travel from surface to a
reflector and back. Depth conversion from TWT to true vertical depth (TVDSS)
requires a velocity model. In the Volve dataset, the Hugin Fm top is at
approximately 3 490–3 510 ms TWT.

### Amplitude
The peak displacement of a seismic wave at a given sample. Amplitude variations
can indicate changes in acoustic impedance caused by lithology or fluid content.
Elevated amplitudes at the Hugin Fm level in the Volve dataset suggest possible
reservoir sands or fluid effects, but require expert analysis before geological
interpretation.

### Acoustic Impedance (AI)
The product of rock density and seismic wave velocity. Reflections are generated
at impedance contrasts between rock layers. The Draupne shale (high AI) overlying
the Hugin sandstone (lower AI) creates the primary reflection at the reservoir level.

### Signal-to-Noise Ratio (SNR)
The ratio of signal energy to background noise in a seismic trace. Low SNR degrades
interpretation quality. The Volve PoC edge inlines (IL 1180–1200) show reduced SNR
due to survey geometry effects — results in that zone should be treated with caution.

### Polarity
The sign convention for a seismic reflection (positive or negative peak first).
SEG-Y polarity must be confirmed on ingest to avoid reversed amplitude
interpretation (high becomes low, and vice versa).

---

## Velocity and Depth Conversion

### Checkshot
A direct velocity measurement taken in a well by recording seismic source travel
time to a downhole receiver at known depths. Checkshots are used to calibrate
the velocity model for depth conversion. Well 15/9-F-1 B provides checkshot data
for the Volve PoC.

### RMS Velocity
Root-mean-square velocity derived from seismic moveout analysis. Used in normal
moveout (NMO) correction and as input to velocity model building. Less accurate
than checkshot-derived velocities at depth.

### Depth Conversion
Converting seismic reflection times (TWT, ms) to subsurface depths (m TVDSS) using
a velocity model. Uncertainty in depth conversion is a key caveat for any
well-placement decision derived from seismic output.

---

## Seismic Attributes

### Amplitude Envelope
The instantaneous amplitude magnitude of a seismic trace, computed as the modulus
of the analytic signal. Bright spots in the envelope may indicate gas or brine
sands. Used in QC to highlight anomalous zones.

### Instantaneous Phase
The phase angle of the analytic signal, independent of amplitude. Useful for
tracking reflector continuity and identifying discontinuities (faults).

### Coherence / Semblance
A measure of trace-to-trace similarity. Low coherence values highlight discontinuities
in the seismic data — potential faults, fracture zones, or erosional channels.
UNet fault detection uses coherence-like features implicitly in its architecture.

---

## Structural Interpretation Concepts

### Fault
A fracture in the rock along which displacement has occurred. In seismic data,
faults appear as discontinuities in reflectors. The UNet model identifies candidate
fault corridors as probability masks — expert review is required before interpreting
as confirmed geological faults.

### Anticline
A convex-upward fold where the oldest rocks are in the core. Anticlines can form
structural traps for hydrocarbons if capped by a seal. The Volve Hugin Fm reservoir
is hosted in a faulted anticline.

### Four-Way Dip Closure
A structural trap type where the reservoir rock dips away from the crest in all
four map directions, creating a fully enclosed structure. Volve has four-way dip
closure at the Hugin Fm level.

### Horst and Graben
Horst: an upthrown block bounded by normal faults on both sides.
Graben: a downthrown block. The Central Graben of the North Sea is a major
extensional basin created during Late Jurassic rifting.

---

## Model Output Concepts

### Prediction Mask
A spatial probability array produced by the UNet model, with the same dimensions
as the input seismic volume. Each voxel contains a probability (0–1) for the
target class (e.g., fault presence, facies type). The prediction mask is stored
as a Zarr array in results storage.

### QC Slice / QC Overlay
A 2D slice through the seismic volume or prediction mask, rendered as an image
for visual quality control. In the DeepSeismic2 workflow, 12 sampled inline slices
are generated as PNG overlays after each inference run.

### Facies
A discrete rock body with a characteristic set of sedimentary properties. In
machine learning seismic interpretation, "facies classification" assigns a
categorical label to each voxel based on its seismic signature. Facies labels
are model outputs — they describe the model's classification, not confirmed geology.

---

## Key Numbers: Volve PoC Reference

| Parameter | Value |
|---|---|
| Survey inline range (PoC subset) | IL 1000–1200 |
| Survey crossline range | XL 950–1100 |
| Sample interval | 4 ms TWT |
| TWT range | 0–4 000 ms |
| Hugin Fm top (TWT, approximate) | ~3 490–3 510 ms |
| Hugin Fm top (TVDSS, well range) | 3 471–3 535 m |
| Primary seal | Draupne Fm (Jurassic shale) |
| Primary bounding fault orientation | NNW–SSE |
| UNet inference time (PoC subset) | ~14 min on GPU |

---

## Important Caveats for Agents

When citing seismic concepts in analyst-facing responses:

1. **Amplitude anomalies are not direct hydrocarbon indicators** — they require
   rock physics analysis and well control to interpret.
2. **UNet fault corridors are candidates** — they are model predictions, not
   confirmed geological structures.
3. **TWT to depth conversion carries uncertainty** — always note the velocity
   model source when quoting depths.
4. **SNR and edge effects matter** — low-confidence zones should be flagged
   explicitly, not silently included in summaries.
5. **Analyst sign-off is mandatory** — no model output should be used for
   operational decisions without qualified geoscientist review.
