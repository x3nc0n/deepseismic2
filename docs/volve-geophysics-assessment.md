# Volve Dataset Assessment from a Geophysicist's Perspective

**Author:** Ash  
**Date:** 2026-06-09

## Executive assessment

For a geophysics-led PoC, the Volve release is credible because it is not just "a seismic cube"; it is a small field-development ecosystem: migrated 3D seismic, angle stacks, at least some pre-stack gather products, velocity volumes, official interpreted horizons, and well packages that support tie and calibration. The most PoC-friendly part of the release is **ST10010**, especially the **final PSDM time-domain post-stack volumes**. They are large enough to be real, but still small enough to subset into a tight demo loop.

My practical recommendation is:

1. **Core PoC:** ST10010 final full-stack time volume + velocity cube + 2-3 calibration wells + official horizons.  
2. **Extended PoC:** add ST10010 near/mid/far time angle stacks.  
3. **Do not start with full raw shots or full pre-stack gathers** unless the explicit demo goal is acquisition QC or pre-stack amplitude work.

---

## 1) Available seismic data in Volve

## 1.1 What surveys appear to be available

Based on the public Volve materials reviewed, the seismic release is anchored on two survey families that matter for a PoC:

- **ST10010**
  - The richest seismic package in the release.
  - Includes raw field data, pre-stack products, post-stack migrated volumes, and velocity cubes.
  - File naming indicates **P-wave, PSDM, Kirchhoff-migrated** deliverables in both **time (T)** and **depth (D)** domains.

- **ST0202 / ST0202R08**
  - Appears in public worked examples and header scans as a **3D marine survey** with **PZ PSDM** products.
  - Includes at least:
    - a representative **post-stack raw PSDM time/depth sample**
    - a **pre-stack CIP gather** example in PP time
  - Community mirrors sometimes describe ST0202 as a "4C" Volve survey, but the direct trace-header evidence I reviewed points to a **streamer-style geometry** (Geco Angler, cables, air-gun source). For PoC planning I would trust the direct SEG-Y header evidence over informal reposts.

### Bottom line

For engineering reality, treat **ST10010 as the primary production-quality 3D survey package** and **ST0202R08 as a useful secondary/baseline/example survey** rather than the main demo target.

---

## 1.2 Acquisition geometry, vintage, and processing level

## ST10010

### Vintage

- Commonly referenced as a **2010 3D survey** over Volve.

### Geometry and product family

The file inventory shows:

- **Raw data** in large shot-domain tiles (`Raw_data/... .sgy`)
- **Prestack data**: `...PZ_PSDM_RMO_CIP_GATHS...PRESTACK_BINGATHERS...`
- **Stacks**:
  - `FULL`, `NEAR`, `MID`, `FAR`, and `NEAR_MID`
  - each in **time** and/or **depth**
  - both **RAW** and **KIRCH FIN** variants
- **Velocity volumes**:
  - migration velocity
  - anisotropy/auxiliary-style cubes (`AZIMUTH`, `DELTA`, `DIP`, `EPSILON`, `RMO`)

### Processing level

The naming convention is unusually informative:

- `PSDM` = pre-stack depth migrated workflow
- `KIRCH` = Kirchhoff migration
- `MIG_FIN` = final migrated product
- `POST_STACK.3D` = post-stack interpretation-ready cube
- `RAW_*` products = less-finished stack variants

This is strong evidence that Volve gives us more than a single post-stack cube; it gives a **processing ladder**, which is excellent for AI/QC demos.

## ST0202R08

Direct header evidence from the PP-time CIP gather example shows:

- **3D**
- **Offshore marine**
- **Vessel:** Geco Angler
- **Source:** 2 air-gun arrays
- **Shot interval:** 25 m (flip-flop)
- **Receiver line spacing:** 400 m
- **Source line spacing:** 100 m
- **Cable length:** 6000 m
- **Record length:** 10.2 s
- **Sample interval:** 4 ms
- **Bin size:** 12.5 m x 12.5 m
- **Processing sequence includes:**
  - designature / zero phase
  - tidal statics
  - HCF 82 Hz
  - PZ summation
  - tau-p deconvolution
  - 3D Kirchhoff depth migration
  - conversion to TWT
  - velocity analysis and RMO correction
  - parabolic Radon demultiple

So ST0202R08 is not a vague example; it is a good-quality processed marine 3D product with explicit acquisition and processing metadata.

---

## 1.3 Data quality, bandwidth, and resolution

## ST10010

I did not find a public text extract with full ST10010 bandwidth statistics, so I would describe it this way:

- **Quality:** high enough for serious research and interpretation; the deliverable set implies a mature processing flow, not a toy dataset.
- **Interpretive value:** high, because it includes:
  - final PSDM stacks
  - angle stacks
  - velocity cubes
  - raw data lineage
- **Bandwidth/resolution:** likely materially better than legacy coarse training cubes, but still North Sea 3D marine seismic with the usual tuning limits. Expect the practical vertical resolution to remain on the order of **tens of meters**, not bed-scale truth.

That means ST10010 is excellent for:

- structure
- faults
- major reservoir-top/near-reservoir horizon interpretation
- angle-stack screening

It is **not** a guarantee of plug-and-play quantitative amplitude reliability without careful calibration.

## ST0202R08

For ST0202R08 we have harder evidence:

- **4 ms sample interval**
- **82 Hz high-cut filter in the documented sequence**
- **12.5 x 12.5 m bins**

That is good enough for a PoC on:

- structural interpretation
- horizon tracking
- fault enhancement
- seismic QC

But not enough, by itself, to promise high-confidence thin-bed quantitative interpretation.

---

## 1.4 Are pre-stack gathers available or only post-stack?

**Pre-stack is available, but unevenly across the release.**

### ST10010

Yes, evidence strongly indicates availability of:

- **raw shot-domain data**
- **pre-stack CIP gathers**
- **post-stack final/raw angle and full stacks**

### ST0202R08

Yes, at least one concrete **pre-stack PP-time CIP gather** example is public and header-scanned.

### Practical interpretation

For the PoC, the answer is:

- **Post-stack is abundant and demo-ready**
- **Pre-stack exists, but should be treated as an advanced lane**

That is enough to support a roadmap where:

- phase 1 = post-stack interpretation automation
- phase 2 = angle-stack / pre-stack amplitude workflows

---

## 1.5 File formats and approximate sizes

## Primary formats

- **SEG-Y / SEGY**
- IBM float encoding appears in at least the inspected ST0202 pre-stack example
- Standard trace-header locations are explicitly documented in the inspected header file

## ST10010 size picture

From the public file list:

- **Seismic.ST10010.zip**: about **2.84 TB compressed**
- **Unzipped total**: about **3.3 TB**

Approximate content split from the inventory:

- **Raw shot-domain SGY tiles:** about **3.0 TB**
- **Post-stack migrated stacks:** about **15.3 GB total**
- **Velocity cubes:** about **0.26 GB total**
- **Remaining balance:** largely pre-stack gather content, logs, docs, and support files

### Individual ST10010 post-stack files

Representative interpretation-friendly files:

- `...KIRCH_FULL_T...POST_STACK...segy` ≈ **0.98 GB**
- `...KIRCH_NEAR_T...POST_STACK...segy` ≈ **0.85 GB**
- `...KIRCH_MID_T...POST_STACK...segy` ≈ **0.98 GB**
- `...KIRCH_FAR_T...POST_STACK...segy` ≈ **0.98 GB**
- `...MIG_VEL...segy` ≈ **0.03 GB**
- `...MIG_VEL-RMO...segy` ≈ **0.12 GB**

## ST0202R08 size picture

From public worked examples:

- representative raw post-stack sample: **~277 MB**
- inspected pre-stack gather example:
  - **4,281,512 traces**
  - **1,126 samples/trace**
  - estimated physical SEG-Y size on disk: **~18.9 GB**

### Conclusion on format/size

The Volve seismic release is in exactly the kind of format mix that creates real operational friction:

- authoritative but awkward **SEG-Y**
- multiple processing states
- multiple domains (time/depth)
- multiple angle variants
- huge raw/pre-stack payloads

That is precisely why it is a good modernization target.

---

## 2) Recommended PoC subset

## 2.1 Best seismic volume(s) for the PoC

### Recommended core subset

**Primary recommendation:**  
**ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy**

### Recommended optional additions

- `ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T...`
- `ST10010ZC11_PZ_PSDM_KIRCH_MID_T...`
- `ST10010ZC11_PZ_PSDM_KIRCH_FAR_T...`
- `ST10010ZC11-MIG-VEL...`

## 2.2 Why this is the best demonstration set

### 1. It is geophysically credible

This is not a downsampled academic toy. It is a final processed PSDM time volume tied to the same field ecosystem as the wells and interpretations.

### 2. It is manageable

The full final time stack is about **1 GB**.  
That is ideal for:

- cloud ingestion
- header scanning
- QC preview generation
- fast attribute runs
- ML inference

### 3. It has a validation path

The broader Volve ecosystem includes:

- official interpreted horizons
- well logs
- reports
- velocity products

That means the demo can answer the most important technical question: **"Is the AI output geologically and geophysically reasonable?"**

### 4. Time-domain is easier for demo storytelling

For a first PoC:

- **time-domain seismic** is easier to tie to interpreted horizons and common interpretation practice
- **depth-domain seismic** is better for some development tasks, but it imports more velocity-model debate into the first demo

## 2.3 Approximate data size for the recommended subset

### Minimal credible subset

- ST10010 final full-stack time cube: **~0.98 GB**
- one migration velocity cube: **~0.03 GB**
- official horizons / metadata / docs: **small**
- 2-3 calibration wells with logs and checkshot support: **small relative to seismic**

**Recommended minimal PoC footprint:** about **1.2-1.5 GB raw**, plus derived previews/metadata.

### Better "AI interpretation" subset

- Full T + Near T + Mid T + Far T + velocity

Approximate total:

- **~3.9-4.2 GB raw seismic**

That is still small enough to be cloud-friendly and much richer for demo purposes.

## 2.4 Well data available for calibration

The Volve field release is known to include the kinds of calibration data a geophysicist wants:

- **well logs** (including sonic and density where available)
- **LAS / DLIS-style log deliveries**
- **checkshot / VSP support**
- **well reports / completion / survey documentation**
- **formation tops and interpretation context**

That is enough to support:

- synthetic seismogram generation
- wavelet estimation
- bulk-shift estimation
- time-depth QC
- seismic-to-well tie scoring

### Practical note

The PoC should not assume every well is equally tie-ready. The best workflow is:

1. rank wells by log completeness
2. require sonic + density + checkshot quality
3. select the best 2-3 wells first

---

## 3) Demo use cases from a geophysics perspective

## 3.1 Seismic QC and data conditioning

### What it demonstrates

That the platform can ingest SEG-Y and automatically answer:

- is the file structurally sane?
- does geometry parse cleanly?
- are there dead traces / amplitude spikes / header inconsistencies?
- what domain is this in?
- is it suitable for interpretation or ML?

### Data needed

- any SEG-Y volume
- preferably ST10010 full T
- header scans, trace statistics, amplitude histograms, quick-look inlines/xlines/timeslices

### What AI could help with

- narrating QC findings in analyst language
- classifying likely issues
- suggesting conditioning steps
- comparing one ingest against previous runs

### Why it is traditionally hard/expensive

Because QC is usually spread across:

- file-format specialists
- processing geophysicists
- interpretation software
- ad hoc scripts

The work is repetitive, but still expert-dependent.

## 3.2 Horizon picking / auto-tracking

### What it demonstrates

That AI can accelerate interpretation of key events such as:

- reservoir top
- major flooding / seal markers
- overburden references

### Data needed

- ST10010 full-stack time volume
- official horizons for validation
- well tops where available

### What AI could help with

- seed suggestion
- auto-tracking along coherent reflectors
- confidence maps
- identifying where tracking should stop

### Why it is traditionally hard/expensive

Because horizon picking is not a single click; it is a repetitive loop of:

- seed
- track
- inspect
- correct
- re-seed near faults / dim zones / polarity changes

This is exactly the kind of "specific series of tasks" that benefits from automation plus explanation.

## 3.3 Fault detection

### What it demonstrates

That the platform can turn a seismic cube into a **fault probability** or **fault enhancement** volume and shorten structural interpretation time.

### Data needed

- ST10010 full-stack time volume
- optional interpreted faults for validation

### What AI could help with

- CNN/segmentation inference
- ranking fault likelihood
- separating acquisition footprint from geology
- generating candidate fault sticks

### Why it is traditionally hard/expensive

Fault interpretation is labor-intensive in dense extensional settings because:

- fault throws vary rapidly
- events are discontinuous near relay zones
- interpreters must constantly reject artifacts

## 3.4 Seismic attribute analysis

### What it demonstrates

That the system can compute standard attributes and then explain what they likely mean geologically.

### Data needed

- ST10010 full-stack time volume
- optionally angle stacks

### Candidate attributes

- coherence / semblance
- dip / azimuth
- curvature
- sweetness
- RMS / envelope / instantaneous frequency

### What AI could help with

- selecting the right attribute set for the objective
- summarizing which attributes actually respond to faults vs noise
- generating interpretation notes

### Why it is traditionally hard/expensive

Because attribute workflows often become kitchen-sink exercises with too many cubes and too little interpretive discipline.

## 3.5 AVO / amplitude analysis

### Should it be in the PoC?

**Only as a phase-2 extension.**

### Why

ST10010 includes near/mid/far stacks and likely enough pre-stack content to support amplitude work, but true AVO credibility requires:

- angle validation
- wavelet consistency checks
- offset/angle balancing
- residual moveout scrutiny
- careful well calibration

### Data needed

- ST10010 near/mid/far time stacks at minimum
- preferably pre-stack gathers
- good sonic/density/checkshot wells

### What AI could help with

- screening for suspect amplitude behavior
- suggesting where amplitudes are robust vs risky
- automating intercept/gradient quicklooks

### Why it is traditionally hard/expensive

Because amplitude interpretation fails quietly when conditioning or calibration is weak. This is high-value, but not first-demo-safe.

## 3.6 Well-seismic tie

### What it demonstrates

That the system can bridge well depth-domain truth and seismic time-domain interpretation.

### Data needed

- ST10010 full T
- sonic + density logs
- checkshot/VSP

### What AI could help with

- selecting the best logs
- wavelet estimation suggestions
- automatic correlation scoring
- identifying bulk shift vs stretch/squeeze behavior
- generating a tie quality summary

### Why it is traditionally hard/expensive

Because well ties are deceptively manual:

- log conditioning
- depth matching
- checkshot QC
- wavelet choice
- phase choice
- stretch/squeeze judgment

This is one of the strongest use cases for a geophysics-savvy AI assistant.

## 3.7 Velocity model QC

### What it demonstrates

That the platform can validate whether the seismic domain, depth conversion, and well time-depth data are mutually consistent.

### Data needed

- ST10010 velocity cube(s)
- ST10010 T and D volumes if desired
- checkshots / markers

### What AI could help with

- flagging suspicious time-depth mismatches
- comparing velocity trends to well control
- summarizing risk around depth prediction

### Why it is traditionally hard/expensive

Because velocity QC is usually hidden inside specialist processing or geomodeling tools, while interpreters see the consequences later.

---

## 4) Regional knowledge requirements

## 4.1 What North Sea / Viking Graben knowledge is traditionally needed

A competent interpreter in this area normally carries a lot of tacit knowledge:

- **rifted North Sea structural style**
  - rotated blocks
  - listric/planar normal faults
  - relay ramps and compartmentalization
- **Jurassic syn-rift stratigraphy**
  - which reflectors are regionally persistent
  - where reservoir/seal relationships are likely
- **North Sea seismic character**
  - marine multiples
  - acquisition footprint risks
  - fault shadow / illumination behavior
  - amplitude dimming across structure vs processing artifacts
- **time-depth expectations**
  - likely velocity behavior through overburden and reservoir section
- **field-development context**
  - which horizons matter operationally
  - which faults actually affect drainage compartments

## 4.2 Where regional knowledge creates barriers to entry

### Barrier 1: knowing what is "normal"

A newcomer cannot easily tell whether a dim event is:

- real lithology/fluid change
- tuning
- mistie
- stretch artifact
- migration issue

### Barrier 2: naming and correlation conventions

North Sea workflows carry strong conventions about:

- horizon names
- marker picks
- reservoir/seal intervals
- which event is used as the main structural reference

### Barrier 3: fault interpretation context

Without local structural intuition, AI outputs can look plausible while being wrong in exactly the way an expert would reject.

## 4.3 Which barriers AI can lower

### Good candidates for AI assistance

- translating local jargon into plain language
- surfacing regional analog expectations
- ranking likely horizon candidates
- explaining probable causes of poor tie / poor amplitude behavior
- auto-generating QC narratives grounded in data

### Harder barriers

AI can **lower entry cost**, but not fully replace:

- regional stratigraphic judgment
- deciding which event is the economically relevant one
- deciding when amplitude is geologically diagnostic versus dangerous

---

## 5) Simplification opportunities

## 5.1 Tasks most amenable to standardization / automation

These are the best targets:

1. **SEG-Y ingest + metadata extraction**
2. **seismic QC reporting**
3. **preview generation**
4. **horizon seed suggestion and auto-tracking assist**
5. **fault probability generation**
6. **attribute generation and ranking**
7. **well-tie setup, scoring, and documentation**
8. **velocity/time-depth consistency checks**

Why these work well:

- they are repetitive
- they have measurable outputs
- they benefit from both ML and LLM explanation
- they are expensive largely because of workflow fragmentation, not because each sub-step is inherently novel

## 5.2 Tasks that must remain expert-guided

These should remain under explicit geophysicist control:

### Final interpretation acceptance

An expert must decide whether a picked horizon or fault surface is geologically defensible.

### Quantitative amplitude claims

Any statement about fluid, rock property, or AVO class needs careful human review.

### Velocity-model signoff

Depth uncertainty has drilling and development consequences. AI can screen; experts must sign off.

### Structural risk judgments

Compartmentalization, trap integrity, and uncertainty ranking remain expert decisions.

## 5.3 Where the workflow has the most friction

The biggest friction is not one algorithm. It is the chain:

1. find the right seismic product  
2. verify domain and processing level  
3. QC geometry and amplitudes  
4. choose calibration wells  
5. tie wells  
6. pick seed horizons  
7. track through faulted areas  
8. compute attributes  
9. decide whether anomalies are geological or processing-related  
10. explain the result to a non-specialist

That chain is exactly what your project is trying to simplify.

---

## Recommended PoC scope in one sentence

**Use ST10010 final full-stack time seismic as the core cube, add ST10010 near/mid/far time stacks as the first extension, and build the demo around automated QC, horizon/fault assistance, and well-tie support before attempting full AVO.**

---

## Key technical conclusions

- **Best first seismic dataset:** ST10010 final PSDM **time** full stack
- **Best extension:** ST10010 near/mid/far time stacks
- **Best first use cases:** seismic QC, horizon tracking, fault detection, attributes, well tie
- **Use with caution:** AVO / quantitative amplitude claims
- **Why Volve is strong for AI:** it includes both the seismic and enough context to validate interpretation quality
- **Why this matters:** the expensive part of interpretation is the linked task sequence, not just the picking itself

---

## Evidence used for this assessment

- Equinor Volve data-sharing page
- Public ST10010 file inventory from `mrava87/equiVolve`
- Public ST0202R08 SEG-Y header scan and manifests from `jonslo/osdu-data-data-definitions`
- Azure documentation using Volve ST0202/ST10010 files as standard SEG-Y conversion examples
- Public Volve-related research/example repositories showing ST10010 angle stacks, horizons, and velocity usage
