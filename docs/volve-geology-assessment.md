# Volve Dataset Assessment — Geologist's Perspective

**Author:** Kane  
**Requested by:** jospaid  
**Date:** 2026-06-09T22:31:20-05:00

## Executive Summary

The Volve dataset is geologically strong for a proof of concept because it combines a classic **South Viking Graben rotated-fault-block trap** with a reservoir interval, the **Hugin Formation**, that is both regionally familiar and interpretationally non-trivial. It has enough stratigraphic structure, enough well control, and enough pre-existing interpretation material to let the team demonstrate real subsurface reasoning rather than generic image processing.

From a geology standpoint, the best PoC is **not** "full-field facies AI" on day one. The most credible path is:

1. build the **structural and stratigraphic framework**  
2. calibrate it with a **well-to-seismic tie**  
3. use that framework for **well correlation and first-pass depositional interpretation**  
4. only then show **facies prediction / classification** as an assisted interpretation layer

That sequence matches how a geologist actually works.

---

## 1. Geological Setting & Available Data

### 1.1 Regional setting

Volve sits in Norwegian block **15/9** in the **South Viking Graben**, part of the broader North Sea rift system. Geologically, this matters because the basin architecture is not random:

- Jurassic extension created **normal-fault-bounded accommodation**
- the field sits on a **rotated structural block**
- faulting influences both **reservoir thickness** and **seismic correlation**
- regional marine flooding and sealing units are well developed, especially the **Upper Jurassic shales**

So even before opening a seismic cube, a basin geologist expects:

- block-bounding normal faults
- rollover / rotated-block geometry
- syn-rift thickness variation
- regionally mappable shale markers
- reservoir compartmentalization by faults and flooding surfaces

### 1.2 Volve stratigraphic story

For a PoC, the useful working stratigraphic column is:

| Stratigraphic level | Geological role at Volve | Why it matters in interpretation |
|---|---|---|
| **Balder / Lista and younger Cenozoic section** | Post-rift cover | Mostly context; less critical for the reservoir story |
| **Base Cretaceous unconformity / overburden markers** | Strong regional mapping horizon | Excellent structural reference and QC surface |
| **Draupne Formation** | Organic-rich Upper Jurassic marine shale; regional seal and source | Key top-seal concept and an important contrast above the reservoir interval |
| **Heather Formation** | Marine mudstone-dominated transition interval | Helps separate reservoir from overlying shale package |
| **Hugin Formation** | Main reservoir; shallow-marine to marginal-marine sandstone system | Core PoC target for tie, correlation, structure, and facies reasoning |
| **Sleipner / associated Jurassic interval below** | Sub-reservoir reference package | Helps bracket the reservoir and tie deeper reflections |
| **Skagerrak Formation** | Older continental to marginal-marine clastic section | Useful deeper framework horizon and structural reference |

From a depositional perspective, the Hugin at Volve should be treated as a **stacked shallow-marine sandstone system with wave, storm, estuarine, and bay-head-delta influence**, not as a simple homogeneous blanket sand. That single observation is why geological context matters: a machine can segment patterns, but a geologist asks whether the package should look **shoreface-clean and laterally persistent**, **heterolithic and estuarine**, or **fault-thickened and compartmentalized**.

### 1.3 Structural setting

The structural interpretation story is strong enough for a live demo because Volve is a textbook rift-basin trap:

- **rotated fault blocks**
- significant **normal-fault compartmentalization**
- likely thickness changes across active faults
- a field-scale geometry that can be explained to non-specialists visually

That is ideal for modernization storytelling. A user can see:

- a seismic reflector offset by faulting
- a tied well penetrating the Hugin
- a mapped surface turning that local observation into a field-scale structural concept

### 1.4 What well data is available

From a geologist's perspective, the public Volve release appears rich enough to support a real interpretation workflow rather than a toy one. Available well-related material includes:

- **composite logs** in LAS form
- **petrophysical interpretation / CPI logs**
- **checkshot data**
- **well headers and trajectories**
- **lithostratigraphic / formation-top tables**
- **drilling and completion documentation**
- **core- and reservoir-description material** used in later public studies

Practically, that means the dataset supports the core geological tasks:

- **correlating tops**
- **building a synthetic tie**
- **checking structural position**
- **interpreting reservoir quality trends**
- **connecting facies ideas back to real borehole evidence**

Important nuance: the "biostratigraphy" value in this dataset is less about one glamorous standalone file and more about the presence of the supporting framework geologists actually use in practice: tops, strat tables, report interpretations, and calibration documents.

### 1.5 Interpretation products in the dataset

The released material is valuable because it appears to include more than raw subsurface data. Public descriptions of the Volve package and Petrel guide indicate the presence of interpretation-ready content such as:

- a **Petrel project / Petrel-ready interpretation package**
- interpreted **horizons**
- interpreted **faults / fault sticks**
- **maps** and surface products
- **facies or grid-property interpretation outputs**
- well-based calibration products suitable for **well-to-seismic ties**

From a PoC point of view, that is excellent. It means the team can validate AI-assisted outputs against:

- existing human interpretation
- well control
- regional expectation

instead of treating the problem as an unlabeled ML sandbox.

### 1.6 Reports and documentation

The Volve release is attractive because it comes with the kinds of reports that make geology transferable:

- field and dataset overview documentation
- seismic handling / Petrel loading guidance
- drilling and completion reports
- petrophysical and reservoir summaries
- production and development context
- public technical papers that reuse the same dataset for Hugin sedimentology, reservoir zonation, and redevelopment studies

Those documents matter because geology is not just pixels and curves. Interpretation quality improves dramatically when the interpreter can ask:

- What was the original development concept?
- Which wells had the best reservoir?
- Where was compartmentalization a known issue?
- Which horizons were considered regionally robust?

---

## 2. Recommended PoC Subset

### 2.1 Best geological subset

If I were choosing a geology-first demo subset, I would start with:

- **Wells:** `15/9-19A`, `15/9-19BT2`, `15/9-19SR`
- **Optional extension wells:** `15/9-F-11T2` and `15/9-F-15A` or `15/9-F-15D`
- **Seismic:** one cropped 3D subvolume around the main Volve fault block that includes the structural crest, one major bounding fault zone, and the tie positions for those wells

### 2.2 Why this is the best story

That subset tells the best geological story because it combines:

1. **good calibration potential**  
   The public mirrors clearly show composite logs, CPI logs, and checkshots for the 15/9-19 wells, which is exactly what you need for a credible tie and correlation exercise.

2. **stratigraphic continuity plus local variation**  
   Multiple penetrations through the Hugin allow the demo to show that the reservoir is continuous enough to correlate, but variable enough to make interpretation meaningful.

3. **structural relevance**  
   A cropped cube around the main rotated block and adjacent fault zone lets the user see real fault displacement and structural relief rather than an arbitrary seismic patch.

4. **validation potential**  
   Existing interpretation products in the Petrel package give the team something to compare against.

### 2.3 Minimal dataset that still demonstrates real geology

The true minimum credible geology subset is:

- **one small 3D seismic crop**
- **three wells with logs**
- **at least one checkshot-supported tie well**
- **formation tops for Hugin, Draupne / BCU-level marker, and Skagerrak**
- **one reference interpretation package with faults and key horizons**

If forced to go smaller, I would choose:

- `15/9-19A`
- `15/9-19BT2`
- `15/9-19SR`
- a seismic window spanning **Top Skagerrak up through Draupne / Base Cretaceous**

That is enough to demonstrate:

- well-to-seismic tie
- structural mapping
- stratigraphic correlation
- first-pass depositional interpretation

without pretending we are doing full-field development geology.

---

## 3. Demo Use Cases — Geology Perspective

## 3.1 Well-to-seismic tie

**What it demonstrates**  
The tie shows how borehole observations become seismic interpretation. It is the moment where geology and geophysics actually connect.

**Data needed**

- sonic and density logs
- checkshot or VSP-style time-depth control
- formation tops
- seismic traces near the tie well

**What AI could help with**

- identify missing inputs for a valid tie
- propose candidate wavelets / tie windows
- summarize mismatch between synthetic and seismic
- explain which reflector is most likely Top Hugin, Draupne, or a deeper marker
- generate QC notes with uncertainty language

**Why it is traditionally hard / expensive**

- requires cross-disciplinary skill
- suffers when metadata are scattered across reports and folders
- manual iteration is slow
- junior interpreters often do not know which mismatch is geological, which is velocity, and which is wavelet

**Geological value**

Very high. This is the best first PoC use case because it grounds everything else.

## 3.2 Structural interpretation — faults and key horizons

**What it demonstrates**  
Mapping the rotated block and major faults shows the fundamental trap geometry of Volve.

**Data needed**

- 3D seismic subvolume
- tied wells
- key tops: Hugin, Draupne / BCU-level surface, Skagerrak or another deeper framework marker
- any reference fault/horizon interpretation

**What AI could help with**

- suggest fault corridors and likely horizon continuations
- flag inconsistent picks across adjacent lines
- compare machine-generated picks against existing interpreted surfaces
- produce an uncertainty map: clear, ambiguous, fault-shadowed

**Why it is traditionally hard / expensive**

- large manual picking effort
- fault interpretation quality depends heavily on experience
- ambiguity increases near fault zones, low continuity, and tuning
- commercial interpretation software and workstation workflows are expensive

**Geological value**

Extremely high. It is visually compelling and easy for a demo audience to understand.

## 3.3 Stratigraphic correlation — well to well

**What it demonstrates**  
Correlation shows that interpretation is not just picking a bright reflector; it is organizing time-equivalent and geologically related packages across space.

**Data needed**

- gamma ray, resistivity, density, neutron, sonic where available
- formation tops and lithostrat tables
- multiple wells through the Hugin interval
- optional core / report descriptions

**What AI could help with**

- generate candidate correlation panels
- identify likely flooding surfaces and reservoir subdivisions
- explain why a sand package should or should not be correlated across a fault block
- summarize alternatives when more than one correlation is plausible

**Why it is traditionally hard / expensive**

- requires experience with North Sea Jurassic stacking patterns
- log motifs are not unique
- faults create repetition and omission
- the best answer is often probabilistic, not deterministic

**Geological value**

High. This is one of the clearest places where LLM reasoning can add value around explanation and hypothesis management.

## 3.4 Facies classification from seismic

**What it demonstrates**  
Shows the modernization story most clearly: machine learning producing a spatially continuous geologic interpretation layer.

**Data needed**

- 3D seismic volume or derived attributes
- well-based labels, interpreted facies, or proxy classes
- structural framework to constrain outputs
- preferably core / log facies definitions

**What AI could help with**

- define geologically sensible class labels
- explain what a predicted class means in depositional terms
- flag geologically impossible juxtapositions
- reconcile ML output with well control and fault offsets

**Why it is traditionally hard / expensive**

- labels are scarce
- seismic facies are not the same as lithofacies
- outputs are easy to oversell
- substantial QC is needed to avoid pattern-recognition theater

**Geological value**

Potentially high, but only after the structural and tie framework exists. For Volve, this should be a **second-phase** demo, not the first geology deliverable.

## 3.5 Depositional environment mapping

**What it demonstrates**  
Turns local observations into a basin story: fairways, cleaner shoreface belts, muddier estuarine or bay-head-delta zones, flooding surfaces.

**Data needed**

- mapped horizons
- well correlation framework
- seismic attributes and/or facies outputs
- geological prior model for Hugin deposition

**What AI could help with**

- propose depositional fairway maps from integrated evidence
- describe expected updip / downdip facies transitions
- explain why certain trends are geologically plausible in the South Viking Graben

**Why it is traditionally hard / expensive**

- requires synthesis, not just picking
- depends heavily on basin knowledge
- easy to produce pretty but geologically weak maps

**Geological value**

High, but only when built on the earlier workflow steps.

## 3.6 Geological reporting

**What it demonstrates**  
That AI can convert technical interpretation work into usable handoff products.

**Data needed**

- tied wells
- interpreted surfaces / faults
- logs, maps, and QC outputs
- analyst notes or metadata

**What AI could help with**

- generate structured summaries
- separate observed facts from interpretation and uncertainty
- produce well, horizon, and area-based notes

**Why it is traditionally hard / expensive**

- geologists spend real time writing updates
- project knowledge is often trapped in scattered slide decks and emails

**Geological value**

Moderate by itself, high as a multiplier. This is a strong support use case, not the core geology demo.

## 3.7 Analog identification

**What it demonstrates**  
That AI can connect Volve to similar rift-basin shallow-marine reservoirs elsewhere.

**Data needed**

- structured geologic metadata
- external analog library
- depositional and structural descriptors

**What AI could help with**

- retrieve comparable Hugin-age or North Sea-style analogs
- explain why they are similar and where the analogy breaks

**Why it is traditionally hard / expensive**

- depends on expert memory and literature familiarity
- analog selection is often informal and poorly documented

**Geological value**

Useful, but not the strongest PoC starting point because it depends on external knowledge curation more than on the Volve dataset itself.

---

## 4. Regional Knowledge Requirements

### 4.1 What Viking Graben knowledge is needed

A competent outsider can read the data. A North Sea interpreter reads the **story** faster because they already know:

- the South Viking Graben is a **rifted basin**, so normal-fault architecture is expected
- **Upper Jurassic marine shales** are regionally important for source and seal
- the **Hugin** is part of a known sandstone fairway, not an isolated local curiosity
- accommodation is strongly shaped by fault activity, so thickness patterns are diagnostic
- the same seismic texture can mean different things depending on whether you are on the crest, flank, or downthrown side of a block

### 4.2 Hugin-specific depositional model knowledge

For the Hugin specifically, regional experience tells a geologist to expect:

- **wave- and storm-reworked shallow-marine sandstones**
- local **estuarine / bay-head-delta** heterogeneity
- flooding-surface-controlled compartmentalization
- cleaner, more laterally continuous shoreface packages in some positions
- muddier, more heterolithic packages in others

That matters because an inexperienced interpreter may wrongly treat all sand-prone intervals as one uniform reservoir unit.

### 4.3 Structural style knowledge

A basin geologist "just knows" certain things here:

- major faults are likely **normal and syn-rift in origin**
- apparent thickness change may reflect **growth across fault blocks**
- reflector terminations matter: onlap, truncation, divergence
- not every seismic discontinuity is a depositional edge; some are structural
- a picked horizon that ignores obvious fault offset is almost certainly wrong

### 4.4 What a geologist just knows in this basin

This is the hidden expertise barrier:

- which horizons are likely to be regionally reliable
- which intervals are below seismic resolution and should not be over-interpreted
- what facies juxtapositions are plausible in a Jurassic North Sea shallow-marine system
- when "bright" means lithology, tuning, fluid, or simply processing artifact
- where structural explanation is more likely than depositional explanation

### 4.5 Where regional knowledge blocks entry

Regional knowledge creates barriers at exactly the moments that matter most:

- choosing which surface to map first
- deciding which log motif is a flooding surface versus local shale break
- recognizing whether a proposed facies map makes paleogeographic sense
- knowing when to trust lateral continuity and when faulting breaks it

This is where an LLM can help most: not by inventing geology, but by surfacing the **basin-appropriate prior model** at the point of interpretation.

---

## 5. Simplification & Standardization Opportunities

### 5.1 Which tasks are repeatable and standardizable

These geological tasks are surprisingly standard:

1. **data inventory and QC**
2. **well selection**
3. **formation-top loading**
4. **time-depth calibration**
5. **key-horizon interpretation**
6. **fault interpretation**
7. **well correlation**
8. **map and cross-section generation**
9. **interpretation summary / uncertainty note**

Each can be templatized as a bounded workflow with required inputs, QC checks, and outputs.

### 5.2 Where interpretation is science versus art

**Mostly science / standardizable**

- loading and validating logs
- calculating synthetics
- checking well trajectories
- tying tops to seismic
- tracking pick consistency
- generating correlation panels
- reporting missing metadata and QC failures

**Interpretive art / expert judgment**

- deciding which correlation is geologically most likely when multiple are possible
- deciding whether a reflector break is structural or stratigraphic
- choosing the most defensible depositional model
- deciding when the seismic supports facies claims and when it does not

The art is real, but the workflow leading up to it is much more standardized than most organizations admit.

### 5.3 How an LLM could bridge the regional knowledge gap

An LLM can help by behaving like a basin-savvy assistant:

- reminding the user what the **expected Hugin depositional model** is
- explaining why **Draupne above Hugin** matters for seal and mapping
- warning that a facies map should honor **fault offsets**
- suggesting the typical order of operations for a Viking Graben interpretation
- translating technical outputs into geologically meaningful language

The right role is not "AI geologist replaces interpreter." The right role is:

> "AI makes the implicit regional context explicit, on demand, and at the exact step where a junior interpreter would otherwise stall."

### 5.4 The standard interpretation workflow, step by step

This is the "amazing specific series of tasks" in practical order:

1. **Inventory the dataset**  
   Confirm seismic volume, well logs, checkshots, tops, trajectories, reports.

2. **Establish the regional frame**  
   Age, basin type, structural style, expected reservoir / seal pair, likely depositional systems.

3. **QC and select wells**  
   Choose the wells with the cleanest Hugin penetration and best time-depth control.

4. **Build the well-to-seismic tie**  
   Create synthetic, align key markers, document uncertainty.

5. **Interpret the first regional / robust horizons**  
   Start with the easiest, most regionally reliable markers before tackling the reservoir top/base.

6. **Interpret major faults**  
   Build structural skeleton first where faulting clearly offsets reflections.

7. **Map reservoir interval surfaces**  
   Pick Top Hugin and supporting deeper / shallower markers with the fault framework honored.

8. **Correlate well to well**  
   Subdivide the reservoir and identify flooding surfaces, cleaner sand packages, and heterolithic breaks.

9. **Create maps and sections**  
   Structure map, isochore / thickness view, correlation panels, inline/xline QC.

10. **Interpret depositional fairways**  
    Translate patterns into shoreface, estuarine, bay-head-delta, or heterolithic trends.

11. **Evaluate uncertainty**  
    Where is the tie weak, the seismic ambiguous, or the facies assignment below resolution?

12. **Write the interpretation note**  
    State observed evidence, inferred geology, risks, and next steps.

That workflow is standardized enough to productize, while still preserving expert judgment at the right points.

---

## 6. Compare and Contrast the Best Use Cases

### 6.1 Shortlist

For this PoC, my top four geology use cases are:

1. **Well-to-seismic tie**
2. **Structural interpretation**
3. **Stratigraphic correlation**
4. **Facies classification from seismic**

### 6.2 Comparison table

| Use case | Feasibility for PoC | Demo impact | AI value-add | Standardization potential | Geologist's verdict |
|---|---|---|---|---|---|
| **Well-to-seismic tie** | **High** — needs only a few good wells plus seismic and checkshots | **High** — easy to explain and visually intuitive | **High** — AI can organize inputs, explain mismatch, and generate QC language | **High** — very repeatable across basins | Best first geology demo |
| **Structural interpretation** | **High** — Volve structure is ideal for this | **Very high** — faults and surfaces are visually compelling | **Moderate to high** — AI can accelerate picking/QC, but humans still validate | **High** — broadly transferable | Best modernization story |
| **Stratigraphic correlation** | **High** — strong well package available | **Moderate to high** — more technical, but geologically rich | **High** — AI reasoning is genuinely useful here | **High** — templatable across many clastic systems | Best expert-assistant story |
| **Facies classification** | **Moderate** — depends on labels and careful geological constraints | **Very high** — flashy and modern | **Moderate** if unconstrained, **high** if tied to geology | **Moderate** — class definitions vary by basin | Strong second-phase demo, risky first demo |

### 6.3 Recommended order of execution

If the team wants the best balance of credibility and demo value, do them in this order:

1. **Well-to-seismic tie**
2. **Structural interpretation**
3. **Stratigraphic correlation**
4. **Facies classification**

That order mirrors real interpretation practice and prevents the PoC from jumping straight to the least constrained problem.

### 6.4 Final recommendation

The highest-value geology PoC is:

> **"AI-assisted Volve interpretation from well tie to structural framework to stratigraphic correlation, with facies classification as an optional, geologically constrained extension."**

That is modern, defensible, and reusable beyond the North Sea.

---

## Bottom Line

Volve is a strong geology dataset for this project because it contains the three things a useful interpretation PoC needs:

- **a real structural story**
- **a real stratigraphic story**
- **enough well and interpretation control to validate what the AI is doing**

The key is to respect geological order of operations. In Volve, the right question is not "Can AI classify facies from seismic?" The right question is:

> "Can AI help a geologist move through the standard interpretation workflow faster, with better regional context and fewer avoidable mistakes?"

On this dataset, the answer should be yes.

---

## References

- Equinor, **Volve data sharing** — dataset overview and access portal: https://www.equinor.com/energy/volve-data-sharing
- Equinor, **Data sharing overview**: https://www.equinor.com/energy/data-sharing
- Norwegian Offshore Directorate / NPD fact pages, **Volve field**: https://factpages.sodir.no/en/field/pageview/all/3420717
- Norwegian petroleum overview, **Volve field**: https://www.norskpetroleum.no/en/facts/field/volve/
- Equinor, **Volve Petrel How to Guide**: https://www.equinor.com/content/dam/statoil/documents/impact/data-village/volve/Volve%20Petrel%20How%20to%20Guide.pdf
- Public Volve mirror used to confirm available well-log-style artifacts (composite logs, CPI logs, checkshots, paths, lithostrat tables): https://github.com/awgeo/Volve_field_data
- Summary source on Hugin / Volve depositional style, especially bay-head-delta and shoreface concepts: https://www.sciencedirect.com/science/article/pii/S0264817220306243
