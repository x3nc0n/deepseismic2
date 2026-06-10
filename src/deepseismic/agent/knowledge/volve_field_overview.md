# Volve Field Overview

**Tags:** geology, geophysics, geoengineering, volve, field-context  
**Document type:** field-overview  
**Updated:** 2026-06-09

---

## Location and Ownership

The Volve field is located in the southern Norwegian North Sea, block 15/9, at
approximately 58°26'N, 1°54'E. Water depth is approximately 80 metres.

The field was operated by Equinor (formerly Statoil) under licence PL 046.
Partners included Shell E&P Norway and Idemitsu Petroleum Norge. Production ran
from 2008 to 2016, after which the field was safely decommissioned.

In 2018, Equinor released the complete Volve dataset as an open-access resource
for research and technology development — the **Equinor Volve Data Village**.
This dataset is the anchor for the DeepSeismic2 proof of concept.

---

## Field History

| Milestone | Date |
|---|---|
| Discovery | 1994 |
| Development plan approved | 2005 |
| First oil | 2008 |
| Cessation of production | 2016 |
| Open data release (Equinor Volve Data Village) | 2018 |

Recoverable reserves at discovery: approximately **186 million BOE** (oil + gas).

---

## Key Formations

### Hugin Formation (primary reservoir)

- **Age:** Late Jurassic, Oxfordian
- **Depth range (TVDSS):** approximately 3 470–3 540 m across the Volve wells
- **Lithology:** Fine- to medium-grained marine to fluvio-deltaic sandstone
- **Porosity:** 18–28%
- **Permeability:** 10–500 mD
- **Net pay:** 10–40 m depending on well location
- **Deposition:** Shallow-marine to fluvio-deltaic; deposited during a period of
  relative sea-level fall on the eastern flank of the Central Graben

The Hugin Fm is the primary interpretation target. The seismic anomaly identified
in the UNet inference pass correlates with this formation in wells 15/9-F-1 B and
15/9-F-4 (Hugin top at ~3 512 and ~3 498 m TVDSS respectively).

### Draupne Formation (primary seal)

- **Age:** Late Jurassic, Kimmeridgian
- **Lithology:** Organic-rich, finely laminated marine shale
- **Role:** Regional seal capping the Hugin Fm reservoir across the Central Graben
- **Note:** The Draupne Fm is also the primary source rock for Jurassic petroleum
  systems in the Norwegian North Sea. Its integrity is essential for trap preservation.

### Shetland Group (secondary seal / overburden)

- **Age:** Cretaceous
- **Lithology:** Chalk, marl, and shale
- **Role:** Forms the main overburden above the Jurassic section; important for
  velocity model building and seismic imaging quality

### Utsira Formation (shallow saline aquifer)

- **Age:** Neogene
- **Depth:** approximately 820 m TVDSS
- **Note:** Used as a CO₂ storage demonstration target at the nearby Sleipner field.
  Not a hydrocarbon reservoir target at Volve.

---

## Structural Setting

Volve sits on a rotated fault block on the eastern margin of the **Utsira High**,
within the Central Graben of the Norwegian North Sea.

**Key structural features:**

- **Trap type:** Faulted anticline with four-way dip closure
- **Main bounding fault:** NNW–SSE trending normal fault on the western flank;
  throws of 100–300 m at Jurassic level
- **Tectonic phase:** Late Jurassic rifting (Oxfordian–Kimmeridgian) created the
  half-graben geometry
- **Post-rift subsidence:** Cretaceous and Palaeogene section onlaps the high;
  structural relief preserved by differential compaction
- **Basement:** Pre-Triassic crystalline basement forms the footwall of the main
  bounding fault

**Structural interpretation notes for analysts:**

The fault corridor identified in the UNet inference pass (IL 1050–1120) is
consistent in orientation with the mapped field-bounding fault system. However,
model output should not be interpreted as confirming fault geometry without
seismic attribute QC and well tie validation.

---

## Well Inventory (PoC Scope)

| Well | Type | TD (m TVDSS) | Hugin Top (m TVDSS) | Log Suite |
|---|---|---|---|---|
| 15/9-F-1 B | Producer | 3 850 | 3 512 | GR, RHOB, NPHI, RT, DTCO |
| 15/9-F-4 | Producer | 3 831 | 3 498 | GR, RHOB, NPHI, RT |
| 15/9-F-11 | Injector | 3 740 | 3 471 | GR, RHOB, NPHI |
| 15/9-F-15 D | Producer | 3 892 | 3 535 | GR, RHOB, NPHI, RT, DTCO, DTSM |

Well 15/9-F-1 B provides the primary formation-top control and checkshot data
for seismic-to-well tie in the PoC workflow.

---

## Data License

The Volve dataset is published under the **Equinor Volve Data Village** open
license for research and development use. Commercial use restrictions apply.
See [equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing)
for the full license terms.
