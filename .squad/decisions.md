# Squad Decisions

## Active Decisions

# Decision: F3 Ingest Contract (Cross-Survey Training)

**Date:** 2026-07-09T17:36:13-05:00  
**Author:** Dallas (Data/ML Engineer)  
**Status:** Proposed — pending coordinator review + infra action  
**Triggered by:** Issue #31 (infra request for F3 data readiness answers)

## Decision

F3 data for cross-survey training must be sourced from the public **OpendTect F3 Demo** dataset (dGB Earth Sciences / TerraNubis, CC BY-SA), ingested using the existing pipeline scripts without modification, and staged at the surveys/f3-demo/ prefix in the staged container.

## Rationale

1. **Real data not present in repo.** Only a synthetic proxy exists in data/f3/ (confirmed by PROXY_DATA_DO_NOT_USE_AS_GROUND_TRUTH.txt). All pipeline code is validated end-to-end against this proxy.

2. **Acquisition contract is already documented.** scripts/download_f3.py (lines 28–120) specifies the source, URL, license, format, and drop paths. No new contract decisions are needed — this formalizes what was already written.

3. **Parser choice is parse_opendtect_fault_sticks.** F3 fault interpretations are distributed as OpendTect ASCII fault-stick exports. The Petrel parser is for Volve's Petrel export format only.

4. **Staged path convention.** Follows the established staged/surveys/{survey_id}/ ADLS convention.

## Contract Summary

| Item | Value |
|------|-------|
| Seismic source | OpendTect F3 Demo (dGB / TerraNubis), SEG-Y |
| Source URL | https://terranubis.com/datainfo/F3-Demo-2020 |
| License | CC BY-SA |
| Fault parser | parse_opendtect_fault_sticks (label_generator.py:346) |
| Fault drop path | data/f3/interpretations/fault_sticks/*.dat (OpendTect ASCII) |
| Staged seismic | staged/surveys/f3-demo/amplitude.zarr (array: mplitude) |
| Staged labels | staged/surveys/f3-demo/fault_label.zarr (array: ault_mask) |

## Leakage Gate

**F3 = training input only. Volve = scoring/evaluation target only.**  
Volve fault sticks must never appear in any F3 training job invocation. Hard rule per issue #24.

---


# Ash SME Review — Fault Detection Metrics & Targets (Issue #37)

**Date:** 2026-07-13  
**Author:** Ash (Geophysicist SME)  
**Relates to:** Issue #37 (metric fix / v0.7.2), Issue #24 (fault over-prediction)  
**Status:** Recommendation — no code changes required

---

## Context

F3 cross-survey UNet fault detection. ~0.08% positive voxels. v0.7.2 fixed the degenerate IoU metric (hardcoded 0.5 threshold on a 97%-background distribution → IoU=0 every epoch). New scoring: IoU@best-threshold from a 0.05–0.95 grid sweep + val Average Precision (PR-AUC). This review assesses whether that is the right approach geophysically and sets a credible target range for the upcoming re-run.

---

## Q1 — Is IoU@best-threshold + AP the right scoring approach?

**Short answer: Yes for a PoC. AP is the primary; IoU is secondary.**

**Reasoning:**

- At ~0.08% prevalence, the PR curve is far more informative than the ROC curve. AP (PR-AUC) integrates precision over all recall levels and is insensitive to the overwhelming background class — it's the correct summary statistic for a severely imbalanced binary task.
- IoU@best-threshold is a useful sanity check and interpretable to non-ML stakeholders ("fraction of fault voxels correctly identified with minimal false alarms"), but it is harsh: a prediction displaced 1 voxel from the interpreter's pick scores zero TP even if geophysically correct. This is a **systematic understatement** given fault-pick uncertainty.
- **What we should NOT use as a standalone metric:** fixed-threshold IoU (the original broken metric), accuracy (97% trivially by predicting all-background), or ROC-AUC (insensitive to class imbalance at this prevalence).

**Alternatives worth tracking but not yet implementing for PoC:**

| Metric | Geophysical case for | Practical obstacle for PoC |
|--------|---------------------|---------------------------|
| Dice/F1@best-T | Less harsh than IoU (2TP in numerator) | Monotonically related to IoU@best-T; adds limited new information |
| Tolerance-band IoU (±2 voxel buffer) | Credits geophysically near-correct predictions; respects picking uncertainty | Requires defining the tolerance parameter a priori — no established value for this volume |
| Skeleton/Hausdorff distance | Measures fault geometry quality, not voxel overlap | Requires thinned/skeletonized fault masks; adds engineering overhead |
| Precision@fixed recall (e.g., P@R=0.5) | Operationally useful: "what precision do we get when we find half the faults?" | Useful once we know R=0.5 is achievable — not yet established |

**Recommendation:** Retain IoU@best-threshold + AP as-is. Document AP as the primary ranking signal in training logs. Add Dice@best-threshold to the logged row for readability (it's computed for free from the same TP/FP/FN accumulators).

---

## Q2 — Single global threshold or depth/bandwidth-dependent?

**Single global threshold is appropriate for this PoC.**

**Geophysical nuance acknowledged:**  
Fault detectability does vary with depth. Q attenuation (North Sea F3 ≈ Q 80–120) reduces bandwidth and lowers S/N with depth. Shallow faults (< ~1.2 s TWT in F3) are imaged at ~50–60 Hz; deeper events may be limited to ~30–40 Hz, lowering vertical resolution and making faults with small throw invisible. This means a single global threshold optimized on the full validation distribution will be dominated by the easiest-to-detect (shallow, large-throw) fault population.

**Practical decision:** A global sweep is still the right call for a PoC because:  
1. We don't have enough labeled data per depth window to optimize per-window thresholds reliably.  
2. The metric is already being swept (not hardcoded) — this is the correct direction.  
3. Depth-dependent thresholding is a product-level refinement.

**Diagnostic signal to log:** If `best_threshold` from the sweep converges below 0.15 or above 0.80 in multiple consecutive epochs, that is diagnostic — not a tuning success. Low threshold → model is in a high-recall/low-precision regime (over-predicting, link to issue #24). High threshold → model is under-predicting or overconfident on background.

---

## Q3 — Credible target ranges for F3 held-out performance

The prior Volve run achieved eval IoU=0.0622, Dice=0.1172, Recall=0.43 on a ~0.16% prevalence held-out region with only ~18 fault stick picks. F3 has a denser, better-documented fault system (multiple published papers have used it for fault detection benchmarks).

**Expected ranges for a competent-but-not-SOTA 3D UNet on F3 (20 epochs, no test-time augmentation, no ensemble):**

| Category | Val AP | Val IoU@best-T | Val Dice@best-T | Interpretation |
|----------|--------|---------------|-----------------|----------------|
| **Broken / degenerate** | < 0.05 | ~0.000 | ~0.000 | No threshold improves on all-background; model not learning faults at all. Investigate pos_weight, data loading, or label integrity. |
| **Clearly learning** | 0.08–0.20 | 0.03–0.10 | 0.06–0.18 | Model detects major faults; precision/recall trade-off still poor. Loss curve should show steady descent. This is the threshold for "model is not broken". |
| **Competent PoC** | 0.20–0.45 | 0.10–0.25 | 0.18–0.40 | Major faults reliably detected; some smaller faults missed; modest over-prediction. Suitable for a demo and for feeding interpretation workflows. |
| **Publishable / near-SOTA** | > 0.45 | > 0.25 | > 0.40 | Consistent with published F3 fault detection results (e.g., Wu 2019, Cunha 2020 range). Requires data augmentation, multi-scale supervision, or semi-supervised labels. |

**"Clearly learning" threshold (minimum bar for the upcoming re-run):**
- Val AP > 0.08 AND val loss still decreasing after epoch 10
- Best threshold should not be pinned at 0.05 (floor) — that would indicate the model is still collapsing to all-positive in the high-recall/zero-precision corner

**Note on Volve baseline:** The historical Volve IoU=0.0622 with only 18 fault picks and a small volume is in the "clearly learning" range. F3 with a proper label set and the fixed metric should reach "competent PoC" within 20–30 epochs.

---

## Q4 — Geophysical Gotchas

**Things that make a GOOD model look BAD:**

1. **Label-dilation mismatch at evaluation time.** Our 3-voxel cubic dilation (7-voxel-wide labels, ~28 ms TWT) is within λ/4, but if the model predicts 3-voxel-wide fault planes it will have artificially low precision against 7-voxel labels. Conversely, if evaluation patches use a different dilation, precision/recall numbers shift. **Always document dilation at label-generation time and apply the same dilation to evaluation labels.**

2. **Amplitude-vs-attribute training mismatch.** F3 OpendTect fault sticks were almost certainly picked on coherence/similarity attribute volumes, not raw amplitude. The model trains on amplitude. Fault expressions in amplitude (reflection discontinuity, pull-up/pull-down) are often spatially displaced by 1–3 voxels from the coherence anomaly center. This generates systematic spatial offsets between predictions and labels → lower IoU, not worse geometry.

3. **Inline spatial split creates regional bias.** The 70/15/15 inline split means the test region is a contiguous block of inlines. If F3's major fault planes are oblique to the inline direction (likely — faults in this field trend NW-SE), the test block may intersect only minor fault segments, and major fault planes will all be in the training set. This inflates apparent train IoU and deflates test IoU.

4. **Label sparsity floor.** At ~0.08% prevalence, even 5 false-positive voxels per patch can crater precision. The threshold sweep will optimize for F1/IoU balance but may select a threshold that excludes real low-confidence fault predictions in structurally complex zones (relay ramps, bifurcations).

**Things that make a BAD model look GOOD:**

5. **Wide dilation makes diffuse predictions acceptable.** A model that predicts broad "fault halos" (e.g., due to insufficient bottleneck capacity) can score well against 7-voxel-wide dilated labels while producing geologically useless output. If IoU looks good but `best_threshold` is low (< 0.2) and the prediction mask is visually diffuse, the model is not learning fault geometry — it's learning "near a fault somewhere."

6. **Gas chimney / acoustic mask correlation.** If F3 test inlines contain gas chimneys that coincidentally overlap with fault locations, a model that detects any coherence breakdown will look correct even if it's detecting gas effects, not faults. This is a real risk in the southern F3 survey area (known shallow gas). Visually inspect predictions against the seismic before declaring success.

7. **Overfitting to fault throw amplitude.** Faults in F3 often have large throws creating strong reflection offsets. A model that simply detects strong amplitude gradients (a proxy for throw) will score well on training data where throw is the dominant fault expression but will fail on low-throw, stratigraphic faults that constitute real geological uncertainty.

---

## Recommendations Summary

| Decision | Recommendation |
|----------|---------------|
| **Primary metric** | Val AP (PR-AUC) — threshold-free, imbalance-robust |
| **Secondary metric** | IoU@best-T and Dice@best-T (same computation, more interpretable) |
| **Threshold strategy** | Single global sweep (0.05–0.95) — appropriate for PoC |
| **Checkpoint selection** | Best-by-val-AP (preferred) or best-by-val-loss fallback — current code uses IoU which is fine but AP is more stable at low counts |
| **"Clearly learning" bar** | Val AP > 0.08 AND best_threshold not pinned at floor |
| **"Competent PoC" bar** | Val AP 0.20–0.45, IoU@best-T 0.10–0.25 |
| **Issue #24 diagnostic** | If best_threshold ≤ 0.10 in steady state → over-prediction problem persists; increase pos_weight or add harder negatives |
| **Label QC before re-run** | Verify dilation value used in F3 label generation matches what training expects; document it |



---



# Decision: F3 Real-Data Ingest Fixes (irregular SEG-Y + F3 fault parser)

**Date:** 2026-07-10T16:24:40-05:00
**Author:** Dallas (Data/ML Engineer)
**Status:** Proposed — pending coordinator review; implemented in PR #32
**Triggered by:** Issue #31 (infra ran documented ingest CLI against REAL F3 Demo 2023 SEG-Y in-VNet; two app-code blockers found)

## Decision

The ingest pipeline handles real F3 irregular geometry and F3's native fault-export
format directly:

1. **Irregular SEG-Y fallback** triggers on any of the known segyio geometry-error
   messages (`inconsistent`, `invalid dimensions`, `should match the number of
   traces`), not just `inconsistent`. The header-driven reconstruction derives the
   IL/XL grid from actual `INLINE_3D`/`CROSSLINE_3D` header min/max (bytes 189/193),
   zero-fills absent boundary traces, and last-wins de-dupes.

2. **Amplitude sidecar carries world georeference.** The loader extracts three
   corner tie-points from `CDP-X`/`CDP-Y` (bytes 181/185, coordinate scalar byte 71
   applied) into `SurveyGeometry.corner_points`, enabling world→(inline, crossline)
   transforms downstream.

3. **F3 fault format is first-class.** `parse_f3_fault_sticks` handles the headerless
   5-column map-coordinate export (X Y Z_ms stick_id point_id), grouping by
   `stick_id`. `generate_fault_label.py --fault-format f3` rasterises via the
   world-coordinate path. Default remains `volve` (index-space, unchanged).

## Rationale

- Real F3 has ~434 irregular/edge traces → `ilines*xlines != tracecount`; the
  structured segyio open fails. The fallback existed but its trigger was too narrow.
- F3's fault export is world-coordinate, not index-space; the world path needs a
  transform, which needs corner-point georeference that the sidecar previously lacked.
- Keeping the Volve default preserves all existing behaviour and tests.

## Team Impact

- **Parker (infra):** re-run the in-VNet ingest against PR branch
  `x3nc0n/f3-irregular-segy-and-fault-parser` (or `main` after merge). After ingest,
  confirm the sidecar's `geometry.corner_points` is non-null and X/Y look like F3 UTM
  metres; flag if CDP coordinate scalar/bytes differ on the licensed file.
- **Anyone consuming `amplitude.json`:** new optional `geometry.corner_points`
  field (`[[x, y, il, xl], ...]` or null).

## Leakage Gate

Unchanged: **F3 = training input only; Volve fault sticks = scoring/eval key only,
never a training input** (#24). Nothing here alters that boundary.

## Evidence

- `src/deepseismic/ingest/segy_loader.py:71` — `_is_irregular_geometry_error`
- `src/deepseismic/ingest/segy_loader.py:270` — broadened `load()` guard
- `src/deepseismic/ingest/segy_loader.py` — `_extract_corner_tie_points` (CDP 181/185)
- `src/deepseismic/ingest/label_generator.py:~403` — `parse_f3_fault_sticks`
- `scripts/generate_fault_label.py` — `--fault-format` flag + `_build_survey_transform`
- `src/tests/test_ingest/test_f3_realdata.py` — 12 regression tests
- PR: https://github.com/x3nc0n/deepseismic2/pull/32


---



# Decision: v0.7.2 Loss-Fallback Bug Fix (patch release needed)

**Date:** 2026-07-13  
**Author:** Dallas  
**Status:** Committed (1d184c6), needs patch release tag

## Context

Verified v0.7.2 fix for issue #37 (cross-survey F3 training producing val IoU=0 for 50 epochs). The 5 shipped items (threshold sweep, AP, robust checkpoint, threshold persistence, unit tests) are substantially correct — but one bug was found in `_select_best_checkpoint`.

## Bug

`not best_saved` guard on the loss fallback caused best.pt to capture only epoch 1 in the all-zero-IoU regime, not the actual best-by-loss checkpoint. In a 50-epoch run where loss typically drops from ~0.9 to ~0.4, the saved checkpoint would reflect the worst-loss epoch.

## Fix Applied

Changed `elif not best_saved and val_metrics["loss"] < best_val_loss:` to `elif val_metrics["iou"] >= best_val_iou and val_metrics["loss"] < best_val_loss:` in `train.py:_select_best_checkpoint`. Added 2 regression tests.

## Recommendation

Tag a patch release (v0.7.3 or similar) from commit `1d184c6` before infra launches the T4 re-run. The fix is surgical and all 389 tests pass.

## De-risk Evidence

Synthetic sparse-positive test (3% faults, sub-0.5 probs):
- IoU@0.5 (old hardcoded): 0.0000  
- IoU@best-thr sweep (0.05): 1.0000  
- AP: 1.0000  

The threshold sweep provably recovers signal that 0.5-threshold would miss.


---



# Hudson Review Verdict — commit 1d184c6 (issue #37 best-checkpoint fix)

**Date:** 2026-07-13  
**Reviewer:** Hudson (Tester/QA)  
**Author:** Dallas  
**Scope:** `_select_best_checkpoint` in `src/deepseismic/training/train.py`; two new regression tests in `test_sprint2_training.py`

---

## VERDICT: ✅ APPROVE

Dallas's fix is logically correct, the regression tests genuinely guard the epoch-1 bug, and the full suite is green. Safe to release as v0.7.3 and request infra re-run on a real T4 GPU.

---

## Evidence

### 1. Logic review

**Old condition (buggy):**
```python
elif not best_saved and val_metrics["loss"] < best_val_loss:
```
After epoch 1 sets `best_saved=True`, this guard is permanently blocked for all subsequent epochs. In a 50-epoch all-zero-IoU run, `best.pt` = epoch-1 checkpoint regardless.

**New condition (fix):**
```python
elif val_metrics["iou"] >= best_val_iou and val_metrics["loss"] < best_val_loss:
```

Verified cases:
| Scenario | Result |
|---|---|
| All-zero IoU, epoch 1 (best_val_loss=inf) | ✅ saves via `0.0 >= 0.0 and loss < inf` |
| All-zero IoU, later epoch with lower loss | ✅ saves via `0.0 >= 0.0 and loss < prev_best` |
| Normal run, IoU improves | ✅ IoU primary branch fires (`iou > best_val_iou`) |
| IoU regresses, loss improves | ✅ blocked by `iou >= best_val_iou` (0.2 >= 0.3 → False) |
| Epoch 1 guarantee (best.pt always written) | ✅ any real loss < inf, iou ≥ 0.0 → saves |

### 2. Revert/restore proof

- Temporarily reverted to `not best_saved` condition
- `test_fallback_updates_best_on_subsequent_loss_improvement_when_iou_zero` → **FAILED** on old code  
- Restored Dallas's fix
- All 4 `TestBestCheckpointSelection` tests → **PASS** ✅

The regression test is a genuine guard, not coverage filler.

### 3. Full suite + lint

- `pytest -m "not integration" -q` → **391 passed**, 2 skipped, 9 deselected ✅
- `ruff check src/ scripts/` → **All checks passed** ✅
- Working tree left exactly as Dallas committed (verified `git diff HEAD` is empty)

### 4. Leakage gate & "best.pt ALWAYS saved" (item-3, issue #37)

- The patch-split leakage test (`test_dilation_zero_no_neighbour_leakage`) remains in the 391-pass suite ✅
- best.pt ALWAYS saved: epoch 1 with initial `best_val_loss=inf` and `best_val_iou=0.0` guarantees the loss-fallback branch fires for any real loss value ✅

---

## No issues found. Recommend: release v0.7.3, request infra T4 re-run.


---



# Decision: UI Design Direction — Impeccable Pass (Amber/Stone/Teal + Barlow)

**Date:** 2026-07-13  
**Author:** Lambert  
**Status:** Proposed (draft PR #TBD → main)

## Context

The Gradio app (`src/deepseismic/ui/gradio_app.py`) had a generic AI-default design: Inter for everything (Impeccable's #1 AI tell), blue/slate Gradio theme, no type hierarchy, no intentional palette. Impeccable was installed and applied to improve the visual design without breaking the backend or test suite.

## Decision

**Adopt the following design direction for the Gradio UI:**

### Typography
- **Heading/label:** Barlow Condensed (500–700 weight) via Google Fonts CSS import — industrial precision, geological survey aesthetic, not on Impeccable reflex-reject list.
- **Body:** Barlow (theme `font`) — humanist sans, readable at tool/dashboard density.
- **Mono:** Fira Code (theme `font_mono`) — characterful, ligature-enabled. Replaces JetBrains Mono.
- **Hierarchy:** All block labels rendered in condensed uppercase via CSS (`.block .label-wrap span`). Header title in display-weight uppercase.

### Color
- `primary_hue`: `amber` — earthy/geological (core samples, sediment cross-sections). Replaces generic `blue`.
- `neutral_hue`: `stone` — warm gray that coheres with amber. Replaces cold `slate`.
- `secondary_hue`: `teal` — subsurface depth accent. Replaces second `slate`.
- Rationale: avoids Impeccable-flagged purple-blue SaaS gradient; grounded in domain (subsurface geoscience).

### Copy / Labels
- "Agent Conversation" → "Analyst Chat"
- "Seismic Inline Viewer" → "Inline Section"
- "Domain Perspective" → "Analyst Perspective"
- "Send" button → "Send ↩" (affordance clarity)
- Viewer empty state: more instructional copy
- Inference accordion: shorter, action-oriented copy
- Title / DESCRIPTION: punchier, concise

### CSS Surface
- Custom CSS constant (`_CUSTOM_CSS`) injected via `gr.Blocks(css=...)`.
- `elem_id` and `elem_classes` added to key components for stable CSS targeting.
- `@import` loads Barlow Condensed from Google Fonts CDN.

## Keep Impeccable installed?
**Yes.** The `.github/skills/impeccable/` files and `.github/hooks/impeccable.json` should stay in the repo. They enable future `/typeset`, `/colorize`, `/polish`, and `/audit` passes on any UI surface in the project. Cost: ~3 MB of JS/MD files in `.github/`.

## Out of scope
- Storage browser reposition (backlog issue #39) — intentionally deferred.
- No backend logic changes. No test suite changes.

## Consequences
- Visual identity is now intentionally geoscience-specific, not generic SaaS.
- Any future Gradio theme changes should start from `amber/stone/teal` as the baseline palette.
- If dark-mode theming is added later, tint the dark surface toward amber (not toward blue).


---



# Decision: Pin gradio<6 (v0.8.1 patch — latent UI container boot bug)

**Date:** 2026-07-14  
**Author:** Parker  
**Status:** Done — v0.8.1 shipped, CD in_progress

## Context

After shipping v0.8.0 (UI redesign, PR #40), post-release smoke testing revealed the deployed Gradio UI container would not boot. The container resolved **gradio 6.17.3** (latest) at build time because:

1. `pyproject.toml` [ui] extra had `gradio>=4.40.0` — **no upper bound**.
2. `docker/Dockerfile.gradio` had an extra `RUN pip install --no-cache-dir gradio matplotlib pillow` that ran *after* `.[ui]` and won the version race, overriding the pyproject range with an unconstrained latest install.

No lock file exists in the repo, so every image build resolved whatever PyPI latest was at that moment.

## Breaking changes in gradio 6

| API | gradio 4/5 | gradio 6 | Impact |
|-----|-----------|---------|--------|
| `gr.Chatbot(type="messages")` | Supported | **Removed** (`TypeError`) | Fatal — crashes at module import |
| `gr.Blocks(theme=..., css=...)` | Supported | **Silently ignored** (moved to `launch()`) | Silent — theme/CSS dropped from v0.8.0 redesign |

The code itself is correct for the gradio API it was written against. The bug is purely a missing version ceiling.

## Decision: Pin `<6`, do NOT migrate to gradio 6 now

**Options considered:**

| Option | Pros | Cons |
|--------|------|------|
| Pin `<6` (chosen) | Minimal-diff patch; ships immediately; unblocks deployed UI | App code stays on gradio 4/5 API; gradio 6 migration deferred |
| Migrate to gradio 6 API | Stay on latest gradio | Non-trivial: `Chatbot` `type=` removal + `Blocks` → `launch()` for theme/css; risk of undiscovered additional API breaks; wrong scope for an emergency patch |

**Rationale for pinning:**
- Pinning is the correct emergency fix for a PoC with no lock file.
- The gradio 4/5 API is stable and will continue to receive security patches within the `<6` range.
- Migrating to gradio 6 is a **feature branch task** — it should be done in a dedicated PR with full UI regression testing, not as a hot patch.

## Changes shipped in v0.8.1

- `pyproject.toml` `[ui]` extra: `gradio>=4.40.0` → `gradio>=4.44.0,<6`
  - Floor raised to 4.44 (the version where `type="messages"` is confirmed solid).
  - Ceiling added: `<6` keeps us on the gradio 4/5 API.
- `docker/Dockerfile.gradio`: removed `gradio` from the explicit `pip install` line.
  - `.[ui]` install on the preceding line now governs the version via pyproject.
  - `matplotlib` and `pillow` kept on the explicit line (pillow not in core deps).

## Verification

Smoke tested on gradio 5.50.0 (resolved by `>=4.44.0,<6`):
- UI import: `python -c "import deepseismic.ui.gradio_app; print('UI import OK')"` — clean ✅
- 391 passed, 2 skipped (non-integration pytest) ✅
- ruff clean ✅

## The Dockerfile unpinned-install foot-gun

**Root pattern to prevent in future:**
> Dockerfile lines of the form `RUN pip install <pkg>` alongside a `pip install .[extra]` are dangerous in lockless repos. When the package also appears in `pyproject.toml`, the explicit Dockerfile line wins the version race at container build time and can silently pull a major-version bump. The pyproject ceiling is effectively bypassed.

**Correct pattern:**
- Let `pip install .[extra]` be the single source of truth for all packages that are already in `pyproject.toml`.
- Only use explicit `RUN pip install <pkg>` for packages that are *not* in `pyproject.toml` (e.g., `pillow` for the Gradio image, system-level tools).

## Follow-up

- Gradio 6 migration: open follow-up issue — update `Chatbot` `type=` param, move `theme`/`css` to `launch()`, run full UI regression. Unblock only after manual UI sign-off.
- Lock file consideration: evaluate `pip-compile` or `uv lock` for reproducible container builds (tracked separately in infra planning).


---



# Decision: v0.7.3 Released — Infra Re-run Requested for Issue #37

**Date:** 2026-07-13  
**Author:** Parker  
**Status:** Done — pending infra re-run results

## Context

Issue #37 (cross-survey F3 training, val IoU=0 for 50 epochs) had two sequential fixes:
- **v0.7.2** (PR #38): Added 0.05–0.95 threshold sweep, val AP, per-epoch best-threshold logging, robust checkpoint selection.
- **v0.7.3** (direct push to main): Fixed a bug in the `_select_best_checkpoint` loss-fallback — the `not best_saved` guard permanently blocked after epoch 1, meaning `best.pt` always captured the worst-loss (epoch-1) checkpoint in a degenerate all-zero-IoU run. Fix: guard now uses `iou >= best_val_iou and loss < best_val_loss` to track the lowest-loss checkpoint across all epochs.

Hudson independently verified: regression test fails on old code, passes on new. 391 passed / 2 skipped. Ruff clean.

## Actions Taken

1. **Committed** Hudson's history notes (unstaged): `docs(squad): hudson review notes for v0.7.3 best-checkpoint fix`
2. **Bumped** `pyproject.toml` version 0.7.2 → 0.7.3
3. **Committed** version bump: `chore(release): v0.7.3 — best-checkpoint loss-fallback fix (#37)`
4. **Pushed** directly to `main` (no branch protection blocking — chore/release convention)
5. **Tagged** GitHub release: https://github.com/x3nc0n/deepseismic2/releases/tag/v0.7.3
6. **CD confirmed** started (`cd.yml` in_progress within ~24s of push — builds ghcr.io/x3nc0n/deepseismic2-api and -ui tagged latest + sha)
7. **Infra notified** via Spava-Corp/deepseismic2-infra#19 — requested warm rebuild from v0.7.3 and re-run of 50-epoch F3 job with new run-id, same training flags
8. **App issue #37** commented — noted v0.7.3 shipped, Hudson-verified, infra pinged; issue left open pending re-run

## Next Steps

- Infra team rebuilds from v0.7.3 / GHCR latest → ACR, runs 50-epoch F3 job
- Expected output: non-zero val IoU@best-threshold, val AP, real best.pt uploaded
- Infra to post metrics table in deepseismic2-infra#19
- App team reviews model quality → feeds issue #24 (Volve scoring readiness)
- Close #37 only after re-run results are reviewed

## Land Path Note

`chore(release)` commits go directly to `main`. Code-fix PRs squash-merge via PR flow. This is the established convention (see v0.7.0/0.7.1/0.7.2 pattern in git log).


---


