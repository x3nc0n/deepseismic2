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
