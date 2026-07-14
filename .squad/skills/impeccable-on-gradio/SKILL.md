# Skill: Applying Impeccable Design Principles to a Gradio UI

**Author:** Lambert | **Date:** 2026-07-13  
**Context:** deepseismic2 — but the pattern is reusable for any Gradio-based AI tool.

---

## Why this is non-obvious

Impeccable (https://impeccable.style) is built for hand-authored HTML/CSS/React frontends. Its deterministic detector (`detect.mjs`) runs on HTML/CSS/JS and returns **0 findings on Python source files**. Its harness commands (`/typeset`, `/colorize`, etc.) expect to inspect and edit component CSS. A Gradio app generates its own DOM — you cannot hand-write arbitrary markup.

This skill documents which Impeccable levers actually work on Gradio, and how to use them.

---

## Installation

```bash
npx impeccable install   # installs to .github/skills/impeccable/
```

Detected harness (GitHub Copilot) and installed to `.github/`. This adds ~3 MB of JS/MD tooling. Keep it — enables future passes.

The CLI `--version` check: `npx impeccable --version` downloads on first use, then cached.

---

## The four real design levers on a Gradio app

### 1. Gradio theme object
The `gr.themes.Base(...)` object is your primary design surface:

```python
_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,   # NOT blue (AI default)
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.stone,   # warm gray, not cold slate
    font=[gr.themes.GoogleFont("Barlow"), "sans-serif"],   # NOT Inter
    font_mono=[gr.themes.GoogleFont("Fira Code"), "monospace"],
    text_size=gr.themes.sizes.text_md,
    radius_size=gr.themes.sizes.radius_sm,  # tighter for technical tools
).set(
    block_label_text_weight="600",
    block_title_text_weight="600",
    block_border_width="1px",
    input_border_width="1px",
)
```

`.set()` exposes CSS variable overrides. Valid names: `body_background_fill`, `block_background_fill`, `button_primary_background_fill`, `block_label_text_size`, `block_label_text_weight`, etc.

### 2. Custom CSS via `gr.Blocks(css=...)`
Inject arbitrary CSS as a string. Use `@import` for additional Google Fonts. Gradio injects this as a `<style>` tag in `<head>`.

```python
_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&display=swap');

:root {
    --ds-font-display: 'Barlow Condensed', sans-serif;
    --ds-track-label: 0.08em;
}
/* Target components by elem_id: */
#ds-chatbot .message { line-height: 1.65; }
.ds-quick-btn button { font-family: var(--ds-font-display) !important; text-transform: uppercase; }
"""

with gr.Blocks(css=_CUSTOM_CSS, ...) as demo:
    ...
```

### 3. `elem_id` and `elem_classes` on components
These create stable CSS hooks — the only reliable way to target specific Gradio components (Gradio's generated class names change between versions):

```python
chatbot = gr.Chatbot(elem_id="ds-chatbot", ...)
btn_status = gr.Button("Status", elem_classes=["ds-quick-btn"], ...)
viewer_status = gr.Markdown("...", elem_id="ds-viewer-status")
```

### 4. Copy / labels / empty states
These are always available:
- Change `label=` on any component
- Change `placeholder=` on textboxes
- Change static markdown text
- Change button text (`.value` at definition time)

---

## Impeccable principles applied manually (single-context fallback)

Since the detector can't scan Python, apply the assessment checklists manually:

### /typeset checklist (applied)
1. **Font choices:** Inter = AI tell #1. Replace with a characterful font matching the domain. For a geoscience tool: `Barlow Condensed` (industrial/precision) + `Barlow` (body) + `Fira Code` (mono).
2. **Hierarchy:** All block labels in condensed uppercase via CSS. Title in display weight. Body text at 0.875–0.9rem.
3. **Sizing:** Body at 0.9rem minimum. Label at 0.67rem condensed caps (distinguishable from body without being too small).
4. **Readability:** Line-height 1.65 for chatbot messages (light-on-dark compensation); 1.5 for description text.

### /colorize checklist (applied)
1. **Strategy:** Restrained (product register). Accent = amber. Neutral = stone.
2. **Domain-specific:** Amber evokes geological core samples, sediment cross-sections. NOT the generic purple-blue SaaS gradient.
3. **60-30-10 rule:** Stone surfaces (60%), slate/warm-gray text (30%), amber accents on interactive elements (10%).

### /layout checklist (applied)  
1. **Header:** Tightened vertical spacing between title and description via CSS (`margin-bottom: 0.15rem`).
2. **Labels:** All in condensed caps → clear visual hierarchy between label and content.
3. **Status lines:** `font-variant-numeric: tabular-nums` for coordinate/count readability.

### /audit — anti-patterns addressed
- ✅ Inter removed (AI tell #1)
- ✅ Generic blue/slate removed
- ✅ No purple-blue gradient
- ✅ Copy improved: shorter, action-oriented, domain-appropriate

---

## What doesn't work on Gradio

- ❌ Hand-written DOM structure — Gradio generates it
- ❌ Container queries — Gradio's component wrappers don't expose container-type
- ❌ CSS Grid page-level restructuring — use Gradio's `gr.Row`/`gr.Column` instead
- ❌ Dark mode tokens — Gradio handles dark mode; `.set()` has `_dark` suffix variants
- ❌ `border-left` accent stripes (Impeccable absolute ban anyway — use surface tints)

---

## Selector reliability notes

Gradio 4.x stable selectors:
- `#<elem_id>` — always reliable if you set `elem_id=`
- `.<elem_class>` — reliable if you set `elem_classes=[]`
- `.block .label-wrap span` — block label text (stable in Gradio 4.x)
- `.message` inside `#chatbot` — chatbot message bubbles
- `.gradio-accordion .label-wrap > span` — accordion headers
- `.gradio-container` — top-level wrapper (always present)

Avoid targeting Gradio's generated utility classes (they begin with `svelte-*` or have hash suffixes) — they change on every Gradio release.

---

## Font loading notes

`@import url(...)` in a `<style>` block works in all modern browsers. Gradio injects the CSS block synchronously at page load, so fonts are requested early. For production, preconnect hint could be added but CSS `@import` is sufficient for a tool app.

`gr.themes.GoogleFont("FontName")` handles the body font; CSS `@import` is needed for any additional weight/style variants (like Barlow Condensed at 500/600/700 that isn't included in the body Barlow request).
