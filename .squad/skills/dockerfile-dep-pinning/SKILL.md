# Skill: Pin Runtime Deps with Upper Bounds — Dockerfile Must Not Override pyproject

**Author:** Parker | **Date:** 2026-07-14  
**Context:** deepseismic2 — but the pattern is universal for any lockless Python project with a Dockerfile.

---

## Why this is non-obvious

In a repo without a lock file (`pip-compile`, `uv lock`, `requirements.txt` with pinned versions), the ONLY protection against silent major-version upgrades at container build time is an upper bound in `pyproject.toml`. That protection is **silently bypassed** if the Dockerfile has a separate `pip install <pkg>` line for the same package — that line wins the version race and resolves the unconstrained latest.

This cost us a broken `latest` deploy in v0.8.0: gradio 6.x resolved at build time because `Dockerfile.gradio` had `pip install gradio` after `pip install .[ui]`.

---

## The foot-gun pattern

```dockerfile
# pyproject.toml has: gradio>=4.44.0,<6
RUN pip install --no-cache-dir ".[ui]"          # installs gradio 5.x ✅
RUN pip install --no-cache-dir gradio matplotlib pillow   # ← OVERRIDES → installs gradio 6.x 💥
```

The second line re-resolves `gradio` with NO constraint, overwriting the version that `.[ui]` installed. Since `pip install <pkg>` with no version spec always pulls the latest, every CI/CD build gets whatever PyPI serves that day.

---

## The correct pattern

**Rule: Let `pyproject.toml` be the single source of truth for every package that already appears there.**

```dockerfile
# ✅ Correct: .[ui] governs gradio via pyproject.toml bound
RUN pip install --no-cache-dir ".[ui]" || pip install --no-cache-dir .
# Only list packages NOT in pyproject.toml in additional pip install lines:
RUN pip install --no-cache-dir pillow   # pillow not in core deps, OK to list separately
```

If you also need `matplotlib` and it IS in pyproject core deps, don't list it separately — it's already installed by `.[extra]`.

---

## Upper bound convention for UI/framework packages

Any package that ships breaking changes between major versions and has slow API surface area (Gradio, Streamlit, FastAPI, etc.) **must** have both a floor AND a ceiling in `pyproject.toml`:

```toml
# ✅ Correct — bounded range
"gradio>=4.44.0,<6"

# ❌ Wrong — no ceiling; will silently break on next major
"gradio>=4.40.0"
```

**When to set the floor:** The earliest version where the feature you depend on is stable. For gradio `type="messages"` on Chatbot: 4.44.

**When to set the ceiling:** One major version above the current API target (`<6` when app uses gradio 4/5 API).

---

## Verification checklist before shipping

After changing a UI/framework dep bound:

```powershell
# 1. Install the constrained range (forces pip to resolve within bounds)
pip install "gradio>=4.44,<6"

# 2. Smoke the import — catches module-level build errors
python -c "import deepseismic.ui.gradio_app; print('UI import OK')"

# 3. Full suite
python -m pytest -m "not integration" -q

# 4. Lint
python -m ruff check src/ scripts/
```

All four must pass before tagging a release.

---

## Gradio 6 API breaks (for future reference)

When the team is ready to migrate to gradio 6:

| Feature | gradio 4/5 | gradio 6 |
|---------|-----------|---------|
| `gr.Chatbot(type="messages")` | Supported | Removed — `type` kwarg gone |
| `gr.Blocks(theme=..., css=...)` | Supported | Silently ignored — move args to `launch()` |

The migration is a feature branch task. Minimum changes:
1. Remove `type="messages"` from `gr.Chatbot(...)` (or use the 6.x equivalent).
2. Move `theme=_THEME, css=_CUSTOM_CSS` from `gr.Blocks(...)` to `demo.launch(...)`.
3. Re-test full UI (theme rendering, font import, custom CSS hooks via `elem_id`).

Do NOT migrate as a hot patch — it requires manual UI sign-off.

---

## Related

- Decision: `.squad/decisions/inbox/parker-gradio6-pin.md`
- Incident: v0.8.0 shipped with broken UI container (gradio 6.17.3 resolved at build time)
- Fix shipped: v0.8.1 (`gradio>=4.44.0,<6` in pyproject + dropped bare `gradio` from Dockerfile.gradio)
