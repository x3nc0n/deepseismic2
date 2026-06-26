"""Gradio demo application for the DeepSeismic Analyst agent.

Provides a minimal but demo-ready interface with:
- **Project picker** — browse ADLS storage to select surveys/datasets
- **Chatbot** — full conversation history with the agent
- **Seismic image** — inline section with optional fault probability overlay
- **Controls** — inline/crossline selector, persona dropdown, quick-action buttons

Run with::

    python src/deepseismic/ui/gradio_app.py

Or with mock mode::

    MOCK_LLM=true python src/deepseismic/ui/gradio_app.py
"""

from __future__ import annotations

import io
import os
from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from deepseismic.ui import _viewer_api as vapi
from deepseismic.ui._viewer_api import ViewerAPIError

matplotlib.use("Agg")  # Non-interactive backend for server rendering

MOCK_MODE: bool = os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")
API_BASE_URL: str = vapi.api_base_url()

# Default checkpoint blob in the 'features' container for live inference.
CHECKPOINT_BLOB: str = os.environ.get(
    "DEEPSEISMIC_CHECKPOINT_BLOB", "checkpoints/unet3d_best.pt"
)

# ---------------------------------------------------------------------------
# Agent singleton (one per server process)
# ---------------------------------------------------------------------------

_agent: Any = None


def _get_agent() -> Any:
    global _agent  # noqa: PLW0603
    if _agent is None:
        from deepseismic.agent.agent import DeepSeismicAgent
        _agent = DeepSeismicAgent()
    return _agent


# ---------------------------------------------------------------------------
# ADLS Project Browser
# ---------------------------------------------------------------------------

def _browse_storage(container: str, prefix: str = "") -> list[dict]:
    """Fetch folder/file listing from the API browse endpoint.

    Best-effort variant (returns [] on error) kept for legacy helpers.  The
    interactive listing uses :func:`vapi.browse` directly so it can surface
    errors instead of silently showing an empty tree.
    """
    try:
        return vapi.browse(container, prefix, API_BASE_URL)
    except ViewerAPIError:
        return []


def _format_browse_tree(container: str, prefix: str = "") -> str:
    """Build a displayable tree view for the current path."""
    items = _browse_storage(container, prefix)
    if not items:
        return "_No items found._"

    lines = []
    for item in items:
        if item["type"] == "folder":
            lines.append(f"📁 **{item['name']}/**")
        else:
            size_mb = (item.get("size") or 0) / 1_048_576
            size_str = f" ({size_mb:.1f} MB)" if size_mb > 0.1 else ""
            lines.append(f"📄 {item['name']}{size_str}")
    return "\n".join(lines)


def _get_project_choices(container: str, prefix: str = "") -> list[str]:
    """Return list of folder names at the current prefix."""
    items = _browse_storage(container, prefix)
    folders = [item["name"] for item in items if item["type"] == "folder"]
    return folders


# ---------------------------------------------------------------------------
# Seismic rendering — real API data (fail-loud); synthetic only in demo mode
# ---------------------------------------------------------------------------

_FAULT_CMAP = LinearSegmentedColormap.from_list(
    "fault", ["#ff000000", "#ff6600cc", "#ffdd00ee"], N=256
)


def _error_image(message: str, dpi: int = 120) -> bytes:
    """Render a dark placeholder image carrying a visible error message."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1520", dpi=dpi)
    ax.set_facecolor("#0d1520")
    ax.axis("off")
    ax.text(
        0.5, 0.5, f"⚠️ {message}",
        ha="center", va="center", color="#fca5a5", fontsize=11, wrap=True,
        transform=ax.transAxes,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_section_image(
    inline: int,
    survey_id: str,
    *,
    show_fault_overlay: bool = True,
    demo_mode: bool = False,
    run_id: str | None = None,
    geom: vapi.SurveyGeometry | None = None,
    dpi: int = 120,
) -> tuple[bytes, str]:
    """Render a seismic inline section as ``(png_bytes, status_markdown)``.

    Real-data path is the default and **fails loud**: on any API error the image
    shows the error and the status string explains it — it does NOT silently draw
    synthetic data.  Synthetic rendering happens only when ``demo_mode`` is set.
    """
    if demo_mode:
        amplitude, extent, title_suffix = _generate_synthetic_section(inline)
        disp = amplitude
        is_real = "synthetic demo"
        status = "🟡 **Demo (synthetic) mode** — not real survey data."
    else:
        try:
            payload = vapi.fetch_inline(survey_id, inline, API_BASE_URL)
        except ViewerAPIError as exc:
            msg = str(exc)
            return _error_image(msg, dpi), f"🔴 **Live data error:** {msg}"
        amp = np.array(payload["amplitude"], dtype=np.float32)  # (n_xl, n_s)
        disp = amp.T  # -> (n_s, n_xl): time on y, crossline on x
        crosslines = payload.get("crossline_coords") or list(range(amp.shape[0]))
        twtt = payload.get("twtt_ms") or list(range(amp.shape[1]))
        extent = [min(crosslines), max(crosslines), max(twtt), min(twtt)]
        title_suffix = f"{survey_id}"
        is_real = "real data"
        status = (
            f"🟢 **Live** — inline {inline} from `{survey_id}` "
            f"({disp.shape[1]}×{disp.shape[0]})."
        )

    # Robust amplitude scaling (real seismic has a wide dynamic range)
    vlim = float(np.percentile(np.abs(disp), 99)) or 1.0

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1520", dpi=dpi)
    ax.set_facecolor("#0d1520")
    ax.imshow(
        disp,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        interpolation="bilinear",
        extent=extent,
    )

    overlay_note = ""
    if show_fault_overlay and demo_mode:
        # Synthetic illustrative overlay — demo only.
        n_samples, n_crosslines = disp.shape
        mask = np.zeros_like(disp)
        xl_fault = int(n_crosslines * 0.37 + (inline - 1000) * 0.03)
        xl_fault = max(5, min(n_crosslines - 5, xl_fault))
        for xl in range(max(0, xl_fault - 8), min(n_crosslines, xl_fault + 8)):
            prob = 1.0 - abs(xl - xl_fault) / 9.0
            mask[int(n_samples * 0.4):, xl] = np.clip(prob * 0.85, 0, 1)
        ax.imshow(
            mask, aspect="auto", cmap=_FAULT_CMAP, vmin=0, vmax=1,
            alpha=0.55, interpolation="bilinear", extent=extent,
        )
    elif show_fault_overlay and run_id and geom is not None:
        # Real fault overlay from a completed UNet3D run.  Pass the absolute
        # inline; the API maps it to the local volume index via the manifest.
        try:
            ov = vapi.fetch_overlay(run_id, inline, API_BASE_URL)
            prob = np.array(ov["fault_probability"], dtype=np.float32).T  # (n_s, n_xl)
            ax.imshow(
                prob, aspect="auto", cmap=_FAULT_CMAP, vmin=0, vmax=1,
                alpha=0.55, interpolation="bilinear", extent=extent,
            )
            overlay_note = f"  ·  fault overlay from run `{run_id[:8]}`"
        except ViewerAPIError as exc:
            overlay_note = f"  ·  ⚠️ overlay unavailable: {exc}"

    ax.set_xlabel("Crossline", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Two-way time (ms)", color="#94a3b8", fontsize=9)
    ax.set_title(
        f"Inline {inline}  —  {title_suffix}  ({is_real})",
        color="#e2e8f0", fontsize=9.5, pad=7,
    )
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read(), status + overlay_note


def _generate_synthetic_section(inline: int) -> tuple[np.ndarray, list, str]:
    """Generate a synthetic seismic section for demo/fallback."""
    rng = np.random.default_rng(inline)
    n_samples, n_crosslines = 200, 150

    data = rng.standard_normal((n_samples, n_crosslines))
    t = np.linspace(-0.05, 0.05, 21)
    peak_freq = 30
    wavelet = (1 - 2 * (np.pi * peak_freq * t) ** 2) * np.exp(
        -((np.pi * peak_freq * t) ** 2)
    )
    from scipy.signal import convolve
    for xl in range(n_crosslines):
        data[:, xl] = convolve(data[:, xl], wavelet, mode="same")

    # Synthetic reflectors
    for layer, amp, width in [
        (60, 0.7, 5), (100, 1.4, 6), (155, 2.0, 8), (168, 3.2, 9), (180, 1.6, 7)
    ]:
        for xl in range(n_crosslines):
            offset = int(xl * 0.04 - n_crosslines * 0.02)
            s = max(0, min(n_samples - width, layer + offset))
            data[s : s + width, xl] += amp * rng.standard_normal()

    # Bright amplitude anomaly
    xl_start, xl_end = int(n_crosslines * 0.3), int(n_crosslines * 0.7)
    data[162:172, xl_start:xl_end] += 3.5

    extent = [950, 1100, 4000, 0]
    return data, extent, "Volve Survey A"


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def _chat(
    message: str,
    history: list[dict[str, str]],
    persona: str,
) -> tuple[list[dict[str, str]], str]:
    """Send a message to the agent and return updated history."""
    import time as _time

    if not message.strip():
        return history, ""

    try:
        agent = _get_agent()

        # Apply persona if changed
        persona_map = {
            "Auto": None,
            "Geophysics (Ash)": "geophysics",
            "Geology (Kane)": "geology",
            "Geoengineering (Brett)": "geoengineering",
        }
        requested = persona_map.get(persona)
        if requested and agent.persona != requested:
            try:
                agent.set_persona(requested)
            except (ValueError, AttributeError):
                pass

        chunks: list[str] = []
        start = _time.monotonic()
        for chunk in agent.chat(message):
            chunks.append(chunk)
            # Guard against exceeding AFD idle timeout (30s default)
            if _time.monotonic() - start > 25:
                chunks.append("\n\n⏱️ _Response truncated — processing took too long._")
                break
        response = "".join(chunks)

    except Exception as exc:
        response = f"⚠️ **Agent error**: {type(exc).__name__}: {exc}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return history, ""


def _quick_action(action: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    """Process a quick-action button and inject the canned question."""
    messages = {
        "Status": "What is the current status of the latest preprocessing and inference run?",
        "Wells": "Show me the well inventory linked to the current survey.",
        "Full Analysis": (
            "Analyze the latest Volve run end-to-end: check data, QC, results, "
            "and give me an analyst handoff note."
        ),
    }
    msg = messages.get(action, "")
    if not msg:
        return history, ""
    return _chat(msg, history, "Auto")


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

TITLE = "🌊 DeepSeismic Analyst — Volve Field PoC"
DESCRIPTION = (
    "**Azure AI Foundry agent** grounded by Azure AI Search over indexed seismic "
    "knowledge and FastAPI tool calls for live run and result data.\n\n"
    + (
        "⚠️ **MOCK MODE** — running offline; no Azure calls are made."
        if MOCK_MODE
        else "● **Live mode** — connected to Azure AI Foundry."
    )
)

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

with gr.Blocks(
    title="DeepSeismic Analyst",
    theme=_THEME,
) as demo:

    gr.Markdown(f"## {TITLE}")
    gr.Markdown(DESCRIPTION)

    # ── Project browser (collapsible) ──────────────────────────────────────
    with gr.Accordion("📂 Project Browser — ADLS Storage", open=True):
        with gr.Row():
            container_dd = gr.Dropdown(
                choices=["raw", "staged", "features", "results", "catalog"],
                value="raw",
                label="Container",
                interactive=True,
                scale=1,
            )
            filter_box = gr.Textbox(
                placeholder="Filter by name...",
                label="Filter",
                interactive=True,
                scale=2,
            )
        breadcrumb = gr.Markdown("📍 **raw:** /")
        browse_listing = gr.Dataframe(
            headers=["Type", "Name", "Size", "Path"],
            datatype=["str", "str", "str", "str"],
            col_count=(4, "fixed"),
            interactive=False,
            label="Select a row and click Open to navigate into folders",
        )
        with gr.Row():
            open_btn = gr.Button("📂 Open", size="sm", variant="primary", scale=2)
            up_btn = gr.Button("⬆️ Up", size="sm", scale=1)
            refresh_btn = gr.Button("🔄", size="sm", scale=1)

    with gr.Row():
        # ── Left column: chat ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=420):
            chatbot = gr.Chatbot(
                label="Agent Conversation",
                type="messages",
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about the Volve survey, runs, wells, or results...",
                    label="Your message",
                    lines=2,
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            with gr.Row():
                btn_status = gr.Button("Status", size="sm")
                btn_wells = gr.Button("Wells", size="sm")
                btn_analyze = gr.Button("Full Analysis", size="sm")

            persona_dd = gr.Dropdown(
                choices=["Auto", "Geophysics (Ash)", "Geology (Kane)", "Geoengineering (Brett)"],
                value="Auto",
                label="Domain Perspective",
                interactive=True,
            )

            clear_btn = gr.Button("Clear Chat", size="sm", variant="secondary")

        # ── Right column: seismic viewer ───────────────────────────────────
        with gr.Column(scale=1, min_width=420):
            with gr.Row():
                survey_dd = gr.Dropdown(
                    choices=[],
                    label="Survey",
                    interactive=True,
                    scale=3,
                )
                refresh_surveys_btn = gr.Button("🔄", size="sm", scale=1)

            seismic_image = gr.Image(
                label="Seismic Inline Viewer",
                type="numpy",
                height=400,
            )

            viewer_status = gr.Markdown("_Select a survey to load real amplitudes._")

            with gr.Row():
                inline_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Inline number",
                    interactive=True,
                )
                overlay_check = gr.Checkbox(
                    value=True,
                    label="Fault overlay",
                    interactive=True,
                )
                demo_check = gr.Checkbox(
                    value=False,
                    label="Demo (synthetic)",
                    interactive=True,
                )

            with gr.Accordion("⚡ Live fault detection (UNet3D)", open=False):
                with gr.Row():
                    run_infer_btn = gr.Button(
                        "Run fault detection", variant="primary", scale=2
                    )
                    check_infer_btn = gr.Button("Check status", scale=1)
                infer_status = gr.Markdown(
                    "_Runs the UNet3D model on the selected survey via the API "
                    f"(checkpoint `{CHECKPOINT_BLOB}`). Results overlay onto the section._"
                )

    # State holding the active survey geometry + last inference run id.
    _viewer_state = gr.State({"survey_id": None, "geom": None, "run_id": None})

    # ── Wire up events ─────────────────────────────────────────────────────

    # -- Project browser events --
    _browser_state = gr.State({"container": "raw", "prefix": ""})

    def _build_listing(state: dict, filter_text: str = "") -> tuple:
        """Return (dataframe_rows, breadcrumb_md, state)."""
        container = state["container"]
        prefix = state["prefix"]
        try:
            items = vapi.browse(container, prefix, API_BASE_URL)
        except ViewerAPIError as exc:
            crumb = f"🔴 **Browse error** for `{container}`: {exc}"
            return [["⚠️", "(error)", "—", ""]], crumb, state

        rows = []
        for item in items:
            name = item["name"]
            # Apply filter
            if filter_text and filter_text.lower() not in name.lower():
                continue
            if item["type"] == "folder":
                rows.append(["📁 Folder", name, "—", item.get("path", f"{prefix}{name}/")])
            else:
                size = item.get("size") or 0
                if size > 1_048_576:
                    size_str = f"{size / 1_048_576:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                rows.append(["📄 File", name, size_str, item.get("path", f"{prefix}{name}")])

        if not rows:
            rows = [["—", "(empty)" if not filter_text else "(no matches)", "—", ""]]

        # Breadcrumb with clickable segments
        parts = prefix.rstrip("/").split("/") if prefix else []
        if parts:
            segments = " › ".join(f"**{p}**" for p in parts)
            crumb = f"📍 `{container}` › {segments}"
        else:
            crumb = f"📍 `{container}` › /"
        return rows, crumb, state

    def _on_container_change(container: str, state: dict) -> tuple:
        state["container"] = container
        state["prefix"] = ""
        rows, crumb, state = _build_listing(state)
        return rows, crumb, state

    def _on_filter(filter_text: str, state: dict) -> tuple:
        rows, crumb, state = _build_listing(state, filter_text)
        return rows, crumb, state

    def _on_open(selected_data, state: dict) -> tuple:
        """Navigate into the selected folder."""
        if selected_data is not None and len(selected_data) > 0:
            try:
                if hasattr(selected_data, 'iloc'):
                    row = selected_data.iloc[0]
                    type_col = str(row.iloc[0]) if len(row) > 0 else ""
                    name = str(row.iloc[1]) if len(row) > 1 else ""
                else:
                    row = selected_data[0] if len(selected_data) > 0 else []
                    type_col = str(row[0]) if len(row) > 0 else ""
                    name = str(row[1]) if len(row) > 1 else ""
                if "Folder" in type_col and name and name != "(empty)":
                    state["prefix"] = state["prefix"] + name + "/"
            except (IndexError, KeyError):
                pass
        rows, crumb, state = _build_listing(state)
        return rows, crumb, state

    def _on_up(state: dict) -> tuple:
        prefix = state["prefix"]
        if prefix:
            parts = prefix.rstrip("/").split("/")
            state["prefix"] = "/".join(parts[:-1]) + "/" if len(parts) > 1 else ""
        rows, crumb, state = _build_listing(state)
        return rows, crumb, state

    container_dd.change(
        _on_container_change,
        inputs=[container_dd, _browser_state],
        outputs=[browse_listing, breadcrumb, _browser_state],
    )
    filter_box.change(
        _on_filter,
        inputs=[filter_box, _browser_state],
        outputs=[browse_listing, breadcrumb, _browser_state],
    )
    open_btn.click(
        _on_open,
        inputs=[browse_listing, _browser_state],
        outputs=[browse_listing, breadcrumb, _browser_state],
    )
    up_btn.click(
        _on_up,
        inputs=[_browser_state],
        outputs=[browse_listing, breadcrumb, _browser_state],
    )
    refresh_btn.click(
        lambda state: _build_listing(state),
        inputs=[_browser_state],
        outputs=[browse_listing, breadcrumb, _browser_state],
    )

    # -- Chat events --
    def _send(message: str, history: list, persona: str) -> tuple:
        new_history, cleared = _chat(message, history, persona)
        return new_history, cleared

    def _png_to_np(png_bytes: bytes) -> np.ndarray:
        import PIL.Image
        return np.array(PIL.Image.open(io.BytesIO(png_bytes)))

    def _clamp_inline(inline: int, state: dict) -> int:
        """Clamp an inline into the loaded survey's valid range.

        The slider keeps ``minimum=0`` so Gradio never server-side rejects a
        stale value (#20 — a raised minimum 422'd on the 2nd interaction with
        an HTML body the browser couldn't parse).  Values outside the survey
        range are clamped here instead.
        """
        geom = state.get("geom")
        if geom is None:
            return int(inline)
        lo, hi, _ = geom.inline_choices_bounds()
        return int(min(max(int(inline), lo), hi))

    def _render(inline: int, show_overlay: bool, demo_mode: bool, state: dict):
        """Render the section for the current survey/run. Returns (image, status)."""
        survey_id = state.get("survey_id")
        if not demo_mode and not survey_id:
            png = _error_image("No survey selected — pick one from the Survey dropdown.")
            return _png_to_np(png), "🔴 **No survey selected.**"
        inline = _clamp_inline(inline, state)
        png, status = _render_section_image(
            int(inline),
            survey_id or "demo",
            show_fault_overlay=show_overlay,
            demo_mode=demo_mode,
            run_id=state.get("run_id"),
            geom=state.get("geom"),
        )
        return _png_to_np(png), status

    def _refresh_surveys(state: dict):
        """Populate the survey dropdown from the API (fail-loud)."""
        try:
            surveys = vapi.list_surveys(API_BASE_URL)
        except ViewerAPIError as exc:
            return gr.update(choices=[], value=None), f"🔴 **Cannot list surveys:** {exc}", state
        if not surveys:
            return gr.update(choices=[], value=None), "🟡 **No surveys ingested yet.**", state
        return (
            gr.update(choices=surveys, value=surveys[0]),
            f"🟢 Found {len(surveys)} survey(s). Select one to load.",
            state,
        )

    def _on_survey_change(survey_id: str, show_overlay: bool, demo_mode: bool, state: dict):
        """Load geometry, reset the inline slider, and render the first inline."""
        state = dict(state)
        state["survey_id"] = survey_id
        state["run_id"] = None
        if not survey_id:
            return gr.update(), None, "🟡 No survey selected.", state, ""
        try:
            geom = vapi.get_survey_geometry(survey_id, API_BASE_URL)
        except ViewerAPIError as exc:
            state["geom"] = None
            png = _error_image(str(exc))
            return gr.update(), _png_to_np(png), f"🔴 {exc}", state, ""
        state["geom"] = geom
        lo, hi, step = geom.inline_choices_bounds()
        img, status = _render(lo, show_overlay, demo_mode, state)
        # Keep minimum=0 — raising it makes Gradio 422 on a stale client value
        # (#20). The handlers clamp the inline into [lo, hi] instead.
        slider = gr.update(minimum=0, maximum=hi, step=step, value=lo)
        return slider, img, status, state, ""

    def _start_inference(inline: int, state: dict):
        survey_id = state.get("survey_id")
        if not survey_id:
            return state, "🔴 Select a survey before running inference."
        inline = _clamp_inline(inline, state)
        try:
            run_id = vapi.start_fault_detection(
                survey_id,
                CHECKPOINT_BLOB,
                API_BASE_URL,
                inline_center=int(inline),
            )
        except ViewerAPIError as exc:
            return state, f"🔴 **Could not start inference:** {exc}"
        state = dict(state)
        state["run_id"] = run_id
        return state, (
            f"🟡 **Queued** run `{run_id[:8]}` on `{survey_id}` around inline "
            f"{int(inline)}. Click **Check status** to poll, then toggle the "
            "fault overlay."
        )

    def _check_inference(inline: int, show_overlay: bool, demo_mode: bool, state: dict):
        run_id = state.get("run_id")
        if not run_id:
            return None, "🟡 No run started yet.", state
        try:
            st = vapi.poll_status(run_id, API_BASE_URL)
        except ViewerAPIError as exc:
            return None, f"🔴 **Status error:** {exc}", state
        status_val = st.get("status", "unknown")
        if status_val == "complete":
            img, vstatus = _render(inline, show_overlay, demo_mode, state)
            return img, f"🟢 **Complete** — run `{run_id[:8]}`. {vstatus}", state
        if status_val in ("failed", "error"):
            return None, f"🔴 **Run failed:** {st.get('error') or 'unknown error'}", state
        return None, f"🟡 **{status_val}** — run `{run_id[:8]}` still processing…", state

    send_btn.click(
        _send,
        inputs=[msg_box, chatbot, persona_dd],
        outputs=[chatbot, msg_box],
    )
    msg_box.submit(
        _send,
        inputs=[msg_box, chatbot, persona_dd],
        outputs=[chatbot, msg_box],
    )

    # -- Seismic viewer events --
    refresh_surveys_btn.click(
        _refresh_surveys,
        inputs=[_viewer_state],
        outputs=[survey_dd, viewer_status, _viewer_state],
    )
    survey_dd.change(
        _on_survey_change,
        inputs=[survey_dd, overlay_check, demo_check, _viewer_state],
        outputs=[inline_slider, seismic_image, viewer_status, _viewer_state, infer_status],
    )
    inline_slider.release(
        _render,
        inputs=[inline_slider, overlay_check, demo_check, _viewer_state],
        outputs=[seismic_image, viewer_status],
    )
    overlay_check.change(
        _render,
        inputs=[inline_slider, overlay_check, demo_check, _viewer_state],
        outputs=[seismic_image, viewer_status],
    )
    demo_check.change(
        _render,
        inputs=[inline_slider, overlay_check, demo_check, _viewer_state],
        outputs=[seismic_image, viewer_status],
    )
    run_infer_btn.click(
        _start_inference,
        inputs=[inline_slider, _viewer_state],
        outputs=[_viewer_state, infer_status],
    )
    check_infer_btn.click(
        _check_inference,
        inputs=[inline_slider, overlay_check, demo_check, _viewer_state],
        outputs=[seismic_image, infer_status, _viewer_state],
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

    quick_btns = [
        (btn_status, "Status"),
        (btn_wells, "Wells"),
        (btn_analyze, "Full Analysis"),
    ]
    for btn, label in quick_btns:
        btn.click(
            lambda h, p, lbl=label: _quick_action(lbl, h),
            inputs=[chatbot, persona_dd],
            outputs=[chatbot, msg_box],
        )

    # Load the project browser and survey list on startup.
    def _initial_load(state: dict):
        bstate = {"container": "raw", "prefix": ""}
        rows, crumb, bstate = _build_listing(bstate)
        survey_update, vstatus, state = _refresh_surveys(state)
        return rows, crumb, bstate, survey_update, vstatus, state

    demo.load(
        _initial_load,
        inputs=[_viewer_state],
        outputs=[
            browse_listing, breadcrumb, _browser_state,
            survey_dd, viewer_status, _viewer_state,
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
