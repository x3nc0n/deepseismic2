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

matplotlib.use("Agg")  # Non-interactive backend for server rendering

MOCK_MODE: bool = os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")

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
    """Fetch folder/file listing from the API browse endpoint."""
    import requests
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/browse/{container}",
            params={"prefix": prefix},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("items", [])
    except Exception:
        pass
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
# Seismic rendering — reads from API or falls back to synthetic
# ---------------------------------------------------------------------------

def _fetch_inline_from_api(survey_id: str, inline: int) -> dict | None:
    """Fetch inline slice data from the API. Returns None on failure."""
    import requests
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/surveys/{survey_id}/inline/{inline}",
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _render_section_image(
    inline: int,
    show_fault_overlay: bool = True,
    survey_id: str = "volve",
    dpi: int = 120,
) -> bytes:
    """Render a seismic inline section as PNG bytes.

    Tries to load real data from the API first; falls back to synthetic
    placeholder if no ingested data is available.
    """
    api_data = _fetch_inline_from_api(survey_id, inline)

    if api_data and api_data.get("amplitude"):
        # Real data path
        amplitude = np.array(api_data["amplitude"], dtype=np.float32)
        crosslines = api_data.get("crossline_coords", list(range(amplitude.shape[1])))
        twtt = api_data.get("twtt_ms", list(range(amplitude.shape[0])))
        title_suffix = f"{survey_id} survey"
        extent = [min(crosslines), max(crosslines), max(twtt), min(twtt)]
    else:
        # Synthetic fallback
        amplitude, extent, title_suffix = _generate_synthetic_section(inline)

    # Normalize for display
    amax = np.abs(amplitude).max()
    if amax > 1e-9:
        amplitude = amplitude / amax

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1520", dpi=dpi)
    ax.set_facecolor("#0d1520")

    ax.imshow(
        amplitude,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.8,
        vmax=0.8,
        interpolation="bilinear",
        extent=extent,
    )

    if show_fault_overlay and not api_data:
        # Only show synthetic fault overlay when using fake data
        n_samples, n_crosslines = amplitude.shape
        mask = np.zeros_like(amplitude)
        xl_fault = int(n_crosslines * 0.37 + (inline - 1000) * 0.03)
        xl_fault = max(5, min(n_crosslines - 5, xl_fault))
        for xl in range(max(0, xl_fault - 8), min(n_crosslines, xl_fault + 8)):
            prob = 1.0 - abs(xl - xl_fault) / 9.0
            mask[int(n_samples * 0.4):, xl] = np.clip(prob * 0.85, 0, 1)
        fault_cmap = LinearSegmentedColormap.from_list(
            "fault", ["#ff000000", "#ff6600cc", "#ffdd00ee"], N=256
        )
        ax.imshow(
            mask,
            aspect="auto",
            cmap=fault_cmap,
            vmin=0, vmax=1,
            alpha=0.55,
            interpolation="bilinear",
            extent=extent,
        )

    ax.set_xlabel("Crossline", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Two-way time (ms)", color="#94a3b8", fontsize=9)
    is_real = "— real data" if api_data else "— synthetic placeholder"
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
    return buf.read()


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
    if not message.strip():
        return history, ""

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
        except ValueError:
            pass

    chunks: list[str] = []
    for chunk in agent.chat(message):
        chunks.append(chunk)
    response = "".join(chunks)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return history, ""


def _update_viewer(
    inline: int,
    show_overlay: bool,
) -> bytes:
    """Return PNG bytes for the seismic inline section."""
    return _render_section_image(inline, show_overlay)


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
    with gr.Accordion("📂 Project Browser — ADLS Storage", open=False):
        with gr.Row():
            container_dd = gr.Dropdown(
                choices=["raw", "staged", "features", "results", "catalog"],
                value="raw",
                label="Container",
                interactive=True,
                scale=1,
            )
            path_display = gr.Textbox(
                value="/",
                label="Current Path",
                interactive=False,
                scale=3,
            )
        with gr.Row():
            folder_dd = gr.Dropdown(
                choices=[],
                label="Navigate to folder",
                interactive=True,
                scale=2,
            )
            nav_btn = gr.Button("Open →", size="sm", scale=1)
            up_btn = gr.Button("↑ Up", size="sm", scale=1)
        browse_output = gr.Markdown("_Select a container to browse._")

    with gr.Row():
        # ── Left column: chat ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=420):
            chatbot = gr.Chatbot(
                label="Agent Conversation",
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
            seismic_image = gr.Image(
                label="Seismic Inline Viewer",
                type="numpy",
                height=400,
            )

            with gr.Row():
                inline_slider = gr.Slider(
                    minimum=1000,
                    maximum=1200,
                    value=1050,
                    step=5,
                    label="Inline number",
                    interactive=True,
                )
                overlay_check = gr.Checkbox(
                    value=True,
                    label="Fault overlay",
                    interactive=True,
                )

            gr.Markdown(
                "_Seismic display is a synthetic placeholder for demo purposes. "
                "Connect `get_inline_section` to real Zarr data for live rendering._",
            )

    # ── Wire up events ─────────────────────────────────────────────────────

    # -- Project browser events --
    _browser_state = gr.State({"container": "raw", "prefix": ""})

    def _refresh_browser(state: dict) -> tuple:
        container = state["container"]
        prefix = state["prefix"]
        tree = _format_browse_tree(container, prefix)
        folders = _get_project_choices(container, prefix)
        path_str = f"/{prefix}" if prefix else "/"
        return tree, gr.update(choices=folders, value=None), path_str

    def _on_container_change(container: str, state: dict) -> tuple:
        state["container"] = container
        state["prefix"] = ""
        tree, folders, path_str = _refresh_browser(state)
        return tree, folders, path_str, state

    def _on_navigate(folder: str | None, state: dict) -> tuple:
        if folder:
            state["prefix"] = state["prefix"] + folder + "/"
        tree, folders, path_str = _refresh_browser(state)
        return tree, folders, path_str, state

    def _on_up(state: dict) -> tuple:
        prefix = state["prefix"]
        if prefix:
            parts = prefix.rstrip("/").split("/")
            state["prefix"] = "/".join(parts[:-1]) + "/" if len(parts) > 1 else ""
        tree, folders, path_str = _refresh_browser(state)
        return tree, folders, path_str, state

    container_dd.change(
        _on_container_change,
        inputs=[container_dd, _browser_state],
        outputs=[browse_output, folder_dd, path_display, _browser_state],
    )
    nav_btn.click(
        _on_navigate,
        inputs=[folder_dd, _browser_state],
        outputs=[browse_output, folder_dd, path_display, _browser_state],
    )
    up_btn.click(
        _on_up,
        inputs=[_browser_state],
        outputs=[browse_output, folder_dd, path_display, _browser_state],
    )

    # -- Chat events --
    def _send(message: str, history: list, persona: str) -> tuple:
        new_history, cleared = _chat(message, history, persona)
        return new_history, cleared

    def _render_pil(inline: int, show_overlay: bool) -> np.ndarray:
        import PIL.Image
        png_bytes = _update_viewer(inline, show_overlay)
        img = PIL.Image.open(io.BytesIO(png_bytes))
        return np.array(img)

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
    inline_slider.release(
        _render_pil,
        inputs=[inline_slider, overlay_check],
        outputs=[seismic_image],
    )
    overlay_check.change(
        _render_pil,
        inputs=[inline_slider, overlay_check],
        outputs=[seismic_image],
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

    # Render default inline on load
    demo.load(
        lambda: _render_pil(1050, True),
        outputs=[seismic_image],
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
