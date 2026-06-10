"""Gradio demo application for the DeepSeismic Analyst agent.

Provides a minimal but demo-ready interface with:
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
# Synthetic seismic rendering
# ---------------------------------------------------------------------------

def _render_section_image(
    inline: int,
    show_fault_overlay: bool = True,
    dpi: int = 120,
) -> bytes:
    """Render a seismic inline section as PNG bytes.

    Returns a synthetic placeholder section for demo purposes.
    Replace with a real Zarr read when live data is available.
    """
    rng = np.random.default_rng(inline)
    n_samples, n_crosslines = 200, 150

    # Bandlimited noise
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

    # Bright amplitude anomaly (Hugin Fm fluid indicator simulation)
    xl_start, xl_end = int(n_crosslines * 0.3), int(n_crosslines * 0.7)
    data[162:172, xl_start:xl_end] += 3.5

    data /= np.abs(data).max() + 1e-9

    # Fault probability mask
    mask = np.zeros((n_samples, n_crosslines), dtype=np.float32)
    if show_fault_overlay:
        xl_fault = int(n_crosslines * 0.37 + (inline - 1000) * 0.03)
        xl_fault = max(5, min(n_crosslines - 5, xl_fault))
        for xl in range(max(0, xl_fault - 8), min(n_crosslines, xl_fault + 8)):
            prob = 1.0 - abs(xl - xl_fault) / 9.0
            mask[80:, xl] = np.clip(prob * 0.85, 0, 1)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1520", dpi=dpi)
    ax.set_facecolor("#0d1520")

    ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.8,
        vmax=0.8,
        interpolation="bilinear",
        extent=[950, 1100, 4000, 0],
    )

    if show_fault_overlay:
        fault_cmap = LinearSegmentedColormap.from_list(
            "fault", ["#ff000000", "#ff6600cc", "#ffdd00ee"], N=256
        )
        ax.imshow(
            mask,
            aspect="auto",
            cmap=fault_cmap,
            vmin=0,
            vmax=1,
            alpha=0.55,
            interpolation="bilinear",
            extent=[950, 1100, 4000, 0],
        )

    ax.axhline(y=3500, color="#22c55e", linewidth=1.2, linestyle="--", alpha=0.85)
    ax.text(1096, 3440, "Hugin Fm top", color="#22c55e", fontsize=8, ha="right")
    ax.axhline(y=3380, color="#94a3b8", linewidth=0.8, linestyle=":", alpha=0.65)
    ax.text(1096, 3320, "Draupne Fm (seal)", color="#94a3b8", fontsize=7.5, ha="right")

    ax.set_xlabel("Crossline", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Two-way time (ms)", color="#94a3b8", fontsize=9)
    ax.set_title(
        f"Inline {inline}  —  Volve Survey A  (synthetic placeholder)",
        color="#e2e8f0",
        fontsize=9.5,
        pad=7,
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


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def _chat(
    message: str,
    history: list[list[str]],
    persona: str,
) -> tuple[list[list[str]], str]:
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

    history = history + [[message, response]]
    return history, ""


def _update_viewer(
    inline: int,
    show_overlay: bool,
) -> bytes:
    """Return PNG bytes for the seismic inline section."""
    return _render_section_image(inline, show_overlay)


def _quick_action(action: str, history: list[list[str]]) -> tuple[list[list[str]], str]:
    """Process a quick-action button and inject the canned question."""
    messages = {
        "📊 Status": "What is the current status of the latest preprocessing and inference run?",
        "🛢 Wells": "Show me the well inventory linked to the current survey.",
        "🔍 Full Analysis": (
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
        (btn_status, "📊 Status"),
        (btn_wells, "🛢 Wells"),
        (btn_analyze, "🔍 Full Analysis"),
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
