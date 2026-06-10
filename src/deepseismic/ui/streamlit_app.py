"""Streamlit demo application for the DeepSeismic Analyst agent.

Provides a two-panel interface:
- **Left panel** — conversation with the agent (chat history + input box)
- **Right panel** — seismic inline viewer (matplotlib) with optional fault overlay
- **Sidebar** — session state, dataset status, persona selector

Run with::

    streamlit run src/deepseismic/ui/streamlit_app.py

Or in mock mode::

    MOCK_LLM=true streamlit run src/deepseismic/ui/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DeepSeismic Analyst",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colour palette and CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #0f1923; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* Chat message bubbles */
    .user-bubble {
        background: #1e3a5f;
        border-radius: 10px 10px 2px 10px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #e8f4fd;
        font-size: 0.93rem;
    }
    .agent-bubble {
        background: #1a2535;
        border-left: 3px solid #3b82f6;
        border-radius: 2px 10px 10px 10px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #cbd5e1;
        font-size: 0.93rem;
    }
    .tool-call-marker {
        color: #64748b;
        font-size: 0.80rem;
        font-family: 'Courier New', monospace;
        background: #0d1520;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 4px 0;
        display: block;
    }
    .status-badge-ok   { color: #22c55e; font-weight: 600; }
    .status-badge-warn { color: #f59e0b; font-weight: 600; }
    .status-badge-mock { color: #a78bfa; font-weight: 600; }
    .section-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin: 12px 0 4px 0;
    }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

MOCK_MODE: bool = os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")


def _init_session() -> None:
    if "agent" not in st.session_state:
        from deepseismic.agent.agent import DeepSeismicAgent
        st.session_state.agent = DeepSeismicAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": str, "content": str}
    if "selected_inline" not in st.session_state:
        st.session_state.selected_inline = 1050
    if "show_fault_overlay" not in st.session_state:
        st.session_state.show_fault_overlay = True


_init_session()

# ---------------------------------------------------------------------------
# Synthetic seismic data (placeholder until live data path is wired up)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _generate_synthetic_section(
    inline: int,
    n_crosslines: int = 150,
    n_samples: int = 200,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a visually plausible synthetic seismic inline section."""
    rng = np.random.default_rng(seed if seed is not None else inline)

    # Base: bandlimited noise convolved with a Ricker wavelet
    data = rng.standard_normal((n_samples, n_crosslines))

    # Ricker wavelet (30 Hz equivalent)
    t = np.linspace(-0.05, 0.05, 21)
    peak_freq = 30
    wavelet = (1 - 2 * (np.pi * peak_freq * t) ** 2) * np.exp(
        -((np.pi * peak_freq * t) ** 2)
    )
    from scipy.signal import convolve
    for xl in range(n_crosslines):
        data[:, xl] = convolve(data[:, xl], wavelet, mode="same")

    # Add layered reflectors (stratigraphic events)
    for layer_sample, amplitude, width in [
        (60, 0.8, 5),    # shallow reflector (Shetland)
        (100, 1.5, 6),   # mid-section
        (155, 2.2, 8),   # Draupne seal
        (168, 3.0, 9),   # Hugin reservoir top
        (180, 1.8, 7),   # Hugin base
    ]:
        for xl in range(n_crosslines):
            # Slight structural dip
            sample_offset = int(xl * 0.04 - n_crosslines * 0.02)
            s = max(0, min(n_samples - width, layer_sample + sample_offset))
            data[s : s + width, xl] += amplitude * rng.standard_normal()

    # Bright spot anomaly near Hugin Fm top (fluid indicator simulation)
    xl_start, xl_end = int(n_crosslines * 0.3), int(n_crosslines * 0.7)
    for xl in range(xl_start, xl_end):
        data[162:172, xl] += 4.0  # elevated amplitude

    # Normalise
    data /= np.abs(data).max() + 1e-9
    return data


@st.cache_data(show_spinner=False)
def _generate_fault_mask(
    inline: int,
    n_crosslines: int = 150,
    n_samples: int = 200,
) -> np.ndarray:
    """Generate a simple probabilistic fault mask for overlay display."""
    mask = np.zeros((n_samples, n_crosslines), dtype=np.float32)
    # Simulated fault corridor around XL 50–70 range
    xl_fault = int(n_crosslines * 0.37 + (inline - 1000) * 0.03)
    xl_fault = max(5, min(n_crosslines - 5, xl_fault))
    for xl in range(max(0, xl_fault - 8), min(n_crosslines, xl_fault + 8)):
        prob = 1.0 - abs(xl - xl_fault) / 9.0
        mask[80:, xl] = np.clip(prob * 0.85, 0, 1)
    return mask


def _render_seismic_section(inline: int, show_overlay: bool) -> None:
    """Render a seismic inline section with optional fault probability overlay."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    data = _generate_synthetic_section(inline)
    n_samples, n_crosslines = data.shape

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0d1520")
    ax.set_facecolor("#0d1520")

    # Seismic wiggle/variable-density display
    ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.8,
        vmax=0.8,
        interpolation="bilinear",
        extent=[950, 1100, 4000, 0],  # XL range, TWT range
    )

    if show_overlay:
        mask = _generate_fault_mask(inline)
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

    # Annotation: Hugin Fm approximate level
    hugin_twt = 3500
    ax.axhline(
        y=hugin_twt,
        color="#22c55e",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
    )
    ax.text(
        1096, hugin_twt - 60, "Hugin Fm top (~3 500 ms)",
        color="#22c55e", fontsize=8, ha="right", va="bottom", alpha=0.9,
    )

    # Draupne seal level
    draupne_twt = 3380
    ax.axhline(y=draupne_twt, color="#94a3b8", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(
        1096, draupne_twt - 50, "Draupne Fm (seal)",
        color="#94a3b8", fontsize=7.5, ha="right", va="bottom", alpha=0.7,
    )

    # Axes
    ax.set_xlabel("Crossline", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Two-way time (ms)", color="#94a3b8", fontsize=9)
    ax.set_title(
        f"Inline {inline}  —  Volve Survey A (synthetic placeholder)",
        color="#e2e8f0",
        fontsize=10,
        pad=8,
    )
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🌊 DeepSeismic Analyst")
    st.caption("Volve Field · Azure AI Foundry Agent")

    # Mock mode badge
    if MOCK_MODE:
        st.markdown(
            '<span class="status-badge-mock">⚠ MOCK MODE</span> — offline, no Azure calls',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge-ok">● Live</span> — Azure AI Foundry',
            unsafe_allow_html=True,
        )

    st.divider()

    # Persona selector
    st.markdown('<p class="section-header">Domain Perspective</p>', unsafe_allow_html=True)
    persona_options = {
        "Auto (context-driven)": None,
        "Ash / Geophysics": "geophysics",
        "Kane / Geology": "geology",
        "Brett / Geoengineering": "geoengineering",
    }
    selected_label = st.selectbox(
        "Perspective",
        list(persona_options.keys()),
        label_visibility="collapsed",
    )
    selected_persona = persona_options[selected_label]
    if selected_persona != st.session_state.agent.persona:
        if selected_persona:
            st.session_state.agent.set_persona(selected_persona)
        else:
            st.session_state.agent.persona = None

    st.divider()

    # Inline selector for the viewer
    st.markdown('<p class="section-header">Seismic Viewer</p>', unsafe_allow_html=True)
    st.session_state.selected_inline = st.slider(
        "Inline number",
        min_value=1000,
        max_value=1200,
        value=st.session_state.selected_inline,
        step=5,
        label_visibility="visible",
    )
    st.session_state.show_fault_overlay = st.checkbox(
        "Show fault probability overlay",
        value=st.session_state.show_fault_overlay,
    )

    st.divider()

    # Session state summary
    st.markdown('<p class="section-header">Session State</p>', unsafe_allow_html=True)
    state = st.session_state.agent.get_state_summary()
    st.caption(f"Thread: `{state['thread_id'] or 'n/a'}`")
    st.caption(f"Dataset: `{state['dataset_id'] or 'not set'}`")
    st.caption(f"Run: `{state['run_id'] or 'not set'}`")
    st.caption(f"Tool calls: {state['tool_calls']}")

    st.divider()

    # Quick action buttons
    st.markdown('<p class="section-header">Quick Actions</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    _btn_status = col_a.button("📊 Status", use_container_width=True)
    _btn_wells = col_b.button("🛢 Wells", use_container_width=True)
    _btn_analyze = st.button("🔍 Full Analysis", use_container_width=True)

    if _btn_status:
        st.session_state._queued_message = (
            "What is the current status of the latest preprocessing and inference run?"
        )
    if _btn_wells:
        st.session_state._queued_message = (
            "Show me the well inventory for the current survey."
        )
    if _btn_analyze:
        st.session_state._queued_message = (
            "Analyze the latest Volve run end-to-end: check data, QC, results, "
            "and give me an analyst handoff note."
        )

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main layout: chat (left) | viewer (right)
# ---------------------------------------------------------------------------

col_chat, col_viewer = st.columns([1, 1], gap="medium")

# ── Seismic viewer (right) ──────────────────────────────────────────────────
with col_viewer:
    st.markdown("#### 📡 Seismic Section")
    _render_seismic_section(
        st.session_state.selected_inline,
        st.session_state.show_fault_overlay,
    )
    if st.session_state.show_fault_overlay:
        st.caption(
            "🟠 Fault probability overlay (UNet candidate — synthetic placeholder). "
            "Requires analyst review."
        )
    st.caption(
        "Seismic display is a synthetic placeholder for demo purposes. "
        "Wire `get_inline_section` to real Zarr data for live rendering."
    )

# ── Chat panel (left) ───────────────────────────────────────────────────────
with col_chat:
    st.markdown("#### 💬 Analyst Chat")

    # Render existing messages
    chat_container = st.container(height=460)
    with chat_container:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(
                    f'<div class="user-bubble">🧑‍💼 {content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Separate tool call markers from main content
                lines = content.split("\n")
                rendered = []
                for line in lines:
                    if line.strip().startswith("> 🔧"):
                        rendered.append(
                            f'<span class="tool-call-marker">{line.strip()}</span>'
                        )
                    else:
                        rendered.append(line)
                body = "\n".join(rendered)
                st.markdown(
                    f'<div class="agent-bubble">{body}</div>',
                    unsafe_allow_html=True,
                )

    # Chat input
    user_input = st.chat_input("Ask about the Volve survey, runs, wells, or results…")

    # Process queued messages from sidebar buttons
    if hasattr(st.session_state, "_queued_message"):
        queued = st.session_state._queued_message
        del st.session_state._queued_message
        user_input = queued

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Agent is thinking…"):
            chunks: list[str] = []
            for chunk in st.session_state.agent.chat(user_input):
                chunks.append(chunk)
            response = "".join(chunks)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
