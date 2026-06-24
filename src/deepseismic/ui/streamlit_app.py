"""Streamlit demo application for the DeepSeismic Analyst agent.

Provides a two-panel interface:
- **Left panel** — conversation with the agent (chat history + input box)
- **Right panel** — seismic inline viewer (matplotlib) with optional fault overlay
- **Sidebar** — session state, dataset status, persona selector

Run with::

    streamlit run src/deepseismic/ui/streamlit_app.py

Or in mock mode::

    MOCK_LLM=true streamlit run src/deepseismic/ui/streamlit_app.py

Data prerequisites
------------------
- Amplitude Zarr:     data/volve/staged/synthetic.zarr
- Fault prob Zarr:    data/volve/staged/fault_prob.zarr  (run scripts/bake_demo_faults.py first)
- Fault mask Zarr:    data/volve/staged/fault_mask.zarr
- Fault sticks:       data/volve/interpretations/fault_sticks/*.dat
"""

from __future__ import annotations

import os
from pathlib import Path

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
# Data paths (relative to repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT  = Path(__file__).resolve().parents[3]
_ZARR_AMP   = _REPO_ROOT / "data/volve/staged/synthetic.zarr"
_ZARR_PROB  = _REPO_ROOT / "data/volve/staged/fault_prob.zarr"
_ZARR_MASK  = _REPO_ROOT / "data/volve/staged/fault_mask.zarr"
_STICKS_DIR = _REPO_ROOT / "data/volve/interpretations/fault_sticks"

# Known amplitude clip values from synthetic.json sidecar (p01/p99)
_AMP_VMIN: float = -0.121
_AMP_VMAX: float = 0.104

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
    if "fault_threshold" not in st.session_state:
        st.session_state.fault_threshold = 0.5


_init_session()

# ---------------------------------------------------------------------------
# Real seismic data readers (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _get_volume_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (inline_arr, crossline_arr, twtt_ms_arr) from the amplitude Zarr."""
    import zarr
    root = zarr.open_group(str(_ZARR_AMP), mode="r")
    return (
        np.asarray(root["inline"][:]),
        np.asarray(root["crossline"][:]),
        np.asarray(root["twtt_ms"][:]),
    )


@st.cache_data(show_spinner=False)
def _get_amplitude_slice(inline_abs: int) -> np.ndarray:
    """Return (n_xl, n_s) float32 amplitude slice for the given absolute inline."""
    import zarr
    root = zarr.open_group(str(_ZARR_AMP), mode="r")
    il_arr, _, _ = _get_volume_coords()
    idx = int(np.clip(np.searchsorted(il_arr, inline_abs), 0, len(il_arr) - 1))
    return np.asarray(root["amplitude"][idx, :, :], dtype=np.float32)


@st.cache_data(show_spinner=False)
def _get_fault_prob_slice(inline_abs: int) -> np.ndarray | None:
    """Return (n_xl, n_s) fault probability slice, or None if bake not available."""
    if not _ZARR_PROB.exists():
        return None
    import zarr
    root = zarr.open_group(str(_ZARR_PROB), mode="r")
    il_arr, _, _ = _get_volume_coords()
    idx = int(np.clip(np.searchsorted(il_arr, inline_abs), 0, len(il_arr) - 1))
    return np.asarray(root["fault_probability"][idx, :, :], dtype=np.float32)


@st.cache_data(show_spinner=False)
def _load_fault_sticks() -> dict[str, np.ndarray]:
    """Parse .dat fault sticks.

    Returns dict of fault_name -> (N, 3) float32 array where columns are:
        [abs_inline, abs_crossline, twt_ms]

    Coordinate mapping for .dat files:
        inline col     = 0-based volume index  →  abs_inline  = 1001 + idx
        crossline col  = 0-based volume index  →  abs_crossline = 1900 + idx
        z_ms col       = SAMPLE INDEX (not true ms)  →  twt_ms = z_sample * 4.0
    Evidence: z_sample values 202-307 -> TWT 808-1228 ms, consistent with the
    UTM-format file (Volve_Fault_Sticks_synthetic.txt) showing Z_ms 700-852 ms.
    If interpreted as true ms (50-77 ms, <7% depth), faults would be unrealistically shallow.
    """
    sticks: dict[str, np.ndarray] = {}
    if not _STICKS_DIR.exists():
        return sticks
    for dat_file in sorted(_STICKS_DIR.glob("*.dat")):
        rows: list[tuple[float, float, float]] = []
        with open(dat_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    il_idx, xl_idx, z_samp = int(parts[0]), int(parts[1]), int(parts[2])
                    # 0-based volume index → absolute coordinate
                    abs_il = 1001 + il_idx
                    abs_xl = 1900 + xl_idx
                    # z column is sample index, not milliseconds
                    twt_ms = float(z_samp) * 4.0
                    rows.append((float(abs_il), float(abs_xl), twt_ms))
        if rows:
            sticks[dat_file.stem] = np.array(rows, dtype=np.float32)
    return sticks


# ---------------------------------------------------------------------------
# Seismic viewer rendering
# ---------------------------------------------------------------------------


def _render_seismic_section(
    inline: int,
    show_overlay: bool,
    fault_threshold: float,
) -> None:
    """Render a real seismic inline section with optional fault probability overlay."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    il_arr, xl_arr, twtt_arr = _get_volume_coords()
    xl_min   = int(xl_arr[0])
    xl_max   = int(xl_arr[-1])
    twt_max  = float(twtt_arr[-1])   # 1996.0 ms

    # Load amplitude: (n_xl, n_s) -> transpose to (n_s, n_xl) for imshow
    amp_slice = _get_amplitude_slice(inline)
    section   = amp_slice.T  # (n_s, n_xl)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0d1520")
    ax.set_facecolor("#0d1520")

    im_amp = ax.imshow(
        section,
        aspect="auto",
        cmap="RdBu_r",
        vmin=_AMP_VMIN,
        vmax=_AMP_VMAX,
        interpolation="bilinear",
        extent=[xl_min, xl_max, twt_max, 0.0],
    )
    fig.colorbar(
        im_amp, ax=ax,
        label="Amplitude (normalised)",
        fraction=0.03, pad=0.02,
    )

    bake_missing = False
    if show_overlay:
        prob_slice = _get_fault_prob_slice(inline)
        if prob_slice is not None:
            prob_disp = prob_slice.T  # (n_s, n_xl)
            fault_cmap = LinearSegmentedColormap.from_list(
                "fault", ["#ff000000", "#ff6600cc", "#ffdd00ee"], N=256
            )
            im_prob = ax.imshow(
                prob_disp,
                aspect="auto",
                cmap=fault_cmap,
                vmin=0,
                vmax=1,
                alpha=0.5,
                interpolation="bilinear",
                extent=[xl_min, xl_max, twt_max, 0.0],
            )
            fig.colorbar(
                im_prob, ax=ax,
                label="Fault probability (UNet3D)",
                fraction=0.03, pad=0.08,
            )

            # Fault stick overlay (interpretation vs ML — the killer feature)
            sticks = _load_fault_sticks()
            legend_added = False
            for _name, stick_arr in sticks.items():
                sel = stick_arr[:, 0] == float(inline)
                if sel.any():
                    ax.scatter(
                        stick_arr[sel, 1],   # abs crossline
                        stick_arr[sel, 2],   # twt_ms
                        c="red", s=30, marker="o", zorder=6, alpha=0.9,
                        label="Interpreted sticks (synthetic GT)" if not legend_added else None,
                    )
                    legend_added = True
            if legend_added:
                ax.legend(
                    loc="lower right", fontsize=7,
                    facecolor="#1a2535", labelcolor="#e2e8f0", edgecolor="#334155",
                )
        else:
            bake_missing = True

    ax.set_xlabel("Crossline", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Two-way time (ms)", color="#94a3b8", fontsize=9)
    ax.set_title(
        f"Inline {inline}  —  Volve Demo Volume (synthetic)",
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

    if bake_missing and show_overlay:
        st.warning(
            "Fault probability Zarr not found.  "
            "Run `python scripts/bake_demo_faults.py` to pre-compute fault detections."
        )

    # Fault fraction readout
    if show_overlay and not bake_missing:
        prob_slice = _get_fault_prob_slice(inline)
        if prob_slice is not None:
            frac = float(np.mean(prob_slice >= fault_threshold))
            st.caption(
                f"Fault voxels in slice at threshold {fault_threshold:.2f}: "
                f"{frac:.1%} of inline"
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Load inline bounds from real Zarr before rendering the slider
_il_arr, _, _ = _get_volume_coords()
_IL_MIN = int(_il_arr[0])    # 1001
_IL_MAX = int(_il_arr[-1])   # 1100

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

    # Inline selector — bounds from real Zarr coordinate array
    st.markdown('<p class="section-header">Seismic Viewer</p>', unsafe_allow_html=True)
    st.session_state.selected_inline = st.slider(
        "Inline number",
        min_value=_IL_MIN,
        max_value=_IL_MAX,
        value=max(_IL_MIN, min(_IL_MAX, st.session_state.selected_inline)),
        step=1,
        label_visibility="visible",
    )
    st.session_state.show_fault_overlay = st.checkbox(
        "Show fault probability overlay",
        value=st.session_state.show_fault_overlay,
    )
    if st.session_state.show_fault_overlay:
        st.session_state.fault_threshold = st.slider(
            "Fault threshold",
            min_value=0.3,
            max_value=0.7,
            value=st.session_state.fault_threshold,
            step=0.05,
            help="Probability cutoff for binary fault fraction readout.",
        )

    st.caption(f"Volume: 100 IL x 200 XL x 500 samp @ 4 ms  |  IL {_IL_MIN}–{_IL_MAX}")

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
        st.session_state.fault_threshold,
    )
    if st.session_state.show_fault_overlay and _ZARR_PROB.exists():
        st.caption(
            "🟠 Fault probability overlay — UNet3D candidate detection. "
            "Requires analyst review. Not a final interpretation."
        )
    st.caption(
        "Synthetic dataset approximating Volve ST10010 geometry — not licensed field data. "
        "UNet3D trained on synthetic fault-stick-derived labels. "
        "Metrics vs training labels only (circular validation — treat as training diagnostics)."
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
