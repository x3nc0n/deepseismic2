"""Pure data-access functions for the DeepSeismic Streamlit viewer.

No Streamlit imports.  No ``@st.cache_data`` decorators.  No sidebar side-effects.
These functions are safe to import in pytest without mocking Streamlit.

Backend resolver
----------------
Controlled by ``DEEPSEISMIC_DATA_BACKEND`` (default ``local``):

``local``
    Reads from repo-relative paths (``data/volve/staged/…``).
    Base directory can be overridden via ``DEEPSEISMIC_DATA_DIR``
    (must contain the same sub-tree: ``staged/``, ``interpretations/``).

``azure``
    Reads Zarr groups via :class:`~deepseismic.storage.blob_client.StorageClient`
    (auto-detects ``STORAGE_CONNECTION_STRING`` or ``DefaultAzureCredential``).
    Env vars and defaults per artifact:

    - ``DEEPSEISMIC_AMP_CONTAINER`` (default ``staged``) +
      ``DEEPSEISMIC_AMP_PREFIX`` (default ``volve/synthetic.zarr``)
    - ``DEEPSEISMIC_FAULT_PROB_CONTAINER`` (default ``results``) +
      ``DEEPSEISMIC_FAULT_PROB_PREFIX`` (default ``volve/fault_prob.zarr``)
    - ``DEEPSEISMIC_FAULT_MASK_CONTAINER`` (default ``results``) +
      ``DEEPSEISMIC_FAULT_MASK_PREFIX`` (default ``volve/fault_mask.zarr``)
    - ``DEEPSEISMIC_STICKS_CONTAINER`` (default ``raw``) +
      ``DEEPSEISMIC_STICKS_PREFIX`` (default
      ``volve/interpretations/fault_sticks``).
      Blobs under this prefix with ``.dat`` suffix are listed and downloaded.

    The amplitude Zarr must contain arrays ``amplitude``, ``inline``, ``crossline``,
    ``twtt_ms`` (see history.md for shape/dtype contract).

Fault-stick coordinate mapping (DO NOT CHANGE — credibility-critical)
----------------------------------------------------------------------
``.dat`` files use 0-based volume indices, not absolute survey coordinates::

    abs_inline    = 1001 + il_idx
    abs_crossline = 1900 + xl_idx
    twt_ms        = z_sample * 4.0    # z column is sample index, NOT true ms

Evidence: z-values 202–307 → TWT 808–1228 ms, consistent with UTM-format file
(Volve_Fault_Sticks_synthetic.txt, Z_ms 700–852 ms).  Treating z as true ms
places faults at 50–77 ms, which is unrealistically shallow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Known amplitude clip values from synthetic.json sidecar (p01/p99)
AMP_VMIN: float = -0.121
AMP_VMAX: float = 0.104

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

_BACKEND_ENV = "DEEPSEISMIC_DATA_BACKEND"


def _backend() -> str:
    return os.environ.get(_BACKEND_ENV, "local").lower()


@dataclass(frozen=True)
class _LocalSources:
    zarr_amp: Path
    zarr_prob: Path
    zarr_mask: Path
    sticks_dir: Path


@dataclass(frozen=True)
class _AzureSources:
    amp_container: str
    amp_prefix: str
    prob_container: str
    prob_prefix: str
    mask_container: str
    mask_prefix: str
    sticks_container: str
    sticks_prefix: str


def _local_sources() -> _LocalSources:
    base = Path(os.environ.get("DEEPSEISMIC_DATA_DIR", str(_REPO_ROOT / "data/volve")))
    return _LocalSources(
        zarr_amp=base / "staged/synthetic.zarr",
        zarr_prob=base / "staged/fault_prob.zarr",
        zarr_mask=base / "staged/fault_mask.zarr",
        sticks_dir=base / "interpretations/fault_sticks",
    )


def _azure_sources() -> _AzureSources:
    return _AzureSources(
        amp_container=os.environ.get("DEEPSEISMIC_AMP_CONTAINER", "staged"),
        amp_prefix=os.environ.get("DEEPSEISMIC_AMP_PREFIX", "volve/synthetic.zarr"),
        prob_container=os.environ.get("DEEPSEISMIC_FAULT_PROB_CONTAINER", "results"),
        prob_prefix=os.environ.get("DEEPSEISMIC_FAULT_PROB_PREFIX", "volve/fault_prob.zarr"),
        mask_container=os.environ.get("DEEPSEISMIC_FAULT_MASK_CONTAINER", "results"),
        mask_prefix=os.environ.get("DEEPSEISMIC_FAULT_MASK_PREFIX", "volve/fault_mask.zarr"),
        sticks_container=os.environ.get("DEEPSEISMIC_STICKS_CONTAINER", "raw"),
        sticks_prefix=os.environ.get(
            "DEEPSEISMIC_STICKS_PREFIX",
            "volve/interpretations/fault_sticks",
        ),
    )


def _storage_client():  # type: ignore[return]
    """Return a :class:`~deepseismic.storage.blob_client.StorageClient` instance."""
    from deepseismic.storage.blob_client import StorageClient

    return StorageClient()


# ---------------------------------------------------------------------------
# Reader: volume coordinates
# ---------------------------------------------------------------------------


def get_volume_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(inline_arr, crossline_arr, twtt_ms_arr)`` from the amplitude Zarr.

    Works with both ``local`` and ``azure`` backends.
    """
    import zarr

    if _backend() == "azure":
        src = _azure_sources()
        client = _storage_client()
        store = client.open_zarr_store(src.amp_container, src.amp_prefix)
        root = zarr.open_group(store=store, mode="r")
    else:
        local = _local_sources()
        root = zarr.open_group(str(local.zarr_amp), mode="r")

    return (
        np.asarray(root["inline"][:]),
        np.asarray(root["crossline"][:]),
        np.asarray(root["twtt_ms"][:]),
    )


# ---------------------------------------------------------------------------
# Reader: amplitude slice
# ---------------------------------------------------------------------------


def get_amplitude_slice(inline_abs: int) -> np.ndarray:
    """Return a ``(n_xl, n_s)`` float32 amplitude slice for *inline_abs*.

    Out-of-range inline values are clamped to the volume extent.
    """
    import zarr

    if _backend() == "azure":
        src = _azure_sources()
        client = _storage_client()
        store = client.open_zarr_store(src.amp_container, src.amp_prefix)
        root = zarr.open_group(store=store, mode="r")
    else:
        local = _local_sources()
        root = zarr.open_group(str(local.zarr_amp), mode="r")

    il_arr, _, _ = get_volume_coords()
    idx = int(np.clip(np.searchsorted(il_arr, inline_abs), 0, len(il_arr) - 1))
    return np.asarray(root["amplitude"][idx, :, :], dtype=np.float32)


# ---------------------------------------------------------------------------
# Reader: fault probability slice
# ---------------------------------------------------------------------------


def get_fault_prob_slice(inline_abs: int) -> np.ndarray | None:
    """Return ``(n_xl, n_s)`` fault probability slice, or ``None`` if absent.

    Returns ``None`` for either backend when the artifact is not available,
    so the viewer renders amplitude-only without raising.
    """
    import zarr

    if _backend() == "azure":
        src = _azure_sources()
        client = _storage_client()
        store = client.open_zarr_store(src.prob_container, src.prob_prefix)
        # Probe for the zarr.json key to detect a missing bake
        try:
            root = zarr.open_group(store=store, mode="r")
            # Force metadata read — raises if store is empty
            _ = root.info
            il_arr, _, _ = get_volume_coords()
            idx = int(np.clip(np.searchsorted(il_arr, inline_abs), 0, len(il_arr) - 1))
            return np.asarray(root["fault_probability"][idx, :, :], dtype=np.float32)
        except Exception:
            return None
    else:
        local = _local_sources()
        if not local.zarr_prob.exists():
            return None
        root = zarr.open_group(str(local.zarr_prob), mode="r")
        il_arr, _, _ = get_volume_coords()
        idx = int(np.clip(np.searchsorted(il_arr, inline_abs), 0, len(il_arr) - 1))
        return np.asarray(root["fault_probability"][idx, :, :], dtype=np.float32)


# ---------------------------------------------------------------------------
# Reader: fault sticks
# ---------------------------------------------------------------------------


def load_fault_sticks() -> dict[str, np.ndarray]:
    """Parse ``.dat`` fault-stick files and return per-fault coordinate arrays.

    Returns a ``dict`` of ``fault_name → (N, 3) float32`` where columns are
    ``[abs_inline, abs_crossline, twt_ms]``.

    Coordinate mapping (canonical — DO NOT change)::

        inline col    (0-based)  →  abs_inline    = 1001 + il_idx
        crossline col (0-based)  →  abs_crossline = 1900 + xl_idx
        z col         (sample#)  →  twt_ms        = z_sample × 4.0

    The azure backend downloads ``.dat`` blobs from
    ``DEEPSEISMIC_STICKS_CONTAINER / DEEPSEISMIC_STICKS_PREFIX``.
    If the sticks container/prefix is unreachable or empty, returns ``{}``.
    """
    if _backend() == "azure":
        return _load_fault_sticks_azure()
    return _load_fault_sticks_local()


def _parse_dat_bytes(content: str) -> list[tuple[float, float, float]]:
    """Parse a single .dat file content string.  Returns list of (il, xl, twt_ms)."""
    rows: list[tuple[float, float, float]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 3:
            il_idx, xl_idx, z_samp = int(parts[0]), int(parts[1]), int(parts[2])
            abs_il = 1001 + il_idx
            abs_xl = 1900 + xl_idx
            twt_ms = float(z_samp) * 4.0  # z column is sample index, NOT true ms
            rows.append((float(abs_il), float(abs_xl), twt_ms))
    return rows


def _load_fault_sticks_local(
    sticks_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    src_dir = sticks_dir or _local_sources().sticks_dir
    sticks: dict[str, np.ndarray] = {}
    if not src_dir.exists():
        return sticks
    for dat_file in sorted(src_dir.glob("*.dat")):
        rows = _parse_dat_bytes(dat_file.read_text(encoding="utf-8"))
        if rows:
            sticks[dat_file.stem] = np.array(rows, dtype=np.float32)
    return sticks


def _load_fault_sticks_azure() -> dict[str, np.ndarray]:
    try:
        src = _azure_sources()
        client = _storage_client()
        blob_names = client.list_blobs(src.sticks_container, prefix=src.sticks_prefix)
        dat_blobs = [n for n in blob_names if n.endswith(".dat")]
        sticks: dict[str, np.ndarray] = {}
        for blob_name in sorted(dat_blobs):
            stem = Path(blob_name).stem
            raw_bytes = client.download_blob(src.sticks_container, blob_name)
            rows = _parse_dat_bytes(raw_bytes.decode("utf-8"))
            if rows:
                sticks[stem] = np.array(rows, dtype=np.float32)
        return sticks
    except Exception:
        return {}
