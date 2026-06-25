"""Seismic data conditioning and quality-control pipeline.

Geophysical conventions assumed throughout
------------------------------------------
- **Phase:** Zero-phase wavelet (symmetric, peak at centre). The symmetry proxy
  in :func:`compute_volume_qc` flags deviations from zero-phase using the
  time-reversal symmetry coefficient of the mean trace: a value near +1.0
  indicates a symmetric (zero-phase) wavelet; ~0.0 indicates a 90°-rotated
  (quadrature-phase) wavelet.
- **Polarity:** SEG normal (American) convention — a hard positive-impedance
  contrast produces a positive amplitude peak on the seismic section.
- **Sample interval:** Read from the ingest sidecar JSON key
  ``geometry.sample_rate_ms``; defaults to 4 ms (250 Hz Nyquist, 125 Hz
  Nyquist) when not supplied.
- **Vertical resolution:** At dominant frequency *f₀* and interval velocity *v*
  the tuning (λ/4) thickness is *v* / (4·*f₀*).  For *f₀* ≈ 30 Hz and
  *v* ≈ 2000 m/s this gives ≈ 17 m — bed pairs thinner than this cannot be
  resolved individually from amplitude alone.
- **Amplitude scale:** Amplitudes are in display units as written by the ingest
  step (not calibrated reflectivity).  The :func:`global_amplitude_normalize`
  function divides by the volume p99 (from the sidecar) to preserve lateral and
  vertical amplitude gradients — unlike the per-patch z-score in
  :mod:`deepseismic.preprocessing.patches` which suppresses those gradients and
  is unsuitable for AVO or attribute studies (GAP-I1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DT_MS: float = 4.0          # 4 ms → 250 Hz sample rate
_DEFAULT_VELOCITY_MS: float = 2000.0 # representative NMO velocity (m/s)
_TRACE_SAMPLE_MAX: int = 500         # max traces used for spectral/autocorr estimates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_sidecar(sidecar_path: Path) -> dict[str, Any]:
    """Return parsed sidecar JSON, or empty dict on failure."""
    try:
        with open(sidecar_path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read sidecar %s: %s", sidecar_path, exc)
        return {}


def _open_amplitude(zarr_path: Path | str) -> zarr.Array:
    """Open the amplitude array from a Zarr group or a bare Zarr array."""
    store = zarr.open(str(zarr_path), mode="r")
    if isinstance(store, zarr.Array):
        return store
    if "amplitude" in store:
        return store["amplitude"]
    # fall back to first array found
    for key in store:
        candidate = store[key]
        if isinstance(candidate, zarr.Array):
            logger.warning("'amplitude' dataset not found; using '%s'", key)
            return candidate
    raise KeyError(f"No array found in Zarr store: {zarr_path}")


def _sample_traces(arr: zarr.Array, n_traces: int = _TRACE_SAMPLE_MAX) -> np.ndarray:
    """Return a (n_traces, n_samples) float64 array of evenly-spaced traces.

    The volume is assumed to be ordered (inline, crossline, sample).
    Traces are drawn from a uniform grid over the inline×crossline plane.
    """
    n_il, n_xl, n_s = arr.shape
    n_total = n_il * n_xl
    step = max(1, n_total // n_traces)
    indices = np.arange(0, n_total, step)[:n_traces]
    il_idx = indices // n_xl
    xl_idx = indices % n_xl
    traces = np.empty((len(indices), n_s), dtype=np.float64)
    for i, (il, xl) in enumerate(zip(il_idx, xl_idx, strict=True)):
        traces[i] = arr[int(il), int(xl), :].astype(np.float64)
    return traces


def _dominant_frequency_hz(traces: np.ndarray, dt_ms: float) -> float:
    """Estimate the dominant (peak-energy) frequency from a set of traces.

    Uses the magnitude spectrum of the mean amplitude spectrum across all
    supplied traces.  *dt_ms* is the sample interval in milliseconds.

    Zero-padding to the next power-of-two improves frequency resolution without
    biasing the peak location.
    """
    n_s = traces.shape[1]
    n_fft = int(2 ** np.ceil(np.log2(n_s)))      # next power-of-two zero-pad
    dt_s = dt_ms * 1e-3
    freqs = np.fft.rfftfreq(n_fft, d=dt_s)        # Hz
    spectra = np.abs(np.fft.rfft(traces, n=n_fft, axis=1))
    mean_spectrum = spectra.mean(axis=0)
    peak_idx = int(np.argmax(mean_spectrum))
    return float(freqs[peak_idx])


def _wavelet_symmetry(traces: np.ndarray) -> float:
    """Zero-phase proxy: time-reversal symmetry coefficient of the mean wavelet.

    A zero-phase wavelet is symmetric in time — w(t) = w(-t) — so its
    normalised dot-product with its own time-reverse equals +1.0.  A
    90°-rotated wavelet (antisymmetric, i.e. its Hilbert transform) yields
    approximately −1.0.  Mixed-phase signals fall between the two.

    The metric is the normalised time-reversal correlation at zero lag::

        C = dot(x, x_reversed) / dot(x, x)

    Range
    -----
    +1.0   Perfectly symmetric → zero-phase (or 180°-rotated if all signs
           flip, but seismic polarity conventions handle that separately)
    ~0.0   Quadrature-phase (90° or 270° rotation)
    −1.0   Perfectly antisymmetric

    Geophysical caveat
    ------------------
    This is an approximate proxy derived from the mean trace over the full
    record length.  It reflects bulk wavelet symmetry, not event-by-event
    phase.  Interference from multiple reflectors can bias the estimate.
    For a definitive phase assessment use well-tie or deterministic
    wavelet extraction.  Values above ~0.8 are consistent with near-zero-phase.

    Implementation
    --------------
    Uses only ``numpy`` (dot products on the mean trace); no scipy required.
    """
    mean_trace = traces.mean(axis=0)
    energy = float(np.dot(mean_trace, mean_trace))
    if energy < 1e-30:
        return float("nan")
    return float(np.dot(mean_trace, mean_trace[::-1])) / energy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_volume_qc(
    zarr_path: Path | str,
    sidecar_path: Path | str | None = None,
    n_trace_samples: int = _TRACE_SAMPLE_MAX,
    reference_velocity_ms: float = _DEFAULT_VELOCITY_MS,
) -> dict[str, Any]:
    """Compute quality-control metrics for a 3-D seismic amplitude volume.

    Parameters
    ----------
    zarr_path:
        Path to the Zarr store containing the amplitude array.
    sidecar_path:
        Path to the JSON sidecar written at ingest time.  When supplied, the
        sample interval and ingest statistics are read from it.  Otherwise
        defaults apply (dt = 4 ms).
    n_trace_samples:
        Number of traces to subsample for spectral and autocorrelation
        estimates.  Default 500.
    reference_velocity_ms:
        Representative interval velocity in m/s used to compute the vertical
        resolution estimate.  Default 2000 m/s (typical shallow marine).

    Returns
    -------
    dict
        QC metrics dictionary with keys:

        ``shape``           (n_inline, n_crossline, n_sample)
        ``dtype``           string
        ``nonzero_fraction`` fraction of non-zero samples  (1.0 = no dead traces)
        ``amp_min``         global amplitude minimum
        ``amp_max``         global amplitude maximum
        ``amp_mean``        global amplitude mean
        ``amp_std``         global amplitude standard deviation
        ``amp_p01``         1st percentile amplitude
        ``amp_p99``         99th percentile amplitude
        ``dt_ms``           sample interval in milliseconds
        ``dominant_freq_hz`` peak-energy frequency estimate (Hz)
        ``vertical_resolution_m`` λ/4 tuning thickness at dominant frequency
        ``wavelet_symmetry``  zero-phase proxy: +1.0 = symmetric (zero-phase),
                              ~0.0 = 90°-rotated, −1.0 = antisymmetric
        ``sidecar_stats_used`` bool — whether sidecar amplitude stats were used

    Geophysical notes
    -----------------
    - Dominant frequency is estimated from the mean amplitude spectrum of
      *n_trace_samples* evenly-spaced traces.  It is not a weighted centroid;
      it reflects the peak-energy lobe of the bandwidth.
    - The wavelet symmetry metric is a quick zero-phase proxy, not a
      full minimum-phase diagnostic.  A value below 0.8 warrants
      further investigation (e.g. deterministic wavelet extraction or well-tie).
    - Vertical resolution assumes a single reference velocity; real resolution
      varies with depth and lithology.
    """
    zarr_path = Path(zarr_path)
    arr = _open_amplitude(zarr_path)

    # --- sidecar ---
    sidecar: dict[str, Any] = {}
    if sidecar_path is None:
        candidate = zarr_path.parent / (zarr_path.stem + ".json")
        if candidate.exists():
            sidecar_path = candidate
    if sidecar_path is not None:
        sidecar = _load_sidecar(Path(sidecar_path))

    dt_ms: float = (
        sidecar.get("geometry", {}).get("sample_rate_ms", _DEFAULT_DT_MS)
    )

    # --- amplitude statistics ---
    # Prefer pre-computed sidecar stats to avoid loading the full volume
    amp_stats = sidecar.get("amplitude_stats", {})
    sidecar_stats_used = bool(amp_stats)

    if sidecar_stats_used:
        amp_min   = float(amp_stats["min"])
        amp_max   = float(amp_stats["max"])
        amp_mean  = float(amp_stats["mean"])
        amp_std   = float(amp_stats["std"])
        amp_p01   = float(amp_stats["p01"])
        amp_p99   = float(amp_stats["p99"])
        nonzero_frac = float(amp_stats.get("nonzero_fraction", float("nan")))
    else:
        logger.info("No sidecar stats — computing from full volume (may be slow)")
        data = arr[:].astype(np.float64)
        amp_min  = float(data.min())
        amp_max  = float(data.max())
        amp_mean = float(data.mean())
        amp_std  = float(data.std())
        amp_p01  = float(np.percentile(data, 1))
        amp_p99  = float(np.percentile(data, 99))
        nonzero_frac = float(np.count_nonzero(data) / data.size)

    # --- spectral and phase estimates from sampled traces ---
    traces = _sample_traces(arr, n_traces=n_trace_samples)
    dominant_freq = _dominant_frequency_hz(traces, dt_ms)
    autocorr_sym  = _wavelet_symmetry(traces)

    # --- resolution estimate ---
    # λ/4 tuning thickness at dominant frequency and reference velocity
    if dominant_freq > 0:
        vert_res_m = reference_velocity_ms / (4.0 * dominant_freq)
    else:
        vert_res_m = float("nan")

    metrics: dict[str, Any] = {
        "shape":                tuple(arr.shape),
        "dtype":                str(arr.dtype),
        "nonzero_fraction":     round(nonzero_frac, 6),
        "amp_min":              amp_min,
        "amp_max":              amp_max,
        "amp_mean":             amp_mean,
        "amp_std":              amp_std,
        "amp_p01":              amp_p01,
        "amp_p99":              amp_p99,
        "dt_ms":                dt_ms,
        "dominant_freq_hz":     round(dominant_freq, 2),
        "vertical_resolution_m": round(vert_res_m, 1),
        "wavelet_symmetry":    round(autocorr_sym, 6),
        "sidecar_stats_used":   sidecar_stats_used,
    }

    _log_qc_report(metrics)
    return metrics


def _log_qc_report(m: dict[str, Any]) -> None:
    """Emit a human-readable QC summary at INFO level."""
    logger.info(
        "=== Volume QC Report ===\n"
        "  Shape (IL × XL × samples): %s\n"
        "  Non-zero fraction:          %.4f\n"
        "  Amplitude  min/max:         %.4g / %.4g\n"
        "             mean ± std:      %.4g ± %.4g\n"
        "             p01 / p99:       %.4g / %.4g\n"
        "  Sample interval:            %.1f ms  (Nyquist %.0f Hz)\n"
        "  Dominant frequency:         %.1f Hz\n"
        "  Vertical resolution (λ/4):  %.1f m  (v_ref=2000 m/s)\n"
        "  Wavelet symmetry (ZP proxy): %.4f  (+1.0=zero-phase, ~0=90°-rot)\n"
        "  Sidecar stats used:         %s",
        m["shape"],
        m["nonzero_fraction"],
        m["amp_min"], m["amp_max"],
        m["amp_mean"], m["amp_std"],
        m["amp_p01"], m["amp_p99"],
        m["dt_ms"], 1000.0 / (2.0 * m["dt_ms"]),
        m["dominant_freq_hz"],
        m["vertical_resolution_m"],
        m["wavelet_symmetry"],
        m["sidecar_stats_used"],
    )


def global_amplitude_normalize(
    volume: np.ndarray,
    p99: float,
    clip: bool = True,
) -> np.ndarray:
    """Amplitude-preserving normalisation by the volume 99th-percentile.

    Divides every sample by *p99* so that the bulk of energy falls in [-1, 1]
    without destroying the lateral or vertical amplitude variation that
    carries stratigraphic and fluid information.

    This is distinct from the per-patch z-score in
    :func:`deepseismic.preprocessing.patches._normalize_patch`, which
    independently normalises each patch and thereby erases inter-patch
    amplitude contrasts (GAP-I1 fix).

    Parameters
    ----------
    volume:
        Input amplitude array, any shape, float32 or float64.
    p99:
        99th-percentile amplitude value for the full volume, typically from
        the ingest sidecar.  Must be positive; a zero or negative value raises
        ``ValueError``.
    clip:
        When ``True`` (default) the result is clipped to [-1.5, +1.5] to
        suppress isolated hot pixels while still preserving amplitude
        hierarchy between patches.

    Returns
    -------
    np.ndarray
        Normalised array, same shape and dtype as *volume*.

    Geophysical note
    ----------------
    Using p99 rather than the absolute maximum limits the influence of
    isolated noise spikes.  The choice of 99th percentile is conventional
    in seismic display and balances outlier robustness with dynamic-range
    preservation.  For AVO workflows consider exposing p95 or a user-
    supplied scalar tied to a well-tie calibration.
    """
    if p99 <= 0.0:
        raise ValueError(
            f"p99 must be positive for amplitude-preserving normalisation; got {p99}"
        )
    out = volume.astype(np.float32, copy=True) / np.float32(p99)
    if clip:
        np.clip(out, -1.5, 1.5, out=out)
    return out


def condition_volume(
    zarr_path: Path | str,
    sidecar_path: Path | str | None = None,
    normalize: bool = False,
    report_path: Path | str | None = None,
    n_trace_samples: int = _TRACE_SAMPLE_MAX,
    reference_velocity_ms: float = _DEFAULT_VELOCITY_MS,
) -> dict[str, Any]:
    """Orchestrate QC computation and optional amplitude conditioning.

    This is the primary entry point for the conditioning stage shown in
    ``docs/architecture/process-architecture.md``.

    Parameters
    ----------
    zarr_path:
        Path to the Zarr amplitude store.
    sidecar_path:
        Path to the ingest sidecar JSON.  Auto-detected from *zarr_path* if
        omitted.
    normalize:
        If ``True``, load the full amplitude volume, apply
        :func:`global_amplitude_normalize`, and store the normalised array
        under the ``amplitude_norm`` key in the same Zarr group.  The
        original ``amplitude`` dataset is never overwritten.
    report_path:
        If supplied, the QC metrics dict is written as JSON to this path.
    n_trace_samples:
        Number of traces sampled for spectral/phase estimates.
    reference_velocity_ms:
        Reference velocity for λ/4 resolution estimate (m/s).

    Returns
    -------
    dict
        QC metrics dict from :func:`compute_volume_qc`, augmented with key
        ``normalised`` (bool) indicating whether conditioning was applied.
    """
    metrics = compute_volume_qc(
        zarr_path,
        sidecar_path=sidecar_path,
        n_trace_samples=n_trace_samples,
        reference_velocity_ms=reference_velocity_ms,
    )

    if normalize:
        p99 = metrics["amp_p99"]
        logger.info("Applying global amplitude normalisation with p99=%.4g", p99)
        arr = _open_amplitude(Path(zarr_path))
        data = arr[:].astype(np.float32)
        normed = global_amplitude_normalize(data, p99=p99)
        # Write normalised array back to same store under a separate key
        store = zarr.open_group(str(zarr_path), mode="a")
        store.array(
            "amplitude_norm",
            normed,
            chunks=arr.chunks,
            dtype=np.float32,
            overwrite=True,
        )
        logger.info(
            "Wrote 'amplitude_norm' dataset to %s (shape=%s, p99-scaled)",
            zarr_path,
            normed.shape,
        )

    metrics["normalised"] = normalize

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("QC report written to %s", report_path)

    return metrics
