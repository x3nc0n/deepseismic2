"""Sprint 2 S2-06 tests: seismic data conditioning and QC pipeline.

Coverage:
- compute_volume_qc: required keys, shape/dtype correctness, sane stat ranges
- _dominant_frequency_hz: recovers a known frequency from a pure sinusoid and
  a Ricker wavelet (both within physically reasonable tolerance)
- _wavelet_symmetry: time-reversal symmetry coefficient; symmetric Gaussian → ~1.0;
  asymmetric causal exponential deviates from 1.0; zero-phase Ricker scores ≥0.5
  higher than its 90°-rotated counterpart
- global_amplitude_normalize: no mutation, ratio preservation, error guards, clipping

All tests use synthetic in-memory data; no dependency on real Volve files.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
import zarr.storage

from deepseismic.preprocessing.pipeline import (
    _dominant_frequency_hz,
    _wavelet_symmetry,
    compute_volume_qc,
    global_amplitude_normalize,
)

# ---------------------------------------------------------------------------
# Expected QC keys from compute_volume_qc
# ---------------------------------------------------------------------------

_REQUIRED_QC_KEYS = frozenset({
    "shape",
    "dtype",
    "nonzero_fraction",
    "amp_min",
    "amp_max",
    "amp_mean",
    "amp_std",
    "amp_p01",
    "amp_p99",
    "dt_ms",
    "dominant_freq_hz",
    "vertical_resolution_m",
    "wavelet_symmetry",
    "sidecar_stats_used",
})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_amplitude_zarr(tmp_path):
    """10×10×64 float32 random amplitude volume in a zarr group."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 10, 64)).astype(np.float32)
    store = zarr.storage.LocalStore(str(tmp_path / "amp.zarr"))
    root = zarr.open_group(store, mode="w")
    root.create_array("amplitude", data=data, chunks=(5, 5, 32))
    return tmp_path / "amp.zarr"


# ---------------------------------------------------------------------------
# compute_volume_qc
# ---------------------------------------------------------------------------


class TestComputeVolumeQC:
    """Tests for compute_volume_qc on a tiny synthetic amplitude zarr."""

    def test_required_keys_present(self, small_amplitude_zarr):
        qc = compute_volume_qc(small_amplitude_zarr)
        assert _REQUIRED_QC_KEYS.issubset(set(qc.keys())), (
            f"Missing keys: {_REQUIRED_QC_KEYS - set(qc.keys())}"
        )

    def test_shape_matches_array(self, small_amplitude_zarr):
        qc = compute_volume_qc(small_amplitude_zarr)
        assert qc["shape"] == (10, 10, 64)

    def test_dtype_is_float_string(self, small_amplitude_zarr):
        qc = compute_volume_qc(small_amplitude_zarr)
        assert isinstance(qc["dtype"], str)
        assert "float" in qc["dtype"]

    def test_amp_stats_ordering_sane(self, small_amplitude_zarr):
        """min ≤ p01 ≤ p99 ≤ max, std > 0, nonzero_fraction ∈ [0,1]."""
        qc = compute_volume_qc(small_amplitude_zarr)
        assert qc["amp_min"] <= qc["amp_p01"]
        assert qc["amp_p01"] <= qc["amp_p99"]
        assert qc["amp_p99"] <= qc["amp_max"]
        assert qc["amp_std"] > 0
        assert 0.0 <= qc["nonzero_fraction"] <= 1.0

    def test_sidecar_stats_used_false_without_sidecar(self, small_amplitude_zarr):
        qc = compute_volume_qc(small_amplitude_zarr)
        assert qc["sidecar_stats_used"] is False


# ---------------------------------------------------------------------------
# _dominant_frequency_hz
# ---------------------------------------------------------------------------


class TestDominantFrequency:
    """Verify _dominant_frequency_hz recovers known frequencies."""

    @staticmethod
    def _sinusoid_traces(
        freq_hz: float,
        n_traces: int,
        n_samples: int,
        dt_ms: float,
    ) -> np.ndarray:
        """Return (n_traces, n_samples) float64 array of pure-sinusoid traces."""
        t_s = np.arange(n_samples) * dt_ms * 1e-3
        sig = np.sin(2 * np.pi * freq_hz * t_s).astype(np.float64)
        return np.tile(sig, (n_traces, 1))

    def test_30hz_sinusoid_recovered(self):
        """30 Hz sinusoid must be recoverable within ±3 Hz."""
        target = 30.0
        traces = self._sinusoid_traces(target, n_traces=5, n_samples=256, dt_ms=4.0)
        est = _dominant_frequency_hz(traces, dt_ms=4.0)
        assert abs(est - target) <= 3.0, f"Expected ~{target} Hz, got {est:.2f} Hz"

    def test_50hz_sinusoid_recovered(self):
        """50 Hz sinusoid must be recoverable within ±4 Hz."""
        target = 50.0
        traces = self._sinusoid_traces(target, n_traces=5, n_samples=256, dt_ms=4.0)
        est = _dominant_frequency_hz(traces, dt_ms=4.0)
        assert abs(est - target) <= 4.0, f"Expected ~{target} Hz, got {est:.2f} Hz"

    def test_ricker_wavelet_dominant_frequency(self):
        """Ricker wavelet peak frequency should be recoverable within ±5 Hz."""

        def ricker(n: int, f_peak: float, dt_s: float) -> np.ndarray:
            t = (np.arange(n) - n // 2) * dt_s
            pf = np.pi * f_peak
            return (1.0 - 2.0 * pf**2 * t**2) * np.exp(-(pf * t) ** 2)

        target = 35.0
        dt_ms = 4.0
        wav = ricker(128, target, dt_ms * 1e-3).astype(np.float64)
        traces = np.tile(wav, (10, 1))
        est = _dominant_frequency_hz(traces, dt_ms=dt_ms)
        assert abs(est - target) <= 5.0, f"Expected ~{target} Hz, got {est:.2f} Hz"


# ---------------------------------------------------------------------------
# _autocorr_symmetry
# ---------------------------------------------------------------------------


class TestWaveletSymmetry:
    """Tests for the zero-phase proxy _wavelet_symmetry.

    The metric is the normalised time-reversal correlation:
        C = dot(x, x_reversed) / dot(x, x)
    Range: +1.0 (symmetric/zero-phase) → ~0.0 (quadrature) → −1.0 (antisymmetric).
    """

    def test_symmetric_wavelet_gives_ratio_near_1(self):
        """Symmetric traces (Gaussian) must give wavelet_symmetry ≈ 1.0."""
        n = 64
        t = np.arange(n) - n // 2
        sym_trace = np.exp(-0.01 * t**2).astype(np.float64)
        traces = np.tile(sym_trace, (5, 1))
        ratio = _wavelet_symmetry(traces)
        assert abs(ratio - 1.0) < 0.05, f"Expected ~1.0 for symmetric wavelet, got {ratio:.4f}"

    def test_asymmetric_signal_deviates_from_1(self):
        """Asymmetric (causal exponential) signal must give ratio well below 1.0.

        A causal exponential is concentrated near sample 0, so its time-reverse
        is concentrated near the last sample.  The two have very low overlap,
        giving a normalised dot product far from 1.0.
        """
        n = 64
        asym_trace = np.exp(-0.1 * np.arange(n, dtype=np.float64))
        traces = np.tile(asym_trace, (5, 1))
        ratio = _wavelet_symmetry(traces)
        assert abs(ratio - 1.0) > 0.1, (
            f"Expected large deviation from 1.0 for asymmetric signal, got {ratio:.4f}"
        )

    def test_zero_phase_ricker_scores_higher_than_rotated(self):
        """Zero-phase Ricker must score ≥ 0.5 higher than its 90°-rotated version.

        Uses odd-length n=129 so the Ricker peak falls on the exact centre
        sample (index 64), giving perfect time-reversal symmetry (score → 1.0).
        The 90°-rotated (quadrature) Ricker is the Hilbert transform, computed
        via numpy FFT — it is antisymmetric, so its time-reversal coefficient
        should be near −1.0.
        """
        n = 129  # odd: centre at index 64 maps to itself under reversal
        f_peak = 35.0
        dt_s = 4e-3
        t = (np.arange(n) - n // 2) * dt_s
        pf = np.pi * f_peak
        ricker = (1.0 - 2.0 * pf**2 * t**2) * np.exp(-(pf * t) ** 2)

        # 90°-rotated Ricker via Hilbert transform (numpy FFT only)
        X = np.fft.fft(ricker)
        N = len(X)  # N == 129 (odd)
        H = np.zeros(N)
        H[0] = 1.0
        H[1 : (N + 1) // 2] = 2.0  # positive freqs; for odd N there is no Nyquist bin
        hilbert = np.imag(np.fft.ifft(X * H))

        score_zero = _wavelet_symmetry(np.tile(ricker, (5, 1)))
        score_rot  = _wavelet_symmetry(np.tile(hilbert, (5, 1)))

        assert score_zero > 0.8, f"Zero-phase Ricker should score >0.8, got {score_zero:.4f}"
        assert score_zero > score_rot + 0.5, (
            f"Zero-phase ({score_zero:.4f}) should exceed rotated ({score_rot:.4f}) by ≥0.5"
        )


# ---------------------------------------------------------------------------
# global_amplitude_normalize
# ---------------------------------------------------------------------------


class TestGlobalAmplitudeNormalize:
    """Tests for global_amplitude_normalize: immutability, ratio, error guards, clipping."""

    def test_does_not_mutate_input_array(self):
        """Input volume must not be modified in place."""
        vol = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        original = vol.copy()
        global_amplitude_normalize(vol, p99=4.0)
        np.testing.assert_array_equal(vol, original)

    def test_relative_amplitude_ratios_preserved(self):
        """Values that differ by 2× before normalisation still differ by 2× after."""
        vol = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        normed = global_amplitude_normalize(vol, p99=4.0, clip=False)
        assert normed[1] / normed[0] == pytest.approx(2.0, rel=1e-5)
        assert normed[2] / normed[0] == pytest.approx(4.0, rel=1e-5)

    def test_p99_value_normalises_to_1(self):
        """A sample equal to p99 must normalise to exactly 1.0."""
        vol = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
        normed = global_amplitude_normalize(vol, p99=4.0, clip=False)
        assert normed[3] == pytest.approx(1.0, rel=1e-5)

    def test_zero_p99_raises_value_error(self):
        """p99 = 0 must raise ValueError (not silently produce inf/nan)."""
        vol = np.ones((3,), dtype=np.float32)
        with pytest.raises(ValueError, match="p99 must be positive"):
            global_amplitude_normalize(vol, p99=0.0)

    def test_negative_p99_raises_value_error(self):
        """p99 < 0 must also raise ValueError."""
        vol = np.ones((3,), dtype=np.float32)
        with pytest.raises(ValueError):
            global_amplitude_normalize(vol, p99=-1.0)

    def test_clip_true_bounds_output(self):
        """With clip=True the output must lie within [-1.5, +1.5]."""
        vol = np.array([-200.0, 0.0, 200.0], dtype=np.float32)
        normed = global_amplitude_normalize(vol, p99=1.0, clip=True)
        assert float(normed.min()) >= -1.5
        assert float(normed.max()) <= 1.5

    def test_clip_false_allows_extremes(self):
        """With clip=False large values must pass through unchanged."""
        vol = np.array([-200.0, 0.0, 200.0], dtype=np.float32)
        normed = global_amplitude_normalize(vol, p99=1.0, clip=False)
        assert float(normed.max()) > 1.5
