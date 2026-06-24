"""Tests for the deepseismic viewer data readers and coordinate mapping.

Coverage
--------
1. Amplitude zarr reader — shape, correct index, out-of-range clamping, NaN/Inf guard.
2. Fault probability zarr reader — shape, values ∈ [0, 1], missing-bake returns None.
3. Fault-stick coordinate mapping — z column is sample index (×4 ms), not raw ms.
   Regression guard: main fault 808–908 ms, antithetic 1200–1228 ms.
4. _write_zarr_volume zarr v3 write/read roundtrip (float32 and uint8, overwrite flag).
5. Viewer module — AST-level import guard: key functions present, no syntax errors.

Coupling note (for Dallas)
--------------------------
_get_amplitude_slice, _get_fault_prob_slice, and _load_fault_sticks are decorated with
@st.cache_data and the module has Streamlit calls + sidebar rendering at import time.
These tests replicate the pure data-path logic directly rather than importing
streamlit_app, which would require a comprehensive Streamlit mock. Recommend Dallas
extract the reader logic into un-decorated helpers in a separate module (e.g.
deepseismic/ui/_data_readers.py) so they can be unit-tested without mocking Streamlit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import zarr

from deepseismic.models.inference import _write_zarr_volume

# ---------------------------------------------------------------------------
# Paths to real staged data (integration-style local reads — no Azurite/GPU)
# ---------------------------------------------------------------------------

_REPO_ROOT  = Path(__file__).parents[3]
_ZARR_AMP   = _REPO_ROOT / "data/volve/staged/synthetic.zarr"
_ZARR_PROB  = _REPO_ROOT / "data/volve/staged/fault_prob.zarr"
_STICKS_DIR = _REPO_ROOT / "data/volve/interpretations/fault_sticks"


# ---------------------------------------------------------------------------
# Pure-logic helpers mirroring the app's reader functions
# (These replicate the logic that lives behind @st.cache_data in streamlit_app.py)
# ---------------------------------------------------------------------------


def _read_inline_array() -> np.ndarray:
    root = zarr.open_group(str(_ZARR_AMP), mode="r")
    return np.asarray(root["inline"][:])


def _amplitude_slice(inline_abs: int) -> np.ndarray:
    """Mirror of _get_amplitude_slice() — reads real zarr, clamps out-of-range inline."""
    root = zarr.open_group(str(_ZARR_AMP), mode="r")
    il_arr = _read_inline_array()
    idx = int(np.clip(np.searchsorted(il_arr, inline_abs), 0, len(il_arr) - 1))
    return np.asarray(root["amplitude"][idx, :, :], dtype=np.float32)


def _fault_prob_slice(inline_abs: int, zarr_path: Path = _ZARR_PROB) -> np.ndarray | None:
    """Mirror of _get_fault_prob_slice() — returns None when zarr is absent."""
    if not zarr_path.exists():
        return None
    root = zarr.open_group(str(zarr_path), mode="r")
    il_arr = _read_inline_array()
    idx = int(np.clip(np.searchsorted(il_arr, inline_abs), 0, len(il_arr) - 1))
    return np.asarray(root["fault_probability"][idx, :, :], dtype=np.float32)


def _parse_fault_sticks() -> dict[str, np.ndarray]:
    """Mirror of _load_fault_sticks() — applies the canonical coordinate mapping."""
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
                    abs_il = 1001 + il_idx       # 0-based vol index → absolute inline
                    abs_xl = 1900 + xl_idx       # 0-based vol index → absolute crossline
                    twt_ms = float(z_samp) * 4.0  # sample index → milliseconds
                    rows.append((float(abs_il), float(abs_xl), twt_ms))
        if rows:
            sticks[dat_file.stem] = np.array(rows, dtype=np.float32)
    return sticks


# ---------------------------------------------------------------------------
# 1. Amplitude reader
# ---------------------------------------------------------------------------


class TestAmplitudeReader:
    """Validate _get_amplitude_slice logic against the real staged zarr."""

    def test_valid_inline_returns_correct_shape(self):
        """Mid-volume inline → (n_xl=200, n_s=500) float32 slice."""
        arr = _amplitude_slice(1050)
        assert arr.shape == (200, 500), f"Expected (200, 500), got {arr.shape}"
        assert arr.dtype == np.float32

    def test_first_inline_index_zero(self):
        """Inline 1001 must map to index 0 (first slice in the volume)."""
        root = zarr.open_group(str(_ZARR_AMP), mode="r")
        expected = np.asarray(root["amplitude"][0, :, :], dtype=np.float32)
        actual = _amplitude_slice(1001)
        np.testing.assert_array_equal(actual, expected)

    def test_last_inline_index_99(self):
        """Inline 1100 must map to index 99 (last slice in the volume)."""
        root = zarr.open_group(str(_ZARR_AMP), mode="r")
        expected = np.asarray(root["amplitude"][99, :, :], dtype=np.float32)
        actual = _amplitude_slice(1100)
        np.testing.assert_array_equal(actual, expected)

    def test_different_inlines_return_different_slices(self):
        """Sampling two different inlines must return different data (not constant)."""
        s1 = _amplitude_slice(1001)
        s99 = _amplitude_slice(1100)
        assert not np.allclose(s1, s99, atol=1e-7), (
            "Inline 1001 and 1100 returned identical slices — index mapping may be broken"
        )

    def test_out_of_range_low_clamps_to_first(self):
        """Inline below the minimum (500 << 1001) must clamp to the first slice."""
        s_low = _amplitude_slice(500)
        s_min = _amplitude_slice(1001)
        np.testing.assert_array_equal(s_low, s_min)

    def test_out_of_range_high_clamps_to_last(self):
        """Inline above the maximum (9999 >> 1100) must clamp to the last slice."""
        s_high = _amplitude_slice(9999)
        s_max = _amplitude_slice(1100)
        np.testing.assert_array_equal(s_high, s_max)

    def test_no_nans_or_infs(self):
        """Amplitude data must be finite (no NaN or Inf)."""
        arr = _amplitude_slice(1050)
        assert not np.any(np.isnan(arr)), "NaN found in amplitude slice"
        assert not np.any(np.isinf(arr)), "Inf found in amplitude slice"


# ---------------------------------------------------------------------------
# 2. Fault probability reader
# ---------------------------------------------------------------------------


class TestFaultProbReader:
    """Validate _get_fault_prob_slice logic against the baked fault_prob.zarr."""

    def test_valid_inline_returns_correct_shape(self):
        """Fault prob slice for a valid inline must have shape (n_xl=200, n_s=500)."""
        arr = _fault_prob_slice(1050)
        assert arr is not None, "fault_prob.zarr exists but reader returned None"
        assert arr.shape == (200, 500), f"Expected (200, 500), got {arr.shape}"

    def test_dtype_is_float32(self):
        arr = _fault_prob_slice(1050)
        assert arr is not None
        assert arr.dtype == np.float32

    def test_values_in_unit_interval(self):
        """All fault probability values must lie in [0, 1]."""
        arr = _fault_prob_slice(1050)
        assert arr is not None
        assert float(arr.min()) >= 0.0 - 1e-6, f"Prob below 0: {arr.min()}"
        assert float(arr.max()) <= 1.0 + 1e-6, f"Prob above 1: {arr.max()}"

    def test_no_nans_or_infs(self):
        arr = _fault_prob_slice(1050)
        assert arr is not None
        assert not np.any(np.isnan(arr)), "NaN in fault probability slice"
        assert not np.any(np.isinf(arr)), "Inf in fault probability slice"

    def test_missing_bake_returns_none(self, tmp_path: Path):
        """When the fault_prob zarr is absent, reader must return None (not raise).

        This guards the graceful-fallback path in _get_fault_prob_slice():
            if not _ZARR_PROB.exists(): return None
        """
        result = _fault_prob_slice(1050, zarr_path=tmp_path / "does_not_exist.zarr")
        assert result is None, "Expected None for missing bake, got data"


# ---------------------------------------------------------------------------
# 3. Fault-stick coordinate mapping
#    Highest-value regression guard: z column is sample index, not true ms.
# ---------------------------------------------------------------------------


class TestFaultStickCoordinateMapping:
    """Pin the canonical coordinate mapping for .dat fault-stick files.

    Regression guard: if the bug of treating z_ms as true milliseconds were
    reintroduced, fault TWT values would be ~202–307 ms (unrealistically shallow).
    These tests fail loudly on that regression.
    """

    @pytest.fixture(scope="class")
    def sticks(self) -> dict[str, np.ndarray]:
        loaded = _parse_fault_sticks()
        assert loaded, f"No .dat files found in {_STICKS_DIR}"
        return loaded

    def test_both_fault_files_loaded(self, sticks: dict[str, np.ndarray]):
        assert "fault_main_normal" in sticks, "fault_main_normal.dat not parsed"
        assert "fault_antithetic" in sticks, "fault_antithetic.dat not parsed"

    def test_twt_not_raw_z_column(self, sticks: dict[str, np.ndarray]):
        """TWT values must be >=800 ms for all sticks.

        If z were used as raw ms (202–307), values would be <<800 ms.
        The >=800 ms guard fails loudly if the sample-index interpretation is lost.
        """
        for name, arr in sticks.items():
            twt = arr[:, 2]
            assert float(twt.min()) >= 800.0, (
                f"{name}: TWT min {twt.min():.1f} ms < 800 ms — "
                "likely raw z_ms bug (z column must be multiplied by 4.0, not used as ms)"
            )

    def test_main_fault_first_row_exact_mapping(self, sticks: dict[str, np.ndarray]):
        """Pin first row of main fault: dat(45, 84, 202) → abs_il=1046, abs_xl=1984, twt=808 ms."""
        row = sticks["fault_main_normal"][0]
        assert row[0] == pytest.approx(1046.0, abs=0.1), f"abs_inline: {row[0]}"
        assert row[1] == pytest.approx(1984.0, abs=0.1), f"abs_crossline: {row[1]}"
        assert row[2] == pytest.approx(808.0, abs=0.1), f"twt_ms: {row[2]}"

    def test_main_fault_twt_band_808_to_908ms(self, sticks: dict[str, np.ndarray]):
        """Main fault TWT must lie in 808–908 ms (z_samples 202–227 × 4 ms/sample)."""
        twt = sticks["fault_main_normal"][:, 2]
        assert float(twt.min()) == pytest.approx(808.0, abs=0.5), f"min TWT: {twt.min()}"
        assert float(twt.max()) == pytest.approx(908.0, abs=0.5), f"max TWT: {twt.max()}"

    def test_antithetic_fault_twt_band_1200_to_1228ms(self, sticks: dict[str, np.ndarray]):
        """Antithetic fault TWT must lie in 1200–1228 ms (z_samples 300–307 × 4 ms/sample)."""
        twt = sticks["fault_antithetic"][:, 2]
        assert float(twt.min()) == pytest.approx(1200.0, abs=0.5), f"min TWT: {twt.min()}"
        assert float(twt.max()) == pytest.approx(1228.0, abs=0.5), f"max TWT: {twt.max()}"

    def test_inline_absolute_mapping(self, sticks: dict[str, np.ndarray]):
        """Inline column (0-based index 45–95) must map to absolute 1046–1096."""
        abs_il = sticks["fault_main_normal"][:, 0]
        assert float(abs_il.min()) == pytest.approx(1046.0, abs=0.1)
        assert float(abs_il.max()) == pytest.approx(1096.0, abs=0.1)

    def test_crossline_absolute_mapping(self, sticks: dict[str, np.ndarray]):
        """Crossline column (0-based index 84–124) must map to absolute 1984–2024."""
        abs_xl = sticks["fault_main_normal"][:, 1]
        assert float(abs_xl.min()) == pytest.approx(1984.0, abs=0.1)
        assert float(abs_xl.max()) == pytest.approx(2024.0, abs=0.1)

    def test_sample_index_formula(self):
        """Unit-level: z_sample × 4.0 == twt_ms for representative values."""
        cases = [(202, 808.0), (227, 908.0), (300, 1200.0), (307, 1228.0)]
        for z_samp, expected_ms in cases:
            twt = float(z_samp) * 4.0
            assert twt == pytest.approx(expected_ms, abs=1e-3), (
                f"z_sample={z_samp}: expected {expected_ms} ms, got {twt}"
            )


# ---------------------------------------------------------------------------
# 4. _write_zarr_volume zarr v3 roundtrip
# ---------------------------------------------------------------------------


class TestWriteZarrVolume:
    """Validate the zarr v3-fixed _write_zarr_volume() in inference.py."""

    def test_float32_roundtrip(self, tmp_path: Path):
        """Written float32 volume reads back with correct name, shape, dtype, and values."""
        rng = np.random.default_rng(42)
        vol = rng.standard_normal((8, 8, 16)).astype(np.float32)
        out_path = tmp_path / "tiny_prob.zarr"

        _write_zarr_volume(vol, out_path, dataset_name="fault_probability", dtype=np.float32)

        root = zarr.open_group(str(out_path), mode="r")
        assert "fault_probability" in list(root.array_keys()), (
            "Array 'fault_probability' not found in written store"
        )
        arr = root["fault_probability"]
        assert arr.shape == (8, 8, 16), f"Shape mismatch: {arr.shape}"
        assert arr.dtype == np.float32
        np.testing.assert_allclose(np.asarray(arr[:]), vol, rtol=1e-5)

    def test_uint8_roundtrip(self, tmp_path: Path):
        """Written uint8 mask reads back with correct dtype and bit-exact values."""
        rng = np.random.default_rng(7)
        mask = (rng.random((8, 8, 16)) > 0.85).astype(np.uint8)
        out_path = tmp_path / "tiny_mask.zarr"

        _write_zarr_volume(mask, out_path, dataset_name="fault_mask", dtype=np.uint8)

        root = zarr.open_group(str(out_path), mode="r")
        arr = root["fault_mask"]
        assert arr.dtype == np.uint8
        np.testing.assert_array_equal(np.asarray(arr[:]), mask)

    def test_no_overwrite_raises(self, tmp_path: Path):
        """Writing to an existing store without overwrite=True must raise."""
        vol = np.zeros((4, 4, 8), dtype=np.float32)
        out_path = tmp_path / "existing.zarr"
        _write_zarr_volume(vol, out_path, dataset_name="fault_probability", dtype=np.float32)
        with pytest.raises(FileExistsError):
            _write_zarr_volume(
                vol, out_path, dataset_name="fault_probability",
                dtype=np.float32, overwrite=False,
            )

    def test_overwrite_true_replaces_data(self, tmp_path: Path):
        """overwrite=True rewrites the store; new values must replace old ones."""
        out_path = tmp_path / "overwrite.zarr"
        vol1 = np.ones((4, 4, 8), dtype=np.float32)
        vol2 = np.zeros((4, 4, 8), dtype=np.float32)
        _write_zarr_volume(vol1, out_path, dataset_name="fault_probability", dtype=np.float32)
        _write_zarr_volume(
            vol2, out_path, dataset_name="fault_probability",
            dtype=np.float32, overwrite=True,
        )
        root = zarr.open_group(str(out_path), mode="r")
        np.testing.assert_array_equal(np.asarray(root["fault_probability"][:]), vol2)

    def test_custom_chunks_preserved(self, tmp_path: Path):
        """Custom chunk shape is written into the store metadata."""
        vol = np.zeros((16, 16, 32), dtype=np.float32)
        out_path = tmp_path / "chunked.zarr"
        _write_zarr_volume(
            vol, out_path, dataset_name="fault_probability",
            dtype=np.float32, chunks=(8, 8, 16),
        )
        root = zarr.open_group(str(out_path), mode="r")
        arr = root["fault_probability"]
        assert arr.chunks == (8, 8, 16), f"Chunk shape mismatch: {arr.chunks}"


# ---------------------------------------------------------------------------
# 5. Viewer module regression guard (AST-level, no Streamlit display required)
# ---------------------------------------------------------------------------


class TestViewerModuleRegression:
    """Guard the viewer module against syntax errors and accidental function removal.

    Rationale: importing streamlit_app.py directly in a test environment requires a
    comprehensive Streamlit mock (the module has top-level @st.cache_data calls,
    sidebar rendering, and a data read that runs at import time). Rather than fighting
    that coupling, we validate the AST — which catches syntax errors and missing
    function definitions without executing Streamlit code.
    """

    _APP_PATH = _REPO_ROOT / "src/deepseismic/ui/streamlit_app.py"

    _EXPECTED_FUNCTIONS = {
        "_get_amplitude_slice",
        "_get_fault_prob_slice",
        "_load_fault_sticks",
        "_get_volume_coords",
        "_render_seismic_section",
    }

    @pytest.fixture(scope="class")
    def app_ast(self) -> ast.Module:
        assert self._APP_PATH.exists(), f"streamlit_app.py not found: {self._APP_PATH}"
        source = self._APP_PATH.read_text(encoding="utf-8")
        return ast.parse(source)  # raises SyntaxError if broken

    def test_no_syntax_errors(self, app_ast: ast.Module):
        """streamlit_app.py must parse without SyntaxError."""
        assert isinstance(app_ast, ast.Module)

    def test_key_reader_functions_present(self, app_ast: ast.Module):
        """All viewer data-reader functions must be defined (no accidental removal)."""
        defined = {
            node.name
            for node in ast.walk(app_ast)
            if isinstance(node, ast.FunctionDef)
        }
        missing = self._EXPECTED_FUNCTIONS - defined
        assert not missing, f"Functions missing from streamlit_app.py: {sorted(missing)}"

    def test_amplitude_reader_uses_correct_array_name(self, app_ast: ast.Module):
        """_get_amplitude_slice must index root['amplitude'], not 'amp' or other alias."""
        source = self._APP_PATH.read_text(encoding="utf-8")
        assert '"amplitude"' in source or "'amplitude'" in source, (
            "String 'amplitude' not found in streamlit_app.py — array name may have changed"
        )

    def test_fault_prob_uses_correct_array_name(self, app_ast: ast.Module):
        """_get_fault_prob_slice must index root['fault_probability'] (not 'fault_prob')."""
        source = self._APP_PATH.read_text(encoding="utf-8")
        assert '"fault_probability"' in source or "'fault_probability'" in source, (
            "String 'fault_probability' not found — array name contract may be broken"
        )
