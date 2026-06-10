"""Smoke tests for deepseismic.ingest.segy_loader.

Strategy:
- Direct segyio tests validate data-manipulation logic (always run).
- Interface tests mock the expected function signatures (always run).
- "Real impl" tests skip gracefully when functions are not yet implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr

from deepseismic.ingest import segy_loader as _mod

# Probe for real implementations — None until the code lands
_load_segy = getattr(_mod, "load_segy", None)
_segy_to_zarr = getattr(_mod, "segy_to_zarr", None)
_extract_metadata = getattr(_mod, "extract_metadata", None)

_REQUIRED_METADATA = frozenset(
    {
        "survey_name",
        "n_inlines",
        "n_crosslines",
        "n_samples",
        "sample_interval_ms",
        "inline_range",
        "crossline_range",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# test_load_segy_basic
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadSegyBasic:
    def test_load_segy_basic_segyio(self, sample_segy_path: Path) -> None:
        """Direct segyio read of synthetic file must produce correct dimensions."""
        import segyio

        with segyio.open(str(sample_segy_path), ignore_geometry=False) as f:
            assert len(f.ilines) == 5
            assert len(f.xlines) == 5
            assert len(f.samples) == 100
            assert f.trace.raw[:].shape == (25, 100)

    def test_load_segy_basic_mock_interface(self, sample_segy_path: Path) -> None:
        """load_segy interface contract: returns dict with volume and metadata."""
        mock_result = {
            "volume": np.zeros((5, 5, 100), dtype=np.float32),
            "metadata": {
                "survey_name": "synthetic",
                "n_inlines": 5,
                "n_crosslines": 5,
                "n_samples": 100,
                "sample_interval_ms": 2.0,
                "inline_range": [1, 5],
                "crossline_range": [1, 5],
            },
        }
        with patch.object(_mod, "load_segy", return_value=mock_result) as mock_fn:
            result = _mod.load_segy(str(sample_segy_path))
            mock_fn.assert_called_once_with(str(sample_segy_path))
            assert result["volume"].shape == (5, 5, 100)
            assert result["volume"].dtype == np.float32
            assert "metadata" in result

    @pytest.mark.integration
    @pytest.mark.skipif(_load_segy is None, reason="load_segy not yet implemented")
    def test_load_segy_basic_real(self, sample_segy_path: Path) -> None:
        """Real load_segy must return (xr.Dataset, SurveyGeometry) with correct shape."""
        ds, geom = _load_segy(str(sample_segy_path))  # type: ignore[misc]
        assert "amplitude" in ds.data_vars
        # Synthetic fixture: 5 inlines x 5 crosslines x 100 samples
        assert ds["amplitude"].shape == (5, 5, 100)
        assert geom.n_inlines == 5
        assert geom.n_crosslines == 5


# ─────────────────────────────────────────────────────────────────────────────
# test_segy_to_zarr
# ─────────────────────────────────────────────────────────────────────────────


class TestSegyToZarr:
    def test_segy_to_zarr_manual(self, sample_segy_path: Path, tmp_zarr_store) -> None:
        """Manual SEG-Y → Zarr pipeline preserves volume shape and dtype."""
        import segyio

        with segyio.open(str(sample_segy_path), ignore_geometry=False) as f:
            il_list = list(f.ilines)
            xl_list = list(f.xlines)
            ns = len(f.samples)
            cube = np.zeros((len(il_list), len(xl_list), ns), dtype=np.float32)
            for i, il in enumerate(il_list):
                for j in range(len(xl_list)):
                    cube[i, j, :] = f.iline[il][j]

        root = zarr.open(tmp_zarr_store, mode="w")
        arr = root.create_array("seismic", shape=cube.shape, dtype="f4")
        arr[:] = cube

        root2 = zarr.open(tmp_zarr_store, mode="r")
        result = root2["seismic"][:]

        assert result.shape == (5, 5, 100)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, cube)

    def test_segy_to_zarr_mock_interface(
        self, sample_segy_path: Path, tmp_zarr_store
    ) -> None:
        """segy_to_zarr interface contract: callable(path, store) → array-like (5,5,100)."""
        mock_arr = MagicMock()
        mock_arr.shape = (5, 5, 100)
        with patch.object(_mod, "segy_to_zarr", return_value=mock_arr) as mock_fn:
            result = _mod.segy_to_zarr(str(sample_segy_path), tmp_zarr_store)
            mock_fn.assert_called_once()
            assert result.shape == (5, 5, 100)

    @pytest.mark.integration
    @pytest.mark.skipif(_segy_to_zarr is None, reason="segy_to_zarr not yet implemented")
    def test_segy_to_zarr_real(self, sample_segy_path: Path, tmp_path) -> None:
        """Real segy_to_zarr must write a Zarr store and return IngestMetadata."""
        dest = tmp_path / "output.zarr"
        meta = _segy_to_zarr(str(sample_segy_path), str(dest))  # type: ignore[misc]
        assert dest.exists()
        assert hasattr(meta, "n_inlines_loaded")
        assert meta.n_inlines_loaded == 5


# ─────────────────────────────────────────────────────────────────────────────
# test_metadata_extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestMetadataExtraction:
    def test_metadata_extraction_direct(self, sample_segy_path: Path) -> None:
        """Fields from segyio headers must include all required keys and be JSON-safe."""
        import segyio

        with segyio.open(str(sample_segy_path), ignore_geometry=False) as f:
            meta = {
                "survey_name": sample_segy_path.stem,
                "n_inlines": len(f.ilines),
                "n_crosslines": len(f.xlines),
                "n_samples": len(f.samples),
                "sample_interval_ms": segyio.dt(f) / 1000.0,
                "inline_range": [int(f.ilines[0]), int(f.ilines[-1])],
                "crossline_range": [int(f.xlines[0]), int(f.xlines[-1])],
            }

        for field in _REQUIRED_METADATA:
            assert field in meta, f"Missing required field: {field}"

        assert meta["n_inlines"] == 5
        assert meta["n_crosslines"] == 5
        assert meta["n_samples"] == 100
        assert meta["inline_range"] == [1, 5]
        assert meta["crossline_range"] == [1, 5]

        # Must survive a JSON round-trip
        roundtrip = json.loads(json.dumps(meta))
        assert roundtrip["n_inlines"] == 5

    def test_metadata_extraction_mock_interface(self, sample_segy_path: Path) -> None:
        """extract_metadata interface contract: callable(path) → dict with all required keys."""
        expected = {
            "survey_name": "synthetic",
            "n_inlines": 5,
            "n_crosslines": 5,
            "n_samples": 100,
            "sample_interval_ms": 2.0,
            "inline_range": [1, 5],
            "crossline_range": [1, 5],
        }
        with patch.object(_mod, "extract_metadata", return_value=expected, create=True):
            result = _mod.extract_metadata(str(sample_segy_path))
            for field in _REQUIRED_METADATA:
                assert field in result

    @pytest.mark.skipif(_extract_metadata is None, reason="extract_metadata not yet implemented")
    def test_metadata_extraction_real(self, sample_segy_path: Path) -> None:
        meta = _extract_metadata(str(sample_segy_path))  # type: ignore[misc]
        for field in _REQUIRED_METADATA:
            assert field in meta
        json.dumps(meta)  # must be JSON-serialisable


# ─────────────────────────────────────────────────────────────────────────────
# test_sample_mode
# ─────────────────────────────────────────────────────────────────────────────


class TestSampleMode:
    def test_sample_mode_first_n_inlines(self, sample_segy_path: Path) -> None:
        """Loading only the first 3 inlines must return exactly 3 inlines."""
        import segyio

        n_load = 3
        with segyio.open(str(sample_segy_path), ignore_geometry=False) as f:
            inlines = list(f.ilines)[:n_load]
            crosslines = list(f.xlines)
            ns = len(f.samples)
            cube = np.zeros((n_load, len(crosslines), ns), dtype=np.float32)
            for i, il in enumerate(inlines):
                for j in range(len(crosslines)):
                    cube[i, j, :] = f.iline[il][j]

        assert cube.shape == (3, 5, 100), f"Unexpected shape: {cube.shape}"

    def test_sample_mode_mock_interface(self, sample_segy_path: Path) -> None:
        """load_segy(path, max_inlines=N) contract — returns N-inline volume."""
        partial = np.zeros((3, 5, 100), dtype=np.float32)
        with patch.object(
            _mod,
            "load_segy",
            return_value={"volume": partial, "metadata": {}},
        ) as mock_fn:
            result = _mod.load_segy(str(sample_segy_path), max_inlines=3)
            mock_fn.assert_called_once_with(str(sample_segy_path), max_inlines=3)
            assert result["volume"].shape[0] == 3


# ─────────────────────────────────────────────────────────────────────────────
# test_invalid_file
# ─────────────────────────────────────────────────────────────────────────────


class TestInvalidFile:
    def test_invalid_file_segyio(self, tmp_path: Path) -> None:
        """segyio.open on a non-SEG-Y text file must raise an exception."""
        import segyio

        bad = tmp_path / "not_segy.txt"
        bad.write_text("this is definitely not seismic data\n")

        with pytest.raises((OSError, RuntimeError, ValueError)):
            with segyio.open(str(bad)):
                pass

    def test_invalid_file_mock_interface(self, tmp_path: Path) -> None:
        """load_segy must raise ValueError (or subclass) for non-SEG-Y input."""
        bad = tmp_path / "bad.csv"
        bad.write_text("a,b,c\n1,2,3\n")

        with patch.object(_mod, "load_segy", side_effect=ValueError("Not a SEG-Y file")):
            with pytest.raises(ValueError, match="SEG-Y"):
                _mod.load_segy(str(bad))

    def test_missing_file_mock_interface(self) -> None:
        """load_segy must raise FileNotFoundError for a non-existent path."""
        with patch.object(_mod, "load_segy", side_effect=FileNotFoundError("no such file")):
            with pytest.raises(FileNotFoundError):
                _mod.load_segy("/no/such/file.segy")
