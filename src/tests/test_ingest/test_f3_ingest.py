"""F3 cross-survey ingest pipeline (issue #24).

Validates the F3 format-proxy generators and the co-registration invariant the
cross-survey training depends on: the rasterised fault-label volume must share
the *exact* grid (shape + inline/crossline bases) as the amplitude volume
produced from the same SEG-Y. These are unit-scale checks on tiny synthetic
proxies — they do NOT use or assert anything about real F3 data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from deepseismic.ingest.label_generator import FaultMaskGenerator
from deepseismic.ingest.segy_loader import segy_to_zarr
from deepseismic.validation import load_volve_fault_sticks

pytest.importorskip("segyio", reason="segyio required for F3 SEG-Y proxy ingest")

from scripts.download_f3 import (  # noqa: E402
    create_proxy_fault_sticks,
    create_proxy_segy,
)

# Tiny dimensions keep the SEG-Y trace loop fast (<1s).
_N_IL, _N_XL, _N_S = 20, 30, 40
_IL_BASE, _XL_BASE = 100, 300  # F3 geometry bases used by the proxy


@pytest.fixture(scope="module")
def f3_proxy(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("f3_proxy")
    segy_path = root / "raw" / "f3_proxy.segy"
    stick_dir = root / "interp" / "fault_sticks"
    amp_zarr = root / "staged" / "amplitude.zarr"
    label_zarr = root / "staged" / "fault_label.zarr"

    create_proxy_segy(
        segy_path,
        n_inlines=_N_IL, n_crosslines=_N_XL, n_samples=_N_S,
    )
    create_proxy_fault_sticks(
        stick_dir,
        n_inlines=_N_IL, n_crosslines=_N_XL, n_samples=_N_S,
    )
    meta = segy_to_zarr(segy_path, amp_zarr, survey_id="f3-proxy-test", overwrite=True)

    return {
        "segy": segy_path,
        "stick_dir": stick_dir,
        "amp_zarr": amp_zarr,
        "label_zarr": label_zarr,
        "geom": meta.geometry,
    }


class TestF3ProxyGenerators:
    def test_segy_ingest_has_f3_geometry(self, f3_proxy: dict) -> None:
        g = f3_proxy["geom"]
        assert (g["n_inlines"], g["n_crosslines"], g["n_samples"]) == (_N_IL, _N_XL, _N_S)
        assert g["inline_min"] == _IL_BASE
        assert g["crossline_min"] == _XL_BASE
        assert g["sample_rate_ms"] == 4.0

    def test_fault_sticks_are_parseable_index_dat(self, f3_proxy: dict) -> None:
        dat_files = sorted(Path(f3_proxy["stick_dir"]).glob("*.dat"))
        assert dat_files, "proxy produced no .dat stick files"
        sticks = load_volve_fault_sticks(f3_proxy["stick_dir"])
        assert sticks, "no sticks parsed from proxy .dat files"
        pts = np.vstack(sticks)
        # All indices must be 0-based and inside the proxy grid.
        assert pts[:, 0].min() >= 0 and pts[:, 0].max() < _N_IL
        assert pts[:, 1].min() >= 0 and pts[:, 1].max() < _N_XL
        assert pts[:, 2].min() >= 0 and pts[:, 2].max() < _N_S

    def test_lower_frequency_than_volve_proxy(self) -> None:
        # F3 proxy default wavelet frequency must be < the Volve proxy's 35 Hz,
        # reflecting F3's older lower-bandwidth acquisition (cross-survey gap).
        import inspect

        from scripts.download_f3 import create_proxy_segy as f3_segy

        sig = inspect.signature(f3_segy)
        assert sig.parameters["freq_hz"].default < 35.0


class TestF3CoRegistration:
    """The label volume must share the amplitude volume's exact grid."""

    def _build_label(self, f3_proxy: dict) -> np.ndarray:
        g = f3_proxy["geom"]
        vol_shape = (g["n_inlines"], g["n_crosslines"], g["n_samples"])
        sticks = load_volve_fault_sticks(f3_proxy["stick_dir"])
        indexed = [[(float(r[0]), float(r[1]), float(r[2])) for r in s] for s in sticks]
        gen = FaultMaskGenerator(
            volume_shape=vol_shape,
            inline_range=(g["inline_min"], g["inline_max"], g["inline_step"]),
            crossline_range=(g["crossline_min"], g["crossline_max"], g["crossline_step"]),
            sample_rate_ms=g["sample_rate_ms"],
            datum_ms=g["datum_ms"],
            dilation_voxels=2,
        )
        gen.add_fault_sticks_in_index_space(indexed, interpolate_between=False)
        gen.to_zarr(f3_proxy["label_zarr"], overwrite=True)
        return gen.mask

    def test_label_grid_matches_amplitude_grid(self, f3_proxy: dict) -> None:
        mask = self._build_label(f3_proxy)
        amp = zarr.open_group(str(f3_proxy["amp_zarr"]), mode="r")["amplitude"]
        assert mask.shape == amp.shape == (_N_IL, _N_XL, _N_S)

    def test_label_has_nonzero_but_sparse_faults(self, f3_proxy: dict) -> None:
        mask = self._build_label(f3_proxy)
        frac = float(mask.mean())
        assert mask.sum() > 0, "co-registration produced an empty fault mask"
        assert frac < 0.20, f"fault fraction {frac:.3f} implausibly dense for a proxy"

    def test_label_zarr_array_contract(self, f3_proxy: dict) -> None:
        self._build_label(f3_proxy)
        root = zarr.open_group(str(f3_proxy["label_zarr"]), mode="r")
        arr = root["fault_mask"]
        assert arr.dtype == np.uint8
        assert arr.shape == (_N_IL, _N_XL, _N_S)
