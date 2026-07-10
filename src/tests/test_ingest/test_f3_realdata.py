"""Regression tests for F3 *real-data* ingest quirks (issue #31).

Two blockers surfaced when infra ran the documented ingest CLI against the real
F3 Demo 2023 SEG-Y in an in-VNet VM:

Blocker 1 — irregular SEG-Y geometry
    Real F3 has ~434 irregular/edge traces, so ``ilines * xlines`` does not equal
    the trace count.  segyio raises ``ValueError("Invalid dimensions ... should
    match the number of traces (N)")`` on a structured open.  The loader's
    irregular-grid fallback must trigger on that message (not only "inconsistent")
    and reconstruct a zero-filled rectangular volume from the true IL/XL headers.

Blocker 2 — F3 5-column map-coordinate fault export
    F3 fault sticks are headerless 5-column ASCII tables (X Y Z_ms stick_id
    point_id) that fit neither the Volve index-space parser nor the OpendTect
    block parser.  ``parse_f3_fault_sticks`` groups points by ``stick_id`` into
    world-coordinate :class:`FaultStick` objects.

These are unit-scale checks on tiny synthetic inputs — they do NOT use or assert
anything about the licensed real F3 data itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepseismic.ingest.label_generator import (
    FaultMaskGenerator,
    FaultPoint,
    SurveyTransform,
    parse_f3_fault_sticks,
)
from deepseismic.ingest.segy_loader import _is_irregular_geometry_error

# ---------------------------------------------------------------------------
# Blocker 1 — error predicate (pure unit, no segyio needed)
# ---------------------------------------------------------------------------

# Verbatim message segyio 1.9.14 emits on the real F3 irregular grid.
_REAL_F3_MSG = (
    "Invalid dimensions, ilines (631) * xlines (951) * offsets (1) "
    "should match the number of traces (600515)"
)


class TestIrregularGeometryPredicate:
    def test_matches_real_f3_invalid_dimensions_message(self) -> None:
        assert _is_irregular_geometry_error(ValueError(_REAL_F3_MSG)) is True

    def test_matches_should_match_number_of_traces_fragment(self) -> None:
        assert _is_irregular_geometry_error(
            ValueError("... should match the number of traces (42)")
        ) is True

    def test_matches_legacy_inconsistent_message(self) -> None:
        assert _is_irregular_geometry_error(ValueError("Inlines inconsistent")) is True

    def test_rejects_unrelated_valueerror(self) -> None:
        # A genuinely unrelated ValueError must still propagate (re-raised).
        unrelated = ValueError("could not convert string to float")
        assert _is_irregular_geometry_error(unrelated) is False


# ---------------------------------------------------------------------------
# Blocker 1 — irregular SEG-Y round-trip through the loader fallback
# ---------------------------------------------------------------------------

segyio = pytest.importorskip("segyio", reason="segyio required for SEG-Y ingest test")

_N_IL, _N_XL, _N_S = 5, 10, 20
_IL0, _XL0 = 100, 300
# Interior cells removed so tracecount (46) != n_il*n_xl (50), while the grid
# corners are retained so IL/XL min-max still span the full 5x10 extent.
_DROPPED = {(101, 302), (102, 305), (103, 301), (101, 308)}


def _make_irregular_segy(dest: Path) -> list[tuple[int, int]]:
    """Write a small SEG-Y whose trace count != n_inlines * n_crosslines."""
    pairs = [
        (il, xl)
        for il in range(_IL0, _IL0 + _N_IL)
        for xl in range(_XL0, _XL0 + _N_XL)
        if (il, xl) not in _DROPPED
    ]
    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
    spec.samples = np.arange(_N_S, dtype=np.float32) * 4.0
    spec.tracecount = len(pairs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with segyio.create(str(dest), spec) as f:
        f.bin.update(hdt=4000, dto=4000)
        for tr, (il, xl) in enumerate(pairs):
            f.header[tr] = {
                segyio.TraceField.INLINE_3D: il,
                segyio.TraceField.CROSSLINE_3D: xl,
                segyio.TraceField.CDP_X: il * 100,
                segyio.TraceField.CDP_Y: xl * 100,
                segyio.TraceField.TRACE_SEQUENCE_FILE: tr + 1,
            }
            # Encode (il, xl) into the sample values so we can verify placement.
            f.trace[tr] = np.full(_N_S, float(il * 1000 + xl), dtype=np.float32)
    return pairs


class TestIrregularSegyLoad:
    def test_structured_open_raises_invalid_dimensions(self, tmp_path: Path) -> None:
        # Sanity: the synthetic file really does trip segyio's structured reader.
        segy = tmp_path / "irr.segy"
        _make_irregular_segy(segy)
        with pytest.raises(ValueError) as exc_info:
            with segyio.open(str(segy), ignore_geometry=False):
                pass
        assert _is_irregular_geometry_error(exc_info.value)

    def test_loader_falls_back_and_zero_fills(self, tmp_path: Path) -> None:
        from deepseismic.ingest.segy_loader import load_segy

        segy = tmp_path / "irr.segy"
        pairs = _make_irregular_segy(segy)

        ds, geom = load_segy(str(segy))
        vol = ds["amplitude"].values

        # Full rectangular grid recovered from the true IL/XL header extent.
        assert vol.shape == (_N_IL, _N_XL, _N_S)
        assert geom.n_inlines == _N_IL
        assert geom.n_crosslines == _N_XL

        present = set(pairs)
        for il_idx in range(_N_IL):
            for xl_idx in range(_N_XL):
                il = _IL0 + il_idx
                xl = _XL0 + xl_idx
                cell = vol[il_idx, xl_idx]
                if (il, xl) in present:
                    np.testing.assert_allclose(cell, float(il * 1000 + xl))
                else:
                    # Missing/edge traces are zero-filled, not dropped or shifted.
                    assert np.all(cell == 0.0), f"expected zero-fill at IL{il}/XL{xl}"

    def test_corner_points_written_from_cdp_headers(self, tmp_path: Path) -> None:
        from deepseismic.ingest.segy_loader import load_segy

        segy = tmp_path / "irr.segy"
        _make_irregular_segy(segy)
        _, geom = load_segy(str(segy))
        assert geom.corner_points is not None
        assert len(geom.corner_points) == 3
        # CDP_X = il*100, CDP_Y = xl*100 (no coordinate scalar) round-trips.
        for x, y, il, xl in geom.corner_points:
            assert x == pytest.approx(il * 100)
            assert y == pytest.approx(xl * 100)


# ---------------------------------------------------------------------------
# Blocker 2 — F3 5-column fault-stick parser
# ---------------------------------------------------------------------------

# Two sticks (ids 0 and 1); real F3 verbatim sample values for stick 0.
_F3_SAMPLE = """\
624690.0625  6074133   294.465  0 0
624643.8125  6074132   368.093  0 1
624597.875   6074130.5 443.721  0 2
624536.0625  6074129   509.341  0 3
624505.1875  6074528   291.795  1 0
624470.0     6074530   360.000  1 1
624440.0     6074532   430.000  1 2
"""


class TestParseF3FaultSticks:
    def test_groups_by_stick_id(self, tmp_path: Path) -> None:
        path = tmp_path / "FaultA.txt"
        path.write_text(_F3_SAMPLE, encoding="utf-8")
        sticks = parse_f3_fault_sticks(path)

        assert len(sticks) == 2
        assert [len(s) for s in sticks] == [4, 3]
        # Fault names carry the file stem + stick_id for cross-file uniqueness.
        assert sticks[0].fault_name == "FaultA_stick0"
        assert sticks[1].fault_name == "FaultA_stick1"

    def test_parses_world_xyz_values(self, tmp_path: Path) -> None:
        path = tmp_path / "FaultA.txt"
        path.write_text(_F3_SAMPLE, encoding="utf-8")
        sticks = parse_f3_fault_sticks(path)

        p0 = sticks[0].points[0]
        assert isinstance(p0, FaultPoint)
        assert p0.x == pytest.approx(624690.0625)
        assert p0.y == pytest.approx(6074133.0)
        assert p0.z_ms == pytest.approx(294.465)
        # Second stick anchored at a distinct Y band (6074528).
        assert sticks[1].points[0].y == pytest.approx(6074528.0)

    def test_orders_points_by_point_id(self, tmp_path: Path) -> None:
        # Shuffle point order in the file; parser must sort by point_id.
        shuffled = "\n".join(
            [
                "624536.0625  6074129   509.341  0 3",
                "624690.0625  6074133   294.465  0 0",
                "624597.875   6074130.5 443.721  0 2",
                "624643.8125  6074132   368.093  0 1",
            ]
        )
        path = tmp_path / "FaultB.txt"
        path.write_text(shuffled, encoding="utf-8")
        sticks = parse_f3_fault_sticks(path)
        assert len(sticks) == 1
        z_vals = [p.z_ms for p in sticks[0].points]
        assert z_vals == pytest.approx([294.465, 368.093, 443.721, 509.341])

    def test_skips_comments_blanks_and_degenerate_sticks(self, tmp_path: Path) -> None:
        content = (
            "# header comment\n"
            "\n"
            "624690.0625  6074133   294.465  0 0\n"
            "624643.8125  6074132   368.093  0 1\n"
            "999999.0     6074999   100.0    5 0\n"  # single-point stick -> dropped
        )
        path = tmp_path / "FaultC.txt"
        path.write_text(content, encoding="utf-8")
        sticks = parse_f3_fault_sticks(path)
        assert len(sticks) == 1
        assert len(sticks[0]) == 2

    def test_rasterises_via_world_transform(self, tmp_path: Path) -> None:
        path = tmp_path / "FaultA.txt"
        path.write_text(_F3_SAMPLE, encoding="utf-8")
        sticks = parse_f3_fault_sticks(path)

        # Synthetic transform: identity-ish mapping placing the sticks inside a
        # small volume.  Solve from three tie-points spanning the XY range.
        transform = SurveyTransform.from_three_points(
            tie_points=[
                (624400.0, 6074100.0, 0, 0),
                (624700.0, 6074100.0, 30, 0),
                (624400.0, 6074600.0, 0, 50),
            ],
            sample_rate_ms=4.0,
            datum_ms=0.0,
        )
        gen = FaultMaskGenerator(
            volume_shape=(40, 60, 200),
            inline_range=(0, 39, 1),
            crossline_range=(0, 59, 1),
            sample_rate_ms=4.0,
            datum_ms=0.0,
            dilation_voxels=1,
        )
        gen.add_fault_sticks(sticks, transform)
        assert gen.mask.sum() > 0, "world-coordinate rasterisation produced empty mask"
        assert gen.mask.dtype == np.uint8
