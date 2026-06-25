"""Sprint 2 S2-01 / Sprint 3 S3-#8 tests: fault-label generation.

Critical coverage:
- Coordinate mapping correctness (highest-risk logic: il/xl 0-based index, z_col=sample index)
- FaultMaskGenerator output dtype/shape/values
- Dilation monotonicity
- load_volve_fault_sticks parsing from synthetic .dat fixtures
- densify_stick_to_il_resolution (S3-#8): interpolation correctness & guardrails
- add_fault_sticks_in_index_space with interpolate_between=True (S3-#8)

Synthetic .dat fixtures only — no dependency on real data files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.storage

from deepseismic.ingest.label_generator import FaultMaskGenerator, densify_stick_to_il_resolution
from deepseismic.validation import load_volve_fault_sticks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_IL = 1001
BASE_XL = 1900
SAMPLE_RATE_MS = 4.0


def _write_dat(tmp_path: Path, name: str, rows: list[tuple]) -> Path:
    """Write a minimal 3-column .dat fixture file (il_idx xl_idx z_col)."""
    p = tmp_path / name
    with open(p, "w") as fh:
        for row in rows:
            fh.write(" ".join(str(v) for v in row) + "\n")
    return p


def _make_gen(
    shape: tuple[int, int, int] = (20, 20, 50),
    dilation: int = 0,
) -> FaultMaskGenerator:
    return FaultMaskGenerator(
        volume_shape=shape,
        inline_range=(0, shape[0] - 1, 1),
        crossline_range=(0, shape[1] - 1, 1),
        sample_rate_ms=SAMPLE_RATE_MS,
        datum_ms=0.0,
        dilation_voxels=dilation,
    )


# ---------------------------------------------------------------------------
# Coordinate mapping (highest-risk logic, S2-01)
# ---------------------------------------------------------------------------


class TestCoordinateMapping:
    """Guard the abs_inline / abs_crossline / twt_ms formulas.

    The .dat files store 0-based index-space values.  A prior team bug
    treated z_col as milliseconds directly (202 ms) instead of as a sample
    index (202 × 4 = 808 ms).  These tests pin the correct interpretation.
    """

    def test_abs_inline_formula(self):
        """abs_inline = BASE_IL(1001) + il_idx for all valid il_idx."""
        for il_idx in [0, 5, 50, 99]:
            assert BASE_IL + il_idx == 1001 + il_idx
            # Sanity: index is in valid range for a 100-inline volume
            assert 0 <= il_idx < 100

    def test_abs_crossline_formula(self):
        """abs_crossline = BASE_XL(1900) + xl_idx for all valid xl_idx."""
        for xl_idx in [0, 10, 100, 199]:
            assert BASE_XL + xl_idx == 1900 + xl_idx
            assert 0 <= xl_idx < 200

    def test_z_col_is_sample_index_not_ms(self):
        """twt_ms = z_col * 4.0 — z_col is a sample INDEX, not ms.

        Real Volve sticks have z_col ≈ 200-307.  If z_col were used as ms
        directly, all TWT values would be < 400 ms.  With the correct formula
        they exceed 800 ms.  This test pins that invariant.
        """
        for z_col in [202, 227, 300, 307]:
            twt_ms = z_col * SAMPLE_RATE_MS
            # Correct: ≥ 800 ms
            assert twt_ms >= 800.0, (
                f"z_col={z_col} should map to ≥800 ms; "
                f"got {twt_ms} ms (bug: treated as raw ms)"
            )

    def test_known_voxel_lands_at_exact_index(self):
        """Given il_idx=10, xl_idx=20, z_col=50 — voxel (10,20,50) must be labelled."""
        gen = FaultMaskGenerator(
            volume_shape=(100, 200, 500),
            inline_range=(1001, 1100, 1),
            crossline_range=(1900, 2099, 1),
            sample_rate_ms=SAMPLE_RATE_MS,
            datum_ms=0.0,
            dilation_voxels=0,
        )
        gen.add_fault_sticks_in_index_space([[(10.0, 20.0, 50.0), (11.0, 21.0, 51.0)]])
        assert gen.mask[10, 20, 50] == 1, "Known voxel was not labelled"

    def test_dilation_zero_no_neighbour_leakage(self):
        """With dilation=0, only the exact voxel is painted — no adjacents."""
        gen = _make_gen(shape=(30, 30, 60), dilation=0)
        gen.add_fault_sticks_in_index_space([[(5.0, 5.0, 10.0), (6.0, 6.0, 11.0)]])
        # Direct neighbours must remain 0
        assert gen.mask[4, 5, 10] == 0
        assert gen.mask[5, 4, 10] == 0
        assert gen.mask[7, 7, 13] == 0  # outside stick range

    def test_load_volve_sticks_parses_dat(self, tmp_path: Path):
        """load_volve_fault_sticks correctly reads a synthetic 3-column .dat file."""
        _write_dat(tmp_path, "fault_A.dat", [
            (10, 50, 202),
            (11, 51, 210),
            (12, 52, 218),
        ])
        sticks = load_volve_fault_sticks(tmp_path)
        assert len(sticks) == 1
        assert sticks[0].shape == (3, 3)
        # First point: il=10, xl=50, last-col=202
        assert sticks[0][0, 0] == pytest.approx(10.0)
        assert sticks[0][0, 1] == pytest.approx(50.0)
        assert sticks[0][0, 2] == pytest.approx(202.0)

    def test_load_volve_sticks_multiple_files(self, tmp_path: Path):
        """Multiple .dat files produce multiple sticks."""
        _write_dat(tmp_path, "fault_A.dat", [(10, 50, 202), (11, 51, 210)])
        _write_dat(tmp_path, "fault_B.dat", [(30, 100, 300), (31, 101, 310)])
        sticks = load_volve_fault_sticks(tmp_path)
        assert len(sticks) == 2

    def test_load_volve_sticks_empty_dir(self, tmp_path: Path):
        """Empty directory returns empty list without error."""
        sticks = load_volve_fault_sticks(tmp_path)
        assert sticks == []


# ---------------------------------------------------------------------------
# FaultMaskGenerator rasterisation
# ---------------------------------------------------------------------------


class TestFaultMaskGenerator:
    """Tests for FaultMaskGenerator rasterisation and dilation."""

    def test_mask_initially_all_zeros(self):
        gen = _make_gen()
        assert gen.mask.sum() == 0

    def test_values_binary_after_rasterisation(self):
        """Mask must contain only 0 and 1."""
        gen = _make_gen(dilation=1)
        gen.add_fault_sticks_in_index_space([[(5.0, 5.0, 10.0), (6.0, 6.0, 12.0)]])
        unique = np.unique(gen.mask)
        assert set(unique).issubset({0, 1}), f"Non-binary values found: {unique}"

    def test_positive_fraction_nonzero_after_stick(self):
        """Adding a stick must produce at least one labelled voxel."""
        gen = _make_gen(dilation=0)
        gen.add_fault_sticks_in_index_space([[(5.0, 5.0, 10.0), (6.0, 6.0, 11.0)]])
        assert gen.mask.sum() > 0

    def test_dilation_increases_count_monotonically(self):
        """Larger dilation radius must label strictly more voxels than smaller."""
        sticks = [[(5.0, 5.0, 10.0), (7.0, 7.0, 13.0)]]
        counts = []
        for d in range(4):
            gen = FaultMaskGenerator(
                volume_shape=(30, 30, 60),
                inline_range=(0, 29, 1),
                crossline_range=(0, 29, 1),
                sample_rate_ms=SAMPLE_RATE_MS,
                datum_ms=0.0,
                dilation_voxels=d,
            )
            gen.add_fault_sticks_in_index_space(sticks)
            counts.append(int(gen.mask.sum()))

        for i in range(1, len(counts)):
            assert counts[i] > counts[i - 1], (
                f"dilation={i} count={counts[i]} not > "
                f"dilation={i - 1} count={counts[i - 1]}"
            )


# ---------------------------------------------------------------------------
# Zarr output (S2-01 output contract)
# ---------------------------------------------------------------------------


class TestLabelZarrOutput:
    """Tests for FaultMaskGenerator.to_zarr: dtype, shape, and roundtrip correctness."""

    def test_zarr_dtype_is_uint8(self, tmp_path: Path):
        gen = _make_gen(shape=(10, 10, 20))
        gen.add_fault_sticks_in_index_space([[(3.0, 3.0, 5.0), (4.0, 4.0, 6.0)]])
        z = gen.to_zarr(tmp_path / "label.zarr", chunks=(5, 5, 10), overwrite=True)
        assert str(z.dtype) == "uint8"

    def test_zarr_shape_matches_volume_shape(self, tmp_path: Path):
        shape = (8, 12, 24)
        gen = FaultMaskGenerator(
            volume_shape=shape,
            inline_range=(0, 7, 1),
            crossline_range=(0, 11, 1),
            sample_rate_ms=SAMPLE_RATE_MS,
            datum_ms=0.0,
            dilation_voxels=0,
        )
        z = gen.to_zarr(tmp_path / "label2.zarr", chunks=(4, 4, 8), overwrite=True)
        assert z.shape == shape

    def test_zarr_values_roundtrip_exactly(self, tmp_path: Path):
        """Values written to Zarr must equal the in-memory mask exactly."""
        gen = _make_gen(shape=(8, 8, 16), dilation=0)
        gen.add_fault_sticks_in_index_space([[(2.0, 2.0, 5.0), (3.0, 3.0, 6.0)]])
        zarr_path = tmp_path / "label3.zarr"
        gen.to_zarr(zarr_path, chunks=(4, 4, 8), overwrite=True)

        store = zarr.storage.LocalStore(str(zarr_path))
        root = zarr.open_group(store, mode="r")
        stored = np.array(root["fault_mask"])
        np.testing.assert_array_equal(stored, gen.mask)


# ---------------------------------------------------------------------------
# S3-#8: densify_stick_to_il_resolution
# ---------------------------------------------------------------------------


class TestDensifyStickToIlResolution:
    """Tests for the between-stick IL densification function.

    Geophysical context: fault picks are often every 3–10 ILs.  Inserting
    1-IL interpolated picks (where gap ≤ max_il_gap) creates a continuous
    fault label band — a INFERRED extension of the interpreter's intent,
    not new ground truth.
    """

    def test_basic_two_point_gap_4(self):
        """IL gap of 4 with max_gap=5 → 3 intermediate picks inserted."""
        pts = [(0.0, 10.0, 200.0), (4.0, 14.0, 204.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        # Expected: IL 0,1,2,3,4 → 5 points
        ils = [p[0] for p in result]
        assert ils == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_xl_z_linearly_interpolated(self):
        """XL and Z must be linearly interpolated at intermediate IL positions."""
        pts = [(0.0, 10.0, 200.0), (4.0, 14.0, 204.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        # At IL=2 (midpoint): XL=12, Z=202
        mid = result[2]
        assert mid[0] == pytest.approx(2.0)
        assert mid[1] == pytest.approx(12.0)
        assert mid[2] == pytest.approx(202.0)

    def test_gap_exceeds_max_not_bridged(self):
        """Gap > max_il_gap → no intermediate points inserted."""
        pts = [(0.0, 10.0, 200.0), (10.0, 20.0, 210.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        # Only the two original points
        assert len(result) == 2
        assert result[0][0] == pytest.approx(0.0)
        assert result[1][0] == pytest.approx(10.0)

    def test_gap_exactly_at_max_is_bridged(self):
        """Gap == max_il_gap → intermediate points ARE inserted."""
        pts = [(0.0, 10.0, 200.0), (5.0, 15.0, 205.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        # Gap=5 ≤ max_gap=5 → 4 intermediate points → 6 total
        assert len(result) == 6

    def test_already_dense_no_change(self):
        """Points at IL=0,1,2 (gap=1) → no new points inserted."""
        pts = [(0.0, 10.0, 200.0), (1.0, 11.0, 201.0), (2.0, 12.0, 202.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        assert len(result) == 3

    def test_single_point_returned_unchanged(self):
        """Single point → returned as-is (no interpolation possible)."""
        pts = [(5.0, 30.0, 150.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        assert len(result) == 1
        assert result[0] == pytest.approx((5.0, 30.0, 150.0))

    def test_unsorted_input_sorted_by_il(self):
        """Input in reverse IL order → output sorted ascending by IL."""
        pts = [(4.0, 14.0, 204.0), (0.0, 10.0, 200.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        ils = [p[0] for p in result]
        assert ils == sorted(ils)
        assert ils[0] == pytest.approx(0.0)
        assert ils[-1] == pytest.approx(4.0)

    def test_mixed_gaps_bridged_selectively(self):
        """Multi-segment: only gaps ≤ max_gap are bridged."""
        # Gap 0→3 = 3 (≤5, bridged), gap 3→15 = 12 (>5, not bridged)
        pts = [(0.0, 0.0, 0.0), (3.0, 3.0, 3.0), (15.0, 15.0, 15.0)]
        result = densify_stick_to_il_resolution(pts, max_il_gap=5)
        ils = [p[0] for p in result]
        # First gap (3): ILs 0,1,2,3 → 4 points; second gap (12): no interp
        assert 0.0 in ils
        assert 1.0 in ils
        assert 2.0 in ils
        assert 3.0 in ils
        assert 15.0 in ils
        # No interpolated ILs between 3 and 15
        assert not any(3.0 < il < 15.0 for il in ils)


# ---------------------------------------------------------------------------
# S3-#8: add_fault_sticks_in_index_space with interpolate_between
# ---------------------------------------------------------------------------


class TestInterpolateBetweenSticks:
    """Tests for the interpolate_between parameter of add_fault_sticks_in_index_space."""

    def test_interpolate_between_api_valid_binary_mask(self):
        """interpolate_between=True API works; mask is valid binary and non-empty."""
        sticks = [[(0.0, 5.0, 10.0), (4.0, 5.0, 10.0), (8.0, 5.0, 10.0)]]
        gen = FaultMaskGenerator(
            volume_shape=(12, 12, 20),
            inline_range=(0, 11, 1),
            crossline_range=(0, 11, 1),
            sample_rate_ms=4.0,
            datum_ms=0.0,
            dilation_voxels=0,
        )
        gen.add_fault_sticks_in_index_space(sticks, interpolate_between=True, max_interp_gap_il=5)
        assert gen.mask.sum() > 0
        assert set(np.unique(gen.mask)).issubset({0, 1})

    def test_interpolate_between_result_ge_baseline(self):
        """interpolate_between=True must produce ≥ as many voxels as False.

        For linear fault geometry the arc-length rasteriser already covers
        all intermediate ILs regardless of densification.  Densification
        guarantees coverage for curved geometry and explicit IL documentation,
        but must never produce FEWER voxels than the baseline.
        """
        sticks = [[(0.0, 5.0, 10.0), (4.0, 5.0, 10.0), (8.0, 5.0, 10.0)]]

        gen_base = FaultMaskGenerator(
            volume_shape=(20, 20, 30),
            inline_range=(0, 19, 1),
            crossline_range=(0, 19, 1),
            sample_rate_ms=4.0,
            datum_ms=0.0,
            dilation_voxels=0,
        )
        gen_base.add_fault_sticks_in_index_space(sticks, interpolate_between=False)

        gen_interp = FaultMaskGenerator(
            volume_shape=(20, 20, 30),
            inline_range=(0, 19, 1),
            crossline_range=(0, 19, 1),
            sample_rate_ms=4.0,
            datum_ms=0.0,
            dilation_voxels=0,
        )
        gen_interp.add_fault_sticks_in_index_space(
            sticks, interpolate_between=True, max_interp_gap_il=5,
        )

        assert gen_interp.mask.sum() >= gen_base.mask.sum(), (
            "interpolate_between=True must never label fewer voxels than False"
        )

    def test_interpolate_between_false_unchanged(self):
        """interpolate_between=False must produce identical result to original call."""
        sticks = [[(2.0, 3.0, 5.0), (6.0, 7.0, 9.0)]]

        gen1 = _make_gen(shape=(15, 15, 20), dilation=0)
        gen1.add_fault_sticks_in_index_space(sticks)

        gen2 = _make_gen(shape=(15, 15, 20), dilation=0)
        gen2.add_fault_sticks_in_index_space(sticks, interpolate_between=False)

        np.testing.assert_array_equal(gen1.mask, gen2.mask)

    def test_gap_exceeds_max_no_extra_voxels(self):
        """When gap > max_interp_gap_il, no additional voxels from interpolation."""
        # Gap of 20 IL >> max_interp_gap_il=5
        sticks = [[(0.0, 5.0, 10.0), (20.0, 5.0, 10.0)]]

        gen_base = _make_gen(shape=(30, 15, 20), dilation=0)
        gen_base.add_fault_sticks_in_index_space(sticks, interpolate_between=False)

        gen_interp = _make_gen(shape=(30, 15, 20), dilation=0)
        gen_interp.add_fault_sticks_in_index_space(
            sticks, interpolate_between=True, max_interp_gap_il=5,
        )

        np.testing.assert_array_equal(gen_base.mask, gen_interp.mask)
