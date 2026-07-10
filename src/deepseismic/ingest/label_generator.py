"""Fault label generator for the deepseismic2 PoC.

Converts Volve fault interpretation files (Petrel fault-stick exports or
OpendTect ASCII exports) into a binary fault-mask volume stored as Zarr.

Pipeline
--------
1. **Parse**  — read fault-stick text files → list of :class:`FaultStick`.
2. **Transform** — map world XYZ coordinates to (inline, crossline, sample)
   index space using an affine :class:`SurveyTransform`.
3. **Rasterise** — interpolate each stick polyline and paint it into a
   ``uint8`` volume, optionally dilated by *N* voxels to account for
   stick-spacing uncertainty.

Partial labelling is inherently supported: only annotated fault sticks
contribute to the mask.  Unannotated regions remain 0 (unknown / background).

Output
------
A ``uint8`` Zarr array of shape ``(n_inlines, n_crosslines, n_samples)``
where ``1 = fault`` and ``0 = background / unlabelled``.

Usage
-----
    from deepseismic.ingest.label_generator import (
        parse_petrel_fault_sticks,
        FaultMaskGenerator,
        SurveyTransform,
    )
    from deepseismic.ingest.segy_loader import load_segy

    # SEG-Y path is supplied by the caller — not hard-coded here.
    # For local dev use the synthetic proxy; for real data supply --source arg.
    segy_path = "path/to/your/survey.segy"   # e.g. ST10010_PSDM_TIME.segy
    _, geom = load_segy(segy_path, sample_mode=True)

    # Build transform from three known tie-points (x, y, inline, crossline)
    transform = SurveyTransform.from_three_points(
        tie_points=[...],
        sample_rate_ms=geom.sample_rate_ms,
        datum_ms=geom.datum_ms,
    )

    sticks = parse_petrel_fault_sticks("data/raw/volve_faults.txt")

    vol_shape = (geom.n_inlines, geom.n_crosslines, geom.n_samples)
    gen = FaultMaskGenerator(
        volume_shape=vol_shape,
        inline_range=(geom.inline_min, geom.inline_max, geom.inline_step),
        crossline_range=(geom.crossline_min, geom.crossline_max, geom.crossline_step),
        sample_rate_ms=geom.sample_rate_ms,
        datum_ms=geom.datum_ms,
        dilation_voxels=1,
    )
    gen.add_fault_sticks(sticks, transform)
    gen.to_zarr("data/staged/fault_mask.zarr", overwrite=True)
    print("Fault voxels:", gen.mask.sum())
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import zarr
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class FaultPoint(NamedTuple):
    """A single XYZ point on a fault stick (world / survey coordinates)."""

    x: float
    y: float
    z_ms: float   # Two-way time in milliseconds; use depth metres for depth volumes


@dataclass
class FaultStick:
    """An ordered polyline of :class:`FaultPoint` objects, named by fault."""

    fault_name: str
    points: list[FaultPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class SurveyTransform:
    """Affine mapping from world XY to inline/crossline survey-number space.

    The transform is defined by::

        inline     = il_origin + il_dx * (X - x0) + il_dy * (Y - y0)
        crossline  = xl_origin + xl_dx * (X - x0) + xl_dy * (Y - y0)

    Construct with :meth:`from_three_points` when three (X, Y, IL, XL) tie-
    points are available from trace headers.

    Parameters
    ----------
    x0, y0:
        World-coordinate origin (one known tie-point).
    il_dx, il_dy:
        Inline direction cosines (change in inline number per metre in X, Y).
    xl_dx, xl_dy:
        Crossline direction cosines.
    il_origin, xl_origin:
        Inline / crossline numbers at the origin ``(x0, y0)``.
    sample_rate_ms:
        Milliseconds per sample.
    datum_ms:
        Recording start time in milliseconds (``DelayRecordingTime``).
    """

    x0: float = 0.0
    y0: float = 0.0
    il_dx: float = 0.0
    il_dy: float = 0.0
    xl_dx: float = 0.0
    xl_dy: float = 0.0
    il_origin: float = 0.0
    xl_origin: float = 0.0
    sample_rate_ms: float = 4.0
    datum_ms: float = 0.0

    def world_to_ilxl(self, x: float, y: float) -> tuple[float, float]:
        """Convert world XY → (inline_number, crossline_number) as floats."""
        dx, dy = x - self.x0, y - self.y0
        il = self.il_origin + self.il_dx * dx + self.il_dy * dy
        xl = self.xl_origin + self.xl_dx * dx + self.xl_dy * dy
        return il, xl

    def z_to_sample(self, z_ms: float) -> float:
        """Convert two-way time (ms) → sample index as float."""
        return (z_ms - self.datum_ms) / self.sample_rate_ms

    @classmethod
    def from_three_points(
        cls,
        tie_points: list[tuple[float, float, int, int]],
        sample_rate_ms: float = 4.0,
        datum_ms: float = 0.0,
    ) -> SurveyTransform:
        """Construct from three (x, y, inline, crossline) tie-points.

        Solves the 2 × 2 linear system that maps (ΔX, ΔY) to (ΔIL, ΔXL).

        Parameters
        ----------
        tie_points:
            At least three ``(x, y, inline, crossline)`` tuples.
        """
        if len(tie_points) < 3:
            raise ValueError("Need at least 3 tie-points to determine the affine transform.")

        (x0, y0, il0, xl0), (x1, y1, il1, xl1), (x2, y2, il2, xl2) = tie_points[:3]

        A = np.array([[x1 - x0, y1 - y0], [x2 - x0, y2 - y0]], dtype=np.float64)
        b_il = np.array([il1 - il0, il2 - il0], dtype=np.float64)
        b_xl = np.array([xl1 - xl0, xl2 - xl0], dtype=np.float64)

        il_vec = np.linalg.solve(A, b_il)
        xl_vec = np.linalg.solve(A, b_xl)

        return cls(
            x0=x0, y0=y0,
            il_dx=float(il_vec[0]), il_dy=float(il_vec[1]),
            xl_dx=float(xl_vec[0]), xl_dy=float(xl_vec[1]),
            il_origin=float(il0), xl_origin=float(xl0),
            sample_rate_ms=sample_rate_ms,
            datum_ms=datum_ms,
        )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_petrel_fault_sticks(
    path: str | Path,
    z_unit: str = "ms",
) -> list[FaultStick]:
    """Parse a Petrel-exported fault stick text file.

    Expected format (whitespace-delimited, comment lines start with ``#``)::

        # Optional header
        FaultName  X  Y  Z
        FaultName  X  Y  Z
        ...

    A new :class:`FaultStick` is started whenever the fault name changes or a
    line with a single ``FAULT`` token is encountered (alternate Petrel style).

    Parameters
    ----------
    path:
        Path to the fault stick export file.
    z_unit:
        Unit of the Z column.  ``'ms'`` (two-way time) or ``'m'`` (depth).
        The Z value is stored as-is; callers are responsible for unit
        consistency with the seismic volume.

    Returns
    -------
    list[FaultStick]
    """
    path = Path(path)
    sticks: list[FaultStick] = []
    current_stick: FaultStick | None = None
    current_fault_name = "UNKNOWN"

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # Comment / header lines
            if line.startswith(("#", "!", "/")):
                continue

            # Petrel style: line containing only "FAULT FaultName"
            if re.match(r"^FAULT\s+\S+$", line, re.IGNORECASE):
                if current_stick and len(current_stick) >= 2:
                    sticks.append(current_stick)
                current_fault_name = line.split()[-1]
                current_stick = FaultStick(fault_name=current_fault_name)
                continue

            parts = line.split()

            if len(parts) == 4:
                # Format: FaultName  X  Y  Z
                try:
                    name = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                if current_stick is None or current_stick.fault_name != name:
                    if current_stick and len(current_stick) >= 2:
                        sticks.append(current_stick)
                    current_stick = FaultStick(fault_name=name)
                current_stick.points.append(FaultPoint(x, y, z))

            elif len(parts) == 3:
                # Continuation line without fault name
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                if current_stick is None:
                    current_stick = FaultStick(fault_name=current_fault_name)
                current_stick.points.append(FaultPoint(x, y, z))

    if current_stick and len(current_stick) >= 2:
        sticks.append(current_stick)

    logger.info("Parsed %d fault sticks from %s", len(sticks), path.name)
    return sticks


def densify_stick_to_il_resolution(
    pts: list[tuple[float, float, float]],
    max_il_gap: int = 5,
) -> list[tuple[float, float, float]]:
    """Insert 1-IL-resolution interpolated picks between sparse fault picks.

    Geophysical assumption
    ----------------------
    Fault geometry is approximately planar between adjacent interpreted sticks.
    Linearly interpolating XL and Z at each intermediate IL position is valid
    when the inline gap is small (≤ ``max_il_gap``).  Larger gaps may represent
    fault segmentation, an interpretation gap, or a data break — these are NOT
    bridged, preserving the interpreter's intent.

    Resolution guardrail
    --------------------
    λ/4 at 36.6 Hz with v = 2 000 m/s ≈ 13.7 m ≈ 3.4 samples (4 ms/sample).
    Rasterising at 1-IL steps ensures the label band is never thinner than the
    minimum resolvable feature.  Dilation (default 3 voxels ≈ 12 ms ≈ ~24 m)
    adds positional uncertainty appropriate for sparse stick spacing.

    Uncertainty note
    ----------------
    Interpolated picks are INFERRED labels, not interpreter-picked ground truth.
    For an original step-5 fault, 4 out of every 5 painted ILs are interpolated
    (~80 % inferred).  Label confidence is proportionally lower at interpolated
    positions.  Quantify this in QC reports; never present interpolated labels
    as equivalent to direct picks.

    Parameters
    ----------
    pts:
        List of ``(il_idx, xl_idx, z_idx)`` tuples (0-based index space).
        Need not be sorted — the function sorts by IL internally.
    max_il_gap:
        Maximum inline gap (in IL index units) to bridge with interpolation.
        Default 5.  Gaps strictly greater than this value are left unconnected.

    Returns
    -------
    list[tuple[float, float, float]]
        Densified list sorted by IL, with 1-IL-step picks inserted in every
        gap whose width is in the range ``(1, max_il_gap]``.
    """
    if len(pts) < 2:
        return list(pts)

    sorted_pts = sorted(pts, key=lambda p: p[0])
    densified: list[tuple[float, float, float]] = []

    for i, (il0, xl0, z0) in enumerate(sorted_pts[:-1]):
        il1, xl1, z1 = sorted_pts[i + 1]
        il_gap = il1 - il0

        densified.append((il0, xl0, z0))

        if 1.0 < il_gap <= max_il_gap:
            n_steps = round(il_gap)
            for k in range(1, n_steps):
                t = k / n_steps
                densified.append((
                    il0 + t * (il1 - il0),
                    xl0 + t * (xl1 - xl0),
                    z0  + t * (z1  - z0),
                ))

    densified.append(sorted_pts[-1])
    return densified


def parse_opendtect_fault_sticks(path: str | Path) -> list[FaultStick]:
    """Parse an OpendTect ASCII fault export.

    OpendTect writes blocks separated by ``Fault: <name>`` headers::

        Fault: FAULT_A
        X  Y  Z  inline  crossline
        ...

    Only the X / Y / Z columns are used; the inline/crossline columns are
    ignored because we re-derive them from the survey transform.

    Parameters
    ----------
    path:
        Path to the OpendTect ASCII export.

    Returns
    -------
    list[FaultStick]
    """
    path = Path(path)
    sticks: list[FaultStick] = []
    current_stick: FaultStick | None = None
    _fault_re = re.compile(r"^Fault\s*:\s*(.+)$", re.IGNORECASE)

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            m = _fault_re.match(line)
            if m:
                if current_stick and len(current_stick) >= 2:
                    sticks.append(current_stick)
                current_stick = FaultStick(fault_name=m.group(1).strip())
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                if current_stick is None:
                    current_stick = FaultStick(fault_name="UNKNOWN")
                current_stick.points.append(FaultPoint(x, y, z))

    if current_stick and len(current_stick) >= 2:
        sticks.append(current_stick)

    logger.info("Parsed %d fault sticks from %s", len(sticks), path.name)
    return sticks


def parse_f3_fault_sticks(path: str | Path) -> list[FaultStick]:
    """Parse an F3 Demo (dGB / TerraNubis) 5-column fault-stick export.

    F3 fault interpretations are distributed as **headerless** ASCII tables
    with five whitespace/tab-delimited columns and world map coordinates::

        624690.0625  6074133   294.465  0 0
        624643.8125  6074132   368.093  0 1
        ...
        624505.1875  6074528   291.795  1 0

    Columns
    -------
    ``col1`` X map coordinate (metres)
    ``col2`` Y map coordinate (metres)
    ``col3`` Z / two-way time (milliseconds)
    ``col4`` stick_id — integer grouping points into distinct fault segments
    ``col5`` point_id — 0-based ordinal of the point within its stick

    Points are grouped by ``stick_id`` into separate :class:`FaultStick`
    objects (one stick = one fault segment) and ordered by ``point_id`` for
    safety.  Blank lines and ``#`` comments are skipped defensively.  Each
    returned stick is named ``"<file-stem>_stick<stick_id>"`` for uniqueness
    across multiple files, and degenerate single-point sticks are dropped
    (``>= 2`` points required), matching :func:`parse_opendtect_fault_sticks`.

    Unlike the Volve/OpendTect index-space path, the returned points carry
    **world XY** coordinates and must be rasterised via
    :meth:`FaultMaskGenerator.add_fault_sticks` with a :class:`SurveyTransform`
    (world→inline/crossline), not the index-space path.

    Parameters
    ----------
    path:
        Path to a single F3 fault-stick ASCII file (e.g. ``FaultA.txt``).

    Returns
    -------
    list[FaultStick]
    """
    path = Path(path)
    # stick_id -> list of (point_id, x, y, z_ms)
    by_stick: dict[int, list[tuple[int, float, float, float]]] = {}

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                z_ms = float(parts[2])
                stick_id = int(float(parts[3]))
                point_id = int(float(parts[4]))
            except ValueError:
                continue
            by_stick.setdefault(stick_id, []).append((point_id, x, y, z_ms))

    stem = path.stem
    sticks: list[FaultStick] = []
    for stick_id in sorted(by_stick):
        ordered = sorted(by_stick[stick_id], key=lambda t: t[0])
        points = [FaultPoint(x, y, z_ms) for (_pid, x, y, z_ms) in ordered]
        if len(points) < 2:
            continue
        sticks.append(FaultStick(fault_name=f"{stem}_stick{stick_id}", points=points))

    logger.info(
        "Parsed %d F3 fault sticks (%d stick_ids) from %s",
        len(sticks), len(by_stick), path.name,
    )
    return sticks


# ---------------------------------------------------------------------------
# Rasteriser
# ---------------------------------------------------------------------------


class FaultMaskGenerator:
    """Rasterise fault sticks onto the seismic grid to produce a binary mask.

    Parameters
    ----------
    volume_shape:
        ``(n_inlines, n_crosslines, n_samples)`` — must match the seismic volume.
    inline_range:
        ``(inline_min, inline_max, inline_step)`` from :class:`SurveyGeometry`.
    crossline_range:
        ``(crossline_min, crossline_max, crossline_step)``.
    sample_rate_ms:
        Milliseconds per sample.
    datum_ms:
        Recording start time in milliseconds.
    dilation_voxels:
        Half-width of the cubic dilation kernel applied around each painted
        voxel.  ``1`` → 3×3×3 neighbourhood.  Increase to 2–3 to compensate
        for coarse fault-stick spacing.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        inline_range: tuple[int, int, int],
        crossline_range: tuple[int, int, int],
        sample_rate_ms: float,
        datum_ms: float = 0.0,
        dilation_voxels: int = 1,
    ) -> None:
        self.shape = volume_shape
        self.il_min, self.il_max, self.il_step = inline_range
        self.xl_min, self.xl_max, self.xl_step = crossline_range
        self.sample_rate_ms = sample_rate_ms
        self.datum_ms = datum_ms
        self.dilation = dilation_voxels

        self._mask = np.zeros(volume_shape, dtype=np.uint8)

    # --- public interface --------------------------------------------------

    @property
    def mask(self) -> np.ndarray:
        """Accumulated binary fault mask, shape ``(n_il, n_xl, n_s)``, dtype uint8."""
        return self._mask

    def add_fault_sticks(
        self,
        sticks: list[FaultStick],
        transform: SurveyTransform,
    ) -> None:
        """Convert world-XYZ sticks to index space and rasterise them.

        Parameters
        ----------
        sticks:
            Parsed fault sticks with world XYZ coordinates.
        transform:
            :class:`SurveyTransform` mapping (X, Y) → (inline, crossline).
        """
        for stick in sticks:
            ilxl_pts: list[tuple[float, float, float]] = []
            for pt in stick.points:
                il_f, xl_f = transform.world_to_ilxl(pt.x, pt.y)
                # Convert survey numbers to 0-based voxel indices
                il_idx = (il_f - self.il_min) / self.il_step
                xl_idx = (xl_f - self.xl_min) / self.xl_step
                s_idx  = transform.z_to_sample(pt.z_ms)
                ilxl_pts.append((il_idx, xl_idx, s_idx))
            self._rasterise_stick(ilxl_pts)

        logger.info(
            "Rasterised %d sticks → %d labelled voxels",
            len(sticks),
            int(self._mask.sum()),
        )

    def add_fault_sticks_in_index_space(
        self,
        sticks_indexed: list[list[tuple[float, float, float]]],
        *,
        interpolate_between: bool = False,
        max_interp_gap_il: int = 5,
    ) -> None:
        """Add pre-indexed sticks given as ``(il_idx, xl_idx, s_idx)`` triplets.

        Useful when caller has already mapped to voxel coordinates (e.g., when
        fault sticks are provided in inline/crossline/sample from an
        interpretation workstation that exports grid coordinates directly).

        Parameters
        ----------
        sticks_indexed:
            Each element is one fault's list of ``(il_idx, xl_idx, s_idx)``
            picks in 0-based index space.
        interpolate_between:
            If ``True``, apply :func:`densify_stick_to_il_resolution` to each
            stick before rasterising.  This inserts 1-IL-resolution picks
            between sparse picks (gap ≤ ``max_interp_gap_il``), increasing the
            positive-voxel fraction by connecting interpreted sticks across
            the inline gap.

            **Geophysical justification:** planar-fault assumption between
            adjacent sticks.  Only valid for gaps ≤ ``max_interp_gap_il``.

            **Uncertainty:** interpolated picks are INFERRED, not picked.
            Treat the resulting labels as lower-confidence ground truth at
            intermediate IL positions.
        max_interp_gap_il:
            Maximum IL gap to bridge when ``interpolate_between=True``.
            Default 5.  Gaps > this are left as-is (possible fault segment
            boundary or interpretation gap).
        """
        raw_count = sum(len(s) for s in sticks_indexed)
        densified_count = 0

        for stick_pts in sticks_indexed:
            pts_to_raster: list[tuple[float, float, float]] = list(stick_pts)
            if interpolate_between and len(stick_pts) >= 2:
                pts_to_raster = densify_stick_to_il_resolution(
                    pts_to_raster, max_il_gap=max_interp_gap_il,
                )
            densified_count += len(pts_to_raster)
            self._rasterise_stick(pts_to_raster)

        if interpolate_between:
            logger.info(
                "Rasterised %d sticks — raw picks: %d → densified picks: %d → %d labelled voxels",
                len(sticks_indexed), raw_count, densified_count, int(self._mask.sum()),
            )
        else:
            logger.info(
                "Rasterised %d sticks → %d labelled voxels",
                len(sticks_indexed), int(self._mask.sum()),
            )

    def to_zarr(
        self,
        output_path: str | Path,
        chunks: tuple[int, int, int] | None = None,
        *,
        overwrite: bool = False,
    ) -> zarr.Array:
        """Write the binary mask to a Zarr store.

        Parameters
        ----------
        output_path:
            Zarr directory path.
        chunks:
            Chunk shape; defaults to ``(64, 64, 128)`` to match seismic data.
        overwrite:
            Overwrite an existing store.

        Returns
        -------
        zarr.Array
            The written ``fault_mask`` dataset.
        """
        chunks = chunks or (64, 64, 128)
        output_path = Path(output_path)

        # zarr v3: LocalStore replaces DirectoryStore; create_array replaces create_dataset
        store = zarr.storage.LocalStore(str(output_path))
        root = zarr.open_group(store, mode="w" if overwrite else "w-")

        z = root.create_array(
            "fault_mask",
            data=self._mask.astype(np.uint8),
            chunks=chunks,
            overwrite=overwrite,
        )
        logger.info(
            "Wrote fault mask → %s  shape=%s  labelled_voxels=%d",
            output_path, z.shape, int(self._mask.sum()),
        )
        return z

    # --- internal ----------------------------------------------------------

    def _rasterise_stick(self, pts: list[tuple[float, float, float]]) -> None:
        """Densely interpolate a stick polyline and paint it into the mask."""
        if len(pts) == 0:
            return
        if len(pts) == 1:
            self._paint_voxel(*pts[0])
            return

        arr = np.array(pts, dtype=np.float64)  # (N, 3)

        # Arc-length parameterisation for uniform-density interpolation
        seg_len = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = arc[-1]

        if total < 1e-9:
            self._paint_voxel(*pts[0])
            return

        # At least one query per voxel diagonal (≈ √3 ≈ 1.73 samples)
        n_q = max(int(total * 2.0), len(pts))
        t_q = np.linspace(0.0, total, n_q)

        interp_il = interp1d(arc, arr[:, 0], kind="linear")(t_q)
        interp_xl = interp1d(arc, arr[:, 1], kind="linear")(t_q)
        interp_s  = interp1d(arc, arr[:, 2], kind="linear")(t_q)

        for il_f, xl_f, s_f in zip(interp_il, interp_xl, interp_s, strict=False):
            self._paint_voxel(il_f, xl_f, s_f)

    def _paint_voxel(self, il_f: float, xl_f: float, s_f: float) -> None:
        """Mark a voxel and its dilation neighbourhood in the mask."""
        n_il, n_xl, n_s = self.shape
        d = self.dilation

        il_c = round(il_f)
        xl_c = round(xl_f)
        s_c  = round(s_f)

        for di in range(-d, d + 1):
            for dj in range(-d, d + 1):
                for dk in range(-d, d + 1):
                    ii = il_c + di
                    jj = xl_c + dj
                    kk = s_c  + dk
                    if 0 <= ii < n_il and 0 <= jj < n_xl and 0 <= kk < n_s:
                        self._mask[ii, jj, kk] = 1
