"""Generate a ground-truth fault-label Zarr from Volve fault-stick .dat files.

Sprint 2 item S2-01.

Reads the real Volve fault-stick .dat files (data/volve/interpretations/fault_sticks/),
maps them to the amplitude volume's index grid, rasterises with dilation, and writes
a uint8 fault-label Zarr alongside the amplitude volume.

Coordinate mapping (verified against synthetic.json geometry)
-------------------------------------------------------------
The .dat files contain three columns:  inline_idx  crossline_idx  z_col

Despite the file comment "z_ms", the z column is a **sample index**, not ms.
(Prior team finding confirmed by the fact that inline/crossline values are
 clearly 0-based volume indices in range 0–99 and 0–199 respectively, not
 absolute survey numbers 1001–1100 / 1900–2099.)

  abs_inline     = base_il  + inline_idx   = 1001 + inline_idx
  abs_crossline  = base_xl  + crossline_idx = 1900 + crossline_idx
  twt_ms         = z_col * sample_rate_ms  = z_col * 4.0

All three values are already in 0-based index space — no further conversion
is needed before calling add_fault_sticks_in_index_space().

Output
------
  data/volve/staged/fault_label.zarr   — uint8, shape (100, 200, 500)
    └── fault_mask                     — the binary label array

Usage
-----
    python scripts/generate_fault_label.py               # default dilation=3
    python scripts/generate_fault_label.py --dilation 2  # lighter dilation
    python scripts/generate_fault_label.py --overwrite   # replace existing

This is the ground-truth label consumed by S2-02 (training) and S2-03 (eval).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from deepseismic.ingest.label_generator import FaultMaskGenerator  # noqa: E402,I001
from deepseismic.validation import load_volve_fault_sticks  # noqa: E402,I001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_fault_label")

# ---------------------------------------------------------------------------
# Default paths (can be overridden via CLI args)
# ---------------------------------------------------------------------------
_DEFAULT_FAULT_STICK_DIR = _REPO_ROOT / "data/volve/interpretations/fault_sticks"
_DEFAULT_AMPLITUDE_JSON  = _REPO_ROOT / "data/volve/staged/synthetic.json"
_DEFAULT_LABEL_OUTPUT    = _REPO_ROOT / "data/volve/staged/fault_label.zarr"


def _load_geometry(json_path: Path) -> dict:
    """Load amplitude volume geometry from its JSON sidecar."""
    with open(json_path) as f:
        meta = json.load(f)
    return meta["geometry"]


def _report_sticks(sticks: list[np.ndarray], geom: dict) -> None:
    """Log per-stick stats and grid-alignment check."""
    n_il  = geom["n_inlines"]
    n_xl  = geom["n_crosslines"]
    n_s   = geom["n_samples"]
    sr    = geom["sample_rate_ms"]
    # Coordinate bases derived from geometry (not hard-coded)
    base_il = geom["inline_min"]
    base_xl = geom["crossline_min"]

    all_pts = np.vstack(sticks) if sticks else np.empty((0, 3))
    n_pts   = len(all_pts)
    logger.info("Total raw stick points: %d across %d sticks", n_pts, len(sticks))

    if n_pts == 0:
        return

    il_min_dat,  il_max_dat  = all_pts[:, 0].min(), all_pts[:, 0].max()
    xl_min_dat,  xl_max_dat  = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min_dat,   z_max_dat   = all_pts[:, 2].min(), all_pts[:, 2].max()

    logger.info(
        "Stick index ranges  →  il:[%.0f–%.0f] xl:[%.0f–%.0f] z:[%.0f–%.0f]",
        il_min_dat, il_max_dat, xl_min_dat, xl_max_dat, z_min_dat, z_max_dat,
    )
    logger.info(
        "Volume index bounds →  il:[0–%d]  xl:[0–%d]  s:[0–%d]",
        n_il - 1, n_xl - 1, n_s - 1,
    )
    logger.info(
        "Abs survey numbers  →  il:[%.0f–%.0f] xl:[%.0f–%.0f]  twt:[%.0f–%.0f] ms",
        base_il + il_min_dat, base_il + il_max_dat,
        base_xl + xl_min_dat, base_xl + xl_max_dat,
        z_min_dat * sr, z_max_dat * sr,
    )

    # Grid alignment check
    out_of_bounds = (
        (all_pts[:, 0] < 0) | (all_pts[:, 0] >= n_il) |
        (all_pts[:, 1] < 0) | (all_pts[:, 1] >= n_xl) |
        (all_pts[:, 2] < 0) | (all_pts[:, 2] >= n_s)
    )
    if out_of_bounds.any():
        logger.warning(
            "GRID ALIGNMENT: %d / %d stick points fall outside the volume!",
            int(out_of_bounds.sum()), n_pts,
        )
    else:
        logger.info("GRID ALIGNMENT: all %d stick points are inside the volume ✓", n_pts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dilation", type=int, default=3,
                        help="Dilation radius in voxels (default: 3). "
                             "Applied as a cubic neighbourhood around each rasterised point. "
                             "Resolution guardrail: ≤3 keeps label band ≤12ms TWT (~24m at 1km/s). "
                             "Do not exceed 3 without geophysical justification (λ/4 ≈ 13.7 m).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing label Zarr.")
    parser.add_argument("--interpolate-between", action="store_true",
                        help="Densify each fault stick to 1-IL resolution. "
                             "Bridges IL gaps ≤ --max-interp-gap (planar-fault assumption). "
                             "⚠ Interpolated picks are INFERRED — see QC report.")
    parser.add_argument("--max-interp-gap", type=int, default=5,
                        help="Max IL gap to bridge with --interpolate-between. "
                             "Default 5. Larger gaps are NOT bridged.")
    parser.add_argument(
        "--fault-stick-dir", default=None,
        help=(
            "Directory containing fault-stick .dat files. "
            f"Default: {_DEFAULT_FAULT_STICK_DIR}"
        ),
    )
    parser.add_argument(
        "--amplitude-json", default=None,
        help=(
            "Path to the amplitude volume JSON sidecar (geometry source). "
            f"Default: {_DEFAULT_AMPLITUDE_JSON}"
        ),
    )
    parser.add_argument(
        "--label-output", default=None,
        help=(
            "Output path for the fault-label Zarr store. "
            f"Default: {_DEFAULT_LABEL_OUTPUT}"
        ),
    )
    args = parser.parse_args()

    FAULT_STICK_DIR = (
        Path(args.fault_stick_dir) if args.fault_stick_dir else _DEFAULT_FAULT_STICK_DIR
    )
    AMPLITUDE_JSON = (
        Path(args.amplitude_json) if args.amplitude_json else _DEFAULT_AMPLITUDE_JSON
    )
    LABEL_OUTPUT = (
        Path(args.label_output) if args.label_output else _DEFAULT_LABEL_OUTPUT
    )

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------
    if not FAULT_STICK_DIR.exists():
        sys.exit(
            f"ERROR: Fault-stick directory not found: {FAULT_STICK_DIR}\n"
            "       Real .dat files are required (S2-01 risk #1 — no fallback)."
        )
    dat_files = sorted(FAULT_STICK_DIR.glob("*.dat"))
    if not dat_files:
        sys.exit(
            f"ERROR: No .dat files found in {FAULT_STICK_DIR}\n"
            "       Real fault sticks are required — no silent fallback to synthetic."
        )
    if not AMPLITUDE_JSON.exists():
        sys.exit(f"ERROR: Amplitude sidecar not found: {AMPLITUDE_JSON}")
    if LABEL_OUTPUT.exists() and not args.overwrite:
        sys.exit(
            f"ERROR: {LABEL_OUTPUT} already exists. Pass --overwrite to replace it.\n"
            "       Tip: python scripts/generate_fault_label.py --overwrite"
        )

    logger.info("Fault-stick dir : %s  (%d .dat files)", FAULT_STICK_DIR, len(dat_files))
    for f in dat_files:
        logger.info("  %s", f.name)
    logger.info("Amplitude JSON  : %s", AMPLITUDE_JSON)
    logger.info("Label output    : %s", LABEL_OUTPUT)
    logger.info("Dilation radius : %d voxels  (neighbourhood=%d³=%d)",
                args.dilation, 2 * args.dilation + 1, (2 * args.dilation + 1) ** 3)

    # ------------------------------------------------------------------
    # Load geometry
    # ------------------------------------------------------------------
    geom = _load_geometry(AMPLITUDE_JSON)
    vol_shape = (geom["n_inlines"], geom["n_crosslines"], geom["n_samples"])
    logger.info(
        "Volume shape    : %s  (il×xl×samples)  sample_rate=%.1f ms  datum=%.1f ms",
        vol_shape, geom["sample_rate_ms"], geom["datum_ms"],
    )

    # ------------------------------------------------------------------
    # Parse fault sticks
    # ------------------------------------------------------------------
    # load_volve_fault_sticks reads col[0]=il_idx, col[1]=xl_idx, col[-1]=z_col
    # For our 3-column files this maps perfectly.
    sticks = load_volve_fault_sticks(FAULT_STICK_DIR)
    if not sticks:
        sys.exit(
            "ERROR: Parsed 0 fault sticks from .dat files.\n"
            "       Check file format (expected 3-col: inline_idx crossline_idx z_col)."
        )

    logger.info("Parsed %d fault sticks", len(sticks))
    _report_sticks(sticks, geom)

    # ------------------------------------------------------------------
    # Coordinate mapping (documented for Ash's review)
    # ------------------------------------------------------------------
    # Each stick is np.ndarray shape (N, 3): [il_idx, xl_idx, z_col]
    # z_col is a SAMPLE INDEX (0-based), not ms, despite the file comment.
    # abs_inline = BASE_IL(1001) + il_idx
    # abs_crossline = BASE_XL(1900) + xl_idx
    # twt_ms = z_col * sample_rate_ms(4.0)
    # No conversion needed — values are already 0-based index space.

    indexed_sticks: list[list[tuple[float, float, float]]] = []
    for stick_arr in sticks:
        pts = [(float(r[0]), float(r[1]), float(r[2])) for r in stick_arr]
        indexed_sticks.append(pts)

    total_raw_pts = sum(len(s) for s in indexed_sticks)
    logger.info("Total raw points to rasterise: %d", total_raw_pts)

    # Detect synthetic proxy directory for QC labeling
    _synth_dir = _REPO_ROOT / "data/volve/interpretations/fault_sticks_synth"
    is_synthetic_proxy = FAULT_STICK_DIR.resolve() == _synth_dir.resolve()
    if is_synthetic_proxy:
        logger.warning(
            "[SYNTHETIC PROXY] Stick dir is fault_sticks_synth/ — output is a "
            "densification proxy for app-readiness testing, NOT real Volve ground truth."
        )

    # ------------------------------------------------------------------
    # Rasterise — baseline (no interpolation) for before/after reporting
    # ------------------------------------------------------------------
    _xl_range = (geom["crossline_min"], geom["crossline_max"], geom["crossline_step"])
    gen = FaultMaskGenerator(
        volume_shape     = vol_shape,
        inline_range     = (geom["inline_min"], geom["inline_max"], geom["inline_step"]),
        crossline_range  = _xl_range,
        sample_rate_ms   = geom["sample_rate_ms"],
        datum_ms         = geom["datum_ms"],
        dilation_voxels  = args.dilation,
    )

    if args.interpolate_between:
        # Run baseline pass first (no interpolation) so we can report before/after
        gen_baseline = FaultMaskGenerator(
            volume_shape    = vol_shape,
            inline_range    = (geom["inline_min"], geom["inline_max"], geom["inline_step"]),
            crossline_range = _xl_range,
            sample_rate_ms  = geom["sample_rate_ms"],
            datum_ms        = geom["datum_ms"],
            dilation_voxels = args.dilation,
        )
        gen_baseline.add_fault_sticks_in_index_space(indexed_sticks, interpolate_between=False)
        baseline_voxels = int(gen_baseline.mask.sum())
        baseline_frac   = baseline_voxels / vol_shape[0] / vol_shape[1] / vol_shape[2]

        gen.add_fault_sticks_in_index_space(
            indexed_sticks,
            interpolate_between=True,
            max_interp_gap_il=args.max_interp_gap,
        )
    else:
        gen.add_fault_sticks_in_index_space(indexed_sticks, interpolate_between=False)
        baseline_voxels = None
        baseline_frac   = None

    mask = gen.mask

    # ------------------------------------------------------------------
    # QC report
    # ------------------------------------------------------------------
    total_voxels  = mask.size
    fault_voxels  = int(mask.sum())
    fault_frac    = fault_voxels / total_voxels

    print()
    print("=" * 60)
    title = "  FAULT LABEL QC REPORT  (S3-#8 dense-label)"
    if is_synthetic_proxy:
        title += "  [SYNTHETIC PROXY]"
    print(title)
    print("=" * 60)
    if is_synthetic_proxy:
        print("  [WARN] SYNTHETIC PROXY -- NOT real Volve ground truth.")
        print("  [WARN] Use ONLY for app-readiness / pipeline validation.")
        print()
    print(f"  Input .dat files       : {len(dat_files)}")
    print(f"  Fault sticks parsed    : {len(sticks)}")
    print(f"  Raw stick points       : {total_raw_pts}")
    print(f"  Volume shape           : {vol_shape}  (il×xl×samples)")
    print(f"  Dilation radius        : {args.dilation} voxels")
    interp_label = (
        "ON  (max IL gap=" + str(args.max_interp_gap) + ")"
        if args.interpolate_between else "OFF"
    )
    print(f"  Between-stick interp   : {interp_label}")
    print()
    if baseline_voxels is not None:
        print("  --- Before densification ---")
        print(f"  Fault voxels (raw)     : {baseline_voxels:,} / {total_voxels:,}")
        print(f"  Fault fraction (raw)   : {baseline_frac:.6f}  ({baseline_frac*100:.4f} %)")
        print()
        print("  --- After between-stick densification ---")
        print(f"  Fault voxels           : {fault_voxels:,} / {total_voxels:,}")
        print(f"  Fault fraction         : {fault_frac:.6f}  ({fault_frac*100:.4f} %)")
        improvement = fault_voxels / max(baseline_voxels, 1)
        print(f"  Improvement            : {improvement:.2f}x")
        print()
        print("  NOTE: For linear fault geometry the arc-length rasteriser")
        print("    already covers all intermediate ILs; densification adds")
        print("    value for curved geometry, explicit IL documentation, and")
        print("    real Petrel multi-stick format.")
        print()
        print("  [WARN] Uncertainty: interpolated picks are INFERRED labels -- lower")
        print("    confidence than direct interpreter picks. For fault IL-step N,")
        print("    ~(N-1)/N of painted ILs are inferred.")
    else:
        print(f"  Fault voxels           : {fault_voxels:,} / {total_voxels:,}")
        print(f"  Fault fraction         : {fault_frac:.6f}  ({fault_frac*100:.4f} %)")
    print()
    print("  Coordinate mapping (for review):")
    print(f"    abs_inline     = {geom['inline_min']} + il_idx")
    print(f"    abs_crossline  = {geom['crossline_min']} + xl_idx")
    print(f"    twt_ms         = z_col x {geom['sample_rate_ms']}  (z_col is sample index)")
    print(f"    datum_ms       = {geom['datum_ms']}")
    print()
    print("  Resolution guardrail   : L/4 ~ 13.7 m ~ 3.4 samples (@36.6 Hz, 2000 m/s)")
    n_label = args.dilation * 2 + 1
    ms_band = n_label * 4
    print(f"  Dilation label band    : {n_label} voxels x 4ms = {ms_band} ms TWT")
    print()

    if fault_voxels == 0:
        print("  [FAIL] Zero fault voxels -- check coordinate mapping.")
        print("=" * 60)
        sys.exit(1)
    elif fault_frac < 0.0001:
        print("  [WARN] Very sparse fault mask (<0.01%) -- pathological for training.")
        print("         Recommend: --interpolate-between or larger dilation.")
    elif fault_frac < 0.005:
        print("  [CAUTION] Sparse fault mask (<0.5%) -- heavy resampling/weighting required.")
    else:
        print("  [PASS] Fault coverage >= 0.5% -- approaching statistically meaningful range.")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Write Zarr
    # ------------------------------------------------------------------
    z_arr = gen.to_zarr(LABEL_OUTPUT, overwrite=args.overwrite)
    logger.info("Wrote fault label -> %s  array='%s'  shape=%s  dtype=%s",
                LABEL_OUTPUT, "fault_mask", z_arr.shape, z_arr.dtype)

    print()
    print("Output written:")
    print(f"  {LABEL_OUTPUT}")
    print(f"  Array: fault_mask  dtype=uint8  shape={z_arr.shape}")
    print()
    print("Regenerate commands:")
    print("  # Real sticks (baseline):")
    print("  python scripts/generate_fault_label.py --overwrite")
    print()
    print("  # Real sticks + between-stick densification:")
    print("  python scripts/generate_fault_label.py --interpolate-between --overwrite")
    print()
    print("  # Synthetic proxy (S3-#8 validation):")
    print("  python scripts/generate_fault_label.py \\")
    print("      --fault-stick-dir data/volve/interpretations/fault_sticks_synth \\")
    print("      --interpolate-between \\")
    print("      --label-output data/volve/staged/fault_label_synth.zarr \\")
    print("      --overwrite")


if __name__ == "__main__":
    main()
