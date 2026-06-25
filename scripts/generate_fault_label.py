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
# Paths
# ---------------------------------------------------------------------------
FAULT_STICK_DIR = _REPO_ROOT / "data/volve/interpretations/fault_sticks"
AMPLITUDE_JSON  = _REPO_ROOT / "data/volve/staged/synthetic.json"
LABEL_OUTPUT    = _REPO_ROOT / "data/volve/staged/fault_label.zarr"

# Coordinate bases: abs_inline = BASE_IL + il_idx, abs_crossline = BASE_XL + xl_idx
BASE_IL = 1001
BASE_XL = 1900


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
        BASE_IL + il_min_dat, BASE_IL + il_max_dat,
        BASE_XL + xl_min_dat, BASE_XL + xl_max_dat,
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
                             "Applied as a cubic neighbourhood around each rasterised point.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing label Zarr.")
    args = parser.parse_args()

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

    # ------------------------------------------------------------------
    # Rasterise
    # ------------------------------------------------------------------
    gen = FaultMaskGenerator(
        volume_shape     = vol_shape,
        inline_range     = (geom["inline_min"], geom["inline_max"], geom["inline_step"]),
        crossline_range  = (geom["crossline_min"], geom["crossline_max"], geom["crossline_step"]),
        sample_rate_ms   = geom["sample_rate_ms"],
        datum_ms         = geom["datum_ms"],
        dilation_voxels  = args.dilation,
    )

    gen.add_fault_sticks_in_index_space(indexed_sticks)
    mask = gen.mask

    # ------------------------------------------------------------------
    # QC report
    # ------------------------------------------------------------------
    total_voxels  = mask.size
    fault_voxels  = int(mask.sum())
    fault_frac    = fault_voxels / total_voxels

    print()
    print("=" * 60)
    print("  FAULT LABEL QC REPORT  (S2-01)")
    print("=" * 60)
    print(f"  Input .dat files       : {len(dat_files)}")
    print(f"  Fault sticks parsed    : {len(sticks)}")
    print(f"  Raw stick points       : {total_raw_pts}")
    print(f"  Volume shape           : {vol_shape}  (il×xl×samples)")
    print(f"  Dilation radius        : {args.dilation} voxels")
    print()
    print(f"  Fault voxels           : {fault_voxels:,} / {total_voxels:,}")
    print(f"  Fault fraction         : {fault_frac:.6f}  ({fault_frac*100:.4f} %)")
    print()
    print("  Coordinate mapping (for review):")
    print(f"    abs_inline     = {BASE_IL} + il_idx  (il_idx in 0–{vol_shape[0]-1})")
    print(f"    abs_crossline  = {BASE_XL} + xl_idx  (xl_idx in 0–{vol_shape[1]-1})")
    print(f"    twt_ms         = z_col × {geom['sample_rate_ms']}  (z_col is sample index)")
    print(f"    datum_ms       = {geom['datum_ms']}")
    print()

    if fault_voxels == 0:
        print("  [FAIL] Zero fault voxels — check coordinate mapping.")
        print("=" * 60)
        sys.exit(1)
    elif fault_frac < 0.0001:
        print("  [WARN] Very sparse fault mask (<0.01%) — consider larger dilation.")
    else:
        print("  [PASS] Non-trivial fault coverage — suitable for training.")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Write Zarr
    # ------------------------------------------------------------------
    z_arr = gen.to_zarr(LABEL_OUTPUT, overwrite=args.overwrite)
    logger.info("Wrote fault label → %s  array='%s'  shape=%s  dtype=%s",
                LABEL_OUTPUT, "fault_mask", z_arr.shape, z_arr.dtype)

    print()
    print("Output written:")
    print(f"  {LABEL_OUTPUT}")
    print(f"  Array: fault_mask  dtype=uint8  shape={z_arr.shape}")
    print()
    print("Regenerate with:")
    print("  python scripts/generate_fault_label.py --overwrite")
    print("  python scripts/generate_fault_label.py --dilation 2 --overwrite")


if __name__ == "__main__":
    main()
