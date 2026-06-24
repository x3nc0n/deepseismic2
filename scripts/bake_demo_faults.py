"""Bake fault inference results for the Streamlit demo viewer.

Runs VolumeInference once over the staged synthetic Zarr volume and writes
probability + binary-mask Zarrs alongside it.  The viewer then reads 2-D slices
directly — no per-slider inference, no server required.

Usage
-----
    python scripts/bake_demo_faults.py                    # first run
    python scripts/bake_demo_faults.py --overwrite        # re-run after retraining

Outputs (relative to repo root)
--------------------------------
    data/volve/staged/fault_prob.zarr   — float32 fault probability, array "fault_probability"
    data/volve/staged/fault_mask.zarr   — uint8  binary mask,         array "fault_mask"

Both stores share the same spatial layout as the amplitude volume:
    shape  (100, 200, 500)  = n_inlines × n_crosslines × n_samples
    index  inline_i, crossline_j, sample_k  (0-based)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve repo root regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from deepseismic.models.inference import run_inference  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bake_demo_faults")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AMPLITUDE_ZARR  = _REPO_ROOT / "data/volve/staged/synthetic.zarr"
CHECKPOINT      = _REPO_ROOT / "checkpoints/latest.pt"
PROB_OUTPUT     = _REPO_ROOT / "data/volve/staged/fault_prob.zarr"
MASK_OUTPUT     = _REPO_ROOT / "data/volve/staged/fault_mask.zarr"

# Patch size: 64³ fits the 100×200×500 volume on all axes.
PATCH_SIZE   = (64, 64, 64)
OVERLAP      = 0.25
BATCH_SIZE   = 4
THRESHOLD    = 0.5


def _qc_check(prob_volume: np.ndarray, binary_mask: np.ndarray) -> bool:
    """Print QC metrics and return True if output looks usable."""
    fault_frac = float(binary_mask.sum()) / binary_mask.size
    prob_min   = float(prob_volume.min())
    prob_max   = float(prob_volume.max())
    prob_mean  = float(prob_volume.mean())
    prob_p10   = float(np.percentile(prob_volume, 10))
    prob_p90   = float(np.percentile(prob_volume, 90))

    print()
    print("=" * 60)
    print("  BAKE QC METRICS")
    print("=" * 60)
    print(f"  Volume shape       : {prob_volume.shape}")
    print(f"  Probability range  : min={prob_min:.4f}  max={prob_max:.4f}")
    print(f"  Probability mean   : {prob_mean:.4f}")
    print(f"  Probability p10/p90: {prob_p10:.4f} / {prob_p90:.4f}")
    print(f"  Fault voxel frac   : {fault_frac:.4f}  ({fault_frac*100:.2f} %)")
    print()

    # Heuristic quality gates
    ok = True
    if prob_max - prob_min < 0.05:
        print("  [FAIL] Probabilities have near-zero range — model may be stuck.")
        ok = False
    if fault_frac < 0.001:
        print("  [FAIL] Fewer than 0.1% fault voxels — model detects nothing.")
        ok = False
    if fault_frac > 0.5:
        print("  [FAIL] More than 50% fault voxels — model detects everything (random weights?).")
        ok = False
    if ok:
        print("  [PASS] Output looks usable for demo.")
    else:
        print()
        print("  Model output fails QC.  Do NOT wire to viewer.  Retrain before demo.")
    print("=" * 60)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output stores.")
    args = parser.parse_args()

    # Pre-flight checks
    if not AMPLITUDE_ZARR.exists():
        sys.exit(f"ERROR: Amplitude Zarr not found: {AMPLITUDE_ZARR}\n"
                 "       Run the ingest pipeline first.")
    if not CHECKPOINT.exists():
        sys.exit(f"ERROR: Checkpoint not found: {CHECKPOINT}\n"
                 "       Train the model first (see src/deepseismic/models/unet.py).")
    if PROB_OUTPUT.exists() and not args.overwrite:
        sys.exit(f"ERROR: {PROB_OUTPUT} already exists.  Pass --overwrite to replace it.")
    if MASK_OUTPUT.exists() and not args.overwrite:
        sys.exit(f"ERROR: {MASK_OUTPUT} already exists.  Pass --overwrite to replace it.")

    logger.info("Amplitude Zarr : %s", AMPLITUDE_ZARR)
    logger.info("Checkpoint     : %s", CHECKPOINT)
    logger.info("Prob output    : %s", PROB_OUTPUT)
    logger.info("Mask output    : %s", MASK_OUTPUT)
    logger.info("Patch size     : %s  overlap=%.2f  batch=%d", PATCH_SIZE, OVERLAP, BATCH_SIZE)

    t0 = time.perf_counter()
    prob_volume, binary_mask = run_inference(
        seismic_zarr    = AMPLITUDE_ZARR,
        checkpoint_path = CHECKPOINT,
        prob_output     = PROB_OUTPUT,
        mask_output     = MASK_OUTPUT,
        patch_size      = PATCH_SIZE,
        overlap         = OVERLAP,
        batch_size      = BATCH_SIZE,
        threshold       = THRESHOLD,
        device          = "cpu",
        overwrite       = args.overwrite,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Inference completed in %.1f s", elapsed)

    ok = _qc_check(prob_volume, binary_mask)
    if not ok:
        sys.exit(1)

    print()
    print("Outputs written:")
    print(f"  {PROB_OUTPUT}")
    print(f"  {MASK_OUTPUT}")
    print()
    print("Next step: launch the Streamlit viewer.")
    print("  streamlit run src/deepseismic/ui/streamlit_app.py")


if __name__ == "__main__":
    main()
