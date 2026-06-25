"""CLI entry point: SEG-Y → Zarr ingest (S3-04).

Wraps :func:`deepseismic.ingest.segy_loader.segy_to_zarr` so it can be called
from the command line during local development or in-VNet Azure ML jobs.

ADLS path convention (infra issue #11)
---------------------------------------
    staged/surveys/{survey_id}/amplitude.zarr

The ``--dest`` default follows this convention when ``--survey-id`` is given.
For in-VNet jobs, mount the ADLS containers and pass an absolute path.

Local smoke-test (format-proxy validation — NOT real results)
--------------------------------------------------------------
Use the synthetic SEG-Y as a format proxy to verify the ingest code path
before real ST10010 data lands in ADLS::

    python scripts/ingest_segy.py \\
        --source data/volve/synthetic_sample.segy \\
        --dest data/volve/staged/smoke_ingest.zarr \\
        --survey-id synthetic-proxy \\
        --sample-mode \\
        --overwrite

    # ⚠️  SYNTHETIC-PROXY ONLY — numbers are NOT from real Volve ST10010 data.
    #    Real ingest must run in-VNet once infra issue #11 lands the SEG-Y.

Real ST10010 ingest (in-VNet only — private endpoint)
------------------------------------------------------
::

    python scripts/ingest_segy.py \\
        --source /mnt/raw/ST10010_PSDM_TIME.segy \\
        --dest /mnt/staged/surveys/volve-st10010/amplitude.zarr \\
        --survey-id volve-st10010 \\
        --overwrite

    # Cheap smoke-ingest (first 50 inlines, ~seconds):
    python scripts/ingest_segy.py \\
        --source /mnt/raw/ST10010_PSDM_TIME.segy \\
        --dest /mnt/staged/surveys/volve-st10010/amplitude_sample.zarr \\
        --survey-id volve-st10010 \\
        --sample-mode --sample-n-inlines 50 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from deepseismic.ingest.segy_loader import segy_to_zarr  # noqa: E402,I001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_segy")


def _build_dest(survey_id: str | None, dest: str | None) -> str:
    """Return the destination zarr path, defaulting to the ADLS convention."""
    if dest:
        return dest
    if survey_id:
        return str(_REPO_ROOT / "data/volve/staged" / f"{survey_id}.zarr")
    # Fallback: a generic output next to the script
    return str(_REPO_ROOT / "data/volve/staged/amplitude.zarr")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the source SEG-Y file.",
    )
    parser.add_argument(
        "--dest", default=None,
        help=(
            "Destination Zarr directory. "
            "Default: data/volve/staged/{survey-id}.zarr "
            "(or data/volve/staged/amplitude.zarr if --survey-id is not set)."
        ),
    )
    parser.add_argument(
        "--survey-id", default=None,
        help=(
            "Survey identifier embedded in the JSON sidecar "
            "(e.g. 'volve-st10010'). Used to derive the default --dest path "
            "and for downstream ADLS path conventions."
        ),
    )
    parser.add_argument(
        "--sample-mode", action="store_true",
        help=(
            "Load only the first --sample-n-inlines inlines. "
            "Use for cheap local smoke-ingest to validate format without "
            "processing the full survey."
        ),
    )
    parser.add_argument(
        "--sample-n-inlines", type=int, default=50,
        help="Number of inlines to load in sample mode (default: 50).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing Zarr store.",
    )
    parser.add_argument(
        "--chunks", nargs=3, type=int, default=[64, 64, 128],
        metavar=("IL", "XL", "S"),
        help="Zarr chunk shape: inline crossline sample (default: 64 64 128).",
    )

    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        sys.exit(f"ERROR: Source SEG-Y not found: {source}")

    dest = _build_dest(args.survey_id, args.dest)
    chunks: tuple[int, int, int] = tuple(args.chunks)  # type: ignore[assignment]

    logger.info("Source SEG-Y     : %s", source)
    logger.info("Destination Zarr : %s", dest)
    logger.info("Survey ID        : %s", args.survey_id or "(not set)")
    logger.info("Sample mode      : %s", args.sample_mode)
    if args.sample_mode:
        logger.info("Sample N inlines : %d", args.sample_n_inlines)
    logger.info("Chunks           : %s", chunks)
    logger.info("Overwrite        : %s", args.overwrite)

    if args.sample_mode:
        logger.info(
            "[WARN] SYNTHETIC-PROXY / SMOKE-INGEST MODE -- results are NOT from real Volve data."
        )

    meta = segy_to_zarr(
        source,
        dest,
        survey_id=args.survey_id,
        chunks=chunks,
        sample_mode=args.sample_mode,
        sample_n_inlines=args.sample_n_inlines,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 60)
    print("  INGEST COMPLETE")
    if args.sample_mode:
        print("  [WARN] SYNTHETIC-PROXY / SMOKE-INGEST -- NOT real Volve results")
    print("=" * 60)
    print(f"  Source        : {meta.source_file}")
    print(f"  Survey ID     : {meta.survey_id or '(not set)'}")
    print(f"  Zarr output   : {meta.zarr_path}")
    print(f"  Inlines loaded: {meta.n_inlines_loaded}")
    print("  Geometry:")
    g = meta.geometry
    print(
        f"    Inlines     : {g['inline_min']}--{g['inline_max']}"
        f"  step={g['inline_step']}"
    )
    print(
        f"    Crosslines  : {g['crossline_min']}--{g['crossline_max']}"
        f"  step={g['crossline_step']}"
    )
    print(
        f"    Samples     : {g['n_samples']}  dt={g['sample_rate_ms']} ms"
        f"  datum={g['datum_ms']} ms"
    )
    p01 = meta.amplitude_stats["p01"]
    p99 = meta.amplitude_stats["p99"]
    print(f"  Amplitude p01 / p99: {p01:.4f} / {p99:.4f}")
    print(f"  SHA-256 (quick)     : {meta.source_sha256 or 'N/A'}")
    print(f"  Sidecar JSON        : {Path(dest).with_suffix('.json')}")
    print("=" * 60)

    # Also write a human-readable summary to stdout as JSON for log capture
    summary = {
        "source": str(source),
        "survey_id": meta.survey_id,
        "dest": dest,
        "n_inlines_loaded": meta.n_inlines_loaded,
        "geometry": meta.geometry,
        "amplitude_stats": meta.amplitude_stats,
        "synthetic_proxy": args.sample_mode,
    }
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
