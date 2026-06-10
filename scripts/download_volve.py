#!/usr/bin/env python3
"""Download a curated Volve dataset subset for deepseismic2.

Quick start - no download needed:
    python scripts/download_volve.py --sample

Full seismic (after portal registration at equinor.com/energy/volve-data-sharing):
    python scripts/download_volve.py --seismic --base-url "https://..."

AVO angle stacks + velocity:
    python scripts/download_volve.py --seismic --base-url "https://..."

Wells + interpretations:
    python scripts/download_volve.py --wells --interpretations --base-url "https://..."

Everything:
    python scripts/download_volve.py --all --base-url "https://..."

Verify downloads:
    python scripts/download_volve.py --verify

--base-url is the storage root URL (with SAS token) from the Equinor portal, or
the Databricks external-location path obtained via databricks_export.py --get-url.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("volve-dl")

# ---------------------------------------------------------------------------
# File catalog - exact names and relative paths inside Equinor storage
# ---------------------------------------------------------------------------

SEISMIC_FILES: list[dict] = [
    {
        "component":   "full_stack",
        "filename":    "ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy",
        "rel_path":    "Seismic.ST10010/Stack/",
        "size_gb":     0.98,
        "description": "ST10010 final Kirchhoff PSDM full-stack time volume (primary PoC target)",
        "priority":    "required",
    },
    {
        "component":   "far_stack",
        "filename":    "ST10010ZC11_PZ_PSDM_KIRCH_FAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy",
        "rel_path":    "Seismic.ST10010/Stack/",
        "size_gb":     0.98,
        "description": "ST10010 FAR angle stack (AVO analysis)",
        "priority":    "optional",
    },
    {
        "component":   "near_stack",
        "filename":    "ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy",
        "rel_path":    "Seismic.ST10010/Stack/",
        "size_gb":     0.85,
        "description": "ST10010 NEAR angle stack",
        "priority":    "optional",
    },
    {
        "component":   "mid_stack",
        "filename":    "ST10010ZC11_PZ_PSDM_KIRCH_MID_T.MIG_FIN.POST_STACK.3D.JS-017536.segy",
        "rel_path":    "Seismic.ST10010/Stack/",
        "size_gb":     0.98,
        "description": "ST10010 MID angle stack",
        "priority":    "optional",
    },
    {
        "component":   "velocity",
        "filename":    "ST10010ZC11_PZ_PSDM_MIG_VEL.segy",
        "rel_path":    "Seismic.ST10010/Velocity/",
        "size_gb":     0.03,
        "description": "Migration velocity cube",
        "priority":    "optional",
    },
]

WELL_FILES: list[dict] = [
    {
        "well":        "15/9-19A",
        "filename":    "15_9-19A.las",
        "rel_path":    "Well_logs/15_9-19A/",
        "description": "Primary producer; complete log suite (GR, RHOB, NPHI, DT, RT)",
    },
    {
        "well":        "15/9-19BT2",
        "filename":    "15_9-19BT2.las",
        "rel_path":    "Well_logs/15_9-19BT2/",
        "description": "Oil producer",
    },
    {
        "well":        "15/9-19SR",
        "filename":    "15_9-19SR.las",
        "rel_path":    "Well_logs/15_9-19SR/",
        "description": "Side-track injector",
    },
]

INTERPRETATION_FILES: list[dict] = [
    {
        "filename":    "Volve_Fault_Sticks.txt",
        "rel_path":    "Interpretations/",
        "description": "Petrel fault-stick export - input to label_generator.py",
    },
    {
        "filename":    "Volve_Horizons.txt",
        "rel_path":    "Interpretations/",
        "description": "Mapped horizons (Base Cretaceous, Hugin Top, Hugin Base)",
    },
]


# ---------------------------------------------------------------------------
# Synthetic SEG-Y generator
# ---------------------------------------------------------------------------

def _ricker_wavelet(n: int, freq_hz: float, sample_rate_ms: float) -> np.ndarray:
    """Ricker (Mexican hat) wavelet, centred, length n."""
    dt = sample_rate_ms / 1000.0
    t = (np.arange(n) - n // 2) * dt
    pi_sq = (np.pi * freq_hz * t) ** 2
    return ((1.0 - 2.0 * pi_sq) * np.exp(-pi_sq)).astype(np.float32)


def create_synthetic_segy(
    dest_path: Path,
    n_inlines: int = 100,
    n_crosslines: int = 200,
    n_samples: int = 500,
    sample_rate_ms: float = 4.0,
    seed: int = 42,
) -> Path:
    """Generate a synthetic 3-D SEG-Y compatible with the deepseismic2 pipeline.

    Default dimensions produce approximately 45 MB. The geometry mirrors real
    ST10010 (inline numbers 1001+, crossline numbers 1900+, 4 ms sample rate)
    so all pipeline code works identically against the synthetic file.

    Two Ricker wavelet reflectors simulate the Hugin Top (~700 ms) and Hugin
    Base (~1000 ms) horizons at realistic S/N ratios.

    Parameters
    ----------
    dest_path:       Output .segy path.
    n_inlines:       Number of inlines (default 100).
    n_crosslines:    Number of crosslines (default 200).
    n_samples:       Samples per trace (default 500, yielding 2000 ms TWT at 4 ms).
    sample_rate_ms:  Sample interval in milliseconds.
    seed:            NumPy random seed for reproducibility.
    """
    try:
        import segyio
    except ImportError:
        logger.error("segyio is required:  pip install segyio")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Survey numbers matching real ST10010 geometry (approximate)
    il_start, xl_start = 1001, 1900
    ilines = np.arange(il_start, il_start + n_inlines,    dtype=np.int32)
    xlines = np.arange(xl_start, xl_start + n_crosslines, dtype=np.int32)

    # Sample positions in ms (used by segyio to set n_samples)
    samples_ms = np.arange(n_samples, dtype=np.float32) * sample_rate_ms

    # Ricker wavelet kernel
    wavelet = _ricker_wavelet(51, 35.0, sample_rate_ms)
    hw = len(wavelet) // 2

    # Reflector positions in sample indices
    twt1 = int(n_samples * 0.35)  # ~700 ms - Hugin Top analogue
    twt2 = int(n_samples * 0.50)  # ~1000 ms - Hugin Base analogue

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format  = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
    spec.samples = samples_ms
    spec.ilines  = ilines
    spec.xlines  = xlines

    total = n_inlines * n_crosslines
    t0 = time.time()
    logger.info(
        "Creating synthetic SEG-Y: %d IL x %d XL x %d samples -> %s",
        n_inlines, n_crosslines, n_samples, dest_path,
    )

    with segyio.create(str(dest_path), spec) as f:
        f.bin.update(
            hdt=int(sample_rate_ms * 1000),
            dto=int(sample_rate_ms * 1000),
        )
        tr = 0
        for il in ilines:
            for xl in xlines:
                trace = np.zeros(n_samples, dtype=np.float32)

                for centre, base_amp in [(twt1, 0.50), (twt2, 0.25)]:
                    amp = base_amp + 0.15 * float(rng.standard_normal())
                    i0 = max(0,         centre - hw)
                    i1 = min(n_samples, centre + hw + 1)
                    w0 = max(0, hw - centre)
                    trace[i0:i1] += amp * wavelet[w0 : w0 + (i1 - i0)]

                trace += (0.015 * rng.standard_normal(n_samples)).astype(np.float32)

                # UTM 32N approximate coordinates for Volve block 15/9
                x_cm = int((456000.0 + (il - il_start) * 12.5) * 100)
                y_cm = int((6470000.0 + (xl - xl_start) * 12.5) * 100)

                f.trace[tr] = trace
                f.header[tr].update({
                    segyio.TraceField.INLINE_3D:            int(il),
                    segyio.TraceField.CROSSLINE_3D:         int(xl),
                    segyio.TraceField.TRACE_SEQUENCE_FILE:  tr + 1,
                    segyio.TraceField.FieldRecord:          int(il),
                    segyio.TraceField.CDP_X:                x_cm,
                    segyio.TraceField.CDP_Y:                y_cm,
                    segyio.TraceField.DelayRecordingTime: 0,
                })
                tr += 1

            if (il - il_start + 1) % 20 == 0:
                pct = tr / total * 100
                logger.info(
                    "  %d/%d inlines - %.0f%% (%.1fs)",
                    il - il_start + 1, n_inlines, pct, time.time() - t0,
                )

    size_mb = dest_path.stat().st_size / 1e6
    logger.info("Done: %.1f MB -> %s", size_mb, dest_path)
    return dest_path


def create_synthetic_fault_sticks(output_path: Path) -> None:
    """Write synthetic Petrel-format fault sticks compatible with label_generator.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synthetic Volve fault sticks (Petrel export format)",
        "# Format: FaultName  X  Y  Z_ms",
        "# Coordinates approximate for synthetic ST10010 geometry",
    ]
    il_start, xl_start = 1001, 1900
    # Main normal fault crossing the survey diagonally
    for step in range(0, 40, 2):
        il = 1041 + step
        xl = 1940 + int(step * 1.5)
        x = 456000.0 + (il - il_start) * 12.5
        y = 6470000.0 + (xl - xl_start) * 12.5
        z = 700.0 + step * 4.0
        lines.append(f"Main_Fault  {x:.1f}  {y:.1f}  {z:.1f}")
    # Antithetic fault
    for step in range(0, 20, 2):
        il = 1071 + step
        xl = 1950 - int(step * 0.8)
        x = 456000.0 + (il - il_start) * 12.5
        y = 6470000.0 + (xl - xl_start) * 12.5
        z = 800.0 + step * 3.5
        lines.append(f"Antithetic_Fault  {x:.1f}  {y:.1f}  {z:.1f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Fault sticks written: %s", output_path)


# ---------------------------------------------------------------------------
# Download infrastructure
# ---------------------------------------------------------------------------

def _sha256_quick(path: Path, chunk_mb: int = 4) -> str:
    """Fast partial SHA-256: first + last N MB (matches segy_loader._file_sha256)."""
    h = hashlib.sha256()
    chunk = chunk_mb * 1024 * 1024
    size = path.stat().st_size
    with path.open("rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk:
            fh.seek(-min(chunk, size), 2)
            h.update(fh.read())
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> bool:
    """Stream-download url to dest with progress logging.  Returns True on success."""
    try:
        import httpx
    except ImportError:
        logger.error("httpx not available.  pip install httpx")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Already present, skipping: %s", dest.name)
        return True

    logger.info("GET  %s", url)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            downloaded = 0
            t0 = last_log = time.time()

            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(chunk_size=1 << 20):  # 1 MB chunks
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log >= 5.0:
                        speed = downloaded / (now - t0) / 1e6
                        if total_bytes:
                            pct = downloaded / total_bytes * 100
                            logger.info(
                                "  %.0f%%  %.1f/%.1f MB  @  %.1f MB/s",
                                pct, downloaded / 1e6, total_bytes / 1e6, speed,
                            )
                        else:
                            logger.info("  %.1f MB  @  %.1f MB/s", downloaded / 1e6, speed)
                        last_log = now

        tmp.rename(dest)
        logger.info("Saved: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed: %s", exc)

    if tmp.exists():
        tmp.unlink()
    return False


# ---------------------------------------------------------------------------
# Component downloaders
# ---------------------------------------------------------------------------

def download_seismic(
    dest_dir: Path,
    base_url: str,
    components: list[str] | None = None,
) -> dict[str, bool]:
    """Download seismic SEG-Y files from base_url.

    Parameters
    ----------
    dest_dir:    Local root; raw SEG-Y files land in dest_dir/raw/.
    base_url:    Root URL from the Equinor portal or Databricks external location.
    components:  Subset of component names.  None = all.
    """
    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for entry in SEISMIC_FILES:
        if components and entry["component"] not in components:
            continue
        fname = entry["filename"]
        url   = f"{base_url.rstrip('/')}/{entry['rel_path'].strip('/')}/{fname}"
        results[fname] = _download_file(url, raw_dir / fname)
    return results


def download_wells(dest_dir: Path, base_url: str | None = None) -> dict[str, bool]:
    """Download LAS well log files."""
    wells_dir = dest_dir / "wells"
    wells_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for entry in WELL_FILES:
        fname = entry["filename"]
        dest  = wells_dir / fname
        ok = False
        if base_url:
            url = f"{base_url.rstrip('/')}/{entry['rel_path'].strip('/')}/{fname}"
            ok  = _download_file(url, dest)
        if not ok:
            logger.warning(
                "Could not download %s -- place it manually at %s\n"
                "  Portal: https://www.equinor.com/energy/volve-data-sharing",
                fname, dest,
            )
        results[fname] = ok
    return results


def download_interpretations(dest_dir: Path, base_url: str | None = None) -> dict[str, bool]:
    """Download fault interpretation and horizon files."""
    interp_dir = dest_dir / "interpretations"
    interp_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for entry in INTERPRETATION_FILES:
        fname = entry["filename"]
        dest  = interp_dir / fname
        ok = False
        if base_url:
            url = f"{base_url.rstrip('/')}/{entry['rel_path'].strip('/')}/{fname}"
            ok  = _download_file(url, dest)
        if not ok:
            logger.warning("Place %s manually at: %s", fname, dest)
        results[fname] = ok
    return results


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_files(dest_dir: Path) -> None:
    """Print a file-presence and size verification report."""
    print(f"\nVerification -- {dest_dir}")
    print("=" * 72)

    raw_dir = dest_dir / "raw"
    print("\nSeismic (raw/):")
    for entry in SEISMIC_FILES:
        path = raw_dir / entry["filename"]
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            expected_mb = entry["size_gb"] * 1024
            delta = abs(size_mb - expected_mb) / expected_mb * 100
            mark = "OK" if delta < 5 else f"size off {delta:.0f}%"
            print(f"  [{mark:<16}]  {path.name[:52]}  {size_mb:7.0f} MB")
        else:
            print(f"  [MISSING         ]  {entry['filename'][:52]}")

    print("\nWells:")
    wells_dir = dest_dir / "wells"
    for entry in WELL_FILES:
        path = wells_dir / entry["filename"]
        mark = f"OK  {path.stat().st_size / 1e3:.0f} kB" if path.exists() else "MISSING"
        print(f"  [{mark:<18}]  {entry['filename']}")

    print("\nInterpretations:")
    interp_dir = dest_dir / "interpretations"
    for entry in INTERPRETATION_FILES:
        path = interp_dir / entry["filename"]
        mark = f"OK  {path.stat().st_size / 1e3:.0f} kB" if path.exists() else "MISSING"
        print(f"  [{mark:<18}]  {entry['filename']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_volve.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    what = p.add_argument_group("what to acquire")
    what.add_argument(
        "--sample", action="store_true",
        help="Generate a synthetic ~45 MB SEG-Y for local testing (no download needed)",
    )
    what.add_argument("--seismic", action="store_true", help="Download ST10010 seismic SEG-Y files")
    what.add_argument("--wells", action="store_true", help="Download LAS well log files")
    what.add_argument("--interpretations", action="store_true", help="Download fault/horizon files")
    what.add_argument("--all", action="store_true",
                      help="Download seismic + wells + interpretations")
    what.add_argument(
        "--components", nargs="+",
        choices=[e["component"] for e in SEISMIC_FILES],
        metavar="COMPONENT",
        help="Specific seismic components (default: all).  Choices: "
             + ", ".join(e["component"] for e in SEISMIC_FILES),
    )

    src = p.add_argument_group("source")
    src.add_argument(
        "--base-url", metavar="URL",
        help=(
            "Root URL of Equinor Volve storage (portal SAS URL or Databricks path).\n"
            "Get it at: https://www.equinor.com/energy/volve-data-sharing"
        ),
    )

    p.add_argument("--dest", default="data/volve", metavar="DIR",
                   help="Local destination directory (default: data/volve)")
    p.add_argument("--verify", action="store_true",
                   help="Report on which files are present and their sizes")
    p.add_argument("--verbose", "-v", action="store_true")

    synth = p.add_argument_group("synthetic sample options (with --sample)")
    synth.add_argument("--sample-inlines",    type=int, default=100, metavar="N",
                       help="Number of inlines (default 100)")
    synth.add_argument("--sample-crosslines", type=int, default=200, metavar="N",
                       help="Number of crosslines (default 200)")
    synth.add_argument("--sample-n-samples",  type=int, default=500, metavar="N",
                       help="Samples per trace (default 500)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Destination: %s", dest)

    did_something = False

    # Synthetic sample
    if args.sample:
        did_something = True
        sample_path = dest / "synthetic_sample.segy"
        if sample_path.exists():
            logger.info(
                "Sample already exists: %s (%.1f MB)",
                sample_path, sample_path.stat().st_size / 1e6,
            )
        else:
            create_synthetic_segy(
                sample_path,
                n_inlines=args.sample_inlines,
                n_crosslines=args.sample_crosslines,
                n_samples=args.sample_n_samples,
            )
        # Also create synthetic fault sticks for testing the label pipeline
        fault_path = dest / "interpretations" / "Volve_Fault_Sticks_synthetic.txt"
        if not fault_path.exists():
            create_synthetic_fault_sticks(fault_path)

        print(f"\n  Synthetic sample ready: {sample_path}")
        print(f"  Size: {sample_path.stat().st_size / 1e6:.1f} MB")
        print("\n  Next steps:")
        print("    jupyter notebook notebooks/01_data_exploration.ipynb")
        print()
        print("  Or in Python:")
        print("    from deepseismic.ingest.segy_loader import load_segy")
        print(f"    ds, geom = load_segy('{sample_path}')")
        print("    print(geom)")

    # Real seismic download
    if args.seismic or args.all:
        did_something = True
        if not args.base_url:
            logger.error(
                "Seismic download requires --base-url.\n\n"
                "  Steps:\n"
                "  1. Accept terms at: https://www.equinor.com/energy/volve-data-sharing\n"
                "  2. Copy the storage root URL from your confirmation email\n"
                "  3. Re-run: python scripts/download_volve.py --seismic --base-url \"<URL>\"\n\n"
                "  Or use --sample for a synthetic test volume (no download needed)."
            )
            if not (args.wells or args.interpretations or args.verify or args.sample):
                return 1
        else:
            res = download_seismic(dest, args.base_url, components=args.components)
            n_ok = sum(res.values())
            logger.info("Seismic: %d / %d files downloaded", n_ok, len(res))

    # Well logs
    if args.wells or args.all:
        did_something = True
        res = download_wells(dest, base_url=args.base_url)
        n_ok = sum(res.values())
        logger.info("Wells: %d / %d files", n_ok, len(res))

    # Interpretations
    if args.interpretations or args.all:
        did_something = True
        res = download_interpretations(dest, base_url=args.base_url)
        n_ok = sum(res.values())
        logger.info("Interpretations: %d / %d files", n_ok, len(res))

    # Verify
    if args.verify:
        did_something = True
        verify_files(dest)

    if not did_something:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
