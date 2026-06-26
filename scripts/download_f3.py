#!/usr/bin/env python3
"""Acquire / scaffold the F3 (Netherlands offshore) dataset for cross-survey training.

Why F3 (issue #24)
------------------
The shipped fault model was trained on **synthetic** seismic and over-predicts on
real Volve data. To evaluate fault detection **without label leakage**, #24 adopts
*cross-survey transfer*: train on **F3**, then blind-infer on **Volve**, using the
Volve fault sticks **only as a scoring key** — never as training input.

F3 and Volve are geophysically different (this is the point — it is an honest test):

============  ==========================================  ============================
Property      F3 (Dutch sector, S. North Sea)             Volve / ST10010 (Norwegian)
============  ==========================================  ============================
Geology       Tertiary prograding deltaic clinoforms;     Jurassic Hugin reservoir in a
              dense small **polygonal** faulting +         **domal horst** bounded by
              shallow gas / bright spots                  large normal faults; salt below
Acquisition   older (~1987), **lower frequency**          modern (2010s), higher freq
Fault style   many small polygonal faults                 few large normal faults
============  ==========================================  ============================

The model must bridge bandwidth/wavelet, amplitude statistics, fault style/scale and
structure — handled at train time by spectral/amplitude domain-normalization + heavy
augmentation (a *separate* #24 step, not this ingest).

Data acquisition contract (REAL data — manual, licensed)
--------------------------------------------------------
F3 is public but must be obtained from the source under its license. This script does
**not** bundle it. Obtain and drop the files into the layout below, then ingest.

  Amplitude (3-D seismic, time):
    Source : OpendTect F3 Demo (dGB Earth Sciences / TerraNubis).
             https://terranubis.com/datainfo/F3-Demo-2020  (free account; CC BY-SA).
    Format : SEG-Y (or OpendTect survey export to SEG-Y).
    Geometry (real F3 Demo): inlines 100-750, crosslines 300-1250,
             ~462 samples @ 4 ms (1848 ms TWT).
    Drop to: data/f3/raw/f3_seismic.segy

  Fault labels (ground truth):
    Option A (sticks)  : OpendTect F3 fault interpretation exported as fault sticks.
                         Parse with deepseismic.ingest.label_generator
                         .parse_opendtect_fault_sticks, then rasterise.
    Option B (volume)  : a public dense F3 fault label volume (e.g. ML-competition
                         derivatives) as SEG-Y/NumPy on the SAME grid as the amplitude.
                         Ingest via a label-volume aligner (TODO when data lands).
    Drop sticks to     : data/f3/interpretations/fault_sticks/*.dat
                         (3-col index format: inline_idx crossline_idx z_sample_idx,
                          0-based on the amplitude grid — same format as the Volve sticks
                          already in this repo, consumed unchanged by
                          scripts/generate_fault_label.py).

Repository layout (created by this script)
------------------------------------------
  data/f3/raw/                              # source SEG-Y (gitignored)
  data/f3/interpretations/fault_sticks/     # fault-stick .dat files (gitignored)
  data/f3/staged/                           # amplitude.zarr + fault_label.zarr (gitignored)

Ingest pipeline (reuses the existing, tested scripts unchanged)
---------------------------------------------------------------
  # 1. amplitude SEG-Y -> zarr (+ JSON geometry sidecar)
  python scripts/ingest_segy.py \\
      --source data/f3/raw/f3_seismic.segy \\
      --dest   data/f3/staged/amplitude.zarr \\
      --survey-id f3-demo --overwrite

  # 2. fault sticks -> co-registered label zarr (geometry from the sidecar)
  python scripts/generate_fault_label.py \\
      --fault-stick-dir data/f3/interpretations/fault_sticks \\
      --amplitude-json  data/f3/staged/amplitude.json \\
      --label-output    data/f3/staged/fault_label.zarr --overwrite

Local pipeline validation (NO real data — clearly-marked format proxy)
----------------------------------------------------------------------
  python scripts/download_f3.py --sample        # writes a small synthetic F3 proxy
  # then run the two ingest commands above -> proves co-registration end-to-end.

  ##  WARNING: --sample output is a SYNTHETIC FORMAT PROXY for pipeline validation.
  ##  It is NOT real F3 data and must NEVER be used as training/eval ground truth.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("f3-dl")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_F3_ROOT = _REPO_ROOT / "data" / "f3"
_RAW_DIR = _F3_ROOT / "raw"
_STICK_DIR = _F3_ROOT / "interpretations" / "fault_sticks"
_STAGED_DIR = _F3_ROOT / "staged"

# ---------------------------------------------------------------------------
# Real-data catalog (documentation; this script does not download these)
# ---------------------------------------------------------------------------
CATALOG: list[dict] = [
    {
        "component": "amplitude",
        "drop_to": "data/f3/raw/f3_seismic.segy",
        "source": "OpendTect F3 Demo (dGB / TerraNubis)",
        "url": "https://terranubis.com/datainfo/F3-Demo-2020",
        "license": "CC BY-SA (dGB F3 Demo terms)",
        "format": "SEG-Y (time)",
        "geometry": "IL 100-750, XL 300-1250, ~462 samples @ 4 ms",
        "priority": "required",
    },
    {
        "component": "fault_labels",
        "drop_to": "data/f3/interpretations/fault_sticks/*.dat",
        "source": "OpendTect F3 fault interpretation (sticks) OR public F3 fault label volume",
        "url": "https://terranubis.com/datainfo/F3-Demo-2020",
        "license": "per source",
        "format": "OpendTect fault sticks  |  label volume (SEG-Y/NumPy on amplitude grid)",
        "geometry": "must co-register to the amplitude grid",
        "priority": "required",
    },
]

# Real F3 Demo geometry (for documentation / proxy realism)
F3_GEOMETRY = {
    "inline_min": 100,
    "crossline_min": 300,
    "sample_rate_ms": 4.0,
}


# ---------------------------------------------------------------------------
# Synthetic format proxy (pipeline validation only — NOT real F3)
# ---------------------------------------------------------------------------
def _ricker_wavelet(n: int, freq_hz: float, sample_rate_ms: float) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet, centred, length n."""
    dt = sample_rate_ms / 1000.0
    t = (np.arange(n) - n // 2) * dt
    pi_sq = (np.pi * freq_hz * t) ** 2
    return ((1.0 - 2.0 * pi_sq) * np.exp(-pi_sq)).astype(np.float32)


def create_proxy_segy(
    dest_path: Path,
    *,
    n_inlines: int = 100,
    n_crosslines: int = 200,
    n_samples: int = 300,
    sample_rate_ms: float = 4.0,
    freq_hz: float = 28.0,
    seed: int = 24,
) -> Path:
    """Write a small synthetic SEG-Y on F3-like geometry for pipeline validation.

    Deliberately lower-frequency than the Volve proxy (``freq_hz`` default 28 Hz vs
    35 Hz) to echo F3's older, lower-bandwidth acquisition — so the cross-survey
    domain gap is visible even in the proxy. This is **not** real F3 data.
    """
    try:
        import segyio
    except ImportError:
        logger.error("segyio is required:  pip install segyio")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    il_start = F3_GEOMETRY["inline_min"]
    xl_start = F3_GEOMETRY["crossline_min"]
    ilines = np.arange(il_start, il_start + n_inlines, dtype=np.int32)
    xlines = np.arange(xl_start, xl_start + n_crosslines, dtype=np.int32)
    samples_ms = np.arange(n_samples, dtype=np.float32) * sample_rate_ms

    wavelet = _ricker_wavelet(51, freq_hz, sample_rate_ms)
    hw = len(wavelet) // 2

    # A few prograding-clinoform-style reflectors (F3 flavour)
    reflectors = [
        (int(n_samples * 0.30), 0.45),
        (int(n_samples * 0.45), 0.35),
        (int(n_samples * 0.62), 0.25),
    ]

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
    spec.samples = samples_ms
    spec.ilines = ilines
    spec.xlines = xlines

    logger.info(
        "Creating SYNTHETIC F3-PROXY SEG-Y: %d IL x %d XL x %d samples -> %s",
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
                # Gentle dip across crosslines to mimic clinoform geometry
                dip = int((xl - xl_start) * 0.05)
                for centre, base_amp in reflectors:
                    c = min(n_samples - 1, centre + dip)
                    amp = base_amp + 0.12 * float(rng.standard_normal())
                    i0 = max(0, c - hw)
                    i1 = min(n_samples, c + hw + 1)
                    w0 = max(0, hw - c)
                    trace[i0:i1] += amp * wavelet[w0 : w0 + (i1 - i0)]
                trace += (0.02 * rng.standard_normal(n_samples)).astype(np.float32)

                f.trace[tr] = trace
                f.header[tr].update({
                    segyio.TraceField.INLINE_3D: int(il),
                    segyio.TraceField.CROSSLINE_3D: int(xl),
                    segyio.TraceField.TRACE_SEQUENCE_FILE: tr + 1,
                    segyio.TraceField.FieldRecord: int(il),
                    segyio.TraceField.DelayRecordingTime: 0,
                })
                tr += 1

    size_mb = dest_path.stat().st_size / 1e6
    logger.info("Done: %.1f MB -> %s", size_mb, dest_path)
    return dest_path


def create_proxy_fault_sticks(
    stick_dir: Path,
    *,
    n_inlines: int = 100,
    n_crosslines: int = 200,
    n_samples: int = 300,
) -> list[Path]:
    """Write synthetic fault sticks in the 3-col **index** .dat format.

    Columns: ``inline_idx  crossline_idx  z_sample_idx`` (all 0-based on the
    amplitude grid), identical to the Volve sticks already in this repo and
    consumed unchanged by ``scripts/generate_fault_label.py``.

    Emits several small faults to echo F3's busier polygonal fault fabric (vs
    Volve's few large normals). **Proxy only — not real F3 interpretation.**
    """
    stick_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write(name: str, points: list[tuple[int, int, int]]) -> None:
        path = stick_dir / name
        lines = [
            "# SYNTHETIC F3-PROXY fault stick (NOT real F3 interpretation)",
            "# Format: inline_idx  crossline_idx  z_sample_idx  (0-based grid indices)",
        ]
        for il, xl, z in points:
            il = int(np.clip(il, 0, n_inlines - 1))
            xl = int(np.clip(xl, 0, n_crosslines - 1))
            z = int(np.clip(z, 0, n_samples - 1))
            lines.append(f"{il}  {xl}  {z}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)

    # A larger normal fault crossing the cube diagonally
    main = [
        (il, int(n_crosslines * 0.30) + int(il * 0.8), int(n_samples * 0.30) + il)
        for il in range(0, n_inlines, 2)
    ]
    _write("f3_proxy_main.dat", main)

    # A couple of smaller polygonal-style faults at different IL bands
    for k, (il0, il1, xl_frac, z_frac) in enumerate(
        [(10, 40, 0.55, 0.40), (50, 80, 0.70, 0.55), (20, 55, 0.45, 0.62)]
    ):
        seg = [
            (il, int(n_crosslines * xl_frac) + (il - il0) // 2,
             int(n_samples * z_frac) + (il - il0))
            for il in range(il0, min(il1, n_inlines))
        ]
        _write(f"f3_proxy_minor_{k}.dat", seg)

    logger.info("Wrote %d proxy fault-stick files -> %s", len(written), stick_dir)
    return written


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def print_catalog() -> None:
    print("=" * 72)
    print("  F3 DATA ACQUISITION CONTRACT  (issue #24 — cross-survey training)")
    print("=" * 72)
    for item in CATALOG:
        print(f"\n  [{item['priority'].upper()}] {item['component']}")
        print(f"    source  : {item['source']}")
        print(f"    url     : {item['url']}")
        print(f"    license : {item['license']}")
        print(f"    format  : {item['format']}")
        print(f"    geometry: {item['geometry']}")
        print(f"    drop to : {item['drop_to']}")
    print("\n  After dropping real files, run the two ingest commands in the module")
    print("  docstring (ingest_segy.py + generate_fault_label.py).")
    print("=" * 72)


def make_sample(overwrite: bool) -> int:
    segy_path = _RAW_DIR / "f3_synthetic_proxy.segy"
    if segy_path.exists() and not overwrite:
        logger.error("%s exists. Pass --overwrite to regenerate.", segy_path)
        return 1
    if segy_path.exists():
        segy_path.unlink()

    print("#" * 72)
    print("#  SYNTHETIC F3 FORMAT PROXY — pipeline validation only.")
    print("#  NOT real F3 data. NEVER use as training/eval ground truth (#24).")
    print("#" * 72)

    create_proxy_segy(segy_path)
    create_proxy_fault_sticks(_STICK_DIR)
    _STAGED_DIR.mkdir(parents=True, exist_ok=True)

    # Drop a marker so anything reading data/f3 knows this is proxy, not real.
    marker = _F3_ROOT / "PROXY_DATA_DO_NOT_USE_AS_GROUND_TRUTH.txt"
    marker.write_text(
        "This data/f3 tree currently holds a SYNTHETIC FORMAT PROXY generated by\n"
        "scripts/download_f3.py --sample. It is NOT real F3 data and must not be\n"
        "used as training or evaluation ground truth. Replace with real F3 data\n"
        "per the acquisition contract in scripts/download_f3.py, then delete this file.\n",
        encoding="utf-8",
    )

    print("\nProxy written:")
    print(f"  SEG-Y       : {segy_path}")
    print(f"  Fault sticks: {_STICK_DIR}")
    print("\nNext (validate the pipeline end-to-end):")
    print("  python scripts/ingest_segy.py --source data/f3/raw/f3_synthetic_proxy.segy \\")
    print("      --dest data/f3/staged/amplitude.zarr --survey-id f3-proxy --overwrite")
    print("  python scripts/generate_fault_label.py \\")
    print("      --fault-stick-dir data/f3/interpretations/fault_sticks \\")
    print("      --amplitude-json data/f3/staged/amplitude.json \\")
    print("      --label-output data/f3/staged/fault_label.zarr --overwrite")
    return 0


def verify() -> int:
    """Report what F3 assets are present and whether they look real or proxy."""
    print("=" * 72)
    print("  F3 ASSET VERIFICATION")
    print("=" * 72)
    proxy_marker = _F3_ROOT / "PROXY_DATA_DO_NOT_USE_AS_GROUND_TRUTH.txt"
    is_proxy = proxy_marker.exists()
    checks = {
        "raw SEG-Y (real)": _RAW_DIR / "f3_seismic.segy",
        "raw SEG-Y (proxy)": _RAW_DIR / "f3_synthetic_proxy.segy",
        "fault sticks dir": _STICK_DIR,
        "staged amplitude.zarr": _STAGED_DIR / "amplitude.zarr",
        "staged fault_label.zarr": _STAGED_DIR / "fault_label.zarr",
    }
    for label, path in checks.items():
        mark = "✓" if path.exists() else "✗"
        print(f"  [{mark}] {label:28s} {path}")
    n_sticks = len(list(_STICK_DIR.glob("*.dat"))) if _STICK_DIR.exists() else 0
    print(f"\n  Fault-stick .dat files: {n_sticks}")
    if is_proxy:
        print("\n  ⚠  PROXY DATA PRESENT — current data/f3 is a synthetic format proxy,")
        print("     NOT real F3. Replace with real data before training/eval (#24).")
    else:
        print("\n  No proxy marker — treat present files as real per the contract.")
    print("=" * 72)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--catalog", action="store_true",
                        help="Print the F3 data acquisition contract and exit.")
    parser.add_argument("--sample", action="store_true",
                        help="Generate a small SYNTHETIC F3 format proxy for pipeline "
                             "validation (NOT real data).")
    parser.add_argument("--verify", action="store_true",
                        help="Report which F3 assets are present (real vs proxy).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing proxy SEG-Y when using --sample.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.catalog:
        print_catalog()
        return 0
    if args.sample:
        return make_sample(args.overwrite)
    if args.verify:
        return verify()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
