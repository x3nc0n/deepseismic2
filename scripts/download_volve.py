"""Download Volve dataset subset for the DeepSeismic2 PoC.

Supports three modes:
  --sample    Generate a synthetic ~50MB SEG-Y for immediate local testing
  --subset    Download specific components (seismic, wells, interpretations)
  --all       Download everything we need

Usage:
  python scripts/download_volve.py --sample              # Quick synthetic data
  python scripts/download_volve.py --subset seismic      # Real ST10010 volume
  python scripts/download_volve.py --subset wells        # Well logs only
  python scripts/download_volve.py --all                 # Everything
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

# Target directory for all Volve data
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "volve"

# Known Volve files and their approximate sizes (for validation)
VOLVE_MANIFEST = {
    "seismic": {
        "description": "ST10010 PSDM time volume (post-stack 3D)",
        "filename": "ST10010ZC11_PZ_PSDM_KIRCH_FAR_D.MIG_FIN.POST_STACK.3D.JS-017536.segy",
        "approx_size_mb": 1024,
        "subdir": "seismic",
    },
    "wells": {
        "description": "Well logs for key Volve wells (15/9-19A, 19BT2, 19SR)",
        "subdir": "wells",
        "files": [
            "15_9-19A/composite.las",
            "15_9-19BT2/composite.las",
            "15_9-19SR/composite.las",
        ],
    },
    "interpretations": {
        "description": "Fault sticks and horizon interpretations",
        "subdir": "interpretations/fault_sticks",
    },
}


def create_synthetic_segy(
    output_path: Path,
    *,
    n_ilines: int = 100,
    n_xlines: int = 200,
    n_samples: int = 500,
) -> None:
    """Generate a synthetic SEG-Y file with realistic seismic character.

    Creates layered reflectivity with synthetic faults for testing the full
    ingest → model pipeline without needing real data.
    """
    try:
        import segyio
    except ImportError:
        print("ERROR: segyio is required. Install with: pip install segyio")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create trace data with layered reflectivity + faults
    print(f"  Generating synthetic volume: {n_ilines}×{n_xlines}×{n_samples}")
    rng = np.random.default_rng(42)

    # Base reflectivity series (layered earth)
    base_reflectivity = np.zeros(n_samples, dtype=np.float32)
    layer_positions = [50, 120, 180, 250, 310, 380, 420, 460]
    for pos in layer_positions:
        base_reflectivity[pos] = rng.uniform(-0.3, 0.3)

    # Ricker wavelet for convolution
    def ricker(f: float, dt: float, length: int) -> np.ndarray:
        t = np.arange(-length // 2, length // 2 + 1) * dt
        pi2 = (np.pi * f * t) ** 2
        return (1 - 2 * pi2) * np.exp(-pi2)

    wavelet = ricker(25.0, 0.004, 64)

    # Generate volume with faults
    data = np.zeros((n_ilines, n_xlines, n_samples), dtype=np.float32)
    for il in range(n_ilines):
        for xl in range(n_xlines):
            # Create local reflectivity with fault offset
            local_r = base_reflectivity.copy()

            # Fault 1: normal fault dipping NE
            fault1_throw = 0
            if il > 40 and xl > 80:
                fault1_throw = int(15 * min((il - 40) / 60, 1.0) * min((xl - 80) / 120, 1.0))

            # Fault 2: smaller antithetic fault
            fault2_throw = 0
            if il > 70 and xl < 60:
                fault2_throw = int(-8 * min((il - 70) / 30, 1.0))

            total_throw = fault1_throw + fault2_throw
            if total_throw != 0:
                local_r = np.roll(local_r, total_throw)

            # Convolve with wavelet + add noise
            trace = np.convolve(local_r, wavelet, mode="same")
            trace += rng.normal(0, 0.02, n_samples).astype(np.float32)
            data[il, xl, :] = trace

    # Write SEG-Y
    spec = segyio.spec()
    spec.sorting = 2  # crossline sorting
    spec.format = 1  # IBM float
    spec.samples = np.arange(n_samples) * 4.0  # 4ms sample rate
    spec.ilines = np.arange(1, n_ilines + 1)
    spec.xlines = np.arange(1, n_xlines + 1)

    print(f"  Writing SEG-Y to: {output_path}")
    with segyio.create(str(output_path), spec) as f:
        for il_idx, il_no in enumerate(spec.ilines):
            for xl_idx, xl_no in enumerate(spec.xlines):
                trace_idx = il_idx * n_xlines + xl_idx
                f.trace[trace_idx] = data[il_idx, xl_idx, :]
                f.header[trace_idx].update({
                    segyio.TraceField.INLINE_3D: int(il_no),
                    segyio.TraceField.CROSSLINE_3D: int(xl_no),
                    segyio.TraceField.CDP_X: 450000 + int(il_no) * 25,
                    segyio.TraceField.CDP_Y: 6470000 + int(xl_no) * 25,
                })

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Created: {output_path.name} ({size_mb:.1f} MB)")


def create_synthetic_fault_sticks(output_path: Path) -> None:
    """Generate synthetic fault interpretation files matching our sample volume."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Fault 1: major normal fault (matches synthetic SEG-Y)
    fault1_lines = [
        "# Fault interpretation: Main_Normal_Fault",
        "# Format: inline crossline z_ms",
        "# Exported from synthetic interpretation",
    ]
    for il in range(45, 100, 5):
        xl = int(80 + (il - 40) * 0.8)
        z = int(200 + (il - 40) * 0.5)
        fault1_lines.append(f"{il} {xl} {z}")

    fault1_path = output_path / "fault_main_normal.dat"
    fault1_path.write_text("\n".join(fault1_lines))

    # Fault 2: antithetic
    fault2_lines = [
        "# Fault interpretation: Antithetic_Fault",
        "# Format: inline crossline z_ms",
    ]
    for il in range(72, 100, 4):
        xl = int(55 - (il - 70) * 0.3)
        z = int(300 + (il - 70) * 0.3)
        fault2_lines.append(f"{il} {xl} {z}")

    fault2_path = output_path / "fault_antithetic.dat"
    fault2_path.write_text("\n".join(fault2_lines))
    print(f"  ✅ Created fault sticks: {fault1_path.name}, {fault2_path.name}")


def create_sample_data() -> None:
    """Create a complete synthetic sample dataset for local testing."""
    print("\n🔬 Creating synthetic Volve sample data...\n")

    # Synthetic SEG-Y
    segy_path = DATA_DIR / "seismic" / "sample_volume.segy"
    create_synthetic_segy(segy_path, n_ilines=100, n_xlines=200, n_samples=500)

    # Synthetic fault interpretations
    faults_dir = DATA_DIR / "interpretations" / "fault_sticks"
    create_synthetic_fault_sticks(faults_dir)

    # README with attribution
    readme = DATA_DIR / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# Volve Sample Data\n\n"
        "This directory contains synthetic sample data generated for local testing.\n\n"
        "**For real Volve data:**\n"
        "- Databricks Marketplace (if you have access)\n"
        "- Equinor: https://www.equinor.com/energy/volve-data-sharing\n\n"
        "## Attribution\n\n"
        "> Real Volve data courtesy of Equinor and the Volve license partners,\n"
        "> released under CC BY-NC-SA 4.0.\n"
    )

    print(f"\n✅ Sample data ready at: {DATA_DIR}")
    print(
        "   Run the pipeline: python -c "
        "\"from deepseismic.ingest.segy_loader import segy_to_zarr; "
        "segy_to_zarr('data/volve/seismic/sample_volume.segy', "
        "'data/volve/zarr/sample/')\""
    )

def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file (for verification)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_existing_data() -> dict[str, bool]:
    """Check which data components are already present."""
    status = {}
    for key, info in VOLVE_MANIFEST.items():
        subdir = DATA_DIR / info["subdir"]
        if key == "seismic":
            has_real = (subdir / info["filename"]).exists()
            has_sample = (DATA_DIR / "seismic" / "sample_volume.segy").exists()
            status[key] = has_real or has_sample
        else:
            status[key] = subdir.exists() and any(subdir.rglob("*"))
    return status


def download_component(component: str) -> None:
    """Download a specific component from Equinor's data portal.

    Note: Equinor's download requires web authentication. This function
    provides instructions rather than automated download since the portal
    requires user consent for the CC BY-NC-SA 4.0 license.
    """
    info = VOLVE_MANIFEST.get(component)
    if not info:
        print(f"ERROR: Unknown component '{component}'. Options: {list(VOLVE_MANIFEST.keys())}")
        sys.exit(1)

    print(f"\n📥 Component: {info['description']}")
    print(f"   Target dir: {DATA_DIR / info['subdir']}")
    print()
    print("   ⚠️  Equinor's portal requires manual license acceptance.")
    print("   Steps:")
    print("   1. Visit: https://www.equinor.com/energy/volve-data-sharing")
    print("   2. Accept the CC BY-NC-SA 4.0 license")
    print("   3. Download the seismic data subset")
    if component == "seismic":
        print(f"   4. Place '{info['filename']}' in:")
        print(f"      {DATA_DIR / info['subdir']}/")
    print()
    print("   Alternatively, if you have Databricks access:")
    print("   - Use scripts/databricks_export.py to export from the Marketplace")
    print("   - See docs/volve-data-acquisition.md for full instructions")

    # Create target directory
    target = DATA_DIR / info["subdir"]
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n   📁 Created: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/generate Volve dataset for DeepSeismic2 PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sample                    # Generate synthetic test data (~50MB)
  %(prog)s --subset seismic            # Instructions for real seismic data
  %(prog)s --subset wells              # Instructions for well logs
  %(prog)s --all                       # Instructions for everything
  %(prog)s --status                    # Check what's already downloaded
        """,
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate synthetic sample data for testing",
    )
    parser.add_argument(
        "--subset", choices=["seismic", "wells", "interpretations"],
        help="Download specific component",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all components",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check what data is available locally",
    )

    args = parser.parse_args()

    if args.status:
        print("\n📊 Local data status:\n")
        status = check_existing_data()
        for component, present in status.items():
            icon = "✅" if present else "❌"
            print(f"   {icon} {component}: {VOLVE_MANIFEST[component]['description']}")
        return

    if args.sample:
        create_sample_data()
        return

    if args.subset:
        download_component(args.subset)
        return

    if args.all:
        for component in VOLVE_MANIFEST:
            download_component(component)
        return

    # Default: show status + suggest action
    parser.print_help()
    print("\n\n📊 Current data status:")
    status = check_existing_data()
    has_any = any(status.values())
    if not has_any:
        print("   No data found. Try: python scripts/download_volve.py --sample")


if __name__ == "__main__":
    main()

