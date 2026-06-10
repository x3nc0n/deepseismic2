"""SEG-Y ingest: load, geometry extraction, Zarr export, metadata sidecar.

Implements the full ingest pipeline for the deepseismic2 PoC:

    SEG-Y file  →  SEGYLoader  →  xarray.Dataset  →  Zarr store + JSON sidecar

Key design decisions
--------------------
- segyio handles all low-level SEG-Y parsing (IBM float, byte-order, trace headers).
- Memory-mapped reads (``f.mmap()``) give sequential-scan performance without loading
  the full file into RAM.
- Sample mode (``--sample-mode``) loads only the first *N* inlines for rapid local
  iteration without the full ~1 GB ST10010 volume.
- Zarr chunks default to (64, 64, 128) — a good starting point for both local
  random access and cloud object-storage prefetch patterns.
- The JSON sidecar is written alongside the Zarr store so pipeline stages
  downstream can read survey bounds, stats, and provenance without opening the
  array data.

Usage
-----
    from deepseismic.ingest.segy_loader import segy_to_zarr

    meta = segy_to_zarr(
        "data/raw/ST10010_PSDM_TIME.segy",
        "data/staged/ST10010.zarr",
        sample_mode=True,
        sample_n_inlines=50,
    )
    print(meta.geometry["n_inlines"], meta.amplitude_stats["p99"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import IO

import numpy as np
import segyio
import xarray as xr
import zarr

logger = logging.getLogger(__name__)

# Default chunk shape: (inline, crossline, sample).
# 64 × 64 × 128 balances random-access latency against storage overhead
# for typical cloud object-storage (Azure Blob 4-MB block alignment).
DEFAULT_CHUNKS: tuple[int, int, int] = (64, 64, 128)


# ---------------------------------------------------------------------------
# Survey geometry
# ---------------------------------------------------------------------------


@dataclass
class SurveyGeometry:
    """Full spatial grid description for a 3-D seismic survey.

    All index ranges are *inclusive* and follow SEG-Y header conventions
    (inline/crossline numbers as recorded, not 0-based indices).
    """

    inline_min: int
    inline_max: int
    inline_step: int
    crossline_min: int
    crossline_max: int
    crossline_step: int
    sample_rate_ms: float   # milliseconds per sample (time domain)
    n_samples: int
    n_inlines: int
    n_crosslines: int
    datum_ms: float = 0.0  # DelayRecordingTime from trace header byte 109

    @property
    def inlines(self) -> np.ndarray:
        """Inline number array matching the volume's first axis."""
        return np.arange(self.inline_min, self.inline_max + 1, self.inline_step)

    @property
    def crosslines(self) -> np.ndarray:
        """Crossline number array matching the volume's second axis."""
        return np.arange(self.crossline_min, self.crossline_max + 1, self.crossline_step)

    @property
    def times_ms(self) -> np.ndarray:
        """Two-way time axis in milliseconds."""
        return self.datum_ms + np.arange(self.n_samples) * self.sample_rate_ms

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestMetadata:
    """Sidecar metadata written as JSON alongside every Zarr export."""

    source_file: str
    source_sha256: str | None
    ingested_at: str
    sample_mode: bool
    n_inlines_loaded: int
    geometry: dict
    amplitude_stats: dict
    zarr_path: str
    zarr_chunks: tuple[int, int, int]
    extra: dict = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        d = asdict(self)
        d["zarr_chunks"] = list(d["zarr_chunks"])
        return json.dumps(d, indent=indent, default=str)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


class SEGYLoader:
    """Load a SEG-Y volume, extract geometry, and export as Zarr + JSON sidecar.

    Supports both file-path and bytes/stream inputs.  Use as a context manager
    to ensure temp-file cleanup when loading from bytes.

    Parameters
    ----------
    path_or_bytes:
        Path to a SEG-Y file *or* raw ``bytes`` / binary file-like object.
        Stream inputs are written to a temporary file so segyio can mmap them.
    sample_mode:
        Load only the first ``sample_n_inlines`` inlines.  Intended for local
        dev without the full dataset.
    sample_n_inlines:
        Inline count when ``sample_mode=True``.
    ignore_geometry:
        Passed to ``segyio.open(ignore_geometry=...)`` for non-standard files
        that lack proper inline/crossline byte locations.
    """

    def __init__(
        self,
        path_or_bytes: str | Path | bytes | IO[bytes],
        *,
        sample_mode: bool = False,
        sample_n_inlines: int = 50,
        ignore_geometry: bool = False,
    ) -> None:
        self._source = path_or_bytes
        self.sample_mode = sample_mode
        self.sample_n_inlines = sample_n_inlines
        self.ignore_geometry = ignore_geometry

        self._tmp_path: Path | None = None
        self._segy_path: Path | None = None

    # --- context manager ---------------------------------------------------

    def __enter__(self) -> "SEGYLoader":
        self._resolve_path()
        return self

    def __exit__(self, *_: object) -> None:
        self._cleanup()

    def _resolve_path(self) -> None:
        src = self._source
        if isinstance(src, (str, Path)):
            self._segy_path = Path(src)
            return

        # Materialise bytes / IO stream → temp file for segyio to mmap
        if isinstance(src, bytes):
            raw = src
        elif isinstance(src, BytesIO):
            raw = src.read()
        else:
            raw = src.read()  # type: ignore[union-attr]

        tmp_dir = Path(tempfile.gettempdir())
        fd, tmp_name = tempfile.mkstemp(suffix=".segy", dir=tmp_dir)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)
        self._tmp_path = Path(tmp_name)
        self._segy_path = self._tmp_path

    def _cleanup(self) -> None:
        if self._tmp_path and self._tmp_path.exists():
            self._tmp_path.unlink(missing_ok=True)

    # --- public API --------------------------------------------------------

    def load(self) -> tuple[xr.Dataset, SurveyGeometry]:
        """Read the SEG-Y into an xarray Dataset.

        Returns
        -------
        ds:
            ``xr.Dataset`` with a single variable ``amplitude``
            dimensioned ``(inline, crossline, twtt_ms)``.
        geom:
            Extracted :class:`SurveyGeometry`.

        Raises
        ------
        RuntimeError
            If called outside a context manager before ``__enter__``.
        """
        if self._segy_path is None:
            self._resolve_path()

        logger.info("Opening SEG-Y: %s", self._segy_path)

        with segyio.open(str(self._segy_path), ignore_geometry=self.ignore_geometry) as f:
            f.mmap()  # memory-map for faster sequential reads
            geom = self._extract_geometry(f)
            data = self._read_traces(f, geom)

        logger.info(
            "Loaded volume shape (IL × XL × S): %s  dtype=%s",
            data.shape,
            data.dtype,
        )
        return self._to_xarray(data, geom), geom

    def to_zarr(
        self,
        output_path: str | Path,
        chunks: tuple[int, int, int] = DEFAULT_CHUNKS,
        *,
        overwrite: bool = False,
    ) -> tuple[zarr.Array, IngestMetadata]:
        """Load the SEG-Y and write a chunked Zarr store plus a JSON sidecar.

        Parameters
        ----------
        output_path:
            Directory path for the Zarr store.
        chunks:
            Chunk shape ``(inline, crossline, sample)``.
        overwrite:
            Overwrite an existing store.

        Returns
        -------
        z : zarr.Array
            The written ``amplitude`` array.
        meta : IngestMetadata
            Fully populated sidecar metadata.
        """
        ds, geom = self.load()
        output_path = Path(output_path)

        amplitude: np.ndarray = ds["amplitude"].values  # (IL, XL, T)

        store = zarr.DirectoryStore(str(output_path))
        root = zarr.open_group(store, mode="w" if overwrite else "w-")

        z = root.create_dataset(
            "amplitude",
            data=amplitude,
            chunks=chunks,
            dtype=np.float32,
            compressor=zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.SHUFFLE),
            overwrite=overwrite,
        )
        logger.info("Wrote amplitude array → %s  shape=%s", output_path, z.shape)

        # Store coordinate 1-D arrays alongside the data for easy access
        root.create_dataset("inline",    data=geom.inlines[:amplitude.shape[0]],  dtype=np.int32)
        root.create_dataset("crossline", data=geom.crosslines[:amplitude.shape[1]], dtype=np.int32)
        root.create_dataset("twtt_ms",   data=geom.times_ms,                      dtype=np.float32)

        amp_stats = _compute_amplitude_stats(amplitude)

        src_name = self._segy_path.name if self._segy_path else "<bytes>"
        sha256 = None
        if self._segy_path and self._segy_path.exists() and self._segy_path != self._tmp_path:
            sha256 = _file_sha256(self._segy_path, quick=True)

        meta = IngestMetadata(
            source_file=src_name,
            source_sha256=sha256,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            sample_mode=self.sample_mode,
            n_inlines_loaded=amplitude.shape[0],
            geometry=geom.to_dict(),
            amplitude_stats=amp_stats,
            zarr_path=str(output_path),
            zarr_chunks=chunks,
        )

        sidecar_path = output_path.with_suffix(".json")
        sidecar_path.write_text(meta.to_json(), encoding="utf-8")
        logger.info("Wrote metadata sidecar → %s", sidecar_path)

        return z, meta

    # --- internal helpers --------------------------------------------------

    def _extract_geometry(self, f: segyio.SegyFile) -> SurveyGeometry:
        inlines = f.ilines
        crosslines = f.xlines

        il_min  = int(inlines[0])
        il_max  = int(inlines[-1])
        il_step = int(inlines[1] - inlines[0]) if len(inlines) > 1 else 1

        xl_min  = int(crosslines[0])
        xl_max  = int(crosslines[-1])
        xl_step = int(crosslines[1] - crosslines[0]) if len(crosslines) > 1 else 1

        # segyio reports sample interval in microseconds
        sample_rate_ms = segyio.tools.dt(f) / 1_000.0
        n_samples = f.samples.size

        # Trace header byte 109 (DelayRecordingTime) gives the recording start
        datum_ms = float(f.header[0].get(segyio.TraceField.DelayRecordingTime, 0))

        return SurveyGeometry(
            inline_min=il_min, inline_max=il_max, inline_step=il_step,
            crossline_min=xl_min, crossline_max=xl_max, crossline_step=xl_step,
            sample_rate_ms=sample_rate_ms,
            n_samples=n_samples,
            n_inlines=len(inlines),
            n_crosslines=len(crosslines),
            datum_ms=datum_ms,
        )

    def _read_traces(self, f: segyio.SegyFile, geom: SurveyGeometry) -> np.ndarray:
        """Read trace data into a ``(n_il, n_xl, n_s)`` float32 array."""
        n_il = geom.n_inlines
        n_xl = geom.n_crosslines
        n_s  = geom.n_samples

        if self.sample_mode:
            n_il = min(n_il, self.sample_n_inlines)
            logger.info("Sample mode: loading %d / %d inlines", n_il, geom.n_inlines)

        volume = np.zeros((n_il, n_xl, n_s), dtype=np.float32)

        for il_idx, il_no in enumerate(f.ilines[:n_il]):
            traces = f.gather[il_no]  # (n_xl_actual, n_s)
            n_loaded = min(traces.shape[0], n_xl)
            volume[il_idx, :n_loaded, :] = traces[:n_loaded].astype(np.float32)

        return volume

    @staticmethod
    def _to_xarray(volume: np.ndarray, geom: SurveyGeometry) -> xr.Dataset:
        n_il, n_xl, _ = volume.shape
        da = xr.DataArray(
            volume,
            dims=["inline", "crossline", "twtt_ms"],
            coords={
                "inline":    geom.inlines[:n_il],
                "crossline": geom.crosslines[:n_xl],
                "twtt_ms":   geom.times_ms,
            },
            attrs={
                "units": "amplitude",
                "sample_rate_ms": geom.sample_rate_ms,
                "datum_ms": geom.datum_ms,
            },
        )
        return xr.Dataset({"amplitude": da})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_amplitude_stats(volume: np.ndarray) -> dict:
    """Descriptive amplitude statistics for the sidecar JSON."""
    flat = volume.ravel()
    nonzero = flat[flat != 0.0]
    return {
        "min":              float(np.min(flat)),
        "max":              float(np.max(flat)),
        "mean":             float(np.mean(flat)),
        "std":              float(np.std(flat)),
        "p01":              float(np.percentile(flat, 1)),
        "p99":              float(np.percentile(flat, 99)),
        "nonzero_fraction": float(nonzero.size / max(flat.size, 1)),
    }


def _file_sha256(path: Path, *, quick: bool = True) -> str:
    """SHA-256 fingerprint of a file.

    Parameters
    ----------
    quick:
        If True, hash only the first + last 4 MB to avoid blocking on large
        SEG-Y files.  The result uniquely identifies accidental re-use of
        wrong files but is *not* a full integrity hash.
    """
    h = hashlib.sha256()
    chunk = 4 * 1024 * 1024
    with path.open("rb") as fh:
        if quick:
            h.update(fh.read(chunk))
            fh.seek(-min(chunk, path.stat().st_size), 2)
            h.update(fh.read())
        else:
            for block in iter(lambda: fh.read(65_536), b""):
                h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


def load_segy(
    path: str | Path,
    *,
    sample_mode: bool = False,
    sample_n_inlines: int = 50,
) -> tuple[xr.Dataset, SurveyGeometry]:
    """Open a SEG-Y file and return an xarray Dataset.

    Parameters
    ----------
    path:
        Path to the .segy / .sgy file.
    sample_mode:
        Load only the first ``sample_n_inlines`` inlines for local dev.
    sample_n_inlines:
        Inline count when ``sample_mode=True``.
    """
    with SEGYLoader(path, sample_mode=sample_mode, sample_n_inlines=sample_n_inlines) as ldr:
        return ldr.load()


def segy_to_zarr(
    source: str | Path,
    dest: str | Path,
    *,
    chunks: tuple[int, int, int] = DEFAULT_CHUNKS,
    sample_mode: bool = False,
    sample_n_inlines: int = 50,
    overwrite: bool = False,
) -> IngestMetadata:
    """End-to-end conversion: SEG-Y → Zarr store + JSON sidecar.

    Parameters
    ----------
    source:
        Path to the source SEG-Y file.
    dest:
        Destination Zarr directory (e.g. ``staged/ST10010.zarr``).
    chunks:
        Zarr chunk shape ``(inline, crossline, sample)``.
    sample_mode:
        Load only the first ``sample_n_inlines`` inlines.
    sample_n_inlines:
        Inline count when ``sample_mode=True``.
    overwrite:
        Overwrite an existing store.

    Returns
    -------
    IngestMetadata
    """
    with SEGYLoader(source, sample_mode=sample_mode, sample_n_inlines=sample_n_inlines) as ldr:
        _, meta = ldr.to_zarr(dest, chunks=chunks, overwrite=overwrite)
    return meta
