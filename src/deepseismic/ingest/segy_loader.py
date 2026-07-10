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

    # Local smoke-ingest of first 50 inlines (cheap, format-proxy only):
    meta = segy_to_zarr(
        "path/to/ST10010_PSDM_TIME.segy",          # supply via --source arg
        "staged/surveys/volve-st10010/amplitude.zarr",
        survey_id="volve-st10010",
        sample_mode=True,
        sample_n_inlines=50,
    )
    print(meta.geometry["n_inlines"], meta.amplitude_stats["p99"])

    # Full ingest (in-VNet, real ST10010 geometry: inlines 9985–10369):
    meta = segy_to_zarr(
        "/mnt/raw/ST10010_PSDM_TIME.segy",
        "staged/surveys/volve-st10010/amplitude.zarr",
        survey_id="volve-st10010",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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

# Substrings that identify a segyio geometry-inference failure caused by an
# irregular trace grid (edge-of-survey traces absent, so ilines*xlines does not
# equal the trace count).  segyio phrases this differently across versions:
#   - "Inlines inconsistent"                      (older, per-inline XL mismatch)
#   - "Invalid dimensions ... should match the number of traces (N)"  (real F3)
# Both mean the same thing: the structured reader can't build a rectangular grid,
# and we should fall back to the header-driven irregular reconstruction.
_IRREGULAR_GEOMETRY_ERROR_MARKERS: tuple[str, ...] = (
    "inconsistent",
    "invalid dimensions",
    "should match the number of traces",
)


def _is_irregular_geometry_error(exc: Exception) -> bool:
    """Return ``True`` if *exc* is a segyio irregular-grid geometry error.

    Used to decide whether :meth:`SEGYLoader.load` should fall back to the
    header-driven irregular-grid reconstruction.  Kept deliberately narrow (a
    fixed set of known substrings) so genuinely unrelated ``ValueError``\\ s are
    still re-raised instead of being silently swallowed.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _IRREGULAR_GEOMETRY_ERROR_MARKERS)


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
    # Optional world-coordinate tie-points ``[[x, y, il, xl], ...]`` derived
    # from CDP-X/CDP-Y (bytes 181/185, coordinate scalar applied).  Populated at
    # ingest when the SEG-Y carries CDP coordinates; enables downstream
    # world→(inline, crossline) transforms (e.g. F3 world-coordinate fault
    # rasterisation).  ``None`` when CDP coordinates are absent/zero.
    corner_points: list[list[float]] | None = None

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
    survey_id: str | None
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

    def __enter__(self) -> SEGYLoader:
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

        try:
            with segyio.open(str(self._segy_path), ignore_geometry=self.ignore_geometry) as f:
                f.mmap()  # memory-map for faster sequential reads
                geom = self._extract_geometry(f)
                data = self._read_traces(f, geom)
        except ValueError as exc:
            # Real surveys often have irregular XL counts per inline (edge-of-survey
            # boundary traces absent).  segyio raises "Inlines inconsistent" OR
            # "Invalid dimensions ... should match the number of traces" in that case
            # (the exact wording differs; real F3 emits the latter).  Fall back to
            # reading all trace headers explicitly and building a padded volume.
            if not _is_irregular_geometry_error(exc):
                raise
            logger.warning(
                "segyio geometry inference failed (%s); falling back to "
                "irregular-grid reconstruction (reads all trace headers).",
                exc,
            )
            with segyio.open(str(self._segy_path), ignore_geometry=True) as f:
                f.mmap()
                geom, ils, xls = self._extract_geometry_irregular(f)
                data = self._read_traces_irregular(f, geom, ils, xls)

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
        survey_id: str | None = None,
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
        survey_id:
            Optional identifier embedded in the JSON sidecar
            (e.g. ``"volve-st10010"``).  Allows downstream pipeline stages
            to verify they are consuming the correct survey artifact.

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

        store = zarr.storage.LocalStore(str(output_path))
        root = zarr.open_group(store, mode="w")

        z = root.create_array(
            "amplitude",
            data=amplitude.astype(np.float32),
            chunks=chunks,
            overwrite=overwrite,
        )
        logger.info("Wrote amplitude array → %s  shape=%s", output_path, z.shape)

        # Store coordinate 1-D arrays alongside the data for easy access
        root.create_array("inline",    data=geom.inlines[:amplitude.shape[0]].astype(np.int32))
        root.create_array("crossline", data=geom.crosslines[:amplitude.shape[1]].astype(np.int32))
        root.create_array("twtt_ms",   data=geom.times_ms.astype(np.float32))

        amp_stats = _compute_amplitude_stats(amplitude)

        src_name = self._segy_path.name if self._segy_path else "<bytes>"
        sha256 = None
        if self._segy_path and self._segy_path.exists() and self._segy_path != self._tmp_path:
            sha256 = _file_sha256(self._segy_path, quick=True)

        meta = IngestMetadata(
            source_file=src_name,
            source_sha256=sha256,
            ingested_at=datetime.now(UTC).isoformat(),
            survey_id=survey_id,
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
            corner_points=self._extract_corner_tie_points(f),
        )

    @staticmethod
    def _extract_corner_tie_points(f: segyio.SegyFile) -> list[list[float]] | None:
        """Derive three world-coordinate tie-points from CDP-X/CDP-Y headers.

        Reads per-trace ``CDP_X`` (byte 181), ``CDP_Y`` (byte 185),
        ``INLINE_3D`` (byte 189) and ``CROSSLINE_3D`` (byte 193) via bulk
        ``attributes()`` calls and returns three non-collinear
        ``[x, y, inline, crossline]`` tie-points anchored at grid corners:
        ``(il_min, xl_min)``, ``(il_max, xl_min)`` and ``(il_min, xl_max)``.

        The SEG-Y coordinate scalar (byte 71, ``SourceGroupScalar``) is applied
        so the returned X/Y are in true map units (metres).  Returns ``None``
        when CDP coordinates are absent or all-zero (no world georeference), in
        which case downstream world-coordinate transforms are unavailable.
        """
        try:
            ils = np.asarray(f.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int64)
            xls = np.asarray(f.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int64)
            cdpx = np.asarray(f.attributes(segyio.TraceField.CDP_X)[:], dtype=np.float64)
            cdpy = np.asarray(f.attributes(segyio.TraceField.CDP_Y)[:], dtype=np.float64)
            scalars = np.asarray(
                f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=np.float64
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read CDP corner coordinates: %s", exc)
            return None

        if ils.size == 0 or (np.all(cdpx == 0.0) and np.all(cdpy == 0.0)):
            logger.info("No CDP-X/CDP-Y georeference present; corner_points=None.")
            return None

        # Apply the SEG-Y coordinate scalar: negative → divide, positive → multiply.
        scalar = float(scalars[0]) if scalars.size else 0.0
        if scalar < 0:
            cdpx = cdpx / abs(scalar)
            cdpy = cdpy / abs(scalar)
        elif scalar > 0:
            cdpx = cdpx * scalar
            cdpy = cdpy * scalar

        il_min, il_max = int(ils.min()), int(ils.max())
        xl_min, xl_max = int(xls.min()), int(xls.max())
        corners = [
            (ils == il_min) & (xls == xl_min),
            (ils == il_max) & (xls == xl_min),
            (ils == il_min) & (xls == xl_max),
        ]
        tie_points: list[list[float]] = []
        for mask in corners:
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                logger.warning(
                    "Missing corner trace for tie-point derivation; corner_points=None."
                )
                return None
            i = int(idx[0])
            tie_points.append(
                [float(cdpx[i]), float(cdpy[i]), int(ils[i]), int(xls[i])]
            )
        logger.info("Derived %d world-coordinate corner tie-points from CDP-X/Y.", len(tie_points))
        return tie_points

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
            traces = np.array(f.iline[il_no], dtype=np.float32)  # (n_xl_actual, n_s)
            n_loaded = min(traces.shape[0], n_xl)
            volume[il_idx, :n_loaded, :] = traces[:n_loaded]

        return volume

    def _extract_geometry_irregular(
        self, f: segyio.SegyFile
    ) -> tuple[SurveyGeometry, np.ndarray, np.ndarray]:
        """Extract geometry from a SEG-Y with an irregular trace grid.

        Reads all trace headers in bulk via :meth:`segyio.SegyFile.attributes`
        (one syscall per field, much faster than per-trace header dicts).

        Returns
        -------
        geom : SurveyGeometry
            Grid extent derived from the actual IL/XL header values.
        ils : np.ndarray[int32]
            Inline number for each trace (length = f.tracecount).
        xls : np.ndarray[int32]
            Crossline number for each trace (length = f.tracecount).
        """
        logger.info(
            "Irregular-grid: reading all %d trace IL/XL headers via attributes()...",
            f.tracecount,
        )
        ils = np.array(f.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int32)
        xls = np.array(f.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int32)

        unique_ils = np.unique(ils)
        unique_xls = np.unique(xls)

        il_min = int(unique_ils[0])
        il_max = int(unique_ils[-1])
        il_step = int(unique_ils[1] - unique_ils[0]) if len(unique_ils) > 1 else 1
        xl_min = int(unique_xls[0])
        xl_max = int(unique_xls[-1])
        xl_step = int(unique_xls[1] - unique_xls[0]) if len(unique_xls) > 1 else 1

        sample_rate_ms = segyio.tools.dt(f) / 1_000.0
        n_samples = f.samples.size
        datum_ms = float(f.header[0].get(segyio.TraceField.DelayRecordingTime, 0))

        geom = SurveyGeometry(
            inline_min=il_min,
            inline_max=il_max,
            inline_step=il_step,
            crossline_min=xl_min,
            crossline_max=xl_max,
            crossline_step=xl_step,
            sample_rate_ms=sample_rate_ms,
            n_samples=n_samples,
            n_inlines=len(unique_ils),
            n_crosslines=len(unique_xls),
            datum_ms=datum_ms,
            corner_points=self._extract_corner_tie_points(f),
        )
        logger.info(
            "Irregular-grid geometry: IL %d–%d (n=%d), XL %d–%d (n=%d), "
            "n_samples=%d, dt=%.1f ms",
            il_min, il_max, geom.n_inlines,
            xl_min, xl_max, geom.n_crosslines,
            n_samples, sample_rate_ms,
        )
        return geom, ils, xls

    def _read_traces_irregular(
        self,
        f: segyio.SegyFile,
        geom: SurveyGeometry,
        ils: np.ndarray,
        xls: np.ndarray,
    ) -> np.ndarray:
        """Fill a zero-padded volume from an irregular-grid SEG-Y.

        Missing IL/XL cells (survey boundary absent traces) remain zero.
        Sample mode restricts to the first ``sample_n_inlines`` inlines.

        Parameters
        ----------
        f : segyio.SegyFile
            Open file handle (ignore_geometry=True).
        geom : SurveyGeometry
            Full-survey geometry from :meth:`_extract_geometry_irregular`.
        ils, xls : np.ndarray[int32]
            Per-trace inline/crossline arrays.
        """
        n_il = geom.n_inlines
        n_xl = geom.n_crosslines
        n_s = geom.n_samples

        if self.sample_mode:
            n_il = min(n_il, self.sample_n_inlines)
            logger.info(
                "Irregular-grid sample mode: loading %d / %d inlines", n_il, geom.n_inlines
            )

        max_il_value = geom.inline_min + (n_il - 1) * geom.inline_step

        # Build O(1) index lookup maps
        il_map = {
            il: idx
            for idx, il in enumerate(
                range(geom.inline_min, geom.inline_max + 1, geom.inline_step)
            )
        }
        xl_map = {
            xl: idx
            for idx, xl in enumerate(
                range(geom.crossline_min, geom.crossline_max + 1, geom.crossline_step)
            )
        }

        volume = np.zeros((n_il, n_xl, n_s), dtype=np.float32)
        n_placed = 0

        logger.info(
            "Irregular-grid: reading %d traces → volume (%d × %d × %d)...",
            f.tracecount, n_il, n_xl, n_s,
        )
        for trace_idx in range(f.tracecount):
            il_val = int(ils[trace_idx])
            if il_val > max_il_value:
                continue
            xl_val = int(xls[trace_idx])
            il_idx = il_map.get(il_val)
            xl_idx = xl_map.get(xl_val)
            if il_idx is None or xl_idx is None:
                continue
            volume[il_idx, xl_idx, :] = f.trace.raw[trace_idx]
            n_placed += 1

        logger.info(
            "Irregular-grid: placed %d / %d traces (%.1f%% fill)",
            n_placed, f.tracecount, 100.0 * n_placed / max(f.tracecount, 1),
        )
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
    survey_id: str | None = None,
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
        Destination Zarr directory (e.g. ``staged/surveys/volve-st10010/amplitude.zarr``).
    survey_id:
        Optional survey identifier embedded in the JSON sidecar
        (e.g. ``"volve-st10010"``).  Used downstream to locate ADLS artifacts;
        does not affect the Zarr store layout.
    chunks:
        Zarr chunk shape ``(inline, crossline, sample)``.
    sample_mode:
        Load only the first ``sample_n_inlines`` inlines.  Use for local
        smoke-ingest validation without the full ST10010 volume (~1 GB).
    sample_n_inlines:
        Inline count when ``sample_mode=True``.
    overwrite:
        Overwrite an existing store.

    Returns
    -------
    IngestMetadata
    """
    with SEGYLoader(source, sample_mode=sample_mode, sample_n_inlines=sample_n_inlines) as ldr:
        _, meta = ldr.to_zarr(dest, chunks=chunks, overwrite=overwrite, survey_id=survey_id)
    return meta
