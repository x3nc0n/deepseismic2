"""Pydantic schema definitions for the deepseismic2 API surface.

All request/response bodies are defined here so route handlers stay lean.
Inline/crossline slices use plain Python lists for numpy-free serialisation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


class SurveyGeometryOut(BaseModel):
    inline_min: int
    inline_max: int
    inline_step: int
    crossline_min: int
    crossline_max: int
    crossline_step: int
    sample_rate_ms: float
    n_samples: int
    n_inlines: int
    n_crosslines: int
    datum_ms: float = 0.0


class AmplitudeStatsOut(BaseModel):
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p99: float
    nonzero_fraction: float


class SurveyMetadata(BaseModel):
    survey_id: str
    source_file: str
    ingested_at: datetime
    sample_mode: bool
    n_inlines_loaded: int
    geometry: SurveyGeometryOut
    amplitude_stats: AmplitudeStatsOut
    zarr_path: str


class SurveyListItem(BaseModel):
    survey_id: str
    source_file: str
    ingested_at: datetime
    n_inlines: int
    n_crosslines: int


class IngestRequest(BaseModel):
    blob_path: str = Field(..., description="SEG-Y blob path in the 'raw' container")
    survey_id: str = Field(..., description="Identifier for the output survey")
    sample_mode: bool = Field(False, description="Load only the first N inlines")
    sample_n_inlines: int = Field(50, ge=1, description="Inline count when sample_mode=True")


class IngestResponse(BaseModel):
    run_id: str
    survey_id: str
    status: str = "pending"
    message: str = "Ingest job queued"


class InlineSlice(BaseModel):
    """Single inline section returned as amplitude arrays for display."""

    survey_id: str
    inline_number: int
    crossline_coords: list[int]
    twtt_ms: list[float]
    amplitude: list[list[float]]  # shape [n_crosslines][n_samples]


class CrosslineSlice(BaseModel):
    """Single crossline section returned as amplitude arrays for display."""

    survey_id: str
    crossline_number: int
    inline_coords: list[int]
    twtt_ms: list[float]
    amplitude: list[list[float]]  # shape [n_inlines][n_samples]


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


class JobStatus(StrEnum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


class InterpretationRequest(BaseModel):
    survey_id: str = Field(..., description="Survey to run fault detection on")
    checkpoint_blob: str = Field(
        "checkpoints/unet3d_best.pt",
        description="Blob path in 'features' container for the model checkpoint",
    )
    patch_size: tuple[int, int, int] = Field((64, 64, 64))
    overlap: float = Field(0.25, ge=0.0, lt=1.0)
    batch_size: int = Field(4, ge=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class InterpretationStatus(BaseModel):
    run_id: str
    survey_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress_pct: float = 0.0
    message: str = ""
    error: str | None = None


class InterpretationResult(BaseModel):
    run_id: str
    survey_id: str
    status: JobStatus
    prob_zarr_path: str
    mask_zarr_path: str
    fault_voxel_fraction: float
    completed_at: datetime
    download_url: str | None = None  # SAS URL when storage is real


class FaultOverlay(BaseModel):
    """Fault probability overlay for a single inline — suitable for seismic viewers."""

    run_id: str
    inline_number: int
    crossline_coords: list[int]
    twtt_ms: list[float]
    fault_probability: list[list[float]]  # shape [n_crosslines][n_samples]
    fault_mask: list[list[int]]           # shape [n_crosslines][n_samples]


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------


class FormationTop(BaseModel):
    formation: str
    depth_m: float
    tvdss_m: float | None = None


class WellListItem(BaseModel):
    well_id: str
    name: str
    latitude: float | None = None
    longitude: float | None = None
    kb_elevation_m: float | None = None
    total_depth_m: float | None = None


class WellMetadata(BaseModel):
    well_id: str
    name: str
    uwi: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    kb_elevation_m: float | None = None
    total_depth_m: float | None = None
    spud_date: str | None = None
    formation_tops: list[FormationTop] = Field(default_factory=list)
    logs_available: list[str] = Field(default_factory=list)


class WellLogCurve(BaseModel):
    mnemonic: str
    unit: str
    description: str
    values: list[float | None]


class WellLog(BaseModel):
    well_id: str
    depth_unit: str = "m"
    depth_values: list[float]
    curves: list[WellLogCurve]
