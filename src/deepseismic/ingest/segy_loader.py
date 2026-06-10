"""SEG-Y ingest entry points.

This module will host the first-pass loader interfaces for reading SEG-Y volumes,
extracting survey metadata, and producing cloud-friendly derivative artifacts such
as Zarr arrays plus JSON or Parquet manifests. Planned interfaces include helpers
for local file loading, ADLS-backed reads, header inspection, and conversion job
submission metadata.
"""
