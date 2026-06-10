# Session Log — Architecture Kickoff

**Date:** 2026-06-10

**Session:** Architecture proposal for DeepSeismic2 PoC

## Overview

Ripley (Lead) generated comprehensive architecture proposal. Team consensus on cloud-native, object-storage-first design with Azure infrastructure.

## Decisions Made

- Storage: Azure Blob/ADLS Gen2 (system of record)
- Format: SEG-Y source + Zarr/JSON/Parquet derived
- Compute: CPU ingest/preprocessing, GPU inference (Azure ML)
- Backend: FastAPI thin service
- Model: UNet baseline
- LLM: Strategic use for assistant tasks only

## Next Steps

- Review architecture proposal (docs/architecture-proposal.md)
- Validate with extended team
- Begin infrastructure setup
