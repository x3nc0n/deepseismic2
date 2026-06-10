# Methodology Overview

DeepSeismic2 follows a constrained end-to-end workflow:
1. Acquire a Volve dataset subset and preserve SEG-Y as the raw source artifact.
2. Extract metadata and generate cloud-friendly derivatives such as Zarr.
3. Apply deterministic preprocessing for conditioning, windowing, and QC.
4. Run baseline model inference using a UNet-style architecture.
5. Publish metadata, run status, and result references through a FastAPI service.
6. Ground an analyst agent with workflow documentation and domain context through Azure AI Search.

The PoC boundary is deliberately narrow: prove one credible workflow rather than a full production interpretation platform.
