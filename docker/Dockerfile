# ===========================================================================
# deepseismic2 — multi-stage Docker build
# Stage 1 (builder): install all dependencies
# Stage 2 (runtime): copy only the installed packages — no build tooling
# ===========================================================================

# ----------- Stage 1: builder ----------- #
FROM python:3.11-slim AS builder

WORKDIR /build

# Upgrade pip once in the builder; layers are cached across rebuilds
RUN pip install --upgrade pip --no-cache-dir

# Copy manifests first so the dependency layer is cached independently
# of application source changes
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

# ----------- Stage 2: runtime ----------- #
FROM python:3.11-slim AS runtime

WORKDIR /app

# Least-privilege: run as a non-root user
RUN groupadd -r deepseismic \
 && useradd  -r -g deepseismic -d /app deepseismic

# Copy installed packages and entry-point scripts from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src ./src

RUN chown -R deepseismic:deepseismic /app

USER deepseismic

EXPOSE 8000

# Health check — hits the /health endpoint defined in FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

CMD ["uvicorn", "deepseismic.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
