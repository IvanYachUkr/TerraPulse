# ===========================================================================
# TerraPulse Dashboard – Multi-Stage Docker Image
# ===========================================================================
# Stages:
#   1. rust-build   → compile terrapulse binary (Linux)
#   2. frontend     → npm ci && npm run build (static dist/)
#   3. runtime      → Python + Rust binary + frontend + data + models
#
# Build:  docker build -t terrapulse .
# Run:    docker run -p 8000:8000 terrapulse
#
# Environment variables (optional overrides):
#   TERRAPULSE_BIN   – path to the terrapulse binary
#   ONNX_MODELS_DIR  – path to ONNX model + scaler + columns JSON
#   DEPLOY_DIR       – scratch directory for deploy jobs
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build the Rust binary
# ---------------------------------------------------------------------------
FROM rust:bookworm AS rust-build

# Install build deps (zstd for S1 COG support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libzstd-dev cmake libssl-dev perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy entire Rust project
COPY terrapulse/ ./

# Build in release mode (ort crate will download ONNX Runtime)
RUN cargo build --release

# Verify the binary works
RUN ./target/release/terrapulse --help

# ONNX Runtime .so is downloaded by ort-sys into cargo's OUT_DIR cache
# Search the entire filesystem to find it
RUN mkdir -p /ort_libs && \
    find / -name "libonnxruntime*.so*" 2>/dev/null -exec cp {} /ort_libs/ \; || true && \
    echo "=== ORT libs found:" && ls -la /ort_libs/ && \
    echo "=== Total files: $(find /ort_libs -type f | wc -l)"

# ---------------------------------------------------------------------------
# Stage 2: Build the frontend
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /frontend
COPY src/dashboard/frontend/package.json src/dashboard/frontend/package-lock.json ./
RUN npm ci --ignore-scripts

COPY src/dashboard/frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 3: Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Install runtime deps for rasterio (GDAL) and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin \
    libzstd1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (slim set for dashboard only)
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy Rust binary
COPY --from=rust-build /build/target/release/terrapulse /usr/local/bin/terrapulse

# Download ONNX Runtime shared lib (ort crate uses load-dynamic / dlopen at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /tmp/ort && \
    curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz -o /tmp/ort.tgz && \
    tar xzf /tmp/ort.tgz -C /tmp/ort --strip-components=1 && \
    cp /tmp/ort/lib/libonnxruntime*.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/ort /tmp/ort.tgz && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ORT_DYLIB_PATH must be a FILE path, not a directory
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

# Copy frontend dist
COPY --from=frontend /frontend/dist /app/src/dashboard/frontend/dist

# Copy Python source
COPY src/ /app/src/

# Copy easy-download helper script
COPY src/easy_download.py /app/easy_download.py

# Copy dashboard data (research JSONs)
COPY src/dashboard/data/ /app/src/dashboard/data/

# Copy ONNX model + scaler + columns
COPY data/pipeline_output/models/onnx/ /app/models/onnx/

# Environment variables
ENV TERRAPULSE_BIN=/usr/local/bin/terrapulse
ENV ONNX_MODELS_DIR=/app/models/onnx
ENV DEPLOY_DIR=/app/deploy_jobs

# Create deploy job scratch dir
RUN mkdir -p /app/deploy_jobs

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/meta')" || exit 1

# Run the API server
CMD ["python", "-m", "uvicorn", "src.dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
