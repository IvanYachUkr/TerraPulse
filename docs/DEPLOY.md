# TerraPulse Deploy: Rust Pipeline & Production Infrastructure

> This document details the high-performance Rust pipeline, Docker containerization, and CI/CD infrastructure that powers TerraPulse's live prediction capability.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Rust Pipeline](#2-rust-pipeline)
3. [COG Reader](#3-cog-reader)
4. [Feature Extraction](#4-feature-extraction)
5. [ONNX Inference](#5-onnx-inference)
6. [SAR Processing](#6-sar-processing)
7. [Dashboard](#7-dashboard)
8. [Docker](#8-docker)
9. [CI/CD](#9-cicd)
10. [Production Fixes](#10-production-fixes)
11. [Testing](#11-testing)

---

## 1. Architecture Overview

```
User selects bbox on map
          ↓
    FastAPI Backend (api.py)
          ↓
    deploy_runner.py
          ↓
    ┌─────────────────────────────┐
    │   terrapulse binary (Rust)  │
    │                             │
    │  Stage 1: Download          │ ← Planetary Computer STAC API
    │    └─ COG tile fetching     │
    │    └─ Cloud masking (SCL)   │
    │    └─ Seasonal compositing  │
    │                             │
    │  Stage 2: Extract           │ ← Feature engineering
    │    └─ Band statistics       │
    │    └─ Spectral indices      │
    │    └─ LBP texture           │
    │    └─ Tasseled Cap          │
    │    └─ Spatial features      │
    │                             │
    │  Stage 3: Predict           │ ← ONNX Runtime
    │    └─ StandardScaler        │
    │    └─ MLP inference         │
    │    └─ Softmax → proportions │
    │                             │
    │  Stage 4: Labels            │ ← WorldCover download
    │  Stage 5: Grid              │ ← GeoJSON output
    └─────────────────────────────┘
          ↓
    JSON results → Dashboard frontend
```

### Why Rust?

The Python pipeline (stackstac + scikit-learn + PyTorch) takes **10–15 minutes** per city-year. The Rust pipeline processes the same workload in **2–3 minutes** — a 5× speedup from:
- Zero-copy TIFF tile parsing (no GDAL overhead)
- Parallel band download with bounded concurrency
- Rayon data-parallel feature extraction across all CPU cores
- Direct ONNX Runtime inference without Python GIL

---

## 2. Rust Pipeline

### Source Files

| File | Lines | Purpose |
|------|------:|---------|
| `main.rs` | ~300 | CLI entry point (clap), pipeline orchestration |
| `composite.rs` | 609 | Scene download, band compositing, seasonal medians |
| `cog.rs` | 798 | Cloud-Optimized GeoTIFF reader (metadata + tile fetching) |
| `extract.rs` | 631 | Feature extraction orchestrator |
| `features.rs` | 1,184 | Core feature computation (indices, LBP, stats, TC) |
| `predict.rs` | 119 | ONNX model loading and inference |
| `sar_download.rs` | 961 | Sentinel-1 SAR data download and processing |
| `reproject.rs` | ~200 | Bilinear reprojection between CRS grids |
| `grid.rs` | ~100 | Grid cell GeoJSON generation |
| `labels.rs` | ~150 | WorldCover label download and aggregation |
| `stac.rs` | ~200 | STAC API client for scene discovery |
| `parquet_io.rs` | 184 | Parquet reading/writing via Arrow |
| `config.rs` | ~50 | Constants (N_CLASSES, GRID_PX, etc.) |

### Pipeline Stages

**Stage 1: Download** (`composite.rs`)
- Query Planetary Computer STAC for Sentinel-2 L2A scenes
- Download bands as COG tiles with bounded concurrency (`buffer_unordered(6)`)
- Apply SCL cloud mask (exclude classes 0, 1, 2, 3, 8, 9, 10, 11)
- Compute per-pixel seasonal median composite (NaN-aware)
- Apply BOA_ADD_OFFSET (−1000) for post-2022 imagery
- Write output as GeoTIFF

**Stage 2: Extract** (`extract.rs` + `features.rs`)
- Load seasonal GeoTIFF composites
- Compute 864 features per cell per season:
  - Band statistics (80): mean/std/min/max/q25/median/q75/finite_frac × 10 bands
  - Spectral indices (75): NDVI, NDWI, NDBI, NDMI, NBR, SAVI, BSI, EVI2, NDRE1, NDRE2, MNDWI, GNDVI, NDTI, IRECI, CRI1
  - Tasseled Cap (6): Brightness/Greenness/Wetness (mean, std)
  - Spatial (8): Sobel edge, Laplacian, Moran's I, NDVI range/IQR
  - LBP texture (55): Multi-band Uniform LBP histograms + entropy
- Parallel cell extraction using Rayon `par_iter`
- Write features to Parquet

**Stage 3: Predict** (`predict.rs`)
- Load StandardScaler (JSON) and MLP model (ONNX)
- Apply standardization: `(x - mean) / scale`
- Run ONNX inference in chunks of 65,536 cells
- Output softmax probabilities → class proportions
- Write predictions to JSON

**Stage 4: Labels** (`labels.rs`)
- Download WorldCover map for the bbox
- Aggregate 10×10 pixel patches to class proportions
- Write labels to JSON

**Stage 5: Grid** (`grid.rs`)
- Generate GeoJSON FeatureCollection with cell geometries
- Include cell metadata (center coordinates, UTM zone)

---

## 3. COG Reader

The `cog.rs` module implements a **from-scratch Cloud-Optimized GeoTIFF reader** — no GDAL dependency required.

### Why Custom?

- GDAL has a ~50 MB runtime dependency
- COG tiles are simple: TIFF header → tile offsets → HTTP range requests → decompress
- Full control over error handling and memory allocation
- Supports Deflate (flate2) and Zstd compression

### Implementation

1. **Metadata parsing**: Read first TIFF IFD to extract dimensions, tile layout, compression, sample format, GeoTransform, and byte order
2. **Tile mapping**: Calculate which tiles overlap the requested pixel bounding box
3. **Concurrent fetch**: Download tiles via HTTP range requests (`reqwest`)
4. **Decompression**: Deflate or Zstd based on TIFF Compression tag
5. **Predictor undo**: Horizontal differencing (predictor=2) for 8/16/32-bit samples, with correct byte order handling for big-endian TIFFs
6. **Assembly**: Place tiles into the output pixel buffer, handling edge tiles

### Byte-Order Awareness

A production fix added the `le` (little-endian) flag to `CogMeta`, threaded through `decode_tile`. This ensures predictor undo uses the correct endianness for both LE and BE TIFFs — critical for Sentinel-1 data which uses big-endian format.

---

## 4. Feature Extraction

### Band Statistics

For each 10×10 pixel cell, compute mean, std, min, max, 25th/50th/75th percentiles, and finite fraction across all bands. Uses NaN-aware sorting and accumulation in f64 for numerical stability.

### Spectral Indices

15 vegetation/water/soil indices computed per pixel, then aggregated to cell-level statistics:

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation greenness |
| NDWI | (Green − NIR) / (Green + NIR) | Water content |
| NDBI | (SWIR1 − NIR) / (SWIR1 + NIR) | Built-up areas |
| NDMI | (NIR − SWIR1) / (NIR + SWIR1) | Moisture |
| NBR | (NIR − SWIR2) / (NIR + SWIR2) | Burn severity |
| SAVI | 1.5(NIR − Red) / (NIR + Red + 0.5) | Soil-adjusted vegetation |
| BSI | (SWIR1 + Red − NIR − Blue) / (SWIR1 + Red + NIR + Blue) | Bare soil |
| EVI2 | 2.5(NIR − Red) / (NIR + 2.4·Red + 1) | Enhanced vegetation |
| NDRE1/2 | (NIR − RE1/RE2) / (NIR + RE1/RE2) | Red-edge vegetation |
| MNDWI | (Green − SWIR1) / (Green + SWIR1) | Modified water |
| GNDVI | (NIR − Green) / (NIR + Green) | Green NDVI |
| NDTI | (SWIR1 − SWIR2) / (SWIR1 + SWIR2) | Tillage index |
| IRECI | (RE3 − Red) / (RE1 / RE2) | Inverted red-edge |
| CRI1 | 1/Green − 1/RE1 | Chlorophyll red-edge |

### LBP Texture

Uniform Local Binary Patterns (P=8, R=1) computed on NIR, NDVI, EVI2, SWIR1, and NDTI bands. Per-patch extraction with reflected boundary handling. Output: 10-bin histogram + entropy per cell per band = 55 features.

### Tasseled Cap

10-band Nedkov 2017 coefficients for Brightness, Greenness, and Wetness. Computed as dot product of all 10 bands with coefficient vectors, then cell-level mean + std = 6 features.

---

## 5. ONNX Inference

### Model Export

The PyTorch MLP (917K parameters, L5 × d1024) is exported to ONNX format with:
- Input: `[batch_size, n_features]` float32
- Output: `[batch_size, N_CLASSES]` float32 (softmax probabilities)

### Rust Integration

The `ort` crate (v2.0.0-rc.9) provides ONNX Runtime bindings with:
- **Dynamic loading**: `load-dynamic` feature avoids static MSVC linking issues
- **Auto-download**: `download-binaries` fetches ONNX Runtime at build time
- **Chunked inference**: Process cells in batches of 65,536 to manage memory

### Scaler

StandardScaler parameters (mean + scale vectors) stored as JSON, applied before inference. Division-by-near-zero protection: if `|scale| < 1e-12`, output 0.0.

---

## 6. SAR Processing

### Sentinel-1 GRD

The `sar_download.rs` module handles Sentinel-1 Ground Range Detected (GRD) imagery:

- **GCP-based geolocation**: S1 GRD COGs use Ground Control Points instead of regular GeoTransforms. The module parses GCPs from TIFF tag 33922, builds a regular grid, and implements inverse bilinear interpolation for pixel ↔ geographic coordinate mapping.
- **Affine fitting**: Least-squares affine transform from GCP grid for reprojection
- **Amplitude normalization**: Raw DN values scaled to [0, 1] via `clamp(raw / 2000, 0, 1)`
- **UTM support**: Full Northern and Southern Hemisphere handling with false northing subtraction

---

## 7. Dashboard

### Two Modes

| Mode | Purpose | Data Source | Needs Rust? |
|------|---------|-------------|:-----------:|
| **Research** | Explore precomputed results, models, evaluation | 22 static JSON files (~50 MB) | ❌ |
| **Deploy** | Live prediction for any bbox on Earth | Satellite download → Rust pipeline | ✅ |

### Frontend Stack

- **React 19** + **Vite** (dev server + bundler)
- **MapLibre GL** + **deck.gl** (WebGL map rendering for 29,946+ cells)
- **Chart.js** (bar charts for model comparison, evaluation metrics)
- **Dark theme** responsive layout

### Backend Stack

- **FastAPI** + **uvicorn** (sub-ms JSON responses)
- **LRU-cached data loading** (all JSON loaded into memory at startup)
- **CORS middleware** for dev/prod flexibility
- **Deploy runner**: Background thread per job, streaming progress from Rust stdout

### Research Dashboard Panels

1. **Map View**: Interactive map of all 29,946 cells with labels/predictions/change/folds
2. **Cell Inspector**: Click any cell → labels (2020/2021), predictions (all 3 models), change, fold
3. **Model Comparison**: R², MAE, Aitchison distance bar charts
4. **Evaluation**: Per-class metrics, stress tests, change detection, failure analysis
5. **Explainability**: Feature importance, SHAP, conformal prediction intervals

### Deploy Dashboard

1. **Region Selector**: Draw bbox on map or enter coordinates
2. **Year Selection**: Choose prediction years (2020–2025)
3. **Live Pipeline**: Progress bar with stage-by-stage updates from Rust
4. **Results**: View predictions, labels, and grid on the map

---

## 8. Docker

### Multi-Stage Build

| Stage | Base Image | Output |
|-------|-----------|--------|
| `rust-build` | `rust:1.83-bookworm` | Linux `terrapulse` binary + ONNX Runtime .so |
| `frontend` | `node:22-slim` | Static `dist/` bundle |
| `runtime` | `python:3.12-slim-bookworm` | Final image with everything |

### Runtime Dependencies

- **GDAL** (`libgdal-dev`): Required for rasterio (anchor GeoTIFF creation in deploy mode)
- **ONNX Runtime**: Bundled by `ort` crate at build time
- **Python packages**: FastAPI, uvicorn, rasterio, numpy, pandas, pyproj, affine, pydantic

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TERRAPULSE_BIN` | `/usr/local/bin/terrapulse` | Path to Rust binary |
| `ONNX_MODELS_DIR` | `/app/models/onnx` | ONNX model + scaler files |
| `DEPLOY_DIR` | `/app/deploy_jobs` | Scratch directory for deploy jobs |
| `ORT_DYLIB_PATH` | `/usr/local/lib` | ONNX Runtime library path |

---

## 9. CI/CD

### GitHub Actions Workflow (`.github/workflows/ci.yml`)

| Trigger | Jobs |
|---------|------|
| Push/PR to `main` | `rust-test`, `frontend-build` |
| Push to `main` only | `docker` (build + push to GHCR) |

**Job Details**:

1. **rust-test**: `cargo test --release` in `terrapulse/` — runs all 17 tests (15 unit + 2 integration)
2. **frontend-build**: `npm ci && npm run build` in `src/dashboard/frontend/`
3. **docker**: Builds multi-stage image and pushes to `ghcr.io/{owner}/terrapulse:latest`

**Caching**: Cargo registry + build artifacts cached via `actions/cache@v4`, npm via `setup-node` cache, Docker layers via BuildKit `type=gha`.

---

## 10. Production Fixes

Eight production fixes were applied and verified (commit `421e551`):

| # | Fix | File | Impact |
|:-:|-----|------|--------|
| 1 | Southern Hemisphere UTM | `sar_download.rs` | False northing subtraction for latitude < 0° |
| 2 | Bounded concurrency | `composite.rs` | `buffer_unordered(6)` prevents OOM during scene downloads |
| 3 | TIFF tag overflow | `composite.rs`, `sar_download.rs` | `ImageWidth`/`ImageLength`/`RowsPerStrip` → LONG type for dimensions > 65,535 |
| 4 | Panic prevention | `labels.rs` | `.unwrap()` → `?` error propagation for UTM conversions |
| 5 | Dynamic pixel size | `grid.rs` | Use anchor GeoTransform instead of hardcoded 10 m |
| 6 | Unused dependency | `Cargo.toml` | Removed `indicatif` |
| 7 | Byte-order predictor | `cog.rs` | Little/big-endian aware predictor undo |
| 8 | Code simplification | `reproject.rs` | `(sx - 0.5 + 0.5).floor()` → `sx.floor()` |

---

## 11. Testing

### Rust Tests (17 total)

**Unit Tests (15)**:
- `cog.rs`: TIFF tag parsing, predictor undo (LE + BE), compression
- `features.rs`: reflect_index, cell_stats_8, LBP LUT
- `sar_download.rs`: UTM conversion (Northern + Southern Hemisphere), affine fitting, GCP grid
- `reproject.rs`: Bilinear interpolation, coordinate transforms

**Integration Tests (2)**:
- `test_terrapulse_cli_help`: Binary starts and shows help
- `test_pipeline_dry_run`: Pipeline entry point (currently a no-op placeholder)

### Verification

- All 17 tests pass with `cargo test --release`
- Release build succeeds: `cargo build --release` (4m 23s)
- Zero compiler warnings after cleanup
