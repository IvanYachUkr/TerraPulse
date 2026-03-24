<p align="center">
  <h1 align="center">TerraPulse</h1>
  <p align="center">
    <strong>Land-cover prediction and change detection from satellite imagery</strong>
  </p>
  <p align="center">
    Sentinel-2 multi-spectral imagery → ML models → interactive dashboard
  </p>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#models">Models</a> •
  <a href="#dashboard">Dashboard</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#repository-structure">Repo Structure</a> •
  <a href="#reproduction">Reproduction</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

TerraPulse is a machine learning system that predicts land-cover composition and land-cover change using ESA WorldCover (2020, 2021) as training labels and Sentinel-2/Sentinel-1 satellite imagery as input. The primary study area is Nuremberg, Germany, but the global deployment model can produce predictions for any region on Earth.

The system classifies each area into 7 land-cover types: tree cover, shrubland, grassland, cropland, built-up, bare/sparse vegetation, and water. Predictions are served through an interactive web dashboard.

**Live demo:** [terrapulse.calmglacier-f6008cba.germanywestcentral.azurecontainerapps.io](https://terrapulse.calmglacier-f6008cba.germanywestcentral.azurecontainerapps.io/)

### Design Choices

- **Rust inference pipeline**: The entire satellite download, compositing, feature extraction, and ONNX inference pipeline is implemented in Rust for maximum speed and efficiency. This allows the dashboard to produce live predictions for arbitrary regions in reasonable time, even for large bounding boxes.
- **Azure deployment**: The application is deployed on Azure Container Apps in the Germany West Central region. This co-locates the service with Microsoft Planetary Computer (the source of Sentinel-2 and Sentinel-1 imagery), which significantly reduces data transfer latency during on-the-fly predictions.
- **Pixel-level resolution**: The Nuremberg model operates at 10m per pixel rather than aggregated cells, allowing the dashboard to render predictions at any resolution the user selects via a zoom slider.
- **Self-contained pipeline**: All data sources (Sentinel-2, Sentinel-1, ESA WorldCover) are publicly available without authentication, so the pipeline can run fully autonomously on any machine.

---

## Models

TerraPulse uses four models, each serving a different purpose:

| # | Purpose | Type | Integration |
|---|---------|------|-------------|
| 1 | Pixel-wise land-cover classification | CatBoost GBDT | Nuremberg tab: per-pixel predictions across years |
| 2 | Cell-level (100m) class-fraction prediction | Tapered MLP (PyTorch/ONNX) | Global tab: live predictions for any region |
| 3 | Pixel-wise binary change likelihood | Random Forest / Logistic Regression | Nuremberg Experimental tab: change heatmap |
| 4 | Pixel-wise new-label prediction (post-change) | Random Forest | Nuremberg Experimental tab: future label prediction |

### Model 1: Pixel-wise CatBoost classifier

Predicts ESA WorldCover land-cover class for each 10m pixel. Trained on ~100 European cities (150K sampled pixels per city), with Nuremberg held out entirely. Uses 217 features per pixel: 10 Sentinel-2 bands, 9 spectral indices, 3 SAR features across 6 temporal slots (2 years x 3 seasons), plus temporal difference features.

The dashboard aggregates per-pixel predictions to whichever resolution the user selects with the zoom slider.

### Model 2: Global MLP

Predicts class-fraction distributions for 100m x 100m cells anywhere on Earth. Architecture: TaperedMLP with layers 1024/512/256/64, GELU activations, ~2.5M parameters. Trained on 92 European cities, validated on 23, tested on 6 held-out cities (including Nuremberg).

Test results (mean across 6 cities): Top-1 accuracy 90.2%, R² 0.676 at 5% threshold. The model is exported to ONNX and runs inside the standalone Rust binary for inference.

Each cell uses 1,764 features: spectral bands and indices, tasseled cap, spatial statistics, multi-band LBP textures, SAR, and phenological descriptors.

### Models 3 and 4: Two-step change prediction

A two-stage pipeline for predicting future land-cover labels. Model 3 predicts how likely each pixel is to change within the next year, visualized as a heatmap. Model 4 then predicts the new label for pixels above a 95% change-likelihood threshold.

Both models use 17 features per pixel (Sentinel-2 bands, NDVI, NDVI std, current label, and district-level socioeconomic statistics). Trained and evaluated using 4-fold spatial cross-validation on Nuremberg.

The Random Forest achieves 98.7% precision and 32.4% recall on change detection, deliberately favoring precision to minimize false positives.

---

## Dashboard

The dashboard has two main tabs:

### Nuremberg Tab

Pixel-level exploration of the Nuremberg study area (29,946 grid cells):

- **Labels view**: ESA WorldCover labels for 2020 and 2021 at adjustable resolution
- **Predictions view**: Model 1 predictions across years (2018 to 2025), with year-to-year comparison showing which pixels changed class
- **Experimental view**: Change heatmap from Model 3 (switch between Random Forest and Logistic Regression outputs), plus Model 4 future label predictions
- **District interaction**: Hover and click on Nuremberg statistical districts, with socioeconomic context (population density, residential units, commercial space)
- **Satellite toggle**: Switch between dark basemap and Esri satellite imagery

### Global Tab

Live prediction for any bounding box worldwide:

1. Draw a region on the map or specify coordinates
2. The Rust pipeline downloads satellite imagery, extracts features, and runs ONNX inference
3. View predictions, compare with WorldCover labels, inspect individual cells
4. Real-time progress updates for each pipeline stage
5. Model evaluation panel with per-class R², confusion matrices, and stress test results

---

## Quickstart

### Option 1: Docker (recommended)

```bash
docker pull ghcr.io/ivanyachukr/terrapulse:latest
docker run -p 8000:8000 ghcr.io/ivanyachukr/terrapulse:latest
# Open http://localhost:8000
```

The Docker image contains everything: the Rust binary, ONNX model, frontend build, precomputed Nuremberg data, and API server.

### Option 2: Build from source

**Prerequisites:** Rust 1.83+, Python 3.12+, Node.js 22+

```bash
git clone https://github.com/IvanYachUkr/TerraPulse.git
cd TerraPulse

# Build the Rust binary
cd terrapulse
cargo build --release
cd ..

# Install Python dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements-docker.txt

# Build the frontend
cd src/dashboard/frontend
npm ci
npm run build
cd ../../..

# Start the server
python -m uvicorn src.dashboard.api:app --port 8000
```

### Option 3: Development mode (hot reload)

```bash
# Terminal 1: API server
python -m uvicorn src.dashboard.api:app --port 8000 --reload

# Terminal 2: Frontend dev server
cd src/dashboard/frontend
npm run dev
# Frontend at http://localhost:5173, proxying API calls to :8000
```

---

## Repository Structure

```
TerraPulse/
│
├── terrapulse/                        # Rust inference pipeline (~6,500 LOC)
│   ├── Cargo.toml                     # Rust package manifest and dependencies
│   ├── Cargo.lock                     # Locked dependency versions
│   ├── helpers/
│   │   └── composite.py               # Python reference implementation for compositing logic
│   └── src/
│       ├── main.rs                    # CLI entry point (subcommands: download, extract, predict, pipeline)
│       ├── composite.rs               # Sentinel-2 scene download, cloud masking, seasonal compositing
│       ├── cog.rs                     # Cloud-Optimized GeoTIFF reader (zero-dependency TIFF parser)
│       ├── config.rs                  # Pipeline configuration and model paths
│       ├── download.rs                # Coordinator for satellite data acquisition
│       ├── extract.rs                 # Feature extraction orchestrator (bands, indices, texture, SAR)
│       ├── features.rs                # Core feature computation (spectral indices, LBP, spatial stats)
│       ├── grid.rs                    # GeoJSON grid generation for bounding boxes
│       ├── labels.rs                  # ESA WorldCover label download and rasterization
│       ├── parquet_io.rs              # Parquet file I/O for feature matrices and predictions
│       ├── predict.rs                 # ONNX Runtime inference (chunked, with scaling)
│       ├── reproject.rs               # CRS reprojection (WGS84 to UTM and back)
│       ├── sar_download.rs            # Sentinel-1 SAR scene download via STAC API
│       ├── sar_features.rs            # SAR feature extraction (VV, VH, ratios, RVI)
│       ├── stac.rs                    # STAC API client for Planetary Computer queries
│       └── tif_reader.rs             # Low-level GeoTIFF band reader
│
├── src/
│   ├── dashboard/
│   │   ├── api.py                     # FastAPI backend (REST API, serves data + spawns Rust pipeline)
│   │   ├── deploy_runner.py           # Pipeline orchestrator (async subprocess management)
│   │   ├── precalculate_experimental.py  # Precompute experimental model predictions
│   │   ├── DATA_FORMAT.md             # Documentation of binary file format
│   │   ├── .gitkeep
│   │   ├── data/
│   │   │   ├── nuremberg_boundary.geojson            # Nuremberg city boundary polygon
│   │   │   └── nuremberg_dashboard/
│   │   │       ├── nuremberg_dashboard_meta.json      # Grid metadata (dimensions, bounds)
│   │   │       ├── experimental_metrics.json          # Metrics for change/experimental models
│   │   │       ├── explainable_metrics.json           # Metrics for RF + LR explainable models
│   │   │       ├── nuremberg_labels_{year}_res{1-10}.bin    # WorldCover labels (2020, 2021)
│   │   │       ├── nuremberg_pred_{year}_res{1-10}.bin      # Model 1 predictions (2018-2025)
│   │   │       ├── experimental_heatmap_2021_res{1-10}.bin  # Model 3 RF change heatmap
│   │   │       ├── experimental_pred_2021_res{1-10}.bin     # Model 4 predicted labels
│   │   │       └── explainable_heatmap_2021_res{1-10}.bin   # Model 3 LR change heatmap
│   │   └── frontend/                  # React + Vite + deck.gl
│   │       ├── index.html             # HTML entry point
│   │       ├── vite.config.js         # Vite build config (API proxy, build settings)
│   │       ├── .gitignore
│   │       ├── public/
│   │       │   ├── nuremberg_stat_bezirke_wgs84.geojson  # District boundaries for map overlay
│   │       │   └── vite.svg           # Favicon
│   │       └── src/
│   │           ├── main.jsx           # React entry point
│   │           ├── App.jsx            # Main app with tab routing (Nuremberg / Global)
│   │           ├── index.css          # Complete design system (dark theme, components, layout)
│   │           ├── hooks/
│   │           │   └── useApi.js      # Custom hook for API calls with loading/error state
│   │           ├── data/
│   │           │   └── trainingRegions.js  # City coordinates for training region map overlay
│   │           └── components/
│   │               ├── Header.jsx            # Top navigation bar with tab switcher
│   │               ├── Sidebar.jsx           # Controls: year picker, resolution, class filter, model stats
│   │               ├── NurembergMapView.jsx  # Nuremberg pixel-level map (labels, predictions, heatmap)
│   │               ├── DeployView.jsx        # Global deployment map (deck.gl + draw region)
│   │               ├── DeployPanel.jsx       # Pipeline controls (bbox input, year selector, run button)
│   │               ├── MapView.jsx           # Shared map utilities and base map component
│   │               ├── CellInspector.jsx     # Per-cell detail panel (predictions across models)
│   │               ├── EvaluationPanel.jsx   # Model evaluation (R², MAE, confusion matrix, stress tests)
│   │               ├── ExplainabilityPanel.jsx  # Feature importance and SHAP visualization
│   │               └── ModelComparison.jsx   # Side-by-side model comparison charts
│   ├── easy_download.py               # Simplified satellite download wrapper
│   └── easy_pipeline.py               # Simplified full pipeline wrapper
│
├── reproduce/                         # Reproduction pipelines
│   ├── requirements.txt               # Python dependencies for reproduction
│   ├── eda_notebook_nuremberg_features.ipynb  # Exploratory data analysis notebook
│   ├── data/                          # Precomputed Nuremberg feature matrices
│   │   ├── nuremberg_features_part{1-3}.parquet  # Feature data (split for Git LFS)
│   │   └── nuremberg_meta.parquet     # Cell metadata (coordinates, grid indices)
│   ├── mlp/                           # MLP model reproduction (Model 2)
│   │   ├── README.md                  # Step-by-step instructions with hardware requirements
│   │   ├── 01_download_data.py        # Download satellite imagery for 120+ cities
│   │   ├── 02_extract_features.py     # Build 1,764-dim feature matrices
│   │   ├── 03_train_bohb_sweep.py     # BOHB hyperparameter sweep (100 trials, 2+ days)
│   │   ├── 04_train_model7.py         # Direct training of deployed Model #7 (~2 hours)
│   │   ├── 05_evaluate_test.py        # Evaluate on 6 held-out test cities
│   │   ├── 06_export_onnx.py          # Export trained model to ONNX
│   │   └── diagrams/                  # EDA plots (band distributions, correlations, indices)
│   └── pixel/                         # Pixel model reproduction (Model 1)
│       ├── README.md                  # Step-by-step instructions with hardware requirements
│       ├── 01_download_data.py        # Download data (same as MLP, can skip if already done)
│       ├── 02_train_catboost.py       # Train CatBoost classifier (4 configs, GPU)
│       └── 03_predict_nuremberg.py    # Generate prediction bins for the dashboard
│
├── scripts/                           # Nuremberg-specific model and data scripts
│   ├── setup_2-step_model.py          # Data preparation for Models 3 + 4 (labels, features, splits)
│   ├── explainable_stage1.py          # Train RF and LR change-prediction models (Models 3 + 4)
│   ├── experimental_nuremberg.py      # Generate experimental predictions for dashboard
│   ├── hpo_experimental.py            # Optuna HPO for RF change model
│   ├── hpo_explainable.py             # Optuna HPO for LR explainable model
│   ├── hpo_plots.py                   # Visualization of HPO results
│   ├── hpo_results.json               # Saved HPO trial results (RF)
│   ├── hpo_explainable_results.json   # Saved HPO trial results (LR)
│   ├── calculate_rf_fcr.py            # Calculate false change rate for RF model
│   ├── nuremberg_geo.py               # Nuremberg geographic utilities
│   ├── nuremberg_stats.py             # District-level statistics computation
│   ├── nuremberg_bezirke_stats_pipeline.py  # Full pipeline: download + process district stats
│   ├── nuremberg_geo_stats_pipeline.py      # Geographic statistics pipeline
│   ├── nuremberg_main_example.py      # Example usage of the Nuremberg model pipeline
│   ├── precompute_district_stats.py   # Precompute per-district aggregated statistics
│   └── rasterize_stats.py            # Rasterize district stats to pixel grid
│
├── models/
│   └── explainable_stage1.joblib      # Trained Models 3 + 4 (RF + LR, serialized)
│
├── data/
│   └── pipeline_output/models/onnx/   # Deployed ONNX model for global predictions
│       ├── mlp_fold_0.onnx            # MLP model weights
│       ├── mlp_scaler_0.json          # Feature standardization parameters
│       └── mlp_cols.json              # Feature column names and ordering
│
├── docs/                              # Extended documentation
│   ├── CLI.md                         # Rust CLI standalone usage and Docker workflows
│   ├── DEPLOY.md                      # Deployment architecture and Docker build
│   ├── DOWNLOAD.md                    # Downloading satellite imagery for any location
│   └── worldcover_class_mapping.md    # ESA 11-class to 7-class remapping table
│
├── figures/
│   └── hpo/                           # HPO visualization plots
│       ├── optimization_history.png   # Trial progression
│       ├── param_vs_value.png         # Parameter importance
│       ├── duration_vs_value.png      # Training time vs score
│       ├── fold_variance.png          # Cross-validation fold variance
│       └── summary_card.png           # HPO summary
│
├── config/
│   └── data_config.yml                # Pipeline config (AOI bounds, bands, seasons, cloud thresholds)
├── config.py                          # Python config (city lists, train/val/test splits)
├── download.py                        # Easy one-file satellite download (uses Docker)
├── pipeline.py                        # Easy one-file full pipeline (uses Docker)
├── verify_api.py                      # API endpoint verification script
├── Dockerfile                         # Multi-stage build (Rust → Node.js → Python runtime)
├── .github/workflows/ci.yml           # CI/CD: build Docker image, push to GHCR
├── .dockerignore                      # Docker build exclusions
├── .gitattributes                     # Git LFS tracking for binary files
├── .gitignore                         # Ignored files and directories
├── pyproject.toml                     # Python project metadata
├── requirements-docker.txt            # Slim Python dependencies for Docker image
├── LICENSE                            # MIT License
├── README.md                          # This file
└── README_TEAM.md                     # Internal team notes
```

---

## Using the Rust Binary

The `terrapulse` binary handles satellite data download, feature extraction, and ONNX inference. It is written in Rust for maximum throughput: parallel scene downloads (tokio), parallel feature extraction (rayon), and chunked ONNX inference.

### Compile from source

```bash
cd terrapulse
cargo build --release
# Binary at: terrapulse/target/release/terrapulse
```

Requires Rust 1.83+ (install via [rustup](https://rustup.rs)).

### Download satellite data only

The easiest way to download satellite imagery is through the provided wrapper script:

```bash
python download.py --bbox 10.95 49.38 11.20 49.52 --output ./satellite_data
python download.py --bbox 2.25 48.81 2.42 48.90 --years 2023 2024 --output ./paris --region paris
```

This script handles Docker image pulling and pipeline invocation automatically. Requires Docker installed.

### Full pipeline (download + extract + predict)

```bash
python pipeline.py --bbox 10.95 49.38 11.20 49.52 --years 2020 2021 --output ./results
```

For standalone CLI usage without Docker, see [docs/CLI.md](docs/CLI.md).

---

## Reproduction

Complete scripts for reproducing both main models are in `reproduce/`. Each subdirectory has its own README with detailed instructions.

### Hardware requirements

- **RAM**: 32 GB minimum (all background applications closed)
- **GPU**: NVIDIA with 6+ GB VRAM and TF32 support (tested on RTX 4070)
- **Storage**: ~300 GB for raw imagery across 100+ cities
- **Data download**: ~48 hours (Planetary Computer rate limits)

### MLP model (Model 2)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r reproduce/requirements.txt

python reproduce/mlp/01_download_data.py       # Download satellite data
python reproduce/mlp/02_extract_features.py     # Build feature matrices
python reproduce/mlp/04_train_model7.py         # Train deployed model (~2h)
python reproduce/mlp/05_evaluate_test.py        # Evaluate on 6 test cities
python reproduce/mlp/06_export_onnx.py          # Export to ONNX format
```

Full BOHB sweep (`03_train_bohb_sweep.py`) takes 2+ days. Use `04_train_model7.py` to directly train the deployed architecture.

See [reproduce/mlp/README.md](reproduce/mlp/README.md) for details.

### Pixel-wise CatBoost model (Model 1)

```bash
python reproduce/pixel/01_download_data.py      # Same data as MLP (skip if done)
python reproduce/pixel/02_train_catboost.py      # Train 4 configs (~1-2h on GPU)
python reproduce/pixel/03_predict_nuremberg.py   # Generate dashboard bins
```

See [reproduce/pixel/README.md](reproduce/pixel/README.md) for details.

### Two-step change model (Models 3 + 4)

The Nuremberg-specific change prediction models are trained via:

```bash
python scripts/setup_2-step_model.py             # Prepare data and spatial splits
python scripts/explainable_stage1.py              # Train RF and LR models
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [CLI.md](docs/CLI.md) | Standalone Rust binary usage and Docker-based workflows |
| [DEPLOY.md](docs/DEPLOY.md) | Deployment architecture, Docker multi-stage build, CI/CD |
| [DOWNLOAD.md](docs/DOWNLOAD.md) | Downloading satellite imagery for any location |
| [WorldCover mapping](docs/worldcover_class_mapping.md) | ESA 11-class to 7-class remapping |
| [Data format](src/dashboard/DATA_FORMAT.md) | Binary file format for dashboard data |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Inference pipeline | Rust (tokio, reqwest, rayon, ort) |
| ML inference | ONNX Runtime |
| Backend | Python (FastAPI, uvicorn) |
| Frontend | React 19, Vite, MapLibre GL, deck.gl, Chart.js |
| Pixel model | CatBoost (GPU training with CUDA) |
| MLP model | PyTorch (mixed-precision FP16, CUDA), exported to ONNX |
| Change models | scikit-learn Random Forest + Logistic Regression |
| Container | Docker (multi-stage: Rust build → Node.js build → Python runtime) |
| CI/CD | GitHub Actions → GitHub Container Registry |
| Deployment | Azure Container Apps (Germany West Central) |
| Data sources | Sentinel-2 L2A, Sentinel-1 GRD, ESA WorldCover 10m (via Planetary Computer STAC API) |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
