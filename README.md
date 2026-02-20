<p align="center">
  <h1 align="center">🌍 TerraPulse</h1>
  <p align="center">
    <strong>Land-cover prediction & change detection from satellite imagery</strong>
  </p>
  <p align="center">
    Sentinel-2 multi-spectral imagery → ML models → interactive dashboard
  </p>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-dashboard">Dashboard</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#%EF%B8%8F-license">License</a>
</p>

---

## ✨ What It Does

TerraPulse predicts the **land-cover composition** of any area on Earth from Sentinel-2 satellite imagery. For each 100 m × 100 m grid cell, it predicts proportions of 7 land-cover classes:

| Class | Example |
|-------|---------|
| 🌳 Tree Cover | Forests, urban parks |
| 🌿 Shrubland | Low woody vegetation |
| 🌾 Grassland | Pastures, meadows |
| 🌽 Cropland | Agricultural fields |
| 🏠 Built-up | Buildings, roads, infrastructure |
| 🪨 Bare/Sparse | Exposed soil, construction sites |
| 💧 Water | Rivers, lakes, ponds |

The production model — a tapered MLP trained on **14 European cities** — achieves **R² = 0.862** with **2.43 pp mean absolute error** on the held-out Nuremberg test region.

---

## 🚀 Quickstart

### Option 1: Docker (Recommended)

```bash
# Pull the pre-built image
docker pull ghcr.io/ivanyachukr/terrapulse:latest

# Run the dashboard
docker run -p 8000:8000 ghcr.io/ivanyachukr/terrapulse:latest

# Open in browser
# → http://localhost:8000
```

The Docker image contains everything: Rust binary, ONNX model, frontend, research data, and API server.

### Option 2: Build from source

**Prerequisites**: Rust 1.83+, Python 3.12+, Node.js 22+

```bash
# Clone
git clone https://github.com/IvanYachUkr/TerraPulse.git
cd TerraPulse

# Build Rust binary
cd terrapulse
cargo build --release
cd ..

# Install Python dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements-docker.txt

# Build frontend
cd src/dashboard/frontend
npm ci
npm run build
cd ../../..

# Run the dashboard
python -m uvicorn src.dashboard.api:app --port 8000
```

### Option 3: Development mode (hot reload)

```bash
# Terminal 1: API server
python -m uvicorn src.dashboard.api:app --port 8000 --reload

# Terminal 2: Frontend dev server
cd src/dashboard/frontend
npm run dev
# → http://localhost:5173 (proxies API to :8000)
```

---

## 📊 Dashboard

The dashboard has two modes:

### Research Mode

Explore precomputed results for the **Nuremberg study area** (29,946 grid cells):

- **Interactive map** — view labels, predictions, change detection, and spatial CV folds
- **3 models** — MLP (R²=0.787), LightGBM (R²=0.736), Ridge (R²=0.423)
- **Cell inspector** — click any cell for detailed predictions across all models
- **Model comparison** — R², MAE, Aitchison distance bar charts
- **Evaluation panel** — per-class metrics, stress tests, failure analysis
- **Explainability** — feature importance, SHAP, conformal prediction intervals

### Deploy Mode

**Live prediction for any region on Earth**:

1. Select a bounding box on the map or enter coordinates
2. Choose prediction years (2020–2025)
3. The Rust pipeline downloads satellite imagery, extracts features, and runs ONNX inference
4. View predictions, labels, and grid on the interactive map
5. Real-time progress updates showing each pipeline stage

---

## 📚 Documentation

Detailed research-paper-style documentation for each component:

| Document | Contents |
|----------|----------|
| **[Research](docs/RESEARCH.md)** | Problem definition, data acquisition, feature engineering, 6-way spatial split comparison, 798+ MLP runs, 480 LightGBM runs, 565 ablation studies, all experiments and results |
| **[Explainability](docs/EXPLAINABILITY.md)** | SHAP analysis (MLP + LightGBM), permutation importance, conformal prediction intervals, stress tests, failure analysis, helpful vs misleading explanations |
| **[Deploy](docs/DEPLOY.md)** | Rust pipeline architecture, COG reader, feature extraction, ONNX inference, Docker multi-stage build, CI/CD, production fixes, testing |

Additional documentation:
- [WorldCover Class Mapping](docs/worldcover_class_mapping.md) — ESA class → model class mapping
- [Project Checklist](project_checklist.md) — Complete phase-by-phase progress tracker

---

## 🏗 Architecture

```
TerraPulse/
├── terrapulse/                    # Rust pipeline (6,500+ LOC)
│   ├── src/
│   │   ├── main.rs                # CLI entry point
│   │   ├── composite.rs           # Scene download & compositing
│   │   ├── cog.rs                 # COG reader (zero-dependency TIFF)
│   │   ├── extract.rs             # Feature extraction orchestrator
│   │   ├── features.rs            # Core feature computation
│   │   ├── predict.rs             # ONNX inference
│   │   ├── sar_download.rs        # Sentinel-1 SAR processing
│   │   ├── reproject.rs           # CRS reprojection
│   │   ├── grid.rs                # GeoJSON grid generation
│   │   ├── labels.rs              # WorldCover label download
│   │   └── ...
│   ├── Cargo.toml
│   └── tests/
├── src/
│   └── dashboard/
│       ├── api.py                 # FastAPI backend
│       ├── deploy_runner.py       # Pipeline orchestrator
│       ├── data/                  # Precomputed research JSONs
│       └── frontend/              # React + Vite + deck.gl
│           ├── src/
│           ├── package.json
│           └── vite.config.js
├── scripts/                       # Training & analysis scripts
├── models/                        # Trained model weights
├── data/                          # Pipeline data (gitignored)
├── docs/                          # Detailed documentation
├── config/data_config.yml         # Pipeline configuration
├── Dockerfile                     # Multi-stage build
├── .github/workflows/ci.yml       # CI/CD pipeline
└── requirements-docker.txt        # Slim Python deps
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Pipeline | Rust (tokio, reqwest, rayon, ort) |
| ML Inference | ONNX Runtime |
| Backend | Python (FastAPI, uvicorn) |
| Frontend | React 19, Vite, MapLibre GL, deck.gl, Chart.js |
| Container | Docker (multi-stage) |
| CI/CD | GitHub Actions → GHCR |
| Data | Sentinel-2 L2A via Planetary Computer |
| Labels | ESA WorldCover 10 m |

### Models

| Model | R² | MAE | Parameters | Context |
|-------|:--:|:---:|:----------:|--------|
| **MLP (production)** | **0.862** | **2.43 pp** | 917K | Multi-city (14 cities → Nuremberg), deployed via ONNX |
| MLP (5-fold CV) | 0.787 | 2.50 pp | 917K | Single-city spatial CV research metric |
| LightGBM | 0.736 | 2.99 pp | — | Tree-based comparison (research) |
| Ridge | 0.423 | 5.63 pp | — | Interpretable baseline (research) |

---

## 🧪 Testing

```bash
# Rust tests (17 total: 15 unit + 2 integration)
cd terrapulse
cargo test --release

# Python tests
pytest tests/
```

---

## 🔧 Configuration

Pipeline configuration is in [`config/data_config.yml`](config/data_config.yml):

- **AOI**: Bounding box, EPSG code
- **Sentinel-2**: Bands, seasons, cloud cover thresholds
- **WorldCover**: Class mapping, tile IDs
- **Grid**: Cell size (100 m), pixel size (10 m)
- **Split**: Block size, number of folds, buffer distance

---

## 🛡️ License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
