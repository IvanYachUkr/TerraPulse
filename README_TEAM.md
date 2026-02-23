# TerraPulse — Team Setup

## Prerequisites

- [Docker Desktop](https://docker.com/get-started) installed and running

## Quick Start

```bash
# 1. Clone this branch
git clone -b Ivan_deploy https://github.com/stellamoR/ml_final_project.git
cd ml_final_project

# 2. Build the Docker image (~5-8 min, compiles Rust + React + Python)
docker build -t terrapulse .

# 3. Run the dashboard
docker run -p 8000:8000 terrapulse

# 4. Open http://localhost:8000
```

## Project Structure

```
terrapulse/src/       ← Rust pipeline (download, extract, predict)
src/dashboard/api.py  ← Python API (FastAPI)
src/dashboard/frontend/src/  ← React frontend (deck.gl map)
Dockerfile            ← Multi-stage build (Rust → Node → Python runtime)
```

## Making Changes

Edit any file, then rebuild:

```bash
docker build -t terrapulse .
docker run -p 8000:8000 terrapulse
```

No Rust/Node/Python install needed — Docker handles all compilation.

### Common edits

| What | Where |
|------|-------|
| Map UI / sidebar | `src/dashboard/frontend/src/components/` |
| API endpoints | `src/dashboard/api.py` |
| Deploy pipeline | `src/dashboard/deploy_runner.py` |
| Rust download/extract/predict | `terrapulse/src/*.rs` |
| Styles | `src/dashboard/frontend/src/index.css` |

## Download Satellite Imagery (no dashboard needed)

```bash
python download.py --bbox 10.95 49.38 11.20 49.52 --output ./satellite_data
```

See [docs/DOWNLOAD.md](docs/DOWNLOAD.md) for details.
