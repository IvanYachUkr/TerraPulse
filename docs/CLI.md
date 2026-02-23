# 🔧 TerraPulse Rust CLI — Standalone Usage

The Docker image ships a standalone Rust binary at `/usr/local/bin/terrapulse` that can be used
**independently of the dashboard**. You can run individual pipeline stages — download satellite imagery,
extract features, or run inference — directly from the command line.

---

## Quick Reference

```bash
# Download Nuremberg satellite imagery (2021) and save to your machine
python download.py --bbox 10.95 49.38 11.20 49.52 --output ./satellite_data

# Multiple years
python download.py --bbox 10.95 49.38 11.20 49.52 --years 2023 2024 --output ./satellite_data
```

That's it. The script handles Docker automatically — pulls the image, runs the pipeline,
and saves GeoTIFF files to the folder you specified.

> **Prerequisites**: [Python 3.7+](https://python.org) and [Docker](https://docker.com/get-started) installed and running.

---

## Easy Download

### Step 1: Get the script

Download [`download.py`](https://raw.githubusercontent.com/IvanYachUkr/TerraPulse/main/download.py)
from the repository (it's a single file, no dependencies beyond Python stdlib).

### Step 2: Run it

```bash
python download.py --bbox 10.95 49.38 11.20 49.52 --output ./nuremberg
```

The script will:
1. ✅ Check that Docker is installed and running
2. ✅ Pull the TerraPulse image (first time only, ~400 MB)
3. ✅ Auto-detect the correct map projection (EPSG)
4. ✅ Create the required anchor reference file
5. ✅ Download Sentinel-2 + Sentinel-1 imagery
6. ✅ Save GeoTIFF files to your specified folder

**Arguments:**

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--bbox` | ✅ | — | Bounding box `[west south east north]` in WGS84 degrees |
| `--years` | | `2021` | Years to download (space-separated) |
| `--output` | ✅ | — | Any folder on your machine (absolute or relative path) |
| `--region` | | `region` | Region name (used in filenames) |

**More examples:**
```bash
# Paris, 2023
python download.py --bbox 2.25 48.81 2.42 48.90 --output ./paris --region paris

# New York, 2022-2024
python download.py --bbox -74.02 40.70 -73.93 40.78 --output C:\Users\me\nyc --region nyc --years 2022 2023 2024
```

> **Tip**: Use Google Maps to find coordinates — right-click any point to copy lat/lon.


---

## Saving Output to Your Local Machine

By default, files created inside a Docker container are **lost** when it exits.
To save downloaded imagery, features, or predictions to your local disk,
mount a folder from your machine using `-v`:

**Linux / macOS:**
```bash
# Creates a "terrapulse_data" folder on your desktop, mapped to /data inside the container
docker run --rm -v ~/Desktop/terrapulse_data:/data \
  --entrypoint terrapulse ghcr.io/ivanyachukr/terrapulse:latest \
  download --raw-dir /data/raw ...

# After the run, files appear at ~/Desktop/terrapulse_data/raw/
```

**Windows (PowerShell):**
```powershell
# Mount a local folder — use full Windows path
docker run --rm -v C:\Users\me\terrapulse_data:/data `
  --entrypoint terrapulse ghcr.io/ivanyachukr/terrapulse:latest `
  download --raw-dir /data/raw ...

# After the run, files appear at C:\Users\me\terrapulse_data\raw\
```

> **Key point**: The `-v HOST_PATH:/data` flag is what lets you choose where
> files are saved on your system. Everything the binary writes to `/data/...`
> inside the container appears at `HOST_PATH/...` on your machine.

---

## Bundled Assets

The image comes with these pre-installed:

| Asset | Path in Container |
|-------|-------------------|
| Rust binary | `/usr/local/bin/terrapulse` |
| ONNX model | `/app/models/onnx/mlp_fold_0.onnx` |
| Feature scaler | `/app/models/onnx/mlp_scaler_0.json` |
| Feature columns | `/app/models/onnx/mlp_cols.json` |
| ONNX Runtime | `/usr/local/lib/libonnxruntime.so` |

---

## Subcommands

### 1. `download` — Download satellite imagery

Downloads Sentinel-2 (optical) and Sentinel-1 (SAR) composites from Microsoft Planetary Computer for a given bounding box and set of years. Produces seasonal GeoTIFF composites.

```bash
# Download Nuremberg imagery for 2023 and 2024
docker run --rm -v $(pwd)/data:/data \
  --entrypoint terrapulse ghcr.io/ivanyachukr/terrapulse:latest \
  download \
    --bbox 10.95 49.38 11.20 49.52 \
    --epsg 32632 \
    --years "2023 2024" \
    --region nuremberg \
    --raw-dir /data/raw \
    --anchor-ref /data/anchor.tif
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--bbox` | ✅ | — | Bounding box `[west south east north]` in WGS84 degrees |
| `--epsg` | | `32632` | Target CRS EPSG code (UTM zone) |
| `--years` | ✅ | — | Space-separated years to download (can be a single year) |
| `--region` | | `nuremberg` | Region name (used in output filenames) |
| `--raw-dir` | ✅ | — | Output directory for raw GeoTIFF files |
| `--anchor-ref` | ✅ | — | Path to anchor reference GeoTIFF (defines the output grid) |

> **Note — Anchor Reference TIF**: The anchor reference defines the spatial grid
> (dimensions, resolution, CRS) that all downloaded imagery is reprojected to.
> If you're running the full `pipeline` subcommand via the dashboard,
> it generates this automatically. For standalone use, you can create one
> from any GeoTIFF covering your region, or use a previously downloaded composite.

> **Note — Minimum Years**: You can download a single year (e.g., `--years "2024"`),
> but the `extract` and `predict` stages require **year pairs** (e.g., `2023_2024`)
> because the model uses temporal change features. So in practice, always download
> **at least 2 consecutive years**.

**Output structure**:
```
raw/
├── nuremberg_spring_2023.tif      # Sentinel-2 optical (spring)
├── nuremberg_summer_2023.tif      # Sentinel-2 optical (summer)
├── nuremberg_autumn_2023.tif      # Sentinel-2 optical (autumn)
├── nuremberg_sar_spring_2023.tif  # Sentinel-1 SAR (spring)
├── nuremberg_sar_summer_2023.tif  # Sentinel-1 SAR (summer)
├── nuremberg_sar_autumn_2023.tif  # Sentinel-1 SAR (autumn)
└── ... (same for 2024)
```

---

### 2. `extract` — Extract features from GeoTIFFs

Reads the seasonal GeoTIFFs produced by `download` and computes per-cell (100 m × 100 m) spectral
and SAR features. Outputs a Parquet file per year pair.

```bash
docker run --rm -v $(pwd)/data:/data \
  --entrypoint terrapulse ghcr.io/ivanyachukr/terrapulse:latest \
  extract \
    --year-pairs "2023_2024" \
    --region nuremberg \
    --raw-dir /data/raw \
    --features-dir /data/features
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--year-pairs` | ✅ | — | Space-separated year pairs, e.g., `"2023_2024 2024_2025"` |
| `--region` | | `nuremberg` | Region name (must match filenames from download) |
| `--raw-dir` | ✅ | — | Directory containing downloaded GeoTIFFs |
| `--features-dir` | ✅ | — | Output directory for feature Parquet files |
| `--min-valid-frac` | | `0.3` | Minimum fraction of valid (non-NaN) pixels per cell |

**Output**: `features/features_rust_2023_2024.parquet`

Each row is one 100 m × 100 m grid cell with ~470 feature columns (spectral indices, percentiles, temporal differences, SAR backscatter statistics).

---

### 3. `predict` — Run model inference

Loads the ONNX model and runs inference on feature Parquets. Produces per-cell land-cover probability predictions.

```bash
docker run --rm -v $(pwd)/data:/data \
  --entrypoint terrapulse ghcr.io/ivanyachukr/terrapulse:latest \
  predict \
    --models-dir /app/models/onnx \
    --features-dir /data/features \
    --output-dir /data/predictions \
    --year-pairs "2023_2024"
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--models-dir` | ✅ | — | Path to ONNX model directory (use `/app/models/onnx` for the bundled model) |
| `--features-dir` | ✅ | — | Directory containing feature Parquets from `extract` |
| `--output-dir` | ✅ | — | Output directory for prediction files |
| `--year-pairs` | ✅ | — | Year pairs to predict |

**Output**:
- `predictions/pred_mlp_2023_2024.parquet` — per-cell probability distributions
- `predictions_2024.json` — JSON format for dashboard consumption

Each cell gets 7 probability values (one per land-cover class: tree, shrubland, grassland, cropland, built-up, bare, water).

---

### 4. `pipeline` — Full end-to-end run

Chains all stages: **download → extract → predict → labels → grid**. Produces everything needed for visualization.

```bash
# Full pipeline for Nuremberg, 2023–2025
docker run --rm -v $(pwd)/data:/data \
  --entrypoint terrapulse ghcr.io/ivanyachukr/terrapulse:latest \
  pipeline \
    --bbox 10.95 49.38 11.20 49.52 \
    --epsg 32632 \
    --years "2023 2024 2025" \
    --region nuremberg \
    --data-dir /data/output \
    --anchor-ref /data/anchor.tif \
    --models-dir /app/models/onnx
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--bbox` | ✅ | — | Bounding box in WGS84 |
| `--epsg` | | `32632` | Target CRS EPSG code |
| `--years` | ✅ | — | Years to process (consecutive pairs derived automatically) |
| `--region` | | `nuremberg` | Region name |
| `--data-dir` | | `data/pipeline_output` | Base directory (`raw/`, `features/`, `predictions/` created inside) |
| `--anchor-ref` | ✅ | — | Path to anchor reference GeoTIFF |
| `--models-dir` | ✅ | — | Path to ONNX model directory |
| `--min-valid-frac` | | `0.3` | Minimum valid pixel fraction |
| `--skip-download` | | `false` | Skip download stage (reuse existing TIFs) |
| `--skip-extract` | | `false` | Skip feature extraction (reuse existing Parquets) |
| `--skip-predict` | | `false` | Skip prediction stage |

**Output structure**:
```
output/
├── raw/                           # Downloaded GeoTIFFs
├── features/                      # Feature Parquets
├── predictions/                   # Prediction Parquets + JSONs
├── labels_2021.json               # WorldCover ground truth (≤2021)
└── grid.json                      # GeoJSON grid for visualization
```

---

## End-to-End Example

A complete workflow for predicting land cover in Nuremberg using only the Docker image:

```bash
# 1. Create a working directory
mkdir terrapulse_run && cd terrapulse_run

# 2. Run the full pipeline
#    Nuremberg bbox: [10.95 49.38 11.20 49.52], UTM zone 32N
docker run --rm -v $(pwd):/data \
  --entrypoint terrapulse ghcr.io/ivanyachukr/terrapulse:latest \
  pipeline \
    --bbox 10.95 49.38 11.20 49.52 \
    --epsg 32632 \
    --years "2023 2024" \
    --region nuremberg \
    --data-dir /data/output \
    --anchor-ref /data/output/raw/nuremberg_spring_2023.tif \
    --models-dir /app/models/onnx

# 3. Results are saved to your local machine at ./output/
ls output/predictions/
```

> **Tip**: The `--anchor-ref` requires an existing GeoTIFF. A practical workaround
> for the first run: run `download` separately first, then use one of the output
> TIFs as the anchor for subsequent `pipeline` runs.

---

## Environment Variables

These are pre-configured in the Docker image:

| Variable | Value | Purpose |
|----------|-------|---------|
| `ORT_DYLIB_PATH` | `/usr/local/lib/libonnxruntime.so` | ONNX Runtime shared library |
| `TERRAPULSE_BIN` | `/usr/local/bin/terrapulse` | Binary location |
| `ONNX_MODELS_DIR` | `/app/models/onnx` | Bundled model directory |
| `DEPLOY_DIR` | `/app/deploy_jobs` | Dashboard deploy scratch dir |

If building from source, you'll need to set `ORT_DYLIB_PATH` to point to your local ONNX Runtime installation.

---

## Recommended: Interactive Shell

Instead of typing `docker run ...` for every command, start **one interactive session**
and run as many commands as you like from inside:

```bash
# Start an interactive shell with your local folder mounted
docker run --rm -it -v $(pwd)/data:/data \
  --entrypoint bash ghcr.io/ivanyachukr/terrapulse:latest
```

Now you're inside the container. Run commands directly — no `docker run` needed:

```bash
# Step 1: Download Nuremberg imagery
terrapulse download \
  --bbox 10.95 49.38 11.20 49.52 \
  --years "2023 2024" \
  --region nuremberg \
  --raw-dir /data/raw \
  --anchor-ref /data/raw/nuremberg_spring_2023.tif

# Step 2: Extract features
terrapulse extract \
  --year-pairs "2023_2024" \
  --region nuremberg \
  --raw-dir /data/raw \
  --features-dir /data/features

# Step 3: Run predictions (model is pre-installed at /app/models/onnx)
terrapulse predict \
  --models-dir /app/models/onnx \
  --features-dir /data/features \
  --output-dir /data/predictions \
  --year-pairs "2023_2024"

# Check results
ls /data/predictions/

# Exit when done — files are saved at ./data/ on your machine
exit
```

All output files (TIFs, Parquets, predictions) are automatically saved
to your local `./data/` folder because of the `-v` mount.
