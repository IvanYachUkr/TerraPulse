# Production Scripts

This folder contains the production-ready scripts for the TerraPulse project, powered by the high-performance Rust feature extractor.

## Components

1.  **`extract.py`**:
    -   Loads Sentinel-2 imagery (GeoTIFFs).
    -   Extracts features using `terrapulse_features` (Rust extension).
    -   Computes Year-Over-Year and Seasonal Deltas.
    -   Outputs a single Parquet file (`data/processed/v2/features_rust_production.parquet`).
    -   **Performance**: ~20x faster than Python implementation.
    -   **Correctness**: Verified to match Python features (< 0.1% diff).

2.  **`train_mlp.py`**:
    -   Trains the "Champion" MLP model (Deep Learning).
    -   Uses PyTorch (CPU/GPU).
    -   Feature Set: `bi_LBP` (Bands + Indices + LBP).
    -   Saves model to `models/production_mlp/`.

3.  **`train_tree.py`**:
    -   Trains the "Best Tree" model (LightGBM).
    -   Feature Set: `VegIdx + RedEdge + TC + NDTI + IRECI + CRI1`.
    -   Saves model to `models/production_tree/`.

## Usage

### 1. Feature Extraction
Run this first to generate the optimized feature dataset.
```bash
python production/extract.py
```

### 2. Training
Train the models using the extracted features.
```bash
python production/train_mlp.py
python production/train_tree.py
```

## Dynamic Inference
To use in a dynamic application (e.g., selecting a region on a map):
```python
from production.extract import extract_features_pipeline

# Dictionary of rasters: (year, season) -> (spectral, valid_fraction)
# spectral: (10, H, W) numpy array
# valid_fraction: (H, W) numpy array
data = {
    (2020, "spring"): load_raster(...),
    ...
}

# Extract DataFrame (feature extraction + deltas)
df_features = extract_features_pipeline(data)

# Load Model
# ... (Standard PyTorch/LightGBM loading)

# Predict
preds = model.predict(df_features)
```
