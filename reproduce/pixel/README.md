# Reproducing the Pixel-Wise CatBoost Classifier (V5)

End-to-end reproducibility for the per-pixel CatBoost land-cover model.

## Overview

| Item | Value |
|------|-------|
| **Model** | CatBoost V5 (`deep_unweighted`): depth=8, trees=3000, lr=0.03 |
| **Features** | 217 per-pixel features: S2 bands + spectral indices + SAR + temporal diffs |
| **Training** | ~100 cities, 150K sampled pixels per city, GPU-accelerated |
| **Validation** | 15 geographically diverse cities |
| **Dashboard** | Nuremberg, 10 resolutions × 8 year-pairs = 80 prediction bins |

## Prerequisites

```bash
pip install numpy pandas catboost rasterio pyarrow scikit-learn

# Rust binary (for data download)
cd terrapulse && cargo build --release
```

## Steps

### 1. Download satellite data
```bash
python reproduce/pixel/01_download_data.py
# Or skip if already run for MLP — same data
```

### 2. Train CatBoost models (4 configs)
```bash
python reproduce/pixel/02_train_catboost.py                 # full training
python reproduce/pixel/02_train_catboost.py --cities munich  # quick test
```

### 3. Generate Nuremberg dashboard predictions
```bash
python reproduce/pixel/03_predict_nuremberg.py
```

## Deployed Config (`deep_unweighted`)

| Param | Value |
|-------|-------|
| Depth | 8 |
| Trees | 3000 |
| Learning rate | 0.03 |
| L2 regularization | 3.0 |
| Class weights | None (unweighted) |
| Early stopping | 80 rounds |

## Output Files

| File | Description |
|------|-------------|
| `data/cities/models_pixel_v5/catboost_pixel_v5_*.cbm` | Trained CatBoost models |
| `data/cities/models_pixel_v5/metrics_pixel_v5.json` | Evaluation metrics |
| `src/dashboard/data/nuremberg_dashboard/nuremberg_pred_*.bin` | Dashboard predictions |
