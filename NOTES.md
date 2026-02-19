# TerraPulse — Project State Notes
_Last updated: 2026-02-19_

## Current Status
**All pipeline stages working end-to-end in pure Rust, verified on Nuremberg.**

### Verification Result (Nuremberg held-out test, 2020-2021)
| Metric | Python pipeline | Rust pipeline |
|--------|----------------|---------------|
| Dominant class accuracy | 92.7% | **93.1%** |
| Mean MAE% | 3.18 | **3.13** |
| Features matched | 1764/1764 | 1764/1764 |
| SAR VV mean | 0.1251 | 0.1253 |

Rust is at parity (slightly better) with the Python reference pipeline.

---

## Key File Locations

### Models (NOT in git — too large)
```
data/pipeline_output/models/onnx/
    mlp_fold_0.onnx          # ONNX model (fold 0 only)
    mlp_fold_0.onnx.data     # external weights file
    mlp_cols.json            # 1764 feature column names (ordered)
    mlp_scaler_0.json        # StandardScaler mean/scale

data/pipeline_output/models/
    mlp_fold_0..4.pt         # PyTorch weights (5 folds, ~22MB each)
    mlp_scaler_0..4.pkl      # Sklearn scalers (5 folds)
    mlp_meta.json            # model architecture metadata
    tree_fold_0..4.pkl       # LightGBM/tree models (5 folds)
    tree_meta.json
```

### Training Data (NOT in git)
```
data/cities/<city>/
    features/features_rust_2020_2021.parquet   # Rust-extracted features
    labels_2020.parquet                         # WorldCover labels
    labels_2021.parquet

data/raw/v2/                   # Python-downloaded Sentinel-2 TIFs (2020-2025)
data/test_rust_v3/raw/         # Rust-downloaded S2+S1 (2020-2021, Nuremberg)
```

### Labels path for Nuremberg
`data/cities/nuremberg/labels_2021.parquet`

---

## Rust Binary Commands

### Build
```powershell
cd terrapulse
cargo build --release -p terrapulse
```
Binary at: `terrapulse/target/release/terrapulse.exe`

### Full pipeline (download + extract + predict)
```powershell
.\terrapulse\target\release\terrapulse.exe pipeline `
  --bbox 11.0754 49.4391 11.1811 49.4762 `
  --years 2020 --years 2021 `
  --region nuremberg `
  --data-dir data\test_rust_v3 `
  --anchor-ref data\raw\v2\sentinel2_nuremberg_2020_spring.tif `
  --models-dir data\pipeline_output\models\onnx
```

### Skip stages (for re-runs)
Add `--skip-download`, `--skip-extract`, or `--skip-predict` as needed.

### Download only (no extract/predict)
```powershell
... --skip-extract --skip-predict
```

---

## What Was Fixed (SAR)

Three bugs fixed to get SAR working in Rust:

1. **OOM crash** — `buffer_unordered(4)` limits concurrent SAR scene downloads
2. **37%→100% coverage** — replaced global affine GCP fit with `GcpGrid` piecewise bilinear interpolation (10×21 GCP grid, Newton's method inverse)
3. **TIFF writer** — switched to `PlanarConfiguration=1` (interleaved `[VV_0,VH_0,VV_1,VH_1,...]`) for compatibility with both rasterio and the Rust `tiff` crate decoder

---

## Dashboard Backend

```
src/dashboard/api.py           # FastAPI routes
src/dashboard/deploy_runner.py # Job orchestration — calls Rust binary via subprocess
```

Python still required for:
- `rasterio`: anchor GeoTIFF creation, WorldCover reprojection, grid GeoJSON CRS transform
- `pandas/numpy`: parquet→JSON conversion
- `fastapi/uvicorn`: HTTP server

---

## Training

Model: **V4 MLP** — depth=5, width=2048, 1764 features (optical + SAR + LBP + Tasseled Cap + phenological)

Training cities (5-fold CV): Berlin, Hamburg, Munich, Frankfurt, Leipzig, ... (see `data/cities/`)
Test/held-out: **Nuremberg**

Best CV baseline: R²=0.759, MAE=2.554% (v6 reference)

To retrain:
```powershell
python rust_features/train_only/train_mlp.py
```

To export to ONNX:
```powershell
python rust_features/production_scr/export_models_onnx.py
```

---

## Git Remote
`https://github.com/IvanYachUkr/TerraPulse.git` (main branch)
