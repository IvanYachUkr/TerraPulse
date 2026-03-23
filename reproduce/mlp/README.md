# Reproducing the MLP Model (V10 BOHB)

End-to-end reproducibility for the deployed MLP land-cover model.

## Overview

| Item | Value |
|------|-------|
| **Deployed model** | Trial #77 — `T_1024_512_256_64` GELU, ~2.5M params |
| **Training data** | ~92 cities × 100m grid cells, features_v7 (Sentinel-2/S1 + texture) |
| **Validation** | 23 cities (label-balanced split) |
| **Test** | 6 held-out cities (nuremberg, ankara, sofia, riga, edinburgh, palermo) |
| **Expected test score** | Combined=0.789 (Top-1=90.2%, R²=0.676 at 5% threshold) |

## Prerequisites

```bash
# Python packages
pip install numpy pandas torch scikit-learn pyarrow rasterio catboost onnx onnxruntime

# For BOHB sweep only
pip install hpbandster ConfigSpace serpent

# Rust binary (must be built once)
cd terrapulse && cargo build --release
```

## Steps

### 1. Download satellite data
```bash
python reproduce/mlp/01_download_data.py                    # all ~120 cities
python reproduce/mlp/01_download_data.py --cities munich     # single city test
python reproduce/mlp/01_download_data.py --list-cities       # show available
```

### 2. Extract features (Rust)
```bash
python reproduce/mlp/02_extract_features.py
```

### 3a. Full BOHB sweep (optional, ~24h on GPU)
```bash
python reproduce/mlp/03_train_bohb_sweep.py --max-trials 100
python reproduce/mlp/03_train_bohb_sweep.py --max-trials 3 --max-budget 10  # quick test
```

### 3b. Train Model #7 directly (recommended, ~2h on GPU)
```bash
python reproduce/mlp/04_train_model7.py
python reproduce/mlp/04_train_model7.py --max-epochs 50  # quick test
```

### 4. Evaluate on test cities
```bash
python reproduce/mlp/05_evaluate_test.py
```

### 5. Export to ONNX
```bash
python reproduce/mlp/06_export_onnx.py
```

## Model #7 Hyperparameters

```json
{
  "arch": "T_1024_512_256_64",
  "activation": "gelu",
  "dropout": 0.3255,
  "input_dropout": 0.0031,
  "lr": 0.00103,
  "weight_decay": 0.000537,
  "mixup_alpha": 0.2975,
  "mixup_prob": 0.4285,
  "label_threshold": 0.021,
  "batch_size": 4096
}
```

## Output Files

| File | Description |
|------|-------------|
| `data/cities/models_v10_bohb/trial_77_T_1024_512_256_64.pt` | PyTorch weights |
| `data/cities/models_v10_bohb/scaler.pkl` | StandardScaler |
| `data/cities/models_v10_bohb/mlp_cols.json` | Feature column order |
| `data/pipeline_output/models/onnx/mlp_fold_0.onnx` | ONNX model |
| `data/pipeline_output/models/onnx/mlp_scaler_0.json` | Scaler for Rust |
| `data/pipeline_output/models/onnx/model_config.json` | Model config |
