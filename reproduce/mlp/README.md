# Reproducing the MLP Model (V10 BOHB)

> [!CAUTION]
> **Minimum Hardware & Time Requirements**
>
> - **Data download**: ~48 hours of real time (120+ cities × 6 seasons × satellite composites from Planetary Computer + ESA WorldCover tiles)
> - **RAM**: Bare minimum **32 GB**, with all background applications (browser, IDE, etc.) closed
> - **GPU**: NVIDIA GPU with **≥6 GB VRAM** and TF32 (TensorFloat-32) support (Ampere architecture or newer). Tested on **GeForce RTX 4070 Laptop GPU**
> - **BOHB sweep** (`03_train_bohb_sweep.py`): **>2 days** on a single GPU to complete 100 trials
> - **Direct Model #7 training** (`04_train_model7.py`): ~2 hours on GPU

End-to-end reproducibility for the deployed MLP land-cover model.

## Overview

| Item | Value |
|------|-------|
| **Deployed model** | Trial #77 — `T_1024_512_256_64` GELU, ~2.5M params |
| **Training data** | ~92 cities × 100m grid cells, features_v7 (Sentinel-2/S1 + texture) |
| **Validation** | 23 cities (label-balanced split) |
| **Test** | 6 held-out cities (nuremberg, ankara_test, sofia_test, riga_test, edinburgh_test, palermo_test) |
| **Expected test score** | Combined=0.789 (Top-1=90.2%, R²=0.676 at 5% threshold) |

## Prerequisites

```bash
# Python packages (install PyTorch with CUDA first)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r reproduce/requirements.txt

# Rust toolchain (latest stable version required)
# Install from https://rustup.rs if not already installed
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
> Steps 4 and 5 work with either path (3a or 3b). No sweep log needed.

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
