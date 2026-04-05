# Raw-Band MLP Comparison Study

End-to-end pipeline for training and comparing pixel-level land cover
classifiers on **raw Sentinel-2 + Sentinel-1 bands** (no handcrafted features).

## Results Summary

| Model | Input | Features | Params | Val Acc | Test Acc |
|-------|-------|----------|--------|---------|----------|
| `mlp_1x1` | Single pixel | 72 | 212K | 78.0% | 79.2% |
| **`mlp_3x3`** | **3×3 patch** | **648** | **1.34M** | **80.1%** | **83.2%** |
| `mlp_5x5` | 5×5 patch | 1800 | 2.52M | 80.3% | 81.7% |
| `mlp_3x3_plus` | 3×3 + 145 indices | 793 | 1.49M | 78.8% | 82.3% |
| `mlp_3x3_plus_big` | 3×3 + 145 indices | 793 | 4.40M | 81.0% | 82.7% |
| CatBoost V5 | Handcrafted features | ~190 | 3000 trees | — | 85.2% |
| **Ensemble (CB=0.7)** | **Both pipelines** | — | — | — | **85.6%** |

### Per-Class Test Accuracy

| Class | mlp_1x1 | mlp_3x3 | mlp_5x5 | 3x3_plus | 3x3_big | CatBoost | Ensemble |
|-------|---------|---------|---------|----------|---------|----------|----------|
| tree_cover | 77.1% | 80.9% | 81.9% | 79.7% | 78.9% | 86.9% | 86.0% |
| shrubland | 80.7% | 77.7% | 78.0% | 80.8% | 76.8% | 5.5% | 20.5% |
| grassland | 54.8% | 64.7% | 59.2% | 62.5% | 65.0% | 76.0% | 75.4% |
| cropland | 79.0% | 80.7% | 80.1% | 83.2% | 80.7% | 78.6% | 79.7% |
| built_up | 91.2% | 92.6% | 89.2% | 90.9% | 92.3% | 90.1% | 91.7% |
| bare_sparse | 59.7% | 59.1% | 73.2% | 63.3% | 67.5% | 28.8% | 39.8% |
| water | 98.8% | 98.9% | 98.8% | 99.1% | 99.2% | 98.9% | 99.0% |

### Per-City Test Accuracy

| City | mlp_1x1 | mlp_3x3 | mlp_5x5 | 3x3_plus | 3x3_big | CatBoost |
|------|---------|---------|---------|----------|---------|----------|
| Nuremberg | 89.7% | 91.2% | 91.0% | 90.9% | 91.1% | 90.6% |
| Ankara | 65.3% | 70.0% | 63.6% | 67.9% | 67.1% | 77.0% |
| Sofia | 80.8% | 81.7% | 81.1% | 81.9% | 81.8% | 82.4% |
| Riga | 85.1% | 87.3% | 86.3% | 86.6% | 87.8% | 87.0% |
| Edinburgh | 89.9% | 90.5% | 90.3% | 90.1% | 90.9% | 90.7% |
| Palermo | 76.2% | 78.5% | 78.0% | 76.5% | 77.7% | 83.3% |

## Key Findings

1. **3×3 spatial context is the sweet spot** — consistent +2% over pixel-level
2. **5×5 hurts** — too much noise for an MLP; convolutions would be needed
3. **Adding handcrafted indices introduces redundancy** — MLPs can't jointly
   exploit raw bands and derived ratios as well as tree-based models
4. **Scaling capacity 3× (1.3M → 4.4M params) doesn't help** test accuracy
5. **All MLPs plateau at 81–83%** — the architecture is the bottleneck
6. **CatBoost and MLP are complementary** on rare classes (shrubland, bare_sparse)
7. **Simple ensemble gives marginal gains** (+0.47%); per-class weighting could do better

## Model Variants

### Raw-band MLPs (no handcrafted features)

| Model | Input | What it tests |
|-------|-------|---------------|
| `mlp_1x1` | Single pixel, 72 raw bands | Baseline: raw bands alone |
| `mlp_3x3` | 3×3 patch, 648 raw bands | Does local spatial context help? |
| `mlp_5x5` | 5×5 patch, 1800 raw bands | Does more spatial context help? |

### Hybrid MLPs (raw bands + engineered)

| Model | Input | What it tests |
|-------|-------|---------------|
| `mlp_3x3_plus` | 3×3 raw + 145 CatBoost indices | Can hybrid features beat raw-only? |
| `mlp_3x3_plus_big` | Same, 3× bigger network | Is the model too small for 793 features? |

### Raw features per pixel (72 total)

- Sentinel-2: 10 spectral bands × 6 temporal slots (2 years × 3 seasons) = 60
- Sentinel-1: VV + VH × 6 temporal slots = 12

### CatBoost-style indices (145 total, center pixel only)

- 9 spectral indices × 6 slots = 54 (NDVI, NDWI, NDBI, NDMI, NBR, BSI, EVI2, NDRE1, NDRE2)
- Seasonal index diffs × 4 transitions × 9 = 36
- Inter-annual index diffs × 3 seasons × 9 = 27
- Range features (spring→autumn) × 4 indices × 2 years = 8
- SAR VV/VH ratios × 6 slots = 6
- SAR seasonal diffs × 4 transitions × 2 bands = 8
- SAR inter-annual diffs × 3 seasons × 2 bands = 6

## Architecture

All MLPs use the expand-contract architecture with:
- LayerNorm + GELU activation + Dropout(0.1)
- Log-softmax output head
- AdamW optimizer, ReduceLROnPlateau scheduler
- Early stopping (patience=20, max 300 epochs)

| Model | Hidden layers |
|-------|---------------|
| `mlp_1x1` | [512, 256, 128, 64] |
| `mlp_3x3` | [1024, 512, 256, 64] |
| `mlp_5x5` | [1024, 512, 256, 64] |
| `mlp_3x3_plus` | [1024, 512, 256, 64] |
| `mlp_3x3_plus_big` | [2048, 1024, 512, 256, 64] |

## Usage

### Train a model

```bash
python reproduce/models/01_train_raw_mlp.py --model mlp_3x3
python reproduce/models/01_train_raw_mlp.py --model mlp_3x3_plus_big
```

Available models: `mlp_1x1`, `mlp_3x3`, `mlp_5x5`, `mlp_3x3_plus`, `mlp_3x3_plus_big`

### Prerequisites

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install rasterio numpy scikit-learn pyarrow
```

## Output Files

| File | Description |
|------|-------------|
| `checkpoints/<model>.pt` | Trained model weights |
| `checkpoints/<model>_scaler.pkl` | StandardScaler for input features |
| `checkpoints/<model>_metrics.json` | Val + test accuracy, per-class, per-city |
| `checkpoints/comparison.json` | All models side-by-side with ensemble + conclusions |

## Train/Val/Test Split

- **Train**: 92 cities (30K pixels/city for 3×3 variants, 200K for 1×1)
- **Val**: 23 cities held from train set (15K pixels/city)
- **Test**: 6 held-out cities (500K pixels/city):
  nuremberg, ankara_test, sofia_test, riga_test, edinburgh_test, palermo_test
