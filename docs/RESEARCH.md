# TerraPulse Research: Land-Cover Prediction from Satellite Imagery

> **Course**: Machine Learning WT 25/26 — Grabocka, Asano, Frey — UTN
> **Objective**: Build a tabular ML system that predicts land-cover composition and change in Nuremberg from Sentinel-2 multi-spectral imagery.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Data Acquisition](#2-data-acquisition)
3. [Label Engineering](#3-label-engineering)
4. [Feature Engineering](#4-feature-engineering)
5. [Dataset Merging & Delta Features](#5-dataset-merging--delta-features)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Train/Test Split Design](#7-traintest-split-design)
8. [Modeling](#8-modeling)
9. [Evaluation Beyond Accuracy](#9-evaluation-beyond-accuracy)
10. [Results Summary](#10-results-summary)

---

## 1. Problem Definition

### Task

Predict the **compositional land-cover proportions** of 100 m × 100 m grid cells from Sentinel-2 multi-spectral satellite imagery. Each cell's prediction is a 7-class probability vector:

| Index | Class | Nuremberg Prevalence |
|:-----:|-------|---------------------:|
| 0 | Tree Cover | 30.9% |
| 1 | Shrubland | 0.5% |
| 2 | Grassland | 13.5% |
| 3 | Cropland | 12.0% |
| 4 | Built-up | 32.5% |
| 5 | Bare/Sparse | 1.0% |
| 6 | Water | 9.5% |

### Spatial Unit

Each grid cell covers **10 × 10 Sentinel-2 pixels** (100 m²). The Nuremberg AOI contains **29,946 cells** (186 cols × 161 rows).

**Grid size justification**: Connected-component analysis of pixel-level WorldCover change (2020 → 2021) showed 76% of changed regions are ≤ 4 pixels (median = 2 px). At 100 m resolution, these tiny artifacts produce only 1–4% proportion noise, naturally smoothed out. Real change (≥ 50 px regions, ~1.5% of all regions) still produces 15–25% proportion shifts per cell. Smaller cells (50 m = 25 px/cell) would amplify label artifacts to ~16% per cell. Larger cells (200 m) would halve sample count to ~7,500 — too few for spatial cross-validation.

### Temporal Setup

| Year | Labels | Sentinel-2 | Purpose |
|------|--------|-------------|---------|
| 2020 | WorldCover v1.0 | Spring/Summer/Autumn composites | Training |
| 2021 | WorldCover v2.0 | Spring/Summer/Autumn composites | Training + Change |
| 2022–2025 | None | Spring/Summer/Autumn composites | Forward prediction |

### Output Head

Multi-output regression with ILR (Isometric Log-Ratio) + softmax to enforce compositional constraints (all proportions sum to 1.0). The Dirichlet head was explored but did not outperform ILR+softmax.

---

## 2. Data Acquisition

### Sentinel-2 L2A

- **Source**: Microsoft Planetary Computer STAC API
- **Collection**: `sentinel-2-l2a` (atmospherically corrected)
- **Bands**: B02 (Blue), B03 (Green), B04 (Red), B05/B06/B07 (Red Edge), B08 (NIR), B8A (Narrow NIR), B11/B12 (SWIR)
- **Resolution**: 10 m native for B02–B04, B08; 20 m for B05–B07, B8A, B11, B12 (resampled to 10 m)
- **Compositing**: Seasonal medians with cloud masking via the Scene Classification Layer (SCL)
- **SCL exclusion**: Classes 0 (no data), 1 (saturated), 2 (dark area), 3 (cloud shadow), 8–10 (cloud/cirrus), 11 (snow)
- **Minimum scenes**: 8 per season (with fallback to cloud_max=50% and ±14 day window expansion)

### WorldCover Labels

- **Source**: ESA WorldCover 10 m (v1.0 for 2020, v2.0 for 2021)
- **Tile**: N48E009
- **Class mapping**: 11 ESA classes → 7 output classes (see [WorldCover Class Mapping](worldcover_class_mapping.md))
- **Known limitation**: v1.0 → v2.0 algorithm changes may create "fake change" signals. This is documented but **not corrected** — no alternative ground truth exists. Treated as label noise.

### BOA Offset Correction

Starting January 2022, Sentinel-2 Level-2A products include a BOA_ADD_OFFSET of −1000 in the metadata. All post-2022 imagery has this offset subtracted before compositing to maintain consistent reflectance values across years.

---

## 3. Label Engineering

Labels are generated per 100 m grid cell by:

1. Downloading ESA WorldCover GeoTIFF tiles
2. Reprojecting to the city's UTM anchor grid (nearest-neighbour)
3. Aggregating the 10 × 10 pixel patch per cell into class proportions
4. Normalizing each row to sum to 1.0

**Outputs**: `labels_2020.parquet`, `labels_2021.parquet`, `labels_change.parquet` (delta = 2021 − 2020)

---

## 4. Feature Engineering

### 4A. Core Features (143 columns per composite)

| Group | Features | Count |
|-------|----------|------:|
| **Band statistics** | mean, std, min, max, q25, median, q75, finite_frac × 10 bands | 80 |
| **Spectral indices** | NDVI, NDWI, NDBI, NDMI, NBR, SAVI, BSI, NDRE1, NDRE2 (mean, std, q25, median, q75) | 45 |
| **Tasseled Cap** | Brightness, Greenness, Wetness (mean, std) using 10-band Nedkov 2017 coefficients | 6 |
| **Spatial** | Sobel edge mean/std/max, Laplacian mean/std, Moran's I (NIR), NDVI range/IQR | 8 |
| **Control** | valid_fraction, low_valid_fraction, reflectance_scale, full_features_computed | 4 |

### Extended Features Added in Later Versions

| Group | Features | Count | Impact |
|-------|----------|------:|--------|
| **Novel indices** | EVI2, GNDVI, MNDWI, NDTI, IRECI, CRI1 (5 stats each) | 30 | +0.032 R² for LightGBM |
| **LBP** (Local Binary Patterns) | Uniform LBP histogram (10 bins) + entropy, per-patch on NIR | 11 | +0.018 R² for MLP |
| **Multi-band LBP** | NDVI, EVI2, SWIR1, NDTI LBP rasters (11 features each) | 44 | Marginal improvement |
| **GLCM** | Co-occurrence texture statistics (contrast, homogeneity, energy, correlation) | Variable | Helped MLPs, hurt trees |
| **Gabor** | Multi-scale/orientation filter responses | Variable | No net benefit |
| **Morphological** | Opening/closing profiles at multiple scales | Variable | No net benefit |
| **Semivariogram** | Spatial autocorrelation decay parameters | Variable | No net benefit |

### 4B. Fixed Bugs (Documented as Data Issues)

1. **Reflectance scaling mismatch**: Raw values 0–10000 vs 0–1 assumptions in texture computation
2. **Cell dropping**: Join corruption when filtering by quality threshold
3. **Cell_id alignment**: Row/column mapping mismatch risk
4. **Imputation scope**: Accidentally touching metadata columns

---

## 5. Dataset Merging & Delta Features

### Merge Strategy

All 6 composites (2 years × 3 seasons) are merged into a single wide table with suffixes `{feature}_{year}_{season}`.

### Delta Features

| Type | Formula | Count | Purpose |
|------|---------|------:|---------|
| **Year-over-Year** | 2021 − 2020 per season | 702 | Temporal change signals |
| **Seasonal contrasts** | Summer − Spring, Autumn − Summer per year | 1,404 | Phenological patterns |

### Final Dataset

- `features_merged_full.parquet`: **29,946 rows × 3,535 columns** (660 MB)
- Breakdown: 1,429 base + 702 YoY + 1,404 seasonal deltas
- 0% NaN after imputation

---

## 6. Exploratory Data Analysis

### Key Findings

- **Class imbalance**: Built-up (32.5%) and Tree Cover (30.9%) dominate; Bare/Sparse is only 1.0%
- **High feature redundancy**: Top-20 Spearman correlations show r > 0.95 among same-band statistics (mean ↔ median)
- **Seasonal drift**: B12 (SWIR2) shows the largest YoY drift, indicating it's most sensitive to atmospheric conditions
- **Quality coupling**: Very low VF cells (< all VF columns) show distinct feature distributions, validating quality-based filtering
- **Spatial autocorrelation**: Moran's I analysis across the full grid confirms strong spatial clustering of features

### Data Issues (Minimum 3 Required)

1. ✅ **Fixed**: Reflectance scale bug
2. ✅ **Fixed**: Cell dropping / join corruption risk
3. ✅ **Fixed**: Imputation scope accidentally touching metadata
4. ⚠️ **Not fixed**: WorldCover v1.0 → v2.0 label-version shift
   - **Justification**: Cannot correct without alternative ground truth. The algorithmic differences between v1.0 and v2.0 may introduce systematic biases in change detection, but we lack pixel-level ground truth to disentangle real change from label artifact. Discussed as a limitation.

---

## 7. Train/Test Split Design

### The Spatial Leakage Problem

Random train/test splits cause **spatial leakage**: nearby cells share similar spectral signatures, inflating test-set performance. We conducted a systematic comparison of 6 splitting strategies using Ridge regression as a diagnostic model.

### 6-Way Leakage Comparison

| Strategy | R² | Gap vs Random | Description |
|----------|:--:|:-------------:|-------------|
| Random | 0.767 | — | Baseline (catastrophically optimistic) |
| Grouped (scattered tiles) | 0.742 | +0.025 | 10×10 cell tiles, random assignment |
| Contiguous (row bands) | 0.691 | +0.076 | Horizontal strip folds |
| **Morton Z-curve** | **0.527** | **+0.240** | Bit-interleaved spatial curve |
| Region growing | 0.429 | +0.338 | Multi-start BFS expansion |
| Region growing + buffer | 0.386 | +0.381 | + 1-tile buffer exclusion zone |

### Chosen Strategy: Region Growing + 1-Tile Buffer

- **5-fold spatial CV** with contiguous, balanced folds
- **Tile-based blocking**: 10 × 10 cells per tile (1 km² blocks), 323 tile groups
- **Multi-start region growing**: 10 restarts with balance + contiguity scoring
- **Chebyshev buffer**: 1-tile exclusion zone between train and test
- **Fold metrics**: connected=YES, max size deviation=2.1%, compactness=1.871
- **14 unit tests** validating partition correctness, tile integrity, connectedness, balance, and determinism

### Design Rationale

Region growing + buffer was chosen despite being the most pessimistic estimator because:
1. It produces **contiguous, geographically meaningful** folds (unlike Morton Z-curve which fragments)
2. The buffer prevents **spatial autocorrelation leakage** at fold boundaries
3. **Balanced fold sizes** (max deviation 2.1%) ensure stable CV estimates
4. It best simulates real deployment: predicting for a **new geographic region** the model has never seen

---

## 8. Modeling

### Overview

| Model | R² (5-fold) | MAE (pp) | Features | Role |
|-------|:-----------:|:--------:|:--------:|------|
| **MLP** (champion) | **0.787 ± 0.038** | **2.50** | 864 (bi_LBP) | Final production model |
| **LightGBM** (best tree) | **0.736 ± 0.044** | **2.99** | 438 | Tree-based comparison |
| ExtraTrees | 0.692 | 3.13 | 2,109 | Explored, superseded |
| Random Forest | 0.684 | 3.23 | 2,109 | Explored, superseded |
| CatBoost | 0.671 | 4.00 | 2,109 | Too slow, dropped |
| **Ridge** | **0.423 ± 0.197** | **5.63** | 864 | Interpretable baseline |

### 8.1 Ridge Regression (Interpretable Baseline)

RidgeCV with spatially-aware cross-validation. R² = 0.423 ± 0.197 — performs poorly as expected: linear models cannot capture the complex nonlinear interactions between spectral bands, indices, and texture features that dominate land-cover prediction.

### 8.2 Feature Ablation with HGBR

Extensive feature ablation was conducted using sklearn's `HistGradientBoostingRegressor` as a fast proxy model (**565 total runs across 5 ablation studies**):

| Study | Runs | Key Finding |
|-------|-----:|-------------|
| Spectral bands vs indices | 95 | Combined > either alone |
| Texture features | 25 | LBP helps, GLCM marginal, HOG/morph hurt trees |
| Novel indices (NDTI, IRECI, CRI1, EVI2) | 315 | +0.032 R² over base |
| Fusion of groups | 80 | RedEdge + VegIdx + TC = 98.6% of full performance at half the features |
| Top-5 combinations | 50 | Optimal: 438 features (VegIdx+RedEdge+TC+NDTI+IRECI+CRI1) |

### 8.3 Tree-Based Models

**Sklearn Trees (28 configs, fold-0)**:
- ExtraTrees (500/1000), RF (500), CatBoost (1000+deeper)
- Trees plateau at R² ≈ 0.67–0.69 regardless of ensemble size or feature set
- CatBoost was **unacceptably slow**: 442s–2,328s vs ExtraTrees 21–239s for similar R²
- Adding texture features (GLCM+LBP) **hurt** tree performance — noise in the split search

**LightGBM Sweep (480 runs = 32 configs × 3 feature sets × 5 folds)**:
- Feature sets: VegIdx+RedEdge+TC (348f), +NDTI+CRI1 (408f), +IRECI (438f)
- Hyperparameters swept: n_estimators, max_depth, learning_rate, num_leaves, min_child_samples, reg_lambda, subsample, colsample_bytree
- **Best: `strong_wide`** (R² = 0.749, MAE = 2.94) — n_estimators=1000, max_depth=6, lr=0.03, num_leaves=255
- LightGBM closes the gap with MLPs significantly (0.749 vs 0.69 for sklearn trees)

**Key tree insight**: LightGBM cannot learn multiplicative feature interactions (axis-aligned splits only). Pre-computed spectral indices are essential. Curated 438-feature set outperforms giving it all 2,109 raw features.

### 8.4 MLP Architecture Search

**798 runs across 14 experiment versions (V5–V17)**:

#### Architectures Tested

| Dimension | Options | Winner | Finding |
|-----------|---------|--------|---------|
| **Type** | Plain vs Residual | Plain (R²=0.864) | Skip connections didn't help |
| **Activation** | SiLU, Mish, GELU, GeGLU, ReLU | SiLU | Most stable, best R² |
| **Depth** | L3, L5, L7, L10, L12, L16, L20 | L5 | Shallow-wide dominates |
| **Width** | d256, d512, d1024, d1536, d2048 | d1024 | Sweet spot |
| **Normalization** | BatchNorm, LayerNorm, None | BatchNorm | Always wins |
| **Output head** | ILR+softmax, Dirichlet | ILR+softmax | Standard approach |

#### Feature Sets Tested for MLP

| Feature Set | # Features | Best R² | Notes |
|-------------|:----------:|:-------:|-------|
| **bi_LBP** | **864** | **0.787** | **Champion — bands+indices+LBP** |
| bi_LBP_all5 | 1,128 | 0.768 | + multi-band LBP, diminishing returns |
| bands_indices | 798 | 0.769 | Spectral only, no texture |
| bi_Gab2_DMP | 1,566 | 0.758 | Complex Gabor + morph DMP |
| bi_LBP_Gab2_DMP | 1,632 | 0.745 | Kitchen sink — too many features |
| all_full | 3,535 | 0.795 | Everything (fold-0 only) |

#### Training Progression (V5–V17)

| Version | Focus | Runs | Key Finding |
|---------|-------|-----:|-------------|
| V4 | Architecture sweep (fold-0) | 1,549 | residual+GELU+BN best on fold-0 |
| V5 | Deep training (2000 ep) | 233 | glcm_lbp L5 d1024 BN = **0.787** |
| V5_arch | Architecture variants | 80 | Confirms plain > residual |
| V6–V7 | Architecture refinement | 100 | SiLU dominates, L5 optimal |
| V9 | Texture ablation | 82 | LBP+GLCM best texture combo |
| V10 | Definitive sweep | 75 | bi_LBP confirmed champion |
| V11–V12 | Gabor/morph/multi-LBP | 92 | Complex textures no benefit |
| V13 | Clean multi-band LBP | 110 | Could not beat V10/V5 |
| V14–V17 | Reproduction + seeds | 26 | Seed 42 = best, R²=0.787 confirmed |

#### Final V4 Production Architecture Sweep

A separate sweep tested **tapered (funnel) architectures** on a 14-city training set:

| Config | Shape | Params | R² | bare_sparse R² |
|--------|-------|-------:|:--:|:--------------:|
| **T_512_256_128** | **1464→512→256→128→6** | **917K** | **0.862** | **0.516** |
| T_768_384_192_96 | 1464→768→384→192→96→6 | 1.52M | 0.852 | 0.464 |
| T_1024_512 | 1464→1024→512→6 | 2.03M | 0.852 | 0.451 |
| 5L×2048w (old) | 1464→2048×5→6 | ~20M | 0.855 | 0.480 |

**Conclusion**: 917K parameters match 20M parameters. The information ceiling is in the **seasonal median composites**, not the model architecture. Breaking the ceiling requires new data sources (SAR, monthly composites).

### Key MLP Findings

1. **Texture features (LBP) help MLPs**: +0.018 R² (0.769 → 0.787)
2. **But too many texture types hurt**: Gabor, HOG, morph, semivariogram add noise
3. **MLP learns index-like relationships internally**: Permutation importance shows raw bands (B02, B03) are top features — the MLP computes its own NDVI/NDRE internally
4. **MLP–Tree gap: +0.038 R²** (0.787 vs 0.749) — MLPs learn cross-spectral interactions that trees cannot model

---

## 9. Evaluation Beyond Accuracy

### 9A. Per-Class Metrics

| Class | R² (MLP) | R² (Ridge) | MLP dominance |
|-------|:--------:|:----------:|:-------------:|
| Tree Cover | 0.963 | 0.922 | +0.041 |
| Grassland | 0.844 | 0.691 | +0.153 |
| Cropland | 0.916 | 0.589 | +0.327 |
| Built-up | 0.961 | 0.881 | +0.080 |
| Bare/Sparse | 0.406 | 0.053 | +0.353 |
| Water | 0.910 | 0.769 | +0.141 |

### 9B. Change Detection Metrics

| Metric | MLP | Ridge |
|--------|:---:|:-----:|
| False-change rate (τ=5%) | 24% | 87% |
| Missed-change rate (τ=5%) | 6.5% | 1.4% |
| Stability MAE (unchanged cells) | 0.83 pp | 4.19 pp |

Ridge over-predicts change everywhere because it cannot model stable areas.

### 9C. Stress Tests

**Gaussian noise injection**: Robust to σ ≤ 0.25 (R² drops 0.004), degrades at σ = 0.5 (−0.02), collapses at σ = 2.0 (R² = −0.06).

**Season dropout** (zeroing all features from one season):

| Season Dropped | R² | Delta |
|----------------|:--:|:-----:|
| 2020_spring | −9.09 | −9.92 (most critical!) |
| 2021_autumn | 0.35 | −0.48 (least critical) |

**Feature group ablation**:
- Drop Bands (480f): R² = −1.44 (catastrophic — bands are essential)
- Drop LBP (66f): R² = 0.47 (−0.36 — biggest impact per feature)
- Drop Indices (192f): R² = 0.61 (−0.23 — indices complement bands)

### 9D. Failure Analysis

- **Best predicted**: Tree Cover (MAE 1.57 pp) and Built-up (2.51 pp) — largest, most distinct classes
- **Most confused**: Cropland (4.66 pp) vs Grassland (4.89 pp) — spectrally similar
- **Worst predicted**: Bare/Sparse (13.81 pp) — rarest class (1%), least training signal

---

## 10. Results Summary

### Final Production Model

- **Architecture**: Plain MLP, L5 × d1024, BatchNorm, SiLU, ILR+softmax head
- **Features**: 864 (bi_LBP: bands + indices + LBP texture)
- **Performance**: R² = 0.787 ± 0.038 across 5 spatial CV folds
- **Training**: ~2000 epochs, AdamW (lr=1e-3, wd=3e-4), batch size 2048

### What Worked

1. **Spatial CV with buffer** — prevented misleading performance estimates
2. **LBP texture** — only texture type that consistently helped
3. **Pre-computed spectral indices** — essential for tree models, learned internally by MLP
4. **Shallow-wide architecture** — L5 × d1024 outperforms deeper or narrower variants
5. **Curated feature sets** — 438 features for trees, 864 for MLP (vs 3,535 available)

### What Didn't Work

1. **Complex texture features** (Gabor, HOG, morphological profiles) — added noise
2. **Deep architectures** (L12+) — no benefit over L5
3. **Residual connections** — plain feedforward was better
4. **Very large models** (20M params) — same R² as 917K params at 16× the cost
5. **CatBoost** — unacceptably slow for marginal improvement

### Known Limitations

1. **WorldCover label shift**: v1.0 → v2.0 algorithm changes create potential fake change signals
2. **Temporal extrapolation**: 2022–2025 predictions use a model trained only on 2020–2021 data
3. **Bare/Sparse prediction**: R² = 0.406 — rarest class remains challenging
4. **Single AOI**: Trained and validated on Nuremberg only (though V4 model uses 14 cities)
