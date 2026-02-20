# TerraPulse Explainability & Uncertainty

> This document details all explainability analyses, uncertainty quantification, and interpretability tests conducted on the TerraPulse land-cover prediction models.

---

## Table of Contents

1. [Overview](#1-overview)
2. [MLP Explainability](#2-mlp-explainability)
3. [LightGBM Explainability](#3-lightgbm-explainability)
4. [MLP vs LightGBM Comparison](#4-mlp-vs-lightgbm-comparison)
5. [Conformal Prediction Intervals](#5-conformal-prediction-intervals)
6. [Stress Tests](#6-stress-tests)
7. [Failure Analysis](#7-failure-analysis)
8. [Helpful vs Misleading Explanations](#8-helpful-vs-misleading-explanations)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. Overview

Explainability is a first-class concern in TerraPulse. Both the champion MLP and the LightGBM comparison model were analyzed with multiple complementary methods:

| Method | MLP | LightGBM | What it reveals |
|--------|:---:|:--------:|-----------------|
| Permutation importance | ✅ | ✅ | Which features are irreplaceable? |
| SHAP | GradientExplainer | TreeExplainer | Per-sample feature attribution |
| Feature ablation | ✅ | ✅ | Group-level importance |
| Conformal prediction | ✅ | ✅ | Prediction interval calibration |
| Stress tests | ✅ | — | Robustness under distribution shift |
| Failure analysis | ✅ | — | Where and why the model fails |

**Scripts**: `scripts/run_explainability_phase10.py` (MLP), `scripts/run_explainability_phase10_tree.py` (LightGBM)
**Outputs**: `reports/phase10/` (MLP), `reports/phase10_tree/` (LightGBM)

---

## 2. MLP Explainability

### 2.1 Permutation Importance

Permutation importance measures R² decrease when a single feature is randomly shuffled, breaking its relationship with the target.

**Top 10 Features (MLP)**:

| Rank | Feature | R² Decrease |
|:----:|---------|:-----------:|
| 1 | B02_q25_2021_autumn | 0.016 |
| 2 | B02_q25_2021_summer | 0.015 |
| 3 | B03_min_2021_spring | 0.014 |
| 4 | B02_q75_2021_autumn | 0.013 |
| 5 | B03_q25_2021_summer | 0.013 |
| 6 | B05_q25_2020_spring | 0.012 |
| 7 | B02_std_2021_autumn | 0.012 |
| 8 | B03_min_2021_autumn | 0.011 |
| 9 | B02_mean_2021_summer | 0.011 |
| 10 | B05_q25_2021_spring | 0.011 |

**Key observation**: 7 of the top 10 features are **raw spectral bands** (B02/B03 — blue/green). The MLP internally learns to compute vegetation index-like relationships from raw band statistics. It doesn't need pre-computed NDVI/NDRE to achieve high performance — it constructs these nonlinear interactions internally.

### 2.2 GradientSHAP

GradientSHAP (approximate Shapley values) was computed on 2,000 test samples with 500 background reference samples.

**Top 10 Features by Mean |SHAP| (MLP)**:

| Rank | Feature | Mean |SHAP| |
|:----:|---------|:------------:|
| 1 | NDRE2_q25_2020_spring | 0.0025 |
| 2 | TC_wet_std_2020_spring | 0.0024 |
| 3 | NDVI_q75_2020_autumn | 0.0024 |
| 4 | B05_q25_2021_spring | 0.0023 |
| 5 | NDRE1_median_2020_spring | 0.0022 |
| 6 | B03_min_2021_spring | 0.0022 |
| 7 | TC_bright_mean_2020_autumn | 0.0021 |
| 8 | NDVI_mean_2020_spring | 0.0021 |
| 9 | B02_q25_2021_autumn | 0.0020 |
| 10 | TC_wet_mean_2020_summer | 0.0020 |

**Key observation**: Unlike permutation importance, SHAP highlights **derived indices** (NDRE2, TC_wet, NDVI) as most impactful per individual prediction. SHAP values are more evenly distributed (top = 0.0025 vs 0.016 for permutation), reflecting the MLP's distributed internal representations.

**Interpretation**: Permutation importance answers "which features are irreplaceable?" (raw bands — can't be reconstructed). SHAP answers "which features drive individual predictions?" (derived indices — directly correlated with vegetation/water/urban signatures).

---

## 3. LightGBM Explainability

### 3.1 Native Gain Importance

Total reduction in loss from splits using each feature. This is LightGBM's built-in importance metric.

**Top 5 Features by Gain**:

| Rank | Feature | % Total Gain |
|:----:|---------|:------------:|
| 1 | CRI1_q75_2021_autumn | 24.3% |
| 2 | NDRE2_q25_2021_summer | 16.1% |
| 3 | CRI1_q75_2020_autumn | 10.2% |
| 4 | TC_wet_mean_2020_autumn | 5.8% |
| 5 | NDTI_q25_2020_spring | 4.2% |

**Striking finding**: Just 3 features account for **50.5% of all gain**. The tree is highly concentrated on a few key features, unlike the MLP's distributed importance.

**CRI1 (Chlorophyll Red-edge Index)** dominates because it effectively separates vegetation types in the red-edge spectral region, which is where the most discriminative information lies for land-cover classification.

### 3.2 Permutation Importance (LightGBM)

| Rank | Feature | R² Decrease |
|:----:|---------|:-----------:|
| 1 | TC_wet_mean_2020_autumn | 0.046 |
| 2 | NDTI_q25_2020_spring | 0.033 |
| 3 | TC_bright_mean_2021_spring | 0.032 |
| 4 | NDRE2_q25_2021_summer | 0.028 |
| 5 | CRI1_q75_2021_autumn | 0.025 |

**Key discord**: CRI1 has **huge gain** importance but only **moderate** permutation importance. This means other features can partially compensate when CRI1 is shuffled. TC_wet has moderate gain but **highest** permutation importance — it is truly irreplaceable.

### 3.3 TreeSHAP (Exact Shapley Values)

TreeSHAP provides **exact** (not approximate) Shapley values for tree ensembles.

**Top features by Mean |SHAP|**: NDRE2_q25 (0.027), CRI1_q75 (0.022), CRI1_q75_2020 (0.011), TC_wet (0.010)

Beeswarm plots show clear nonlinear patterns for tree_cover (NDVI-driven) and bare_sparse (CRI1-driven).

---

## 4. MLP vs LightGBM Comparison

| Aspect | MLP (864 features) | LightGBM (438 features) |
|--------|:------------------:|:-----------------------:|
| **Permutation top** | Raw bands (B02, B03, B05) | Derived indices (TC_wet, NDTI, CRI1) |
| **SHAP top** | NDRE2, TC_wet, NDVI | NDRE2, CRI1, TC_wet |
| **Concentration** | Even (top = 1.6% R² drop) | Concentrated (top = 4.6% R² drop) |
| **Collinearity issue** | Yes (B02 mean/median) | Avoided (curated features) |
| **SHAP method** | GradientExplainer (approximate) | TreeExplainer (exact) |

### Core Insight

The MLP learns to **compute index-like relationships** from raw bands internally. LightGBM cannot do this — its axis-aligned splits cannot model multiplicative interactions (band ratios). This is why:
- **MLP works well with raw bands** — it constructs its own indices
- **LightGBM needs pre-computed indices** — especially CRI1, NDRE2, TC_wet
- The MLP–Tree gap (+0.038 R²) comes from these learned cross-spectral interactions

---

## 5. Conformal Prediction Intervals

All models use **split-conformal calibration** at 90% nominal coverage on fold-0 holdout data. For each class, the nonconformity score is the absolute residual, and the prediction interval is `[ŷ − q, ŷ + q]` where `q` is the 90th percentile of calibration residuals.

### Results

| Model | Coverage Range | Interval Width Range | Assessment |
|-------|:--------------:|:--------------------:|:----------:|
| Ridge | 71–83% | 0.002–29 pp | Wide, unreliable |
| ElasticNet | 71–84% | 0.005–30 pp | Wide, unreliable |
| MLP | 71–80% | 0.03–14 pp | Tightest, best calibrated |
| CatBoost | 68–80% | 0.0003–21 pp | Moderate |
| Random Forest | 69–80% | 0.0003–16 pp | Moderate |
| ExtraTrees | 65–75% | 0.0003–13 pp | Worst coverage |

### Critical Caveat

> **⚠️ All models fall below the 90% nominal coverage.** Conformal intervals are too narrow, likely due to distribution shift between calibration and test splits (spatial CV creates geographic separation). This means the prediction intervals are **anti-conservative** — they claim more certainty than warranted. Conservative interpretation is recommended.

### Dashboard Integration

Conformal prediction data is served via the API (`/api/conformal`) and displayed in the Evaluation panel showing per-model, per-class coverage and interval widths.

---

## 6. Stress Tests

Three types of stress tests assess model robustness under simulated deployment conditions.

### 6.1 Gaussian Noise Injection

Additive Gaussian noise N(0, σ²) applied to all features:

| σ | R² | Δ R² | Assessment |
|:-:|:--:|:----:|:----------:|
| 0.00 | 0.833 | — | Baseline |
| 0.10 | 0.831 | −0.002 | Essentially unchanged |
| 0.25 | 0.829 | −0.004 | Still robust |
| 0.50 | 0.813 | −0.020 | Slight degradation |
| 1.00 | 0.704 | −0.129 | Significant drop |
| 2.00 | −0.060 | −0.893 | Collapse |

**Practical significance**: Sentinel-2 measurement noise is typically σ < 0.1 in normalized reflectance. The model is robust well beyond expected sensor noise.

### 6.2 Season Dropout

All features from one season zeroed (simulating missing satellite revisits):

| Season Dropped | R² | Δ R² | Interpretation |
|----------------|:--:|:----:|----------------|
| 2020_spring | −9.09 | −9.92 | **Most critical** — spring vegetation greening is essential |
| 2021_summer | −1.81 | −2.64 | Summer peak vegetation highly informative |
| 2020_autumn | −0.65 | −1.49 | Senescence contributes moderately |
| 2020_summer | −0.56 | −1.40 | Partially redundant with 2021_summer |
| 2021_spring | 0.00 | −0.83 | Partially redundant with 2020_spring |
| 2021_autumn | 0.35 | −0.48 | **Least critical** — most redundant season |

**Key finding**: Spring 2020 is **catastrophically important** — losing it destroys the model. This makes sense: spring greening is the strongest discriminator between vegetation types (deciduous forests leaf-out while cropland is bare).

### 6.3 Feature Group Ablation

Entire feature groups zeroed:

| Group Dropped | Features | R² | Δ R² | Assessment |
|---------------|:--------:|:--:|:----:|:----------:|
| Bands | 480 | −1.44 | −2.27 | **Catastrophic** — bands are essential |
| LBP | 66 | 0.47 | −0.36 | Biggest impact per feature |
| Indices | 192 | 0.61 | −0.23 | Indices complement bands |
| Tasseled Cap | 36 | 0.78 | −0.05 | Minor contribution |
| Spatial | 48 | 0.81 | −0.02 | Marginal contribution |

**LBP has the highest impact-per-feature ratio**: 66 features contribute −0.36 R² → each LBP feature is ~0.0055 R²/feature. Compare to bands: 480 features for −2.27 R² → 0.0047 R²/feature.

---

## 7. Failure Analysis

### Error by Dominant Class

| Dominant Class | MAE (pp) | Interpretation |
|---------------|:--------:|----------------|
| Tree Cover | 1.57 | Lowest error — largest, most spectrally distinct |
| Built-up | 2.51 | Well predicted despite heterogeneity |
| Cropland | 4.66 | Confused with grassland |
| Grassland | 4.89 | Confused with cropland |
| Water | 5.12 | High when mixed with other classes |
| Bare/Sparse | 13.81 | **Worst** — rarest class (1%), least signal |

### Cropland–Grassland Confusion

The largest systematic error is between cropland and grassland. Both classes:
- Share green vegetation spectral signatures
- Differ mainly in temporal phenology (crop rotation vs perennial grass)
- Are best distinguished by spring/summer transitions (hence spring dropout being catastrophic)

### Fold Variation

R² ranges from 0.753 to 0.844 across 5 spatial folds, with MAE from 1.18 to 3.61 pp. This variation reflects genuine geographic heterogeneity in Nuremberg — some folds contain more urban/industrial areas (easier to predict) while others contain mixed suburban/agricultural transitions (harder).

---

## 8. Helpful vs Misleading Explanations

### 8.1 Helpful Explanation

**NDVI_mean → tree_cover prediction**

For a high-tree sample (63% tree cover), NDVI features have consistent SHAP direction: high NDVI pushes tree_cover prediction upward. This confirms the model learned the real physical relationship between the Normalized Difference Vegetation Index and canopy density. This relationship is:
- **Physically grounded** in leaf chlorophyll absorption
- **Consistent** across seasons and years
- **Directionally correct** in both MLP and LightGBM

### 8.2 Misleading Explanation

**B02_mean vs B02_median — SHAP credit splitting**

B02_mean and B02_median have near-perfect correlation (r > 0.97 in all seasons) but SHAP assigns wildly different importance to them (ratio ranges 0.3x–1.2x across classes and seasons). A naive interpretation would conclude that one statistic "matters" and the other "doesn't."

**Why this is misleading**: This is the classical **SHAP credit-splitting artifact** among collinear features. When two features carry the same information, Shapley values split credit roughly equally but unstably — small perturbations can flip which feature gets more credit. The correct interpretation is that **the information carried by blue band reflectance matters**, not that mean or median specifically matters.

**Note**: This issue is **avoided by design** in LightGBM's curated 438-feature set, which excludes redundant band statistics.

---

## 9. Key Takeaways

1. **Complementary explanation methods are essential**: Permutation importance, SHAP, and gain importance tell different stories that together give a complete picture.

2. **MLPs learn implicit feature engineering**: Raw bands are most important by permutation, but derived indices drive predictions — the MLP computes index-like relationships internally.

3. **Conformal intervals are anti-conservative**: Spatial CV creates distribution shift that makes calibrated intervals too narrow. Real deployment uncertainty is higher than reported.

4. **Spring is the most critical season**: Season dropout shows 2020_spring is catastrophically important. Satellite monitoring systems must ensure spring acquisitions.

5. **LBP texture features are efficient**: Highest impact-per-feature ratio of any group.

6. **Bare/sparse is the hardest class**: 1% prevalence, 13.81 pp MAE. More labeled examples of rare classes would improve this.
