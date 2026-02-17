# MLP Architecture Experiments – V4 Production Pipeline
*Date: 2026-02-17 | Dataset: 14-city train, Nuremberg val | 1464 features*

## Summary

**Best model: `T_512_256_128` (Tapered MLP, funnel architecture)**
- Shape: `1464 → 512 → 256 → 128 → 6`
- Parameters: 916,870 (22× smaller than the original 20M-param model)
- **R² = 0.8619 | MAE = 2.43pp**
- bare_sparse R² = 0.516 (best ever, +0.036 over previous best)
- Saved to: `models/best_mlp/`

---

## Experiment 1: Regularization Tuning (5L × 2048w)

Starting point: the V4 pipeline's default 5L × 2048w MLP (~20M params).

| Config | Best Epoch | Best Val Loss | Pattern |
|--------|-----------|---------------|---------|
| dropout=0.15, wd=1e-4 (old) | 4-7 | 0.4337 | Peaked immediately |
| dropout=0.30, wd=1e-3 (new) | 15 | 0.4299 | Improved longer |

**Finding:** Stronger regularization extended useful training from 4 → 15 epochs,
but the model still converges quickly — the bottleneck is data, not capacity.

---

## Experiment 2: Small vs Large MLP

Comparing the 20M-param model against a much smaller 3L × 512w model.

| Model | Params | R² | MAE | bare_sparse | Train time |
|-------|-------:|-----:|------:|------:|----------:|
| 3L × 512w (small) | 1.26M | 0.8549 | 2.48pp | 0.486 | 217s |
| 5L × 2048w (big) | ~20M | 0.8554 | 2.46pp | 0.480 | 564s |

**Finding:** The small model matches the big one almost exactly with 16× fewer
parameters, proving that model capacity is NOT the bottleneck.

---

## Experiment 3: Weight Analysis (3L × 512w)

Analyzed the trained weights to understand capacity utilization.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Dead weights (<0.001) | 2.9% | Model is well-utilized |
| Near-zero (<0.01) | 29.1% | ~30% of weights contribute very little |
| Kurtosis | 336.5 | Extremely heavy-tailed distribution |
| Feature importance CV | 0.18 | All 1464 features used fairly uniformly |
| Features with <1% of max importance | 0 | Every feature contributes |

**Finding:** The model uses all features uniformly — no wasted features, no
wasted capacity. The ceiling is in the data, not the model.

---

## Experiment 4: Constant-Width Architecture Sweep

Tested depth vs width tradeoffs with all 1464 features (including 36 previously
excluded spatial features: edge, laplacian, Moran's I).

| Config | Params | R² | MAE | bare_sparse | Time |
|--------|-------:|-----:|------:|------:|-----:|
| 2L × 1024w (short+wide) | 2.56M | 0.8510 | 2.52pp | 0.445 | 53s |
| 3L × 512w (baseline) | 1.28M | 0.8503 | 2.38pp | 0.434 | 76s |
| **5L × 256w (narrow+deep)** | **642K** | **0.8544** | **2.38pp** | **0.463** | 88s |
| 7L × 128w (very deep) | 289K | 0.8436 | 2.46pp | 0.412 | 104s |

**Findings:**
- Depth > width for this problem (5L×256 beats 2L×1024)
- 7L×128 is too narrow, especially for bare_sparse
- 5L×256w matches the 20M-param model with 30× fewer params
- 36 newly included features (edge, lap, Moran's I) contribute to the total

---

## Experiment 5: Tapered (Funnel) Architecture Sweep

Tested wide→narrow funnel architectures where the first layer compresses
1464 features and subsequent layers progressively narrow.

| Config | Shape | Params | R² | bare_sparse | Time |
|--------|-------|-------:|-----:|------:|------:|
| **T_512_256_128** | **1464→512→256→128→6** | **917K** | **0.8619** | **0.516** | 71s |
| T_768_384_192_96 | 1464→768→384→192→96→6 | 1.52M | 0.8520 | 0.464 | 75s |
| T_1024_512 | 1464→1024→512→6 | 2.03M | 0.8520 | 0.451 | 74s |
| T_1024_256 | 1464→1024→256→6 | 1.77M | 0.8505 | 0.440 | 62s |
| T_1024_512_256_128 | 1464→1024→512→256→128→6 | 2.19M | 0.8499 | 0.431 | 69s |
| T_1024_512_256 | 1464→1024→512→256→6 | 2.16M | 0.8493 | 0.436 | 55s |
| T_768_384_192 | 1464→768→384→192→6 | 1.50M | 0.8354 | 0.359 | 64s |
| T_512_256_128_64 | 1464→512→256→128→64→6 | 925K | 0.8067 | 0.179 | 110s |

**Findings:**
- **Funnel 512→256→128 is the clear winner** — best R² and best bare_sparse
- Starting wider (1024→) wastes parameters without improving accuracy
- 128 is the minimum useful layer width — going to 64 destroys bare_sparse
- The funnel shape naturally matches the problem: compress 1464 features →
  progressively abstract → predict 6 classes

---

## Per-Class Comparison: Best Models

| Class | T_512_256_128 | 5L×2048w (old) | Δ |
|-------|:---:|:---:|:---:|
| tree_cover | 0.9668 | 0.9656 | +0.001 |
| built_up | 0.9657 | 0.9631 | +0.003 |
| cropland | 0.9281 | 0.9302 | -0.002 |
| water | 0.9331 | 0.9307 | +0.002 |
| grassland | 0.8613 | 0.8628 | -0.002 |
| **bare_sparse** | **0.5161** | 0.4799 | **+0.036** |

---

## Unlocked Features

36 features were previously excluded by `build_bi_lbp()` prefix filtering:
- `edge_mean/std/max` (18 features) — Sobel edge magnitude statistics
- `lap_abs_mean/std` (12 features) — Laplacian statistics
- `morans_I_NIR` (6 features) — Moran's I spatial autocorrelation of NIR

These are now included in the best model (1464 total vs 1428 before).

---

## Training Efficiency

Reduced early stopping patience from 30,000 steps (~159 epochs) to 5,000 steps
(~27 epochs). This cut training time from 10+ minutes per config to ~1 minute,
enabling the 12-config sweep in under 15 minutes total.

All models converge by epoch 15-30, confirming that short patience is sufficient.

---

## Best Model Details

```
Architecture: TaperedMLP (funnel)
Shape:        1464 → 512 → 256 → 128 → 6
Parameters:   916,870
Activation:   SiLU
Norm:         BatchNorm
Dropout:      0.15
Input dropout: 0.05
Weight decay: 3e-4
Learning rate: 1e-3
Batch size:   2048
Seed offset:  100
Patience:     5000 steps (~27 epochs)
Best epoch:   23
Features:     1464 (all, including edge/laplacian/Moran's I)
Train cities: 14 (bremen, hamburg, duesseldorf, leipzig, rostock,
              amsterdam, hambach_mine, welzow_mine, amiens,
              magdeburg, ulm, salzburg, schwerin, malmo)
Val city:     nuremberg
```

## Conclusion

We have reached the **information ceiling of seasonal median composites**.
Evidence:
1. 917K params matches 20M params → capacity is not limiting
2. All 1464 features used uniformly → no unused feature capacity
3. Multiple architectures converge to R² ≈ 0.85-0.86 → same optimum
4. Training always converges in 15-30 epochs → signal learned fast

**Breaking the ceiling requires new data sources** (e.g., Sentinel-1 SAR,
monthly composites) rather than further architecture tuning.
