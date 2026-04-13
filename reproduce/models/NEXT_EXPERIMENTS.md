# Next Experiments: Loss Functions & Ensembles

## Current State

Best single model: **SSNet V2 = 86.32%**, SSNet V3 = 86.39% test.  
Best ensemble: CB+MLP 70/30 = 85.63% (doesn't include SSNet).  
V4 iterative resampling currently training.

### Per-Class Accuracy Across Models (test %)

| Class | SSNet V3 | CatBoost | MLP 3×3 | TempCNN 3×3 | TempCNN 1×1 | Pixels |
|-------|----------|----------|---------|-------------|-------------|--------|
| tree_cover | **89.3** | 86.9 | 80.9 | 77.0 | 77.6 | 896K |
| shrubland | 19.3 | 5.5 | 77.7 | 83.7 | **87.7** | 22K |
| grassland | **76.1** | 76.0 | 64.7 | 66.5 | 60.2 | 539K |
| cropland | **81.6** | 78.6 | 80.7 | 80.7 | 79.5 | 275K |
| built_up | 90.0 | 90.1 | 92.6 | **94.0** | 92.8 | 851K |
| bare_sparse | 36.6 | 28.8 | 59.1 | **60.6** | 60.0 | 41K |
| water | **99.0** | 98.9 | 98.9 | 99.2 | 99.1 | 375K |
| **Overall** | **86.4** | 85.2 | 83.2 | 82.9 | 81.1 | 3.0M |

---

## Experiment A: Loss Function Variants on SSNet

**Problem:** Unweighted CE gives SSNet 89.3% tree but 19.3% shrubland.
Inverse-freq weights crashed majority classes. Need a middle ground.

### A1. Sqrt-dampened class weights

```python
counts = np.bincount(train_y, minlength=N_CLASSES)
weights = 1.0 / np.sqrt(counts / counts.sum())
weights /= weights.sum() / N_CLASSES  # normalize to mean=1
```

Gentler than `1/freq` — boosts rare classes moderately without destroying majority.
Expected: shrubland 40-50%, tree stays >85%.

### A2. Focal loss (γ=2)

```python
# Standard focal loss: -α_t * (1-p_t)^γ * log(p_t)
# Down-weights easy (high-confidence) examples, focuses on hard ones
# No need for class weights — difficulty-aware by design
```

Self-balancing: the model's own confidence determines weighting.
Well-proven for class imbalance (Lin et al., RetinaNet).

### A3. Label-smoothed CE

```python
loss = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Prevents overconfident predictions, acts as regularization.
Won't fix imbalance directly but may improve calibration.

### Plan

1. Take V4 best checkpoint (or V2 checkpoint if V4 isn't done)
2. Train 3 variants: sqrt-weights, focal, label-smoothing
3. Each variant: 10 epochs, patience=5, same data (round 0 seed)
4. Compare per-class accuracy — especially shrubland/bare vs tree/grass

---

## Experiment B: Multi-Model Ensemble

**Key insight:** Every model excels at different classes. We have 5 trained
models with saved checkpoints + scalers. No training needed — just inference
and weight optimization.

### Available checkpoints

| Model | Checkpoint | Scaler | Best at |
|-------|-----------|--------|---------|
| SSNet V2/V3/V4 | `ssnet.pt` | `ssnet_scaler.pkl` | tree, grass, crop, water |
| CatBoost V5 | (in pixel pipeline) | — | grass, tree |
| MLP 3×3 | `mlp_3x3.pt` | `mlp_3x3_scaler.pkl` | shrub, bare |
| TempCNN 3×3 | `tempcnn_3x3.pt` | `tempcnn_3x3_scaler.pkl` | built_up, shrub |
| TempCNN 1×1 | `tempcnn_1x1.pt` | `tempcnn_1x1_scaler.pkl` | shrubland (87.7%) |

### B1. Uniform probability blend (3-way)

SSNet + CatBoost + TempCNN 3×3 probability blend.
Sweep weights on val set, pick best.

### B2. Per-class best-model selection

For each pixel, use the model that performs best on that class (oracle).
This gives an upper bound on what a per-class ensemble could achieve:

| Class | Best model | Expected |
|-------|-----------|----------|
| tree_cover | SSNet | 89.3% |
| shrubland | TempCNN 1×1 | 87.7% |
| grassland | SSNet | 76.1% |
| cropland | SSNet | 81.6% |
| built_up | TempCNN 3×3 | 94.0% |
| bare_sparse | TempCNN 3×3 | 60.6% |
| water | SSNet | 99.0% |

But we can't know the true class at test time — we'd need a learned stacker.

### B3. Learned stacker (logistic regression)

1. Run all 5 models on val set → 5 probability vectors per pixel
2. Train a logistic regression (or small MLP) to combine them
3. Evaluate on test set

This can learn "trust SSNet for tree, trust TempCNN for shrubland" automatically.

### Plan

1. Write `06_ensemble_ssnet.py`
2. Load all 5 model checkpoints
3. Generate probability predictions on aligned val + test pixels
4. Optimize: sweep weights (B1), per-class argmax (B2), learned stacker (B3)
5. Report best ensemble

---

## Priority Order

1. **V4 iterative resampling** — currently running, wait for results
2. **B1/B2 ensemble** — quick win, no training, just inference + optimization
3. **A1 sqrt weights on V4** — quick experiment, 1 training run
4. **A2 focal loss on V4** — if sqrt doesn't work
5. **B3 learned stacker** — if uniform blend is promising
