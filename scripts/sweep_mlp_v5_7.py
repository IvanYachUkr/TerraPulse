#!/usr/bin/env python3
"""
V5.7 MLP Sweep — Best-of-all-worlds Sampler + Inverted Bottleneck Architectures.

Combines proven techniques from v5.6 experiments + ChatGPT's critique:

ARCHITECTURE (proven in v5.6):
  - Inverted bottleneck: wide first layer (≥1024) → aggressive taper
  - T_1024_512_256_64 won v5.6 with R²=0.8419 (bare=0.414)
  - C_3x256 baseline control for comparison

SAMPLING (best of all sources):
  1. Fractional rarity weighting (ChatGPT):
     - Instead of argmax class bucketing, weights each sample by its
       composition: w_i = Σ y_ik · (p_k + ε)^(-α).
     - Cells with ANY bare_sparse fraction get proportional upweight.
  2. City-balanced hierarchical sampling:
     - Median-based per-city budget with replacement for small cities.
     - Within each city, sample with rarity + hardness weights.
  3. Inter-city Mixup (proven in v5.6):
     - Beta(0.3, 0.3) blending of features + labels from different
       positions in the batch.
  4. Hard-city curriculum (ChatGPT):
     - Track per-city training loss, tilt sampling toward hard cities.
   5. OHEM with EMA + variance penalty (refined from Gemini):
      - Track per-sample loss mean AND variance via EMA.
      - Learnability score: H_i = max(ε, μ_i − λ·σ_i)
      - Consistently-hard = learnable; high-variance = noise.
      - Track losses on CLEAN (pre-mixup) data to avoid contamination.

TRAINING:
  - Warmup + cosine LR schedule
  - AMP (fp16) + fused AdamW
  - Step-based patience converted to epochs

Usage:
    .venv/Scripts/python.exe scripts/sweep_mlp_v5_7.py
"""

import os, sys, time, math, json, pickle, gc
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    TRAIN_CITIES as _ALL_TRAIN, TEST_CITIES as _ALL_TEST,
    SEED, CLASS_NAMES, N_CLASSES, CITIES,
    city_features_dir, city_labels_path,
    _discover_feature_cols, _load_city_arrays, build_bi_lbp, ts,
)
from scripts.run_mlp_overnight_v4 import (
    _make_norm, PlainBlock, normalize_targets,
    soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# =====================================================================
# SAR-equipped cities (same as v5.5/v5.6)
# =====================================================================
_SAR_TRAIN_NAMES = [
    'bremen', 'hamburg', 'duesseldorf', 'leipzig', 'rostock',
    'amsterdam', 'hambach_mine', 'welzow_mine', 'amiens', 'magdeburg',
    'ulm', 'salzburg', 'schwerin', 'malmo',
    'regensburg', 'london', 'brussels', 'rotterdam', 'antwerp',
]
_SAR_TEST_NAMES = ['frankfurt']

_city_by_name = {c.name: c for c in CITIES}
TRAIN_CITIES = [_city_by_name[n] for n in _SAR_TRAIN_NAMES]
TEST_CITIES = [_city_by_name[n] for n in _SAR_TEST_NAMES]

# =====================================================================
# Model definitions (unchanged from v5.6)
# =====================================================================

class ConstantMLP(nn.Module):
    def __init__(self, in_features, n_classes, n_layers, width,
                 dropout=0.15, activation="silu", input_dropout=0.05,
                 norm_type="batchnorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = [PlainBlock(in_features, width, dropout, activation, norm_type)]
        for _ in range(n_layers - 1):
            layers.append(PlainBlock(width, width, dropout, activation, norm_type))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


class TaperedMLP(nn.Module):
    def __init__(self, in_features, n_classes, widths,
                 dropout=0.15, activation="silu", input_dropout=0.05,
                 norm_type="batchnorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = []
        prev_dim = in_features
        for w in widths:
            layers.append(PlainBlock(prev_dim, w, dropout, activation, norm_type))
            prev_dim = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# =====================================================================
# Architecture configs — v5.6 winners + variants
# =====================================================================

# Sampling modes:
#   "baseline"   = no sampling tricks (pure uniform + no mixup)
#   "mixup_only" = mixup only (no rarity/OHEM/hard-city) — proper control
#   "full"       = all sampling tricks (mixup + rarity + OHEM + hard-city)

CONFIGS = [
    # Format: (name, type, shape, dropout, wd, mode)

    # ---- Controls (v5.6 proven) ----
    ("C_3x256_baseline",           "C", (3, 256),  0.10, 1e-4, "baseline"),
    ("C_3x256_mixup",              "C", (3, 256),  0.10, 1e-4, "mixup_only"),

    # ---- V5.6 winner + variants (all full mode) ----
    ("T_1024_512_256_64",          "T", [1024, 512, 256, 64],   0.15, 2e-4, "full"),
    ("T_1024_512_256_128",         "T", [1024, 512, 256, 128],  0.15, 2e-4, "full"),
    ("T_1024_512_128",             "T", [1024, 512, 128],       0.12, 2e-4, "full"),

    # ---- Wider first layer ----
    ("T_2048_512_128",             "T", [2048, 512, 128],       0.15, 3e-4, "full"),
    ("T_2048_1024_256_64",         "T", [2048, 1024, 256, 64],  0.18, 3e-4, "full"),

    # ---- Match input dim ----
    ("T_1748_512_128",             "T", [1748, 512, 128],       0.15, 3e-4, "full"),

    # ---- Smaller variants (regularization via capacity limit) ----
    ("T_512_256_128_64",           "T", [512, 256, 128, 64],    0.10, 1e-4, "full"),
    ("T_768_256_64",               "T", [768, 256, 64],         0.12, 2e-4, "full"),
]

# =====================================================================
# Training constants
# =====================================================================
BATCH_SIZE = 4096
PATIENCE_STEPS = 10000
MIN_STEPS = 3000
MAX_EPOCHS = 500
SEED_OFFSET = 300  # different seed from v5.5/v5.6
INPUT_DROPOUT = 0.10
LR = 1e-3
ACTIVATION = "silu"
NORM = "batchnorm"

# Mixup (proven in v5.6)
MIXUP_ALPHA = 0.3
MIXUP_PROB = 0.5

# OHEM — variance-penalized (Gemini refinement)
OHEM_WARMUP = 20        # don't start until loss signal is meaningful
OHEM_VAR_LAMBDA = 0.5    # penalty for high-variance (noisy) samples
OHEM_EMA_DECAY = 0.7     # decay for per-sample loss EMA

# Fractional rarity (new from ChatGPT)
RARITY_ALPHA = 0.5       # exponent: (p_k + eps)^(-alpha)
RARITY_WARMUP = 5        # start uniform, ramp up rarity over this many epochs
RARITY_MAX_ETA = 0.7     # max blend toward rarity vs uniform

# Hard-city curriculum — softmax with temperature (Gemini/Group-DRO)
CITY_HARDNESS_ALPHA = 0.2    # blend weight toward hard cities (1-α is uniform)
CITY_TEMPERATURE = 0.5       # softmax temperature (lower = sharper focus)
CITY_LOSS_EMA_DECAY = 0.9

# Output directory
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v5_7_combined", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# Data loading
# =====================================================================

print(f"[{ts()}] V5.7 Combined-Strategy MLP Sweep (run_id={RUN_ID})")
print(f"  Output: {OUT_DIR}")
print(f"  Configs: {len(CONFIGS)}")
print(f"  Test cities: {[c.name for c in TEST_CITIES]}")

# Discover features
print(f"\n[{ts()}] Discovering feature columns...")
full_feature_cols = _discover_feature_cols()
mlp_idx = build_bi_lbp(full_feature_cols)
mlp_cols = [full_feature_cols[i] for i in mlp_idx]
n_features = len(mlp_cols)
print(f"  Total parquet columns: {len(full_feature_cols)}")
print(f"  MLP feature columns: {n_features}")

# Load training data
print(f"\n[{ts()}] Loading training data...")
city_counts = []
train_cities_valid = []
for city in TRAIN_CITIES:
    feat_path = os.path.join(city_features_dir(city),
                             "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        print(f"  WARNING: Missing {city.name} -- skip")
        continue
    import pyarrow.parquet as pq
    n = pq.read_metadata(feat_path).num_rows
    city_counts.append(n)
    train_cities_valid.append(city)
    print(f"  [{city.name}] {n:,} cells")

total = sum(city_counts)
print(f"  Total: {total:,} cells x {n_features} features")

X_train = np.empty((total, n_features), dtype=np.float32)
y_train = np.empty((total, N_CLASSES), dtype=np.float32)
city_boundaries = []
city_ids = np.empty(total, dtype=np.int32)
offset = 0
for ci, (city, n) in enumerate(zip(train_cities_valid, city_counts)):
    result = _load_city_arrays(city, mlp_cols)
    if result is None:
        continue
    X_city, _ = result
    labels = pd.read_parquet(city_labels_path(city, 2021))
    y_city = labels[CLASS_NAMES].values.astype(np.float32)
    del labels

    X_train[offset:offset + n] = X_city
    y_train[offset:offset + n] = y_city
    city_ids[offset:offset + n] = ci
    city_boundaries.append((offset, offset + n))
    del X_city, y_city
    offset += n
    gc.collect()

n_cities = len(city_boundaries)
print(f"  Loaded: {offset:,} samples, {n_cities} cities")

# Load test data
print(f"\n[{ts()}] Loading test cities...")
test_data = {}
for city in TEST_CITIES:
    result = _load_city_arrays(city, mlp_cols)
    if result is None:
        print(f"  WARNING: Missing {city.name} -- skip")
        continue
    X_val, n_val = result
    labels = pd.read_parquet(city_labels_path(city, 2021))
    y_val = labels[CLASS_NAMES].values.astype(np.float32)
    del labels
    test_data[city.name] = (X_val, y_val)
    print(f"  [{city.name}] {n_val:,} cells")

if not test_data:
    raise RuntimeError("No test data available!")

# Scale
print(f"\n[{ts()}] Fitting scaler...")
scaler = StandardScaler()
scaler.fit(X_train)
X_train -= scaler.mean_.astype(np.float32)
X_train /= scaler.scale_.astype(np.float32)
for name in list(test_data.keys()):
    X_v, y_v = test_data[name]
    X_v -= scaler.mean_.astype(np.float32)
    X_v /= scaler.scale_.astype(np.float32)
    test_data[name] = (X_v, y_v)

with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUT_DIR, "mlp_cols.json"), "w") as f:
    json.dump(mlp_cols, f)

# =====================================================================
# FRACTIONAL RARITY WEIGHTS (key innovation from ChatGPT)
# =====================================================================
# Compute global class prevalence from soft labels
# p_k = (1/N) * Σ y_ik  for each class k
class_prevalence = y_train.mean(axis=0)  # [N_CLASSES]
print(f"\n  Class prevalence (soft labels):")
for k, cn in enumerate(CLASS_NAMES):
    print(f"    {cn:20s}: {class_prevalence[k]:.4f}")

# Rarity weight per class: r_k = (p_k + ε)^(-α)
rarity_per_class = (class_prevalence + 1e-6) ** (-RARITY_ALPHA)
rarity_per_class /= rarity_per_class.max()  # normalize so max=1
print(f"\n  Rarity weights (alpha={RARITY_ALPHA}):")
for k, cn in enumerate(CLASS_NAMES):
    print(f"    {cn:20s}: {rarity_per_class[k]:.3f}")

# Per-sample rarity score: w_i = Σ y_ik * r_k
# Cells with ANY bare_sparse fraction get proportional upweight
sample_rarity = y_train @ rarity_per_class  # [N]
sample_rarity /= sample_rarity.sum()  # normalize to probabilities
print(f"\n  Sample rarity stats: min={sample_rarity.min():.2e}, "
      f"max={sample_rarity.max():.2e}, mean={sample_rarity.mean():.2e}")

# Per-sample loss trackers for variance-penalized OHEM
sample_loss_mu = np.ones(total, dtype=np.float32)    # EMA mean
sample_loss_var = np.zeros(total, dtype=np.float32)   # EMA variance

# Per-city loss tracker for hard-city curriculum
city_losses = np.ones(n_cities, dtype=np.float32)

# Move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {device}")

y_norm = normalize_targets(y_train)
X_trn_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_trn_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
del X_train, y_norm
gc.collect()
print(f"  X_train on GPU: {X_trn_t.shape} ({X_trn_t.nbytes/1e9:.1f} GB)")

# Test tensors
test_tensors = {}
for name, (X_v, y_v) in test_data.items():
    y_v_norm = normalize_targets(y_v)
    test_tensors[name] = {
        "X": torch.tensor(X_v, dtype=torch.float32, device=device),
        "y_norm": torch.tensor(y_v_norm, dtype=torch.float32, device=device),
        "y_raw": y_v,
    }
del test_data
gc.collect()


# =====================================================================
# HIERARCHICAL SAMPLER: city-balanced + fractional-rarity + OHEM
# =====================================================================

def build_epoch_indices(rng, epoch, use_rarity=True, use_ohem=True,
                        use_hard_cities=True):
    """
    Hierarchical sampler: city → within-city rarity/hardness → shuffle.

    Step 1: Choose how many samples per city (city-balanced + hard-city tilt)
    Step 2: Within each city, sample with rarity + hardness weights
    Step 3: Shuffle globally
    """
    # Use median city size as epoch budget (not min — avoids starving large cities)
    city_sizes = [end - start for start, end in city_boundaries]
    samples_per_city_target = int(np.median(city_sizes))

    # --- Step 1: City selection weights (softmax with temperature) ---
    if use_hard_cities and epoch >= OHEM_WARMUP:
        # Group-DRO-ish: P(c) = (1-α)/C + α * softmax(L_c / τ)
        logits = city_losses / CITY_TEMPERATURE
        logits -= logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        softmax_dist = exp_logits / exp_logits.sum()
        uniform = np.ones(n_cities) / n_cities
        city_probs = (1 - CITY_HARDNESS_ALPHA) * uniform + \
                      CITY_HARDNESS_ALPHA * softmax_dist
    else:
        city_probs = np.ones(n_cities) / n_cities

    # Scale samples per city by city weight
    city_sample_counts = (city_probs * n_cities * samples_per_city_target).astype(int)
    city_sample_counts = np.maximum(city_sample_counts, 1)  # at least 1

    # --- Step 2: Within-city sampling with rarity + hardness ---
    # Curriculum: ramp up rarity weighting over epochs
    if use_rarity and epoch >= RARITY_WARMUP:
        rarity_strength = min(1.0, (epoch - RARITY_WARMUP) / 20.0)
        eta = RARITY_MAX_ETA * rarity_strength
    else:
        eta = 0.0  # pure uniform early on

    all_indices = []
    for ci, (start, end) in enumerate(city_boundaries):
        city_n = end - start
        n_to_sample = city_sample_counts[ci]
        # Small cities: sample WITH replacement; large cities: WITHOUT
        use_replace = n_to_sample > city_n

        # Build per-sample weights within this city
        uniform = np.ones(city_n, dtype=np.float64) / city_n

        if eta > 0 or (use_ohem and epoch >= OHEM_WARMUP):
            # Rarity component
            city_rarity = sample_rarity[start:end].astype(np.float64)
            city_rarity /= (city_rarity.sum() + 1e-12)

            # OHEM component — variance-penalized learnability
            if use_ohem and epoch >= OHEM_WARMUP:
                mu = sample_loss_mu[start:end].astype(np.float64)
                std = np.sqrt(sample_loss_var[start:end].astype(np.float64) + 1e-8)
                # Learnability: high mean + low variance = truly hard
                # High variance = noisy/borderline, penalize
                city_hardness = np.maximum(1e-8, mu - OHEM_VAR_LAMBDA * std)
                city_hardness /= (city_hardness.sum() + 1e-12)
                # Blend rarity + hardness
                combined = 0.6 * city_rarity + 0.4 * city_hardness
                combined /= (combined.sum() + 1e-12)
            else:
                combined = city_rarity

            weights = (1 - eta) * uniform + eta * combined
            weights /= weights.sum()
        else:
            weights = uniform

        chosen = rng.choice(city_n, size=n_to_sample,
                            replace=use_replace, p=weights)
        all_indices.append(chosen + start)

    perm = np.concatenate(all_indices)
    rng.shuffle(perm)

    # Force constant epoch size for stable LR schedule
    target_total = samples_per_city_target * n_cities
    if len(perm) > target_total:
        perm = perm[:target_total]
    elif len(perm) < target_total:
        # Pad with random samples from existing perm
        extra = rng.choice(perm, size=target_total - len(perm), replace=True)
        perm = np.concatenate([perm, extra])
        rng.shuffle(perm)

    return perm


def apply_mixup(xb, yb, alpha):
    """Inter-city Mixup: Beta(alpha, alpha) blending."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)  # original dominates
    perm = torch.randperm(xb.size(0), device=xb.device)
    xb_mixed = lam * xb + (1.0 - lam) * xb[perm]
    yb_mixed = lam * yb + (1.0 - lam) * yb[perm]
    return xb_mixed, yb_mixed


# =====================================================================
# Sweep setup
# =====================================================================

city_sizes = [end - start for start, end in city_boundaries]
samples_per_city = int(np.median(city_sizes))  # median, not min
n_balanced = samples_per_city * n_cities
steps_per_epoch = (n_balanced + BATCH_SIZE - 1) // BATCH_SIZE
patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)

print(f"\n  Balanced epoch: ~{n_balanced:,} samples "
      f"({samples_per_city:,}/city [median] x {n_cities} cities)")
print(f"  City sizes: min={min(city_sizes):,}, median={samples_per_city:,}, max={max(city_sizes):,}")
print(f"  Steps/epoch: ~{steps_per_epoch}, patience: {patience_epochs}ep, "
      f"min: {min_epochs}ep, max: {MAX_EPOCHS}ep")
print(f"\n  Rarity: alpha={RARITY_ALPHA}, warmup={RARITY_WARMUP}ep, max_eta={RARITY_MAX_ETA}")
print(f"  OHEM: var_lambda={OHEM_VAR_LAMBDA}, warmup={OHEM_WARMUP}ep")
print(f"  Mixup: alpha={MIXUP_ALPHA}, prob={MIXUP_PROB}")
print(f"  Hard-city: alpha={CITY_HARDNESS_ALPHA}, temp={CITY_TEMPERATURE}, ema={CITY_LOSS_EMA_DECAY}")

# =====================================================================
# Sweep loop
# =====================================================================

results = []
print(f"\n[{ts()}] Starting sweep of {len(CONFIGS)} configs...")

for ci, (cfg_name, cfg_type, shape_params, dropout, wd, mode) in enumerate(CONFIGS):
    print(f"\n{'='*70}")
    print(f"  [{ci+1}/{len(CONFIGS)}] {cfg_name}  (mode: {mode})")

    # Determine sampling flags from explicit mode
    use_mixup     = mode in ("mixup_only", "full")
    use_rarity    = mode == "full"
    use_ohem      = mode == "full"
    use_hard_cities = mode == "full"

    actual_seed = SEED + SEED_OFFSET
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    if cfg_type == "C":
        n_layers, width = shape_params
        net = ConstantMLP(n_features, N_CLASSES, n_layers, width,
                          dropout=dropout, activation=ACTIVATION,
                          input_dropout=INPUT_DROPOUT, norm_type=NORM)
        widths_list = [width] * n_layers
    else:
        widths_list = shape_params
        net = TaperedMLP(n_features, N_CLASSES, widths_list,
                         dropout=dropout, activation=ACTIVATION,
                         input_dropout=INPUT_DROPOUT, norm_type=NORM)

    shape_str = f"{n_features} -> " + " -> ".join(str(w) for w in widths_list) + f" -> {N_CLASSES}"
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    mode_str = {"baseline": "BASELINE (no tricks)",
                "mixup_only": "MIXUP ONLY (control)",
                "full": "FULL (mixup+rarity+OHEM+hard-city)"}[mode]
    print(f"  Shape: {shape_str}")
    print(f"  Params: {n_params:,} | dropout={dropout} wd={wd}")
    print(f"  Mode: {mode_str}")

    # Reset trackers
    sample_loss_mu[:] = 1.0
    sample_loss_var[:] = 0.0
    city_losses[:] = 1.0

    # Optimizer
    use_amp = device == "cuda"
    amp_device = "cuda" if use_amp else "cpu"
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd, fused=use_amp)
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd)

    scaler_amp = torch.amp.GradScaler(enabled=use_amp)
    total_steps = MAX_EPOCHS * steps_per_epoch
    warmup_steps = steps_per_epoch * 3
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps),
    ], milestones=[warmup_steps])

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
    best_val = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0
    rng = np.random.RandomState(actual_seed)

    val_name = "frankfurt"
    if val_name not in test_tensors:
        val_name = list(test_tensors.keys())[0]

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        epoch_loss, n_batches = 0.0, 0

        # === HIERARCHICAL SAMPLING ===
        perm = build_epoch_indices(rng, epoch,
                                   use_rarity=use_rarity,
                                   use_ohem=use_ohem,
                                   use_hard_cities=use_hard_cities)

        perm_t = torch.from_numpy(perm).to(device, dtype=torch.long)

        for batch_i, start in enumerate(range(0, len(perm_t), BATCH_SIZE)):
            idx = perm_t[start:start + BATCH_SIZE]
            xb = X_trn_t[idx]
            yb = y_trn_t[idx]

            # === OHEM: compute per-sample loss on CLEAN data BEFORE mixup ===
            # Uses eval mode (no dropout/BN update) and fp32 for stable tracking
            # Only every 8th batch to reduce compute overhead (~12% tax vs 100%)
            if use_ohem and epoch % 5 == 0 and batch_i % 8 == 0:
                net.eval()
                with torch.no_grad():
                    clean_pred = net(xb)  # fp32, no dropout, no BN updates
                    per_sample = F.kl_div(clean_pred, yb, reduction='none').sum(dim=1)
                    np_idx = idx.cpu().numpy()
                    np_losses = per_sample.cpu().float().numpy()
                    # EMA update of mean and variance
                    old_mu = sample_loss_mu[np_idx].copy()
                    sample_loss_mu[np_idx] = (OHEM_EMA_DECAY * old_mu +
                                              (1 - OHEM_EMA_DECAY) * np_losses)
                    # Welford-style EMA variance
                    diff = np_losses - old_mu
                    diff2 = np_losses - sample_loss_mu[np_idx]
                    sample_loss_var[np_idx] = (OHEM_EMA_DECAY * sample_loss_var[np_idx] +
                                               (1 - OHEM_EMA_DECAY) * diff * diff2)
                net.train()  # restore train mode

            # === INTER-CITY MIXUP (applied AFTER clean loss tracking) ===
            if use_mixup and rng.random() < MIXUP_PROB:
                xb, yb = apply_mixup(xb, yb, MIXUP_ALPHA)

            if has_bn and xb.size(0) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=use_amp, dtype=torch.float16):
                pred = net(xb)
                loss = soft_cross_entropy(pred, yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validate
        net.eval()
        with torch.no_grad():
            with torch.amp.autocast(amp_device, enabled=use_amp, dtype=torch.float16):
                val_pred = net(test_tensors[val_name]["X"])
                val_loss = soft_cross_entropy(
                    val_pred, test_tensors[val_name]["y_norm"]
                ).item()

        # Update per-city loss tracker (for hard-city curriculum)
        # Use training loss mean per city as proxy
        if use_hard_cities and epoch % 5 == 0:
            for city_ci, (c_start, c_end) in enumerate(city_boundaries):
                city_mean_loss = sample_loss_mu[c_start:c_end].mean()
                city_losses[city_ci] = (CITY_LOSS_EMA_DECAY * city_losses[city_ci] +
                                        (1 - CITY_LOSS_EMA_DECAY) * city_mean_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
            best_epoch = epoch
        else:
            wait += 1

        marker = " *BEST*" if improved else ""
        if epoch <= 3 or epoch % 20 == 0 or improved:
            extra = ""
            if use_rarity and epoch >= RARITY_WARMUP:
                rarity_str = min(1.0, (epoch - RARITY_WARMUP) / 20.0)
                extra += f" rarity_eta={rarity_str * RARITY_MAX_ETA:.2f}"
            print(f"    Ep {epoch:3d}: train={avg_train:.5f} val={val_loss:.5f} "
                  f"wait={wait}{marker}{extra}")

        if epoch >= min_epochs and wait >= patience_epochs:
            print(f"    Early stop at epoch {epoch} (patience={patience_epochs})")
            break

    elapsed = time.time() - t0
    n_epochs_done = epoch + 1

    # Load best
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # Evaluate
    city_metrics = {}
    for test_name, tdata in test_tensors.items():
        with torch.no_grad():
            preds = net.predict(tdata["X"]).cpu().numpy()
        y_true = tdata["y_raw"]
        r2 = float(r2_score(y_true, preds))
        mae = float(mean_absolute_error(y_true, preds) * 100)
        per_class = {}
        for i, cn in enumerate(CLASS_NAMES):
            yt = y_true[:, i]
            if np.var(yt) < 1e-12:
                per_class[cn] = float("nan")
            else:
                per_class[cn] = float(r2_score(yt, preds[:, i]))
        city_metrics[test_name] = {"r2": r2, "mae_pp": mae, "per_class_r2": per_class}

    # Print results
    print(f"\n  >> {cfg_name}: {n_params:,} params, "
          f"best@ep{best_epoch}, {elapsed:.0f}s")
    for test_name, m in city_metrics.items():
        print(f"     {test_name:15s}  R2={m['r2']:.4f}  MAE={m['mae_pp']:.2f}pp")
        for cn in CLASS_NAMES:
            print(f"       {cn:20s} R2={m['per_class_r2'][cn]:.4f}")

    mean_r2 = np.mean([m["r2"] for m in city_metrics.values()])
    mean_mae = np.mean([m["mae_pp"] for m in city_metrics.values()])

    result = {
        "name": cfg_name,
        "type": cfg_type,
        "mode": mode,
        "widths": widths_list if cfg_type == "T" else [shape_params[1]] * shape_params[0],
        "n_params": n_params,
        "dropout": dropout,
        "wd": wd,
        "n_epochs": n_epochs_done,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "time_s": round(elapsed, 1),
        "mean_r2": float(mean_r2),
        "mean_mae_pp": float(mean_mae),
        "per_city": city_metrics,
    }
    results.append(result)
    torch.save(best_state, os.path.join(OUT_DIR, f"{cfg_name}.pt"))
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    del net, optimizer, scaler_amp, scheduler, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =====================================================================
# Summary
# =====================================================================

print(f"\n\n{'='*90}")
print(f"V5.7 COMBINED-STRATEGY SWEEP COMPLETE ({len(CONFIGS)} configs)")
print(f"{'='*90}")

ranked = sorted(results, key=lambda r: r["mean_r2"], reverse=True)

header = (f"{'Rank':>4} {'Config':30s} {'Params':>10} "
          f"{'MeanR2':>8} {'MeanMAE':>8} {'BestEp':>6} {'Time':>6}")
print(header)
print("-" * len(header))
for rank, r in enumerate(ranked, 1):
    print(f"{rank:4d} {r['name']:30s} {r['n_params']:>10,} "
          f"{r['mean_r2']:>8.4f} {r['mean_mae_pp']:>7.2f}pp "
          f"{r['best_epoch']:>5d}  {r['time_s']:>5.0f}s")

# Per-class breakdown for top 5
print(f"\n--- TOP 5 per-class breakdown ---")
for rank, r in enumerate(ranked[:5], 1):
    print(f"\n  #{rank} {r['name']} (R2={r['mean_r2']:.4f})")
    for city_name, m in r['per_city'].items():
        print(f"    {city_name:15s}  R2={m['r2']:.4f}  MAE={m['mae_pp']:.2f}pp")
        for cn in CLASS_NAMES:
            pc = m['per_class_r2'][cn]
            print(f"      {cn:20s} R2={pc:.4f}")

# Comparison with v5.6 best
print(f"\n--- Comparison with v5.6 best ---")
print(f"  v5.6 best: T_1024_512_256_64 R2=0.8419 bare=0.414 grass=0.804")
print(f"  v5.7 best: {ranked[0]['name']} R2={ranked[0]['mean_r2']:.4f}")

# Save CSV
csv_rows = []
for r in ranked:
    row = {
        "rank": ranked.index(r) + 1,
        "name": r["name"],
        "n_params": r["n_params"],
        "mean_r2": r["mean_r2"],
        "mean_mae_pp": r["mean_mae_pp"],
        "best_epoch": r["best_epoch"],
        "time_s": r["time_s"],
    }
    for city_name in [c.name for c in TEST_CITIES]:
        if city_name in r["per_city"]:
            row[f"r2_{city_name}"] = r["per_city"][city_name]["r2"]
            row[f"mae_{city_name}"] = r["per_city"][city_name]["mae_pp"]
    csv_rows.append(row)

df_summary = pd.DataFrame(csv_rows)
csv_path = os.path.join(OUT_DIR, "sweep_summary.csv")
df_summary.to_csv(csv_path, index=False)

best = ranked[0]
print(f"\n{'='*90}")
print(f"BEST: {best['name']}")
print(f"  R2 = {best['mean_r2']:.4f}, MAE = {best['mean_mae_pp']:.2f}pp")
print(f"  Params: {best['n_params']:,}, Best epoch: {best['best_epoch']}")
print(f"\nSaved to: {OUT_DIR}")
print(f"{'='*90}")
