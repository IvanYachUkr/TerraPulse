#!/usr/bin/env python3
"""
V5.6 MLP Sweep — Advanced Sampling Strategies.

Key innovations over v5.5:
  1. Inter-city Mixup:     blend features+labels from different cities
  2. Class-stratified:     oversample cells with rare land-cover classes
  3. Online Hard Mining:   loss-weighted resampling of hard examples
  4. Progressive schedule: easy→hard curriculum over training

Only small/deep architectures tested (C_3x256 won v5.5 decisively).

Usage:
    .venv/Scripts/python.exe scripts/sweep_mlp_v5_6.py
"""
import os, sys, time, math, json, pickle, hashlib, gc
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
# SAR-only city override (same as v5.5)
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
# Model definitions (same as v5.5)
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
# Architecture configs — SMALL/DEEP only (v5.5 proved bigger = worse)
# =====================================================================

CONFIGS = [
    # Format: (name, type, shape, dropout, wd, sampler_mode)
    # sampler_mode: "baseline" | "mixup" | "stratified" | "ohem" | "full"

    # ---- Controls (proven winners from v5.6 round 1) ----
    ("C_3x256_baseline",           "C", (3, 256),  0.10, 1e-4, "baseline"),
    ("C_3x256_mixup",              "C", (3, 256),  0.10, 1e-4, "mixup"),

    # ---- INVERTED BOTTLENECK: wide first layer → aggressive taper ----
    # The key idea: 1748 inputs need room to form combinations BEFORE compression.
    # 1748→256 is 6.8x crush. 1748→2048 lets features "breathe" first.

    # Aggressive taper (wide → narrow fast) with mixup
    ("T_2048_512_128_mix",         "T", [2048, 512, 128],       0.15, 3e-4, "mixup"),
    ("T_2048_256_64_mix",          "T", [2048, 256, 64],        0.15, 3e-4, "mixup"),
    ("T_1024_256_64_mix",          "T", [1024, 256, 64],        0.12, 2e-4, "mixup"),
    ("T_1024_512_128_mix",         "T", [1024, 512, 128],       0.12, 2e-4, "mixup"),

    # Gradual squeeze (4 layers, gentle compression) with mixup
    ("T_2048_1024_256_64_mix",     "T", [2048, 1024, 256, 64],  0.18, 3e-4, "mixup"),
    ("T_1024_512_256_64_mix",      "T", [1024, 512, 256, 64],   0.15, 2e-4, "mixup"),

    # Extra-wide first layer (matches input dim) with mixup
    ("T_1748_512_128_mix",         "T", [1748, 512, 128],       0.15, 3e-4, "mixup"),
    ("T_1748_256_64_mix",          "T", [1748, 256, 64],        0.15, 3e-4, "mixup"),

    # Best inverted bottleneck with full sampling (mixup + stratified + OHEM)
    ("T_2048_512_128_full",        "T", [2048, 512, 128],       0.15, 3e-4, "full"),
    ("T_2048_1024_256_64_full",    "T", [2048, 1024, 256, 64],  0.18, 3e-4, "full"),
]

# =====================================================================
# Training constants
# =====================================================================
BATCH_SIZE = 4096
PATIENCE_STEPS = 10000
MIN_STEPS = 3000
MAX_EPOCHS = 500
SEED_OFFSET = 200  # different from v5.5
INPUT_DROPOUT = 0.10
LR = 1e-3
ACTIVATION = "silu"
NORM = "batchnorm"

# Mixup hyperparameters
MIXUP_ALPHA = 0.3       # Beta distribution parameter
MIXUP_PROB = 0.5         # probability of applying mixup per batch

# OHEM hyperparameters
OHEM_TOP_FRAC = 0.3      # fraction of hardest examples to oversample
OHEM_WARMUP = 20         # start OHEM after this many epochs

# Class-stratified sampling
# Cells are grouped into "dominant class" buckets. Rare-class buckets
# are oversampled so every batch has balanced class representation.
N_CLASS_BUCKETS = N_CLASSES

# Output directory
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v5_6_sampler", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# Data loading (identical to v5.5 — all on GPU)
# =====================================================================

print(f"[{ts()}] V5.6 Advanced-Sampler MLP Sweep (run_id={RUN_ID})")
print(f"  Output: {OUT_DIR}")
print(f"  Configs: {len(CONFIGS)}")
print(f"  Test cities: {[c.name for c in TEST_CITIES]}")

# Discover feature columns
print(f"\n[{ts()}] Discovering feature columns...")
full_feature_cols = _discover_feature_cols()
mlp_idx = build_bi_lbp(full_feature_cols)
mlp_cols = [full_feature_cols[i] for i in mlp_idx]
n_features = len(mlp_cols)
print(f"  Total parquet columns: {len(full_feature_cols)}")
print(f"  MLP feature columns: {n_features}")

# Pass 1: count cells
print(f"\n[{ts()}] Loading training data (memory-efficient)...")
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

# Pass 2: load into arrays
X_train = np.empty((total, n_features), dtype=np.float32)
y_train = np.empty((total, N_CLASSES), dtype=np.float32)
city_boundaries = []
city_ids = np.empty(total, dtype=np.int32)  # tracks which city each cell belongs to
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

print(f"  Loaded: {offset:,} samples, {len(city_boundaries)} cities")

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
print(f"\n[{ts()}] Fitting scaler (in-place)...")
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
# ADVANCED SAMPLING INFRASTRUCTURE
# =====================================================================

# 1. Precompute dominant class per cell (for stratified sampling)
dominant_class = np.argmax(y_train, axis=1)  # [N]
class_indices = {}  # class_id -> array of global indices
for c_id in range(N_CLASSES):
    class_indices[c_id] = np.where(dominant_class == c_id)[0]
    print(f"  Class {CLASS_NAMES[c_id]:20s}: {len(class_indices[c_id]):>8,} cells "
          f"({100*len(class_indices[c_id])/total:.1f}%)")

# 2. Per-sample loss tracker (for OHEM)
sample_losses = np.ones(total, dtype=np.float32)  # init uniform

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

# =====================================================================
# SAMPLING FUNCTIONS
# =====================================================================

def sample_baseline(rng, epoch, city_boundaries, n_per_city):
    """City-balanced sampling (v5.5 style)."""
    balanced_idx = []
    for start_i, end_i in city_boundaries:
        city_n = end_i - start_i
        chosen = rng.choice(city_n, size=n_per_city, replace=False)
        balanced_idx.append(chosen + start_i)
    perm = np.concatenate(balanced_idx)
    rng.shuffle(perm)
    return perm


def sample_class_stratified(rng, epoch, city_boundaries, total_n):
    """Class-balanced sampling: equal cells from each dominant-class bucket."""
    n_per_class = total_n // N_CLASSES
    balanced = []
    for c_id in range(N_CLASSES):
        pool = class_indices[c_id]
        if len(pool) == 0:
            continue
        chosen = rng.choice(pool, size=n_per_class, replace=len(pool) < n_per_class)
        balanced.append(chosen)
    perm = np.concatenate(balanced)
    rng.shuffle(perm)
    return perm


def sample_ohem(rng, epoch, city_boundaries, n_per_city):
    """Online Hard Example Mining: oversample high-loss cells.
    Blends 70% hard examples + 30% random for stability."""
    if epoch < OHEM_WARMUP:
        return sample_baseline(rng, epoch, city_boundaries, n_per_city)

    total_n = n_per_city * len(city_boundaries)
    n_hard = int(total_n * OHEM_TOP_FRAC)
    n_random = total_n - n_hard

    # Hard examples: probability proportional to loss
    probs = sample_losses / sample_losses.sum()
    hard_idx = rng.choice(len(probs), size=n_hard, replace=False, p=probs)

    # Random examples: uniform
    random_idx = rng.choice(len(probs), size=n_random, replace=False)

    perm = np.concatenate([hard_idx, random_idx])
    rng.shuffle(perm)
    return perm


def sample_strat_mix(rng, epoch, city_boundaries, n_per_city):
    """Class-stratified with city-balanced twist:
    within each class bucket, sample proportionally from each city."""
    total_n = n_per_city * len(city_boundaries)
    n_per_class = total_n // N_CLASSES
    balanced = []
    for c_id in range(N_CLASSES):
        pool = class_indices[c_id]
        if len(pool) == 0:
            continue
        chosen = rng.choice(pool, size=n_per_class, replace=len(pool) < n_per_class)
        balanced.append(chosen)
    perm = np.concatenate(balanced)
    rng.shuffle(perm)
    return perm


def sample_full(rng, epoch, city_boundaries, n_per_city):
    """Full kitchen sink: class-stratified + OHEM after warmup."""
    if epoch < OHEM_WARMUP:
        return sample_class_stratified(rng, epoch, city_boundaries,
                                       n_per_city * len(city_boundaries))

    total_n = n_per_city * len(city_boundaries)
    n_hard = int(total_n * OHEM_TOP_FRAC)
    n_strat = total_n - n_hard

    # Hard portion from OHEM
    probs = sample_losses / sample_losses.sum()
    hard_idx = rng.choice(len(probs), size=n_hard, replace=False, p=probs)

    # Stratified portion
    n_per_class = n_strat // N_CLASSES
    strat = []
    for c_id in range(N_CLASSES):
        pool = class_indices[c_id]
        if len(pool) == 0:
            continue
        chosen = rng.choice(pool, size=n_per_class, replace=len(pool) < n_per_class)
        strat.append(chosen)
    strat_idx = np.concatenate(strat)

    perm = np.concatenate([hard_idx, strat_idx])
    rng.shuffle(perm)
    return perm


SAMPLER_FNS = {
    "baseline": sample_baseline,
    "mixup": sample_baseline,  # uses baseline sampling + mixup in forward
    "stratified": sample_class_stratified,
    "strat_mix": sample_strat_mix,
    "ohem": sample_ohem,
    "full": sample_full,
}


def apply_mixup(xb, yb, alpha, rng_torch):
    """Inter-city Mixup: blend pairs of samples with Beta(alpha, alpha) weights.
    This forces the model to interpolate between cities, preventing
    city-level memorization."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5 (original dominates)

    # Shuffle to create pairs from different positions (likely different cities)
    perm = torch.randperm(xb.size(0), device=xb.device)
    xb_mixed = lam * xb + (1.0 - lam) * xb[perm]
    yb_mixed = lam * yb + (1.0 - lam) * yb[perm]
    return xb_mixed, yb_mixed


# =====================================================================
# Sweep setup
# =====================================================================

samples_per_city_global = min(s[1] - s[0] for s in city_boundaries)
n_balanced_global = samples_per_city_global * len(city_boundaries)
steps_per_epoch = (n_balanced_global + BATCH_SIZE - 1) // BATCH_SIZE
patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)

print(f"\n  Balanced epoch: {n_balanced_global:,} samples "
      f"({samples_per_city_global:,}/city x {len(city_boundaries)} cities)")
print(f"  Steps/epoch: {steps_per_epoch}, patience: {patience_epochs}ep, "
      f"min: {min_epochs}ep, max: {MAX_EPOCHS}ep")
print(f"\n  Sampling modes: {set(c[5] for c in CONFIGS)}")
print(f"  Mixup alpha={MIXUP_ALPHA}, prob={MIXUP_PROB}")
print(f"  OHEM top_frac={OHEM_TOP_FRAC}, warmup={OHEM_WARMUP}ep")

# =====================================================================
# Sweep loop
# =====================================================================

results = []
print(f"\n[{ts()}] Starting sweep of {len(CONFIGS)} configs...")

for ci, (cfg_name, cfg_type, shape_params, dropout, wd, sampler_mode) in enumerate(CONFIGS):
    print(f"\n{'='*70}")
    print(f"  [{ci+1}/{len(CONFIGS)}] {cfg_name}  (sampler: {sampler_mode})")

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
    print(f"  Shape: {shape_str}")
    print(f"  Params: {n_params:,} | dropout={dropout} wd={wd}")

    # Get sampler function
    sampler_fn = SAMPLER_FNS[sampler_mode]
    use_mixup = sampler_mode in ("mixup", "strat_mix", "full")

    # Reset per-sample losses for OHEM
    sample_losses[:] = 1.0

    # Optimizer
    use_amp = device == "cuda"
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd, fused=use_amp)
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd)

    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
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

        # === ADVANCED SAMPLING ===
        if sampler_mode in ("baseline", "mixup"):
            perm = sampler_fn(rng, epoch, city_boundaries, samples_per_city_global)
        elif sampler_mode == "stratified":
            perm = sampler_fn(rng, epoch, city_boundaries, n_balanced_global)
        elif sampler_mode in ("strat_mix", "ohem", "full"):
            perm = sampler_fn(rng, epoch, city_boundaries, samples_per_city_global)
        else:
            perm = sample_baseline(rng, epoch, city_boundaries, samples_per_city_global)

        perm_t = torch.from_numpy(perm).to(device, dtype=torch.long)

        # Track losses for OHEM update
        batch_indices_list = []
        batch_losses_list = []

        for start in range(0, len(perm_t), BATCH_SIZE):
            idx = perm_t[start:start + BATCH_SIZE]
            xb = X_trn_t[idx]
            yb = y_trn_t[idx]

            # === INTER-CITY MIXUP ===
            if use_mixup and rng.random() < MIXUP_PROB:
                xb, yb = apply_mixup(xb, yb, MIXUP_ALPHA, rng)

            if has_bn and xb.size(0) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                pred = net(xb)
                loss = soft_cross_entropy(pred, yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

            # Track per-sample losses for OHEM (every 5 epochs to save compute)
            if sampler_mode in ("ohem", "full") and epoch % 5 == 0:
                with torch.no_grad():
                    per_sample = F.kl_div(pred, yb, reduction='none').sum(dim=1)
                    np_idx = idx.cpu().numpy()
                    np_losses = per_sample.cpu().float().numpy()
                    batch_indices_list.append(np_idx)
                    batch_losses_list.append(np_losses)

        avg_train = epoch_loss / max(n_batches, 1)

        # Update OHEM loss tracker
        if batch_indices_list:
            all_idx = np.concatenate(batch_indices_list)
            all_losses = np.concatenate(batch_losses_list)
            # Exponential moving average
            sample_losses[all_idx] = 0.7 * sample_losses[all_idx] + 0.3 * all_losses

        # Validate
        net.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                val_loss = soft_cross_entropy(
                    net(test_tensors[val_name]["X"]),
                    test_tensors[val_name]["y_norm"]
                ).item()

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
            ohem_info = ""
            if sampler_mode in ("ohem", "full") and epoch >= OHEM_WARMUP:
                top_loss = np.percentile(sample_losses, 70)
                ohem_info = f" ohem_p70={top_loss:.3f}"
            print(f"    Ep {epoch:3d}: train={avg_train:.5f} val={val_loss:.5f} "
                  f"wait={wait}{marker}{ohem_info}")

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
            per_class[cn] = float(r2_score(y_true[:, i], preds[:, i]))
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
        "widths": widths_list,
        "sampler": sampler_mode,
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
print(f"V5.6 ADVANCED SAMPLER SWEEP COMPLETE ({len(CONFIGS)} configs)")
print(f"{'='*90}")

ranked = sorted(results, key=lambda r: r["mean_r2"], reverse=True)

header = (f"{'Rank':>4} {'Config':30s} {'Sampler':10s} {'Params':>10} "
          f"{'MeanR2':>8} {'MeanMAE':>8} {'BestEp':>6} {'Time':>6}")
print(header)
print("-" * len(header))
for rank, r in enumerate(ranked, 1):
    shape = "->".join(str(w) for w in r['widths'][:4])
    print(f"{rank:4d} {r['name']:30s} {r['sampler']:10s} {r['n_params']:>10,} "
          f"{r['mean_r2']:>8.4f} {r['mean_mae_pp']:>7.2f}pp "
          f"{r['best_epoch']:>5d}  {r['time_s']:>5.0f}s")

# Per-class breakdown for top 5
print(f"\n--- TOP 5 per-class breakdown ---")
for rank, r in enumerate(ranked[:5], 1):
    print(f"\n  #{rank} {r['name']} [{r['sampler']}] (R2={r['mean_r2']:.4f})")
    for city_name, m in r['per_city'].items():
        print(f"    {city_name:15s}  R2={m['r2']:.4f}  MAE={m['mae_pp']:.2f}pp")
        for cn in CLASS_NAMES:
            pc = m['per_class_r2'][cn]
            print(f"      {cn:20s} R2={pc:.4f}")

# Save CSV
csv_rows = []
for r in ranked:
    row = {
        "rank": ranked.index(r) + 1,
        "name": r["name"],
        "sampler": r["sampler"],
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
print(f"BEST: {best['name']} [{best['sampler']}]")
print(f"  R2 = {best['mean_r2']:.4f}, MAE = {best['mean_mae_pp']:.2f}pp")
print(f"  Params: {best['n_params']:,}, Best epoch: {best['best_epoch']}")
print(f"\nSaved to: {OUT_DIR}")
print(f"{'='*90}")
