#!/usr/bin/env python3
"""
V6 MLP Sweep — SAR-Only Cities + Advanced Sampling + SAR Temporal Features.

Combines:
  - From v5:  CPU-based memory-efficient data loading  (50+ cities → CPU arrays
              → GPU mini-batches), multi-city test evaluation, broad arch configs.
  - From v5.7: Fractional rarity weighting, city-balanced hierarchical sampler,
              variance-penalized OHEM, hard-city curriculum (Group-DRO), inter-city
              Mixup, proper AMP device handling, gradient clipping, GradScaler fix.

Key v6 changes:
  - SAR-aware city filtering via parquet schema check
  - Feature columns = INTERSECTION across all selected cities (no mismatch)
  - Robust 2-pass loading with trim-to-offset (no uninitialized tail)
  - Test data kept on CPU; transferred to GPU one city at a time
  - CPU batch transfers with from_numpy + non_blocking (marginal gain)
  - OHEM variance clamped to >= 0 before sqrt (no NaN)
  - OHEM tracks per-sample soft CE instead of KL (no entropy bias)
  - y_train freed after rarity computation to save RAM

Usage:
    .venv/Scripts/python.exe scripts/sweep_mlp_v6.py
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
    SEED, CLASS_NAMES, N_CLASSES,
    city_features_dir, city_labels_path,
    _discover_feature_cols, _load_city_arrays, build_bi_lbp, ts,
)
try:
    from scripts.run_multi_city_pipeline_v5 import CONTROL_COLS
except ImportError:
    CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                    "reflectance_scale", "full_features_computed"}
from scripts.run_mlp_overnight_v4 import (
    _make_norm, PlainBlock, normalize_targets,
    soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pyarrow.parquet as pq

# =====================================================================
# SAR-aware city filtering + feature-column intersection (Bug #2 fix)
# =====================================================================

def _city_parquet_path(city):
    return os.path.join(city_features_dir(city),
                        "features_rust_2020_2021.parquet")


def _city_feature_cols(city):
    """Return set of numeric feature column names from a city's parquet."""
    feat_path = _city_parquet_path(city)
    if not os.path.exists(feat_path):
        return None
    schema = pq.read_schema(feat_path)
    numeric_types = {'float', 'double', 'int32', 'int64', 'float32', 'float64'}
    cols = set()
    for f in schema:
        type_str = str(f.type).lower()
        is_num = any(t in type_str for t in numeric_types)
        if is_num and f.name not in CONTROL_COLS:
            cols.add(f.name)
    return cols


def _city_has_sar(city) -> bool:
    """Check if a city's parquet contains SAR features."""
    cols = _city_feature_cols(city)
    if cols is None:
        return False
    return any(c.startswith("SAR_") for c in cols)


print(f"[{ts()}] V6 MLP Sweep — filtering cities with SAR features...")
TRAIN_CITIES = [c for c in _ALL_TRAIN if _city_has_sar(c)]
TEST_CITIES  = [c for c in _ALL_TEST  if _city_has_sar(c)]
print(f"  SAR train cities: {len(TRAIN_CITIES)} / {len(_ALL_TRAIN)}")
print(f"  SAR test cities:  {len(TEST_CITIES)} / {len(_ALL_TEST)}")
if not TRAIN_CITIES:
    raise RuntimeError("No training cities with SAR features found!")
if not TEST_CITIES:
    print("  WARNING: No test cities with SAR — using last 2 train cities")
    TEST_CITIES = TRAIN_CITIES[-2:]
    TRAIN_CITIES = TRAIN_CITIES[:-2]

print(f"  Train: {[c.name for c in TRAIN_CITIES]}")
print(f"  Test:  {[c.name for c in TEST_CITIES]}")

# Build feature columns as INTERSECTION across ALL selected cities (Bug #2)
print(f"\n[{ts()}] Building feature column intersection across all cities...")
all_city_col_sets = []
for city in TRAIN_CITIES + TEST_CITIES:
    cols = _city_feature_cols(city)
    if cols is not None:
        all_city_col_sets.append(cols)
        print(f"  [{city.name}] {len(cols)} columns")

common_cols = set.intersection(*all_city_col_sets) if all_city_col_sets else set()
# Preserve ordering from the first city's schema
first_schema = pq.read_schema(_city_parquet_path(TRAIN_CITIES[0]))
full_feature_cols = [f.name for f in first_schema if f.name in common_cols]

# Apply build_bi_lbp selection
mlp_idx = build_bi_lbp(full_feature_cols)
mlp_cols = [full_feature_cols[i] for i in mlp_idx]
n_features = len(mlp_cols)

# Report SAR temporal features included
sar_temporal = [c for c in mlp_cols if "temporal" in c or "summer_winter" in c]
print(f"  Common columns across all cities: {len(full_feature_cols)}")
print(f"  MLP feature columns: {n_features}")
print(f"  SAR temporal features: {len(sar_temporal)}")
if sar_temporal:
    for f in sar_temporal[:10]:
        print(f"    {f}")
    if len(sar_temporal) > 10:
        print(f"    ... and {len(sar_temporal) - 10} more")

# =====================================================================
# Model definitions (from v5/v5.7)
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
# Architecture configs — merged from v5 + v5.7
# =====================================================================
# Format: (name, type, shape, dropout, wd, mode)
# mode: "baseline" / "mixup_only" / "full"

CONFIGS = [
    # ==== REMAINING 8 full-mode configs from original sweep ====
    ("T_2048_1024_512_256",        "T", [2048, 1024, 512, 256], 0.25, 1e-3, "full"),
    ("C_5x2048_full",              "C", (5, 2048),              0.30, 1e-3, "full"),
    ("C_4x2048_full",              "C", (4, 2048),              0.25, 1e-3, "full"),
    ("C_5x1024_full",              "C", (5, 1024),              0.20, 5e-4, "full"),
    ("T_512_256_128_64",           "T", [512, 256, 128, 64],    0.10, 1e-4, "full"),
    ("T_768_256_64",               "T", [768, 256, 64],         0.12, 2e-4, "full"),
    ("T_2048_1024_512_256_128",    "T", [2048, 1024, 512, 256, 128], 0.25, 1e-3, "full"),
    ("T_3072_1536_768_384",        "T", [3072, 1536, 768, 384],     0.30, 1e-3, "full"),

    # ==== MIXUP-ONLY variants of ALL full architectures ====
    # Already-completed architectures (re-run with mixup_only)
    ("T_1024_512_256_64_mixup",    "T", [1024, 512, 256, 64],   0.15, 2e-4, "mixup_only"),
    ("T_1024_512_256_128_mixup",   "T", [1024, 512, 256, 128],  0.15, 2e-4, "mixup_only"),
    ("T_1024_512_128_mixup",       "T", [1024, 512, 128],       0.12, 2e-4, "mixup_only"),
    ("T_2048_512_128_mixup",       "T", [2048, 512, 128],       0.15, 3e-4, "mixup_only"),
    ("T_2048_1024_256_64_mixup",   "T", [2048, 1024, 256, 64],  0.18, 3e-4, "mixup_only"),
    ("T_2048_1024_512_mixup",      "T", [2048, 1024, 512],      0.25, 1e-3, "mixup_only"),
    # Remaining architectures (mixup_only)
    ("T_2048_1024_512_256_mixup",  "T", [2048, 1024, 512, 256], 0.25, 1e-3, "mixup_only"),
    ("C_5x2048_mixup",             "C", (5, 2048),              0.30, 1e-3, "mixup_only"),
    ("C_4x2048_mixup",             "C", (4, 2048),              0.25, 1e-3, "mixup_only"),
    ("C_5x1024_mixup",             "C", (5, 1024),              0.20, 5e-4, "mixup_only"),
    ("T_512_256_128_64_mixup",     "T", [512, 256, 128, 64],    0.10, 1e-4, "mixup_only"),
    ("T_768_256_64_mixup",         "T", [768, 256, 64],         0.12, 2e-4, "mixup_only"),
    ("T_2048_1024_512_256_128_mx", "T", [2048, 1024, 512, 256, 128], 0.25, 1e-3, "mixup_only"),
    ("T_3072_1536_768_384_mixup",  "T", [3072, 1536, 768, 384],     0.30, 1e-3, "mixup_only"),
]

# =====================================================================
# Training constants
# =====================================================================
BATCH_SIZE = 4096
PATIENCE_STEPS = 10000
MIN_STEPS = 3000
MAX_EPOCHS = 500
SEED_OFFSET = 600   # different seed from all prior sweeps
INPUT_DROPOUT = 0.10
LR = 1e-3
ACTIVATION = "silu"
NORM = "batchnorm"

# Mixup
MIXUP_ALPHA = 0.3
MIXUP_PROB = 0.5

# OHEM — variance-penalized
OHEM_WARMUP = 20
OHEM_VAR_LAMBDA = 0.5
OHEM_EMA_DECAY = 0.7

# Fractional rarity
RARITY_ALPHA = 0.5
RARITY_WARMUP = 5
RARITY_MAX_ETA = 0.7

# Hard-city curriculum
CITY_HARDNESS_ALPHA = 0.2
CITY_TEMPERATURE = 0.5
CITY_LOSS_EMA_DECAY = 0.9

# Output directory
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v6_sweep", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# Data loading (memory-efficient CPU path from v5, with Bug #1 fix)
# =====================================================================

print(f"\n[{ts()}] V6 MLP Sweep starting (run_id={RUN_ID})")
print(f"  Output: {OUT_DIR}")
print(f"  Configs: {len(CONFIGS)}")
print(f"  Test cities: {[c.name for c in TEST_CITIES]}")

# Pass 1: count cells per city
print(f"\n[{ts()}] Loading training data (memory-efficient)...")
city_counts = []
train_cities_valid = []
for city in TRAIN_CITIES:
    feat_path = _city_parquet_path(city)
    if not os.path.exists(feat_path):
        print(f"  WARNING: Missing {city.name} -- skip")
        continue
    n = pq.read_metadata(feat_path).num_rows
    city_counts.append(n)
    train_cities_valid.append(city)
    print(f"  [{city.name}] {n:,} cells")

total = sum(city_counts)
print(f"  Total: {total:,} cells x {n_features} features")
print(f"  X_train will use {total * n_features * 4 / 1e9:.1f} GB RAM")

# Pass 2: pre-allocate and fill
X_train = np.empty((total, n_features), dtype=np.float32)
y_train = np.empty((total, N_CLASSES), dtype=np.float32)
city_boundaries = []
city_ids = np.empty(total, dtype=np.int32)
offset = 0
cities_loaded = []
for ci, (city, n) in enumerate(zip(train_cities_valid, city_counts)):
    result = _load_city_arrays(city, mlp_cols)
    if result is None:
        print(f"  WARNING: {city.name} returned None from loader -- skip")
        continue
    X_city, _ = result

    label_path = city_labels_path(city, 2021)
    if not os.path.exists(label_path):
        print(f"  WARNING: Missing labels for {city.name} -- skip")
        del X_city
        continue
    labels = pd.read_parquet(label_path)
    y_city = labels[CLASS_NAMES].values.astype(np.float32)
    del labels

    # Normalize label rows to sum to 1.0; drop zero-coverage cells
    row_sums = y_city.sum(axis=1, keepdims=True)
    valid_mask = (row_sums.ravel() > 0)
    if not valid_mask.all():
        X_city = X_city[valid_mask]
        y_city = y_city[valid_mask]
        row_sums = row_sums[valid_mask]
    y_city = y_city / np.maximum(row_sums, 1e-8)

    actual_n = min(n, X_city.shape[0], y_city.shape[0])
    X_train[offset:offset + actual_n] = X_city[:actual_n]
    y_train[offset:offset + actual_n] = y_city[:actual_n]
    city_ids[offset:offset + actual_n] = len(city_boundaries)
    city_boundaries.append((offset, offset + actual_n))
    cities_loaded.append(city)
    del X_city, y_city
    offset += actual_n
    gc.collect()

# Bug #1 fix: trim arrays to actual offset (no uninitialized garbage)
if offset < total:
    print(f"  NOTE: Trimming arrays from {total:,} to {offset:,} "
          f"({total - offset:,} cells skipped)")
    X_train = X_train[:offset]
    y_train = y_train[:offset]
    city_ids = city_ids[:offset]
    total = offset

n_cities = len(city_boundaries)
print(f"  Loaded: {total:,} samples, {n_cities} cities")

# Load test data (Bug #3 fix: keep on CPU, not GPU)
print(f"\n[{ts()}] Loading test cities (CPU-resident)...")
test_data = {}
for city in TEST_CITIES:
    result = _load_city_arrays(city, mlp_cols)
    if result is None:
        print(f"  WARNING: Missing {city.name} -- skip")
        continue
    X_val, n_val = result
    label_path = city_labels_path(city, 2021)
    if not os.path.exists(label_path):
        print(f"  WARNING: Missing labels for {city.name} -- skip")
        del X_val
        continue
    labels = pd.read_parquet(label_path)
    y_val = labels[CLASS_NAMES].values.astype(np.float32)
    del labels

    # Normalize label rows to sum to 1.0; drop zero-coverage cells
    row_sums = y_val.sum(axis=1, keepdims=True)
    valid_mask = (row_sums.ravel() > 0)
    if not valid_mask.all():
        X_val = X_val[valid_mask]
        y_val = y_val[valid_mask]
        row_sums = row_sums[valid_mask]
    y_val = y_val / np.maximum(row_sums, 1e-8)

    test_data[city.name] = (X_val, y_val)
    print(f"  [{city.name}] {n_val:,} cells")

if not test_data:
    raise RuntimeError("No test data available!")

# Scale in-place (compute mean/std in float32 to avoid float64 OOM)
print(f"\n[{ts()}] Fitting scaler (in-place, float32)...")
scaler = StandardScaler()
# partial_fit in chunks to avoid sklearn's internal float64 copy
SCALER_CHUNK = 200_000
for sc_start in range(0, total, SCALER_CHUNK):
    scaler.partial_fit(X_train[sc_start:sc_start + SCALER_CHUNK])
X_train -= scaler.mean_.astype(np.float32)
X_train /= scaler.scale_.astype(np.float32)
print(f"  Scaled {total:,} x {n_features} in-place")

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
# FRACTIONAL RARITY WEIGHTS
# =====================================================================
class_prevalence = y_train.mean(axis=0)
print(f"\n  Class prevalence (soft labels):")
for k, cn in enumerate(CLASS_NAMES):
    print(f"    {cn:20s}: {class_prevalence[k]:.4f}")

rarity_per_class = (class_prevalence + 1e-6) ** (-RARITY_ALPHA)
rarity_per_class /= rarity_per_class.max()
print(f"\n  Rarity weights (alpha={RARITY_ALPHA}):")
for k, cn in enumerate(CLASS_NAMES):
    print(f"    {cn:20s}: {rarity_per_class[k]:.3f}")

sample_rarity = y_train @ rarity_per_class
sample_rarity /= sample_rarity.sum()

# Perf #6: free y_train now — we only need X_trn_np and y_trn_np from here
y_norm = normalize_targets(y_train)
del y_train   # saves N * N_CLASSES * 4 bytes of RAM
gc.collect()

# Per-sample loss trackers for OHEM
sample_loss_mu  = np.ones(total, dtype=np.float32)
sample_loss_var = np.zeros(total, dtype=np.float32)

# Per-city loss tracker for hard-city curriculum
city_losses = np.ones(n_cities, dtype=np.float32)

# Training data stays on CPU (too large for GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
amp_device = "cuda" if device == "cuda" else "cpu"
use_amp = device == "cuda"
print(f"\n  Device: {device}")
print(f"  Training data: CPU (batches transferred to GPU)")

X_trn_np = X_train    # already scaled in-place
y_trn_np = y_norm
del y_norm

# Test data stays on CPU as numpy; moved to GPU per-city during eval
test_arrays = {}
for name, (X_v, y_v) in test_data.items():
    y_v_norm = normalize_targets(y_v)
    test_arrays[name] = {
        "X": X_v,                  # numpy, CPU
        "y_norm": y_v_norm,        # numpy, CPU
        "y_raw": y_v,              # numpy, CPU
    }
del test_data
gc.collect()

# Pick early-stop city (stays on CPU; validated in chunks)
val_name = list(test_arrays.keys())[0]
for preferred in ["nuremberg", "frankfurt", "munich"]:
    if preferred in test_arrays:
        val_name = preferred
        break
print(f"  Early-stop city: {val_name} "
      f"({test_arrays[val_name]['X'].shape[0]:,} cells, CPU-chunked)")

# Compute epoch geometry
city_sizes = [end - start for start, end in city_boundaries]
samples_per_city = int(np.median(city_sizes))
n_balanced = samples_per_city * n_cities
steps_per_epoch = (n_balanced + BATCH_SIZE - 1) // BATCH_SIZE
patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)

print(f"\n  Balanced epoch: ~{n_balanced:,} samples "
      f"({samples_per_city:,}/city [median] x {n_cities} cities)")
print(f"  City sizes: min={min(city_sizes):,}, median={samples_per_city:,}, "
      f"max={max(city_sizes):,}")
print(f"  Steps/epoch: ~{steps_per_epoch}, patience: {patience_epochs}ep, "
      f"min: {min_epochs}ep, max: {MAX_EPOCHS}ep")
print(f"\n  Rarity: alpha={RARITY_ALPHA}, warmup={RARITY_WARMUP}ep, "
      f"max_eta={RARITY_MAX_ETA}")
print(f"  OHEM: var_lambda={OHEM_VAR_LAMBDA}, warmup={OHEM_WARMUP}ep")
print(f"  Mixup: alpha={MIXUP_ALPHA}, prob={MIXUP_PROB}")
print(f"  Hard-city: alpha={CITY_HARDNESS_ALPHA}, temp={CITY_TEMPERATURE}")

# =====================================================================
# HIERARCHICAL SAMPLER (from v5.7)
# =====================================================================

def build_epoch_indices(rng, epoch, use_rarity=True, use_ohem=True,
                        use_hard_cities=True):
    """City-balanced hierarchical sampler with rarity + OHEM + hard-city."""
    city_sizes_local = [end - start for start, end in city_boundaries]
    samples_per_city_target = int(np.median(city_sizes_local))

    # City selection weights (softmax with temperature)
    if use_hard_cities and epoch >= OHEM_WARMUP:
        logits = city_losses / CITY_TEMPERATURE
        logits -= logits.max()
        exp_logits = np.exp(logits)
        softmax_dist = exp_logits / exp_logits.sum()
        uniform = np.ones(n_cities) / n_cities
        city_probs = ((1 - CITY_HARDNESS_ALPHA) * uniform +
                      CITY_HARDNESS_ALPHA * softmax_dist)
    else:
        city_probs = np.ones(n_cities) / n_cities

    city_sample_counts = (city_probs * n_cities * samples_per_city_target).astype(int)
    city_sample_counts = np.maximum(city_sample_counts, 1)

    # Within-city sampling
    if use_rarity and epoch >= RARITY_WARMUP:
        rarity_strength = min(1.0, (epoch - RARITY_WARMUP) / 20.0)
        eta = RARITY_MAX_ETA * rarity_strength
    else:
        eta = 0.0

    all_indices = []
    for ci, (start, end) in enumerate(city_boundaries):
        city_n = end - start
        n_to_sample = city_sample_counts[ci]
        use_replace = n_to_sample > city_n

        uniform = np.ones(city_n, dtype=np.float64) / city_n

        if eta > 0 or (use_ohem and epoch >= OHEM_WARMUP):
            city_rarity = sample_rarity[start:end].astype(np.float64)
            city_rarity /= (city_rarity.sum() + 1e-12)

            if use_ohem and epoch >= OHEM_WARMUP:
                mu = sample_loss_mu[start:end].astype(np.float64)
                # Bug #4 fix: clamp variance to >= 0 before sqrt
                var = np.maximum(
                    sample_loss_var[start:end].astype(np.float64), 0.0)
                std = np.sqrt(var + 1e-8)
                city_hardness = np.maximum(1e-8, mu - OHEM_VAR_LAMBDA * std)
                city_hardness /= (city_hardness.sum() + 1e-12)
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

    # Force constant epoch size
    target_total = samples_per_city_target * n_cities
    if len(perm) > target_total:
        perm = perm[:target_total]
    elif len(perm) < target_total:
        extra = rng.choice(perm, size=target_total - len(perm), replace=True)
        perm = np.concatenate([perm, extra])
        rng.shuffle(perm)

    return perm


def apply_mixup(xb, yb, alpha):
    """Inter-city Mixup: Beta(alpha, alpha) blending."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]


# =====================================================================
# Sweep loop
# =====================================================================

results = []
print(f"\n[{ts()}] Starting sweep of {len(CONFIGS)} configs...")

for ci, (cfg_name, cfg_type, shape_params, dropout, wd, mode) in enumerate(CONFIGS):
    print(f"\n{'='*70}")
    print(f"  [{ci+1}/{len(CONFIGS)}] {cfg_name}  (mode: {mode})")

    use_mixup      = mode in ("mixup_only", "full")
    use_rarity     = mode == "full"
    use_ohem       = mode == "full"
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

    shape_str = (f"{n_features} -> " +
                 " -> ".join(str(w) for w in widths_list) +
                 f" -> {N_CLASSES}")
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    mode_str = {"baseline": "BASELINE (no tricks)",
                "mixup_only": "MIXUP ONLY (control)",
                "full": "FULL (mixup+rarity+OHEM+hard-city)"}[mode]
    print(f"  Shape: {shape_str}")
    print(f"  Params: {n_params:,} | dropout={dropout} wd={wd}")
    print(f"  Mode: {mode_str}")

    # Reset trackers per config
    sample_loss_mu[:] = 1.0
    sample_loss_var[:] = 0.0
    city_losses[:] = 1.0

    # Optimizer
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

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        epoch_loss, n_batches = 0.0, 0

        # Hierarchical sampling (v5.7)
        perm = build_epoch_indices(rng, epoch,
                                   use_rarity=use_rarity,
                                   use_ohem=use_ohem,
                                   use_hard_cities=use_hard_cities)

        for batch_i, start in enumerate(range(0, len(perm), BATCH_SIZE)):
            idx = perm[start:start + BATCH_SIZE]

            # from_numpy avoids one copy vs torch.tensor; non_blocking
            # is marginal without pinned memory but doesn't hurt
            xb = torch.from_numpy(X_trn_np[idx]).to(
                device, non_blocking=True)
            yb = torch.from_numpy(y_trn_np[idx]).to(
                device, non_blocking=True)

            # Bug #7: OHEM tracks per-sample soft CE (not KL) for unbiased
            # hardness scores — avoids per-sample entropy constant in KL
            if use_ohem and epoch % 5 == 0 and batch_i % 8 == 0:
                net.eval()
                with torch.no_grad():
                    clean_logp = net(xb)   # log-softmax output
                    # soft CE = -sum(y * log(p)) per sample
                    per_sample = -(yb * clean_logp).sum(dim=1)
                    np_losses = per_sample.cpu().float().numpy()
                    old_mu = sample_loss_mu[idx].copy()
                    sample_loss_mu[idx] = (OHEM_EMA_DECAY * old_mu +
                                           (1 - OHEM_EMA_DECAY) * np_losses)
                    diff  = np_losses - old_mu
                    diff2 = np_losses - sample_loss_mu[idx]
                    sample_loss_var[idx] = (
                        OHEM_EMA_DECAY * sample_loss_var[idx] +
                        (1 - OHEM_EMA_DECAY) * diff * diff2)
                net.train()

            # Mixup (after clean loss tracking)
            if use_mixup and rng.random() < MIXUP_PROB:
                xb, yb = apply_mixup(xb, yb, MIXUP_ALPHA)

            if has_bn and xb.size(0) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=use_amp,
                                    dtype=torch.float16):
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

        # Validate (chunked CPU→GPU to avoid OOM on large cities)
        net.eval()
        val_X_np = test_arrays[val_name]["X"]
        val_y_np = test_arrays[val_name]["y_norm"]
        val_losses_chunks = []
        with torch.no_grad():
            for vs in range(0, val_X_np.shape[0], 131072):
                vxb = torch.from_numpy(val_X_np[vs:vs+131072]).to(device)
                vyb = torch.from_numpy(val_y_np[vs:vs+131072]).to(device)
                with torch.amp.autocast(amp_device, enabled=use_amp,
                                        dtype=torch.float16):
                    val_losses_chunks.append(
                        soft_cross_entropy(net(vxb), vyb).item())
                del vxb, vyb
        val_loss = float(np.mean(val_losses_chunks))

        # Hard-city curriculum update
        if use_hard_cities and epoch % 5 == 0:
            for city_ci, (c_start, c_end) in enumerate(city_boundaries):
                city_mean_loss = sample_loss_mu[c_start:c_end].mean()
                city_losses[city_ci] = (
                    CITY_LOSS_EMA_DECAY * city_losses[city_ci] +
                    (1 - CITY_LOSS_EMA_DECAY) * city_mean_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone()
                         for k, v in net.state_dict().items()}
            wait = 0
            best_epoch = epoch
        else:
            wait += 1

        marker = " *BEST*" if improved else ""
        if epoch <= 3 or epoch % 20 == 0 or improved:
            extra = ""
            if use_rarity and epoch >= RARITY_WARMUP:
                rs = min(1.0, (epoch - RARITY_WARMUP) / 20.0)
                extra += f" rarity_eta={rs * RARITY_MAX_ETA:.2f}"
            print(f"    Ep {epoch:3d}: train={avg_train:.5f} "
                  f"val={val_loss:.5f} wait={wait}{marker}{extra}")

        if epoch >= min_epochs and wait >= patience_epochs:
            print(f"    Early stop at epoch {epoch} "
                  f"(patience={patience_epochs})")
            break

    elapsed = time.time() - t0
    n_epochs_done = epoch + 1

    # Load best
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # Evaluate on ALL test cities (chunked to avoid OOM)
    city_metrics = {}
    for test_name, tdata in test_arrays.items():
        X_np = tdata["X"]
        all_preds = []
        with torch.no_grad():
            for es in range(0, X_np.shape[0], 131072):
                chunk = torch.from_numpy(X_np[es:es+131072]).to(device)
                all_preds.append(net.predict(chunk).cpu().numpy())
                del chunk
        preds = np.concatenate(all_preds, axis=0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        y_true = tdata["y_raw"]
        mae = float(mean_absolute_error(y_true, preds) * 100)
        per_class = {}
        for i, cn in enumerate(CLASS_NAMES):
            yt = y_true[:, i]
            if np.var(yt) < 1e-12:
                per_class[cn] = float("nan")
            else:
                per_class[cn] = float(r2_score(yt, preds[:, i]))
        # Mean R2 only over classes with actual variance (skip NaN)
        valid_r2s = [v for v in per_class.values() if not np.isnan(v)]
        r2 = float(np.mean(valid_r2s)) if valid_r2s else float("nan")
        city_metrics[test_name] = {
            "r2": r2, "mae_pp": mae, "per_class_r2": per_class
        }

    # Print results
    print(f"\n  >> {cfg_name}: {n_params:,} params, "
          f"best@ep{best_epoch}, {elapsed:.0f}s")
    for test_name, m in city_metrics.items():
        print(f"     {test_name:15s}  R2={m['r2']:.4f}  "
              f"MAE={m['mae_pp']:.2f}pp")
        for cn in CLASS_NAMES:
            print(f"       {cn:20s} R2={m['per_class_r2'][cn]:.4f}")

    mean_r2 = float(np.nanmean([m["r2"] for m in city_metrics.values()]))
    mean_mae = float(np.nanmean([m["mae_pp"] for m in city_metrics.values()]))

    # Prevalence-weighted R2 (fairer for rare classes)
    weighted_r2_cities = []
    for cname, m in city_metrics.items():
        pc = m["per_class_r2"]
        vals = np.array([pc[cn] for cn in CLASS_NAMES])
        valid = ~np.isnan(vals)
        w = class_prevalence.copy()
        w[~valid] = 0
        if w.sum() > 0:
            w = w / w.sum()
            weighted_r2_cities.append(float(np.nansum(vals * w)))
    weighted_r2 = float(np.mean(weighted_r2_cities)) if weighted_r2_cities else float("nan")

    result = {
        "name": cfg_name,
        "type": cfg_type,
        "mode": mode,
        "widths": (widths_list if cfg_type == "T"
                   else [shape_params[1]] * shape_params[0]),
        "n_params": n_params,
        "dropout": dropout,
        "wd": wd,
        "n_epochs": n_epochs_done,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "time_s": round(elapsed, 1),
        "mean_r2": float(mean_r2),
        "weighted_r2": weighted_r2,
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
print(f"V6 SAR SWEEP COMPLETE ({len(CONFIGS)} configs)")
print(f"{'='*90}")

ranked = sorted(results, key=lambda r: r["mean_r2"], reverse=True)

header = (f"{'Rank':>4} {'Config':30s} {'Mode':8s} {'Params':>10} "
          f"{'MeanR2':>8} {'MeanMAE':>8} {'BestEp':>6} {'Time':>6}")
print(header)
print("-" * len(header))
for rank, r in enumerate(ranked, 1):
    print(f"{rank:4d} {r['name']:30s} {r['mode']:8s} "
          f"{r['n_params']:>10,} "
          f"{r['mean_r2']:>8.4f} {r['mean_mae_pp']:>7.2f}pp "
          f"{r['best_epoch']:>5d}  {r['time_s']:>5.0f}s")

# Per-class breakdown for top 5
print(f"\n--- TOP 5 per-class breakdown ---")
for rank, r in enumerate(ranked[:5], 1):
    print(f"\n  #{rank} {r['name']} (R2={r['mean_r2']:.4f}, mode={r['mode']})")
    for city_name, m in r['per_city'].items():
        print(f"    {city_name:15s}  R2={m['r2']:.4f}  "
              f"MAE={m['mae_pp']:.2f}pp")
        for cn in CLASS_NAMES:
            pc = m['per_class_r2'][cn]
            print(f"      {cn:20s} R2={pc:.4f}")

# Save CSV
csv_rows = []
for r in ranked:
    row = {
        "rank": ranked.index(r) + 1,
        "name": r["name"],
        "mode": r["mode"],
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

reports_dir = os.path.join(PROJECT_ROOT, "reports", "v6_sweep")
os.makedirs(reports_dir, exist_ok=True)
df_summary.to_csv(os.path.join(reports_dir, f"sweep_{RUN_ID}.csv"),
                   index=False)

best = ranked[0]
print(f"\n{'='*90}")
print(f"BEST CONFIG: {best['name']} (mode={best['mode']})")
print(f"  Mean R2 = {best['mean_r2']:.4f}, Mean MAE = {best['mean_mae_pp']:.2f}pp")
print(f"  Params: {best['n_params']:,}, Best epoch: {best['best_epoch']}")
print(f"  SAR temporal features included: {len(sar_temporal)}")
print(f"\nAll results saved to: {OUT_DIR}")
print(f"{'='*90}")
