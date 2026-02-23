#!/usr/bin/env python3
"""
V9 MLP Sweep — Self-contained, no exec/monkey-patching bugs.

Key improvements over V8:
  - Label denoising: sweep thresholds 3/5/10% (zero micro-classes, renormalize)
  - 15 diverse val cities across all climate zones (on VRAM, not CPU)
  - No unseen-city testing during sweep (saves RAM; test after)
  - Output hardcoded to models_v9_sweep/<threshold>/ (no v6 output-dir bug)
  - SWA (Stochastic Weight Averaging) on all configs — free generalization boost
  - SAM (Sharpness-Aware Minimization) variants for top architectures
  - 14 configs: 11 mixup-only + 3 SAM variants

Usage:
    .venv/Scripts/python.exe scripts/sweep_mlp_v9.py
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
    CITIES, SEED, CLASS_NAMES, N_CLASSES,
    city_features_dir, city_labels_path,
    _discover_feature_cols, _load_city_arrays, build_bi_lbp, ts,
)
try:
    from scripts.run_multi_city_pipeline_v5 import CONTROL_COLS
except ImportError:
    CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                    "reflectance_scale", "full_features_computed"}
from scripts.run_mlp_overnight_v4 import (
    _make_norm, PlainBlock, normalize_targets, soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pyarrow.parquet as pq

# =====================================================================
# V9 city splits — 15 diverse val cities, rest = train
# =====================================================================

VAL_CITY_NAMES = {
    # Central European
    'munich',               # built_up 40%, reference city
    'vojvodina_cropland',   # cropland 52%
    'hortobagy_puszta',     # grassland 80%, steppe
    # Mediterranean / Southern
    'seville',              # built_up 30%, Mediterranean urban
    'crete_phrygana',       # grassland+shrub, coastal Mediterranean
    'tabernas_desert',      # bare_sparse 38%, arid
    'sardinia_maquis',      # tree_cover 72%, island maquis
    'camargue_wetland',     # water 64%, wetland
    'pyrenees_meadows',     # grassland 44% + bare 26%, mountain
    # NW European
    'ireland_bog_pasture',  # grassland 77%
    'danish_farmland',      # cropland 50%, northern agriculture
    # Nordic / Arctic
    'stockholm',            # tree_cover 48%, Nordic urban
    'finnish_lakeland',     # tree_cover 72%, boreal forest/lake
    'iceland_highlands',    # bare_sparse 67%, extreme landscape
    'lapland_tundra',       # tree_cover 72%, sub-arctic boreal
    # NOTE: corsica_interior kept in TRAIN (highest shrubland 19.5%)
}

# Nuremberg excluded from everything (unseen test)
EXCLUDED_CITY_NAMES = {'nuremberg'}

# Override features dir to use features_v7/ (same data as v8, no duplication)
# MUST monkey-patch v5_mod so _load_city_arrays() picks it up
import scripts.run_multi_city_pipeline_v5 as v5_mod

def _city_features_dir_v9(city):
    """Point to features_v7/ (reuse V7 data)."""
    return os.path.join(v5_mod.city_dir(city), "features_v7")

v5_mod.city_features_dir = _city_features_dir_v9

# =====================================================================
# Feature path helpers
# =====================================================================

def _city_parquet_path(city):
    return os.path.join(_city_features_dir_v9(city),
                        "features_rust_2020_2021.parquet")


def _city_feature_cols(city):
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
    cols = _city_feature_cols(city)
    if cols is None:
        return False
    return any(c.startswith("SAR_") for c in cols)


# =====================================================================
# Model definitions (same as v6/v8 — reproducing here for clarity)
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
# SAM Optimizer — Sharpness-Aware Minimization
# =====================================================================

class SAM:
    """Lightweight SAM wrapper around any base optimizer."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        # Force fused=False — fused AdamW conflicts with SAM perturbation
        kwargs.pop('fused', None)
        self.base = base_optimizer_cls(params, **kwargs)
        self.rho = rho
        self.param_groups = self.base.param_groups
        self.state = self.base.state
        self._e_w = {}  # perturbation storage (avoids state init issues)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)   # ascent step
                # Store perturbation in our own dict (not base.state
                # which may not be initialized yet)
                self._e_w[p] = e_w.clone()

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self._e_w[p])  # back to original
        self.base.step()
        self._e_w.clear()

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)


# =====================================================================
# Architecture configs — 11 mixup + 3 SAM variants = 14 configs
# =====================================================================
# Format: (name, type, shape, dropout, wd, use_sam)
#   Trimmed: T_1024_512_256_128 (redundant with T_1024_512_256_64)
#            T_2048_1024_512 (middle child, keep 4L and 5L)
#            C_4x2048 (too similar to C_5x2048)

CONFIGS = [
    # --- Standard mixup configs ---
    ("T_1024_512_256_64",       "T", [1024, 512, 256, 64],       0.15, 2e-4, False),
    ("T_1024_512_128",          "T", [1024, 512, 128],            0.12, 2e-4, False),
    ("T_1024_512_256_128",      "T", [1024, 512, 256, 128],       0.15, 2e-4, False),
    ("T_768_256_64",            "T", [768, 256, 64],              0.12, 2e-4, False),
    ("T_512_256_128_64",        "T", [512, 256, 128, 64],         0.10, 1e-4, False),
    ("T_2048_512_128",          "T", [2048, 512, 128],            0.15, 3e-4, False),
    ("T_2048_1024_256_64",      "T", [2048, 1024, 256, 64],       0.18, 3e-4, False),
    ("T_2048_1024_512_256",     "T", [2048, 1024, 512, 256],      0.25, 1e-3, False),
    ("T_2048_1024_512_256_128", "T", [2048, 1024, 512, 256, 128], 0.25, 1e-3, False),
    ("T_3072_1536_768_384",     "T", [3072, 1536, 768, 384],      0.30, 1e-3, False),
    ("C_5x2048",                "C", (5, 2048),                   0.30, 1e-3, False),
    ("C_5x1024",                "C", (5, 1024),                   0.20, 5e-4, False),
    # --- SAM variants (seek flat minima, ~2x train time) ---
    ("T_1024_512_256_64_SAM",   "T", [1024, 512, 256, 64],       0.15, 2e-4, True),
    ("T_2048_1024_256_64_SAM",  "T", [2048, 1024, 256, 64],       0.18, 3e-4, True),
    ("C_5x1024_SAM",            "C", (5, 1024),                   0.20, 5e-4, True),
]

# =====================================================================
# Training constants
# =====================================================================
BATCH_SIZE = 4096
PATIENCE_STEPS = 10000
MIN_STEPS = 3000
MAX_EPOCHS = 500
SEED_OFFSET = 900  # distinct from v6 (600), v7, v8
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

# Label thresholds to sweep (dropped 8% — too close to 5% and 10%)
LABEL_THRESHOLDS = [0.03, 0.05, 0.10]

# SWA config
SWA_START_FRAC = 0.75  # collect SWA weights in last 25% of training
SAM_RHO = 0.05         # SAM perturbation radius

# Output directory
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
V9_SWEEP_DIR = os.path.join(CITIES_DIR, "models_v9_sweep")
os.makedirs(V9_SWEEP_DIR, exist_ok=True)

# =====================================================================
# City filtering — SAR features required
# =====================================================================

print(f"[{ts()}] V9 MLP Sweep — filtering cities with SAR features...")

ALL_TRAIN = []
ALL_VAL = []
for c in CITIES:
    if c.name in EXCLUDED_CITY_NAMES:
        continue
    if c.name in VAL_CITY_NAMES:
        ALL_VAL.append(c)
    else:
        ALL_TRAIN.append(c)

# Only keep cities with SAR features
TRAIN_CITIES = [c for c in ALL_TRAIN if _city_has_sar(c)]
VAL_CITIES = [c for c in ALL_VAL if _city_has_sar(c)]

print(f"  Train cities: {len(TRAIN_CITIES)} / {len(ALL_TRAIN)} (with SAR)")
print(f"  Val cities:   {len(ALL_VAL)} -> {len(VAL_CITIES)} (with SAR)")
print(f"  Excluded:     {EXCLUDED_CITY_NAMES}")

if not TRAIN_CITIES:
    raise RuntimeError("No training cities with SAR features found!")
if not VAL_CITIES:
    raise RuntimeError("No validation cities with SAR features found!")

print(f"  Train: {[c.name for c in TRAIN_CITIES]}")
print(f"  Val:   {[c.name for c in VAL_CITIES]}")

# Build feature columns as INTERSECTION across all cities
print(f"\n[{ts()}] Building feature column intersection...")
all_city_col_sets = []
for city in TRAIN_CITIES + VAL_CITIES:
    cols = _city_feature_cols(city)
    if cols is not None:
        all_city_col_sets.append(cols)
        print(f"  [{city.name}] {len(cols)} columns")

common_cols = set.intersection(*all_city_col_sets) if all_city_col_sets else set()
first_schema = pq.read_schema(_city_parquet_path(TRAIN_CITIES[0]))
full_feature_cols = [f.name for f in first_schema if f.name in common_cols]

mlp_idx = build_bi_lbp(full_feature_cols)
mlp_cols = [full_feature_cols[i] for i in mlp_idx]
n_features = len(mlp_cols)

sar_temporal = [c for c in mlp_cols if "temporal" in c or "summer_winter" in c]
print(f"  Common columns: {len(full_feature_cols)}")
print(f"  MLP features: {n_features}")
print(f"  SAR temporal: {len(sar_temporal)}")


# =====================================================================
# Data loading — train on CPU, val on VRAM
# =====================================================================

def load_city_data(city, threshold=0.0):
    """Load features + labels for a city, apply label threshold."""
    result = _load_city_arrays(city, mlp_cols)
    if result is None:
        return None, None

    X_city, _ = result
    label_path = city_labels_path(city, 2021)
    if not os.path.exists(label_path):
        del X_city
        return None, None

    labels = pd.read_parquet(label_path)
    y_city = labels[CLASS_NAMES].values.astype(np.float32)
    del labels

    # Normalize to sum to 1.0; drop zero-coverage cells
    row_sums = y_city.sum(axis=1, keepdims=True)
    valid = (row_sums.ravel() > 0)
    if not valid.all():
        X_city = X_city[valid]
        y_city = y_city[valid]
        row_sums = row_sums[valid]
    y_city = y_city / np.maximum(row_sums, 1e-8)

    # Label denoising: zero out micro-classes below threshold, renormalize
    if threshold > 0:
        y_city[y_city < threshold] = 0.0
        row_sums2 = y_city.sum(axis=1, keepdims=True)
        valid2 = (row_sums2.ravel() > 0)
        if not valid2.all():
            X_city = X_city[valid2]
            y_city = y_city[valid2]
            row_sums2 = row_sums2[valid2]
        y_city = y_city / np.maximum(row_sums2, 1e-8)

    # Align lengths
    n = min(X_city.shape[0], y_city.shape[0])
    return X_city[:n], y_city[:n]


def apply_mixup(xb, yb, alpha):
    """Inter-city Mixup: Beta(alpha, alpha) blending."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]


# =====================================================================
# Run one threshold experiment
# =====================================================================

def run_threshold_sweep(threshold, device):
    """Run all 14 configs for one label threshold. Returns results list."""
    thresh_str = f"{int(threshold * 100):02d}"
    out_dir = os.path.join(V9_SWEEP_DIR, f"thresh_{thresh_str}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*90}")
    print(f"  THRESHOLD = {threshold:.0%} — Output: {out_dir}")
    print(f"{'#'*90}")

    # ---- Load training data ----
    print(f"\n[{ts()}] Loading training data (threshold={threshold:.0%})...")
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

    # Pre-allocate and fill
    total_est = sum(city_counts)
    X_train = np.empty((total_est, n_features), dtype=np.float32)
    y_train = np.empty((total_est, N_CLASSES), dtype=np.float32)
    city_boundaries = []
    city_ids = np.empty(total_est, dtype=np.int32)
    offset = 0
    cities_loaded = []

    for ci, (city, n) in enumerate(zip(train_cities_valid, city_counts)):
        X_city, y_city = load_city_data(city, threshold)
        if X_city is None:
            print(f"  WARNING: {city.name} returned None -- skip")
            continue

        actual_n = min(n, X_city.shape[0])
        X_train[offset:offset + actual_n] = X_city[:actual_n]
        y_train[offset:offset + actual_n] = y_city[:actual_n]
        city_ids[offset:offset + actual_n] = len(city_boundaries)
        city_boundaries.append((offset, offset + actual_n))
        cities_loaded.append(city)
        del X_city, y_city
        offset += actual_n
        gc.collect()

    # Trim arrays
    if offset < total_est:
        print(f"  Trimming: {total_est:,} -> {offset:,} ({total_est - offset:,} skipped)")
        X_train = X_train[:offset]
        y_train = y_train[:offset]
        city_ids = city_ids[:offset]
    total = offset

    n_cities = len(city_boundaries)
    print(f"  Loaded: {total:,} samples, {n_cities} cities")

    # ---- Scale training data ----
    print(f"\n[{ts()}] Fitting scaler...")
    scaler = StandardScaler()
    SCALER_CHUNK = 200_000
    for sc_start in range(0, total, SCALER_CHUNK):
        scaler.partial_fit(X_train[sc_start:sc_start + SCALER_CHUNK])
    scaler_mean = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)
    X_train -= scaler_mean
    X_train /= scaler_scale
    print(f"  Scaled {total:,} x {n_features} in-place")

    # Save scaler + cols
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)

    # ---- Load val data -> VRAM ----
    print(f"\n[{ts()}] Loading val cities -> VRAM...")
    val_tensors = {}
    val_vram_bytes = 0
    for city in VAL_CITIES:
        X_val, y_val = load_city_data(city, threshold)
        if X_val is None:
            print(f"  WARNING: Missing val {city.name} -- skip")
            continue
        # Scale
        X_val -= scaler_mean
        X_val /= scaler_scale
        y_val_norm = normalize_targets(y_val)

        # Move to VRAM
        X_t = torch.tensor(X_val, device=device)
        y_norm_t = torch.tensor(y_val_norm, device=device)
        y_raw_np = y_val.copy()  # keep on CPU for R2/MAE

        vram_used = (X_t.nbytes + y_norm_t.nbytes) / 1e6
        val_vram_bytes += X_t.nbytes + y_norm_t.nbytes
        val_tensors[city.name] = {
            "X": X_t,             # GPU tensor
            "y_norm": y_norm_t,   # GPU tensor
            "y_raw": y_raw_np,    # CPU numpy
        }
        print(f"  [{city.name}] {X_val.shape[0]:,} cells -> VRAM ({vram_used:.1f} MB)")
        del X_val, y_val, y_val_norm

    print(f"  Total val VRAM: {val_vram_bytes / 1e9:.2f} GB")
    gc.collect()

    # Rarity weights
    class_prevalence = y_train.mean(axis=0)
    print(f"\n  Class prevalence (after {threshold:.0%} threshold):")
    for k, cn in enumerate(CLASS_NAMES):
        print(f"    {cn:20s}: {class_prevalence[k]:.4f}")

    rarity_per_class = (class_prevalence + 1e-6) ** (-RARITY_ALPHA)
    rarity_per_class /= rarity_per_class.max()

    sample_rarity = y_train @ rarity_per_class
    sample_rarity /= sample_rarity.sum()

    y_norm = normalize_targets(y_train)
    del y_train
    gc.collect()

    # Per-sample loss trackers
    sample_loss_mu = np.ones(total, dtype=np.float32)
    sample_loss_var = np.zeros(total, dtype=np.float32)
    city_losses = np.ones(n_cities, dtype=np.float32)

    X_trn_np = X_train
    y_trn_np = y_norm
    del y_norm

    use_amp = device == "cuda"
    amp_device = "cuda" if device == "cuda" else "cpu"

    # Early-stop city selection
    val_name = list(val_tensors.keys())[0]
    for preferred in ["munich", "seville", "stockholm"]:
        if preferred in val_tensors:
            val_name = preferred
            break
    print(f"  Early-stop city: {val_name} "
          f"({val_tensors[val_name]['X'].shape[0]:,} cells, VRAM)")

    # Epoch geometry
    city_sizes = [end - start for start, end in city_boundaries]
    samples_per_city = int(np.median(city_sizes))
    n_balanced = samples_per_city * n_cities
    steps_per_epoch = (n_balanced + BATCH_SIZE - 1) // BATCH_SIZE
    patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
    min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)

    print(f"\n  Balanced epoch: ~{n_balanced:,} samples")
    print(f"  Steps/epoch: ~{steps_per_epoch}, patience: {patience_epochs}ep")

    # =====================================================================
    # Hierarchical sampler
    # =====================================================================
    def build_epoch_indices(rng, epoch):
        city_sizes_local = [end - start for start, end in city_boundaries]
        spc = int(np.median(city_sizes_local))

        # Hard-city curriculum
        if epoch >= OHEM_WARMUP:
            logits = city_losses / CITY_TEMPERATURE
            logits -= logits.max()
            exp_l = np.exp(logits)
            sm = exp_l / exp_l.sum()
            uniform = np.ones(n_cities) / n_cities
            city_probs = (1 - CITY_HARDNESS_ALPHA) * uniform + CITY_HARDNESS_ALPHA * sm
        else:
            city_probs = np.ones(n_cities) / n_cities

        city_sample_counts = (city_probs * n_cities * spc).astype(int)
        city_sample_counts = np.maximum(city_sample_counts, 1)

        # Rarity + OHEM within city
        rarity_strength = min(1.0, max(0, (epoch - RARITY_WARMUP)) / 20.0)
        eta = RARITY_MAX_ETA * rarity_strength if epoch >= RARITY_WARMUP else 0.0

        all_indices = []
        for ci, (start, end) in enumerate(city_boundaries):
            city_n = end - start
            n_to_sample = city_sample_counts[ci]
            use_replace = n_to_sample > city_n
            uniform = np.ones(city_n, dtype=np.float64) / city_n

            if eta > 0 or epoch >= OHEM_WARMUP:
                city_rarity = sample_rarity[start:end].astype(np.float64)
                city_rarity /= (city_rarity.sum() + 1e-12)

                if epoch >= OHEM_WARMUP:
                    mu = sample_loss_mu[start:end].astype(np.float64)
                    var = np.maximum(sample_loss_var[start:end].astype(np.float64), 0.0)
                    std = np.sqrt(var + 1e-8)
                    hardness = np.maximum(1e-8, mu - OHEM_VAR_LAMBDA * std)
                    hardness /= (hardness.sum() + 1e-12)
                    combined = 0.6 * city_rarity + 0.4 * hardness
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

        target = spc * n_cities
        if len(perm) > target:
            perm = perm[:target]
        elif len(perm) < target:
            extra = rng.choice(perm, size=target - len(perm), replace=True)
            perm = np.concatenate([perm, extra])
            rng.shuffle(perm)
        return perm

    # =====================================================================
    # Sweep loop
    # =====================================================================
    results = []
    print(f"\n[{ts()}] Starting sweep — {len(CONFIGS)} configs, threshold={threshold:.0%}")

    for ci, (cfg_name, cfg_type, shape_params, dropout, wd, use_sam) in enumerate(CONFIGS):
        print(f"\n{'='*70}")
        sam_tag = ' [SAM]' if use_sam else ''
        print(f"  [{ci+1}/{len(CONFIGS)}] {cfg_name}{sam_tag}  (threshold={threshold:.0%})")

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
        print(f"  Shape: {shape_str}")
        print(f"  Params: {n_params:,} | dropout={dropout} wd={wd} SAM={use_sam}")

        # Reset trackers
        sample_loss_mu[:] = 1.0
        sample_loss_var[:] = 0.0
        city_losses[:] = 1.0

        # Optimizer: SAM wraps AdamW, or plain AdamW
        if use_sam:
            try:
                optimizer = SAM(net.parameters(), torch.optim.AdamW,
                                rho=SAM_RHO, lr=LR, weight_decay=wd, fused=use_amp)
            except TypeError:
                optimizer = SAM(net.parameters(), torch.optim.AdamW,
                                rho=SAM_RHO, lr=LR, weight_decay=wd)
        else:
            try:
                optimizer = torch.optim.AdamW(
                    net.parameters(), lr=LR, weight_decay=wd, fused=use_amp)
            except TypeError:
                optimizer = torch.optim.AdamW(
                    net.parameters(), lr=LR, weight_decay=wd)

        # SAM doesn't support GradScaler well; disable AMP for SAM
        use_amp_this = use_amp and (not use_sam)
        scaler_amp = torch.amp.GradScaler(enabled=use_amp_this)
        total_steps = MAX_EPOCHS * steps_per_epoch
        warmup_steps = steps_per_epoch * 3
        base_opt = optimizer.base if use_sam else optimizer
        scheduler = torch.optim.lr_scheduler.SequentialLR(base_opt, [
            torch.optim.lr_scheduler.LinearLR(
                base_opt, start_factor=0.01, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                base_opt, T_max=total_steps - warmup_steps),
        ], milestones=[warmup_steps])

        # SWA state
        swa_state = None
        swa_count = 0
        swa_start_epoch = int(MAX_EPOCHS * SWA_START_FRAC)

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

            perm = build_epoch_indices(rng, epoch)

            for batch_i, start_idx in enumerate(range(0, len(perm), BATCH_SIZE)):
                idx = perm[start_idx:start_idx + BATCH_SIZE]
                xb = torch.from_numpy(X_trn_np[idx]).to(device, non_blocking=True)
                yb = torch.from_numpy(y_trn_np[idx]).to(device, non_blocking=True)

                # OHEM tracking
                if epoch % 5 == 0 and batch_i % 8 == 0:
                    net.eval()
                    with torch.no_grad():
                        clean_logp = net(xb)
                        per_sample = -(yb * clean_logp).sum(dim=1)
                        np_losses = per_sample.cpu().float().numpy()
                        old_mu = sample_loss_mu[idx].copy()
                        sample_loss_mu[idx] = (OHEM_EMA_DECAY * old_mu +
                                               (1 - OHEM_EMA_DECAY) * np_losses)
                        diff = np_losses - old_mu
                        diff2 = np_losses - sample_loss_mu[idx]
                        sample_loss_var[idx] = (
                            OHEM_EMA_DECAY * sample_loss_var[idx] +
                            (1 - OHEM_EMA_DECAY) * diff * diff2)
                    net.train()

                # Mixup
                if rng.random() < MIXUP_PROB:
                    xb, yb = apply_mixup(xb, yb, MIXUP_ALPHA)

                if has_bn and xb.size(0) < 2:
                    continue

                if use_sam:
                    # SAM: two forward-backward passes
                    optimizer.zero_grad(set_to_none=True)
                    pred1 = net(xb)
                    loss1 = soft_cross_entropy(pred1, yb)
                    loss1.backward()
                    optimizer.first_step()  # ascent

                    optimizer.zero_grad(set_to_none=True)
                    pred2 = net(xb)
                    loss2 = soft_cross_entropy(pred2, yb)
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.second_step()  # descent from perturbed point
                    scheduler.step()
                    epoch_loss += loss1.item()
                else:
                    # Standard: single forward-backward
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(amp_device, enabled=use_amp_this,
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

            # Validate on VRAM (no CPU->GPU transfer needed!)
            net.eval()

            # SWA: accumulate weights after swa_start_epoch
            if epoch >= swa_start_epoch:
                with torch.no_grad():
                    if swa_state is None:
                        swa_state = {k: v.cpu().clone().float()
                                     for k, v in net.state_dict().items()}
                        swa_count = 1
                    else:
                        for k, v in net.state_dict().items():
                            swa_state[k].add_(v.cpu().float())
                        swa_count += 1

            val_X = val_tensors[val_name]["X"]       # already on GPU
            val_y = val_tensors[val_name]["y_norm"]   # already on GPU
            val_losses = []
            with torch.no_grad():
                for vs in range(0, val_X.shape[0], 131072):
                    vxb = val_X[vs:vs+131072]
                    vyb = val_y[vs:vs+131072]
                    with torch.amp.autocast(amp_device, enabled=use_amp_this,
                                            dtype=torch.float16):
                        val_losses.append(
                            soft_cross_entropy(net(vxb), vyb).item())
            val_loss = float(np.mean(val_losses))

            # Hard-city curriculum update
            if epoch % 5 == 0:
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

            if epoch <= 3 or epoch % 20 == 0 or improved:
                marker = " *BEST*" if improved else ""
                print(f"    Ep {epoch:3d}: train={avg_train:.5f} "
                      f"val={val_loss:.5f} wait={wait}{marker}")

            if epoch >= min_epochs and wait >= patience_epochs:
                print(f"    Early stop at epoch {epoch} "
                      f"(patience={patience_epochs})")
                break

        elapsed = time.time() - t0
        n_epochs_done = epoch + 1

        # Use SWA weights if available, else best checkpoint
        if swa_state is not None and swa_count > 1:
            swa_avg = {k: (v / swa_count) for k, v in swa_state.items()}
            net.load_state_dict({k: v.to(device) for k, v in swa_avg.items()})
            # Update BatchNorm stats with SWA weights
            if has_bn:
                torch.optim.swa_utils.update_bn(
                    torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(
                            torch.from_numpy(X_trn_np[:min(200000, total)])),
                        batch_size=BATCH_SIZE, shuffle=True),
                    net, device=device)
            print(f"    SWA: averaged {swa_count} checkpoints")
        elif best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        # Evaluate on ALL val cities (already on VRAM)
        city_metrics = {}
        for vname, vdata in val_tensors.items():
            all_preds = []
            with torch.no_grad():
                for es in range(0, vdata["X"].shape[0], 131072):
                    chunk = vdata["X"][es:es+131072]
                    all_preds.append(net.predict(chunk).cpu().numpy())
            preds = np.concatenate(all_preds, axis=0)

            y_true = vdata["y_raw"]
            mae = float(mean_absolute_error(y_true, preds) * 100)
            per_class = {}
            for i, cn in enumerate(CLASS_NAMES):
                yt = y_true[:, i]
                if np.var(yt) < 1e-4:  # skip degenerate classes
                    per_class[cn] = float("nan")
                else:
                    per_class[cn] = float(r2_score(yt, preds[:, i]))
            valid_r2s = [v for v in per_class.values() if not np.isnan(v)]
            r2 = float(np.mean(valid_r2s)) if valid_r2s else float("nan")

            # Per-city prevalence-weighted R2 (uses city's own class distribution)
            city_prev = y_true.mean(axis=0)
            vals = np.array([per_class[cn] for cn in CLASS_NAMES])
            vmask = ~np.isnan(vals)
            w_city = city_prev.copy()
            w_city[~vmask] = 0
            if w_city.sum() > 0:
                w_city = w_city / w_city.sum()
                city_wr2 = float(np.nansum(vals * w_city))
            else:
                city_wr2 = float("nan")

            city_metrics[vname] = {
                "r2": r2, "weighted_r2": city_wr2,
                "mae_pp": mae, "per_class_r2": per_class
            }

        # Print results — weighted R2 as primary metric
        print(f"\n  >> {cfg_name}: {n_params:,} params, "
              f"best@ep{best_epoch}, {elapsed:.0f}s")
        for vname, m in city_metrics.items():
            print(f"     {vname:25s}  wR2={m['weighted_r2']:.4f}  "
                  f"R2={m['r2']:.4f}  MAE={m['mae_pp']:.2f}pp")

        mean_r2 = float(np.nanmean([m["r2"] for m in city_metrics.values()]))
        mean_mae = float(np.nanmean([m["mae_pp"] for m in city_metrics.values()]))

        # Overall weighted R2: average of per-city weighted R2s
        city_wr2s = [m["weighted_r2"] for m in city_metrics.values()
                     if not np.isnan(m["weighted_r2"])]
        weighted_r2 = float(np.mean(city_wr2s)) if city_wr2s else float("nan")

        # Save final state (SWA or best)
        final_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        result = {
            "name": cfg_name,
            "type": cfg_type,
            "threshold": threshold,
            "use_sam": use_sam,
            "swa_epochs": swa_count,
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
        torch.save(final_state, os.path.join(out_dir, f"{cfg_name}.pt"))
        with open(os.path.join(out_dir, "sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        del net, optimizer, scaler_amp, scheduler, best_state, final_state
        del swa_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary for this threshold ----
    print(f"\n\n{'='*90}")
    print(f"THRESHOLD {threshold:.0%} COMPLETE ({len(CONFIGS)} configs)")
    print(f"{'='*90}")

    ranked = sorted(results, key=lambda r: r["mean_r2"], reverse=True)
    header = (f"{'Rank':>4} {'Config':30s} {'Params':>10} "
              f"{'MeanR2':>8} {'MeanMAE':>8} {'BestEp':>6} {'Time':>6}")
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:4d} {r['name']:30s} "
              f"{r['n_params']:>10,} "
              f"{r['mean_r2']:>8.4f} {r['mean_mae_pp']:>7.2f}pp "
              f"{r['best_epoch']:>5d}  {r['time_s']:>5.0f}s")

    best = ranked[0]
    print(f"\nBEST: {best['name']} (R2={best['mean_r2']:.4f})")

    # CSV summary
    csv_rows = []
    for r in ranked:
        row = {
            "rank": ranked.index(r) + 1,
            "name": r["name"],
            "threshold": r["threshold"],
            "n_params": r["n_params"],
            "mean_r2": r["mean_r2"],
            "weighted_r2": r["weighted_r2"],
            "mean_mae_pp": r["mean_mae_pp"],
            "best_epoch": r["best_epoch"],
            "time_s": r["time_s"],
        }
        for vn in [c.name for c in VAL_CITIES]:
            if vn in r["per_city"]:
                row[f"r2_{vn}"] = r["per_city"][vn]["r2"]
                row[f"mae_{vn}"] = r["per_city"][vn]["mae_pp"]
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)

    # Free train data for this threshold
    del X_train, X_trn_np, y_trn_np, sample_rarity, sample_loss_mu, sample_loss_var
    del city_losses, city_ids
    for vd in val_tensors.values():
        del vd["X"], vd["y_norm"]
    del val_tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{ts()}] V9 MLP Sweep starting")
    print(f"  Device: {device}")
    print(f"  Thresholds: {LABEL_THRESHOLDS}")
    print(f"  Configs: {len(CONFIGS)}")
    print(f"  Total runs: {len(CONFIGS) * len(LABEL_THRESHOLDS)}")
    print(f"  Output: {V9_SWEEP_DIR}")

    all_results = {}
    for threshold in LABEL_THRESHOLDS:
        results = run_threshold_sweep(threshold, device)
        all_results[f"thresh_{int(threshold*100):02d}"] = results

    # Final cross-threshold summary
    print(f"\n\n{'='*90}")
    print(f"V9 SWEEP COMPLETE — ALL THRESHOLDS")
    print(f"{'='*90}")
    for thresh_key, results in all_results.items():
        best = max(results, key=lambda r: r["mean_r2"])
        print(f"  {thresh_key}: best={best['name']} R2={best['mean_r2']:.4f} "
              f"MAE={best['mean_mae_pp']:.2f}pp")

    # Save cross-threshold summary
    with open(os.path.join(V9_SWEEP_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {V9_SWEEP_DIR}")
    print(f"{'='*90}")
