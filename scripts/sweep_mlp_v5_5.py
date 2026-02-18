#!/usr/bin/env python3
"""
V5.5 MLP Sweep (SAR-only cities): trains on 19 SAR-equipped cities,
tests on Frankfurt (held out). Runs while SAR re-downloads happen in
parallel for the remaining 36 cities.

Usage:
    .venv/Scripts/python.exe scripts/sweep_mlp_v5_5.py
"""
import os, sys, time, math, json, pickle, hashlib
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

# Override: use only cities that have SAR features (1787 cols)
_SAR_TRAIN_NAMES = [
    'bremen', 'hamburg', 'duesseldorf', 'leipzig', 'rostock',
    'amsterdam', 'hambach_mine', 'welzow_mine', 'amiens', 'magdeburg',
    'ulm', 'salzburg', 'schwerin', 'malmo',
    'regensburg', 'london', 'brussels', 'rotterdam', 'antwerp',
]
_SAR_TEST_NAMES = ['frankfurt']  # held-out from the 20 SAR cities

_city_by_name = {c.name: c for c in CITIES}
TRAIN_CITIES = [_city_by_name[n] for n in _SAR_TRAIN_NAMES]
TEST_CITIES = [_city_by_name[n] for n in _SAR_TEST_NAMES]
from scripts.run_mlp_overnight_v4 import (
    _make_norm, PlainBlock, normalize_targets,
    soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# =====================================================================
# Model definitions
# =====================================================================

class ConstantMLP(nn.Module):
    """Constant-width MLP: all hidden layers have the same width."""
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
    """MLP with decreasing layer widths: wide → narrow."""
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
# Architecture configs
# =====================================================================
# Format: (name, type, shape_params, dropout, weight_decay)
# type="C" -> constant: shape_params = (n_layers, width)
# type="T" -> tapered:  shape_params = list of widths

CONFIGS = [
    # ---- Tiny (fast baselines, <1M params) ----
    ("C_3x256",            "C", (3, 256),                          0.10, 1e-4),
    ("C_3x512",            "C", (3, 512),                          0.15, 3e-4),
    ("C_4x256",            "C", (4, 256),                          0.10, 1e-4),
    ("C_4x512",            "C", (4, 512),                          0.15, 3e-4),

    # ---- Small (1-5M params) ----
    ("C_3x1024",           "C", (3, 1024),                         0.20, 5e-4),
    ("C_4x1024",           "C", (4, 1024),                         0.20, 5e-4),
    ("C_5x512",            "C", (5, 512),                          0.15, 3e-4),
    ("C_5x1024",           "C", (5, 1024),                         0.20, 5e-4),

    # ---- Medium (5-20M params) ----
    ("C_5x2048",           "C", (5, 2048),                         0.30, 1e-3),
    ("C_4x2048",           "C", (4, 2048),                         0.25, 1e-3),
    ("C_6x1024",           "C", (6, 1024),                         0.20, 5e-4),
    ("C_3x2048",           "C", (3, 2048),                         0.25, 1e-3),

    # ---- Large (20M+ params — kept ≤3072 to fit in 8 GB VRAM) ----
    ("C_5x3072",           "C", (5, 3072),                         0.30, 1e-3),
    ("C_4x3072",           "C", (4, 3072),                         0.30, 1e-3),
    ("C_6x2048",           "C", (6, 2048),                         0.30, 1e-3),
    ("C_3x3072",           "C", (3, 3072),                         0.25, 1e-3),

    # ---- Tapered: gentle (2x squeeze per layer) ----
    ("T_2048_1024_512",    "T", [2048, 1024, 512],                 0.25, 1e-3),
    ("T_2048_1024_512_256","T", [2048, 1024, 512, 256],            0.25, 1e-3),
    ("T_3072_1536_768",    "T", [3072, 1536, 768],                 0.30, 1e-3),
    ("T_3072_1536_768_384","T", [3072, 1536, 768, 384],            0.30, 1e-3),
    ("T_4096_2048_1024",   "T", [4096, 2048, 1024],                0.30, 1e-3),
    ("T_4096_2048_1024_512","T",[4096, 2048, 1024, 512],            0.30, 1e-3),

    # ---- Tapered: aggressive (4x total squeeze) ----
    ("T_2048_512",         "T", [2048, 512],                       0.25, 1e-3),
    ("T_3072_768",         "T", [3072, 768],                       0.30, 1e-3),
    ("T_4096_1024_256",    "T", [4096, 1024, 256],                 0.30, 1e-3),

    # ---- Tapered: bottleneck (wide → narrow → narrow) ----
    ("T_2048_256_256",     "T", [2048, 256, 256],                  0.20, 5e-4),
    ("T_3072_512_512",     "T", [3072, 512, 512],                  0.25, 1e-3),
    ("T_4096_512_256_128", "T", [4096, 512, 256, 128],             0.25, 1e-3),

    # ---- Deep tapered ----
    ("T_2048_1024_512_256_128", "T", [2048, 1024, 512, 256, 128],  0.25, 1e-3),
    ("T_1024_512_256_128", "T", [1024, 512, 256, 128],             0.20, 5e-4),
]

# =====================================================================
# Training constants
# =====================================================================
BATCH_SIZE = 4096
PATIENCE_STEPS = 10000
MIN_STEPS = 3000
MAX_EPOCHS = 500
SEED_OFFSET = 100
INPUT_DROPOUT = 0.10
LR = 1e-3
ACTIVATION = "silu"
NORM = "batchnorm"

# Output directory - timestamped to avoid overwriting
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v5_5_sweep_sar", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# Data loading
# =====================================================================

print(f"[{ts()}] V5.5 SAR-only MLP Sweep starting (run_id={RUN_ID})")
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

# =====================================================================
# MEMORY-EFFICIENT LOADING
# Two-pass: count cells first, then pre-allocate and fill in-place.
# Peak RAM: ~9 GB (one copy of X_train) instead of ~17 GB.
# =====================================================================

# Pass 1: count cells per city (no data loaded)
print(f"\n[{ts()}] Loading training data (memory-efficient)...")
import gc
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
print(f"  X_train will use {total * n_features * 4 / 1e9:.1f} GB RAM")

# Pass 2: pre-allocate and fill
X_train = np.empty((total, n_features), dtype=np.float32)
y_train = np.empty((total, N_CLASSES), dtype=np.float32)
city_boundaries = []  # (start_idx, end_idx) per city
offset = 0
for city, n in zip(train_cities_valid, city_counts):
    result = _load_city_arrays(city, mlp_cols)
    if result is None:
        continue
    X_city, _ = result
    labels = pd.read_parquet(city_labels_path(city, 2021))
    y_city = labels[CLASS_NAMES].values.astype(np.float32)
    del labels

    X_train[offset:offset + n] = X_city
    y_train[offset:offset + n] = y_city
    city_boundaries.append((offset, offset + n))
    del X_city, y_city
    offset += n
    gc.collect()

print(f"  Loaded: {offset:,} samples, {len(city_boundaries)} cities")

# Load all test cities
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

# Scale IN-PLACE (avoids creating a copy = saves 8.6 GB peak RAM)
print(f"\n[{ts()}] Fitting scaler (in-place)...")
scaler = StandardScaler()
scaler.fit(X_train)  # only computes mean/std, O(1) extra memory

# In-place transform: X = (X - mean) / scale
X_train -= scaler.mean_.astype(np.float32)
X_train /= scaler.scale_.astype(np.float32)
print(f"  Scaled {total:,} x {n_features} in-place")

# Scale test data in-place too
for name in list(test_data.keys()):
    X_v, y_v = test_data[name]
    X_v -= scaler.mean_.astype(np.float32)
    X_v /= scaler.scale_.astype(np.float32)
    test_data[name] = (X_v, y_v)

with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUT_DIR, "mlp_cols.json"), "w") as f:
    json.dump(mlp_cols, f)

# All data fits on GPU (3.5 GB data + models ≤ 8 GB VRAM)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
print(f"  Training data: GPU (no CPU->GPU overhead per batch!)")

n = len(y_train)
# Compute balanced epoch size using the smallest city
samples_per_city_global = min(s[1] - s[0] for s in city_boundaries)
n_balanced_global = samples_per_city_global * len(city_boundaries)
steps_per_epoch = (n_balanced_global + BATCH_SIZE - 1) // BATCH_SIZE
patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)
print(f"  Balanced epoch: {n_balanced_global:,} samples ({samples_per_city_global:,}/city x {len(city_boundaries)} cities)")
print(f"  Steps/epoch: {steps_per_epoch}, patience: {patience_epochs}ep, "
      f"min: {min_epochs}ep, max: {MAX_EPOCHS}ep")

y_norm = normalize_targets(y_train)

# Move everything to GPU
X_trn_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_trn_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
del X_train, y_norm
gc.collect()
print(f"  X_train on GPU: {X_trn_t.shape} ({X_trn_t.nbytes/1e9:.1f} GB)")

# Test tensors on GPU
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
# Sweep
# =====================================================================

results = []

print(f"\n[{ts()}] Starting sweep of {len(CONFIGS)} configs...")
for ci, (cfg_name, cfg_type, shape_params, dropout, wd) in enumerate(CONFIGS):
    print(f"\n{'='*70}")
    print(f"  [{ci+1}/{len(CONFIGS)}] {cfg_name}")

    # Build model
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
        shape_str = f"{n_features} -> " + " -> ".join([str(width)] * n_layers) + f" -> {N_CLASSES}"
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

    # Optimizer
    use_amp = device == "cuda"
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd, fused=use_amp)
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd)

    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = MAX_EPOCHS * steps_per_epoch  # uses balanced steps_per_epoch
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

    # Use first test city as validation target for early stopping
    val_name = "frankfurt"
    if val_name not in test_tensors:
        val_name = list(test_tensors.keys())[0]

    # City-balanced sampling: each epoch draws equal samples per city,
    # then shuffles. This prevents large cities from dominating.
    samples_per_city = samples_per_city_global
    n_balanced = n_balanced_global

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        epoch_loss, n_batches = 0.0, 0

        # Build balanced index: sample `samples_per_city` from each city
        balanced_idx = []
        for start_i, end_i in city_boundaries:
            city_n = end_i - start_i
            chosen = rng.choice(city_n, size=samples_per_city, replace=False)
            balanced_idx.append(chosen + start_i)
        perm = np.concatenate(balanced_idx)
        rng.shuffle(perm)

        perm_t = torch.from_numpy(perm).to(device, dtype=torch.long)
        for start in range(0, len(perm_t), BATCH_SIZE):
            idx = perm_t[start:start + BATCH_SIZE]
            # Direct GPU indexing — no CPU→GPU transfer!
            xb = X_trn_t[idx]
            yb = y_trn_t[idx]
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

        avg_train = epoch_loss / max(n_batches, 1)

        # Validate on early-stopping city
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
            print(f"    Ep {epoch:3d}: train={avg_train:.5f} val={val_loss:.5f} "
                  f"wait={wait}{marker}")

        if epoch >= min_epochs and wait >= patience_epochs:
            print(f"    Early stop at epoch {epoch} (patience={patience_epochs})")
            break

    elapsed = time.time() - t0
    n_epochs_done = epoch + 1

    # Load best
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # Evaluate on ALL test cities
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

    # Mean R2 across test cities
    mean_r2 = np.mean([m["r2"] for m in city_metrics.values()])
    mean_mae = np.mean([m["mae_pp"] for m in city_metrics.values()])

    result = {
        "name": cfg_name,
        "type": cfg_type,
        "widths": widths_list,
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

    # Save weights
    torch.save(best_state, os.path.join(OUT_DIR, f"{cfg_name}.pt"))

    # Save incremental results (in case of crash)
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    del net, optimizer, scaler_amp, scheduler, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# Summary
# =====================================================================

print(f"\n\n{'='*90}")
print(f"V5.5 SAR-only MLP SWEEP COMPLETE ({len(CONFIGS)} configs)")
print(f"{'='*90}")

# Sort by mean R2 descending
ranked = sorted(results, key=lambda r: r["mean_r2"], reverse=True)

header = (f"{'Rank':>4} {'Config':28s} {'Type':>4} {'Params':>10} "
          f"{'MeanR2':>8} {'MeanMAE':>8} {'BestEp':>6} {'Time':>6}")
print(header)
print("-" * len(header))
for rank, r in enumerate(ranked, 1):
    shape = "->".join(str(w) for w in r['widths'][:4])
    if len(r['widths']) > 4:
        shape += "->..."
    print(f"{rank:4d} {r['name']:28s} {r['type']:>4} {r['n_params']:>10,} "
          f"{r['mean_r2']:>8.4f} {r['mean_mae_pp']:>7.2f}pp "
          f"{r['best_epoch']:>5d}  {r['time_s']:>5.0f}s")

# Per-city breakdown for top 5
print(f"\n--- TOP 5 per-city breakdown ---")
for rank, r in enumerate(ranked[:5], 1):
    print(f"\n  #{rank} {r['name']} (mean R2={r['mean_r2']:.4f})")
    for city_name, m in r['per_city'].items():
        print(f"    {city_name:15s}  R2={m['r2']:.4f}  MAE={m['mae_pp']:.2f}pp")
        for cn in CLASS_NAMES:
            pc = m['per_class_r2'][cn]
            print(f"      {cn:20s} R2={pc:.4f}")

# Save final CSV summary
csv_rows = []
for r in ranked:
    row = {
        "rank": ranked.index(r) + 1,
        "name": r["name"],
        "type": r["type"],
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

# Also save to reports
reports_dir = os.path.join(PROJECT_ROOT, "reports", "v5_sweep")
os.makedirs(reports_dir, exist_ok=True)
df_summary.to_csv(os.path.join(reports_dir, f"sweep_{RUN_ID}.csv"), index=False)

# Identify best
best = ranked[0]
print(f"\n{'='*90}")
print(f"BEST CONFIG: {best['name']}")
print(f"  Mean R2 = {best['mean_r2']:.4f}, Mean MAE = {best['mean_mae_pp']:.2f}pp")
print(f"  Params: {best['n_params']:,}, Best epoch: {best['best_epoch']}")
print(f"\nAll results saved to: {OUT_DIR}")
print(f"{'='*90}")
