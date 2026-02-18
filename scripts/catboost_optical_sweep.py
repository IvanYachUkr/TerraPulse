#!/usr/bin/env python3
"""
CatBoost optical-only sweep V2 — MultiRMSE + normalized predictions.

Improvements over V1:
  1. MultiRMSE: single multi-output model (learns cross-class correlations)
  2. Post-hoc normalization: clip negatives, renormalize to sum=1

Same 10 configs, same 19 training cities, test on Frankfurt.
"""

import os
import sys
import json
import time
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error

# =====================================================================
# CONFIG
# =====================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cities")

TEST_CITY = "frankfurt"
CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction"}

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)

CONFIGS = [
    # (name, depth, lr, l2_leaf, iterations)
    ("shallow_fast",     4, 0.10, 3.0,  1000),
    ("shallow_slow",     4, 0.03, 3.0,  3000),
    ("mid_fast",         6, 0.10, 3.0,  1000),
    ("mid_slow",         6, 0.03, 3.0,  3000),
    ("mid_reg",          6, 0.05, 10.0, 2000),
    ("deep_fast",        8, 0.10, 3.0,  1000),
    ("deep_slow",        8, 0.03, 3.0,  3000),
    ("deep_reg",         8, 0.05, 10.0, 2000),
    ("vdeep_slow",      10, 0.03, 5.0,  3000),
    ("vdeep_reg",       10, 0.01, 10.0, 5000),
]

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities",
                       "models_catboost_optical_v2", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)


def ts():
    return datetime.now().strftime("%H:%M:%S")


def normalize_preds(pred):
    """Clip negatives and normalize rows to sum to 1."""
    pred = np.clip(pred, 0.0, None)
    s = pred.sum(axis=1, keepdims=True)
    pred = np.divide(pred, s, out=np.zeros_like(pred), where=s > 0)
    return pred


# =====================================================================
# DATA LOADING
# =====================================================================

print(f"[{ts()}] CatBoost OPTICAL V2 sweep (MultiRMSE + norm) run={RUN_ID}")
print(f"  Output: {OUT_DIR}")
print(f"  Configs: {len(CONFIGS)}")
print(f"  Test city: {TEST_CITY}")

# Find all cities with SAR features (same city set for fair comparison)
print(f"\n[{ts()}] Scanning for cities...")
all_cities = sorted([d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d, "features"))])

sar_cities = []
optical_cols = None
for city in all_cities:
    pq_path = os.path.join(DATA_DIR, city, "features",
                           "features_rust_2020_2021.parquet")
    if not os.path.exists(pq_path):
        continue
    schema = pq.read_schema(pq_path)
    city_sar = [f.name for f in schema if "SAR" in f.name]
    if len(city_sar) > 0:
        sar_cities.append(city)
        if optical_cols is None:
            numeric_types = {'float', 'double', 'int32', 'int64',
                             'float32', 'float64'}
            optical_cols = []
            for f in schema:
                type_str = str(f.type).lower()
                is_num = any(t in type_str for t in numeric_types)
                if (is_num and f.name not in CONTROL_COLS
                        and "SAR" not in f.name):
                    optical_cols.append(f.name)

if optical_cols is None:
    raise RuntimeError("No cities with features found!")

n_optical = len(optical_cols)
train_cities = [c for c in sar_cities if c != TEST_CITY]
print(f"  Optical features: {n_optical}")
print(f"  Training cities: {len(train_cities)}")
print(f"  Test city: {TEST_CITY}")

# Load training data
print(f"\n[{ts()}] Loading training data...")
train_dfs_X = []
train_dfs_y = []
for city in train_cities:
    pq_path = os.path.join(DATA_DIR, city, "features",
                           "features_rust_2020_2021.parquet")
    label_path = os.path.join(DATA_DIR, city, "labels_2021.parquet")
    if not os.path.exists(label_path):
        print(f"  WARNING: no labels for {city}, skip")
        continue

    X = pd.read_parquet(pq_path, columns=optical_cols)
    y = pd.read_parquet(label_path, columns=CLASS_NAMES)
    mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[mask].values.astype(np.float32)
    y = y[mask].values.astype(np.float32)
    train_dfs_X.append(X)
    train_dfs_y.append(y)
    print(f"  [{city}] {len(X):,} cells")
    del X, y
    gc.collect()

X_train = np.concatenate(train_dfs_X)
y_train = np.concatenate(train_dfs_y)
del train_dfs_X, train_dfs_y
gc.collect()
print(f"  Total training: {X_train.shape[0]:,} x {X_train.shape[1]} features")

# Load test data
print(f"\n[{ts()}] Loading test data ({TEST_CITY})...")
pq_path = os.path.join(DATA_DIR, TEST_CITY, "features",
                       "features_rust_2020_2021.parquet")
label_path = os.path.join(DATA_DIR, TEST_CITY, "labels_2021.parquet")
X_test = pd.read_parquet(pq_path, columns=optical_cols).values.astype(np.float32)
y_test = pd.read_parquet(label_path,
                         columns=CLASS_NAMES).values.astype(np.float32)
X_train = np.nan_to_num(X_train, 0.0)
X_test = np.nan_to_num(X_test, 0.0)
print(f"  Test: {X_test.shape[0]:,} x {X_test.shape[1]} features")

# Save feature list
with open(os.path.join(OUT_DIR, "optical_cols.json"), "w") as f:
    json.dump(optical_cols, f, indent=2)

# Create pools for MultiRMSE
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)
print(f"  Pools created")

# =====================================================================
# SWEEP
# =====================================================================

results = []

for ci, (name, depth, lr, l2, iters) in enumerate(CONFIGS):
    print(f"\n{'='*70}")
    print(f"  [{ci+1}/{len(CONFIGS)}] {name}")
    print(f"  depth={depth}, lr={lr}, l2={l2}, iters={iters}")
    print(f"  loss=MultiRMSE (joint 6-output)")
    t0 = time.time()

    model = CatBoostRegressor(
        iterations=iters,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2,
        task_type="GPU",
        devices="0",
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
    )
    model.fit(train_pool, eval_set=test_pool, verbose=0)
    best_iter = model.get_best_iteration()
    print(f"  best_iter={best_iter}")

    raw_preds = model.predict(X_test)
    all_preds = normalize_preds(raw_preds)
    del model
    gc.collect()

    elapsed = time.time() - t0

    # Metrics on normalized predictions
    r2 = float(r2_score(y_test, all_preds))
    mae = float(mean_absolute_error(y_test, all_preds) * 100)

    # Also compute raw (un-normalized) R2 for comparison
    r2_raw = float(r2_score(y_test, raw_preds))

    per_class = {}
    per_class_raw = {}
    for ki, cname in enumerate(CLASS_NAMES):
        yt = y_test[:, ki]
        if np.var(yt) < 1e-12:
            per_class[cname] = float("nan")
            per_class_raw[cname] = float("nan")
        else:
            per_class[cname] = float(r2_score(yt, all_preds[:, ki]))
            per_class_raw[cname] = float(r2_score(yt, raw_preds[:, ki]))

    print(f"\n  >> {name}: R2={r2:.4f} (raw={r2_raw:.4f})  "
          f"MAE={mae:.2f}pp  ({elapsed:.0f}s)")
    for cname in CLASS_NAMES:
        print(f"     {cname:20s} R2={per_class[cname]:.4f} "
              f"(raw={per_class_raw[cname]:.4f})")

    result = {
        "name": name,
        "depth": depth,
        "lr": lr,
        "l2_leaf_reg": l2,
        "iterations": iters,
        "best_iteration": best_iter,
        "time_s": round(elapsed, 1),
        "mean_r2": r2,
        "mean_r2_raw": r2_raw,
        "mean_mae_pp": mae,
        "per_class_r2": per_class,
        "per_class_r2_raw": per_class_raw,
    }
    results.append(result)

    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    gc.collect()

# =====================================================================
# SUMMARY
# =====================================================================

print(f"\n{'='*70}")
print("FINAL LEADERBOARD (OPTICAL CatBoost V2: MultiRMSE + norm)")
print(f"{'='*70}")

ranked = sorted(results, key=lambda r: -r["mean_r2"])
for rank, r in enumerate(ranked, 1):
    print(f"  {rank:2d} {r['name']:20s}  d={r['depth']}  lr={r['lr']:.2f}"
          f"  R2={r['mean_r2']:.4f} (raw={r['mean_r2_raw']:.4f})"
          f"  MAE={r['mean_mae_pp']:.2f}pp  {r['time_s']:.0f}s")

print(f"\nBest per-class R2:")
best = ranked[0]
for cn in CLASS_NAMES:
    v = best["per_class_r2"][cn]
    vr = best["per_class_r2_raw"][cn]
    print(f"  {cn:20s}: {v:.4f} (raw={vr:.4f})")

print(f"\n  SAR-only CatBoost:       R2~0.42")
print(f"  Optical CatBoost V1:     R2~0.81  (6 independent models)")
print(f"  Optical CatBoost V2:     R2={best['mean_r2']:.4f}  (MultiRMSE+norm)")
print(f"  MLP v5.7 best (all):     R2=0.8407")
print(f"\nSaved to: {OUT_DIR}")
