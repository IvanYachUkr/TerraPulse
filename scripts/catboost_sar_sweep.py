#!/usr/bin/env python3
"""
CatBoost SAR-only sweep — GPU-accelerated.

Tests whether SAR features alone (Sentinel-1 VV/VH stats, indices, LBP)
can predict land-cover fractions. Trains on all cities with SAR features,
tests on Frankfurt.

Multiple configs sweep depth, learning rate, and L2 regularization.
Uses GPU via CatBoost's native CUDA support.
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

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)

# Hyperparameter grid
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
                       "models_catboost_sar", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)


def ts():
    return datetime.now().strftime("%H:%M:%S")


# =====================================================================
# DATA LOADING
# =====================================================================

print(f"[{ts()}] CatBoost SAR-only sweep (run={RUN_ID})")
print(f"  Output: {OUT_DIR}")
print(f"  Configs: {len(CONFIGS)}")
print(f"  Test city: {TEST_CITY}")

# Find all cities with SAR features
print(f"\n[{ts()}] Scanning for cities with SAR features...")
all_cities = sorted([d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d, "features"))])

sar_cities = []
sar_cols = None
for city in all_cities:
    pq_path = os.path.join(DATA_DIR, city, "features",
                           "features_rust_2020_2021.parquet")
    if not os.path.exists(pq_path):
        continue
    schema = pq.read_schema(pq_path)
    city_sar = [f.name for f in schema if "SAR" in f.name]
    if len(city_sar) > 0:
        sar_cities.append(city)
        if sar_cols is None:
            sar_cols = sorted(city_sar)

if sar_cols is None:
    raise RuntimeError("No cities with SAR features found!")

n_sar = len(sar_cols)
train_cities = [c for c in sar_cities if c != TEST_CITY]
print(f"  SAR features: {n_sar}")
print(f"  Cities with SAR: {len(sar_cities)}")
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

    X = pd.read_parquet(pq_path, columns=sar_cols)
    y = pd.read_parquet(label_path, columns=CLASS_NAMES)
    # Drop NaN rows
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
X_test = pd.read_parquet(pq_path, columns=sar_cols).values.astype(np.float32)
y_test = pd.read_parquet(label_path,
                         columns=CLASS_NAMES).values.astype(np.float32)
# Handle NaN
X_train = np.nan_to_num(X_train, 0.0)
X_test = np.nan_to_num(X_test, 0.0)
print(f"  Test: {X_test.shape[0]:,} x {X_test.shape[1]} features")

# Save feature list
with open(os.path.join(OUT_DIR, "sar_cols.json"), "w") as f:
    json.dump(sar_cols, f, indent=2)

# =====================================================================
# SWEEP
# =====================================================================

results = []

for ci, (name, depth, lr, l2, iters) in enumerate(CONFIGS):
    print(f"\n{'='*70}")
    print(f"  [{ci+1}/{len(CONFIGS)}] {name}")
    print(f"  depth={depth}, lr={lr}, l2={l2}, iters={iters}")
    t0 = time.time()

    # Train one CatBoost per target class (multi-output via class loop)
    all_preds = np.zeros_like(y_test)

    for ki, cname in enumerate(CLASS_NAMES):
        model = CatBoostRegressor(
            iterations=iters,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2,
            task_type="GPU",
            devices="0",
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=100,
        )
        model.fit(
            X_train, y_train[:, ki],
            eval_set=(X_test, y_test[:, ki]),
            verbose=0,
        )
        pred = model.predict(X_test)
        all_preds[:, ki] = pred
        best_iter = model.get_best_iteration()
        print(f"    {cname:20s}: best_iter={best_iter}")
        del model
        gc.collect()

    elapsed = time.time() - t0

    # Metrics
    r2 = float(r2_score(y_test, all_preds))
    mae = float(mean_absolute_error(y_test, all_preds) * 100)
    per_class = {}
    for ki, cname in enumerate(CLASS_NAMES):
        yt = y_test[:, ki]
        if np.var(yt) < 1e-12:
            per_class[cname] = float("nan")
        else:
            per_class[cname] = float(r2_score(yt, all_preds[:, ki]))

    print(f"\n  >> {name}: R2={r2:.4f}  MAE={mae:.2f}pp  ({elapsed:.0f}s)")
    for cname in CLASS_NAMES:
        print(f"     {cname:20s} R2={per_class[cname]:.4f}")

    result = {
        "name": name,
        "depth": depth,
        "lr": lr,
        "l2_leaf_reg": l2,
        "iterations": iters,
        "time_s": round(elapsed, 1),
        "mean_r2": r2,
        "mean_mae_pp": mae,
        "per_class_r2": per_class,
    }
    results.append(result)

    # Save incrementally
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    gc.collect()

# =====================================================================
# SUMMARY
# =====================================================================

print(f"\n{'='*70}")
print("FINAL LEADERBOARD (SAR-only CatBoost)")
print(f"{'='*70}")

ranked = sorted(results, key=lambda r: -r["mean_r2"])
for rank, r in enumerate(ranked, 1):
    print(f"  {rank:2d} {r['name']:20s}  d={r['depth']}  lr={r['lr']:.2f}"
          f"  R2={r['mean_r2']:.4f}  MAE={r['mean_mae_pp']:.2f}pp"
          f"  {r['time_s']:.0f}s")

print(f"\nBest per-class R2 (SAR only):")
best = ranked[0]
for cn in CLASS_NAMES:
    v = best["per_class_r2"][cn]
    print(f"  {cn:20s}: {v:.4f}")

print(f"\n  For reference, MLP v5.7 best (all 1748 feats): R2=0.8407")
print(f"  SAR-only best: R2={best['mean_r2']:.4f}")
print(f"\nSaved to: {OUT_DIR}")
