#!/usr/bin/env python3
"""
Per-Pixel Land Cover Classification V3 — CatBoost with native CUDA.

Reuses V2's cached data (217 features, ~10s load) and adds CatBoost
with native CUDA GPU training for fast iteration.

Usage:
    # Quick run with GPU (loads cached data)
    python scripts/pixel_classifier_v3.py

    # Sweep multiple configs
    python scripts/pixel_classifier_v3.py --sweep

    # Force CPU
    python scripts/pixel_classifier_v3.py --device cpu

    # Custom trees
    python scripts/pixel_classifier_v3.py --depth 8 --trees 1500 --lr 0.05
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Reuse V2 infrastructure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.pixel_classifier_v2 import (
    SEED, N_CLASSES, CLASS_NAMES, CITIES_DIR,
    load_data_cache, save_data_cache, load_cities_sequential,
    build_pixel_features, stratified_subsample, evaluate_model,
    ts,
)
from scripts.run_multi_city_pipeline_v5 import CITIES

EXCLUDED_CITY_NAMES = {"nuremberg"}
VAL_CITY_NAMES = {
    "finnish_lakeland", "danish_farmland", "tabernas_desert",
    "sardinia_maquis", "crete_phrygana", "iceland_highlands",
    "lapland_tundra", "ireland_bog_pasture", "hortobagy_puszta",
    "vojvodina_cropland", "camargue_wetland", "pyrenees_meadows",
    "munich", "seville", "stockholm",
}


def train_catboost(X_train, y_train, X_val, y_val, config, feature_names=None):
    """Train CatBoost with native CUDA. Falls back to CPU on failure."""
    cname = config.get('name', 'default')
    device = config.get('device', 'GPU')

    # Class weights (inverse frequency)
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    class_weights = {int(c): total / (len(classes) * cnt)
                     for c, cnt in zip(classes, counts)}

    def _make_params(task_type):
        params = {
            'iterations': config.get('trees', 2000),
            'depth': config.get('depth', 8),
            'learning_rate': config.get('lr', 0.05),
            'l2_leaf_reg': config.get('l2_leaf_reg', 3.0),
            'random_seed': SEED,
            'task_type': task_type,
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'class_weights': class_weights,
            'verbose': 100,
            'early_stopping_rounds': 80,
            'use_best_model': True,
            'auto_class_weights': None,
        }
        if task_type == 'GPU':
            params['devices'] = '0'
        return params

    # Try GPU first, fall back to CPU
    devices_to_try = ['GPU', 'CPU'] if device == 'GPU' else ['CPU']

    for dev in devices_to_try:
        try:
            print(f"\n  [{ts()}] Training CatBoost [{cname}] on {dev}...")
            print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
            print(f"  Val:   {X_val.shape[0]:,}")
            print(f"  Config: depth={config.get('depth', 8)}, "
                  f"trees={config.get('trees', 2000)}, lr={config.get('lr', 0.05)}")

            params = _make_params(dev)
            model = CatBoostClassifier(**params)

            train_pool = Pool(X_train, y_train,
                            feature_names=feature_names)
            val_pool = Pool(X_val, y_val,
                          feature_names=feature_names)

            model.fit(train_pool, eval_set=val_pool)

            best_iter = model.get_best_iteration()
            print(f"  Best iteration: {best_iter}")
            return model

        except Exception as e:
            print(f"\n  [!] {dev} training FAILED: {e}")
            if dev != devices_to_try[-1]:
                print(f"  [!] Falling back to CPU...")
            else:
                raise


def evaluate_catboost(model, X, y):
    """Evaluate CatBoost model and print metrics."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = model.predict(X).flatten().astype(int)
    acc = accuracy_score(y, y_pred)

    present = sorted(set(y) | set(y_pred))
    names = [CLASS_NAMES[c] for c in present]

    report = classification_report(y, y_pred, labels=present,
                                   target_names=names, digits=4)
    cm = confusion_matrix(y, y_pred, labels=present)

    print(f"\n  Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n{report}")

    print(f"\n  Confusion Matrix:")
    header = "  " + " ".join(f"{n[:6]:>6}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>6}" for v in row)
        print(f"  {names[i]:<12} {row_str}")

    return {
        'accuracy': float(acc),
        'report': report,
        'cm': cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-pixel classifier V3 (CatBoost)")
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'],
                        help='GPU (CUDA) or CPU')
    parser.add_argument('--sweep', action='store_true',
                        help='Try multiple configs')
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--trees', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--max-pixels-per-city', type=int, default=50000)
    parser.add_argument('--min-per-class', type=int, default=500)
    args = parser.parse_args()

    np.random.seed(SEED)
    out_dir = os.path.join(CITIES_DIR, "models_pixel_v3")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Per-Pixel Land Cover Classification V3 (CatBoost)")
    print(f"  Device: {args.device} | Sweep: {args.sweep}")
    print(f"  Default: depth={args.depth}, trees={args.trees}, lr={args.lr}")
    print(f"{'='*70}\n")

    # --- Load cached data (from V2) ---
    t_load = time.time()
    cached = load_data_cache(args.max_pixels_per_city, args.min_per_class)

    if cached is not None:
        X_train, y_train, X_val, y_val, feat_names = cached
    else:
        print("No cached data found! Run V2 first or loading from scratch...")
        train_cities = [c for c in CITIES
                        if c.name not in EXCLUDED_CITY_NAMES
                        and c.name not in VAL_CITY_NAMES]
        val_cities = [c for c in CITIES if c.name in VAL_CITY_NAMES]

        print(f"\n[{ts()}] Loading TRAIN ({len(train_cities)} cities, sequential)...")
        all_X_train, all_y_train, feat_names = load_cities_sequential(
            train_cities, args.max_pixels_per_city, args.min_per_class)

        print(f"\n[{ts()}] Loading VAL ({len(val_cities)} cities, sequential)...")
        all_X_val, all_y_val, _ = load_cities_sequential(
            val_cities, args.max_pixels_per_city, args.min_per_class)

        if not all_X_train or not all_X_val:
            print("ERROR: No data!"); return

        X_train = np.concatenate(all_X_train).astype(np.float32)
        y_train = np.concatenate(all_y_train)
        X_val = np.concatenate(all_X_val).astype(np.float32)
        y_val = np.concatenate(all_y_val)
        del all_X_train, all_y_train, all_X_val, all_y_val; gc.collect()

        save_data_cache(X_train, y_train, X_val, y_val, feat_names,
                        args.max_pixels_per_city, args.min_per_class)

    load_time = time.time() - t_load
    print(f"\n[{ts()}] Data loaded in {load_time:.1f}s")
    print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    print(f"  Val:   {X_val.shape[0]:,} x {X_val.shape[1]}")

    for sn, sy in [("Train", y_train), ("Val", y_val)]:
        cls, cnt = np.unique(sy, return_counts=True); tot = cnt.sum()
        print(f"\n  {sn} distribution:")
        for c, n in zip(cls, cnt):
            print(f"    {CLASS_NAMES[c]:>15}: {n:>8,} ({100*n/tot:5.1f}%)")

    # --- Train ---
    results = {}

    if args.sweep:
        configs = [
            {'name': 'shallow_fast', 'depth': 6, 'trees': 1500,
             'lr': 0.08, 'device': args.device},
            {'name': 'default', 'depth': 8, 'trees': 2000,
             'lr': 0.05, 'device': args.device},
            {'name': 'deep', 'depth': 10, 'trees': 2000,
             'lr': 0.03, 'device': args.device},
            {'name': 'v1_style', 'depth': 8, 'trees': 1000,
             'lr': 0.05, 'device': args.device},
        ]
    else:
        configs = [
            {'name': 'default', 'depth': args.depth, 'trees': args.trees,
             'lr': args.lr, 'device': args.device},
        ]

    for config in configs:
        cname = config['name']
        print(f"\n{'='*70}")
        print(f"  CatBoost [{cname}]")
        print(f"{'='*70}")

        t0 = time.time()
        model = train_catboost(X_train, y_train, X_val, y_val, config,
                               feat_names)
        elapsed = time.time() - t0
        print(f"\n  [{ts()}] Trained in {elapsed:.1f}s")

        print(f"\n  --- [{cname}] Validation ---")
        metrics = evaluate_catboost(model, X_val, y_val)
        results[f'catboost_{cname}'] = metrics

        # Feature importance top 20
        if feat_names:
            imp = model.get_feature_importance()
            top_idx = np.argsort(imp)[::-1][:20]
            print(f"\n  Top 20 Features:")
            for rank, idx in enumerate(top_idx):
                fname = feat_names[idx] if idx < len(feat_names) else f"f_{idx}"
                print(f"    {rank+1:2d}. {fname:<35s}: {imp[idx]:>6.1f}")

        # Save model
        path = os.path.join(out_dir, f"catboost_pixel_v3_{cname}.cbm")
        model.save_model(path)
        print(f"\n  Saved: {path}")

        # Also save as pickle for compatibility
        pkl_path = os.path.join(out_dir, f"catboost_pixel_v3_{cname}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)

    # Save metrics
    def jsonify(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(os.path.join(out_dir, "metrics_pixel_v3.json"), 'w') as f:
        json.dump(results, f, indent=2, default=jsonify)

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
