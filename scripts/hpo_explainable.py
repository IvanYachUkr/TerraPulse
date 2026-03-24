#!/usr/bin/env python3
"""
hpo_explainable.py – Optuna HPO for the Linear Stage 1 Model

This script tunes the Logistic Regression model parameters (C, penalty, solver)
to maximize the Macro F1-score for change detection, preventing the model from
simply predicting "No Change" everywhere.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd

import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# --- Paths & Grid ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
S2_DIR = DATA_DIR / "sentinel2"
WC_DIR = DATA_DIR / "worldcover"
STATS_DIR = DATA_DIR / "nuremberg_stats"
RESULTS_FILE = SCRIPT_DIR / "hpo_explainable_results.json"

ANCHOR_CRS = "EPSG:32632"
ANCHOR_TRANSFORM = Affine(10.0, 0.0, 641740.0, 0.0, -10.0, 5492260.0)
ANCHOR_W, ANCHOR_H = 2550, 2850
WC_TO_CLASS = {10: 0, 20: 1, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 1}

def remap_wc(arr):
    out = np.full_like(arr, 255, dtype=np.uint8)
    for k, v in WC_TO_CLASS.items(): out[arr == k] = v
    return out

def reproject_to_grid(path, categorical=False):
    with rasterio.open(path) as src:
        dtype = np.uint8 if categorical else np.float32
        resampling = Resampling.nearest if categorical else Resampling.bilinear
        dst = np.full((src.count, ANCHOR_H, ANCHOR_W), 0 if categorical else np.nan, dtype=dtype)
        for i in range(src.count):
            reproject(rasterio.band(src, i+1), dst[i], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ANCHOR_TRANSFORM, dst_crs=ANCHOR_CRS, resampling=resampling)
        return dst if not categorical else dst[0]

def build_features():
    # Load Boundary
    geo_p = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_boundary.geojson"
    gdf = gpd.read_file(geo_p).to_crs(ANCHOR_CRS)
    mask = rasterize([(g, 1) for g in gdf.geometry], out_shape=(ANCHOR_H, ANCHOR_W), transform=ANCHOR_TRANSFORM) == 1
    
    # Load Data
    s2_20 = reproject_to_grid(S2_DIR / "s2_median_2020.tif")
    stats = reproject_to_grid(STATS_DIR / "rasterized_stats.tif")
    std_p = S2_DIR / "s2_ndvi_std_2020.tif"
    std = reproject_to_grid(std_p) if std_p.exists() else np.zeros((1, ANCHOR_H, ANCHOR_W))
    wc_20 = remap_wc(reproject_to_grid(WC_DIR / "worldcover_2020.tif", True))
    wc_21 = remap_wc(reproject_to_grid(WC_DIR / "worldcover_2021.tif", True))

    valid = (mask & np.all(np.isfinite(s2_20), axis=0) & np.all(np.isfinite(stats), axis=0))
    
    features = []
    for i in range(10): features.append(s2_20[i][valid])
    b4, b8 = s2_20[2][valid], s2_20[6][valid]
    features.append((b8 - b4) / (b8 + b4 + 1e-8))
    features.append(wc_20[valid])
    for i in range(4): features.append(stats[i][valid])
    features.append(std[0][valid])
    
    X = np.column_stack(features)
    y = (wc_21[valid] != wc_20[valid]).astype(int)
    return X, y

def save_optuna_results(study, X_shape, start_time):
    out = {
        "metadata": {
            "started": start_time,
            "finished": datetime.now().isoformat(),
            "n_pixels": X_shape[0],
            "n_features": X_shape[1],
            "n_folds": 4
        },
        "best_macro_f1": study.best_value,
        "best_params": study.best_params,
        "trials": []
    }
    
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            out["trials"].append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "fold_f1s": t.user_attrs.get("fold_f1s", [])
            })
            
    with open(RESULTS_FILE, 'w') as f:
        json.dump(out, f, indent=2)

# --- Global Variables for Objective ---
X_bal = None
y_bal = None

def objective(trial):
    c_val = trial.suggest_float("C", 1e-4, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    
    # solver selection based on penalty
    if penalty == "l1":
        solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
    else:
        solver = trial.suggest_categorical("solver_l2", ["lbfgs", "liblinear", "saga"])
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    fold_f1s = []
    
    scaler = StandardScaler()
    
    for train_idx, test_idx in cv.split(X_bal, y_bal):
        X_tr, X_te = X_bal[train_idx], X_bal[test_idx]
        y_tr, y_te = y_bal[train_idx], y_bal[test_idx]
        
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        model = LogisticRegression(
            C=c_val,
            penalty=penalty,
            class_weight=class_weight,
            solver=solver,
            max_iter=1000,
            random_state=42,
            n_jobs=-1 if solver in ['lbfgs', 'saga'] else None
        )
        
        # Train & Predict
        model.fit(X_tr_sc, y_tr)
        y_pred = model.predict(X_te_sc)
        
        # Calculate Macro F1
        score = f1_score(y_te, y_pred, average='macro')
        fold_f1s.append(score)
        
    mean_f1 = np.mean(fold_f1s)
    trial.set_user_attr("fold_f1s", fold_f1s)
    
    return mean_f1

def main():
    print("--- Executing Linear Model HPO ---")
    start_time = datetime.now().isoformat()
    
    global X_bal, y_bal
    
    # Load & Engineer Features
    print("Loading data...")
    X, y = build_features()
    print(f"Loaded {X.shape[0]} valid pixels.")
    
    # Balanced Downsampling
    chg_idx = np.where(y == 1)[0]
    sta_idx = np.where(y == 0)[0]
    np.random.seed(42)
    sta_sample = np.random.choice(sta_idx, size=len(chg_idx), replace=False)
    
    bal_mask = np.concatenate([chg_idx, sta_sample])
    X_bal, y_bal = X[bal_mask], y[bal_mask]
    print(f"Balanced Dataset: {len(X_bal)} samples ({len(chg_idx)} changes, {len(sta_sample)} stable)")
    
    # Run Optuna Study
    print("Starting Optuna optimization (50 trials)...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    
    # Fast parallel execution with joblib isn't heavily needed if n_trials is small, 
    # but we will just optimize sequentially since each trial is fast.
    study.optimize(objective, n_trials=50, n_jobs=4, show_progress_bar=True)
    
    print(f"\nOptimization Finished! Best Macro F1: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    save_optuna_results(study, X.shape, start_time)
    print(f"Saved results to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
