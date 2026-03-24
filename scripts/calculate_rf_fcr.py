#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier

# Add scripts dir to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from experimental_nuremberg import (
    load_multi_year_aligned, 
    build_boundary_mask, 
    build_feature_matrix,
    remap_worldcover,
    reproject_to_anchor,
    WC_DIR,
    DASHBOARD_DATA
)

def calculate_fcr():
    print("Loading data for FCR calculation...")
    boundary = build_boundary_mask()
    wc_2020_path = WC_DIR / "worldcover_2020.tif"
    wc_2021_path = WC_DIR / "worldcover_2021.tif"
    
    s_2020, _, stats, std = load_multi_year_aligned(2020)
    X, y_row, flat_idx = build_feature_matrix(s_2020, stats, std, boundary, wc_2020_path)
    
    y_start = remap_worldcover(reproject_to_anchor(wc_2020_path, categorical=True)).flatten()[np.where(boundary.flatten())[0]]
    y_target = remap_worldcover(reproject_to_anchor(wc_2021_path, categorical=True)).flatten()[np.where(boundary.flatten())[0]]
    
    mask_in_valid = np.isin(np.where(boundary.flatten())[0], flat_idx)
    y_start, y_target = y_start[mask_in_valid], y_target[mask_in_valid]
    
    is_changed = (y_target != y_start).astype(np.int8)
    
    # 4-Fold Spatial Holdout
    sort_idx = np.argsort(y_row)
    fold_ids = np.zeros(len(y_row), dtype=int)
    fold_ids[sort_idx] = np.arange(len(y_row)) * 4 // len(y_row)
    
    fcrs = []
    
    # Use params from experimental_nuremberg.py
    # clf1 = RandomForestClassifier(n_estimators=50, max_depth=35, max_features=0.785, n_jobs=-1, random_state=42)
    
    print("Evaluating FCR across folds...")
    for f in range(4):
        train_mask = fold_ids != f
        test_mask = fold_ids == f
        
        X_tr, y_tr_chg = X[train_mask], is_changed[train_mask]
        X_te, y_te_chg = X[test_mask], is_changed[test_mask]
        
        # Balance training
        chg_idx = np.where(y_tr_chg == 1)[0]
        sta_idx = np.where(y_tr_chg == 0)[0]
        sta_sample = np.random.default_rng(42).choice(sta_idx, size=len(chg_idx), replace=False)
        bal_idx = np.concatenate([chg_idx, sta_sample])
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=35, max_features=0.785, n_jobs=-1, random_state=42)
        clf.fit(X_tr[bal_idx], y_tr_chg[bal_idx])
        
        # Predict on test set
        # Using 95% threshold as in production
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.95).astype(np.int8)
        
        # FCR = Pred_Change when True_NoChange
        mask_no_change = (y_te_chg == 0)
        fcr = (preds[mask_no_change] == 1).mean()
        fcrs.append(fcr)
        print(f"  Fold {f}: FCR = {fcr:.4f}")

    mean_fcr = np.mean(fcrs)
    print(f"\nMean False Change Rate for RF: {mean_fcr:.4f}")
    
    # Update experimental_metrics.json
    metrics_path = DASHBOARD_DATA / "experimental_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        metrics["false_change_rate"] = mean_fcr
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Updated {metrics_path}")
    else:
        print(f"Warning: {metrics_path} not found.")

if __name__ == "__main__":
    calculate_fcr()
