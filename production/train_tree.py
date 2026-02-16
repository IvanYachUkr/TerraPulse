#!/usr/bin/env python3
"""
Production Training Script for LightGBM (Best Tree Model).

Train the final LightGBM model on extracted features.
Uses the "strong_wide" configuration and "VegIdx+RedEdge+TC+NDTI+IRECI+CRI1" feature set.
"""

import os
import sys
import re
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_V2_DIR, PROJECT_ROOT

# Hardcoded classes
FINAL_CLASSES = [
    "tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"
]

def build_feature_set(feat_cols):
    """Select the best feature set: VegIdx + RedEdge + TC + NDTI + IRECI + CRI1."""

    # 1. Base groups from extract_features.py naming
    band_pat = re.compile(r'^B(05|06|07|8A)_')

    veg_idx = [c for c in feat_cols
               if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"])
               and not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr")]

    rededge = [c for c in feat_cols if band_pat.match(c)]

    tc = [c for c in feat_cols if c.startswith("TC_")]

    # 2. Novel indices
    novel = [c for c in feat_cols
             if any(c.startswith(p) for p in ["NDTI_", "IRECI_", "CRI1_"])]

    # Combine
    selected = veg_idx + rededge + tc + novel
    return sorted(list(set(selected)))

def main():
    # ── Config ──
    # Best config from sweep (strong_wide)
    LGBM_PARAMS = dict(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        num_leaves=255,
        min_child_samples=20,
        reg_lambda=3.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        verbosity=-1,
        n_jobs=-1,
        random_state=42
    )

    # ── Load Data ──
    path = os.path.join(PROCESSED_V2_DIR, "features_rust_production.parquet")
    if not os.path.exists(path):
        # Fallback
        # Note: Standard pipeline splits features into two files (full + v2).
        # We need both.
        print("Production parquet not found. Attempting to load from merged sets...")
        path_full = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")
        path_v2 = os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet")

        if os.path.exists(path_full) and os.path.exists(path_v2):
            df_full = pd.read_parquet(path_full)
            df_v2 = pd.read_parquet(path_v2)
            # Merge
            df = df_full.merge(df_v2, on="cell_id", suffixes=("", "_v2"))
            # Handle potential duplicate columns if any (v2 shouldn't overlap full usually)
        elif os.path.exists(path_full):
            df = pd.read_parquet(path_full)
            print("Warning: V2 features (NDTI/IRECI/CRI1) missing. Model performance may degrade.")
        else:
            print("No features found.")
            return
    else:
        print(f"Loading features from {path}...")
        df = pd.read_parquet(path)

    # Select features
    CONTROL = ["cell_id", "valid_fraction", "low_valid_fraction", "reflectance_scale", "full_features_computed"]
    all_numeric = [c for c in df.columns if c not in CONTROL and pd.api.types.is_numeric_dtype(df[c])]

    selected_cols = build_feature_set(all_numeric)
    print(f"Selected {len(selected_cols)} features for Tree model.")

    X = df[selected_cols].values.astype(np.float32)
    X = np.nan_to_num(X)

    # Labels
    labels_path = os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet")
    if os.path.exists(labels_path):
        y_df = pd.read_parquet(labels_path)
        y = y_df[FINAL_CLASSES].values.astype(np.float32)
    else:
        print("Labels not found.")
        return

    # Train
    print("Training LightGBM (MultiOutput)...")
    model = MultiOutputRegressor(lgb.LGBMRegressor(**LGBM_PARAMS))
    model.fit(X, y)

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "models", "production_tree")
    os.makedirs(out_dir, exist_ok=True)

    # Save using pickle
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"features": selected_cols, "params": LGBM_PARAMS}, f)

    print(f"Model saved to {out_dir}")

if __name__ == "__main__":
    main()
