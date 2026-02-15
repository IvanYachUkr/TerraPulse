#!/usr/bin/env python3
"""
Train & save the final LightGBM champion model (best config from sweep).

Produces:
    models/final_tree/fold_{i}.pkl        — fitted MultiOutputRegressor per fold
    models/final_tree/oof_predictions.parquet — OOF predictions
    models/final_tree/meta.json           — model metadata + per-fold metrics
"""

import json
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_V2_DIR, PROJECT_ROOT
from src.splitting import get_fold_indices
from src.models.evaluation import evaluate_model

SEED = 42
N_FOLDS = 5
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
OUT_DIR = os.path.join(PROJECT_ROOT, "models", "final_tree")
os.makedirs(OUT_DIR, exist_ok=True)

FULL_PQ = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")
V2_PQ = os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet")

NOVEL_INDICES = ["NDTI", "IRECI", "CRI1"]

# Best config from sweep: "base" with VegIdx+RedEdge+TC+NDTI+IRECI+CRI1
BEST_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=20, reg_lambda=0.1,
    subsample=0.85, colsample_bytree=0.85, verbosity=-1,
    random_state=SEED, n_jobs=-1,
)


def build_feature_groups(feat_cols):
    """Same logic as lgbm_sweep.py."""
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    veg_idx = [c for c in feat_cols
               if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"])
               and not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr")]
    rededge = [c for c in feat_cols if band_pat.match(c)]
    tc = [c for c in feat_cols if c.startswith("TC_")]
    return {"VegIdx": veg_idx, "RedEdge": rededge, "TC": tc}


def get_novel_cols(v2_cols, index_name):
    return [c for c in v2_cols if c.startswith(f"{index_name}_")]


def main():
    t0 = time.time()
    print("=" * 70)
    print("Train Final LightGBM Champion")
    print("=" * 70)

    # ── Build feature list ──
    full_cols = [c for c in pq.read_schema(FULL_PQ).names if c != "cell_id"]
    v2_cols = [c for c in pq.read_schema(V2_PQ).names if c != "cell_id"]

    groups = build_feature_groups(full_cols)
    novel_cols = {idx: get_novel_cols(v2_cols, idx) for idx in NOVEL_INDICES}

    base_cols = groups["VegIdx"] + groups["RedEdge"] + groups["TC"]
    novel_extra = [c for idx in NOVEL_INDICES for c in novel_cols[idx]]
    feature_cols = base_cols + novel_extra
    print(f"Feature set: VegIdx+RedEdge+TC+NDTI+IRECI+CRI1 = {len(feature_cols)} features")

    # ── Load data ──
    print("Loading data...")
    base_needed = sorted(set(base_cols))
    novel_needed = sorted(set(novel_extra))

    base_df = pd.read_parquet(FULL_PQ, columns=["cell_id"] + base_needed)
    if novel_needed:
        v2_df = pd.read_parquet(V2_PQ, columns=["cell_id"] + novel_needed)
        merged = base_df.merge(v2_df, on="cell_id", how="inner")
    else:
        merged = base_df

    X_all = np.nan_to_num(merged[feature_cols].values.astype(np.float32), 0.0)
    cell_ids = merged["cell_id"].values

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        split_meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    print(f"Data: {X_all.shape[0]} cells, {X_all.shape[1]} features")

    # ── Train per fold ──
    oof = np.zeros_like(y)
    oof_mask = np.zeros(len(y), dtype=bool)
    fold_metrics = []

    for fold_id in range(N_FOLDS):
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id,
            split_meta["tile_cols"], split_meta["tile_rows"],
            buffer_tiles=1,
        )
        print(f"\nFold {fold_id}: train={len(train_idx)}, test={len(test_idx)}")

        t_fold = time.time()
        model = MultiOutputRegressor(lgb.LGBMRegressor(**BEST_PARAMS))
        model.fit(X_all[train_idx], y[train_idx])

        y_pred = np.clip(model.predict(X_all[test_idx]), 0, 100)
        oof[test_idx] = y_pred
        oof_mask[test_idx] = True

        summary, per_class = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
        r2 = summary["r2_uniform"]
        mae = summary["mae_mean_pp"]
        elapsed = time.time() - t_fold
        print(f"  R2={r2:.4f}  MAE={mae:.3f}pp  ({elapsed:.1f}s)")

        fold_metrics.append({
            "fold": fold_id,
            "r2_uniform": r2,
            "mae_mean_pp": mae,
            "time_s": elapsed,
        })

        # Save model
        model_path = os.path.join(OUT_DIR, f"fold_{fold_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved {model_path}")

    # ── Save OOF predictions ──
    oof_df = pd.DataFrame({"cell_id": cell_ids})
    for ci, cn in enumerate(CLASS_NAMES):
        oof_df[f"{cn}_pred"] = oof[:, ci]
    oof_path = os.path.join(OUT_DIR, "oof_predictions.parquet")
    oof_df.to_parquet(oof_path, index=False)
    print(f"\nSaved OOF predictions: {oof_path}")

    # ── Aggregate metrics ──
    r2_all = [fm["r2_uniform"] for fm in fold_metrics]
    mae_all = [fm["mae_mean_pp"] for fm in fold_metrics]
    print(f"\nMean R2: {np.mean(r2_all):.4f} +/- {np.std(r2_all):.4f}")
    print(f"Mean MAE: {np.mean(mae_all):.3f}pp")

    # ── Save metadata ──
    meta = {
        "model": "LightGBM",
        "config": "base",
        "feature_set": "VegIdx+RedEdge+TC+NDTI+IRECI+CRI1",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "seed": SEED,
        "r2_mean": float(np.mean(r2_all)),
        "r2_std": float(np.std(r2_all)),
        "mae_mean_pp": float(np.mean(mae_all)),
        "fold_metrics": fold_metrics,
        "hyperparameters": {k: v for k, v in BEST_PARAMS.items()
                           if k not in ("verbosity", "random_state", "n_jobs")},
    }
    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
