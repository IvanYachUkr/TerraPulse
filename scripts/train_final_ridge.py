#!/usr/bin/env python3
"""
Train & save interpretable Ridge regression model for the report.

Uses the same bi_LBP feature set (864 features) as the MLP champion
for apples-to-apples comparison.

Produces:
    models/final_ridge/fold_{i}.pkl       - Ridge model per fold
    models/final_ridge/scaler_{i}.pkl     - StandardScaler per fold
    models/final_ridge/oof_predictions.parquet
    models/final_ridge/meta.json          - metadata + per-fold metrics
    models/final_ridge/coefficients.csv   - feature coefficients (avg across folds)
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from run_mlp_overnight_v4 import (
    PROJECT_ROOT, CLASS_NAMES, N_FOLDS, CONTROL_COLS,
    partition_features,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

SEED = 42
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final_ridge")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_bi_lbp(full_feature_cols):
    groups = partition_features(full_feature_cols)
    base_idx = groups["bands_indices"]
    lbp_idx = [i for i, c in enumerate(full_feature_cols) if c.startswith("LBP_")]
    return sorted(set(base_idx) | set(lbp_idx))


def main():
    print("Loading data...")
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    fs_idx = build_bi_lbp(full_feature_cols)
    fs_cols = [full_feature_cols[i] for i in fs_idx]
    n_features = len(fs_idx)
    print("Features: {} (bi_LBP)".format(n_features))

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    cell_ids = feat_df["cell_id"].values if "cell_id" in feat_df.columns else np.arange(len(feat_df))
    del feat_df

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    np.random.seed(SEED)

    oof_preds = np.full((len(y), len(CLASS_NAMES)), np.nan, dtype=np.float32)
    fold_metrics = []
    all_coefs = []

    for fold_id in range(N_FOLDS):
        print("\n--- Fold {} ---".format(fold_id))
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )
        print("  Train: {}, Test: {}".format(len(train_idx), len(test_idx)))

        X_fs = X_all[:, fs_idx]
        scaler = StandardScaler()
        X_trn = scaler.fit_transform(X_fs[train_idx])
        X_test = scaler.transform(X_fs[test_idx])

        # Save scaler
        with open(os.path.join(MODEL_DIR, "scaler_{}.pkl".format(fold_id)), "wb") as f:
            pickle.dump(scaler, f)

        # Train Ridge with built-in CV for alpha
        ridge = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            cv=5,
        )
        ridge.fit(X_trn, y[train_idx])
        print("  Best alpha: {}".format(ridge.alpha_))

        # Save model
        with open(os.path.join(MODEL_DIR, "fold_{}.pkl".format(fold_id)), "wb") as f:
            pickle.dump(ridge, f)

        # Predict
        preds = ridge.predict(X_test).clip(0, 1).astype(np.float32)
        # Renormalize to sum to 1
        row_sums = preds.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
        preds = preds / row_sums

        oof_preds[test_idx] = preds

        metrics, _ = evaluate_model(y[test_idx], preds, CLASS_NAMES)
        r2 = metrics["r2_uniform"]
        mae = metrics["mae_mean_pp"]
        fold_metrics.append({
            "fold": fold_id, "r2": r2, "mae": mae,
            "alpha": float(ridge.alpha_),
        })
        print("  R2={:.4f}  MAE={:.2f}pp".format(r2, mae))

        all_coefs.append(ridge.coef_)

    # ── Save OOF predictions ──
    oof_df = pd.DataFrame(oof_preds, columns=[c + "_pred" for c in CLASS_NAMES])
    oof_df.insert(0, "cell_id", cell_ids)
    oof_df.to_parquet(os.path.join(MODEL_DIR, "oof_predictions.parquet"), index=False)

    # ── Save coefficients ──
    mean_coefs = np.mean(all_coefs, axis=0)  # shape: (n_classes, n_features)
    coef_df_records = []
    for j, feat in enumerate(fs_cols):
        row = {"feature": feat}
        for k, cls in enumerate(CLASS_NAMES):
            row[cls] = mean_coefs[k, j]
        row["abs_mean"] = np.mean(np.abs(mean_coefs[:, j]))
        coef_df_records.append(row)
    coef_df = pd.DataFrame(coef_df_records).sort_values("abs_mean", ascending=False)
    coef_df.to_csv(os.path.join(MODEL_DIR, "coefficients.csv"), index=False)

    # ── Save metadata ──
    r2_mean = np.mean([m["r2"] for m in fold_metrics])
    r2_std = np.std([m["r2"] for m in fold_metrics])
    mae_mean = np.mean([m["mae"] for m in fold_metrics])
    meta_out = {
        "model": "RidgeCV",
        "feature_set": "bi_LBP",
        "feature_cols": fs_cols,
        "n_features": n_features,
        "seed": SEED,
        "r2_mean": round(r2_mean, 4),
        "r2_std": round(r2_std, 4),
        "mae_mean_pp": round(mae_mean, 2),
        "fold_metrics": fold_metrics,
        "data_source": "features_merged_full.parquet",
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RIDGE -- 5-fold CV")
    print("=" * 60)
    for m in fold_metrics:
        print("  Fold {}: R2={:.4f}  MAE={:.2f}pp  alpha={}".format(
            m["fold"], m["r2"], m["mae"], m["alpha"]))
    print("  " + "-" * 35)
    print("  Mean:   R2={:.4f} +/- {:.4f}  MAE={:.2f}pp".format(r2_mean, r2_std, mae_mean))
    print("=" * 60)
    print("\nTop 20 features by |coef|:")
    print(coef_df.head(20)[["feature", "abs_mean"]].to_string(index=False))
    print("\nAll saved to: {}".format(MODEL_DIR))


if __name__ == "__main__":
    main()
