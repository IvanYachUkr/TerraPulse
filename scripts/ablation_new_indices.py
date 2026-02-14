"""
Ablation of the 6 NEW spectral indices across ALL 5 folds.

New indices: EVI2, MNDWI, GNDVI, NDTI, IRECI, CRI1
(each has 5 stats × 6 composites = 30 features)

Tests every possible combination (2^6 - 1 = 63) individually and combined.
Uses the same HistGradientBoostingRegressor hyperparameters as previous
ablation scripts for consistency.

Output: reports/phase8/tables/ablation_new_indices.csv
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROJECT_ROOT  # noqa: E402
from src.splitting import get_fold_indices  # noqa: E402
from src.models.evaluation import evaluate_model  # noqa: E402

SEED = 42
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
PARQUET_PATH = os.path.join(V2_DIR, "features_bands_indices_v2.parquet")
OUT_CSV = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables",
                       "ablation_new_indices.csv")

# The 6 new indices (not in features_merged_full.parquet)
NEW_INDICES = ["EVI2", "MNDWI", "GNDVI", "NDTI", "IRECI", "CRI1"]


def get_index_columns(all_cols, index_name):
    """Get all columns belonging to a specific index."""
    return [c for c in all_cols if c.startswith(f"{index_name}_")]


def main():
    t0_total = time.time()

    # Load the v2 feature file
    print("Loading features_bands_indices_v2.parquet...", flush=True)
    feat_df = pd.read_parquet(PARQUET_PATH)
    all_cols = [c for c in feat_df.columns if c != "cell_id"]

    # Identify columns for each new index
    index_cols = {}
    for idx_name in NEW_INDICES:
        cols = get_index_columns(all_cols, idx_name)
        index_cols[idx_name] = cols
        print(f"  {idx_name}: {len(cols)} features", flush=True)

    # Build ALL combinations (2^6 - 1 = 63)
    experiments = []
    for r in range(1, len(NEW_INDICES) + 1):
        for combo in itertools.combinations(NEW_INDICES, r):
            name = "+".join(combo)
            cols = []
            for idx_name in combo:
                cols.extend(index_cols[idx_name])
            experiments.append((name, cols))

    print(f"\n{len(experiments)} experiments (all combos of {len(NEW_INDICES)} indices)",
          flush=True)

    # Prepare feature matrix (only new index columns)
    all_new_cols = []
    for idx_name in NEW_INDICES:
        all_new_cols.extend(index_cols[idx_name])

    X_all = np.nan_to_num(feat_df[all_new_cols].values.astype(np.float32), 0.0)
    # Map column names to indices for fast slicing
    col_to_idx = {c: i for i, c in enumerate(all_new_cols)}

    # Load splits + labels
    print("Loading splits + labels...", flush=True)
    labels_df = pd.read_parquet(os.path.join(V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    # Run all folds × all experiments
    print(f"\n{'='*80}", flush=True)
    print(f"{'Fold':>4s} {'Index Combo':<45s} {'N feat':>7s} {'R2':>8s} {'MAE':>8s} {'Time':>6s}",
          flush=True)
    print(f"{'='*80}", flush=True)

    results = []
    for fold_id in range(5):
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )
        print(f"\n--- Fold {fold_id} (train={len(train_idx)}, test={len(test_idx)}) ---",
              flush=True)

        for name, cols in experiments:
            t0 = time.time()

            # Select columns by index
            feat_idx = [col_to_idx[c] for c in cols]
            X = X_all[:, feat_idx]

            model = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=300, max_depth=6, learning_rate=0.05,
                    min_samples_leaf=20, l2_regularization=0.1,
                    random_state=SEED, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30,
                )
            )
            model.fit(X[train_idx], y[train_idx])
            y_pred = np.clip(model.predict(X[test_idx]), 0, 100)

            summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
            r2 = summary["r2_uniform"]
            mae = summary["mae_mean_pp"]
            elapsed = time.time() - t0

            print(f"  F{fold_id} {name:<45s} {len(cols):>7d} {r2:>8.4f} {mae:>8.3f} {elapsed:>5.1f}s",
                  flush=True)
            results.append({
                "fold": fold_id, "group": name, "n_features": len(cols),
                "n_indices": name.count("+") + 1,
                "r2": r2, "mae": mae, "time": elapsed,
            })

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}", flush=True)
    print(f"Total time: {time.time() - t0_total:.0f}s", flush=True)

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY (mean across 5 folds):", flush=True)
    print(f"{'='*80}", flush=True)
    res_df = pd.DataFrame(results)
    summary = res_df.groupby("group").agg(
        n_features=("n_features", "first"),
        n_indices=("n_indices", "first"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
        mean_mae=("mae", "mean"),
    ).sort_values("mean_r2", ascending=False)
    print(summary.to_string(), flush=True)

    # Best per number of indices
    print(f"\n{'='*80}", flush=True)
    print("BEST COMBO PER SIZE:", flush=True)
    print(f"{'='*80}", flush=True)
    for n in range(1, 7):
        sub = summary[summary["n_indices"] == n]
        if len(sub) > 0:
            best = sub.sort_values("mean_r2", ascending=False).iloc[0]
            print(f"  {n} index(es): {best.name:<45s} R2={best['mean_r2']:.4f} "
                  f"(±{best['std_r2']:.4f}) feat={int(best['n_features'])}",
                  flush=True)


if __name__ == "__main__":
    main()
