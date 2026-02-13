"""
Texture ablation across ALL 5 folds.
Tests: Morph alone, GLCM alone, Gabor-s1 (small kernel), Morph+GLCM, Morph+GLCM+Gabor-s1
Each tree trains ONLY on that texture group — pure isolation.
Outputs CSV for report generation.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_V2_DIR, PROJECT_ROOT  # noqa: E402
from src.splitting import get_fold_indices  # noqa: E402
from src.models.evaluation import evaluate_model  # noqa: E402

SEED = 42
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
PARQUET_PATH = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")
OUT_CSV = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables",
                       "ablation_texture_all_folds.csv")


def main():
    t0_total = time.time()

    # Read schema
    schema = pq.read_schema(PARQUET_PATH)
    feat_cols = [c for c in schema.names if c != "cell_id"]

    # Define texture groups
    glcm = [c for c in feat_cols if "GLCM_" in c]
    morph = [c for c in feat_cols if "MP_" in c]
    gabor_s1 = [c for c in feat_cols if "Gabor_s1_" in c]  # Only sigma=1 (small kernel)

    experiments = {
        "GLCM": glcm,
        "Morph": morph,
        "Gabor_s1": gabor_s1,
        "GLCM+Morph": glcm + morph,
        "GLCM+Morph+Gabor_s1": glcm + morph + gabor_s1,
    }

    print("Texture groups:", flush=True)
    for name, cols in experiments.items():
        print(f"  {name}: {len(cols)} features", flush=True)

    # Load splits + labels
    print("\nLoading splits + labels...", flush=True)
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    # Run all folds × all experiments
    print(f"\n{'='*75}", flush=True)
    print(f"{'Fold':>4s} {'Texture Group':<25s} {'N feat':>7s} {'R2':>8s} {'MAE':>8s} {'Time':>6s}",
          flush=True)
    print(f"{'='*75}", flush=True)

    results = []
    for fold_id in range(5):
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )

        for name, cols in experiments.items():
            t0 = time.time()
            df = pd.read_parquet(PARQUET_PATH, columns=cols)
            X = np.nan_to_num(df.values.astype(np.float32), 0.0)

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

            print(f"  F{fold_id} {name:<25s} {len(cols):>7d} {r2:>8.4f} {mae:>8.3f} {elapsed:>5.1f}s",
                  flush=True)
            results.append({
                "fold": fold_id, "group": name, "n_features": len(cols),
                "r2": r2, "mae": mae, "time": elapsed,
            })

            del df, X, model

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}", flush=True)
    print(f"Total time: {time.time() - t0_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
