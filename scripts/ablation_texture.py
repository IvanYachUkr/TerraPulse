"""
Pure texture ablation: train trees on EACH texture group ALONE on fold 4.
No base features — purely isolating each texture type's predictive power.
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
from src.config import PROCESSED_V2_DIR  # noqa: E402
from src.splitting import get_fold_indices  # noqa: E402
from src.models.evaluation import evaluate_model  # noqa: E402

SEED = 42
FOLD_ID = 4
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
PARQUET_PATH = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")


def main():
    t0_total = time.time()

    # Read column names from schema (no data loaded)
    schema = pq.read_schema(PARQUET_PATH)
    feat_cols = [c for c in schema.names if c != "cell_id"]

    # Define texture groups
    groups = {
        "GLCM":      [c for c in feat_cols if "GLCM_" in c],
        "LBP":       [c for c in feat_cols if "LBP_" in c],
        "Gabor":     [c for c in feat_cols if "Gabor_" in c],
        "HOG":       [c for c in feat_cols if "HOG_" in c],
        "Morph":     [c for c in feat_cols if "MP_" in c],
        "Semivario": [c for c in feat_cols if "SV_" in c],
    }

    for name, cols in groups.items():
        print(f"  {name}: {len(cols)} features", flush=True)

    # Load splits + labels
    print("Loading splits + labels...", flush=True)
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    train_idx, test_idx = get_fold_indices(
        tiles, folds_arr, FOLD_ID, meta["tile_cols"], meta["tile_rows"],
        buffer_tiles=1,
    )
    print(f"Fold {FOLD_ID}: train={len(train_idx)}, test={len(test_idx)}", flush=True)

    # Run each texture group ALONE
    print(f"\n{'='*55}", flush=True)
    print(f"{'Texture Group':<15s} {'N feat':>7s} {'R²':>8s} {'MAE':>8s} {'Time':>6s}", flush=True)
    print(f"{'='*55}", flush=True)

    results = []
    for name, cols in groups.items():
        t0 = time.time()

        # Load ONLY this texture group's columns
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

        print(f"{name:<15s} {len(cols):>7d} {r2:>8.4f} {mae:>8.3f} {elapsed:>5.1f}s", flush=True)
        results.append({"group": name, "n_features": len(cols), "r2": r2, "mae": mae})

        del df, X, model

    print(f"\nTotal time: {time.time() - t0_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
