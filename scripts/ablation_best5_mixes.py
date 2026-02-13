"""
Best-5 feature group mix ablation across ALL 5 folds.

Tests every useful combination of the top 5 feature groups:
  Texture:  Morph, GLCM
  Spectral: Veg idx, Red-edge (B05-07,8A), Tasseled Cap

Combinations tested (10 total):
  - 1 texture + 1 spectral  (2 x 3 = 6)
  - 2 textures + 2 spectral (C(3,2) = 3)
  - All 5 combined           (1)

Uses the same HistGradientBoostingRegressor hyperparameters as the
previous ablation scripts for consistency.

Outputs CSV to reports/phase8/tables/ablation_best5_mixes.csv
"""

import itertools
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
                       "ablation_best5_mixes.csv")


def build_feature_groups(feat_cols):
    """Identify the 5 best feature groups from column names."""
    import re
    band_pat = re.compile(r'^B(05|06|07|8A)_')

    morph = [c for c in feat_cols if "MP_" in c]
    glcm = [c for c in feat_cols if "GLCM_" in c]

    # Vegetation indices: NDVI (excluding range/iqr which are spatial), SAVI, NDRE
    veg_idx = [c for c in feat_cols
               if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"])
               and not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr")]

    # Red-edge bands: B05, B06, B07, B8A
    rededge = [c for c in feat_cols if band_pat.match(c)]

    # Tasseled Cap
    tc = [c for c in feat_cols if c.startswith("TC_")]

    return {
        "Morph": morph,
        "GLCM": glcm,
        "VegIdx": veg_idx,
        "RedEdge": rededge,
        "TC": tc,
    }


def build_experiments(groups):
    """Build all requested combinations."""
    textures = ["Morph", "GLCM"]
    spectrals = ["VegIdx", "RedEdge", "TC"]

    experiments = []

    # --- 1 texture + 1 spectral (6 combos) ---
    for tex in textures:
        for spec in spectrals:
            name = f"{tex}+{spec}"
            cols = groups[tex] + groups[spec]
            experiments.append((name, cols))

    # --- 2 textures + 2 spectral (3 combos) ---
    for spec_pair in itertools.combinations(spectrals, 2):
        name = "GLCM+Morph+" + "+".join(spec_pair)
        cols = groups["GLCM"] + groups["Morph"]
        for s in spec_pair:
            cols = cols + groups[s]
        experiments.append((name, cols))

    # --- All 5 combined (1 combo) ---
    name = "GLCM+Morph+VegIdx+RedEdge+TC"
    cols = []
    for g in ["GLCM", "Morph", "VegIdx", "RedEdge", "TC"]:
        cols = cols + groups[g]
    experiments.append((name, cols))

    return experiments


def main():
    t0_total = time.time()

    # Read schema to get feature columns
    schema = pq.read_schema(PARQUET_PATH)
    feat_cols = [c for c in schema.names if c != "cell_id"]

    groups = build_feature_groups(feat_cols)
    print("Feature groups:", flush=True)
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)} features", flush=True)

    experiments = build_experiments(groups)
    print(f"\n{len(experiments)} experiments:", flush=True)
    for name, cols in experiments:
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

    # Run all folds x all experiments
    print(f"\n{'='*80}", flush=True)
    print(f"{'Fold':>4s} {'Feature Mix':<40s} {'N feat':>7s} {'R2':>8s} {'MAE':>8s} {'Time':>6s}",
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

            print(f"  F{fold_id} {name:<40s} {len(cols):>7d} {r2:>8.4f} {mae:>8.3f} {elapsed:>5.1f}s",
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

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY (mean across 5 folds):", flush=True)
    print(f"{'='*80}", flush=True)
    res_df = pd.DataFrame(results)
    summary = res_df.groupby("group").agg(
        n_features=("n_features", "first"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
        mean_mae=("mae", "mean"),
    ).sort_values("mean_r2", ascending=False)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
