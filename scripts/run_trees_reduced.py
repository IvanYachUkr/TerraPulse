"""
Run trees + CatBoost on reduced feature sets (bands+indices = 798 features).

Compared to the previous overnight run (2,109 core features), this tests
whether removing the 1,251 delta features improves tree model performance,
as discovered in the MLP sweep.

Runs: ExtraTrees, RF, CatBoost with default HPs on bands+indices (798 features).
Also tests bands-only (480) and indices-only (318) as ablations.

Usage:
    python scripts/run_trees_reduced.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CFG, PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.boosting import CatBoostModel
from src.models.evaluation import evaluate_model
from src.models.forests import ExtraTreesModel, RandomForestModel
from src.models.spatial_diagnostics import compute_residual_morans_i
from src.splitting import get_fold_indices
from src.transforms import helmert_basis, ilr_forward

CLASS_NAMES = CFG["worldcover"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
SPLIT_CFG = CFG["split"]
SEED = SPLIT_CFG["seed"]
N_FOLDS = SPLIT_CFG["n_folds"]

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
OUT_CSV = os.path.join(OUT_DIR, "trees_reduced_features.csv")


# =====================================================================
# Feature partitioning
# =====================================================================

def partition_features(all_cols):
    """Split feature columns into semantic groups."""
    bands, indices, deltas, other = [], [], [], []
    band_prefixes = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
    index_prefixes = {
        "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
        "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
    }
    for i, col in enumerate(all_cols):
        prefix = col.split("_")[0]
        if col.startswith("delta"):
            deltas.append(i)
        elif prefix in band_prefixes:
            bands.append(i)
        elif prefix in index_prefixes:
            indices.append(i)
        else:
            other.append(i)

    return {
        "all_core": list(range(len(all_cols))),
        "bands_and_indices": bands + indices,
        "bands_only": bands,
        "indices_only": indices,
    }


# =====================================================================
# Model configurations
# =====================================================================

def build_models():
    """Return list of (name, model_instance) to test."""
    basis = helmert_basis(N_CLASSES)
    return [
        ("extratrees_500", ExtraTreesModel(n_estimators=500, basis=basis)),
        ("rf_500", RandomForestModel(n_estimators=500, basis=basis)),
        ("catboost_1000", CatBoostModel(iterations=1000, basis=basis)),
        # Wider trees to test if more features per split helps
        ("extratrees_500_third", ExtraTreesModel(
            n_estimators=500, max_features=0.33, basis=basis)),
        ("rf_500_third", RandomForestModel(
            n_estimators=500, max_features=0.33, basis=basis)),
        # More trees for stability
        ("extratrees_1000", ExtraTreesModel(n_estimators=1000, basis=basis)),
        # CatBoost with more depth
        ("catboost_deeper", CatBoostModel(
            iterations=1000, depth=8, learning_rate=0.05, basis=basis)),
    ]


# =====================================================================
# Main
# =====================================================================

def main():
    # Load data
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_core.parquet"))
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))

    feature_cols = [c for c in feat_df.columns
                    if c not in CONTROL_COLS
                    and feat_df[c].dtype in ("float64", "float32", "int64")]
    X_all = feat_df[feature_cols].values.astype(np.float64)
    y = labels_df[CLASS_NAMES].values.astype(np.float64)

    basis = helmert_basis(N_CLASSES)
    z = ilr_forward(y, basis=basis)

    fold_assignments = split_df["fold_region_growing"].values
    tile_groups = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)
    n_tc, n_tr = meta["tile_cols"], meta["tile_rows"]

    feat_groups = partition_features(feature_cols)
    print(f"Feature groups: " + ", ".join(f"{k}={len(v)}" for k, v in feat_groups.items()))

    # Train/test split (fold 0 = test)
    train_idx, test_idx = get_fold_indices(
        tile_groups, fold_assignments, 0, n_tc, n_tr, buffer_tiles=1,
    )

    # Validation split for CatBoost early stopping
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_idx))
    n_val = max(int(len(train_idx) * 0.15), 100)
    val_idx = train_idx[perm[:n_val]]
    trn_idx = train_idx[perm[n_val:]]
    print(f"Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Load existing results if resuming
    os.makedirs(OUT_DIR, exist_ok=True)
    existing_results = []
    completed_names = set()
    if os.path.exists(OUT_CSV):
        existing_df = pd.read_csv(OUT_CSV)
        existing_results = existing_df.to_dict("records")
        completed_names = set(existing_df["run_name"].values)
        print(f"Found {len(completed_names)} completed runs, resuming...")

    results = existing_results

    # Run all combinations
    for feat_name, feat_idx in feat_groups.items():
        if len(feat_idx) == 0:
            continue

        X = X_all[:, feat_idx]
        n_features = X.shape[1]

        for model_name, model in build_models():
            run_name = f"{model_name}_feat_{feat_name}"
            if run_name in completed_names:
                print(f"SKIP {run_name} (already done)")
                continue

            print(f"\n--- {run_name} ({n_features} features) ---")
            t0 = time.time()

            # Train
            is_catboost = "catboost" in model_name
            if is_catboost:
                model.fit(X[trn_idx], z[trn_idx],
                          X_val=X[val_idx], z_val=z[val_idx])
            else:
                model.fit(X[train_idx], z[train_idx])

            # Predict
            y_pred = model.predict_proportions(X[test_idx])
            elapsed = time.time() - t0

            # Evaluate
            summary, detail = evaluate_model(y[test_idx], y_pred, CLASS_NAMES, model_name=run_name)

            # Spatial diagnostics
            mi_result, mi_mean = compute_residual_morans_i(
                y[test_idx], y_pred, tile_groups[test_idx], n_tc, n_tr, CLASS_NAMES,
            )

            # Feature importance
            imp = model.feature_importances_
            top10_features = [feature_cols[feat_idx[i]] for i in np.argsort(imp)[-10:][::-1]]

            summary.update({
                "run_name": run_name,
                "model_type": model_name.split("_")[0],
                "feature_set": feat_name,
                "n_features": n_features,
                "morans_i_mean": mi_mean,
                "elapsed_s": round(elapsed, 1),
                "top10_features": "; ".join(top10_features),
            })

            results.append(summary)
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)

            r2 = summary["r2_uniform"]
            mae = summary["mae_mean_pp"]
            print(f"  R2={r2:.4f}  MAE={mae:.2f}pp  Moran={mi_mean:.4f}  time={elapsed:.0f}s")

            # Reset model for next run
            model = None

    # Final summary
    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("RESULTS (sorted by R2):")
    print(df.sort_values("r2_uniform", ascending=False)[
        ["run_name", "r2_uniform", "mae_mean_pp", "aitchison_mean",
         "n_features", "morans_i_mean", "elapsed_s"]
    ].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
