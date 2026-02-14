#!/usr/bin/env python3
"""
MLP V8: Auto-select best MLP feature set, then run tree models on it.

Waits for V7 to finish, compares V5, V6, V7 MLP results to pick the
winning feature set, then trains tree/CatBoost models on that feature
set across all 5 spatial folds.

Feature sets compared:
  - V5:  bands_indices_glcm_lbp (from features_merged_full.parquet)
  - V6:  bi_glcm_morph          (bands + indices + GLCM + Morph, no LBP)
  - V7:  bi_glcm_morph_v2       (V6 + NDTI + IRECI + CRI1 from v2)

Tree models:
  - ExtraTrees (500, 1000 estimators)
  - RandomForest (500)
  - CatBoost (1000 iterations, also deeper variant)

Output:
    reports/phase8/tables/mlp_v8_trees.csv
    reports/phase8/tables/mlp_v8_trees_summary.csv

Usage:
    .venv\\Scripts\\python.exe scripts/run_mlp_v8_trees.py
    .venv\\Scripts\\python.exe scripts/run_mlp_v8_trees.py --no-wait
    .venv\\Scripts\\python.exe scripts/run_mlp_v8_trees.py --force-features bi_glcm_morph
"""

import argparse
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
from src.splitting import get_fold_indices
from src.transforms import helmert_basis, ilr_forward

# Reuse partition_features from V4
sys.path.insert(0, os.path.dirname(__file__))
from run_mlp_overnight_v4 import partition_features as mlp_partition_features

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
V8_CSV = os.path.join(OUT_DIR, "mlp_v8_trees.csv")
V8_SUMMARY = os.path.join(OUT_DIR, "mlp_v8_trees_summary.csv")

# Previous run CSVs
V5_CSV = os.path.join(OUT_DIR, "mlp_v5_deep.csv")
V6_CSV = os.path.join(OUT_DIR, "mlp_v6_arch.csv")
V7_CSV = os.path.join(OUT_DIR, "mlp_v7_arch.csv")
V7_EXPECTED_RUNS = 50

# Novel v2 index prefixes (same as V7)
NOVEL_V2_PREFIXES = ["NDTI_", "IRECI_", "CRI1_"]


# =====================================================================
# Wait for V7
# =====================================================================

def wait_for_v7(poll_interval=60):
    """Poll V7 CSV until all runs are done."""
    print(f"Waiting for V7 to finish ({V7_EXPECTED_RUNS} runs)...", flush=True)
    while True:
        if os.path.exists(V7_CSV):
            try:
                df = pd.read_csv(V7_CSV)
                n = len(df)
                if n >= V7_EXPECTED_RUNS:
                    n_valid = df["r2_uniform"].notna().sum()
                    print(f"  V7 done! {n_valid} valid = {n} total", flush=True)
                    return
                print(f"  V7: {n}/{V7_EXPECTED_RUNS} runs done, waiting...", flush=True)
            except Exception:
                pass
        else:
            print(f"  V7 CSV not found yet, waiting...", flush=True)
        time.sleep(poll_interval)


# =====================================================================
# Feature set comparison and selection
# =====================================================================

def compare_and_select_features():
    """Compare V5, V6, V7 MLP results to pick the best feature set.

    Returns the winning feature set name.
    """
    print("\n" + "=" * 70, flush=True)
    print("COMPARING MLP RESULTS: V5 vs V6 vs V7", flush=True)
    print("=" * 70, flush=True)

    results = {}

    # V5: best feature set is bands_indices_glcm_lbp
    if os.path.exists(V5_CSV):
        v5 = pd.read_csv(V5_CSV)
        v5v = v5[v5.r2_uniform.notna()]
        # Get mean R2 of the best config in V5
        v5_agg = v5v.groupby("name").agg(
            r2=("r2_uniform", "mean"), n=("fold", "count"),
            feat=("feature_set", "first")
        )
        v5_best = v5_agg[v5_agg.n == v5_agg.n.max()].sort_values("r2", ascending=False).iloc[0]
        results["v5_glcm_lbp"] = {
            "r2": v5_best.r2, "feat": v5_best.feat, "config": v5_best.name,
            "folds": int(v5_best.n)
        }
        print(f"  V5 best: R2={v5_best.r2:.4f}  {v5_best.name} ({v5_best.feat})", flush=True)

    # V6: bi_glcm_morph
    if os.path.exists(V6_CSV):
        v6 = pd.read_csv(V6_CSV)
        v6v = v6[v6.r2_uniform.notna()]
        v6_agg = v6v.groupby("name").agg(
            r2=("r2_uniform", "mean"), n=("fold", "count"),
            feat=("feature_set", "first")
        )
        v6_best = v6_agg[v6_agg.n == v6_agg.n.max()].sort_values("r2", ascending=False).iloc[0]
        results["v6_bi_glcm_morph"] = {
            "r2": v6_best.r2, "feat": v6_best.feat, "config": v6_best.name,
            "folds": int(v6_best.n)
        }
        print(f"  V6 best: R2={v6_best.r2:.4f}  {v6_best.name} ({v6_best.feat})", flush=True)

    # V7: bi_glcm_morph + v2 novel
    if os.path.exists(V7_CSV):
        v7 = pd.read_csv(V7_CSV)
        v7v = v7[v7.r2_uniform.notna()]
        v7_agg = v7v.groupby("name").agg(
            r2=("r2_uniform", "mean"), n=("fold", "count"),
            feat=("feature_set", "first")
        )
        v7_best = v7_agg[v7_agg.n == v7_agg.n.max()].sort_values("r2", ascending=False).iloc[0]
        results["v7_bi_glcm_morph_v2"] = {
            "r2": v7_best.r2, "feat": v7_best.feat, "config": v7_best.name,
            "folds": int(v7_best.n)
        }
        print(f"  V7 best: R2={v7_best.r2:.4f}  {v7_best.name} ({v7_best.feat})", flush=True)

    if not results:
        raise RuntimeError("No MLP results found!")

    # Pick winner
    winner_key = max(results, key=lambda k: results[k]["r2"])
    winner = results[winner_key]
    print(f"\n  >>> WINNER: {winner_key} with R2={winner['r2']:.4f}", flush=True)
    print(f"      Feature set: {winner['feat']}", flush=True)
    print(f"      Best config: {winner['config']}", flush=True)
    print(f"      Folds used:  {winner['folds']}", flush=True)

    return winner_key, winner


# =====================================================================
# Feature set builders
# =====================================================================

def build_v5_features(full_feature_cols):
    """bands_indices + GLCM + LBP (V5 champion feature set)."""
    groups = mlp_partition_features(full_feature_cols)
    return groups["bands_indices_glcm_lbp"]


def build_v6_features(full_feature_cols):
    """bands_indices + GLCM + Morph (no LBP)."""
    groups = mlp_partition_features(full_feature_cols)
    base_idx = set(groups["bands_indices"])
    glcm_idx = {i for i, c in enumerate(full_feature_cols) if "GLCM_" in c}
    morph_idx = {i for i, c in enumerate(full_feature_cols) if "MP_" in c}
    return sorted(base_idx | glcm_idx | morph_idx)


def build_v7_features(full_feature_cols, v2_only_cols):
    """V6 features + NDTI/IRECI/CRI1 from v2."""
    bi_glcm_morph = build_v6_features(full_feature_cols)
    novel_v2 = [c for c in v2_only_cols
                if any(c.startswith(p) for p in NOVEL_V2_PREFIXES)]
    n_base = len(full_feature_cols)
    novel_indices = list(range(n_base, n_base + len(novel_v2)))
    return bi_glcm_morph + novel_indices, novel_v2


# =====================================================================
# Tree models
# =====================================================================

def build_models():
    """Return list of (name, model_instance) to test."""
    basis = helmert_basis(N_CLASSES)
    return [
        ("extratrees_500", ExtraTreesModel(n_estimators=500, basis=basis)),
        ("rf_500", RandomForestModel(n_estimators=500, basis=basis)),
        ("catboost_1000", CatBoostModel(iterations=1000, basis=basis)),
        ("extratrees_500_third", ExtraTreesModel(
            n_estimators=500, max_features=0.33, basis=basis)),
        ("rf_500_third", RandomForestModel(
            n_estimators=500, max_features=0.33, basis=basis)),
        ("extratrees_1000", ExtraTreesModel(n_estimators=1000, basis=basis)),
        ("catboost_deeper", CatBoostModel(
            iterations=1000, depth=8, learning_rate=0.05, basis=basis)),
    ]


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="V8: Auto-select features, run trees")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--force-features", type=str, default=None,
                        choices=["v5_glcm_lbp", "v6_bi_glcm_morph", "v7_bi_glcm_morph_v2"],
                        help="Override auto-selection")
    args = parser.parse_args()

    # ── Wait for V7 ──
    if not args.no_wait:
        wait_for_v7(poll_interval=60)
    else:
        print("Skipping V7 wait (--no-wait)", flush=True)

    # ── Compare and select features ──
    if args.force_features:
        winner_key = args.force_features
        winner = {"feat": winner_key, "r2": None, "config": "forced"}
        print(f"  Forced feature set: {winner_key}", flush=True)
    else:
        winner_key, winner = compare_and_select_features()

    folds_to_run = args.folds if args.folds else list(range(N_FOLDS))

    # ── Load data ──
    print("\nLoading data...", flush=True)
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # Build feature indices based on winner
    needs_v2 = winner_key == "v7_bi_glcm_morph_v2"

    if needs_v2:
        v2_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet"))
        v2_only_cols = [c for c in v2_df.columns
                        if c != "cell_id" and c not in feat_df.columns]
        feat_idx, novel_v2 = build_v7_features(full_feature_cols, v2_only_cols)
        base_arr = feat_df[full_feature_cols].values.astype(np.float64)
        v2_arr = v2_df[novel_v2].values.astype(np.float64)
        X_all = np.hstack([base_arr, v2_arr])
        all_cols = full_feature_cols + novel_v2
        del v2_df, base_arr, v2_arr
    elif winner_key == "v6_bi_glcm_morph":
        feat_idx = build_v6_features(full_feature_cols)
        X_all = feat_df[full_feature_cols].values.astype(np.float64)
        all_cols = full_feature_cols
    else:  # v5_glcm_lbp
        feat_idx = build_v5_features(full_feature_cols)
        X_all = feat_df[full_feature_cols].values.astype(np.float64)
        all_cols = full_feature_cols

    del feat_df

    np.nan_to_num(X_all, copy=False)
    n_features = len(feat_idx)
    feat_col_names = [all_cols[i] for i in feat_idx]
    print(f"  Winner: {winner_key} -> {n_features} features", flush=True)

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float64)

    basis = helmert_basis(N_CLASSES)
    z = ilr_forward(y, basis=basis)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)
    n_tc, n_tr = meta["tile_cols"], meta["tile_rows"]

    print(f"Data loaded: X={X_all.shape}, y={y.shape}", flush=True)

    # ── Resume logic ──
    results = []
    done_keys = set()
    if os.path.exists(V8_CSV):
        df_old = pd.read_csv(V8_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["run_name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs already done", flush=True)

    models = build_models()
    total_runs = len(models) * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"V8 TREES — {winner_key} ({n_features} features)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Models:  {len(models)}", flush=True)
    print(f"  Folds:   {folds_to_run}", flush=True)
    print(f"  Total:   {total_runs} runs", flush=True)
    print(f"  MLP winner R2: {winner.get('r2', 'N/A')}", flush=True)
    print(f"{'='*70}\n", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    times = []
    n_skipped = 0

    for fold_id in folds_to_run:
        print(f"\n{'='*80}", flush=True)
        print(f"FOLD {fold_id}/{N_FOLDS-1}", flush=True)
        print(f"{'='*80}", flush=True)

        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, n_tc, n_tr, buffer_tiles=1,
        )

        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx = train_idx[perm[:n_val]]
        trn_idx = train_idx[perm[n_val:]]
        print(f"  Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}", flush=True)

        # Select features
        X = X_all[:, feat_idx]

        for model_name, model in build_models():
            run_name = f"{model_name}_{winner_key}"

            if (run_name, fold_id) in done_keys:
                n_skipped += 1
                continue

            t0 = time.time()
            try:
                is_catboost = "catboost" in model_name

                if is_catboost:
                    model.fit(X[trn_idx], z[trn_idx],
                              X_val=X[val_idx], z_val=z[val_idx])
                else:
                    model.fit(X[train_idx], z[train_idx])

                y_pred = model.predict_proportions(X[test_idx])
                elapsed = time.time() - t0

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)

                # Feature importance
                imp = model.feature_importances_
                top10 = [feat_col_names[i] for i in np.argsort(imp)[-10:][::-1]]

                summary.update({
                    "run_name": run_name, "fold": fold_id,
                    "stage": "v8", "model_type": model_name.split("_")[0],
                    "feature_set": winner_key,
                    "n_features": n_features,
                    "elapsed_s": round(elapsed, 1),
                    "top10_features": "; ".join(top10),
                    "mlp_winner_r2": winner.get("r2"),
                })

                results.append(summary)
                done_keys.add((run_name, fold_id))
                times.append(elapsed)

                avg_time = sum(times) / len(times)
                remaining = max(0, total_runs - n_skipped - len(times))
                eta_h = remaining * avg_time / 3600

                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {run_name:50s} "
                      f"R2={r2:.4f}  MAE={mae:.2f}pp  "
                      f"{elapsed:.0f}s  ETA={eta_h:.1f}h", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                results.append({"run_name": run_name, "fold": fold_id,
                                "stage": "v8", "error": str(e)})
                done_keys.add((run_name, fold_id))
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {run_name:50s} "
                      f"ERROR: {e} ({elapsed:.0f}s)", flush=True)

            if (n_skipped + len(times)) % 3 == 0:
                pd.DataFrame(results).to_csv(V8_CSV, index=False)

            # Free model memory
            model = None

        pd.DataFrame(results).to_csv(V8_CSV, index=False)

    # Final save
    pd.DataFrame(results).to_csv(V8_CSV, index=False)
    print(f"\nResults saved to {V8_CSV}", flush=True)

    # ── Summary ──
    df = pd.DataFrame(results)
    valid = df[df["r2_uniform"].notna()]
    if len(valid) > 0:
        agg = (valid.groupby("run_name")
               .agg(
                   r2_mean=("r2_uniform", "mean"),
                   r2_std=("r2_uniform", "std"),
                   mae_mean=("mae_mean_pp", "mean"),
                   folds=("fold", "count"),
                   model_type=("model_type", "first"),
               )
               .sort_values("r2_mean", ascending=False))
        agg.to_csv(V8_SUMMARY)
        print(f"Summary saved to {V8_SUMMARY}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"V8 TREE RESULTS — {winner_key} ({n_features} features)", flush=True)
        print(f"{'='*70}", flush=True)
        for _, row in agg.iterrows():
            print(f"  {row.name:50s} R2={row['r2_mean']:.4f}+/-{row['r2_std']:.4f}  "
                  f"MAE={row['mae_mean']:.2f}  folds={int(row['folds'])}  "
                  f"[{row['model_type']}]", flush=True)

    n_errors = len(df) - len(valid)
    if n_errors:
        print(f"\nERRORS: {n_errors}", flush=True)

    total_time = sum(times)
    if total_time > 0:
        print(f"\nTotal compute time: {total_time/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
