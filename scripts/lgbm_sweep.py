"""
LightGBM hyperparameter sweep on the best feature set
(VegIdx+RedEdge+TC + NDTI+IRECI+CRI1, 438 features).

Also tests the top-3 feature sets to confirm ranking holds with LightGBM.

Output: reports/phase8/tables/lgbm_sweep.csv
"""

import itertools
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_V2_DIR, PROJECT_ROOT  # noqa: E402
from src.splitting import get_fold_indices  # noqa: E402
from src.models.evaluation import evaluate_model  # noqa: E402

SEED = 42
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
FULL_PQ = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")
V2_PQ = os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet")
OUT_CSV = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables",
                       "lgbm_sweep.csv")

NOVEL_INDICES = ["NDTI", "IRECI", "CRI1"]


# ── Feature set builders ────────────────────────────────────────────────────

def build_feature_groups(feat_cols):
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    veg_idx = [c for c in feat_cols
               if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"])
               and not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr")]
    rededge = [c for c in feat_cols if band_pat.match(c)]
    tc = [c for c in feat_cols if c.startswith("TC_")]
    glcm = [c for c in feat_cols if "GLCM_" in c]
    morph = [c for c in feat_cols if "MP_" in c]
    return {
        "VegIdx": veg_idx, "RedEdge": rededge, "TC": tc,
        "GLCM": glcm, "Morph": morph,
    }


def get_novel_cols(v2_cols, index_name):
    return [c for c in v2_cols if c.startswith(f"{index_name}_")]


# ── Hyperparameter configs ──────────────────────────────────────────────────

def generate_configs():
    """~30 configs covering key LightGBM hyperparameter axes."""
    configs = []

    # Base config (similar to our HistGBR baseline)
    base = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, reg_lambda=0.1,
        subsample=0.85, colsample_bytree=0.85, verbosity=-1,
        random_state=SEED, n_jobs=-1,
    )
    configs.append(("base", base.copy()))

    # --- Axis 1: n_estimators ---
    for n in [200, 1000, 2000]:
        c = base.copy()
        c["n_estimators"] = n
        configs.append((f"n{n}", c))

    # --- Axis 2: max_depth ---
    for d in [4, 8, 12, -1]:
        c = base.copy()
        c["max_depth"] = d
        label = f"depth{d}" if d > 0 else "depth_none"
        configs.append((label, c))

    # --- Axis 3: num_leaves ---
    for nl in [15, 63, 127, 255]:
        c = base.copy()
        c["num_leaves"] = nl
        configs.append((f"leaves{nl}", c))

    # --- Axis 4: learning_rate ---
    for lr in [0.01, 0.1, 0.2]:
        c = base.copy()
        c["learning_rate"] = lr
        configs.append((f"lr{lr}", c))

    # --- Axis 5: regularization ---
    for reg in [0.0, 1.0, 5.0, 10.0]:
        c = base.copy()
        c["reg_lambda"] = reg
        configs.append((f"reg{reg}", c))

    # --- Axis 6: subsampling ---
    for ss in [0.6, 0.7, 1.0]:
        c = base.copy()
        c["subsample"] = ss
        c["subsample_freq"] = 1 if ss < 1.0 else 0
        configs.append((f"ss{ss}", c))

    # --- Axis 7: colsample ---
    for cs in [0.5, 0.7, 1.0]:
        c = base.copy()
        c["colsample_bytree"] = cs
        configs.append((f"cs{cs}", c))

    # --- Axis 8: min_child_samples ---
    for mcs in [5, 50, 100]:
        c = base.copy()
        c["min_child_samples"] = mcs
        configs.append((f"mcs{mcs}", c))

    # --- Strong combos (tuned guesses) ---
    # High-capacity
    configs.append(("strong_deep", dict(
        n_estimators=1000, max_depth=-1, learning_rate=0.03,
        num_leaves=127, min_child_samples=10, reg_lambda=1.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))
    # Conservative
    configs.append(("strong_conservative", dict(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        num_leaves=63, min_child_samples=30, reg_lambda=5.0,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))
    # Fast shallow
    configs.append(("strong_shallow", dict(
        n_estimators=2000, max_depth=4, learning_rate=0.02,
        num_leaves=15, min_child_samples=20, reg_lambda=0.5,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))
    # Wide leaves
    configs.append(("strong_wide", dict(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        num_leaves=255, min_child_samples=20, reg_lambda=3.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    return configs


def main():
    t0_total = time.time()

    # --- Load column lists ---
    full_cols = [c for c in pq.read_schema(FULL_PQ).names if c != "cell_id"]
    v2_cols = [c for c in pq.read_schema(V2_PQ).names if c != "cell_id"]

    groups = build_feature_groups(full_cols)
    novel_cols = {idx: get_novel_cols(v2_cols, idx) for idx in NOVEL_INDICES}

    # Top 3 feature sets to test
    feature_sets = {
        "VegIdx+RedEdge+TC+NDTI+IRECI+CRI1": (
            groups["VegIdx"] + groups["RedEdge"] + groups["TC"],
            [c for idx in NOVEL_INDICES for c in novel_cols[idx]],
        ),
        "VegIdx+RedEdge+TC+NDTI+CRI1": (
            groups["VegIdx"] + groups["RedEdge"] + groups["TC"],
            [c for idx in ["NDTI", "CRI1"] for c in novel_cols[idx]],
        ),
        "VegIdx+RedEdge+TC": (
            groups["VegIdx"] + groups["RedEdge"] + groups["TC"],
            [],
        ),
    }

    for name, (base, novel) in feature_sets.items():
        print(f"  {name}: {len(base) + len(novel)} features "
              f"({len(base)} base + {len(novel)} novel)", flush=True)

    configs = generate_configs()
    print(f"\n{len(configs)} hyperparameter configs", flush=True)
    total_runs = len(configs) * len(feature_sets) * 5
    print(f"Total runs: {len(configs)} configs x {len(feature_sets)} feat sets "
          f"x 5 folds = {total_runs}", flush=True)

    # --- Load data ---
    print("\nLoading data...", flush=True)
    all_base_needed = list(set(
        c for _, (base, _) in feature_sets.items() for c in base
    ))
    all_novel_needed = list(set(
        c for _, (_, novel) in feature_sets.items() for c in novel
    ))
    all_base_needed.sort()
    all_novel_needed.sort()

    base_df = pd.read_parquet(FULL_PQ, columns=["cell_id"] + all_base_needed)
    if all_novel_needed:
        v2_df = pd.read_parquet(V2_PQ, columns=["cell_id"] + all_novel_needed)
        merged = base_df.merge(v2_df, on="cell_id", how="inner")
    else:
        merged = base_df

    all_feat_cols = all_base_needed + all_novel_needed
    col_to_idx = {c: i for i, c in enumerate(all_feat_cols)}
    X_all = np.nan_to_num(merged[all_feat_cols].values.astype(np.float32), 0.0)

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    print(f"Data: {X_all.shape[0]} cells, {X_all.shape[1]} max features", flush=True)

    # --- Run sweep ---
    results = []
    run_num = 0

    for fold_id in range(5):
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )
        print(f"\n{'='*100}", flush=True)
        print(f"Fold {fold_id} (train={len(train_idx)}, test={len(test_idx)})",
              flush=True)
        print(f"{'='*100}", flush=True)

        for feat_name, (base_cols, novel_extra) in feature_sets.items():
            feat_idx = [col_to_idx[c] for c in base_cols + novel_extra]
            X = X_all[:, feat_idx]
            n_feat = len(feat_idx)

            for cfg_name, params in configs:
                run_num += 1
                t0 = time.time()

                model = MultiOutputRegressor(
                    lgb.LGBMRegressor(**params)
                )
                model.fit(X[train_idx], y[train_idx])
                y_pred = np.clip(model.predict(X[test_idx]), 0, 100)

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                elapsed = time.time() - t0

                if run_num % 10 == 1 or run_num <= 3:
                    print(f"  [{run_num:>4d}/{total_runs}] F{fold_id} "
                          f"{feat_name:<40s} {cfg_name:<22s} "
                          f"{n_feat:>4d}f  R2={r2:.4f}  MAE={mae:.3f}  "
                          f"{elapsed:.1f}s", flush=True)

                results.append({
                    "fold": fold_id,
                    "feature_set": feat_name,
                    "config": cfg_name,
                    "n_features": n_feat,
                    "r2": r2, "mae": mae, "time": elapsed,
                    **{k: v for k, v in params.items()
                       if k not in ("verbosity", "random_state", "n_jobs")},
                })

    # --- Save ---
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}", flush=True)
    print(f"Total time: {time.time() - t0_total:.0f}s", flush=True)

    # --- Summary ---
    print(f"\n{'='*100}", flush=True)
    print("TOP 10 CONFIGS (mean R2 across 5 folds):", flush=True)
    print(f"{'='*100}", flush=True)

    summary = res_df.groupby(["feature_set", "config"]).agg(
        n_features=("n_features", "first"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
        mean_mae=("mae", "mean"),
        mean_time=("time", "mean"),
    ).sort_values("mean_r2", ascending=False)

    for i, (idx, row) in enumerate(summary.head(10).iterrows()):
        feat, cfg = idx
        print(f"  {i+1:>2d}. {feat:<40s} {cfg:<22s} "
              f"{int(row.n_features):>4d}f  R2={row.mean_r2:.4f} "
              f"(+/-{row.std_r2:.4f})  MAE={row.mean_mae:.3f}  "
              f"{row.mean_time:.1f}s", flush=True)

    # Best per feature set
    print(f"\nBest config per feature set:", flush=True)
    for feat_name in feature_sets:
        sub = summary.loc[feat_name]
        best = sub.sort_values("mean_r2", ascending=False).head(1)
        idx = best.index[0]
        row = best.iloc[0]
        print(f"  {feat_name:<40s} -> {idx:<22s} "
              f"R2={row.mean_r2:.4f} (+/-{row.std_r2:.4f})", flush=True)


if __name__ == "__main__":
    main()
