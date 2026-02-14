"""
Fusion test: do genuinely novel indices (NDTI, IRECI, CRI1) improve
existing best feature sets?

Two base feature sets (from features_merged_full.parquet):
  A) VegIdx + RedEdge + TC           (~348 features, best spectral combo)
  B) GLCM + Morph + VegIdx + RedEdge + TC  (~633 features, best overall)

Novel indices to fuse (from features_bands_indices_v2.parquet):
  NDTI  — SWIR1/SWIR2 ratio (unique band pair)
  IRECI — multi-band red-edge chlorophyll (unique combo)
  CRI1  — carotenoid reciprocal difference (unique transform)

Tests: each base × {none, +NDTI, +IRECI, +CRI1, +NDTI+IRECI,
       +NDTI+CRI1, +IRECI+CRI1, +all three} = 2 × 8 = 16 experiments × 5 folds

Output: reports/phase8/tables/ablation_fusion_novel.csv
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
from sklearn.ensemble import HistGradientBoostingRegressor
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
                       "ablation_fusion_novel.csv")

NOVEL_INDICES = ["NDTI", "IRECI", "CRI1"]


def build_base_groups(feat_cols):
    """Identify existing best feature groups from the full merged parquet."""
    band_pat = re.compile(r'^B(05|06|07|8A)_')

    morph = [c for c in feat_cols if "MP_" in c]
    glcm = [c for c in feat_cols if "GLCM_" in c]

    veg_idx = [c for c in feat_cols
               if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"])
               and not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr")]

    rededge = [c for c in feat_cols if band_pat.match(c)]
    tc = [c for c in feat_cols if c.startswith("TC_")]

    base_a = veg_idx + rededge + tc
    base_b = glcm + morph + veg_idx + rededge + tc

    return {
        "VegIdx+RedEdge+TC": base_a,
        "GLCM+Morph+VegIdx+RedEdge+TC": base_b,
    }


def get_novel_cols(v2_cols, index_name):
    """Get columns for a novel index from the v2 parquet."""
    return [c for c in v2_cols if c.startswith(f"{index_name}_")]


def main():
    t0_total = time.time()

    # --- Load column lists ---
    full_cols = [c for c in pq.read_schema(FULL_PQ).names if c != "cell_id"]
    v2_cols = [c for c in pq.read_schema(V2_PQ).names if c != "cell_id"]

    bases = build_base_groups(full_cols)
    novel_cols = {idx: get_novel_cols(v2_cols, idx) for idx in NOVEL_INDICES}

    print("Base feature sets:", flush=True)
    for name, cols in bases.items():
        print(f"  {name}: {len(cols)} features", flush=True)

    print("\nNovel indices to fuse:", flush=True)
    for name, cols in novel_cols.items():
        print(f"  {name}: {len(cols)} features", flush=True)

    # --- Build experiments: each base × subsets of novel indices ---
    novel_combos = [()]  # empty = base only
    for r in range(1, len(NOVEL_INDICES) + 1):
        for combo in itertools.combinations(NOVEL_INDICES, r):
            novel_combos.append(combo)
    # novel_combos: (), (NDTI,), (IRECI,), (CRI1,), (NDTI,IRECI), ...

    experiments = []
    for base_name, base_cols in bases.items():
        for novel_combo in novel_combos:
            if len(novel_combo) == 0:
                name = f"{base_name} (base)"
                extra_cols = []
            else:
                suffix = "+".join(novel_combo)
                name = f"{base_name} + {suffix}"
                extra_cols = []
                for idx in novel_combo:
                    extra_cols.extend(novel_cols[idx])
            experiments.append((name, base_name, base_cols, extra_cols, novel_combo))

    print(f"\n{len(experiments)} experiments:", flush=True)
    for name, _, base_cols, extra_cols, _ in experiments:
        print(f"  {name}: {len(base_cols) + len(extra_cols)} features", flush=True)

    # --- Load data ---
    print("\nLoading full parquet (base features)...", flush=True)
    all_base_cols = list(set(c for _, _, cols, _, _ in experiments for c in cols))
    all_base_cols.sort()
    base_df = pd.read_parquet(FULL_PQ, columns=["cell_id"] + all_base_cols)

    print("Loading v2 parquet (novel indices)...", flush=True)
    all_novel_cols = list(set(c for idx in NOVEL_INDICES for c in novel_cols[idx]))
    all_novel_cols.sort()
    v2_df = pd.read_parquet(V2_PQ, columns=["cell_id"] + all_novel_cols)

    # Merge on cell_id
    merged = base_df.merge(v2_df, on="cell_id", how="inner")
    print(f"Merged: {len(merged)} cells, {len(merged.columns)-1} total columns", flush=True)

    # Build numpy arrays
    all_feat_cols = all_base_cols + all_novel_cols
    col_to_idx = {c: i for i, c in enumerate(all_feat_cols)}
    X_all = np.nan_to_num(merged[all_feat_cols].values.astype(np.float32), 0.0)

    # Splits + labels
    print("Loading splits + labels...", flush=True)
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    # --- Run experiments ---
    print(f"\n{'='*90}", flush=True)
    print(f"{'Fold':>4s} {'Experiment':<55s} {'N feat':>7s} {'R2':>8s} {'MAE':>8s} "
          f"{'Time':>6s}", flush=True)
    print(f"{'='*90}", flush=True)

    results = []
    for fold_id in range(5):
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )
        print(f"\n--- Fold {fold_id} (train={len(train_idx)}, test={len(test_idx)}) ---",
              flush=True)

        for name, base_name, base_cols, extra_cols, novel_combo in experiments:
            t0 = time.time()
            feat_idx = [col_to_idx[c] for c in base_cols + extra_cols]
            X = X_all[:, feat_idx]
            n_feat = len(feat_idx)

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

            print(f"  F{fold_id} {name:<55s} {n_feat:>7d} {r2:>8.4f} {mae:>8.3f} "
                  f"{elapsed:>5.1f}s", flush=True)

            novel_str = "+".join(novel_combo) if novel_combo else "none"
            results.append({
                "fold": fold_id,
                "base": base_name,
                "novel_added": novel_str,
                "group": name,
                "n_base_features": len(base_cols),
                "n_novel_features": len(extra_cols),
                "n_features": n_feat,
                "r2": r2, "mae": mae, "time": elapsed,
            })

    # --- Save ---
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}", flush=True)
    print(f"Total time: {time.time() - t0_total:.0f}s", flush=True)

    # --- Summary ---
    res_df = pd.DataFrame(results)

    for base_name in bases:
        print(f"\n{'='*90}", flush=True)
        print(f"Base: {base_name}", flush=True)
        print(f"{'='*90}", flush=True)
        sub = res_df[res_df["base"] == base_name]
        summary = sub.groupby("novel_added").agg(
            n_features=("n_features", "first"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_mae=("mae", "mean"),
        ).sort_values("mean_r2", ascending=False)
        print(summary.to_string(), flush=True)

        # Compute delta from base
        base_r2 = summary.loc["none", "mean_r2"]
        print(f"\n  Δ R² vs base ({base_r2:.4f}):", flush=True)
        for idx, row in summary.iterrows():
            delta = row["mean_r2"] - base_r2
            sign = "+" if delta >= 0 else ""
            print(f"    {idx:<30s} {sign}{delta:.4f}", flush=True)


if __name__ == "__main__":
    main()
