#!/usr/bin/env python3
"""
Production LightGBM Training — Champion Config (standalone).

Champion: VegIdx + RedEdge + TC + NDTI + IRECI + CRI1
  - LightGBM via MultiOutputRegressor
  - n_estimators=500, max_depth=6, lr=0.05, num_leaves=31
  - subsample=0.85, colsample_bytree=0.85, reg_lambda=0.1

Fully standalone — no project imports required.

Produces:
    models/final_tree_prod/fold_{i}.pkl
    models/final_tree_prod/oof_predictions.parquet
    models/final_tree_prod/meta.json
"""

import json
import os
import pickle
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# =====================================================================
# Constants
# =====================================================================

SEED = 42
N_FOLDS = 5
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)

BEST_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    reg_lambda=0.1,
    subsample=0.85,
    colsample_bytree=0.85,
    verbosity=-1,
    random_state=SEED,
    n_jobs=-1,
)

# Novel indices from v2 feature set
NOVEL_INDICES = ["NDTI", "IRECI", "CRI1"]


# =====================================================================
# Spatial Splitting (inlined)
# =====================================================================

@lru_cache(maxsize=8)
def _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles):
    neighbors = {}
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile_id = tr * n_tile_cols + tc
            nbrs = set()
            for dr in range(-buffer_tiles, buffer_tiles + 1):
                for dc in range(-buffer_tiles, buffer_tiles + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                        nbrs.add(nr * n_tile_cols + nc)
            neighbors[tile_id] = frozenset(nbrs)
    return neighbors


def get_fold_indices(groups, fold_assignments, fold_idx,
                     n_tile_cols, n_tile_rows, buffer_tiles=1):
    """Train/test split with spatial buffer to prevent leakage."""
    test_mask = fold_assignments == fold_idx

    if buffer_tiles > 0:
        test_tiles = set(np.unique(groups[test_mask]))
        tile_neighbors = _precompute_tile_neighbors(
            n_tile_cols, n_tile_rows, buffer_tiles
        )
        buffer_tiles_set = set()
        for tt in test_tiles:
            for nbr in tile_neighbors.get(tt, set()):
                if nbr not in test_tiles:
                    buffer_tiles_set.add(nbr)
        buffer_mask = np.isin(groups, list(buffer_tiles_set))
        train_mask = (~test_mask) & (~buffer_mask)
    else:
        train_mask = ~test_mask

    return np.where(train_mask)[0], np.where(test_mask)[0]


# =====================================================================
# Evaluation (inlined)
# =====================================================================

def evaluate_predictions(y_true, y_pred, class_names):
    """Compute R2 (uniform) and MAE (pp)."""
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    mae = float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")) * 100
    per_class = {
        cn: {
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
            "mae_pp": float(mean_absolute_error(y_true[:, i], y_pred[:, i])) * 100,
        }
        for i, cn in enumerate(class_names)
    }
    return {"r2_uniform": r2, "mae_mean_pp": mae, "per_class": per_class}


# =====================================================================
# Feature Selection
# =====================================================================

def build_feature_groups(feat_cols):
    """Select VegIdx + RedEdge + TC feature columns."""
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    veg_idx = [
        c for c in feat_cols
        if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"])
        and not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr")
    ]
    rededge = [c for c in feat_cols if band_pat.match(c)]
    tc = [c for c in feat_cols if c.startswith("TC_")]
    return {"VegIdx": veg_idx, "RedEdge": rededge, "TC": tc}


def build_tree_features(feat_cols, v2_cols=None):
    """
    Build the champion tree feature set: VegIdx + RedEdge + TC + NDTI + IRECI + CRI1.

    Works with either:
    - Single Rust production parquet (all features in one file)
    - Legacy two-parquet setup (feat_cols from full, v2_cols from v2)
    """
    groups = build_feature_groups(feat_cols)
    base_cols = groups["VegIdx"] + groups["RedEdge"] + groups["TC"]

    # Novel indices — look in feat_cols first, fall back to v2_cols
    novel_cols = []
    for idx_name in NOVEL_INDICES:
        found = [c for c in feat_cols if c.startswith(f"{idx_name}_")]
        if found:
            novel_cols.extend(found)
        elif v2_cols is not None:
            novel_cols.extend(c for c in v2_cols if c.startswith(f"{idx_name}_"))

    return base_cols + novel_cols


# =====================================================================
# Main
# =====================================================================

def main():
    t_start = time.time()

    # ── Paths (auto-detect project root) ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    for _ in range(5):
        if os.path.isdir(os.path.join(project_root, "data")):
            break
        project_root = os.path.dirname(project_root)

    processed_dir = os.path.join(project_root, "data", "processed", "v2")
    model_dir = os.path.join(project_root, "models", "final_tree_prod")
    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ──
    print("=" * 60)
    print("Production LightGBM Training — Champion Config")
    print("=" * 60)

    # Try Rust production parquet first (single file, has all features)
    rust_path = os.path.join(processed_dir, "features_rust_production.parquet")
    full_path = os.path.join(processed_dir, "features_merged_full.parquet")
    v2_path = os.path.join(processed_dir, "features_bands_indices_v2.parquet")

    if os.path.exists(rust_path):
        print(f"Loading features from {os.path.basename(rust_path)}...")
        import pyarrow.parquet as pq
        all_cols = [c for c in pq.read_schema(rust_path).names if c != "cell_id"]
        feature_cols = build_tree_features(all_cols)
        df = pd.read_parquet(rust_path, columns=["cell_id"] + feature_cols)
        print(f"  Source: Rust production parquet")
    else:
        # Legacy: merge two parquets
        print(f"Loading features from legacy parquets...")
        import pyarrow.parquet as pq
        full_cols = [c for c in pq.read_schema(full_path).names if c != "cell_id"]
        v2_cols = [c for c in pq.read_schema(v2_path).names if c != "cell_id"]
        feature_cols = build_tree_features(full_cols, v2_cols)

        # Split into which file has which columns
        full_set = set(full_cols)
        v2_set = set(v2_cols)
        from_full = [c for c in feature_cols if c in full_set]
        from_v2 = [c for c in feature_cols if c in v2_set and c not in full_set]

        base_df = pd.read_parquet(full_path, columns=["cell_id"] + sorted(set(from_full)))
        if from_v2:
            v2_df = pd.read_parquet(v2_path, columns=["cell_id"] + sorted(set(from_v2)))
            df = base_df.merge(v2_df, on="cell_id", how="inner")
            del v2_df
        else:
            df = base_df
        del base_df
        print(f"  Source: merged full + v2 parquets")

    n_features = len(feature_cols)
    print(f"  Feature set: VegIdx+RedEdge+TC+NDTI+IRECI+CRI1 = {n_features} features")

    X_all = np.nan_to_num(df[feature_cols].values.astype(np.float32), 0.0)
    cell_ids = df["cell_id"].values
    del df

    # Labels + splits
    labels_df = pd.read_parquet(os.path.join(processed_dir, "labels_2021.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)
    del labels_df

    split_df = pd.read_parquet(os.path.join(processed_dir, "split_spatial.parquet"))
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    del split_df

    with open(os.path.join(processed_dir, "split_spatial_meta.json")) as f:
        split_meta = json.load(f)

    print(f"  Data: {X_all.shape[0]} cells, {n_features} features, {N_CLASSES} classes")

    # ── Train 5 folds ──
    oof = np.zeros_like(y)
    oof_mask = np.zeros(len(y), dtype=bool)
    fold_metrics = []

    for fold_id in range(N_FOLDS):
        print(f"\n{'='*50}")
        print(f"Fold {fold_id}")
        print(f"{'='*50}")

        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id,
            split_meta["tile_cols"], split_meta["tile_rows"],
            buffer_tiles=1,
        )
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

        t0 = time.time()
        model = MultiOutputRegressor(lgb.LGBMRegressor(**BEST_PARAMS))
        model.fit(X_all[train_idx], y[train_idx])

        y_pred = np.clip(model.predict(X_all[test_idx]), 0, 100)
        oof[test_idx] = y_pred
        oof_mask[test_idx] = True

        elapsed = time.time() - t0

        metrics = evaluate_predictions(y[test_idx], y_pred, CLASS_NAMES)
        r2 = metrics["r2_uniform"]
        mae = metrics["mae_mean_pp"]
        fold_metrics.append({
            "fold": fold_id, "r2": r2, "mae": mae, "time_s": round(elapsed, 1),
        })
        print(f"  R2={r2:.4f}  MAE={mae:.2f}pp  time={elapsed:.1f}s")

        # Save model
        with open(os.path.join(model_dir, f"fold_{fold_id}.pkl"), "wb") as f:
            pickle.dump(model, f)

    # ── Save OOF predictions ──
    oof_df = pd.DataFrame({"cell_id": cell_ids})
    for ci, cn in enumerate(CLASS_NAMES):
        oof_df[f"{cn}_pred"] = oof[:, ci]
    oof_df.to_parquet(os.path.join(model_dir, "oof_predictions.parquet"), index=False)

    # ── Save metadata ──
    r2_vals = [m["r2"] for m in fold_metrics]
    mae_vals = [m["mae"] for m in fold_metrics]

    meta = {
        "model": "LightGBM",
        "config": "VegIdx+RedEdge+TC+NDTI+IRECI+CRI1",
        "feature_cols": feature_cols,
        "n_features": n_features,
        "seed": SEED,
        "r2_mean": round(float(np.mean(r2_vals)), 4),
        "r2_std": round(float(np.std(r2_vals)), 4),
        "mae_mean_pp": round(float(np.mean(mae_vals)), 2),
        "fold_metrics": fold_metrics,
        "hyperparameters": {
            k: v for k, v in BEST_PARAMS.items()
            if k not in ("verbosity", "random_state", "n_jobs")
        },
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ──
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"FINAL TREE CHAMPION — 5-fold CV")
    print(f"{'='*60}")
    for m in fold_metrics:
        print(f"  Fold {m['fold']}: R2={m['r2']:.4f}  MAE={m['mae']:.2f}pp  "
              f"time={m['time_s']}s")
    print(f"  {'-'*40}")
    print(f"  Mean: R2={np.mean(r2_vals):.4f} +/- {np.std(r2_vals):.4f}  "
          f"MAE={np.mean(mae_vals):.2f}pp")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.0f}s")
    print(f"Saved to: {model_dir}")


if __name__ == "__main__":
    main()
