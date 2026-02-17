#!/usr/bin/env python3
"""
LightGBM sweep on multi-city V4 features.

Uses the SAME features and cities as run_multi_city_pipeline_v4.py but trains
LightGBM with multiple configs to see if trees can match/beat the MLP now
that there is more training data + richer features (phenology, LBP, etc.).

Evaluates on 3 held-out test cities (Nuremberg, Frankfurt, Munich).
Also tests two feature sets:
  - "mlp_feats" : same large feature set the MLP uses (bands+indices+LBP+pheno)
  - "tree_feats": the smaller curated set from V4 pipeline

Output: data/cities/models_v4/lgbm_sweep_results.csv
"""

import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config (mirror from run_multi_city_pipeline_v4.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

SEED = 42
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)

_BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                  "B11", "B12"}
_INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI2", "CRI1", "MNDWI", "GNDVI", "NDTI", "IRECI", "TC",
}

@dataclass
class CityConfig:
    name: str
    bbox: List[float]
    epsg: int
    wc_tile: str
    is_test: bool = False

CITIES = [
    CityConfig("bremen",      [8.65, 53.00, 8.90, 53.14], 32632, "N51E006"),
    CityConfig("hamburg",     [9.80, 53.40, 10.15, 53.58], 32632, "N51E009"),
    CityConfig("duesseldorf", [6.70, 51.15, 6.90, 51.28], 32632, "N51E006"),
    CityConfig("leipzig",     [12.25, 51.27, 12.50, 51.40], 32633, "N51E012"),
    CityConfig("rostock",     [12.00, 54.05, 12.20, 54.18], 32633, "N54E012"),
    CityConfig("amsterdam",   [4.75, 52.30, 4.95, 52.40], 32631, "N51E003"),
    CityConfig("hambach_mine",[6.40, 50.85, 6.60, 50.98], 32632, "N48E006"),
    CityConfig("welzow_mine", [14.10, 51.50, 14.35, 51.65], 32633, "N51E012"),
    CityConfig("amiens",      [2.15, 49.82, 2.42, 49.96], 32631, "N48E000"),
    CityConfig("magdeburg",   [11.50, 52.05, 11.75, 52.20], 32632, "N51E009"),
    CityConfig("ulm",         [9.85, 48.33, 10.10, 48.47], 32632, "N48E009"),
    CityConfig("salzburg",    [12.95, 47.73, 13.15, 47.87], 32633, "N45E012"),
    CityConfig("schwerin",    [11.30, 53.55, 11.55, 53.70], 32632, "N51E009"),
    CityConfig("malmo",       [12.90, 55.53, 13.15, 55.68], 32633, "N54E012"),
    CityConfig("nuremberg",   [10.95, 49.38, 11.20, 49.52], 32632, "N48E009",
               is_test=True),
    CityConfig("frankfurt",   [8.55, 50.05, 8.80, 50.18], 32632, "N48E006",
               is_test=True),
    CityConfig("munich",      [11.45, 48.08, 11.70, 48.22], 32632, "N48E009",
               is_test=True),
]

TRAIN_CITIES = [c for c in CITIES if not c.is_test]
TEST_CITIES = [c for c in CITIES if c.is_test]

CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
MODELS_DIR = os.path.join(CITIES_DIR, "models_v4")


# ---------------------------------------------------------------------------
# Feature selection (identical to V4 pipeline)
# ---------------------------------------------------------------------------
def build_bi_lbp(feature_cols):
    """MLP feature set: bands + indices + LBP + pheno (NO deltas)."""
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in _BAND_PREFIXES or prefix in _INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
        elif "_pheno_" in col:
            selected.append(i)
    return sorted(set(selected))


def build_tree_features(feature_cols):
    """Original V4 tree feature set (curated subset)."""
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    novel = ["NDTI", "IRECI", "CRI1"]
    selected = []
    for c in feature_cols:
        if "_pheno_" in c:
            selected.append(c)
            continue
        if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"]):
            if not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr"):
                selected.append(c)
                continue
        if band_pat.match(c):
            selected.append(c)
            continue
        if c.startswith("TC_"):
            selected.append(c)
            continue
        for idx in novel:
            if c.startswith(f"{idx}_"):
                selected.append(c)
                break
    return selected


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def city_features_dir(city):
    return os.path.join(CITIES_DIR, city.name, "features")


def city_labels_path(city, year=2021):
    return os.path.join(CITIES_DIR, city.name,
                        f"labels_{year}.parquet")


def discover_feature_cols():
    """Read column names from first available parquet."""
    for city in TRAIN_CITIES:
        path = os.path.join(city_features_dir(city),
                            "features_rust_2020_2021.parquet")
        if os.path.exists(path):
            control = {"cell_id", "valid_fraction", "low_valid_fraction",
                        "reflectance_scale", "full_features_computed"}
            cols = pd.read_parquet(path, columns=None).columns.tolist()
            return [c for c in cols if c not in control]
    raise RuntimeError("No feature parquets found!")


def load_city_data(city, cols):
    """Load features + labels for a city. Returns (X, y) or None."""
    feat_path = os.path.join(city_features_dir(city),
                             "features_rust_2020_2021.parquet")
    labels_path = city_labels_path(city)
    if not os.path.exists(feat_path) or not os.path.exists(labels_path):
        return None

    cols_to_read = [c for c in cols if c != "cell_id"]
    df = pd.read_parquet(feat_path, columns=cols_to_read)
    X = np.nan_to_num(df.values.astype(np.float32), 0.0)
    del df

    labels = pd.read_parquet(labels_path)
    y = labels[CLASS_NAMES].values.astype(np.float32)
    del labels

    return X, y


def evaluate(y_true, y_pred):
    """R2 (uniform average) and MAE (pp)."""
    from sklearn.metrics import mean_absolute_error, r2_score
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    mae = float(mean_absolute_error(y_true, y_pred,
                                     multioutput="uniform_average")) * 100
    per_class = {}
    for i, cn in enumerate(CLASS_NAMES):
        per_class[cn] = {
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
            "mae_pp": float(mean_absolute_error(y_true[:, i],
                                                 y_pred[:, i])) * 100,
        }
    return r2, mae, per_class


# ---------------------------------------------------------------------------
# LightGBM configs
# ---------------------------------------------------------------------------
def generate_configs():
    """Focused set of LightGBM configs — best from previous sweep + variants."""
    configs = []

    # 1. V4 production config (current best = "strong_wide")
    configs.append(("v4_production", dict(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        num_leaves=255, min_child_samples=20, reg_lambda=3.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    # 2. Strong deep (unlimited depth)
    configs.append(("strong_deep", dict(
        n_estimators=1000, max_depth=-1, learning_rate=0.03,
        num_leaves=127, min_child_samples=10, reg_lambda=1.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    # 3. Conservative (more regularized)
    configs.append(("strong_conservative", dict(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        num_leaves=63, min_child_samples=30, reg_lambda=5.0,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    # 4. More trees + lower LR (potentially better generalization)
    configs.append(("n2000_lr002", dict(
        n_estimators=2000, max_depth=6, learning_rate=0.02,
        num_leaves=255, min_child_samples=20, reg_lambda=3.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    # 5. Wider leaves + more trees
    configs.append(("n2000_leaves512", dict(
        n_estimators=2000, max_depth=8, learning_rate=0.02,
        num_leaves=512, min_child_samples=20, reg_lambda=2.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    # 6. Very deep, many trees, low LR
    configs.append(("n3000_deep_lowlr", dict(
        n_estimators=3000, max_depth=-1, learning_rate=0.01,
        num_leaves=255, min_child_samples=15, reg_lambda=2.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    # 7. High regularization
    configs.append(("high_reg", dict(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        num_leaves=127, min_child_samples=30, reg_lambda=10.0,
        reg_alpha=1.0,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    # 8. Shallow fast (baseline comparison)
    configs.append(("shallow_fast", dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, reg_lambda=0.1,
        subsample=0.85, colsample_bytree=0.85,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )))

    return configs


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("LightGBM SWEEP on V4 multi-city features")
    print("=" * 70)

    # --- Discover features ---
    print("\nDiscovering feature columns...")
    all_feature_cols = discover_feature_cols()

    mlp_idx = build_bi_lbp(all_feature_cols)
    mlp_cols = [all_feature_cols[i] for i in mlp_idx]
    tree_cols = build_tree_features(all_feature_cols)
    print(f"  MLP feature set:  {len(mlp_cols)} features")
    print(f"  Tree feature set: {len(tree_cols)} features (V4 curated)")

    # Union of all needed columns
    all_needed = sorted(set(mlp_cols) | set(tree_cols))
    mlp_indices_in_union = [all_needed.index(c) for c in mlp_cols]
    tree_indices_in_union = [all_needed.index(c) for c in tree_cols]

    # --- Load training data ---
    print(f"\nLoading training data ({len(TRAIN_CITIES)} cities)...")
    X_parts, y_parts = [], []
    total = 0
    for city in TRAIN_CITIES:
        result = load_city_data(city, all_needed)
        if result is None:
            print(f"  WARNING: {city.name} missing -- skip")
            continue
        X_c, y_c = result
        X_parts.append(X_c)
        y_parts.append(y_c)
        total += len(y_c)
        print(f"  {city.name}: {len(y_c)} cells (total={total})")

    X_train_all = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    del X_parts, y_parts
    print(f"  Total training: {len(y_train)} cells")

    # --- Load test data ---
    print(f"\nLoading test data ({len(TEST_CITIES)} cities)...")
    test_data = {}
    for city in TEST_CITIES:
        result = load_city_data(city, all_needed)
        if result is None:
            print(f"  WARNING: {city.name} missing -- skip")
            continue
        X_c, y_c = result
        test_data[city.name] = (X_c, y_c)
        print(f"  {city.name}: {len(y_c)} cells")

    # Combine test data for aggregate metrics
    X_test_all = np.concatenate([v[0] for v in test_data.values()], axis=0)
    y_test = np.concatenate([v[1] for v in test_data.values()], axis=0)
    print(f"  Total test: {len(y_test)} cells")

    # --- Prepare feature set views ---
    feature_sets = {
        "mlp_feats": (mlp_indices_in_union, mlp_cols),
        "tree_feats": (tree_indices_in_union, tree_cols),
    }

    # --- Run sweep ---
    configs = generate_configs()
    results = []
    n_runs = len(configs) * len(feature_sets)

    print(f"\n{'='*70}")
    print(f"Running {n_runs} experiments "
          f"({len(configs)} configs x {len(feature_sets)} feature sets)")
    print(f"{'='*70}")

    run_idx = 0
    for feat_name, (feat_indices, feat_col_names) in feature_sets.items():
        X_train = X_train_all[:, feat_indices]
        X_test = X_test_all[:, feat_indices]
        n_feats = len(feat_indices)

        # Per-city test views
        city_test_views = {}
        for cname, (X_c, y_c) in test_data.items():
            city_test_views[cname] = (X_c[:, feat_indices], y_c)

        for cfg_name, params in configs:
            run_idx += 1
            label = f"{feat_name}/{cfg_name}"
            print(f"\n[{run_idx}/{n_runs}] {label} ({n_feats} features)")

            t0 = time.time()
            model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # In-sample
            y_pred_train = np.clip(model.predict(X_train), 0, 1)
            r2_train, mae_train, _ = evaluate(y_train, y_pred_train)

            # Combined test set
            y_pred_test = np.clip(model.predict(X_test), 0, 1)
            r2_test, mae_test, per_class = evaluate(y_test, y_pred_test)

            print(f"  Train: R2={r2_train:.4f} MAE={mae_train:.2f}pp "
                  f"({train_time:.0f}s)")
            print(f"  Test:  R2={r2_test:.4f} MAE={mae_test:.2f}pp")

            row = {
                "feature_set": feat_name,
                "config": cfg_name,
                "n_features": n_feats,
                "r2_train": r2_train,
                "mae_train_pp": mae_train,
                "r2_test": r2_test,
                "mae_test_pp": mae_test,
                "train_time_s": train_time,
            }

            # Per-class test metrics
            for cn in CLASS_NAMES:
                row[f"r2_test_{cn}"] = per_class[cn]["r2"]
                row[f"mae_test_{cn}_pp"] = per_class[cn]["mae_pp"]

            # Per-city test metrics
            for cname, (X_ct, y_ct) in city_test_views.items():
                y_p_ct = np.clip(model.predict(X_ct), 0, 1)
                r2_ct, mae_ct, _ = evaluate(y_ct, y_p_ct)
                row[f"r2_{cname}"] = r2_ct
                row[f"mae_{cname}_pp"] = mae_ct
                print(f"  {cname}: R2={r2_ct:.4f} MAE={mae_ct:.2f}pp")

            results.append(row)

            # Save best model per feature set
            if all(r["feature_set"] != feat_name or
                   r["r2_test"] <= r2_test for r in results):
                best_path = os.path.join(
                    MODELS_DIR, f"lgbm_best_{feat_name}.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump(model, f)

    # --- Save results ---
    df = pd.DataFrame(results)
    out_path = os.path.join(MODELS_DIR, "lgbm_sweep_results.csv")
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {out_path}")
    print(f"{'='*70}")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY (sorted by test R2)")
    print(f"{'='*70}")
    df_sorted = df.sort_values("r2_test", ascending=False)
    for _, row in df_sorted.iterrows():
        print(f"  {row['feature_set']:12s} / {row['config']:22s}  "
              f"R2={row['r2_test']:.4f}  MAE={row['mae_test_pp']:.2f}pp  "
              f"({row['n_features']} feats, {row['train_time_s']:.0f}s)")

    # Compare to MLP (if meta exists)
    mlp_meta_path = os.path.join(MODELS_DIR, "mlp_meta.json")
    if os.path.exists(mlp_meta_path):
        with open(mlp_meta_path) as f:
            mlp_meta = json.load(f)
        print(f"\n  MLP reference: R2={mlp_meta.get('r2_insample', '?')} "
              f"(in-sample)")

    best = df_sorted.iloc[0]
    print(f"\n  BEST: {best['feature_set']}/{best['config']} "
          f"R2={best['r2_test']:.4f} MAE={best['mae_test_pp']:.2f}pp")


if __name__ == "__main__":
    main()
