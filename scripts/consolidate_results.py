#!/usr/bin/env python3
"""
Consolidate all experiment results into master tables.

Produces:
    reports/phase8/tables/all_mlp_results.csv       — all MLP runs (per-fold)
    reports/phase8/tables/all_mlp_summary.csv        — MLP 5-fold CV summaries
    reports/phase8/tables/all_tree_results.csv       — all tree runs (per-fold)
    reports/phase8/tables/all_tree_summary.csv        — tree 5-fold CV summaries
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")

COMMON_COLS = ["version", "name", "fold", "feature_set", "n_features",
               "r2_uniform", "mae_mean_pp"]


def parse_old_model_name(model_str):
    """Parse V5-V9 style model name like 'bands_indices_plain_silu_L5_d512_bn__fold0'."""
    parts = model_str.split("__")
    name = parts[0]
    fold = int(parts[1].replace("fold", "")) if len(parts) > 1 else -1
    return name, fold


def extract_feature_set_from_name(name):
    """Try to extract feature set from old-style config names."""
    known_sets = [
        "bands_indices_glcm_lbp", "bands_indices_texture",
        "bands_indices_hog", "full_no_deltas", "bands_indices",
        "all_full", "top500_full", "texture_all",
    ]
    for s in sorted(known_sets, key=len, reverse=True):
        if name.startswith(s + "_"):
            return s
    return "unknown"


def load_old_format(csv_path, version):
    """Load V5-V9 style CSV (model column with fold encoded)."""
    df = pd.read_csv(csv_path)
    if "model" not in df.columns:
        return None

    records = []
    for _, row in df.iterrows():
        name, fold = parse_old_model_name(row["model"])
        fs = extract_feature_set_from_name(name)
        records.append({
            "version": version,
            "name": name,
            "fold": fold,
            "feature_set": fs,
            "n_features": row.get("n_features", np.nan),
            "r2_uniform": row["r2_uniform"],
            "mae_mean_pp": row["mae_mean_pp"],
        })
    return pd.DataFrame(records)


def load_new_format(csv_path, version):
    """Load V10+ style CSV (name, fold, feature_set columns)."""
    df = pd.read_csv(csv_path)
    if "name" not in df.columns and "model" not in df.columns:
        return None

    name_col = "name" if "name" in df.columns else "model"
    out = pd.DataFrame({
        "version": version,
        "name": df[name_col],
        "fold": df.get("fold", -1),
        "feature_set": df.get("feature_set", "unknown"),
        "n_features": df.get("n_features", np.nan),
        "r2_uniform": df["r2_uniform"],
        "mae_mean_pp": df["mae_mean_pp"],
    })
    return out


def summarize(df):
    """Create 5-fold CV summary from per-fold results."""
    g = df.groupby(["version", "name", "feature_set"]).agg(
        r2_mean=("r2_uniform", "mean"),
        r2_std=("r2_uniform", "std"),
        mae_mean=("mae_mean_pp", "mean"),
        n_folds=("fold", "count"),
        n_features=("n_features", "first"),
    ).reset_index().sort_values("r2_mean", ascending=False)
    return g


def main():
    # ── MLP results ──
    mlp_versions = {
        "V5_arch": "mlp_v5_5_arch.csv",
        "V5_deep": "mlp_v5_deep.csv",
        "V6_arch": "mlp_v6_arch.csv",
        "V7_arch": "mlp_v7_arch.csv",
        "V9_texture": "mlp_v9_texture_ablation.csv",
        "V10": "mlp_v10_definitive.csv",
        "V11": "mlp_v11_new_texture.csv",
        "V12": "mlp_v12_multiband_lbp.csv",
        "V12b": "mlp_v12b_arch_sweep.csv",
        "V13": "mlp_v13_multiband_lbp_clean.csv",
        "V14": "mlp_v14_reproduce.csv",
        "V15": "mlp_v15_perpatch_lbp.csv",
        "V16": "mlp_v16_rust_reproduce.csv",
        "V17": "mlp_v17_multiseed.csv",
    }

    old_versions = {"V5_arch", "V5_deep", "V6_arch", "V7_arch", "V9_texture", "V10"}
    all_mlp = []

    for ver, fname in mlp_versions.items():
        path = os.path.join(TABLE_DIR, fname)
        if not os.path.exists(path):
            print("  SKIP (missing): {}".format(fname))
            continue
        if ver in old_versions:
            df = load_old_format(path, ver)
        else:
            df = load_new_format(path, ver)
        if df is not None and len(df) > 0:
            all_mlp.append(df)
            print("  Loaded {:12s}: {:4d} runs  ({})".format(ver, len(df), fname))

    mlp_df = pd.concat(all_mlp, ignore_index=True)
    mlp_df.to_csv(os.path.join(TABLE_DIR, "all_mlp_results.csv"), index=False)
    print("\nMLP total: {} runs".format(len(mlp_df)))

    # MLP summary (only configs with >= 5 folds)
    mlp_summary = summarize(mlp_df)
    mlp_summary.to_csv(os.path.join(TABLE_DIR, "all_mlp_summary.csv"), index=False)
    print("MLP summary: {} configs".format(len(mlp_summary)))

    # Top 15 MLP configs
    top = mlp_summary[mlp_summary["n_folds"] >= 5].head(15)
    print("\nTop 15 MLP configs (5-fold CV):")
    print(top[["version", "name", "feature_set", "n_features", "r2_mean", "r2_std", "mae_mean"]].to_string(index=False))

    # ── Tree results ──
    # 1. LightGBM sweep (already has fold column)
    lgbm_path = os.path.join(TABLE_DIR, "lgbm_sweep.csv")
    lgbm_df = pd.read_csv(lgbm_path)
    lgbm_out = pd.DataFrame({
        "version": "LGBM",
        "model_type": "LightGBM",
        "config": lgbm_df["config"],
        "feature_set": lgbm_df["feature_set"],
        "n_features": lgbm_df["n_features"],
        "fold": lgbm_df["fold"],
        "r2_uniform": lgbm_df["r2"],
        "mae_mean_pp": lgbm_df["mae"],
    })

    # 2. V8 trees (fold-0 only, old format)
    v8_path = os.path.join(TABLE_DIR, "mlp_v8_trees.csv")
    v8_df = pd.read_csv(v8_path)
    v8_records = []
    for _, row in v8_df.iterrows():
        name, fold = parse_old_model_name(row["model"])
        # Parse model type from name
        mt = "unknown"
        for prefix in ["extratrees", "rf", "catboost"]:
            if name.startswith(prefix):
                mt = prefix
                break
        v8_records.append({
            "version": "V8",
            "model_type": mt,
            "config": name,
            "feature_set": "v5_glcm_lbp",
            "n_features": row.get("n_features", np.nan),
            "fold": fold,
            "r2_uniform": row["r2_uniform"],
            "mae_mean_pp": row["mae_mean_pp"],
        })
    v8_out = pd.DataFrame(v8_records)

    # 3. Trees reduced features (fold-0 only, different format)
    trees_path = os.path.join(TABLE_DIR, "trees_reduced_features.csv")
    trees_df = pd.read_csv(trees_path)
    trees_out = pd.DataFrame({
        "version": "V4_trees",
        "model_type": trees_df["model_type"],
        "config": trees_df["run_name"],
        "feature_set": trees_df["feature_set"],
        "n_features": trees_df["n_features"],
        "fold": 0,
        "r2_uniform": trees_df["r2_uniform"],
        "mae_mean_pp": trees_df["mae_mean_pp"],
    })

    tree_df = pd.concat([lgbm_out, v8_out, trees_out], ignore_index=True)
    tree_df.to_csv(os.path.join(TABLE_DIR, "all_tree_results.csv"), index=False)
    print("\nTree total: {} runs".format(len(tree_df)))

    # Tree summary
    tree_summary = tree_df.groupby(["version", "model_type", "config", "feature_set"]).agg(
        r2_mean=("r2_uniform", "mean"),
        r2_std=("r2_uniform", "std"),
        mae_mean=("mae_mean_pp", "mean"),
        n_folds=("fold", "count"),
        n_features=("n_features", "first"),
    ).reset_index().sort_values("r2_mean", ascending=False)
    tree_summary.to_csv(os.path.join(TABLE_DIR, "all_tree_summary.csv"), index=False)
    print("Tree summary: {} configs".format(len(tree_summary)))

    top_trees = tree_summary[tree_summary["n_folds"] >= 5].head(15)
    print("\nTop 15 tree configs (5-fold CV):")
    print(top_trees[["version", "model_type", "config", "feature_set", "n_features",
                      "r2_mean", "r2_std", "mae_mean"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print("  MLP champion:  R2 = 0.787  MAE = 2.50pp  (bi_LBP, 864 feat)")
    print("  LightGBM best: R2 = 0.749  MAE = 2.94pp  (438 feat)")
    print("  Ridge:         R2 = 0.423  MAE = 5.63pp  (bi_LBP, 864 feat)")
    print("=" * 70)
    print("\nOutput files:")
    print("  {}".format(os.path.join(TABLE_DIR, "all_mlp_results.csv")))
    print("  {}".format(os.path.join(TABLE_DIR, "all_mlp_summary.csv")))
    print("  {}".format(os.path.join(TABLE_DIR, "all_tree_results.csv")))
    print("  {}".format(os.path.join(TABLE_DIR, "all_tree_summary.csv")))


if __name__ == "__main__":
    main()
