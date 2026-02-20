#!/usr/bin/env python3
"""
Generate SHAP deep-dive plots for the dashboard:
  1. Compute TreeSHAP for LightGBM (per-sample values)
  2. Generate beeswarm summary plots for MLP and LightGBM (1 per class)
  3. Generate dependence scatter plots (top-5 features per class)

Output: src/dashboard/data/shap_plots/*.png
"""

import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROC = os.path.join(ROOT, "data", "processed", "v2")
OUT_DIR = os.path.join(ROOT, "src", "dashboard", "data", "shap_plots")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
CLASS_LABELS = ["Tree Cover", "Grassland", "Cropland", "Built-up", "Bare/Sparse", "Water"]
CLASS_COLORS = ["#2d6a4f", "#6a994e", "#f4a261", "#e76f51", "#d4a373", "#0096c7"]

# Dark theme for dashboard
DARK_BG = "#0f172a"
DARK_CARD = "#1e293b"
DARK_TEXT = "#e2e8f0"
DARK_SUBTEXT = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_CARD,
    "text.color": DARK_TEXT,
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": DARK_SUBTEXT,
    "ytick.color": DARK_SUBTEXT,
    "axes.edgecolor": "#334155",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": DARK_BG,
    "font.family": "sans-serif",
    "font.size": 10,
})


def short_name(fname, max_len=25):
    """Shorten feature name for axis labels."""
    # Replace common prefixes
    s = fname.replace("_mean_", "_m_").replace("_std_", "_s_")
    s = s.replace("_median_", "_md_").replace("_q25_", "_q1_").replace("_q75_", "_q3_")
    s = s.replace("_finite_frac_", "_ff_").replace("_range_", "_rng_")
    s = s.replace("_min_", "_mn_").replace("_max_", "_mx_")
    if len(s) > max_len:
        s = s[:max_len-1] + "."
    return s


# =====================================================================
# Step 1: Load or compute SHAP caches
# =====================================================================

def load_mlp_shap():
    """Load MLP DeepSHAP cache."""
    cache_path = os.path.join(ROOT, "reports", "phase10", "tables", "_shap_cache.pkl")
    print(f"Loading MLP SHAP cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    # shap_values: list of 6 arrays, each (2000, 864)
    # explain_data: (2000, 864)
    # explain_idx: (2000,)
    return cache["shap_values"], cache["explain_data"], cache["explain_idx"]


def compute_tree_shap():
    """Compute TreeSHAP for LightGBM and cache results."""
    cache_path = os.path.join(ROOT, "reports", "phase10_tree", "tables", "_treeshap_cache.pkl")
    if os.path.exists(cache_path):
        print(f"Loading existing TreeSHAP cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["shap_values"], cache["explain_data"], cache["explain_idx"]

    print("Computing TreeSHAP for LightGBM (this may take a minute)...")
    import shap

    # Load tree model (fold 0 for SHAP)
    tree_dir = os.path.join(ROOT, "models", "final_tree")
    with open(os.path.join(tree_dir, "meta.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    with open(os.path.join(tree_dir, "fold_0.pkl"), "rb") as f:
        model = pickle.load(f)

    # Load features
    feat_path = os.path.join(PROC, "features_merged_full.parquet")
    feat_df = pd.read_parquet(feat_path)

    # Tree model may need features from v2 parquet too
    import pyarrow.parquet as pq
    v2_path = os.path.join(PROC, "features_bands_indices_v2.parquet")
    v2_cols_available = [c for c in pq.read_schema(v2_path).names if c != "cell_id"]
    tree_in_full = [c for c in feature_cols if c in feat_df.columns]
    tree_in_v2 = [c for c in feature_cols if c not in feat_df.columns and c in v2_cols_available]

    if tree_in_v2:
        v2_df = pd.read_parquet(v2_path, columns=["cell_id"] + tree_in_v2)
        tree_df = feat_df[["cell_id"] + tree_in_full].merge(v2_df, on="cell_id", how="inner")
    else:
        tree_df = feat_df[["cell_id"] + tree_in_full]

    X_all = tree_df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)

    # Sample 2000 points
    rng = np.random.RandomState(42)
    n_samples = min(2000, len(X_all))
    idx = rng.choice(len(X_all), n_samples, replace=False)
    X_explain = X_all[idx]

    # LightGBM is a MultiOutputRegressor — compute SHAP per estimator
    shap_values_per_class = []
    for ci, est in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_explain)
        shap_values_per_class.append(sv)
        print(f"  Class {ci}/{len(model.estimators_)}: {CLASS_NAMES[ci]} done")

    # Save cache
    cache = {
        "shap_values": shap_values_per_class,
        "explain_data": X_explain,
        "explain_idx": idx,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Saved TreeSHAP cache to {cache_path}")

    return shap_values_per_class, X_explain, idx


# =====================================================================
# Step 2: Generate beeswarm plots
# =====================================================================

def make_beeswarm(shap_values, X_explain, feature_names, class_idx, class_label, model_name, top_n=15):
    """
    Custom beeswarm-style plot: horizontal strip plot of SHAP values
    colored by feature value, sorted by mean |SHAP|.
    """
    sv = shap_values[class_idx]  # (n_samples, n_features)
    n_features = sv.shape[1]

    # Get top features by mean |SHAP|
    mean_abs = np.mean(np.abs(sv), axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1.2))

    for row_i, feat_i in enumerate(top_idx):
        shap_col = sv[:, feat_i]
        feat_col = X_explain[:, feat_i]

        # Normalize feature values to [0, 1] for color mapping
        fmin, fmax = np.nanmin(feat_col), np.nanmax(feat_col)
        if fmax - fmin > 1e-10:
            feat_norm = (feat_col - fmin) / (fmax - fmin)
        else:
            feat_norm = np.full_like(feat_col, 0.5)

        # Jitter y position
        y_pos = top_n - 1 - row_i
        jitter = np.random.RandomState(feat_i).uniform(-0.3, 0.3, len(shap_col))

        # Color: blue (low) -> red (high)
        cmap = plt.cm.RdBu_r
        colors = cmap(feat_norm)

        # Sort by |shap| so extreme points draw on top
        order = np.argsort(np.abs(shap_col))
        ax.scatter(
            shap_col[order], y_pos + jitter[order],
            c=colors[order], s=4, alpha=0.6, linewidths=0, rasterized=True,
        )

    # Labels
    labels = [short_name(feature_names[i]) for i in top_idx]
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.axvline(0, color="#475569", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=9)
    ax.set_title(f"{model_name} - {class_label}", fontsize=11, fontweight="bold", pad=10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=8, color=DARK_SUBTEXT)
    cbar.ax.tick_params(colors=DARK_SUBTEXT, labelsize=8)

    ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
    fig.tight_layout()
    return fig


# =====================================================================
# Step 3: Generate dependence plots
# =====================================================================

def make_dependence(shap_values, X_explain, feature_names, class_idx, feat_idx, class_label, model_name):
    """
    SHAP dependence plot: feature value (x) vs SHAP value (y),
    colored by the most correlated interaction feature.
    """
    sv_class = shap_values[class_idx]
    shap_col = sv_class[:, feat_idx]
    feat_col = X_explain[:, feat_idx]
    fname = feature_names[feat_idx]

    # Find interaction feature (highest abs correlation with SHAP values)
    n_feat = sv_class.shape[1]
    best_corr = 0
    best_interact_idx = feat_idx
    for j in range(n_feat):
        if j == feat_idx:
            continue
        c = np.abs(np.corrcoef(X_explain[:, j], shap_col)[0, 1])
        if not np.isnan(c) and c > best_corr:
            best_corr = c
            best_interact_idx = j

    interact_col = X_explain[:, best_interact_idx]
    interact_name = feature_names[best_interact_idx]

    # Normalize interaction feature for color
    imin, imax = np.nanmin(interact_col), np.nanmax(interact_col)
    if imax - imin > 1e-10:
        interact_norm = (interact_col - imin) / (imax - imin)
    else:
        interact_norm = np.full_like(interact_col, 0.5)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.cm.RdBu_r
    colors = cmap(interact_norm)

    order = np.argsort(np.abs(shap_col))
    ax.scatter(feat_col[order], shap_col[order], c=colors[order],
               s=8, alpha=0.6, linewidths=0, rasterized=True)

    ax.axhline(0, color="#475569", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(short_name(fname, 35), fontsize=9)
    ax.set_ylabel(f"SHAP value for {class_label}", fontsize=9)
    ax.set_title(f"{model_name} - {class_label}", fontsize=10, fontweight="bold", pad=8)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label(short_name(interact_name, 25), fontsize=8, color=DARK_SUBTEXT)
    cbar.ax.tick_params(colors=DARK_SUBTEXT, labelsize=7)

    fig.tight_layout()
    return fig


# =====================================================================
# Main
# =====================================================================

def generate_all_plots(shap_values, X_explain, feature_names, model_tag, model_label, top_dep=5):
    """Generate full set of plots for one model."""
    n_classes = len(CLASS_NAMES)

    for ci in range(n_classes):
        cls = CLASS_NAMES[ci]
        cl = CLASS_LABELS[ci]

        # Beeswarm
        fig = make_beeswarm(shap_values, X_explain, feature_names, ci, cl, model_label)
        path = os.path.join(OUT_DIR, f"beeswarm_{model_tag}_{cls}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {os.path.basename(path)}")

        # Top-N dependence plots
        sv_class = shap_values[ci]
        mean_abs = np.mean(np.abs(sv_class), axis=0)
        top_feat_idx = np.argsort(mean_abs)[-top_dep:][::-1]

        for rank, fi in enumerate(top_feat_idx):
            fig = make_dependence(shap_values, X_explain, feature_names, ci, fi, cl, model_label)
            path = os.path.join(OUT_DIR, f"dep_{model_tag}_{cls}_{rank}.png")
            fig.savefig(path)
            plt.close(fig)

        print(f"  Saved {top_dep} dependence plots for {cls}")

    # Also generate a manifest JSON for the frontend
    manifest = {"model": model_tag, "label": model_label, "classes": []}
    for ci in range(n_classes):
        cls = CLASS_NAMES[ci]
        sv_class = shap_values[ci]
        mean_abs = np.mean(np.abs(sv_class), axis=0)
        top_feat_idx = np.argsort(mean_abs)[-top_dep:][::-1]

        class_info = {
            "class": cls,
            "label": CLASS_LABELS[ci],
            "beeswarm": f"beeswarm_{model_tag}_{cls}.png",
            "dependence": [
                {
                    "feature": feature_names[fi],
                    "file": f"dep_{model_tag}_{cls}_{rank}.png",
                }
                for rank, fi in enumerate(top_feat_idx)
            ],
        }
        manifest["classes"].append(class_info)
    return manifest


def main():
    sep = "=" * 60
    print(sep)
    print("SHAP Deep-Dive Plot Generation")
    print(sep)

    # ── MLP ──
    print("\n--- MLP (DeepSHAP) ---")
    mlp_sv, mlp_X, mlp_idx = load_mlp_shap()
    with open(os.path.join(ROOT, "models", "final_mlp", "meta.json")) as f:
        mlp_meta = json.load(f)
    mlp_features = mlp_meta["feature_cols"]
    print(f"  {len(mlp_sv)} classes, {mlp_sv[0].shape[0]} samples, {mlp_sv[0].shape[1]} features")

    mlp_manifest = generate_all_plots(mlp_sv, mlp_X, mlp_features, "mlp", "MLP")

    # ── LightGBM ──
    print("\n--- LightGBM (TreeSHAP) ---")
    tree_sv, tree_X, tree_idx = compute_tree_shap()
    with open(os.path.join(ROOT, "models", "final_tree", "meta.json")) as f:
        tree_meta = json.load(f)
    tree_features = tree_meta["feature_cols"]
    print(f"  {len(tree_sv)} classes, {tree_sv[0].shape[0]} samples, {tree_sv[0].shape[1]} features")

    tree_manifest = generate_all_plots(tree_sv, tree_X, tree_features, "tree", "LightGBM")

    # Save manifest
    manifest = {"mlp": mlp_manifest, "tree": tree_manifest}
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest.json")

    # Count output files
    pngs = [f for f in os.listdir(OUT_DIR) if f.endswith(".png")]
    print(f"\n{sep}")
    print(f"DONE: {len(pngs)} PNG files generated in {OUT_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
