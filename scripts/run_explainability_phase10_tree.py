#!/usr/bin/env python3
"""
Phase 10 (Tree) — LightGBM Explainability.

Sections:
  1 — Native feature importance (gain + split) + permutation importance
  2 — TreeSHAP exact values (global + per-class)
  3 — Helpful vs misleading explanation (same structure as MLP Phase 10)

Outputs to reports/phase10_tree/tables/ and reports/phase10_tree/figures/.
"""

import json
import os
import pickle
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.config import PROCESSED_V2_DIR, PROJECT_ROOT
from src.splitting import get_fold_indices

# --- Paths ---
TABLE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase10_tree", "tables")
FIG_DIR   = os.path.join(PROJECT_ROOT, "reports", "phase10_tree", "figures")
TREE_DIR  = os.path.join(PROJECT_ROOT, "models", "final_tree")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES = ["tree_cover", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
CLASS_LABELS = ["Tree Cover", "Grassland", "Cropland",
                "Built-up", "Bare/Sparse", "Water"]
N_CLASSES = len(CLASS_NAMES)
N_FOLDS = 5
SEED = 42

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
})

BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
ORANGE = "#ea580c"


# =====================================================================
# Model & data loading
# =====================================================================

def load_tree_fold(fold_id=0):
    """Load a single fold's LightGBM MultiOutputRegressor model."""
    with open(os.path.join(TREE_DIR, "meta.json")) as f:
        meta = json.load(f)

    with open(os.path.join(TREE_DIR, "fold_{}.pkl".format(fold_id)), "rb") as f:
        model = pickle.load(f)

    return model, meta


def load_fold_data(feature_cols, fold_id=0):
    """Load features, labels, and split info for a specific fold."""
    import pyarrow.parquet as pq

    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))

    # Some LightGBM features may be in v2 file
    v2_pq = os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet")
    v2_cols_available = [c for c in pq.read_schema(v2_pq).names if c != "cell_id"]

    in_full = [c for c in feature_cols if c in feat_df.columns]
    in_v2 = [c for c in feature_cols if c not in feat_df.columns and c in v2_cols_available]

    if in_v2:
        v2_df = pd.read_parquet(v2_pq, columns=["cell_id"] + in_v2)
        combined = feat_df[["cell_id"] + in_full].merge(v2_df, on="cell_id", how="inner")
    else:
        combined = feat_df[["cell_id"] + in_full]

    X_all = combined[feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df, combined

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        split_meta = json.load(f)

    train_idx, test_idx = get_fold_indices(
        tiles, folds_arr, fold_id, split_meta["tile_cols"], split_meta["tile_rows"],
        buffer_tiles=1,
    )

    return X_all, y, train_idx, test_idx


def r2_uniform(y_true, y_pred):
    """Multi-output R2 (uniform average)."""
    from sklearn.metrics import r2_score
    return float(r2_score(y_true, y_pred, multioutput="uniform_average"))


# =====================================================================
# Section 1: Native + Permutation importance
# =====================================================================

def section_1(model, X_all, y, train_idx, test_idx, feature_cols):
    """Native LightGBM importance + permutation importance."""
    print("\n" + "=" * 70)
    print("SECTION 1: Feature Importance (Native + Permutation)")
    print("=" * 70)

    X_test = X_all[test_idx]
    y_test = y[test_idx]

    # --- 1a: Native LightGBM importance ---
    print("  Computing native LightGBM importance (gain + split)...")

    # MultiOutputRegressor wraps N_CLASSES separate LGBMRegressors
    estimators = model.estimators_
    gain_imp = np.zeros(len(feature_cols))
    split_imp = np.zeros(len(feature_cols))

    for est in estimators:
        gain_imp += est.booster_.feature_importance(importance_type="gain")
        split_imp += est.booster_.feature_importance(importance_type="split")

    # Average across outputs
    gain_imp /= len(estimators)
    split_imp /= len(estimators)

    # Normalize
    gain_imp_norm = gain_imp / gain_imp.sum() if gain_imp.sum() > 0 else gain_imp
    split_imp_norm = split_imp / split_imp.sum() if split_imp.sum() > 0 else split_imp

    native_df = pd.DataFrame({
        "feature": feature_cols,
        "gain_importance": gain_imp,
        "gain_normalized": gain_imp_norm,
        "split_importance": split_imp,
        "split_normalized": split_imp_norm,
    }).sort_values("gain_normalized", ascending=False)
    native_df.to_csv(os.path.join(TABLE_DIR, "native_importance.csv"), index=False)
    print("  Saved native_importance.csv")

    print("\n  Top-10 by gain importance:")
    for _, r in native_df.head(10).iterrows():
        print("    {:45s}  gain={:.5f}  split={:.5f}".format(
            r.feature, r.gain_normalized, r.split_normalized))

    # --- 1b: Permutation importance ---
    print("\n  Computing permutation importance (3 repeats)...")
    from sklearn.metrics import r2_score

    base_pred = np.clip(model.predict(X_test), 0, 100)
    base_r2 = r2_uniform(y_test, base_pred)

    n_repeats = 3
    rng = np.random.RandomState(SEED)
    importances = np.zeros((len(feature_cols), n_repeats))

    for rep in range(n_repeats):
        for fi in range(len(feature_cols)):
            X_perm = X_test.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            perm_pred = np.clip(model.predict(X_perm), 0, 100)
            perm_r2 = r2_uniform(y_test, perm_pred)
            importances[fi, rep] = base_r2 - perm_r2
        print("    Repeat {}/{} done".format(rep + 1, n_repeats))

    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importances.mean(axis=1),
        "importance_std": importances.std(axis=1),
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(os.path.join(TABLE_DIR, "permutation_importance.csv"), index=False)
    print("  Saved permutation_importance.csv")

    print("\n  Top-10 permutation importance:")
    for _, r in perm_df.head(10).iterrows():
        print("    {:45s}  {:.5f} +/- {:.5f}".format(
            r.feature, r.importance_mean, r.importance_std))

    # --- Figures ---
    _fig_1_native_importance(native_df)
    _fig_2_permutation_importance(perm_df)
    _fig_3_gain_vs_perm(native_df, perm_df)

    return native_df, perm_df


def _fig_1_native_importance(native_df):
    """Top-30 native gain importance."""
    top = native_df.head(30)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    y_pos = np.arange(len(top))
    ax1.barh(y_pos, top.gain_normalized, color=GREEN, alpha=0.8, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top.feature, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Normalized Gain Importance")
    ax1.set_title("Top-30 by Gain")
    ax1.grid(True, alpha=0.3, axis="x")

    # Match order for split
    split_vals = [float(top[top.feature == f].split_normalized.values[0]) for f in top.feature]
    ax2.barh(y_pos, split_vals, color=BLUE, alpha=0.8, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top.feature, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Normalized Split Count")
    ax2.set_title("Top-30 by Split Count")
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle("LightGBM Native Feature Importance (fold-0)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig01_native_importance.png"))
    plt.close(fig)
    print("  Saved fig01_native_importance.png")


def _fig_2_permutation_importance(perm_df):
    """Top-30 permutation importance."""
    top = perm_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.importance_mean, xerr=top.importance_std,
            color=ORANGE, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.feature, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Permutation Importance (R2 decrease)")
    ax.set_title("Top-30 Features by Permutation Importance (LightGBM, fold-0)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig02_permutation_importance.png"))
    plt.close(fig)
    print("  Saved fig02_permutation_importance.png")


def _fig_3_gain_vs_perm(native_df, perm_df):
    """Scatter: gain importance vs permutation importance."""
    merged = native_df[["feature", "gain_normalized"]].merge(
        perm_df[["feature", "importance_mean"]], on="feature")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(merged.gain_normalized, merged.importance_mean,
               alpha=0.4, s=20, color=BLUE)

    # Label top-10 by either metric
    top_gain = set(native_df.head(10).feature)
    top_perm = set(perm_df.head(10).feature)
    top_both = top_gain | top_perm
    for _, r in merged.iterrows():
        if r.feature in top_both:
            ax.annotate(r.feature, (r.gain_normalized, r.importance_mean),
                        fontsize=5, alpha=0.7, rotation=15)

    ax.set_xlabel("Normalized Gain Importance")
    ax.set_ylabel("Permutation Importance (R2 decrease)")
    ax.set_title("Gain vs Permutation Importance (LightGBM)")
    ax.grid(True, alpha=0.3)

    # correlation
    r, _ = stats.pearsonr(merged.gain_normalized, merged.importance_mean)
    ax.text(0.05, 0.95, "Pearson r = {:.3f}".format(r),
            transform=ax.transAxes, fontsize=10, va="top")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig03_gain_vs_permutation.png"))
    plt.close(fig)
    print("  Saved fig03_gain_vs_permutation.png")


# =====================================================================
# Section 2: TreeSHAP
# =====================================================================

def section_2(model, X_all, y, train_idx, test_idx, feature_cols):
    """Exact TreeSHAP values for each per-class LightGBM estimator."""
    print("\n" + "=" * 70)
    print("SECTION 2: TreeSHAP (Exact)")
    print("=" * 70)

    X_test = X_all[test_idx]
    X_train = X_all[train_idx]

    rng = np.random.RandomState(SEED)
    explain_idx = rng.choice(len(X_test), min(2000, len(X_test)), replace=False)
    X_explain = X_test[explain_idx]

    estimators = model.estimators_  # list of N_CLASSES LGBMRegressors

    # Compute TreeSHAP per class
    shap_values_per_class = []
    for ci, est in enumerate(estimators):
        print("  TreeSHAP for class {} ({})...".format(ci, CLASS_NAMES[ci]))
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_explain)
        shap_values_per_class.append(sv)
        print("    shape: {}".format(sv.shape))

    # Global mean |SHAP| across all classes
    mean_abs_shap = np.zeros(len(feature_cols))
    for ci in range(N_CLASSES):
        mean_abs_shap += np.mean(np.abs(shap_values_per_class[ci]), axis=0)
    mean_abs_shap /= N_CLASSES

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    })
    for ci, cn in enumerate(CLASS_NAMES):
        shap_df["shap_{}".format(cn)] = np.mean(
            np.abs(shap_values_per_class[ci]), axis=0)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(TABLE_DIR, "treeshap_global_importance.csv"), index=False)
    print("  Saved treeshap_global_importance.csv")

    print("\n  Top-10 TreeSHAP global importance:")
    for _, r in shap_df.head(10).iterrows():
        print("    {:45s}  {:.6f}".format(r.feature, r.mean_abs_shap))

    # --- Figures ---
    _fig_4_treeshap_global(shap_df)
    _fig_5_beeswarm(shap_values_per_class, feature_cols, X_explain,
                     class_idx=0, class_name="tree_cover")
    _fig_6_beeswarm(shap_values_per_class, feature_cols, X_explain,
                     class_idx=4, class_name="bare_sparse")
    _fig_7_shap_per_class_heatmap(shap_df)

    return shap_df, shap_values_per_class, X_explain, explain_idx


def _fig_4_treeshap_global(shap_df):
    """Top-30 TreeSHAP global importance bar chart."""
    top = shap_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.mean_abs_shap, color=GREEN, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.feature, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (averaged across all 6 classes)")
    ax.set_title("Top-30 Features by TreeSHAP Global Importance (LightGBM)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig04_treeshap_global_top30.png"))
    plt.close(fig)
    print("  Saved fig04_treeshap_global_top30.png")


def _fig_5_beeswarm(shap_values, feature_cols, X_explain, class_idx, class_name):
    """SHAP beeswarm for a specific class output."""
    sv = shap_values[class_idx]  # (N, F)

    explanation = shap.Explanation(
        values=sv,
        data=X_explain,
        feature_names=feature_cols,
    )

    fig_num = 5 if class_idx == 0 else 6
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("TreeSHAP Values for {} Output (LightGBM, fold-0)".format(class_name))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,
                "fig{:02d}_treeshap_beeswarm_{}.png".format(fig_num, class_name)))
    plt.close("all")
    print("  Saved fig{:02d}_treeshap_beeswarm_{}.png".format(fig_num, class_name))


def _fig_6_beeswarm(shap_values, feature_cols, X_explain, class_idx, class_name):
    """Alias for beeswarm with different class."""
    _fig_5_beeswarm(shap_values, feature_cols, X_explain, class_idx, class_name)


def _fig_7_shap_per_class_heatmap(shap_df):
    """Heatmap: top-20 features x 6 classes."""
    top = shap_df.head(20)
    shap_cols = ["shap_{}".format(cn) for cn in CLASS_NAMES]
    data = top[shap_cols].values

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.feature.values, fontsize=7)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "{:.4f}".format(data[i, j]),
                    ha="center", va="center", fontsize=6,
                    color="white" if data[i, j] > data.max() * 0.6 else "black")

    fig.colorbar(im, label="Mean |SHAP|")
    ax.set_title("TreeSHAP Importance by Class (Top-20 Features)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig07_shap_per_class_heatmap.png"))
    plt.close(fig)
    print("  Saved fig07_shap_per_class_heatmap.png")


# =====================================================================
# Section 3: Helpful vs Misleading Explanation
# =====================================================================

def section_3(shap_values, feature_cols, X_explain, explain_idx,
              X_all, y, test_idx):
    """Construct helpful and misleading explanation examples."""
    print("\n" + "=" * 70)
    print("SECTION 3: Helpful vs Misleading Explanation")
    print("=" * 70)

    y_test = y[test_idx]

    # --- Helpful: NDVI -> tree_cover ---
    print("\n  Helpful explanation: NDVI_mean -> tree_cover")
    tree_idx = 0

    tree_true = y_test[explain_idx, tree_idx]
    high_tree_mask = tree_true > 0.6
    if high_tree_mask.sum() > 0:
        high_tree_positions = np.where(high_tree_mask)[0]
        sv_tree = shap_values[tree_idx][high_tree_positions]
        abs_total = np.sum(np.abs(sv_tree), axis=1)
        median_idx = high_tree_positions[np.argsort(abs_total)[len(abs_total) // 2]]
    else:
        median_idx = 0

    sample_shap = shap_values[tree_idx][median_idx]
    sample_data = X_explain[median_idx]

    ndvi_cols = [i for i, c in enumerate(feature_cols) if "NDVI" in c.upper() and "mean" in c.lower()]
    ndvi_names = [feature_cols[i] for i in ndvi_cols]
    ndvi_shap = [float(sample_shap[i]) for i in ndvi_cols]

    sample_tree_val = float(tree_true[median_idx]) if high_tree_mask.sum() > 0 else float(y_test[explain_idx[median_idx], 0])
    print("    Sample tree_cover = {:.3f}".format(sample_tree_val))
    for name, sv in zip(ndvi_names, ndvi_shap):
        print("    SHAP({}): {:+.6f}".format(name, sv))

    helpful_info = {
        "type": "helpful",
        "model": "LightGBM",
        "explanation": "NDVI_mean features have positive TreeSHAP for high tree_cover predictions",
        "rationale": "High NDVI physically indicates dense vegetation - the model learned a real physical relationship",
        "sample_tree_cover": sample_tree_val,
        "ndvi_features_shap": {n: s for n, s in zip(ndvi_names, ndvi_shap)},
    }

    # Waterfall plot
    explanation = shap.Explanation(
        values=sample_shap,
        base_values=0.0,
        data=sample_data,
        feature_names=feature_cols,
    )
    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title("Helpful: TreeSHAP for tree_cover (high-tree sample, LightGBM)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig08_helpful_waterfall.png"))
    plt.close("all")
    print("  Saved fig08_helpful_waterfall.png")

    # --- Misleading: Correlated band statistics ---
    print("\n  Misleading explanation: Correlated B02 band features")

    b02_mean_cols = [i for i, c in enumerate(feature_cols) if "B02_mean" in c]
    b02_median_cols = [i for i, c in enumerate(feature_cols) if "B02_median" in c]

    corr_pairs = []
    for mi in b02_mean_cols:
        mean_name = feature_cols[mi]
        season = mean_name.replace("B02_mean_", "")
        median_name = "B02_median_" + season
        if median_name in feature_cols:
            mdi = feature_cols.index(median_name)
            r, _ = stats.pearsonr(X_all[:, mi], X_all[:, mdi])
            shap_mean_imp = float(np.mean(np.abs(shap_values[0][:, mi])))
            shap_median_imp = float(np.mean(np.abs(shap_values[0][:, mdi])))
            corr_pairs.append({
                "season": season,
                "mean_feature": mean_name,
                "median_feature": median_name,
                "pearson_r": float(r),
                "shap_mean": shap_mean_imp,
                "shap_median": shap_median_imp,
                "shap_ratio": shap_mean_imp / max(shap_median_imp, 1e-10),
            })
            print("    {} vs {}: r={:.4f}, SHAP ratio={:.1f}x".format(
                mean_name, median_name, r, shap_mean_imp / max(shap_median_imp, 1e-10)))

    misleading_info = {
        "type": "misleading",
        "model": "LightGBM",
        "explanation": "B02_mean and B02_median are near-perfectly correlated (r>0.99) "
                       "but TreeSHAP assigns different importance to each",
        "note": "TreeSHAP is exact for trees, but correlated features still cause "
                "arbitrary credit splitting in the tree structure",
        "pairs": corr_pairs,
    }

    with open(os.path.join(TABLE_DIR, "helpful_example.json"), "w") as f:
        json.dump(helpful_info, f, indent=2)
    with open(os.path.join(TABLE_DIR, "misleading_example.json"), "w") as f:
        json.dump(misleading_info, f, indent=2)
    print("  Saved helpful_example.json, misleading_example.json")

    # Fig 9: SHAP credit split
    if corr_pairs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        seasons = [p["season"] for p in corr_pairs]
        shap_means = [p["shap_mean"] for p in corr_pairs]
        shap_medians = [p["shap_median"] for p in corr_pairs]
        correlations = [p["pearson_r"] for p in corr_pairs]

        x = np.arange(len(seasons))
        width = 0.35
        ax1.bar(x - width/2, shap_means, width, label="B02_mean", color=BLUE, alpha=0.8)
        ax1.bar(x + width/2, shap_medians, width, label="B02_median", color=RED, alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(seasons, rotation=30, ha="right", fontsize=7)
        ax1.set_ylabel("Mean |TreeSHAP| for tree_cover")
        ax1.set_title("TreeSHAP Credit Split\n(near-identical features)")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, correlations, color=GREEN, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(seasons, rotation=30, ha="right", fontsize=7)
        ax2.set_ylabel("Pearson Correlation")
        ax2.set_title("Correlation Between B02_mean & B02_median")
        ax2.set_ylim(0.95, 1.0)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle("LightGBM: TreeSHAP Credit Split Among Correlated Features",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig09_shap_credit_split.png"))
        plt.close(fig)
        print("  Saved fig09_shap_credit_split.png")

    return helpful_info, misleading_info


# =====================================================================
# Main
# =====================================================================

def main():
    t0 = time.time()
    print("Phase 10 (Tree) -- LightGBM Explainability")
    print("=" * 70)

    fold_id = 0
    print("Using fold {} for all analyses".format(fold_id))

    # Load model and data
    print("Loading LightGBM model and data...")
    model, meta = load_tree_fold(fold_id)
    feature_cols = meta["feature_cols"]
    X_all, y, train_idx, test_idx = load_fold_data(feature_cols, fold_id)
    print("  Train: {}, Test: {}, Features: {}".format(
        len(train_idx), len(test_idx), len(feature_cols)))

    # Section 1: Native + permutation importance
    native_df, perm_df = section_1(model, X_all, y, train_idx, test_idx, feature_cols)

    # Section 2: TreeSHAP
    shap_df, shap_values, X_explain, explain_idx = section_2(
        model, X_all, y, train_idx, test_idx, feature_cols)

    # Section 3: Helpful vs misleading
    helpful, misleading = section_3(
        shap_values, feature_cols, X_explain, explain_idx,
        X_all, y, test_idx)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("PHASE 10 (TREE) COMPLETE -- {:.1f}s".format(elapsed))
    print("=" * 70)
    print("Tables: {}".format(TABLE_DIR))
    print("Figures: {}".format(FIG_DIR))
    n_figs = len([f for f in os.listdir(FIG_DIR) if f.endswith(".png")])
    n_tables = len([f for f in os.listdir(TABLE_DIR)])
    print("  {} figures + {} tables generated".format(n_figs, n_tables))


if __name__ == "__main__":
    main()
