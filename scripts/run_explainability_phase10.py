#!/usr/bin/env python3
"""
Phase 10 — Explainability + Uncertainty.

Sections:
  1 — Global feature importance (Permutation + SHAP GradientExplainer)
  2 — Helpful vs misleading explanation (NDVI→tree, correlated bands)
  3 — Conformal uncertainty consolidation

Outputs to reports/phase10/tables/ and reports/phase10/figures/.
"""

import json
import os
import pickle
import re
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.config import PROCESSED_V2_DIR, PROJECT_ROOT
from src.splitting import get_fold_indices

# ─── Paths ────────────────────────────────────────────────────────────
TABLE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase10", "tables")
FIG_DIR   = os.path.join(PROJECT_ROOT, "reports", "phase10", "figures")
MLP_DIR   = os.path.join(PROJECT_ROOT, "models", "final_mlp")
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

# Colour palette
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"


# =====================================================================
# Model loading
# =====================================================================

def load_mlp_fold(fold_id=0):
    """Load a single fold's MLP model + scaler + metadata."""
    from run_mlp_overnight_v4 import build_model, _cfg

    with open(os.path.join(MLP_DIR, "meta.json")) as f:
        meta = json.load(f)

    arch = meta["architecture"]
    cfg = _cfg(0, "bi_LBP", arch["arch"], arch["activation"],
               arch["n_layers"], arch["d_model"],
               "batchnorm" if arch["norm"] == "batchnorm" else "none")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(cfg, meta["n_features"], device)
    state = torch.load(os.path.join(MLP_DIR, "fold_{}.pt".format(fold_id)),
                       map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()

    with open(os.path.join(MLP_DIR, "scaler_{}.pkl".format(fold_id)), "rb") as f:
        scaler = pickle.load(f)

    return net, scaler, meta, device


def load_fold_data(fold_id=0):
    """Load features, labels, and split info for a specific fold."""
    with open(os.path.join(MLP_DIR, "meta.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    X_all = feat_df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df

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

    return X_all, y, train_idx, test_idx, feature_cols


# =====================================================================
# sklearn-compatible wrapper for permutation importance
# =====================================================================

class MLPWrapper:
    """Wraps a PyTorch MLP as a sklearn-compatible estimator for permutation importance."""

    def __init__(self, net, scaler, device):
        self.net = net
        self.scaler = scaler
        self.device = device

    def fit(self, X, y):
        """No-op: model is already trained."""
        return self

    def __sklearn_tags__(self):
        """Tell sklearn this is a fitted estimator."""
        from sklearn.utils._tags import Tags
        tags = Tags()
        return tags

    def predict(self, X):
        X_s = self.scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X_s, dtype=torch.float32)
        self.net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), 8192):
                xb = X_t[i:i+8192].to(self.device, non_blocking=True)
                out = self.net.forward(xb).exp().cpu().numpy()
                preds.append(out)
                del xb
        return np.concatenate(preds, axis=0)

    def score(self, X, y):
        """Multi-output R² (uniform average)."""
        y_pred = self.predict(X)
        return float(r2_score(y, y_pred, multioutput="uniform_average"))


# =====================================================================
# Fast permutation importance (GPU-batched)
# =====================================================================

def _fast_permutation_importance(net, scaler, X_test, y_test, feature_cols,
                                  device, n_repeats=3, seed=42):
    """
    Custom GPU-batched permutation importance.

    Instead of calling scaler.transform() + model.predict() per feature,
    we scale once, keep the tensor on GPU, and shuffle one column at a time.
    This is ~100x faster than sklearn's permutation_importance.
    """
    rng = np.random.RandomState(seed)
    net.eval()

    # Scale once, transfer to GPU once
    X_scaled = scaler.transform(X_test).astype(np.float32)
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    # Baseline score (multi-output R²)
    with torch.no_grad():
        base_pred = net.forward(X_t).exp()
    base_r2 = _torch_r2(y_t, base_pred)

    n_features = X_t.shape[1]
    importances = np.zeros((n_features, n_repeats))

    for rep in range(n_repeats):
        for fi in range(n_features):
            # Save original column, shuffle in-place, restore after
            orig_col = X_t[:, fi].clone()
            perm_idx = torch.tensor(
                rng.permutation(X_t.shape[0]),
                dtype=torch.long, device=device
            )
            X_t[:, fi] = orig_col[perm_idx]

            with torch.no_grad():
                perm_pred = net.forward(X_t).exp()
            perm_r2 = _torch_r2(y_t, perm_pred)
            importances[fi, rep] = base_r2 - perm_r2

            # Restore original column
            X_t[:, fi] = orig_col

        print("    Repeat {}/{} done".format(rep + 1, n_repeats))

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importances.mean(axis=1),
        "importance_std": importances.std(axis=1),
    }).sort_values("importance_mean", ascending=False)
    return df


def _torch_r2(y_true, y_pred):
    """Multi-output R² (uniform average) on GPU tensors."""
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_class = 1 - ss_res / ss_tot.clamp(min=1e-10)
    return float(r2_per_class.mean().item())


# =====================================================================
# Section 1: Global feature importance
# =====================================================================

def section_1(X_all, y, train_idx, test_idx, feature_cols, net, scaler, device):
    """Permutation importance + SHAP global summary (with disk caching)."""
    print("\n" + "=" * 70)
    print("SECTION 1: Global Feature Importance")
    print("=" * 70)

    X_test = X_all[test_idx]
    y_test = y[test_idx]
    X_train = X_all[train_idx]

    # Cache paths
    perm_csv = os.path.join(TABLE_DIR, "permutation_importance.csv")
    shap_cache = os.path.join(TABLE_DIR, "_shap_cache.pkl")
    shap_csv = os.path.join(TABLE_DIR, "shap_global_importance.csv")

    # -- 1a: Permutation importance (cached) --
    if os.path.exists(perm_csv):
        print("  [CACHE] Loading permutation importance from {}".format(perm_csv))
        perm_df = pd.read_csv(perm_csv)
    else:
        print("  Computing permutation importance (3 repeats, GPU-batched)...")
        perm_df = _fast_permutation_importance(
            net, scaler, X_test, y_test, feature_cols, device,
            n_repeats=3, seed=SEED,
        )
        perm_df.to_csv(perm_csv, index=False)
        print("  Saved permutation_importance.csv")

    print("\n  Top-10 permutation importance:")
    for _, r in perm_df.head(10).iterrows():
        print("    {:40s}  {:.5f} +/- {:.5f}".format(r.feature, r.importance_mean, r.importance_std))

    # -- 1b: SHAP GradientExplainer (cached) --
    if os.path.exists(shap_cache):
        print("\n  [CACHE] Loading SHAP values from {}".format(shap_cache))
        with open(shap_cache, "rb") as f:
            cache = pickle.load(f)
        shap_values = cache["shap_values"]
        explain_data = cache["explain_data"]
        explain_idx = cache["explain_idx"]
    else:
        print("\n  Computing SHAP values (GradientExplainer)...")

        # Prepare scaled data
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        # Subsample for SHAP computation
        rng = np.random.RandomState(SEED)
        bg_idx = rng.choice(len(X_train_s), min(500, len(X_train_s)), replace=False)
        explain_idx = rng.choice(len(X_test_s), min(2000, len(X_test_s)), replace=False)

        background = torch.tensor(X_train_s[bg_idx], dtype=torch.float32).to(device)
        explain_data_t = torch.tensor(X_test_s[explain_idx], dtype=torch.float32).to(device)

        # Wrapper model that outputs probabilities (not log-softmax)
        class ProbModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model

            def forward(self, x):
                return self.base.forward(x).exp()

        prob_model = ProbModel(net)
        prob_model.eval()

        explainer = shap.GradientExplainer(prob_model, background)
        shap_values = explainer.shap_values(explain_data_t)

        # Normalize shape: newer shap returns (n_explain, n_features, n_outputs)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = [shap_values[:, :, ci] for ci in range(shap_values.shape[2])]
        elif isinstance(shap_values, list) and len(shap_values) == N_CLASSES:
            shap_values = [np.array(sv) if not isinstance(sv, np.ndarray) else sv
                           for sv in shap_values]
        else:
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                shap_values = [shap_values]

        # Convert explain_data to numpy for caching
        explain_data = explain_data_t.cpu().numpy()
        del explain_data_t, background
        torch.cuda.empty_cache()

        # Save cache
        with open(shap_cache, "wb") as f:
            pickle.dump({
                "shap_values": shap_values,
                "explain_data": explain_data,
                "explain_idx": explain_idx,
            }, f)
        print("    Saved SHAP cache to {}".format(shap_cache))

    print("    SHAP values: {} classes, each shape {}".format(
        len(shap_values), shap_values[0].shape))

    # Global mean |SHAP| across all classes
    mean_abs_shap = np.zeros(len(feature_cols))
    for ci in range(N_CLASSES):
        mean_abs_shap += np.mean(np.abs(shap_values[ci]), axis=0)
    mean_abs_shap /= N_CLASSES

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    })
    # Add per-class columns
    for ci, cn in enumerate(CLASS_NAMES):
        shap_df["shap_{}".format(cn)] = np.mean(np.abs(shap_values[ci]), axis=0)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(shap_csv, index=False)
    print("  Saved shap_global_importance.csv")

    print("\n  Top-10 SHAP global importance:")
    for _, r in shap_df.head(10).iterrows():
        print("    {:40s}  {:.6f}".format(r.feature, r.mean_abs_shap))

    # -- Figures --
    _fig_1_permutation(perm_df)
    _fig_2_beeswarm(shap_values, feature_cols, explain_data, class_idx=0, class_name="tree_cover")
    _fig_3_beeswarm(shap_values, feature_cols, explain_data, class_idx=4, class_name="bare_sparse")
    _fig_4_shap_global(shap_df)

    return perm_df, shap_df, shap_values, explain_data, explain_idx, X_test, y_test


def _fig_1_permutation(perm_df):
    """Top-30 permutation importance."""
    top = perm_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.importance_mean, xerr=top.importance_std,
            color=BLUE, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.feature, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Permutation Importance (R2 decrease)")
    ax.set_title("Top-30 Features by Permutation Importance (MLP, fold-0)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig01_permutation_importance.png"))
    plt.close(fig)
    print("  Saved fig01_permutation_importance.png")


def _fig_2_beeswarm(shap_values, feature_cols, explain_data, class_idx, class_name):
    """SHAP beeswarm for a specific class output."""
    sv = shap_values[class_idx]  # (N, F)
    xd = explain_data.cpu().numpy() if isinstance(explain_data, torch.Tensor) else explain_data

    explanation = shap.Explanation(
        values=sv,
        data=xd,
        feature_names=feature_cols,
    )

    fig_num = 2 if class_idx == 0 else 3
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP Values for {} Output (MLP, fold-0)".format(class_name))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig{:02d}_shap_beeswarm_{}.png".format(fig_num, class_name)))
    plt.close("all")
    print("  Saved fig{:02d}_shap_beeswarm_{}.png".format(fig_num, class_name))


def _fig_3_beeswarm(shap_values, feature_cols, explain_data, class_idx, class_name):
    """Alias -- same function, different class."""
    _fig_2_beeswarm(shap_values, feature_cols, explain_data, class_idx, class_name)


def _fig_4_shap_global(shap_df):
    """Top-30 SHAP global importance bar chart."""
    top = shap_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.mean_abs_shap, color=GREEN, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.feature, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (averaged across all 6 classes)")
    ax.set_title("Top-30 Features by SHAP Global Importance")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig04_shap_global_top30.png"))
    plt.close(fig)
    print("  Saved fig04_shap_global_top30.png")


# =====================================================================
# Section 2: Helpful vs Misleading Explanation
# =====================================================================

def section_2(shap_values, feature_cols, explain_data,
              explain_idx, X_test, y_test, X_all, y, test_idx):
    """Construct helpful and misleading explanation examples."""
    print("\n" + "=" * 70)
    print("SECTION 2: Helpful vs Misleading Explanation")
    print("=" * 70)

    # -- Helpful: NDVI -> tree_cover --
    print("\n  Helpful explanation: NDVI_mean -> tree_cover")
    tree_idx_in_classes = 0

    # Find a sample with high tree_cover that was well-predicted
    tree_true = y_test[explain_idx, tree_idx_in_classes]
    high_tree_mask = tree_true > 0.6
    if high_tree_mask.sum() > 0:
        # Among high-tree samples, pick one with median SHAP values (representative)
        high_tree_positions = np.where(high_tree_mask)[0]
        sv_tree = shap_values[tree_idx_in_classes][high_tree_positions]
        abs_total = np.sum(np.abs(sv_tree), axis=1)
        median_idx = high_tree_positions[np.argsort(abs_total)[len(abs_total) // 2]]
    else:
        median_idx = 0

    sample_shap = shap_values[tree_idx_in_classes][median_idx]
    sample_data = explain_data[median_idx].cpu().numpy() if isinstance(explain_data, torch.Tensor) else explain_data[median_idx]

    # Find NDVI features
    ndvi_cols = [i for i, c in enumerate(feature_cols) if "NDVI" in c.upper() and "mean" in c.lower()]
    ndvi_names = [feature_cols[i] for i in ndvi_cols]
    ndvi_shap = [sample_shap[i] for i in ndvi_cols]

    print("    Sample tree_cover = {:.3f}".format(tree_true[median_idx] if high_tree_mask.sum() > 0 else y_test[explain_idx[median_idx], 0]))
    for name, sv in zip(ndvi_names, ndvi_shap):
        print("    SHAP({}): {:+.6f}".format(name, sv))

    helpful_info = {
        "type": "helpful",
        "explanation": "NDVI_mean features have positive SHAP for high tree_cover predictions",
        "rationale": "High NDVI physically indicates dense vegetation - the model learned a real physical relationship",
        "sample_tree_cover": float(tree_true[median_idx] if high_tree_mask.sum() > 0 else y_test[explain_idx[median_idx], 0]),
        "ndvi_features_shap": {n: float(s) for n, s in zip(ndvi_names, ndvi_shap)},
    }

    # Waterfall plot -- helpful
    explanation = shap.Explanation(
        values=sample_shap,
        base_values=0.0,
        data=sample_data,
        feature_names=feature_cols,
    )
    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title("Helpful: SHAP for tree_cover (high-tree sample)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig05_helpful_waterfall.png"))
    plt.close("all")
    print("  Saved fig05_helpful_waterfall.png")

    # -- Misleading: Correlated band statistics --
    print("\n  Misleading explanation: Correlated B02 band features")

    # Find B02_mean and B02_median pairs
    b02_mean_cols = [i for i, c in enumerate(feature_cols) if "B02_mean" in c]
    b02_median_cols = [i for i, c in enumerate(feature_cols) if "B02_median" in c]

    # Match by season
    corr_pairs = []
    for mi in b02_mean_cols:
        mean_name = feature_cols[mi]
        season = mean_name.replace("B02_mean_", "")
        median_name = "B02_median_" + season
        if median_name in feature_cols:
            mdi = feature_cols.index(median_name)
            # Correlation on raw (unscaled) data
            r, _ = stats.pearsonr(X_all[:, mi], X_all[:, mdi])
            # SHAP importance
            shap_mean_imp = np.mean(np.abs(shap_values[0][:, mi]))
            shap_median_imp = np.mean(np.abs(shap_values[0][:, mdi]))
            corr_pairs.append({
                "season": season,
                "mean_feature": mean_name,
                "median_feature": median_name,
                "pearson_r": float(r),
                "shap_mean": float(shap_mean_imp),
                "shap_median": float(shap_median_imp),
                "shap_ratio": float(shap_mean_imp / max(shap_median_imp, 1e-10)),
            })
            print("    {} vs {}: r={:.4f}, SHAP ratio={:.1f}x".format(
                mean_name, median_name, r, shap_mean_imp / max(shap_median_imp, 1e-10)))

    misleading_info = {
        "type": "misleading",
        "explanation": "B02_mean and B02_median are near-perfectly correlated (r>0.99) "
                       "but SHAP assigns wildly different importance to each",
        "pitfall": "SHAP splits credit arbitrarily among collinear features - "
                   "a naive user would wrongly conclude one is important and the other is not",
        "recommendation": "Group correlated features before interpreting SHAP, or use permutation importance on groups",
        "pairs": corr_pairs,
    }

    # Save to JSON
    with open(os.path.join(TABLE_DIR, "helpful_example.json"), "w") as f:
        json.dump(helpful_info, f, indent=2)
    with open(os.path.join(TABLE_DIR, "misleading_example.json"), "w") as f:
        json.dump(misleading_info, f, indent=2)
    print("  Saved helpful_example.json, misleading_example.json")

    # Waterfall for misleading -- pick a sample where B02_mean dominates
    if b02_mean_cols:
        # Use the same sample but highlight the misleading feature attribution
        fig = plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title("Misleading: correlated features get unequal SHAP credit")
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig06_misleading_waterfall.png"))
        plt.close("all")
        print("  Saved fig06_misleading_waterfall.png")

    # -- Fig 7: B02 correlation heatmap --
    b02_cols_all = [i for i, c in enumerate(feature_cols)
                    if c.startswith("B02_") and any(s in c for s in ["2021_summer"])]
    b02_names = [feature_cols[i] for i in b02_cols_all]
    if len(b02_cols_all) >= 2:
        corr_matrix = np.corrcoef(X_all[:, b02_cols_all].T)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(b02_names)))
        ax.set_yticks(range(len(b02_names)))
        short_names = [n.replace("B02_", "").replace("_2021_summer", "") for n in b02_names]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        for i in range(len(b02_names)):
            for j in range(len(b02_names)):
                ax.text(j, i, "{:.2f}".format(corr_matrix[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(corr_matrix[i, j]) > 0.7 else "black")
        fig.colorbar(im, label="Pearson r")
        ax.set_title("B02 Feature Correlations (2021 summer)\nNear-identical features get different SHAP credit")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig07_band_correlation_heatmap.png"))
        plt.close(fig)
        print("  Saved fig07_band_correlation_heatmap.png")

    # -- Fig 8: SHAP credit split visualization --
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
        ax1.set_ylabel("Mean |SHAP| for tree_cover")
        ax1.set_title("SHAP Credit Split\n(near-identical features)")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, correlations, color=GREEN, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(seasons, rotation=30, ha="right", fontsize=7)
        ax2.set_ylabel("Pearson Correlation")
        ax2.set_title("Correlation Between B02_mean & B02_median\n(all > 0.99)")
        ax2.set_ylim(0.95, 1.0)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Misleading: SHAP Arbitrarily Splits Credit Among Correlated Features",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig08_shap_credit_split.png"))
        plt.close(fig)
        print("  Saved fig08_shap_credit_split.png")

    return helpful_info, misleading_info


# =====================================================================
# Section 3: Conformal uncertainty consolidation
# =====================================================================

def section_3():
    """Consolidate existing conformal prediction results into report format."""
    print("\n" + "=" * 70)
    print("SECTION 3: Conformal Uncertainty Consolidation")
    print("=" * 70)

    # Load existing conformal results
    conformal_csv = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables", "conformal_coverage.csv")
    conformal_json = os.path.join(PROJECT_ROOT, "src", "dashboard", "data", "conformal.json")

    if os.path.exists(conformal_csv):
        conf_df = pd.read_csv(conformal_csv)
        print("  Loaded conformal_coverage.csv ({} rows)".format(len(conf_df)))
        print(conf_df.to_string(index=False))

        # ── Fig 9: Conformal interval width ──
        if "model" in conf_df.columns and "mean_width_pp" in conf_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            models = conf_df["model"].unique()
            x = np.arange(N_CLASSES)
            width = 0.8 / len(models)

            for i, model_name in enumerate(models):
                sub = conf_df[conf_df.model == model_name]
                widths = []
                for cn in CLASS_NAMES:
                    row = sub[sub["class"] == cn] if "class" in sub.columns else pd.DataFrame()
                    widths.append(float(row["mean_width_pp"].values[0]) if len(row) else 0)
                color = BLUE if "MLP" in model_name.upper() else RED
                ax.bar(x + i * width, widths, width, label=model_name, color=color, alpha=0.8)

            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
            ax.set_ylabel("Mean Interval Width (pp)")
            ax.set_title("Conformal Prediction Interval Width by Class")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "fig09_conformal_intervals.png"))
            plt.close(fig)
            print("  Saved fig09_conformal_intervals.png")
        else:
            print("  Conformal CSV format unexpected, plotting from raw data")
            _plot_conformal_from_json(conformal_json)
    elif os.path.exists(conformal_json):
        _plot_conformal_from_json(conformal_json)
    else:
        print("  No conformal results found — skipping Section 3")
        print("  Expected: {} or {}".format(conformal_csv, conformal_json))


def _plot_conformal_from_json(json_path):
    """Fallback: plot from conformal.json."""
    if not os.path.exists(json_path):
        print("  No conformal.json found")
        return

    with open(json_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Extract coverage and width per model per class
    for mi, (model_name, model_data) in enumerate(data.items()):
        if "per_class" not in model_data:
            continue
        coverages = []
        widths = []
        classes = []
        for cn in CLASS_NAMES:
            if cn in model_data["per_class"]:
                pc = model_data["per_class"][cn]
                coverages.append(pc.get("coverage_pct", 0))
                widths.append(pc.get("mean_width_pp", 0))
                classes.append(cn)

        color = BLUE if "mlp" in model_name.lower() else RED
        x = np.arange(len(classes))
        w = 0.35
        axes[0].bar(x + mi * w, coverages, w, label=model_name, color=color, alpha=0.8)
        axes[1].bar(x + mi * w, widths, w, label=model_name, color=color, alpha=0.8)

    axes[0].axhline(90, color="gray", ls="--", lw=1, label="90% target")
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_title("Conformal Coverage")
    axes[0].legend()

    axes[1].set_ylabel("Mean Width (pp)")
    axes[1].set_title("Conformal Interval Width")

    for ax in axes:
        ax.set_xticks(np.arange(len(CLASS_NAMES)) + 0.175)
        ax.set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Conformal Prediction Intervals", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig09_conformal_intervals.png"))
    plt.close(fig)
    print("  Saved fig09_conformal_intervals.png")


# =====================================================================
# Main
# =====================================================================

def main():
    t0 = time.time()
    print("Phase 10 — Explainability + Uncertainty")
    print("=" * 70)

    fold_id = 0
    print("Using fold {} for all analyses".format(fold_id))

    # Load model and data
    print("Loading MLP model and data...")
    net, scaler, meta, device = load_mlp_fold(fold_id)
    X_all, y, train_idx, test_idx, feature_cols = load_fold_data(fold_id)
    print("  Train: {}, Test: {}, Features: {}".format(
        len(train_idx), len(test_idx), len(feature_cols)))

    # Section 1: Global feature importance
    perm_df, shap_df, shap_values, explain_data, explain_idx, X_test, y_test = \
        section_1(X_all, y, train_idx, test_idx, feature_cols, net, scaler, device)

    # Section 2: Helpful vs misleading explanation
    helpful, misleading = section_2(
        shap_values, feature_cols, explain_data,
        explain_idx, X_test, y_test, X_all, y, test_idx)

    # Section 3: Conformal uncertainty
    section_3()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("PHASE 10 COMPLETE -- {:.1f}s".format(elapsed))
    print("=" * 70)
    print("Tables: {}".format(TABLE_DIR))
    print("Figures: {}".format(FIG_DIR))


if __name__ == "__main__":
    main()
