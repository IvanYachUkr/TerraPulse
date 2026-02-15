#!/usr/bin/env python3
"""
Phase 9 — Evaluation Beyond Accuracy.

Sections:
  A — Standard per-class metrics (MLP vs LightGBM vs Ridge)
  B — Change-specific metrics (false change, stability, calibration)
  C — Stress tests (noise injection, season dropout, feature-group ablation)
  D — Spatial failure analysis maps

Outputs to reports/phase9/tables/ and reports/phase9/figures/.
"""

import json
import os
import pickle
import re
import sys
import time
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.evaluation import (
    evaluate_model, r2_per_class, mae_per_class, rmse_per_class,
    r2_uniform, mae_mean, aitchison_mean, simplex_validity,
)
from src.splitting import get_fold_indices

# ─── Paths ────────────────────────────────────────────────────────────
TABLE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase9", "tables")
FIG_DIR   = os.path.join(PROJECT_ROOT, "reports", "phase9", "figures")
MLP_DIR   = os.path.join(PROJECT_ROOT, "models", "final_mlp")
TREE_DIR  = os.path.join(PROJECT_ROOT, "models", "final_tree")
RIDGE_DIR = os.path.join(PROJECT_ROOT, "models", "final_ridge")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES = ["tree_cover", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
CLASS_LABELS = ["Tree Cover", "Grassland", "Cropland",
                "Built-up", "Bare/Sparse", "Water"]
N_CLASSES = len(CLASS_NAMES)
N_FOLDS = 5

# ─── Plot styling ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

MODEL_COLORS = {"MLP": "#2563eb", "LightGBM": "#16a34a", "Ridge": "#dc2626"}
CLASS_COLORS = ["#228B22", "#90EE90", "#DAA520", "#DC143C", "#DEB887", "#4169E1"]


# =====================================================================
# Data loading
# =====================================================================

def load_data():
    """Load labels, OOF predictions, split info, and grid geometry."""
    labels_2021 = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    labels_2020 = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2020.parquet"))
    labels_change = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_change.parquet"))
    split = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    grid = gpd.read_file(os.path.join(PROCESSED_V2_DIR, "grid.gpkg"))

    mlp_oof = pd.read_parquet(os.path.join(MLP_DIR, "oof_predictions.parquet"))
    tree_oof = pd.read_parquet(os.path.join(TREE_DIR, "oof_predictions.parquet"))
    ridge_oof = pd.read_parquet(os.path.join(RIDGE_DIR, "oof_predictions.parquet"))

    # Align by cell_id
    y_true = labels_2021[CLASS_NAMES].values.astype(np.float64)
    y_2020 = labels_2020[CLASS_NAMES].values.astype(np.float64)

    y_pred_mlp = mlp_oof[[c + "_pred" for c in CLASS_NAMES]].values.astype(np.float64)
    y_pred_tree = tree_oof[[c + "_pred" for c in CLASS_NAMES]].values.astype(np.float64)
    y_pred_ridge = ridge_oof[[c + "_pred" for c in CLASS_NAMES]].values.astype(np.float64)

    cell_ids = labels_2021["cell_id"].values
    folds = split["fold_region_growing"].values

    return {
        "y_true": y_true, "y_2020": y_2020,
        "y_pred_mlp": y_pred_mlp, "y_pred_tree": y_pred_tree,
        "y_pred_ridge": y_pred_ridge,
        "labels_change": labels_change, "cell_ids": cell_ids,
        "folds": folds, "split": split, "grid": grid,
    }


# =====================================================================
# Section A: Standard per-class metrics
# =====================================================================

def kl_divergence(p, q, eps=1e-10):
    """KL(p || q) per sample. p, q shape (N, C)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    # Re-normalize after clipping
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=1)


def section_a(data):
    """Standard per-class metrics and comparison table."""
    print("\n" + "=" * 70)
    print("SECTION A: Standard Per-Class Metrics")
    print("=" * 70)

    y_true = data["y_true"]
    models = {
        "MLP": data["y_pred_mlp"],
        "LightGBM": data["y_pred_tree"],
        "Ridge": data["y_pred_ridge"],
    }

    # ── Per-class metrics table ──
    rows = []
    for model_name, y_pred in models.items():
        r2_cls = r2_per_class(y_true, y_pred, CLASS_NAMES)
        mae_cls = mae_per_class(y_true, y_pred, CLASS_NAMES)
        rmse_cls = rmse_per_class(y_true, y_pred, CLASS_NAMES)
        for c in CLASS_NAMES:
            rows.append({
                "model": model_name, "class": c,
                "r2": r2_cls[c], "mae_pp": mae_cls[c], "rmse_pp": rmse_cls[c],
            })
    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(os.path.join(TABLE_DIR, "per_class_metrics.csv"), index=False)
    print("  Saved per_class_metrics.csv")

    # ── Aggregate summary ──
    summary_rows = []
    for model_name, y_pred in models.items():
        sv = simplex_validity(y_pred)
        kl = kl_divergence(y_true, y_pred)
        from src.transforms import aitchison_distance
        aitch = aitchison_distance(y_true, y_pred, eps=1e-6)
        summary_rows.append({
            "model": model_name,
            "r2_uniform": r2_uniform(y_true, y_pred),
            "mae_mean_pp": mae_mean(y_true, y_pred),
            "aitchison_mean": float(np.mean(aitch)),
            "aitchison_median": float(np.median(aitch)),
            "kl_mean": float(np.mean(kl)),
            "kl_median": float(np.median(kl)),
            **sv,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLE_DIR, "aggregate_metrics.csv"), index=False)
    print("  Saved aggregate_metrics.csv")

    # Print to console
    print("\n  Per-class R2:")
    for c in CLASS_NAMES:
        mlp_r2 = detail_df[(detail_df.model == "MLP") & (detail_df["class"] == c)]["r2"].values[0]
        tree_r2 = detail_df[(detail_df.model == "LightGBM") & (detail_df["class"] == c)]["r2"].values[0]
        ridge_r2 = detail_df[(detail_df.model == "Ridge") & (detail_df["class"] == c)]["r2"].values[0]
        print("    {:15s}  MLP={:+.4f}  LGBM={:+.4f}  Ridge={:+.4f}".format(
            c, mlp_r2, tree_r2, ridge_r2))

    for _, row in summary_df.iterrows():
        print("\n  {}: R2={:.4f} MAE={:.2f}pp Aitchison={:.3f} KL={:.4f}".format(
            row.model, row.r2_uniform, row.mae_mean_pp, row.aitchison_mean, row.kl_mean))

    # ── Figures ──
    _fig_a1_per_class_r2(detail_df)
    _fig_a2_error_violins(y_true, models)
    _fig_a3_scatter_hexbin(y_true, models)
    _fig_a4_aitchison_hist(y_true, models)

    return detail_df, summary_df


def _fig_a1_per_class_r2(detail_df):
    """Bar chart: per-class R2 for MLP vs Ridge."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_CLASSES)
    model_list = list(MODEL_COLORS.keys())
    n_models = len(model_list)
    width = 0.8 / n_models

    for i, model in enumerate(model_list):
        vals = [detail_df[(detail_df.model == model) & (detail_df["class"] == c)]["r2"].values[0]
                for c in CLASS_NAMES]
        ax.bar(x + i * width, vals, width, label=model, color=MODEL_COLORS[model], alpha=0.85)

    ax.set_ylabel("R2")
    ax.set_title("Per-Class R2: MLP vs LightGBM vs Ridge")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylim(min(0, ax.get_ylim()[0] - 0.05), 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig01_per_class_r2.png"))
    plt.close(fig)
    print("  Saved fig01_per_class_r2.png")


def _fig_a2_error_violins(y_true, models):
    """Violin plots of per-sample absolute error per class."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.ravel()

    for ci, (cn, cl) in enumerate(zip(CLASS_NAMES, CLASS_LABELS)):
        ax = axes[ci]
        data_list = []
        labels = []
        for model_name, y_pred in models.items():
            err = np.abs(y_true[:, ci] - y_pred[:, ci]) * 100  # to pp
            data_list.append(err)
            labels.append(model_name)

        parts = ax.violinplot(data_list, showmedians=True, showextrema=False)
        for j, pc in enumerate(parts["bodies"]):
            color = MODEL_COLORS[labels[j]]
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")

        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(cl)
        ax.set_ylabel("Absolute Error (pp)" if ci % 3 == 0 else "")

    fig.suptitle("Per-Class Error Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig02_error_violins.png"))
    plt.close(fig)
    print("  Saved fig02_error_violins.png")


def _fig_a3_scatter_hexbin(y_true, models):
    """Hex-binned scatter: predicted vs actual for each class."""
    n_models = len(models)
    fig, axes = plt.subplots(n_models, N_CLASSES, figsize=(20, 7))

    for mi, (model_name, y_pred) in enumerate(models.items()):
        for ci in range(N_CLASSES):
            ax = axes[mi, ci]
            ax.hexbin(y_true[:, ci] * 100, y_pred[:, ci] * 100,
                      gridsize=40, cmap="YlOrRd", mincnt=1)
            lim = max(y_true[:, ci].max(), y_pred[:, ci].max()) * 100
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
            if mi == 0:
                ax.set_title(CLASS_LABELS[ci], fontsize=9)
            if ci == 0:
                ax.set_ylabel("{}\nPredicted (%)".format(model_name), fontsize=9)
            if mi == n_models - 1:
                ax.set_xlabel("Actual (%)", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("Predicted vs Actual Per Class", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig03_scatter_hexbin.png"))
    plt.close(fig)
    print("  Saved fig03_scatter_hexbin.png")


def _fig_a4_aitchison_hist(y_true, models):
    """Histogram of Aitchison distances."""
    from src.transforms import aitchison_distance
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, y_pred in models.items():
        d = aitchison_distance(y_true, y_pred, eps=1e-6)
        ax.hist(d, bins=80, alpha=0.55, label="{} (mean={:.3f})".format(model_name, d.mean()),
                color=MODEL_COLORS[model_name], density=True)

    ax.set_xlabel("Aitchison Distance")
    ax.set_ylabel("Density")
    ax.set_title("Aitchison Distance Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig04_aitchison_hist.png"))
    plt.close(fig)
    print("  Saved fig04_aitchison_hist.png")


# =====================================================================
# Section B: Change-specific metrics
# =====================================================================

def section_b(data):
    """Change-specific metrics: false-change, stability, calibration."""
    print("\n" + "=" * 70)
    print("SECTION B: Change-Specific Metrics")
    print("=" * 70)

    y_true = data["y_true"]
    y_2020 = data["y_2020"]

    # True change: actual 2021 - actual 2020
    delta_true = y_true - y_2020  # shape (N, 6)

    models = {
        "MLP": data["y_pred_mlp"],
        "LightGBM": data["y_pred_tree"],
        "Ridge": data["y_pred_ridge"],
    }

    rows = []
    for model_name, y_pred in models.items():
        # Predicted change: predicted 2021 - actual 2020
        delta_pred = y_pred - y_2020

        # Total absolute change per cell
        total_delta_true = np.sum(np.abs(delta_true), axis=1)
        total_delta_pred = np.sum(np.abs(delta_pred), axis=1)

        # Thresholds for change/no-change classification
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]
        for tau in thresholds:
            stable_mask = total_delta_true < tau
            changed_mask = total_delta_true >= tau

            n_stable = stable_mask.sum()
            n_changed = changed_mask.sum()

            # False-change: stable in truth, model predicts change
            if n_stable > 0:
                false_change_rate = np.mean(total_delta_pred[stable_mask] >= tau) * 100
                stability_mae = np.mean(np.abs(y_pred[stable_mask] - y_true[stable_mask])) * 100
            else:
                false_change_rate = np.nan
                stability_mae = np.nan

            # Missed-change: changed in truth, model predicts stable
            if n_changed > 0:
                missed_change_rate = np.mean(total_delta_pred[changed_mask] < tau) * 100
            else:
                missed_change_rate = np.nan

            rows.append({
                "model": model_name,
                "threshold": tau,
                "n_stable": int(n_stable),
                "n_changed": int(n_changed),
                "false_change_rate_pct": round(false_change_rate, 2),
                "missed_change_rate_pct": round(missed_change_rate, 2),
                "stability_mae_pp": round(stability_mae, 3),
            })

        # Per-class delta correlation
        for ci, cn in enumerate(CLASS_NAMES):
            corr_p, _ = stats.pearsonr(delta_true[:, ci], delta_pred[:, ci])
            corr_s, _ = stats.spearmanr(delta_true[:, ci], delta_pred[:, ci])
            rows_delta = rows  # reuse for printing
            print("  {} delta corr {:15s}: Pearson={:.4f}  Spearman={:.4f}".format(
                model_name, cn, corr_p, corr_s))

    change_df = pd.DataFrame(rows)
    change_df.to_csv(os.path.join(TABLE_DIR, "change_metrics.csv"), index=False)
    print("  Saved change_metrics.csv")

    # Print summary at key threshold
    for model_name in ["MLP", "LightGBM", "Ridge"]:
        sub = change_df[(change_df.model == model_name) & (change_df.threshold == 0.05)]
        if len(sub):
            r = sub.iloc[0]
            print("  {} (tau=0.05): false_change={:.1f}%  missed_change={:.1f}%  stability_MAE={:.3f}pp".format(
                model_name, r.false_change_rate_pct, r.missed_change_rate_pct, r.stability_mae_pp))

    # Per-class delta R2
    delta_r2_rows = []
    for model_name, y_pred in models.items():
        delta_pred = y_pred - y_2020
        for ci, cn in enumerate(CLASS_NAMES):
            r2_val = float(1.0 - np.sum((delta_true[:, ci] - delta_pred[:, ci]) ** 2) /
                          (np.sum((delta_true[:, ci] - delta_true[:, ci].mean()) ** 2) + 1e-10))
            delta_r2_rows.append({"model": model_name, "class": cn, "delta_r2": r2_val})
    delta_r2_df = pd.DataFrame(delta_r2_rows)

    # ── Figures ──
    _fig_b5_change_tradeoff(change_df)
    _fig_b6_change_calibration(data, models)
    _fig_b7_delta_correlation(data, models)

    return change_df


def _fig_b5_change_tradeoff(change_df):
    """False-change vs missed-change tradeoff curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in ["MLP", "LightGBM", "Ridge"]:
        sub = change_df[change_df.model == model_name].sort_values("threshold")
        ax.plot(sub.false_change_rate_pct, sub.missed_change_rate_pct,
                "o-", label=model_name, color=MODEL_COLORS[model_name], markersize=6)
        # Annotate key thresholds
        for _, r in sub.iterrows():
            if r.threshold in [0.02, 0.05, 0.10]:
                ax.annotate("{:.0f}%".format(r.threshold * 100),
                            (r.false_change_rate_pct, r.missed_change_rate_pct),
                            textcoords="offset points", xytext=(8, 4), fontsize=7)

    ax.set_xlabel("False Change Rate (%)")
    ax.set_ylabel("Missed Change Rate (%)")
    ax.set_title("Change Detection Tradeoff (varying threshold)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig05_change_tradeoff.png"))
    plt.close(fig)
    print("  Saved fig05_change_tradeoff.png")


def _fig_b6_change_calibration(data, models):
    """Change-magnitude calibration: true delta deciles vs predicted delta."""
    y_2020 = data["y_2020"]
    y_true = data["y_true"]
    delta_true_total = np.sum(np.abs(y_true - y_2020), axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_pred in models.items():
        delta_pred_total = np.sum(np.abs(y_pred - y_2020), axis=1)

        # Bin true delta into deciles
        n_bins = 10
        bin_edges = np.percentile(delta_true_total, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10
        bin_idx = np.digitize(delta_true_total, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bin_mean_true = []
        bin_mean_pred = []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                bin_mean_true.append(delta_true_total[mask].mean() * 100)
                bin_mean_pred.append(delta_pred_total[mask].mean() * 100)

        ax.plot(bin_mean_true, bin_mean_pred, "o-", label=model_name,
                color=MODEL_COLORS[model_name], markersize=6)

    # Perfect calibration line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="Perfect")

    ax.set_xlabel("True |Delta| (pp, decile mean)")
    ax.set_ylabel("Predicted |Delta| (pp, decile mean)")
    ax.set_title("Change-Magnitude Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig06_change_calibration.png"))
    plt.close(fig)
    print("  Saved fig06_change_calibration.png")


def _fig_b7_delta_correlation(data, models):
    """Per-class delta Pearson correlation bar chart."""
    y_2020 = data["y_2020"]
    y_true = data["y_true"]
    delta_true = y_true - y_2020

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_CLASSES)
    n_models = len(models)
    width = 0.8 / n_models

    for i, (model_name, y_pred) in enumerate(models.items()):
        delta_pred = y_pred - y_2020
        corrs = []
        for ci in range(N_CLASSES):
            corr, _ = stats.pearsonr(delta_true[:, ci], delta_pred[:, ci])
            corrs.append(corr)
        ax.bar(x + i * width, corrs, width, label=model_name,
               color=MODEL_COLORS[model_name], alpha=0.85)

    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Per-Class Change (Delta) Correlation")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig07_delta_correlation.png"))
    plt.close(fig)
    print("  Saved fig07_delta_correlation.png")


# =====================================================================
# Section C: Stress tests
# =====================================================================

def load_mlp_for_inference():
    """Load MLP models, scalers, and feature info for re-inference."""
    with open(os.path.join(MLP_DIR, "meta.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    arch = meta["architecture"]

    # Import model builder
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    from run_mlp_overnight_v4 import build_model, _cfg, _predict_batched, normalize_targets

    cfg = _cfg(0, "bi_LBP", arch["arch"], arch["activation"],
               arch["n_layers"], arch["d_model"],
               "batchnorm" if arch["norm"] == "batchnorm" else "none")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_models = []
    fold_scalers = []
    for fold_id in range(N_FOLDS):
        # Load scaler
        with open(os.path.join(MLP_DIR, "scaler_{}.pkl".format(fold_id)), "rb") as f:
            scaler = pickle.load(f)
        fold_scalers.append(scaler)

        # Load model
        net = build_model(cfg, len(feature_cols), device)
        state = torch.load(os.path.join(MLP_DIR, "fold_{}.pt".format(fold_id)),
                           map_location=device, weights_only=True)
        net.load_state_dict(state)
        net.eval()
        fold_models.append(net)

    return {
        "models": fold_models, "scalers": fold_scalers,
        "feature_cols": feature_cols, "cfg": cfg, "device": device,
        "predict_fn": _predict_batched,
    }


def run_mlp_inference(mlp_info, X_features, split_df, meta_split):
    """Run MLP inference on given features, return OOF predictions."""
    from run_mlp_overnight_v4 import _predict_batched

    device = mlp_info["device"]
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values

    oof = np.full((len(X_features), N_CLASSES), np.nan, dtype=np.float32)

    for fold_id in range(N_FOLDS):
        _, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta_split["tile_cols"], meta_split["tile_rows"],
            buffer_tiles=1,
        )

        scaler = mlp_info["scalers"][fold_id]
        X_test_s = scaler.transform(X_features[test_idx]).astype(np.float32)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

        net = mlp_info["models"][fold_id]
        preds = _predict_batched(net, X_test_t, device)
        oof[test_idx] = preds

    return oof


def load_tree_for_inference():
    """Load LightGBM models and feature info for re-inference."""
    with open(os.path.join(TREE_DIR, "meta.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    fold_models = []
    for fold_id in range(N_FOLDS):
        with open(os.path.join(TREE_DIR, "fold_{}.pkl".format(fold_id)), "rb") as f:
            model = pickle.load(f)
        fold_models.append(model)

    return {
        "models": fold_models,
        "feature_cols": feature_cols,
    }


def run_tree_inference(tree_info, X_features, split_df, meta_split):
    """Run LightGBM inference on given features, return OOF predictions."""
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values

    oof = np.full((len(X_features), N_CLASSES), np.nan, dtype=np.float32)

    for fold_id in range(N_FOLDS):
        _, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta_split["tile_cols"], meta_split["tile_rows"],
            buffer_tiles=1,
        )

        model = tree_info["models"][fold_id]
        preds = np.clip(model.predict(X_features[test_idx]), 0, 100)
        oof[test_idx] = preds

    return oof


def section_c(data):
    """Stress tests: noise injection, season dropout, feature-group ablation."""
    print("\n" + "=" * 70)
    print("SECTION C: Stress Tests (MLP + LightGBM)")
    print("=" * 70)

    # ── Load features and both models ──
    print("  Loading features and models...")
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))

    # Also load V2 features for LightGBM novel indices
    import pyarrow.parquet as pq
    v2_pq = os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet")
    v2_cols_available = [c for c in pq.read_schema(v2_pq).names if c != "cell_id"]

    mlp_info = load_mlp_for_inference()
    tree_info = load_tree_for_inference()

    mlp_feat_cols = mlp_info["feature_cols"]
    tree_feat_cols = tree_info["feature_cols"]

    # MLP features (all from features_merged_full)
    X_mlp = feat_df[mlp_feat_cols].values.astype(np.float32)
    np.nan_to_num(X_mlp, copy=False)

    # LightGBM features (some from features_merged_full, some from V2)
    tree_in_full = [c for c in tree_feat_cols if c in feat_df.columns]
    tree_in_v2 = [c for c in tree_feat_cols if c not in feat_df.columns and c in v2_cols_available]

    if tree_in_v2:
        v2_df = pd.read_parquet(v2_pq, columns=["cell_id"] + tree_in_v2)
        tree_df = feat_df[["cell_id"] + tree_in_full].merge(v2_df, on="cell_id", how="inner")
    else:
        tree_df = feat_df[["cell_id"] + tree_in_full]
    X_tree = tree_df[tree_feat_cols].values.astype(np.float32)
    np.nan_to_num(X_tree, copy=False)

    split_df = data["split"]
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta_split = json.load(f)
    y_true = data["y_true"]

    # Define model configs
    model_configs = [
        ("MLP", mlp_info, mlp_feat_cols, X_mlp,
         lambda info, X, s, m: run_mlp_inference(info, X, s, m)),
        ("LightGBM", tree_info, tree_feat_cols, X_tree,
         lambda info, X, s, m: run_tree_inference(info, X, s, m)),
    ]

    all_noise_rows = []
    all_season_rows = []
    all_ablation_rows = []

    for model_name, m_info, feat_cols, X_full, infer_fn in model_configs:
        print("\n  -- {} --".format(model_name))

        # Baseline
        oof_baseline = infer_fn(m_info, X_full, split_df, meta_split)
        r2_base = r2_uniform(y_true, oof_baseline)
        mae_base = mae_mean(y_true, oof_baseline)
        print("  Baseline: R2={:.4f}  MAE={:.2f}pp".format(r2_base, mae_base))

        # ── Test 1: Gaussian noise injection ──
        print("  Test 1: Gaussian noise injection")
        feat_stds = np.std(X_full, axis=0)
        noise_levels = [0.1, 0.25, 0.5, 1.0, 2.0]

        all_noise_rows.append({"model": model_name, "noise_sigma": 0.0,
                               "r2": r2_base, "mae_pp": mae_base})
        for sigma in noise_levels:
            rng = np.random.RandomState(42)
            noise = rng.randn(*X_full.shape).astype(np.float32) * feat_stds * sigma
            X_noisy = X_full + noise
            oof_noisy = infer_fn(m_info, X_noisy, split_df, meta_split)
            r2_n = r2_uniform(y_true, oof_noisy)
            mae_n = mae_mean(y_true, oof_noisy)
            all_noise_rows.append({"model": model_name, "noise_sigma": sigma,
                                   "r2": r2_n, "mae_pp": mae_n})
            print("    sigma={:.2f}: R2={:.4f} (delta={:+.4f})  MAE={:.2f}pp".format(
                sigma, r2_n, r2_n - r2_base, mae_n))

        # ── Test 2: Season dropout ──
        print("  Test 2: Season dropout")
        season_patterns = {}
        for c in feat_cols:
            m = re.search(r"(2020|2021)_(spring|summer|autumn)", c)
            if m:
                key = "{}_{}".format(m.group(1), m.group(2))
                season_patterns.setdefault(key, []).append(feat_cols.index(c))

        all_season_rows.append({"model": model_name, "season_dropped": "none",
                                "r2": r2_base, "mae_pp": mae_base, "n_zeroed": 0})
        for season, idxs in sorted(season_patterns.items()):
            X_dropped = X_full.copy()
            X_dropped[:, idxs] = 0.0
            oof_dropped = infer_fn(m_info, X_dropped, split_df, meta_split)
            r2_d = r2_uniform(y_true, oof_dropped)
            mae_d = mae_mean(y_true, oof_dropped)
            all_season_rows.append({"model": model_name, "season_dropped": season,
                                    "r2": r2_d, "mae_pp": mae_d, "n_zeroed": len(idxs)})
            print("    Drop {}: R2={:.4f} (delta={:+.4f})  MAE={:.2f}pp  ({} features zeroed)".format(
                season, r2_d, r2_d - r2_base, mae_d, len(idxs)))

        # ── Test 3: Feature group ablation ──
        print("  Test 3: Feature group ablation")
        group_defs = {
            "LBP":      [i for i, c in enumerate(feat_cols) if c.startswith("LBP_")],
            "Indices":  [i for i, c in enumerate(feat_cols)
                         if any(x in c.lower() for x in ["ndvi", "ndbi", "ndwi", "ndmi",
                                "evi", "savi", "bsi", "mndwi", "gndvi", "ndti", "ireci", "cri"])],
            "Bands":    [i for i, c in enumerate(feat_cols)
                         if any(x in c for x in ["B02", "B03", "B04", "B05", "B06", "B07",
                                "B08", "B8A", "B11", "B12"])],
        }

        all_ablation_rows.append({"model": model_name, "group_dropped": "none",
                                  "r2": r2_base, "mae_pp": mae_base,
                                  "n_zeroed": 0, "n_remaining": len(feat_cols)})
        for group_name, idxs in sorted(group_defs.items()):
            if not idxs:
                continue
            X_dropped = X_full.copy()
            X_dropped[:, idxs] = 0.0
            oof_dropped = infer_fn(m_info, X_dropped, split_df, meta_split)
            r2_d = r2_uniform(y_true, oof_dropped)
            mae_d = mae_mean(y_true, oof_dropped)
            all_ablation_rows.append({
                "model": model_name, "group_dropped": group_name,
                "r2": r2_d, "mae_pp": mae_d,
                "n_zeroed": len(idxs), "n_remaining": len(feat_cols) - len(idxs),
            })
            print("    Drop {}: R2={:.4f} (delta={:+.4f})  MAE={:.2f}pp  ({} features zeroed)".format(
                group_name, r2_d, r2_d - r2_base, mae_d, len(idxs)))

    # ── Save ──
    noise_df = pd.DataFrame(all_noise_rows)
    noise_df.to_csv(os.path.join(TABLE_DIR, "stress_noise.csv"), index=False)
    season_df = pd.DataFrame(all_season_rows)
    season_df.to_csv(os.path.join(TABLE_DIR, "stress_season_dropout.csv"), index=False)
    ablation_df = pd.DataFrame(all_ablation_rows)
    ablation_df.to_csv(os.path.join(TABLE_DIR, "stress_feature_ablation.csv"), index=False)

    # ── Figures ──
    _fig_c8_noise_degradation(noise_df)
    _fig_c9_season_dropout(season_df)
    _fig_c10_feature_ablation(ablation_df)

    # Cleanup
    del feat_df, X_mlp, X_tree
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return noise_df, season_df, ablation_df


def _fig_c8_noise_degradation(noise_df):
    """R2 degradation curve vs noise level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model_name in noise_df.model.unique():
        sub = noise_df[noise_df.model == model_name]
        color = MODEL_COLORS.get(model_name, "gray")
        ax1.plot(sub.noise_sigma, sub.r2, "o-", color=color,
                 markersize=8, linewidth=2, label=model_name)
        ax1.fill_between(sub.noise_sigma, sub.r2,
                         sub.r2.iloc[0], alpha=0.08, color=color)
        ax2.plot(sub.noise_sigma, sub.mae_pp, "o-", color=color,
                 markersize=8, linewidth=2, label=model_name)

    ax1.set_xlabel("Noise Level (sigma x feature_std)")
    ax1.set_ylabel("R2")
    ax1.set_title("R2 Degradation Under Gaussian Noise")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Noise Level (sigma x feature_std)")
    ax2.set_ylabel("MAE (pp)")
    ax2.set_title("MAE Increase Under Gaussian Noise")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Stress Test: Gaussian Noise Injection", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig08_noise_degradation.png"))
    plt.close(fig)
    print("  Saved fig08_noise_degradation.png")


def _fig_c9_season_dropout(season_df):
    """Season dropout impact grouped bar chart — MLP vs LightGBM."""
    models = season_df.model.unique()
    seasons = [s for s in season_df.season_dropped.unique() if s != "none"]
    x = np.arange(len(seasons))
    n_models = len(models)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for mi, model_name in enumerate(models):
        sub = season_df[season_df.model == model_name]
        base_r2 = sub[sub.season_dropped == "none"].r2.values[0]
        deltas = []
        for s in seasons:
            row = sub[sub.season_dropped == s]
            deltas.append(float(row.r2.values[0] - base_r2) if len(row) else 0.0)
        color = MODEL_COLORS.get(model_name, "gray")
        bars = ax.bar(x + mi * width, deltas, width, label=model_name,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, deltas):
            ax.annotate("{:.3f}".format(val),
                        (bar.get_x() + bar.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, -12 if val < 0 else 5),
                        ha="center", fontsize=7)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(seasons, rotation=30, ha="right")
    ax.set_ylabel("R2 Change from Baseline")
    ax.set_title("Impact of Dropping Each Season's Features")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig09_season_dropout.png"))
    plt.close(fig)
    print("  Saved fig09_season_dropout.png")


def _fig_c10_feature_ablation(ablation_df):
    """Feature group ablation grouped bar chart — MLP vs LightGBM."""
    models = ablation_df.model.unique()
    groups = [g for g in ablation_df.group_dropped.unique() if g != "none"]
    y = np.arange(len(groups))
    n_models = len(models)
    height = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, model_name in enumerate(models):
        sub = ablation_df[ablation_df.model == model_name]
        base_r2 = sub[sub.group_dropped == "none"].r2.values[0]
        deltas = []
        labels = []
        for g in groups:
            row = sub[sub.group_dropped == g]
            if len(row):
                deltas.append(float(row.r2.values[0] - base_r2))
                labels.append("{} ({} feat)".format(g, int(row.n_zeroed.values[0])))
            else:
                deltas.append(0.0)
                labels.append(g)
        color = MODEL_COLORS.get(model_name, "gray")
        bars = ax.barh(y + mi * height, deltas, height, label=model_name,
                       color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, deltas):
            ax.annotate("{:+.4f}".format(val),
                        (val, bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(5, 0),
                        va="center", fontsize=8)

    ax.set_yticks(y + height * (n_models - 1) / 2)
    ax.set_yticklabels(groups)
    ax.set_xlabel("R2 Change from Baseline")
    ax.set_title("Feature Group Ablation (zeroing out)")
    ax.axvline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig10_feature_ablation.png"))
    plt.close(fig)
    print("  Saved fig10_feature_ablation.png")


# =====================================================================
# Section D: Failure analysis maps
# =====================================================================

def section_d(data):
    """Spatial failure analysis: error heatmap, error by land-cover, fold maps."""
    print("\n" + "=" * 70)
    print("SECTION D: Spatial Failure Analysis (MLP + LightGBM)")
    print("=" * 70)

    y_true = data["y_true"]
    grid = data["grid"]
    folds = data["folds"]

    models_d = {
        "MLP": data["y_pred_mlp"],
        "LightGBM": data["y_pred_tree"],
    }

    # Dominant land-cover class
    dominant_class = np.argmax(y_true, axis=1)

    from src.transforms import aitchison_distance

    all_failure_rows = []

    for model_name, y_pred in models_d.items():
        print("\n  -- {} --".format(model_name))
        cell_mae = np.mean(np.abs(y_pred - y_true), axis=1) * 100
        aitch = aitchison_distance(y_true, y_pred, eps=1e-6)
        pred_entropy = -np.sum(np.clip(y_pred, 1e-10, 1) * np.log(np.clip(y_pred, 1e-10, 1)), axis=1)

        # ── Failure table ──
        for ci, cn in enumerate(CLASS_NAMES):
            mask = dominant_class == ci
            if mask.sum() > 0:
                all_failure_rows.append({
                    "model": model_name,
                    "dominant_class": cn,
                    "n_cells": int(mask.sum()),
                    "mae_pp": float(cell_mae[mask].mean()),
                    "mae_std_pp": float(cell_mae[mask].std()),
                    "aitchison_mean": float(aitch[mask].mean()),
                    "r2_uniform": float(r2_uniform(y_true[mask], y_pred[mask])),
                })

        # Print summary
        for ci, cn in enumerate(CLASS_NAMES):
            mask = dominant_class == ci
            if mask.sum() > 0:
                sub = [r for r in all_failure_rows
                       if r["model"] == model_name and r["dominant_class"] == cn]
                if sub:
                    r = sub[0]
                    print("    {:15s}: n={:5d}  MAE={:.2f}pp  Aitchison={:.3f}  R2={:.4f}".format(
                        cn, r["n_cells"], r["mae_pp"], r["aitchison_mean"], r["r2_uniform"]))

        # ── Fold-level metrics ──
        for fold_id in range(N_FOLDS):
            mask = folds == fold_id
            if mask.sum() > 0:
                r2_f = float(r2_uniform(y_true[mask], y_pred[mask]))
                mae_f = float(mae_mean(y_true[mask], y_pred[mask]))
                print("  Fold {}: n={:5d}  R2={:.4f}  MAE={:.2f}pp".format(
                    fold_id, int(mask.sum()), r2_f, mae_f))

        # ── Figures (per model, with suffix) ──
        suffix = "_mlp" if model_name == "MLP" else "_lgbm"
        _fig_d11_error_heatmap(grid, cell_mae, model_name, suffix)
        _fig_d12_error_by_landcover(cell_mae, dominant_class, model_name, suffix)
        _fig_d13_error_vs_aitchison(cell_mae, aitch, pred_entropy, model_name, suffix)
        _fig_d14_fold_error_map(grid, cell_mae, folds, model_name, suffix)

    failure_df = pd.DataFrame(all_failure_rows)
    failure_df.to_csv(os.path.join(TABLE_DIR, "failure_by_landcover.csv"), index=False)
    print("  Saved failure_by_landcover.csv")

    return failure_df


def _fig_d11_error_heatmap(grid, cell_mae, model_name, suffix):
    """Spatial error heatmap."""
    gdf = grid.copy()
    gdf["mae_pp"] = cell_mae

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(column="mae_pp", cmap="YlOrRd", ax=ax, legend=True,
             legend_kwds={"label": "MAE (pp)", "shrink": 0.6},
             vmin=0, vmax=np.percentile(cell_mae, 95))
    ax.set_title("{} Prediction Error Heatmap (MAE per cell)".format(model_name))
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    fig.tight_layout()
    fname = "fig11_error_heatmap{}.png".format(suffix)
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)
    print("  Saved {}".format(fname))


def _fig_d12_error_by_landcover(cell_mae, dominant_class, model_name, suffix):
    """Box plots of error by dominant land-cover class."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_by_class = []
    labels = []
    for ci, (cn, cl) in enumerate(zip(CLASS_NAMES, CLASS_LABELS)):
        mask = dominant_class == ci
        if mask.sum() > 0:
            data_by_class.append(cell_mae[mask])
            labels.append("{}\n(n={:,})".format(cl, mask.sum()))

    bp = ax.boxplot(data_by_class, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], CLASS_COLORS[:len(data_by_class)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("MAE (pp)")
    ax.set_title("Prediction Error by Dominant Land-Cover Class ({})".format(model_name))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fname = "fig12_error_by_landcover{}.png".format(suffix)
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)
    print("  Saved {}".format(fname))


def _fig_d13_error_vs_aitchison(cell_mae, aitch, pred_entropy, model_name, suffix):
    """Scatter: MAE vs Aitchison distance, colored by prediction entropy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MAE vs Aitchison
    ax1.hexbin(aitch, cell_mae, gridsize=50, cmap="YlOrRd", mincnt=1)
    ax1.set_xlabel("Aitchison Distance")
    ax1.set_ylabel("MAE (pp)")
    ax1.set_title("Error vs Compositional Distance")

    # MAE vs prediction entropy
    ax2.hexbin(pred_entropy, cell_mae, gridsize=50, cmap="YlOrRd", mincnt=1)
    ax2.set_xlabel("Prediction Entropy")
    ax2.set_ylabel("MAE (pp)")
    ax2.set_title("Error vs Prediction Uncertainty")

    fig.suptitle("{} Error Diagnostic Scatterplots".format(model_name), fontsize=14, y=1.02)
    fig.tight_layout()
    fname = "fig13_error_vs_aitchison{}.png".format(suffix)
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)
    print("  Saved {}".format(fname))


def _fig_d14_fold_error_map(grid, cell_mae, folds, model_name, suffix):
    """5-panel fold error map."""
    fig, axes = plt.subplots(1, N_FOLDS, figsize=(25, 8))

    vmax = np.percentile(cell_mae, 95)

    for fold_id in range(N_FOLDS):
        ax = axes[fold_id]
        mask = folds == fold_id
        gdf = grid.copy()
        gdf["mae_pp"] = np.nan
        gdf.loc[mask, "mae_pp"] = cell_mae[mask]

        # Plot background in gray
        grid.plot(ax=ax, color="#f0f0f0", edgecolor="none")
        # Plot fold cells colored by error
        gdf[mask].plot(column="mae_pp", cmap="YlOrRd", ax=ax,
                       vmin=0, vmax=vmax, legend=(fold_id == N_FOLDS - 1),
                       legend_kwds={"label": "MAE (pp)", "shrink": 0.8})

        ax.set_title("Fold {} (n={:,})".format(fold_id, mask.sum()), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Per-Fold Prediction Error Maps ({})".format(model_name), fontsize=14)
    fig.tight_layout()
    fname = "fig14_fold_error_map{}.png".format(suffix)
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)
    print("  Saved {}".format(fname))


# =====================================================================
# Main
# =====================================================================

def main():
    t0 = time.time()
    print("Phase 9 — Evaluation Beyond Accuracy")
    print("=" * 70)

    data = load_data()

    # Section A: Standard per-class metrics
    detail_df, summary_df = section_a(data)

    # Section B: Change-specific metrics
    change_df = section_b(data)

    # Section C: Stress tests
    noise_df, season_df, ablation_df = section_c(data)

    # Section D: Failure analysis maps
    failure_df = section_d(data)

    # ── Final summary ──
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("PHASE 9 COMPLETE — {:.1f}s".format(elapsed))
    print("=" * 70)
    print("Tables: {}".format(TABLE_DIR))
    print("Figures: {}".format(FIG_DIR))
    print("  14 figures + 7 CSV tables generated")


if __name__ == "__main__":
    main()
