"""
Phase 8: Model training, evaluation, and benchmarking.

Trains all model families on ILR-transformed land-cover proportions
using the spatial region-growing split from Phase 7.

Outputs:
  reports/phase8/tables/model_benchmark.csv         -- cross-model metrics
  reports/phase8/tables/model_benchmark_per_class.csv
  reports/phase8/tables/coefficient_stability.csv    -- Ridge/ElasticNet
  reports/phase8/tables/morans_i_residuals.csv       -- spatial diagnostics
  reports/phase8/tables/conformal_coverage.csv       -- prediction intervals
  reports/phase8/figures/*.png
  data/processed/v2/predictions_*.parquet            -- per-model predictions

Usage:
    python scripts/run_phase8.py                        # all models, core features
    python scripts/run_phase8.py --models ridge,catboost
    python scripts/run_phase8.py --feature-set full
    python scripts/run_phase8.py --skip-hpo             # default params, fast
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CFG, PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.conformal import (
    calibrate_conformal,
    conformal_coverage_report,
    predict_interval,
)
from src.models.evaluation import evaluate_model, simplex_validity
from src.models.spatial_diagnostics import compute_residual_morans_i
from src.splitting import get_fold_indices
from src.transforms import helmert_basis, ilr_forward, ilr_inverse, pivot_basis

# Silence optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =====================================================================
# Config
# =====================================================================

SPLIT_CFG = CFG["split"]
N_FOLDS = SPLIT_CFG["n_folds"]
SEED = SPLIT_CFG["seed"]

CLASS_NAMES = CFG["worldcover"]["class_names"]
N_CLASSES = len(CLASS_NAMES)

# Merged feature control columns (not model features)
CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}

# Directories
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
TBL_DIR = os.path.join(REPORTS_DIR, "tables")


# =====================================================================
# Helpers
# =====================================================================

def save_fig(fig, name, dpi=150):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  * Saved figure: {path}")


def save_table(df, name):
    os.makedirs(TBL_DIR, exist_ok=True)
    path = os.path.join(TBL_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  * Saved table: {path}")


# =====================================================================
# Data loading
# =====================================================================

def load_data(feature_set="core"):
    """Load features, labels, and split assignments."""
    print("=" * 70)
    print(f"Loading data (feature_set={feature_set})")
    print("=" * 70)

    # Features
    feat_path = os.path.join(PROCESSED_V2_DIR, f"features_merged_{feature_set}.parquet")
    feat_df = pd.read_parquet(feat_path)
    print(f"  Features: {feat_df.shape}")

    # Labels (2021 proportions)
    labels_path = os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet")
    labels_df = pd.read_parquet(labels_path)
    print(f"  Labels: {labels_df.shape}")

    # Split
    split_path = os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet")
    split_df = pd.read_parquet(split_path)
    print(f"  Split: {split_df.shape}")

    # Align on cell_id
    assert (feat_df["cell_id"].values == labels_df["cell_id"].values).all(), \
        "cell_id mismatch between features and labels"
    assert (feat_df["cell_id"].values == split_df["cell_id"].values).all(), \
        "cell_id mismatch between features and split"

    # Extract arrays
    feature_cols = [c for c in feat_df.columns
                    if c not in CONTROL_COLS and feat_df[c].dtype in ("float64", "float32", "int64")]
    X = feat_df[feature_cols].values.astype(np.float64)
    y = labels_df[CLASS_NAMES].values.astype(np.float64)

    cell_ids = feat_df["cell_id"].values
    fold_assignments = split_df["fold_region_growing"].values
    tile_groups = split_df["tile_group"].values

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Feature columns: {len(feature_cols)}")

    # NaN check
    n_nan = np.isnan(X).sum()
    assert n_nan == 0, f"X has {n_nan} NaN values"

    # Split metadata
    meta_path = os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")
    with open(meta_path) as f:
        split_meta = json.load(f)

    n_tile_cols = split_meta["tile_cols"]
    n_tile_rows = split_meta["tile_rows"]

    return (X, y, cell_ids, fold_assignments, tile_groups,
            feature_cols, n_tile_cols, n_tile_rows)


# =====================================================================
# Model factory
# =====================================================================

def build_model(model_name, **hparams):
    """Instantiate a model by name."""
    if model_name == "ridge":
        from src.models.linear import RidgeModel
        return RidgeModel(alpha=hparams.get("alpha", 1.0))

    elif model_name == "elasticnet":
        from src.models.linear import ElasticNetModel
        return ElasticNetModel(
            alpha=hparams.get("alpha", 0.1),
            l1_ratio=hparams.get("l1_ratio", 0.5),
        )

    elif model_name == "extratrees":
        from src.models.forests import ExtraTreesModel
        return ExtraTreesModel(
            n_estimators=hparams.get("n_estimators", 500),
            max_features=hparams.get("max_features", "sqrt"),
            min_samples_leaf=hparams.get("min_samples_leaf", 1),
        )

    elif model_name == "rf":
        from src.models.forests import RandomForestModel
        return RandomForestModel(
            n_estimators=hparams.get("n_estimators", 500),
            max_features=hparams.get("max_features", "sqrt"),
            min_samples_leaf=hparams.get("min_samples_leaf", 1),
        )

    elif model_name == "catboost":
        from src.models.boosting import CatBoostModel
        return CatBoostModel(
            iterations=hparams.get("iterations", 1000),
            depth=hparams.get("depth", 6),
            learning_rate=hparams.get("learning_rate", 0.1),
            l2_leaf_reg=hparams.get("l2_leaf_reg", 3.0),
            random_strength=hparams.get("random_strength", 1.0),
        )

    elif model_name == "mlp":
        from src.models.mlp_torch import SoftmaxMLP
        return SoftmaxMLP(
            n_classes=N_CLASSES,
            hidden_dim=hparams.get("hidden_dim", 256),
            n_layers=hparams.get("n_layers", 3),
            dropout=hparams.get("dropout", 0.15),
            lr=hparams.get("lr", 1e-3),
            weight_decay=hparams.get("weight_decay", 1e-4),
        )

    elif model_name == "dirichlet_mlp":
        from src.models.mlp_torch import DirichletMLP
        return DirichletMLP(
            n_classes=N_CLASSES,
            hidden_dim=hparams.get("hidden_dim", 256),
            n_layers=hparams.get("n_layers", 3),
            dropout=hparams.get("dropout", 0.15),
            lr=hparams.get("lr", 1e-3),
            weight_decay=hparams.get("weight_decay", 1e-4),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# MLP-based models that train on raw proportions, not ILR
MLP_MODELS = {"mlp", "dirichlet_mlp"}

# All available model names
ALL_MODELS = ["ridge", "elasticnet", "extratrees", "rf", "catboost", "mlp"]


# =====================================================================
# Hyperparameter optimization
# =====================================================================

def _inner_cv_score(model_name, hparams, X, z, y, fold_assignments,
                    tile_groups, n_tile_cols, n_tile_rows,
                    inner_folds=(1, 2, 3, 4), buffer_tiles=1):
    """
    Evaluate hyperparameters via inner CV (folds 1*4).

    Returns mean R* (uniform) across inner folds.
    """
    from sklearn.metrics import r2_score
    from src.transforms import ilr_inverse

    scores = []
    is_mlp_model = model_name in MLP_MODELS

    for val_fold in inner_folds:
        # Train on all inner folds except val_fold
        train_folds = [f for f in inner_folds if f != val_fold]
        val_mask = fold_assignments == val_fold

        # Build train mask: all inner folds except val, with buffer against val
        train_mask = np.isin(fold_assignments, train_folds)

        # Apply buffer: exclude train tiles near val tiles
        if buffer_tiles > 0:
            from src.splitting import _precompute_tile_neighbors
            val_tiles = set(np.unique(tile_groups[val_mask]))
            tile_nbrs = _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles)
            buffer_set = set()
            for vt in val_tiles:
                for nbr in tile_nbrs.get(vt, set()):
                    if nbr not in val_tiles:
                        buffer_set.add(nbr)
            buffer_mask = np.isin(tile_groups, list(buffer_set))
            train_mask = train_mask & (~buffer_mask)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) < 10 or len(val_idx) < 10:
            continue

        model = build_model(model_name, **hparams)

        if is_mlp_model:
            model.fit(X[train_idx], y[train_idx],
                      X_val=X[val_idx], z_val_or_y=y[val_idx])
            y_pred = model.predict_proportions(X[val_idx])
        elif model_name == "catboost":
            model.fit(X[train_idx], z[train_idx],
                      X_val=X[val_idx], z_val=z[val_idx])
            y_pred = model.predict_proportions(X[val_idx])
        else:
            model.fit(X[train_idx], z[train_idx])
            y_pred = model.predict_proportions(X[val_idx])

        try:
            score = r2_score(y[val_idx], y_pred, multioutput="uniform_average")
            if np.isfinite(score):
                scores.append(score)
        except ValueError:
            pass

    return float(np.mean(scores)) if scores else -999.0


def tune_ridge(X, z, y, fold_assignments, tile_groups,
               n_tile_cols, n_tile_rows, buffer_tiles=1):
    """Grid search over alpha for Ridge."""
    print("  * Tuning Ridge (grid search over alpha)...")
    alphas = [1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
    best_alpha, best_score = 1.0, -999.0
    for alpha in alphas:
        score = _inner_cv_score(
            "ridge", {"alpha": alpha}, X, z, y,
            fold_assignments, tile_groups, n_tile_cols, n_tile_rows,
            buffer_tiles=buffer_tiles,
        )
        if score > best_score:
            best_score = score
            best_alpha = alpha
    print(f"     Best alpha={best_alpha}, inner R*={best_score:.4f}")
    return {"alpha": best_alpha}


def tune_elasticnet(X, z, y, fold_assignments, tile_groups,
                    n_tile_cols, n_tile_rows, n_trials=30, buffer_tiles=1):
    """Optuna HPO for ElasticNet."""
    print(f"  * Tuning ElasticNet ({n_trials} Optuna trials)...")

    def objective(trial):
        params = {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        }
        return _inner_cv_score(
            "elasticnet", params, X, z, y,
            fold_assignments, tile_groups, n_tile_cols, n_tile_rows,
            buffer_tiles=buffer_tiles,
        )

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     Best: {study.best_params}, inner R*={study.best_value:.4f}")
    return study.best_params


def tune_forest(model_name, X, z, y, fold_assignments, tile_groups,
                n_tile_cols, n_tile_rows, n_trials=20, buffer_tiles=1):
    """Randomized search for tree ensembles."""
    print(f"  * Tuning {model_name} ({n_trials} random combos)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500, 800]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 3, 5, 10]),
        }
        return _inner_cv_score(
            model_name, params, X, z, y,
            fold_assignments, tile_groups, n_tile_cols, n_tile_rows,
            buffer_tiles=buffer_tiles,
        )

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     Best: {study.best_params}, inner R*={study.best_value:.4f}")
    return study.best_params


def tune_catboost(X, z, y, fold_assignments, tile_groups,
                  n_tile_cols, n_tile_rows, n_trials=50, buffer_tiles=1):
    """Optuna HPO for CatBoost."""
    print(f"  * Tuning CatBoost ({n_trials} Optuna trials)...")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 2000, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        }
        return _inner_cv_score(
            "catboost", params, X, z, y,
            fold_assignments, tile_groups, n_tile_cols, n_tile_rows,
            buffer_tiles=buffer_tiles,
        )

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     Best: {study.best_params}, inner R*={study.best_value:.4f}")
    return study.best_params


def tune_model(model_name, X, z, y, fold_assignments, tile_groups,
               n_tile_cols, n_tile_rows, n_optuna_trials=50, buffer_tiles=1):
    """Dispatch HPO by model name. Returns best hparams dict."""
    if model_name == "ridge":
        return tune_ridge(X, z, y, fold_assignments, tile_groups,
                          n_tile_cols, n_tile_rows, buffer_tiles)
    elif model_name == "elasticnet":
        return tune_elasticnet(X, z, y, fold_assignments, tile_groups,
                               n_tile_cols, n_tile_rows, n_optuna_trials, buffer_tiles)
    elif model_name in ("extratrees", "rf"):
        return tune_forest(model_name, X, z, y, fold_assignments, tile_groups,
                           n_tile_cols, n_tile_rows, n_trials=20, buffer_tiles=buffer_tiles)
    elif model_name == "catboost":
        return tune_catboost(X, z, y, fold_assignments, tile_groups,
                             n_tile_cols, n_tile_rows, n_optuna_trials, buffer_tiles)
    elif model_name in MLP_MODELS:
        # MLP uses manual defaults + early stopping; no Optuna for now
        return {}
    else:
        return {}


# =====================================================================
# Full 5-fold CV evaluation
# =====================================================================

def run_full_cv(model_name, hparams, X, z, y, fold_assignments,
                tile_groups, n_tile_cols, n_tile_rows, buffer_tiles=1):
    """
    Run full 5-fold CV and collect per-fold metrics.

    Returns
    -------
    fold_results : list of dict (per-fold summary metrics)
    fold_predictions : dict, fold_idx * y_pred
    fold_details : list of pd.DataFrame (per-class per-fold)
    """
    is_mlp_model = model_name in MLP_MODELS
    fold_results = []
    fold_predictions = {}
    fold_details = []

    for fold_idx in range(N_FOLDS):
        train_idx, test_idx = get_fold_indices(
            tile_groups, fold_assignments, fold_idx,
            n_tile_cols, n_tile_rows, buffer_tiles=buffer_tiles,
        )

        model = build_model(model_name, **hparams)

        if is_mlp_model:
            # Use ~20% of train as validation for early stopping
            rng = np.random.RandomState(SEED + fold_idx)
            n_train = len(train_idx)
            perm = rng.permutation(n_train)
            n_val = max(int(n_train * 0.15), 100)
            val_sub = train_idx[perm[:n_val]]
            train_sub = train_idx[perm[n_val:]]
            model.fit(X[train_sub], y[train_sub],
                      X_val=X[val_sub], z_val_or_y=y[val_sub])
            y_pred = model.predict_proportions(X[test_idx])
        elif model_name == "catboost":
            rng = np.random.RandomState(SEED + fold_idx)
            n_train = len(train_idx)
            perm = rng.permutation(n_train)
            n_val = max(int(n_train * 0.15), 100)
            val_sub = train_idx[perm[:n_val]]
            train_sub = train_idx[perm[n_val:]]
            model.fit(X[train_sub], z[train_sub],
                      X_val=X[val_sub], z_val=z[val_sub])
            y_pred = model.predict_proportions(X[test_idx])
        else:
            model.fit(X[train_idx], z[train_idx])
            y_pred = model.predict_proportions(X[test_idx])

        summary, detail = evaluate_model(
            y[test_idx], y_pred, CLASS_NAMES, model_name=model_name,
        )
        summary["fold"] = fold_idx
        summary["train_size"] = len(train_idx)
        summary["test_size"] = len(test_idx)

        # Moran's I
        mi_result, mi_mean = compute_residual_morans_i(
            y[test_idx], y_pred, tile_groups[test_idx],
            n_tile_cols, n_tile_rows, CLASS_NAMES,
        )
        summary["morans_i_mean"] = mi_mean
        for cls, val in mi_result.items():
            summary[f"morans_i_{cls}"] = val

        fold_results.append(summary)
        fold_predictions[fold_idx] = y_pred
        detail["fold"] = fold_idx
        fold_details.append(detail)

    return fold_results, fold_predictions, fold_details


# =====================================================================
# Holdout evaluation (fold 0)
# =====================================================================

def run_holdout(model_name, hparams, X, z, y, cell_ids,
                fold_assignments, tile_groups, n_tile_cols, n_tile_rows,
                buffer_tiles=1, feature_set="core"):
    """
    Train on folds 1*4, test on fold 0 (untouched holdout).

    Returns summary, per-class detail, y_pred, conformal report.
    """
    is_mlp_model = model_name in MLP_MODELS

    # Train on folds 1*4
    train_mask = np.isin(fold_assignments, [1, 2, 3, 4])
    test_mask = fold_assignments == 0

    # Apply buffer against fold 0
    if buffer_tiles > 0:
        from src.splitting import _precompute_tile_neighbors
        test_tiles = set(np.unique(tile_groups[test_mask]))
        nbrs = _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles)
        buf_set = set()
        for tt in test_tiles:
            for nbr_tile in nbrs.get(tt, set()):
                if nbr_tile not in test_tiles:
                    buf_set.add(nbr_tile)
        buf_mask = np.isin(tile_groups, list(buf_set))
        train_mask = train_mask & (~buf_mask)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    # For early stopping: split a val portion from train
    model = build_model(model_name, **hparams)

    if is_mlp_model:
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_sub = train_idx[perm[:n_val]]
        train_sub = train_idx[perm[n_val:]]
        model.fit(X[train_sub], y[train_sub],
                  X_val=X[val_sub], z_val_or_y=y[val_sub])
    elif model_name == "catboost":
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_sub = train_idx[perm[:n_val]]
        train_sub = train_idx[perm[n_val:]]
        model.fit(X[train_sub], z[train_sub],
                  X_val=X[val_sub], z_val=z[val_sub])
    else:
        model.fit(X[train_idx], z[train_idx])

    y_pred = model.predict_proportions(X[test_idx])

    # Metrics
    summary, detail = evaluate_model(
        y[test_idx], y_pred, CLASS_NAMES, model_name=model_name,
    )
    summary["fold"] = "holdout"
    summary["train_size"] = len(train_idx)
    summary["test_size"] = len(test_idx)

    # Moran's I
    mi_result, mi_mean = compute_residual_morans_i(
        y[test_idx], y_pred, tile_groups[test_idx],
        n_tile_cols, n_tile_rows, CLASS_NAMES,
    )
    summary["morans_i_mean"] = mi_mean

    # Conformal: calibrate on inner residuals, test on fold 0
    # Use fold 4 as calibration set
    cal_mask = fold_assignments == 4
    cal_idx = np.where(cal_mask)[0]
    if is_mlp_model:
        y_cal_pred = model.predict_proportions(X[cal_idx])
    else:
        y_cal_pred = model.predict_proportions(X[cal_idx])
    quantiles = calibrate_conformal(y[cal_idx], y_cal_pred, alpha=0.1)
    lower, upper = predict_interval(y_pred, quantiles)
    conf_report = conformal_coverage_report(y[test_idx], lower, upper, CLASS_NAMES)

    # Save predictions
    pred_df = pd.DataFrame(y_pred, columns=CLASS_NAMES)
    pred_df["cell_id"] = cell_ids[test_idx]
    cols = ["cell_id"] + CLASS_NAMES
    pred_path = os.path.join(
        PROCESSED_V2_DIR, f"predictions_{model_name}_{feature_set}.parquet",
    )
    pred_df[cols].to_parquet(pred_path, index=False)
    print(f"  * Predictions: {pred_path}")

    return summary, detail, y_pred, conf_report, model


# =====================================================================
# Coefficient stability (linear models)
# =====================================================================

def compute_coefficient_stability(model_name, hparams, X, z, y,
                                  fold_assignments, tile_groups,
                                  n_tile_cols, n_tile_rows,
                                  feature_cols, buffer_tiles=1):
    """
    Fit model on each fold, collect coefficients, compute stability.

    Returns DataFrame with feature, mean_coef, std_coef, sign_consistency.
    """
    from src.models.linear import RidgeModel

    all_coefs = []

    for fold_idx in range(N_FOLDS):
        train_idx, _ = get_fold_indices(
            tile_groups, fold_assignments, fold_idx,
            n_tile_cols, n_tile_rows, buffer_tiles=buffer_tiles,
        )
        model = build_model(model_name, **hparams)
        model.fit(X[train_idx], z[train_idx])
        coefs = model.coef_  # (D-1, n_features) or similar
        # Average across ILR coordinates for interpretability
        coef_avg = coefs.mean(axis=0) if coefs.ndim == 2 else coefs
        all_coefs.append(coef_avg)

    coef_matrix = np.array(all_coefs)  # (n_folds, n_features)
    mean_coef = coef_matrix.mean(axis=0)
    std_coef = coef_matrix.std(axis=0)

    # Sign consistency: fraction of folds where sign matches majority
    signs = np.sign(coef_matrix)
    majority_sign = np.sign(signs.sum(axis=0))
    sign_match = (signs == majority_sign[None, :]).mean(axis=0)

    stability_df = pd.DataFrame({
        "feature": feature_cols[:len(mean_coef)],
        "mean_coef": mean_coef,
        "std_coef": std_coef,
        "sign_consistency": sign_match,
        "abs_mean_coef": np.abs(mean_coef),
    })
    stability_df = stability_df.sort_values("abs_mean_coef", ascending=False)

    return stability_df


# =====================================================================
# Plotting
# =====================================================================

def plot_model_comparison(benchmark_df):
    """Bar chart comparing models on key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 8 * Model Comparison (Holdout Fold 0)", fontsize=14, fontweight="bold")

    models = benchmark_df["model"].values
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for ax, metric, label in zip(
        axes,
        ["r2_uniform", "mae_mean_pp", "aitchison_mean"],
        ["R* (uniform avg)", "MAE (pp)", "Aitchison distance"],
    ):
        vals = benchmark_df[metric].values
        bars = ax.barh(models, vals, color=colors)
        ax.set_xlabel(label)
        ax.set_title(label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    save_fig(fig, "model_comparison")


def plot_cv_boxplot(cv_results_df):
    """Boxplot of 5-fold CV R* per model."""
    models = cv_results_df["model"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    data_per_model = []
    labels = []
    for m in models:
        vals = cv_results_df[cv_results_df["model"] == m]["r2_uniform"].dropna().values
        data_per_model.append(vals)
        labels.append(m)

    bp = ax.boxplot(data_per_model, labels=labels, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("R* (uniform avg)")
    ax.set_title("5-Fold Spatial CV * R* Distribution")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "cv_boxplot")


def plot_per_class_r2(benchmark_df):
    """Grouped bar chart of R* per class per model."""
    models = benchmark_df["model"].values
    r2_cols = [f"r2_{c}" for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        row = benchmark_df[benchmark_df["model"] == model].iloc[0]
        vals = [row.get(c, 0) for c in r2_cols]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.8)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel("R*")
    ax.set_title("Per-Class R* * Holdout Fold 0")
    ax.legend(loc="lower right")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "per_class_r2")


def plot_conformal_summary(conformal_reports):
    """Coverage and width summary per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(conformal_reports.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    coverages = [conformal_reports[m]["aggregate"]["marginal_coverage_pct"] for m in models]
    widths = [conformal_reports[m]["aggregate"]["mean_width_pp"] for m in models]

    axes[0].barh(models, coverages, color=colors)
    axes[0].axvline(90, color="red", linestyle="--", label="Target 90%")
    axes[0].set_xlabel("Marginal Coverage (%)")
    axes[0].set_title("Conformal Coverage")
    axes[0].legend()

    axes[1].barh(models, widths, color=colors)
    axes[1].set_xlabel("Mean Interval Width (pp)")
    axes[1].set_title("Conformal Interval Width")

    plt.tight_layout()
    save_fig(fig, "conformal_summary")


# =====================================================================
# Main
# =====================================================================

def main(feature_set="core", model_names=None, skip_hpo=False,
         n_optuna_trials=50, buffer_tiles=1):
    """Main Phase 8 pipeline."""
    t0 = time.time()

    if model_names is None:
        model_names = ALL_MODELS

    # Load data
    (X, y, cell_ids, fold_assignments, tile_groups,
     feature_cols, n_tile_cols, n_tile_rows) = load_data(feature_set)

    # ILR forward transform
    basis = helmert_basis(N_CLASSES)
    z = ilr_forward(y, basis=basis)
    print(f"  ILR shape: {z.shape} (Helmert basis)")

    # Results collectors
    all_holdout_summaries = []
    all_cv_results = []
    all_conformal = {}
    all_details = []

    for model_name in model_names:
        print()
        print("=" * 70)
        print(f"Model: {model_name.upper()}")
        print("=" * 70)
        t_model = time.time()

        # -- HPO --
        if skip_hpo:
            best_hparams = {}
            print("  >> Skipping HPO (using defaults)")
        else:
            best_hparams = tune_model(
                model_name, X, z, y, fold_assignments, tile_groups,
                n_tile_cols, n_tile_rows, n_optuna_trials, buffer_tiles,
            )

        # -- Full 5-fold CV --
        print(f"  * Running 5-fold spatial CV...")
        fold_results, _, fold_detail_list = run_full_cv(
            model_name, best_hparams, X, z, y, fold_assignments,
            tile_groups, n_tile_cols, n_tile_rows, buffer_tiles,
        )
        cv_df = pd.DataFrame(fold_results)
        mean_r2 = cv_df["r2_uniform"].mean()
        std_r2 = cv_df["r2_uniform"].std()
        print(f"     CV R* = {mean_r2:.4f} * {std_r2:.4f}")
        cv_df["model"] = model_name
        all_cv_results.append(cv_df)

        # -- Holdout evaluation --
        print(f"  * Holdout evaluation (fold 0)...")
        (holdout_summary, holdout_detail, y_pred_holdout,
         conf_report, trained_model) = run_holdout(
            model_name, best_hparams, X, z, y, cell_ids,
            fold_assignments, tile_groups, n_tile_cols, n_tile_rows,
            buffer_tiles, feature_set,
        )
        holdout_summary["hparams"] = json.dumps(best_hparams)
        holdout_summary["cv_r2_mean"] = mean_r2
        holdout_summary["cv_r2_std"] = std_r2
        all_holdout_summaries.append(holdout_summary)
        all_conformal[model_name] = conf_report
        holdout_detail["model"] = model_name
        all_details.append(holdout_detail)

        sv = simplex_validity(y_pred_holdout)
        print(f"     R*  = {holdout_summary['r2_uniform']:.4f}")
        print(f"     MAE = {holdout_summary['mae_mean_pp']:.2f} pp")
        print(f"     Aitchison = {holdout_summary['aitchison_mean']:.4f}")
        print(f"     Simplex valid = {sv['pct_fully_valid']:.1f}%")
        print(f"     Moran's I (mean) = {holdout_summary['morans_i_mean']:.4f}")
        print(f"     Conformal coverage = "
              f"{conf_report['aggregate']['marginal_coverage_pct']:.1f}%")

        # -- Coefficient stability (linear models only) --
        if model_name in ("ridge", "elasticnet") and not skip_hpo:
            print(f"  * Computing coefficient stability...")
            stability_df = compute_coefficient_stability(
                model_name, best_hparams, X, z, y,
                fold_assignments, tile_groups, n_tile_cols, n_tile_rows,
                feature_cols, buffer_tiles,
            )
            save_table(stability_df.head(50), f"coefficient_stability_{model_name}")

        elapsed = time.time() - t_model
        print(f"  T  {model_name}: {elapsed:.1f}s")

    # -- Save aggregate tables --
    print()
    print("=" * 70)
    print("Saving results")
    print("=" * 70)

    benchmark_df = pd.DataFrame(all_holdout_summaries)
    save_table(benchmark_df, "model_benchmark")

    cv_all = pd.concat(all_cv_results, ignore_index=True)
    save_table(cv_all, "cv_results")

    detail_all = pd.concat(all_details, ignore_index=True)
    save_table(detail_all, "model_benchmark_per_class")

    # Conformal
    conf_rows = []
    for m, report in all_conformal.items():
        for cls, vals in report["per_class"].items():
            conf_rows.append({"model": m, "class": cls, **vals})
    if conf_rows:
        save_table(pd.DataFrame(conf_rows), "conformal_coverage")

    # -- Plots --
    if len(all_holdout_summaries) >= 2:
        plot_model_comparison(benchmark_df)
        plot_cv_boxplot(cv_all)
        plot_per_class_r2(benchmark_df)
        if all_conformal:
            plot_conformal_summary(all_conformal)

    # -- Metadata --
    metadata = {
        "feature_set": feature_set,
        "n_features": X.shape[1],
        "n_cells": X.shape[0],
        "n_folds": N_FOLDS,
        "buffer_tiles": buffer_tiles,
        "ilr_basis": "helmert",
        "models": model_names,
        "skip_hpo": skip_hpo,
        "n_optuna_trials": n_optuna_trials,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    meta_path = os.path.join(REPORTS_DIR, "metadata.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  * Metadata: {meta_path}")

    total = time.time() - t0
    print()
    print(f"OK Phase 8 complete in {total:.1f}s")
    print(f"   Models: {', '.join(model_names)}")
    print(f"   Best holdout R*: {benchmark_df['r2_uniform'].max():.4f} "
          f"({benchmark_df.loc[benchmark_df['r2_uniform'].idxmax(), 'model']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8: Model Training")
    parser.add_argument("--feature-set", default="core",
                        choices=["core", "full"],
                        help="Which merged feature set to use (default: core)")
    parser.add_argument("--models", default="all",
                        help="Comma-separated model names, or 'all' (default: all)")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="Skip hyperparameter optimization (use defaults)")
    parser.add_argument("--n-optuna-trials", type=int, default=50,
                        help="Number of Optuna trials for CatBoost/ElasticNet")
    parser.add_argument("--buffer-tiles", type=int, default=1,
                        help="Tile-level Chebyshev buffer (default: 1)")
    args = parser.parse_args()

    models = ALL_MODELS if args.models == "all" else args.models.split(",")
    main(
        feature_set=args.feature_set,
        model_names=models,
        skip_hpo=args.skip_hpo,
        n_optuna_trials=args.n_optuna_trials,
        buffer_tiles=args.buffer_tiles,
    )
