"""
Phase 8: Model evaluation metrics for compositional regression.

All metrics operate on proportion space (after ILR inverse), making
them comparable across models regardless of training space.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.transforms import aitchison_distance


# =====================================================================
# Per-class metrics
# =====================================================================

def r2_per_class(y_true, y_pred, class_names=None):
    """
    R² per land-cover class.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (N, C)
    class_names : list of str, optional

    Returns
    -------
    dict : class_name → R² value
    """
    C = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]
    return {
        name: float(r2_score(y_true[:, i], y_pred[:, i]))
        for i, name in enumerate(class_names)
    }


def mae_per_class(y_true, y_pred, class_names=None):
    """
    MAE per class in percentage points (multiply by 100).

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (N, C)
    class_names : list of str, optional

    Returns
    -------
    dict : class_name → MAE in pp
    """
    C = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]
    return {
        name: float(mean_absolute_error(y_true[:, i], y_pred[:, i])) * 100
        for i, name in enumerate(class_names)
    }


def rmse_per_class(y_true, y_pred, class_names=None):
    """
    RMSE per class in percentage points.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (N, C)
    class_names : list of str, optional

    Returns
    -------
    dict : class_name → RMSE in pp
    """
    C = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]
    return {
        name: float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))) * 100
        for i, name in enumerate(class_names)
    }


# =====================================================================
# Aggregate metrics
# =====================================================================

def r2_uniform(y_true, y_pred):
    """R² averaged uniformly across all output classes."""
    return float(r2_score(y_true, y_pred, multioutput="uniform_average"))


def r2_weighted(y_true, y_pred):
    """R² weighted by per-class variance."""
    return float(r2_score(y_true, y_pred, multioutput="variance_weighted"))


def mae_mean(y_true, y_pred):
    """Mean MAE across classes, in percentage points."""
    return float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")) * 100


def rmse_mean(y_true, y_pred):
    """Mean RMSE across classes, in percentage points."""
    per_class = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return float(per_class.mean()) * 100


def aitchison_mean(y_true, y_pred, eps=1e-6):
    """Mean Aitchison distance across samples."""
    d = aitchison_distance(y_true, y_pred, eps=eps)
    return float(np.mean(d))


# =====================================================================
# Simplex validity
# =====================================================================

def simplex_validity(y_pred, tol=1e-4):
    """
    Check how many predictions are valid on the simplex.

    Parameters
    ----------
    y_pred : ndarray, shape (N, C)
    tol : float
        Tolerance for sum-to-one check.

    Returns
    -------
    dict with keys:
        pct_valid_sum : % of rows with |sum - 1| < tol
        pct_valid_range : % of values in [0, 1]
        pct_fully_valid : % of rows that pass both
        mean_sum : mean row sum
        std_sum : std of row sums
    """
    row_sums = y_pred.sum(axis=1)
    valid_sum = np.abs(row_sums - 1.0) < tol
    valid_range = (y_pred >= -tol).all(axis=1) & (y_pred <= 1.0 + tol).all(axis=1)

    return {
        "pct_valid_sum": float(valid_sum.mean()) * 100,
        "pct_valid_range": float(valid_range.mean()) * 100,
        "pct_fully_valid": float((valid_sum & valid_range).mean()) * 100,
        "mean_sum": float(row_sums.mean()),
        "std_sum": float(row_sums.std()),
    }


# =====================================================================
# Combined evaluation
# =====================================================================

def evaluate_model(y_true, y_pred, class_names=None, model_name="model"):
    """
    Compute all metrics for a single model's predictions.

    Parameters
    ----------
    y_true : ndarray, shape (N, C)
        True proportions.
    y_pred : ndarray, shape (N, C)
        Predicted proportions (on the simplex).
    class_names : list of str, optional
    model_name : str

    Returns
    -------
    summary : dict
        Flat dictionary of all metric values.
    detail : pd.DataFrame
        Per-class metrics table with columns:
        class, r2, mae_pp, rmse_pp
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(y_true.shape[1])]

    r2_cls = r2_per_class(y_true, y_pred, class_names)
    mae_cls = mae_per_class(y_true, y_pred, class_names)
    rmse_cls = rmse_per_class(y_true, y_pred, class_names)
    sv = simplex_validity(y_pred)

    summary = {
        "model": model_name,
        "r2_uniform": r2_uniform(y_true, y_pred),
        "r2_weighted": r2_weighted(y_true, y_pred),
        "mae_mean_pp": mae_mean(y_true, y_pred),
        "rmse_mean_pp": rmse_mean(y_true, y_pred),
        "aitchison_mean": aitchison_mean(y_true, y_pred),
        **{f"r2_{k}": v for k, v in r2_cls.items()},
        **{f"mae_{k}_pp": v for k, v in mae_cls.items()},
        **sv,
    }

    detail = pd.DataFrame({
        "class": class_names,
        "r2": [r2_cls[c] for c in class_names],
        "mae_pp": [mae_cls[c] for c in class_names],
        "rmse_pp": [rmse_cls[c] for c in class_names],
    })

    return summary, detail
