"""
Phase 8: Split conformal prediction intervals.

Provides distribution-free prediction intervals for each land-cover
class proportion at a specified coverage level.
"""

import numpy as np


def calibrate_conformal(y_cal_true, y_cal_pred, alpha=0.1):
    """
    Calibrate conformal prediction intervals from calibration residuals.

    Uses absolute residual as nonconformity score.

    Parameters
    ----------
    y_cal_true : ndarray (N_cal, C)
        True proportions on calibration set.
    y_cal_pred : ndarray (N_cal, C)
        Predicted proportions on calibration set.
    alpha : float
        Miscoverage level (0.1 = 90% coverage target).

    Returns
    -------
    quantiles : ndarray (C,)
        Per-class quantile thresholds for interval construction.
    """
    scores = np.abs(y_cal_true - y_cal_pred)  # (N_cal, C)
    N_cal = scores.shape[0]

    # Finite-sample correction: ceil((N+1)(1-α)) / N
    q_level = min(np.ceil((N_cal + 1) * (1 - alpha)) / N_cal, 1.0)

    quantiles = np.quantile(scores, q_level, axis=0)  # (C,)
    return quantiles


def predict_interval(y_pred, quantiles):
    """
    Construct prediction intervals from predictions and calibration quantiles.

    Parameters
    ----------
    y_pred : ndarray (N, C)
    quantiles : ndarray (C,)

    Returns
    -------
    lower : ndarray (N, C)
    upper : ndarray (N, C)
    """
    lower = np.clip(y_pred - quantiles, 0.0, 1.0)
    upper = np.clip(y_pred + quantiles, 0.0, 1.0)
    return lower, upper


def conformal_coverage_report(y_test, lower, upper, class_names=None):
    """
    Evaluate conformal prediction intervals.

    Parameters
    ----------
    y_test : ndarray (N, C)
    lower, upper : ndarray (N, C)
    class_names : list of str, optional

    Returns
    -------
    report : dict
        Per-class and aggregate coverage and width metrics.
    """
    C = y_test.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    covered = (y_test >= lower) & (y_test <= upper)  # (N, C)
    widths = upper - lower  # (N, C)

    report = {"per_class": {}}
    for c in range(C):
        report["per_class"][class_names[c]] = {
            "coverage_pct": float(covered[:, c].mean()) * 100,
            "mean_width_pp": float(widths[:, c].mean()) * 100,
            "median_width_pp": float(np.median(widths[:, c])) * 100,
        }

    report["aggregate"] = {
        "coverage_pct": float(covered.all(axis=1).mean()) * 100,  # joint
        "marginal_coverage_pct": float(covered.mean()) * 100,
        "mean_width_pp": float(widths.mean()) * 100,
    }

    return report
