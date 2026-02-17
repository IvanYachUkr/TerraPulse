#!/usr/bin/env python3
"""
Cross-City Evaluation: Compare multi-city vs single-city models on Nuremberg.

Runs automatically after the multi-city pipeline finishes.
Loads Nuremberg predictions from both pipelines, compares R2/MAE per class
against 2021 ground-truth labels.

Output: data/cities/cross_city_evaluation.json + console summary.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.evaluation import evaluate_model

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]

# Paths
SINGLE_CITY_DIR = os.path.join(
    PROJECT_ROOT, "data", "pipeline_output", "predictions")
MULTI_CITY_DIR = os.path.join(
    PROJECT_ROOT, "data", "cities", "nuremberg", "predictions")
LABELS_SINGLE = os.path.join(
    PROJECT_ROOT, "data", "pipeline_output", "labels_2021.parquet")
LABELS_MULTI = os.path.join(
    PROJECT_ROOT, "data", "cities", "nuremberg", "labels_2021.parquet")

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "cities",
                           "cross_city_evaluation.json")


def load_labels():
    """Load Nuremberg 2021 labels from whichever path exists."""
    for p in [LABELS_MULTI, LABELS_SINGLE,
              os.path.join(PROJECT_ROOT, "data", "processed", "v2",
                           "labels_2021.parquet")]:
        if os.path.exists(p):
            return pd.read_parquet(p)
    raise FileNotFoundError("No labels file found for Nuremberg 2021")


def evaluate_predictions(pred_path, labels_df):
    """Evaluate predictions against labels, return metrics dict."""
    if not os.path.exists(pred_path):
        return None

    pred_df = pd.read_parquet(pred_path)
    pred_cols = [f"{cn}_pred" for cn in CLASS_NAMES]

    # Align by cell_id if present
    if "cell_id" in pred_df.columns and "cell_id" in labels_df.columns:
        merged = pd.merge(pred_df, labels_df[["cell_id"] + CLASS_NAMES],
                          on="cell_id", how="inner")
        y_true = merged[CLASS_NAMES].values
        y_pred = merged[pred_cols].values
    else:
        n = min(len(pred_df), len(labels_df))
        y_true = labels_df[CLASS_NAMES].values[:n]
        y_pred = pred_df[pred_cols].values[:n]

    summary, detail = evaluate_model(y_true, y_pred, CLASS_NAMES)
    return {
        "r2": summary["r2_uniform"],
        "mae_pp": summary["mae_mean_pp"],
        "per_class_r2": {cn: float(summary[f"r2_{cn}"])
                         for cn in CLASS_NAMES},
        "per_class_mae": {cn: float(summary[f"mae_{cn}_pp"])
                          for cn in CLASS_NAMES},
        "n_cells": len(y_true),
    }


def main():
    print("=" * 70)
    print("CROSS-CITY EVALUATION: Multi-City vs Single-City on Nuremberg")
    print("=" * 70)

    labels = load_labels()
    print(f"Labels: {len(labels)} cells")

    results = {}
    models = ["tree", "mlp"]
    tag = "2020_2021"  # Labeled year-pair

    for model in models:
        print(f"\n--- {model.upper()} ({tag}) ---")

        # Single-city model
        single_path = os.path.join(
            SINGLE_CITY_DIR, f"predictions_{model}_{tag}.parquet")
        single = evaluate_predictions(single_path, labels)
        if single:
            results[f"{model}_single"] = single
            print(f"  Single-city: R2={single['r2']:.4f}  "
                  f"MAE={single['mae_pp']:.2f}pp  ({single['n_cells']} cells)")
        else:
            print(f"  Single-city: NOT FOUND ({single_path})")

        # Multi-city model
        multi_path = os.path.join(
            MULTI_CITY_DIR, f"predictions_{model}_{tag}.parquet")
        multi = evaluate_predictions(multi_path, labels)
        if multi:
            results[f"{model}_multi"] = multi
            print(f"  Multi-city:  R2={multi['r2']:.4f}  "
                  f"MAE={multi['mae_pp']:.2f}pp  ({multi['n_cells']} cells)")
        else:
            print(f"  Multi-city:  NOT FOUND ({multi_path})")

        # Delta
        if single and multi:
            dr2 = multi["r2"] - single["r2"]
            dmae = multi["mae_pp"] - single["mae_pp"]
            arrow = "+" if dr2 > 0 else ""
            print(f"  Delta:       R2 {arrow}{dr2:.4f}  "
                  f"MAE {'+' if dmae > 0 else ''}{dmae:.2f}pp")

            print(f"\n  Per-class R2:")
            print(f"  {'Class':<15} {'Single':>8} {'Multi':>8} {'Delta':>8}")
            print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
            for cn in CLASS_NAMES:
                s = single["per_class_r2"][cn]
                m = multi["per_class_r2"][cn]
                d = m - s
                marker = ">>>" if d > 0.02 else ("<<<" if d < -0.02 else "")
                print(f"  {cn:<15} {s:>8.4f} {m:>8.4f} {d:>+8.4f} {marker}")

    # Also evaluate on unlabeled year-pairs (just check prediction quality)
    print(f"\n{'='*70}")
    print("PREDICTION AVAILABILITY (all year-pairs)")
    print(f"{'='*70}")
    for model in models:
        for y in range(2020, 2025):
            tag = f"{y}_{y+1}"
            s_exists = os.path.exists(
                os.path.join(SINGLE_CITY_DIR,
                             f"predictions_{model}_{tag}.parquet"))
            m_exists = os.path.exists(
                os.path.join(MULTI_CITY_DIR,
                             f"predictions_{model}_{tag}.parquet"))
            print(f"  {model:>5} {tag}: "
                  f"single={'OK' if s_exists else 'MISSING':>7}  "
                  f"multi={'OK' if m_exists else 'MISSING':>7}")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT_PATH}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
