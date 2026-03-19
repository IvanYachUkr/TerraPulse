#!/usr/bin/env python3
"""
Fused MLP Evaluation -- 6 Held-Out Test Cities

Compares the fused MLP against the deployed V10 model (#7, trial 77)
on the same 6 cities used in the original evaluation.

Usage:
    .venv\\Scripts\\python -u mlp/eval_test.py
    .venv\\Scripts\\python -u mlp/eval_test.py --fused-dir data/cities/models_fused_mlp_v3
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mlp.config import CITIES_DIR, N_CLASSES, CLASS_NAMES
from mlp.model import TaperedMLP, PlainBlock
from mlp.fused_model import HybridMaskedFusionMLP, make_fused_model, split_feature_indices

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# TEST CITIES (same 6 as V10 eval)
# =====================================================================

TEST_CITIES = [
    "nuremberg", "ankara_test", "sofia_test",
    "riga_test", "edinburgh_test", "palermo_test",
]

FIXED_THRESHOLDS = [0.0, 0.05, 0.10]

# V10 deployed model
V10_DIR = os.path.join(CITIES_DIR, "models_v10_bohb")
DEPLOYED_TRIAL = 77

# =====================================================================
# EVALUATION FUNCTIONS
# =====================================================================

def compute_r2(preds, y, threshold):
    """Mean per-class R2, evaluating only pixels where y >= threshold."""
    r2s = []
    for ci in range(N_CLASSES):
        mk = y[:, ci] >= threshold if threshold > 0 else np.ones(len(y), dtype=bool)
        if mk.sum() < 50:
            continue
        yt = y[mk, ci]
        if np.var(yt) < 1e-8:
            continue
        r2s.append(r2_score(yt, preds[mk, ci]))
    return np.mean(r2s) if r2s else 0


def compute_topk_accuracy(preds, y, k, threshold=0.01):
    """Top-K set-match accuracy."""
    n_above = (y >= threshold).sum(axis=1)
    mask = n_above >= k
    if mask.sum() < 50:
        return float("nan")
    true_topk = np.argsort(-y[mask], axis=1)[:, :k]
    pred_topk = np.argsort(-preds[mask], axis=1)[:, :k]
    return np.mean([set(t) == set(p) for t, p in zip(true_topk, pred_topk)])


# =====================================================================
# MODEL LOADERS
# =====================================================================

def load_v10_deployed():
    """Load the deployed V10 model (#7, trial 77)."""
    trials = [json.loads(l) for l in open(os.path.join(V10_DIR, "trial_log.jsonl")) if l.strip()]
    # Find trial 77
    trial = None
    for t in trials:
        if t["trial"] == DEPLOYED_TRIAL:
            trial = t
            break
    if trial is None:
        raise ValueError(f"Trial {DEPLOYED_TRIAL} not found in trial_log.jsonl")

    model_path = os.path.join(V10_DIR, f"trial_{trial['trial']}_{trial['arch']}.pt")
    cols_path = os.path.join(V10_DIR, "mlp_cols.json")
    scaler_path = os.path.join(V10_DIR, "scaler.pkl")

    with open(cols_path) as f:
        mlp_cols = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    net = TaperedMLP(
        len(mlp_cols), N_CLASSES, trial["widths"],
        dropout=trial["config"]["dropout"],
        activation=trial["config"]["activation"],
        input_dropout=trial["config"]["input_dropout"],
    ).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()

    return {
        "name": f"V10-deployed #{DEPLOYED_TRIAL}",
        "net": net,
        "cols": mlp_cols,
        "scaler": scaler,
        "thresh": max(trial["config"]["label_threshold"], 0.01),
        "predict_fn": lambda net, x: net.forward(x).exp(),  # log_softmax -> probs
    }


def load_fused_model(fused_dir):
    """Load the fused/hybrid MLP from a training output directory."""
    cols_path = os.path.join(fused_dir, "mlp_cols.json")
    scaler_path = os.path.join(fused_dir, "scaler.pkl")
    model_path = os.path.join(fused_dir, "best_model.pt")
    results_path = os.path.join(fused_dir, "results.json")

    with open(cols_path) as f:
        mlp_cols = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(results_path) as f:
        results = json.load(f)

    # Support both old (combined_widths) and new (joint_widths) formats
    joint_widths = results.get("joint_widths", results.get("combined_widths", [1024, 512, 256]))
    refiner_widths = results.get("refiner_widths", [256, 128])
    cfg = results.get("config", {})

    net = make_fused_model(
        len(mlp_cols), N_CLASSES, mlp_cols,
        optical_widths=results["optical_widths"],
        sar_widths=results["sar_widths"],
        joint_widths=joint_widths,
        fusion_widths=results["fusion_widths"],
        refiner_widths=refiner_widths,
        dropout=cfg.get("dropout", 0.072),
        activation=cfg.get("activation", "gelu"),
        input_dropout=cfg.get("input_dropout", 0.06),
        rare_class_scale=results.get("rare_class_weights", None),
        gate_strength=cfg.get("gate_strength", 0.80),
        gate_floor=cfg.get("gate_floor", 0.10),
        refine_strength=cfg.get("refine_strength", 0.60),
        optical_branch_drop=cfg.get("optical_branch_drop", 0.04),
        sar_branch_drop=cfg.get("sar_branch_drop", 0.04),
        joint_branch_drop=cfg.get("joint_branch_drop", 0.10),
    ).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()

    arch = (f"opt={'x'.join(map(str, results['optical_widths']))} "
            f"sar={'x'.join(map(str, results['sar_widths']))} "
            f"joint={'x'.join(map(str, joint_widths))} "
            f"fus={'x'.join(map(str, results['fusion_widths']))} "
            f"ref={'x'.join(map(str, refiner_widths))}")

    return {
        "name": f"Hybrid MLP ({arch})",
        "net": net,
        "cols": mlp_cols,
        "scaler": scaler,
        "thresh": max(cfg.get("label_threshold", 0.015), 0.01),
        "predict_fn": lambda net, x: net.predict(x),
        "val_results": results,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-dir", type=str,
                        default=os.path.join(CITIES_DIR, "models_fused_mlp_v3"))
    args = parser.parse_args()

    t0 = time.time()
    sep = "=" * 90

    print(f"\n{sep}")
    print("  Fused MLP vs V10 Deployed -- 6 Held-Out Test Cities")
    print(f"{sep}")

    # Load models
    models = []
    try:
        models.append(load_fused_model(args.fused_dir))
        print(f"  Fused model loaded from: {args.fused_dir}")
        if "val_results" in models[-1]:
            vr = models[-1]["val_results"]
            print(f"    Val: combined={vr['combined']:.4f} top1={vr['top1_acc']:.4f} "
                  f"R2={vr['mean_r2']:.4f} | {vr['n_params']:,} params")
    except Exception as e:
        print(f"  WARNING: Could not load fused model: {e}")

    try:
        models.append(load_v10_deployed())
        print(f"  V10 deployed model loaded (trial {DEPLOYED_TRIAL})")
    except Exception as e:
        print(f"  WARNING: Could not load V10 model: {e}")

    if not models:
        print("  No models loaded, exiting.")
        return

    # Evaluate
    print(f"\n  Running inference on {len(TEST_CITIES)} test cities...", flush=True)
    all_preds = {}
    all_labels = {}
    all_top1 = {}

    for city in TEST_CITIES:
        feat_path = os.path.join(CITIES_DIR, city, "features_v7", "features_rust_2020_2021.parquet")
        label_path = os.path.join(CITIES_DIR, city, "labels_2021.parquet")
        if not os.path.exists(feat_path) or not os.path.exists(label_path):
            print(f"    {city}: SKIP (missing data)")
            continue

        df_lab = pd.read_parquet(label_path)
        y = df_lab[CLASS_NAMES].values.astype(np.float32)
        rs = y.sum(axis=1)
        valid = rs > 0
        y = y[valid]
        y = y / y.sum(axis=1, keepdims=True)
        all_labels[city] = y

        for m in models:
            df_feat = pd.read_parquet(feat_path)
            for col in m["cols"]:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            X = m["scaler"].transform(df_feat[m["cols"]].values.astype(np.float32)[valid])
            del df_feat

            net = m["net"]
            net.eval()
            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            with torch.no_grad():
                preds = m["predict_fn"](net, X_t).cpu().numpy()
            del X_t, X
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_preds[(m["name"], city)] = preds
            all_top1[(m["name"], city)] = float((y.argmax(1) == preds.argmax(1)).mean())

        print(f"    {city} ({valid.sum():,} px) -- done", flush=True)

    model_names = [m["name"] for m in models]

    # --- Per-city Top-1 ---
    W = 14
    print(f"\n{sep}")
    print(f"  TOP-1 ACCURACY PER CITY")
    print(f"{sep}")
    header = f"{'Model':40s}" + "".join(f"{c:>{W}s}" for c in TEST_CITIES) + f"{'MEAN':>{W}s}"
    print(header)
    print("-" * len(header))
    for mn in model_names:
        row = f"{mn:40s}"
        vals = []
        for city in TEST_CITIES:
            if city not in all_labels:
                row += f"{'--':>{W}s}"
                continue
            v = all_top1[(mn, city)]
            row += f"{v:>{W}.4f}"
            vals.append(v)
        row += f"{np.mean(vals):>{W}.4f}"
        print(row)

    # --- Rankings at each threshold ---
    for thresh in FIXED_THRESHOLDS:
        tl = f"{int(thresh*100)}%" if thresh > 0 else "0% (no filter)"
        print(f"\n{sep}")
        print(f"  RANKING @ THRESHOLD = {tl}")
        print(f"{sep}")
        print(f"{'Rank':<5s} {'Model':40s} {'Combined':>10s} {'Top-1':>10s} {'R2':>10s} "
              f"{'Top-2':>10s} {'Top-3':>10s}")
        print("-" * 95)

        ranks = []
        for mn in model_names:
            t1_vals, r2_vals, t2_vals, t3_vals = [], [], [], []
            for city in TEST_CITIES:
                if city not in all_labels:
                    continue
                y = all_labels[city]
                p = all_preds[(mn, city)]
                t1_vals.append(all_top1[(mn, city)])
                r2_vals.append(compute_r2(p, y, thresh))
                t2_vals.append(compute_topk_accuracy(p, y, 2, max(thresh, 0.01)))
                t3_vals.append(compute_topk_accuracy(p, y, 3, max(thresh, 0.01)))

            mean_t1 = np.mean(t1_vals)
            mean_r2 = np.mean(r2_vals)
            mean_t2 = np.nanmean(t2_vals)
            mean_t3 = np.nanmean(t3_vals)
            comb = 0.5 * mean_t1 + 0.5 * max(0, mean_r2)
            ranks.append((mn, comb, mean_t1, mean_r2, mean_t2, mean_t3))

        ranks.sort(key=lambda x: x[1], reverse=True)
        for i, (mn, c, t1, r2, t2, t3) in enumerate(ranks, 1):
            print(f"{i:<5d} {mn:40s} {c:>10.4f} {t1:>10.4f} {r2:>10.4f} "
                  f"{t2:>10.4f} {t3:>10.4f}")

    # --- Per-class R2 detail ---
    for thresh in [0.05]:
        tl = f"{int(thresh*100)}%"
        print(f"\n{sep}")
        print(f"  PER-CLASS R2 (threshold={tl})")
        print(f"{sep}")
        short = [c[:10] for c in CLASS_NAMES]
        print(f"{'Model':40s}" + "".join(f"{c:>12s}" for c in short) + f"{'MEAN':>10s}")
        print("-" * 130)
        for m in models:
            mn = m["name"]
            row = f"{mn:40s}"
            all_r2s = []
            for ci in range(N_CLASSES):
                ci_r2s = []
                for city in TEST_CITIES:
                    if city not in all_labels:
                        continue
                    y = all_labels[city]
                    p = all_preds[(mn, city)]
                    mk = y[:, ci] >= thresh
                    if mk.sum() < 50 or np.var(y[mk, ci]) < 1e-8:
                        continue
                    ci_r2s.append(r2_score(y[mk, ci], p[mk, ci]))
                if ci_r2s:
                    r2 = np.mean(ci_r2s)
                    all_r2s.append(r2)
                    row += f"{r2:>12.4f}"
                else:
                    row += f"{'N/A':>12s}"
            row += f"{np.mean(all_r2s):>10.4f}" if all_r2s else f"{'--':>10s}"
            print(row)

    elapsed = time.time() - t0
    print(f"\n{sep}")
    print(f"  Evaluation complete in {elapsed:.0f}s")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
