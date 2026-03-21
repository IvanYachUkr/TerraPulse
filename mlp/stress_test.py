#!/usr/bin/env python3
"""
Stress Testing the Deployed V10 MLP (Trial 77)  --  Report-grade version

Four experiments on 6 held-out test cities (221K+ cells):
  1. Gaussian noise injection  (sigma sweep, 10 seeds, with std/CI)
  2. Season dropout            (6 individual + 2 full-year + 3 cross-year)
  3. Feature-group ablation    (7 groups + 3 combos)
  4. Per-city breakdown        (every perturbation broken out by geography)

Methodology notes:
  - Perturbations are applied in z-score space (after StandardScaler).
    Zeroing a feature in z-space = setting it to its training-set mean.
  - Noise sigma is in standard-deviation units of the z-scored features.
  - R2 is computed per-class with a 5% threshold (y >= 0.05), then averaged.
  - For noise experiments, results are averaged over N_SEEDS draws and
    standard deviations + 95% CIs are reported.

Usage:
    .venv\\Scripts\\python -u mlp/stress_test.py
    .venv\\Scripts\\python -u mlp/stress_test.py --seeds 10
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mlp.config import CITIES_DIR, N_CLASSES, CLASS_NAMES
from mlp.model import TaperedMLP, PlainBlock  # noqa: F401 (needed for unpickling)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# Constants
# =====================================================================

TEST_CITIES = [
    "nuremberg", "ankara_test", "sofia_test",
    "riga_test", "edinburgh_test", "palermo_test",
]

V10_DIR = os.path.join(CITIES_DIR, "models_v10_bohb")
DEPLOYED_TRIAL = 77
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "stress_test_v10")

R2_THRESHOLD = 0.05  # minimum class fraction for R2 evaluation
LABEL_THRESHOLD = 0.021  # deployed model_config.json label_threshold (classes below this are zeroed)

# Feature-group definitions
INDICES = {"NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
           "NDRE1", "NDRE2", "EVI2", "MNDWI", "GNDVI", "NDTI", "IRECI", "CRI1"}
BANDS = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
SPATIAL_PREFIXES = ("edge_", "lap_", "morans_", "NDVI_range", "NDVI_iqr")
SEASONS = ["spring", "summer", "autumn"]
YEARS = [2020, 2021]


def classify_column(col: str) -> str:
    if col.startswith("SAR_"):
        return "SAR"
    if col.startswith("LBP_"):
        return "LBP"
    if col.startswith("TC_"):
        return "Tasseled_Cap"
    if any(col.startswith(p) for p in SPATIAL_PREFIXES):
        return "Spatial"
    if "_pheno_" in col and not col.startswith("SAR_"):
        return "Phenological"
    if any(col.startswith(idx + "_") for idx in INDICES):
        return "Indices"
    if any(col.startswith(b + "_") for b in BANDS):
        return "Bands"
    return "Other"


# =====================================================================
# Model & data loading
# =====================================================================

def load_model():
    trials = [json.loads(l) for l in open(os.path.join(V10_DIR, "trial_log.jsonl")) if l.strip()]
    trial = next(t for t in trials if t["trial"] == DEPLOYED_TRIAL)

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
    return net, mlp_cols, scaler, trial


def load_test_data(mlp_cols, scaler):
    data = {}
    for city in TEST_CITIES:
        feat_path = os.path.join(CITIES_DIR, city, "features_v7",
                                 "features_rust_2020_2021.parquet")
        label_path = os.path.join(CITIES_DIR, city, "labels_2021.parquet")
        if not os.path.exists(feat_path) or not os.path.exists(label_path):
            print(f"  SKIP {city} (missing data)")
            continue

        df_feat = pd.read_parquet(feat_path)
        for col in mlp_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        df_lab = pd.read_parquet(label_path)
        y = df_lab[CLASS_NAMES].values.astype(np.float32)
        rs = y.sum(axis=1)
        valid = rs > 0
        y = y[valid]
        y = y / y.sum(axis=1, keepdims=True)

        X_raw = df_feat[mlp_cols].values.astype(np.float32)[valid]
        X_scaled = scaler.transform(X_raw)
        data[city] = (X_scaled, y)
        del df_feat, df_lab, X_raw
        print(f"  {city}: {X_scaled.shape[0]:,} cells")
    return data


# =====================================================================
# Inference & metrics
# =====================================================================

@torch.no_grad()
def predict(net, X_np, batch_size=8192):
    net.eval()
    preds = []
    for i in range(0, len(X_np), batch_size):
        x = torch.from_numpy(X_np[i:i+batch_size].astype(np.float32)).to(device)
        preds.append(net(x).exp().cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_topk_setmatch(preds, y, k, threshold=LABEL_THRESHOLD):
    """Top-K set-match: fraction of cells where the top-k predicted classes
    exactly match the top-k true classes (by fraction). Only evaluates cells
    that have >= k classes above threshold."""
    n_above = (y >= threshold).sum(axis=1)
    mask = n_above >= k
    if mask.sum() < 50:
        return float("nan")
    true_topk = np.argsort(-y[mask], axis=1)[:, :k]
    pred_topk = np.argsort(-preds[mask], axis=1)[:, :k]
    return float(np.mean([set(t) == set(p) for t, p in zip(true_topk, pred_topk)]))


def compute_metrics(preds, y):
    """Compute top-1, top-2, top-3 accuracy, mean R2, MAE, and per-class R2."""
    top1 = float((y.argmax(1) == preds.argmax(1)).mean())
    top2 = compute_topk_setmatch(preds, y, 2)
    top3 = compute_topk_setmatch(preds, y, 3)
    mae = float(mean_absolute_error(y, preds))

    r2s = []
    per_class = {}
    for ci, cname in enumerate(CLASS_NAMES):
        mk = y[:, ci] >= R2_THRESHOLD
        if mk.sum() < 50 or np.var(y[mk, ci]) < 1e-8:
            per_class[cname] = float("nan")
            continue
        r2 = r2_score(y[mk, ci], preds[mk, ci])
        r2s.append(r2)
        per_class[cname] = r2

    return {
        "top1_acc": top1,
        "top2_acc": top2,
        "top3_acc": top3,
        "mean_r2": float(np.mean(r2s)) if r2s else 0.0,
        "mae": mae,
        "per_class_r2": per_class,
    }


def run_perturbed(net, data, perturb_fn):
    """Run inference with a perturbation function applied to X.
    Returns {city: metrics_dict}."""
    city_metrics = {}
    for city, (X, y) in data.items():
        X_mod = perturb_fn(X)
        preds = predict(net, X_mod)
        city_metrics[city] = compute_metrics(preds, y)
    return city_metrics


def agg_cities(city_metrics):
    """Aggregate metrics across cities (mean + per-city detail)."""
    keys = ["top1_acc", "top2_acc", "top3_acc", "mean_r2", "mae"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in city_metrics.values() if not np.isnan(m.get(k, float("nan")))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    per_class = {}
    for cname in CLASS_NAMES:
        vals = [m["per_class_r2"].get(cname, float("nan"))
                for m in city_metrics.values()]
        finite = [v for v in vals if np.isfinite(v)]
        per_class[cname] = float(np.mean(finite)) if finite else float("nan")
    agg["per_class_r2"] = per_class
    agg["per_city"] = {c: m for c, m in city_metrics.items()}
    return agg


def paired_ttest(baseline_cities, perturbed_cities, metric="mean_r2"):
    """Paired t-test across cities: H0 = no difference from baseline."""
    cities = sorted(set(baseline_cities) & set(perturbed_cities))
    if len(cities) < 3:
        return float("nan"), float("nan")
    base = [baseline_cities[c][metric] for c in cities]
    pert = [perturbed_cities[c][metric] for c in cities]
    t_stat, p_val = scipy_stats.ttest_rel(base, pert)
    return float(t_stat), float(p_val)


# =====================================================================
# Experiment 1: Gaussian noise injection (with CI)
# =====================================================================

def exp_noise(net, data, n_seeds=10):
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    results = []

    for sigma in sigmas:
        if sigma == 0:
            cm = run_perturbed(net, data, lambda X: X)
            agg = agg_cities(cm)
            agg["sigma"] = 0.0
            agg["r2_ci95"] = 0.0
            agg["r2_seeds_std"] = 0.0
            results.append(agg)
            baseline_cm = cm
            print(f"  sigma=0.00 (baseline): R2={agg['mean_r2']:.4f}  "
                  f"MAE={agg['mae']:.4f}  Top1={agg['top1_acc']:.4f}  "
                  f"Top2={agg['top2_acc']:.4f}  Top3={agg['top3_acc']:.4f}")
        else:
            seed_r2s = []
            seed_maes = []
            seed_top1s = []
            seed_top2s = []
            seed_top3s = []
            all_seed_cm = []

            for seed in range(n_seeds):
                rng = np.random.RandomState(42 + seed)

                def perturb(X, _rng=rng, _s=sigma):
                    return X + _rng.normal(0, _s, X.shape).astype(np.float32)

                cm = run_perturbed(net, data, perturb)
                agg_s = agg_cities(cm)
                seed_r2s.append(agg_s["mean_r2"])
                seed_maes.append(agg_s["mae"])
                seed_top1s.append(agg_s["top1_acc"])
                seed_top2s.append(agg_s["top2_acc"])
                seed_top3s.append(agg_s["top3_acc"])
                all_seed_cm.append(cm)

            # Use the mean-seed result for the per-city/per-class details
            # (take the median seed for representative per-city breakdown)
            median_idx = int(np.argsort(seed_r2s)[len(seed_r2s) // 2])
            agg = agg_cities(all_seed_cm[median_idx])

            # Override aggregates with proper mean +/- CI across seeds
            agg["mean_r2"] = float(np.mean(seed_r2s))
            agg["mae"] = float(np.mean(seed_maes))
            agg["top1_acc"] = float(np.mean(seed_top1s))
            agg["top2_acc"] = float(np.mean(seed_top2s))
            agg["top3_acc"] = float(np.mean(seed_top3s))
            agg["r2_seeds_std"] = float(np.std(seed_r2s, ddof=1))
            agg["r2_ci95"] = float(1.96 * np.std(seed_r2s, ddof=1) / np.sqrt(n_seeds))
            agg["sigma"] = sigma

            # Paired t-test vs baseline (using median seed)
            t_stat, p_val = paired_ttest(baseline_cm, all_seed_cm[median_idx])
            agg["ttest_t"] = t_stat
            agg["ttest_p"] = p_val

            results.append(agg)
            print(f"  sigma={sigma:.2f}: R2={agg['mean_r2']:.4f} +/- {agg['r2_ci95']:.4f} (95%CI)  "
                  f"MAE={agg['mae']:.4f}  Top1={agg['top1_acc']:.4f}  "
                  f"Top2={agg['top2_acc']:.4f}  Top3={agg['top3_acc']:.4f}  p={p_val:.4f}")

    return results, baseline_cm


# =====================================================================
# Experiment 2: Season dropout (individual + full-year + cross-year)
# =====================================================================

def exp_season_dropout(net, data, mlp_cols, baseline_cm):
    results = []

    # Build column-to-season mapping
    col_season_year = {}
    for i, col in enumerate(mlp_cols):
        for s in SEASONS:
            for y in YEARS:
                if f"_{y}_{s}" in col:
                    col_season_year.setdefault((y, s), []).append(i)

    # Baseline
    agg_base = agg_cities(baseline_cm)
    agg_base["dropped"] = "none (baseline)"
    agg_base["n_zeroed"] = 0
    results.append(agg_base)
    print(f"  baseline: R2={agg_base['mean_r2']:.4f}  "
          f"MAE={agg_base['mae']:.4f}  Top1={agg_base['top1_acc']:.4f}  "
          f"Top2={agg_base['top2_acc']:.4f}  Top3={agg_base['top3_acc']:.4f}")

    # --- Individual year x season ---
    for yr in YEARS:
        for s in SEASONS:
            key = (yr, s)
            if key not in col_season_year:
                continue
            indices = col_season_year[key]
            label = f"{yr}_{s}"

            def perturb(X, _idx=indices):
                X2 = X.copy()
                X2[:, _idx] = 0.0
                return X2

            cm = run_perturbed(net, data, perturb)
            agg = agg_cities(cm)
            agg["dropped"] = label
            agg["n_zeroed"] = len(indices)

            t_stat, p_val = paired_ttest(baseline_cm, cm)
            agg["ttest_t"] = t_stat
            agg["ttest_p"] = p_val
            results.append(agg)
            print(f"  drop {label:16s} ({len(indices):4d} cols): "
                  f"R2={agg['mean_r2']:.4f}  MAE={agg['mae']:.4f}  "
                  f"Top1={agg['top1_acc']:.4f}  Top2={agg['top2_acc']:.4f}  "
                  f"Top3={agg['top3_acc']:.4f}  p={p_val:.4f}")

    # --- Full year (all 3 seasons of one year) ---
    for yr in YEARS:
        indices = []
        for s in SEASONS:
            indices.extend(col_season_year.get((yr, s), []))
        label = f"all_{yr}"

        def perturb(X, _idx=indices):
            X2 = X.copy()
            X2[:, _idx] = 0.0
            return X2

        cm = run_perturbed(net, data, perturb)
        agg = agg_cities(cm)
        agg["dropped"] = label
        agg["n_zeroed"] = len(indices)
        t_stat, p_val = paired_ttest(baseline_cm, cm)
        agg["ttest_t"] = t_stat
        agg["ttest_p"] = p_val
        results.append(agg)
        print(f"  drop {label:16s} ({len(indices):4d} cols): "
              f"R2={agg['mean_r2']:.4f}  MAE={agg['mae']:.4f}  "
              f"Top1={agg['top1_acc']:.4f}  Top2={agg['top2_acc']:.4f}  "
              f"Top3={agg['top3_acc']:.4f}  p={p_val:.4f}")

    # --- Cross-year same season (e.g. both springs) ---
    for s in SEASONS:
        indices = []
        for yr in YEARS:
            indices.extend(col_season_year.get((yr, s), []))
        label = f"both_{s}"

        def perturb(X, _idx=indices):
            X2 = X.copy()
            X2[:, _idx] = 0.0
            return X2

        cm = run_perturbed(net, data, perturb)
        agg = agg_cities(cm)
        agg["dropped"] = label
        agg["n_zeroed"] = len(indices)
        t_stat, p_val = paired_ttest(baseline_cm, cm)
        agg["ttest_t"] = t_stat
        agg["ttest_p"] = p_val
        results.append(agg)
        print(f"  drop {label:16s} ({len(indices):4d} cols): "
              f"R2={agg['mean_r2']:.4f}  MAE={agg['mae']:.4f}  "
              f"Top1={agg['top1_acc']:.4f}  Top2={agg['top2_acc']:.4f}  "
              f"Top3={agg['top3_acc']:.4f}  p={p_val:.4f}")

    return results


# =====================================================================
# Experiment 3: Feature-group ablation
# =====================================================================

def exp_ablation(net, data, mlp_cols, baseline_cm):
    results = []

    group_indices = defaultdict(list)
    for i, col in enumerate(mlp_cols):
        group_indices[classify_column(col)].append(i)

    # Baseline
    agg_base = agg_cities(baseline_cm)
    agg_base["removed"] = "none (baseline)"
    agg_base["n_zeroed"] = 0
    results.append(agg_base)
    print(f"  baseline: R2={agg_base['mean_r2']:.4f}  "
          f"MAE={agg_base['mae']:.4f}  Top1={agg_base['top1_acc']:.4f}  "
          f"Top2={agg_base['top2_acc']:.4f}  Top3={agg_base['top3_acc']:.4f}")

    # Individual groups
    for group in sorted(group_indices.keys()):
        indices = group_indices[group]

        def perturb(X, _idx=indices):
            X2 = X.copy()
            X2[:, _idx] = 0.0
            return X2

        cm = run_perturbed(net, data, perturb)
        agg = agg_cities(cm)
        agg["removed"] = group
        agg["n_zeroed"] = len(indices)
        t_stat, p_val = paired_ttest(baseline_cm, cm)
        agg["ttest_t"] = t_stat
        agg["ttest_p"] = p_val
        results.append(agg)
        print(f"  remove {group:20s} ({len(indices):4d} cols): "
              f"R2={agg['mean_r2']:.4f}  MAE={agg['mae']:.4f}  "
              f"Top1={agg['top1_acc']:.4f}  Top2={agg['top2_acc']:.4f}  "
              f"Top3={agg['top3_acc']:.4f}  p={p_val:.4f}")

    # Multi-group combos
    combos = [
        ("Bands+Indices", ["Bands", "Indices"]),
        ("SAR+LBP", ["SAR", "LBP"]),
        ("Spatial+Pheno+TC", ["Spatial", "Phenological", "Tasseled_Cap"]),
    ]
    for combo_name, groups in combos:
        indices = []
        for g in groups:
            indices.extend(group_indices.get(g, []))
        if not indices:
            continue

        def perturb(X, _idx=indices):
            X2 = X.copy()
            X2[:, _idx] = 0.0
            return X2

        cm = run_perturbed(net, data, perturb)
        agg = agg_cities(cm)
        agg["removed"] = combo_name
        agg["n_zeroed"] = len(indices)
        t_stat, p_val = paired_ttest(baseline_cm, cm)
        agg["ttest_t"] = t_stat
        agg["ttest_p"] = p_val
        results.append(agg)
        print(f"  remove {combo_name:20s} ({len(indices):4d} cols): "
              f"R2={agg['mean_r2']:.4f}  MAE={agg['mae']:.4f}  "
              f"Top1={agg['top1_acc']:.4f}  Top2={agg['top2_acc']:.4f}  "
              f"Top3={agg['top3_acc']:.4f}  p={p_val:.4f}")

    return results


# =====================================================================
# Save all results
# =====================================================================

def save_results(noise_res, season_res, ablation_res, output_dir, n_seeds):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Helper: flatten per-class R2 into columns ----
    def flatten_row(r, extra_cols):
        row = dict(extra_cols)
        for k in ["mean_r2", "mae", "top1_acc", "top2_acc", "top3_acc"]:
            row[k] = r[k]
        for k in ["r2_seeds_std", "r2_ci95", "ttest_t", "ttest_p",
                   "mean_r2_std", "top1_acc_std", "mae_std"]:
            if k in r:
                row[k] = r[k]
        for cname in CLASS_NAMES:
            row[f"r2_{cname}"] = r["per_class_r2"].get(cname, float("nan"))
        return row

    # ---- Helper: flatten per-city detail ----
    def flatten_per_city(r, condition_label, condition_key, condition_val):
        rows = []
        if "per_city" not in r:
            return rows
        for city, cm in r["per_city"].items():
            row = {condition_key: condition_val, "city": city}
            for k in ["top1_acc", "top2_acc", "top3_acc", "mean_r2", "mae"]:
                row[k] = cm[k]
            for cname in CLASS_NAMES:
                row[f"r2_{cname}"] = cm["per_class_r2"].get(cname, float("nan"))
            rows.append(row)
        return rows

    # ==== 1. Noise injection ====
    noise_rows = [flatten_row(r, {"sigma": r["sigma"]}) for r in noise_res]
    pd.DataFrame(noise_rows).to_csv(
        os.path.join(output_dir, "noise_injection.csv"), index=False, float_format="%.6f")

    noise_city_rows = []
    for r in noise_res:
        noise_city_rows.extend(flatten_per_city(r, "sigma", "sigma", r["sigma"]))
    if noise_city_rows:
        pd.DataFrame(noise_city_rows).to_csv(
            os.path.join(output_dir, "noise_injection_per_city.csv"),
            index=False, float_format="%.6f")

    # ==== 2. Season dropout ====
    season_rows = [flatten_row(r, {"dropped": r["dropped"], "n_zeroed": r["n_zeroed"]})
                   for r in season_res]
    pd.DataFrame(season_rows).to_csv(
        os.path.join(output_dir, "season_dropout.csv"), index=False, float_format="%.6f")

    season_city_rows = []
    for r in season_res:
        season_city_rows.extend(
            flatten_per_city(r, "dropped", "dropped", r["dropped"]))
    if season_city_rows:
        pd.DataFrame(season_city_rows).to_csv(
            os.path.join(output_dir, "season_dropout_per_city.csv"),
            index=False, float_format="%.6f")

    # ==== 3. Feature ablation ====
    abl_rows = [flatten_row(r, {"removed": r["removed"], "n_zeroed": r["n_zeroed"]})
                for r in ablation_res]
    pd.DataFrame(abl_rows).to_csv(
        os.path.join(output_dir, "feature_ablation.csv"), index=False, float_format="%.6f")

    abl_city_rows = []
    for r in ablation_res:
        abl_city_rows.extend(
            flatten_per_city(r, "removed", "removed", r["removed"]))
    if abl_city_rows:
        pd.DataFrame(abl_city_rows).to_csv(
            os.path.join(output_dir, "feature_ablation_per_city.csv"),
            index=False, float_format="%.6f")

    # ==== Combined JSON ====
    combined = {
        "model": f"V10 trial {DEPLOYED_TRIAL}",
        "test_cities": TEST_CITIES,
        "n_noise_seeds": n_seeds,
        "r2_threshold": R2_THRESHOLD,
        "methodology": {
            "perturbation_space": "z-scored (post-StandardScaler)",
            "zeroing_semantics": "equivalent to mean imputation in raw feature space",
            "noise_sigma_units": "standard deviations of z-scored features",
            "r2_computation": f"per-class, evaluated on samples where y >= {R2_THRESHOLD}",
            "topk_threshold": LABEL_THRESHOLD,
            "topk_semantics": "Top-K set-match: cells with >=K classes above label_threshold; checks if top-K predicted classes match top-K true classes by set equality",
            "significance_test": "paired two-sided t-test across 6 test cities (scipy.stats.ttest_rel)",
        },
        "noise_injection": noise_res,
        "season_dropout": season_res,
        "feature_ablation": ablation_res,
    }
    # Strip per_city from JSON (too verbose), keep in per-city CSVs
    for section in ["noise_injection", "season_dropout", "feature_ablation"]:
        for entry in combined[section]:
            entry.pop("per_city", None)

    with open(os.path.join(output_dir, "stress_results.json"), "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_dir}")
    print(f"  Files:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath)
        print(f"    {fname:45s} {size:>8,d} bytes")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Stress test V10 deployed MLP (report-grade)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of noise seeds for averaging (default 10)")
    args = parser.parse_args()

    t0 = time.time()
    sep = "=" * 90

    print(f"\n{sep}")
    print("  V10 MLP Stress Test (Report-Grade) -- Trial 77 on 6 Held-Out Test Cities")
    print(f"{sep}\n")

    print("Loading model...")
    net, mlp_cols, scaler, trial_info = load_model()
    print(f"  Arch: {trial_info['arch']}  |  Params: {trial_info['n_params']:,}")
    print(f"  Activation: {trial_info['config']['activation']}  "
          f"|  Dropout: {trial_info['config']['dropout']:.4f}")
    print(f"  Device: {device}\n")

    print("Loading test data...")
    data = load_test_data(mlp_cols, scaler)
    total_cells = sum(X.shape[0] for X, _ in data.values())
    print(f"  Total: {total_cells:,} cells across {len(data)} cities\n")

    # --- Exp 1: Noise ---
    print(f"{sep}")
    print(f"  Experiment 1: Gaussian Noise Injection ({args.seeds} seeds per sigma)")
    print(f"{sep}")
    noise_results, baseline_cm = exp_noise(net, data, n_seeds=args.seeds)

    # --- Exp 2: Season dropout ---
    print(f"\n{sep}")
    print("  Experiment 2: Season Dropout (individual + full-year + cross-year)")
    print(f"{sep}")
    season_results = exp_season_dropout(net, data, mlp_cols, baseline_cm)

    # --- Exp 3: Feature ablation ---
    print(f"\n{sep}")
    print("  Experiment 3: Feature-Group Ablation (zero entire groups)")
    print(f"{sep}")
    ablation_results = exp_ablation(net, data, mlp_cols, baseline_cm)

    # --- Save ---
    print(f"\n{sep}")
    print("  Saving results...")
    print(f"{sep}")
    save_results(noise_results, season_results, ablation_results,
                 OUTPUT_DIR, args.seeds)

    elapsed = time.time() - t0
    print(f"\n  Stress test complete in {elapsed:.0f}s")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
