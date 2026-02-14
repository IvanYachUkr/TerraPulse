#!/usr/bin/env python3
"""
MLP V5.5: Architecture Sweep — Deep vs Wide vs Deep+Wide.

Purpose: Compare three architecture families on the best feature
combination (GLCM+Morph+VegIdx+RedEdge+TC) from the ablation study,
plus an extended set that adds new v2 indices (EVI2, GNDVI, NDTI,
IRECI, CRI1, MNDWI + edge/lap/morans).

Architecture families:
  - Deep:       L12/L16/L20, d256  (many blocks, narrow)
  - Wide:       L5, d1024/d2048     (few blocks, very wide)
  - Deep+Wide:  L10/L11/L12, d1024  (balanced depth + width)

Feature sets:
  - best5:     GLCM + Morph + VegIdx + RedEdge + TC  (from merged_full)
  - best5_v2:  same + 180 new v2 indices              (merged_full + v2 extras)

Output:
    reports/phase8/tables/mlp_v5_5_arch.csv           — per-fold results
    reports/phase8/tables/mlp_v5_5_arch_summary.csv   — mean ± std

Usage:
    .venv\\Scripts\\python.exe scripts/run_mlp_v5_5_arch_sweep.py
    .venv\\Scripts\\python.exe scripts/run_mlp_v5_5_arch_sweep.py --max-epochs 3000
    .venv\\Scripts\\python.exe scripts/run_mlp_v5_5_arch_sweep.py --folds 0 1
"""

import argparse
import json as _json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ── Reuse V4/V5 infrastructure ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from run_mlp_overnight_v4 import (
    # Constants
    PROJECT_ROOT, CLASS_NAMES, N_FOLDS, CONTROL_COLS, SEED,
    # Model building
    build_model, make_name, _cfg,
    # Training
    train_model, normalize_targets, _predict_batched,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

# V5.5 output paths
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V55_CSV = os.path.join(OUT_DIR, "mlp_v5_5_arch.csv")
V55_SUMMARY = os.path.join(OUT_DIR, "mlp_v5_5_arch_summary.csv")


# =====================================================================
# Feature group builder (matching ablation_best5_mixes.py)
# =====================================================================

def build_feature_groups(feat_cols):
    """Identify the best-5 feature groups from column names.

    Groups: GLCM, Morph (MP), VegIdx (NDVI/SAVI/NDRE*),
            RedEdge (B05-B07, B8A), TC.
    """
    band_pat = re.compile(r'^B(05|06|07|8A)_')

    morph = [c for c in feat_cols if "MP_" in c]
    glcm = [c for c in feat_cols if "GLCM_" in c]

    veg_idx = [c for c in feat_cols
               if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"])
               and not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr")]

    rededge = [c for c in feat_cols if band_pat.match(c)]
    tc = [c for c in feat_cols if c.startswith("TC_")]

    return {
        "Morph": morph, "GLCM": glcm, "VegIdx": veg_idx,
        "RedEdge": rededge, "TC": tc,
    }


def get_best5_cols(feat_cols):
    """Return sorted column indices for GLCM+Morph+VegIdx+RedEdge+TC."""
    groups = build_feature_groups(feat_cols)
    selected = []
    for g in ["GLCM", "Morph", "VegIdx", "RedEdge", "TC"]:
        selected.extend(groups[g])
    # Deduplicate preserving order
    seen = set()
    out = []
    for c in selected:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# =====================================================================
# Architecture configs
# =====================================================================

FEATURE_SETS = ["best5", "best5_v2"]


def generate_v55_configs():
    """Architecture sweep: deep, wide, deep+wide. SiLU activation only."""
    configs = []
    rid = 0

    for fs in FEATURE_SETS:
        # ── Deep family: many blocks, narrow width (d256) ──
        configs.append(_cfg(rid, fs, "residual", "silu",  12, 256, "batchnorm")); rid += 1
        configs.append(_cfg(rid, fs, "residual", "silu",  16, 256, "batchnorm", lr=5e-4)); rid += 1
        configs.append(_cfg(rid, fs, "residual", "silu",  20, 256, "batchnorm", lr=5e-4)); rid += 1

        # ── Wide family: 5 layers, very wide ──
        configs.append(_cfg(rid, fs, "plain", "silu",  5, 1024, "batchnorm")); rid += 1
        configs.append(_cfg(rid, fs, "plain", "silu",  5, 2048, "batchnorm", lr=5e-4)); rid += 1

        # ── Deep+Wide family: L10-L12 x d1024 ──
        configs.append(_cfg(rid, fs, "residual", "silu", 10, 1024, "batchnorm", lr=5e-4)); rid += 1
        configs.append(_cfg(rid, fs, "residual", "silu", 11, 1024, "batchnorm", lr=5e-4)); rid += 1
        configs.append(_cfg(rid, fs, "residual", "silu", 12, 1024, "batchnorm", lr=5e-4)); rid += 1

    return configs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MLP V5.5: Architecture Sweep")
    parser.add_argument("--max-epochs", type=int, default=2000,
                        help="Max epochs (default: 2000 — rely on early stopping)")
    parser.add_argument("--patience-steps", type=int, default=5000,
                        help="Early stopping patience in steps (default: 5000)")
    parser.add_argument("--min-steps", type=int, default=2000,
                        help="Min steps before early stopping (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Override which folds to run (default: all 5)")
    args = parser.parse_args()

    folds_to_run = args.folds if args.folds else list(range(N_FOLDS))

    configs = generate_v55_configs()
    total_configs = len(configs)
    total_runs = total_configs * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V5.5 — ARCHITECTURE SWEEP: Deep vs Wide vs Deep+Wide", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Configs:        {total_configs}", flush=True)
    print(f"  Folds:          {folds_to_run}", flush=True)
    print(f"  Total runs:     {total_runs}", flush=True)
    print(f"  Max epochs:     {args.max_epochs}", flush=True)
    print(f"  Patience:       {args.patience_steps} steps", flush=True)
    print(f"  Min steps:      {args.min_steps}", flush=True)
    print(f"  Output CSV:     {V55_CSV}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Show all configs
    print("Configs:", flush=True)
    for c in configs:
        print(f"  {make_name(c)}", flush=True)
    print(flush=True)

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── Load data ──
    print("Loading data...", flush=True)

    # 1) Base merged features
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # 2) V2 extended bands/indices
    v2_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet"))
    v2_only_cols = [c for c in v2_df.columns
                    if c != "cell_id" and c not in feat_df.columns]
    print(f"  V2 new columns: {len(v2_only_cols)}", flush=True)

    # Merge v2 extras into feat_df (aligned by row order — same cell ordering)
    for c in v2_only_cols:
        feat_df[c] = v2_df[c].values
    del v2_df

    all_cols = full_feature_cols + v2_only_cols

    # 3) Build feature-set column selections
    best5_cols = get_best5_cols(full_feature_cols)
    best5_indices = [all_cols.index(c) for c in best5_cols]

    # best5_v2 = best5 + v2-only columns that match our target groups + all new indices
    best5_v2_cols = best5_cols + v2_only_cols
    best5_v2_indices = [all_cols.index(c) for c in best5_v2_cols]

    feat_set_map = {
        "best5": best5_indices,
        "best5_v2": best5_v2_indices,
    }
    for name, idx_list in feat_set_map.items():
        print(f"  Feature set '{name}': {len(idx_list)} features", flush=True)

    # Build numpy arrays
    X_all = feat_df[all_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    print(f"Data loaded: X={X_all.shape}, y={y.shape}", flush=True)

    # ── Resume logic ──
    results = []
    done_keys = set()
    if os.path.exists(V55_CSV):
        df_old = pd.read_csv(V55_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs already done", flush=True)

    # ── Cross-validation loop ──
    os.makedirs(OUT_DIR, exist_ok=True)
    run_idx = 0
    n_skipped = 0
    times = []

    for fold_id in folds_to_run:
        print(f"\n{'='*80}", flush=True)
        print(f"FOLD {fold_id}/{N_FOLDS-1}", flush=True)
        print(f"{'='*80}", flush=True)

        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )

        # Internal val split (15% of train)
        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx = train_idx[perm[:n_val]]
        trn_idx = train_idx[perm[n_val:]]
        print(f"  Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}", flush=True)

        # Pre-scale all feature sets for this fold
        print("  Pre-scaling feature sets -> GPU...", flush=True)
        scaled_cache = {}
        for fs_name, feat_idx in feat_set_map.items():
            X = X_all[:, feat_idx]
            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X[val_idx]).astype(np.float32)
            X_test_s = scaler.transform(X[test_idx]).astype(np.float32)
            scaled_cache[fs_name] = (
                torch.tensor(X_trn_s, dtype=torch.float32).to(device),
                torch.tensor(X_val_s, dtype=torch.float32).to(device),
                torch.tensor(X_test_s, dtype=torch.float32),  # CPU for batched predict
                len(feat_idx),
            )
            print(f"    {fs_name}: {len(feat_idx)} features", flush=True)

        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        for ci, cfg in enumerate(configs):
            name = make_name(cfg)
            fs = cfg["feature_set"]
            run_idx += 1

            if (name, fold_id) in done_keys:
                n_skipped += 1
                continue

            if fs not in scaled_cache:
                continue

            X_trn_t, X_val_t, X_test_t, n_features = scaled_cache[fs]

            t0 = time.time()
            try:
                torch.manual_seed(SEED + fold_id)
                net = build_model(cfg, n_features, device)

                epochs_done, best_val_loss, final_model = train_model(
                    net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                    lr=cfg["lr"],
                    weight_decay=cfg.get("weight_decay", 1e-4),
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience_steps=args.patience_steps,
                    min_steps=args.min_steps,
                    mixup_alpha=cfg.get("mixup_alpha", 0),
                    use_swa=cfg.get("use_swa", False),
                    use_cosine=cfg.get("use_cosine", True),
                )

                # Evaluate on TEST set
                y_pred = _predict_batched(final_model, X_test_t, device)
                elapsed = time.time() - t0

                del net, final_model

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)

                # Classify architecture family
                n_layers = cfg["n_layers"]
                d_model = cfg["d_model"]
                if d_model >= 1024 and n_layers <= 5 and cfg["arch"] == "plain":
                    arch_family = "wide"
                elif d_model <= 256 and n_layers >= 12:
                    arch_family = "deep"
                else:
                    arch_family = "deep+wide"

                summary.update({
                    "name": name, "fold": fold_id, "stage": "v5_5_arch",
                    "feature_set": fs, "arch": cfg["arch"],
                    "activation": cfg["activation"],
                    "norm_type": cfg.get("norm_type", "layernorm"),
                    "n_layers": n_layers, "d_model": d_model,
                    "arch_family": arch_family,
                    "lr": cfg["lr"], "n_features": n_features,
                    "epochs": epochs_done, "elapsed_s": round(elapsed, 1),
                })

                results.append(summary)
                done_keys.add((name, fold_id))
                times.append(elapsed)

                # Running average ETA
                avg_time = sum(times) / len(times)
                remaining = max(0, total_runs - n_skipped - len(times))
                eta_h = remaining * avg_time / 3600

                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                early_tag = f"ep={epochs_done:3d}" if epochs_done < args.max_epochs else f"ep={epochs_done:3d}(MAX)"
                print(f"  [{n_skipped+len(times):4d}/{total_runs}] F{fold_id} {name:60s} "
                      f"R2={r2:.4f}  MAE={mae:.2f}pp  {early_tag}  "
                      f"{elapsed:.0f}s  ETA={eta_h:.1f}h  [{arch_family}]", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                results.append({"name": name, "fold": fold_id,
                                "stage": "v5_5_arch", "error": str(e)})
                done_keys.add((name, fold_id))
                print(f"  [{n_skipped+len(times):4d}/{total_runs}] F{fold_id} {name:60s} "
                      f"ERROR: {e} ({elapsed:.0f}s)", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()

            # Save periodically
            if (n_skipped + len(times)) % 5 == 0:
                pd.DataFrame(results).to_csv(V55_CSV, index=False)

        # End-of-fold save and cleanup
        pd.DataFrame(results).to_csv(V55_CSV, index=False)
        del scaled_cache, y_trn_t, y_val_t
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final save
    pd.DataFrame(results).to_csv(V55_CSV, index=False)
    print(f"\nResults saved to {V55_CSV}", flush=True)

    # ── Summary ──
    df = pd.DataFrame(results)
    valid = df[df["r2_uniform"].notna()]
    if len(valid) > 0:
        agg = (valid.groupby("name")
               .agg(
                   r2_mean=("r2_uniform", "mean"),
                   r2_std=("r2_uniform", "std"),
                   mae_mean=("mae_mean_pp", "mean"),
                   epochs_mean=("epochs", "mean"),
                   folds=("fold", "count"),
                   feature_set=("feature_set", "first"),
                   arch=("arch", "first"),
                   activation=("activation", "first"),
                   arch_family=("arch_family", "first"),
               )
               .sort_values("r2_mean", ascending=False))
        agg.to_csv(V55_SUMMARY)
        print(f"Summary saved to {V55_SUMMARY}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"V5.5 ARCHITECTURE SWEEP RESULTS", flush=True)
        print(f"{'='*70}", flush=True)
        for _, row in agg.iterrows():
            folds_str = f"folds={int(row['folds'])}"
            print(f"  {row.name:60s} R2={row['r2_mean']:.4f}+/-{row['r2_std']:.4f}  "
                  f"MAE={row['mae_mean']:.2f}  ep={row['epochs_mean']:.0f}  "
                  f"{folds_str}  [{row['arch_family']}]", flush=True)

        # Best per architecture family
        print(f"\nBEST PER ARCHITECTURE FAMILY:", flush=True)
        for fam in ["deep", "wide", "deep+wide"]:
            fam_agg = agg[agg["arch_family"] == fam]
            if len(fam_agg) > 0:
                best = fam_agg.iloc[0]
                print(f"  {fam:12s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  {best.name}", flush=True)

        # Best per feature set
        print(f"\nBEST PER FEATURE SET:", flush=True)
        for fs in sorted(valid["feature_set"].unique()):
            fs_agg = agg[agg["feature_set"] == fs]
            if len(fs_agg) > 0:
                best = fs_agg.iloc[0]
                print(f"  {fs:20s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  {best.name}", flush=True)

        # Cross-table: family x feature set
        print(f"\nFAMILY × FEATURE SET (mean R²):", flush=True)
        for fam in ["deep", "wide", "deep+wide"]:
            for fs in FEATURE_SETS:
                subset = agg[(agg["arch_family"] == fam) & (agg["feature_set"] == fs)]
                if len(subset) > 0:
                    best = subset.iloc[0]
                    print(f"  {fam:12s} × {fs:12s}: R2={best['r2_mean']:.4f}  {best.name}", flush=True)

        # Convergence stats
        n_early = len(valid[valid["epochs"] < args.max_epochs])
        n_maxed = len(valid[valid["epochs"] >= args.max_epochs])
        print(f"\nCONVERGENCE: {n_early}/{len(valid)} early-stopped, "
              f"{n_maxed}/{len(valid)} hit cap ({args.max_epochs}ep)", flush=True)

    n_errors = len(df) - len(valid)
    if n_errors:
        print(f"\nERRORS: {n_errors}", flush=True)
        for _, row in df[df["r2_uniform"].isna()].iterrows():
            print(f"  {row.get('name', '?')} F{row.get('fold', '?')}: {row.get('error', '?')}", flush=True)

    total_time = sum(times)
    if total_time > 0:
        print(f"\nTotal compute time: {total_time/3600:.1f}h", flush=True)
    elif n_skipped == total_runs:
        print("\nAll runs already done (resumed from CSV)", flush=True)


if __name__ == "__main__":
    main()
