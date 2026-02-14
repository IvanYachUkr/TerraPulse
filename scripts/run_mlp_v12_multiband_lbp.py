#!/usr/bin/env python3
"""
MLP V12: Multi-band LBP features — test whether LBP on multiple
spectral bands/indices outperforms NIR-only LBP.

Feature sets:
  - bi_LBP_multi  (bands+indices + old NIR LBP + new multi-band LBP)
  - bi_mLBP_only  (bands+indices + new multi-band LBP, NO old LBP)

Architectures:  plain silu L5, batchnorm, d1024 / d1536

Total: 4 configs × 5 folds = 20 runs

Requires:
  - features_merged_full.parquet (existing)
  - features_lbp_multiband.parquet (from extract_lbp_multiband.py)

Output: reports/phase8/tables/mlp_v12_multiband_lbp.csv
"""

import argparse
import json as _json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from run_mlp_overnight_v4 import (
    PROJECT_ROOT, CLASS_NAMES, N_FOLDS, CONTROL_COLS, SEED,
    build_model, _cfg,
    train_model, normalize_targets, _predict_batched,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V12_CSV = os.path.join(OUT_DIR, "mlp_v12_multiband_lbp.csv")


# =====================================================================
# Data loading
# =====================================================================

def build_v12_data(device):
    """Load features from Rust-extracted features_v3.parquet.

    The Rust output already contains bands, indices, TC, spatial, and
    multi-band LBP (NIR, NDVI, EVI2, SWIR1, NDTI) — everything V12 needs.

    Returns: X_all (numpy), feature_sets dict, all_cols list
    """
    feat_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    print(f"Loading Rust features: {feat_path}", flush=True)
    feat_df = pd.read_parquet(feat_path)

    from pandas.api.types import is_numeric_dtype
    all_feature_cols = [c for c in feat_df.columns
                        if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # Partition by name patterns
    lbp_nir_cols = [c for c in all_feature_cols if c.startswith("LBP_NIR_")]
    lbp_multi_cols = [c for c in all_feature_cols
                      if c.startswith("LBP_") and not c.startswith("LBP_NIR_")]
    base_cols = [c for c in all_feature_cols
                 if not c.startswith("LBP_")]

    print(f"  Base (bands+indices+TC+spatial): {len(base_cols)} cols", flush=True)
    print(f"  LBP NIR:                         {len(lbp_nir_cols)} cols", flush=True)
    print(f"  LBP Multi (NDVI,EVI2,SWIR1,NDTI):{len(lbp_multi_cols)} cols", flush=True)

    # Order: base, LBP_NIR, LBP_multi
    ordered_cols = base_cols + lbp_nir_cols + lbp_multi_cols
    X_all = feat_df[ordered_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df

    n_base = len(base_cols)
    n_nir = len(lbp_nir_cols)
    n_multi = len(lbp_multi_cols)

    base_idx = list(range(n_base))
    nir_idx = list(range(n_base, n_base + n_nir))
    multi_idx = list(range(n_base + n_nir, n_base + n_nir + n_multi))

    feature_sets = {
        # Multi-band LBP only (no old NIR LBP)
        "bi_mLBP_only": sorted(set(base_idx) | set(multi_idx)),
        # NIR LBP + multi-band LBP (full LBP complement)
        "bi_LBP_mLBP":  sorted(set(base_idx) | set(nir_idx) | set(multi_idx)),
    }

    for name, idx in feature_sets.items():
        print(f"  Feature set '{name}': {len(idx)} features", flush=True)

    return X_all, feature_sets, ordered_cols


# =====================================================================
# V12 configs
# =====================================================================

def build_v12_configs():
    """Build V12 configs: 4 total (2 feature sets × 2 widths)."""
    configs = []

    # ── bi_mLBP_only: new multi-band LBP, no old NIR LBP ──

    # 1. Baseline: V10 champion arch
    c1 = _cfg(0, "bi_mLBP_only", "plain", "silu", 5, 1024, "batchnorm")
    configs.append(("bi_mLBP_only_plain_silu_L5_d1024_bn", c1, "bi_mLBP_only"))

    # 2. Wider
    c2 = _cfg(0, "bi_mLBP_only", "plain", "silu", 5, 1536, "batchnorm")
    configs.append(("bi_mLBP_only_plain_silu_L5_d1536_bn", c2, "bi_mLBP_only"))

    # ── bi_LBP_mLBP: old NIR LBP + new multi-band ──

    # 3. Baseline
    c3 = _cfg(0, "bi_LBP_mLBP", "plain", "silu", 5, 1024, "batchnorm")
    configs.append(("bi_LBP_mLBP_plain_silu_L5_d1024_bn", c3, "bi_LBP_mLBP"))

    # 4. Wider
    c4 = _cfg(0, "bi_LBP_mLBP", "plain", "silu", 5, 1536, "batchnorm")
    configs.append(("bi_LBP_mLBP_plain_silu_L5_d1536_bn", c4, "bi_LBP_mLBP"))

    return configs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="V12: Multi-band LBP sweep")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for LBP extraction")
    args = parser.parse_args()

    folds_to_run = args.folds if args.folds else list(range(N_FOLDS))



    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Load data
    X_all, feature_sets, full_cols = build_v12_data(device)

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    # Build configs
    configs = build_v12_configs()
    total_runs = len(configs) * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V12 — Multi-Band LBP Sweep", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Configs:  {len(configs)}", flush=True)
    print(f"  Folds:    {folds_to_run}", flush=True)
    print(f"  Total:    {total_runs} runs", flush=True)
    print(f"  Device:   {device}", flush=True)
    for name, cfg, fs_name in configs:
        n_feat = len(feature_sets[fs_name])
        print(f"    {name:50s}  {n_feat:5d}f", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Resume
    results = []
    done_keys = set()
    if os.path.exists(V12_CSV):
        df_old = pd.read_csv(V12_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs already done", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    times = []
    run_idx = 0

    for fold_id in folds_to_run:
        print(f"\n{'='*80}", flush=True)
        print(f"FOLD {fold_id}", flush=True)
        print(f"{'='*80}", flush=True)

        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )

        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx = train_idx[perm[:n_val]]
        trn_idx = train_idx[perm[n_val:]]
        print(f"  Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}",
              flush=True)

        # Targets
        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        for name, cfg, fs_name in configs:
            run_idx += 1
            if (name, fold_id) in done_keys:
                continue

            fs_idx = feature_sets[fs_name]
            n_features = len(fs_idx)

            X_fs = X_all[:, fs_idx]
            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X_fs[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X_fs[val_idx]).astype(np.float32)
            X_test_s = scaler.transform(X_fs[test_idx]).astype(np.float32)

            X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device)
            X_val_t_fs = torch.tensor(X_val_s, dtype=torch.float32).to(device)

            t0 = time.time()
            net = build_model(cfg, n_features, device)

            n_epochs, best_val, trained_net = train_model(
                net, X_trn_t, y_trn_t, X_val_t_fs, y_val_t,
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-4),
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience_steps=args.patience_steps,
                min_steps=args.min_steps,
            )

            # Predict
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
            preds = _predict_batched(trained_net, X_test_t, device)
            elapsed = time.time() - t0

            # Evaluate
            metrics, _ = evaluate_model(y[test_idx], preds)
            r2 = metrics["r2_uniform"]
            mae = metrics["mae_mean_pp"]

            # ETA
            times.append(elapsed)
            runs_left = total_runs - run_idx
            eta_h = (np.mean(times) * runs_left) / 3600

            rec = {
                "name": name,
                "fold": fold_id,
                "feature_set": fs_name,
                "n_features": n_features,
                "arch": f"{cfg['arch']}_L{cfg['n_layers']}_d{cfg['d_model']}",
                "r2_uniform": r2,
                "mae_mean_pp": mae,
                "best_val_loss": best_val,
                "n_epochs": n_epochs,
                "elapsed_s": elapsed,
                "n_train": len(trn_idx),
                "n_test": len(test_idx),
            }
            for cn in CLASS_NAMES:
                if f"r2_{cn}" in metrics:
                    rec[f"r2_{cn}"] = metrics[f"r2_{cn}"]

            results.append(rec)
            done_keys.add((name, fold_id))

            # Save after each run
            pd.DataFrame(results).to_csv(V12_CSV, index=False)

            print(f"  [{run_idx:3d}/{total_runs}] F{fold_id} {name:50s} "
                  f"R2={r2:.4f}  MAE={mae:.2f}pp  "
                  f"ep={n_epochs}  {elapsed:.0f}s  "
                  f"ETA={eta_h:.1f}h", flush=True)

            # Free GPU
            del net, trained_net, X_trn_t, X_val_t_fs, X_test_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"V12 COMPLETE — {len(results)} runs", flush=True)
    print(f"{'='*70}", flush=True)

    df = pd.DataFrame(results)
    if len(df) > 0:
        summary = (df.groupby("name")
                   .agg(r2_mean=("r2_uniform", "mean"),
                        r2_std=("r2_uniform", "std"),
                        mae_mean=("mae_mean_pp", "mean"),
                        mae_std=("mae_mean_pp", "std"),
                        n_folds=("fold", "count"))
                   .sort_values("r2_mean", ascending=False))
        summary_csv = V12_CSV.replace(".csv", "_summary.csv")
        summary.to_csv(summary_csv)
        print(summary.to_string())
        print(f"\nSaved: {V12_CSV}")
        print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
