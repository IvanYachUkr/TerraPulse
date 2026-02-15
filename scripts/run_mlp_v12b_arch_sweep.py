#!/usr/bin/env python3
"""
MLP V12b: Architecture sweep on the winning Rust feature set (bi_LBP_mLBP).

Fixed: all 1,344 Rust-extracted features (bands+indices+TC+spatial+5-band LBP)
Sweep:
  1. plain  silu L5  d1024 bn  (V12 baseline, already done)
  2. plain  silu L5  d1536 bn  (V12 best,     already done)
  3. plain  silu L5  d2048 bn  (wider)
  4. plain  silu L7  d1024 bn  (deeper)
  5. plain  silu L7  d1536 bn  (deeper+wider)
  6. residual silu L10 d512 bn  (deep residual)
  7. residual silu L20 d256 bn  (V10 runner-up arch)
  8. residual silu L20 d512 bn  (deep+wide residual)

Total: 6 new configs × 5 folds = 30 runs  (~30-40 min on CUDA)

Output: reports/phase8/tables/mlp_v12b_arch_sweep.csv
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
V12B_CSV = os.path.join(OUT_DIR, "mlp_v12b_arch_sweep.csv")


def build_data(device):
    """Load all 1,344 Rust features (bi_LBP_mLBP)."""
    feat_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    print(f"Loading Rust features: {feat_path}", flush=True)
    feat_df = pd.read_parquet(feat_path)

    from pandas.api.types import is_numeric_dtype
    feature_cols = [c for c in feat_df.columns
                    if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    X = feat_df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    del feat_df

    print(f"  Features: {len(feature_cols)}", flush=True)
    return X, feature_cols


def build_configs():
    """Architecture sweep — all use bi_LBP_mLBP feature set."""
    fs = "bi_LBP_mLBP"
    configs = []

    # Width sweep (plain L5)
    for d in [2048]:
        c = _cfg(0, fs, "plain", "silu", 5, d, "batchnorm")
        configs.append((f"plain_silu_L5_d{d}_bn", c))

    # Depth sweep (plain d1024 / d1536)
    for L in [7]:
        for d in [1024, 1536]:
            c = _cfg(0, fs, "plain", "silu", L, d, "batchnorm")
            configs.append((f"plain_silu_L{L}_d{d}_bn", c))

    # Residual architectures
    for L, d in [(10, 512), (20, 256), (20, 512)]:
        c = _cfg(0, fs, "residual", "silu", L, d, "batchnorm")
        c["lr"] = 0.0005  # lower LR for deep residual (from V10)
        configs.append((f"residual_silu_L{L}_d{d}_bn_lr5e-4", c))

    return configs


def main():
    parser = argparse.ArgumentParser(description="V12b: Arch sweep on Rust features")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
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

    X, feature_cols = build_data(device)
    n_features = X.shape[1]

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    configs = build_configs()
    total_runs = len(configs) * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V12b — Architecture Sweep on Rust Features ({n_features}f)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Configs:  {len(configs)}", flush=True)
    print(f"  Folds:    {folds_to_run}", flush=True)
    print(f"  Total:    {total_runs} runs", flush=True)
    print(f"  Device:   {device}", flush=True)
    for name, cfg in configs:
        arch = cfg.get("arch", "plain")
        print(f"    {name}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Resume
    results = []
    done_keys = set()
    if os.path.exists(V12B_CSV):
        df_old = pd.read_csv(V12B_CSV)
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

        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        # Pre-scale once per fold (all configs use same features)
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X[trn_idx]).astype(np.float32)
        X_val_s = scaler.transform(X[val_idx]).astype(np.float32)
        X_test_s = scaler.transform(X[test_idx]).astype(np.float32)

        X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)

        for name, cfg in configs:
            run_idx += 1
            if (name, fold_id) in done_keys:
                continue

            t0 = time.time()
            net = build_model(cfg, n_features, device)

            n_epochs, best_val, trained_net = train_model(
                net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-4),
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience_steps=args.patience_steps,
                min_steps=args.min_steps,
            )

            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
            preds = _predict_batched(trained_net, X_test_t, device)
            elapsed = time.time() - t0

            metrics, _ = evaluate_model(y[test_idx], preds)
            r2 = metrics["r2_uniform"]
            mae = metrics["mae_mean_pp"]

            times.append(elapsed)
            runs_left = total_runs - run_idx
            eta_h = (np.mean(times) * runs_left) / 3600

            rec = {
                "name": name,
                "fold": fold_id,
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
            pd.DataFrame(results).to_csv(V12B_CSV, index=False)

            print(f"  [{run_idx:3d}/{total_runs}] F{fold_id} {name:45s} "
                  f"R2={r2:.4f}  MAE={mae:.2f}pp  "
                  f"ep={n_epochs}  {elapsed:.0f}s  "
                  f"ETA={eta_h:.1f}h", flush=True)

            del net, trained_net, X_test_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del X_trn_t, X_val_t

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"V12b COMPLETE — {len(results)} runs", flush=True)
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
        summary_csv = V12B_CSV.replace(".csv", "_summary.csv")
        summary.to_csv(summary_csv)
        print(summary.to_string())
        print(f"\nSaved: {V12B_CSV}")
        print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
