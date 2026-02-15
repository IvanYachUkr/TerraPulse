#!/usr/bin/env python3
"""
MLP V13: Multi-band LBP on V10-matched feature base.

Uses partition_features() to select EXACTLY the same bands_indices subset
that V10 used, plus LBP.  Tests whether multi-band LBP improves over
single-band (NIR-only) LBP.

Feature sets:
  - bi_LBP_NIR   : V10 champion set (bands+indices + NIR LBP only)
  - bi_LBP_all5  : same base + ALL 5 LBP bands (NIR, NDVI, EVI2, SWIR1, NDTI)

Architectures: plain silu L5 d1024 bn  (V10 champion)
               plain silu L5 d1536 bn  (wider)

Total: 4 configs x 5 folds = 20 runs

Waits for V12b to finish before starting.
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
    partition_features,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V13_CSV = os.path.join(OUT_DIR, "mlp_v13_multiband_lbp_clean.csv")
V12B_CSV = os.path.join(OUT_DIR, "mlp_v12b_arch_sweep.csv")


def build_v13_data():
    """Select V10-equivalent features from Rust output + multi-band LBP.

    Uses partition_features() to pick bands_indices, then REMOVES indices
    that the old pipeline never computed (MNDWI, CRI1) so we get the
    exact same 798 base features as V10.
    """
    feat_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    print(f"Loading Rust features: {feat_path}", flush=True)
    feat_df = pd.read_parquet(feat_path)

    from pandas.api.types import is_numeric_dtype
    all_cols = [c for c in feat_df.columns
                if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # Use V10's partition logic to select bands + indices
    groups = partition_features(all_cols)
    bi_cols = [all_cols[i] for i in groups["bands_indices"]]

    # Remove indices that the old pipeline never computed.
    # The Rust extractor added MNDWI and CRI1 but V10's parquet didn't have them.
    EXCLUDE_PREFIXES = {"MNDWI", "CRI1"}
    bi_cols = [c for c in bi_cols if c.split("_")[0] not in EXCLUDE_PREFIXES]

    # LBP columns
    lbp_nir = [c for c in all_cols if c.startswith("LBP_NIR_")]
    lbp_multi = [c for c in all_cols
                 if c.startswith("LBP_") and not c.startswith("LBP_NIR_")]

    print(f"  bands_indices (V10-matched): {len(bi_cols)} cols", flush=True)
    print(f"  LBP NIR:                     {len(lbp_nir)} cols", flush=True)
    print(f"  LBP multi:                   {len(lbp_multi)} cols "
          f"(NDVI, EVI2, SWIR1, NDTI)", flush=True)

    # Build ordered column lists for each feature set
    bi_lbp_all5_cols = bi_cols + lbp_nir + lbp_multi

    X_all = feat_df[bi_lbp_all5_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df

    n_bi = len(bi_cols)
    n_nir = len(lbp_nir)
    n_multi = len(lbp_multi)

    bi_idx = list(range(n_bi))
    nir_idx = list(range(n_bi, n_bi + n_nir))
    multi_idx = list(range(n_bi + n_nir, n_bi + n_nir + n_multi))

    feature_sets = {
        # Exact V10 champion set: 798 bi + 66 NIR LBP = 864
        "bi_LBP_NIR":  sorted(set(bi_idx) | set(nir_idx)),
        # V10 base + all 5 LBP bands: 798 bi + 330 LBP = 1128
        "bi_LBP_all5": sorted(set(bi_idx) | set(nir_idx) | set(multi_idx)),
    }

    for name, idx in feature_sets.items():
        print(f"  Feature set '{name}': {len(idx)} features", flush=True)

    return X_all, feature_sets, bi_lbp_all5_cols


def build_v13_configs():
    """22 configs: 11 architectures × 2 feature sets.

    Architectures:
      Plain:    L3/L5/L7 × d512/d1024/d1536/d2048  (selected combos)
      Residual: L10 d512, L20 d256, L20 d512
    """
    archs = [
        # (type, activation, layers, width, norm, lr_override)
        ("plain", "silu", 3, 1024,  "batchnorm", None),
        ("plain", "silu", 3, 2048,  "batchnorm", None),
        ("plain", "silu", 5, 512,   "batchnorm", None),
        ("plain", "silu", 5, 1024,  "batchnorm", None),   # V10 champion
        ("plain", "silu", 5, 1536,  "batchnorm", None),
        ("plain", "silu", 5, 2048,  "batchnorm", None),
        ("plain", "silu", 7, 1024,  "batchnorm", None),
        ("plain", "silu", 7, 1536,  "batchnorm", None),
        ("residual", "silu", 10, 512, "batchnorm", 0.0005),
        ("residual", "silu", 20, 256, "batchnorm", 0.0005),  # V10 runner-up
        ("residual", "silu", 20, 512, "batchnorm", 0.0005),
    ]

    configs = []
    for fs in ["bi_LBP_NIR", "bi_LBP_all5"]:
        for arch_type, act, L, d, norm, lr in archs:
            c = _cfg(0, fs, arch_type, act, L, d, norm)
            if lr is not None:
                c["lr"] = lr
            lr_tag = f"_lr{lr}" if lr else ""
            name = f"{fs}_{arch_type}_{act}_L{L}_d{d}_bn{lr_tag}"
            configs.append((name, c, fs))

    return configs


def main():
    parser = argparse.ArgumentParser(description="V13: Multi-band LBP clean test")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for V12b")
    args = parser.parse_args()

    folds_to_run = args.folds if args.folds else list(range(N_FOLDS))

    # Wait for V12b to finish (30 runs)
    if not args.no_wait and os.path.exists(V12B_CSV):
        while True:
            try:
                df_check = pd.read_csv(V12B_CSV)
                if len(df_check) >= 30:
                    print(f"V12b complete ({len(df_check)} runs). Starting V13.",
                          flush=True)
                    break
            except Exception:
                pass
            print("Waiting for V12b to finish...", flush=True)
            time.sleep(30)
    elif not args.no_wait:
        print("V12b CSV not found, proceeding without waiting.", flush=True)

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

    X_all, feature_sets, full_cols = build_v13_data()

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    configs = build_v13_configs()
    total_runs = len(configs) * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V13 — Multi-Band LBP Clean Test", flush=True)
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
    if os.path.exists(V13_CSV):
        df_old = pd.read_csv(V13_CSV)
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
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)

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
                "feature_set": fs_name,
                "n_features": n_features,
                "arch": f"plain_L{cfg['n_layers']}_d{cfg['d_model']}",
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
            pd.DataFrame(results).to_csv(V13_CSV, index=False)

            print(f"  [{run_idx:3d}/{total_runs}] F{fold_id} {name:50s} "
                  f"R2={r2:.4f}  MAE={mae:.2f}pp  "
                  f"ep={n_epochs}  {elapsed:.0f}s  "
                  f"ETA={eta_h:.1f}h", flush=True)

            del net, trained_net, X_trn_t, X_val_t, X_test_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"V13 COMPLETE — {len(results)} runs", flush=True)
    print(f"{'='*70}", flush=True)

    df = pd.DataFrame(results)
    if len(df) > 0:
        summary = (df.groupby("name")
                   .agg(r2_mean=("r2_uniform", "mean"),
                        r2_std=("r2_uniform", "std"),
                        mae_mean=("mae_mean_pp", "mean"),
                        mae_std=("mae_mean_pp", "std"),
                        n_features=("n_features", "first"),
                        n_folds=("fold", "count"))
                   .sort_values("r2_mean", ascending=False))
        summary_csv = V13_CSV.replace(".csv", "_summary.csv")
        summary.to_csv(summary_csv)
        print(summary.to_string())
        print(f"\nSaved: {V13_CSV}")
        print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
