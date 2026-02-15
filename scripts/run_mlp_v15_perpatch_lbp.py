#!/usr/bin/env python3
"""
MLP V15: Per-patch LBP from Rust — test whether multi-band LBP helps
when computed the same way as V10's Python LBP.

Data source:  features_v4.parquet (Rust, per-patch LBP)

Feature sets:
  - bi_LBP_NIR:  V10-matched base (798) + NIR LBP (66) = 864
  - bi_LBP_all5: V10-matched base (798) + 5-band LBP (330) = 1128

Architecture: plain_silu_L5_d1024_bn (V10 champion)

Total: 2 configs × 5 folds = 10 runs
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
V15_CSV = os.path.join(OUT_DIR, "mlp_v15_perpatch_lbp.csv")


def build_v15_data():
    """Build feature sets from v4 parquet (Rust with per-patch LBP)."""
    feat_path = os.path.join(PROCESSED_V2_DIR, "features_v4.parquet")
    print(f"Loading Rust per-patch features: {feat_path}", flush=True)
    feat_df = pd.read_parquet(feat_path)

    from pandas.api.types import is_numeric_dtype
    all_cols = [c for c in feat_df.columns
                if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # Use V10's partition logic
    groups = partition_features(all_cols)
    bi_cols = [all_cols[i] for i in groups["bands_indices"]]

    # Remove indices V10 never had
    EXCLUDE_PREFIXES = {"MNDWI", "CRI1"}
    bi_cols = [c for c in bi_cols if c.split("_")[0] not in EXCLUDE_PREFIXES]

    # LBP columns
    lbp_nir = [c for c in all_cols if c.startswith("LBP_NIR_")]
    lbp_multi = [c for c in all_cols
                 if c.startswith("LBP_") and not c.startswith("LBP_NIR_")]

    print(f"  bands_indices (V10-matched): {len(bi_cols)} cols", flush=True)
    print(f"  LBP NIR:                     {len(lbp_nir)} cols", flush=True)
    print(f"  LBP multi:                   {len(lbp_multi)} cols", flush=True)

    bi_lbp_all5_cols = bi_cols + lbp_nir + lbp_multi
    X_all = feat_df[bi_lbp_all5_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df

    n_bi = len(bi_cols)
    n_nir = len(lbp_nir)
    n_multi = len(lbp_multi)

    feature_sets = {
        "bi_LBP_NIR":  list(range(n_bi + n_nir)),
        "bi_LBP_all5": list(range(n_bi + n_nir + n_multi)),
    }

    for name, idx in feature_sets.items():
        print(f"  Feature set '{name}': {len(idx)} features", flush=True)

    return X_all, feature_sets, bi_lbp_all5_cols


def main():
    parser = argparse.ArgumentParser(description="V15: Per-patch multi-band LBP")
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

    X_all, feature_sets, full_cols = build_v15_data()

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    # 2 configs: champion arch × 2 feature sets
    configs = []
    for fs in ["bi_LBP_NIR", "bi_LBP_all5"]:
        cfg = _cfg(0, fs, "plain", "silu", 5, 1024, "batchnorm")
        name = f"{fs}_plain_silu_L5_d1024_bn"
        configs.append((name, cfg, fs))

    total_runs = len(configs) * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V15 — Per-Patch Multi-Band LBP", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Data:     features_v4.parquet (Rust, per-patch LBP)", flush=True)
    print(f"  Configs:  {len(configs)}", flush=True)
    print(f"  Folds:    {folds_to_run}", flush=True)
    print(f"  Total:    {total_runs} runs", flush=True)
    print(f"  Device:   {device}", flush=True)
    for name, cfg, fs in configs:
        print(f"    {name:50s} {len(feature_sets[fs])}f", flush=True)
    print(f"{'='*70}\n", flush=True)

    results = []
    done_keys = set()
    if os.path.exists(V15_CSV):
        df_old = pd.read_csv(V15_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs done", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    times = []
    run_idx = 0

    for fold_id in folds_to_run:
        print(f"\n--- FOLD {fold_id} ---", flush=True)

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
            torch.manual_seed(SEED + fold_id)
            net = build_model(cfg, n_features, device)

            n_epochs, best_val, trained_net = train_model(
                net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-4),
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience_steps=args.patience_steps,
                min_steps=args.min_steps,
                mixup_alpha=0,
                use_swa=False,
                use_cosine=True,
            )

            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
            preds = _predict_batched(trained_net, X_test_t, device)
            elapsed = time.time() - t0

            metrics, _ = evaluate_model(y[test_idx], preds, CLASS_NAMES)
            r2 = metrics["r2_uniform"]
            mae = metrics["mae_mean_pp"]

            times.append(elapsed)
            remaining = total_runs - run_idx
            eta_h = (np.mean(times) * remaining) / 3600

            rec = {
                "name": name, "fold": fold_id,
                "feature_set": fs_name, "n_features": n_features,
                "r2_uniform": r2, "mae_mean_pp": mae,
                "best_val_loss": best_val, "n_epochs": n_epochs,
                "elapsed_s": round(elapsed, 1),
            }
            results.append(rec)
            done_keys.add((name, fold_id))
            pd.DataFrame(results).to_csv(V15_CSV, index=False)

            print(f"  [{run_idx:2d}/{total_runs}] F{fold_id} {name:50s} "
                  f"R2={r2:.4f}  MAE={mae:.2f}pp  ep={n_epochs}  "
                  f"{elapsed:.0f}s  ETA={eta_h:.1f}h", flush=True)

            del net, trained_net, X_trn_t, X_val_t, X_test_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"V15 RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    df = pd.DataFrame(results)
    if len(df) > 0:
        for fs in ["bi_LBP_NIR", "bi_LBP_all5"]:
            sub = df[df["feature_set"] == fs]
            if len(sub) > 0:
                r2_m, r2_s = sub["r2_uniform"].mean(), sub["r2_uniform"].std()
                mae_m = sub["mae_mean_pp"].mean()
                n_f = sub["n_features"].iloc[0]
                print(f"  {fs:20s} ({n_f:4d}f): R2={r2_m:.4f}+-{r2_s:.4f}  "
                      f"MAE={mae_m:.2f}pp  ({len(sub)} folds)", flush=True)
                for _, row in sub.sort_values("fold").iterrows():
                    print(f"    F{int(row['fold'])}: R2={row['r2_uniform']:.4f}  "
                          f"MAE={row['mae_mean_pp']:.2f}", flush=True)

        print(f"\n  V10 ref (Python LBP):  R2=0.787  (bi_LBP, 864f)")
        print(f"  V14 ref (reproduced):  R2=0.787  (bi_LBP, 864f)")

    print(f"\n  Saved: {V15_CSV}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
