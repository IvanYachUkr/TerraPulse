#!/usr/bin/env python3
"""
MLP V17: Multi-seed replication of V10 champion result.

Goal: prove that R²≈0.787 is achievable by sampling multiple random seeds.

Config (identical to V10 champion):
    bi_LBP (864 feat) × plain_silu_L5_d1024_bn × ILR head
    Data: features_merged_full.parquet (Python-extracted)

Seeds: 5 different seeds × 5 CV folds = 25 runs.
ETA:   ~2h on CUDA, ~16h on CPU.

Output:
    reports/phase8/tables/mlp_v17_multiseed.csv
    reports/phase8/tables/mlp_v17_multiseed_summary.csv
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
    PROJECT_ROOT, CLASS_NAMES, N_FOLDS, CONTROL_COLS,
    build_model, _cfg,
    train_model, normalize_targets, _predict_batched,
    partition_features,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V17_CSV = os.path.join(OUT_DIR, "mlp_v17_multiseed.csv")
V17_SUMMARY = os.path.join(OUT_DIR, "mlp_v17_multiseed_summary.csv")

# Seeds to test — seed 42 is the original V10 seed
SEEDS = [42, 123, 456, 789, 2024]


def build_bi_lbp(full_feature_cols):
    """Exact V10 bi_LBP feature set: bands_indices + LBP_ columns."""
    groups = partition_features(full_feature_cols)
    base_idx = groups["bands_indices"]
    lbp_idx = [i for i, c in enumerate(full_feature_cols) if c.startswith("LBP_")]
    return sorted(set(base_idx) | set(lbp_idx))


def main():
    parser = argparse.ArgumentParser(description="V17: Multi-seed V10 replication")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds (default: 42 123 456 789 2024)")
    args = parser.parse_args()

    folds_to_run = args.folds if args.folds else list(range(N_FOLDS))
    seeds = args.seeds if args.seeds else SEEDS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data (once, shared across all seeds)
    print("Loading data...", flush=True)
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    fs_idx = build_bi_lbp(full_feature_cols)
    n_features = len(fs_idx)

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    # Champion config (identical to V10)
    cfg = _cfg(0, "bi_LBP", "plain", "silu", 5, 1024, "batchnorm")
    name = "bi_LBP_plain_silu_L5_d1024_bn"

    total_runs = len(seeds) * len(folds_to_run)

    print("\n" + "=" * 70, flush=True)
    print("MLP V17 — Multi-Seed Replication of V10 Champion", flush=True)
    print("=" * 70, flush=True)
    print("  Config:     {}".format(name), flush=True)
    print("  Features:   {} (bi_LBP)".format(n_features), flush=True)
    print("  Seeds:      {}".format(seeds), flush=True)
    print("  Folds:      {}".format(folds_to_run), flush=True)
    print("  Total runs: {}".format(total_runs), flush=True)
    print("  Device:     {}".format(device), flush=True)
    print("=" * 70 + "\n", flush=True)

    # Resume support
    results = []
    done_keys = set()
    if os.path.exists(V17_CSV):
        df_old = pd.read_csv(V17_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["seed"].astype(int), df_old["fold"].astype(int)))
        print("Resuming: {} runs already done".format(len(results)), flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    times = []
    run_count = 0

    for seed in seeds:
        print("\n" + "=" * 70, flush=True)
        print("SEED = {}".format(seed), flush=True)
        print("=" * 70, flush=True)

        # Global seed setup (matches V10 exactly)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        for fold_id in folds_to_run:
            run_count += 1
            if (seed, fold_id) in done_keys:
                print("  S{} F{} already done, skipping".format(seed, fold_id), flush=True)
                continue

            print("\n  --- SEED {} / FOLD {} ---".format(seed, fold_id), flush=True)

            train_idx, test_idx = get_fold_indices(
                tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
                buffer_tiles=1,
            )

            # Val split (deterministic per seed+fold)
            rng = np.random.RandomState(seed + fold_id)
            perm = rng.permutation(len(train_idx))
            n_val = max(int(len(train_idx) * 0.15), 100)
            val_idx = train_idx[perm[:n_val]]
            trn_idx = train_idx[perm[n_val:]]
            print("  Train: {}, Val: {}, Test: {}".format(
                len(trn_idx), len(val_idx), len(test_idx)), flush=True)

            X_fs = X_all[:, fs_idx]
            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X_fs[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X_fs[val_idx]).astype(np.float32)
            X_test_s = scaler.transform(X_fs[test_idx]).astype(np.float32)

            X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)

            y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
            y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

            t0 = time.time()
            torch.manual_seed(seed + fold_id)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + fold_id)
            net = build_model(cfg, n_features, device)

            n_epochs, best_val, trained_net = train_model(
                net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                lr=cfg["lr"],
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

            rec = {
                "seed": seed, "name": name, "fold": fold_id,
                "feature_set": "bi_LBP", "n_features": n_features,
                "r2_uniform": r2, "mae_mean_pp": mae,
                "best_val_loss": best_val, "n_epochs": n_epochs,
                "elapsed_s": round(elapsed, 1),
                "n_train": len(trn_idx), "n_test": len(test_idx),
            }
            results.append(rec)
            done_keys.add((seed, fold_id))
            times.append(elapsed)

            avg_t = sum(times) / len(times)
            remaining = max(0, total_runs - len(done_keys))
            eta_min = remaining * avg_t / 60
            print("  R2={:.4f}  MAE={:.2f}pp  epochs={}  time={:.0f}s  ETA={:.0f}min".format(
                r2, mae, n_epochs, elapsed, eta_min), flush=True)

            # Save after each fold
            pd.DataFrame(results).to_csv(V17_CSV, index=False)

            del net, trained_net, X_trn_t, X_val_t, X_test_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Final summary ──
    df = pd.DataFrame(results)
    df.to_csv(V17_CSV, index=False)

    print("\n" + "=" * 70, flush=True)
    print("V17 MULTI-SEED RESULTS", flush=True)
    print("=" * 70, flush=True)

    # Per-seed summary
    seed_summary = []
    for s in seeds:
        mask = df["seed"] == s
        if mask.sum() == 0:
            continue
        r2_vals = df.loc[mask, "r2_uniform"]
        mae_vals = df.loc[mask, "mae_mean_pp"]
        r2m = r2_vals.mean()
        r2s = r2_vals.std()
        maem = mae_vals.mean()
        seed_summary.append({
            "seed": s,
            "r2_mean": r2m, "r2_std": r2s,
            "mae_mean": maem,
            "n_folds": mask.sum(),
        })
        fold_str = "  ".join(
            "F{}={:.3f}".format(int(r.fold), r.r2_uniform)
            for _, r in df[mask].sort_values("fold").iterrows()
        )
        marker = " <-- BEST" if r2m == max(d["r2_mean"] for d in seed_summary) else ""
        print("  Seed {:5d}:  R2 = {:.4f} +/- {:.4f}  MAE = {:.2f}pp  [{}]{}".format(
            s, r2m, r2s, maem, fold_str, marker), flush=True)

    best_seed_rec = max(seed_summary, key=lambda d: d["r2_mean"])
    print("\n  BEST SEED: {}  R2 = {:.4f}".format(
        best_seed_rec["seed"], best_seed_rec["r2_mean"]), flush=True)
    print("  V10 REF:   42  R2 = 0.7872", flush=True)
    print("  Delta:     {:+.4f}".format(best_seed_rec["r2_mean"] - 0.7872), flush=True)

    pd.DataFrame(seed_summary).to_csv(V17_SUMMARY, index=False)
    print("\n  Results: {}".format(V17_CSV), flush=True)
    print("  Summary: {}".format(V17_SUMMARY), flush=True)
    print("=" * 70, flush=True)

    if times:
        print("\nTotal compute: {:.1f}min".format(sum(times) / 60), flush=True)


if __name__ == "__main__":
    main()
