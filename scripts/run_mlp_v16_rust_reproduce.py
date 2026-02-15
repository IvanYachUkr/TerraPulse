#!/usr/bin/env python3
"""
MLP V16: Reproduce V10 champion using Rust-extracted features.

Single config: bi_LBP_NIR (864 feat) × plain_silu_L5_d1024_bn × ILR head.
Data source:   features_v4.parquet (Rust per-patch, with TC/block-reduce/LBP fixes).

Expected R² ≈ 0.787 ± 0.038 (V10 reference).
Comparison:    V14 used Python features → R² ≈ 0.787 ± 0.034 (V14 result).
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
V16_CSV = os.path.join(OUT_DIR, "mlp_v16_rust_reproduce.csv")


def build_v10_matched_features(all_cols):
    """Build V10-equivalent feature set from Rust V4 features.

    V10 bi_LBP = bands_indices (798) + LBP NIR (66) = 864.
    Rust V4 has extra indices (CRI1, MNDWI, EVI2, GNDVI, NDTI) and spatial
    features that V10 never had. We exclude those to match V10 exactly.
    """
    groups = partition_features(all_cols)
    bi_idx = groups["bands_indices"]
    bi_cols = [all_cols[i] for i in bi_idx]

    # Remove indices V10 never had
    EXCLUDE_PREFIXES = {"MNDWI", "CRI1"}
    bi_cols_filtered = [c for c in bi_cols if c.split("_")[0] not in EXCLUDE_PREFIXES]

    # LBP NIR only — now named LBP_u8_* and LBP_entropy_* (after rename)
    # Exclude multi-band LBP (LBP_NDVI_, LBP_EVI2_, LBP_SWIR1_, LBP_NDTI_)
    MULTI_LBP_PREFIXES = ("LBP_NDVI_", "LBP_EVI2_", "LBP_SWIR1_", "LBP_NDTI_")
    lbp_nir_cols = [c for c in all_cols
                    if c.startswith("LBP_")
                    and not any(c.startswith(p) for p in MULTI_LBP_PREFIXES)]

    selected = bi_cols_filtered + lbp_nir_cols
    selected_idx = [all_cols.index(c) for c in selected]
    return sorted(selected_idx), selected


def main():
    parser = argparse.ArgumentParser(description="V16: Reproduce V10 with Rust features")
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

    # Load RUST features
    print("Loading Rust features (features_v4.parquet)...", flush=True)
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_v4.parquet"))
    from pandas.api.types import is_numeric_dtype
    all_feature_cols = [c for c in feat_df.columns
                        if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    fs_idx, fs_cols = build_v10_matched_features(all_feature_cols)
    n_features = len(fs_idx)

    X_all = feat_df[all_feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    del feat_df

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    # Champion config — exact V10/V14
    cfg = _cfg(0, "bi_LBP_NIR", "plain", "silu", 5, 1024, "batchnorm")
    name = "bi_LBP_NIR_plain_silu_L5_d1024_bn"

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V16 — Reproduce V10 Champion with Rust Features", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Config:     {name}", flush=True)
    print(f"  Features:   {n_features} (bi_LBP_NIR from Rust V4)", flush=True)
    print(f"  Folds:      {folds_to_run}", flush=True)
    print(f"  Device:     {device}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Resume
    results = []
    done_keys = set()
    if os.path.exists(V16_CSV):
        df_old = pd.read_csv(V16_CSV)
        results = df_old.to_dict("records")
        done_keys = set(df_old["fold"].astype(int))
        print(f"Resuming: {len(results)} folds already done", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_r2 = []
    all_mae = []

    for fold_id in folds_to_run:
        if fold_id in done_keys:
            old = [r for r in results if r["fold"] == fold_id]
            if old:
                all_r2.append(old[0]["r2_uniform"])
                all_mae.append(old[0]["mae_mean_pp"])
            print(f"  F{fold_id} already done, skipping", flush=True)
            continue

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
        torch.manual_seed(SEED + fold_id)
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
        all_r2.append(r2)
        all_mae.append(mae)

        rec = {
            "name": name, "fold": fold_id,
            "feature_set": "bi_LBP_NIR", "n_features": n_features,
            "data_source": "rust_v4",
            "r2_uniform": r2, "mae_mean_pp": mae,
            "best_val_loss": best_val, "n_epochs": n_epochs,
            "elapsed_s": round(elapsed, 1),
            "n_train": len(trn_idx), "n_test": len(test_idx),
        }
        results.append(rec)
        done_keys.add(fold_id)
        pd.DataFrame(results).to_csv(V16_CSV, index=False)

        print(f"  R2={r2:.4f}  MAE={mae:.2f}pp  epochs={n_epochs}  "
              f"time={elapsed:.0f}s", flush=True)

        del net, trained_net, X_trn_t, X_val_t, X_test_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print(f"V16 RESULTS — V10 Reproduction with Rust Features", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Config:     {name}", flush=True)
    print(f"  Features:   {n_features} (bi_LBP_NIR)", flush=True)
    print(f"  Data:       features_v4.parquet (Rust per-patch)", flush=True)

    if all_r2:
        r2_mean = np.mean(all_r2)
        r2_std = np.std(all_r2)
        mae_mean = np.mean(all_mae)
        print(f"\n  R² mean:    {r2_mean:.4f} ± {r2_std:.4f}", flush=True)
        print(f"  MAE mean:   {mae_mean:.2f} pp", flush=True)
        print(f"\n  V10 ref:    R² = 0.787 ± 0.038", flush=True)
        print(f"  V14 ref:    R² = 0.787 ± 0.034 (Python features)", flush=True)
        print(f"  V16 delta:  {r2_mean - 0.787:+.4f}", flush=True)

        for i, (r, m) in enumerate(zip(all_r2, all_mae)):
            print(f"    F{i}: R2={r:.4f}  MAE={m:.2f}", flush=True)

    print(f"\n  Saved: {V16_CSV}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
