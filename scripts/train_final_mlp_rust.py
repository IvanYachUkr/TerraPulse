#!/usr/bin/env python3
"""
Train final MLP using the baseline's 864 feature columns,
but sourced from the Rust parquet (features_v3_rust.parquet).

Steps:
  1. Load 864 column names from models/final_mlp/meta.json
  2. Load features_v3_rust.parquet, rename LBP_NIR_* -> LBP_*
  3. Impute NaN with column medians
  4. Train 5-fold MLP with identical seeds/arch
  5. Compare to baseline R²=0.7872
"""

import json, os, pickle, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from run_mlp_overnight_v4 import (
    PROJECT_ROOT, CLASS_NAMES, N_FOLDS,
    build_model, _cfg, train_model, normalize_targets, _predict_batched,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

SEED = 42
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final_mlp_rust")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Get the exact 864 columns from the baseline
    with open(os.path.join(PROJECT_ROOT, "models", "final_mlp", "meta.json")) as f:
        baseline = json.load(f)
    target_cols = baseline["feature_cols"]
    print(f"Baseline features: {len(target_cols)}")

    # 2. Load Rust parquet and rename LBP_NIR_* -> LBP_*
    rust_pq = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    print(f"Loading: {rust_pq}")
    df = pd.read_parquet(rust_pq)
    rename = {c: c.replace("LBP_NIR_", "LBP_") for c in df.columns if "LBP_NIR_" in c}
    df = df.rename(columns=rename)
    print(f"  Renamed {len(rename)} LBP_NIR columns")

    # Check coverage
    missing = [c for c in target_cols if c not in df.columns]
    print(f"  Missing: {len(missing)}")
    if missing:
        print(f"    Examples: {missing[:10]}")
    use_cols = [c for c in target_cols if c in df.columns]
    print(f"  Using: {len(use_cols)} / {len(target_cols)} features")

    # 3. Impute NaN with column medians
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_total = 0
    for c in use_cols:
        n = df[c].isna().sum()
        if n > 0:
            nan_total += n
            med = df[c].median()
            df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)
    print(f"  Imputed {nan_total} NaN values")

    # Quick comparison with merged parquet values
    py_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    X_rs = df[use_cols].values.astype(np.float32)
    X_py = py_df[use_cols].values.astype(np.float32)
    np.nan_to_num(X_rs, copy=False)
    np.nan_to_num(X_py, copy=False)
    adiff = np.abs(X_rs - X_py)
    print(f"\n  Rust vs merged_full comparison ({len(use_cols)} cols):")
    print(f"    Max abs diff:  {adiff.max():.8f}")
    print(f"    Mean abs diff: {adiff.mean():.8f}")
    print(f"    Identical cells: {(adiff.max(axis=1) == 0).sum()} / {len(X_rs)}")
    del py_df, X_py, adiff

    cell_ids = df["cell_id"].values if "cell_id" in df.columns else np.arange(len(df))
    X_all = X_rs
    n_features = X_all.shape[1]
    del df

    # Load labels & splits
    y = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))[CLASS_NAMES].values.astype(np.float32)
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    # 4. Train
    cfg = _cfg(0, "bi_LBP", "plain", "silu", 5, 1024, "batchnorm")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except: pass

    oof_preds = np.full((len(y), len(CLASS_NAMES)), np.nan, dtype=np.float32)
    fold_metrics = []

    for fold_id in range(N_FOLDS):
        print(f"\n--- Fold {fold_id} ---")
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"], buffer_tiles=1)
        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx, trn_idx = train_idx[perm[:n_val]], train_idx[perm[n_val:]]

        scaler = StandardScaler()
        X_trn = scaler.fit_transform(X_all[trn_idx]).astype(np.float32)
        X_val = scaler.transform(X_all[val_idx]).astype(np.float32)
        X_tst = scaler.transform(X_all[test_idx]).astype(np.float32)

        with open(os.path.join(MODEL_DIR, f"scaler_{fold_id}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        X_trn_t = torch.tensor(X_trn).to(device)
        X_val_t = torch.tensor(X_val).to(device)
        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        t0 = time.time()
        torch.manual_seed(SEED + fold_id)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_id)
        net = build_model(cfg, n_features, device)

        n_epochs, best_val, trained_net = train_model(
            net, X_trn_t, y_trn_t, X_val_t, y_val_t,
            lr=cfg["lr"], weight_decay=1e-3,
            batch_size=4096, max_epochs=2000, patience_steps=5000, min_steps=2000,
            mixup_alpha=0, use_swa=False, use_cosine=True)
        elapsed = time.time() - t0

        torch.save(trained_net.state_dict(), os.path.join(MODEL_DIR, f"fold_{fold_id}.pt"))

        preds = _predict_batched(trained_net, torch.tensor(X_tst), device)
        oof_preds[test_idx] = preds

        metrics, _ = evaluate_model(y[test_idx], preds, CLASS_NAMES)
        r2, mae = metrics["r2_uniform"], metrics["mae_mean_pp"]
        fold_metrics.append({"fold": fold_id, "r2": r2, "mae": mae,
                             "epochs": n_epochs, "val_loss": best_val, "time_s": round(elapsed, 1)})
        print(f"  R2={r2:.4f}  MAE={mae:.2f}pp  epochs={n_epochs}  time={elapsed:.0f}s")

        del net, trained_net, X_trn_t, X_val_t
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 5. Compare with baseline
    print("\n" + "=" * 60)
    print("Rust (864 baseline cols) vs Python baseline")
    print("=" * 60)
    print(f"{'Fold':<8} {'Rust R2':>10} {'Py R2':>10} {'Diff':>10}")
    print("-" * 38)
    for fm in fold_metrics:
        py_fold = [m for m in baseline["fold_metrics"] if m["fold"] == fm["fold"]][0]
        print(f"Fold {fm['fold']:<3} {fm['r2']:>10.4f} {py_fold['r2']:>10.4f} {fm['r2']-py_fold['r2']:>+10.4f}")

    rs_r2 = np.mean([m["r2"] for m in fold_metrics])
    py_r2 = baseline["r2_mean"]
    print("-" * 38)
    print(f"{'Mean':<8} {rs_r2:>10.4f} {py_r2:>10.4f} {rs_r2-py_r2:>+10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
