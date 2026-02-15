#!/usr/bin/env python3
"""
Train & save the final MLP champion model (V10 config).

Produces:
    models/final_mlp/fold_{i}.pt          — model state dicts
    models/final_mlp/scaler_{i}.pkl       — StandardScaler per fold
    models/final_mlp/oof_predictions.parquet — OOF predictions
    models/final_mlp/meta.json            — model metadata + per-fold metrics
"""

import json
import os
import pickle
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

SEED = 42
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final_mlp")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_bi_lbp(full_feature_cols):
    groups = partition_features(full_feature_cols)
    base_idx = groups["bands_indices"]
    lbp_idx = [i for i, c in enumerate(full_feature_cols) if c.startswith("LBP_")]
    return sorted(set(base_idx) | set(lbp_idx))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    # ── Load data ──
    print("Loading data...")
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    fs_idx = build_bi_lbp(full_feature_cols)
    fs_cols = [full_feature_cols[i] for i in fs_idx]
    n_features = len(fs_idx)
    print("Features: {} (bi_LBP)".format(n_features))

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    cell_ids = feat_df["cell_id"].values if "cell_id" in feat_df.columns else np.arange(len(feat_df))
    del feat_df

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    # ── Champion config ──
    cfg = _cfg(0, "bi_LBP", "plain", "silu", 5, 1024, "batchnorm")

    # ── Train all 5 folds ──
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    oof_preds = np.full((len(y), len(CLASS_NAMES)), np.nan, dtype=np.float32)
    fold_metrics = []

    for fold_id in range(N_FOLDS):
        print("\n--- Fold {} ---".format(fold_id))
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )

        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx = train_idx[perm[:n_val]]
        trn_idx = train_idx[perm[n_val:]]
        print("  Train: {}, Val: {}, Test: {}".format(len(trn_idx), len(val_idx), len(test_idx)))

        X_fs = X_all[:, fs_idx]
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X_fs[trn_idx]).astype(np.float32)
        X_val_s = scaler.transform(X_fs[val_idx]).astype(np.float32)
        X_test_s = scaler.transform(X_fs[test_idx]).astype(np.float32)

        # Save scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler_{}.pkl".format(fold_id))
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        t0 = time.time()
        torch.manual_seed(SEED + fold_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold_id)
        net = build_model(cfg, n_features, device)

        n_epochs, best_val, trained_net = train_model(
            net, X_trn_t, y_trn_t, X_val_t, y_val_t,
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 1e-4),
            batch_size=2048,
            max_epochs=2000,
            patience_steps=5000,
            min_steps=2000,
            mixup_alpha=0,
            use_swa=False,
            use_cosine=True,
        )
        elapsed = time.time() - t0

        # Save model weights
        model_path = os.path.join(MODEL_DIR, "fold_{}.pt".format(fold_id))
        torch.save(trained_net.state_dict(), model_path)
        print("  Saved: {}".format(model_path))

        # OOF predictions on test set
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        preds = _predict_batched(trained_net, X_test_t, device)
        oof_preds[test_idx] = preds

        metrics, _ = evaluate_model(y[test_idx], preds, CLASS_NAMES)
        r2 = metrics["r2_uniform"]
        mae = metrics["mae_mean_pp"]
        fold_metrics.append({
            "fold": fold_id, "r2": r2, "mae": mae,
            "epochs": n_epochs, "val_loss": best_val, "time_s": round(elapsed, 1),
        })
        print("  R2={:.4f}  MAE={:.2f}pp  epochs={}  time={:.0f}s".format(
            r2, mae, n_epochs, elapsed))

        del net, trained_net, X_trn_t, X_val_t, X_test_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save OOF predictions ──
    oof_df = pd.DataFrame(oof_preds, columns=[c + "_pred" for c in CLASS_NAMES])
    oof_df.insert(0, "cell_id", cell_ids)
    oof_path = os.path.join(MODEL_DIR, "oof_predictions.parquet")
    oof_df.to_parquet(oof_path, index=False)
    print("\nOOF predictions saved: {}".format(oof_path))

    # ── Save metadata ──
    r2_mean = np.mean([m["r2"] for m in fold_metrics])
    r2_std = np.std([m["r2"] for m in fold_metrics])
    mae_mean = np.mean([m["mae"] for m in fold_metrics])
    meta_out = {
        "model": "PlainMLP",
        "config": "bi_LBP_plain_silu_L5_d1024_bn",
        "feature_set": "bi_LBP",
        "feature_cols": fs_cols,
        "n_features": n_features,
        "seed": SEED,
        "r2_mean": round(r2_mean, 4),
        "r2_std": round(r2_std, 4),
        "mae_mean_pp": round(mae_mean, 2),
        "fold_metrics": fold_metrics,
        "architecture": {
            "arch": "plain", "activation": "silu",
            "n_layers": 5, "d_model": 1024, "norm": "batchnorm",
            "dropout": 0.15, "head": "ILR_softmax",
        },
        "training": {
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 2048,
            "max_epochs": 2000, "patience_steps": 5000, "min_steps": 2000,
            "scheduler": "cosine_warmup_3ep",
        },
        "data_source": "features_merged_full.parquet",
    }
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL MLP CHAMPION — 5-fold CV")
    print("=" * 60)
    for m in fold_metrics:
        print("  Fold {}: R2={:.4f}  MAE={:.2f}pp  epochs={}".format(
            m["fold"], m["r2"], m["mae"], m["epochs"]))
    print("  " + "-" * 35)
    print("  Mean:   R2={:.4f} ± {:.4f}  MAE={:.2f}pp".format(r2_mean, r2_std, mae_mean))
    print("=" * 60)
    print("\nAll saved to: {}".format(MODEL_DIR))


if __name__ == "__main__":
    main()
