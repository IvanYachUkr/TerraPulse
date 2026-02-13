#!/usr/bin/env python3
"""
MLP V5: Deep Training — let every config train until convergence.

Purpose: Give carefully selected configs unlimited training time
(2000 epoch cap, 5000-step patience) across 5 folds to find the
true ceiling of MLP performance on this task.

Configs selected based on V4 search + CV insights:
  - Tier 1: Proven CV winners (still improving at 300 epochs)
  - Tier 2: Deep architectures (need more epochs to converge)
  - Tier 3: GeGLU recovery (avoid BatchNorm to prevent NaN)
  - Tier 4: Wider models (d1024, never tested)

Feature sets: only the top 3 competitive sets.

Output:
    reports/phase8/tables/mlp_v5_deep.csv           — per-fold results
    reports/phase8/tables/mlp_v5_deep_summary.csv   — mean ± std

Usage:
    .venv\\Scripts\\python.exe scripts/run_mlp_v5_deep_train.py
    .venv\\Scripts\\python.exe scripts/run_mlp_v5_deep_train.py --max-epochs 3000
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# ── Reuse V4 infrastructure ──────────────────────────────────────────
# Add scripts dir to path so we can import from V4
sys.path.insert(0, os.path.dirname(__file__))

import json as _json
from sklearn.preprocessing import StandardScaler

from run_mlp_overnight_v4 import (
    # Constants
    PROJECT_ROOT, CLASS_NAMES, N_FOLDS, CONTROL_COLS, SEED,
    # Feature sets
    partition_features,
    # Model building
    build_model, make_name, _cfg,
    # Training
    train_model, normalize_targets, _predict_batched,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

# V5 output paths
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V5_CSV = os.path.join(OUT_DIR, "mlp_v5_deep.csv")
V5_SUMMARY = os.path.join(OUT_DIR, "mlp_v5_deep_summary.csv")


# =====================================================================
# V5 Config Selection — hand-picked for deep training
# =====================================================================

def generate_v5_configs():
    """Curated configs for unlimited training."""
    configs = []
    rid = 0

    # Only the 3 competitive feature sets
    feature_sets = ["bands_indices", "bands_indices_glcm_lbp", "full_no_deltas"]

    for fs in feature_sets:

        # ── Tier 1: Proven CV winners (undertrained at 300 epochs) ──
        # #1 overall: glcm_lbp + plain mish
        configs.append(_cfg(rid, fs, "plain", "mish", 5, 512, "batchnorm")); rid += 1
        # #2 overall: residual silu deep
        configs.append(_cfg(rid, fs, "residual", "silu", 10, 256, "none")); rid += 1
        # #3: plain silu wide (stable across folds)
        configs.append(_cfg(rid, fs, "plain", "silu", 5, 512, "none")); rid += 1
        # #4: plain silu + BN
        configs.append(_cfg(rid, fs, "plain", "silu", 5, 512, "batchnorm")); rid += 1
        # Stable: residual silu wide + low LR
        configs.append(_cfg(rid, fs, "residual", "silu", 6, 512, "batchnorm", lr=5e-4)); rid += 1

        # ── Tier 2: Deep architectures (may need 500+ epochs) ──
        # Deep residual — 12 blocks
        configs.append(_cfg(rid, fs, "residual", "gelu", 12, 256, "batchnorm")); rid += 1
        configs.append(_cfg(rid, fs, "residual", "silu", 12, 256, "none")); rid += 1
        configs.append(_cfg(rid, fs, "residual", "mish", 12, 256, "batchnorm")); rid += 1
        # Very deep — 16 blocks (new, never tested!)
        configs.append(_cfg(rid, fs, "residual", "gelu", 16, 256, "batchnorm", lr=5e-4)); rid += 1
        configs.append(_cfg(rid, fs, "residual", "silu", 16, 256, "none", lr=5e-4)); rid += 1

        # ── Tier 3: GeGLU recovery (NO batchnorm to prevent NaN) ──
        # GeGLU plain
        configs.append(_cfg(rid, fs, "plain", "geglu", 5, 256, "none")); rid += 1
        configs.append(_cfg(rid, fs, "plain", "geglu", 5, 512, "none")); rid += 1
        # GeGLU residual
        configs.append(_cfg(rid, fs, "residual", "geglu", 6, 256, "none")); rid += 1
        configs.append(_cfg(rid, fs, "residual", "geglu", 6, 512, "none")); rid += 1
        configs.append(_cfg(rid, fs, "residual", "geglu", 4, 256, "none")); rid += 1
        # GeGLU deep
        configs.append(_cfg(rid, fs, "residual", "geglu", 10, 256, "none")); rid += 1
        # GeGLU with layernorm (should be safe, not batchnorm)
        configs.append(_cfg(rid, fs, "residual", "geglu", 6, 256, "layernorm")); rid += 1

        # ── Tier 4: Wider models (d1024, never tested) ──
        configs.append(_cfg(rid, fs, "plain", "silu", 5, 1024, "batchnorm")); rid += 1
        configs.append(_cfg(rid, fs, "residual", "gelu", 6, 1024, "none", lr=5e-4)); rid += 1
        configs.append(_cfg(rid, fs, "residual", "silu", 6, 1024, "batchnorm", lr=5e-4)); rid += 1

    return configs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MLP V5: Deep Training")
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

    configs = generate_v5_configs()
    total_configs = len(configs)
    total_runs = total_configs * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V5 — DEEP TRAINING", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Configs:        {total_configs}", flush=True)
    print(f"  Folds:          {folds_to_run}", flush=True)
    print(f"  Total runs:     {total_runs}", flush=True)
    print(f"  Max epochs:     {args.max_epochs}", flush=True)
    print(f"  Patience:       {args.patience_steps} steps", flush=True)
    print(f"  Min steps:      {args.min_steps}", flush=True)
    print(f"  Output CSV:     {V5_CSV}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Show all configs
    print("Configs:", flush=True)
    for c in configs:
        print(f"  {make_name(c)}", flush=True)
    print(flush=True)

    # ── Load data (matching V4 exactly) ──
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

    print("Loading data...", flush=True)
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))

    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    y = labels_df[CLASS_NAMES].values.astype(np.float32)
    del feat_df

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    feat_groups = partition_features(full_feature_cols)

    print(f"Data loaded: X={X_all.shape}, y={y.shape}", flush=True)

    # ── Resume logic ──
    results = []
    done_keys = set()
    if os.path.exists(V5_CSV):
        df_old = pd.read_csv(V5_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs already done", flush=True)

    # ── Cross-validation loop ──
    run_idx = 0
    n_skipped = 0
    times = []

    for fold_id in folds_to_run:
        print(f"\n{'='*80}", flush=True)
        print(f"FOLD {fold_id}/{N_FOLDS-1}", flush=True)
        print(f"{'='*80}", flush=True)

        # Get train/test indices with buffer (matching V4)
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
        for feat_set in ["bands_indices", "bands_indices_glcm_lbp", "full_no_deltas"]:
            feat_idx = feat_groups.get(feat_set)
            if feat_idx is None or len(feat_idx) == 0:
                continue
            X = X_all[:, feat_idx]
            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X[val_idx]).astype(np.float32)
            X_test_s = scaler.transform(X[test_idx]).astype(np.float32)
            scaled_cache[feat_set] = (
                torch.tensor(X_trn_s, dtype=torch.float32).to(device),
                torch.tensor(X_val_s, dtype=torch.float32).to(device),
                torch.tensor(X_test_s, dtype=torch.float32),  # CPU for batched predict
                len(feat_idx),
            )
            print(f"    {feat_set}: {len(feat_idx)} features", flush=True)

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

                # Evaluate on TEST set (matching V4)
                y_pred = _predict_batched(final_model, X_test_t, device)
                elapsed = time.time() - t0

                del net, final_model

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
                summary.update({
                    "name": name, "fold": fold_id, "stage": "v5_deep",
                    "feature_set": fs, "arch": cfg["arch"],
                    "activation": cfg["activation"],
                    "norm_type": cfg.get("norm_type", "layernorm"),
                    "n_layers": cfg["n_layers"], "d_model": cfg["d_model"],
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
                      f"{elapsed:.0f}s  ETA={eta_h:.1f}h", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                results.append({"name": name, "fold": fold_id,
                                "stage": "v5_deep", "error": str(e)})
                done_keys.add((name, fold_id))
                print(f"  [{n_skipped+len(times):4d}/{total_runs}] F{fold_id} {name:60s} "
                      f"ERROR: {e} ({elapsed:.0f}s)", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()

            # Save periodically
            if (n_skipped + len(times)) % 5 == 0:
                pd.DataFrame(results).to_csv(V5_CSV, index=False)

        # End-of-fold save and cleanup
        pd.DataFrame(results).to_csv(V5_CSV, index=False)
        del scaled_cache, y_trn_t, y_val_t
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final save
    pd.DataFrame(results).to_csv(V5_CSV, index=False)
    print(f"\nResults saved to {V5_CSV}", flush=True)

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
               )
               .sort_values("r2_mean", ascending=False))
        agg.to_csv(V5_SUMMARY)
        print(f"Summary saved to {V5_SUMMARY}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"V5 DEEP TRAINING RESULTS", flush=True)
        print(f"{'='*70}", flush=True)
        for _, row in agg.iterrows():
            folds_str = f"folds={int(row['folds'])}"
            print(f"  {row.name:60s} R2={row['r2_mean']:.4f}+/-{row['r2_std']:.4f}  "
                  f"MAE={row['mae_mean']:.2f}  ep={row['epochs_mean']:.0f}  {folds_str}", flush=True)

        # Best per feature set
        print(f"\nBEST PER FEATURE SET:", flush=True)
        for fs in sorted(valid["feature_set"].unique()):
            fs_agg = agg[agg["feature_set"] == fs]
            if len(fs_agg) > 0:
                best = fs_agg.iloc[0]
                print(f"  {fs:30s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  {best.name}", flush=True)

        # Best per activation
        print(f"\nBEST PER ACTIVATION:", flush=True)
        for act in sorted(valid["activation"].unique()):
            act_agg = agg[agg["activation"] == act]
            if len(act_agg) > 0:
                best = act_agg.iloc[0]
                print(f"  {act:8s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  {best.name}", flush=True)

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
