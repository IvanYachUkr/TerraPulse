#!/usr/bin/env python3
"""
MLP V11: New texture features (Gabor v2 + Morph DMP) + SiLU Dirichlet.

Waits for V10 to finish and texture_v2 extraction to complete before starting.

Tests 7 configs × 5 folds = 35 runs:
  Feature sets:
    - bi_Gab2_DMP          (bands+indices + new Gabor v2 + Morph DMP)
    - bi_LBP_Gab2_DMP      (above + LBP)
  Architectures:
    - plain silu L5 d1024 bn    (V10 champion, baseline)
    - plain silu L5 d1536 bn    (wider for more features)
    - plain silu L7 d1024 bn    (deeper)
  Heads:
    - ILR (softmax + soft_cross_entropy)
    - Dirichlet (softplus+1, NLL)

Output: reports/phase8/tables/mlp_v11_new_texture.csv
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
    partition_features, dirichlet_nll,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V11_CSV = os.path.join(OUT_DIR, "mlp_v11_new_texture.csv")
V10_CSV = os.path.join(OUT_DIR, "mlp_v10_definitive.csv")
TEXTURE_V2_PATH = os.path.join(PROCESSED_V2_DIR, "features_texture_v2.parquet")


# =====================================================================
# Wait for dependencies
# =====================================================================

def wait_for_v10(poll_seconds=60):
    """Wait for V10 to finish (all 75 runs in CSV)."""
    if not os.path.exists(V10_CSV):
        print("V10 CSV not found, waiting...", flush=True)
    while True:
        if os.path.exists(V10_CSV):
            try:
                df = pd.read_csv(V10_CSV)
                n = len(df)
                if n >= 75:
                    print(f"V10 complete ({n} runs). Proceeding.", flush=True)
                    return
                else:
                    print(f"V10: {n}/75 runs done. Waiting {poll_seconds}s...",
                          flush=True)
            except Exception:
                pass
        time.sleep(poll_seconds)


def wait_for_texture_v2(poll_seconds=30):
    """Wait for texture v2 parquet to exist."""
    while not os.path.exists(TEXTURE_V2_PATH):
        print(f"Waiting for {TEXTURE_V2_PATH}...", flush=True)
        time.sleep(poll_seconds)
    # Wait a bit more to ensure file is fully written
    time.sleep(5)
    print(f"Texture v2 parquet found: {TEXTURE_V2_PATH}", flush=True)


# =====================================================================
# Feature sets: base + LBP from merged, new Gabor/Morph from texture_v2
# =====================================================================

def build_v11_data(device):
    """Load and merge features from merged_full + texture_v2.

    Returns: X_all (numpy), feature_sets dict, full_cols list
    """
    print("Loading base features...", flush=True)
    feat_df = pd.read_parquet(
        os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))

    from pandas.api.types import is_numeric_dtype
    old_feature_cols = [c for c in feat_df.columns
                        if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # Identify base (bands+indices) and LBP column indices
    groups = partition_features(old_feature_cols)
    base_cols = [old_feature_cols[i] for i in groups["bands_indices"]]
    lbp_cols = [c for c in old_feature_cols if c.startswith("LBP_")]

    print(f"  Base (bands+indices): {len(base_cols)} cols", flush=True)
    print(f"  LBP: {len(lbp_cols)} cols", flush=True)

    # Load texture v2
    print("Loading texture v2 features...", flush=True)
    tex_df = pd.read_parquet(TEXTURE_V2_PATH)
    tex_cols = [c for c in tex_df.columns if c != "cell_id"]
    print(f"  Texture v2: {len(tex_cols)} cols", flush=True)

    # Merge on cell_id
    merged = feat_df[["cell_id"] + base_cols + lbp_cols].merge(
        tex_df, on="cell_id", how="inner")
    assert len(merged) == len(feat_df), \
        f"Merge lost rows: {len(merged)} vs {len(feat_df)}"

    # Build combined feature columns
    all_feature_cols = base_cols + lbp_cols + tex_cols
    X_all = merged[all_feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)

    del feat_df, tex_df, merged

    # Build feature set index maps
    n_base = len(base_cols)
    n_lbp = len(lbp_cols)
    n_tex = len(tex_cols)

    base_idx = list(range(n_base))
    lbp_idx = list(range(n_base, n_base + n_lbp))
    tex_idx = list(range(n_base + n_lbp, n_base + n_lbp + n_tex))

    feature_sets = {
        "bi_Gab2_DMP": sorted(set(base_idx) | set(tex_idx)),
        "bi_LBP_Gab2_DMP": sorted(set(base_idx) | set(lbp_idx) | set(tex_idx)),
    }

    for name, idx in feature_sets.items():
        print(f"  Feature set '{name}': {len(idx)} features", flush=True)

    return X_all, feature_sets, all_feature_cols


# =====================================================================
# V11 configs: 7 total
# =====================================================================

def build_v11_configs():
    """Build the 7 V11 configs."""
    configs = []

    # ── bi_Gab2_DMP: 5 configs ──

    # 1. Core test: same arch as V10 champion, new features, ILR
    c1 = _cfg(0, "bi_Gab2_DMP", "plain", "silu", 5, 1024, "batchnorm")
    configs.append(("bi_Gab2_DMP_plain_silu_L5_d1024_bn", c1, "bi_Gab2_DMP"))

    # 2. Dirichlet head: same arch, uncertainty
    c2 = _cfg(0, "bi_Gab2_DMP", "plain", "silu", 5, 1024, "batchnorm",
              head_type="dirichlet")
    c2["head_type"] = "dirichlet"
    configs.append(("bi_Gab2_DMP_plain_silu_L5_d1024_bn_dirichlet", c2, "bi_Gab2_DMP"))

    # 3. Wider: d1536 for 57% more features
    c3 = _cfg(0, "bi_Gab2_DMP", "plain", "silu", 5, 1536, "batchnorm")
    configs.append(("bi_Gab2_DMP_plain_silu_L5_d1536_bn", c3, "bi_Gab2_DMP"))

    # 4. Deeper: L7 d1024
    c4 = _cfg(0, "bi_Gab2_DMP", "plain", "silu", 7, 1024, "batchnorm")
    configs.append(("bi_Gab2_DMP_plain_silu_L7_d1024_bn", c4, "bi_Gab2_DMP"))

    # 5. Optional: wider + Dirichlet
    c5 = _cfg(0, "bi_Gab2_DMP", "plain", "silu", 5, 1536, "batchnorm",
              head_type="dirichlet")
    c5["head_type"] = "dirichlet"
    configs.append(("bi_Gab2_DMP_plain_silu_L5_d1536_bn_dirichlet", c5, "bi_Gab2_DMP"))

    # ── bi_LBP_Gab2_DMP: 2 configs ──

    # 6. LBP + new textures: baseline arch
    c6 = _cfg(0, "bi_LBP_Gab2_DMP", "plain", "silu", 5, 1024, "batchnorm")
    configs.append(("bi_LBP_Gab2_DMP_plain_silu_L5_d1024_bn", c6, "bi_LBP_Gab2_DMP"))

    # 7. Optional: LBP + new textures + wider
    c7 = _cfg(0, "bi_LBP_Gab2_DMP", "plain", "silu", 5, 1536, "batchnorm")
    configs.append(("bi_LBP_Gab2_DMP_plain_silu_L5_d1536_bn", c7, "bi_LBP_Gab2_DMP"))

    return configs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="V11: New texture + Dirichlet sweep")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for V10/texture extraction")
    args = parser.parse_args()

    folds_to_run = args.folds if args.folds else list(range(N_FOLDS))

    # Wait for dependencies
    if not args.no_wait:
        wait_for_texture_v2(poll_seconds=30)
        wait_for_v10(poll_seconds=60)

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
    X_all, feature_sets, full_cols = build_v11_data(device)

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    # Build configs
    configs = build_v11_configs()
    total_runs = len(configs) * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V11 — New Texture + Dirichlet Sweep", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Configs:  {len(configs)}", flush=True)
    print(f"  Folds:    {folds_to_run}", flush=True)
    print(f"  Total:    {total_runs} runs", flush=True)
    print(f"  Device:   {device}", flush=True)
    for name, cfg, fs_name in configs:
        n_feat = len(feature_sets[fs_name])
        head = cfg.get("head_type", "softmax")
        print(f"    {name:55s}  {n_feat:5d}f  {head}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Resume
    results = []
    done_keys = set()
    if os.path.exists(V11_CSV):
        df_old = pd.read_csv(V11_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs already done", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    times = []
    n_skipped = 0
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
                n_skipped += 1
                continue

            fs_idx = feature_sets[fs_name]
            n_features = len(fs_idx)
            is_dir = cfg.get("head_type", "softmax") == "dirichlet"

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
            metrics = evaluate_model(y[test_idx], preds)
            r2 = metrics["r2_uniform"]
            mae = metrics["mae_mean_pp"]

            # ETA
            times.append(elapsed)
            runs_left = total_runs - run_idx
            eta_h = (np.mean(times) * runs_left) / 3600
            head_str = "DIR" if is_dir else "ILR"

            rec = {
                "name": name,
                "fold": fold_id,
                "feature_set": fs_name,
                "n_features": n_features,
                "head_type": "dirichlet" if is_dir else "softmax",
                "arch": f"{cfg['arch']}_L{cfg['n_layers']}_d{cfg['d_model']}",
                "r2_uniform": r2,
                "mae_mean_pp": mae,
                "best_val_loss": best_val,
                "n_epochs": n_epochs,
                "elapsed_s": elapsed,
                "n_train": len(trn_idx),
                "n_test": len(test_idx),
            }
            # Per-class R²
            for cn in CLASS_NAMES:
                if f"r2_{cn}" in metrics:
                    rec[f"r2_{cn}"] = metrics[f"r2_{cn}"]

            results.append(rec)
            done_keys.add((name, fold_id))

            # Save after each run
            pd.DataFrame(results).to_csv(V11_CSV, index=False)

            print(f"  [{run_idx:3d}/{total_runs}] F{fold_id} {name:55s} "
                  f"R2={r2:.4f}  MAE={mae:.2f}pp  "
                  f"ep={n_epochs}  {elapsed:.0f}s  {head_str}  "
                  f"ETA={eta_h:.1f}h", flush=True)

            # Free GPU
            del net, trained_net, X_trn_t, X_val_t_fs, X_test_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"V11 COMPLETE — {len(results)} runs", flush=True)
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
        summary_csv = V11_CSV.replace(".csv", "_summary.csv")
        summary.to_csv(summary_csv)
        print(summary.to_string())
        print(f"\nSaved: {V11_CSV}")
        print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
