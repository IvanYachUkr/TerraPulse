#!/usr/bin/env python3
"""
MLP V10: Definitive sweep — best texture combos × architectures × heads.

Tests 3 feature sets:
  - bi_LBP:          bands + indices + LBP         (864 feat)
  - bi_MP_Gabor:     bands + indices + MP + Gabor   (996 feat)
  - bi_LBP_MP_Gabor: bands + indices + LBP+MP+Gabor (1062 feat)

Architectures (V4-style build_model):
  A1: plain   silu L5  d1024 batchnorm          (V5 champion)
  A2: plain   silu L5  d512  batchnorm          (V5 runner-up, often best MAE)
  A3: residual silu L20 d256  batchnorm lr=5e-4 (V6 deep champion)

Output heads:
  H1: ILR softmax (V4/V5 training pipeline — build_model + train_model)
  H2: SoftmaxMLP  (GeGLU, KL loss — src/models/mlp_torch.py)
  H3: DirichletMLP (GeGLU, Dirichlet NLL — free uncertainty)

Total: 3 feats × 3 archs × 1 ILR + 3 feats × 2 GeGLU-heads × 1 arch = 9 + 6 = 15 configs × 5 folds = 75 runs
(GeGLU heads only tested with one representative architecture: L3 h1024)

ETA: ~2.5h.

Output:
    reports/phase8/tables/mlp_v10_definitive.csv
    reports/phase8/tables/mlp_v10_definitive_summary.csv
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
    build_model, make_name, _cfg,
    train_model, normalize_targets, _predict_batched,
    partition_features,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.models.mlp_torch import SoftmaxMLP, DirichletMLP
from src.splitting import get_fold_indices

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V10_CSV = os.path.join(OUT_DIR, "mlp_v10_definitive.csv")
V10_SUMMARY = os.path.join(OUT_DIR, "mlp_v10_definitive_summary.csv")

TEXTURE_GROUPS = {
    "LBP": "LBP_",
    "MP": "MP_",
    "Gabor": "Gabor_",
}


# =====================================================================
# Feature set builders
# =====================================================================

def get_base_and_texture_indices(full_feature_cols):
    """Get base band+index indices and texture group indices."""
    groups = partition_features(full_feature_cols)
    base_idx = groups["bands_indices"]
    tex_idx = {}
    for grp, prefix in TEXTURE_GROUPS.items():
        tex_idx[grp] = [i for i, c in enumerate(full_feature_cols) if c.startswith(prefix)]
    return base_idx, tex_idx


def build_feature_sets(full_feature_cols):
    """Return dict of feature_set_name -> sorted list of column indices."""
    base_idx, tex_idx = get_base_and_texture_indices(full_feature_cols)

    sets = {}
    # 1. bands + indices + LBP
    sets["bi_LBP"] = sorted(set(base_idx) | set(tex_idx["LBP"]))
    # 2. bands + indices + MP + Gabor
    sets["bi_MP_Gabor"] = sorted(set(base_idx) | set(tex_idx["MP"]) | set(tex_idx["Gabor"]))
    # 3. bands + indices + LBP + MP + Gabor
    sets["bi_LBP_MP_Gabor"] = sorted(
        set(base_idx) | set(tex_idx["LBP"]) | set(tex_idx["MP"]) | set(tex_idx["Gabor"])
    )
    return sets


# =====================================================================
# V4-style configs (ILR + build_model)
# =====================================================================

def v4_configs(feat_set_name):
    """Return list of (name, cfg_dict) for V4-style architectures."""
    cfgs = []

    # A1: V5 champion — plain silu L5 d1024 bn
    c1 = _cfg(0, feat_set_name, "plain", "silu", 5, 1024, "batchnorm")
    cfgs.append((f"{feat_set_name}_plain_silu_L5_d1024_bn", c1))

    # A2: V5 runner-up — plain silu L5 d512 bn
    c2 = _cfg(0, feat_set_name, "plain", "silu", 5, 512, "batchnorm")
    cfgs.append((f"{feat_set_name}_plain_silu_L5_d512_bn", c2))

    # A3: V6 deep champion — residual silu L20 d256 bn lr=5e-4
    c3 = _cfg(0, feat_set_name, "residual", "silu", 20, 256, "batchnorm", lr=5e-4)
    cfgs.append((f"{feat_set_name}_residual_silu_L20_d256_bn_lr0.0005", c3))

    return cfgs


# =====================================================================
# GeGLU configs (SoftmaxMLP / DirichletMLP)
# =====================================================================

def geglu_configs(feat_set_name):
    """Return list of (name, head_class, kwargs) for GeGLU-based models."""
    configs = []
    # Representative GeGLU architecture: L3 h1024 (similar capacity to V5 winner)
    base_kw = dict(
        n_classes=6, hidden_dim=1024, n_layers=3,
        dropout=0.15, lr=1e-3, weight_decay=1e-4,
        batch_size=2048, max_epochs=500, patience=30,
    )
    configs.append((
        f"{feat_set_name}_softmax_geglu_L3_h1024",
        SoftmaxMLP, base_kw.copy(),
    ))
    configs.append((
        f"{feat_set_name}_dirichlet_geglu_L3_h1024",
        DirichletMLP, base_kw.copy(),
    ))
    return configs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="V10: Definitive feature+arch+head sweep")
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

    # Load data
    print("Loading data...", flush=True)
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    feature_sets = build_feature_sets(full_feature_cols)

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

    # Build run list
    v4_runs = []
    for fs_name, fs_idx in feature_sets.items():
        for name, cfg in v4_configs(fs_name):
            v4_runs.append((name, cfg, fs_name, fs_idx))

    geglu_runs = []
    for fs_name, fs_idx in feature_sets.items():
        for name, head_cls, kw in geglu_configs(fs_name):
            geglu_runs.append((name, head_cls, kw, fs_name, fs_idx))

    total_configs = len(v4_runs) + len(geglu_runs)
    total_runs = total_configs * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V10 — Definitive Sweep", flush=True)
    print(f"{'='*70}", flush=True)
    for fs_name, fs_idx in feature_sets.items():
        print(f"  {fs_name:20s}: {len(fs_idx)} features", flush=True)
    print(f"  V4-style configs:   {len(v4_runs)}", flush=True)
    print(f"  GeGLU configs:      {len(geglu_runs)}", flush=True)
    print(f"  Folds:              {folds_to_run}", flush=True)
    print(f"  Total runs:         {total_runs}", flush=True)
    print(f"  Device:             {device}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Resume
    results = []
    done_keys = set()
    if os.path.exists(V10_CSV):
        df_old = pd.read_csv(V10_CSV)
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
        print(f"  Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}", flush=True)

        # Pre-compute ILR targets for V4-style runs
        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        # ── V4-style runs ──
        for name, cfg, fs_name, fs_idx in v4_runs:
            run_idx += 1
            if (name, fold_id) in done_keys:
                n_skipped += 1
                continue

            n_features = len(fs_idx)
            X_fs = X_all[:, fs_idx]
            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X_fs[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X_fs[val_idx]).astype(np.float32)
            X_test_s = scaler.transform(X_fs[test_idx]).astype(np.float32)

            X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device)
            X_val_t_fs = torch.tensor(X_val_s, dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

            t0 = time.time()
            try:
                torch.manual_seed(SEED + fold_id)
                net = build_model(cfg, n_features, device)

                epochs_done, best_val_loss, final_model = train_model(
                    net, X_trn_t, y_trn_t, X_val_t_fs, y_val_t,
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

                y_pred = _predict_batched(final_model, X_test_t, device)
                elapsed = time.time() - t0
                del net, final_model

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
                summary.update({
                    "name": name, "fold": fold_id, "stage": "v10",
                    "head": "ilr_v4", "feature_set": fs_name,
                    "n_features": n_features,
                    "epochs": epochs_done,
                    "elapsed_s": round(elapsed, 1),
                })
                results.append(summary)
                done_keys.add((name, fold_id))
                times.append(elapsed)

                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                avg_t = sum(times) / len(times)
                remaining = max(0, total_runs - n_skipped - len(times))
                eta_h = remaining * avg_t / 3600
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {name:50s} "
                      f"R2={r2:.4f}  MAE={mae:.2f}pp  ep={epochs_done}  "
                      f"{elapsed:.0f}s  ETA={eta_h:.1f}h", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                results.append({"name": name, "fold": fold_id,
                                "stage": "v10", "head": "ilr_v4", "error": str(e)})
                done_keys.add((name, fold_id))
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {name:50s} "
                      f"ERROR: {e} ({elapsed:.0f}s)", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()

            del X_trn_t, X_val_t_fs, X_test_t

        # ── GeGLU-style runs (SoftmaxMLP / DirichletMLP) ──
        for name, head_cls, kw, fs_name, fs_idx in geglu_runs:
            run_idx += 1
            if (name, fold_id) in done_keys:
                n_skipped += 1
                continue

            n_features = len(fs_idx)
            X_fs = X_all[:, fs_idx].astype(np.float64)

            t0 = time.time()
            try:
                model = head_cls(
                    **kw, device=device, random_state=SEED + fold_id,
                )
                model.fit(
                    X_fs[trn_idx], y[trn_idx],
                    X_val=X_fs[val_idx], z_val_or_y=y[val_idx],
                )

                y_pred = model.predict_proportions(X_fs[test_idx])
                elapsed = time.time() - t0
                epochs_done = len(model.train_losses)

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
                head_name = "softmax_geglu" if head_cls == SoftmaxMLP else "dirichlet_geglu"
                summary.update({
                    "name": name, "fold": fold_id, "stage": "v10",
                    "head": head_name, "feature_set": fs_name,
                    "n_features": n_features,
                    "epochs": epochs_done,
                    "elapsed_s": round(elapsed, 1),
                })
                results.append(summary)
                done_keys.add((name, fold_id))
                times.append(elapsed)

                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                avg_t = sum(times) / len(times)
                remaining = max(0, total_runs - n_skipped - len(times))
                eta_h = remaining * avg_t / 3600
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {name:50s} "
                      f"R2={r2:.4f}  MAE={mae:.2f}pp  ep={epochs_done}  "
                      f"{elapsed:.0f}s  ETA={eta_h:.1f}h", flush=True)

                del model

            except Exception as e:
                elapsed = time.time() - t0
                head_name = "softmax_geglu" if head_cls == SoftmaxMLP else "dirichlet_geglu"
                results.append({"name": name, "fold": fold_id,
                                "stage": "v10", "head": head_name, "error": str(e)})
                done_keys.add((name, fold_id))
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {name:50s} "
                      f"ERROR: {e} ({elapsed:.0f}s)", flush=True)

            if device == "cuda":
                torch.cuda.empty_cache()

            if (n_skipped + len(times)) % 5 == 0:
                pd.DataFrame(results).to_csv(V10_CSV, index=False)

        pd.DataFrame(results).to_csv(V10_CSV, index=False)

    # Final save
    pd.DataFrame(results).to_csv(V10_CSV, index=False)
    print(f"\nResults saved to {V10_CSV}", flush=True)

    # Summary
    df = pd.DataFrame(results)
    valid = df[df["r2_uniform"].notna()]
    if len(valid) > 0:
        agg = (valid.groupby(["name", "head", "feature_set"])
               .agg(
                   r2_mean=("r2_uniform", "mean"),
                   r2_std=("r2_uniform", "std"),
                   mae_mean=("mae_mean_pp", "mean"),
                   mae_std=("mae_mean_pp", "std"),
                   n_feat=("n_features", "first"),
                   folds=("fold", "count"),
               )
               .sort_values("r2_mean", ascending=False))
        agg.to_csv(V10_SUMMARY)
        print(f"Summary saved to {V10_SUMMARY}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"V10 DEFINITIVE RESULTS", flush=True)
        print(f"{'='*70}", flush=True)

        for head_label in ["ilr_v4", "softmax_geglu", "dirichlet_geglu"]:
            sub = agg.xs(head_label, level="head") if head_label in agg.index.get_level_values("head") else pd.DataFrame()
            if len(sub) > 0:
                sub = sub.sort_values("r2_mean", ascending=False)
                print(f"\n  --- {head_label.upper()} ---", flush=True)
                for (nm, fs), row in sub.iterrows():
                    std_s = f"{row['r2_std']:.4f}" if pd.notna(row['r2_std']) else "n/a"
                    ms = f"{row['mae_std']:.2f}" if pd.notna(row['mae_std']) else "n/a"
                    print(f"    {nm:50s} ({fs}, {int(row['n_feat'])}f)  "
                          f"R2={row['r2_mean']:.4f}+-{std_s}  MAE={row['mae_mean']:.2f}+-{ms}pp  "
                          f"f={int(row['folds'])}", flush=True)

        print(f"\n  --- OVERALL TOP 10 (by R2) ---", flush=True)
        for (nm, head, fs), row in agg.head(10).iterrows():
            std_s = f"{row['r2_std']:.4f}" if pd.notna(row['r2_std']) else "n/a"
            print(f"    [{head:15s}] {nm:50s} R2={row['r2_mean']:.4f}+-{std_s}  "
                  f"MAE={row['mae_mean']:.2f}pp", flush=True)

        print(f"\n  --- OVERALL TOP 10 (by MAE) ---", flush=True)
        agg_mae = agg.sort_values("mae_mean")
        for (nm, head, fs), row in agg_mae.head(10).iterrows():
            ms = f"{row['mae_std']:.2f}" if pd.notna(row['mae_std']) else "n/a"
            print(f"    [{head:15s}] {nm:50s} MAE={row['mae_mean']:.2f}+-{ms}pp  "
                  f"R2={row['r2_mean']:.4f}", flush=True)

    total_time = sum(times)
    if total_time > 0:
        print(f"\nTotal compute time: {total_time/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
