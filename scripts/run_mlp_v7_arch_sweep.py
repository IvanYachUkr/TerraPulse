#!/usr/bin/env python3
"""
MLP V7: Architecture Sweep with bi_glcm_morph + novel v2 indices.

Auto-waits for V6 to finish, then runs the same architectures on an
extended feature set:
  bands_indices + GLCM + Morph + NDTI + IRECI + CRI1
  (no MNDWI — too noisy/harmful per ablation)

Novel v2 indices rationale:
  - NDTI:  SWIR1/SWIR2 ratio — genuinely new band pair
  - IRECI: Red-edge chlorophyll with B07 — novel multi-band combo
  - CRI1:  Reciprocal difference — nonlinear carotenoid signal

Output:
    reports/phase8/tables/mlp_v7_arch.csv
    reports/phase8/tables/mlp_v7_arch_summary.csv

Usage:
    .venv\\Scripts\\python.exe scripts/run_mlp_v7_arch_sweep.py
    .venv\\Scripts\\python.exe scripts/run_mlp_v7_arch_sweep.py --no-wait
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

# ── Reuse V4/V5 infrastructure ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from run_mlp_overnight_v4 import (
    PROJECT_ROOT, CLASS_NAMES, N_FOLDS, CONTROL_COLS, SEED,
    build_model, make_name, _cfg,
    train_model, normalize_targets, _predict_batched,
    partition_features,
)
from src.config import PROCESSED_V2_DIR
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

# Output paths
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V7_CSV = os.path.join(OUT_DIR, "mlp_v7_arch.csv")
V7_SUMMARY = os.path.join(OUT_DIR, "mlp_v7_arch_summary.csv")

# V6 output to wait for
V6_CSV = os.path.join(OUT_DIR, "mlp_v6_arch.csv")
V6_EXPECTED_RUNS = 50  # 10 configs x 5 folds

# Novel v2 index prefixes to include (no MNDWI — too noisy)
NOVEL_V2_PREFIXES = ["NDTI_", "IRECI_", "CRI1_"]


# =====================================================================
# Wait for V6
# =====================================================================

def wait_for_v6(poll_interval=60):
    """Poll V6 CSV until all runs are done."""
    print(f"Waiting for V6 to finish ({V6_EXPECTED_RUNS} runs)...", flush=True)
    while True:
        if os.path.exists(V6_CSV):
            try:
                df = pd.read_csv(V6_CSV)
                n = len(df)
                if n >= V6_EXPECTED_RUNS:
                    n_valid = df["r2_uniform"].notna().sum()
                    n_err = df["r2_uniform"].isna().sum()
                    print(f"  V6 done! {n_valid} valid + {n_err} errors = {n} total",
                          flush=True)
                    return
                print(f"  V6: {n}/{V6_EXPECTED_RUNS} runs done, waiting...", flush=True)
            except Exception:
                pass
        else:
            print(f"  V6 CSV not found yet, waiting...", flush=True)
        time.sleep(poll_interval)


# =====================================================================
# Feature set: bands_indices_glcm_morph + novel v2
# =====================================================================

def build_feature_set(full_feature_cols, v2_only_cols):
    """Build bi_glcm_morph + NDTI/IRECI/CRI1 feature indices.

    Returns:
        feat_indices: list of int indices into the combined column list
        all_cols: the combined column list (full_feature_cols + selected v2)
    """
    feat_groups = partition_features(full_feature_cols)

    # Base: bands + indices + GLCM + Morph
    base_idx = set(feat_groups["bands_indices"])
    glcm_idx = {i for i, c in enumerate(full_feature_cols) if "GLCM_" in c}
    morph_idx = {i for i, c in enumerate(full_feature_cols) if "MP_" in c}
    bi_glcm_morph = sorted(base_idx | glcm_idx | morph_idx)

    # Novel v2 columns (NDTI, IRECI, CRI1 only — skip MNDWI)
    novel_v2 = [c for c in v2_only_cols
                if any(c.startswith(p) for p in NOVEL_V2_PREFIXES)]

    # Combined column list
    all_cols = full_feature_cols + novel_v2
    # Indices: bi_glcm_morph from base + appended novel v2
    n_base = len(full_feature_cols)
    novel_indices = list(range(n_base, n_base + len(novel_v2)))
    feat_indices = bi_glcm_morph + novel_indices

    return feat_indices, all_cols, novel_v2


# =====================================================================
# Architecture configs (same as V6)
# =====================================================================

FEATURE_SET = "bi_glcm_morph_v2"


def generate_v7_configs():
    """Same architectures as V6, on the extended feature set."""
    configs = []
    rid = 0
    fs = FEATURE_SET

    # ── V5 champion configs ──
    configs.append(_cfg(rid, fs, "plain", "silu", 5, 1024, "batchnorm")); rid += 1
    configs.append(_cfg(rid, fs, "plain", "silu", 5, 512, "batchnorm")); rid += 1
    configs.append(_cfg(rid, fs, "residual", "silu", 10, 256, "none")); rid += 1

    # ── Deep family ──
    configs.append(_cfg(rid, fs, "residual", "silu", 12, 256, "batchnorm")); rid += 1
    configs.append(_cfg(rid, fs, "residual", "silu", 16, 256, "batchnorm", lr=5e-4)); rid += 1
    configs.append(_cfg(rid, fs, "residual", "silu", 20, 256, "batchnorm", lr=5e-4)); rid += 1

    # ── Wide family ──
    configs.append(_cfg(rid, fs, "plain", "silu", 5, 2048, "batchnorm", lr=5e-4)); rid += 1

    # ── Deep+Wide family ──
    configs.append(_cfg(rid, fs, "residual", "silu", 10, 1024, "batchnorm", lr=5e-4)); rid += 1
    configs.append(_cfg(rid, fs, "residual", "silu", 11, 1024, "batchnorm", lr=5e-4)); rid += 1
    configs.append(_cfg(rid, fs, "residual", "silu", 12, 1024, "batchnorm", lr=5e-4)); rid += 1

    return configs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MLP V7: bi_glcm_morph + v2 novel")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for V6 to finish")
    args = parser.parse_args()

    # ── Wait for V6 ──
    if not args.no_wait:
        wait_for_v6(poll_interval=60)
    else:
        print("Skipping V6 wait (--no-wait)", flush=True)

    folds_to_run = args.folds if args.folds else list(range(N_FOLDS))

    configs = generate_v7_configs()
    total_configs = len(configs)
    total_runs = total_configs * len(folds_to_run)

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V7 — bi_glcm_morph + NDTI + IRECI + CRI1", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Configs:        {total_configs}", flush=True)
    print(f"  Folds:          {folds_to_run}", flush=True)
    print(f"  Total runs:     {total_runs}", flush=True)
    print(f"  Max epochs:     {args.max_epochs}", flush=True)
    print(f"  Patience:       {args.patience_steps} steps", flush=True)
    print(f"  Min steps:      {args.min_steps}", flush=True)
    print(f"  Output CSV:     {V7_CSV}", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("Configs:", flush=True)
    for c in configs:
        print(f"  {make_name(c)}", flush=True)
    print(flush=True)

    # ── Device ──
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

    # ── Load data ──
    print("Loading data...", flush=True)

    # Base features
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # V2 extras
    v2_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_bands_indices_v2.parquet"))
    v2_only_cols = [c for c in v2_df.columns
                    if c != "cell_id" and c not in feat_df.columns]

    # Build combined feature set
    feat_idx, all_cols, novel_v2 = build_feature_set(full_feature_cols, v2_only_cols)
    print(f"  bi_glcm_morph base: from {len(full_feature_cols)} full cols", flush=True)
    print(f"  Novel v2 added: {len(novel_v2)} cols ({', '.join(sorted(set(c.split('_')[0] for c in novel_v2)))})",
          flush=True)
    print(f"  Total feature set: {len(feat_idx)} features", flush=True)

    # Build combined array
    base_arr = feat_df[full_feature_cols].values.astype(np.float32)
    v2_arr = v2_df[novel_v2].values.astype(np.float32) if novel_v2 else np.empty((len(feat_df), 0), dtype=np.float32)
    X_all = np.hstack([base_arr, v2_arr])
    np.nan_to_num(X_all, copy=False)
    del feat_df, v2_df, base_arr, v2_arr

    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = _json.load(f)

    print(f"Data loaded: X={X_all.shape}, y={y.shape}", flush=True)

    # ── Resume logic ──
    results = []
    done_keys = set()
    if os.path.exists(V7_CSV):
        df_old = pd.read_csv(V7_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs already done", flush=True)

    # ── CV loop ──
    os.makedirs(OUT_DIR, exist_ok=True)
    run_idx = 0
    n_skipped = 0
    times = []

    for fold_id in folds_to_run:
        print(f"\n{'='*80}", flush=True)
        print(f"FOLD {fold_id}/{N_FOLDS-1}", flush=True)
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

        # Scale
        print("  Scaling -> GPU...", flush=True)
        X_fs = X_all[:, feat_idx]
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X_fs[trn_idx]).astype(np.float32)
        X_val_s = scaler.transform(X_fs[val_idx]).astype(np.float32)
        X_test_s = scaler.transform(X_fs[test_idx]).astype(np.float32)

        X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        n_features = len(feat_idx)
        print(f"    {n_features} features", flush=True)

        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        for ci, cfg in enumerate(configs):
            name = make_name(cfg)
            run_idx += 1

            if (name, fold_id) in done_keys:
                n_skipped += 1
                continue

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

                y_pred = _predict_batched(final_model, X_test_t, device)
                elapsed = time.time() - t0
                del net, final_model

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)

                n_layers = cfg["n_layers"]
                d_model = cfg["d_model"]
                if cfg["arch"] == "plain" and d_model >= 1024:
                    arch_family = "wide"
                elif cfg["arch"] == "plain":
                    arch_family = "v5_champion"
                elif d_model <= 256 and n_layers >= 12:
                    arch_family = "deep"
                elif d_model <= 256:
                    arch_family = "v5_champion"
                else:
                    arch_family = "deep+wide"

                summary.update({
                    "name": name, "fold": fold_id, "stage": "v7",
                    "feature_set": FEATURE_SET, "arch": cfg["arch"],
                    "activation": cfg["activation"],
                    "norm_type": cfg.get("norm_type", "layernorm"),
                    "n_layers": n_layers, "d_model": d_model,
                    "arch_family": arch_family,
                    "lr": cfg["lr"], "n_features": n_features,
                    "epochs": epochs_done, "elapsed_s": round(elapsed, 1),
                })

                results.append(summary)
                done_keys.add((name, fold_id))
                times.append(elapsed)

                avg_time = sum(times) / len(times)
                remaining = max(0, total_runs - n_skipped - len(times))
                eta_h = remaining * avg_time / 3600

                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                early_tag = f"ep={epochs_done:3d}" if epochs_done < args.max_epochs else f"ep={epochs_done:3d}(MAX)"
                print(f"  [{n_skipped+len(times):4d}/{total_runs}] F{fold_id} {name:60s} "
                      f"R2={r2:.4f}  MAE={mae:.2f}pp  {early_tag}  "
                      f"{elapsed:.0f}s  ETA={eta_h:.1f}h  [{arch_family}]", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                results.append({"name": name, "fold": fold_id,
                                "stage": "v7", "error": str(e)})
                done_keys.add((name, fold_id))
                print(f"  [{n_skipped+len(times):4d}/{total_runs}] F{fold_id} {name:60s} "
                      f"ERROR: {e} ({elapsed:.0f}s)", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()

            if (n_skipped + len(times)) % 5 == 0:
                pd.DataFrame(results).to_csv(V7_CSV, index=False)

        pd.DataFrame(results).to_csv(V7_CSV, index=False)
        del X_trn_t, X_val_t, X_test_t, y_trn_t, y_val_t
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final save
    pd.DataFrame(results).to_csv(V7_CSV, index=False)
    print(f"\nResults saved to {V7_CSV}", flush=True)

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
                   arch=("arch", "first"),
                   arch_family=("arch_family", "first"),
               )
               .sort_values("r2_mean", ascending=False))
        agg.to_csv(V7_SUMMARY)
        print(f"Summary saved to {V7_SUMMARY}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"V7 RESULTS — bi_glcm_morph + NDTI + IRECI + CRI1", flush=True)
        print(f"{'='*70}", flush=True)
        for _, row in agg.iterrows():
            fs_str = f"folds={int(row['folds'])}"
            print(f"  {row.name:60s} R2={row['r2_mean']:.4f}+/-{row['r2_std']:.4f}  "
                  f"MAE={row['mae_mean']:.2f}  ep={row['epochs_mean']:.0f}  "
                  f"{fs_str}  [{row['arch_family']}]", flush=True)

        print(f"\nBEST PER ARCHITECTURE FAMILY:", flush=True)
        for fam in ["v5_champion", "deep", "wide", "deep+wide"]:
            fam_agg = agg[agg["arch_family"] == fam]
            if len(fam_agg) > 0:
                best = fam_agg.iloc[0]
                print(f"  {fam:12s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  {best.name}", flush=True)

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
