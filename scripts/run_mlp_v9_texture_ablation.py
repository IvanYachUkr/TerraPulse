#!/usr/bin/env python3
"""
MLP V9: Texture ablation — which texture groups actually help?

Base: bands + indices (798 features).
Adds all singles, pairs, and triples of 6 texture groups:
  GLCM (60), LBP (66), MP (54), HOG (204), Gabor (144), SV (42)

Architecture: V5 champion — plain silu L5 d1024 batchnorm.
Folds: 0, 2 only (fast probe).

Combos: 6 singles + 15 pairs + 20 triples = 41 configs × 2 folds = 82 runs.
ETA: ~2h.

Output:
    reports/phase8/tables/mlp_v9_texture_ablation.csv
    reports/phase8/tables/mlp_v9_texture_ablation_summary.csv
"""

import argparse
import json as _json
import os
import sys
import time
from itertools import combinations

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
from src.splitting import get_fold_indices

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
V9_CSV = os.path.join(OUT_DIR, "mlp_v9_texture_ablation.csv")
V9_SUMMARY = os.path.join(OUT_DIR, "mlp_v9_texture_ablation_summary.csv")

TEXTURE_GROUPS = ["GLCM", "LBP", "MP", "HOG", "Gabor", "SV"]


# =====================================================================
# Feature set building
# =====================================================================

def get_texture_indices(full_feature_cols):
    """Return dict mapping texture group name -> list of column indices."""
    result = {}
    for grp in TEXTURE_GROUPS:
        prefix = grp + "_"
        result[grp] = [i for i, c in enumerate(full_feature_cols) if c.startswith(prefix)]
    return result


def generate_combos():
    """Generate all singles, pairs, triples of texture groups."""
    combos = []
    for r in range(1, 4):  # 1, 2, 3
        for combo in combinations(TEXTURE_GROUPS, r):
            combos.append(combo)
    return combos


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="V9: Texture ablation")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 2])
    args = parser.parse_args()

    folds_to_run = args.folds
    combos = generate_combos()

    print(f"\n{'='*70}", flush=True)
    print(f"MLP V9 — Texture Ablation", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Architecture: plain silu L5 d1024 batchnorm (V5 champion)", flush=True)
    print(f"  Base:         bands + indices", flush=True)
    print(f"  Combos:       {len(combos)} (6 singles + 15 pairs + 20 triples)", flush=True)
    print(f"  Folds:        {folds_to_run}", flush=True)
    print(f"  Total runs:   {len(combos) * len(folds_to_run)}", flush=True)
    print(f"{'='*70}\n", flush=True)

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

    # Load data
    print("Loading data...", flush=True)
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    # Get base (bands+indices) and texture group indices
    feat_groups = partition_features(full_feature_cols)
    base_idx = feat_groups["bands_indices"]
    texture_idx = get_texture_indices(full_feature_cols)

    print(f"  Base (bands+indices): {len(base_idx)} features", flush=True)
    for grp, idx in texture_idx.items():
        print(f"  {grp:8s}: {len(idx)} features", flush=True)

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

    print(f"Data loaded: X={X_all.shape}, y={y.shape}\n", flush=True)

    # Resume logic
    results = []
    done_keys = set()
    if os.path.exists(V9_CSV):
        df_old = pd.read_csv(V9_CSV)
        results = df_old.to_dict("records")
        done_keys = set(zip(df_old["name"], df_old["fold"].astype(int)))
        print(f"Resuming: {len(results)} runs already done", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    total_runs = len(combos) * len(folds_to_run)
    times = []
    n_skipped = 0

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

        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        for combo in combos:
            combo_name = "+".join(combo)
            name = f"bi_{combo_name}_plain_silu_L5_d1024_bn"

            if (name, fold_id) in done_keys:
                n_skipped += 1
                continue

            # Build feature indices for this combo
            feat_idx = list(base_idx)
            for grp in combo:
                feat_idx.extend(texture_idx[grp])
            feat_idx = sorted(set(feat_idx))
            n_features = len(feat_idx)

            # Scale
            X_fs = X_all[:, feat_idx]
            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X_fs[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X_fs[val_idx]).astype(np.float32)
            X_test_s = scaler.transform(X_fs[test_idx]).astype(np.float32)

            X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

            t0 = time.time()
            try:
                cfg = _cfg(0, name, "plain", "silu", 5, 1024, "batchnorm")
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
                    mixup_alpha=0,
                    use_swa=False,
                    use_cosine=True,
                )

                y_pred = _predict_batched(final_model, X_test_t, device)
                elapsed = time.time() - t0
                del net, final_model

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)

                combo_size = len(combo)
                summary.update({
                    "name": name, "fold": fold_id, "stage": "v9",
                    "texture_combo": combo_name,
                    "combo_size": combo_size,
                    "n_features": n_features,
                    "n_texture_features": n_features - len(base_idx),
                    "epochs": epochs_done,
                    "elapsed_s": round(elapsed, 1),
                })

                results.append(summary)
                done_keys.add((name, fold_id))
                times.append(elapsed)

                avg_time = sum(times) / len(times)
                remaining = max(0, total_runs - n_skipped - len(times))
                eta_h = remaining * avg_time / 3600

                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {combo_name:20s} "
                      f"({n_features:4d}f) R2={r2:.4f}  MAE={mae:.2f}pp  "
                      f"ep={epochs_done:3d}  {elapsed:.0f}s  ETA={eta_h:.1f}h", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                results.append({"name": name, "fold": fold_id,
                                "stage": "v9", "error": str(e)})
                done_keys.add((name, fold_id))
                print(f"  [{n_skipped+len(times):3d}/{total_runs}] F{fold_id} {combo_name:20s} "
                      f"ERROR: {e} ({elapsed:.0f}s)", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()

            del X_trn_t, X_val_t, X_test_t

            if (n_skipped + len(times)) % 5 == 0:
                pd.DataFrame(results).to_csv(V9_CSV, index=False)

        pd.DataFrame(results).to_csv(V9_CSV, index=False)
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final save
    pd.DataFrame(results).to_csv(V9_CSV, index=False)
    print(f"\nResults saved to {V9_CSV}", flush=True)

    # Summary
    df = pd.DataFrame(results)
    valid = df[df["r2_uniform"].notna()]
    if len(valid) > 0:
        agg = (valid.groupby(["texture_combo", "combo_size"])
               .agg(
                   r2_mean=("r2_uniform", "mean"),
                   r2_std=("r2_uniform", "std"),
                   mae_mean=("mae_mean_pp", "mean"),
                   n_feat=("n_features", "first"),
                   n_tex=("n_texture_features", "first"),
                   folds=("fold", "count"),
               )
               .sort_values("r2_mean", ascending=False))
        agg.to_csv(V9_SUMMARY)
        print(f"Summary saved to {V9_SUMMARY}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"TEXTURE ABLATION RESULTS", flush=True)
        print(f"{'='*70}", flush=True)

        for size_label, size in [("SINGLES", 1), ("PAIRS", 2), ("TRIPLES", 3)]:
            sub = agg.xs(size, level="combo_size") if size in agg.index.get_level_values("combo_size") else pd.DataFrame()
            if len(sub) > 0:
                sub = sub.sort_values("r2_mean", ascending=False)
                print(f"\n  --- {size_label} ---", flush=True)
                for combo_name, row in sub.iterrows():
                    std_s = f"{row['r2_std']:.4f}" if pd.notna(row['r2_std']) else "n/a"
                    print(f"    {combo_name:30s} ({int(row['n_feat'])}f, +{int(row['n_tex'])}tex)  "
                          f"R2={row['r2_mean']:.4f}+-{std_s}  MAE={row['mae_mean']:.2f}pp", flush=True)

        print(f"\n  --- OVERALL TOP 10 ---", flush=True)
        for (combo_name, size), row in agg.head(10).iterrows():
            std_s = f"{row['r2_std']:.4f}" if pd.notna(row['r2_std']) else "n/a"
            print(f"    {combo_name:30s} (size={size}, {int(row['n_feat'])}f)  "
                  f"R2={row['r2_mean']:.4f}+-{std_s}", flush=True)

    total_time = sum(times)
    if total_time > 0:
        print(f"\nTotal compute time: {total_time/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
