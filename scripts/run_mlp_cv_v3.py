"""
MLP Cross-Validation V3: full 5-fold spatial CV for all architectures.

Runs every V3 config across all 5 spatial folds to produce
robust mean +/- std R^2 for each architecture/feature combination.

No screening -- every config gets a full 300-epoch run.
GPU pre-loading and index-batching for speed.

Output:
    reports/phase8/tables/mlp_cv_v3.csv          -- per-fold results
    reports/phase8/tables/mlp_cv_v3_summary.csv   -- mean +/- std aggregated

Usage:
    .venv\\Scripts\\python.exe scripts/run_mlp_cv_v3.py
    .venv\\Scripts\\python.exe scripts/run_mlp_cv_v3.py --folds 0 1 2
    .venv\\Scripts\\python.exe scripts/run_mlp_cv_v3.py --top-n 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CFG, PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

# Re-use V3's building blocks, configs, and training code
from scripts.run_mlp_mega_sweep_v3 import (
    PlainMLP, ResMLP, PlainBlock, GeGLUBlock, ResMLPBlock,
    cosine_warmup_scheduler, mixup_batch, normalize_targets,
    soft_cross_entropy, train_model, build_model,
    partition_features, top_k_by_variance,
    FEATURE_SETS, generate_configs, make_name, _cfg,
    CONTROL_COLS,
)

CLASS_NAMES = CFG["worldcover"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
SPLIT_CFG = CFG["split"]
SEED = SPLIT_CFG["seed"]
N_FOLDS = 5

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
OUT_CSV = os.path.join(OUT_DIR, "mlp_cv_v3.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "mlp_cv_v3_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)),
                        help="Which folds to run (default: all 5)")
    parser.add_argument("--top-n", type=int, default=0,
                        help="If >0, only runs the top-N configs from V3 sweep results")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    # ---- Device + seeds ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ---- Load data ----
    print("Loading data...")
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))

    core_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_core.parquet"))
    from pandas.api.types import is_numeric_dtype
    core_cols = [c for c in core_df.columns
                 if c not in CONTROL_COLS and is_numeric_dtype(core_df[c])]
    del core_df

    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    y = labels_df[CLASS_NAMES].values.astype(np.float32)
    del feat_df

    print(f"Full data: X={X_all.shape}, y={y.shape}")

    # ---- Split metadata ----
    folds = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    # ---- Feature groups ----
    feat_groups = partition_features(core_cols, full_feature_cols)
    # top500 computed on first fold's train split, then reused
    # (small bias, but consistent across folds)

    # ---- Generate configs ----
    configs = generate_configs()

    # ---- Optionally filter to top-N from V3 sweep ----
    if args.top_n > 0:
        v3_csv = os.path.join(OUT_DIR, "mlp_mega_sweep_v3.csv")
        if os.path.exists(v3_csv):
            v3_df = pd.read_csv(v3_csv)
            top_names = set(
                v3_df.sort_values("r2_uniform", ascending=False)
                .head(args.top_n)["name"].values
            )
            # Extract arch keys from top configs (ignoring feature set)
            top_base_names = set()
            for n in top_names:
                # Remove feature set prefix to get arch portion
                for fs in FEATURE_SETS:
                    if n.startswith(fs + "_"):
                        top_base_names.add(n[len(fs) + 1:])
                        break

            configs = [c for c in configs
                       if make_name(c).split("_", 1)[-1].replace(
                           c["feature_set"] + "_", "", 1
                       ) in top_base_names
                       or make_name(c) in top_names]
            print(f"Filtered to top-{args.top_n} architectures: {len(configs)} configs")
        else:
            print(f"Warning: --top-n specified but {v3_csv} not found, running all configs")

    total_runs = len(configs) * len(args.folds)
    print(f"\nCV plan: {len(configs)} configs x {len(args.folds)} folds = {total_runs} total runs")

    # ---- Resume ----
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    completed = set()
    if os.path.exists(OUT_CSV):
        df_old = pd.read_csv(OUT_CSV)
        results = df_old.to_dict("records")
        completed = {(r["name"], r["fold"]) for r in results}
        print(f"Found {len(completed)} completed (config, fold) pairs, resuming...")

    # ---- Cross-validation loop ----
    t0_total = time.time()
    n_done = len(completed)

    for fold_id in args.folds:
        print(f"\n{'='*80}")
        print(f"FOLD {fold_id}/{N_FOLDS-1}")
        print(f"{'='*80}")

        # ---- Train/val/test split for this fold ----
        train_idx, test_idx = get_fold_indices(
            tiles, folds, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )
        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx = train_idx[perm[:n_val]]
        trn_idx = train_idx[perm[n_val:]]
        print(f"  Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # ---- top500 for this fold (train-only variance) ----
        feat_groups["top500_full"] = top_k_by_variance(X_all[trn_idx], 500)

        # ---- Pre-scale and move to GPU for this fold ----
        print("  Pre-scaling feature sets -> GPU...")
        scaled_cache = {}
        for feat_set in FEATURE_SETS:
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
                torch.tensor(X_test_s, dtype=torch.float32).to(device),
                len(feat_idx),
            )

        y_trn_t = torch.tensor(normalize_targets(y[trn_idx], N_CLASSES)).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx], N_CLASSES)).to(device)

        # ---- Run all configs for this fold ----
        for ci, cfg in enumerate(configs):
            name = make_name(cfg)
            if (name, fold_id) in completed:
                continue

            feat_set = cfg["feature_set"]
            if feat_set not in scaled_cache:
                continue

            X_trn_t, X_val_t, X_test_t, n_features = scaled_cache[feat_set]

            torch.manual_seed(SEED + fold_id)
            net = build_model(cfg, n_features, device)
            n_params = sum(p.numel() for p in net.parameters())

            t0 = time.time()
            epochs, best_val_loss, final_model = train_model(
                net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                batch_size=args.batch_size, max_epochs=args.max_epochs,
                patience=args.patience,
                mixup_alpha=cfg.get("mixup_alpha", 0),
                use_swa=cfg.get("use_swa", False),
                use_cosine=cfg.get("use_cosine", True),
            )

            # Evaluate
            final_model.eval()
            with torch.no_grad():
                y_pred = final_model.predict(X_test_t).cpu().numpy()
            elapsed = time.time() - t0

            # Free GPU memory
            del net, final_model
            if device == "cuda":
                torch.cuda.empty_cache()

            summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
            summary.update({
                "name": name,
                "fold": fold_id,
                "feature_set": feat_set,
                "arch": cfg["arch"],
                "activation": cfg["activation"],
                "n_layers": cfg["n_layers"],
                "d_model": cfg["d_model"],
                "expansion": cfg.get("expansion", 0),
                "dropout": cfg["dropout"],
                "input_dropout": cfg.get("input_dropout", 0),
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "mixup_alpha": cfg.get("mixup_alpha", 0),
                "use_swa": cfg.get("use_swa", False),
                "n_features": n_features,
                "n_params": n_params,
                "epochs": epochs,
                "best_val_loss": round(best_val_loss, 6),
                "elapsed_s": round(elapsed, 1),
            })

            results.append(summary)
            n_done += 1

            # Save after every result (resumable)
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)

            r2 = summary["r2_uniform"]
            mae = summary["mae_mean_pp"]
            ait = summary.get("aitchison_mean", float("nan"))
            print(f"  [{n_done:4d}/{total_runs}] F{fold_id} {name:55s}  "
                  f"R2={r2:.4f}  MAE={mae:.2f}pp  Ait={ait:.4f}  "
                  f"ep={epochs:3d}  {elapsed:.0f}s")

        # Free fold-level GPU tensors
        del scaled_cache, y_trn_t, y_val_t
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- Aggregate summary ----
    total_time = time.time() - t0_total
    df = pd.DataFrame(results)

    print(f"\n{'='*100}")
    print(f"CV COMPLETE: {len(df)} results in {total_time/3600:.1f} hours")
    print(f"{'='*100}")

    if len(df) > 0 and "fold" in df.columns:
        # Per-config aggregation across folds
        agg = df.groupby("name").agg(
            r2_mean=("r2_uniform", "mean"),
            r2_std=("r2_uniform", "std"),
            mae_mean=("mae_mean_pp", "mean"),
            mae_std=("mae_mean_pp", "std"),
            ait_mean=("aitchison_mean", "mean"),
            ait_std=("aitchison_mean", "std"),
            n_folds=("fold", "count"),
            epochs_mean=("epochs", "mean"),
            elapsed_mean=("elapsed_s", "mean"),
            n_features=("n_features", "first"),
            n_params=("n_params", "first"),
            feature_set=("feature_set", "first"),
            arch=("arch", "first"),
            activation=("activation", "first"),
        ).sort_values("r2_mean", ascending=False).reset_index()

        agg.to_csv(SUMMARY_CSV, index=False)
        print(f"\nSummary saved to {SUMMARY_CSV}")

        print("\nTOP 15 by mean R2 (across folds):")
        for _, r in agg.head(15).iterrows():
            print(f"  {r['name']:55s}  "
                  f"R2={r['r2_mean']:.4f}+/-{r['r2_std']:.4f}  "
                  f"MAE={r['mae_mean']:.2f}  "
                  f"folds={r['n_folds']:.0f}  "
                  f"feat={r['n_features']:.0f}")

        print("\nBEST PER FEATURE SET:")
        for fs in sorted(agg["feature_set"].unique()):
            best = agg[agg["feature_set"] == fs].iloc[0]
            print(f"  {fs:30s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  "
                  f"name={best['name']}")


if __name__ == "__main__":
    main()
