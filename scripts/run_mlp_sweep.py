"""
Quick MLP sweep: test multiple architectures with huge batch sizes.

Runs independently of run_phase8.py, can run in parallel with trees.
Saves results to reports/phase8/tables/mlp_sweep.csv

Usage:
    python scripts/run_mlp_sweep.py              # full sweep
    python scripts/run_mlp_sweep.py --quick       # 3 configs only
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CFG, PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.evaluation import evaluate_model
from src.models.mlp_torch import SoftmaxMLP, DirichletMLP
from src.splitting import get_fold_indices
from src.transforms import helmert_basis, ilr_forward

CLASS_NAMES = CFG["worldcover"]["class_names"]
SPLIT_CFG = CFG["split"]
SEED = SPLIT_CFG["seed"]
CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}


def load_data():
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_core.parquet"))
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))

    feature_cols = [c for c in feat_df.columns
                    if c not in CONTROL_COLS and feat_df[c].dtype in ("float64", "float32", "int64")]
    X = feat_df[feature_cols].values.astype(np.float64)
    y = labels_df[CLASS_NAMES].values.astype(np.float64)
    fold_assignments = split_df["fold_region_growing"].values
    tile_groups = split_df["tile_group"].values

    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    return X, y, fold_assignments, tile_groups, meta["tile_cols"], meta["tile_rows"]


# =====================================================================
# Sweep configs
# =====================================================================

FULL_CONFIGS = [
    # name, model_class, kwargs
    ("softmax_h128_L2_bs512",    SoftmaxMLP,   dict(hidden_dim=128, n_layers=2, batch_size=512)),
    ("softmax_h256_L3_bs512",    SoftmaxMLP,   dict(hidden_dim=256, n_layers=3, batch_size=512)),
    ("softmax_h512_L3_bs512",    SoftmaxMLP,   dict(hidden_dim=512, n_layers=3, batch_size=512)),
    ("softmax_h256_L4_bs512",    SoftmaxMLP,   dict(hidden_dim=256, n_layers=4, batch_size=512)),
    ("softmax_h128_L2_full",     SoftmaxMLP,   dict(hidden_dim=128, n_layers=2, batch_size=30000)),
    ("softmax_h256_L3_full",     SoftmaxMLP,   dict(hidden_dim=256, n_layers=3, batch_size=30000)),
    ("softmax_h512_L3_full",     SoftmaxMLP,   dict(hidden_dim=512, n_layers=3, batch_size=30000)),
    ("softmax_h512_L4_full",     SoftmaxMLP,   dict(hidden_dim=512, n_layers=4, batch_size=30000)),
    ("softmax_h256_L3_drop30",   SoftmaxMLP,   dict(hidden_dim=256, n_layers=3, dropout=0.30)),
    ("softmax_h256_L3_drop05",   SoftmaxMLP,   dict(hidden_dim=256, n_layers=3, dropout=0.05)),
    ("softmax_h256_L3_lr5e4",    SoftmaxMLP,   dict(hidden_dim=256, n_layers=3, lr=5e-4)),
    ("softmax_h256_L3_lr3e3",    SoftmaxMLP,   dict(hidden_dim=256, n_layers=3, lr=3e-3)),
    ("softmax_h1024_L3_full",    SoftmaxMLP,   dict(hidden_dim=1024, n_layers=3, batch_size=30000)),
    ("dirichlet_h256_L3_bs512",  DirichletMLP, dict(hidden_dim=256, n_layers=3, batch_size=512)),
    ("dirichlet_h256_L3_full",   DirichletMLP, dict(hidden_dim=256, n_layers=3, batch_size=30000)),
    ("dirichlet_h512_L3_full",   DirichletMLP, dict(hidden_dim=512, n_layers=3, batch_size=30000)),
]

QUICK_CONFIGS = FULL_CONFIGS[:3]  # just 3 for smoke test


def run_sweep(configs, X, y, fold_assignments, tile_groups, n_tc, n_tr):
    """Train each config on folds 1-4, evaluate on fold 0."""
    # Get train/test split (fold 0 = test)
    train_idx, test_idx = get_fold_indices(
        tile_groups, fold_assignments, 0, n_tc, n_tr, buffer_tiles=1,
    )

    # Validation split from train
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_idx))
    n_val = max(int(len(train_idx) * 0.15), 100)
    val_idx = train_idx[perm[:n_val]]
    trn_idx = train_idx[perm[n_val:]]

    print(f"Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"{'='*80}")

    results = []

    for name, model_cls, kwargs in configs:
        print(f"\n--- {name} ---")
        t0 = time.time()

        model = model_cls(
            n_classes=len(CLASS_NAMES),
            max_epochs=300,
            patience=20,
            random_state=SEED,
            **kwargs,
        )

        model.fit(X[trn_idx], y[trn_idx], X_val=X[val_idx], z_val_or_y=y[val_idx])

        y_pred = model.predict_proportions(X[test_idx])
        elapsed = time.time() - t0

        summary, detail = evaluate_model(y[test_idx], y_pred, CLASS_NAMES, model_name=name)
        summary["epochs"] = len(model.train_losses)
        summary["elapsed_s"] = round(elapsed, 1)
        summary["batch_size"] = kwargs.get("batch_size", 512)
        summary["hidden_dim"] = kwargs.get("hidden_dim", 256)
        summary["n_layers"] = kwargs.get("n_layers", 3)
        summary["dropout"] = kwargs.get("dropout", 0.15)
        summary["lr"] = kwargs.get("lr", 1e-3)
        summary["model_class"] = model_cls.__name__

        results.append(summary)

        print(f"  R2={summary['r2_uniform']:.4f}  MAE={summary['mae_mean_pp']:.2f}pp  "
              f"Aitchison={summary['aitchison_mean']:.4f}  "
              f"epochs={summary['epochs']}  time={elapsed:.1f}s")

        # Save incrementally
        out_dir = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(
            os.path.join(out_dir, "mlp_sweep.csv"), index=False,
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="3 configs only")
    args = parser.parse_args()

    X, y, folds, tiles, n_tc, n_tr = load_data()
    print(f"Data: X={X.shape}, y={y.shape}")

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS
    print(f"Running {len(configs)} MLP configs")

    df = run_sweep(configs, X, y, folds, tiles, n_tc, n_tr)

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (sorted by R2):")
    print(df[["model", "r2_uniform", "mae_mean_pp", "aitchison_mean",
              "epochs", "elapsed_s", "batch_size", "hidden_dim"]].sort_values(
                  "r2_uniform", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
