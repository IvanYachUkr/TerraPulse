"""
Comprehensive overnight MLP sweep.

Tests ~160 configurations varying:
- Architecture: hidden_dim, n_layers, activation functions
- Training: learning rate, weight decay, L1 penalty
- Feature sets: all, bands-only, indices-only, deltas-only, no-deltas, top-K

Saves results incrementally to reports/phase8/tables/mlp_mega_sweep.csv

Expected runtime: ~5-6 hours at ~2 min/config on CPU.

Usage:
    python scripts/run_mlp_mega_sweep.py
"""

import argparse
import itertools
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CFG, PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

CLASS_NAMES = CFG["worldcover"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
SPLIT_CFG = CFG["split"]
SEED = SPLIT_CFG["seed"]

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
OUT_CSV = os.path.join(OUT_DIR, "mlp_mega_sweep.csv")

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}


# =====================================================================
# Flexible MLP with multiple activation functions
# =====================================================================

class GeGLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear(x)
        gate, value = h.chunk(2, dim=-1)
        return self.dropout(self.norm(F.gelu(gate) * value))


class SiLUBlock(nn.Module):
    """SiLU (Swish) activation block with LayerNorm."""
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.silu(self.linear(x))))


class ReLUBlock(nn.Module):
    """ReLU + BatchNorm block."""
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.relu(self.linear(x))))


class GELUBlock(nn.Module):
    """GELU + LayerNorm block."""
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.gelu(self.linear(x))))


class MishBlock(nn.Module):
    """Mish activation + LayerNorm block."""
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.mish(self.linear(x))))


BLOCK_MAP = {
    "geglu": GeGLUBlock,
    "silu": SiLUBlock,
    "relu": ReLUBlock,
    "gelu": GELUBlock,
    "mish": MishBlock,
}


class FlexibleMLP(nn.Module):
    """Configurable MLP with softmax output head."""

    def __init__(self, input_dim, n_classes, hidden_dim=256,
                 n_layers=3, dropout=0.15, activation="geglu"):
        super().__init__()
        block_cls = BLOCK_MAP[activation]
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(block_cls(current_dim, hidden_dim, dropout))
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(x)), dim=-1)

    def predict(self, x):
        return F.softmax(self.head(self.backbone(x)), dim=-1)


# =====================================================================
# Trainer with L1 regularization support
# =====================================================================

def train_model(net, X_trn, y_trn, X_val, y_val,
                lr=1e-3, weight_decay=1e-4, l1_lambda=0.0,
                batch_size=1024, max_epochs=300, patience=20,
                device="cpu"):
    """Train with AdamW, ReduceLROnPlateau, early stopping, optional L1."""
    eps = 1e-7
    y_trn_s = np.clip(y_trn, eps, 1.0)
    y_trn_s = y_trn_s / y_trn_s.sum(axis=1, keepdims=True)
    y_val_s = np.clip(y_val, eps, 1.0)
    y_val_s = y_val_s / y_val_s.sum(axis=1, keepdims=True)

    X_t = torch.tensor(X_trn, dtype=torch.float32)
    y_t = torch.tensor(y_trn_s, dtype=torch.float32)
    X_vt = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_vt = torch.tensor(y_val_s, dtype=torch.float32).to(device)

    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                    shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_val = float("inf")
    no_improve = 0
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        net.train()
        epoch_loss = 0.0
        n_b = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            log_pred = net(xb)
            loss = F.kl_div(log_pred, yb, reduction="batchmean", log_target=False)

            # L1 regularization on first layer weights (feature selection)
            if l1_lambda > 0:
                first_linear = None
                for module in net.backbone[0].modules():
                    if isinstance(module, nn.Linear):
                        first_linear = module
                        break
                if first_linear is not None:
                    loss = loss + l1_lambda * first_linear.weight.abs().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        avg_train = epoch_loss / max(n_b, 1)
        train_losses.append(avg_train)

        net.eval()
        with torch.no_grad():
            log_val = net(X_vt)
            val_loss = F.kl_div(log_val, y_vt, reduction="batchmean", log_target=False).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    return len(train_losses), best_val


# =====================================================================
# Feature group partitioning
# =====================================================================

def partition_features(all_cols):
    """Split feature columns into semantic groups."""
    bands = []
    indices = []
    deltas = []
    tasseled_cap = []
    other = []

    band_prefixes = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
    index_prefixes = {"NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI", "NDRE1", "NDRE2",
                      "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI"}

    for i, col in enumerate(all_cols):
        prefix = col.split("_")[0]
        if col.startswith("delta"):
            deltas.append(i)
        elif prefix in band_prefixes:
            bands.append(i)
        elif prefix in index_prefixes:
            indices.append(i)
        elif prefix == "TC":
            tasseled_cap.append(i)
        else:
            other.append(i)

    return {
        "all": list(range(len(all_cols))),
        "bands": bands,
        "indices": indices + tasseled_cap,
        "deltas": deltas,
        "no_deltas": bands + indices + tasseled_cap + other,
        "bands_and_indices": bands + indices + tasseled_cap,
        "other": other,
    }


def top_k_by_variance(X, k=500):
    """Return indices of top-K features by variance."""
    var = np.var(X, axis=0)
    return np.argsort(var)[-k:][::-1]


# =====================================================================
# Configuration generator
# =====================================================================

def generate_configs():
    """
    ~160 configs covering architecture, training, and feature ablation.

    Section 1: Architecture sweep (fixed features=all, lr=1e-3)
    Section 2: Training hyperparam sweep (fixed arch, features=all)
    Section 3: Feature ablation (fixed arch + training)
    Section 4: Deep dive on best activation patterns
    """
    configs = []
    run_id = 0

    # ── Section 1: Architecture sweep (~60 configs) ──
    # Systematic grid over activation × depth × width
    for act in ["geglu", "gelu", "silu", "relu", "mish"]:
        for n_layers in [2, 3, 4, 5]:
            for hidden_dim in [128, 256, 512]:
                configs.append({
                    "run_id": run_id,
                    "section": "architecture",
                    "name": f"arch_{act}_L{n_layers}_h{hidden_dim}",
                    "activation": act,
                    "n_layers": n_layers,
                    "hidden_dim": hidden_dim,
                    "dropout": 0.15,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "l1_lambda": 0.0,
                    "feature_set": "all",
                })
                run_id += 1

    # Extra wide networks
    for act in ["geglu", "gelu", "silu"]:
        configs.append({
            "run_id": run_id,
            "section": "architecture",
            "name": f"arch_{act}_L3_h1024",
            "activation": act,
            "n_layers": 3,
            "hidden_dim": 1024,
            "dropout": 0.15,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "feature_set": "all",
        })
        run_id += 1

    # Very deep networks
    for act in ["geglu", "gelu", "silu"]:
        for n_layers in [6, 8]:
            configs.append({
                "run_id": run_id,
                "section": "architecture",
                "name": f"arch_{act}_L{n_layers}_h256",
                "activation": act,
                "n_layers": n_layers,
                "hidden_dim": 256,
                "dropout": 0.15,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "l1_lambda": 0.0,
                "feature_set": "all",
            })
            run_id += 1

    # ── Section 2: Training hyperparams (~30 configs) ──
    # Using geglu L3 h256 as reference arch
    for lr in [3e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
        for wd in [1e-5, 1e-4, 1e-3]:
            configs.append({
                "run_id": run_id,
                "section": "training",
                "name": f"train_lr{lr}_wd{wd}",
                "activation": "geglu",
                "n_layers": 3,
                "hidden_dim": 256,
                "dropout": 0.15,
                "lr": lr,
                "weight_decay": wd,
                "l1_lambda": 0.0,
                "feature_set": "all",
            })
            run_id += 1

    # Dropout sweep
    for drop in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]:
        configs.append({
            "run_id": run_id,
            "section": "training",
            "name": f"train_drop{drop}",
            "activation": "geglu",
            "n_layers": 3,
            "hidden_dim": 256,
            "dropout": drop,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "feature_set": "all",
        })
        run_id += 1

    # L1 regularization (feature selection via weight sparsity)
    for l1 in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
        configs.append({
            "run_id": run_id,
            "section": "l1_regularization",
            "name": f"l1_{l1}",
            "activation": "geglu",
            "n_layers": 3,
            "hidden_dim": 256,
            "dropout": 0.15,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "l1_lambda": l1,
            "feature_set": "all",
        })
        run_id += 1

    # ── Section 3: Feature ablation (~30 configs) ──
    # Using geglu L3 h256 as reference arch
    for feat_set in ["all", "bands", "indices", "deltas", "no_deltas",
                     "bands_and_indices", "top500", "top300", "top100"]:
        for act in ["geglu", "gelu", "silu"]:
            configs.append({
                "run_id": run_id,
                "section": "feature_ablation",
                "name": f"feat_{feat_set}_{act}_L3_h256",
                "activation": act,
                "n_layers": 3,
                "hidden_dim": 256,
                "dropout": 0.15,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "l1_lambda": 0.0,
                "feature_set": feat_set,
            })
            run_id += 1

    print(f"Generated {len(configs)} configurations")
    return configs


# =====================================================================
# Main sweep
# =====================================================================

def load_data():
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_core.parquet"))
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))

    feature_cols = [c for c in feat_df.columns
                    if c not in CONTROL_COLS
                    and feat_df[c].dtype in ("float64", "float32", "int64")]

    X = feat_df[feature_cols].values.astype(np.float64)
    y = labels_df[CLASS_NAMES].values.astype(np.float64)
    fold_assignments = split_df["fold_region_growing"].values
    tile_groups = split_df["tile_group"].values

    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    return X, y, fold_assignments, tile_groups, meta["tile_cols"], meta["tile_rows"], feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0,
                        help="Resume from this run_id (skip earlier configs)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    X_all, y, folds, tiles, n_tc, n_tr, feature_cols = load_data()
    print(f"Data: X={X_all.shape}, y={y.shape}")

    # Feature group indices
    feat_groups = partition_features(feature_cols)
    feat_groups["top500"] = top_k_by_variance(X_all, 500).tolist()
    feat_groups["top300"] = top_k_by_variance(X_all, 300).tolist()
    feat_groups["top100"] = top_k_by_variance(X_all, 100).tolist()

    print(f"Feature groups: " + ", ".join(
        f"{k}={len(v)}" for k, v in feat_groups.items()
    ))

    # Train/val/test split (fold 0 = test)
    train_idx, test_idx = get_fold_indices(tiles, folds, 0, n_tc, n_tr, buffer_tiles=1)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_idx))
    n_val = max(int(len(train_idx) * 0.15), 100)
    val_idx = train_idx[perm[:n_val]]
    trn_idx = train_idx[perm[n_val:]]
    print(f"Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    configs = generate_configs()

    # Load existing results if resuming
    os.makedirs(OUT_DIR, exist_ok=True)
    existing_results = []
    completed_names = set()
    if os.path.exists(OUT_CSV):
        existing_df = pd.read_csv(OUT_CSV)
        existing_results = existing_df.to_dict("records")
        completed_names = set(existing_df["name"].values)
        print(f"Found {len(completed_names)} completed configs, resuming...")

    results = existing_results

    t0_total = time.time()

    for cfg in configs:
        if cfg["run_id"] < args.start_from:
            continue
        if cfg["name"] in completed_names:
            print(f"[{cfg['run_id']:3d}] SKIP {cfg['name']} (already done)")
            continue

        # Select features
        feat_set = cfg["feature_set"]
        feat_idx = feat_groups.get(feat_set, feat_groups["all"])
        if len(feat_idx) == 0:
            print(f"[{cfg['run_id']:3d}] SKIP {cfg['name']} (empty feature set)")
            continue

        X = X_all[:, feat_idx]
        n_features = X.shape[1]

        # Scale
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X[trn_idx])
        X_val_s = scaler.transform(X[val_idx])
        X_test_s = scaler.transform(X[test_idx])

        # Build model
        torch.manual_seed(SEED)
        net = FlexibleMLP(
            input_dim=n_features,
            n_classes=N_CLASSES,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            activation=cfg["activation"],
        ).to(device)

        n_params = sum(p.numel() for p in net.parameters())

        # Train
        t0 = time.time()
        epochs, best_val_loss = train_model(
            net, X_trn_s, y[trn_idx], X_val_s, y[val_idx],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            l1_lambda=cfg["l1_lambda"],
            batch_size=1024,
            max_epochs=300,
            patience=20,
            device=device,
        )

        # Evaluate
        net.eval()
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = net.predict(X_test_t).cpu().numpy()
        elapsed = time.time() - t0

        summary, detail = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
        summary.update({
            "name": cfg["name"],
            "section": cfg["section"],
            "activation": cfg["activation"],
            "n_layers": cfg["n_layers"],
            "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "l1_lambda": cfg["l1_lambda"],
            "feature_set": feat_set,
            "n_features": n_features,
            "n_params": n_params,
            "epochs": epochs,
            "best_val_loss": round(best_val_loss, 6),
            "elapsed_s": round(elapsed, 1),
        })

        results.append(summary)

        # Save incrementally
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

        r2 = summary["r2_uniform"]
        mae = summary["mae_mean_pp"]
        print(f"[{cfg['run_id']:3d}] {cfg['name']:40s}  "
              f"R2={r2:.4f}  MAE={mae:.2f}pp  "
              f"feat={n_features:4d}  params={n_params:,}  "
              f"ep={epochs:3d}  {elapsed:.0f}s")

    # Final summary
    total_time = time.time() - t0_total
    df = pd.DataFrame(results)

    print(f"\n{'='*90}")
    print(f"SWEEP COMPLETE: {len(results)} configs in {total_time/3600:.1f} hours")
    print(f"{'='*90}")

    # Top 10 by R2
    df_sorted = df.sort_values("r2_uniform", ascending=False)
    print("\nTOP 10 by R2:")
    print(df_sorted[["name", "r2_uniform", "mae_mean_pp", "aitchison_mean",
                      "n_features", "epochs", "elapsed_s"]].head(10).to_string(index=False))

    # Best per section
    print("\nBEST PER SECTION:")
    for section in df["section"].unique():
        best = df[df["section"] == section].sort_values("r2_uniform", ascending=False).iloc[0]
        print(f"  {section:25s}: {best['name']:40s} R2={best['r2_uniform']:.4f}")

    # Feature importance via L1
    l1_runs = df[df["section"] == "l1_regularization"]
    if len(l1_runs) > 0:
        print("\nL1 REGULARIZATION EFFECT:")
        for _, row in l1_runs.iterrows():
            print(f"  l1={row['l1_lambda']:.0e}: R2={row['r2_uniform']:.4f} MAE={row['mae_mean_pp']:.2f}pp")


if __name__ == "__main__":
    main()
