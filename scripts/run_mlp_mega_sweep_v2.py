"""
MLP Mega Sweep V2: Full feature set + texture features.

Builds on overnight sweep findings:
- Bands+indices (798 core) beat all 2109 core features
- GeGLU trained best small model, ReLU/SiLU best overall
- Now tests: full feature set (3535), texture-only subsets (HOG, GLCM, etc.),
  and combinations of core winners + texture features

Feature sets tested:
- bands_indices (798 core):           baseline winner
- texture_all (570 full):             HOG+Gabor+LBP+GLCM+MP+SV
- bands_indices_texture (1368):       core winner + all texture
- bands_indices_hog (1002):           core winner + HOG
- bands_indices_glcm_lbp (924):       core winner + GLCM + LBP
- full_no_deltas (1428 full):         everything except delta
- top500_full, top300_full:           top-K from full set by variance
- extra_only (1425):                  all full-only features (no core)
- all_full (3535):                    everything

Saves to reports/phase8/tables/mlp_mega_sweep_v2.csv

Usage:
    python scripts/run_mlp_mega_sweep_v2.py
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
OUT_CSV = os.path.join(OUT_DIR, "mlp_mega_sweep_v2.csv")

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}


# =====================================================================
# Flexible MLP (same as v1, copied for independence)
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
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.silu(self.linear(x))))


class ReLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.relu(self.linear(x))))


class GELUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.gelu(self.linear(x))))


class MishBlock(nn.Module):
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
# Trainer
# =====================================================================

def train_model(net, X_trn, y_trn, X_val, y_val,
                lr=1e-3, weight_decay=1e-4,
                batch_size=1024, max_epochs=300, patience=20,
                device="cpu"):
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
    n_epochs_done = 0

    for epoch in range(max_epochs):
        net.train()
        epoch_loss = 0.0
        n_b = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            log_pred = net(xb)
            loss = F.kl_div(log_pred, yb, reduction="batchmean", log_target=False)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        n_epochs_done = epoch + 1

        net.eval()
        with torch.no_grad():
            log_val = net(X_vt)
            val_loss = F.kl_div(log_val, y_vt, reduction="batchmean", log_target=False).item()
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

    return n_epochs_done, best_val


# =====================================================================
# Feature group partitioning
# =====================================================================

def partition_features(core_cols, full_cols):
    """Split columns into semantic groups spanning both core and full."""
    band_prefixes = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
    index_prefixes = {
        "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
        "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
    }

    # Core groups (by index into full_cols)
    bands_idx, indices_idx, deltas_idx = [], [], []
    hog_idx, gabor_idx, lbp_idx, glcm_idx, mp_idx, sv_idx = [], [], [], [], [], []
    other_idx = []

    for i, col in enumerate(full_cols):
        prefix = col.split("_")[0]
        if col.startswith("delta"):
            deltas_idx.append(i)
        elif prefix in band_prefixes:
            bands_idx.append(i)
        elif prefix in index_prefixes:
            indices_idx.append(i)
        elif prefix == "HOG":
            hog_idx.append(i)
        elif prefix == "Gabor":
            gabor_idx.append(i)
        elif prefix == "LBP":
            lbp_idx.append(i)
        elif prefix == "GLCM":
            glcm_idx.append(i)
        elif prefix == "MP":
            mp_idx.append(i)
        elif prefix == "SV":
            sv_idx.append(i)
        else:
            other_idx.append(i)

    texture_all = hog_idx + gabor_idx + lbp_idx + glcm_idx + mp_idx + sv_idx
    bands_indices = bands_idx + indices_idx

    core_set = set(core_cols)
    extra_only = [i for i, c in enumerate(full_cols) if c not in core_set]

    return {
        "bands_indices": bands_indices,
        "texture_all": texture_all,
        "bands_indices_texture": bands_indices + texture_all,
        "bands_indices_hog": bands_indices + hog_idx,
        "bands_indices_glcm_lbp": bands_indices + glcm_idx + lbp_idx,
        "full_no_deltas": bands_indices + texture_all + other_idx,
        "extra_only": extra_only,
        "all_full": list(range(len(full_cols))),
        "hog_only": hog_idx,
        "glcm_only": glcm_idx,
        "gabor_only": gabor_idx,
    }


def top_k_by_variance(X, k):
    var = np.var(X, axis=0)
    return np.argsort(var)[-k:][::-1].tolist()


# =====================================================================
# Config generator
# =====================================================================

def generate_configs():
    configs = []
    run_id = 0

    # All feature sets to test
    feature_sets = [
        "bands_indices",
        "texture_all",
        "bands_indices_texture",
        "bands_indices_hog",
        "bands_indices_glcm_lbp",
        "full_no_deltas",
        "extra_only",
        "all_full",
        "top500_full",
        "top300_full",
        "hog_only",
        "glcm_only",
        "gabor_only",
    ]

    # Architecture configs to sweep per feature set
    arch_configs = [
        # (activation, n_layers, hidden_dim, dropout, lr)
        ("geglu", 3, 256, 0.15, 1e-3),   # best small
        ("silu", 3, 256, 0.15, 1e-3),     # best overall
        ("relu", 3, 256, 0.15, 1e-3),     # close second
        ("gelu", 3, 256, 0.15, 1e-3),     # comparison
        ("geglu", 3, 128, 0.15, 1e-3),    # smaller
        ("silu", 3, 512, 0.15, 1e-3),     # bigger
        ("relu", 5, 256, 0.15, 1e-3),     # deeper
        ("silu", 4, 256, 0.15, 1e-3),     # medium deep
        ("geglu", 2, 256, 0.15, 1e-3),    # shallow
        ("relu", 3, 256, 0.10, 1e-3),     # less dropout
        ("silu", 3, 256, 0.15, 5e-3),     # higher LR
        ("geglu", 3, 256, 0.15, 5e-4),    # lower LR
    ]

    for feat_set in feature_sets:
        for act, nl, hd, drop, lr in arch_configs:
            name = f"{feat_set}_{act}_L{nl}_h{hd}"
            if drop != 0.15:
                name += f"_d{drop}"
            if lr != 1e-3:
                name += f"_lr{lr}"

            configs.append({
                "run_id": run_id,
                "name": name,
                "feature_set": feat_set,
                "activation": act,
                "n_layers": nl,
                "hidden_dim": hd,
                "dropout": drop,
                "lr": lr,
                "weight_decay": 1e-4,
            })
            run_id += 1

    print(f"Generated {len(configs)} configurations")
    return configs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load FULL feature set
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))

    # Also load core columns for reference
    core_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_core.parquet"))
    core_cols = [c for c in core_df.columns
                 if c not in CONTROL_COLS and core_df[c].dtype in ("float64", "float32", "int64")]
    del core_df  # free memory

    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS
                         and feat_df[c].dtype in ("float64", "float32", "int64")]

    X_all = feat_df[full_feature_cols].values.astype(np.float64)
    y = labels_df[CLASS_NAMES].values.astype(np.float64)
    del feat_df  # free memory

    print(f"Full data: X={X_all.shape}, y={y.shape}")

    folds = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        m = json.load(f)

    # Feature groups
    feat_groups = partition_features(core_cols, full_feature_cols)
    feat_groups["top500_full"] = top_k_by_variance(X_all, 500)
    feat_groups["top300_full"] = top_k_by_variance(X_all, 300)

    print("Feature groups: " + ", ".join(f"{k}={len(v)}" for k, v in feat_groups.items()))

    # Train/val/test split
    train_idx, test_idx = get_fold_indices(tiles, folds, 0, m["tile_cols"], m["tile_rows"], buffer_tiles=1)
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
            continue

        feat_set = cfg["feature_set"]
        feat_idx = feat_groups.get(feat_set)
        if feat_idx is None or len(feat_idx) == 0:
            print(f"[{cfg['run_id']:3d}] SKIP {cfg['name']} (unknown/empty feature set)")
            continue

        X = X_all[:, feat_idx]
        n_features = X.shape[1]

        # Scale
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X[trn_idx])
        X_val_s = scaler.transform(X[val_idx])
        X_test_s = scaler.transform(X[test_idx])

        # Build + train
        torch.manual_seed(SEED)
        net = FlexibleMLP(
            input_dim=n_features, n_classes=N_CLASSES,
            hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], activation=cfg["activation"],
        ).to(device)

        n_params = sum(p.numel() for p in net.parameters())

        t0 = time.time()
        epochs, best_val_loss = train_model(
            net, X_trn_s, y[trn_idx], X_val_s, y[val_idx],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            batch_size=1024, max_epochs=300, patience=20, device=device,
        )

        # Evaluate
        net.eval()
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = net.predict(X_test_t).cpu().numpy()
        elapsed = time.time() - t0

        summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
        summary.update({
            "name": cfg["name"],
            "feature_set": feat_set,
            "activation": cfg["activation"],
            "n_layers": cfg["n_layers"],
            "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "n_features": n_features,
            "n_params": n_params,
            "epochs": epochs,
            "best_val_loss": round(best_val_loss, 6),
            "elapsed_s": round(elapsed, 1),
        })

        results.append(summary)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

        r2 = summary["r2_uniform"]
        mae = summary["mae_mean_pp"]
        print(f"[{cfg['run_id']:3d}] {cfg['name']:55s}  "
              f"R2={r2:.4f}  MAE={mae:.2f}pp  feat={n_features:4d}  "
              f"ep={epochs:3d}  {elapsed:.0f}s")

    # Final summary
    total_time = time.time() - t0_total
    df = pd.DataFrame(results)

    print(f"\n{'='*90}")
    print(f"SWEEP V2 COMPLETE: {len(results)} configs in {total_time/3600:.1f} hours")
    print(f"{'='*90}")

    df_sorted = df.sort_values("r2_uniform", ascending=False)
    print("\nTOP 15 by R2:")
    print(df_sorted[["name", "r2_uniform", "mae_mean_pp", "aitchison_mean",
                      "n_features", "epochs", "elapsed_s"]].head(15).to_string(index=False))

    print("\nBEST PER FEATURE SET:")
    for fs in df["feature_set"].unique():
        best = df[df["feature_set"] == fs].sort_values("r2_uniform", ascending=False).iloc[0]
        print(f"  {fs:30s}: R2={best['r2_uniform']:.4f}  feat={best['n_features']}  "
              f"arch={best['activation']}_L{best['n_layers']}_h{best['hidden_dim']}")


if __name__ == "__main__":
    main()
