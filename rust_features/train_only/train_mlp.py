#!/usr/bin/env python3
"""
Production MLP Training — Champion Config (standalone, speed-optimized).

Champion: bi_LBP_plain_silu_L5_d1024_bn
  - PlainMLP: 5 layers, d=1024, SiLU, BatchNorm, dropout=0.15
  - Training: cosine warmup (3 ep), AMP, fused AdamW, patience=5000 steps
  - Features: bands_indices + all LBP columns

Speed optimizations applied:
  - AMP (fp16) on CUDA
  - Fused AdamW kernel
  - TF32 matmul
  - torch.compile (if PyTorch 2.x)
  - set_to_none=True for zero_grad
  - Non-blocking GPU transfers
  - Batched inference for large test sets

Fully standalone — no project imports required.

Produces:
    models/final_mlp_prod/fold_{i}.pt
    models/final_mlp_prod/scaler_{i}.pkl
    models/final_mlp_prod/oof_predictions.parquet
    models/final_mlp_prod/meta.json
"""

import json
import math
import os
import pickle
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# =====================================================================
# Constants
# =====================================================================

SEED = 42
N_FOLDS = 5
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}

# Feature group prefixes (for partition_features / build_bi_lbp)
BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
}

# =====================================================================
# Model Architecture (exact champion config)
# =====================================================================

class PlainBlock(nn.Module):
    """Linear -> SiLU -> BatchNorm -> Dropout."""

    def __init__(self, in_dim, out_dim, dropout=0.15, activation="silu",
                 norm_type="batchnorm"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act_fn = {
            "gelu": lambda x: F.gelu(x, approximate="tanh"),
            "silu": F.silu, "relu": F.relu, "mish": F.mish,
        }[activation]

    def forward(self, x):
        return self.dropout(self.norm(self.act_fn(self.linear(x))))


class PlainMLP(nn.Module):
    def __init__(self, in_features, n_classes, hidden=1024, n_layers=5,
                 dropout=0.15, activation="silu", norm_type="batchnorm"):
        super().__init__()
        layers = [PlainBlock(in_features, hidden, dropout, activation, norm_type)]
        for _ in range(n_layers - 1):
            layers.append(PlainBlock(hidden, hidden, dropout, activation, norm_type))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(x)), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# =====================================================================
# Training Utilities
# =====================================================================

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.001, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def normalize_targets(y):
    """Clip, normalize to simplex, add label smoothing."""
    y = np.clip(y, 0, None).astype(np.float32)
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    y = y / row_sums
    y = y + 1e-7
    y = y / y.sum(axis=1, keepdims=True)
    return y


def soft_cross_entropy(log_pred, target):
    return -(target * log_pred).sum(dim=-1).mean()


def refresh_bn_stats(model, X, batch_size=2048):
    """Re-compute BatchNorm running stats after SWA (not used here but kept)."""
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not bn_layers:
        return
    model.eval()
    for bn in bn_layers:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            _ = model(X[i:i + batch_size])
    model.eval()


def predict_batched(model, X_cpu, device, batch_size=65536):
    """Batched inference — handles large test sets without OOM."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, X_cpu.size(0), batch_size):
            xb = X_cpu[i:i + batch_size].to(device, non_blocking=True)
            parts.append(model.predict(xb).cpu())
            del xb
    return torch.cat(parts, dim=0).numpy()


def train_model(net, X_trn, y_trn, X_val, y_val, *,
                lr=1e-3, weight_decay=1e-4, batch_size=2048,
                max_epochs=2000, patience_steps=5000, min_steps=2000):
    """Train with AMP, fused AdamW, cosine warmup, step-based early stopping."""
    device = X_trn.device
    use_amp = X_trn.is_cuda

    # Fused AdamW (PyTorch 2.x CUDA) — faster kernel
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay, fused=use_amp,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n = X_trn.size(0)
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = max_epochs * steps_per_epoch
    scheduler = cosine_warmup_scheduler(optimizer, steps_per_epoch * 3, total_steps)

    patience_epochs = max(math.ceil(patience_steps / steps_per_epoch), 5)
    min_epochs = max(math.ceil(min_steps / steps_per_epoch), 3)

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=device)

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_trn[idx], y_trn[idx]
            if has_bn and xb.size(0) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logp = net(xb)
                loss = soft_cross_entropy(logp, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validation
        net.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            val_loss = soft_cross_entropy(net(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if (epoch + 1) >= min_epochs and wait >= patience_epochs:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    return epoch + 1, best_val, net


# =====================================================================
# Spatial Splitting (inlined from src/splitting.py)
# =====================================================================

@lru_cache(maxsize=8)
def _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles):
    neighbors = {}
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile_id = tr * n_tile_cols + tc
            nbrs = set()
            for dr in range(-buffer_tiles, buffer_tiles + 1):
                for dc in range(-buffer_tiles, buffer_tiles + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                        nbrs.add(nr * n_tile_cols + nc)
            neighbors[tile_id] = frozenset(nbrs)
    return neighbors


def get_fold_indices(groups, fold_assignments, fold_idx,
                     n_tile_cols, n_tile_rows, buffer_tiles=1):
    """Train/test split with spatial buffer to prevent leakage."""
    test_mask = fold_assignments == fold_idx

    if buffer_tiles > 0:
        test_tiles = set(np.unique(groups[test_mask]))
        tile_neighbors = _precompute_tile_neighbors(
            n_tile_cols, n_tile_rows, buffer_tiles
        )
        buffer_tiles_set = set()
        for tt in test_tiles:
            for nbr in tile_neighbors.get(tt, set()):
                if nbr not in test_tiles:
                    buffer_tiles_set.add(nbr)
        buffer_mask = np.isin(groups, list(buffer_tiles_set))
        train_mask = (~test_mask) & (~buffer_mask)
    else:
        train_mask = ~test_mask

    return np.where(train_mask)[0], np.where(test_mask)[0]


# =====================================================================
# Evaluation (inlined from src/models/evaluation.py)
# =====================================================================

def evaluate_predictions(y_true, y_pred, class_names):
    """Compute R2 (uniform), MAE (pp), and per-class metrics."""
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    mae = float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")) * 100
    per_class = {
        cn: {
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
            "mae_pp": float(mean_absolute_error(y_true[:, i], y_pred[:, i])) * 100,
        }
        for i, cn in enumerate(class_names)
    }
    return {"r2_uniform": r2, "mae_mean_pp": mae, "per_class": per_class}


# =====================================================================
# Feature Selection
# =====================================================================

def build_bi_lbp(feature_cols):
    """Select bands + indices + TC + all LBP columns (matching champion config)."""
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in BAND_PREFIXES or prefix in INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
    return sorted(set(selected))


# =====================================================================
# Main
# =====================================================================

def main():
    t_start = time.time()

    # ── Paths (auto-detect project root) ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Walk up to find project root (contains data/ directory)
    project_root = script_dir
    for _ in range(5):
        if os.path.isdir(os.path.join(project_root, "data")):
            break
        project_root = os.path.dirname(project_root)

    processed_dir = os.path.join(project_root, "data", "processed", "v2")
    model_dir = os.path.join(project_root, "models", "final_mlp_prod")
    os.makedirs(model_dir, exist_ok=True)

    # ── Device setup ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── Load data ──
    print("Loading data...")

    # Try Rust production parquet first, fall back to standard
    feat_path = os.path.join(processed_dir, "features_rust_production.parquet")
    if not os.path.exists(feat_path):
        feat_path = os.path.join(processed_dir, "features_merged_full.parquet")
    print(f"  Features: {os.path.basename(feat_path)}")

    feat_df = pd.read_parquet(feat_path)
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [
        c for c in feat_df.columns
        if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])
    ]

    # Build bi_LBP feature set
    fs_idx = build_bi_lbp(full_feature_cols)
    fs_cols = [full_feature_cols[i] for i in fs_idx]
    n_features = len(fs_idx)
    print(f"  Features selected: {n_features} (bi_LBP)")

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    np.nan_to_num(X_all, copy=False)
    cell_ids = feat_df["cell_id"].values if "cell_id" in feat_df.columns else np.arange(len(feat_df))
    del feat_df

    labels_df = pd.read_parquet(os.path.join(processed_dir, "labels_2021.parquet"))
    y = labels_df[CLASS_NAMES].values.astype(np.float32)
    del labels_df

    split_df = pd.read_parquet(os.path.join(processed_dir, "split_spatial.parquet"))
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    del split_df

    with open(os.path.join(processed_dir, "split_spatial_meta.json")) as f:
        split_meta = json.load(f)

    print(f"  Data: {X_all.shape[0]} cells, {y.shape[1]} classes")

    # ── Train 5 folds ──
    oof_preds = np.full((len(y), N_CLASSES), np.nan, dtype=np.float32)
    fold_metrics = []

    for fold_id in range(N_FOLDS):
        print(f"\n{'='*50}")
        print(f"Fold {fold_id}")
        print(f"{'='*50}")

        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id,
            split_meta["tile_cols"], split_meta["tile_rows"],
            buffer_tiles=1,
        )

        # Validation split from training set
        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx = train_idx[perm[:n_val]]
        trn_idx = train_idx[perm[n_val:]]
        print(f"  Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # Feature subset + scaling
        X_fs = X_all[:, fs_idx]
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X_fs[trn_idx]).astype(np.float32)
        X_val_s = scaler.transform(X_fs[val_idx]).astype(np.float32)
        X_test_s = scaler.transform(X_fs[test_idx]).astype(np.float32)

        # Save scaler
        with open(os.path.join(model_dir, f"scaler_{fold_id}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        # To GPU
        X_trn_t = torch.tensor(X_trn_s, dtype=torch.float32).to(device, non_blocking=True)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device, non_blocking=True)
        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device, non_blocking=True)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device, non_blocking=True)

        # Build model
        torch.manual_seed(SEED + fold_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold_id)

        net = PlainMLP(
            n_features, N_CLASSES,
            hidden=1024, n_layers=5,
            dropout=0.15, activation="silu", norm_type="batchnorm",
        ).to(device)

        # Optional: torch.compile for PyTorch 2.x
        compiled_net = net
        try:
            compiled_net = torch.compile(net, mode="reduce-overhead")
            print("  torch.compile enabled (reduce-overhead)")
        except Exception:
            compiled_net = net

        # Train
        t0 = time.time()
        n_epochs, best_val, trained_net = train_model(
            compiled_net, X_trn_t, y_trn_t, X_val_t, y_val_t,
            lr=1e-3, weight_decay=1e-4, batch_size=2048,
            max_epochs=2000, patience_steps=5000, min_steps=2000,
        )
        elapsed = time.time() - t0

        # Save model (unwrap compiled if needed)
        save_net = trained_net
        if hasattr(trained_net, "_orig_mod"):
            save_net = trained_net._orig_mod
        torch.save(save_net.state_dict(), os.path.join(model_dir, f"fold_{fold_id}.pt"))

        # OOF predictions
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        preds = predict_batched(trained_net, X_test_t, device)
        oof_preds[test_idx] = preds

        metrics = evaluate_predictions(y[test_idx], preds, CLASS_NAMES)
        r2 = metrics["r2_uniform"]
        mae = metrics["mae_mean_pp"]
        fold_metrics.append({
            "fold": fold_id, "r2": r2, "mae": mae,
            "epochs": n_epochs, "val_loss": best_val, "time_s": round(elapsed, 1),
        })
        print(f"  R2={r2:.4f}  MAE={mae:.2f}pp  epochs={n_epochs}  time={elapsed:.0f}s")

        # Cleanup
        del net, compiled_net, trained_net, X_trn_t, X_val_t, X_test_t, y_trn_t, y_val_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save OOF predictions ──
    oof_df = pd.DataFrame(oof_preds, columns=[c + "_pred" for c in CLASS_NAMES])
    oof_df.insert(0, "cell_id", cell_ids)
    oof_df.to_parquet(os.path.join(model_dir, "oof_predictions.parquet"), index=False)

    # ── Save metadata ──
    r2_vals = [m["r2"] for m in fold_metrics]
    mae_vals = [m["mae"] for m in fold_metrics]

    meta = {
        "model": "PlainMLP",
        "config": "bi_LBP_plain_silu_L5_d1024_bn",
        "feature_set": "bi_LBP",
        "feature_cols": fs_cols,
        "n_features": n_features,
        "seed": SEED,
        "r2_mean": round(float(np.mean(r2_vals)), 4),
        "r2_std": round(float(np.std(r2_vals)), 4),
        "mae_mean_pp": round(float(np.mean(mae_vals)), 2),
        "fold_metrics": fold_metrics,
        "architecture": {
            "arch": "plain", "activation": "silu",
            "n_layers": 5, "d_model": 1024, "norm": "batchnorm",
            "dropout": 0.15,
        },
        "training": {
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 2048,
            "max_epochs": 2000, "patience_steps": 5000, "min_steps": 2000,
            "scheduler": "cosine_warmup_3ep",
        },
        "data_source": os.path.basename(feat_path),
        "speed_optimizations": [
            "AMP_fp16", "fused_AdamW", "TF32_matmul",
            "torch_compile", "set_to_none_zero_grad",
        ],
    }
    with open(os.path.join(model_dir, f"meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ──
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"FINAL MLP CHAMPION — 5-fold CV")
    print(f"{'='*60}")
    for m in fold_metrics:
        print(f"  Fold {m['fold']}: R2={m['r2']:.4f}  MAE={m['mae']:.2f}pp  "
              f"epochs={m['epochs']}  time={m['time_s']}s")
    print(f"  {'-'*40}")
    print(f"  Mean: R2={np.mean(r2_vals):.4f} +/- {np.std(r2_vals):.4f}  "
          f"MAE={np.mean(mae_vals):.2f}pp")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.0f}s")
    print(f"Saved to: {model_dir}")


if __name__ == "__main__":
    main()
