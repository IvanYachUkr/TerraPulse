#!/usr/bin/env python3
"""
Step 4: Train Model #7 directly with exact hyperparameters.

Skips the full BOHB sweep and trains the winning configuration directly:
  Architecture: T_1024_512_256_64 (TaperedMLP, ~2.5M params)
  Activation:   GELU
  Dropout:      0.3255
  Input dropout: 0.0031
  LR:           0.00103
  Weight decay: 0.000537
  Mixup alpha:  0.2975
  Mixup prob:   0.4285
  Batch size:   4096
  Label thresh: 0.021

Usage:
    python 04_train_model7.py                   # train with max 300 epochs
    python 04_train_model7.py --max-epochs 50   # quick test
"""

import os, sys, time, math, json, pickle, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pyarrow.parquet as pq

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
OUT_DIR = os.path.join(CITIES_DIR, "models_v10_bohb")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_CLASSES = 7
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                "reflectance_scale", "full_features_computed"}

# ===== MODEL #7 EXACT HYPERPARAMETERS (from trial 77 of BOHB sweep) =====
MODEL7_CONFIG = {
    "arch": "T_1024_512_256_64",
    "widths": [1024, 512, 256, 64],
    "activation": "gelu",
    "dropout": 0.3255,
    "input_dropout": 0.0031,
    "lr": 0.00103,
    "weight_decay": 0.000537,
    "mixup_alpha": 0.2975,
    "mixup_prob": 0.4285,
    "label_threshold": 0.021,
    "batch_size": 4096,
}

def ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Import city list from step 1
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
step1 = import_module("01_download_data")
ALL_CITIES = step1.CITIES

EXCLUDED_CITY_NAMES = {
    "nuremberg",
    "ankara_test", "sofia_test", "riga_test", "edinburgh_test", "palermo_test",
}
VAL_CITY_NAMES = {
    "alentejo_portugal", "andalusia_olives", "berlin", "bordeaux",
    "central_spain_plateau", "corsica_interior", "dresden", "dutch_polders",
    "ebro_delta", "estonian_plains", "helsinki", "iceland_highlands",
    "ireland_bog_pasture", "jaen_olives", "madrid", "marseille",
    "northern_sweden", "paris_south", "peloponnese_rural", "po_valley_rural",
    "rostock", "uppland_farmland", "vojvodina_cropland",
}


# ---------------------------------------------------------------------------
# Model architecture (identical to sweep)
# ---------------------------------------------------------------------------
def _make_norm(norm_type, dim):
    if norm_type == "batchnorm": return nn.BatchNorm1d(dim)
    elif norm_type == "layernorm": return nn.LayerNorm(dim)
    return nn.Identity()


class PlainBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15, activation="gelu",
                 norm_type="batchnorm"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = _make_norm(norm_type, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = {
            "gelu": lambda x: F.gelu(x, approximate="tanh"),
            "silu": F.silu, "relu": F.relu, "mish": F.mish,
        }[activation]

    def forward(self, x):
        return self.dropout(self.norm(self.act_fn(self.linear(x))))


class TaperedMLP(nn.Module):
    def __init__(self, in_features, n_classes, widths,
                 dropout=0.15, activation="gelu", input_dropout=0.05,
                 norm_type="batchnorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers, prev = [], in_features
        for w in widths:
            layers.append(PlainBlock(prev, w, dropout, activation, norm_type))
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def export_forward(self, x):
        return torch.softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def normalize_targets(y):
    y = np.clip(y, 0, None).astype(np.float32)
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    y = y / row_sums + 1e-7
    y = y / y.sum(axis=1, keepdims=True)
    return y.astype(np.float32)


def soft_cross_entropy(log_pred, target):
    return -(target * log_pred).sum(dim=-1).mean()


def apply_mixup(xb, yb, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]


# Feature selection
_BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
_INDEX_PREFIXES = {"NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
                   "NDRE1", "NDRE2", "EVI2", "CRI1", "MNDWI", "GNDVI", "NDTI", "IRECI", "TC"}


def build_bi_lbp(feature_cols):
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"): continue
        prefix = col.split("_")[0]
        if prefix in _BAND_PREFIXES or prefix in _INDEX_PREFIXES: selected.append(i)
        elif prefix == "LBP": selected.append(i)
        elif "_pheno_" in col: selected.append(i)
        elif prefix == "SAR": selected.append(i)
    return sorted(set(selected))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Model #7 directly")
    parser.add_argument("--max-epochs", type=int, default=300)
    args = parser.parse_args()

    cfg = MODEL7_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  Training Model #7 (T_1024_512_256_64 GELU)")
    print(f"  Device: {device}, Max epochs: {args.max_epochs}")
    print(f"{'='*70}\n")

    # --- Filter cities with SAR ---
    def city_has_sar(city):
        fp = os.path.join(CITIES_DIR, city.name, "features_v7",
                          "features_rust_2020_2021.parquet")
        if not os.path.exists(fp): return False
        return any(f.name.startswith("SAR_") for f in pq.read_schema(fp))

    sar_cities = [c for c in ALL_CITIES if city_has_sar(c)]
    train_cities = [c for c in sar_cities
                    if c.name not in VAL_CITY_NAMES and c.name not in EXCLUDED_CITY_NAMES]
    val_cities = [c for c in sar_cities if c.name in VAL_CITY_NAMES]
    print(f"  Train: {len(train_cities)} cities, Val: {len(val_cities)} cities")

    # --- Build feature columns ---
    all_col_sets = []
    for city in train_cities + val_cities:
        fp = os.path.join(CITIES_DIR, city.name, "features_v7",
                          "features_rust_2020_2021.parquet")
        if not os.path.exists(fp): continue
        schema = pq.read_schema(fp)
        numeric_types = {'float', 'double', 'int32', 'int64', 'float32', 'float64'}
        cols = {f.name for f in schema
                if any(t in str(f.type).lower() for t in numeric_types)
                and f.name not in CONTROL_COLS}
        all_col_sets.append(cols)

    common = set.intersection(*all_col_sets)
    first_schema = pq.read_schema(os.path.join(
        CITIES_DIR, train_cities[0].name, "features_v7", "features_rust_2020_2021.parquet"))
    full_cols = [f.name for f in first_schema if f.name in common]
    mlp_idx = build_bi_lbp(full_cols)
    mlp_cols = [full_cols[i] for i in mlp_idx]
    n_features = len(mlp_cols)
    print(f"  Features: {n_features}")

    # --- Load training data ---
    print(f"\n[{ts()}] Loading training data...")
    X_parts, y_parts = [], []
    for city in train_cities:
        fp = os.path.join(CITIES_DIR, city.name, "features_v7",
                          "features_rust_2020_2021.parquet")
        lp = os.path.join(CITIES_DIR, city.name, "labels_2021.parquet")
        if not os.path.exists(fp) or not os.path.exists(lp): continue
        df = pd.read_parquet(fp, columns=[c for c in mlp_cols if c != "cell_id"])
        X = np.nan_to_num(df.values.astype(np.float32), 0.0)
        labels = pd.read_parquet(lp)
        y = labels[CLASS_NAMES].values.astype(np.float32)
        rs = y.sum(axis=1, keepdims=True)
        valid = rs.ravel() > 0
        if not valid.all():
            X, y, rs = X[valid], y[valid], rs[valid]
        y = y / np.maximum(rs, 1e-8)
        X_parts.append(X); y_parts.append(y)
        del df, labels, X, y; gc.collect()
        print(f"  [{city.name}] {len(X_parts[-1]):,} cells")

    X_all = np.concatenate(X_parts); del X_parts
    y_all = np.concatenate(y_parts); del y_parts
    gc.collect()
    print(f"  Total: {len(X_all):,} x {n_features}")

    # --- Scale ---
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)
    with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT_DIR, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)

    # --- Apply label threshold ---
    thresh = cfg["label_threshold"]
    y_work = y_all.copy()
    y_work[y_work < thresh] = 0.0
    row_sums = y_work.sum(axis=1, keepdims=True)
    valid_idx = np.where(row_sums.ravel() > 0)[0]
    y_work = y_work[valid_idx]
    y_work = y_work / np.maximum(y_work.sum(axis=1, keepdims=True), 1e-8)
    y_norm = normalize_targets(y_work)
    n_samples = len(y_norm)
    print(f"  After threshold ({thresh}): {n_samples:,} samples")

    # --- Load val data ---
    print(f"\n[{ts()}] Loading validation data...")
    val_X_list, val_y_list = [], []
    for city in val_cities:
        fp = os.path.join(CITIES_DIR, city.name, "features_v7",
                          "features_rust_2020_2021.parquet")
        lp = os.path.join(CITIES_DIR, city.name, "labels_2021.parquet")
        if not os.path.exists(fp) or not os.path.exists(lp): continue
        df = pd.read_parquet(fp, columns=[c for c in mlp_cols if c != "cell_id"])
        X_v = np.nan_to_num(df.values.astype(np.float32), 0.0)
        labels = pd.read_parquet(lp)
        y_v = labels[CLASS_NAMES].values.astype(np.float32)
        rs = y_v.sum(axis=1, keepdims=True)
        valid = rs.ravel() > 0
        if not valid.all():
            X_v, y_v, rs = X_v[valid], y_v[valid], rs[valid]
        y_v = y_v / np.maximum(rs, 1e-8)
        X_v = scaler.transform(X_v).astype(np.float32)
        val_X_list.append(X_v); val_y_list.append(y_v)
        del df, labels; gc.collect()

    X_val = np.concatenate(val_X_list); del val_X_list
    y_val = np.concatenate(val_y_list); del val_y_list
    y_val_norm = normalize_targets(y_val)
    print(f"  Val: {len(X_val):,} cells")

    # Use berlin as early-stop city if available
    val_X_gpu = torch.from_numpy(X_val).to(device)
    val_y_gpu = torch.from_numpy(y_val_norm).to(device)

    # --- Build model ---
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    net = TaperedMLP(n_features, N_CLASSES, cfg["widths"],
                     dropout=cfg["dropout"], activation=cfg["activation"],
                     input_dropout=cfg["input_dropout"],
                     norm_type="batchnorm").to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n  Model: {cfg['arch']} | {n_params:,} params")

    batch_size = cfg["batch_size"]
    try:
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                       weight_decay=cfg["weight_decay"],
                                       fused=(device == "cuda"))
    except TypeError:
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                       weight_decay=cfg["weight_decay"])

    use_amp = device == "cuda"
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = (n_samples + batch_size - 1) // batch_size
    total_steps = args.max_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * 3

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                           total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)),
    ], milestones=[warmup_steps])

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
    patience = max(math.ceil(5000 / steps_per_epoch), 5)
    min_epochs = max(math.ceil(1500 / steps_per_epoch), 3)

    best_val, best_state, wait = float("inf"), None, 0
    rng = np.random.RandomState(SEED)

    print(f"\n[{ts()}] Training...")
    for epoch in range(args.max_epochs):
        net.train()
        perm = np.random.permutation(n_samples)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            x_idx = valid_idx[idx]
            xb = torch.from_numpy(np.array(X_all[x_idx])).to(device, non_blocking=True)
            yb = torch.from_numpy(y_norm[idx]).to(device, non_blocking=True)

            if cfg["mixup_alpha"] > 0 and rng.random() < cfg["mixup_prob"]:
                xb, yb = apply_mixup(xb, yb, cfg["mixup_alpha"])
            if has_bn and xb.size(0) < 2: continue

            optimizer.zero_grad(set_to_none=True)
            amp_device = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(amp_device, enabled=use_amp, dtype=torch.float16):
                loss = soft_cross_entropy(net(xb), yb)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        net.eval()
        with torch.no_grad():
            amp_device = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(amp_device, enabled=use_amp, dtype=torch.float16):
                val_loss = soft_cross_entropy(net(val_X_gpu), val_y_gpu).item()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch <= 3 or epoch % 20 == 0 or improved:
            marker = " *" if improved else ""
            avg = epoch_loss / max(n_batches, 1)
            print(f"  Ep {epoch:3d}: train={avg:.5f} val={val_loss:.5f} wait={wait}{marker}")

        if epoch >= min_epochs and wait >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # Save
    save_path = os.path.join(OUT_DIR, f"trial_77_{cfg['arch']}.pt")
    torch.save(best_state or net.state_dict(), save_path)
    print(f"\n  Model saved: {save_path}")
    print(f"  Best val loss: {best_val:.5f}")

    # Quick eval
    with torch.no_grad():
        preds = net.predict(val_X_gpu).cpu().numpy()
    top1 = (y_val.argmax(1) == preds.argmax(1)).mean()
    r2_vals = []
    for ci in range(N_CLASSES):
        mk = y_val[:, ci] >= 0.01
        if mk.sum() < 50: continue
        yt = y_val[mk, ci]
        if np.var(yt) < 1e-8: continue
        r2_vals.append(r2_score(yt, preds[mk, ci]))
    mean_r2 = np.mean(r2_vals) if r2_vals else 0
    combined = 0.5 * top1 + 0.5 * max(0, mean_r2)

    print(f"\n  Val Results:")
    print(f"    Top-1 accuracy: {top1:.4f} ({top1*100:.2f}%)")
    print(f"    Mean R²:        {mean_r2:.4f}")
    print(f"    Combined:       {combined:.4f}")
    print(f"\n  Expected (from original sweep):")
    print(f"    Combined ~0.6678, deployed on ONNX with threshold=0.021")

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
