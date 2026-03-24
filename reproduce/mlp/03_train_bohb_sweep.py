#!/usr/bin/env python3
"""
Step 3: Full BOHB hyperparameter sweep for the MLP model.

Reproduces the V10 BOHB sweep that found trial #77 (deployed model).
Searches over 4 architectures, dropout, LR, mixup, activation, etc.

Requirements: pip install hpbandster ConfigSpace serpent

Usage:
    python 03_train_bohb_sweep.py                           # full 100 trials
    python 03_train_bohb_sweep.py --max-trials 5 --max-budget 20  # quick test
"""

import os, sys, time, math, json, pickle, gc, logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pyarrow.parquet as pq

# BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

# Fix numpy serialization for BOHB
import serpent
for np_type in [np.int32, np.int64, np.float32, np.float64,
                np.bool_, np.str_, np.bytes_, np.ndarray, np.intc, np.intp]:
    try:
        def _ser(obj, s, o, i):
            if isinstance(obj, np.integer): s._serialize(int(obj), o, i)
            elif isinstance(obj, np.floating): s._serialize(float(obj), o, i)
            elif isinstance(obj, np.bool_): s._serialize(bool(obj), o, i)
            elif isinstance(obj, (np.str_, np.bytes_)): s._serialize(str(obj), o, i)
            elif isinstance(obj, np.ndarray): s._serialize(obj.tolist(), o, i)
            else: s._serialize(str(obj), o, i)
        serpent.register_class(np_type, _ser)
    except Exception:
        pass

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
V10_DIR = os.path.join(CITIES_DIR, "models_v10_bohb")
os.makedirs(V10_DIR, exist_ok=True)

SEED = 42
N_CLASSES = 7
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                "reflectance_scale", "full_features_computed"}

def ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# City definitions and splits
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
# Model architecture
# ---------------------------------------------------------------------------
def _make_norm(norm_type, dim):
    if norm_type == "layernorm": return nn.LayerNorm(dim)
    elif norm_type == "batchnorm": return nn.BatchNorm1d(dim)
    else: return nn.Identity()


class PlainBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15, activation="gelu",
                 norm_type="layernorm"):
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
                 dropout=0.15, activation="silu", input_dropout=0.05,
                 norm_type="batchnorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = []
        prev_dim = in_features
        for w in widths:
            layers.append(PlainBlock(prev_dim, w, dropout, activation, norm_type))
            prev_dim = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
BASE_ARCHS = {
    "T_512_256_128_64":   [512, 256, 128, 64],
    "T_1024_512_256_64":  [1024, 512, 256, 64],
    "T_2048_512_128":     [2048, 512, 128],
    "T_2048_1024_512":    [2048, 1024, 512],
}


def normalize_targets(y):
    y = np.clip(y, 0, None).astype(np.float32)
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    y = y / row_sums
    y = y + 1e-7
    y = y / y.sum(axis=1, keepdims=True)
    return y.astype(np.float32)


def soft_cross_entropy(log_pred, target):
    return -(target * log_pred).sum(dim=-1).mean()


def apply_mixup(xb, yb, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
_BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                  "B11", "B12"}
_INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI2", "CRI1", "MNDWI", "GNDVI", "NDTI", "IRECI", "TC",
}


def build_bi_lbp(feature_cols):
    """Feature selection: bands + indices + LBP + SAR + pheno."""
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in _BAND_PREFIXES or prefix in _INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
        elif "_pheno_" in col:
            selected.append(i)
        elif prefix == "SAR":
            selected.append(i)
    return sorted(set(selected))


def city_has_sar(city):
    feat_path = os.path.join(CITIES_DIR, city.name, "features_v7",
                             "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        return False
    schema = pq.read_schema(feat_path)
    return any(f.name.startswith("SAR_") for f in schema)


def city_feature_cols(city):
    feat_path = os.path.join(CITIES_DIR, city.name, "features_v7",
                             "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        return None
    schema = pq.read_schema(feat_path)
    numeric_types = {'float', 'double', 'int32', 'int64', 'float32', 'float64'}
    cols = set()
    for f in schema:
        type_str = str(f.type).lower()
        if any(t in type_str for t in numeric_types) and f.name not in CONTROL_COLS:
            cols.add(f.name)
    return cols


def load_city_arrays(city, columns):
    feat_path = os.path.join(CITIES_DIR, city.name, "features_v7",
                             "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        return None
    cols_to_read = [c for c in columns if c != "cell_id"]
    df = pd.read_parquet(feat_path, columns=cols_to_read)
    arr = np.nan_to_num(df.values.astype(np.float32), 0.0)
    n_cells = len(df)
    del df
    return arr, n_cells


def city_labels_path(city, year=2021):
    return os.path.join(CITIES_DIR, city.name, f"labels_{year}.parquet")


# ---------------------------------------------------------------------------
# Setup data
# ---------------------------------------------------------------------------
print(f"[{ts()}] V10 BOHB — filtering cities with SAR features...")
ALL_SAR_CITIES = [c for c in ALL_CITIES if city_has_sar(c)]
TRAIN_CITIES = [c for c in ALL_SAR_CITIES
                if c.name not in VAL_CITY_NAMES and c.name not in EXCLUDED_CITY_NAMES]
VAL_CITIES = [c for c in ALL_SAR_CITIES if c.name in VAL_CITY_NAMES]
print(f"  Train cities: {len(TRAIN_CITIES)}")
print(f"  Val cities: {len(VAL_CITIES)}")

# Feature columns = intersection across all cities
print(f"\n[{ts()}] Building feature column intersection...")
all_col_sets = []
for city in TRAIN_CITIES + VAL_CITIES:
    cols = city_feature_cols(city)
    if cols is not None:
        all_col_sets.append(cols)

common_cols = set.intersection(*all_col_sets) if all_col_sets else set()
first_schema = pq.read_schema(os.path.join(
    CITIES_DIR, TRAIN_CITIES[0].name, "features_v7", "features_rust_2020_2021.parquet"))
full_feature_cols = [f.name for f in first_schema if f.name in common_cols]
mlp_idx = build_bi_lbp(full_feature_cols)
mlp_cols = [full_feature_cols[i] for i in mlp_idx]
n_features = len(mlp_cols)
print(f"  MLP features: {n_features}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

# Load training data to memmap
print(f"\n[{ts()}] Loading training data (memmap)...")
city_counts = []
train_cities_valid = []
for city in TRAIN_CITIES:
    feat_path = os.path.join(CITIES_DIR, city.name, "features_v7",
                             "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        continue
    n = pq.read_metadata(feat_path).num_rows
    city_counts.append(n)
    train_cities_valid.append(city)

total_est = sum(city_counts)
_mmap_dir = os.path.join(V10_DIR, "_mmap_cache")
os.makedirs(_mmap_dir, exist_ok=True)
X_mmap_path = os.path.join(_mmap_dir, "X_train.dat")
y_mmap_path = os.path.join(_mmap_dir, "y_train.dat")
X_train_global = np.memmap(X_mmap_path, dtype=np.float32, mode='w+',
                            shape=(total_est, n_features))
y_train_global = np.memmap(y_mmap_path, dtype=np.float32, mode='w+',
                            shape=(total_est, N_CLASSES))

offset = 0
for ci, (city, n) in enumerate(zip(train_cities_valid, city_counts)):
    result = load_city_arrays(city, mlp_cols)
    if result is None:
        continue
    X_city, _ = result
    label_path = city_labels_path(city, 2021)
    if not os.path.exists(label_path):
        del X_city
        continue
    labels = pd.read_parquet(label_path)
    y_city = labels[CLASS_NAMES].values.astype(np.float32)
    del labels
    row_sums = y_city.sum(axis=1, keepdims=True)
    valid = (row_sums.ravel() > 0)
    if not valid.all():
        X_city = X_city[valid]
        y_city = y_city[valid]
        row_sums = row_sums[valid]
    y_city = y_city / np.maximum(row_sums, 1e-8)
    actual_n = min(n, X_city.shape[0], y_city.shape[0])
    X_train_global[offset:offset + actual_n] = X_city[:actual_n]
    y_train_global[offset:offset + actual_n] = y_city[:actual_n]
    del X_city, y_city
    offset += actual_n
    gc.collect()
    print(f"  [{city.name}] {actual_n:,} cells")

if offset < total_est:
    print(f"  Trimming from {total_est:,} to {offset:,}")
X_train_global.flush()
y_train_global.flush()
X_train_global = np.memmap(X_mmap_path, dtype=np.float32, mode='r+',
                            shape=(offset, n_features))
y_train_global = np.memmap(y_mmap_path, dtype=np.float32, mode='r+',
                            shape=(offset, N_CLASSES))
total = offset
print(f"  Total: {total:,} x {n_features}")

# Scale
print(f"\n[{ts()}] Fitting scaler...")
scaler = StandardScaler()
SCALER_CHUNK = 200_000
for sc in range(0, total, SCALER_CHUNK):
    scaler.partial_fit(X_train_global[sc:sc + SCALER_CHUNK])
scaler_mean = scaler.mean_.astype(np.float32)
scaler_scale = scaler.scale_.astype(np.float32)
X_train_global -= scaler_mean
X_train_global /= scaler_scale

with open(os.path.join(V10_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(V10_DIR, "mlp_cols.json"), "w") as f:
    json.dump(mlp_cols, f)

# Load val data to GPU
print(f"\n[{ts()}] Loading val cities to VRAM...")
val_tensors = {}
for city in VAL_CITIES:
    result = load_city_arrays(city, mlp_cols)
    if result is None:
        continue
    X_v, _ = result
    label_path = city_labels_path(city, 2021)
    if not os.path.exists(label_path):
        del X_v
        continue
    labels = pd.read_parquet(label_path)
    y_v = labels[CLASS_NAMES].values.astype(np.float32)
    del labels
    row_sums = y_v.sum(axis=1, keepdims=True)
    valid = (row_sums.ravel() > 0)
    if not valid.all():
        X_v = X_v[valid]
        y_v = y_v[valid]
        row_sums = row_sums[valid]
    y_v = y_v / np.maximum(row_sums, 1e-8)
    X_v = (X_v - scaler_mean) / scaler_scale
    y_v_norm = normalize_targets(y_v)
    val_tensors[city.name] = {
        "X": torch.from_numpy(X_v).to(device),
        "y_norm": torch.from_numpy(y_v_norm).to(device),
        "y_raw": y_v,
    }
    del X_v, y_v_norm
    print(f"  [{city.name}] {val_tensors[city.name]['X'].shape[0]:,} cells -> VRAM")
    gc.collect()

val_name = "berlin"
if val_name not in val_tensors:
    val_name = list(val_tensors.keys())[0]
print(f"  Early-stop city: {val_name}")
gc.collect()


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def compute_objective(net, val_data, label_threshold, dev):
    all_top1_correct, all_top1_total, all_r2_values = 0, 0, []
    net.eval()
    for city_name, data in val_data.items():
        with torch.no_grad():
            preds = net.predict(data["X"]).cpu().numpy()
        y_raw = data["y_raw"]
        true_top1 = y_raw.argmax(axis=1)
        pred_top1 = preds.argmax(axis=1)
        all_top1_correct += (true_top1 == pred_top1).sum()
        all_top1_total += len(true_top1)
        for cls_i in range(N_CLASSES):
            yt = y_raw[:, cls_i]
            mask = yt >= label_threshold
            if mask.sum() < 50: continue
            yt_masked = yt[mask]
            if np.var(yt_masked) < 1e-8: continue
            all_r2_values.append(r2_score(yt_masked, preds[mask, cls_i]))
    top1_acc = all_top1_correct / max(all_top1_total, 1)
    mean_r2 = np.mean(all_r2_values) if all_r2_values else 0.0
    return 0.5 * top1_acc + 0.5 * max(0.0, mean_r2), top1_acc, mean_r2


# ---------------------------------------------------------------------------
# BOHB ConfigSpace
# ---------------------------------------------------------------------------
def get_configspace():
    cs = CS.ConfigurationSpace(seed=SEED)
    cs.add([
        CSH.CategoricalHyperparameter("arch", choices=list(BASE_ARCHS.keys())),
        CSH.UniformFloatHyperparameter("dropout", 0.05, 0.35, default_value=0.15),
        CSH.UniformFloatHyperparameter("input_dropout", 0.0, 0.15, default_value=0.05),
        CSH.UniformFloatHyperparameter("lr", 1e-4, 5e-3, default_value=1e-3, log=True),
        CSH.UniformFloatHyperparameter("weight_decay", 1e-5, 1e-2, default_value=1e-4, log=True),
        CSH.UniformFloatHyperparameter("mixup_alpha", 0.0, 0.5, default_value=0.3),
        CSH.UniformFloatHyperparameter("label_threshold", 0.0, 0.10, default_value=0.0),
        CSH.UniformFloatHyperparameter("mixup_prob", 0.3, 0.8, default_value=0.5),
        CSH.CategoricalHyperparameter("activation", choices=["silu", "gelu", "mish"],
                                       default_value="silu"),
        CSH.CategoricalHyperparameter("batch_size", choices=[2048, 4096],
                                       default_value=4096),
    ])
    return cs


# ---------------------------------------------------------------------------
# BOHB Worker
# ---------------------------------------------------------------------------
class MLPWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.trial_count = 0

    def compute(self, config, budget, **kwargs):
        self.trial_count += 1
        budget = int(budget)
        arch_name = config["arch"]
        widths = BASE_ARCHS[arch_name]

        print(f"\n{'='*60}")
        print(f"  Trial {self.trial_count} | {arch_name} | {budget} epochs")
        print(f"  lr={config['lr']:.5f} wd={config['weight_decay']:.5f}")
        print(f"{'='*60}")

        t0 = time.time()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        thresh = config["label_threshold"]
        if thresh > 0:
            y_work = np.array(y_train_global)
            y_work[y_work < thresh] = 0.0
            row_sums = y_work.sum(axis=1, keepdims=True)
            valid_idx = np.where(row_sums.ravel() > 0)[0]
            X_work = X_train_global
            y_work = y_work[valid_idx]
            y_work = y_work / np.maximum(y_work.sum(axis=1, keepdims=True), 1e-8)
            use_valid_idx = True
        else:
            X_work = X_train_global
            y_work = y_train_global
            valid_idx = None
            use_valid_idx = False

        y_norm = normalize_targets(y_work)
        n_samples = len(y_norm)

        net = TaperedMLP(n_features, N_CLASSES, widths,
                         dropout=config["dropout"],
                         activation=config["activation"],
                         input_dropout=config["input_dropout"],
                         norm_type="batchnorm").to(self.device)

        n_params = sum(p.numel() for p in net.parameters())
        batch_size = config["batch_size"]

        try:
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=config["lr"],
                weight_decay=config["weight_decay"],
                fused=(self.device == "cuda"))
        except TypeError:
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=config["lr"],
                weight_decay=config["weight_decay"])

        use_amp = self.device == "cuda"
        grad_scaler = torch.amp.GradScaler(enabled=use_amp)
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size
        total_steps = budget * steps_per_epoch
        warmup_steps = steps_per_epoch * 3

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup_steps, 1)),
        ], milestones=[warmup_steps])

        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
        val_X_gpu = val_tensors[val_name]["X"]
        val_y_gpu = val_tensors[val_name]["y_norm"]

        best_val = float("inf")
        best_state = None
        wait = 0
        patience = max(math.ceil(5000 / steps_per_epoch), 5)
        min_epochs = max(math.ceil(1500 / steps_per_epoch), 3)
        rng = np.random.RandomState(SEED)

        for epoch in range(budget):
            net.train()
            perm = np.random.permutation(n_samples)
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                x_idx = valid_idx[idx] if use_valid_idx else idx
                xb = torch.from_numpy(np.array(X_work[x_idx])).to(self.device, non_blocking=True)
                yb = torch.from_numpy(y_norm[idx]).to(self.device, non_blocking=True)

                if config["mixup_alpha"] > 0 and rng.random() < config["mixup_prob"]:
                    xb, yb = apply_mixup(xb, yb, config["mixup_alpha"])

                if has_bn and xb.size(0) < 2:
                    continue

                optimizer.zero_grad(set_to_none=True)
                amp_device = "cuda" if use_amp else "cpu"
                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=torch.float16):
                    pred = net(xb)
                    loss = soft_cross_entropy(pred, yb)

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

            if epoch >= min_epochs and wait >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        elapsed = time.time() - t0
        combined, top1_acc, mean_r2 = compute_objective(
            net, val_tensors, max(config["label_threshold"], 0.01), self.device)

        print(f"  >> Result: combined={combined:.4f} top1={top1_acc:.4f} "
              f"R2={mean_r2:.4f} {elapsed:.0f}s")

        state_path = os.path.join(V10_DIR, f"trial_{self.trial_count}_{arch_name}.pt")
        torch.save(best_state or net.state_dict(), state_path)

        trial_result = {
            "trial": int(self.trial_count),
            "arch": str(arch_name),
            "widths": [int(w) for w in widths],
            "config": {k: (int(v) if isinstance(v, np.integer)
                          else float(v) if isinstance(v, np.floating)
                          else str(v) if isinstance(v, (np.str_, np.bytes_))
                          else v) for k, v in dict(config).items()},
            "budget": int(budget),
            "combined": float(combined),
            "top1_acc": float(top1_acc),
            "mean_r2": float(mean_r2),
            "val_loss": float(best_val),
            "n_params": int(n_params),
            "time_s": float(round(elapsed, 1)),
        }
        log_path = os.path.join(V10_DIR, "trial_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(trial_result) + "\n")

        del net, optimizer, grad_scaler, scheduler, best_state, y_norm
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        return {"loss": -combined, "info": trial_result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="V10 BOHB MLP Sweep")
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--min-budget", type=int, default=15)
    parser.add_argument("--max-budget", type=int, default=300)
    parser.add_argument("--eta", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  V10 BOHB MLP Sweep (Reproducibility)")
    print(f"  Max trials: {args.max_trials}, Budget: [{args.min_budget}, {args.max_budget}]")
    print(f"  Features: {n_features}, Train: {total:,}")
    print(f"{'='*70}\n")

    logging.getLogger("hpbandster").setLevel(logging.WARNING)

    NS = hpns.NameServer(run_id="v10_bohb", host="127.0.0.1", port=None)
    NS.start()

    worker = MLPWorker(nameserver="127.0.0.1", nameserver_port=NS.port,
                        run_id="v10_bohb")
    worker.run(background=True)

    bohb = BOHB(configspace=get_configspace(), run_id="v10_bohb",
                nameserver="127.0.0.1", nameserver_port=NS.port,
                min_budget=args.min_budget, max_budget=args.max_budget,
                eta=args.eta)

    result = bohb.run(n_iterations=args.max_trials)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()
    best_config = id2config[incumbent]["config"]
    best_runs = result.get_runs_by_id(incumbent)
    best_loss = min(r.loss for r in best_runs)

    print(f"\n{'='*70}")
    print(f"  BOHB COMPLETE — Best combined: {-best_loss:.4f}")
    for k, v in sorted(best_config.items()):
        print(f"    {k:20s}: {v}")
    print(f"{'='*70}")

    with open(os.path.join(V10_DIR, "best_config.json"), "w") as f:
        json.dump({"config": best_config, "loss": float(best_loss)}, f, indent=2)


if __name__ == "__main__":
    main()
