#!/usr/bin/env python3
"""
V10 MLP BOHB Sweep — Hyperparameter optimization using BOHB.

Built on V8's proven training loop (mixup_only mode with working cosine LR).
Searches hyperparams around the top-4 universally-good V8 architectures.

Search space:
  - Architecture: 4 TaperedMLP base shapes from V8 top performers
  - Dropout, input_dropout, weight_decay, learning_rate (continuous)
  - Mixup alpha, label_threshold (continuous)
  - Batch size, activation (categorical)

Objective: combined metric of
  (1) Top-1 argmax accuracy on val cities
  (2) R² for class fractions ≥ label_threshold

Uses BOHB (Bayesian Optimization + HyperBand) for efficient exploration:
  - Min budget = 15 epochs (quick reject bad configs)
  - Max budget = 300 epochs (full training for promising configs)

Usage:
    .venv\\Scripts\\python -u scripts/sweep_mlp_v10_bohb.py
    .venv\\Scripts\\python -u scripts/sweep_mlp_v10_bohb.py --max-trials 5 --max-budget 20  # dry run
"""

import os, sys, time, math, json, pickle, gc, logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# V8 imports (same data pipeline that produced the working model)
# ---------------------------------------------------------------------------
import scripts.run_multi_city_pipeline_v5 as v5_mod
from scripts.run_multi_city_pipeline_v5 import (
    TRAIN_CITIES as _ALL_TRAIN, TEST_CITIES as _ALL_TEST,
    SEED, CLASS_NAMES, N_CLASSES,
    city_features_dir, city_labels_path,
    _load_city_arrays, build_bi_lbp, ts,
)
try:
    from scripts.run_multi_city_pipeline_v5 import CONTROL_COLS
except ImportError:
    CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                    "reflectance_scale", "full_features_computed"}
from scripts.run_mlp_overnight_v4 import (
    PlainBlock, normalize_targets, soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pyarrow.parquet as pq

# BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

# Fix: Register serpent serializers for numpy types that ConfigSpace uses
# internally. Without this, Pyro4 RPCs crash on numpy.str_, numpy.int64, etc.
import serpent

def _numpy_serializer(obj, serpent_serializer, outputstream, indentlevel):
    """Serpent serializer for numpy scalar types → native Python."""
    if isinstance(obj, np.integer):
        serpent_serializer._serialize(int(obj), outputstream, indentlevel)
    elif isinstance(obj, np.floating):
        serpent_serializer._serialize(float(obj), outputstream, indentlevel)
    elif isinstance(obj, np.bool_):
        serpent_serializer._serialize(bool(obj), outputstream, indentlevel)
    elif isinstance(obj, (np.str_, np.bytes_)):
        serpent_serializer._serialize(str(obj), outputstream, indentlevel)
    elif isinstance(obj, np.ndarray):
        serpent_serializer._serialize(obj.tolist(), outputstream, indentlevel)
    else:
        serpent_serializer._serialize(str(obj), outputstream, indentlevel)

for np_type in [np.int32, np.int64, np.float32, np.float64,
                np.bool_, np.str_, np.bytes_, np.ndarray,
                np.intc, np.intp]:
    try:
        serpent.register_class(np_type, _numpy_serializer)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Monkey-patch: use features_v7/ (same as V8)
# ---------------------------------------------------------------------------
def _city_features_dir_v10(city):
    return os.path.join(v5_mod.city_dir(city), "features_v7")

v5_mod.city_features_dir = _city_features_dir_v10

CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
V10_DIR = os.path.join(CITIES_DIR, "models_v10_bohb")
os.makedirs(V10_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# City filtering (SAR required — same as V8)
# ---------------------------------------------------------------------------
def _city_parquet_path(city):
    return os.path.join(_city_features_dir_v10(city),
                        "features_rust_2020_2021.parquet")

def _city_feature_cols(city):
    feat_path = _city_parquet_path(city)
    if not os.path.exists(feat_path):
        return None
    schema = pq.read_schema(feat_path)
    numeric_types = {'float', 'double', 'int32', 'int64', 'float32', 'float64'}
    cols = set()
    for f in schema:
        type_str = str(f.type).lower()
        is_num = any(t in type_str for t in numeric_types)
        if is_num and f.name not in CONTROL_COLS:
            cols.add(f.name)
    return cols

def _city_has_sar(city) -> bool:
    cols = _city_feature_cols(city)
    if cols is None:
        return False
    return any(c.startswith("SAR_") for c in cols)

# ---------------------------------------------------------------------------
# V10 city splits — 23 val cities (GPT-optimized for label balance)
# Mean train-vs-val class gap: 0.041pp (vs 3.26pp before)
# Worst single-class gap: 0.079pp
# ---------------------------------------------------------------------------
VAL_CITY_NAMES = {
    'alentejo_portugal',
    'andalusia_olives',
    'berlin',
    'bordeaux',
    'central_spain_plateau',
    'corsica_interior',
    'dresden',
    'dutch_polders',
    'ebro_delta',
    'estonian_plains',
    'helsinki',
    'iceland_highlands',
    'ireland_bog_pasture',
    'jaen_olives',
    'madrid',
    'marseille',
    'northern_sweden',
    'paris_south',
    'peloponnese_rural',
    'po_valley_rural',
    'rostock',
    'uppland_farmland',
    'vojvodina_cropland',
}
EXCLUDED_CITY_NAMES = {'nuremberg'}

print(f"[{ts()}] V10 BOHB -- filtering cities with SAR features...")
ALL_SAR_CITIES = [c for c in list(_ALL_TRAIN) + list(_ALL_TEST) if _city_has_sar(c)]
TRAIN_CITIES = [c for c in ALL_SAR_CITIES
                if c.name not in VAL_CITY_NAMES and c.name not in EXCLUDED_CITY_NAMES]
VAL_CITIES = [c for c in ALL_SAR_CITIES if c.name in VAL_CITY_NAMES]
print(f"  Train cities: {len(TRAIN_CITIES)}")
print(f"  Val cities: {len(VAL_CITIES)} ({[c.name for c in VAL_CITIES]})")

# Feature columns = intersection across ALL cities (train + val)
print(f"\n[{ts()}] Building feature column intersection...")
all_col_sets = []
for city in TRAIN_CITIES + VAL_CITIES:
    cols = _city_feature_cols(city)
    if cols is not None:
        all_col_sets.append(cols)

common_cols = set.intersection(*all_col_sets) if all_col_sets else set()
first_schema = pq.read_schema(_city_parquet_path(TRAIN_CITIES[0]))
full_feature_cols = [f.name for f in first_schema if f.name in common_cols]
mlp_idx = build_bi_lbp(full_feature_cols)
mlp_cols = [full_feature_cols[i] for i in mlp_idx]
n_features = len(mlp_cols)
print(f"  MLP features: {n_features}")

# ---------------------------------------------------------------------------
# Model definitions (exactly as V8/V6)
# ---------------------------------------------------------------------------
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
# Top-4 universally good base architectures from V8
# ---------------------------------------------------------------------------
BASE_ARCHS = {
    "T_512_256_128_64":   [512, 256, 128, 64],    # 1.1M params, best min_r2
    "T_1024_512_256_64":  [1024, 512, 256, 64],   # 2.5M params, deployed
    "T_2048_512_128":     [2048, 512, 128],        # 4.7M params, good balance
    "T_2048_1024_512":    [2048, 1024, 512],       # 6.2M params, highest min_r2
}

# ---------------------------------------------------------------------------
# Mixup (from V6/V8)
# ---------------------------------------------------------------------------
def apply_mixup(xb, yb, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]

# ---------------------------------------------------------------------------
# Data loading — training data to memmap, val data to GPU
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {device}")

print(f"\n[{ts()}] Loading training data (memmap)...")
city_counts = []
train_cities_valid = []
for city in TRAIN_CITIES:
    feat_path = _city_parquet_path(city)
    if not os.path.exists(feat_path):
        continue
    n = pq.read_metadata(feat_path).num_rows
    city_counts.append(n)
    train_cities_valid.append(city)

total_est = sum(city_counts)
print(f"  Pre-allocating memmap ({total_est:,} x {n_features})...")
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
    result = _load_city_arrays(city, mlp_cols)
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

# Load val data directly to GPU (like V9)
print(f"\n[{ts()}] Loading val cities to VRAM...")
val_tensors = {}
for city in VAL_CITIES:
    result = _load_city_arrays(city, mlp_cols)
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
    # Preload to GPU
    val_tensors[city.name] = {
        "X": torch.from_numpy(X_v).to(device),
        "y_norm": torch.from_numpy(y_v_norm).to(device),
        "y_raw": y_v,  # keep on CPU for R2 eval
    }
    del X_v, y_v_norm
    print(f"  [{city.name}] {val_tensors[city.name]['X'].shape[0]:,} cells -> VRAM")
    gc.collect()

# Early-stop city
val_name = "berlin"
if val_name not in val_tensors:
    val_name = list(val_tensors.keys())[0]
print(f"  Early-stop city: {val_name} (on GPU)")
gc.collect()

# ---------------------------------------------------------------------------
# Custom objective function
# ---------------------------------------------------------------------------
def compute_objective(net, val_data, label_threshold, dev):
    """
    Combined objective on GPU-resident val tensors:
      (1) Top-1 argmax accuracy
      (2) R2 for class fractions >= label_threshold
    """
    all_top1_correct = 0
    all_top1_total = 0
    all_r2_values = []

    net.eval()
    for city_name, data in val_data.items():
        X_gpu = data["X"]       # already on GPU
        y_raw = data["y_raw"]   # numpy on CPU for R2

        # Predict on GPU directly
        with torch.no_grad():
            preds = net.predict(X_gpu).cpu().numpy()

        # (1) Top-1 accuracy
        true_top1 = y_raw.argmax(axis=1)
        pred_top1 = preds.argmax(axis=1)
        all_top1_correct += (true_top1 == pred_top1).sum()
        all_top1_total += len(true_top1)

        # (2) R2 for meaningful classes
        for cls_i in range(N_CLASSES):
            yt = y_raw[:, cls_i]
            mask = yt >= label_threshold
            if mask.sum() < 50:
                continue
            yt_masked = yt[mask]
            if np.var(yt_masked) < 1e-8:
                continue
            r2 = r2_score(yt_masked, preds[mask, cls_i])
            all_r2_values.append(r2)

    top1_acc = all_top1_correct / max(all_top1_total, 1)
    mean_r2 = np.mean(all_r2_values) if all_r2_values else 0.0
    combined = 0.5 * top1_acc + 0.5 * max(0.0, mean_r2)
    return combined, top1_acc, mean_r2

# ---------------------------------------------------------------------------
# BOHB ConfigSpace
# ---------------------------------------------------------------------------
def get_configspace():
    cs = CS.ConfigurationSpace(seed=SEED)

    # Architecture choice
    arch = CSH.CategoricalHyperparameter(
        "arch", choices=list(BASE_ARCHS.keys()))

    # Continuous hyperparams
    dropout = CSH.UniformFloatHyperparameter("dropout", 0.05, 0.35, default_value=0.15)
    input_dropout = CSH.UniformFloatHyperparameter("input_dropout", 0.0, 0.15, default_value=0.05)
    lr = CSH.UniformFloatHyperparameter("lr", 1e-4, 5e-3, default_value=1e-3, log=True)
    weight_decay = CSH.UniformFloatHyperparameter("weight_decay", 1e-5, 1e-2, default_value=1e-4, log=True)
    mixup_alpha = CSH.UniformFloatHyperparameter("mixup_alpha", 0.0, 0.5, default_value=0.3)
    label_threshold = CSH.UniformFloatHyperparameter("label_threshold", 0.0, 0.10, default_value=0.0)
    mixup_prob = CSH.UniformFloatHyperparameter("mixup_prob", 0.3, 0.8, default_value=0.5)

    # Categorical
    activation = CSH.CategoricalHyperparameter("activation", choices=["silu", "gelu", "mish"],
                                                default_value="silu")
    batch_size = CSH.CategoricalHyperparameter("batch_size", choices=[2048, 4096],
                                                default_value=4096)

    cs.add([arch, dropout, input_dropout, lr, weight_decay,
            mixup_alpha, label_threshold, mixup_prob,
            activation, batch_size])
    return cs

# ---------------------------------------------------------------------------
# BOHB Worker
# ---------------------------------------------------------------------------
class MLPWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trial_count = 0

    def compute(self, config, budget, **kwargs):
        """Train one MLP config for `budget` epochs and return objective."""
        self.trial_count += 1
        budget = int(budget)
        arch_name = config["arch"]
        widths = BASE_ARCHS[arch_name]

        print(f"\n{'='*60}")
        print(f"  Trial {self.trial_count} | {arch_name} | {budget} epochs")
        print(f"  lr={config['lr']:.5f} wd={config['weight_decay']:.5f} "
              f"do={config['dropout']:.2f} ido={config['input_dropout']:.2f}")
        print(f"  mixup_a={config['mixup_alpha']:.2f} "
              f"mixup_p={config['mixup_prob']:.2f} "
              f"label_thresh={config['label_threshold']:.3f}")
        print(f"  act={config['activation']} bs={config['batch_size']}")
        print(f"{'='*60}")

        t0 = time.time()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # Apply label denoising
        thresh = config["label_threshold"]
        if thresh > 0:
            # Y is small (3.2M x 7 = 86MB), safe to copy
            y_work = np.array(y_train_global)
            y_work[y_work < thresh] = 0.0
            row_sums = y_work.sum(axis=1, keepdims=True)
            valid_idx = np.where(row_sums.ravel() > 0)[0]
            # Use index array instead of boolean mask to avoid full memmap copy
            X_work = X_train_global  # stays as memmap, indexed per batch
            y_work = y_work[valid_idx]
            y_work = y_work / np.maximum(y_work.sum(axis=1, keepdims=True), 1e-8)
            use_valid_idx = True
        else:
            X_work = X_train_global
            y_work = y_train_global
            valid_idx = None
            use_valid_idx = False

        y_norm = normalize_targets(y_work)
        n_samples = len(y_norm)  # filtered count when thresh > 0

        # Build model
        net = TaperedMLP(
            n_features, N_CLASSES, widths,
            dropout=config["dropout"],
            activation=config["activation"],
            input_dropout=config["input_dropout"],
            norm_type="batchnorm"
        ).to(self.device)

        n_params = sum(p.numel() for p in net.parameters())
        batch_size = config["batch_size"]

        # Optimizer (same as V8)
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
        amp_device = "cuda" if use_amp else "cpu"
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

        # Val data already on GPU
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
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                # Translate indices: Y is indexed directly, X needs valid_idx mapping
                x_idx = valid_idx[idx] if use_valid_idx else idx
                xb = torch.from_numpy(np.array(X_work[x_idx])).to(
                    self.device, non_blocking=True)
                yb = torch.from_numpy(y_norm[idx]).to(self.device, non_blocking=True)

                # Mixup
                if config["mixup_alpha"] > 0 and rng.random() < config["mixup_prob"]:
                    xb, yb = apply_mixup(xb, yb, config["mixup_alpha"])

                if has_bn and xb.size(0) < 2:
                    continue

                optimizer.zero_grad(set_to_none=True)
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

            avg_train = epoch_loss / max(n_batches, 1)

            # Validate (all on GPU — no transfer)
            net.eval()
            with torch.no_grad():
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
                print(f"    Ep {epoch:3d}: train={avg_train:.5f} "
                      f"val={val_loss:.5f} wait={wait}{marker}")

            if epoch >= min_epochs and wait >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

        # Load best
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        elapsed = time.time() - t0

        # Compute combined objective
        combined, top1_acc, mean_r2 = compute_objective(
            net, val_tensors,
            max(config["label_threshold"], 0.01),
            self.device
        )

        print(f"\n  >> Result: combined={combined:.4f} "
              f"top1={top1_acc:.4f} R2={mean_r2:.4f} "
              f"val_loss={best_val:.5f} {elapsed:.0f}s")

        # Save best models
        state_path = os.path.join(V10_DIR, f"trial_{self.trial_count}_{arch_name}.pt")
        torch.save(best_state or net.state_dict(), state_path)

        # Log to JSON — convert numpy types to native Python for serialization
        native_config = {}
        for k, v in dict(config).items():
            if isinstance(v, (np.integer,)):
                native_config[k] = int(v)
            elif isinstance(v, (np.floating,)):
                native_config[k] = float(v)
            elif isinstance(v, (np.str_, np.bytes_)):
                native_config[k] = str(v)
            else:
                native_config[k] = v

        trial_result = {
            "trial": int(self.trial_count),
            "arch": str(arch_name),
            "widths": [int(w) for w in widths],
            "config": native_config,
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

        # Cleanup
        del net, optimizer, grad_scaler, scheduler, best_state
        del y_norm
        if thresh > 0:
            del X_work, y_work
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "loss": -combined,  # BOHB minimizes
            "info": trial_result,
        }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="V10 BOHB MLP Sweep")
    parser.add_argument("--max-trials", type=int, default=100,
                        help="Maximum BOHB iterations (default: 100)")
    parser.add_argument("--min-budget", type=int, default=15,
                        help="Min epochs for HyperBand (default: 15)")
    parser.add_argument("--max-budget", type=int, default=300,
                        help="Max epochs for HyperBand (default: 300)")
    parser.add_argument("--eta", type=int, default=3,
                        help="HyperBand eta (default: 3)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  V10 BOHB MLP Sweep")
    print(f"{'='*70}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Max trials: {args.max_trials}")
    print(f"  Budget range: [{args.min_budget}, {args.max_budget}] epochs")
    print(f"  Eta: {args.eta}")
    print(f"  Output: {V10_DIR}")
    print(f"  Features: {n_features}")
    print(f"  Train samples: {total:,}")
    print(f"  Val city: {val_name}")
    print(f"  Architectures: {list(BASE_ARCHS.keys())}")
    print(f"{'='*70}\n")

    # Suppress hpbandster/Pyro noise
    logging.getLogger("hpbandster").setLevel(logging.WARNING)

    # Start nameserver
    NS = hpns.NameServer(run_id="v10_bohb", host="127.0.0.1", port=None)
    NS.start()

    # Start worker
    worker = MLPWorker(nameserver="127.0.0.1",
                       nameserver_port=NS.port,
                       run_id="v10_bohb")
    worker.run(background=True)

    # Run BOHB
    bohb = BOHB(
        configspace=get_configspace(),
        run_id="v10_bohb",
        nameserver="127.0.0.1",
        nameserver_port=NS.port,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        eta=args.eta,
    )

    result = bohb.run(n_iterations=args.max_trials)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Extract best
    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()

    best_config = id2config[incumbent]["config"]
    best_runs = result.get_runs_by_id(incumbent)
    best_loss = min(r.loss for r in best_runs)

    print(f"\n{'='*70}")
    print(f"  BOHB COMPLETE")
    print(f"{'='*70}")
    print(f"  Best combined objective: {-best_loss:.4f}")
    print(f"  Best config:")
    for k, v in sorted(best_config.items()):
        print(f"    {k:20s}: {v}")

    # Save best config — convert numpy types for JSON
    native_best = {}
    for k, v in best_config.items():
        if isinstance(v, (np.integer,)):
            native_best[k] = int(v)
        elif isinstance(v, (np.floating,)):
            native_best[k] = float(v)
        elif isinstance(v, (np.str_, np.bytes_)):
            native_best[k] = str(v)
        else:
            native_best[k] = v
    with open(os.path.join(V10_DIR, "best_config.json"), "w") as f:
        json.dump({"config": native_best, "loss": float(best_loss)}, f, indent=2)

    # Read trial log for summary
    log_path = os.path.join(V10_DIR, "trial_log.jsonl")
    if os.path.exists(log_path):
        trials = []
        with open(log_path) as f:
            for line in f:
                trials.append(json.loads(line))
        ranked = sorted(trials, key=lambda t: t["combined"], reverse=True)
        print(f"\n  Top 5 trials:")
        for i, t in enumerate(ranked[:5], 1):
            print(f"    {i}. {t['arch']:25s} combined={t['combined']:.4f} "
                  f"top1={t['top1_acc']:.4f} R2={t['mean_r2']:.4f} "
                  f"budget={t['budget']}")

    print(f"\n  Results saved to: {V10_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
