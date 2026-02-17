"""Quick experiment: small MLP (3L x 512w) vs large (5L x 2048w).

Reuses V4 pipeline data loading, saves to models_v4_small/.
"""
import os, sys, time, math, json, pickle
import numpy as np
import torch

# -- Add project root --
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v4 import (
    TRAIN_CITIES, TEST_CITIES, SEED, CLASS_NAMES, N_CLASSES,
    MODELS_DIR as ORIG_MODELS_DIR,
    city_features_dir, city_labels_path,
    _discover_feature_cols, _load_city_arrays,
    build_bi_lbp,
    ts,
)
from scripts.run_mlp_overnight_v4 import (
    build_model, _cfg, normalize_targets, soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler


# ---------- Config ----------
WIDTH = 512
DEPTH = 3
DROPOUT = 0.20
INPUT_DROPOUT = 0.05
WEIGHT_DECAY = 5e-4
LR = 1e-3
BATCH_SIZE = 2048
PATIENCE_STEPS = 30000
MIN_STEPS = 3000
MAX_EPOCHS = 3000
SEED_OFFSET = 100

OUT_DIR = os.path.join(os.path.dirname(ORIG_MODELS_DIR), "models_v4_small")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"=== Small MLP Experiment: {DEPTH}L x {WIDTH}w ===")
print(f"  dropout={DROPOUT}, input_dropout={INPUT_DROPOUT}, wd={WEIGHT_DECAY}")
print(f"  Output: {OUT_DIR}")

# ---------- Load data ----------
print(f"\n[{ts()}] Discovering features...")
full_feature_cols = _discover_feature_cols()
mlp_idx = build_bi_lbp(full_feature_cols)
mlp_cols = [full_feature_cols[i] for i in mlp_idx]
n_features = len(mlp_cols)
print(f"  MLP features: {n_features}")

print(f"[{ts()}] Loading training data...")
X_parts, y_parts = [], []
total = 0
for city in TRAIN_CITIES:
    result = _load_city_arrays(city, mlp_cols)
    if result is None:
        print(f"  WARNING: Missing {city.name}, skip")
        continue
    X_city, n = result
    labels = __import__('pandas').read_parquet(
        city_labels_path(city, 2021))
    y_city = labels[CLASS_NAMES].values.astype(np.float32)
    del labels
    X_parts.append(X_city)
    y_parts.append(y_city)
    total += n
    print(f"  [{city.name}] {n} cells (total={total})")

X = np.concatenate(X_parts, axis=0); del X_parts
y = np.concatenate(y_parts, axis=0); del y_parts
print(f"  Total: {len(y)} samples, {n_features} features")

# Load validation
print(f"[{ts()}] Loading Nuremberg validation...")
val_city = TEST_CITIES[0]  # nuremberg
val_result = _load_city_arrays(val_city, mlp_cols)
X_val_raw, n_val = val_result
labels_val = __import__('pandas').read_parquet(
    city_labels_path(val_city, 2021))
y_val = labels_val[CLASS_NAMES].values.astype(np.float32)
del labels_val
print(f"  Nuremberg: {n_val} cells")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
del X, X_val_raw

with open(os.path.join(OUT_DIR, "mlp_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---------- Train ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[{ts()}] Training on {device}")

actual_seed = SEED + SEED_OFFSET
torch.manual_seed(actual_seed)
np.random.seed(actual_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(actual_seed)

cfg = _cfg(0, "bi_LBP", "plain", "silu", DEPTH, WIDTH, "batchnorm",
           dropout=DROPOUT, input_dropout=INPUT_DROPOUT,
           weight_decay=WEIGHT_DECAY, lr=LR)
net = build_model(cfg, n_features, device)
n_params = sum(p.numel() for p in net.parameters())
print(f"  Model: {DEPTH}L x {WIDTH}w, {n_params:,} parameters")

y_norm = normalize_targets(y)
y_val_norm = normalize_targets(y_val)

X_trn_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
y_trn_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
y_val_t = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
del X_scaled, y_norm, X_val_scaled, y_val_norm

use_amp = device == "cuda"
wd = cfg["weight_decay"]
try:
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["lr"], weight_decay=wd, fused=use_amp)
except TypeError:
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["lr"], weight_decay=wd)

scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

n = len(y)
steps_per_epoch = (n + BATCH_SIZE - 1) // BATCH_SIZE
total_steps = MAX_EPOCHS * steps_per_epoch

warmup_steps = steps_per_epoch * 3
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
    torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps),
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps),
], milestones=[warmup_steps])

patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)
print(f"  steps/epoch={steps_per_epoch}, patience={patience_epochs}ep, "
      f"min={min_epochs}ep")

has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules())
best_val = float("inf")
best_state = None
wait = 0
rng = np.random.RandomState(actual_seed)

t0 = time.time()
for epoch in range(MAX_EPOCHS):
    net.train()
    epoch_loss = 0.0
    n_batches = 0
    perm = rng.permutation(n)
    for start in range(0, n, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        xb = X_trn_t[idx_t]
        yb = y_trn_t[idx_t]
        if has_bn and xb.size(0) < 2:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            pred = net(xb)
            loss = soft_cross_entropy(pred, yb)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_train = epoch_loss / max(n_batches, 1)

    # Validate
    net.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            val_pred = net(X_val_t)
            val_loss = soft_cross_entropy(val_pred, y_val_t).item()

    improved = val_loss < best_val
    if improved:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        wait = 0
    else:
        wait += 1

    marker = " *BEST*" if improved else ""
    if epoch <= 10 or epoch % 10 == 0 or improved:
        print(f"    Epoch {epoch:4d}: train={avg_train:.5f} val={val_loss:.5f} "
              f"wait={wait}{marker}")

    if epoch >= min_epochs and wait >= patience_epochs:
        print(f"    Early stopping at epoch {epoch} (patience={patience_epochs})")
        break

elapsed = time.time() - t0

# Restore best
net.load_state_dict(best_state)
net.eval()

# Evaluate — net.predict() returns 0-1 probabilities, y_val is 0-1 fractions
with torch.no_grad():
    preds = net.predict(X_val_t).cpu().numpy()
# y_val is already in 0-1 scale
from sklearn.metrics import r2_score, mean_absolute_error
r2_overall = r2_score(y_val, preds)
mae_overall = mean_absolute_error(y_val, preds) * 100  # convert to pp
print(f"\n  Result: R2={r2_overall:.4f} MAE={mae_overall:.2f}pp "
      f"({epoch+1} epochs, {elapsed:.0f}s)")

per_class = {}
for i, cn in enumerate(CLASS_NAMES):
    r2c = r2_score(y_val[:, i], preds[:, i])
    per_class[cn] = r2c
    print(f"    {cn:<20} R2={r2c:.4f}")

# Save
torch.save(net.state_dict(), os.path.join(OUT_DIR, "mlp_seed0.pt"))
meta = {
    "model": "PlainMLP", "training": "small_experiment",
    "arch": f"{DEPTH}L x {WIDTH}w",
    "n_params": n_params,
    "n_features": n_features,
    "dropout": DROPOUT, "input_dropout": INPUT_DROPOUT,
    "weight_decay": WEIGHT_DECAY, "lr": LR,
    "r2_nuremberg": r2_overall,
    "mae_nuremberg": mae_overall,
    "per_class_r2": per_class,
    "n_epochs": epoch + 1,
    "best_val_loss": best_val,
    "time_s": round(elapsed, 1),
}
with open(os.path.join(OUT_DIR, "mlp_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n  Saved to {OUT_DIR}")
print("  Done!")
