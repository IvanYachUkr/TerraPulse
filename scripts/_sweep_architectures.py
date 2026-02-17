"""Architecture sweep: test short+wide vs narrow+deep with ALL features.

Uses all 1464 features (including edge, laplacian, Moran's I).
Low patience (5000 steps = ~27 epochs) for fast iteration.
"""
import os, sys, time, math, json, pickle
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v4 import (
    TRAIN_CITIES, TEST_CITIES, SEED, CLASS_NAMES,
    city_features_dir, city_labels_path,
    _discover_feature_cols, _load_city_arrays,
    build_bi_lbp, ts,
)
from scripts.run_mlp_overnight_v4 import (
    build_model, _cfg, normalize_targets, soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- Configs to sweep ----------
CONFIGS = [
    # name, depth, width, dropout, wd
    ("2L_1024w",  2, 1024, 0.20, 5e-4),  # short + wide
    ("3L_512w",   3,  512, 0.20, 5e-4),  # baseline small (reproduce)
    ("5L_256w",   5,  256, 0.15, 3e-4),  # narrow + deep
    ("7L_128w",   7,  128, 0.10, 1e-4),  # very narrow + very deep
]

BATCH_SIZE = 2048
PATIENCE_STEPS = 5000   # fast iteration: ~27 epochs
MIN_STEPS = 2000
MAX_EPOCHS = 500
SEED_OFFSET = 100
INPUT_DROPOUT = 0.05
LR = 1e-3

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v4_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load ALL features (no filtering) ----------
print(f"[{ts()}] Discovering ALL features...")
all_feature_cols = _discover_feature_cols()  # returns all numeric non-control cols
n_features = len(all_feature_cols)
print(f"  Total features: {n_features}")

# Also count the MLP-excluded ones
mlp_idx = build_bi_lbp(all_feature_cols)
mlp_cols_set = set(all_feature_cols[i] for i in mlp_idx)
extra = [c for c in all_feature_cols if c not in mlp_cols_set]
print(f"  Previously excluded (now included): {len(extra)}")
for c in extra[:10]:
    print(f"    {c}")
if len(extra) > 10:
    print(f"    ... and {len(extra)-10} more")

# Load training data (ALL columns)
print(f"\n[{ts()}] Loading training data ({n_features} cols)...")
X_parts, y_parts = [], []
total = 0
for city in TRAIN_CITIES:
    result = _load_city_arrays(city, all_feature_cols)
    if result is None:
        print(f"  SKIP {city.name}")
        continue
    X_city, n = result
    labels = __import__('pandas').read_parquet(city_labels_path(city, 2021))
    y_city = labels[CLASS_NAMES].values.astype(np.float32)
    del labels
    X_parts.append(X_city)
    y_parts.append(y_city)
    total += n
    print(f"  [{city.name}] {n} cells (total={total})")

X = np.concatenate(X_parts, axis=0); del X_parts
y = np.concatenate(y_parts, axis=0); del y_parts

# Load validation
val_city = TEST_CITIES[0]
X_val, n_val = _load_city_arrays(val_city, all_feature_cols)
labels_val = __import__('pandas').read_parquet(city_labels_path(val_city, 2021))
y_val = labels_val[CLASS_NAMES].values.astype(np.float32)
del labels_val

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
del X, X_val

device = "cuda" if torch.cuda.is_available() else "cpu"
n = len(y)
steps_per_epoch = (n + BATCH_SIZE - 1) // BATCH_SIZE
patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)

print(f"\n  Total: {n} samples, {n_features} features")
print(f"  steps/epoch={steps_per_epoch}, patience={patience_epochs}ep")
print(f"  Sweeping {len(CONFIGS)} architectures on {device}")

# Preload tensors
y_norm = normalize_targets(y)
y_val_norm = normalize_targets(y_val)
X_trn_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
y_trn_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
y_val_t = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
del X_scaled, y_norm, X_val_scaled, y_val_norm

results = []

for cfg_name, depth, width, dropout, wd in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  {cfg_name}: {depth}L x {width}w (do={dropout}, wd={wd})")
    print(f"{'='*60}")

    actual_seed = SEED + SEED_OFFSET
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    cfg = _cfg(0, "bi_LBP", "plain", "silu", depth, width, "batchnorm",
               dropout=dropout, input_dropout=INPUT_DROPOUT,
               weight_decay=wd, lr=LR)
    net = build_model(cfg, n_features, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {n_params:,}")

    use_amp = device == "cuda"
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd, fused=use_amp)
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=LR, weight_decay=wd)

    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = MAX_EPOCHS * steps_per_epoch
    warmup_steps = steps_per_epoch * 3
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps),
    ], milestones=[warmup_steps])

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
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
        if epoch <= 5 or epoch % 10 == 0 or improved:
            print(f"    Ep {epoch:3d}: train={avg_train:.5f} val={val_loss:.5f} "
                  f"wait={wait}{marker}")

        if epoch >= min_epochs and wait >= patience_epochs:
            print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # Evaluate
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        preds = net.predict(X_val_t).cpu().numpy()

    r2 = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds) * 100
    per_class = {}
    for i, cn in enumerate(CLASS_NAMES):
        per_class[cn] = r2_score(y_val[:, i], preds[:, i])

    print(f"\n  >> {cfg_name}: R2={r2:.4f} MAE={mae:.2f}pp "
          f"({epoch+1}ep, {n_params:,} params, {elapsed:.0f}s)")
    for cn in CLASS_NAMES:
        print(f"       {cn:20s} R2={per_class[cn]:.4f}")

    results.append({
        "name": cfg_name, "depth": depth, "width": width,
        "n_params": n_params, "dropout": dropout, "wd": wd,
        "r2": r2, "mae": mae, "per_class_r2": per_class,
        "n_epochs": epoch + 1, "best_val": best_val,
        "time_s": round(elapsed, 1),
    })

    # Save best model
    torch.save(best_state, os.path.join(OUT_DIR, f"mlp_{cfg_name}.pt"))
    del net, optimizer, scaler_amp, scheduler, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print("SWEEP SUMMARY")
print(f"{'='*60}")
print(f"{'Config':15s} {'Params':>10s} {'R2':>8s} {'MAE':>8s} {'Epochs':>6s} {'Time':>6s}")
for r in results:
    print(f"{r['name']:15s} {r['n_params']:>10,} {r['r2']:>8.4f} "
          f"{r['mae']:>7.2f}pp {r['n_epochs']:>5d}  {r['time_s']:>5.0f}s")

# Save results
with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"\nSaved to {OUT_DIR}")
