"""Tapered MLP sweep: wide input → progressively narrower layers.

Tests funnel architectures where the first layer is wide (to compress
1464 features) and subsequent layers narrow down.
"""
import os, sys, time, math, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v4 import (
    TRAIN_CITIES, TEST_CITIES, SEED, CLASS_NAMES,
    city_features_dir, city_labels_path,
    _discover_feature_cols, _load_city_arrays, ts,
)
from scripts.run_mlp_overnight_v4 import (
    _make_norm, PlainBlock, normalize_targets,
    soft_cross_entropy,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


class TaperedMLP(nn.Module):
    """MLP with decreasing layer widths: wide input compression → narrow output.
    
    Example: widths=[1024, 512, 256, 128] creates:
      1464 → 1024 → 512 → 256 → 128 → 6
    """
    def __init__(self, in_features, n_classes, widths, dropout=0.15,
                 activation="silu", input_dropout=0.05, norm_type="batchnorm"):
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


# ---------- Configs ----------
CONFIGS = [
    # name, widths, dropout, wd
    ("T_1024_512_256",      [1024, 512, 256],           0.20, 5e-4),
    ("T_1024_512_256_128",  [1024, 512, 256, 128],      0.20, 5e-4),
    ("T_768_384_192",       [768, 384, 192],            0.20, 5e-4),
    ("T_768_384_192_96",    [768, 384, 192, 96],        0.15, 3e-4),
    ("T_512_256_128",       [512, 256, 128],            0.15, 3e-4),
    ("T_512_256_128_64",    [512, 256, 128, 64],        0.15, 3e-4),
    ("T_1024_256",          [1024, 256],                0.20, 5e-4),   # aggressive 4x squeeze
    ("T_1024_512",          [1024, 512],                0.20, 5e-4),   # gentle 2x squeeze
]

BATCH_SIZE = 2048
PATIENCE_STEPS = 5000
MIN_STEPS = 2000
MAX_EPOCHS = 500
SEED_OFFSET = 100
INPUT_DROPOUT = 0.05
LR = 1e-3

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v4_tapered")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load ALL features ----------
print(f"[{ts()}] Loading ALL 1464 features...")
all_feature_cols = _discover_feature_cols()
n_features = len(all_feature_cols)

X_parts, y_parts = [], []
total = 0
for city in TRAIN_CITIES:
    result = _load_city_arrays(city, all_feature_cols)
    if result is None:
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

val_city = TEST_CITIES[0]
X_val, n_val = _load_city_arrays(val_city, all_feature_cols)
labels_val = __import__('pandas').read_parquet(city_labels_path(val_city, 2021))
y_val = labels_val[CLASS_NAMES].values.astype(np.float32)
del labels_val

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
del X, X_val

device = "cuda" if torch.cuda.is_available() else "cpu"
n = len(y)
steps_per_epoch = (n + BATCH_SIZE - 1) // BATCH_SIZE
patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)

y_norm = normalize_targets(y)
y_val_norm = normalize_targets(y_val)
X_trn_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
y_trn_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
y_val_t = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
del X_scaled, y_norm, X_val_scaled, y_val_norm

print(f"\n  {n} samples, {n_features} features, patience={patience_epochs}ep")
print(f"  Sweeping {len(CONFIGS)} tapered architectures on {device}\n")

results = []

for cfg_name, widths, dropout, wd in CONFIGS:
    print(f"{'='*60}")
    shape_str = " -> ".join([str(n_features)] + [str(w) for w in widths] + ["6"])
    print(f"  {cfg_name}: {shape_str}")
    print(f"  dropout={dropout}, wd={wd}")

    actual_seed = SEED + SEED_OFFSET
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    net = TaperedMLP(n_features, len(CLASS_NAMES), widths,
                     dropout=dropout, input_dropout=INPUT_DROPOUT,
                     norm_type="batchnorm").to(device)
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

    best_val = float("inf")
    best_state = None
    wait = 0
    rng = np.random.RandomState(actual_seed)
    best_epoch = 0

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        epoch_loss, n_batches = 0.0, 0
        perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            xb, yb = X_trn_t[idx_t], y_trn_t[idx_t]
            if xb.size(0) < 2:
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
                val_loss = soft_cross_entropy(net(X_val_t), y_val_t).item()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
            best_epoch = epoch
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
          f"(best@ep{best_epoch}, {n_params:,} params, {elapsed:.0f}s)")
    for cn in CLASS_NAMES:
        print(f"       {cn:20s} R2={per_class[cn]:.4f}")

    results.append({
        "name": cfg_name, "widths": widths, "n_params": n_params,
        "dropout": dropout, "wd": wd,
        "r2": r2, "mae": mae, "per_class_r2": per_class,
        "n_epochs": epoch + 1, "best_epoch": best_epoch,
        "best_val": best_val, "time_s": round(elapsed, 1),
    })

    torch.save(best_state, os.path.join(OUT_DIR, f"mlp_{cfg_name}.pt"))
    del net, optimizer, scaler_amp, scheduler, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Summary
print(f"\n{'='*70}")
print("TAPERED SWEEP SUMMARY")
print(f"{'='*70}")
print(f"{'Config':25s} {'Shape':30s} {'Params':>9s} {'R2':>7s} {'MAE':>7s} {'BestEp':>6s} {'Time':>5s}")
for r in results:
    shape = "->".join(str(w) for w in r['widths'])
    print(f"{r['name']:25s} {shape:30s} {r['n_params']:>9,} {r['r2']:>7.4f} "
          f"{r['mae']:>6.2f}pp {r['best_epoch']:>5d}  {r['time_s']:>4.0f}s")

with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"\nSaved to {OUT_DIR}")
