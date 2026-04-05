#!/usr/bin/env python3
"""
Step 1: Train raw-band MLP models (1×1, 3×3, 5×5).

Trains three MLP variants sequentially on pixel-level land cover
classification using raw S2+S1 bands (no handcrafted features).

Each model is fully trained and freed before the next one starts.
Data is loaded directly from TIF files into RAM — no disk cache.

Pixel budgets are calibrated to fill ~10 GB RAM for training data:
  - mlp_1x1 (72 features):   350K px/city → ~9.3 GB train data
  - mlp_3x3 (648 features):   40K px/city → ~9.6 GB train data
  - mlp_5x5 (1800 features):  15K px/city → ~9.9 GB train data

Usage:
    python 01_train_raw_mlp.py                        # train all 3
    python 01_train_raw_mlp.py --models mlp_1x1       # just one
    python 01_train_raw_mlp.py --max-epochs 50        # quick test
"""

import argparse
import gc
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Setup ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    CITIES_DIR, SEED, N_CLASSES, CLASS_NAMES, N_RAW_FEATURES,
    get_train_cities, get_val_cities, get_test_cities,
    city_has_raw_tifs,
)
from reproduce.models.shared.data import (
    extract_pixels_for_city,
)
from reproduce.models.architectures.mlp import (
    MODEL_CONFIGS, build_model,
)


OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")

# Pixel budgets calibrated to fill ~10 GB RAM for train data
# Val uses fewer pixels to keep GPU memory reasonable
MODEL_SPECS = {
    "mlp_1x1":      {"variant": "1x1",      "n_feat": 72,   "train_px": 200_000, "val_px": 100_000},
    "mlp_3x3":      {"variant": "3x3",      "n_feat": 648,  "train_px":  30_000, "val_px":  15_000},
    "mlp_3x3_plus":     {"variant": "3x3_plus", "n_feat": 793,  "train_px":  30_000, "val_px":  15_000},
    "mlp_3x3_plus_big": {"variant": "3x3_plus", "n_feat": 793,  "train_px":  30_000, "val_px":  15_000},
    "mlp_5x5":          {"variant": "5x5",      "n_feat": 1800, "train_px":  12_000, "val_px":   8_000},
}


def ts():
    return time.strftime("%H:%M:%S")


# ── Data Loading (streaming from TIFs, one city at a time) ───────────────────

def load_split(cities, max_pixels, variant, label="split"):
    """
    Load features for all cities in a split. One city in RAM at a time.
    Returns X (N, n_feat), y (N,) as numpy arrays.
    """
    feat_key = f"feat_{variant}"
    pad = 2 if variant == "5x5" else (1 if variant in ("3x3", "3x3_plus") else 0)
    all_X, all_y = [], []

    for i, city in enumerate(cities):
        if not city_has_raw_tifs(city):
            continue

        city_rng = np.random.RandomState(SEED + hash(city.name) % 10000)
        result = extract_pixels_for_city(city, max_pixels=max_pixels,
                                         pad=pad, rng=city_rng)
        if result is None:
            print(f"  [{i+1}/{len(cities)}] {city.name:25s} — SKIP")
            continue

        all_X.append(result[feat_key])
        all_y.append(result["labels"])
        n = result["n_pixels"]
        print(f"  [{i+1}/{len(cities)}] {city.name:25s} — {n:>7,} px")

        del result
        gc.collect()

    if not all_X:
        return None, None

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)
    del all_X, all_y
    gc.collect()

    print(f"  {label}: {X.shape[0]:,} × {X.shape[1]} ({X.nbytes / 1e9:.2f} GB)")
    return X, y


# ── Training Utilities ───────────────────────────────────────────────────────

def compute_class_weights(y, n_classes):
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    weights = np.ones(n_classes, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        weights[c] = total / (n_classes * cnt)
    return weights


# ── Training Loop ────────────────────────────────────────────────────────────

def train_single_model(model_name, device, max_epochs, batch_size, lr,
                       weight_decay):
    """
    Train a single MLP model:
      1. Load val from TIFs → scale → GPU
      2. Free val from CPU
      3. Load train from TIFs → CPU
      4. Train (batch CPU → GPU)
      5. Save + cleanup
    """
    spec = MODEL_SPECS[model_name]
    variant = spec["variant"]
    n_feat = spec["n_feat"]
    train_px = spec["train_px"]
    val_px = spec["val_px"]
    cfg = MODEL_CONFIGS[model_name]

    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"  {cfg['description']}")
    print(f"  Features: {n_feat}, Train px/city: {train_px:,}, Val px/city: {val_px:,}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # ── Step 1: Load VALIDATION → GPU ────────────────────────────────────
    print(f"\n[{ts()}] Loading validation data...")
    val_cities = get_val_cities()
    X_val, y_val = load_split(val_cities, val_px, variant, "Val")
    if X_val is None:
        print("  ERROR: No val data!")
        return None

    # Fit scaler on val first (we'll refit on train later)
    # Just need val to be on GPU before we load train
    print(f"\n[{ts()}] Loading training data...")
    train_cities = get_train_cities()
    X_train, y_train = load_split(train_cities, train_px, variant, "Train")
    if X_train is None:
        print("  ERROR: No train data!")
        del X_val, y_val
        return None

    # Fit scaler on train
    print(f"\n[{ts()}] Fitting scaler & transferring val to GPU...")
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Scale and move val to GPU, then free CPU copy
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_val_gpu = torch.from_numpy(X_val_scaled).to(device)
    y_val_gpu = torch.from_numpy(y_val.astype(np.int64)).to(device)
    del X_val, y_val, X_val_scaled
    gc.collect()

    print(f"  Val on GPU: {X_val_gpu.shape}")

    # Scale train in-place
    X_train = scaler.transform(X_train).astype(np.float32)

    n_train = len(X_train)
    n_val = len(X_val_gpu)

    # Class distribution
    for sn, sy in [("Train", y_train), ("Val", y_val_gpu.cpu().numpy())]:
        cls, cnt = np.unique(sy, return_counts=True)
        total = cnt.sum()
        dist = " ".join(f"{CLASS_NAMES[c][:4]}={100*n/total:.1f}%"
                        for c, n in zip(cls, cnt))
        print(f"  {sn}: {dist}")

    # Class weights
    class_weights = compute_class_weights(y_train, N_CLASSES)
    weights_gpu = torch.from_numpy(class_weights).to(device)

    # ── Step 2: Build model ──────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model, desc = build_model(model_name, N_CLASSES, device)
    n_params = model.n_params()
    print(f"\n  Model: {n_params:,} parameters")

    criterion = nn.CrossEntropyLoss(weight=weights_gpu)

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            fused=(device == "cuda"))
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = device == "cuda"
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * 3

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)),
    ], milestones=[warmup_steps])

    # ── Step 3: Train ────────────────────────────────────────────────────
    patience = max(math.ceil(5000 / steps_per_epoch), 20)
    min_epochs = max(math.ceil(3000 / steps_per_epoch), 15)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    print(f"\n[{ts()}] Training ({max_epochs} max epochs, patience={patience})...\n")

    for epoch in range(max_epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(y_train[idx].astype(np.int64)).to(
                device, non_blocking=True)

            if xb.size(0) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            amp_device = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(amp_device, enabled=use_amp,
                                     dtype=torch.float16):
                logits = model.backbone(model.input_drop(xb))
                logits = model.head(logits)
                loss = criterion(logits, yb)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            amp_device = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(amp_device, enabled=use_amp,
                                     dtype=torch.float16):
                val_logits = model.head(model.backbone(model.input_drop(X_val_gpu)))
                val_loss = criterion(val_logits, y_val_gpu).item()
                val_acc = (val_logits.argmax(1) == y_val_gpu).float().mean().item()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch <= 3 or epoch % 10 == 0 or improved:
            avg = epoch_loss / max(n_batches, 1)
            marker = " *" if improved else ""
            print(f"  Ep {epoch:3d}: train={avg:.5f} val={val_loss:.5f} "
                  f"acc={val_acc:.4f} wait={wait}{marker}")

        if epoch >= min_epochs and wait >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Step 4: Save ─────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, f"{model_name}.pt")
    torch.save(best_state or model.state_dict(), save_path)

    with open(os.path.join(OUT_DIR, f"{model_name}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Final val accuracy
    with torch.no_grad():
        preds = model.predict(X_val_gpu).cpu().numpy()
    y_val_np = y_val_gpu.cpu().numpy()
    top1 = (y_val_np == preds.argmax(1)).mean()

    result = {
        "model": model_name,
        "variant": variant,
        "n_features": n_feat,
        "n_params": n_params,
        "n_train": n_train,
        "n_val": n_val,
        "train_px_per_city": train_px,
        "val_px_per_city": val_px,
        "best_val_loss": float(best_val_loss),
        "val_accuracy": float(top1),
        "epochs_trained": epoch + 1,
    }

    for ci in range(N_CLASSES):
        mask = y_val_np == ci
        if mask.sum() > 0:
            result[f"acc_{CLASS_NAMES[ci]}"] = float(
                (preds[mask].argmax(1) == ci).mean())

    with open(os.path.join(OUT_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Model saved: {save_path}")
    print(f"  Val accuracy: {top1:.4f} ({top1*100:.2f}%)")
    for ci in range(N_CLASSES):
        k = f"acc_{CLASS_NAMES[ci]}"
        if k in result:
            print(f"    {CLASS_NAMES[ci]:>15}: {result[k]:.4f}")

    # ── Step 5: Test evaluation ───────────────────────────────────────────
    print(f"\n[{ts()}] Evaluating on test cities...")
    del X_train, y_train, X_val_gpu, y_val_gpu, weights_gpu
    del optimizer, grad_scaler, scheduler, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    test_result = evaluate_on_test(model, scaler, variant, device)
    if test_result:
        result["test_accuracy"] = test_result["overall_accuracy"]
        result["test_per_class"] = test_result["per_class"]
        result["test_per_city"] = test_result["per_city"]

        with open(os.path.join(OUT_DIR, f"{model_name}_metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


# ── Test Evaluation ──────────────────────────────────────────────────────────

def evaluate_on_test(model, scaler, variant, device):
    """
    Evaluate model on ALL pixels from each test city.
    Loads one city at a time to avoid memory issues.
    """
    from reproduce.models.shared.data import (
        load_raw_feature_cube, load_pixel_labels,
    )

    test_cities = get_test_cities()
    available = [c for c in test_cities if city_has_raw_tifs(c)]

    if not available:
        print("  No test cities with raw TIFs")
        return None

    all_correct = 0
    all_total = 0
    class_correct = np.zeros(N_CLASSES, dtype=np.int64)
    class_total = np.zeros(N_CLASSES, dtype=np.int64)
    per_city = {}

    model.eval()
    feat_key = f"feat_{variant}"
    pad = 2 if variant == "5x5" else (1 if variant in ("3x3", "3x3_plus") else 0)

    # Cap test pixels based on feature size to prevent OOM
    # 1×1: use all pixels (tiny features), 3×3/5×5: cap to fit RAM
    test_max_px = {
        "1x1":      5_000_000,   # 5M × 72 × 4  = 1.4 GB
        "3x3":        500_000,   # 500K × 648 × 4 = 1.3 GB
        "3x3_plus":   500_000,   # 500K × 771 × 4 = 1.5 GB
        "5x5":        200_000,   # 200K × 1800 × 4 = 1.4 GB
    }[variant]

    for city in available:
        result = extract_pixels_for_city(
            city, max_pixels=test_max_px, pad=pad,
            rng=np.random.RandomState(SEED))
        if result is None:
            print(f"  {city.name:25s} — SKIP")
            continue

        X_city = result[feat_key].astype(np.float32)
        y_city = result["labels"]
        n = result["n_pixels"]
        del result

        # Scale
        X_city = scaler.transform(X_city).astype(np.float32)

        # Predict in batches
        BATCH = 32768
        preds = []
        with torch.no_grad():
            for start in range(0, n, BATCH):
                xb = torch.from_numpy(X_city[start:start+BATCH]).to(device)
                pb = model.predict(xb).cpu().numpy()
                preds.append(pb.argmax(1))
        pred_classes = np.concatenate(preds)
        del X_city, preds
        gc.collect()

        correct = (pred_classes == y_city).sum()
        city_acc = correct / n
        per_city[city.name] = {"accuracy": float(city_acc), "n_pixels": int(n)}

        all_correct += correct
        all_total += n

        for ci in range(N_CLASSES):
            mask = y_city == ci
            class_total[ci] += mask.sum()
            class_correct[ci] += (pred_classes[mask] == ci).sum()

        del y_city, pred_classes
        gc.collect()

        print(f"  {city.name:25s} — {n:>9,} px, acc={city_acc:.4f}")

    overall = all_correct / max(all_total, 1)
    per_class_acc = {}
    print(f"\n  Test Overall: {overall:.4f} ({overall*100:.2f}%)")
    for ci in range(N_CLASSES):
        if class_total[ci] > 0:
            acc = class_correct[ci] / class_total[ci]
            per_class_acc[CLASS_NAMES[ci]] = float(acc)
            print(f"    {CLASS_NAMES[ci]:>15}: {acc:.4f} ({class_total[ci]:,} px)")

    return {
        "overall_accuracy": float(overall),
        "total_pixels": int(all_total),
        "per_class": per_class_acc,
        "per_city": per_city,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train raw-band MLP models")
    parser.add_argument("--models", nargs="*",
                        default=["mlp_1x1", "mlp_3x3", "mlp_3x3_plus", "mlp_3x3_plus_big", "mlp_5x5"],
                        choices=["mlp_1x1", "mlp_3x3", "mlp_3x3_plus", "mlp_3x3_plus_big", "mlp_5x5"])
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  Raw-Band MLP Training")
    print(f"  Device: {device}")
    print(f"  Models: {args.models}")
    print(f"  Max epochs: {args.max_epochs}, Batch: {args.batch_size}")
    print(f"{'='*70}\n")

    all_results = {}

    for model_name in args.models:
        result = train_single_model(
            model_name, device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if result:
            all_results[model_name] = result

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<12} {'Feat':>6} {'Params':>10} {'Train':>10} {'Val Acc':>8}")
        print(f"  {'-'*50}")
        for name, r in all_results.items():
            print(f"  {name:<12} {r['n_features']:>6} {r['n_params']:>10,} "
                  f"{r['n_train']:>10,} {r['val_accuracy']:>8.4f}")

        with open(os.path.join(OUT_DIR, "comparison.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
