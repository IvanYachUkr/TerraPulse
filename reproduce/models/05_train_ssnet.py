#!/usr/bin/env python3
"""
Train SpectralSpatialNet for pixel-wise land cover classification.

Architecture: depthwise spatial conv + temporal attention + index branch.

Memory budget: ~6 GB RAM total
  - Train: 15K px/city × 92 cities × 648 raw = 3.6 GB
  - Indices computed in 500K chunks from raw features = +0.8 GB
  - Val: 8K px/city × 23 cities × 648 = 0.47 GB + indices 0.06 GB

Usage:
    python 05_train_ssnet.py
    python 05_train_ssnet.py --max-epochs 10   # smoke test
"""

import argparse, gc, json, math, os, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    SEED, N_CLASSES, CLASS_NAMES,
    get_train_cities, get_val_cities, get_test_cities, city_has_raw_tifs,
)
from reproduce.models.shared.data import (
    extract_pixels_for_city, compute_center_indices,
)
from reproduce.models.architectures.spectral_spatial import SpectralSpatialNetV2

OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")

TRAIN_PX = 60_000    # per city (fp16 storage: ~8.7 GB total)
VAL_PX = 10_000      # per city
PAD = 1              # for 3×3 patches
N_INDICES = 145


def ts():
    return time.strftime("%H:%M:%S")


def compute_indices_chunked(X_1x1, chunk_size=500_000):
    """Compute center-pixel indices in chunks to avoid OOM."""
    N = X_1x1.shape[0]
    result = np.empty((N, N_INDICES), dtype=np.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        result[start:end] = compute_center_indices(X_1x1[start:end])
    return result


def load_split(cities, max_px, label="split", use_fp16=False):
    """Load 3×3 patches and compute center-pixel indices."""
    all_patches, all_y = [], []

    for i, city in enumerate(cities):
        if not city_has_raw_tifs(city):
            continue
        rng = np.random.RandomState(SEED + hash(city.name) % 10000)
        result = extract_pixels_for_city(city, max_pixels=max_px,
                                         pad=PAD, rng=rng)
        if result is None:
            print(f"  [{i+1}/{len(cities)}] {city.name:25s} - SKIP")
            continue

        store_dtype = np.float16 if use_fp16 else np.float32
        all_patches.append(result["feat_3x3"].astype(store_dtype))
        all_y.append(result["labels"])
        print(f"  [{i+1}/{len(cities)}] {city.name:25s} - {result['n_pixels']:>7,} px")
        del result; gc.collect()

    if not all_patches:
        return None, None, None

    patches = np.concatenate(all_patches)
    y = np.concatenate(all_y).astype(np.int32)
    del all_patches, all_y; gc.collect()

    # Extract center pixel — need float32 for index computation (division ratios)
    center_1x1 = patches[:, 4*72:5*72].astype(np.float32)

    print(f"  Computing indices ({patches.shape[0]:,} pixels)...")
    indices = compute_indices_chunked(center_1x1)
    if use_fp16:
        indices = indices.astype(np.float16)
    del center_1x1; gc.collect()

    mem = (patches.nbytes + indices.nbytes) / 1e9
    dt = 'fp16' if use_fp16 else 'fp32'
    print(f"  {label}: {patches.shape[0]:,} px [{dt}], patches {patches.nbytes/1e9:.2f} GB + "
          f"indices {indices.nbytes/1e9:.2f} GB = {mem:.2f} GB total")
    return patches, indices, y


def train(device, max_epochs, batch_size, lr, weight_decay):
    print(f"\n{'='*70}")
    print(f"  SpectralSpatialNet Training")
    print(f"  Train: {TRAIN_PX:,} px/city, Val: {VAL_PX:,} px/city")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # ── Load data ──
    print(f"\n[{ts()}] Loading validation data...")
    val_cities = get_val_cities()
    val_patches, val_indices, val_y = load_split(val_cities, VAL_PX, "Val", use_fp16=False)
    if val_patches is None:
        print("ERROR: No val data!"); return None

    print(f"\n[{ts()}] Loading training data (fp16 to save RAM)...")
    train_cities = get_train_cities()
    train_patches, train_indices, train_y = load_split(train_cities, TRAIN_PX, "Train", use_fp16=True)
    if train_patches is None:
        print("ERROR: No train data!"); return None

    # ── Scale features ──
    # Fit scaler on float32 subsample (StandardScaler needs precision)
    print(f"\n[{ts()}] Fitting scalers...")
    n_train = len(train_y)
    scaler_sample = min(500_000, n_train)
    scaler_idx = np.random.RandomState(SEED).choice(n_train, scaler_sample, replace=False)

    patch_scaler = StandardScaler()
    patch_scaler.fit(train_patches[scaler_idx].astype(np.float32))
    # Transform val to float32 (goes to GPU)
    val_patches = patch_scaler.transform(val_patches).astype(np.float32)
    # Transform train in-place to fp16
    # Process in chunks to avoid fp32 peak
    SCALE_CHUNK = 200_000
    for s in range(0, n_train, SCALE_CHUNK):
        e = min(s + SCALE_CHUNK, n_train)
        train_patches[s:e] = patch_scaler.transform(
            train_patches[s:e].astype(np.float32)).astype(np.float16)

    idx_scaler = StandardScaler()
    idx_scaler.fit(train_indices[scaler_idx].astype(np.float32))
    val_indices = idx_scaler.transform(val_indices).astype(np.float32)
    for s in range(0, n_train, SCALE_CHUNK):
        e = min(s + SCALE_CHUNK, n_train)
        train_indices[s:e] = idx_scaler.transform(
            train_indices[s:e].astype(np.float32)).astype(np.float16)

    n_train = len(train_y)
    n_val = len(val_y)

    # Convert train to torch tensors on CPU (avoids numpy→torch each batch)
    print(f"  Converting train to torch tensors...")
    train_patches_t = torch.from_numpy(train_patches).pin_memory()
    train_indices_t = torch.from_numpy(train_indices).pin_memory()
    train_y_t = torch.from_numpy(train_y.astype(np.int64)).pin_memory()
    del train_patches, train_indices; gc.collect()
    print(f"  Train tensors: patches {train_patches_t.shape} ({train_patches_t.dtype}), "
          f"indices {train_indices_t.shape}")

    # Move val to GPU (small enough: ~0.6 GB)
    val_patches_gpu = torch.from_numpy(val_patches).to(device)
    val_indices_gpu = torch.from_numpy(val_indices).to(device)
    val_y_gpu = torch.from_numpy(val_y.astype(np.int64)).to(device)
    del val_patches, val_indices, val_y; gc.collect()
    print(f"  Val on GPU: patches {val_patches_gpu.shape}, indices {val_indices_gpu.shape}")

    # Class distribution
    for sn, sy in [("Train", train_y), ("Val", val_y_gpu.cpu().numpy())]:
        cls, cnt = np.unique(sy, return_counts=True)
        total = cnt.sum()
        dist = " ".join(f"{CLASS_NAMES[c][:4]}={100*n/total:.1f}%"
                        for c, n in zip(cls, cnt))
        print(f"  {sn}: {dist}")

    # No class weights — matches CatBoost's plain CrossEntropy
    weights_gpu = None

    # ── Build model ──
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = SpectralSpatialNetV2(
        n_bands=12, n_timesteps=6, n_indices=N_INDICES,
        spatial_dims=(48, 96, 192), temporal_dim=192,
        n_attn_layers=3, n_heads=8,
        n_classes=N_CLASSES, dropout=0.15,
    ).to(device)
    n_params = model.n_params()
    print(f"\n  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    # ── Train loop ──
    patience = 5
    min_epochs = 10
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    print(f"\n[{ts()}] Training ({max_epochs} max epochs, patience={patience})...\n")

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xp = train_patches_t[idx].to(device, non_blocking=True)
            xi = train_indices_t[idx].to(device, non_blocking=True)
            yb = train_y_t[idx].to(device, non_blocking=True)

            if xp.size(0) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            amp_dev = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(amp_dev, enabled=use_amp, dtype=torch.float16):
                logits = model(xp, xi)
                loss = nn.functional.nll_loss(logits, yb, weight=weights_gpu)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # ── Validation (batched on GPU) ──
        model.eval()
        val_loss_sum, val_correct = 0.0, 0
        VB = 32768
        with torch.no_grad():
            amp_dev = "cuda" if use_amp else "cpu"
            for vs in range(0, n_val, VB):
                ve = min(vs + VB, n_val)
                with torch.amp.autocast(amp_dev, enabled=use_amp, dtype=torch.float16):
                    vout = model(val_patches_gpu[vs:ve], val_indices_gpu[vs:ve])
                    val_loss_sum += nn.functional.nll_loss(
                        vout, val_y_gpu[vs:ve], weight=weights_gpu,
                        reduction='sum').item()
                val_correct += (vout.argmax(1) == val_y_gpu[vs:ve]).sum().item()
        val_loss = val_loss_sum / n_val
        val_acc = val_correct / n_val

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            # Save best model + scalers to disk immediately
            os.makedirs(OUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "ssnet.pt"))
            with open(os.path.join(OUT_DIR, "ssnet_scaler.pkl"), "wb") as f:
                pickle.dump({"patches": patch_scaler, "indices": idx_scaler}, f)
            wait = 0
        else:
            wait += 1

        if epoch <= 5 or epoch % 10 == 0 or improved:
            avg = epoch_loss / max(n_batches, 1)
            marker = " *" if improved else ""
            print(f"  Ep {epoch:3d}: train={avg:.5f} val={val_loss:.5f} "
                  f"acc={val_acc:.4f} wait={wait}{marker}")

        if epoch >= min_epochs and wait >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Load best checkpoint from disk ──
    best_path = os.path.join(OUT_DIR, "ssnet.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    # Final val metrics (batched)
    all_preds = []
    with torch.no_grad():
        for vs in range(0, n_val, 32768):
            ve = min(vs + 32768, n_val)
            pb = model.predict(val_patches_gpu[vs:ve], val_indices_gpu[vs:ve])
            all_preds.append(pb.cpu().numpy())
    preds = np.concatenate(all_preds)
    val_y_np = val_y_gpu.cpu().numpy()
    top1 = (val_y_np == preds.argmax(1)).mean()

    result = {
        "model": "ssnet",
        "n_params": n_params,
        "n_train": n_train,
        "n_val": n_val,
        "best_val_loss": float(best_val_loss),
        "val_accuracy": float(top1),
        "epochs_trained": epoch + 1,
    }
    for ci in range(N_CLASSES):
        mask = val_y_np == ci
        if mask.sum() > 0:
            result[f"acc_{CLASS_NAMES[ci]}"] = float(
                (preds[mask].argmax(1) == ci).mean())

    print(f"\n  Model saved: ssnet.pt")
    print(f"  Val accuracy: {top1:.4f} ({top1*100:.2f}%)")
    for ci in range(N_CLASSES):
        k = f"acc_{CLASS_NAMES[ci]}"
        if k in result:
            print(f"    {CLASS_NAMES[ci]:>15}: {result[k]:.4f}")

    # ── Test ──
    print(f"\n[{ts()}] Evaluating on test cities...")
    del train_patches_t, train_indices_t, train_y_t, train_y
    del val_patches_gpu, val_indices_gpu, val_y_gpu, val_y_np, weights_gpu
    del optimizer, grad_scaler, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    test_result = evaluate_test(model, patch_scaler, idx_scaler, device)
    if test_result:
        result["test_accuracy"] = test_result["overall_accuracy"]
        result["test_per_class"] = test_result["per_class"]
        result["test_per_city"] = test_result["per_city"]

    with open(os.path.join(OUT_DIR, "ssnet_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def evaluate_test(model, patch_scaler, idx_scaler, device):
    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    if not test_cities:
        print("  No test cities"); return None

    all_correct, all_total = 0, 0
    class_correct = np.zeros(N_CLASSES, dtype=np.int64)
    class_total = np.zeros(N_CLASSES, dtype=np.int64)
    per_city = {}
    model.eval()

    for city in test_cities:
        rng = np.random.RandomState(SEED)
        result = extract_pixels_for_city(city, max_pixels=500_000,
                                         pad=PAD, rng=rng)
        if result is None:
            print(f"  {city.name:25s} - SKIP"); continue

        patches = result["feat_3x3"].astype(np.float32)
        y = result["labels"]
        n = result["n_pixels"]
        del result; gc.collect()

        # Compute indices from center pixel (pixel 4 in 3×3)
        center = patches[:, 4*72:5*72].copy()
        indices = compute_indices_chunked(center)
        del center; gc.collect()

        # Scale
        patches = patch_scaler.transform(patches).astype(np.float32)
        indices = idx_scaler.transform(indices).astype(np.float32)

        # Predict in batches
        BATCH = 16384
        preds = []
        with torch.no_grad():
            for s in range(0, n, BATCH):
                xp = torch.from_numpy(patches[s:s+BATCH]).to(device)
                xi = torch.from_numpy(indices[s:s+BATCH]).to(device)
                pb = model.predict(xp, xi).cpu().numpy()
                preds.append(pb.argmax(1))
        pred_classes = np.concatenate(preds)
        del patches, indices, preds; gc.collect()

        correct = (pred_classes == y).sum()
        city_acc = correct / n
        per_city[city.name] = {"accuracy": float(city_acc), "n_pixels": int(n)}
        all_correct += correct
        all_total += n
        for ci in range(N_CLASSES):
            mask = y == ci
            class_total[ci] += mask.sum()
            class_correct[ci] += (pred_classes[mask] == ci).sum()
        del y, pred_classes; gc.collect()
        print(f"  {city.name:25s} - {n:>9,} px, acc={city_acc:.4f}")

    overall = all_correct / max(all_total, 1)
    per_class = {}
    print(f"\n  Test Overall: {overall:.4f} ({overall*100:.2f}%)")
    for ci in range(N_CLASSES):
        if class_total[ci] > 0:
            acc = class_correct[ci] / class_total[ci]
            per_class[CLASS_NAMES[ci]] = float(acc)
            print(f"    {CLASS_NAMES[ci]:>15}: {acc:.4f} ({class_total[ci]:,} px)")

    return {"overall_accuracy": float(overall), "per_class": per_class,
            "per_city": per_city}


def main():
    parser = argparse.ArgumentParser(description="Train SpectralSpatialNet")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    result = train(device, args.max_epochs, args.batch_size,
                   args.lr, args.weight_decay)

    if result:
        print(f"\n{'='*70}")
        print(f"  DONE: SpectralSpatialNet")
        print(f"  Val: {result['val_accuracy']:.4f}")
        if "test_accuracy" in result:
            print(f"  Test: {result['test_accuracy']:.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
