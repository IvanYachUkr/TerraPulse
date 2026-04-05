#!/usr/bin/env python3
"""
Train TempCNN models for pixel-wise land cover classification.

Applies 1D convolutions along the temporal axis instead of treating
features as a flat vector. Reuses the same train/val/test split and
data pipeline as the MLP study.

Memory strategy (same as MLP pipeline):
  1. Load val data -> scale -> move to GPU -> free CPU copy
  2. Load train data -> keep on CPU
  3. Train with CPU->GPU batches

Variants:
  tempcnn_1x1:       (6, 12) center pixel, ~100K params
  tempcnn_3x3:       (6, 108) 3x3 patch, ~500K params
  tempcnn_1x1_plus:  (6, 22) temporal + (85,) cross-time, ~200K params

Usage:
    python 03_train_tempcnn.py --model tempcnn_1x1
    python 03_train_tempcnn.py --model tempcnn_3x3
    python 03_train_tempcnn.py --model tempcnn_1x1_plus
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
    SEED, N_CLASSES, CLASS_NAMES, N_RAW_FEATURES,
    get_train_cities, get_val_cities, get_test_cities, city_has_raw_tifs,
)
from reproduce.models.shared.data import (
    extract_pixels_for_city, compute_center_indices, N_HYBRID_EXTRA,
)
from reproduce.models.architectures.tempcnn import (
    MODEL_CONFIGS, build_tempcnn, TempCNNPlus,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")

# ── Per-model specs ──
MODEL_SPECS = {
    "tempcnn_1x1":      {"variant": "1x1", "train_px": 200_000, "val_px": 100_000, "pad": 0},
    "tempcnn_3x3":      {"variant": "3x3", "train_px":  30_000, "val_px":  15_000, "pad": 1},
    "tempcnn_1x1_plus": {"variant": "1x1", "train_px": 200_000, "val_px": 100_000, "pad": 0},
}

N_BANDS_PER_STEP = 12        # S2(10) + S1(2)
N_TEMPORAL_SLOTS = 6         # 2 years x 3 seasons
N_TEMPORAL_PLUS_CH = 22      # 12 raw + 9 indices + 1 SAR ratio
N_CROSSTIME = 85             # diffs, ranges, SAR diffs


def ts():
    return time.strftime("%H:%M:%S")


# ── Reshape functions ────────────────────────────────────────────────────────

def reshape_to_temporal(X_flat, variant):
    """Reshape flat features to (N, T, C) for temporal convolution."""
    N = X_flat.shape[0]

    if variant == "1x1":
        # (N, 72) -> (N, 6, 12)
        return X_flat.reshape(N, N_TEMPORAL_SLOTS, N_BANDS_PER_STEP)
    elif variant == "3x3":
        # (N, 648) -> (N, 6, 108)  [9 pixels x 12 bands per time step]
        # Original layout: for each pixel in 3x3, bands are interleaved as
        # [year0_spring_B02..B12_VV_VH, year0_summer_..., ..., year1_autumn_...]
        # We need to deinterleave: group all 9 pixels' bands for each time step
        n_pix = 9
        out = np.empty((N, N_TEMPORAL_SLOTS, n_pix * N_BANDS_PER_STEP), dtype=np.float32)
        for px in range(n_pix):
            px_start = px * (N_TEMPORAL_SLOTS * N_BANDS_PER_STEP)
            for t in range(N_TEMPORAL_SLOTS):
                src_start = px_start + t * N_BANDS_PER_STEP
                dst_start = px * N_BANDS_PER_STEP
                out[:, t, dst_start:dst_start + N_BANDS_PER_STEP] = \
                    X_flat[:, src_start:src_start + N_BANDS_PER_STEP]
        return out
    else:
        raise ValueError(f"Unknown variant: {variant}")


def build_plus_features(feat_1x1):
    """
    Build temporal + cross-time features for tempcnn_1x1_plus.

    Per-slot (22 per step, 6 steps = 132 total):
      - 12 raw bands
      - 9 spectral indices (NDVI, NDWI, etc.)
      - 1 SAR VV/VH ratio

    Cross-time (85 total):
      - 36 seasonal index diffs
      - 27 inter-annual index diffs
      - 8 range features
      - 8 SAR seasonal diffs
      - 6 SAR inter-annual diffs

    Returns: (temporal: (N, 6, 22), crosstime: (N, 85))
    """
    N = feat_1x1.shape[0]

    # Compute indices in chunks to avoid OOM (nan_to_num allocates bool array)
    CHUNK = 500_000
    extra = np.empty((N, 145), dtype=np.float32)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        extra[start:end] = compute_center_indices(feat_1x1[start:end])
    print(f"    Computed indices: {extra.shape}")

    # Layout of compute_center_indices output:
    # [0:54]   indices per slot (9 x 6 = 54)
    # [54:90]  seasonal diffs (36)
    # [90:117] inter-annual diffs (27)
    # [117:125] range (8)
    # [125:131] SAR ratios per slot (6)
    # [131:139] SAR seasonal diffs (8)
    # [139:145] SAR inter-annual diffs (6)

    # Build per-slot temporal features
    temporal = np.empty((N, 6, N_TEMPORAL_PLUS_CH), dtype=np.float32)
    for slot in range(6):
        off = slot * N_BANDS_PER_STEP
        # 12 raw bands
        temporal[:, slot, :12] = feat_1x1[:, off:off + 12]
        # 9 indices
        temporal[:, slot, 12:21] = extra[:, slot * 9:(slot + 1) * 9]
        # 1 SAR ratio
        temporal[:, slot, 21] = extra[:, 125 + slot]

    # Cross-time features
    crosstime = np.concatenate([
        extra[:, 54:90],    # 36 seasonal diffs
        extra[:, 90:117],   # 27 inter-annual diffs
        extra[:, 117:125],  # 8 range
        extra[:, 131:139],  # 8 SAR seasonal diffs
        extra[:, 139:145],  # 6 SAR inter-annual diffs
    ], axis=1)  # (N, 85)

    del extra; gc.collect()
    return temporal, crosstime


# ── Data loading ─────────────────────────────────────────────────────────────

def load_split(cities, max_pixels, model_name, label="split"):
    """Load and reshape features for a split."""
    spec = MODEL_SPECS[model_name]
    variant = spec["variant"]
    pad = spec["pad"]
    is_plus = model_name == "tempcnn_1x1_plus"

    feat_key = "feat_1x1" if variant == "1x1" else "feat_3x3"
    all_X, all_y = [], []

    for i, city in enumerate(cities):
        if not city_has_raw_tifs(city):
            continue
        city_rng = np.random.RandomState(SEED + hash(city.name) % 10000)
        result = extract_pixels_for_city(city, max_pixels=max_pixels,
                                         pad=pad, rng=city_rng)
        if result is None:
            print(f"  [{i+1}/{len(cities)}] {city.name:25s} - SKIP")
            continue
        all_X.append(result[feat_key])
        all_y.append(result["labels"])
        print(f"  [{i+1}/{len(cities)}] {city.name:25s} - {result['n_pixels']:>7,} px")
        del result; gc.collect()

    if not all_X:
        return None, None, None

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)
    del all_X, all_y; gc.collect()

    print(f"  {label}: {X.shape[0]:,} x {X.shape[1]} ({X.nbytes / 1e9:.2f} GB)")
    return X, y, variant


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model_name, device, max_epochs, batch_size, lr, weight_decay):
    spec = MODEL_SPECS[model_name]
    variant = spec["variant"]
    is_plus = model_name == "tempcnn_1x1_plus"

    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"  {MODEL_CONFIGS[model_name]['description']}")
    print(f"  Variant: {variant}, Train px/city: {spec['train_px']:,}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # ── 1. Load VALIDATION ────────────────────────────────────────────────
    print(f"\n[{ts()}] Loading validation data...")
    val_cities = get_val_cities()
    X_val_raw, y_val, _ = load_split(val_cities, spec["val_px"], model_name, "Val")
    if X_val_raw is None:
        print("  ERROR: No val data!"); return None

    # ── 2. Load TRAINING ──────────────────────────────────────────────────
    print(f"\n[{ts()}] Loading training data...")
    train_cities = get_train_cities()
    X_train_raw, y_train, _ = load_split(train_cities, spec["train_px"], model_name, "Train")
    if X_train_raw is None:
        print("  ERROR: No train data!"); del X_val_raw, y_val; return None

    # ── 3. Fit scaler on flat features → transform → reshape ─────────────
    print(f"\n[{ts()}] Fitting scaler & reshaping...")
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_train_raw = scaler.transform(X_train_raw).astype(np.float32)
    del X_val_raw; gc.collect()

    # Reshape for temporal processing
    if is_plus:
        # Unscale for index computation (indices need raw reflectances)
        # Actually, compute indices BEFORE scaling
        # ... We need the unscaled features for index computation.
        # Let's redo: compute plus features from unscaled, then scale separately.
        pass  # handled below

    if is_plus:
        # For plus model, we need to recompute from unscaled features
        # So let's reload cleanly
        del X_val_scaled, X_train_raw
        gc.collect()

        # Reload and compute plus features before scaling
        print(f"\n[{ts()}] Rebuilding plus features (unscaled for index computation)...")

        # Val
        X_val_raw2, y_val2, _ = load_split(val_cities, spec["val_px"], model_name, "Val (re)")
        val_temporal, val_crosstime = build_plus_features(X_val_raw2)
        del X_val_raw2; gc.collect()

        # Train
        X_train_raw2, y_train2, _ = load_split(train_cities, spec["train_px"], model_name, "Train (re)")
        train_temporal, train_crosstime = build_plus_features(X_train_raw2)
        del X_train_raw2; gc.collect()
        y_val = y_val2; y_train = y_train2

        # Scale temporal and crosstime separately
        n_t = train_temporal.shape[0]
        temporal_flat_train = train_temporal.reshape(n_t, -1)
        temporal_flat_val = val_temporal.reshape(len(y_val), -1)

        scaler_temporal = StandardScaler()
        scaler_temporal.fit(temporal_flat_train)
        temporal_flat_train = scaler_temporal.transform(temporal_flat_train).astype(np.float32)
        temporal_flat_val = scaler_temporal.transform(temporal_flat_val).astype(np.float32)

        scaler_crosstime = StandardScaler()
        scaler_crosstime.fit(train_crosstime)
        train_crosstime = scaler_crosstime.transform(train_crosstime).astype(np.float32)
        val_crosstime_s = scaler_crosstime.transform(val_crosstime).astype(np.float32)

        train_temporal = temporal_flat_train.reshape(n_t, 6, N_TEMPORAL_PLUS_CH)
        val_temporal_s = temporal_flat_val.reshape(len(y_val), 6, N_TEMPORAL_PLUS_CH)

        # Keep val on CPU (GPU too small for full val set)
        X_val_temporal = val_temporal_s
        X_val_crosstime = val_crosstime_s
        y_val_np = y_val.astype(np.int64)
        del temporal_flat_val, val_temporal, val_crosstime
        gc.collect()

        print(f"  Val on CPU: temporal {X_val_temporal.shape}, crosstime {X_val_crosstime.shape}")

        # Scalers to save
        scaler_to_save = {"temporal": scaler_temporal, "crosstime": scaler_crosstime}

    else:
        # Standard reshape
        X_val_temporal = reshape_to_temporal(X_val_scaled, variant)
        del X_val_scaled

        X_val_crosstime = None
        y_val_np = y_val.astype(np.int64)
        del y_val; gc.collect()

        print(f"  Val on CPU: {X_val_temporal.shape}")

        train_crosstime = None

        # Reshape train
        X_train_temporal = reshape_to_temporal(X_train_raw, variant)
        del X_train_raw; gc.collect()
        train_temporal = X_train_temporal

        scaler_to_save = scaler

    n_train = len(y_train)
    n_val = len(y_val_np)

    # Class distribution
    for sn, sy in [("Train", y_train), ("Val", y_val_np)]:
        cls, cnt = np.unique(sy, return_counts=True)
        total = cnt.sum()
        dist = " ".join(f"{CLASS_NAMES[c][:4]}={100*n/total:.1f}%"
                        for c, n in zip(cls, cnt))
        print(f"  {sn}: {dist}")

    # Class weights
    classes, counts = np.unique(y_train, return_counts=True)
    cw = np.ones(N_CLASSES, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        cw[c] = counts.sum() / (N_CLASSES * cnt)
    weights_gpu = torch.from_numpy(cw).to(device)

    # ── 4. Build model ────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model, desc = build_tempcnn(model_name, N_CLASSES, device)
    n_params = model.n_params()
    print(f"\n  Model: {n_params:,} parameters")

    criterion = nn.CrossEntropyLoss(weight=weights_gpu)
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

    # ── 5. Train ──────────────────────────────────────────────────────────
    patience = 20
    min_epochs = 15
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    print(f"\n[{ts()}] Training ({max_epochs} max epochs, patience={patience})...\n")

    for epoch in range(max_epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]

            xb_t = torch.from_numpy(train_temporal[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(y_train[idx].astype(np.int64)).to(device, non_blocking=True)

            if xb_t.size(0) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            amp_device = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(amp_device, enabled=use_amp, dtype=torch.float16):
                if is_plus:
                    xb_c = torch.from_numpy(train_crosstime[idx]).to(device, non_blocking=True)
                    logits = model(xb_t, xb_c)
                else:
                    logits = model(xb_t)
                loss = nn.functional.nll_loss(logits, yb, weight=weights_gpu)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # ── Validation (batched, val stays on CPU) ──
        model.eval()
        val_loss_sum, val_correct, val_counted = 0.0, 0, 0
        VAL_BATCH = 65536
        with torch.no_grad():
            amp_device = "cuda" if use_amp else "cpu"
            for vs in range(0, n_val, VAL_BATCH):
                ve = min(vs + VAL_BATCH, n_val)
                vt = torch.from_numpy(X_val_temporal[vs:ve]).to(device)
                vy = torch.from_numpy(y_val_np[vs:ve]).to(device)
                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=torch.float16):
                    if is_plus:
                        vc = torch.from_numpy(X_val_crosstime[vs:ve]).to(device)
                        vout = model(vt, vc)
                    else:
                        vout = model(vt)
                    val_loss_sum += nn.functional.nll_loss(
                        vout, vy, weight=weights_gpu, reduction='sum').item()
                val_correct += (vout.argmax(1) == vy).sum().item()
                val_counted += ve - vs
        val_loss = val_loss_sum / val_counted
        val_acc = val_correct / val_counted

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

    # ── 6. Save ───────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(best_state or model.state_dict(),
               os.path.join(OUT_DIR, f"{model_name}.pt"))
    with open(os.path.join(OUT_DIR, f"{model_name}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_to_save, f)

    # Final val metrics (batched)
    all_preds = []
    with torch.no_grad():
        for vs in range(0, n_val, 65536):
            ve = min(vs + 65536, n_val)
            vt = torch.from_numpy(X_val_temporal[vs:ve]).to(device)
            if is_plus:
                vc = torch.from_numpy(X_val_crosstime[vs:ve]).to(device)
                pb = model.predict(vt, vc).cpu().numpy()
            else:
                pb = model.predict(vt).cpu().numpy()
            all_preds.append(pb)
    preds = np.concatenate(all_preds)
    top1 = (y_val_np == preds.argmax(1)).mean()

    result = {
        "model": model_name,
        "n_params": n_params,
        "n_train": n_train,
        "n_val": n_val,
        "best_val_loss": float(best_val_loss),
        "val_accuracy": float(top1),
        "epochs_trained": epoch + 1,
    }
    for ci in range(N_CLASSES):
        mask = y_val_np == ci
        if mask.sum() > 0:
            result[f"acc_{CLASS_NAMES[ci]}"] = float(
                (preds[mask].argmax(1) == ci).mean())

    print(f"\n  Model saved: {model_name}.pt")
    print(f"  Val accuracy: {top1:.4f} ({top1*100:.2f}%)")
    for ci in range(N_CLASSES):
        k = f"acc_{CLASS_NAMES[ci]}"
        if k in result:
            print(f"    {CLASS_NAMES[ci]:>15}: {result[k]:.4f}")

    # ── 7. Test evaluation ────────────────────────────────────────────────
    print(f"\n[{ts()}] Evaluating on test cities...")
    del X_val_temporal, y_val_np, weights_gpu, optimizer, grad_scaler, scheduler
    if X_val_crosstime is not None:
        del X_val_crosstime
    del best_state, train_temporal
    if train_crosstime is not None:
        del train_crosstime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    test_result = evaluate_test(model, scaler_to_save, model_name, device)
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

def evaluate_test(model, scaler_obj, model_name, device):
    spec = MODEL_SPECS[model_name]
    variant = spec["variant"]
    is_plus = model_name == "tempcnn_1x1_plus"
    pad = spec["pad"]

    feat_key = "feat_1x1" if variant == "1x1" else "feat_3x3"
    test_max_px = 5_000_000 if variant == "1x1" else 500_000

    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    if not test_cities:
        print("  No test cities"); return None

    all_correct, all_total = 0, 0
    class_correct = np.zeros(N_CLASSES, dtype=np.int64)
    class_total = np.zeros(N_CLASSES, dtype=np.int64)
    per_city = {}
    model.eval()

    for city in test_cities:
        result = extract_pixels_for_city(
            city, max_pixels=test_max_px, pad=pad,
            rng=np.random.RandomState(SEED))
        if result is None:
            print(f"  {city.name:25s} - SKIP"); continue

        X_raw = result[feat_key].astype(np.float32)
        y = result["labels"]
        n = result["n_pixels"]
        del result; gc.collect()

        if is_plus:
            temporal, crosstime = build_plus_features(X_raw)
            del X_raw

            st = scaler_obj["temporal"]
            sc = scaler_obj["crosstime"]
            temporal_flat = st.transform(temporal.reshape(n, -1)).astype(np.float32)
            temporal = temporal_flat.reshape(n, 6, N_TEMPORAL_PLUS_CH)
            crosstime = sc.transform(crosstime).astype(np.float32)

            BATCH = 32768
            preds = []
            with torch.no_grad():
                for s in range(0, n, BATCH):
                    xt = torch.from_numpy(temporal[s:s+BATCH]).to(device)
                    xc = torch.from_numpy(crosstime[s:s+BATCH]).to(device)
                    pb = model.predict(xt, xc).cpu().numpy()
                    preds.append(pb.argmax(1))
            del temporal, crosstime
        else:
            if isinstance(scaler_obj, dict):
                X_scaled = X_raw  # shouldn't happen for non-plus
            else:
                X_scaled = scaler_obj.transform(X_raw).astype(np.float32)
            del X_raw
            X_temporal = reshape_to_temporal(X_scaled, variant)
            del X_scaled

            BATCH = 32768
            preds = []
            with torch.no_grad():
                for s in range(0, n, BATCH):
                    xb = torch.from_numpy(X_temporal[s:s+BATCH]).to(device)
                    pb = model.predict(xb).cpu().numpy()
                    preds.append(pb.argmax(1))
            del X_temporal

        pred_classes = np.concatenate(preds)
        correct = (pred_classes == y).sum()
        city_acc = correct / n
        per_city[city.name] = {"accuracy": float(city_acc), "n_pixels": int(n)}
        all_correct += correct
        all_total += n
        for ci in range(N_CLASSES):
            mask = y == ci
            class_total[ci] += mask.sum()
            class_correct[ci] += (pred_classes[mask] == ci).sum()
        del y, pred_classes, preds; gc.collect()
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TempCNN models")
    parser.add_argument("--model", default="tempcnn_1x1",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    result = train_model(
        args.model, device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if result:
        print(f"\n{'='*70}")
        print(f"  DONE: {args.model}")
        print(f"  Val: {result['val_accuracy']:.4f}")
        if "test_accuracy" in result:
            print(f"  Test: {result['test_accuracy']:.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
