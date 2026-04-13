#!/usr/bin/env python3
"""
Train SpectralSpatialNet V4 with iterative data resampling.

Each round:
  1. Samples a fresh 60K pixels/city from training TIFs (different random seed)
  2. Refits StandardScalers on the new sample
  3. Warm-starts model from best weights, fresh optimizer
  4. Trains until inner patience exhausted

This exposes the model to much more total data across rounds without
needing more RAM — each round uses the same ~8.7 GB memory budget.

Usage:
    python 05_train_ssnet.py                  # fresh start, 10 rounds
    python 05_train_ssnet.py --resume         # continue from last checkpoint
    python 05_train_ssnet.py --n-rounds 20    # more rounds
"""

import argparse, gc, json, math, os, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
VAL_CACHE_DIR = os.path.join(OUT_DIR, "val_raw_cache")

TRAIN_PX = 60_000    # per city per round (fp16 storage: ~8.7 GB)
VAL_PX = 5_000       # per city (on GPU, keep small for VRAM)
PAD = 1              # for 3×3 patches
N_INDICES = 145
SCALE_CHUNK = 200_000


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


def load_split(cities, max_px, label="split", use_fp16=False, round_seed=None):
    """Load 3×3 patches and compute center-pixel indices.

    Args:
        round_seed: If provided, each city gets seed = round_seed + city_index.
                    This ensures different data each round while being deterministic.
    """
    all_patches, all_y = [], []

    for i, city in enumerate(cities):
        if not city_has_raw_tifs(city):
            continue
        if round_seed is not None:
            city_seed = round_seed + i
        else:
            city_seed = SEED + abs(hash(city.name)) % 10000
        rng = np.random.RandomState(city_seed)
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

    # Extract center pixel for index computation
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


def print_class_dist(label, y):
    """Print class distribution."""
    cls, cnt = np.unique(y, return_counts=True)
    total = cnt.sum()
    dist = " ".join(f"{CLASS_NAMES[c][:4]}={100*n/total:.1f}%"
                    for c, n in zip(cls, cnt))
    print(f"  {label}: {dist}")


def fit_and_apply_scalers(train_patches, train_indices, n_train):
    """Fit scalers on a subsample, apply to train in-place (fp16)."""
    scaler_sample = min(500_000, n_train)
    scaler_idx = np.random.RandomState(SEED).choice(n_train, scaler_sample, replace=False)

    patch_scaler = StandardScaler()
    patch_scaler.fit(train_patches[scaler_idx].astype(np.float32))
    for s in range(0, n_train, SCALE_CHUNK):
        e = min(s + SCALE_CHUNK, n_train)
        train_patches[s:e] = patch_scaler.transform(
            train_patches[s:e].astype(np.float32)).astype(np.float16)

    idx_scaler = StandardScaler()
    idx_scaler.fit(train_indices[scaler_idx].astype(np.float32))
    for s in range(0, n_train, SCALE_CHUNK):
        e = min(s + SCALE_CHUNK, n_train)
        train_indices[s:e] = idx_scaler.transform(
            train_indices[s:e].astype(np.float32)).astype(np.float16)

    return patch_scaler, idx_scaler


def scale_val_to_gpu(patch_scaler, idx_scaler, device):
    """Load raw val from disk cache, scale with given scalers, move to GPU."""
    val_p = np.load(os.path.join(VAL_CACHE_DIR, "patches.npy"))
    val_i = np.load(os.path.join(VAL_CACHE_DIR, "indices.npy"))
    val_patches_gpu = torch.from_numpy(
        patch_scaler.transform(val_p).astype(np.float32)).to(device)
    val_indices_gpu = torch.from_numpy(
        idx_scaler.transform(val_i).astype(np.float32)).to(device)
    del val_p, val_i; gc.collect()
    return val_patches_gpu, val_indices_gpu


def validate(model, val_patches_gpu, val_indices_gpu, val_y_gpu, n_val, use_amp):
    """Run batched validation, return (accuracy, loss)."""
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
                    vout, val_y_gpu[vs:ve], reduction='sum').item()
            val_correct += (vout.argmax(1) == val_y_gpu[vs:ve]).sum().item()
    return val_correct / n_val, val_loss_sum / n_val


# ═══════════════════════════════════════════════════════════════════════
#  Main training function
# ═══════════════════════════════════════════════════════════════════════

def train(device, n_rounds, max_epochs_per_round, batch_size, lr,
          weight_decay, inner_patience, resume=False):
    train_cities = get_train_cities()
    val_cities = get_val_cities()

    print(f"\n{'='*70}")
    print(f"  SpectralSpatialNet V4 — Iterative Resampling")
    print(f"  Rounds: {n_rounds}, Max epochs/round: {max_epochs_per_round}")
    print(f"  Train: {TRAIN_PX:,} px/city, Val: {VAL_PX:,} px/city")
    print(f"  Inner patience: {inner_patience}, Batch: {batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # ── Load val ONCE, cache raw to disk, keep labels on GPU ──
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)
    val_cache_exists = all(
        os.path.exists(os.path.join(VAL_CACHE_DIR, f))
        for f in ["patches.npy", "indices.npy", "y.npy"]
    )

    if val_cache_exists:
        print(f"\n[{ts()}] Loading cached val data...")
        val_y = np.load(os.path.join(VAL_CACHE_DIR, "y.npy"))
    else:
        print(f"\n[{ts()}] Loading validation data (one-time)...")
        val_patches, val_indices, val_y = load_split(
            val_cities, VAL_PX, "Val", use_fp16=False)
        if val_patches is None:
            print("ERROR: No val data!"); return None
        np.save(os.path.join(VAL_CACHE_DIR, "patches.npy"), val_patches)
        np.save(os.path.join(VAL_CACHE_DIR, "indices.npy"), val_indices)
        np.save(os.path.join(VAL_CACHE_DIR, "y.npy"), val_y)
        del val_patches, val_indices; gc.collect()
        print(f"  Val raw cached to {VAL_CACHE_DIR}")

    n_val = len(val_y)
    val_y_gpu = torch.from_numpy(val_y.astype(np.int64)).to(device)
    print_class_dist("Val", val_y)
    del val_y; gc.collect()

    # ── Build model (V2 architecture: fast, proven) ──
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    model = SpectralSpatialNetV2(
        n_bands=12, n_timesteps=6, n_indices=N_INDICES,
        spatial_dims=(32, 64, 128), expand_ratio=4, temporal_dim=128,
        n_attn_layers=2, n_heads=8,
        n_classes=N_CLASSES, dropout=0.15,
    ).to(device)
    use_amp = device == "cuda"
    n_params = model.n_params()
    print(f"\n  Model: {n_params:,} parameters")

    # ── Resume support ──
    start_round = 0
    global_best_val_acc = 0.0
    ckpt_path = os.path.join(OUT_DIR, "ssnet_training.pt")
    best_model_path = os.path.join(OUT_DIR, "ssnet.pt")
    round_model_path = os.path.join(OUT_DIR, "ssnet_round.pt")

    if resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_round = ckpt.get("round_idx", 0) + 1
        global_best_val_acc = ckpt.get("global_best_val_acc", 0.0)
        print(f"  Resuming from round {start_round}, "
              f"global best acc={global_best_val_acc:.4f}")
        del ckpt; gc.collect()
    elif resume and os.path.exists(best_model_path):
        print(f"  Loading model weights from {best_model_path}")
        model.load_state_dict(torch.load(
            best_model_path, map_location=device, weights_only=True))
        print(f"  Weights loaded — starting iterative training from round 0")

    total_epochs_trained = 0

    # ═══════════════════════════════════════════════════════
    #  OUTER LOOP: iterate over data rounds
    # ═══════════════════════════════════════════════════════
    for round_idx in range(start_round, n_rounds):
        print(f"\n{'='*70}")
        print(f"  ROUND {round_idx+1}/{n_rounds}")
        print(f"{'='*70}")

        # ── Sample fresh training data ──
        # Use a prime multiplier for maximal seed diversity across rounds
        round_seed = SEED + round_idx * 7919
        print(f"\n[{ts()}] Loading training data (round seed={round_seed})...")
        train_patches, train_indices, train_y = load_split(
            train_cities, TRAIN_PX, f"Train R{round_idx+1}",
            use_fp16=True, round_seed=round_seed)
        if train_patches is None:
            print("ERROR: No train data!"); continue

        n_train = len(train_y)
        print_class_dist(f"Train R{round_idx+1}", train_y)

        # ── Fit scalers on this round's data ──
        print(f"\n[{ts()}] Fitting scalers...")
        patch_scaler, idx_scaler = fit_and_apply_scalers(
            train_patches, train_indices, n_train)

        # ── Scale val and move to GPU ──
        val_patches_gpu, val_indices_gpu = scale_val_to_gpu(
            patch_scaler, idx_scaler, device)
        print(f"  Val on GPU: {val_patches_gpu.shape}")

        # ── Fresh optimizer + scheduler (model stays warm) ──
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        grad_scaler = torch.amp.GradScaler(enabled=use_amp)

        steps_per_epoch = (n_train + batch_size - 1) // batch_size
        round_total_steps = max_epochs_per_round * steps_per_epoch
        warmup_steps = steps_per_epoch  # 1 epoch warmup (gentle for warm-start)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(round_total_steps - warmup_steps, 1)),
        ], milestones=[warmup_steps])

        # ── Inner training loop ──
        round_best_val_acc = 0.0
        wait = 0
        min_epochs = 3

        print(f"\n[{ts()}] Training round {round_idx+1} "
              f"(max {max_epochs_per_round} epochs, patience={inner_patience})...\n")

        for epoch in range(max_epochs_per_round):
            model.train()
            perm = np.random.permutation(n_train)
            epoch_loss, n_batches = 0.0, 0

            batch_iter = tqdm(
                range(0, n_train, batch_size),
                desc=f"  R{round_idx+1} Ep{epoch:2d}",
                total=steps_per_epoch,
                ncols=100,
                leave=False,
                file=sys.stderr,
                mininterval=10,
            )
            for start in batch_iter:
                idx = perm[start:start + batch_size]
                xp = torch.from_numpy(
                    train_patches[idx]).to(device, non_blocking=True)
                xi = torch.from_numpy(
                    train_indices[idx]).to(device, non_blocking=True)
                yb = torch.from_numpy(
                    train_y[idx].astype(np.int64)).to(device, non_blocking=True)

                if xp.size(0) < 2:
                    continue

                optimizer.zero_grad(set_to_none=True)
                amp_dev = "cuda" if use_amp else "cpu"
                with torch.amp.autocast(amp_dev, enabled=use_amp,
                                        dtype=torch.float16):
                    logits = model(xp, xi)
                    loss = nn.functional.nll_loss(logits, yb)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            # ── Validation ──
            val_acc, val_loss = validate(
                model, val_patches_gpu, val_indices_gpu, val_y_gpu,
                n_val, use_amp)
            total_epochs_trained += 1

            # Track improvement
            improved_round = val_acc > round_best_val_acc
            improved_global = val_acc > global_best_val_acc

            if improved_round:
                round_best_val_acc = val_acc
                # Save round-best model (for warm-starting next round)
                torch.save(model.state_dict(), round_model_path)
                wait = 0

            if improved_global:
                global_best_val_acc = val_acc
                # Save global-best model + scalers (for final inference)
                os.makedirs(OUT_DIR, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                with open(os.path.join(OUT_DIR, "ssnet_scaler.pkl"), "wb") as f:
                    pickle.dump({"patches": patch_scaler,
                                 "indices": idx_scaler}, f)
                # Save full state for resume
                torch.save({
                    "model": model.state_dict(),
                    "round_idx": round_idx,
                    "global_best_val_acc": global_best_val_acc,
                }, ckpt_path)

            if not improved_round:
                wait += 1

            # Print epoch summary
            avg = epoch_loss / max(n_batches, 1)
            marker = ""
            if improved_global:
                marker = " ** GLOBAL"
            elif improved_round:
                marker = " *"
            print(f"  R{round_idx+1} Ep{epoch:2d}: "
                  f"loss={avg:.5f} val_loss={val_loss:.5f} "
                  f"acc={val_acc:.4f} (best_round={round_best_val_acc:.4f} "
                  f"best_global={global_best_val_acc:.4f}) "
                  f"wait={wait}{marker}")

            if epoch >= min_epochs and wait >= inner_patience:
                print(f"  Early stop round {round_idx+1} at epoch {epoch}")
                break

        # ── End of round: load round-best weights for next round ──
        if os.path.exists(round_model_path):
            model.load_state_dict(torch.load(
                round_model_path, map_location=device, weights_only=True))

        # ── Free round data ──
        del train_patches, train_indices, train_y
        del val_patches_gpu, val_indices_gpu
        del optimizer, grad_scaler, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\n  Round {round_idx+1} done: "
              f"round_best={round_best_val_acc:.4f}, "
              f"global_best={global_best_val_acc:.4f}, "
              f"total_epochs={total_epochs_trained}")

    # ═══════════════════════════════════════════════════════
    #  Final evaluation
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  All {n_rounds} rounds complete.")
    print(f"  Global best val accuracy: {global_best_val_acc:.4f}")
    print(f"  Total epochs trained: {total_epochs_trained}")
    print(f"{'='*70}")

    # Load global best model + its scalers for test eval
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(
            best_model_path, map_location=device, weights_only=True))
    model.eval()

    with open(os.path.join(OUT_DIR, "ssnet_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    patch_scaler = sc["patches"]
    idx_scaler = sc["indices"]

    # Final val metrics with best model + scalers
    val_patches_gpu, val_indices_gpu = scale_val_to_gpu(
        patch_scaler, idx_scaler, device)
    val_acc, val_loss = validate(
        model, val_patches_gpu, val_indices_gpu, val_y_gpu, n_val, use_amp)

    all_preds = []
    with torch.no_grad():
        for vs in range(0, n_val, 32768):
            ve = min(vs + 32768, n_val)
            pb = model.predict(val_patches_gpu[vs:ve], val_indices_gpu[vs:ve])
            all_preds.append(pb.cpu().numpy())
    preds = np.concatenate(all_preds)
    val_y_np = val_y_gpu.cpu().numpy()

    result = {
        "model": "ssnet_v4_iterative",
        "n_params": n_params,
        "n_rounds": n_rounds,
        "total_epochs": total_epochs_trained,
        "val_accuracy": float(val_acc),
        "global_best_val_acc": float(global_best_val_acc),
    }
    print(f"\n  Val accuracy (best model): {val_acc:.4f} ({val_acc*100:.2f}%)")
    for ci in range(N_CLASSES):
        mask = val_y_np == ci
        if mask.sum() > 0:
            acc = float((preds[mask].argmax(1) == ci).mean())
            result[f"acc_{CLASS_NAMES[ci]}"] = acc
            print(f"    {CLASS_NAMES[ci]:>15}: {acc:.4f}")

    # ── Test evaluation ──
    print(f"\n[{ts()}] Evaluating on test cities...")
    del val_patches_gpu, val_indices_gpu, val_y_np
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
    parser = argparse.ArgumentParser(
        description="Train SpectralSpatialNet V4 with iterative resampling")
    parser.add_argument("--n-rounds", type=int, default=10,
                        help="Number of data resampling rounds")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Max epochs per round")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--inner-patience", type=int, default=4,
                        help="Early stop patience within each round")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    result = train(device, args.n_rounds, args.max_epochs, args.batch_size,
                   args.lr, args.weight_decay, args.inner_patience,
                   resume=args.resume)

    if result:
        print(f"\n{'='*70}")
        print(f"  DONE: SpectralSpatialNet V4")
        print(f"  Val: {result['val_accuracy']:.4f}")
        if "test_accuracy" in result:
            print(f"  Test: {result['test_accuracy']:.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
