#!/usr/bin/env python3
"""
Train SpectralSpatialNet V5 with iterative resampling and robust supervision.

Main changes versus V4:
  1. Fixed scalers fitted once on a representative train subset.
  2. Center-aware V5 architecture with auxiliary branch heads.
  3. EMA teacher + confidence-aware bootstrapped soft targets.
  4. Loss capping for suspicious / likely noisy samples.
  5. Branch dropout to stop the handcrafted index branch from dominating.
  6. D4 spatial augmentation (rotations / flips) on 3x3 patches.
  7. Model selection by harmonic mean of overall accuracy and balanced accuracy.

This is designed as a "last serious try" model for noisy EO labels:
it should improve robustness without the pathological overprediction that often
appears with naive class weighting.
"""

import argparse
import copy
import gc
import json
import math
import os
import pickle
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from reproduce.models.architectures.spectral_spatial_v5 import SpectralSpatialNetV5

OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")
VAL_CACHE_DIR = os.path.join(OUT_DIR, "val_raw_cache_v5")
FIXED_SCALER_PATH = os.path.join(OUT_DIR, "ssnet_v5_fixed_scaler.pkl")
BEST_MODEL_PATH = os.path.join(OUT_DIR, "ssnet_v5.pt")
ROUND_MODEL_PATH = os.path.join(OUT_DIR, "ssnet_v5_round.pt")
CKPT_PATH = os.path.join(OUT_DIR, "ssnet_v5_training.pt")
METRICS_PATH = os.path.join(OUT_DIR, "ssnet_v5_metrics.json")

TRAIN_PX = 60_000
VAL_PX = 5_000
PAD = 1
N_INDICES = 145
SCALE_CHUNK = 200_000
SCALER_SAMPLE_PX = 20_000  # per city, only used once to fit fixed scalers


def ts():
    return time.strftime("%H:%M:%S")


def compute_indices_chunked(X_1x1, chunk_size=500_000):
    N = X_1x1.shape[0]
    result = np.empty((N, N_INDICES), dtype=np.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        result[start:end] = compute_center_indices(X_1x1[start:end])
    return result


def load_split(cities, max_px, label="split", use_fp16=False, round_seed=None):
    all_patches, all_y = [], []

    for i, city in enumerate(cities):
        if not city_has_raw_tifs(city):
            continue
        if round_seed is not None:
            city_seed = round_seed + i
        else:
            city_seed = SEED + abs(hash(city.name)) % 10000
        rng = np.random.RandomState(city_seed)
        result = extract_pixels_for_city(city, max_pixels=max_px, pad=PAD, rng=rng)
        if result is None:
            print(f"  [{i+1}/{len(cities)}] {city.name:25s} - SKIP")
            continue

        store_dtype = np.float16 if use_fp16 else np.float32
        all_patches.append(result["feat_3x3"].astype(store_dtype))
        all_y.append(result["labels"])
        print(f"  [{i+1}/{len(cities)}] {city.name:25s} - {result['n_pixels']:>7,} px")
        del result
        gc.collect()

    if not all_patches:
        return None, None, None

    patches = np.concatenate(all_patches)
    y = np.concatenate(all_y).astype(np.int32)
    del all_patches, all_y
    gc.collect()

    center_1x1 = patches[:, 4 * 72:5 * 72].astype(np.float32)
    print(f"  Computing indices ({patches.shape[0]:,} pixels)...")
    indices = compute_indices_chunked(center_1x1)
    if use_fp16:
        indices = indices.astype(np.float16)
    del center_1x1
    gc.collect()

    mem = (patches.nbytes + indices.nbytes) / 1e9
    dt = 'fp16' if use_fp16 else 'fp32'
    print(f"  {label}: {patches.shape[0]:,} px [{dt}], patches {patches.nbytes/1e9:.2f} GB + "
          f"indices {indices.nbytes/1e9:.2f} GB = {mem:.2f} GB total")
    return patches, indices, y


def print_class_dist(label, y):
    cls, cnt = np.unique(y, return_counts=True)
    total = cnt.sum()
    dist = " ".join(f"{CLASS_NAMES[c][:4]}={100*n/total:.1f}%" for c, n in zip(cls, cnt))
    print(f"  {label}: {dist}")


def load_or_fit_fixed_scalers(train_cities):
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FIXED_SCALER_PATH):
        print(f"\n[{ts()}] Loading fixed scalers from {FIXED_SCALER_PATH}...")
        with open(FIXED_SCALER_PATH, "rb") as f:
            sc = pickle.load(f)
        return sc["patches"], sc["indices"]

    print(f"\n[{ts()}] Fitting fixed scalers once on representative train subset...")
    patches, indices, y = load_split(
        train_cities, SCALER_SAMPLE_PX, "Scaler sample", use_fp16=False,
        round_seed=SEED + 123_457,
    )
    if patches is None:
        raise RuntimeError("No data available to fit fixed scalers.")

    patch_scaler = StandardScaler()
    patch_scaler.fit(patches)

    idx_scaler = StandardScaler()
    idx_scaler.fit(indices)

    with open(FIXED_SCALER_PATH, "wb") as f:
        pickle.dump({"patches": patch_scaler, "indices": idx_scaler}, f)

    print_class_dist("Scaler sample", y)
    print(f"  Fixed scalers saved to {FIXED_SCALER_PATH}")
    del patches, indices, y
    gc.collect()
    return patch_scaler, idx_scaler


def apply_scaler_inplace(arr, scaler, target_dtype=np.float16):
    n = arr.shape[0]
    for s in range(0, n, SCALE_CHUNK):
        e = min(s + SCALE_CHUNK, n)
        arr[s:e] = scaler.transform(arr[s:e].astype(np.float32)).astype(target_dtype)


def scale_val_to_gpu(patch_scaler, idx_scaler, device):
    val_p = np.load(os.path.join(VAL_CACHE_DIR, "patches.npy"))
    val_i = np.load(os.path.join(VAL_CACHE_DIR, "indices.npy"))
    val_patches_gpu = torch.from_numpy(patch_scaler.transform(val_p).astype(np.float32)).to(device)
    val_indices_gpu = torch.from_numpy(idx_scaler.transform(val_i).astype(np.float32)).to(device)
    del val_p, val_i
    gc.collect()
    return val_patches_gpu, val_indices_gpu


def ensure_val_cache(val_cities):
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)
    files = ["patches.npy", "indices.npy", "y.npy"]
    val_cache_exists = all(os.path.exists(os.path.join(VAL_CACHE_DIR, f)) for f in files)

    if val_cache_exists:
        print(f"\n[{ts()}] Loading cached val labels...")
        val_y = np.load(os.path.join(VAL_CACHE_DIR, "y.npy"))
        return val_y

    print(f"\n[{ts()}] Loading validation data (one-time)...")
    val_patches, val_indices, val_y = load_split(val_cities, VAL_PX, "Val", use_fp16=False)
    if val_patches is None:
        raise RuntimeError("No validation data.")
    np.save(os.path.join(VAL_CACHE_DIR, "patches.npy"), val_patches)
    np.save(os.path.join(VAL_CACHE_DIR, "indices.npy"), val_indices)
    np.save(os.path.join(VAL_CACHE_DIR, "y.npy"), val_y)
    del val_patches, val_indices
    gc.collect()
    print(f"  Val raw cached to {VAL_CACHE_DIR}")
    return val_y


def random_d4_augment_flat(patches, n_timesteps=6, n_bands=12, p=0.75):
    if torch.rand((), device=patches.device).item() >= p:
        return patches

    n = patches.shape[0]
    x = patches.reshape(n, 9, n_timesteps, n_bands)
    x = x.permute(0, 2, 3, 1).contiguous().reshape(n, n_timesteps, n_bands, 3, 3)

    k = int(torch.randint(0, 4, (1,), device=patches.device).item())
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if torch.rand((), device=patches.device).item() < 0.5:
        x = torch.flip(x, dims=(-1,))

    x = x.reshape(n, n_timesteps, n_bands, 9)
    x = x.permute(0, 3, 1, 2).contiguous().reshape(n, -1)
    return x


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


class RobustBootstrapCriterion:
    def __init__(self,
                 n_classes,
                 label_smoothing=0.02,
                 bootstrap_start_epoch=2,
                 bootstrap_alpha_max=0.20,
                 teacher_conf_thresh=0.65,
                 teacher_label_prob_thresh=0.50,
                 loss_cap=2.5,
                 spatial_aux_weight=0.20,
                 index_aux_weight=0.05):
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.bootstrap_start_epoch = bootstrap_start_epoch
        self.bootstrap_alpha_max = bootstrap_alpha_max
        self.teacher_conf_thresh = teacher_conf_thresh
        self.teacher_label_prob_thresh = teacher_label_prob_thresh
        self.loss_cap = loss_cap
        self.spatial_aux_weight = spatial_aux_weight
        self.index_aux_weight = index_aux_weight

    def _smoothed_targets(self, y):
        eps = self.label_smoothing
        off = eps / max(self.n_classes - 1, 1)
        target = torch.full((y.size(0), self.n_classes), off, device=y.device, dtype=torch.float32)
        target.scatter_(1, y.unsqueeze(1), 1.0 - eps)
        return target

    def _soft_ce(self, logits, soft_target):
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_target * log_probs).sum(dim=-1)

    def _teacher_mix(self, base_target, y, teacher_probs, epoch):
        if teacher_probs is None or epoch < self.bootstrap_start_epoch:
            alpha = torch.zeros(y.size(0), device=y.device, dtype=torch.float32)
            return base_target, alpha

        conf, _ = teacher_probs.max(dim=1)
        y_prob = teacher_probs.gather(1, y.unsqueeze(1)).squeeze(1)

        conf_strength = ((conf - self.teacher_conf_thresh) / (1.0 - self.teacher_conf_thresh)).clamp(0.0, 1.0)
        suspect = ((self.teacher_label_prob_thresh - y_prob) / self.teacher_label_prob_thresh).clamp(0.0, 1.0)
        alpha = self.bootstrap_alpha_max * conf_strength * suspect
        soft_target = (1.0 - alpha.unsqueeze(1)) * base_target + alpha.unsqueeze(1) * teacher_probs
        return soft_target, alpha

    def __call__(self, outputs, y, teacher_probs=None, epoch=0):
        base_target = self._smoothed_targets(y)
        soft_target, alpha = self._teacher_mix(base_target, y, teacher_probs, epoch)

        main_loss = self._soft_ce(outputs["logits"], soft_target)
        spatial_loss = self._soft_ce(outputs["spatial_logits"], soft_target)
        index_loss = self._soft_ce(outputs["index_logits"], soft_target)

        if epoch >= self.bootstrap_start_epoch:
            main_loss = torch.clamp(main_loss, max=self.loss_cap)
            spatial_loss = torch.clamp(spatial_loss, max=self.loss_cap)
            index_loss = torch.clamp(index_loss, max=self.loss_cap)

        total = (
            main_loss.mean()
            + self.spatial_aux_weight * spatial_loss.mean()
            + self.index_aux_weight * index_loss.mean()
        )
        stats = {
            "main": float(main_loss.mean().detach().cpu()),
            "spatial_aux": float(spatial_loss.mean().detach().cpu()),
            "index_aux": float(index_loss.mean().detach().cpu()),
            "boot_alpha": float(alpha.mean().detach().cpu()),
        }
        return total, stats


def confusion_from_preds(pred, target, n_classes):
    idx = target * n_classes + pred
    cm = torch.bincount(idx, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes).cpu().numpy()


def metrics_from_confusion(conf):
    total = conf.sum()
    tp = np.diag(conf).astype(np.float64)
    row_sum = conf.sum(axis=1).astype(np.float64)
    col_sum = conf.sum(axis=0).astype(np.float64)

    overall = float(tp.sum() / max(total, 1))
    valid = row_sum > 0
    recall = np.zeros_like(tp)
    precision = np.zeros_like(tp)
    f1 = np.zeros_like(tp)

    recall[valid] = tp[valid] / row_sum[valid]
    precision[col_sum > 0] = tp[col_sum > 0] / col_sum[col_sum > 0]
    denom = precision + recall
    good = denom > 0
    f1[good] = 2.0 * precision[good] * recall[good] / denom[good]

    balanced = float(recall[valid].mean()) if valid.any() else 0.0
    macro_f1 = float(f1[valid].mean()) if valid.any() else 0.0
    score = float((2.0 * overall * balanced) / max(overall + balanced, 1e-12))

    return {
        "overall_acc": overall,
        "balanced_acc": balanced,
        "macro_f1": macro_f1,
        "score": score,
        "per_class_recall": recall.tolist(),
    }


def validate(model, val_patches_gpu, val_indices_gpu, val_y_gpu, n_val, n_classes, use_amp):
    model.eval()
    loss_sum = 0.0
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    VB = 32768

    with torch.no_grad():
        amp_dev = "cuda" if use_amp else "cpu"
        for vs in range(0, n_val, VB):
            ve = min(vs + VB, n_val)
            with torch.amp.autocast(amp_dev, enabled=use_amp, dtype=torch.float16):
                out = model(val_patches_gpu[vs:ve], val_indices_gpu[vs:ve])
                logits = out["logits"]
                loss_sum += F.cross_entropy(logits, val_y_gpu[vs:ve], reduction='sum').item()
            pred = logits.argmax(dim=1)
            conf += confusion_from_preds(pred, val_y_gpu[vs:ve], n_classes)

    metrics = metrics_from_confusion(conf)
    metrics["loss"] = loss_sum / n_val
    return metrics


def train(device, n_rounds, max_epochs_per_round, batch_size, lr,
          weight_decay, inner_patience, resume=False):
    train_cities = get_train_cities()
    val_cities = get_val_cities()

    print(f"\n{'='*78}")
    print(f"  SpectralSpatialNet V5 — Robust Iterative Resampling")
    print(f"  Rounds: {n_rounds}, Max epochs/round: {max_epochs_per_round}")
    print(f"  Train: {TRAIN_PX:,} px/city, Val: {VAL_PX:,} px/city")
    print(f"  Inner patience: {inner_patience}, Batch: {batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*78}")

    val_y = ensure_val_cache(val_cities)
    n_val = len(val_y)
    val_y_gpu = torch.from_numpy(val_y.astype(np.int64)).to(device)
    print_class_dist("Val", val_y)
    del val_y
    gc.collect()

    patch_scaler, idx_scaler = load_or_fit_fixed_scalers(train_cities)
    val_patches_gpu, val_indices_gpu = scale_val_to_gpu(patch_scaler, idx_scaler, device)
    print(f"\n  Val on GPU: {tuple(val_patches_gpu.shape)}, {tuple(val_indices_gpu.shape)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    model = SpectralSpatialNetV5(
        n_bands=12,
        n_timesteps=6,
        n_indices=N_INDICES,
        spatial_dims=(32, 64, 128),
        expand_ratio=4,
        temporal_dim=128,
        n_attn_layers=2,
        n_heads=4,
        n_classes=N_CLASSES,
        dropout=0.15,
        spatial_branch_drop=0.10,
        index_branch_drop=0.25,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()

    use_amp = device == "cuda"
    n_params = model.n_params()
    print(f"\n  Model: {n_params:,} parameters")

    criterion = RobustBootstrapCriterion(
        n_classes=N_CLASSES,
        label_smoothing=0.02,
        bootstrap_start_epoch=2,
        bootstrap_alpha_max=0.20,
        teacher_conf_thresh=0.65,
        teacher_label_prob_thresh=0.50,
        loss_cap=2.5,
        spatial_aux_weight=0.20,
        index_aux_weight=0.05,
    )

    start_round = 0
    best_score = 0.0
    best_val_acc = 0.0
    best_bal_acc = 0.0

    if resume and os.path.exists(CKPT_PATH):
        print(f"  Resuming from {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        start_round = ckpt.get("round_idx", 0) + 1
        best_score = ckpt.get("best_score", 0.0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_bal_acc = ckpt.get("best_bal_acc", 0.0)
        print(f"  Resume round={start_round}, best_score={best_score:.4f}, "
              f"best_acc={best_val_acc:.4f}, best_bal={best_bal_acc:.4f}")
        del ckpt
        gc.collect()
    elif resume and os.path.exists(BEST_MODEL_PATH):
        print(f"  Loading best EMA weights from {BEST_MODEL_PATH}")
        best_state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        ema_model.load_state_dict(best_state)

    total_epochs_trained = 0

    for round_idx in range(start_round, n_rounds):
        print(f"\n{'='*78}")
        print(f"  ROUND {round_idx+1}/{n_rounds}")
        print(f"{'='*78}")

        round_seed = SEED + round_idx * 7919
        print(f"\n[{ts()}] Loading training data (round seed={round_seed})...")
        train_patches, train_indices, train_y = load_split(
            train_cities, TRAIN_PX, f"Train R{round_idx+1}", use_fp16=True,
            round_seed=round_seed,
        )
        if train_patches is None:
            print("ERROR: No train data!")
            continue

        n_train = len(train_y)
        print_class_dist(f"Train R{round_idx+1}", train_y)

        print(f"\n[{ts()}] Applying fixed scalers...")
        apply_scaler_inplace(train_patches, patch_scaler, target_dtype=np.float16)
        apply_scaler_inplace(train_indices, idx_scaler, target_dtype=np.float16)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        grad_scaler = torch.amp.GradScaler(enabled=use_amp)

        steps_per_epoch = (n_train + batch_size - 1) // batch_size
        round_total_steps = max_epochs_per_round * steps_per_epoch
        warmup_steps = steps_per_epoch

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.10, total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(round_total_steps - warmup_steps, 1),
                ),
            ],
            milestones=[warmup_steps],
        )

        round_best_score = 0.0
        round_best_metrics = None
        wait = 0
        min_epochs = 4

        print(f"\n[{ts()}] Training round {round_idx+1} "
              f"(max {max_epochs_per_round} epochs, patience={inner_patience})...\n")

        for epoch in range(max_epochs_per_round):
            model.train()
            perm = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            running_main = 0.0
            running_spatial = 0.0
            running_index = 0.0
            running_alpha = 0.0

            batch_iter = tqdm(
                range(0, n_train, batch_size),
                desc=f"  R{round_idx+1} Ep{epoch:02d}",
                total=steps_per_epoch,
                ncols=110,
                leave=False,
                file=sys.stderr,
                mininterval=10,
            )

            for start in batch_iter:
                idx = perm[start:start + batch_size]
                xp = torch.from_numpy(train_patches[idx]).to(device, non_blocking=True)
                xi = torch.from_numpy(train_indices[idx]).to(device, non_blocking=True)
                yb = torch.from_numpy(train_y[idx].astype(np.int64)).to(device, non_blocking=True)
                if xp.size(0) < 2:
                    continue

                xp_teacher = xp.float()
                xp_student = random_d4_augment_flat(xp_teacher.clone(), n_timesteps=6, n_bands=12, p=0.75)
                xi_student = xi.float()

                with torch.no_grad():
                    teacher_out = ema_model(xp_teacher, xi_student)
                    teacher_probs = F.softmax(teacher_out["logits"], dim=-1)

                optimizer.zero_grad(set_to_none=True)
                amp_dev = "cuda" if use_amp else "cpu"
                with torch.amp.autocast(amp_dev, enabled=use_amp, dtype=torch.float16):
                    outputs = model(xp_student, xi_student)
                    loss, loss_stats = criterion(outputs, yb, teacher_probs=teacher_probs, epoch=epoch)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                update_ema(ema_model, model, decay=0.999)

                epoch_loss += loss.item()
                n_batches += 1
                running_main += loss_stats["main"]
                running_spatial += loss_stats["spatial_aux"]
                running_index += loss_stats["index_aux"]
                running_alpha += loss_stats["boot_alpha"]

            val_metrics = validate(
                ema_model, val_patches_gpu, val_indices_gpu, val_y_gpu,
                n_val, N_CLASSES, use_amp,
            )
            total_epochs_trained += 1

            improved_round = val_metrics["score"] > round_best_score
            improved_global = val_metrics["score"] > best_score

            if improved_round:
                round_best_score = val_metrics["score"]
                round_best_metrics = dict(val_metrics)
                torch.save(ema_model.state_dict(), ROUND_MODEL_PATH)
                wait = 0
            else:
                wait += 1

            if improved_global:
                best_score = val_metrics["score"]
                best_val_acc = val_metrics["overall_acc"]
                best_bal_acc = val_metrics["balanced_acc"]
                os.makedirs(OUT_DIR, exist_ok=True)
                torch.save(ema_model.state_dict(), BEST_MODEL_PATH)
                torch.save({
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "round_idx": round_idx,
                    "best_score": best_score,
                    "best_val_acc": best_val_acc,
                    "best_bal_acc": best_bal_acc,
                }, CKPT_PATH)

            avg = epoch_loss / max(n_batches, 1)
            marker = " ** GLOBAL" if improved_global else (" *" if improved_round else "")
            print(
                f"  R{round_idx+1} Ep{epoch:02d}: "
                f"loss={avg:.5f} "
                f"main={running_main/max(n_batches,1):.4f} "
                f"sp={running_spatial/max(n_batches,1):.4f} "
                f"idx={running_index/max(n_batches,1):.4f} "
                f"alpha={running_alpha/max(n_batches,1):.3f} | "
                f"val_loss={val_metrics['loss']:.5f} "
                f"acc={val_metrics['overall_acc']:.4f} "
                f"bal={val_metrics['balanced_acc']:.4f} "
                f"mf1={val_metrics['macro_f1']:.4f} "
                f"score={val_metrics['score']:.4f} "
                f"(best_round={round_best_score:.4f} best_global={best_score:.4f}) "
                f"wait={wait}{marker}"
            )

            if epoch >= min_epochs and wait >= inner_patience:
                print(f"  Early stop round {round_idx+1} at epoch {epoch}")
                break

        if os.path.exists(ROUND_MODEL_PATH):
            best_round_state = torch.load(ROUND_MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(best_round_state)
            ema_model.load_state_dict(best_round_state)

        del train_patches, train_indices, train_y
        del optimizer, grad_scaler, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        rb = round_best_metrics or {"overall_acc": 0.0, "balanced_acc": 0.0, "score": 0.0}
        print(f"\n  Round {round_idx+1} done: "
              f"acc={rb['overall_acc']:.4f}, bal={rb['balanced_acc']:.4f}, "
              f"score={rb['score']:.4f}, best_global={best_score:.4f}, "
              f"total_epochs={total_epochs_trained}")

    print(f"\n{'='*78}")
    print(f"  All {n_rounds} rounds complete.")
    print(f"  Best score: {best_score:.4f}")
    print(f"  Best val acc: {best_val_acc:.4f}")
    print(f"  Best balanced acc: {best_bal_acc:.4f}")
    print(f"  Total epochs trained: {total_epochs_trained}")
    print(f"{'='*78}")

    if os.path.exists(BEST_MODEL_PATH):
        best_state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        ema_model.load_state_dict(best_state)
    ema_model.eval()

    final_val = validate(ema_model, val_patches_gpu, val_indices_gpu, val_y_gpu, n_val, N_CLASSES, use_amp)
    print(f"\n  Final best-model val: acc={final_val['overall_acc']:.4f}, "
          f"bal={final_val['balanced_acc']:.4f}, mf1={final_val['macro_f1']:.4f}, "
          f"score={final_val['score']:.4f}")

    test_result = evaluate_test(ema_model, patch_scaler, idx_scaler, device)

    result = {
        "model": "ssnet_v5_robust",
        "n_params": n_params,
        "n_rounds": n_rounds,
        "total_epochs": total_epochs_trained,
        "val_accuracy": float(final_val["overall_acc"]),
        "val_balanced_accuracy": float(final_val["balanced_acc"]),
        "val_macro_f1": float(final_val["macro_f1"]),
        "val_score": float(final_val["score"]),
        "best_score": float(best_score),
        "best_val_acc": float(best_val_acc),
        "best_bal_acc": float(best_bal_acc),
    }
    if test_result:
        result["test_accuracy"] = test_result["overall_accuracy"]
        result["test_balanced_accuracy"] = test_result["balanced_accuracy"]
        result["test_macro_f1"] = test_result["macro_f1"]
        result["test_per_class"] = test_result["per_class"]
        result["test_per_city"] = test_result["per_city"]

    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def evaluate_test(model, patch_scaler, idx_scaler, device):
    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    if not test_cities:
        print("  No test cities")
        return None

    all_conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    per_city = {}
    model.eval()

    for city in test_cities:
        rng = np.random.RandomState(SEED)
        result = extract_pixels_for_city(city, max_pixels=500_000, pad=PAD, rng=rng)
        if result is None:
            print(f"  {city.name:25s} - SKIP")
            continue

        patches = result["feat_3x3"].astype(np.float32)
        y = result["labels"]
        n = result["n_pixels"]
        del result
        gc.collect()

        center = patches[:, 4 * 72:5 * 72].copy()
        indices = compute_indices_chunked(center)
        del center
        gc.collect()

        patches = patch_scaler.transform(patches).astype(np.float32)
        indices = idx_scaler.transform(indices).astype(np.float32)

        BATCH = 16384
        city_conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for s in range(0, n, BATCH):
                xp = torch.from_numpy(patches[s:s+BATCH]).to(device)
                xi = torch.from_numpy(indices[s:s+BATCH]).to(device)
                pred = model.predict(xp, xi).argmax(dim=1)
                target = torch.from_numpy(y[s:s+BATCH].astype(np.int64)).to(device)
                city_conf += confusion_from_preds(pred, target, N_CLASSES)

        city_metrics = metrics_from_confusion(city_conf)
        per_city[city.name] = {
            "accuracy": float(city_metrics["overall_acc"]),
            "balanced_accuracy": float(city_metrics["balanced_acc"]),
            "macro_f1": float(city_metrics["macro_f1"]),
            "score": float(city_metrics["score"]),
            "n_pixels": int(n),
        }
        all_conf += city_conf
        print(f"  {city.name:25s} - {n:>9,} px, "
              f"acc={city_metrics['overall_acc']:.4f}, "
              f"bal={city_metrics['balanced_acc']:.4f}, "
              f"score={city_metrics['score']:.4f}")
        del patches, indices, y
        gc.collect()

    metrics = metrics_from_confusion(all_conf)
    per_class = {}
    print(f"\n  Test Overall: acc={metrics['overall_acc']:.4f}, "
          f"bal={metrics['balanced_acc']:.4f}, mf1={metrics['macro_f1']:.4f}, "
          f"score={metrics['score']:.4f}")
    for ci in range(N_CLASSES):
        per_class[CLASS_NAMES[ci]] = float(metrics["per_class_recall"][ci])
        print(f"    {CLASS_NAMES[ci]:>15}: {metrics['per_class_recall'][ci]:.4f}")

    return {
        "overall_accuracy": float(metrics["overall_acc"]),
        "balanced_accuracy": float(metrics["balanced_acc"]),
        "macro_f1": float(metrics["macro_f1"]),
        "score": float(metrics["score"]),
        "per_class": per_class,
        "per_city": per_city,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SpectralSpatialNet V5")
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--inner-patience", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    result = train(
        device=device,
        n_rounds=args.n_rounds,
        max_epochs_per_round=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        inner_patience=args.inner_patience,
        resume=args.resume,
    )

    if result:
        print(f"\n{'='*78}")
        print("  DONE: SpectralSpatialNet V5")
        print(f"  Val acc:  {result['val_accuracy']:.4f}")
        print(f"  Val bal:  {result['val_balanced_accuracy']:.4f}")
        print(f"  Val mf1:  {result['val_macro_f1']:.4f}")
        print(f"  Val score:{result['val_score']:.4f}")
        if "test_accuracy" in result:
            print(f"  Test acc: {result['test_accuracy']:.4f}")
            print(f"  Test bal: {result['test_balanced_accuracy']:.4f}")
            print(f"  Test mf1: {result['test_macro_f1']:.4f}")
        print(f"{'='*78}")


if __name__ == "__main__":
    main()
