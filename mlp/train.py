"""
Training loop, loss functions, augmentation, and evaluation.
"""

import math
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from .config import N_CLASSES, SEED


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def normalize_targets(y: np.ndarray) -> np.ndarray:
    """Clip, smooth, and re-normalise label fractions to valid distribution.

    Adds a small epsilon so no class has exactly 0 probability
    (required for KL-divergence / soft cross-entropy stability).
    """
    y = np.clip(y, 0, None).astype(np.float32)
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    y = y / row_sums
    y = y + 1e-7
    y = y / y.sum(axis=1, keepdims=True)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def soft_cross_entropy(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy loss for fractional label targets."""
    return -(target * log_pred).sum(dim=-1).mean()


def distribution_l1(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between predicted and target distributions."""
    pred = log_pred.exp()
    return (pred - target).abs().sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def apply_mixup(xb, yb, alpha: float):
    """Apply mixup augmentation to a batch."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_objective(
    net: nn.Module,
    val_data: Dict[str, dict],
    label_threshold: float,
    device: str,
) -> Tuple[float, float, float]:
    """Combined objective on GPU-resident val tensors.

    Returns (combined, top1_accuracy, mean_r2).
    """
    all_top1_correct = 0
    all_top1_total = 0
    all_r2_values = []

    net.eval()
    for city_name, data in val_data.items():
        X_gpu = data["X"]
        y_raw = data["y_raw"]  # numpy on CPU

        with torch.no_grad():
            preds = net.predict(X_gpu).cpu().numpy()

        # Top-1 accuracy
        true_top1 = y_raw.argmax(axis=1)
        pred_top1 = preds.argmax(axis=1)
        all_top1_correct += (true_top1 == pred_top1).sum()
        all_top1_total += len(true_top1)

        # R² for classes above threshold
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
# Training loop
# ---------------------------------------------------------------------------

def train_one_config(
    net: nn.Module,
    X_train: np.ndarray,       # memmap float32
    y_train_norm: np.ndarray,  # normalised targets
    val_tensors: Dict[str, dict],
    config: dict,
    device: str,
    *,
    val_city_name: str = "berlin",
) -> dict:
    """Train one model configuration to completion.

    Config keys:
        lr, weight_decay, batch_size, max_epochs,
        mixup_alpha, mixup_prob, label_threshold

    Returns dict with training results.
    """
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Unpack config
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    batch_size = config.get("batch_size", 4096)
    max_epochs = config.get("max_epochs", 300)
    mixup_alpha = config.get("mixup_alpha", 0.3)
    mixup_prob = config.get("mixup_prob", 0.5)
    label_threshold = config.get("label_threshold", 0.0)

    # Label denoising
    if label_threshold > 0:
        y_work = np.array(y_train_norm)
        y_work[y_work < label_threshold] = 0.0
        row_sums = y_work.sum(axis=1, keepdims=True)
        valid_idx = np.where(row_sums.ravel() > 0)[0]
        y_work = y_work[valid_idx]
        y_work = y_work / np.maximum(y_work.sum(axis=1, keepdims=True), 1e-8)
        use_valid_idx = True
    else:
        y_work = y_train_norm
        valid_idx = None
        use_valid_idx = False

    n_samples = len(y_work)

    # Optimizer
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay,
            fused=(device == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay,
        )

    use_amp = device == "cuda"
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = (n_samples + batch_size - 1) // batch_size
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * 3

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)),
    ], milestones=[warmup_steps])

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())

    # Validation tensors (already on GPU)
    if val_city_name not in val_tensors:
        val_city_name = list(val_tensors.keys())[0]
    val_X_gpu = val_tensors[val_city_name]["X"]
    val_y_gpu = val_tensors[val_city_name]["y_norm"]

    dist_l1_weight = config.get("dist_l1_weight", 0.5)

    best_score = -float("inf")
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    patience = config.get("patience", 35)
    min_epochs = max(math.ceil(1500 / steps_per_epoch), 3)
    rng = np.random.RandomState(SEED)

    for epoch in range(max_epochs):
        net.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            x_idx = valid_idx[idx] if use_valid_idx else idx
            xb = torch.from_numpy(np.array(X_train[x_idx])).to(
                device, non_blocking=True)
            yb = torch.from_numpy(y_work[idx]).to(device, non_blocking=True)

            # Mixup
            if mixup_alpha > 0 and rng.random() < mixup_prob:
                xb, yb = apply_mixup(xb, yb, mixup_alpha)

            if has_bn and xb.size(0) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=use_amp, dtype=torch.float16):
                pred = net(xb)
                loss_ce = soft_cross_entropy(pred, yb)
                loss_l1 = distribution_l1(pred, yb)
                loss = loss_ce + dist_l1_weight * loss_l1

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validate using combined metric across ALL val cities
        net.eval()
        val_ce = 0.0
        val_l1 = 0.0
        val_n = 0
        with torch.no_grad():
            with torch.amp.autocast(device, enabled=use_amp, dtype=torch.float16):
                for data in val_tensors.values():
                    n = data["X"].shape[0]
                    pred_v = net(data["X"])
                    val_ce += soft_cross_entropy(pred_v, data["y_norm"]).item() * n
                    val_l1 += distribution_l1(pred_v, data["y_norm"]).item() * n
                    val_n += n
        val_ce /= max(val_n, 1)
        val_l1 /= max(val_n, 1)
        val_loss = val_ce + dist_l1_weight * val_l1

        combined, top1_acc, mean_r2 = compute_objective(
            net, val_tensors, max(label_threshold, 0.01), device
        )

        improved = (
            (combined > best_score + 1e-4)
            or (combined >= best_score - 1e-4 and val_loss < best_val_loss - 1e-8)
        )
        if improved:
            best_score = float(combined)
            best_val_loss = float(val_loss)
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        marker = " *" if improved else ""
        if epoch <= 3 or epoch % 10 == 0 or improved:
            print(f"    Ep {epoch:3d}: trn={avg_train:.4f} "
                  f"val[ce={val_ce:.4f} l1={val_l1:.4f}] "
                  f"score[comb={combined:.4f} top1={top1_acc:.4f} r2={mean_r2:.4f}] "
                  f"lr={cur_lr:.6f} w={wait}{marker}")

        if epoch >= min_epochs and wait >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    # Restore best
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    elapsed = time.time() - t0

    # Full objective
    combined, top1_acc, mean_r2 = compute_objective(
        net, val_tensors,
        max(label_threshold, 0.01),
        device,
    )

    results = {
        "combined": float(combined),
        "top1_acc": float(top1_acc),
        "mean_r2": float(mean_r2),
        "val_loss": float(best_val_loss),
        "n_params": sum(p.numel() for p in net.parameters()),
        "epochs": epoch + 1,
        "time_s": round(elapsed, 1),
        "best_combined": float(best_score),
    }
    print(f"\n  >> Result: combined={combined:.4f} "
          f"top1={top1_acc:.4f} R2={mean_r2:.4f} "
          f"val_loss={best_val_loss:.5f} {elapsed:.0f}s")

    return results
