"""
Training loop for the hybrid masked fusion MLP with split-head architecture.

Main ideas:
  - Phase 1: pretrain the SAR mask head on presence / compatibility detection
  - Phase 2: joint training of optical + SAR + joint interaction branch + fusion head
  - Split-head: common classes (5) use softmax CE, rare classes (2: shrub, bare) use L1
  - Refiner and rare-class emphasis are introduced gradually, not from epoch 0
  - The SAR mask is supervised with masked BCE on the ORIGINAL batch (never mixup)
  - Validation is stationary, using recombined 7-class predictions
  - Faster cosine LR scheduler for quicker convergence

The SAR gate uses detached mask outputs inside the forward pass, so classification
losses do not directly rewrite the semantics of the mask head.
"""

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SEED
from .train import compute_objective


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def soft_cross_entropy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy for fractional distribution targets."""
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def distribution_l1_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between predicted and target class fractions."""
    probs = torch.softmax(logits, dim=-1)
    return torch.abs(probs - target).sum(dim=-1).mean()


def weighted_soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Class-weighted soft CE, normalized row-wise to keep scale sane."""
    weighted_target = target * class_weights.unsqueeze(0)
    weighted_target = weighted_target / weighted_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    logp = F.log_softmax(logits, dim=-1)
    return -(weighted_target * logp).sum(dim=-1).mean()


def build_presence_targets(
    soft_targets: torch.Tensor,
    neg_threshold: float = 0.005,
    pos_threshold: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create binary presence labels with an ignored uncertainty band."""
    positive = (soft_targets >= pos_threshold).float()
    valid = ((soft_targets <= neg_threshold) | (soft_targets >= pos_threshold)).float()
    return positive, valid


def masked_presence_bce_loss(
    mask_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    neg_threshold: float = 0.005,
    pos_threshold: float = 0.02,
    pos_weight: torch.Tensor = None,
) -> torch.Tensor:
    """Masked BCE loss for the SAR compatibility head."""
    binary_targets, valid_mask = build_presence_targets(
        soft_targets,
        neg_threshold=neg_threshold,
        pos_threshold=pos_threshold,
    )
    loss = F.binary_cross_entropy_with_logits(
        mask_logits,
        binary_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = loss * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return loss.sum() / denom


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def apply_mixup(xb: torch.Tensor, yb: torch.Tensor, alpha: float):
    """Standard mixup for the MAIN distribution task only."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1.0 - lam) * xb[perm], lam * yb + (1.0 - lam) * yb[perm]


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def build_optimizer(net: nn.Module, config: dict, device: str):
    """AdamW with mildly slower SAR learning rates after mask pretraining."""
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0068)
    sar_backbone_lr_scale = config.get("sar_backbone_lr_scale", 0.55)
    mask_head_lr_scale = config.get("mask_head_lr_scale", 0.45)
    freeze_sar_backbone = config.get("freeze_sar_backbone", False)
    freeze_mask_head = config.get("freeze_mask_head", False)

    param_groups = []

    def add_group(params, group_lr):
        params = [p for p in params if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": group_lr, "weight_decay": weight_decay})

    add_group(net.optical_branch.parameters(), lr)
    add_group(net.joint_branch.parameters(), lr)
    add_group(net.fusion_backbone.parameters(), lr)
    add_group(net.common_head.parameters(), lr)
    add_group(net.rare_head.parameters(), lr)
    add_group(net.refiner_backbone.parameters(), lr)
    add_group(net.refiner_common_head.parameters(), lr)
    add_group(net.refiner_rare_head.parameters(), lr)

    if freeze_sar_backbone:
        for p in net.sar_branch.backbone.parameters():
            p.requires_grad = False
    if freeze_mask_head:
        for p in net.sar_branch.mask_head.parameters():
            p.requires_grad = False

    add_group(net.sar_branch.backbone.parameters(), lr * sar_backbone_lr_scale)
    add_group(net.sar_branch.mask_head.parameters(), lr * mask_head_lr_scale)

    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            fused=(device == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups)
    return optimizer


# ---------------------------------------------------------------------------
# Phase 1: SAR pretraining
# ---------------------------------------------------------------------------

def pretrain_sar(
    net: nn.Module,
    X_train: np.ndarray,
    y_train_norm: np.ndarray,
    val_tensors: Dict[str, dict],
    config: dict,
    device: str,
) -> dict:
    """Pretrain the SAR branch on compatibility / presence prediction."""
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n_epochs = config.get("sar_pretrain_epochs", 12)
    lr = config.get("lr", 1e-3) * 1.5
    batch_size = config.get("batch_size", 2048)
    neg_threshold = config.get("presence_neg_threshold", 0.005)
    pos_threshold = config.get("presence_pos_threshold", 0.02)

    mask_pos_weight = config.get("mask_pos_weight", None)
    if mask_pos_weight is not None:
        mask_pos_weight = torch.tensor(mask_pos_weight, dtype=torch.float32, device=device)

    sar_params = list(net.sar_branch.parameters())
    optimizer = torch.optim.AdamW(sar_params, lr=lr, weight_decay=1e-3)
    use_amp = device == "cuda"
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)

    n_samples = len(y_train_norm)
    print(
        f"    SAR pretrain: {n_epochs} epochs, lr={lr:.5f}, "
        f"{sum(p.numel() for p in sar_params):,} params"
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        net.sar_branch.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(np.array(X_train[idx])).to(device, non_blocking=True)
            yb = torch.from_numpy(y_train_norm[idx]).to(device, non_blocking=True)

            x_sar = xb[:, net.sar_idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=use_amp, dtype=torch.float16):
                _, mask_logits, _ = net.sar_branch(x_sar)
                loss = masked_presence_bce_loss(
                    mask_logits,
                    yb,
                    neg_threshold=neg_threshold,
                    pos_threshold=pos_threshold,
                    pos_weight=mask_pos_weight,
                )

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sar_params, 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        net.sar_branch.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            with torch.amp.autocast(device, enabled=use_amp, dtype=torch.float16):
                for data in val_tensors.values():
                    x_sar = data["X"][:, net.sar_idx]
                    _, mask_logits, _ = net.sar_branch(x_sar)
                    batch_loss = masked_presence_bce_loss(
                        mask_logits,
                        data["y_norm"],
                        neg_threshold=neg_threshold,
                        pos_threshold=pos_threshold,
                        pos_weight=mask_pos_weight,
                    ).item()
                    n = x_sar.shape[0]
                    val_loss += batch_loss * n
                    val_n += n
        val_loss /= max(val_n, 1)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.sar_branch.state_dict().items()}

        marker = " *" if improved else ""
        print(f"      SAR Ep {epoch:2d}: trn={avg_train:.4f} val={val_loss:.4f}{marker}")

    if best_state is not None:
        net.sar_branch.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"    SAR pretrain done in {elapsed:.0f}s (best val={best_val:.4f})")
    return {
        "sar_pretrain_val": float(best_val),
        "sar_pretrain_time": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Phase 2: Main training
# ---------------------------------------------------------------------------

def train_fused(
    net: nn.Module,
    X_train: np.ndarray,
    y_train_norm: np.ndarray,
    val_tensors: Dict[str, dict],
    config: dict,
    device: str,
    output_dir: str = None,
) -> dict:
    """Train the hybrid masked fusion model."""
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 2048)
    max_epochs = config.get("max_epochs", 300)
    mixup_alpha = config.get("mixup_alpha", 0.34)
    mixup_prob = config.get("mixup_prob", 0.56)
    mixup_off_last_epochs = config.get("mixup_off_last_epochs", 20)
    label_threshold = config.get("label_threshold", 0.015)

    dist_l1_weight = config.get("dist_l1_weight", 0.18)
    main_weight = config.get("main_weight", 1.0)
    base_weight = config.get("base_weight", 0.35)
    optical_aux_weight = config.get("optical_aux_weight", 0.18)
    rare_weight = config.get("rare_weight", 0.22)
    optical_rare_weight = config.get("optical_rare_weight", 0.08)
    mask_loss_weight = config.get("mask_loss_weight", 0.18)

    gate_warmup = config.get("gate_warmup_epochs", 10)
    refiner_start_epoch = config.get("refiner_start_epoch", 8)
    refiner_warmup = config.get("refiner_warmup_epochs", 12)

    patience = config.get("patience", 35)
    min_epochs = config.get("min_epochs", 20)

    neg_threshold = config.get("presence_neg_threshold", 0.005)
    pos_threshold = config.get("presence_pos_threshold", 0.02)

    rare_class_weights = config.get("rare_class_weights", None)
    if rare_class_weights is None:
        rare_class_weights = np.ones(y_train_norm.shape[1], dtype=np.float32)
    rare_class_weights = torch.tensor(rare_class_weights, dtype=torch.float32, device=device)

    mask_pos_weight = config.get("mask_pos_weight", None)
    if mask_pos_weight is not None:
        mask_pos_weight = torch.tensor(mask_pos_weight, dtype=torch.float32, device=device)

    # Main-task denoising of tiny fractions only for the main distribution objective.
    # The mask task keeps the original normalized labels.
    if label_threshold > 0:
        y_main = np.array(y_train_norm, copy=True)
        y_main[y_main < label_threshold] = 0.0
        row_sums = y_main.sum(axis=1, keepdims=True)
        valid_idx = np.where(row_sums.ravel() > 0)[0]
        y_main = y_main[valid_idx]
        y_main = y_main / np.maximum(y_main.sum(axis=1, keepdims=True), 1e-8)
        use_valid_idx = True
    else:
        y_main = y_train_norm
        valid_idx = None
        use_valid_idx = False

    n_samples = len(y_main)
    optimizer = build_optimizer(net, config, device)
    trainable_params = [p for p in net.parameters() if p.requires_grad]

    print(
        f"    Main training: lr={lr:.5f}, batch={batch_size}, epochs={max_epochs}, "
        f"trainable_params={sum(p.numel() for p in trainable_params):,}"
    )

    use_amp = device == "cuda"
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = max((n_samples + batch_size - 1) // batch_size, 1)
    # Faster LR schedule: cosine over ~100 epochs worth of steps instead of max_epochs
    cosine_epochs = config.get("cosine_epochs", 100)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = min(steps_per_epoch * 3, max(total_steps - 1, 1))
    cosine_steps = max(cosine_epochs * steps_per_epoch - warmup_steps, 1)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.05,
                total_iters=max(warmup_steps, 1),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
            ),
        ],
        milestones=[max(warmup_steps, 1)],
    )

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())

    best_score = -float("inf")
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    def get_gate_scale(epoch: int) -> float:
        if gate_warmup <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(gate_warmup))

    def get_refine_scale(epoch: int) -> float:
        if epoch < refiner_start_epoch:
            return 0.0
        if refiner_warmup <= 0:
            return 1.0
        return min(1.0, float(epoch - refiner_start_epoch + 1) / float(refiner_warmup))

    rare_l1_weight = config.get("rare_l1_weight", 0.30)

    for epoch in range(max_epochs):
        gate_scale = get_gate_scale(epoch)
        refine_scale = get_refine_scale(epoch)
        enable_mixup = epoch < (max_epochs - mixup_off_last_epochs)

        net.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            x_idx = valid_idx[idx] if use_valid_idx else idx

            xb_orig = torch.from_numpy(np.array(X_train[x_idx])).to(device, non_blocking=True)
            yb_main_orig = torch.from_numpy(y_main[idx]).to(device, non_blocking=True)
            yb_mask_orig = torch.from_numpy(y_train_norm[x_idx]).to(device, non_blocking=True)

            xb = xb_orig
            yb_main = yb_main_orig
            if enable_mixup and mixup_alpha > 0 and np.random.rand() < mixup_prob:
                xb, yb_main = apply_mixup(xb, yb_main, mixup_alpha)

            if has_bn and xb.size(0) < 2:
                continue

            # Split targets into common (re-normalized) and rare (raw fractions)
            yb_rare = yb_main[:, net.rare_idx]           # [B, 2] raw fractions
            yb_common_raw = yb_main[:, net.common_idx]   # [B, 5]
            yb_common = yb_common_raw / yb_common_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=use_amp, dtype=torch.float16):
                out = net(
                    xb,
                    gate_scale=gate_scale,
                    refine_scale=refine_scale,
                    detach_mask_inputs=True,
                    apply_branch_dropout=True,
                )

                # Common class losses (softmax CE + L1)
                loss_common = (
                    soft_cross_entropy_from_logits(out["final_common_logits"], yb_common)
                    + dist_l1_weight * distribution_l1_from_logits(out["final_common_logits"], yb_common)
                )

                loss_base_common = (
                    soft_cross_entropy_from_logits(out["common_logits"], yb_common)
                    + dist_l1_weight * distribution_l1_from_logits(out["common_logits"], yb_common)
                )

                # Rare class loss (L1 on sigmoid predictions vs raw fractions)
                rare_preds = torch.sigmoid(out["final_rare_logits"])
                loss_rare = torch.abs(rare_preds - yb_rare).mean()

                rare_base_preds = torch.sigmoid(out["rare_logits"])
                loss_rare_base = torch.abs(rare_base_preds - yb_rare).mean()

                # Full 7-class L1 using recombined predictions
                full_preds = net._recombine(out)
                loss_full_l1 = torch.abs(full_preds - yb_main).sum(dim=-1).mean()

                # Optical aux loss (still 7-class softmax for the optical branch)
                loss_opt = soft_cross_entropy_from_logits(out["optical_logits"], yb_main)

                # Mask loss always uses the original, unmixed targets and a no-dropout forward.
                out_nomix = net(
                    xb_orig,
                    gate_scale=gate_scale,
                    refine_scale=refine_scale,
                    detach_mask_inputs=True,
                    apply_branch_dropout=False,
                )
                loss_mask = masked_presence_bce_loss(
                    out_nomix["mask_logits"],
                    yb_mask_orig,
                    neg_threshold=neg_threshold,
                    pos_threshold=pos_threshold,
                    pos_weight=mask_pos_weight,
                )

                loss = (
                    main_weight * loss_common
                    + base_weight * loss_base_common
                    + rare_l1_weight * (loss_rare + 0.5 * loss_rare_base)
                    + dist_l1_weight * 0.5 * loss_full_l1
                    + optical_aux_weight * loss_opt
                    + mask_loss_weight * loss_mask
                )

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_mask_loss += loss_mask.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_mask = epoch_mask_loss / max(n_batches, 1)

        # Stationary validation using recombined 7-class predictions.
        net.eval()
        val_full_l1 = 0.0
        val_rare_l1 = 0.0
        val_opt_ce = 0.0
        val_mask = 0.0
        val_n = 0

        with torch.no_grad():
            with torch.amp.autocast(device, enabled=use_amp, dtype=torch.float16):
                for data in val_tensors.values():
                    out = net(
                        data["X"],
                        gate_scale=1.0,
                        refine_scale=1.0,
                        detach_mask_inputs=True,
                        apply_branch_dropout=False,
                    )
                    yv = data["y_norm"]
                    n = data["X"].shape[0]

                    # Recombined full 7-class L1
                    full_preds = net._recombine(out)
                    val_full_l1 += torch.abs(full_preds - yv).sum(dim=-1).mean().item() * n

                    # Rare class L1 specifically
                    rare_preds = torch.sigmoid(out["final_rare_logits"])
                    val_rare_l1 += torch.abs(rare_preds - yv[:, net.rare_idx]).mean().item() * n

                    val_opt_ce += soft_cross_entropy_from_logits(out["optical_logits"], yv).item() * n
                    val_mask += masked_presence_bce_loss(
                        out["mask_logits"],
                        yv,
                        neg_threshold=neg_threshold,
                        pos_threshold=pos_threshold,
                        pos_weight=mask_pos_weight,
                    ).item() * n
                    val_n += n

        val_full_l1 /= max(val_n, 1)
        val_rare_l1 /= max(val_n, 1)
        val_opt_ce /= max(val_n, 1)
        val_mask /= max(val_n, 1)
        val_loss = val_full_l1

        combined, top1_acc, mean_r2 = compute_objective(
            net, val_tensors, max(label_threshold, 0.01), device
        )

        improved = (
            combined > best_score + 1e-8
            or (
                abs(combined - best_score) <= 1e-8
                and val_loss < best_val_loss - 1e-8
            )
        )

        if improved:
            best_score = float(combined)
            best_val_loss = float(val_loss)
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
            if output_dir is not None:
                import os
                torch.save(best_state, os.path.join(output_dir, "best_model.pt"))
        else:
            wait += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        marker = " *" if improved else ""
        print(
            f"    Ep {epoch:3d}: trn={avg_train:.4f} mask_trn={avg_mask:.4f} "
            f"val[l1={val_full_l1:.4f} rare={val_rare_l1:.4f} "
            f"opt={val_opt_ce:.4f} mask={val_mask:.4f}] "
            f"score[comb={combined:.4f} top1={top1_acc:.4f} r2={mean_r2:.4f}] "
            f"gate={gate_scale:.2f} ref={refine_scale:.2f} mixup={'on' if enable_mixup else 'off'} "
            f"lr={cur_lr:.6f} w={wait}{marker}"
        )

        if epoch >= min_epochs and wait >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    elapsed = time.time() - t0
    combined, top1_acc, mean_r2 = compute_objective(
        net, val_tensors, max(label_threshold, 0.01), device
    )

    # Recompute stationary validation loss on best checkpoint.
    final_l1 = 0.0
    final_rare_l1 = 0.0
    final_mask = 0.0
    val_n = 0

    with torch.no_grad():
        with torch.amp.autocast(device, enabled=use_amp, dtype=torch.float16):
            for data in val_tensors.values():
                out = net(
                    data["X"],
                    gate_scale=1.0,
                    refine_scale=1.0,
                    detach_mask_inputs=True,
                    apply_branch_dropout=False,
                )
                yv = data["y_norm"]
                n = data["X"].shape[0]
                full_preds = net._recombine(out)
                final_l1 += torch.abs(full_preds - yv).sum(dim=-1).mean().item() * n
                rare_preds = torch.sigmoid(out["final_rare_logits"])
                final_rare_l1 += torch.abs(rare_preds - yv[:, net.rare_idx]).mean().item() * n
                final_mask += masked_presence_bce_loss(
                    out["mask_logits"],
                    yv,
                    neg_threshold=neg_threshold,
                    pos_threshold=pos_threshold,
                    pos_weight=mask_pos_weight,
                ).item() * n
                val_n += n

    final_l1 /= max(val_n, 1)
    final_rare_l1 /= max(val_n, 1)
    final_mask /= max(val_n, 1)

    results = {
        "combined": float(combined),
        "top1_acc": float(top1_acc),
        "mean_r2": float(mean_r2),
        "val_loss": float(final_l1),
        "val_l1": float(final_l1),
        "val_rare_l1": float(final_rare_l1),
        "val_mask": float(final_mask),
        "n_params": sum(p.numel() for p in net.parameters()),
        "epochs": epoch + 1,
        "time_s": round(elapsed, 1),
        "best_combined": float(best_score),
        "best_val_loss": float(best_val_loss),
    }

    print(
        f"\n  >> Result: combined={combined:.4f} top1={top1_acc:.4f} "
        f"R2={mean_r2:.4f} val_l1={final_l1:.5f} rare_l1={final_rare_l1:.5f} {elapsed:.0f}s"
    )
    return results
