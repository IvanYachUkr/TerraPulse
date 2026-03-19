#!/usr/bin/env python3
"""
Hybrid masked fusion MLP training script.

This version keeps the large all-feature interaction branch while still making
SAR act like a real compatibility prior instead of a decorative side quest.

Stages:
    Phase 1: pretrain SAR mask head on class presence / compatibility
    Phase 2: main training of optical + SAR + joint branch + fusion + refiner

Usage:
    .venv\Scripts\python -u mlp/run_fused.py
    .venv\Scripts\python -u mlp/run_fused.py --output-dir data/cities/models_hybrid_masked_fusion_v1
    .venv\Scripts\python -u mlp/run_fused.py --joint-widths 1280 640 320 --fusion-widths 640 320
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from mlp.config import (  # noqa: E402
    ALL_TRAIN, ALL_TEST, VAL_CITY_NAMES, EXCLUDED_CITY_NAMES,
    CITIES_DIR, N_CLASSES, CLASS_NAMES,
)
from mlp.features import select_features, get_common_columns, city_has_sar  # noqa: E402
from mlp.fused_model import make_fused_model, split_feature_indices  # noqa: E402
from mlp.data import (  # noqa: E402
    build_memmap_dataset, fit_scaler, apply_scaler_inplace, load_val_to_gpu,
)
from mlp.train import normalize_targets  # noqa: E402
from mlp.fused_train import pretrain_sar, train_fused  # noqa: E402


def _ts():
    return time.strftime("%H:%M:%S")


def compute_rare_class_weights(
    y_train_norm: np.ndarray,
    power: float = 0.50,
    max_weight: float = 3.0,
):
    """Inverse-frequency-like weights for rare-class refinement."""
    class_mass = y_train_norm.mean(axis=0).astype(np.float32)
    class_mass = np.clip(class_mass, 1e-6, None)

    weights = (class_mass.mean() / class_mass) ** power
    weights = np.clip(weights, 1.0, max_weight)
    weights = weights / weights.mean()
    return weights.astype(np.float32), class_mass.astype(np.float32)


def compute_mask_pos_weight(
    y_train_norm: np.ndarray,
    neg_threshold: float = 0.005,
    pos_threshold: float = 0.02,
    max_pos_weight: float = 6.0,
):
    """Class-wise positive weights for the SAR mask BCE."""
    pos = (y_train_norm >= pos_threshold).sum(axis=0).astype(np.float32)
    neg = (y_train_norm <= neg_threshold).sum(axis=0).astype(np.float32)

    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, max_pos_weight)
    return pos_weight.astype(np.float32), pos.astype(np.float32), neg.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Masked Fusion MLP Training")

    # Architecture
    parser.add_argument("--optical-widths", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--sar-widths", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--joint-widths", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument("--fusion-widths", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--refiner-widths", type=int, nargs="+", default=[256, 128])

    # Regularization
    parser.add_argument("--dropout", type=float, default=0.072)
    parser.add_argument("--input-dropout", type=float, default=0.06)
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=["silu", "gelu", "relu", "mish"])
    parser.add_argument("--optical-branch-drop", type=float, default=0.04)
    parser.add_argument("--sar-branch-drop", type=float, default=0.04)
    parser.add_argument("--joint-branch-drop", type=float, default=0.10)

    # Training
    parser.add_argument("--lr", type=float, default=0.0010)
    parser.add_argument("--weight-decay", type=float, default=0.0068)
    parser.add_argument("--sar-backbone-lr-scale", type=float, default=0.55)
    parser.add_argument("--mask-head-lr-scale", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--mixup-alpha", type=float, default=0.34)
    parser.add_argument("--mixup-prob", type=float, default=0.56)
    parser.add_argument("--mixup-off-last-epochs", type=int, default=20)
    parser.add_argument("--label-threshold", type=float, default=0.015)

    # Loss weights
    parser.add_argument("--dist-l1-weight", type=float, default=0.18)
    parser.add_argument("--main-weight", type=float, default=1.0)
    parser.add_argument("--base-weight", type=float, default=0.35)
    parser.add_argument("--optical-aux-weight", type=float, default=0.18)
    parser.add_argument("--rare-weight", type=float, default=0.22)
    parser.add_argument("--optical-rare-weight", type=float, default=0.08)
    parser.add_argument("--mask-loss-weight", type=float, default=0.18)

    # SAR mask / gating
    parser.add_argument("--sar-pretrain-epochs", type=int, default=12)
    parser.add_argument("--presence-neg-threshold", type=float, default=0.005)
    parser.add_argument("--presence-pos-threshold", type=float, default=0.02)
    parser.add_argument("--mask-max-pos-weight", type=float, default=6.0)
    parser.add_argument("--gate-strength", type=float, default=0.80)
    parser.add_argument("--gate-floor", type=float, default=0.10)
    parser.add_argument("--gate-warmup", type=int, default=10)

    # Refiner / rare-class focus
    parser.add_argument("--refine-strength", type=float, default=0.60)
    parser.add_argument("--rare-power", type=float, default=0.50)
    parser.add_argument("--rare-max-weight", type=float, default=3.0)
    parser.add_argument("--refiner-start-epoch", type=int, default=8)
    parser.add_argument("--refiner-warmup", type=int, default=12)

    # Optional freezing if you want to test ablations instead of peace of mind
    parser.add_argument("--freeze-sar-backbone", action="store_true")
    parser.add_argument("--freeze-mask-head", action="store_true")

    # Early stopping
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-epochs", type=int, default=20)

    # Output
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(CITIES_DIR, "models_hybrid_masked_fusion_mlp")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sep = "=" * 76
    print(f"\n{sep}")
    print("  Hybrid Masked Fusion MLP -- optical specialist + SAR mask + joint branch")
    print(f"{sep}")
    print(f"  Device:            {device}")
    print(f"  Optical:           {args.optical_widths}")
    print(f"  SAR:               {args.sar_widths} (compatibility mask)")
    print(f"  Joint:             {args.joint_widths}")
    print(f"  Fusion:            {args.fusion_widths}")
    print(f"  Refiner:           {args.refiner_widths}")
    print(f"  Branch dropout:    opt={args.optical_branch_drop} sar={args.sar_branch_drop} joint={args.joint_branch_drop}")
    print(f"  Gate:              strength={args.gate_strength} floor={args.gate_floor} warmup={args.gate_warmup}")
    print(f"  Refiner start:     epoch {args.refiner_start_epoch} + {args.refiner_warmup} warmup")
    print(f"  SAR pretrain:      {args.sar_pretrain_epochs} epochs")
    print(f"  Freeze SAR trunk:  {args.freeze_sar_backbone}")
    print(f"  Freeze mask head:  {args.freeze_mask_head}")
    print(f"  Output:            {output_dir}")
    print(f"{sep}\n")

    # ---- Filter cities ----
    print(f"[{_ts()}] Filtering cities with SAR features...")
    all_sar = [c for c in ALL_TRAIN + ALL_TEST if city_has_sar(c)]
    train_cities = [c for c in all_sar
                    if c.name not in VAL_CITY_NAMES
                    and c.name not in EXCLUDED_CITY_NAMES]
    val_cities = [c for c in all_sar if c.name in VAL_CITY_NAMES]
    print(f"  Train: {len(train_cities)} cities")
    print(f"  Val:   {len(val_cities)} cities")

    # ---- Feature columns ----
    print(f"\n[{_ts()}] Building feature column intersection...")
    all_columns = get_common_columns(train_cities + val_cities)
    mlp_cols = select_features(all_columns)
    n_features = len(mlp_cols)

    idx = split_feature_indices(mlp_cols)
    n_optical = len(idx["optical"])
    n_sar = len(idx["sar"])
    print(f"  Total features: {n_features}")
    print(f"    Optical: {n_optical}")
    print(f"    SAR:     {n_sar}")

    # ---- Build training data ----
    mmap_dir = os.path.join(output_dir, "_mmap_cache")
    print(f"\n[{_ts()}] Loading training data (memmap)...")
    X_path, y_path, n_samples = build_memmap_dataset(train_cities, mlp_cols, mmap_dir)

    X_train = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(n_samples, n_features))
    y_train = np.memmap(y_path, dtype=np.float32, mode="r+", shape=(n_samples, N_CLASSES))

    # ---- Scaler ----
    print(f"\n[{_ts()}] Fitting scaler...")
    scaler = fit_scaler(X_train)
    apply_scaler_inplace(X_train, scaler)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)

    # ---- Val data ----
    print(f"\n[{_ts()}] Loading validation data to VRAM...")
    val_tensors = load_val_to_gpu(val_cities, mlp_cols, scaler, device)
    gc.collect()

    y_train_norm = normalize_targets(np.array(y_train))

    # ---- Class statistics ----
    rare_class_weights, class_mass = compute_rare_class_weights(
        y_train_norm,
        power=args.rare_power,
        max_weight=args.rare_max_weight,
    )
    mask_pos_weight, pos_counts, neg_counts = compute_mask_pos_weight(
        y_train_norm,
        neg_threshold=args.presence_neg_threshold,
        pos_threshold=args.presence_pos_threshold,
        max_pos_weight=args.mask_max_pos_weight,
    )

    print(f"\n[{_ts()}] Class statistics:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:>14s} | mean_mass={class_mass[i]:.5f} "
            f"rare_w={rare_class_weights[i]:.3f} "
            f"mask_pos_w={mask_pos_weight[i]:.3f} "
            f"pos={int(pos_counts[i])} neg={int(neg_counts[i])}"
        )

    # ---- Build model ----
    print(f"\n[{_ts()}] Building hybrid masked fusion model...")
    net = make_fused_model(
        n_features,
        N_CLASSES,
        mlp_cols,
        optical_widths=args.optical_widths,
        sar_widths=args.sar_widths,
        joint_widths=args.joint_widths,
        fusion_widths=args.fusion_widths,
        refiner_widths=args.refiner_widths,
        dropout=args.dropout,
        activation=args.activation,
        input_dropout=args.input_dropout,
        optical_branch_drop=args.optical_branch_drop,
        sar_branch_drop=args.sar_branch_drop,
        joint_branch_drop=args.joint_branch_drop,
        rare_class_scale=rare_class_weights.tolist(),
        gate_strength=args.gate_strength,
        gate_floor=args.gate_floor,
        refine_strength=args.refine_strength,
    ).to(device)

    n_params = sum(p.numel() for p in net.parameters())
    fusion_in = args.optical_widths[-1] + args.sar_widths[-1] + args.joint_widths[-1] + N_CLASSES
    fusion_hidden = args.fusion_widths[-1] if args.fusion_widths else fusion_in
    refiner_in = fusion_hidden + args.joint_widths[-1] + N_CLASSES + N_CLASSES + N_CLASSES
    print(f"  Optical branch:   {n_optical} -> {' -> '.join(map(str, args.optical_widths))} -> aux({N_CLASSES})")
    print(f"  SAR branch:       {n_sar} -> {' -> '.join(map(str, args.sar_widths))} -> mask({N_CLASSES})")
    print(f"  Joint branch:     {n_features} -> {' -> '.join(map(str, args.joint_widths))}")
    print(f"  Fusion head:      {fusion_in} -> {' -> '.join(map(str, args.fusion_widths))} -> base({N_CLASSES})")
    print(f"  Refiner head:     {refiner_in} -> {' -> '.join(map(str, args.refiner_widths))} -> residual({N_CLASSES})")
    print(f"  Total parameters: {n_params:,}")

    # ---- Phase 1: SAR pretraining ----
    if args.sar_pretrain_epochs > 0:
        print(f"\n[{_ts()}] Phase 1: Pretrain SAR mask branch...")
        sar_results = pretrain_sar(
            net,
            X_train,
            y_train_norm,
            val_tensors,
            {
                "sar_pretrain_epochs": args.sar_pretrain_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "presence_neg_threshold": args.presence_neg_threshold,
                "presence_pos_threshold": args.presence_pos_threshold,
                "mask_pos_weight": mask_pos_weight.tolist(),
            },
            device,
        )
    else:
        sar_results = {}

    # ---- Phase 2: Main training ----
    print(f"\n[{_ts()}] Phase 2: Main training...")
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "sar_backbone_lr_scale": args.sar_backbone_lr_scale,
        "mask_head_lr_scale": args.mask_head_lr_scale,
        "freeze_sar_backbone": args.freeze_sar_backbone,
        "freeze_mask_head": args.freeze_mask_head,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "mixup_alpha": args.mixup_alpha,
        "mixup_prob": args.mixup_prob,
        "mixup_off_last_epochs": args.mixup_off_last_epochs,
        "label_threshold": args.label_threshold,
        "dist_l1_weight": args.dist_l1_weight,
        "main_weight": args.main_weight,
        "base_weight": args.base_weight,
        "optical_aux_weight": args.optical_aux_weight,
        "rare_weight": args.rare_weight,
        "optical_rare_weight": args.optical_rare_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "gate_warmup_epochs": args.gate_warmup,
        "refiner_start_epoch": args.refiner_start_epoch,
        "refiner_warmup_epochs": args.refiner_warmup,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "presence_neg_threshold": args.presence_neg_threshold,
        "presence_pos_threshold": args.presence_pos_threshold,
        "rare_class_weights": rare_class_weights.tolist(),
        "mask_pos_weight": mask_pos_weight.tolist(),
    }
    results = train_fused(
        net,
        X_train,
        y_train_norm,
        val_tensors,
        config,
        device,
        output_dir=output_dir,
    )

    # ---- Save ----
    model_path = os.path.join(output_dir, "best_model.pt")
    torch.save(net.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    results_out = {
        "optical_widths": args.optical_widths,
        "sar_widths": args.sar_widths,
        "joint_widths": args.joint_widths,
        "fusion_widths": args.fusion_widths,
        "refiner_widths": args.refiner_widths,
        "config": config,
        "n_features": n_features,
        "n_optical": n_optical,
        "n_sar": n_sar,
        "class_mass": class_mass.tolist(),
        "rare_class_weights": rare_class_weights.tolist(),
        "mask_pos_weight": mask_pos_weight.tolist(),
        **sar_results,
        **results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)

    if args.export:
        from mlp.export import export_all
        export_all(net, scaler, mlp_cols, n_features)

    print(f"\n{sep}")
    print(f"  DONE -- {results['time_s']:.0f}s | {n_params:,} params")
    print(f"  combined={results['combined']:.4f}  top1={results['top1_acc']:.4f}  R2={results['mean_r2']:.4f}")
    print(f"{sep}")


if __name__ == "__main__":
    main()
