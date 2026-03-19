#!/usr/bin/env python3
"""
MLP Training — clean entry point.

Usage:
    # Single training run with default architecture
    .venv\\Scripts\\python -u mlp/run.py

    # Custom architecture
    .venv\\Scripts\\python -u mlp/run.py --widths 2048 1024 512 --lr 5e-4

    # Quick dry run (2 epochs)
    .venv\\Scripts\\python -u mlp/run.py --max-epochs 2

    # Train and export ONNX
    .venv\\Scripts\\python -u mlp/run.py --export

    # Custom output directory
    .venv\\Scripts\\python -u mlp/run.py --output-dir data/cities/models_v11
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

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from mlp.config import (
    ALL_TRAIN, ALL_TEST, VAL_CITY_NAMES, EXCLUDED_CITY_NAMES,
    CITIES_DIR, N_CLASSES, SEED, CLASS_NAMES,
)
from mlp.features import select_features, get_common_columns, city_has_sar
from mlp.model import make_model
from mlp.data import (
    build_memmap_dataset, fit_scaler, apply_scaler_inplace, load_val_to_gpu,
)
from mlp.train import normalize_targets, train_one_config


def _ts():
    return time.strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="MLP Training")
    parser.add_argument("--widths", type=int, nargs="+",
                        default=[1024, 512, 256, 64],
                        help="Hidden layer widths (default: 1024 512 256 64)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="silu",
                        choices=["silu", "gelu", "relu", "mish"])
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--label-threshold", type=float, default=0.0)
    parser.add_argument("--export", action="store_true",
                        help="Export ONNX model after training")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/cities/models_mlp)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(CITIES_DIR, "models_mlp")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sep = "=" * 70
    print(f"\n{sep}")
    print("  MLP Training -- Clean Module")
    print(f"{sep}")
    print(f"  Device:     {device}")
    print(f"  Widths:     {args.widths}")
    print(f"  LR:         {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Output:     {output_dir}")
    print(f"{sep}\n")

    # ---- Filter cities (SAR required) ----
    print(f"[{_ts()}] Filtering cities with SAR features...")
    all_sar = [c for c in ALL_TRAIN + ALL_TEST if city_has_sar(c)]
    train_cities = [c for c in all_sar
                    if c.name not in VAL_CITY_NAMES
                    and c.name not in EXCLUDED_CITY_NAMES]
    val_cities = [c for c in all_sar if c.name in VAL_CITY_NAMES]
    print(f"  Train: {len(train_cities)} cities")
    print(f"  Val:   {len(val_cities)} cities ({[c.name for c in val_cities]})")

    # ---- Feature columns ----
    print(f"\n[{_ts()}] Building feature column intersection...")
    all_columns = get_common_columns(train_cities + val_cities)
    mlp_cols = select_features(all_columns)
    n_features = len(mlp_cols)
    print(f"  MLP features: {n_features}")

    # Show feature breakdown
    n_quality = sum(1 for c in mlp_cols if c in ("valid_fraction", "low_valid_fraction"))
    n_spatial = sum(1 for c in mlp_cols if c.split("_")[0] in ("edge", "lap", "morans"))
    n_sar = sum(1 for c in mlp_cols if c.startswith("SAR_"))
    n_pheno = sum(1 for c in mlp_cols if "_pheno_" in c)
    print(f"    Quality columns: {n_quality}")
    print(f"    Spatial texture: {n_spatial}")
    print(f"    SAR features:    {n_sar}")
    print(f"    Phenological:    {n_pheno}")

    # ---- Build training data (memmap) ----
    mmap_dir = os.path.join(output_dir, "_mmap_cache")
    print(f"\n[{_ts()}] Loading training data (memmap)...")
    X_path, y_path, n_samples = build_memmap_dataset(
        train_cities, mlp_cols, mmap_dir,
    )

    X_train = np.memmap(X_path, dtype=np.float32, mode="r+",
                        shape=(n_samples, n_features))
    y_train = np.memmap(y_path, dtype=np.float32, mode="r+",
                        shape=(n_samples, N_CLASSES))

    # ---- Fit scaler ----
    print(f"\n[{_ts()}] Fitting scaler...")
    scaler = fit_scaler(X_train)
    apply_scaler_inplace(X_train, scaler)

    # Save scaler and columns
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)

    # ---- Load val data to GPU ----
    print(f"\n[{_ts()}] Loading validation data to VRAM...")
    val_tensors = load_val_to_gpu(val_cities, mlp_cols, scaler, device)
    gc.collect()

    # ---- Normalise training targets ----
    y_train_norm = normalize_targets(np.array(y_train))

    # ---- Build model ----
    print(f"\n[{_ts()}] Building model...")
    net = make_model(
        n_features, N_CLASSES, args.widths,
        dropout=args.dropout, activation=args.activation,
        input_dropout=args.input_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Architecture: {n_features} -> {' -> '.join(map(str, args.widths))} -> {N_CLASSES}")
    print(f"  Parameters: {n_params:,}")

    # ---- Train ----
    print(f"\n[{_ts()}] Training...")
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "mixup_alpha": args.mixup_alpha,
        "mixup_prob": args.mixup_prob,
        "label_threshold": args.label_threshold,
    }
    results = train_one_config(
        net, X_train, y_train_norm, val_tensors, config, device,
    )

    # Save model
    model_path = os.path.join(output_dir, "best_model.pt")
    torch.save(net.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    # Save results
    results_out = {
        "widths": args.widths,
        "config": config,
        "n_features": n_features,
        **results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)

    # ---- Export ONNX ----
    if args.export:
        from mlp.export import export_all
        export_all(net, scaler, mlp_cols, n_features)

    print(f"\n{sep}")
    print(f"  DONE -- {results['time_s']:.0f}s")
    print(f"  combined={results['combined']:.4f}  "
          f"top1={results['top1_acc']:.4f}  R2={results['mean_r2']:.4f}")
    print(f"{sep}")


if __name__ == "__main__":
    main()
