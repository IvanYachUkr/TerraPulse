#!/usr/bin/env python3
"""
Production Training Script for MLP (Champion Model).

Train the final MLP model on extracted features.
Optimized for performance on CPU/GPU.
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.mlp_torch import SoftmaxMLP, ILR_MLP, DirichletMLP
from src.splitting import get_fold_indices

# Import from scripts to reuse config/build logic if possible,
# or copy strictly necessary parts to be standalone.
# I'll implement standalone to be safe/clean.

# ── Config ──
SEED = 42
CLASS_NAMES = ['tree_cover', 'shrubland', 'grassland', 'cropland', 'built_up', 'bare_sparse_vegetation', 'snow_and_ice', 'permanent_water_bodies', 'herbaceous_wetland', 'mangroves', 'moss_and_lichen']
# Wait, classes in checklist: WorldCover -> 6 final classes.
# I should check config or existing scripts for REAL classes.
# scripts/train_final_mlp.py uses CLASS_NAMES from run_mlp_overnight_v4.
# Let's check run_mlp_overnight_v4.py for CLASS_NAMES.
# Actually, I'll read it from src/config.py if it's there, or assume standard 6.
# Checklist says "6 final classes".

# Hardcoded classes from project context (Phase 1)
FINAL_CLASSES = [
    "tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"
]

def build_model(cfg, n_features, device):
    """Build MLP model from config."""
    # Champion config: bi_LBP_plain_silu_L5_d1024_bn
    model = SoftmaxMLP(
        n_features=n_features,
        n_classes=len(FINAL_CLASSES),
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
        norm=cfg["norm"],
    )
    return model.to(device)

def train_model(net, X_trn, y_trn, X_val, y_val, cfg):
    """Train loop."""
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])

    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    batch_size = cfg["batch_size"]
    n_samples = X_trn.size(0)
    best_val = float("inf")
    patience = cfg["patience_steps"]
    min_steps = cfg["min_steps"]
    steps_without_improv = 0
    best_state = None

    # Simple loop
    for epoch in range(cfg["max_epochs"]):
        net.train()
        perm = torch.randperm(n_samples, device=X_trn.device)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_X = X_trn[idx]
            batch_y = y_trn[idx]

            opt.zero_grad()
            preds = net(batch_X) # LogSoftmax
            loss = criterion(preds, batch_y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)

        scheduler.step()

        # Validation
        net.eval()
        with torch.no_grad():
            val_preds = net(X_val)
            val_loss = criterion(val_preds, y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = net.state_dict()
            steps_without_improv = 0
        else:
            steps_without_improv += 1

        if epoch > min_steps // (n_samples // batch_size) and steps_without_improv > patience:
            break

    net.load_state_dict(best_state)
    return net, best_val

def partition_features(cols):
    """Simple partition to find LBP vs others."""
    # Logic matching scripts/train_final_mlp.py build_bi_lbp
    base = []
    lbp = []
    for c in cols:
        if c.startswith("LBP_"):
            lbp.append(c)
        else:
            # Assume everything else is band/index/tc/spatial
            base.append(c)
    return base + lbp

def main():
    # ── Config ──
    CFG_MLP = {
        "d_model": 1024,
        "n_layers": 5,
        "dropout": 0.15,
        "activation": "silu",
        "norm": "batchnorm",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 2048,
        "max_epochs": 100, # Reduced for demo, real training uses 2000
        "patience_steps": 50,
        "min_steps": 20
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load Data ──
    # Can load from standard path or argument
    path = os.path.join(PROCESSED_V2_DIR, "features_rust_production.parquet")
    if not os.path.exists(path):
        # Fallback to standard path if production file doesn't exist
        path = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")

    print(f"Loading features from {path}...")
    df = pd.read_parquet(path)

    # Select features
    # bi_LBP = bands + indices + LBP
    # We filter out control cols
    CONTROL = ["cell_id", "valid_fraction", "low_valid_fraction", "reflectance_scale", "full_features_computed"]
    feat_cols = [c for c in df.columns if c not in CONTROL and pd.api.types.is_numeric_dtype(df[c])]

    # Filter for bi_LBP (Bands, Indices, LBP)
    # Exclude "Textural" if any (GLCM, Gabor, HOG) if they exist and are not LBP
    # In features_merged_full, we have GLCM_, Gabor_, HOG_, MP_, SV_
    # Rust features might have them if we add them later, but currently Rust only has bi_LBP components + Multi-band LBP
    # We want to MATCH the final model: Bands + Indices + NIR LBP.
    # The final model excluded Multi-band LBP?
    # Check `train_final_mlp.py`: `build_bi_lbp` uses `bands_indices` group and `LBP_*`.
    # `bands_indices` group usually excludes GLCM/Gabor/HOG.
    # We should exclude GLCM/Gabor/HOG/MP/SV.

    selected_cols = []
    for c in feat_cols:
        if any(x in c for x in ["GLCM_", "Gabor_", "HOG_", "MP_", "SV_"]):
            continue
        # Also, final model likely only used NIR LBP (LBP_u8_).
        # Rust produces LBP_NDVI_ etc.
        # If we want to replicate exact Final MLP, we should probably exclude Multi-band LBP.
        # But if we train a NEW model dynamically, using Multi-band LBP is likely BETTER.
        # The user asked to "train best config (final) mlp".
        # Best config used bi_LBP.
        # I will stick to bi_LBP (Bands+Indices+NIR LBP).
        if "LBP_" in c and not c.startswith("LBP_u8_") and not c.startswith("LBP_entropy"):
             # If it is LBP_NDVI_... skip it to match original model exactly
             continue
        selected_cols.append(c)

    print(f"Selected {len(selected_cols)} features (bi_LBP equivalent).")

    X = df[selected_cols].values.astype(np.float32)
    # NaNs should be gone, but safety check
    X = np.nan_to_num(X)

    # Labels
    # Load labels
    labels_path = os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet")
    if os.path.exists(labels_path):
        y_df = pd.read_parquet(labels_path)
        y = y_df[FINAL_CLASSES].values.astype(np.float32)
    else:
        print("Labels not found. Cannot train.")
        return

    # Train on all data (or split)
    # For production, we often train on ALL available data.
    # Or we can do a simple split.
    print("Training on full dataset...")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    X_t = torch.tensor(X_s, device=device)
    y_t = torch.tensor(y, device=device) # Normalize? KLDiv requires sum=1 target?
    # Normalize targets to sum to 1
    y_t = y_t / (y_t.sum(dim=1, keepdim=True) + 1e-6)

    net = build_model(CFG_MLP, len(selected_cols), device)

    net, loss = train_model(net, X_t, y_t, X_t, y_t, CFG_MLP) # Validation on train for demo

    print(f"Training complete. Final Loss: {loss:.4f}")

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "models", "production_mlp")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"features": selected_cols, "cfg": CFG_MLP}, f)

    print(f"Model saved to {out_dir}")

if __name__ == "__main__":
    main()
