#!/usr/bin/env python3
"""
Generate full-resolution pixel-wise prediction maps for Nuremberg.

Produces 3 PNG images:
  1. nuremberg_v3.png  - class map from V3 (SpectralSpatialNetV2, expand_ratio=2)
  2. nuremberg_v5.png  - class map from V5 (SpectralSpatialNetV5)
  3. nuremberg_diff.png - agreement/disagreement overlay

Diff visualization:
  - Pixels where V3 and V5 AGREE: shown in the class color at 40% brightness (dimmed)
  - Pixels where they DISAGREE: shown in V5's predicted color at FULL brightness
  This makes disagreement pixels visually "pop" against the dim background.
"""

import gc
import os
import pickle
import sys
import time

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    SEED, N_CLASSES, CLASS_NAMES, get_train_cities, get_val_cities, get_test_cities,
)
from reproduce.models.shared.data import load_raw_feature_cube, compute_center_indices

CKPT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "prediction_maps")

# Class colors (RGB) — standard EO land cover scheme
CLASS_COLORS = np.array([
    [  0, 128,   0],   # tree_cover   — forest green
    [170, 160,  60],   # shrubland    — olive
    [152, 230, 100],   # grassland    — lime green
    [255, 225,  80],   # cropland     — golden yellow
    [220,  40,  40],   # built_up     — red
    [180, 180, 180],   # bare_sparse  — gray
    [ 30,  80, 220],   # water        — deep blue
], dtype=np.uint8)

NODATA_COLOR = np.array([30, 30, 30], dtype=np.uint8)  # near-black


def ts():
    return time.strftime("%H:%M:%S")


def load_v3_model(device):
    """Load V3 = SpectralSpatialNetV2 with expand_ratio=2, bigger dims."""
    from reproduce.models.architectures.spectral_spatial import SpectralSpatialNetV2

    model = SpectralSpatialNetV2(
        spatial_dims=(48, 96, 192),
        expand_ratio=4,
        temporal_dim=192,
        n_attn_layers=3,
    ).to(device)
    state = torch.load(
        os.path.join(CKPT_DIR, "ssnet_v3_ep3_backup.pt"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    print(f"  V3 loaded: {model.n_params():,} params")

    with open(os.path.join(CKPT_DIR, "ssnet_scaler_v3_backup.pkl"), "rb") as f:
        sc = pickle.load(f)
    return model, sc["patches"], sc["indices"]


def load_v5_model(device):
    """Load V5 = SpectralSpatialNetV5 with default config."""
    from reproduce.models.architectures.spectral_spatial_v5 import SpectralSpatialNetV5

    model = SpectralSpatialNetV5(
        n_bands=12, n_timesteps=6, n_indices=145,
        spatial_dims=(32, 64, 128), expand_ratio=4,
        temporal_dim=128, n_attn_layers=2, n_heads=4,
        n_classes=7, dropout=0.15,
        spatial_branch_drop=0.10, index_branch_drop=0.25,
    ).to(device)
    state = torch.load(
        os.path.join(CKPT_DIR, "ssnet_v5.pt"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    print(f"  V5 loaded: {model.n_params():,} params")

    with open(os.path.join(CKPT_DIR, "ssnet_v5_fixed_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    return model, sc["patches"], sc["indices"]


def extract_row_patches(cube, H, W, row, pad=1):
    """
    Extract 3×3 patches for ALL pixels in a row.
    cube: (H, W, 72) with NaN-pad border
    Returns: (W_orig, 648) patches, (W_orig, 72) center pixels
    """
    # cube is already padded: shape (H+2, W+2, 72)
    patches = np.empty((W, 9 * 72), dtype=np.float32)
    r = row + pad  # offset for padding

    for px_col in range(W):
        c = px_col + pad
        patch_3x3 = cube[r-1:r+2, c-1:c+2, :]  # (3, 3, 72)
        patches[px_col] = patch_3x3.reshape(-1)

    centers = patches[:, 4*72:5*72].copy()
    return patches, centers


def predict_full_raster(model, patch_scaler, idx_scaler, cube, H, W, device,
                        model_name="model", is_v5=False):
    """
    Run model on every pixel. Returns (H, W) int8 class predictions.
    Processes row-by-row to stay within memory.
    """
    # Pad cube with NaN border
    padded = np.full((H + 2, W + 2, 72), np.nan, dtype=np.float32)
    padded[1:H+1, 1:W+1, :] = cube

    predictions = np.full((H, W), 255, dtype=np.uint8)  # 255 = nodata
    BATCH = 8192

    print(f"  [{ts()}] Predicting {model_name}: {H}×{W} = {H*W:,} pixels")

    with torch.no_grad():
        for row in range(H):
            if row % 200 == 0:
                print(f"    row {row}/{H} ({100*row/H:.0f}%)", flush=True)

            patches, centers = extract_row_patches(padded, H, W, row)

            # Check for all-NaN pixels
            valid_mask = np.isfinite(centers).any(axis=1)
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) == 0:
                continue

            valid_patches = patches[valid_idx]
            valid_centers = centers[valid_idx]

            # Replace NaN with 0 before scaling
            np.nan_to_num(valid_patches, nan=0.0, copy=False)
            np.nan_to_num(valid_centers, nan=0.0, copy=False)

            # Compute indices
            indices = compute_center_indices(valid_centers)

            # Scale
            valid_patches = patch_scaler.transform(valid_patches).astype(np.float32)
            indices = idx_scaler.transform(indices).astype(np.float32)

            # Predict in batches
            row_preds = np.empty(len(valid_idx), dtype=np.uint8)
            for bs in range(0, len(valid_idx), BATCH):
                be = min(bs + BATCH, len(valid_idx))
                xp = torch.from_numpy(valid_patches[bs:be]).to(device)
                xi = torch.from_numpy(indices[bs:be]).to(device)

                with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                    if is_v5:
                        out = model(xp, xi)
                        logits = out["logits"]
                    else:
                        logits = model(xp, xi)

                pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                row_preds[bs:be] = pred

            predictions[row, valid_idx] = row_preds

    print(f"  [{ts()}] Done {model_name}")
    return predictions


def class_map_to_rgb(pred_map):
    """Convert (H,W) uint8 class map to (H,W,3) RGB image."""
    H, W = pred_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for ci in range(N_CLASSES):
        mask = pred_map == ci
        rgb[mask] = CLASS_COLORS[ci]
    nodata = pred_map == 255
    rgb[nodata] = NODATA_COLOR
    return rgb


def diff_map_to_rgb(v3_pred, v5_pred):
    """
    Create diff visualization:
      - AGREE: V3 class color at 40% brightness (dimmed)
      - DISAGREE: V5 class color at full brightness (pops out)
    """
    H, W = v3_pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    agree = v3_pred == v5_pred
    disagree = ~agree & (v3_pred != 255) & (v5_pred != 255)

    # Agreed pixels: dimmed class color
    for ci in range(N_CLASSES):
        mask = agree & (v3_pred == ci)
        rgb[mask] = (CLASS_COLORS[ci].astype(np.float32) * 0.4).astype(np.uint8)

    # Disagreed pixels: full-brightness V5 color
    for ci in range(N_CLASSES):
        mask = disagree & (v5_pred == ci)
        rgb[mask] = CLASS_COLORS[ci]

    # Nodata
    nodata = (v3_pred == 255) | (v5_pred == 255)
    rgb[nodata] = NODATA_COLOR

    return rgb


def add_legend(rgb, title=""):
    """Add a legend bar at the bottom of the image."""
    H, W, _ = rgb.shape
    legend_h = 40
    new = np.zeros((H + legend_h, W, 3), dtype=np.uint8)
    new[:H] = rgb
    new[H:] = 20  # dark background

    # Draw color swatches
    swatch_w = min(W // (N_CLASSES + 1), 120)
    x_start = 10
    for ci in range(N_CLASSES):
        x = x_start + ci * (swatch_w + 5)
        new[H+8:H+28, x:x+swatch_w] = CLASS_COLORS[ci]

    return new


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find Nuremberg city config
    all_cities = get_train_cities() + get_val_cities() + get_test_cities()
    nuremberg = None
    for c in all_cities:
        if c.name == "nuremberg":
            nuremberg = c
            break
    if nuremberg is None:
        print("ERROR: Nuremberg not found")
        return

    # Load raw data
    print(f"\n[{ts()}] Loading Nuremberg raw data...")
    cube, H, W = load_raw_feature_cube(nuremberg)
    if cube is None:
        print("ERROR: Could not load Nuremberg data")
        return
    print(f"  Raster: {H} × {W} = {H*W:,} pixels")

    # --- V3 Predictions ---
    print(f"\n[{ts()}] Loading V3 model...")
    v3_model, v3_ps, v3_is = load_v3_model(device)
    v3_pred = predict_full_raster(v3_model, v3_ps, v3_is, cube, H, W, device,
                                   model_name="V3", is_v5=False)
    del v3_model, v3_ps, v3_is
    torch.cuda.empty_cache()
    gc.collect()

    v3_rgb = class_map_to_rgb(v3_pred)
    v3_rgb = add_legend(v3_rgb, "V3")
    img = Image.fromarray(v3_rgb)
    v3_path = os.path.join(OUT_DIR, "nuremberg_v3.png")
    img.save(v3_path)
    print(f"  Saved: {v3_path}")
    del v3_rgb, img
    gc.collect()

    # --- V5 Predictions ---
    print(f"\n[{ts()}] Loading V5 model...")
    v5_model, v5_ps, v5_is = load_v5_model(device)
    v5_pred = predict_full_raster(v5_model, v5_ps, v5_is, cube, H, W, device,
                                   model_name="V5", is_v5=True)
    del v5_model, v5_ps, v5_is, cube
    torch.cuda.empty_cache()
    gc.collect()

    v5_rgb = class_map_to_rgb(v5_pred)
    v5_rgb = add_legend(v5_rgb, "V5")
    img = Image.fromarray(v5_rgb)
    v5_path = os.path.join(OUT_DIR, "nuremberg_v5.png")
    img.save(v5_path)
    print(f"  Saved: {v5_path}")
    del v5_rgb, img
    gc.collect()

    # --- Diff Map ---
    print(f"\n[{ts()}] Creating diff map...")
    valid_mask = (v3_pred != 255) & (v5_pred != 255)
    n_total = valid_mask.sum()
    n_agree = ((v3_pred == v5_pred) & valid_mask).sum()
    n_disagree = n_total - n_agree
    print(f"  Valid pixels:  {n_total:,}")
    print(f"  Agree:         {n_agree:,} ({100*n_agree/max(n_total,1):.1f}%)")
    print(f"  Disagree:      {n_disagree:,} ({100*n_disagree/max(n_total,1):.1f}%)")

    # Per-class disagreement breakdown
    print(f"\n  Disagreement breakdown (V3 class -> V5 changed to):")
    for ci in range(N_CLASSES):
        v3_is_ci = (v3_pred == ci) & valid_mask
        disagree_ci = v3_is_ci & (v5_pred != ci)
        if v3_is_ci.sum() > 0:
            pct = 100 * disagree_ci.sum() / v3_is_ci.sum()
            print(f"    {CLASS_NAMES[ci]:15s}: {disagree_ci.sum():>7,} / {v3_is_ci.sum():>9,} ({pct:.1f}%)")

    diff_rgb = diff_map_to_rgb(v3_pred, v5_pred)
    diff_rgb = add_legend(diff_rgb, "Diff")
    img = Image.fromarray(diff_rgb)
    diff_path = os.path.join(OUT_DIR, "nuremberg_diff.png")
    img.save(diff_path)
    print(f"  Saved: {diff_path}")

    print(f"\n[{ts()}] All done!")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
