#!/usr/bin/env python3
"""
Generate satellite-overlay prediction maps for Nuremberg.

Creates 3 high-res PNGs (4x upscaled):
  1. nuremberg_v3_overlay.png  - Satellite + V3 predictions at 50% opacity
  2. nuremberg_v5_overlay.png  - Satellite + V5 predictions at 50% opacity
  3. nuremberg_diff_overlay.png - Satellite base, disagreement pixels highlighted

For the diff map:
  - Where V3 and V5 AGREE: pure satellite image, no overlay
  - Where they DISAGREE: V5 class color at 70% opacity + thin white border
  This makes disagreement areas pop against the natural satellite background.
"""

import gc
import os
import sys
import time

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    N_CLASSES, CLASS_NAMES,
    get_train_cities, get_val_cities, get_test_cities,
)
from reproduce.models.shared.data import load_raw_feature_cube

PRED_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "prediction_maps")
OUT_DIR = PRED_DIR  # same directory
SCALE = 4  # upscale factor

# Class colors (RGB) - saturated for visibility on satellite
CLASS_COLORS = np.array([
    [ 20, 200,  20],   # tree_cover   - bright green
    [210, 180,  50],   # shrubland    - golden olive
    [140, 255, 100],   # grassland    - lime
    [255, 240,  50],   # cropland     - yellow
    [255,  40,  40],   # built_up     - bright red
    [200, 200, 200],   # bare_sparse  - white-gray
    [ 40, 100, 255],   # water        - bright blue
], dtype=np.uint8)

NODATA_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def ts():
    return time.strftime("%H:%M:%S")


def make_satellite_rgb(cube, H, W):
    """
    Extract true-color (R=B04, G=B03, B=B02) from summer 2021.
    Apply percentile stretch for contrast.
    """
    # Summer 2021 = timestep 4 (0=2020spring, 1=2020summer, 2=2020autumn,
    #                            3=2021spring, 4=2021summer, 5=2021autumn)
    slot = 4
    off = slot * 12
    red   = cube[:, :, off + 2].copy()  # B04
    green = cube[:, :, off + 1].copy()  # B03
    blue  = cube[:, :, off + 0].copy()  # B02

    rgb = np.stack([red, green, blue], axis=-1)  # (H, W, 3)

    # Replace NaN
    nan_mask = ~np.isfinite(rgb)
    rgb[nan_mask] = 0.0

    # Percentile stretch per channel (2-98%)
    for ch in range(3):
        band = rgb[:, :, ch]
        valid = band[band > 0]
        if len(valid) == 0:
            continue
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        if hi - lo < 1e-6:
            continue
        band = np.clip((band - lo) / (hi - lo), 0, 1)
        rgb[:, :, ch] = band

    # Gamma correction for brightness (satellite images tend to be dark)
    rgb = np.power(rgb, 0.7)

    return (rgb * 255).clip(0, 255).astype(np.uint8)


def reconstruct_class_map(png_path):
    """Load a prediction PNG and reverse-map colors to class IDs."""
    img = np.array(Image.open(png_path))
    # Remove legend bar (40px at bottom)
    H = img.shape[0] - 40
    rgb = img[:H]

    pred = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    # Original render colors (must match render_predictions.py)
    orig_colors = np.array([
        [  0, 128,   0], [170, 160,  60], [152, 230, 100], [255, 225,  80],
        [220,  40,  40], [180, 180, 180], [ 30,  80, 220],
    ], dtype=np.uint8)

    for ci in range(7):
        mask = np.all(rgb == orig_colors[ci], axis=2)
        pred[mask] = ci
    return pred


def upscale_nearest(arr, scale):
    """Upscale 2D or 3D array by repeating pixels."""
    if arr.ndim == 2:
        return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
    else:
        return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def blend_overlay(base_rgb, class_map, alpha=0.5):
    """
    Blend satellite base with class-colored overlay.
    base_rgb: (H, W, 3) uint8
    class_map: (H, W) uint8 class IDs
    Returns: (H, W, 3) uint8
    """
    result = base_rgb.copy()
    a = alpha

    for ci in range(N_CLASSES):
        mask = class_map == ci
        if not mask.any():
            continue
        color = CLASS_COLORS[ci].astype(np.float32)
        result[mask] = (
            (1 - a) * base_rgb[mask].astype(np.float32) + a * color
        ).clip(0, 255).astype(np.uint8)

    return result


def make_diff_overlay(base_rgb, v3_pred, v5_pred):
    """
    Satellite base with disagreement highlighted.
    - Agree: pure satellite (untouched)
    - Disagree: V5 class color at 70% opacity + white 1px border
    """
    H, W = v3_pred.shape
    result = base_rgb.copy()

    valid = (v3_pred != 255) & (v5_pred != 255)
    disagree = (v3_pred != v5_pred) & valid

    # Dilate disagree mask by 1px for border
    border = np.zeros_like(disagree)
    border[1:, :] |= disagree[:-1, :]
    border[:-1, :] |= disagree[1:, :]
    border[:, 1:] |= disagree[:, :-1]
    border[:, :-1] |= disagree[:, 1:]
    border = border & ~disagree  # border only, not interior

    # Fill disagreement with V5 color at 70% opacity
    for ci in range(N_CLASSES):
        mask = disagree & (v5_pred == ci)
        if not mask.any():
            continue
        color = CLASS_COLORS[ci].astype(np.float32)
        result[mask] = (
            0.3 * base_rgb[mask].astype(np.float32) + 0.7 * color
        ).clip(0, 255).astype(np.uint8)

    # White border around disagreement
    result[border] = (
        0.5 * result[border].astype(np.float32) + 0.5 * 255
    ).clip(0, 255).astype(np.uint8)

    return result


def add_legend_bar(img, title="", bar_height=60):
    """Add a legend bar at the bottom."""
    H, W, _ = img.shape
    new_H = H + bar_height
    canvas = np.zeros((new_H, W, 3), dtype=np.uint8)
    canvas[:H] = img
    canvas[H:] = 25  # dark background

    # Swatch layout
    swatch_w = 30
    swatch_h = 16
    gap = 8
    text_space = 0  # we can't render text without PIL.ImageDraw, just swatches
    x = 20
    y_top = H + (bar_height - swatch_h) // 2

    for ci in range(N_CLASSES):
        canvas[y_top:y_top+swatch_h, x:x+swatch_w] = CLASS_COLORS[ci]
        x += swatch_w + gap

    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[{ts()}] Starting overlay generation (scale={SCALE}x)")

    # 1. Load satellite image
    print(f"[{ts()}] Loading Nuremberg raw data...")
    all_cities = get_train_cities() + get_val_cities() + get_test_cities()
    nuremberg = [c for c in all_cities if c.name == "nuremberg"][0]
    cube, H, W = load_raw_feature_cube(nuremberg)
    print(f"  Raster: {H} x {W}")

    print(f"[{ts()}] Building true-color satellite composite...")
    sat_rgb = make_satellite_rgb(cube, H, W)
    del cube
    gc.collect()
    print(f"  Satellite RGB: {sat_rgb.shape}, dtype={sat_rgb.dtype}")

    # 2. Load predictions from existing PNGs
    print(f"[{ts()}] Reconstructing class maps from PNGs...")
    v3_pred = reconstruct_class_map(os.path.join(PRED_DIR, "nuremberg_v3.png"))
    v5_pred = reconstruct_class_map(os.path.join(PRED_DIR, "nuremberg_v5.png"))
    print(f"  V3 pred: {v3_pred.shape}, V5 pred: {v5_pred.shape}")
    assert v3_pred.shape == (H, W), f"V3 shape mismatch: {v3_pred.shape} vs ({H},{W})"

    # 3. Upscale everything
    print(f"[{ts()}] Upscaling {SCALE}x...")
    sat_up = upscale_nearest(sat_rgb, SCALE)
    v3_up = upscale_nearest(v3_pred, SCALE)
    v5_up = upscale_nearest(v5_pred, SCALE)
    print(f"  Upscaled: {sat_up.shape}")
    del sat_rgb, v3_pred, v5_pred
    gc.collect()

    # 4. Generate Image 1: Satellite + V3 at 50% opacity
    print(f"[{ts()}] Creating V3 overlay...")
    v3_overlay = blend_overlay(sat_up, v3_up, alpha=0.5)
    v3_overlay = add_legend_bar(v3_overlay, "V3")
    Image.fromarray(v3_overlay).save(
        os.path.join(OUT_DIR, "nuremberg_v3_overlay.png"),
        optimize=True,
    )
    print(f"  Saved nuremberg_v3_overlay.png")
    del v3_overlay
    gc.collect()

    # 5. Generate Image 2: Satellite + V5 at 50% opacity
    print(f"[{ts()}] Creating V5 overlay...")
    v5_overlay = blend_overlay(sat_up, v5_up, alpha=0.5)
    v5_overlay = add_legend_bar(v5_overlay, "V5")
    Image.fromarray(v5_overlay).save(
        os.path.join(OUT_DIR, "nuremberg_v5_overlay.png"),
        optimize=True,
    )
    print(f"  Saved nuremberg_v5_overlay.png")
    del v5_overlay
    gc.collect()

    # 6. Generate Image 3: Satellite + disagreement highlights
    print(f"[{ts()}] Creating diff overlay...")
    diff_overlay = make_diff_overlay(sat_up, v3_up, v5_up)
    diff_overlay = add_legend_bar(diff_overlay, "Diff")
    Image.fromarray(diff_overlay).save(
        os.path.join(OUT_DIR, "nuremberg_diff_overlay.png"),
        optimize=True,
    )
    print(f"  Saved nuremberg_diff_overlay.png")

    # Stats
    valid = (v3_up != 255) & (v5_up != 255)
    agree = (v3_up == v5_up) & valid
    disagree = ~agree & valid
    print(f"\n  Agreement: {agree.sum():,} / {valid.sum():,} "
          f"({100*agree.sum()/valid.sum():.1f}%)")
    print(f"  Disagreement: {disagree.sum():,} ({100*disagree.sum()/valid.sum():.1f}%)")

    print(f"\n[{ts()}] All overlays saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
