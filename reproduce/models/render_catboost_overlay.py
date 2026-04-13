#!/usr/bin/env python3
"""
Generate satellite-overlay prediction map for Nuremberg using the CatBoost
pixel_v5 model (the one used in the dashboard).

Produces:
  nuremberg_catboost_v5_overlay.png  - Satellite + CatBoost predictions at 50% opacity
"""

import gc
import os
import sys
import time

import numpy as np
from PIL import Image
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    N_CLASSES, CLASS_NAMES,
    get_train_cities, get_val_cities, get_test_cities,
)
from reproduce.models.shared.data import load_raw_feature_cube

# Import pixel v5 feature building
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "mlp"))
from importlib import import_module
step1 = import_module("01_download_data")
CITY_MAP = step1.CITY_MAP

sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "pixel"))
step2 = import_module("02_train_catboost")
step3 = import_module("03_predict_nuremberg")

PRED_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "prediction_maps")
CATBOOST_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_pixel_v5")
MODEL_NAME = "catboost_pixel_v5_deep_unweighted.cbm"
SCALE = 4

# Same saturated class colors as the other overlays
CLASS_COLORS = np.array([
    [ 20, 200,  20],   # tree_cover
    [210, 180,  50],   # shrubland
    [140, 255, 100],   # grassland
    [255, 240,  50],   # cropland
    [255,  40,  40],   # built_up
    [200, 200, 200],   # bare_sparse
    [ 40, 100, 255],   # water
], dtype=np.uint8)


def ts():
    return time.strftime("%H:%M:%S")


def make_satellite_rgb(cube, H, W):
    """True-color composite from summer 2021."""
    slot = 4  # 2021 summer
    off = slot * 12
    red   = cube[:, :, off + 2].copy()
    green = cube[:, :, off + 1].copy()
    blue  = cube[:, :, off + 0].copy()
    rgb = np.stack([red, green, blue], axis=-1)
    nan_mask = ~np.isfinite(rgb)
    rgb[nan_mask] = 0.0
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
    rgb = np.power(rgb, 0.7)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def upscale_nearest(arr, scale):
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def blend_overlay(base_rgb, class_map, alpha=0.5):
    result = base_rgb.copy()
    for ci in range(N_CLASSES):
        mask = class_map == ci
        if not mask.any():
            continue
        color = CLASS_COLORS[ci].astype(np.float32)
        result[mask] = (
            (1 - alpha) * base_rgb[mask].astype(np.float32) + alpha * color
        ).clip(0, 255).astype(np.uint8)
    return result


def add_legend_bar(img, bar_height=60):
    H, W, _ = img.shape
    canvas = np.zeros((H + bar_height, W, 3), dtype=np.uint8)
    canvas[:H] = img
    canvas[H:] = 25
    swatch_w = 30
    swatch_h = 16
    gap = 8
    x = 20
    y_top = H + (bar_height - swatch_h) // 2
    for ci in range(N_CLASSES):
        canvas[y_top:y_top+swatch_h, x:x+swatch_w] = CLASS_COLORS[ci]
        x += swatch_w + gap
    return canvas


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    print(f"[{ts()}] CatBoost pixel_v5 overlay generation")

    # Load CatBoost model
    model_path = os.path.join(CATBOOST_DIR, MODEL_NAME)
    print(f"[{ts()}] Loading CatBoost model: {MODEL_NAME}")
    model = CatBoostClassifier()
    model.load_model(model_path)
    print(f"  Model loaded ({model.tree_count_} trees)")

    # Load satellite imagery for base
    print(f"[{ts()}] Loading Nuremberg raw data for satellite base...")
    all_cities = get_train_cities() + get_val_cities() + get_test_cities()
    nuremberg = [c for c in all_cities if c.name == "nuremberg"][0]
    cube, H, W = load_raw_feature_cube(nuremberg)
    print(f"  Raster: {H} x {W}")

    print(f"[{ts()}] Building satellite RGB composite...")
    sat_rgb = make_satellite_rgb(cube, H, W)
    del cube
    gc.collect()

    # Build CatBoost features for the 2020-2021 year pair
    print(f"[{ts()}] Building CatBoost features (2020-2021)...")
    raw_dir = step3._raw_dir_nuremberg()
    anchor = os.path.join(raw_dir, "sentinel2_nuremberg_2020_spring.tif")
    feat_cube, valid, feat_names = step3.build_prediction_features(
        raw_dir, (2020, 2021), anchor, H, W)
    print(f"  Features: {feat_cube.shape[-1]} per pixel, valid: {valid.sum():,}/{H*W:,}")

    # Run predictions
    print(f"[{ts()}] Predicting {valid.sum():,} pixels with CatBoost...")
    flat_X = feat_cube.reshape(-1, feat_cube.shape[-1])
    flat_valid = valid.reshape(-1)
    del feat_cube
    gc.collect()

    pred_flat = np.full(H * W, 255, dtype=np.uint8)
    X_predict = flat_X[flat_valid]
    CHUNK = 500_000
    all_preds = []
    for i in range(0, len(X_predict), CHUNK):
        chunk_end = min(i + CHUNK, len(X_predict))
        print(f"    Chunk {i:,}-{chunk_end:,}...")
        all_preds.append(model.predict(X_predict[i:chunk_end]).flatten().astype(np.uint8))
    pred_flat[flat_valid] = np.concatenate(all_preds)
    pred_2d = pred_flat.reshape(H, W)
    del flat_X, flat_valid, X_predict, all_preds
    gc.collect()

    # Check class distribution
    for ci in range(N_CLASSES):
        n = (pred_2d == ci).sum()
        print(f"    {CLASS_NAMES[ci]:15s}: {n:>9,} ({100*n/valid.sum():.1f}%)")

    # Save raw prediction map (matching the other formats)
    print(f"[{ts()}] Saving raw prediction map...")
    from PIL import Image as PILImage
    orig_colors = np.array([
        [  0, 128,   0], [170, 160,  60], [152, 230, 100], [255, 225,  80],
        [220,  40,  40], [180, 180, 180], [ 30,  80, 220],
    ], dtype=np.uint8)
    raw_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    raw_rgb[pred_2d == 255] = [30, 30, 30]
    for ci in range(N_CLASSES):
        raw_rgb[pred_2d == ci] = orig_colors[ci]
    # Add legend
    legend_h = 40
    raw_full = np.zeros((H + legend_h, W, 3), dtype=np.uint8)
    raw_full[:H] = raw_rgb
    raw_full[H:] = 20
    swatch_w = min(W // 8, 120)
    for ci in range(N_CLASSES):
        x = 10 + ci * (swatch_w + 5)
        raw_full[H+8:H+28, x:x+swatch_w] = orig_colors[ci]
    PILImage.fromarray(raw_full).save(os.path.join(PRED_DIR, "nuremberg_catboost_v5.png"))
    print(f"  Saved nuremberg_catboost_v5.png")
    del raw_rgb, raw_full
    gc.collect()

    # Upscale and create overlay
    print(f"[{ts()}] Upscaling {SCALE}x and creating overlay...")
    sat_up = upscale_nearest(sat_rgb, SCALE)
    pred_up = upscale_nearest(pred_2d, SCALE)
    del sat_rgb, pred_2d
    gc.collect()

    overlay = blend_overlay(sat_up, pred_up, alpha=0.5)
    overlay = add_legend_bar(overlay)

    out_path = os.path.join(PRED_DIR, "nuremberg_catboost_v5_overlay.png")
    Image.fromarray(overlay).save(out_path, optimize=True)
    print(f"  Saved nuremberg_catboost_v5_overlay.png")

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
