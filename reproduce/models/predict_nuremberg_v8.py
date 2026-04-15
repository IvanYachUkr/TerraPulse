#!/usr/bin/env python3
"""
Generate Nuremberg dashboard predictions using SSNet V8.

For each year pair (y, y+1), loads the 6-slot S2+S1 raw data cube from the
dashboard-scale rasters (2850×2550), applies per-band mean NaN fill (matching
the Rust `clean_band_nan_fill` logic), runs SSNet V8 inference per pixel with
3×3 patches, applies the Nuremberg city boundary mask, and writes multi-
resolution .bin files for the dashboard.

NaN fill strategy (CRITICAL):
  - For each of the 72 bands independently, NaN values are replaced with the
    raster-wide mean of that band's finite values.
  - This exactly replicates the Rust `clean_band_nan_fill()` function.
  - Zero fill is NEVER used — zeros would be misinterpreted as water.

Border padding:
  - 3×3 patches at the raster edge use `np.pad(mode='edge')` (replicate
    boundary pixels) instead of zero-padding.
  - During training, border pixels were excluded entirely; edge replication
    is the most conservative approximation for inference.

Validity mask:
  - Pixels where >50% of the 72 raw bands were originally NaN are marked
    invalid and assigned class 255 (transparent).
  - This matches the training filter in shared/data.py:369.

Class mapping:
  - The model outputs 7 classes (0–6).  The dashboard uses 6 classes where
    shrubland (model class 1) is remapped to grassland (dashboard class 1).
  - Dashboard mapping: 0=tree, 1=grassland(+shrub), 2=cropland, 3=built_up,
    4=bare_sparse, 5=water, 255=outside.

Usage:
    python predict_nuremberg_v8.py
"""

import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np
import rasterio
import torch

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    SEED, N_CLASSES, S2_BANDS, S1_BANDS, S2_NODATA, S1_NODATA,
    N_RAW_FEATURES,
)
from reproduce.models.shared.data import compute_center_indices

# ── Paths ────────────────────────────────────────────────────────────────────

CKPT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "cities",
                       "nuremberg_dashboard_v2", "raw")
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "src", "dashboard", "data",
                             "nuremberg_dashboard")
BOUNDARY_GEOJSON = os.path.join(PROJECT_ROOT,
                                "nuremberg_stat_bezirke_wgs84.geojson")

# ── Constants ────────────────────────────────────────────────────────────────

SEASONS = ["spring", "summer", "autumn"]
CITY_TAG = "nuremberg_dashboard"
RESOLUTIONS = list(range(1, 11))  # 1..10, matching dashboard metadata

# Year pairs: each pair (y, y+1) uses 6 temporal slots and is labeled as y+1
# in the dashboard.
PREDICT_YEAR_PAIRS = [(y, y + 1) for y in range(2017, 2025)]

# Model-to-dashboard class mapping:
#   model class 0 (tree_cover)  → dashboard 0 (tree_cover)
#   model class 1 (shrubland)   → dashboard 1 (grassland)  ← MERGED
#   model class 2 (grassland)   → dashboard 1 (grassland)  ← MERGED
#   model class 3 (cropland)    → dashboard 2 (cropland)
#   model class 4 (built_up)    → dashboard 3 (built_up)
#   model class 5 (bare_sparse) → dashboard 4 (bare_sparse)
#   model class 6 (water)       → dashboard 5 (water)
MODEL_TO_DASHBOARD = np.array([0, 1, 1, 2, 3, 4, 5], dtype=np.uint8)


def ts():
    return time.strftime("%H:%M:%S")


# ── TIF Loading ──────────────────────────────────────────────────────────────

def _load_tif(path, nodata_val):
    """Load a single TIF, replace nodata sentinel with NaN."""
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    data[data == nodata_val] = np.nan
    return data


def load_year_pair_cube(year1, year2):
    """
    Load the raw (H, W, 72) feature cube for a year pair.

    Feature layout per temporal slot (12 bands):
        10 S2 bands (B02..B12), then 2 S1 bands (VV, VH)
    Temporal slots: y1_spring, y1_summer, y1_autumn,
                    y2_spring, y2_summer, y2_autumn
    Total: 6 × 12 = 72 features per pixel.

    Returns (cube, H, W) with NaN for missing/nodata pixels.
    """
    bands_list = []
    ref_shape = None

    for year in [year1, year2]:
        for season in SEASONS:
            tag = f"{CITY_TAG}_{year}_{season}"

            # ── Sentinel-2 (10 spectral bands) ──
            s2_path = os.path.join(RAW_DIR, f"sentinel2_{tag}.tif")
            s2 = _load_tif(s2_path, S2_NODATA)

            if s2 is not None and s2.shape[0] >= 10:
                if ref_shape is None:
                    ref_shape = (s2.shape[1], s2.shape[2])
                for bi in range(10):
                    bands_list.append(s2[bi])
            else:
                if ref_shape is None:
                    print(f"    ERROR: Cannot determine raster shape "
                          f"(missing {s2_path})")
                    return None, 0, 0
                for _ in range(10):
                    bands_list.append(
                        np.full(ref_shape, np.nan, dtype=np.float32))

            # ── Sentinel-1 (VV, VH) ──
            s1_path = os.path.join(RAW_DIR, f"sentinel1_{tag}.tif")
            s1 = _load_tif(s1_path, S1_NODATA)

            if s1 is not None and s1.shape[0] >= 2:
                for bi in range(2):
                    bands_list.append(s1[bi])
            else:
                for _ in range(2):
                    bands_list.append(
                        np.full(ref_shape, np.nan, dtype=np.float32))

    if ref_shape is None or len(bands_list) != N_RAW_FEATURES:
        return None, 0, 0

    H, W = ref_shape
    cube = np.stack(bands_list, axis=-1)  # (H, W, 72)
    del bands_list
    return cube, H, W


# ── NaN Fill (Rust-style per-band mean) ──────────────────────────────────────

def per_band_mean_nan_fill(cube):
    """
    Replace NaN values with the per-band raster-wide mean of finite values.

    This exactly replicates the Rust `clean_band_nan_fill()`:
        let fill = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
        raw.iter().map(|&v| if v.is_finite() { v } else { fill }).collect()

    For bands that are 100% NaN (e.g. missing SAR for a whole season),
    the fill value defaults to 0.0 — same as the Rust fallback.

    Args:
        cube: (H, W, 72) array, modified in-place.

    Returns:
        nan_frac: (H, W) array — fraction of bands that were NaN per pixel
                  (computed BEFORE filling, used for validity masking).
    """
    H, W, n_bands = cube.shape

    # Compute per-pixel NaN fraction BEFORE filling
    nan_frac = np.isnan(cube).sum(axis=-1).astype(np.float32) / n_bands

    # Fill each band independently
    for b in range(n_bands):
        band = cube[:, :, b]
        finite_mask = np.isfinite(band)
        n_finite = finite_mask.sum()
        if n_finite > 0:
            fill_val = np.float64(band[finite_mask].sum()) / n_finite
            fill_val = np.float32(fill_val)
        else:
            fill_val = np.float32(0.0)
        band[~finite_mask] = fill_val

    return nan_frac


# ── Nuremberg Boundary Mask ──────────────────────────────────────────────────

def build_nuremberg_mask(anchor_tif_path):
    """
    Rasterize the Nuremberg district boundaries onto the anchor grid.

    Returns (H, W) bool array: True = inside Nuremberg, False = outside.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    if not os.path.exists(BOUNDARY_GEOJSON):
        print(f"  WARNING: Boundary file not found: {BOUNDARY_GEOJSON}")
        print(f"  All pixels will be predicted (no boundary mask).")
        return None

    with rasterio.open(anchor_tif_path) as src:
        transform = src.transform
        H, W = src.height, src.width
        crs = src.crs

    gdf = gpd.read_file(BOUNDARY_GEOJSON).to_crs(crs)
    mask = rasterize(
        gdf.geometry,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    inside = mask > 0
    print(f"  Boundary mask: {inside.sum():,} inside / "
          f"{(~inside).sum():,} outside of {H*W:,} total")
    return inside


# ── Model Loading ────────────────────────────────────────────────────────────

def load_ssnet_v8(device):
    """Load the trained SSNet V8 model and its scalers."""
    from reproduce.models.architectures.spectral_spatial_v8 import (
        SpectralSpatialNetV8,
    )

    model = SpectralSpatialNetV8(
        n_bands=12, n_timesteps=6, n_indices=145,
        spatial_dims=(32, 64, 128), expand_ratio=4,
        temporal_dim=128, n_attn_layers=2, n_heads=4,
        n_classes=N_CLASSES, dropout=0.12,
        prior_hidden=96,
    ).to(device)

    state = torch.load(os.path.join(CKPT_DIR, "ssnet_v8.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with open(os.path.join(CKPT_DIR, "ssnet_v8_fixed_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)

    return model, sc["patches"], sc["indices"]


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_full_raster(model, patch_scaler, idx_scaler, cube, valid_mask,
                        H, W, device, batch_size=4096):
    """
    Run SSNet V8 on every valid pixel in the filled cube.

    Args:
        cube: (H, W, 72) — already NaN-filled (per-band mean).
        valid_mask: (H, W) bool — True for pixels to predict.

    Returns:
        pred: (H, W) uint8 — model class index for valid pixels,
              255 for invalid/outside pixels.
    """
    # Pad with edge replication for 3×3 neighborhood extraction.
    # This avoids zero-padding which would create artificial water signatures
    # at the raster boundaries.
    padded = np.pad(cube, ((1, 1), (1, 1), (0, 0)), mode='edge')

    pred = np.full((H, W), 255, dtype=np.uint8)

    total_valid = valid_mask.sum()
    processed = 0

    for r in range(H):
        # Check how many valid pixels this row has
        row_valid = valid_mask[r]
        n_row_valid = row_valid.sum()
        if n_row_valid == 0:
            continue

        # Extract 3×3 patches for ALL columns, then filter to valid
        row_patches = np.empty((W, 9 * 72), dtype=np.float32)
        for c in range(W):
            patch = padded[r:r+3, c:c+3, :]  # (3, 3, 72)
            row_patches[c] = patch.reshape(-1)

        # Center pixel raw values (position 4 in 3×3 = index 4*72 : 5*72)
        centers = row_patches[:, 4*72:5*72].copy()

        # Only process valid pixels
        valid_cols = row_valid
        vp = row_patches[valid_cols]
        vc = centers[valid_cols]
        n_valid = len(vp)

        # Compute spectral indices from center pixels
        indices = compute_center_indices(vc)

        # Scale
        patches_s = patch_scaler.transform(vp).astype(np.float32)
        indices_s = idx_scaler.transform(indices).astype(np.float32)
        center_raw = vc.astype(np.float32)

        # Inference in batches
        preds_valid = np.empty(n_valid, dtype=np.uint8)
        with torch.no_grad():
            for s in range(0, n_valid, batch_size):
                e = min(s + batch_size, n_valid)
                xp = torch.from_numpy(patches_s[s:e]).to(device)
                xi = torch.from_numpy(indices_s[s:e]).to(device)
                xc = torch.from_numpy(center_raw[s:e]).to(device)
                out = model(xp, xi, xc)
                logits = out["logits"]
                preds_valid[s:e] = (logits.argmax(dim=1)
                                    .cpu().numpy().astype(np.uint8))
                del xp, xi, xc, out, logits

        pred[r, valid_cols] = preds_valid
        processed += n_valid

        if r % 200 == 0 and r > 0:
            print(f"    Row {r}/{H} — {processed:,}/{total_valid:,} pixels")

    return pred


# ── Downsampling ─────────────────────────────────────────────────────────────

def downsample_majority(pred, H, W, res):
    """Downsample by majority vote in res×res blocks, ignoring class 255."""
    bh, bw = H // res, W // res
    out = np.full((bh, bw), 255, dtype=np.uint8)
    for r in range(bh):
        for c in range(bw):
            block = pred[r*res:(r+1)*res, c*res:(c+1)*res]
            valid = block[block < 255]
            if len(valid) == 0:
                continue
            classes, counts = np.unique(valid, return_counts=True)
            out[r, c] = classes[counts.argmax()]
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SSNet V8 Nuremberg dashboard predictions")
    parser.add_argument("--device", default="auto",
                        help="cuda or cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*72}")
    print(f"  SSNet V8 — Nuremberg Dashboard Predictions")
    print(f"  Device: {device}")
    print(f"  Raw dir: {RAW_DIR}")
    print(f"  Output:  {DASHBOARD_DIR}")
    print(f"{'='*72}\n")

    # Verify raw data exists
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: Raw dir not found: {RAW_DIR}")
        sys.exit(1)

    # Load model
    print(f"[{ts()}] Loading SSNet V8...")
    model, patch_scaler, idx_scaler = load_ssnet_v8(device)
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Build city boundary mask (from first available anchor TIF)
    anchor_path = os.path.join(
        RAW_DIR, f"sentinel2_{CITY_TAG}_2020_spring.tif")
    print(f"\n[{ts()}] Building Nuremberg boundary mask...")
    inside_city = build_nuremberg_mask(anchor_path)

    with rasterio.open(anchor_path) as src:
        H, W = src.height, src.width
    print(f"  Dashboard grid: {H}×{W} = {H*W:,} pixels")

    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    for y1, y2 in PREDICT_YEAR_PAIRS:
        dashboard_year = y2  # dashboard labels predictions by second year
        print(f"\n{'-'*72}")
        print(f"  [{ts()}] Year pair ({y1}, {y2}) -> dashboard year {dashboard_year}")
        print(f"{'-'*72}")

        # Load raw cube
        print(f"  [{ts()}] Loading raw cube...")
        cube, cube_H, cube_W = load_year_pair_cube(y1, y2)
        if cube is None:
            print(f"  ERROR: Could not load data. Skipping.")
            continue
        assert cube_H == H and cube_W == W, \
            f"Grid mismatch: {cube_H}×{cube_W} vs {H}×{W}"

        # Per-band mean NaN fill + compute validity mask
        print(f"  [{ts()}] Applying per-band mean NaN fill (Rust-style)...")
        nan_frac = per_band_mean_nan_fill(cube)

        # Validity: <50% NaN bands (same threshold as training)
        pixel_valid = nan_frac < 0.5
        print(f"    Valid pixels (NaN<50%): {pixel_valid.sum():,} / {H*W:,}")

        # Combine with city boundary mask
        if inside_city is not None:
            predict_mask = pixel_valid & inside_city
        else:
            predict_mask = pixel_valid
        print(f"    Pixels to predict: {predict_mask.sum():,}")

        # Run SSNet V8
        print(f"  [{ts()}] Running SSNet V8 inference...")
        pred_7class = predict_full_raster(
            model, patch_scaler, idx_scaler,
            cube, predict_mask, H, W, device,
            batch_size=args.batch_size,
        )
        del cube
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Remap 7-class → 6-class dashboard mapping
        # Only remap predicted pixels (not 255)
        predicted = pred_7class < N_CLASSES
        pred_dashboard = np.full((H, W), 255, dtype=np.uint8)
        pred_dashboard[predicted] = MODEL_TO_DASHBOARD[pred_7class[predicted]]

        # Quick class distribution check
        for ci in range(6):
            n = (pred_dashboard == ci).sum()
            if n > 0:
                print(f"    Dashboard class {ci}: {n:,} px")

        # Generate all resolutions and save
        for res in RESOLUTIONS:
            if res == 1:
                out = pred_dashboard
                bh, bw = H, W
            else:
                bh, bw = H // res, W // res
                out = downsample_majority(pred_dashboard, H, W, res)

            fname = f"nuremberg_ssnet_v8_pred_{dashboard_year}_res{res}.bin"
            path = os.path.join(DASHBOARD_DIR, fname)
            out.tofile(path)
            actual_bytes = os.path.getsize(path)
            expected_bytes = bh * bw
            assert actual_bytes == expected_bytes, \
                f"Size mismatch: {actual_bytes} != {expected_bytes} " \
                f"for {bh}×{bw}"
            if res <= 3 or res == 10:
                print(f"    {fname} ({bh}×{bw}, {actual_bytes:,} B)")

        del pred_7class, pred_dashboard
        gc.collect()

    print(f"\n[{ts()}] All predictions saved to: {DASHBOARD_DIR}")
    total_files = len(PREDICT_YEAR_PAIRS) * len(RESOLUTIONS)
    print(f"  Total: {total_files} bin files "
          f"(prefix: nuremberg_ssnet_v8_pred_)")


if __name__ == "__main__":
    main()
