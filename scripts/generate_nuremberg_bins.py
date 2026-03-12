#!/usr/bin/env python3
"""
generate_nuremberg_bins.py – Regenerate pixel-level Nuremberg prediction .bin
files using the CatBoost V5 pixel classifier (217 features).

Reads raw Sentinel-2 (10 bands) and Sentinel-1 (VV, VH) rasters per season,
builds the 217-feature matrix per pixel, runs the CatBoost model (GPU), and
writes multi-resolution .bin files to a validation directory.

Usage:
    python scripts/generate_nuremberg_bins.py              # all year pairs
    python scripts/generate_nuremberg_bins.py 2020 2021    # single year pair
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from catboost import CatBoostClassifier
from scipy.stats import mode as scipy_mode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "cities" / "nuremberg_dashboard" / "raw"
WC_DIR = PROJECT_DIR / "data" / "worldcover"
# Write to a NEW directory for validation — do NOT overwrite existing bins
OUTPUT_DIR = PROJECT_DIR / "data" / "cities" / "nuremberg_bins_new"
MODEL_PATH = PROJECT_DIR / "data" / "cities" / "models_pixel_v5" / "catboost_pixel_v5_deep_unweighted.cbm"

# Dashboard anchor grid
ANCHOR_W = 2550
ANCHOR_H = 2850
NODATA = -9999.0

# S2 band names (index 0-9 in the raster)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEASONS = ["spring", "summer", "autumn"]

# WorldCover class mapping (7 classes, model output)
# Model outputs 0-6: tree, shrubland, grassland, cropland, built_up, bare_sparse, water
# Dashboard uses 0-5: tree, grassland, cropland, built_up, bare_sparse, water (shrubland → grassland)
MODEL_TO_DASHBOARD = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# ESA WorldCover codes → dashboard 6-class scheme
WC_TO_CLASS = {10: 0, 20: 1, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 1}


def ts():
    return time.strftime("%H:%M:%S")


def load_raster(path, bands=None):
    """Load raster bands as float32 array of shape (bands, H, W)."""
    with rasterio.open(path) as src:
        if bands is not None:
            return src.read(bands).astype(np.float32)
        return src.read().astype(np.float32)


def compute_indices(s2):
    """Compute spectral indices from 10-band S2 array.

    Input s2 shape: (10, H, W) — bands B02-B12 in S2_BANDS order.
    Returns dict of index_name → (H, W) array.
    """
    # Band indices in S2_BANDS order
    B04 = s2[2]  # Red
    B05 = s2[3]  # Red Edge 1
    B06 = s2[4]  # Red Edge 2
    B08 = s2[6]  # NIR
    B8A = s2[7]  # NIR narrow
    B11 = s2[8]  # SWIR 1
    B12 = s2[9]  # SWIR 2
    B03 = s2[1]  # Green

    eps = 1e-10

    ndvi = (B08 - B04) / (B08 + B04 + eps)
    ndwi = (B03 - B08) / (B03 + B08 + eps)
    ndbi = (B11 - B08) / (B11 + B08 + eps)
    ndmi = (B08 - B11) / (B08 + B11 + eps)
    nbr = (B08 - B12) / (B08 + B12 + eps)
    bsi = ((B11 + B04) - (B08 + B03)) / ((B11 + B04) + (B08 + B03) + eps)
    evi2 = 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1.0 + eps)
    ndre1 = (B08 - B05) / (B08 + B05 + eps)
    ndre2 = (B08 - B06) / (B08 + B06 + eps)

    return {
        "NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "NDMI": ndmi,
        "NBR": nbr, "BSI": bsi, "EVI2": evi2, "NDRE1": ndre1, "NDRE2": ndre2,
    }


INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDMI", "NBR", "BSI", "EVI2", "NDRE1", "NDRE2"]


def build_features_for_year_pair(year1, year2):
    """Build the 217-feature pixel matrix for a year pair.

    Features match the CatBoost V5 model's expected order.
    Returns (features, valid_mask) where features has shape (n_valid, 217)
    and valid_mask is (H, W) boolean.
    """
    print(f"[{ts()}] Building features for {year1}+{year2}...")

    # Load all rasters for both years
    data = {}  # (year, season) → {"s2": (10, H, W), "sar": (2, H, W), "indices": dict}
    for year in [year1, year2]:
        for season in SEASONS:
            s2_path = RAW_DIR / f"sentinel2_nuremberg_dashboard_{year}_{season}.tif"
            sar_path = RAW_DIR / f"sentinel1_nuremberg_dashboard_{year}_{season}.tif"

            if not s2_path.exists():
                print(f"  WARNING: Missing {s2_path.name}")
                return None, None
            if not sar_path.exists():
                print(f"  WARNING: Missing {sar_path.name}")
                return None, None

            s2 = load_raster(s2_path, bands=list(range(1, 11)))  # 10 bands
            sar = load_raster(sar_path)  # VV, VH

            # Replace nodata with NaN
            s2[s2 == NODATA] = np.nan
            sar[sar == NODATA] = np.nan

            indices = compute_indices(s2)
            data[(year, season)] = {"s2": s2, "sar": sar, "indices": indices}

    # Build valid mask: all bands finite in all seasons/years
    valid = np.ones((ANCHOR_H, ANCHOR_W), dtype=bool)
    for key, d in data.items():
        valid &= np.all(np.isfinite(d["s2"]), axis=0)
        valid &= np.all(np.isfinite(d["sar"]), axis=0)

    n_valid = valid.sum()
    print(f"  Valid pixels: {n_valid:,} / {ANCHOR_H * ANCHOR_W:,}")

    # Build features in exact model order
    features = []

    # --- Per season per year: 22 features ---
    for year in [year1, year2]:
        for season in SEASONS:
            d = data[(year, season)]
            # 10 S2 bands
            for i in range(10):
                features.append(d["s2"][i][valid])
            # 9 spectral indices
            for idx_name in INDEX_NAMES:
                features.append(d["indices"][idx_name][valid])
            # 3 SAR: VV, VH, VV/VH ratio
            vv = d["sar"][0][valid]
            vh = d["sar"][1][valid]
            features.append(vv)
            features.append(vh)
            features.append(vv / (vh + 1e-10))

    # --- Intra-annual index diffs (per year, 18 features) ---
    for year in [year1, year2]:
        for idx_name in INDEX_NAMES:
            sp = data[(year, "spring")]["indices"][idx_name][valid]
            sm = data[(year, "summer")]["indices"][idx_name][valid]
            au = data[(year, "autumn")]["indices"][idx_name][valid]
            features.append(sm - sp)  # diff_summer_spring
        for idx_name in INDEX_NAMES:
            sp = data[(year, "spring")]["indices"][idx_name][valid]
            sm = data[(year, "summer")]["indices"][idx_name][valid]
            au = data[(year, "autumn")]["indices"][idx_name][valid]
            features.append(au - sm)  # diff_autumn_summer

    # --- Interannual index diffs (27 features) ---
    for season in SEASONS:
        for idx_name in INDEX_NAMES:
            v1 = data[(year1, season)]["indices"][idx_name][valid]
            v2 = data[(year2, season)]["indices"][idx_name][valid]
            features.append(v2 - v1)

    # --- Index ranges (per year, 4 features each = 8 total) ---
    for year in [year1, year2]:
        for idx_name in ["NDVI", "NDWI", "EVI2", "BSI"]:
            vals = np.stack([
                data[(year, s)]["indices"][idx_name][valid] for s in SEASONS
            ])
            features.append(vals.max(axis=0) - vals.min(axis=0))

    # --- SAR diffs (intra-annual, 4 per year = 8) ---
    for year in [year1, year2]:
        for sar_idx, sar_name in enumerate(["VV", "VH"]):
            sp = data[(year, "spring")]["sar"][sar_idx][valid]
            sm = data[(year, "summer")]["sar"][sar_idx][valid]
            au = data[(year, "autumn")]["sar"][sar_idx][valid]
            features.append(sm - sp)  # diff_summer_spring
            features.append(au - sm)  # diff_autumn_summer

    # --- SAR interannual diffs (6 features) ---
    for season in SEASONS:
        for sar_idx, sar_name in enumerate(["VV", "VH"]):
            v1 = data[(year1, season)]["sar"][sar_idx][valid]
            v2 = data[(year2, season)]["sar"][sar_idx][valid]
            features.append(v2 - v1)

    X = np.column_stack(features).astype(np.float32)
    print(f"  Feature matrix: {X.shape}")

    # Cleanup rasters immediately
    del data, features
    gc.collect()

    return X, valid


def generate_bins(pred_map, year_label, n_classes=6):
    """Write multi-resolution .bin files for predictions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for res in range(1, 11):
        if res == 1:
            out = pred_map
        else:
            rh = ANCHOR_H // res
            rw = ANCHOR_W // res
            # Crop to exact block size, reshape into blocks, take mode
            cropped = pred_map[:rh * res, :rw * res]
            blocks = cropped.reshape(rh, res, rw, res)
            # For each block, find the most common non-255 value
            out = np.full((rh, rw), 255, dtype=np.uint8)
            for cls in range(n_classes):
                counts = np.sum(blocks == cls, axis=(1, 3))
                # Track which class has the most votes per block
                if cls == 0:
                    best_counts = counts.copy()
                    out[:] = 0
                else:
                    better = counts > best_counts
                    out[better] = cls
                    best_counts = np.maximum(best_counts, counts)
            # Where no valid pixel existed, keep 255
            valid_counts = np.sum(blocks != 255, axis=(1, 3))
            out[valid_counts == 0] = 255

        fname = f"nuremberg_pred_{year_label}_res{res}.bin"
        (OUTPUT_DIR / fname).write_bytes(out.tobytes())
        print(f"  Wrote {fname} ({out.shape[0]}×{out.shape[1]})")


def generate_label_bins(wc_path, year_label):
    """Generate label .bin files from WorldCover raster."""
    from rasterio.warp import reproject, Resampling
    from affine import Affine

    ANCHOR_TRANSFORM = Affine(10.0, 0.0, 641740.0, 0.0, -10.0, 5492260.0)

    with rasterio.open(wc_path) as src:
        wc_arr = np.zeros((ANCHOR_H, ANCHOR_W), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=wc_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ANCHOR_TRANSFORM,
            dst_crs="EPSG:32632",
            resampling=Resampling.nearest,
        )

    # Remap to 6-class scheme
    labels = np.full_like(wc_arr, 255, dtype=np.uint8)
    for wc_id, cls_idx in WC_TO_CLASS.items():
        labels[wc_arr == wc_id] = cls_idx

    for res in range(1, 11):
        if res == 1:
            out = labels
        else:
            rh = ANCHOR_H // res
            rw = ANCHOR_W // res
            out = np.full((rh, rw), 255, dtype=np.uint8)
            for r in range(rh):
                for c in range(rw):
                    block = labels[r * res:(r + 1) * res, c * res:(c + 1) * res]
                    valid_b = block[block != 255]
                    if len(valid_b) > 0:
                        out[r, c] = scipy_mode(valid_b, keepdims=False).mode

        fname = f"nuremberg_labels_{year_label}_res{res}.bin"
        (OUTPUT_DIR / fname).write_bytes(out.tobytes())
    print(f"  Wrote labels for {year_label}")


def main():
    print("=" * 60)
    print("Regenerating Nuremberg Predictions with CatBoost V5 (GPU)")
    print("=" * 60)

    # Load model with GPU
    print(f"\n[{ts()}] Loading CatBoost model (GPU)...")
    model = CatBoostClassifier(task_type='GPU')
    model.load_model(str(MODEL_PATH))
    print(f"  Features: {len(model.feature_names_)}")
    print(f"  Classes: {model.classes_}")

    # Year pairs
    all_pairs = [
        (2019, 2020),
        (2020, 2021),
        (2021, 2022),
        (2022, 2023),
        (2023, 2024),
        (2024, 2025),
    ]

    # Allow selecting a single year pair via CLI
    if len(sys.argv) == 3:
        y1, y2 = int(sys.argv[1]), int(sys.argv[2])
        year_pairs = [(y1, y2)]
        print(f"  Processing single pair: {y1}+{y2}")
    else:
        year_pairs = all_pairs
        print(f"  Processing all {len(year_pairs)} pairs")

    for y1, y2 in year_pairs:
        print(f"\n{'='*60}")
        print(f"  Year pair: {y1} + {y2}")
        print(f"{'='*60}")

        X, valid = build_features_for_year_pair(y1, y2)
        if X is None:
            print(f"  SKIPPED (missing data)")
            continue

        # Replace any remaining NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[{ts()}] Running CatBoost GPU on {X.shape[0]:,} pixels...")
        start = time.time()
        preds_raw = model.predict(X).flatten().astype(int)
        elapsed = time.time() - start
        print(f"  Predicted in {elapsed:.1f}s")

        # Remap 7-class model output to 6-class dashboard scheme
        preds_dashboard = np.array([MODEL_TO_DASHBOARD[p] for p in preds_raw], dtype=np.uint8)

        # Build full prediction map
        pred_map = np.full((ANCHOR_H, ANCHOR_W), 255, dtype=np.uint8)
        pred_map[valid] = preds_dashboard

        # Generate bins — label the prediction with the LATER year
        generate_bins(pred_map, y2)

        # Also generate for the earlier year from same pair
        if y1 >= 2020:
            generate_bins(pred_map, y1)

        del X, valid, preds_raw, preds_dashboard, pred_map
        gc.collect()

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
