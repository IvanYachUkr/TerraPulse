#!/usr/bin/env python3
"""
generate_nuremberg_bins_filled.py – Generate Nuremberg prediction bins
WITH pixel-level filling for S2 NODATA gaps.

Filling strategy (applied per-pixel, per-band):
  1. If a pixel is NODATA in one S2 season but valid in the other same-year
     seasons → fill with the mean of the valid same-year seasons.
  2. If still NODATA → fill from the corresponding season of the OTHER year
     in the pair (temporal neighbor).
  3. If still NODATA → fill with the mean of ALL valid seasons across both years.
  4. If ALL seasons are NODATA for a pixel → fill with the spatial median of
     a 5×5 neighborhood from the best available season.

S1 (SAR) always has full coverage so no filling needed there.

Outputs bins to: data/cities/nuremberg_bins_filled/
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "cities" / "nuremberg_dashboard" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "cities" / "nuremberg_bins_filled"
MODEL_PATH = PROJECT_DIR / "data" / "cities" / "models_pixel_v5" / "catboost_pixel_v5_deep_unweighted.cbm"

# Dashboard anchor grid
ANCHOR_W = 2550
ANCHOR_H = 2850
NODATA = -9999.0

# S2 band names (index 0-9 in the raster)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEASONS = ["spring", "summer", "autumn"]

# 7-class model → 6-class dashboard mapping
MODEL_TO_DASHBOARD = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDMI", "NBR", "BSI", "EVI2", "NDRE1", "NDRE2"]


def ts():
    return time.strftime("%H:%M:%S")


def load_raster(path, bands=None):
    """Load raster bands as float32 array of shape (bands, H, W)."""
    with rasterio.open(path) as src:
        if bands is not None:
            return src.read(bands).astype(np.float32)
        return src.read().astype(np.float32)


def fill_s2_nodata(s2_data, year_seasons):
    """Fill S2 NODATA pixels using multi-season/multi-year strategy.
    
    s2_data: dict mapping (year, season) → (10, H, W) float32 array
    year_seasons: list of (year, season) tuples
    
    Modifies arrays in-place. Returns count of filled pixels.
    """
    H, W = list(s2_data.values())[0].shape[1:]
    n_bands = 10
    total_filled = 0
    
    # Group by year
    years = sorted(set(y for y, s in year_seasons))
    
    for year in years:
        for season in SEASONS:
            key = (year, season)
            if key not in s2_data:
                continue
            arr = s2_data[key]  # (10, H, W)
            bad_mask = np.any(arr == NODATA, axis=0)  # (H, W)
            n_bad = bad_mask.sum()
            if n_bad == 0:
                continue
            
            print(f"    Filling {year} {season}: {n_bad:,} bad pixels")
            filled_count = 0
            
            # Strategy 1: fill from other same-year seasons
            other_same_year = []
            for s in SEASONS:
                if s != season and (year, s) in s2_data:
                    other = s2_data[(year, s)]
                    other_valid = np.all(other != NODATA, axis=0) & np.all(np.isfinite(other), axis=0)
                    other_same_year.append((other, other_valid))
            
            if other_same_year:
                # For each bad pixel, average the valid same-year seasons
                still_bad = bad_mask.copy()
                fill_sum = np.zeros((n_bands, H, W), dtype=np.float64)
                fill_count = np.zeros((H, W), dtype=np.int32)
                for other_arr, other_valid in other_same_year:
                    can_contribute = still_bad & other_valid
                    for b in range(n_bands):
                        fill_sum[b] += np.where(can_contribute, other_arr[b], 0)
                    fill_count += can_contribute.astype(np.int32)
                
                can_fill = still_bad & (fill_count > 0)
                if can_fill.any():
                    for b in range(n_bands):
                        arr[b] = np.where(
                            can_fill,
                            (fill_sum[b] / np.maximum(fill_count, 1)).astype(np.float32),
                            arr[b]
                        )
                    n_filled = can_fill.sum()
                    filled_count += n_filled
                    still_bad &= ~can_fill
            
            # Strategy 2: fill from corresponding season of the other year
            other_year = [y for y in years if y != year]
            for oy in other_year:
                if (oy, season) in s2_data:
                    still_bad = np.any(arr == NODATA, axis=0)
                    if not still_bad.any():
                        break
                    donor = s2_data[(oy, season)]
                    donor_valid = np.all(donor != NODATA, axis=0) & np.all(np.isfinite(donor), axis=0)
                    can_fill = still_bad & donor_valid
                    if can_fill.any():
                        for b in range(n_bands):
                            arr[b] = np.where(can_fill, donor[b], arr[b])
                        n_filled = can_fill.sum()
                        filled_count += n_filled
            
            # Strategy 3: fill from ANY valid season across both years
            still_bad = np.any(arr == NODATA, axis=0)
            if still_bad.any():
                fill_sum = np.zeros((n_bands, H, W), dtype=np.float64)
                fill_count = np.zeros((H, W), dtype=np.int32)
                for k, other_arr in s2_data.items():
                    if k == key:
                        continue
                    other_valid = np.all(other_arr != NODATA, axis=0) & np.all(np.isfinite(other_arr), axis=0)
                    can_contribute = still_bad & other_valid
                    for b in range(n_bands):
                        fill_sum[b] += np.where(can_contribute, other_arr[b], 0)
                    fill_count += can_contribute.astype(np.int32)
                
                can_fill = still_bad & (fill_count > 0)
                if can_fill.any():
                    for b in range(n_bands):
                        arr[b] = np.where(
                            can_fill,
                            (fill_sum[b] / np.maximum(fill_count, 1)).astype(np.float32),
                            arr[b]
                        )
                    n_filled = can_fill.sum()
                    filled_count += n_filled
            
            # Strategy 4: nearest-neighbor spatial fill (fast, O(n))
            # Copies band values from closest valid pixel
            still_bad = np.any(arr == NODATA, axis=0)
            if still_bad.any():
                from scipy.ndimage import distance_transform_edt
                valid_mask = ~still_bad
                _, nearest_idx = distance_transform_edt(
                    ~valid_mask, return_distances=True, return_indices=True
                )
                for b in range(n_bands):
                    arr[b][still_bad] = arr[b][
                        nearest_idx[0][still_bad], nearest_idx[1][still_bad]
                    ]
                n_spatial = still_bad.sum()
                filled_count += n_spatial
            
            remaining = np.any(arr == NODATA, axis=0).sum()
            total_filled += filled_count
            print(f"      Filled {filled_count:,}, remaining bad: {remaining:,}")
    
    return total_filled


def compute_indices(s2):
    """Compute spectral indices from 10-band S2 array."""
    B04 = s2[2]; B05 = s2[3]; B06 = s2[4]; B08 = s2[6]
    B8A = s2[7]; B11 = s2[8]; B12 = s2[9]; B03 = s2[1]
    eps = 1e-10
    return {
        "NDVI": (B08 - B04) / (B08 + B04 + eps),
        "NDWI": (B03 - B08) / (B03 + B08 + eps),
        "NDBI": (B11 - B08) / (B11 + B08 + eps),
        "NDMI": (B08 - B11) / (B08 + B11 + eps),
        "NBR":  (B08 - B12) / (B08 + B12 + eps),
        "BSI":  ((B11 + B04) - (B08 + B03)) / ((B11 + B04) + (B08 + B03) + eps),
        "EVI2": 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1.0 + eps),
        "NDRE1": (B08 - B05) / (B08 + B05 + eps),
        "NDRE2": (B08 - B06) / (B08 + B06 + eps),
    }


def build_features_with_filling(year1, year2):
    """Build feature matrix with S2 filling for ALL pixels (zero empty output)."""
    print(f"[{ts()}] Building features for {year1}+{year2} WITH filling...")

    # Load all S2 rasters
    s2_data = {}
    sar_data = {}
    for year in [year1, year2]:
        for season in SEASONS:
            s2_path = RAW_DIR / f"sentinel2_nuremberg_dashboard_{year}_{season}.tif"
            sar_path = RAW_DIR / f"sentinel1_nuremberg_dashboard_{year}_{season}.tif"
            if not s2_path.exists() or not sar_path.exists():
                print(f"  WARNING: Missing {s2_path.name} or {sar_path.name}")
                return None
            s2_data[(year, season)] = load_raster(s2_path, bands=list(range(1, 11)))
            sar_data[(year, season)] = load_raster(sar_path)
            s2_data[(year, season)][s2_data[(year, season)] == NODATA] = NODATA
            sar_data[(year, season)][sar_data[(year, season)] == NODATA] = np.nan

    # Count original bad pixels
    original_bad = np.zeros((ANCHOR_H, ANCHOR_W), dtype=bool)
    for key, arr in s2_data.items():
        original_bad |= np.any(arr == NODATA, axis=0)
    print(f"  Original bad S2 pixels: {original_bad.sum():,} / {ANCHOR_H * ANCHOR_W:,}")

    # Apply S2 filling (strategies 1-4)
    year_seasons = list(s2_data.keys())
    n_filled = fill_s2_nodata(s2_data, year_seasons)
    print(f"  Total S2 pixels filled: {n_filled:,}")

    # Final safety: nearest-neighbor for any stragglers (should be 0 after strategy 4)
    from scipy.ndimage import distance_transform_edt
    for key in s2_data:
        arr = s2_data[key]
        still_bad = np.any(arr == NODATA, axis=0)
        if still_bad.any():
            print(f"    Final nearest-neighbor fill for {key}: {still_bad.sum():,} pixels")
            valid_mask = ~still_bad
            _, nearest_idx = distance_transform_edt(~valid_mask, return_distances=True, return_indices=True)
            for b in range(arr.shape[0]):
                arr[b][still_bad] = arr[b][nearest_idx[0][still_bad], nearest_idx[1][still_bad]]
        arr[~np.isfinite(arr)] = 0.0
    for key in sar_data:
        sar_data[key] = np.nan_to_num(sar_data[key], nan=0.0)

    print(f"  ALL pixels will get predictions (zero empty)")

    # Compute indices
    data = {}
    for key in s2_data:
        data[key] = {
            "s2": s2_data[key],
            "sar": sar_data[key],
            "indices": compute_indices(s2_data[key]),
        }

    # Build features for ALL pixels (flatten entire H*W grid)
    total_px = ANCHOR_H * ANCHOR_W
    flat = slice(None)  # select all pixels

    features = []
    for year in [year1, year2]:
        for season in SEASONS:
            d = data[(year, season)]
            for i in range(10):
                features.append(d["s2"][i].ravel())
            for idx_name in INDEX_NAMES:
                features.append(d["indices"][idx_name].ravel())
            vv = d["sar"][0].ravel()
            vh = d["sar"][1].ravel()
            features.append(vv)
            features.append(vh)
            features.append(vv / (vh + 1e-10))

    # Intra-annual diffs
    for year in [year1, year2]:
        for idx_name in INDEX_NAMES:
            sp = data[(year, "spring")]["indices"][idx_name].ravel()
            sm = data[(year, "summer")]["indices"][idx_name].ravel()
            features.append(sm - sp)
        for idx_name in INDEX_NAMES:
            sm = data[(year, "summer")]["indices"][idx_name].ravel()
            au = data[(year, "autumn")]["indices"][idx_name].ravel()
            features.append(au - sm)

    # Interannual diffs
    for season in SEASONS:
        for idx_name in INDEX_NAMES:
            v1 = data[(year1, season)]["indices"][idx_name].ravel()
            v2 = data[(year2, season)]["indices"][idx_name].ravel()
            features.append(v2 - v1)

    # Index ranges
    for year in [year1, year2]:
        for idx_name in ["NDVI", "NDWI", "EVI2", "BSI"]:
            vals = np.stack([data[(year, s)]["indices"][idx_name].ravel() for s in SEASONS])
            features.append(vals.max(axis=0) - vals.min(axis=0))

    # SAR diffs
    for year in [year1, year2]:
        for sar_idx in range(2):
            sp = data[(year, "spring")]["sar"][sar_idx].ravel()
            sm = data[(year, "summer")]["sar"][sar_idx].ravel()
            au = data[(year, "autumn")]["sar"][sar_idx].ravel()
            features.append(sm - sp)
            features.append(au - sm)

    # SAR interannual diffs
    for season in SEASONS:
        for sar_idx in range(2):
            v1 = data[(year1, season)]["sar"][sar_idx].ravel()
            v2 = data[(year2, season)]["sar"][sar_idx].ravel()
            features.append(v2 - v1)

    X = np.column_stack(features).astype(np.float32)
    print(f"  Feature matrix: {X.shape} (ALL pixels)")

    del data, s2_data, sar_data, features
    gc.collect()

    return X


def generate_bins(pred_map, year_label, n_classes=6):
    """Write multi-resolution .bin files for predictions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for res in range(1, 11):
        if res == 1:
            out = pred_map
        else:
            rh = ANCHOR_H // res
            rw = ANCHOR_W // res
            cropped = pred_map[:rh * res, :rw * res]
            blocks = cropped.reshape(rh, res, rw, res)
            out = np.full((rh, rw), 255, dtype=np.uint8)
            for cls in range(n_classes):
                counts = np.sum(blocks == cls, axis=(1, 3))
                if cls == 0:
                    best_counts = counts.copy()
                    out[:] = 0
                else:
                    better = counts > best_counts
                    out[better] = cls
                    best_counts = np.maximum(best_counts, counts)
            valid_counts = np.sum(blocks != 255, axis=(1, 3))
            out[valid_counts == 0] = 255
        fname = f"nuremberg_pred_{year_label}_res{res}.bin"
        (OUTPUT_DIR / fname).write_bytes(out.tobytes())
        print(f"  Wrote {fname} ({out.shape[0]}x{out.shape[1]})")


def main():
    print("=" * 60)
    print("Nuremberg Bins — FULL COVERAGE (zero empty pixels)")
    print("=" * 60)

    print(f"\n[{ts()}] Loading CatBoost model (GPU)...")
    model = CatBoostClassifier(task_type='GPU')
    model.load_model(str(MODEL_PATH))
    print(f"  Features: {len(model.feature_names_)}")
    print(f"  Classes: {model.classes_}")

    # Only process pre-2022 year pairs
    year_pairs = [
        (2019, 2020),
        (2020, 2021),
    ]

    if len(sys.argv) == 3:
        y1, y2 = int(sys.argv[1]), int(sys.argv[2])
        year_pairs = [(y1, y2)]

    for y1, y2 in year_pairs:
        print(f"\n{'='*60}")
        print(f"  Year pair: {y1} + {y2}")
        print(f"{'='*60}")

        X = build_features_with_filling(y1, y2)
        if X is None:
            print(f"  SKIPPED (missing data)")
            continue

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[{ts()}] Running CatBoost GPU on {X.shape[0]:,} pixels...")
        start = time.time()
        preds_raw = model.predict(X).flatten().astype(int)
        elapsed = time.time() - start
        print(f"  Predicted in {elapsed:.1f}s")

        preds_dashboard = np.array([MODEL_TO_DASHBOARD[p] for p in preds_raw], dtype=np.uint8)
        pred_map = preds_dashboard.reshape(ANCHOR_H, ANCHOR_W)

        empty_count = (pred_map == 255).sum()
        print(f"  Empty pixels in output: {empty_count}")

        # Generate bins for the later year (y2) only.
        # y1's prediction comes from the pair where y1 is the second year.
        generate_bins(pred_map, y2)

        # Save a visualization PNG for comparison
        save_comparison(pred_map, y2)

        del X, preds_raw, preds_dashboard, pred_map
        gc.collect()

    print(f"\n[{ts()}] Done! Bins in {OUTPUT_DIR}")


def save_comparison(pred_map, year_label):
    """Save a side-by-side comparison PNG: original vs filled."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # 6 dashboard classes
    colors = ['#2d6a4f', '#6a994e', '#f4a261', '#e76f51', '#d4a373', '#0096c7']
    labels = ['Tree Cover', 'Grassland', 'Cropland', 'Built-up', 'Bare/Sparse', 'Water']
    cmap = ListedColormap(colors)

    # Load original bin for comparison
    orig_dir = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_dashboard"
    orig_path = orig_dir / f"nuremberg_pred_{year_label}_res1.bin"
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    
    if orig_path.exists():
        orig = np.frombuffer(orig_path.read_bytes(), dtype=np.uint8).reshape(ANCHOR_H, ANCHOR_W)
        orig_rgb = np.full((ANCHOR_H, ANCHOR_W, 3), 0.15, dtype=np.float32)
        for i, color in enumerate(colors):
            r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
            mask = orig == i
            orig_rgb[mask] = [r, g, b]
        orig_empty = (orig == 255).sum()
        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"Original {year_label}\n({orig_empty:,} empty pixels)", fontsize=14)
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Original not found', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title(f"Original {year_label}")
        axes[0].axis('off')

    # Filled version
    filled_rgb = np.full((ANCHOR_H, ANCHOR_W, 3), 0.15, dtype=np.float32)
    for i, color in enumerate(colors):
        r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
        mask = pred_map == i
        filled_rgb[mask] = [r, g, b]
    filled_empty = (pred_map == 255).sum()
    axes[1].imshow(filled_rgb)
    axes[1].set_title(f"Filled {year_label}\n({filled_empty:,} empty pixels)", fontsize=14)
    axes[1].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    legend_elements.append(Patch(facecolor='#262626', label='No Data'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=7, fontsize=11)

    plt.suptitle(f"Nuremberg Predictions {year_label} — Original vs Filled", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    out_path = OUTPUT_DIR / f"comparison_{year_label}.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {out_path}")


if __name__ == "__main__":
    main()
