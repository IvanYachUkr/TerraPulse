#!/usr/bin/env python3
"""
Extract multi-band LBP features: LBP histograms on 5 bands/indices.

Saves to:  PROCESSED_V2_DIR / features_lbp_multiband.parquet

Bands:
  1. NIR (B08)  — 10m native, canopy texture
  2. NDVI       — 10m derived, vegetation density patterns
  3. EVI2       — 10m derived, better dynamic range in dense vegetation
  4. NDTI       — 20m derived (SWIR1/SWIR2), cropland vs bare soil
  5. SWIR1 (B11)— 20m native, moisture/built-up texture

Per band: 10 histogram bins + 1 entropy = 11 features
Total:    5 bands × 11 features × 6 seasons = 330 features
"""

import argparse
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import PROCESSED_V2_DIR
from src.features.extract_features import (
    load_sentinel_v2,
    compute_grid_shape,
    extract_cell_patch,
    cell_valid_fraction,
    detect_reflectance_scale,
    _normalize_reflectance,
    BAND_INDEX,
    SENTINEL_YEARS,
    SEASON_ORDER,
    MIN_VALID_FRAC,
)

EPS = 1e-10
V2_DIR = PROCESSED_V2_DIR
GRID_PATH = os.path.join(V2_DIR, "grid.gpkg")

# LBP config
LBP_P = 8  # number of neighbors
LBP_R = 1  # radius
LBP_BINS = LBP_P + 2  # 10 uniform patterns + 1 non-uniform


# =====================================================================
# Helpers
# =====================================================================

def _clean_band(arr):
    """Fill NaN, clip to [0, 1]."""
    fill = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
    out = np.where(np.isfinite(arr), arr, fill)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _compute_lbp_features(img, band_name):
    """Compute LBP histogram + entropy for a single 2D image."""
    feats = {}
    lbp = local_binary_pattern(img, LBP_P, LBP_R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS,
                           range=(0, LBP_BINS), density=True)
    for i in range(LBP_BINS):
        feats[f"LBP_{band_name}_u{LBP_P}_{i}"] = float(hist[i])
    feats[f"LBP_{band_name}_entropy"] = float(
        -np.sum(hist * np.log(hist + EPS)))
    return feats


# =====================================================================
# Multi-band LBP extraction per cell
# =====================================================================

def lbp_multiband_features(patch_ref: np.ndarray) -> dict:
    """Compute LBP histograms on 5 bands/indices."""
    feats = {}

    # 1. NIR (B08) — native 10m
    nir = _clean_band(patch_ref[BAND_INDEX["B08"]])
    feats.update(_compute_lbp_features(nir, "NIR"))

    # 2. Red (B04) — needed for NDVI/EVI2
    red = _clean_band(patch_ref[BAND_INDEX["B04"]])

    # 3. NDVI — derived 10m
    ndvi_raw = (nir - red) / (nir + red + EPS)
    ndvi_01 = np.clip((ndvi_raw + 1.0) / 2.0, 0.0, 1.0)  # shift to [0,1]
    feats.update(_compute_lbp_features(ndvi_01, "NDVI"))

    # 4. EVI2 — derived 10m, less saturated than NDVI
    evi2_raw = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + EPS)
    evi2_01 = np.clip((evi2_raw + 0.5) / 1.5, 0.0, 1.0)  # typical range ~[-0.5, 1]
    feats.update(_compute_lbp_features(evi2_01, "EVI2"))

    # 5. SWIR1 (B11) — 20m upsampled to 10m
    swir1 = _clean_band(patch_ref[BAND_INDEX["B11"]])
    feats.update(_compute_lbp_features(swir1, "SWIR1"))

    # 6. NDTI — derived from SWIR1/SWIR2, 20m effective
    swir2 = _clean_band(patch_ref[BAND_INDEX["B12"]])
    ndti_raw = (swir1 - swir2) / (swir1 + swir2 + EPS)
    ndti_01 = np.clip((ndti_raw + 1.0) / 2.0, 0.0, 1.0)
    feats.update(_compute_lbp_features(ndti_01, "NDTI"))

    return feats  # 5 bands × 11 = 55 features per season


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract first season only for testing")
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Band LBP Feature Extraction")
    print(f"  Bands:    NIR, NDVI, EVI2, SWIR1, NDTI")
    print(f"  LBP:      P={LBP_P}, R={LBP_R}, uniform")
    print(f"  Per band: {LBP_BINS} hist bins + 1 entropy = 11 features")
    print(f"  Per season: 5 bands × 11 = 55 features")
    print(f"  Total:    55 × 6 seasons = 330 features")
    print("=" * 60)

    # Load grid
    print(f"Loading grid from {GRID_PATH}", flush=True)
    grid = gpd.read_file(GRID_PATH)
    grid = grid.sort_values("cell_id").reset_index(drop=True)
    n_cells = len(grid)
    print(f"  {n_cells} cells", flush=True)

    jobs = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]
    if args.dry_run:
        jobs = jobs[:1]
        print(f"  DRY RUN: only {jobs[0]}")

    all_season_dfs = []
    total_t0 = time.time()

    for year, season in jobs:
        suffix = f"{year}_{season}"
        print(f"\n--- {suffix} ---", flush=True)

        spectral, vf = load_sentinel_v2(year, season)
        n_rows, n_cols = compute_grid_shape(spectral)
        scale = detect_reflectance_scale(spectral)
        print(f"  Spectral: {spectral.shape}, scale={scale}", flush=True)

        records = []
        t0 = time.time()

        for row in grid.itertuples(index=False):
            cell_id = int(row.cell_id)
            row_idx = cell_id // n_cols
            col_idx = cell_id % n_cols

            v = cell_valid_fraction(vf, row_idx, col_idx)
            patch = extract_cell_patch(spectral, row_idx, col_idx)
            patch_ref = _normalize_reflectance(patch, scale)

            rec = {"cell_id": cell_id}

            if v < MIN_VALID_FRAC:
                # Low quality: NaN placeholder
                dummy = lbp_multiband_features(patch_ref)
                rec.update({k: np.nan for k in dummy})
            else:
                rec.update(lbp_multiband_features(patch_ref))

            records.append(rec)

            if (cell_id + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (cell_id + 1) / elapsed
                eta_s = (n_cells - cell_id - 1) / rate
                print(f"    {cell_id+1:6d}/{n_cells} ({rate:.0f} cells/s, "
                      f"ETA {eta_s:.0f}s)", flush=True)

        df = pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)

        # Add season suffix
        feat_cols = [c for c in df.columns if c != "cell_id"]
        rename_map = {c: f"{c}_{suffix}" for c in feat_cols}
        df = df.rename(columns=rename_map)
        df = df.replace([np.inf, -np.inf], np.nan)

        elapsed = time.time() - t0
        print(f"  Done: {len(feat_cols)} features × {len(df)} cells "
              f"in {elapsed:.1f}s", flush=True)

        all_season_dfs.append(df)

    # Merge all seasons
    print("\nMerging seasons...", flush=True)
    merged = all_season_dfs[0]
    for df in all_season_dfs[1:]:
        merged = merged.merge(df, on="cell_id", how="outer")

    # Impute NaN with column median
    feat_cols = [c for c in merged.columns if c != "cell_id"]
    nan_count = 0
    for c in feat_cols:
        n = merged[c].isna().sum()
        if n > 0:
            nan_count += n
            med = merged[c].median()
            merged[c] = merged[c].fillna(med if np.isfinite(med) else 0.0)

    if nan_count > 0:
        print(f"  Imputed {nan_count} NaN values (median)", flush=True)

    out_path = os.path.join(PROCESSED_V2_DIR, "features_lbp_multiband.parquet")
    merged.to_parquet(out_path, index=False)

    total_elapsed = time.time() - total_t0
    n_new_feats = len(feat_cols)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  Saved:    {out_path}")
    print(f"  Shape:    {merged.shape[0]} cells × {merged.shape[1]} columns")
    print(f"  Features: {n_new_feats}")
    print(f"  Size:     {size_mb:.1f} MB")
    print(f"  Time:     {total_elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
