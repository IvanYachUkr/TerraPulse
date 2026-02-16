"""
Production Feature Extraction (Rust-accelerated).

Handles extraction of Sentinel-2 features using the optimized Rust extension,
followed by calculation of Year-Over-Year (YoY) and Seasonal deltas.

Can be imported for dynamic inference or run as a script for batch processing.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
import terrapulse_features

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG, PROJECT_ROOT

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "v2")

SENTINEL_YEARS = CFG["sentinel2"]["years"]
SEASON_ORDER = CFG["sentinel2"]["season_order"]
BAND_NAMES = CFG["sentinel2"]["bands"]
MIN_VALID_FRAC = float(CFG["quality"]["min_valid_fraction"])

GRID_PX = 10

def load_sentinel_raster(path: str):
    """Load Sentinel-2 raster (bands + valid_fraction)."""
    with rasterio.open(path) as ds:
        data = ds.read()
        nodata = ds.nodata

    spectral = data[:len(BAND_NAMES)].astype(np.float32)
    vf = data[len(BAND_NAMES)].astype(np.float32)

    if nodata is not None:
        spectral = np.where(spectral == nodata, np.nan, spectral)
        vf = np.where(vf == nodata, np.nan, vf)

    return spectral, vf

def detect_scale(spectral: np.ndarray) -> float:
    """Detect if reflectance is 0..10000 or 0..1."""
    # Check NIR band (index 6 for B08)
    nir = spectral[6]
    p95 = np.nanpercentile(nir, 95) if np.isfinite(nir).any() else 0
    if p95 > 2.0:
        return 10000.0
    return 1.0

def compute_deltas(merged: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Compute YoY and Seasonal deltas."""

    # 1. YoY Deltas (2021 - 2020)
    # Assumes 2020 and 2021 are present in SENTINEL_YEARS
    yoy_data = {}
    years = sorted(SENTINEL_YEARS)
    if len(years) >= 2:
        y0, y1 = years[:2]
        for season in SEASON_ORDER:
            for feat in feature_cols:
                c_new = f"{feat}_{y1}_{season}"
                c_old = f"{feat}_{y0}_{season}"
                if c_new in merged.columns and c_old in merged.columns:
                    yoy_data[f"delta_yoy_{season}_{feat}"] = merged[c_new] - merged[c_old]

    yoy_df = pd.DataFrame(yoy_data, index=merged.index)

    # 2. Seasonal Deltas (within each year)
    seasonal_data = {}
    for year in SENTINEL_YEARS:
        for i in range(len(SEASON_ORDER)):
            for j in range(i + 1, len(SEASON_ORDER)):
                s_a = SEASON_ORDER[i]
                s_b = SEASON_ORDER[j]
                for feat in feature_cols:
                    c_b = f"{feat}_{year}_{s_b}"
                    c_a = f"{feat}_{year}_{s_a}"
                    if c_b in merged.columns and c_a in merged.columns:
                        seasonal_data[f"delta_{year}_{s_b}_vs_{s_a}_{feat}"] = (
                            merged[c_b] - merged[c_a]
                        )

    seasonal_df = pd.DataFrame(seasonal_data, index=merged.index)

    return pd.concat([merged, yoy_df, seasonal_df], axis=1)

def extract_features_pipeline(rasters_by_key: dict):
    """
    Extract features for a set of rasters.

    Args:
        rasters_by_key: Dict of {(year, season): (spectral, vf)}
                        spectral: (B, H, W) float32
                        vf: (H, W) float32

    Returns:
        pd.DataFrame: Merged features with deltas.
    """
    dfs = []
    base_feature_names = terrapulse_features.feature_names()
    n_cells_total = 0

    keys = sorted(rasters_by_key.keys())

    for year, season in keys:
        spectral, vf = rasters_by_key[(year, season)]
        scale = detect_scale(spectral)

        _, H, W = spectral.shape
        n_rows = H // GRID_PX
        n_cols = W // GRID_PX
        n_cells = n_rows * n_cols

        if n_cells_total == 0:
            n_cells_total = n_cells
        elif n_cells != n_cells_total:
            raise ValueError(f"Grid mismatch: {year}_{season} has {n_cells} cells, expected {n_cells_total}")

        # Call Rust
        # Returns flat array: [cell0_feat0, cell0_feat1... cell1_feat0...]
        flat_feats = terrapulse_features.extract_season(spectral, vf, n_rows, n_cols, scale)

        # Reshape to (n_cells, n_feats)
        feats_arr = flat_feats.reshape(n_cells, -1)

        # DataFrame
        col_names = [f"{n}_{year}_{season}" for n in base_feature_names]
        df = pd.DataFrame(feats_arr, columns=col_names)

        # Control columns (only need once really, but kept for compatibility)
        # We can calculate valid_fraction if needed, or rely on Rust/Python to pass it.
        # Rust doesn't return it currently.
        # But we can compute it quickly in Python.

        # Calculate cell valid fraction
        vf_cells = vf.reshape(n_rows, GRID_PX, n_cols, GRID_PX).transpose(0, 2, 1, 3).reshape(n_cells, -1)
        valid_frac = np.nanmean(vf_cells, axis=1)
        low_vf = (valid_frac < MIN_VALID_FRAC).astype(int)

        df["cell_id"] = np.arange(n_cells)
        # We add suffix to control cols to avoid collision during merge?
        # Actually standard compute_deltas keeps cell_id unsuffixed.

        dfs.append(df)

    # Merge all
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="cell_id")

    # Compute Deltas
    # Identify feature columns (excluding cell_id)
    # Actually compute_deltas function expects unsuffixed feature names list
    # But here we have suffixed columns.
    # We can reconstruct the logic.

    final_df = compute_deltas(merged, base_feature_names)

    # Impute NaNs (median)
    # Simple imputation for production safety
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    cols_to_impute = [c for c in final_df.columns if c != "cell_id"]
    final_df[cols_to_impute] = final_df[cols_to_impute].fillna(final_df[cols_to_impute].median())
    final_df[cols_to_impute] = final_df[cols_to_impute].fillna(0.0) # Fallback

    return final_df

def main():
    print("Loading data...")
    data = {}
    for year in SENTINEL_YEARS:
        for season in SEASON_ORDER:
            path = os.path.join(RAW_DIR, f"sentinel2_nuremberg_{year}_{season}.tif")
            if os.path.exists(path):
                print(f"  {year} {season}")
                data[(year, season)] = load_sentinel_raster(path)
            else:
                print(f"  Warning: Missing {path}")

    if not data:
        print("No data found.")
        return

    print("Extracting features...")
    df = extract_features_pipeline(data)

    out_path = os.path.join(PROCESSED_DIR, "features_rust_production.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()
