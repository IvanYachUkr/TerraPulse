"""
Production Feature Extraction (Rust-accelerated).

Extracts Sentinel-2 features using the optimized Rust extension.
Uses extract_all_seasons for batch processing (single LBP LUT build,
better memory reuse).

Output: one parquet with columns:
    {feature}_{year}_{season}  for each of 224 base features x N year-seasons
    + cell_id

No deltas are computed — neither the final MLP nor the final tree use them.

Can be imported for dynamic inference or run as a script for batch processing.
"""

import os
import sys
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

GRID_PX = 10


def load_sentinel_raster(path: str):
    """Load Sentinel-2 raster, returning spectral bands as float32.

    The valid_fraction band (last band) is discarded — the Rust extractor
    handles NaN-awareness internally per cell.
    """
    with rasterio.open(path) as ds:
        data = ds.read()
        nodata = ds.nodata

    spectral = data[:len(BAND_NAMES)].astype(np.float32)

    if nodata is not None:
        spectral = np.where(spectral == nodata, np.nan, spectral)

    return spectral


def detect_scale(spectral: np.ndarray) -> float:
    """Detect if reflectance is 0..10000 or 0..1 using NIR band p95."""
    nir = spectral[6]  # B08
    finite = nir[np.isfinite(nir)]
    if len(finite) == 0:
        return 1.0
    p95 = np.percentile(finite, 95)
    return 10000.0 if p95 > 2.0 else 1.0


def extract_features_pipeline(rasters_by_key: dict) -> pd.DataFrame:
    """
    Extract features for a set of rasters using batch Rust extraction.

    Args:
        rasters_by_key: Dict of {(year, season): spectral_array}
                        spectral_array shape: (n_bands, H, W) float32

    Returns:
        pd.DataFrame with columns: cell_id + {feature}_{year}_{season} for all
        year-season combinations. NaN/Inf values are median-imputed.
    """
    base_feature_names = terrapulse_features.feature_names()
    n_base = len(base_feature_names)

    # Sort keys for deterministic ordering
    keys = sorted(rasters_by_key.keys())
    n_seasons = len(keys)

    # Validate grid consistency and detect scale from first raster
    first_spec = rasters_by_key[keys[0]]
    _, H, W = first_spec.shape
    n_rows = H // GRID_PX
    n_cols = W // GRID_PX
    n_cells = n_rows * n_cols
    scale = detect_scale(first_spec)

    # Build spectral list in sorted key order
    spectral_list = []
    for key in keys:
        spec = rasters_by_key[key]
        if spec.shape[1] != H or spec.shape[2] != W:
            raise ValueError(
                f"Grid mismatch: {key} has shape {spec.shape}, "
                f"expected (*, {H}, {W})"
            )
        # Ensure contiguous C-order for Rust
        spectral_list.append(np.ascontiguousarray(spec))

    # Call Rust batch extractor
    # Output layout: for each cell, all seasons' features are contiguous
    # [cell0_s0_f0..f223, cell0_s1_f0..f223, ..., cell1_s0_f0..f223, ...]
    flat = terrapulse_features.extract_all_seasons(
        spectral_list, n_rows, n_cols, scale
    )

    # Reshape: (n_cells, n_seasons, n_base_features)
    feats_3d = flat.reshape(n_cells, n_seasons, n_base)

    # Build column names
    col_names = []
    for si, (year, season) in enumerate(keys):
        for fname in base_feature_names:
            col_names.append(f"{fname}_{year}_{season}")

    # Flatten to (n_cells, n_seasons * n_base_features)
    feats_2d = feats_3d.reshape(n_cells, -1)

    df = pd.DataFrame(feats_2d, columns=col_names)
    df.insert(0, "cell_id", np.arange(n_cells))

    # Impute NaN/Inf for production safety
    feat_cols = [c for c in df.columns if c != "cell_id"]
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    medians = df[feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(medians).fillna(0.0)

    return df


def main():
    """Batch extraction for all available year-season rasters."""
    print("=" * 60)
    print("Production Feature Extraction (Rust)")
    print("=" * 60)

    print(f"Base features per cell: {terrapulse_features.n_features_per_cell()}")
    print(f"Years: {SENTINEL_YEARS}, Seasons: {SEASON_ORDER}")

    data = {}
    for year in SENTINEL_YEARS:
        for season in SEASON_ORDER:
            path = os.path.join(
                RAW_DIR, f"sentinel2_nuremberg_{year}_{season}.tif"
            )
            if os.path.exists(path):
                print(f"  Loading {year} {season}...")
                data[(year, season)] = load_sentinel_raster(path)
            else:
                print(f"  Warning: Missing {path}")

    if not data:
        print("No data found.")
        return

    print(f"\nExtracting features for {len(data)} rasters...")
    df = extract_features_pipeline(data)

    out_path = os.path.join(PROCESSED_DIR, "features_rust_production.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
