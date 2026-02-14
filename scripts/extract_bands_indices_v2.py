"""
Lean feature extraction: band statistics + spectral indices only.

Produces a single merged parquet with per-season band stats + all 15 indices
(original 9 + EVI2, MNDWI, GNDVI, NDTI, IRECI, CRI1), plus Tasseled Cap
and basic spatial features. NO texture features (GLCM, Gabor, LBP, HOG,
morphological profiles, semivariogram).

Output: data/processed/v2/features_bands_indices_v2.parquet

Usage:
    python scripts/extract_bands_indices_v2.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG, PROJECT_ROOT  # noqa: E402
from src.features.extract_features import (  # noqa: E402
    SENTINEL_YEARS,
    SEASON_ORDER,
    BAND_NAMES,
    V2_DIR,
    GRID_PATH,
    MIN_VALID_FRAC,
    load_sentinel_v2,
    compute_grid_shape,
    extract_cell_patch,
    cell_valid_fraction,
    detect_reflectance_scale,
    _normalize_reflectance,
    band_statistics,
    spectral_indices,
    tasseled_cap,
    spatial_simple,
)

import geopandas as gpd  # noqa: E402

OUT_PATH = os.path.join(V2_DIR, "features_bands_indices_v2.parquet")


def extract_lean(patch_ref: np.ndarray) -> dict:
    """Band stats + spectral indices + tasseled cap + spatial. No texture."""
    feats = {}
    feats.update(band_statistics(patch_ref))
    feats.update(spectral_indices(patch_ref))
    feats.update(tasseled_cap(patch_ref))
    feats.update(spatial_simple(patch_ref))
    return feats


def process_composite(year: int, season: str, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extract lean features for one composite."""
    tag = f"{year}_{season}"
    print(f"\n{'='*60}")
    print(f"Processing {tag}")
    print(f"{'='*60}")

    spectral, vf = load_sentinel_v2(year, season)
    n_rows, n_cols = compute_grid_shape(spectral)
    scale = detect_reflectance_scale(spectral)
    print(f"  Shape: {spectral.shape}, scale={scale}")

    records = []
    for row in grid.itertuples(index=False):
        cell_id = int(row.cell_id)
        row_idx = cell_id // n_cols
        col_idx = cell_id % n_cols

        patch = extract_cell_patch(spectral, row_idx, col_idx)
        patch_ref = _normalize_reflectance(patch, scale)

        rec = {"cell_id": cell_id}
        rec.update(extract_lean(patch_ref))
        records.append(rec)

        if (cell_id + 1) % 10000 == 0:
            print(f"  {cell_id + 1}/{len(grid)} cells done")

    df = pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Suffix columns with season_year tag
    feat_cols = [c for c in df.columns if c != "cell_id"]
    df = df.rename(columns={c: f"{c}_{tag}" for c in feat_cols})

    print(f"  {len(df)} cells, {len(feat_cols)} features -> {len(df.columns)} columns")
    return df


def main():
    t0 = time.time()
    print("=" * 60)
    print("LEAN EXTRACTION: band stats + indices (v2)")
    print("=" * 60)

    # Load grid
    print("Loading grid...")
    grid = gpd.read_file(GRID_PATH)
    grid = grid.sort_values("cell_id").reset_index(drop=True)
    print(f"  {len(grid)} cells")

    # Process all composites and merge
    merged = pd.DataFrame({"cell_id": grid["cell_id"].values})

    for year in SENTINEL_YEARS:
        for season in SEASON_ORDER:
            df = process_composite(year, season, grid)
            merged = merged.merge(df, on="cell_id", how="left")

    # Impute NaN with median
    feat_cols = [c for c in merged.columns if c != "cell_id"]
    med = merged[feat_cols].median(numeric_only=True)
    merged[feat_cols] = merged[feat_cols].fillna(med)

    # Save
    os.makedirs(V2_DIR, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"Output: {OUT_PATH}")
    print(f"Shape: {merged.shape[0]} cells x {merged.shape[1]} columns")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Features per composite: {len(feat_cols) // len(SENTINEL_YEARS) // len(SEASON_ORDER)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
