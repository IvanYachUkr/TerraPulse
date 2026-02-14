#!/usr/bin/env python3
"""
Extract v2 texture features: proper Gabor (complex) + Morph DMP.

Saves to:  PROCESSED_V2_DIR / features_texture_v2.parquet

Gabor v2 (64 feat/season × 6 seasons = 384 total):
  - 2 bands (NIR, NDVI) × 2 σ (1, 2) × 4 θ (0,45,90,135) × 4 stats
  - Stats: magnitude mean, magnitude std, magnitude energy, phase coherence

Morph DMP v2 (64 feat/season × 6 seasons = 384 total):
  - 2 bands (NDVI, NIR) × 3 radii (1,2,3)
  - Per radius: peak mean/std/max, valley mean/std/max, peak_frac, valley_frac = 8
  - Cross-scale DMP: 2 pairs × 2 residual types × 2 stats × 2 bands = 16/season

Total: 128 feat/season × 6 = 768 new features
"""

import argparse
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import disk, opening, closing

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

try:
    from skimage.filters import gabor_kernel
except ImportError:
    from skimage.filters import gabor_kernel

EPS = 1e-10

V2_DIR = os.path.join(PROCESSED_V2_DIR)
GRID_PATH = os.path.join(V2_DIR, "grid.gpkg")

CHANGE_THRESHOLD = 0.05  # for "fraction changed" features


# =====================================================================
# Helpers
# =====================================================================

def _norm01(arr):
    """Normalize array to [0, 1] range."""
    mn, mx = float(arr.min()), float(arr.max())
    if (mx - mn) > 1e-10:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)


def _get_nir_ndvi(patch_ref):
    """Extract cleaned NIR and NDVI from a reflectance-normalized patch."""
    nir = patch_ref[BAND_INDEX["B08"]].astype(np.float32)
    red = patch_ref[BAND_INDEX["B04"]].astype(np.float32)

    nir = np.where(np.isfinite(nir), nir,
                   np.nanmean(nir) if np.isfinite(nir).any() else 0.0)
    nir = np.clip(nir, 0.0, 1.0)

    red_c = np.where(np.isfinite(red), red,
                     np.nanmean(red) if np.isfinite(red).any() else 0.0)
    red_c = np.clip(red_c, 0.0, 1.0)

    ndvi = (nir - red_c) / (nir + red_c + EPS)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return nir, ndvi


# =====================================================================
# Gabor v2: proper complex response (magnitude + phase coherence)
# =====================================================================

def gabor_features_v2(patch_ref: np.ndarray) -> dict:
    """
    Proper Gabor using full complex kernel.

    For each (band, sigma, theta):
      - Compute real and imaginary convolution responses
      - magnitude = sqrt(real² + imag²)
      - phase = atan2(imag, real)
      - Stats: mag_mean, mag_std, mag_energy, phase_coherence

    phase_coherence = |mean(e^{j*phase})| = resultant length
      High = organized texture, Low = random pattern
    """
    feats = {}
    nir, ndvi = _get_nir_ndvi(patch_ref)

    bands = {"NIR": _norm01(nir), "NDVI": _norm01(ndvi)}

    for band_name, img in bands.items():
        for sigma in [1.0, 2.0]:
            for theta_deg in [0, 45, 90, 135]:
                theta = np.deg2rad(theta_deg)
                kernel_complex = gabor_kernel(
                    frequency=0.3, theta=theta,
                    sigma_x=sigma, sigma_y=sigma,
                )
                kernel_real = np.real(kernel_complex)
                kernel_imag = np.imag(kernel_complex)

                resp_real = ndimage.convolve(img, kernel_real, mode="reflect")
                resp_imag = ndimage.convolve(img, kernel_imag, mode="reflect")

                # Magnitude (phase-invariant texture energy)
                mag = np.sqrt(resp_real**2 + resp_imag**2)

                # Phase coherence (circular resultant length)
                phase = np.arctan2(resp_imag, resp_real)
                # R = |mean(e^{j*phase})| = sqrt(mean(cos)^2 + mean(sin)^2)
                cos_mean = float(np.mean(np.cos(phase)))
                sin_mean = float(np.mean(np.sin(phase)))
                coherence = np.sqrt(cos_mean**2 + sin_mean**2)

                prefix = f"Gab2_{band_name}_s{int(sigma)}_t{theta_deg}"
                feats[f"{prefix}_mag_mean"] = float(np.mean(mag))
                feats[f"{prefix}_mag_std"] = float(np.std(mag))
                feats[f"{prefix}_mag_energy"] = float(np.mean(mag**2))
                feats[f"{prefix}_phase_coh"] = float(coherence)

    return feats  # 2 bands × 2σ × 4θ × 4 stats = 64 features/season


# =====================================================================
# Morph DMP v2: peak/valley residuals + multi-scale DMP
# =====================================================================

def morph_dmp_features_v2(patch_ref: np.ndarray) -> dict:
    """
    Differential Morphological Profile (DMP) features.

    For each (band, radius):
      peak   = original - opening  (bright structures removed)
      valley = closing - original  (dark structures filled)
      Stats: mean, std, max, fraction > threshold

    Cross-scale DMP (consecutive residual differences):
      DMP_peak_r{r2}_vs_r{r1}  = |peak_r2| - |peak_r1|
      DMP_valley_r{r2}_vs_r{r1} = |valley_r2| - |valley_r1|
      Stats: mean, std
    """
    feats = {}
    nir, ndvi = _get_nir_ndvi(patch_ref)

    # Check validity
    if not (np.isfinite(nir).any() and np.isfinite(ndvi).any()):
        # Return NaN dict
        for band_name in ["NDVI", "NIR"]:
            for r in [1, 2, 3]:
                for res_type in ["peak", "valley"]:
                    for stat in ["mean", "std", "max", "frac"]:
                        feats[f"DMP_{band_name}_{res_type}_r{r}_{stat}"] = np.nan
            for r_pair in ["r2_vs_r1", "r3_vs_r2"]:
                for res_type in ["peak", "valley"]:
                    for stat in ["mean", "std"]:
                        feats[f"DMP_{band_name}_{res_type}_{r_pair}_{stat}"] = np.nan
        return feats

    # Prepare images (morph needs non-negative)
    ndvi_img = (ndvi + 1.0) / 2.0  # shift [-1,1] → [0,1]
    nir_img = nir  # already [0,1]

    bands = {"NDVI": ndvi_img, "NIR": nir_img}
    radii = [1, 2, 3]

    for band_name, img in bands.items():
        residuals = {"peak": {}, "valley": {}}

        for r in radii:
            se = disk(r)
            op = opening(img, se)
            cl = closing(img, se)

            peak = img - op       # bright stuff removed by opening
            valley = cl - img     # dark stuff filled by closing

            residuals["peak"][r] = peak
            residuals["valley"][r] = valley

            prefix = f"DMP_{band_name}"
            feats[f"{prefix}_peak_r{r}_mean"] = float(np.mean(peak))
            feats[f"{prefix}_peak_r{r}_std"] = float(np.std(peak))
            feats[f"{prefix}_peak_r{r}_max"] = float(np.max(peak))
            feats[f"{prefix}_peak_r{r}_frac"] = float(
                np.mean(peak > CHANGE_THRESHOLD))

            feats[f"{prefix}_valley_r{r}_mean"] = float(np.mean(valley))
            feats[f"{prefix}_valley_r{r}_std"] = float(np.std(valley))
            feats[f"{prefix}_valley_r{r}_max"] = float(np.max(valley))
            feats[f"{prefix}_valley_r{r}_frac"] = float(
                np.mean(valley > CHANGE_THRESHOLD))

        # Cross-scale DMP: differences between consecutive scales
        for r1, r2 in [(1, 2), (2, 3)]:
            for res_type in ["peak", "valley"]:
                diff = np.abs(residuals[res_type][r2]) - np.abs(residuals[res_type][r1])
                prefix = f"DMP_{band_name}_{res_type}_r{r2}_vs_r{r1}"
                feats[f"{prefix}_mean"] = float(np.mean(diff))
                feats[f"{prefix}_std"] = float(np.std(diff))

    # Per band: 3 radii × 2 types × 4 stats + 2 pairs × 2 types × 2 stats = 24 + 8 = 32
    # × 2 bands = 64 features/season
    return feats


# =====================================================================
# Main extraction loop
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract first season only for testing")
    args = parser.parse_args()

    print("=" * 60)
    print("Texture v2 Feature Extraction")
    print("  Gabor:  complex magnitude + phase coherence (64/season)")
    print("  Morph:  DMP peak/valley + cross-scale (64/season)")
    print("  Total:  128/season × 6 seasons = 768 features")
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
                # Low quality: generate keys with NaN
                dummy_gabor = gabor_features_v2(patch_ref)
                dummy_morph = morph_dmp_features_v2(patch_ref)
                rec.update({k: np.nan for k in dummy_gabor})
                rec.update({k: np.nan for k in dummy_morph})
            else:
                rec.update(gabor_features_v2(patch_ref))
                rec.update(morph_dmp_features_v2(patch_ref))

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
        print(f"  Done: {len(feat_cols)} features × {len(df)} cells in {elapsed:.1f}s",
              flush=True)

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

    out_path = os.path.join(PROCESSED_V2_DIR, "features_texture_v2.parquet")
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
