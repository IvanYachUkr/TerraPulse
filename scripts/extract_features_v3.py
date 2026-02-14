#!/usr/bin/env python3
"""
Feature Extraction v3 — Raster-native vectorized.

ZERO cell-by-cell Python loops. All ops on full raster arrays,
then reshaped to per-cell aggregates.

Target runtime: ~2-5s per season, <30s total for all 6 seasons.
(vs minutes per season with the cell-loop v1 extractor)

Features per season (224):
  Band statistics:     10 bands × 8 stats = 80
  Spectral indices:    15 indices × 5 stats = 75
  Tasseled Cap:        3 components × 2 stats = 6
  Spatial descriptors: 8 (edges, Laplacian, Moran's I, NDVI spread)
  Multi-band LBP:      5 targets × 11 features = 55

Total: 224 × 6 seasons = 1,344 features

Output: PROCESSED_V2_DIR / features_v3.parquet
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import local_binary_pattern

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import PROCESSED_V2_DIR
from src.features.extract_features import (
    load_sentinel_v2,
    detect_reflectance_scale,
    BAND_NAMES,
    BAND_INDEX,
    SENTINEL_YEARS,
    SEASON_ORDER,
    MIN_VALID_FRAC,
    TC_BRIGHTNESS,
    TC_GREENNESS,
    TC_WETNESS,
)

EPS = 1e-10
GRID_PX = 10  # pixels per cell side

# Band indices (avoid dict lookups in hot paths)
_B02 = BAND_INDEX["B02"]
_B03 = BAND_INDEX["B03"]
_B04 = BAND_INDEX["B04"]
_B05 = BAND_INDEX["B05"]
_B06 = BAND_INDEX["B06"]
_B07 = BAND_INDEX["B07"]
_B08 = BAND_INDEX["B08"]
_B11 = BAND_INDEX["B11"]
_B12 = BAND_INDEX["B12"]

# LBP config
LBP_P, LBP_R = 8, 1
LBP_N_BINS = LBP_P + 2  # 10 uniform + non-uniform


# =====================================================================
# Core reshape utilities
# =====================================================================

def _to_cells(arr_2d, nr, nc):
    """Reshape (H, W) → (n_cells, GRID_PX²) contiguous array.

    This is THE key operation: it turns per-pixel rasters into
    per-cell pixel vectors for bulk stat computation.
    """
    gp = GRID_PX
    return (arr_2d
            .reshape(nr, gp, nc, gp)
            .transpose(0, 2, 1, 3)
            .reshape(nr * nc, gp * gp))


def _to_patches(arr_2d, nr, nc):
    """Reshape (H, W) → (n_cells, GRID_PX, GRID_PX) keeping 2D layout.

    Needed for Moran's I where spatial layout within cell matters.
    """
    gp = GRID_PX
    return (arr_2d
            .reshape(nr, gp, nc, gp)
            .transpose(0, 2, 1, 3)
            .reshape(nr * nc, gp, gp))


def _fill_nan(arr_2d):
    """Replace NaN with global mean (for convolution-based features)."""
    if not np.isfinite(arr_2d).all():
        fill = np.nanmean(arr_2d)
        if not np.isfinite(fill):
            fill = 0.0
        return np.where(np.isfinite(arr_2d), arr_2d, fill)
    return arr_2d


def _safe_ratio(a, b):
    """(a - b) / (a + b + EPS) with NaN propagation."""
    m = np.isfinite(a) & np.isfinite(b)
    out = np.full_like(a, np.nan, dtype=np.float32)
    out[m] = (a[m] - b[m]) / (a[m] + b[m] + EPS)
    return out


# =====================================================================
# Bulk statistics — sort-based (>>10x faster than nanpercentile)
# =====================================================================

# For 100 pixels per cell, sorted positions for percentiles
_N_PIX = GRID_PX * GRID_PX  # 100
_Q25_IDX = int(0.25 * (_N_PIX - 1))   # 24
_Q50_IDX = int(0.50 * (_N_PIX - 1))   # 49
_Q75_IDX = int(0.75 * (_N_PIX - 1))   # 74


def _fast_percentiles(cells):
    """Compute q25, median, q75 via sort (not nanpercentile).

    Strategy: replace NaN with +inf, sort, index sorted positions.
    NaN-heavy cells get +inf values which we clip afterwards.
    Much faster than nanpercentile for small fixed-size vectors.

    Input:  (n_cells, 100) float32
    Output: q25, median, q75 — each (n_cells,) float32
    """
    # Replace NaN with very large value so they sort to the end
    filled = np.where(np.isfinite(cells), cells, np.inf)
    sorted_arr = np.sort(filled, axis=1)  # (n_cells, 100)

    q25 = sorted_arr[:, _Q25_IDX]
    med = sorted_arr[:, _Q50_IDX]
    q75 = sorted_arr[:, _Q75_IDX]

    # If a cell had >75% NaN, the percentile positions hit +inf → replace with NaN
    q25 = np.where(np.isfinite(q25), q25, np.nan)
    med = np.where(np.isfinite(med), med, np.nan)
    q75 = np.where(np.isfinite(q75), q75, np.nan)

    return q25, med, q75


def _band_stats(cells):
    """8 stats from (n_cells, n_pixels) → list of (name, array).

    Uses sort-based percentiles instead of nanpercentile for speed.
    """
    mean = np.nanmean(cells, axis=1)
    std = np.nanstd(cells, axis=1)
    mn = np.nanmin(cells, axis=1)
    mx = np.nanmax(cells, axis=1)
    q25, med, q75 = _fast_percentiles(cells)
    finite_frac = np.isfinite(cells).mean(axis=1)
    return [
        ("mean", mean), ("std", std), ("min", mn), ("max", mx),
        ("median", med), ("q25", q25), ("q75", q75),
        ("finite_frac", finite_frac),
    ]


def _index_stats(cells):
    """5 stats from (n_cells, n_pixels) → list of (name, array)."""
    mean = np.nanmean(cells, axis=1)
    std = np.nanstd(cells, axis=1)
    q25, med, q75 = _fast_percentiles(cells)
    return [
        ("mean", mean), ("std", std),
        ("median", med), ("q25", q25), ("q75", q75),
    ]


# =====================================================================
# LBP: full-raster then vectorized per-cell histograms
# =====================================================================

def _lbp_histograms(img_2d, nr, nc):
    """Compute LBP on full raster, vectorized per-cell histograms.

    Returns: (n_cells, LBP_N_BINS) histograms, (n_cells,) entropy
    """
    gp = GRID_PX
    n_cells = nr * nc
    n_pixels = gp * gp

    lbp = local_binary_pattern(img_2d, LBP_P, LBP_R, method="uniform")
    lbp_int = lbp.astype(np.int32)

    # Reshape to (n_cells, n_pixels)
    lbp_flat = _to_cells(lbp_int, nr, nc)

    # Vectorized histogram: count each bin across all cells simultaneously
    # For 10 bins, 10 broadcasts on (30k, 100) is trivially fast
    hist = np.empty((n_cells, LBP_N_BINS), dtype=np.float32)
    for b in range(LBP_N_BINS):
        hist[:, b] = (lbp_flat == b).sum(axis=1)
    hist /= n_pixels  # density normalization

    # Entropy: -sum(p * log(p))
    entropy = -np.sum(hist * np.log(hist + EPS), axis=1)

    return hist, entropy


# =====================================================================
# Feature extraction for one season
# =====================================================================

def extract_one_season(year, season, nr, nc):
    """Extract all features for one season, fully vectorized.

    Returns: list of (column_name, values_array_shape_n_cells)
    """
    t0 = time.time()
    n_cells = nr * nc

    # ── Load & normalize ──
    spectral, vf = load_sentinel_v2(year, season)
    scale = detect_reflectance_scale(spectral)
    ref = spectral.astype(np.float32)
    if scale != 1.0:
        ref = ref / scale

    features = []  # [(name, array_n_cells), ...]

    # ==================================================================
    # 1. BAND STATISTICS  (10 bands × 8 stats = 80 features)
    # ==================================================================
    t1 = time.time()
    for bi, bname in enumerate(BAND_NAMES):
        cells = _to_cells(ref[bi], nr, nc)
        for stat_name, arr in _band_stats(cells):
            features.append((f"{bname}_{stat_name}", arr.astype(np.float32)))
    dt_bands = time.time() - t1

    # ==================================================================
    # 2. SPECTRAL INDICES  (15 indices × 5 stats = 75 features)
    # ==================================================================
    t2 = time.time()

    # Extract full-raster bands (these are views, no copy)
    blue = ref[_B02]
    green = ref[_B03]
    red = ref[_B04]
    nir = ref[_B08]
    re1 = ref[_B05]
    re2 = ref[_B06]
    re3 = ref[_B07]
    swir1 = ref[_B11]
    swir2 = ref[_B12]

    # Compute all indices on full rasters (vectorized)
    indices = {}
    indices["NDVI"] = _safe_ratio(nir, red)
    indices["NDWI"] = _safe_ratio(green, nir)
    indices["NDBI"] = _safe_ratio(swir1, nir)
    indices["NDMI"] = _safe_ratio(nir, swir1)
    indices["NBR"] = _safe_ratio(nir, swir2)
    indices["NDRE1"] = _safe_ratio(nir, re1)
    indices["NDRE2"] = _safe_ratio(nir, re2)
    indices["MNDWI"] = _safe_ratio(green, swir1)
    indices["GNDVI"] = _safe_ratio(nir, green)
    indices["NDTI"] = _safe_ratio(swir1, swir2)

    # SAVI: 1.5 * (NIR - Red) / (NIR + Red + 0.5 + EPS)
    m_sr = np.isfinite(nir) & np.isfinite(red)
    savi = np.full_like(nir, np.nan, dtype=np.float32)
    savi[m_sr] = 1.5 * (nir[m_sr] - red[m_sr]) / (nir[m_sr] + red[m_sr] + 0.5 + EPS)
    indices["SAVI"] = savi

    # BSI: ((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue)+EPS)
    m_bsi = np.isfinite(swir1) & np.isfinite(red) & np.isfinite(nir) & np.isfinite(blue)
    bsi = np.full_like(nir, np.nan, dtype=np.float32)
    bsi[m_bsi] = ((swir1[m_bsi] + red[m_bsi]) - (nir[m_bsi] + blue[m_bsi])) / \
                 ((swir1[m_bsi] + red[m_bsi]) + (nir[m_bsi] + blue[m_bsi]) + EPS)
    indices["BSI"] = bsi

    # EVI2: 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1.0 + EPS)
    evi2 = np.full_like(nir, np.nan, dtype=np.float32)
    evi2[m_sr] = 2.5 * (nir[m_sr] - red[m_sr]) / (nir[m_sr] + 2.4 * red[m_sr] + 1.0 + EPS)
    indices["EVI2"] = evi2

    # IRECI: (B07 - B04) / (B05 / (B06 + EPS) + EPS)
    m_ir = np.isfinite(re3) & np.isfinite(red) & np.isfinite(re1) & np.isfinite(re2)
    ireci = np.full_like(nir, np.nan, dtype=np.float32)
    denom_ir = re1[m_ir] / (re2[m_ir] + EPS)
    ireci[m_ir] = (re3[m_ir] - red[m_ir]) / (denom_ir + EPS)
    indices["IRECI"] = ireci

    # CRI1: (1/Green) - (1/RE1)
    m_cr = np.isfinite(green) & np.isfinite(re1) & (green > EPS) & (re1 > EPS)
    cri1 = np.full_like(nir, np.nan, dtype=np.float32)
    cri1[m_cr] = (1.0 / green[m_cr]) - (1.0 / re1[m_cr])
    indices["CRI1"] = cri1

    # Compute stats for each index
    for idx_name, idx_arr in indices.items():
        cells = _to_cells(idx_arr, nr, nc)
        for stat_name, arr in _index_stats(cells):
            features.append((f"{idx_name}_{stat_name}", arr.astype(np.float32)))

    dt_indices = time.time() - t2

    # ==================================================================
    # 3. TASSELED CAP  (3 × 2 = 6 features)
    # ==================================================================
    t3 = time.time()

    # Vectorized: reshape (10, H, W) → (H*W, 10), matmul, reshape back
    n_bands = ref.shape[0]
    H, W = ref.shape[1], ref.shape[2]
    pixels = ref.reshape(n_bands, H * W).T  # (H*W, 10)
    m_finite = np.all(np.isfinite(pixels), axis=1)

    for tc_name, tc_coeff in [("TC_bright", TC_BRIGHTNESS),
                               ("TC_green", TC_GREENNESS),
                               ("TC_wet", TC_WETNESS)]:
        tc_vals = np.full(H * W, np.nan, dtype=np.float32)
        if m_finite.any():
            tc_vals[m_finite] = pixels[m_finite] @ tc_coeff
        tc_2d = tc_vals.reshape(H, W)
        cells = _to_cells(tc_2d, nr, nc)
        features.append((f"{tc_name}_mean", np.nanmean(cells, axis=1).astype(np.float32)))
        features.append((f"{tc_name}_std", np.nanstd(cells, axis=1).astype(np.float32)))

    dt_tc = time.time() - t3

    # ==================================================================
    # 4. SPATIAL DESCRIPTORS  (8 features)
    # ==================================================================
    t4 = time.time()

    nir_clean = _fill_nan(np.clip(nir, 0.0, 1.0))
    red_clean = _fill_nan(np.clip(red, 0.0, 1.0))

    # Sobel edges on full raster
    sobel_x = ndimage.sobel(nir_clean, axis=1)
    sobel_y = ndimage.sobel(nir_clean, axis=0)
    edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    edge_cells = _to_cells(edge, nr, nc)
    features.append(("edge_mean", edge_cells.mean(axis=1).astype(np.float32)))
    features.append(("edge_std", edge_cells.std(axis=1).astype(np.float32)))
    features.append(("edge_max", edge_cells.max(axis=1).astype(np.float32)))

    # Laplacian
    lap = ndimage.laplace(nir_clean)
    lap_cells = _to_cells(lap, nr, nc)
    features.append(("lap_abs_mean", np.abs(lap_cells).mean(axis=1).astype(np.float32)))
    features.append(("lap_std", lap_cells.std(axis=1).astype(np.float32)))

    # Moran's I — FULLY VECTORIZED over all cells
    patches = _to_patches(nir_clean, nr, nc)  # (n_cells, 10, 10)
    cell_mean = patches.mean(axis=(-2, -1), keepdims=True)
    z = patches - cell_mean
    denom = (z ** 2).sum(axis=(-2, -1))  # (n_cells,)
    h_sum = (z[:, :, :-1] * z[:, :, 1:]).sum(axis=(-2, -1))
    v_sum = (z[:, :-1, :] * z[:, 1:, :]).sum(axis=(-2, -1))
    n_px = GRID_PX * GRID_PX
    W_pairs = GRID_PX * (GRID_PX - 1) * 2
    morans = np.where(denom > 1e-10,
                      (n_px / W_pairs) * (h_sum + v_sum) / denom,
                      0.0)
    features.append(("morans_I_NIR", morans.astype(np.float32)))

    # NDVI range & IQR
    ndvi_arr = indices["NDVI"]
    ndvi_cells = _to_cells(ndvi_arr, nr, nc)
    ndvi_min = np.nanmin(ndvi_cells, axis=1)
    ndvi_max = np.nanmax(ndvi_cells, axis=1)
    ndvi_q25, _, ndvi_q75 = _fast_percentiles(ndvi_cells)
    features.append(("NDVI_range", (ndvi_max - ndvi_min).astype(np.float32)))
    features.append(("NDVI_iqr", (ndvi_q75 - ndvi_q25).astype(np.float32)))

    dt_spatial = time.time() - t4

    # ==================================================================
    # 5. MULTI-BAND LBP  (5 bands × 11 = 55 features)
    # ==================================================================
    t5 = time.time()

    # Prepare clean images for LBP
    ndvi_img = np.clip((ndvi_arr + 1.0) / 2.0, 0.0, 1.0)
    ndvi_img = _fill_nan(ndvi_img)

    evi2_clean = _fill_nan(np.clip((evi2 + 0.5) / 1.5, 0.0, 1.0))
    swir1_clean = _fill_nan(np.clip(swir1, 0.0, 1.0))
    ndti_img = _fill_nan(np.clip((indices["NDTI"] + 1.0) / 2.0, 0.0, 1.0))

    lbp_targets = [
        ("NIR", nir_clean),
        ("NDVI", ndvi_img),
        ("EVI2", evi2_clean),
        ("SWIR1", swir1_clean),
        ("NDTI", ndti_img),
    ]

    for band_name, img in lbp_targets:
        hist, entropy = _lbp_histograms(img, nr, nc)
        for b in range(LBP_N_BINS):
            features.append((f"LBP_{band_name}_u{LBP_P}_{b}", hist[:, b]))
        features.append((f"LBP_{band_name}_entropy", entropy.astype(np.float32)))

    dt_lbp = time.time() - t5

    # ==================================================================
    # Control columns: valid_fraction, low_valid_fraction
    # ==================================================================
    vf_cells = _to_cells(
        np.where(np.isfinite(vf), vf, 0.0).astype(np.float32), nr, nc)
    valid_frac = vf_cells.mean(axis=1)
    low_vf = (valid_frac < MIN_VALID_FRAC).astype(np.float32)

    elapsed = time.time() - t0
    print(f"  {year}_{season}: {elapsed:.2f}s "
          f"(bands={dt_bands:.2f} idx={dt_indices:.2f} "
          f"tc={dt_tc:.2f} spatial={dt_spatial:.2f} lbp={dt_lbp:.2f})",
          flush=True)

    return features, valid_frac, low_vf


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature Extraction v3 — raster-native vectorized")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract first season only for testing")
    args = parser.parse_args()

    print("=" * 60)
    print("Feature Extraction v3 — Raster-Native Vectorized")
    print("  Band stats:     10 × 8 = 80 features/season")
    print("  Spectral idx:   15 × 5 = 75 features/season")
    print("  Tasseled Cap:    3 × 2 =  6 features/season")
    print("  Spatial:                    8 features/season")
    print("  Multi-band LBP:  5 × 11 = 55 features/season")
    print("  " + "-" * 45)
    print("  Total:                   224 features/season")
    print("  Seasons:                   6")
    print("  Grand total:           1,344 features")
    print("=" * 60)

    # Determine grid dimensions from first composite
    from src.features.extract_features import compute_grid_shape
    spectral_probe, _ = load_sentinel_v2(SENTINEL_YEARS[0], SEASON_ORDER[0])
    nr, nc = compute_grid_shape(spectral_probe)
    n_cells = nr * nc
    del spectral_probe
    print(f"  Grid: {nr} × {nc} = {n_cells} cells "
          f"({nr * GRID_PX} × {nc * GRID_PX} pixels)", flush=True)

    jobs = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]
    if args.dry_run:
        jobs = jobs[:1]
        print(f"  DRY RUN: only {jobs[0]}")

    total_t0 = time.time()
    all_season_dfs = []

    for year, season in jobs:
        suffix = f"{year}_{season}"
        features, valid_frac, low_vf = extract_one_season(year, season, nr, nc)

        # Build DataFrame for this season
        data = {"cell_id": np.arange(n_cells, dtype=np.int32)}
        for name, arr in features:
            data[f"{name}_{suffix}"] = arr

        # Control columns (only from first season, same for all)
        if len(all_season_dfs) == 0:
            data["valid_fraction"] = valid_frac
            data["low_valid_fraction"] = low_vf

        df = pd.DataFrame(data)
        df = df.replace([np.inf, -np.inf], np.nan)
        all_season_dfs.append(df)

    # Merge all seasons on cell_id
    print("\nMerging seasons...", flush=True)
    merged = all_season_dfs[0]
    for df in all_season_dfs[1:]:
        merged = merged.merge(df, on="cell_id", how="outer")

    # Impute NaN with column median
    feat_cols = [c for c in merged.columns
                 if c not in ("cell_id", "valid_fraction", "low_valid_fraction")]
    nan_count = 0
    for c in feat_cols:
        n = merged[c].isna().sum()
        if n > 0:
            nan_count += n
            med = merged[c].median()
            merged[c] = merged[c].fillna(med if np.isfinite(med) else 0.0)

    if nan_count > 0:
        print(f"  Imputed {nan_count} NaN values (median)", flush=True)

    # Save
    out_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    merged.to_parquet(out_path, index=False)

    total_elapsed = time.time() - total_t0
    n_feats = len(feat_cols)
    size_mb = os.path.getsize(out_path) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  Saved:    {out_path}")
    print(f"  Shape:    {merged.shape[0]} cells × {merged.shape[1]} columns")
    print(f"  Features: {n_feats}")
    print(f"  Size:     {size_mb:.1f} MB")
    print(f"  Time:     {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
