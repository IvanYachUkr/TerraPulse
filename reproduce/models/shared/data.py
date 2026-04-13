"""
Data loading utilities for raw-band models.

Handles:
 - Reading raw S2+S1 TIFs for a city
 - Building per-pixel feature cubes (H, W, 72)
 - Loading WorldCover labels
 - Extracting neighborhood patches (1×1, 3×3, 5×5)
 - Memory-mapped dataset management
"""

import gc
import math
import os
import urllib.request

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .config import (
    CITIES_DIR, WC_TILES_DIR,
    S2_BANDS, S1_BANDS, YEARS, SEASONS,
    S2_NODATA, S1_NODATA, WC_CLASS_MAP, N_CLASSES,
    N_RAW_FEATURES, SEED,
    raw_dir,
)


# ── TIF Loading ──────────────────────────────────────────────────────────────

def _load_tif(path, nodata_val):
    """Load a single TIF, replace nodata with NaN."""
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    data[data == nodata_val] = np.nan
    return data


def load_raw_feature_cube(city):
    """
    Load all S2+S1 TIFs for a city and stack into (H, W, 72) array.

    Feature order: for each (year, season):
        10 S2 bands, then 2 S1 bands
    Total: 6 temporal slots × 12 bands = 72

    Returns: (cube, H, W) or (None, 0, 0) if missing data
    """
    rd = raw_dir(city)
    if rd is None:
        return None, 0, 0

    bands_list = []
    ref_shape = None

    for year in YEARS:
        for season in SEASONS:
            tag = f"{city.name}_{year}_{season}"

            # ── Sentinel-2 (10 spectral bands, skip SCL at index 10) ──
            s2_path = os.path.join(rd, f"sentinel2_{tag}.tif")
            s2 = _load_tif(s2_path, S2_NODATA)

            if s2 is not None and s2.shape[0] >= 10:
                if ref_shape is None:
                    ref_shape = (s2.shape[1], s2.shape[2])  # (H, W)
                for bi in range(10):
                    bands_list.append(s2[bi])
            else:
                if ref_shape is None:
                    return None, 0, 0
                for _ in range(10):
                    bands_list.append(np.full(ref_shape, np.nan, dtype=np.float32))

            # ── Sentinel-1 (VV, VH) ──
            s1_path = os.path.join(rd, f"sentinel1_{tag}.tif")
            s1 = _load_tif(s1_path, S1_NODATA)

            if s1 is not None and s1.shape[0] >= 2:
                for bi in range(2):
                    bands_list.append(s1[bi])
            else:
                for _ in range(2):
                    bands_list.append(np.full(ref_shape, np.nan, dtype=np.float32))

    if ref_shape is None or len(bands_list) != N_RAW_FEATURES:
        return None, 0, 0

    H, W = ref_shape
    cube = np.stack(bands_list, axis=-1)  # (H, W, 72)
    del bands_list
    return cube, H, W


# ── WorldCover Labels ────────────────────────────────────────────────────────

def _wc_tiles_for_bbox(bbox):
    """Compute ESA WorldCover 3×3 degree tile IDs covering a bbox."""
    west, south, east, north = bbox
    lat_lo = int(math.floor(south / 3.0)) * 3
    lat_hi = int(math.floor(north / 3.0)) * 3
    lon_lo = int(math.floor(west / 3.0)) * 3
    lon_hi = int(math.floor(east / 3.0)) * 3
    tiles = []
    for lat in range(lat_lo, lat_hi + 1, 3):
        for lon in range(lon_lo, lon_hi + 1, 3):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return tiles


def _download_wc_tile(tile, year=2021):
    """Download a WorldCover tile if not already present."""
    os.makedirs(WC_TILES_DIR, exist_ok=True)
    version = "v100" if year == 2020 else "v200"
    filename = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    path = os.path.join(WC_TILES_DIR, filename)
    if os.path.exists(path):
        return path
    url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
           f"{version}/{year}/map/{filename}")
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        return None


def load_pixel_labels(city, year=2021):
    """
    Load WorldCover per-pixel labels for a city, reprojected to match
    the S2 anchor grid.

    Returns: (H, W) uint8 array with class indices (0-6), or None.
    """
    rd = raw_dir(city)
    if rd is None:
        return None

    # Find an anchor TIF for grid alignment
    anchor_path = None
    for y in YEARS:
        for s in SEASONS:
            p = os.path.join(rd, f"sentinel2_{city.name}_{y}_{s}.tif")
            if os.path.exists(p):
                anchor_path = p
                break
        if anchor_path:
            break
    if anchor_path is None:
        return None

    with rasterio.open(anchor_path) as ref:
        anchor_crs = ref.crs
        anchor_transform = ref.transform
        anchor_w = ref.width
        anchor_h = ref.height

    # Build reprojected WorldCover
    tiles = _wc_tiles_for_bbox(city.bbox)
    dst_array = np.zeros((anchor_h, anchor_w), dtype=np.uint8)
    found_any = False

    for tile in tiles:
        wc_path = _download_wc_tile(tile, year)
        if wc_path is None:
            continue
        found_any = True
        tmp = np.zeros_like(dst_array)
        with rasterio.open(wc_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=tmp,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=anchor_transform, dst_crs=anchor_crs,
                dst_nodata=0, resampling=Resampling.nearest,
            )
        mask = (dst_array == 0) & (tmp > 0)
        dst_array[mask] = tmp[mask]

    if not found_any:
        return None

    # Map ESA codes → 7 classes
    label_array = np.full((anchor_h, anchor_w), 255, dtype=np.uint8)
    for wc_code, our_class in WC_CLASS_MAP.items():
        label_array[dst_array == wc_code] = our_class

    return label_array


# ── Spectral Index Computation ───────────────────────────────────────────────

INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDMI", "NBR", "BSI", "EVI2",
               "NDRE1", "NDRE2"]
# Number of extra features per center pixel for hybrid mode
# 54 indices + 36 seasonal + 27 inter-annual + 8 range + 6 SAR ratio
#   + 8 SAR seasonal + 6 SAR inter-annual = 145
N_HYBRID_EXTRA = 145


def _safe_ratio(a, b, eps=1e-10):
    """Compute (a - b) / (a + b), safe against division by zero."""
    denom = a + b
    mask = np.abs(denom) > eps
    result = np.full_like(a, 0.0, dtype=np.float32)
    result[mask] = (a[mask] - b[mask]) / denom[mask]
    return result


def compute_center_indices(raw_1x1):
    """
    Given (N, 72) raw features for center pixels, compute ALL CatBoost-style
    engineered features: spectral indices, temporal diffs, ranges, SAR ratios
    and SAR temporal diffs.

    Returns: (N, 145) float32 array
    """
    N = raw_1x1.shape[0]

    # ── 1. Spectral indices per time slot (9 indices × 6 slots = 54) ──
    indices_per_slot = []  # list of (N, 9) arrays
    for slot in range(6):
        off = slot * 12
        B02 = raw_1x1[:, off + 0]
        B03 = raw_1x1[:, off + 1]
        B04 = raw_1x1[:, off + 2]
        B05 = raw_1x1[:, off + 3]
        B06 = raw_1x1[:, off + 4]
        B08 = raw_1x1[:, off + 6]
        B11 = raw_1x1[:, off + 8]
        B12 = raw_1x1[:, off + 9]

        ndvi  = _safe_ratio(B08, B04)
        ndwi  = _safe_ratio(B03, B08)
        ndbi  = _safe_ratio(B11, B08)
        ndmi  = _safe_ratio(B08, B11)
        nbr   = _safe_ratio(B08, B12)
        bsi   = _safe_ratio(B11 + B04, B08 + B02)
        denom_evi = B08 + 2.4 * B04 + 1.0
        evi2  = np.where(np.abs(denom_evi) > 1e-10,
                         2.5 * (B08 - B04) / denom_evi, 0.0).astype(np.float32)
        ndre1 = _safe_ratio(B08, B05)
        ndre2 = _safe_ratio(B08, B06)

        slot_indices = np.column_stack([
            ndvi, ndwi, ndbi, ndmi, nbr, bsi, evi2, ndre1, ndre2
        ])  # (N, 9)
        indices_per_slot.append(slot_indices)

    all_indices = np.concatenate(indices_per_slot, axis=1)  # (N, 54)

    # ── 2. Seasonal diffs (within-year): 4 pairs × 9 = 36 ──
    # Slots: 0=2020_spring, 1=2020_summer, 2=2020_autumn,
    #        3=2021_spring, 4=2021_summer, 5=2021_autumn
    seasonal_diffs = np.concatenate([
        indices_per_slot[1] - indices_per_slot[0],  # 2020 spring→summer
        indices_per_slot[2] - indices_per_slot[1],  # 2020 summer→autumn
        indices_per_slot[4] - indices_per_slot[3],  # 2021 spring→summer
        indices_per_slot[5] - indices_per_slot[4],  # 2021 summer→autumn
    ], axis=1)  # (N, 36)

    # ── 3. Inter-annual diffs: 3 seasons × 9 = 27 ──
    interannual_diffs = np.concatenate([
        indices_per_slot[3] - indices_per_slot[0],  # spring 2021-2020
        indices_per_slot[4] - indices_per_slot[1],  # summer 2021-2020
        indices_per_slot[5] - indices_per_slot[2],  # autumn 2021-2020
    ], axis=1)  # (N, 27)

    # ── 4. Seasonal range (autumn - spring): 2 years × 4 indices = 8 ──
    # NDVI=0, NDWI=1, EVI2=6, BSI=5 in the 9-index layout
    range_idx = [0, 1, 6, 5]  # NDVI, NDWI, EVI2, BSI
    range_feats = np.concatenate([
        indices_per_slot[2][:, range_idx] - indices_per_slot[0][:, range_idx],  # 2020
        indices_per_slot[5][:, range_idx] - indices_per_slot[3][:, range_idx],  # 2021
    ], axis=1)  # (N, 8)

    # ── 5. SAR VV/VH ratio per slot: 6 ──
    sar_ratios = []
    for slot in range(6):
        off = slot * 12
        vv = raw_1x1[:, off + 10]
        vh = raw_1x1[:, off + 11]
        ratio = np.where(np.abs(vh) > 1e-10, vv / vh, 0.0).astype(np.float32)
        sar_ratios.append(ratio[:, None])
    sar_ratios = np.concatenate(sar_ratios, axis=1)  # (N, 6)

    # ── 6. SAR temporal diffs (within-year): 4 pairs × 2 bands = 8 ──
    sar_seasonal = []
    for f_slot, t_slot in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        for band_off in [10, 11]:  # VV, VH
            diff = raw_1x1[:, t_slot*12+band_off] - raw_1x1[:, f_slot*12+band_off]
            sar_seasonal.append(diff[:, None])
    sar_seasonal = np.concatenate(sar_seasonal, axis=1)  # (N, 8)

    # ── 7. SAR inter-annual diffs: 3 seasons × 2 bands = 6 ──
    sar_interannual = []
    for s in range(3):
        for band_off in [10, 11]:
            diff = raw_1x1[:, (3+s)*12+band_off] - raw_1x1[:, s*12+band_off]
            sar_interannual.append(diff[:, None])
    sar_interannual = np.concatenate(sar_interannual, axis=1)  # (N, 6)

    # Stack all: 54 + 36 + 27 + 8 + 6 + 8 + 6 = 145
    result = np.concatenate([all_indices, seasonal_diffs, interannual_diffs,
                             range_feats, sar_ratios, sar_seasonal,
                             sar_interannual], axis=1)
    np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return result  # (N, 145)


# ── Pixel Extraction ─────────────────────────────────────────────────────────

def extract_pixels_for_city(city, max_pixels=100_000, pad=2, rng=None):
    """
    Extract raw-band features for sampled pixels from a single city.

    For each sampled pixel, extracts:
      - 1×1: the pixel's 72 raw features
      - 3×3: 72 × 9 = 648 features (flattened neighborhood)  [if pad >= 1]
      - 5×5: 72 × 25 = 1800 features (flattened neighborhood) [if pad >= 2]
      - 3x3_plus: 648 + 123 = 771 (3×3 raw + center indices) [if pad >= 1]

    Args:
        city: CityConfig object
        max_pixels: max pixels to sample
        pad: neighborhood padding (2 for 5×5)
        rng: numpy random state

    Returns:
        dict with keys:
          'feat_1x1': (N, 72)  float32
          'feat_3x3': (N, 648) float32  [or None if pad < 1]
          'feat_3x3_plus': (N, 771) float32  [or None if pad < 1]
          'feat_5x5': (N, 1800) float32 [or None if pad < 2]
          'labels':   (N,)     int32
          'n_pixels': int
        or None if data unavailable
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    cube, H, W = load_raw_feature_cube(city)
    if cube is None:
        return None

    labels = load_pixel_labels(city)
    if labels is None:
        del cube
        return None

    # Ensure shapes match
    if labels.shape != (H, W):
        min_h = min(H, labels.shape[0])
        min_w = min(W, labels.shape[1])
        cube = cube[:min_h, :min_w, :]
        labels = labels[:min_h, :min_w]
        H, W = min_h, min_w

    # Valid pixels: have a class label and <50% NaN features
    valid_label = labels < N_CLASSES
    nan_frac = np.isnan(cube).sum(axis=-1) / cube.shape[-1]
    valid_features = nan_frac < 0.5
    valid = valid_label & valid_features

    # Avoid border pixels (need room for neighborhood patch)
    if pad > 0:
        valid[:pad, :] = False
        valid[-pad:, :] = False
        valid[:, :pad] = False
        valid[:, -pad:] = False

    valid_idx = np.argwhere(valid)  # (N_valid, 2)   [row, col]
    n_valid = len(valid_idx)

    if n_valid == 0:
        del cube, labels
        return None

    # Subsample
    n_sample = min(max_pixels, n_valid)
    chosen = rng.choice(n_valid, n_sample, replace=False)
    coords = valid_idx[chosen]  # (n_sample, 2)

    # Replace NaN with 0 before extraction
    np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # Extract features
    y = labels[coords[:, 0], coords[:, 1]].astype(np.int32)

    # 1×1
    feat_1x1 = cube[coords[:, 0], coords[:, 1], :]  # (N, 72)

    # 3×3 (only if pad >= 1, meaning border pixels are excluded)
    feat_3x3 = None
    feat_3x3_plus = None
    if pad >= 1:
        feat_3x3 = np.empty((n_sample, N_RAW_FEATURES * 9), dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            patch = cube[r - 1:r + 2, c - 1:c + 2, :]  # (3, 3, 72)
            feat_3x3[i] = patch.reshape(-1)

        # Hybrid: 3×3 raw + center-pixel spectral indices
        center_indices = compute_center_indices(feat_1x1)  # (N, 123)
        feat_3x3_plus = np.concatenate([feat_3x3, center_indices], axis=1)
        del center_indices

    # 5×5 (only if pad >= 2, meaning enough border room)
    feat_5x5 = None
    if pad >= 2:
        feat_5x5 = np.empty((n_sample, N_RAW_FEATURES * 25), dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            patch = cube[r - 2:r + 3, c - 2:c + 3, :]  # (5, 5, 72)
            feat_5x5[i] = patch.reshape(-1)

    del cube, labels, valid_idx
    gc.collect()

    return {
        "feat_1x1": feat_1x1,
        "feat_3x3": feat_3x3,
        "feat_3x3_plus": feat_3x3_plus,
        "feat_5x5": feat_5x5,
        "labels": y,
        "n_pixels": n_sample,
        "rows": coords[:, 0].copy(),
        "cols": coords[:, 1].copy(),
    }

