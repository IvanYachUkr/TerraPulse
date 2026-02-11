"""
Phase 4: Feature Engineering (v2) - Anchor-correct, join-safe, scale-safe.

Key guarantees:
  - NEVER drops cell_ids. All outputs contain every grid cell.
  - cell_id -> (row_idx, col_idx) computed from cell_id (row-major).
  - Reflectance scaling auto-detected (likely 0..10000) and normalized to 0..1
    for texture features and any method assuming reflectance ranges.
  - OSM features are cached once (same for all composites).

Feature categories:
  Core (~150 features):
    1. Per-band statistics (mean, std, min, max, median, q25, q75, finite_frac)
    2. Spectral indices (NDVI, NDWI, NDBI, NDMI, NBR, SAVI, BSI, NDRE1, NDRE2)
    3. Tasseled Cap (brightness, greenness, wetness)
    4. Spatial (edges, Laplacian, Moran's I, NDVI range/iqr)
  Full (adds ~80 features):
    5. GLCM texture (NIR + NDVI)
    6. Gabor wavelets
    7. LBP texture histogram
    8. HOG features
    9. Morphological profiles on NDVI
   10. Semivariogram features

Usage:
  python src/features/extract_features.py --year 2020 --season spring
  python src/features/extract_features.py --all --feature-set core
  python src/features/extract_features.py --all --feature-set full
"""

import argparse
import os
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor_kernel

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

# -- Configuration via data_config.yml -----------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CFG, PROJECT_ROOT  # noqa: E402

# Paths
RAW_V2_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "v2")
OSM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "osm")
V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
GRID_PATH = os.path.join(V2_DIR, "grid.gpkg")
OSM_CACHE_PATH = os.path.join(V2_DIR, "osm_features.parquet")

GRID_SIZE_M = int(CFG["grid"]["size_m"])
PIXEL_SIZE = float(CFG["grid"]["pixel_size"])
GRID_PX = int(GRID_SIZE_M // PIXEL_SIZE)  # 10 pixels per cell side

# Sentinel-2 band names (in order as stored in our GeoTIFFs)
BAND_NAMES = CFG["sentinel2"]["bands"]
BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}
NODATA = float(CFG["sentinel2"]["nodata"])

# Quality thresholds
MIN_VALID_FRAC = float(CFG["quality"]["min_valid_fraction"])
MAX_DROP_PCT = float(CFG["quality"]["max_drop_pct"])

# Bands at native 20m (upsampled to 10m in our GeoTIFFs)
BANDS_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}

# Seasons/years from config
SENTINEL_YEARS = CFG["sentinel2"]["years"]
SEASON_ORDER = CFG["sentinel2"]["season_order"]

# Tasseled Cap coefficients for Sentinel-2 (Nedkov, 2017)
TC_BRIGHTNESS = np.array(
    [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949, 0.1822, 0.0031, 0.1112, 0.0825],
    dtype=np.float32,
)
TC_GREENNESS = np.array(
    [-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.0469, -0.0322, -0.0015, -0.0693, -0.0180],
    dtype=np.float32,
)
TC_WETNESS = np.array(
    [0.2578, 0.2305, 0.0883, 0.1071, -0.7611, 0.0882, 0.4572, -0.0021, -0.4064, 0.0117],
    dtype=np.float32,
)

EPS = 1e-10


# =============================================================================
# IO
# =============================================================================

def load_sentinel_v2(year: int, season: str):
    """Load v2 Sentinel-2 composite.

    GeoTIFF has 11 bands: 1-10 spectral, 11 = VALID_FRACTION.
    Returns:
      spectral: (10, H, W) float32 with nodata -> NaN
      valid_fraction: (H, W) float32 with nodata -> NaN
    """
    path = os.path.join(RAW_V2_DIR, f"sentinel2_nuremberg_{year}_{season}.tif")
    assert os.path.exists(path), f"Missing composite: {path}"

    with rasterio.open(path) as ds:
        data = ds.read()  # (11, H, W)
        nodata_val = ds.nodata

    spectral = data[:len(BAND_NAMES)].astype(np.float32)
    valid_fraction = data[len(BAND_NAMES)].astype(np.float32)

    if nodata_val is not None:
        spectral = np.where(spectral == nodata_val, np.nan, spectral)
        valid_fraction = np.where(valid_fraction == nodata_val, np.nan, valid_fraction)

    return spectral, valid_fraction


def detect_reflectance_scale(spectral: np.ndarray) -> float:
    """Heuristic: Sentinel-2 reflectance is often scaled (0..10000).
    If 95th percentile of NIR > 2, treat as scaled integers and divide by 10000.
    """
    x = spectral[BAND_INDEX["B08"]]
    p = np.nanpercentile(x, 95) if np.isfinite(x).any() else np.nan
    if np.isfinite(p) and p > 2.0:
        return 10000.0
    return 1.0


# =============================================================================
# Grid indexing (anchor-safe)
# =============================================================================

def compute_grid_shape(spectral: np.ndarray):
    """Compute number of 100m cells in cols/rows from raster shape."""
    H, W = spectral.shape[1], spectral.shape[2]
    assert H % GRID_PX == 0 and W % GRID_PX == 0, (
        f"Raster not divisible by GRID_PX={GRID_PX}: {H}x{W}"
    )
    return H // GRID_PX, W // GRID_PX


def extract_cell_patch(spectral: np.ndarray, row_idx: int, col_idx: int) -> np.ndarray:
    r0 = row_idx * GRID_PX
    c0 = col_idx * GRID_PX
    return spectral[:, r0:r0 + GRID_PX, c0:c0 + GRID_PX]


def cell_valid_fraction(vf: np.ndarray, row_idx: int, col_idx: int) -> float:
    r0 = row_idx * GRID_PX
    c0 = col_idx * GRID_PX
    patch = vf[r0:r0 + GRID_PX, c0:c0 + GRID_PX]
    finite = patch[np.isfinite(patch)]
    return float(np.mean(finite)) if finite.size else 0.0


# =============================================================================
# Helpers
# =============================================================================

def _block_reduce_mean(arr2d: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample 2D array by factor via block nanmean (tolerant to single NaN)."""
    H, W = arr2d.shape
    H2, W2 = (H // factor) * factor, (W // factor) * factor
    x = arr2d[:H2, :W2].reshape(H2 // factor, factor, W2 // factor, factor)
    return np.nanmean(x, axis=(1, 3))


def _finite_stats(x: np.ndarray):
    """Return common stats on finite values, else NaNs."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                    median=np.nan, q25=np.nan, q75=np.nan)
    return dict(
        mean=float(np.mean(v)),
        std=float(np.std(v)),
        min=float(np.min(v)),
        max=float(np.max(v)),
        median=float(np.median(v)),
        q25=float(np.percentile(v, 25)),
        q75=float(np.percentile(v, 75)),
    )


def _normalize_reflectance(patch: np.ndarray, scale: float) -> np.ndarray:
    """Normalize spectral patch to 0..1 reflectance if scale != 1."""
    if scale == 1.0:
        return patch
    return patch / scale


# =============================================================================
# Feature blocks — CORE
# =============================================================================

def band_statistics(patch: np.ndarray) -> dict:
    """Per-band stats. 20m bands block-reduced for variability stats."""
    feats = {}
    for i, name in enumerate(BAND_NAMES):
        band = patch[i].astype(np.float32)

        if name in BANDS_20M and band.shape[0] >= 2 and band.shape[1] >= 2:
            band = _block_reduce_mean(band, factor=2)

        st = _finite_stats(band)
        feats[f"{name}_mean"] = st["mean"]
        feats[f"{name}_std"] = st["std"]
        feats[f"{name}_min"] = st["min"]
        feats[f"{name}_max"] = st["max"]
        feats[f"{name}_median"] = st["median"]
        feats[f"{name}_q25"] = st["q25"]
        feats[f"{name}_q75"] = st["q75"]

        # Missingness per band (important when you stop dropping cells)
        finite = np.isfinite(band).sum()
        feats[f"{name}_finite_frac"] = float(finite) / float(band.size)

    return feats


def spectral_indices(patch: np.ndarray) -> dict:
    """Indices: NDVI, NDWI, NDBI, NDMI, NBR, SAVI, BSI, NDRE1, NDRE2."""
    blue = patch[BAND_INDEX["B02"]]
    green = patch[BAND_INDEX["B03"]]
    red = patch[BAND_INDEX["B04"]]
    nir = patch[BAND_INDEX["B08"]]
    re1 = patch[BAND_INDEX["B05"]]
    re2 = patch[BAND_INDEX["B06"]]
    swir1 = patch[BAND_INDEX["B11"]]
    swir2 = patch[BAND_INDEX["B12"]]

    def safe_ratio(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        out = np.full_like(a, np.nan, dtype=np.float32)
        out[m] = (a[m] - b[m]) / (a[m] + b[m] + EPS)
        return out

    ndvi = safe_ratio(nir, red)
    ndwi = safe_ratio(green, nir)
    ndbi = safe_ratio(swir1, nir)
    ndmi = safe_ratio(nir, swir1)
    nbr = safe_ratio(nir, swir2)
    ndre1 = safe_ratio(nir, re1)
    ndre2 = safe_ratio(nir, re2)

    savi = np.full_like(ndvi, np.nan, dtype=np.float32)
    m = np.isfinite(nir) & np.isfinite(red)
    savi[m] = 1.5 * (nir[m] - red[m]) / (nir[m] + red[m] + 0.5 + EPS)

    bsi = np.full_like(ndvi, np.nan, dtype=np.float32)
    m2 = np.isfinite(swir1) & np.isfinite(red) & np.isfinite(nir) & np.isfinite(blue)
    bsi[m2] = ((swir1[m2] + red[m2]) - (nir[m2] + blue[m2])) / (
        (swir1[m2] + red[m2]) + (nir[m2] + blue[m2]) + EPS
    )

    feats = {}
    for name, arr in [
        ("NDVI", ndvi), ("NDWI", ndwi), ("NDBI", ndbi), ("NDMI", ndmi),
        ("NBR", nbr), ("SAVI", savi), ("BSI", bsi), ("NDRE1", ndre1), ("NDRE2", ndre2),
    ]:
        st = _finite_stats(arr)
        feats[f"{name}_mean"] = st["mean"]
        feats[f"{name}_std"] = st["std"]
        feats[f"{name}_median"] = st["median"]
        feats[f"{name}_q25"] = st["q25"]
        feats[f"{name}_q75"] = st["q75"]

    return feats


def tasseled_cap(patch: np.ndarray) -> dict:
    """Tasseled Cap on per-pixel spectra, then stats."""
    n_bands = patch.shape[0]
    X = patch.reshape(n_bands, -1).T.astype(np.float32)
    m = np.all(np.isfinite(X), axis=1)
    if m.sum() == 0:
        return {k: np.nan for k in [
            "TC_bright_mean", "TC_bright_std",
            "TC_green_mean", "TC_green_std",
            "TC_wet_mean", "TC_wet_std",
        ]}
    Xv = X[m]
    bright = Xv @ TC_BRIGHTNESS
    green = Xv @ TC_GREENNESS
    wet = Xv @ TC_WETNESS
    return {
        "TC_bright_mean": float(np.mean(bright)),
        "TC_bright_std": float(np.std(bright)),
        "TC_green_mean": float(np.mean(green)),
        "TC_green_std": float(np.std(green)),
        "TC_wet_mean": float(np.mean(wet)),
        "TC_wet_std": float(np.std(wet)),
    }


def spatial_simple(patch: np.ndarray) -> dict:
    """Cheap spatial descriptors: edges, Laplacian, Moran's I, NDVI range/iqr."""
    feats = {}
    nir = patch[BAND_INDEX["B08"]].astype(np.float32)
    red = patch[BAND_INDEX["B04"]].astype(np.float32)

    # Fill NaN for convolution-based features
    nir_f = np.where(np.isfinite(nir), nir,
                     np.nanmean(nir) if np.isfinite(nir).any() else 0.0)

    sobel_x = ndimage.sobel(nir_f, axis=1)
    sobel_y = ndimage.sobel(nir_f, axis=0)
    edge = np.sqrt(sobel_x**2 + sobel_y**2)
    feats["edge_mean"] = float(np.mean(edge))
    feats["edge_std"] = float(np.std(edge))
    feats["edge_max"] = float(np.max(edge))

    lap = ndimage.laplace(nir_f)
    feats["lap_abs_mean"] = float(np.mean(np.abs(lap)))
    feats["lap_std"] = float(np.std(lap))

    # Moran's I
    z = nir_f - float(np.mean(nir_f))
    denom = float(np.sum(z**2))
    if denom > 1e-10:
        h_sum = float(np.sum(z[:, :-1] * z[:, 1:]))
        v_sum = float(np.sum(z[:-1, :] * z[1:, :]))
        n = z.size
        W = (z.shape[0] * (z.shape[1] - 1)) + ((z.shape[0] - 1) * z.shape[1])
        feats["morans_I_NIR"] = (n / W) * (h_sum + v_sum) / denom
    else:
        feats["morans_I_NIR"] = 0.0

    # NDVI spread
    m = np.isfinite(nir) & np.isfinite(red)
    if m.any():
        ndvi = (nir[m] - red[m]) / (nir[m] + red[m] + EPS)
        feats["NDVI_range"] = float(np.max(ndvi) - np.min(ndvi))
        feats["NDVI_iqr"] = float(np.percentile(ndvi, 75) - np.percentile(ndvi, 25))
    else:
        feats["NDVI_range"] = np.nan
        feats["NDVI_iqr"] = np.nan

    return feats


# =============================================================================
# Feature blocks — FULL (heavier)
# =============================================================================

def glcm_features(patch_ref: np.ndarray) -> dict:
    """GLCM texture on NIR and NDVI using reflectance-scaled [0,1] values."""
    feats = {}
    nir = patch_ref[BAND_INDEX["B08"]].astype(np.float32)
    red = patch_ref[BAND_INDEX["B04"]].astype(np.float32)
    ndvi = (nir - red) / (nir + red + EPS)

    for name, arr, vmin, vmax in [("NIR", nir, 0.0, 1.0), ("NDVI", ndvi, -1.0, 1.0)]:
        arr_clean = np.where(np.isfinite(arr), arr, (vmin + vmax) / 2)
        clipped = np.clip(arr_clean, vmin, vmax)
        q = ((clipped - vmin) / (vmax - vmin + EPS) * 31).astype(np.uint8)

        glcm = graycomatrix(
            q, distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=32, symmetric=True, normed=True,
        )
        for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
            v = float(np.mean(graycoprops(glcm, prop)))
            feats[f"GLCM_{name}_{prop}"] = v if np.isfinite(v) else np.nan

    return feats


def gabor_features(patch_ref: np.ndarray) -> dict:
    """Gabor features on NIR reflectance [0,1]."""
    feats = {}
    nir = patch_ref[BAND_INDEX["B08"]].astype(np.float32)
    nir = np.where(np.isfinite(nir), nir,
                   np.nanmean(nir) if np.isfinite(nir).any() else 0.0)
    nir = np.clip(nir, 0.0, 1.0)

    mn, mx = float(nir.min()), float(nir.max())
    nirn = (nir - mn) / (mx - mn + EPS) if (mx - mn) > 1e-10 else np.zeros_like(nir)

    for sigma in [1.0, 2.0, 4.0]:
        for theta_deg in [0, 45, 90, 135]:
            theta = np.deg2rad(theta_deg)
            kernel = np.real(gabor_kernel(frequency=0.3, theta=theta,
                                         sigma_x=sigma, sigma_y=sigma))
            resp = ndimage.convolve(nirn, kernel, mode="reflect")
            feats[f"Gabor_s{int(sigma)}_t{theta_deg}_mean"] = float(np.mean(resp))
            feats[f"Gabor_s{int(sigma)}_t{theta_deg}_std"] = float(np.std(resp))
    return feats


def lbp_features(patch_ref: np.ndarray) -> dict:
    """Local Binary Pattern histogram on NIR."""
    from skimage.feature import local_binary_pattern

    feats = {}
    nir = patch_ref[BAND_INDEX["B08"]].astype(np.float32)
    nir = np.where(np.isfinite(nir), nir,
                   np.nanmean(nir) if np.isfinite(nir).any() else 0.0)
    nir = np.clip(nir, 0.0, 1.0)

    P, R = 8, 1
    lbp = local_binary_pattern(nir, P=P, R=R, method="uniform")
    bins = int(P + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    for i in range(bins):
        feats[f"LBP_u8_{i}"] = float(hist[i])
    feats["LBP_entropy"] = float(-np.sum(hist * np.log(hist + EPS)))
    return feats


def hog_features(patch_ref: np.ndarray) -> dict:
    """Tiny HOG on NIR (10x10 is small, so keep config coarse)."""
    from skimage.feature import hog

    feats = {}
    nir = patch_ref[BAND_INDEX["B08"]].astype(np.float32)
    nir = np.where(np.isfinite(nir), nir,
                   np.nanmean(nir) if np.isfinite(nir).any() else 0.0)
    nir = np.clip(nir, 0.0, 1.0)

    vec = hog(
        nir, orientations=8,
        pixels_per_cell=(5, 5), cells_per_block=(1, 1),
        block_norm="L2-Hys", feature_vector=True,
    )
    for i, v in enumerate(vec):
        feats[f"HOG_{i}"] = float(v)
    feats["HOG_mean"] = float(np.mean(vec))
    feats["HOG_std"] = float(np.std(vec))
    return feats


def morph_profile_features(patch_ref: np.ndarray) -> dict:
    """Morphological profiles on NDVI (opening/closing at multiple radii)."""
    from skimage.morphology import disk, opening, closing

    feats = {}
    nir = patch_ref[BAND_INDEX["B08"]].astype(np.float32)
    red = patch_ref[BAND_INDEX["B04"]].astype(np.float32)
    m = np.isfinite(nir) & np.isfinite(red)
    if not m.any():
        return {f"MP_{op}_r{r}_mean": np.nan
                for r in [1, 2, 3] for op in ["open", "close", "grad"]}

    ndvi = np.zeros_like(nir)
    ndvi[m] = (nir[m] - red[m]) / (nir[m] + red[m] + EPS)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Shift to non-negative for morphology
    img = (ndvi + 1.0) / 2.0

    for r in [1, 2, 3]:
        se = disk(r)
        op = opening(img, se)
        cl = closing(img, se)
        feats[f"MP_open_r{r}_mean"] = float(np.mean(op))
        feats[f"MP_close_r{r}_mean"] = float(np.mean(cl))
        feats[f"MP_grad_r{r}_mean"] = float(np.mean(cl - op))
    return feats


def semivariogram_features(patch_ref: np.ndarray) -> dict:
    """Empirical semivariogram on NIR (lags 1..4) + exponential fit."""
    from scipy.optimize import curve_fit

    feats = {}
    nir = patch_ref[BAND_INDEX["B08"]].astype(np.float32)
    if not np.isfinite(nir).any():
        return {k: np.nan for k in [
            "SV_gamma1", "SV_gamma2", "SV_gamma3", "SV_gamma4",
            "SV_nugget", "SV_sill", "SV_range",
        ]}

    z = np.where(np.isfinite(nir), nir, np.nanmean(nir))
    z = np.clip(z, 0.0, 1.0)

    def gamma_at_lag(d: int) -> float:
        vals = []
        vals.append((z[:, :-d] - z[:, d:]) ** 2)
        vals.append((z[:-d, :] - z[d:, :]) ** 2)
        vals.append((z[:-d, :-d] - z[d:, d:]) ** 2)
        vals.append((z[d:, :-d] - z[:-d, d:]) ** 2)
        v = np.concatenate([x.ravel() for x in vals])
        return 0.5 * float(np.mean(v)) if v.size else np.nan

    lags = np.array([1, 2, 3, 4], dtype=np.float32)
    gammas = np.array([gamma_at_lag(int(d)) for d in lags], dtype=np.float32)

    for i, g in enumerate(gammas, start=1):
        feats[f"SV_gamma{i}"] = float(g)

    # Fit exponential model: nugget + sill*(1-exp(-h/range))
    def model(h, nugget, sill, rng):
        return nugget + sill * (1.0 - np.exp(-h / (rng + EPS)))

    if np.isfinite(gammas).sum() >= 3:
        nug0 = float(gammas[0])
        sill0 = float(np.nanmax(gammas) - nug0)
        try:
            popt, _ = curve_fit(
                model, lags, gammas,
                p0=[nug0, max(sill0, 1e-6), 2.0],
                bounds=([0.0, 0.0, 0.1], [1.0, 1.0, 20.0]),
                maxfev=2000,
            )
            feats["SV_nugget"] = float(popt[0])
            feats["SV_sill"] = float(popt[1])
            feats["SV_range"] = float(popt[2])
        except Exception:
            feats["SV_nugget"] = np.nan
            feats["SV_sill"] = np.nan
            feats["SV_range"] = np.nan
    else:
        feats["SV_nugget"] = np.nan
        feats["SV_sill"] = np.nan
        feats["SV_range"] = np.nan

    return feats


# =============================================================================
# OSM features (cached)
# =============================================================================

def _sindex_query(gdf, geom):
    candidates = list(gdf.sindex.query(geom, predicate="intersects"))
    if not candidates:
        return gdf.iloc[[]]
    sub = gdf.iloc[candidates]
    return sub[sub.intersects(geom)]


def load_osm_data() -> dict:
    osm_data = {}
    for name in ["buildings", "roads", "landuse", "natural", "water"]:
        path = os.path.join(OSM_DIR, f"{name}.gpkg")
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            if gdf.crs and gdf.crs.to_epsg() != 32632:
                gdf = gdf.to_crs(epsg=32632)
            _ = gdf.sindex
            osm_data[name] = gdf
            print(f"  Loaded OSM {name}: {len(gdf)} features")
        else:
            osm_data[name] = None
            print(f"  OSM {name}: not found")
    return osm_data


def compute_osm_features(grid: gpd.GeoDataFrame, osm_data: dict) -> pd.DataFrame:
    """Extract OSM features for all grid cells (cached once)."""
    b = osm_data.get("buildings")
    r = osm_data.get("roads")
    lu = osm_data.get("landuse")
    w = osm_data.get("water")
    records = []

    for idx, row in enumerate(grid.itertuples(index=False)):
        cell_id = int(row.cell_id)
        geom = row.geometry
        cell_area = float(geom.area)
        rec = {"cell_id": cell_id}

        # Buildings
        if b is not None and len(b) > 0:
            hits = _sindex_query(b, geom)
            rec["osm_building_count"] = int(len(hits))
            if len(hits) > 0:
                inter = hits.geometry.intersection(geom)
                area = float(inter.area.sum())
                rec["osm_building_area_frac"] = area / cell_area
                rec["osm_building_mean_area"] = area / len(hits)
            else:
                rec["osm_building_area_frac"] = 0.0
                rec["osm_building_mean_area"] = 0.0
        else:
            rec["osm_building_count"] = 0
            rec["osm_building_area_frac"] = 0.0
            rec["osm_building_mean_area"] = 0.0

        # Roads
        if r is not None and len(r) > 0:
            hits = _sindex_query(r, geom)
            rec["osm_road_count"] = int(len(hits))
            if len(hits) > 0:
                inter = hits.geometry.intersection(geom)
                rec["osm_road_length"] = float(inter.length.sum())
            else:
                rec["osm_road_length"] = 0.0
        else:
            rec["osm_road_count"] = 0
            rec["osm_road_length"] = 0.0

        # Land use
        if lu is not None and len(lu) > 0 and "landuse" in lu.columns:
            hits = _sindex_query(lu, geom)
            if len(hits) > 0:
                dominant = hits["landuse"].mode()
                rec["osm_landuse_type"] = str(dominant.iloc[0]) if len(dominant) > 0 else "unknown"
                rec["osm_landuse_count"] = int(len(hits))
            else:
                rec["osm_landuse_type"] = "none"
                rec["osm_landuse_count"] = 0
        else:
            rec["osm_landuse_type"] = "none"
            rec["osm_landuse_count"] = 0

        # Water
        if w is not None and len(w) > 0:
            centroid = geom.centroid
            candidates = list(w.sindex.query(centroid.buffer(5000), predicate="intersects"))
            nearby = w.iloc[candidates] if candidates else w
            rec["osm_water_min_dist"] = float(nearby.geometry.distance(centroid).min())
            rec["osm_water_intersects"] = int(nearby.intersects(geom).any())
        else:
            rec["osm_water_min_dist"] = 99999.0
            rec["osm_water_intersects"] = 0

        records.append(rec)

        if (idx + 1) % 5000 == 0:
            print(f"    OSM: {idx + 1}/{len(grid)} cells done")

    df = pd.DataFrame(records)

    # One-hot landuse
    if "osm_landuse_type" in df.columns:
        dummies = pd.get_dummies(df["osm_landuse_type"], prefix="osm_lu", dtype=np.float32)
        df = df.drop(columns=["osm_landuse_type"])
        df = pd.concat([df, dummies], axis=1)

    return df.sort_values("cell_id").reset_index(drop=True)


# =============================================================================
# Orchestration
# =============================================================================

def extract_features_for_cell(patch_raw: np.ndarray, feature_set: str,
                              scale: float) -> dict:
    """Compute selected feature blocks for one cell patch."""
    patch_ref = _normalize_reflectance(patch_raw, scale)

    feats = {}
    feats.update(band_statistics(patch_ref))
    feats.update(spectral_indices(patch_ref))
    feats.update(tasseled_cap(patch_ref))
    feats.update(spatial_simple(patch_ref))

    if feature_set == "full":
        feats.update(glcm_features(patch_ref))
        feats.update(gabor_features(patch_ref))
        feats.update(lbp_features(patch_ref))
        feats.update(hog_features(patch_ref))
        feats.update(morph_profile_features(patch_ref))
        feats.update(semivariogram_features(patch_ref))

    return feats


# Columns that should NEVER be imputed (control/metadata, not features)
_IMPUTE_EXCLUDE = {"cell_id", "valid_fraction", "low_valid_fraction",
                   "reflectance_scale", "full_features_computed"}


def impute_df(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Impute numeric feature columns only (excludes control columns)."""
    if strategy == "none":
        return df
    out = df.copy()
    num_cols = [c for c in out.columns if c not in _IMPUTE_EXCLUDE
                and pd.api.types.is_numeric_dtype(out[c])]
    if strategy == "zero":
        out[num_cols] = out[num_cols].fillna(0.0)
    elif strategy == "median":
        med = out[num_cols].median(numeric_only=True)
        out[num_cols] = out[num_cols].fillna(med)
    else:
        raise ValueError(f"Unknown impute strategy: {strategy}")
    return out


def process_one_composite(year: int, season: str, grid: gpd.GeoDataFrame,
                          osm_df: pd.DataFrame, feature_set: str, impute: str):
    """Extract features for a single year/season composite."""
    print(f"\n{'='*60}")
    print(f"Processing {year}/{season}")
    print(f"{'='*60}")

    spectral, vf = load_sentinel_v2(year, season)
    n_rows, n_cols = compute_grid_shape(spectral)
    assert len(grid) == n_rows * n_cols, (
        f"Grid size mismatch: {len(grid)} != {n_rows}*{n_cols}"
    )

    scale = detect_reflectance_scale(spectral)
    print(f"  Spectral shape: {spectral.shape} | "
          f"detected scale={scale} (divide by this for reflectance)")

    records = []
    low_vf_count = 0

    for row in grid.itertuples(index=False):
        cell_id = int(row.cell_id)
        row_idx = cell_id // n_cols
        col_idx = cell_id % n_cols

        v = cell_valid_fraction(vf, row_idx, col_idx)
        low = int(v < MIN_VALID_FRAC)
        low_vf_count += low

        patch = extract_cell_patch(spectral, row_idx, col_idx)
        rec = {
            "cell_id": cell_id,
            "valid_fraction": float(v),
            "low_valid_fraction": low,
            "reflectance_scale": float(scale),
            "full_features_computed": 0,
        }

        # If quality is low, keep the row but only compute core features.
        # Heavy features remain NaN. This keeps cell_id sets identical.
        if low:
            rec.update(extract_features_for_cell(patch, feature_set="core", scale=scale))
        else:
            rec.update(extract_features_for_cell(patch, feature_set=feature_set, scale=scale))
            if feature_set == "full":
                rec["full_features_computed"] = 1

        records.append(rec)

        if (cell_id + 1) % 5000 == 0:
            print(f"  Features: {cell_id + 1}/{len(grid)} cells done")

    df = pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)

    drop_pct = 100.0 * low_vf_count / len(grid)
    print(f"  low_valid_fraction: {low_vf_count}/{len(grid)} ({drop_pct:.1f}%)")
    if drop_pct > MAX_DROP_PCT:
        print(f"  WARNING: low_valid_fraction {drop_pct:.1f}% > {MAX_DROP_PCT}% threshold!")

    # Merge OSM (time-invariant features)
    if osm_df is not None and len(osm_df) > 0:
        df = df.merge(osm_df, on="cell_id", how="left")

    # Clean inf -> nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute
    df = impute_df(df, impute)

    os.makedirs(V2_DIR, exist_ok=True)
    out_path = os.path.join(V2_DIR, f"features_{year}_{season}_{feature_set}.parquet")
    df.to_parquet(out_path, index=False)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    n_feats = len([c for c in df.columns if c not in ("cell_id", "year", "season")])
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Shape: {df.shape[0]} cells x {df.shape[1]} columns ({n_feats} features)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Extract v2 features (per season)")
    parser.add_argument("--year", type=int, help="Year to process")
    parser.add_argument("--season", type=str, help="Season to process")
    parser.add_argument("--all", action="store_true", help="Process all composites")
    parser.add_argument("--skip-osm", action="store_true", help="Skip OSM features")
    parser.add_argument("--feature-set", choices=["core", "full"], default="core")
    parser.add_argument("--impute", choices=["none", "median", "zero"], default="median")
    args = parser.parse_args()

    if args.all:
        jobs = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]
    elif args.year is not None and args.season is not None:
        jobs = [(args.year, args.season)]
    else:
        parser.error("Specify --year and --season, or --all")

    # Load grid (once, sorted by cell_id for deterministic row/col mapping)
    print("Loading v2 grid...")
    grid = gpd.read_file(GRID_PATH)
    grid = grid.sort_values("cell_id").reset_index(drop=True)
    assert (grid["cell_id"].values == np.arange(len(grid))).all(), (
        "Grid cell_id is not contiguous 0..N-1"
    )
    print(f"  {len(grid)} cells")

    # OSM features (cached once, reused for all composites)
    osm_df = None
    if not args.skip_osm:
        if os.path.exists(OSM_CACHE_PATH):
            print(f"Loading cached OSM features: {OSM_CACHE_PATH}")
            osm_df = pd.read_parquet(OSM_CACHE_PATH)
            osm_df = osm_df.sort_values("cell_id").reset_index(drop=True)
            # Sanity: catch stale cache
            assert len(osm_df) == len(grid), f"OSM cache size {len(osm_df)} != grid {len(grid)}"
            assert osm_df["cell_id"].is_unique, "OSM cache has duplicate cell_ids"
            assert (osm_df["cell_id"].values == np.arange(len(grid))).all(), "OSM cache cell_ids don't match grid"
        else:
            print("Computing OSM features (will be cached)...")
            osm_data = load_osm_data()
            osm_df = compute_osm_features(grid, osm_data)
            osm_df.to_parquet(OSM_CACHE_PATH, index=False)
            print(f"  Saved OSM cache -> {OSM_CACHE_PATH}")

    for year, season in jobs:
        process_one_composite(
            year=year, season=season, grid=grid,
            osm_df=osm_df, feature_set=args.feature_set, impute=args.impute,
        )

    print(f"\n{'='*60}")
    print(f"DONE! Processed {len(jobs)} composite(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
