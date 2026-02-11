"""
Phase 3: Feature Engineering

Extracts ~163 tabular features per 100m grid cell from Sentinel-2 imagery + OSM data.

Feature categories:
  1. Per-band statistics (mean, std, min, max, median) -- ~50 features
  2. Spectral indices (NDVI, NDBI, NDWI, SAVI, BSI)   -- ~25 features
  3. Tasseled Cap (brightness, greenness, wetness)     -- ~6 features
  4. GLCM texture features                             -- ~10 features
  5. Gabor wavelet features                            -- ~24 features
  6. Spatial autocorrelation / edge density             -- ~8 features
  7. OSM-derived features                              -- ~40 features

Correctness notes:
  - 20m bands (B05-B12) block-reduced to native 5x5 for std/min/max
  - GLCM uses symmetric=True, normed=True, fixed quantization bins
  - Moran's I counts each adjacency pair once (no 2x factor)
  - OSM uses sindex.query() for O(log n) spatial indexing

Usage:
    python src/features/extract_features.py [--year 2020]

Outputs:
    data/processed/features_{year}.parquet
"""

import argparse
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor_kernel

warnings.filterwarnings("ignore")

# -- Configuration ----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GRID_PATH = os.path.join(PROCESSED_DIR, "grid.gpkg")

GRID_SIZE_M = 100
PIXEL_SIZE = 10
GRID_PX = GRID_SIZE_M // PIXEL_SIZE  # 10 pixels per cell side

# Sentinel-2 band names (in order as stored in our GeoTIFFs)
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}

# Bands at native 20m (upsampled to 10m in our GeoTIFFs)
BANDS_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}

# Tasseled Cap coefficients for Sentinel-2 (Nedkov, 2017)
# Order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
TC_BRIGHTNESS = np.array(
    [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949, 0.1822, 0.0031, 0.1112, 0.0825]
)
TC_GREENNESS = np.array(
    [-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.0469, -0.0322, -0.0015, -0.0693, -0.0180]
)
TC_WETNESS = np.array(
    [0.2578, 0.2305, 0.0883, 0.1071, -0.7611, 0.0882, 0.4572, -0.0021, -0.4064, 0.0117]
)


def load_sentinel(year: int) -> tuple:
    """Load Sentinel-2 raster for a given year, return (bands_array, transform)."""
    path = os.path.join(RAW_DIR, f"sentinel2_nuremberg_{year}.tif")
    with rasterio.open(path) as ds:
        data = ds.read()  # shape: (10, H, W)
        transform = ds.transform
    return data, transform


def extract_cell_patch(bands: np.ndarray, transform: rasterio.Affine, geom) -> np.ndarray:
    """Extract a 10x10 pixel patch for a grid cell geometry."""
    origin_x = transform.c
    origin_y = transform.f
    px = abs(transform.a)

    col_start = int((geom.bounds[0] - origin_x) / px)
    row_start = int((origin_y - geom.bounds[3]) / px)

    patch = bands[:, row_start : row_start + GRID_PX, col_start : col_start + GRID_PX]
    return patch  # shape: (10, 10, 10)


# -- Helpers ----------------------------------------------------------------


def _block_reduce_mean(arr: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample 2D array by factor using block mean (undo upsampling artifacts)."""
    H, W = arr.shape
    H2, W2 = (H // factor) * factor, (W // factor) * factor
    return arr[:H2, :W2].reshape(H2 // factor, factor, W2 // factor, factor).mean(axis=(1, 3))


# -- Feature extraction functions ------------------------------------------


def band_statistics(patch: np.ndarray) -> dict:
    """Tier 1: Per-band stats. 20m bands block-reduced to native 5x5 grid
    so that std/min/max reflect true variability, not upsampling artifacts."""
    features = {}
    for i, name in enumerate(BAND_NAMES):
        band = patch[i].astype(np.float32)
        if name in BANDS_20M and band.shape[0] >= 2 and band.shape[1] >= 2:
            band = _block_reduce_mean(band, factor=2)
        flat = band.ravel()
        features[f"{name}_mean"] = float(np.mean(flat))
        features[f"{name}_std"] = float(np.std(flat))
        features[f"{name}_min"] = float(np.min(flat))
        features[f"{name}_max"] = float(np.max(flat))
        features[f"{name}_median"] = float(np.median(flat))
    return features


def spectral_indices(patch: np.ndarray) -> dict:
    """Tier 1: Spectral index features."""
    eps = 1e-10
    blue = patch[BAND_INDEX["B02"]].astype(np.float32)
    green = patch[BAND_INDEX["B03"]].astype(np.float32)
    red = patch[BAND_INDEX["B04"]].astype(np.float32)
    nir = patch[BAND_INDEX["B08"]].astype(np.float32)
    swir1 = patch[BAND_INDEX["B11"]].astype(np.float32)

    # NDVI: vegetation
    ndvi = (nir - red) / (nir + red + eps)
    # NDBI: built-up
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    # NDWI: water
    ndwi = (green - nir) / (green + nir + eps)
    # SAVI: soil-adjusted vegetation (L=0.5)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    # BSI: bare soil index
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)

    features = {}
    for name, arr in [
        ("NDVI", ndvi),
        ("NDBI", ndbi),
        ("NDWI", ndwi),
        ("SAVI", savi),
        ("BSI", bsi),
    ]:
        flat = arr.ravel()
        features[f"{name}_mean"] = float(np.mean(flat))
        features[f"{name}_std"] = float(np.std(flat))
        features[f"{name}_median"] = float(np.median(flat))
        features[f"{name}_q25"] = float(np.percentile(flat, 25))
        features[f"{name}_q75"] = float(np.percentile(flat, 75))
    return features


def tasseled_cap(patch: np.ndarray) -> dict:
    """Tier 2: Tasseled Cap transformation (brightness, greenness, wetness)."""
    n_bands = patch.shape[0]
    pixels = patch.reshape(n_bands, -1).T.astype(np.float32)  # (100, 10)

    brightness = pixels @ TC_BRIGHTNESS
    greenness = pixels @ TC_GREENNESS
    wetness = pixels @ TC_WETNESS

    features = {}
    for name, arr in [("TC_bright", brightness), ("TC_green", greenness), ("TC_wet", wetness)]:
        features[f"{name}_mean"] = float(np.mean(arr))
        features[f"{name}_std"] = float(np.std(arr))
    return features


def glcm_features(patch: np.ndarray) -> dict:
    """Tier 2: GLCM texture on NIR and NDVI.
    Uses symmetric=True, normed=True for cross-patch comparability.
    Uses fixed quantization bins (NDVI in [-1,1], reflectance in [0,1])."""
    features = {}
    eps = 1e-10

    nir = patch[BAND_INDEX["B08"]].astype(np.float32)
    red = patch[BAND_INDEX["B04"]].astype(np.float32)
    ndvi = (nir - red) / (nir + red + eps)

    for name, arr, vmin, vmax in [("NIR", nir, 0.0, 1.0), ("NDVI", ndvi, -1.0, 1.0)]:
        # Fixed-range quantization (not per-patch min-max) for comparability
        clipped = np.clip(arr, vmin, vmax)
        quantized = ((clipped - vmin) / (vmax - vmin + eps) * 31).astype(np.uint8)

        glcm = graycomatrix(
            quantized,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=32,
            symmetric=True,
            normed=True,
        )

        for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
            val = float(np.mean(graycoprops(glcm, prop)))
            features[f"GLCM_{name}_{prop}"] = val if np.isfinite(val) else 0.0

    return features


def gabor_features(patch: np.ndarray) -> dict:
    """Tier 3: Gabor wavelet features at multiple scales and orientations."""
    features = {}
    nir = patch[BAND_INDEX["B08"]].astype(np.float64)

    # Normalize to [0, 1]
    nir_min, nir_max = nir.min(), nir.max()
    if nir_max - nir_min > 1e-10:
        nir_norm = (nir - nir_min) / (nir_max - nir_min)
    else:
        nir_norm = np.zeros_like(nir)

    for sigma in [1.0, 2.0, 4.0]:
        for theta_deg in [0, 45, 90, 135]:
            theta = np.deg2rad(theta_deg)
            kernel = np.real(
                gabor_kernel(frequency=0.3, theta=theta, sigma_x=sigma, sigma_y=sigma)
            )
            response = ndimage.convolve(nir_norm, kernel, mode="reflect")
            features[f"Gabor_s{sigma:.0f}_t{theta_deg}_mean"] = float(np.mean(response))
            features[f"Gabor_s{sigma:.0f}_t{theta_deg}_std"] = float(np.std(response))

    return features


def spatial_features(patch: np.ndarray) -> dict:
    """Tier 2: Spatial autocorrelation and edge density."""
    features = {}
    nir = patch[BAND_INDEX["B08"]].astype(np.float64)

    # Edge density (Sobel magnitude)
    sobel_x = ndimage.sobel(nir, axis=1)
    sobel_y = ndimage.sobel(nir, axis=0)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    features["edge_density_mean"] = float(np.mean(edge_mag))
    features["edge_density_std"] = float(np.std(edge_mag))
    features["edge_density_max"] = float(np.max(edge_mag))

    # Local variance (Laplacian)
    laplacian = ndimage.laplace(nir)
    features["laplacian_mean"] = float(np.mean(np.abs(laplacian)))
    features["laplacian_std"] = float(np.std(laplacian))

    # Moran's I for NIR band (rook adjacency, each pair counted once)
    n = nir.size
    mean_val = np.mean(nir)
    dev = nir - mean_val
    denom = float(np.sum(dev**2))
    if denom > 1e-10:
        h_sum = float(np.sum(dev[:, :-1] * dev[:, 1:]))  # horizontal pairs
        v_sum = float(np.sum(dev[:-1, :] * dev[1:, :]))  # vertical pairs
        # Count each pair once (no factor of 2)
        n_pairs = (nir.shape[0] * (nir.shape[1] - 1)) + ((nir.shape[0] - 1) * nir.shape[1])
        morans_i = (n / n_pairs) * (h_sum + v_sum) / denom
    else:
        morans_i = 0.0
    features["morans_I_NIR"] = float(morans_i)

    # NDVI spatial heterogeneity
    eps = 1e-10
    red = patch[BAND_INDEX["B04"]].astype(np.float64)
    ndvi = (nir - red) / (nir + red + eps)
    features["NDVI_spatial_range"] = float(np.max(ndvi) - np.min(ndvi))
    features["NDVI_spatial_iqr"] = float(np.percentile(ndvi, 75) - np.percentile(ndvi, 25))

    return features


# -- OSM features -----------------------------------------------------------


def _sindex_query(gdf, geom):
    """Use the actual spatial index to pre-filter candidates, then precise test."""
    candidates = list(gdf.sindex.query(geom, predicate="intersects"))
    if not candidates:
        return gdf.iloc[[]]
    return gdf.iloc[candidates][gdf.iloc[candidates].intersects(geom)]


def osm_features(cell_geom, osm_data: dict) -> dict:
    """OSM-derived features per grid cell using sindex.query() for O(log n) lookups."""
    features = {}
    cell_area = cell_geom.area  # m^2

    # Buildings
    b = osm_data.get("buildings")
    if b is not None and len(b) > 0:
        hits = _sindex_query(b, cell_geom)
        features["osm_building_count"] = len(hits)
        if len(hits) > 0:
            bldg_area = hits.geometry.intersection(cell_geom).area.sum()
            features["osm_building_area_frac"] = bldg_area / cell_area
            features["osm_building_mean_area"] = bldg_area / len(hits)
        else:
            features["osm_building_area_frac"] = 0.0
            features["osm_building_mean_area"] = 0.0
    else:
        features["osm_building_count"] = 0
        features["osm_building_area_frac"] = 0.0
        features["osm_building_mean_area"] = 0.0

    # Roads
    r = osm_data.get("roads")
    if r is not None and len(r) > 0:
        hits = _sindex_query(r, cell_geom)
        if len(hits) > 0:
            clipped = hits.geometry.intersection(cell_geom)
            features["osm_road_length"] = float(clipped.length.sum())
            features["osm_road_count"] = len(hits)
        else:
            features["osm_road_length"] = 0.0
            features["osm_road_count"] = 0
    else:
        features["osm_road_length"] = 0.0
        features["osm_road_count"] = 0

    # Land use
    lu = osm_data.get("landuse")
    if lu is not None and len(lu) > 0 and "landuse" in lu.columns:
        hits = _sindex_query(lu, cell_geom)
        if len(hits) > 0:
            dominant = hits["landuse"].mode()
            features["osm_landuse_type"] = dominant.iloc[0] if len(dominant) > 0 else "unknown"
            features["osm_landuse_count"] = len(hits)
        else:
            features["osm_landuse_type"] = "none"
            features["osm_landuse_count"] = 0
    else:
        features["osm_landuse_type"] = "none"
        features["osm_landuse_count"] = 0

    # Water proximity (tiered search: 5km buffer first, then full dataset)
    w = osm_data.get("water")
    if w is not None and len(w) > 0:
        centroid = cell_geom.centroid
        # Try 5km buffer first for speed
        candidates = list(w.sindex.query(centroid.buffer(5000), predicate="intersects"))
        if not candidates:
            # Fallback: search full dataset (slower but always correct)
            candidates = list(range(len(w)))
        nearby = w.iloc[candidates]
        features["osm_water_min_dist"] = float(nearby.geometry.distance(centroid).min())
        features["osm_water_intersects"] = int(nearby.intersects(cell_geom).any())
    else:
        features["osm_water_min_dist"] = 99999.0
        features["osm_water_intersects"] = 0

    return features


def load_osm_data() -> dict:
    """Load all OSM GeoPackage files, reproject to EPSG:32632."""
    osm_dir = os.path.join(RAW_DIR, "osm")
    osm_data = {}

    for name in ["buildings", "roads", "landuse", "natural", "water"]:
        path = os.path.join(osm_dir, f"{name}.gpkg")
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            if gdf.crs and gdf.crs.to_epsg() != 32632:
                gdf = gdf.to_crs(epsg=32632)
            # Build spatial index
            _ = gdf.sindex
            osm_data[name] = gdf
            print(f"  Loaded OSM {name}: {len(gdf)} features")
        else:
            osm_data[name] = None
            print(f"  OSM {name}: not found")

    return osm_data


def extract_osm_with_spatial_index(grid: gpd.GeoDataFrame, osm_data: dict) -> pd.DataFrame:
    """Extract OSM features using spatial index for efficiency."""
    print("  Extracting OSM features with spatial indexing...")
    records = []

    for idx, row in grid.iterrows():
        features = osm_features(row.geometry, osm_data)
        features["cell_id"] = row.cell_id
        records.append(features)

        if (idx + 1) % 5000 == 0:
            print(f"    {idx + 1}/{len(grid)} cells done")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Extract features for a given year")
    parser.add_argument("--year", type=int, default=2020, help="Year to process")
    parser.add_argument("--skip-osm", action="store_true", help="Skip OSM features (faster)")
    args = parser.parse_args()

    year = args.year
    print(f"{'='*60}")
    print(f"Phase 3: Feature Engineering for {year}")
    print(f"{'='*60}")

    # Load grid
    print("\nLoading grid...")
    grid = gpd.read_file(GRID_PATH)
    print(f"  {len(grid)} cells")

    # Load Sentinel-2
    print(f"\nLoading Sentinel-2 {year}...")
    bands, transform = load_sentinel(year)
    print(f"  Shape: {bands.shape}")

    # Load OSM data (once, same for all years)
    osm_data = {}
    if not args.skip_osm:
        print("\nLoading OSM data...")
        osm_data = load_osm_data()

    # Extract features for each cell
    print(f"\nExtracting features for {len(grid)} cells...")
    all_records = []

    for idx, row in grid.iterrows():
        cell_id = row.cell_id
        geom = row.geometry

        # Extract pixel patch
        patch = extract_cell_patch(bands, transform, geom)

        if patch.shape != (10, GRID_PX, GRID_PX):
            # Edge cell with incomplete data
            continue

        # Compute all spectral features
        record = {"cell_id": cell_id}
        record.update(band_statistics(patch))
        record.update(spectral_indices(patch))
        record.update(tasseled_cap(patch))
        record.update(glcm_features(patch))
        record.update(gabor_features(patch))
        record.update(spatial_features(patch))

        all_records.append(record)

        if (idx + 1) % 5000 == 0:
            print(f"  Spectral features: {idx + 1}/{len(grid)} cells done")

    spectral_df = pd.DataFrame(all_records)
    print(f"  Spectral features: {spectral_df.shape[1] - 1} features x {len(spectral_df)} cells")

    # OSM features (separate because of spatial join overhead)
    if not args.skip_osm and osm_data:
        osm_df = extract_osm_with_spatial_index(grid, osm_data)

        # Handle categorical landuse: one-hot encode
        if "osm_landuse_type" in osm_df.columns:
            landuse_dummies = pd.get_dummies(
                osm_df["osm_landuse_type"], prefix="osm_lu", dtype=np.float32
            )
            osm_df = osm_df.drop(columns=["osm_landuse_type"])
            osm_df = pd.concat([osm_df, landuse_dummies], axis=1)

        print(f"  OSM features: {osm_df.shape[1] - 1} features x {len(osm_df)} cells")

        # Merge spectral + OSM
        features_df = spectral_df.merge(osm_df, on="cell_id", how="left")
    else:
        features_df = spectral_df

    # Replace inf/nan
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    if nan_cols:
        print(f"\n  Warning: NaN in {len(nan_cols)} columns, filling with 0")
        features_df = features_df.fillna(0)

    # Save
    output_path = os.path.join(PROCESSED_DIR, f"features_{year}.parquet")
    features_df.to_parquet(output_path, index=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  Shape: {features_df.shape[0]} cells x {features_df.shape[1]} columns")
    print(f"  Features: {features_df.shape[1] - 1} (excluding cell_id)")

    # Quick summary
    print(f"\n{'='*60}")
    print("Feature summary:")
    print(f"{'='*60}")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in sorted(numeric_cols):
        if col == "cell_id":
            continue
        print(
            f"  {col:<35} mean={features_df[col].mean():>10.4f}"
            f"  std={features_df[col].std():>10.4f}"
        )


if __name__ == "__main__":
    main()
