#!/usr/bin/env python3
"""
Multi-City Pipeline V5: SAR integration + phenological features.

Builds on V4 with the addition of Sentinel-1 SAR features:
  - Stage 1b: Download S1 GRD composites (VV/VH) per season
  - Updated Rust terrapulse binary extracts 48 SAR features/season
  - Wider MLP (5L x 3072w) to leverage expanded feature set
  - SAR phenological features (curvature/slope/amplitude/peak)

> [!WARNING]
> Sentinel-1B failed in Dec 2021. S1 data may be sparse for seasons
> in late 2022 and 2023. The pipeline handles missing SAR gracefully.

Cities (training, 14):
    Bremen, Hamburg, Duesseldorf, Leipzig, Rostock, Amsterdam,
    Hambach Mine, Welzow Mine, Amiens, Magdeburg, Ulm,
    Salzburg, Schwerin, Malmo

City (validation/test):
    Nuremberg

Usage:
    .venv/Scripts/python.exe scripts/run_multi_city_pipeline_v5.py

Runtime: ~45-60 min (most cities cached, includes SAR download)
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from math import ceil, floor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                  "B11", "B12"]
SENTINEL_RES = 10
SENTINEL_NODATA = -9999
MIN_SCENES = 8
SEASON_DATES = {
    "spring": ("04-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "autumn": ("09-01", "10-31"),
}
SEASONS = ["spring", "summer", "autumn"]
SCL_EXCLUDE = [0, 1, 2, 3, 8, 9, 10, 11]

ALL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
WC_YEARS = [2020, 2021]
GRID_PX = 10  # pixels per 100m cell side
GRID_SIZE_M = GRID_PX * SENTINEL_RES

WC_CLASS_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4,
                60: 5, 70: 5, 100: 5, 80: 6}
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)

SEED = 42
N_FOLDS = 5
N_SEEDS = 1  # single seed for iteration
MLP_SEED_OFFSET = 100  # best seed from sweep
MLP_DEPTH = 5
MLP_WIDTH = 3072
MAX_EPOCHS = 3000
PATIENCE_STEPS = 30000
MIN_STEPS = 3000
CHECKPOINT_EVERY = 10  # epochs

CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                "reflectance_scale", "full_features_computed"}

# Rust CLI binary path
TERRAPULSE_BIN = os.path.join(PROJECT_ROOT, "terrapulse", "target", "release",
                              "terrapulse.exe")

# ---------------------------------------------------------------------------
# City Configurations
# ---------------------------------------------------------------------------

@dataclass
class CityConfig:
    name: str
    bbox: List[float]  # [west, south, east, north] WGS84
    epsg: int
    wc_tile: str
    is_test: bool = False

CITIES = [
    # === ORIGINAL TRAINING CITIES (V1-V3) ===
    CityConfig("bremen",      [8.65, 53.00, 8.90, 53.14], 32632, "N51E006"),
    CityConfig("hamburg",     [9.80, 53.40, 10.15, 53.58], 32632, "N51E009"),
    CityConfig("duesseldorf", [6.70, 51.15, 6.90, 51.28], 32632, "N51E006"),
    CityConfig("leipzig",     [12.25, 51.27, 12.50, 51.40], 32633, "N51E012"),
    CityConfig("rostock",     [12.00, 54.05, 12.20, 54.18], 32633, "N54E012"),
    CityConfig("amsterdam",   [4.75, 52.30, 4.95, 52.40], 32631, "N51E003"),
    CityConfig("hambach_mine",[6.40, 50.85, 6.60, 50.98], 32632, "N48E006"),
    CityConfig("welzow_mine", [14.10, 51.50, 14.35, 51.65], 32633, "N51E012"),
    CityConfig("amiens",      [2.15, 49.82, 2.42, 49.96], 32631, "N48E000"),
    CityConfig("magdeburg",   [11.50, 52.05, 11.75, 52.20], 32632, "N51E009"),
    CityConfig("ulm",         [9.85, 48.33, 10.10, 48.47], 32632, "N48E009"),
    CityConfig("salzburg",    [12.95, 47.73, 13.15, 47.87], 32633, "N45E012"),
    CityConfig("schwerin",    [11.30, 53.55, 11.55, 53.70], 32632, "N51E009"),
    CityConfig("malmo",       [12.90, 55.53, 13.15, 55.68], 32633, "N54E012"),
    # V4 cities → training (except Nuremberg → back to test)
    CityConfig("frankfurt",   [8.55, 50.05, 8.80, 50.18], 32632, "N48E006"),
    CityConfig("regensburg",  [12.00, 48.95, 12.20, 49.07], 32633, "N48E012"),

    # === NEW — Northwestern Europe ===
    CityConfig("london",      [-0.25, 51.42, 0.05, 51.56], 32631, "N51W003"),
    CityConfig("brussels",    [4.30, 50.80, 4.50, 50.92], 32631, "N48E003"),
    CityConfig("rotterdam",   [4.35, 51.85, 4.55, 51.97], 32631, "N51E003"),
    CityConfig("antwerp",     [4.30, 51.15, 4.50, 51.28], 32631, "N51E003"),
    CityConfig("paris_south", [2.25, 48.75, 2.50, 48.89], 32631, "N48E000"),

    # === NEW — Central Europe ===
    CityConfig("berlin",      [13.30, 52.45, 13.55, 52.58], 32633, "N51E012"),
    CityConfig("vienna",      [16.30, 48.15, 16.55, 48.28], 32633, "N48E015"),
    CityConfig("zurich",      [8.45, 47.32, 8.65, 47.44], 32632, "N45E006"),
    CityConfig("munich_north",[11.45, 48.22, 11.70, 48.36], 32632, "N48E009"),
    CityConfig("stuttgart",   [9.10, 48.72, 9.30, 48.84], 32632, "N48E009"),
    CityConfig("innsbruck",   [11.30, 47.22, 11.55, 47.35], 32632, "N45E009"),

    # === NEW — Eastern Europe ===
    CityConfig("krakow",      [19.85, 50.02, 20.10, 50.14], 32634, "N48E018"),
    CityConfig("budapest",    [19.00, 47.42, 19.20, 47.55], 32634, "N45E018"),
    CityConfig("bratislava",  [17.05, 48.10, 17.25, 48.22], 32633, "N48E015"),

    # === NEW — Nordic ===
    CityConfig("helsinki",    [24.85, 60.13, 25.10, 60.27], 32635, "N60E024"),
    CityConfig("copenhagen",  [12.48, 55.62, 12.68, 55.74], 32633, "N54E012"),
    CityConfig("gothenburg",  [11.90, 57.65, 12.10, 57.78], 32632, "N57E009"),

    # === NEW — Southern Europe ===
    CityConfig("barcelona",   [2.05, 41.32, 2.30, 41.45], 32631, "N39E000"),
    CityConfig("madrid",      [-3.80, 40.35, -3.55, 40.48], 32630, "N39W006"),
    CityConfig("lisbon",      [-9.20, 38.68, -8.95, 38.82], 32629, "N36W012"),
    CityConfig("rome",        [12.40, 41.82, 12.60, 41.95], 32633, "N39E012"),
    CityConfig("milan",       [9.10, 45.42, 9.35, 45.55], 32632, "N45E009"),
    CityConfig("lyon",        [4.75, 45.70, 4.95, 45.82], 32631, "N45E003"),
    CityConfig("toulouse",    [1.35, 43.55, 1.55, 43.68], 32631, "N42E000"),
    CityConfig("athens",      [23.65, 37.92, 23.85, 38.05], 32635, "N36E021"),

    # === NEW — Specialty (bare land / cropland) ===
    CityConfig("almeria_coast",[-2.50, 36.78, -2.30, 36.90], 32630, "N36W003"),
    CityConfig("central_hungary",[19.50, 47.10, 19.75, 47.24], 32634, "N45E018"),

    # === NEW — Fill gaps to reach 50 training ===
    CityConfig("dublin",      [-6.35, 53.30, -6.15, 53.42], 32629, "N51W009"),
    CityConfig("marseille",   [5.30, 43.25, 5.50, 43.38], 32631, "N42E003"),
    CityConfig("naples",      [14.18, 40.80, 14.38, 40.93], 32633, "N39E012"),
    CityConfig("valencia",    [-0.45, 39.42, -0.25, 39.55], 32630, "N39W003"),
    CityConfig("bordeaux",    [-0.65, 44.80, -0.45, 44.93], 32630, "N42W003"),
    CityConfig("oslo",        [10.65, 59.87, 10.85, 60.00], 32632, "N57E009"),
    CityConfig("gdansk",      [18.55, 54.32, 18.75, 54.45], 32634, "N54E018"),

    # === 6 VALIDATION CITIES (diverse biomes) ===
    CityConfig("munich",      [11.45, 48.08, 11.70, 48.22], 32632, "N48E009",
               is_test=True),
    CityConfig("nuremberg",   [10.95, 49.38, 11.20, 49.52], 32632, "N48E009",
               is_test=True),
    CityConfig("warsaw",      [20.90, 52.15, 21.15, 52.30], 32634, "N51E018",
               is_test=True),
    CityConfig("prague",      [14.35, 50.02, 14.55, 50.15], 32633, "N48E012",
               is_test=True),
    CityConfig("seville",     [-6.05, 37.32, -5.85, 37.45], 32630, "N36W009",
               is_test=True),
    CityConfig("stockholm",   [17.95, 59.28, 18.20, 59.42], 32633, "N57E015",
               is_test=True),
]

TRAIN_CITIES = [c for c in CITIES if not c.is_test]
TEST_CITIES = [c for c in CITIES if c.is_test]

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
WC_TILES_DIR = os.path.join(CITIES_DIR, "worldcover_tiles")
MODELS_DIR = os.path.join(CITIES_DIR, "models_v5")


def city_dir(city: CityConfig) -> str:
    return os.path.join(CITIES_DIR, city.name)


def city_raw_dir(city: CityConfig) -> str:
    return os.path.join(city_dir(city), "raw")


def city_features_dir(city: CityConfig) -> str:
    return os.path.join(city_dir(city), "features")


def city_predictions_dir(city: CityConfig) -> str:
    return os.path.join(city_dir(city), "predictions")


def city_anchor_path(city: CityConfig) -> str:
    return os.path.join(city_dir(city), f"anchor_{city.name}.tif")


def city_labels_path(city: CityConfig, year: int) -> str:
    return os.path.join(city_dir(city), f"labels_{year}.parquet")


def ensure_all_dirs():
    """Create all output directories."""
    dirs = [CITIES_DIR, WC_TILES_DIR, MODELS_DIR]
    for c in CITIES:
        dirs.extend([city_dir(c), city_raw_dir(c), city_features_dir(c),
                     city_predictions_dir(c)])
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def ts():
    return time.strftime("%H:%M:%S")


# ===========================================================================
# STAGE 0: CREATE ANCHORS
# ===========================================================================

def create_anchor(city: CityConfig):
    """Create anchor GeoTIFF for a city (deterministic bbox -> grid)."""
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    path = city_anchor_path(city)
    if os.path.exists(path):
        with rasterio.open(path) as src:
            nc = src.width // GRID_PX
            nr = src.height // GRID_PX
            print(f"  [{city.name}] Anchor exists "
                  f"({src.width}x{src.height} px, {nc}x{nr}={nc*nr} cells)")
        return

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(city.epsg)
    west, south, east, north = city.bbox
    left, bottom, right, top = transform_bounds(
        src_crs, dst_crs, west, south, east, north, densify_pts=21)

    # Snap to pixel grid
    ps = float(SENTINEL_RES)
    left_s = floor(left / ps) * ps
    bottom_s = floor(bottom / ps) * ps
    right_s = ceil(right / ps) * ps
    top_s = ceil(top / ps) * ps

    # Pad to block multiples
    w0 = round((right_s - left_s) / ps)
    h0 = round((top_s - bottom_s) / ps)
    width = ceil(w0 / GRID_PX) * GRID_PX
    height = ceil(h0 / GRID_PX) * GRID_PX

    right_f = left_s + width * ps
    bottom_f = top_s - height * ps
    transform = Affine(ps, 0.0, left_s, 0.0, -ps, top_s)

    data = np.full((1, height, width), SENTINEL_NODATA, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="float32", crs=dst_crs, transform=transform,
        nodata=SENTINEL_NODATA, compress="lzw",
    ) as dst:
        dst.write(data)

    nc = width // GRID_PX
    nr = height // GRID_PX
    print(f"  [{city.name}] Created anchor: {width}x{height} px, "
          f"{nc}x{nr}={nc*nr} cells (EPSG:{city.epsg})")


def stage_anchors():
    print(f"\n{'='*70}")
    print("STAGE 0: CREATE ANCHOR GRIDS")
    print(f"{'='*70}")
    for city in CITIES:
        create_anchor(city)
    print(f"  [OK] All anchors ready.")


# ===========================================================================
# STAGE 1: DOWNLOAD SENTINEL-2
# ===========================================================================

def download_season(city: CityConfig, year: int, season: str):
    """Download one Sentinel-2 composite for a city-year-season."""
    import planetary_computer
    import pystac_client
    import rasterio
    import stackstac
    import xarray as xr
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    path = os.path.join(city_raw_dir(city),
                        f"sentinel2_{city.name}_{year}_{season}.tif")
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{city.name}/{year}/{season}] Exists ({mb:.1f} MB)")
        return

    # Read anchor for warp target
    anchor_path = city_anchor_path(city)
    with rasterio.open(anchor_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    bbox = city.bbox
    target_epsg = city.epsg
    nodata = SENTINEL_NODATA

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    start_date = f"{year}-{SEASON_DATES[season][0]}"
    end_date = f"{year}-{SEASON_DATES[season][1]}"

    # Progressive cloud relaxation
    items = None
    for cloud_max in [40, 50, 60]:
        items = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": cloud_max}},
        ).item_collection()
        if len(items) >= MIN_SCENES:
            break

    # Widen window if still few
    if len(items) < MIN_SCENES:
        from datetime import datetime, timedelta
        orig_s = datetime.strptime(start_date, "%Y-%m-%d")
        orig_e = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (orig_s - timedelta(days=14)).strftime("%Y-%m-%d")
        end_date = (orig_e + timedelta(days=14)).strftime("%Y-%m-%d")
        items = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": 60}},
        ).item_collection()

    n_scenes = len(items)
    if n_scenes == 0:
        print(f"  [{city.name}/{year}/{season}] WARNING: No scenes!")
        return
    print(f"  [{city.name}/{year}/{season}] {n_scenes} scenes, compositing...")

    # Stack spectral + SCL
    warnings.filterwarnings("ignore", module="stackstac")
    spectral = stackstac.stack(
        items, assets=SENTINEL_BANDS, bounds_latlon=bbox,
        resolution=SENTINEL_RES, epsg=target_epsg, dtype="float64",
        fill_value=np.nan, resampling=Resampling.bilinear, chunksize=1024,
        rescale=False,
    )
    scl = stackstac.stack(
        items, assets=["SCL"], bounds_latlon=bbox,
        resolution=SENTINEL_RES, epsg=target_epsg, dtype="float64",
        fill_value=np.nan, resampling=Resampling.nearest, chunksize=1024,
        rescale=False,
    ).sel(band="SCL")

    spectral, scl = xr.align(spectral, scl, join="exact")
    spectral = spectral.sel(band=SENTINEL_BANDS)

    # SCL mask
    import dask.array as da
    scl_vals = scl.data
    valid = xr.DataArray(da.isfinite(scl_vals),
                         coords=scl.coords, dims=scl.dims)
    for cls in SCL_EXCLUDE:
        valid = valid & (scl != cls)

    valid_fraction_xr = valid.mean(dim="time").astype("float32")
    composite_xr = (spectral.where(valid)
                    .median(dim="time", skipna=True)
                    .astype("float32"))

    print(f"  [{city.name}/{year}/{season}] Computing median composite...")
    composite = composite_xr.compute().values
    valid_fraction = valid_fraction_xr.compute().values

    # Source transform
    xs = np.asarray(composite_xr.coords["x"].values)
    ys = np.asarray(composite_xr.coords["y"].values)
    rx = float(np.abs(xs[1] - xs[0]))
    ry = float(np.abs(ys[1] - ys[0]))
    left = float(xs.min()) - rx / 2
    right = float(xs.max()) + rx / 2
    bottom = float(ys.min()) - ry / 2
    top = float(ys.max()) + ry / 2
    src_transform = rasterio.transform.from_bounds(
        left, bottom, right, top, len(xs), len(ys))
    src_crs = CRS.from_epsg(target_epsg)

    # Warp to anchor
    comp_clean = np.where(np.isnan(composite), nodata, composite).astype(
        np.float32)
    vf_clean = np.where(np.isnan(valid_fraction), nodata,
                        valid_fraction).astype(np.float32)

    n_spectral = len(SENTINEL_BANDS)
    warped = np.full((n_spectral, dst_height, dst_width), nodata,
                     dtype=np.float32)
    for i in range(n_spectral):
        reproject(
            source=comp_clean[i], destination=warped[i],
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata, dst_nodata=nodata,
        )

    vf_warped = np.full((dst_height, dst_width), nodata, dtype=np.float32)
    reproject(
        source=vf_clean, destination=vf_warped,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata, dst_nodata=nodata,
    )
    vf_mask = vf_warped != nodata
    vf_warped[vf_mask] = np.clip(vf_warped[vf_mask], 0.0, 1.0)

    # Write 11-band TIF (10 spectral + valid_fraction)
    with rasterio.open(
        path, "w", driver="GTiff", height=dst_height, width=dst_width,
        count=n_spectral + 1, dtype="float32", crs=dst_crs,
        transform=dst_transform, compress="lzw", nodata=nodata,
    ) as dst:
        for i in range(n_spectral):
            dst.write(warped[i], i + 1)
            dst.set_band_description(i + 1, SENTINEL_BANDS[i])
        dst.write(vf_warped, n_spectral + 1)
        dst.set_band_description(n_spectral + 1, "VALID_FRACTION")

    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [{city.name}/{year}/{season}] Saved ({mb:.1f} MB)")


def stage_download():
    print(f"\n{'='*70}")
    print("STAGE 1: DOWNLOAD SENTINEL-2 COMPOSITES")
    print(f"{'='*70}")
    total = len(CITIES) * len(ALL_YEARS) * len(SEASONS)
    done = 0
    for city in CITIES:
        for year in ALL_YEARS:
            for season in SEASONS:
                for attempt in range(3):
                    try:
                        download_season(city, year, season)
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(f"  RETRY {attempt+1}/2: {city.name}/{year}/{season} ({e})")
                            time.sleep(30)
                        else:
                            print(f"  FAILED after 3 attempts: {city.name}/{year}/{season}")
                            print(f"    Error: {e}")
                done += 1
    print(f"\n[{ts()}] Download S2 stage complete ({done} TIFs).")


# ===========================================================================
# STAGE 1b: DOWNLOAD SENTINEL-1 SAR
# ===========================================================================

SAR_BANDS = ["vv", "vh"]
SAR_NODATA = -9999

def download_sar_season(city: CityConfig, year: int, season: str):
    """Download one Sentinel-1 GRD composite for a city-year-season.

    Uses rasterio WarpedVRT to reproject GCP-referenced S1 GRD scenes
    (Planetary Computer S1 GRD has no CRS/transform — only GCPs).

    Downloads VV/VH bands, converts linear power to dB, clamps to
    [-30, 0] dB, then scales to [0, 1]. Output: 2-band GeoTIFF aligned
    to the city anchor grid.
    """
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import reproject

    path = os.path.join(city_raw_dir(city),
                        f"sentinel1_{city.name}_{year}_{season}.tif")
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [SAR {city.name}/{year}/{season}] Exists ({mb:.1f} MB)")
        return

    anchor_path = city_anchor_path(city)
    with rasterio.open(anchor_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    bbox = city.bbox
    target_epsg = city.epsg
    nodata = SAR_NODATA

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    start_date = f"{year}-{SEASON_DATES[season][0]}"
    end_date = f"{year}-{SEASON_DATES[season][1]}"

    # Search for ascending orbit IW GRD
    items = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "sar:instrument_mode": {"eq": "IW"},
            "sat:orbit_state": {"eq": "ascending"},
        },
    ).item_collection()

    if len(items) < 3:
        items_desc = catalog.search(
            collections=["sentinel-1-grd"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"sar:instrument_mode": {"eq": "IW"}},
        ).item_collection()
        if len(items_desc) > len(items):
            items = items_desc

    n_scenes = len(items)
    if n_scenes == 0:
        print(f"  [SAR {city.name}/{year}/{season}] WARNING: No S1 scenes!")
        return
    print(f"  [SAR {city.name}/{year}/{season}] {n_scenes} scenes, "
          f"reading directly...")

    # Read each scene with WarpedVRT (handles GCPs automatically)
    all_vv = []
    all_vh = []
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="1",
        VSI_CACHE=True,
        VSI_CACHE_SIZE=5000000,
    )

    with env:
        for i, item in enumerate(items):
            vv_asset = item.assets.get("vv")
            vh_asset = item.assets.get("vh")
            if not vv_asset or not vh_asset:
                continue

            try:
                # S1 GRD scenes have CRS=None with GCPs in EPSG:4326.
                # Two-step WarpedVRT is needed:
                #   1) Regularize GCPs → EPSG:4326 grid
                #   2) Reproject 4326 → anchor CRS (e.g. EPSG:32632)
                # Single-step WarpedVRT silently produces all-zeros.

                # Read VV
                with rasterio.open(vv_asset.href) as src:
                    gcp_crs = src.gcps[1] if src.gcps[0] else src.crs
                    with WarpedVRT(src, src_crs=gcp_crs,
                                   crs=gcp_crs) as vrt_4326:
                        with WarpedVRT(
                            vrt_4326,
                            crs=dst_crs,
                            transform=dst_transform,
                            width=dst_width,
                            height=dst_height,
                            resampling=Resampling.bilinear,
                        ) as vrt:
                            vv_data = vrt.read(1).astype(np.float32)

                # Read VH
                with rasterio.open(vh_asset.href) as src:
                    gcp_crs = src.gcps[1] if src.gcps[0] else src.crs
                    with WarpedVRT(src, src_crs=gcp_crs,
                                   crs=gcp_crs) as vrt_4326:
                        with WarpedVRT(
                            vrt_4326,
                            crs=dst_crs,
                            transform=dst_transform,
                            width=dst_width,
                            height=dst_height,
                            resampling=Resampling.bilinear,
                        ) as vrt:
                            vh_data = vrt.read(1).astype(np.float32)

                # Zero means nodata in S1 GRD (uint16 source)
                vv_data[vv_data == 0] = np.nan
                vh_data[vh_data == 0] = np.nan

                all_vv.append(vv_data)
                all_vh.append(vh_data)

                if (i + 1) % 5 == 0 or i == n_scenes - 1:
                    print(f"    Read {i+1}/{n_scenes} scenes...")

            except Exception as e:
                print(f"    Scene {i} failed: {e}")
                continue

    if not all_vv:
        print(f"  [SAR {city.name}/{year}/{season}] ERROR: No scenes readable!")
        return

    print(f"  [SAR {city.name}/{year}/{season}] Computing median from "
          f"{len(all_vv)} scenes...")

    # Stack and take median
    vv_stack = np.stack(all_vv, axis=0)
    vh_stack = np.stack(all_vh, axis=0)
    del all_vv, all_vh

    vv_median = np.nanmedian(vv_stack, axis=0)
    vh_median = np.nanmedian(vh_stack, axis=0)
    del vv_stack, vh_stack

    composite = np.stack([vv_median, vh_median], axis=0)  # [2, H, W]
    del vv_median, vh_median

    # Scale raw uint16 DN (amplitude) values to [0, 1].
    # S1 GRD from Planetary Computer stores raw digital numbers:
    #   p1~55, p50~155, p99~910 (uint16 range).
    # These are NOT calibrated σ0, so don't do 10*log10.
    # We clip to [0, MAX_DN] and linearly scale to [0, 1].
    MAX_DN = 2000.0  # generous clip — p99.9 is ~1500
    valid_mask = np.isfinite(composite) & (composite > 0)
    composite_scaled = np.full_like(composite, nodata)
    composite_scaled[valid_mask] = np.clip(
        composite[valid_mask] / MAX_DN, 0.0, 1.0
    )

    # Write 2-band TIF
    n_sar = len(SAR_BANDS)
    with rasterio.open(
        path, "w", driver="GTiff", height=dst_height, width=dst_width,
        count=n_sar, dtype="float32", crs=dst_crs,
        transform=dst_transform, compress="lzw", nodata=nodata,
    ) as dst:
        for i in range(n_sar):
            dst.write(composite_scaled[i], i + 1)
            dst.set_band_description(i + 1, SAR_BANDS[i].upper())

    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [SAR {city.name}/{year}/{season}] Saved ({mb:.1f} MB)")


def stage_download_sar():
    print(f"\n{'='*70}")
    print("STAGE 1b: DOWNLOAD SENTINEL-1 SAR COMPOSITES")
    print(f"{'='*70}")
    total = len(CITIES) * len(ALL_YEARS) * len(SEASONS)
    done = 0
    for city in CITIES:
        for year in ALL_YEARS:
            for season in SEASONS:
                for attempt in range(3):
                    try:
                        download_sar_season(city, year, season)
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(f"  RETRY {attempt+1}/2: SAR {city.name}/"
                                  f"{year}/{season} ({e})")
                            time.sleep(30)
                        else:
                            print(f"  FAILED SAR after 3 attempts: "
                                  f"{city.name}/{year}/{season}")
                            print(f"    Error: {e}")
                done += 1
    print(f"\n[{ts()}] SAR download stage complete ({done} TIFs).")


# ===========================================================================
# STAGE 2: WORLDCOVER LABELS
# ===========================================================================

def download_worldcover_tile(tile: str, year: int) -> str:
    """Download a WorldCover tile from ESA S3 (if not cached)."""
    version = "v100" if year == 2020 else "v200"
    filename = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    path = os.path.join(WC_TILES_DIR, filename)
    if os.path.exists(path):
        return path

    url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
           f"{version}/{year}/map/{filename}")
    print(f"  Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, path)
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Downloaded: {filename} ({mb:.1f} MB)")
    except Exception as e:
        print(f"  ERROR: WorldCover download failed: {e}")
        if os.path.exists(path):
            os.remove(path)
        return ""
    return path


def reproject_worldcover_to_anchor(wc_paths, anchor_path):
    """Reproject WorldCover from EPSG:4326 to city anchor grid.

    wc_paths can be a single path (str) or a list of paths.  When
    multiple tiles are provided the results are merged by taking the
    maximum (non-zero) pixel value per location.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    if isinstance(wc_paths, str):
        wc_paths = [wc_paths]

    with rasterio.open(anchor_path) as ref:
        anchor = {
            "crs": ref.crs, "transform": ref.transform,
            "width": ref.width, "height": ref.height,
        }

    dst_array = np.zeros((anchor["height"], anchor["width"]), dtype=np.uint8)
    for wc_path in wc_paths:
        tmp = np.zeros_like(dst_array)
        with rasterio.open(wc_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=tmp,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=anchor["transform"], dst_crs=anchor["crs"],
                dst_nodata=0, resampling=Resampling.nearest,
            )
        # Merge: keep non-zero pixels from each tile
        mask = (dst_array == 0) & (tmp > 0)
        dst_array[mask] = tmp[mask]
    return dst_array, anchor["width"] // GRID_PX, anchor["height"] // GRID_PX


def aggregate_labels(wc_array, n_cols, n_rows):
    """Compute class proportions per 100m cell from WorldCover 10m pixels."""
    total_px = GRID_PX * GRID_PX
    records = []
    cell_id = 0
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            r0, c0 = row_idx * GRID_PX, col_idx * GRID_PX
            patch = wc_array[r0:r0 + GRID_PX, c0:c0 + GRID_PX]
            proportions = np.zeros(N_CLASSES, dtype=np.float32)
            mapped = 0
            for wc_code, our_class in WC_CLASS_MAP.items():
                count = int(np.sum(patch == wc_code))
                proportions[our_class] += count
                mapped += count
            if total_px > 0:
                proportions /= total_px
            # Normalize so rows sum to 1.0 (handles remaining unmapped pixels)
            row_sum = proportions.sum()
            if row_sum > 0:
                proportions /= row_sum
            coverage = mapped / total_px if total_px > 0 else 0.0
            record = {
                "cell_id": cell_id,
                "mapped_pixels": mapped,
                "coverage": float(coverage),
            }
            for i, name in enumerate(CLASS_NAMES):
                record[name] = float(proportions[i])
            records.append(record)
            cell_id += 1
    return pd.DataFrame(records)


def _wc_tiles_for_bbox(bbox):
    """Compute all ESA WorldCover 3°×3° tile IDs covering a WGS84 bbox.

    ESA tile IDs use the LOWER-LEFT corner, e.g. N48E009 covers
    lat [48,51), lon [9,12).  For negative longitudes the convention
    is W with the absolute value of the lower-left corner.
    """
    import math
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
            tile = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            tiles.append(tile)
    return tiles


def _get_wc_mosaic(city: CityConfig, year: int):
    """Download all needed WC tiles for a city and return list of paths."""
    tiles = _wc_tiles_for_bbox(city.bbox)
    paths = []
    for tile in tiles:
        p = download_worldcover_tile(tile, year)
        if p:
            paths.append(p)
    return paths


def create_labels(city: CityConfig):
    """Create labels for a city from WorldCover tiles."""
    for year in WC_YEARS:
        path = city_labels_path(city, year)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"  [{city.name}/{year}] Labels exist ({len(df)} cells)")
            continue

        wc_paths = _get_wc_mosaic(city, year)
        if not wc_paths:
            print(f"  [{city.name}/{year}] WARNING: No WorldCover available")
            continue

        anchor = city_anchor_path(city)
        print(f"  [{city.name}/{year}] Reprojecting WorldCover ({len(wc_paths)} tile(s))...")
        wc_array, n_cols, n_rows = reproject_worldcover_to_anchor(
            wc_paths, anchor)
        print(f"  [{city.name}/{year}] Aggregating {n_cols}x{n_rows}"
              f"={n_cols*n_rows} cells...")
        labels_df = aggregate_labels(wc_array, n_cols, n_rows)

        # Print summary
        for name in CLASS_NAMES:
            col = labels_df[name]
            print(f"    {name:<15} mean={col.mean():.3f}")
        labels_df.to_parquet(path, index=False)
        print(f"  [{city.name}/{year}] Saved: {path}")


def stage_labels():
    print(f"\n{'='*70}")
    print("STAGE 2: WORLDCOVER LABELS")
    print(f"{'='*70}")
    for city in CITIES:
        create_labels(city)
    print(f"  [OK] All labels ready.")


# ===========================================================================
# STAGE 3: EXTRACT FEATURES (Rust CLI)
# ===========================================================================

def stage_extract():
    print(f"\n{'='*70}")
    print("STAGE 3: EXTRACT FEATURES (Rust CLI)")
    print(f"{'='*70}")

    if not os.path.exists(TERRAPULSE_BIN):
        print(f"  ERROR: Rust binary not found: {TERRAPULSE_BIN}")
        print(f"  Build with: cargo build --release -p terrapulse")
        return

    for city in CITIES:
        year_pairs = " ".join(f"{y}_{y+1}" for y in range(2020, 2025))
        raw = city_raw_dir(city)
        feat = city_features_dir(city)

        # Check if all already extracted
        all_done = all(
            os.path.exists(
                os.path.join(feat, f"features_rust_{y}_{y+1}.parquet"))
            for y in range(2020, 2025)
        )
        if all_done:
            print(f"  [{city.name}] All features already extracted -- skip")
            continue

        print(f"  [{city.name}] Extracting features...")
        cmd = [
            TERRAPULSE_BIN, "extract",
            "--year-pairs", year_pairs,
            "--region", city.name,
            "--raw-dir", raw,
            "--features-dir", feat,
        ]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [{city.name}] ERROR: {result.stderr[-500:]}")
        else:
            print(f"  [{city.name}] Done in {elapsed:.1f}s")
            # Print last few lines of stdout
            for line in result.stdout.strip().split("\n")[-3:]:
                print(f"    {line}")

    print(f"\n[{ts()}] Extract stage complete.")


# ===========================================================================
# STAGE 4: TRAIN MODELS
# ===========================================================================

# Feature selection (same logic as run_overnight_pipeline.py)
_BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                  "B11", "B12"}
_INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI2", "CRI1", "MNDWI", "GNDVI", "NDTI", "IRECI", "TC",
}


def build_bi_lbp(feature_cols):
    """Feature selection: bands + indices + LBP + SAR + pheno."""
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in _BAND_PREFIXES or prefix in _INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
        elif "_pheno_" in col:
            selected.append(i)
        # V5: SAR features
        elif prefix == "SAR":
            selected.append(i)
    return sorted(set(selected))





def _discover_feature_cols():
    """Read the column schema from the first available training parquet.

    Returns the full list of numeric feature column names (excludes control
    cols like cell_id, valid_fraction, etc.) without loading any data.
    """
    from pandas.api.types import is_numeric_dtype

    for city in TRAIN_CITIES:
        feat_path = os.path.join(city_features_dir(city),
                                 "features_rust_2020_2021.parquet")
        if os.path.exists(feat_path):
            import pyarrow.parquet as pq
            schema = pq.read_schema(feat_path)
            # Filter to numeric feature columns
            numeric_types = {'float', 'double', 'int32', 'int64', 'float32',
                             'float64'}
            feature_cols = []
            for f in schema:
                type_str = str(f.type).lower()
                is_num = any(t in type_str for t in numeric_types)
                if is_num and f.name not in CONTROL_COLS:
                    feature_cols.append(f.name)
            return feature_cols
    raise RuntimeError("No training parquet found to discover columns")


def _load_city_arrays(city, columns):
    """Load only the specified columns from a city parquet as float32 array.

    Returns (X, n_cells) or None if missing.
    """
    feat_path = os.path.join(city_features_dir(city),
                             "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        return None

    available = set(pd.read_parquet(feat_path, columns=["cell_id"]).columns)
    # Read only the needed columns
    cols_to_read = [c for c in columns if c != "cell_id"]
    df = pd.read_parquet(feat_path, columns=cols_to_read)
    arr = np.nan_to_num(df.values.astype(np.float32), 0.0)
    n_cells = len(df)
    del df
    return arr, n_cells






def load_nuremberg_validation_data(mlp_cols):
    """Load Nuremberg features + labels as validation set."""
    test_city = TEST_CITIES[0]  # Nuremberg
    feat_path = os.path.join(city_features_dir(test_city),
                             "features_rust_2020_2021.parquet")
    labels_path = city_labels_path(test_city, 2021)

    if not os.path.exists(feat_path) or not os.path.exists(labels_path):
        raise RuntimeError(f"Missing Nuremberg data: {feat_path} / {labels_path}")

    feats = pd.read_parquet(feat_path)
    labels = pd.read_parquet(labels_path)

    available = [c for c in mlp_cols if c in feats.columns]
    X_val = np.nan_to_num(feats[available].values.astype(np.float32), 0.0)
    y_val = labels[CLASS_NAMES].values.astype(np.float32)

    print(f"  Nuremberg validation: {len(X_val)} cells")
    return X_val, y_val


def train_production_mlp(X_train, y_train, X_val, y_val,
                         n_features, device,
                         seed_offset=0):
    """
    Production MLP training with stratified batching and checkpoints.

    Key differences from train_model:
      - Stratified batch sampling (balanced by class + city)
      - Longer patience (10K steps)
      - Checkpoints saved every CHECKPOINT_EVERY epochs
      - Nuremberg as validation
    """
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scripts.run_mlp_overnight_v4 import (
        build_model, _cfg, normalize_targets, soft_cross_entropy,
        cosine_warmup_scheduler,
    )

    actual_seed = SEED + seed_offset
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    cfg = _cfg(0, "bi_LBP", "plain", "silu", MLP_DEPTH, MLP_WIDTH, "batchnorm",
               dropout=0.30, input_dropout=0.10, weight_decay=1e-3)
    # Note: V5 uses 3072w (vs V4's 2048w) for expanded SAR features
    net = build_model(cfg, n_features, device)

    y_norm = normalize_targets(y_train)
    y_val_norm = normalize_targets(y_val)

    X_trn_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_trn_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_norm, dtype=torch.float32, device=device)

    use_amp = device == "cuda"
    wd = cfg["weight_decay"]
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg["lr"], weight_decay=wd, fused=use_amp)
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg["lr"], weight_decay=wd)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_size = 2048
    n = len(X_train)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = MAX_EPOCHS * steps_per_epoch
    scheduler = cosine_warmup_scheduler(
        optimizer, steps_per_epoch * 3, total_steps)

    patience_epochs = max(math.ceil(PATIENCE_STEPS / steps_per_epoch), 5)
    min_epochs = max(math.ceil(MIN_STEPS / steps_per_epoch), 3)

    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
    loss_fn = soft_cross_entropy

    best_val = float("inf")
    best_state = None
    wait = 0
    n_epochs_done = 0
    checkpoint_dir = os.path.join(MODELS_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    rng = np.random.RandomState(actual_seed)

    for epoch in range(MAX_EPOCHS):
        net.train()
        epoch_loss = 0.0
        n_batches = 0

        # Standard random permutation batching
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            xb = X_trn_t[idx_t]
            yb = y_trn_t[idx_t]
            if has_bn and xb.size(0) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logp = net(xb)
                loss = loss_fn(logp, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate on Nuremberg
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(X_val_t), y_val_t).item()

        n_epochs_done = epoch + 1

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items()}
            wait = 0
            marker = " *BEST*"
        else:
            wait += 1
            marker = ""

        if epoch % 10 == 0 or marker:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"    Epoch {epoch:>4}: train_loss={avg_loss:.5f} "
                  f"val_loss={val_loss:.5f} wait={wait}{marker}")

        # Save checkpoint
        if n_epochs_done % CHECKPOINT_EVERY == 0:
            cp_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_seed{seed_offset}_epoch{n_epochs_done}.pt")
            torch.save(net.state_dict(), cp_path)

        # Early stopping
        if (n_epochs_done >= min_epochs
                and wait >= patience_epochs):
            print(f"    Early stopping at epoch {n_epochs_done} "
                  f"(patience={patience_epochs})")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    del X_trn_t, y_trn_t, X_val_t, y_val_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return net, n_epochs_done, best_val


def stage_train():
    print(f"\n{'='*70}")
    print("STAGE 4: TRAIN MLP")
    print(f"{'='*70}")

    from sklearn.preprocessing import StandardScaler

    # Check if already trained
    mlp_done = os.path.exists(os.path.join(MODELS_DIR, "mlp_meta.json"))
    if mlp_done:
        print("  MLP already trained -- skip")
        return

    # Discover feature columns from parquet schema (no data loaded)
    print(f"  [{ts()}] Discovering feature columns...")
    full_feature_cols = _discover_feature_cols()

    # Build feature selection list (MLP only)
    mlp_idx = build_bi_lbp(full_feature_cols)
    mlp_cols = [full_feature_cols[i] for i in mlp_idx]
    n_mlp = len(mlp_cols)
    print(f"  MLP features: {n_mlp}")

    # Load data per city
    print(f"  [{ts()}] Loading training data ({n_mlp} cols)...")
    X_parts = []
    y_parts = []
    total_cells = 0

    for city in TRAIN_CITIES:
        result = _load_city_arrays(city, mlp_cols)
        if result is None:
            print(f"  WARNING: Missing features for {city.name} -- skip")
            continue

        X_city, n_cells = result

        # Load labels
        labels_path = city_labels_path(city, 2021)
        if not os.path.exists(labels_path):
            print(f"  WARNING: Missing labels for {city.name} -- skip")
            del X_city
            continue
        labels = pd.read_parquet(labels_path)
        y_city = labels[CLASS_NAMES].values.astype(np.float32)
        del labels

        X_parts.append(X_city)
        y_parts.append(y_city)
        total_cells += n_cells
        print(f"  [{city.name}] {n_cells} cells loaded (total={total_cells})")

    if not X_parts:
        raise RuntimeError("No training data available!")

    # Concatenate all cities
    X_mlp = np.concatenate(X_parts, axis=0)
    del X_parts
    y = np.concatenate(y_parts, axis=0)
    del y_parts

    n_total = len(y)
    print(f"  Total training samples: {n_total}")
    print(f"  Labels shape: {y.shape}")

    # --- Train MLP ---
    if not mlp_done:
        import torch
        from scripts.run_mlp_overnight_v4 import _predict_batched

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  [{ts()}] Training MLP production model on {device}")
        print(f"  Architecture: {MLP_DEPTH}L x {MLP_WIDTH}w")
        print(f"  Seeds: {N_SEEDS}, Patience: {PATIENCE_STEPS} steps")

        # Load Nuremberg validation data
        X_val_raw, y_val = load_nuremberg_validation_data(mlp_cols)

        # Fit scaler on training data
        scaler = StandardScaler()
        X_trn_scaled = scaler.fit_transform(X_mlp).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)

        with open(os.path.join(MODELS_DIR, "mlp_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        seed_offsets = [MLP_SEED_OFFSET + i * 100 for i in range(N_SEEDS)]
        seed_metrics = []

        for seed_idx, seed_offset in enumerate(seed_offsets):
            print(f"\n  === MLP Seed {seed_idx} (offset={seed_offset}) ===")
            t0 = time.time()
            trained_net, n_epochs, best_val = train_production_mlp(
                X_trn_scaled, y, X_val_scaled, y_val,
                n_mlp, device,
                seed_offset=seed_offset)
            elapsed = time.time() - t0

            # Save model
            torch.save(trained_net.state_dict(),
                       os.path.join(MODELS_DIR, f"mlp_seed{seed_idx}.pt"))

            # Evaluate on Nuremberg
            preds = _predict_batched(
                trained_net, torch.tensor(X_val_scaled), device)
            from src.models.evaluation import evaluate_model
            summary, _ = evaluate_model(y_val, preds, CLASS_NAMES)
            r2 = summary["r2_uniform"]
            mae = summary["mae_mean_pp"]
            print(f"  Seed {seed_idx}: R2={r2:.4f} MAE={mae:.2f}pp "
                  f"epochs={n_epochs} ({elapsed:.0f}s)")
            for cn in CLASS_NAMES:
                print(f"    {cn:<15} R2={summary[f'r2_{cn}']:.4f}")

            seed_metrics.append({
                "seed": seed_idx, "seed_offset": seed_offset,
                "r2_nuremberg": r2, "mae_nuremberg": mae,
                "n_epochs": n_epochs, "best_val_loss": best_val,
                "time_s": round(elapsed, 1),
            })

            del trained_net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate 3-seed ensemble on Nuremberg
        print(f"\n  [{ts()}] Evaluating 3-seed ensemble on Nuremberg...")
        preds_sum = np.zeros((len(X_val_scaled), N_CLASSES), dtype=np.float32)
        for seed_idx in range(N_SEEDS):
            from scripts.run_mlp_overnight_v4 import build_model, _cfg
            cfg = _cfg(0, "bi_LBP", "plain", "silu",
                       MLP_DEPTH, MLP_WIDTH, "batchnorm",
                       dropout=0.30, input_dropout=0.10)
            net = build_model(cfg, n_mlp, device)
            net.load_state_dict(torch.load(
                os.path.join(MODELS_DIR, f"mlp_seed{seed_idx}.pt"),
                map_location=device, weights_only=True))
            net.eval()
            preds_sum += _predict_batched(
                net, torch.tensor(X_val_scaled), device)
            del net

        ensemble_preds = preds_sum / N_SEEDS
        ens_summary, _ = evaluate_model(y_val, ensemble_preds, CLASS_NAMES)
        print(f"  Ensemble: R2={ens_summary['r2_uniform']:.4f} "
              f"MAE={ens_summary['mae_mean_pp']:.2f}pp")
        for cn in CLASS_NAMES:
            print(f"    {cn:<15} R2={ens_summary[f'r2_{cn}']:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mlp_meta = {
            "model": "MLP", "training": "production_v5",
            "arch": {"depth": MLP_DEPTH, "width": MLP_WIDTH},
            "cities": [c.name for c in TRAIN_CITIES],
            "feature_cols": mlp_cols, "n_features": n_mlp,
            "n_seeds": N_SEEDS,
            "validation_city": "nuremberg",
            "r2_nuremberg_ensemble": float(ens_summary["r2_uniform"]),
            "mae_nuremberg_ensemble": float(ens_summary["mae_mean_pp"]),
            "per_class_r2": {cn: float(ens_summary[f"r2_{cn}"])
                            for cn in CLASS_NAMES},
            "seed_metrics": seed_metrics,
        }
        with open(os.path.join(MODELS_DIR, "mlp_meta.json"), "w") as f:
            json.dump(mlp_meta, f, indent=2)

    print(f"\n[{ts()}] Train stage complete.")


# ===========================================================================
# STAGE 5: PREDICT
# ===========================================================================




def predict_with_mlp_ensemble(X, n_features, device):
    """Predict with 3-seed MLP ensemble (no folds)."""
    import torch
    from scripts.run_mlp_overnight_v4 import build_model, _cfg, _predict_batched

    with open(os.path.join(MODELS_DIR, "mlp_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X).astype(np.float32)
    preds_sum = np.zeros((len(X), N_CLASSES), dtype=np.float32)

    for seed_idx in range(N_SEEDS):
        model_path = os.path.join(MODELS_DIR, f"mlp_seed{seed_idx}.pt")
        if not os.path.exists(model_path):
            continue

        cfg = _cfg(0, "bi_LBP", "plain", "silu",
                   MLP_DEPTH, MLP_WIDTH, "batchnorm",
                   dropout=0.30, input_dropout=0.10)
        net = build_model(cfg, n_features, device)
        net.load_state_dict(torch.load(model_path, map_location=device,
                                       weights_only=True))
        net.eval()
        preds = _predict_batched(net, torch.tensor(X_scaled), device)
        preds_sum += preds
        del net

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds_sum / N_SEEDS


def stage_predict():
    print(f"\n{'='*70}")
    print("STAGE 5: PREDICT ALL CITIES (MLP only)")
    print(f"{'='*70}")

    # Load model metadata
    with open(os.path.join(MODELS_DIR, "mlp_meta.json")) as f:
        mlp_meta = json.load(f)
    mlp_cols = mlp_meta["feature_cols"]
    n_mlp = mlp_meta["n_features"]

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    print(f"  Device: {device}")

    year_pairs = [(y, y + 1) for y in range(2020, 2025)]

    for city in CITIES:
        print(f"\n  === {city.name.upper()} "
              f"{'(VALIDATION)' if city.is_test else ''} ===")
        pred_dir = city_predictions_dir(city)

        for prev_year, curr_year in year_pairs:
            tag = f"{prev_year}_{curr_year}"
            mlp_path = os.path.join(pred_dir,
                                    f"predictions_mlp_{tag}.parquet")

            if os.path.exists(mlp_path):
                continue

            feat_path = os.path.join(city_features_dir(city),
                                     f"features_rust_{tag}.parquet")
            if not os.path.exists(feat_path):
                print(f"  [{tag}] Missing features -- skip")
                continue

            merged = pd.read_parquet(feat_path)
            cell_ids = merged["cell_id"].values
            merged_cols_set = set(merged.columns)

            X_mlp = np.nan_to_num(
                merged[[c for c in mlp_cols if c in merged_cols_set]]
                .values.astype(np.float32), 0.0)

            preds_all = predict_with_mlp_ensemble(
                X_mlp, n_mlp, device)

            mlp_df = pd.DataFrame({"cell_id": cell_ids})
            for ci, cn in enumerate(CLASS_NAMES):
                mlp_df[f"{cn}_pred"] = preds_all[:, ci]
            mlp_df["prev_year"] = prev_year
            mlp_df["curr_year"] = curr_year
            mlp_df.to_parquet(mlp_path, index=False)

            print(f"  [{city.name}/{tag}] Done")

    print(f"\n[{ts()}] Predict stage complete.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-city V5 pipeline (SAR integration)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-download-sar", action="store_true",
                        help="Skip SAR download stage only")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")

    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--only-cities", nargs="+", default=None,
                        help="Only process these cities (for partial runs)")
    args = parser.parse_args()

    # Filter cities if requested
    global CITIES, TRAIN_CITIES, TEST_CITIES
    if args.only_cities:
        names = set(args.only_cities)
        CITIES = [c for c in CITIES if c.name in names]
        TRAIN_CITIES = [c for c in CITIES if not c.is_test]
        TEST_CITIES = [c for c in CITIES if c.is_test]

    t_total = time.time()
    print(f"[{ts()}] Multi-City Pipeline V5 (SAR Integration) starting")
    print(f"  Cities: {[c.name for c in CITIES]}")
    print(f"  Training: {[c.name for c in TRAIN_CITIES]}")
    print(f"  Validation: {[c.name for c in TEST_CITIES]}")
    print(f"  MLP: {MLP_DEPTH}L x {MLP_WIDTH}w (wider for SAR), "
          f"{N_SEEDS} seed(s), dropout=0.30, wd=1e-3")
    print(f"  Patience: {PATIENCE_STEPS} steps, checkpoints every "
          f"{CHECKPOINT_EVERY} epochs")

    ensure_all_dirs()

    # Stage 0: Anchors
    stage_anchors()

    # Stage 1: Download S2
    if not args.skip_download:
        stage_download()
    else:
        print("\n  S2 DOWNLOAD skipped")

    # Stage 1b: Download SAR
    if not args.skip_download and not args.skip_download_sar:
        stage_download_sar()
    else:
        print("\n  SAR DOWNLOAD skipped")

    # Stage 2: Labels
    stage_labels()

    # Stage 3: Extract
    if not args.skip_extract:
        stage_extract()
    else:
        print("\n  EXTRACT skipped")

    # Stage 4: Train MLP
    if not args.skip_train:
        stage_train()
    else:
        print("\n  TRAIN skipped")

    # Stage 5: Predict
    if not args.skip_predict:
        stage_predict()
    else:
        print("\n  PREDICT skipped")

    total = time.time() - t_total
    hours = int(total // 3600)
    mins = int((total % 3600) // 60)
    print(f"\n{'='*70}")
    print(f"MULTI-CITY PIPELINE V5 COMPLETE in {hours}h {mins}m")
    print(f"  Data:        {CITIES_DIR}")
    print(f"  Models:      {MODELS_DIR}")
    for c in CITIES:
        n_pred = len([f for f in os.listdir(city_predictions_dir(c))
                      if f.endswith(".parquet")])
        print(f"  {c.name:<15} {n_pred} prediction files")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

