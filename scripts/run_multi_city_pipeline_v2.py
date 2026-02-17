#!/usr/bin/env python3
"""
Multi-City Pipeline V2: Download -> Anchor -> Labels -> Extract -> Train -> Predict.

Train on 14 European cities across 7 countries, predict on all 15
(Nuremberg = held-out test). Includes MLP architecture sweep
(baseline, deep, wide, deep_wide).

Cities (training):
    v1: Bremen, Hamburg, Düsseldorf, Leipzig, Rostock, Amsterdam
    v2: + Almería, Murcia, Amiens, Magdeburg, Ulm, Salzburg, Schwerin, Malmö

City (test):
    Nuremberg

Usage:
    .venv/Scripts/python.exe scripts/run_multi_city_pipeline_v2.py
    # Skip already-done stages:
    .venv/Scripts/python.exe scripts/run_multi_city_pipeline_v2.py --skip-download

Runtime: ~5-6 hours (downloads ~2h, training ~3h with arch sweep)
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

WC_CLASS_MAP = {10: 0, 30: 1, 90: 1, 40: 2, 50: 3, 60: 4, 80: 5}
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)

SEED = 42
N_FOLDS = 5
N_SEEDS = 3  # multi-seed MLP ensemble

# Architecture sweep configs
MLP_ARCHS = [
    {"name": "baseline",  "depth": 5, "width": 1024},
    {"name": "deep",      "depth": 7, "width": 1024},
    {"name": "wide",      "depth": 5, "width": 2048},
    {"name": "deep_wide", "depth": 7, "width": 2048},
]

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
    # --- V1 training cities ---
    CityConfig("bremen",      [8.65, 53.00, 8.90, 53.14], 32632, "N51E006"),
    CityConfig("hamburg",     [9.80, 53.40, 10.15, 53.58], 32632, "N51E009"),
    CityConfig("duesseldorf", [6.70, 51.15, 6.90, 51.28], 32632, "N51E006"),
    CityConfig("leipzig",     [12.25, 51.27, 12.50, 51.40], 32633, "N51E012"),
    CityConfig("rostock",     [12.00, 54.05, 12.20, 54.18], 32633, "N54E012"),
    CityConfig("amsterdam",   [4.75, 52.30, 4.95, 52.40], 32631, "N51E003"),
    # --- V2 additions: bare/sparse boosters ---
    CityConfig("almeria",     [-2.55, 36.78, -2.30, 36.92], 32630, "N36W003"),
    CityConfig("murcia",      [-1.25, 37.92, -1.00, 38.06], 32630, "N36W003"),
    # --- V2 additions: cropland boosters ---
    CityConfig("amiens",      [2.15, 49.82, 2.42, 49.96], 32631, "N48E000"),
    CityConfig("magdeburg",   [11.50, 52.05, 11.75, 52.20], 32632, "N51E009"),
    CityConfig("ulm",         [9.85, 48.33, 10.10, 48.47], 32632, "N48E009"),
    # --- V2 additions: grassland booster ---
    CityConfig("salzburg",    [12.95, 47.73, 13.15, 47.87], 32633, "N45E012"),
    # --- V2 additions: diversity + water ---
    CityConfig("schwerin",    [11.30, 53.55, 11.55, 53.70], 32632, "N51E009"),
    CityConfig("malmo",       [12.90, 55.53, 13.15, 55.68], 32633, "N54E012"),
    # --- Test city ---
    CityConfig("nuremberg",   [10.95, 49.38, 11.20, 49.52], 32632, "N48E009",
               is_test=True),
]

TRAIN_CITIES = [c for c in CITIES if not c.is_test]
TEST_CITIES = [c for c in CITIES if c.is_test]

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
WC_TILES_DIR = os.path.join(CITIES_DIR, "worldcover_tiles")
MODELS_DIR = os.path.join(CITIES_DIR, "models_v2")


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
                download_season(city, year, season)
                done += 1
    print(f"\n[{ts()}] Download stage complete ({done} TIFs).")


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


def reproject_worldcover_to_anchor(wc_path: str, anchor_path: str):
    """Reproject WorldCover from EPSG:4326 to city anchor grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(anchor_path) as ref:
        anchor = {
            "crs": ref.crs, "transform": ref.transform,
            "width": ref.width, "height": ref.height,
        }

    dst_array = np.zeros((anchor["height"], anchor["width"]), dtype=np.uint8)
    with rasterio.open(wc_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst_array,
            src_transform=src.transform, src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=anchor["transform"], dst_crs=anchor["crs"],
            dst_nodata=0, resampling=Resampling.nearest,
        )
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


def create_labels(city: CityConfig):
    """Create labels for a city from WorldCover tiles."""
    for year in WC_YEARS:
        path = city_labels_path(city, year)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"  [{city.name}/{year}] Labels exist ({len(df)} cells)")
            continue

        wc_path = download_worldcover_tile(city.wc_tile, year)
        if not wc_path:
            print(f"  [{city.name}/{year}] WARNING: No WorldCover available")
            continue

        anchor = city_anchor_path(city)
        print(f"  [{city.name}/{year}] Reprojecting WorldCover...")
        wc_array, n_cols, n_rows = reproject_worldcover_to_anchor(
            wc_path, anchor)
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
    "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
}


def build_bi_lbp(feature_cols):
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in _BAND_PREFIXES or prefix in _INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
    return sorted(set(selected))


def build_tree_features(feature_cols):
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    novel = ["NDTI", "IRECI", "CRI1"]
    selected = []
    for c in feature_cols:
        if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"]):
            if not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr"):
                selected.append(c)
                continue
        if band_pat.match(c):
            selected.append(c)
            continue
        if c.startswith("TC_"):
            selected.append(c)
            continue
        for idx in novel:
            if c.startswith(f"{idx}_"):
                selected.append(c)
                break
    return selected


def load_multi_city_training_data():
    """Load and concatenate features + labels from all training cities."""
    from pandas.api.types import is_numeric_dtype

    all_features = []
    all_labels = []
    city_ids = []
    offset = 0

    for city in TRAIN_CITIES:
        # Load features for 2020_2021 year-pair
        feat_path = os.path.join(city_features_dir(city),
                                 "features_rust_2020_2021.parquet")
        if not os.path.exists(feat_path):
            print(f"  WARNING: Missing features for {city.name} -- skip")
            continue

        df = pd.read_parquet(feat_path)
        # Reindex cell_id to be globally unique
        df["cell_id"] = df["cell_id"] + offset
        df["city"] = city.name

        # Load labels (2021 = the "current year")
        labels_path = city_labels_path(city, 2021)
        if not os.path.exists(labels_path):
            print(f"  WARNING: Missing labels for {city.name} -- skip")
            continue
        labels = pd.read_parquet(labels_path)
        labels["cell_id"] = labels["cell_id"] + offset

        all_features.append(df)
        all_labels.append(labels)
        city_ids.append(city.name)
        offset += len(df)
        print(f"  [{city.name}] {len(df)} cells loaded (offset={offset})")

    if not all_features:
        raise RuntimeError("No training data available!")

    merged_features = pd.concat(all_features, ignore_index=True)
    merged_labels = pd.concat(all_labels, ignore_index=True)

    print(f"  Total training cells: {len(merged_features)} from "
          f"{len(city_ids)} cities")
    return merged_features, merged_labels, city_ids


def train_tree_model(X_train, y_train, fold_id):
    """Train LightGBM with best sweep config (strong_wide)."""
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor

    params = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        num_leaves=255, min_child_samples=20, reg_lambda=3.0,
        subsample=0.8, colsample_bytree=0.7, verbosity=-1,
        random_state=SEED + fold_id, n_jobs=-1,
    )
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    model.fit(X_train, y_train)
    return model


def train_mlp_model(X_train, y_train, X_val, y_val, fold_id, n_features,
                    device, seed_offset=0, depth=5, width=1024):
    """Train MLP with specified seed and architecture."""
    import torch
    from scripts.run_mlp_overnight_v4 import (
        build_model, _cfg, train_model, normalize_targets,
    )

    actual_seed = SEED + fold_id + seed_offset

    cfg = _cfg(0, "bi_LBP", "plain", "silu", depth, width, "batchnorm")

    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    net = build_model(cfg, n_features, device)

    X_trn_t = torch.tensor(X_train).to(device)
    y_trn_t = torch.tensor(normalize_targets(y_train)).to(device)
    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(normalize_targets(y_val)).to(device)

    n_epochs, best_val, trained_net = train_model(
        net, X_trn_t, y_trn_t, X_val_t, y_val_t,
        lr=cfg["lr"], weight_decay=1e-4,
        batch_size=2048, max_epochs=2000, patience_steps=5000,
        min_steps=2000, mixup_alpha=0, use_swa=False, use_cosine=True,
    )

    del X_trn_t, y_trn_t, X_val_t, y_val_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trained_net, n_epochs, best_val


def stage_train():
    print(f"\n{'='*70}")
    print("STAGE 4: TRAIN MODELS (multi-city)")
    print(f"{'='*70}")

    from pandas.api.types import is_numeric_dtype
    from sklearn.preprocessing import StandardScaler

    # Check if already trained
    tree_done = os.path.exists(os.path.join(MODELS_DIR, "tree_meta.json"))
    all_mlp_done = all(
        os.path.exists(os.path.join(MODELS_DIR, f"arch_{a['name']}",
                                    "mlp_meta.json"))
        for a in MLP_ARCHS
    )
    if tree_done and all_mlp_done:
        print("  All models already trained -- skip")
        return

    # Load multi-city data
    print(f"  [{ts()}] Loading multi-city training data...")
    merged, labels_df, city_ids = load_multi_city_training_data()

    # Feature selection
    full_feature_cols = [
        c for c in merged.columns
        if c not in CONTROL_COLS and c not in {"city"} and
        is_numeric_dtype(merged[c])
    ]

    mlp_idx = build_bi_lbp(full_feature_cols)
    mlp_cols = [full_feature_cols[i] for i in mlp_idx]
    tree_cols = build_tree_features(full_feature_cols)
    n_mlp = len(mlp_cols)
    n_tree = len(tree_cols)
    print(f"  MLP features: {n_mlp}")
    print(f"  Tree features: {n_tree}")

    X_mlp = np.nan_to_num(merged[mlp_cols].values.astype(np.float32), 0.0)
    X_tree = np.nan_to_num(merged[tree_cols].values.astype(np.float32), 0.0)
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    n_total = len(y)
    print(f"  Total samples: {n_total}")
    print(f"  Labels shape: {y.shape}")

    # Random 5-fold split (multi-city: real test = Nuremberg)
    rng = np.random.RandomState(SEED)
    fold_assignments = rng.randint(0, N_FOLDS, size=n_total)

    # --- Train LightGBM ---
    if not tree_done:
        print(f"\n  [{ts()}] Training LightGBM ({N_FOLDS} folds)...")
        tree_fold_metrics = []

        for fold_id in range(N_FOLDS):
            train_mask = fold_assignments != fold_id
            test_mask = fold_assignments == fold_id

            t0 = time.time()
            model = train_tree_model(
                X_tree[train_mask], y[train_mask], fold_id)
            y_pred = np.clip(model.predict(X_tree[test_mask]), 0, 100)

            from src.models.evaluation import evaluate_model
            summary, _ = evaluate_model(y[test_mask], y_pred, CLASS_NAMES)
            elapsed = time.time() - t0

            tree_fold_metrics.append({
                "fold": fold_id,
                "r2": summary["r2_uniform"],
                "mae": summary["mae_mean_pp"],
                "time_s": round(elapsed, 1),
            })

            # Save model
            with open(os.path.join(MODELS_DIR,
                      f"tree_fold_{fold_id}.pkl"), "wb") as f:
                pickle.dump(model, f)

            print(f"    Fold {fold_id}: R2={summary['r2_uniform']:.4f} "
                  f"MAE={summary['mae_mean_pp']:.2f}pp ({elapsed:.0f}s)")

        tree_r2 = np.mean([m["r2"] for m in tree_fold_metrics])
        tree_meta = {
            "model": "LightGBM", "training": "multi_city",
            "cities": [c.name for c in TRAIN_CITIES],
            "feature_cols": tree_cols, "n_features": n_tree,
            "r2_mean": float(tree_r2),
            "fold_metrics": tree_fold_metrics,
        }
        with open(os.path.join(MODELS_DIR, "tree_meta.json"), "w") as f:
            json.dump(tree_meta, f, indent=2)
        print(f"  Tree mean R2: {tree_r2:.4f}")

    # --- Train MLP (multi-arch, multi-seed) ---
    for arch in MLP_ARCHS:
        arch_name = arch["name"]
        arch_dir = os.path.join(MODELS_DIR, f"arch_{arch_name}")
        os.makedirs(arch_dir, exist_ok=True)

        mlp_done_arch = os.path.exists(
            os.path.join(arch_dir, "mlp_meta.json"))
        if mlp_done_arch:
            print(f"\n  [{ts()}] MLP arch '{arch_name}' already trained -- skip")
            continue

        import torch
        from scripts.run_mlp_overnight_v4 import _predict_batched

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  [{ts()}] Training MLP arch '{arch_name}' "
              f"(depth={arch['depth']}, width={arch['width']}) on {device}")
        print(f"  {N_SEEDS} seeds x {N_FOLDS} folds = "
              f"{N_SEEDS * N_FOLDS} models")

        mlp_fold_metrics = []
        seed_offsets = [0, 100, 200][:N_SEEDS]

        for fold_id in range(N_FOLDS):
            train_mask = fold_assignments != fold_id
            test_mask = fold_assignments == fold_id
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            # Split train into train/val (85/15)
            rng_fold = np.random.RandomState(SEED + fold_id)
            perm = rng_fold.permutation(len(train_indices))
            n_val = max(int(len(train_indices) * 0.15), 100)
            val_idx = train_indices[perm[:n_val]]
            trn_idx = train_indices[perm[n_val:]]

            scaler = StandardScaler()
            X_trn = scaler.fit_transform(X_mlp[trn_idx]).astype(np.float32)
            X_val = scaler.transform(X_mlp[val_idx]).astype(np.float32)
            X_tst = scaler.transform(X_mlp[test_indices]).astype(np.float32)

            # Save scaler (shared across seeds for same fold)
            with open(os.path.join(arch_dir,
                      f"mlp_scaler_{fold_id}.pkl"), "wb") as f:
                pickle.dump(scaler, f)

            fold_preds = np.zeros((len(test_indices), N_CLASSES),
                                  dtype=np.float32)

            for seed_idx, seed_offset in enumerate(seed_offsets):
                print(f"\n  --- {arch_name} Fold {fold_id} Seed {seed_idx} "
                      f"(offset={seed_offset}) ---")

                t0 = time.time()
                trained_net, n_epochs, best_val = train_mlp_model(
                    X_trn, y[trn_idx], X_val, y[val_idx],
                    fold_id, n_mlp, device, seed_offset=seed_offset,
                    depth=arch["depth"], width=arch["width"])
                elapsed = time.time() - t0

                # Save model
                torch.save(trained_net.state_dict(),
                           os.path.join(arch_dir,
                                        f"mlp_seed{seed_idx}_fold_{fold_id}.pt"))

                # Evaluate this seed
                preds = _predict_batched(
                    trained_net, torch.tensor(X_tst), device)
                fold_preds += preds

                from src.models.evaluation import evaluate_model
                summary, _ = evaluate_model(y[test_indices], preds,
                                            CLASS_NAMES)
                print(f"    R2={summary['r2_uniform']:.4f} "
                      f"MAE={summary['mae_mean_pp']:.2f}pp "
                      f"epochs={n_epochs} ({elapsed:.0f}s)")

                del trained_net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Evaluate seed ensemble for this fold
            fold_preds /= N_SEEDS
            ens_summary, _ = evaluate_model(y[test_indices], fold_preds,
                                            CLASS_NAMES)
            mlp_fold_metrics.append({
                "fold": fold_id,
                "r2_ensemble": ens_summary["r2_uniform"],
                "mae_ensemble": ens_summary["mae_mean_pp"],
                "n_seeds": N_SEEDS,
            })
            print(f"  {arch_name} Fold {fold_id} ensemble: "
                  f"R2={ens_summary['r2_uniform']:.4f} "
                  f"MAE={ens_summary['mae_mean_pp']:.2f}pp")

        mlp_r2 = np.mean([m["r2_ensemble"] for m in mlp_fold_metrics])
        mlp_meta = {
            "model": "MLP", "training": "multi_city_v2",
            "arch": arch,
            "cities": [c.name for c in TRAIN_CITIES],
            "feature_cols": mlp_cols, "n_features": n_mlp,
            "n_seeds": N_SEEDS, "r2_mean": float(mlp_r2),
            "fold_metrics": mlp_fold_metrics,
        }
        with open(os.path.join(arch_dir, "mlp_meta.json"), "w") as f:
            json.dump(mlp_meta, f, indent=2)
        print(f"  [{arch_name}] MLP ensemble mean R2: {mlp_r2:.4f}")

    print(f"\n[{ts()}] Train stage complete.")


# ===========================================================================
# STAGE 5: PREDICT
# ===========================================================================

def predict_with_tree(X, fold_id):
    path = os.path.join(MODELS_DIR, f"tree_fold_{fold_id}.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return np.clip(model.predict(X), 0, 100).astype(np.float32)


def predict_with_mlp_ensemble(X, fold_id, n_features, device, arch_dir, arch):
    """Predict with multi-seed MLP ensemble for one fold."""
    import torch
    from scripts.run_mlp_overnight_v4 import build_model, _cfg, _predict_batched

    with open(os.path.join(arch_dir,
              f"mlp_scaler_{fold_id}.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X).astype(np.float32)
    preds_sum = np.zeros((len(X), N_CLASSES), dtype=np.float32)

    for seed_idx in range(N_SEEDS):
        model_path = os.path.join(arch_dir,
                                  f"mlp_seed{seed_idx}_fold_{fold_id}.pt")
        if not os.path.exists(model_path):
            continue

        cfg = _cfg(0, "bi_LBP", "plain", "silu",
                   arch["depth"], arch["width"], "batchnorm")
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
    print("STAGE 5: PREDICT ALL CITIES")
    print(f"{'='*70}")

    # Load tree feature column list
    with open(os.path.join(MODELS_DIR, "tree_meta.json")) as f:
        tree_meta = json.load(f)
    tree_cols = tree_meta["feature_cols"]

    # Find best MLP architecture by R2
    best_arch = None
    best_r2 = -1
    all_archs = []
    for arch in MLP_ARCHS:
        arch_dir = os.path.join(MODELS_DIR, f"arch_{arch['name']}")
        meta_path = os.path.join(arch_dir, "mlp_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            r2 = meta["r2_mean"]
            all_archs.append((arch, r2, arch_dir, meta))
            print(f"  Arch '{arch['name']}': R2={r2:.4f}")
            if r2 > best_r2:
                best_r2 = r2
                best_arch = arch

    if best_arch is None:
        print("  ERROR: No trained MLP architectures found!")
        return

    print(f"  Best arch: '{best_arch['name']}' (R2={best_r2:.4f})")
    best_arch_dir = os.path.join(MODELS_DIR, f"arch_{best_arch['name']}")
    with open(os.path.join(best_arch_dir, "mlp_meta.json")) as f:
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
        print(f"\n  === {city.name.upper()} {'(TEST)' if city.is_test else ''} ===")
        pred_dir = city_predictions_dir(city)

        for prev_year, curr_year in year_pairs:
            tag = f"{prev_year}_{curr_year}"
            tree_path = os.path.join(pred_dir,
                                     f"predictions_tree_{tag}.parquet")
            mlp_path = os.path.join(pred_dir,
                                    f"predictions_mlp_{tag}.parquet")

            if os.path.exists(tree_path) and os.path.exists(mlp_path):
                continue

            feat_path = os.path.join(city_features_dir(city),
                                     f"features_rust_{tag}.parquet")
            if not os.path.exists(feat_path):
                print(f"  [{tag}] Missing features -- skip")
                continue

            merged = pd.read_parquet(feat_path)
            cell_ids = merged["cell_id"].values
            merged_cols_set = set(merged.columns)

            # Tree predictions
            if not os.path.exists(tree_path):
                X_tree = np.nan_to_num(
                    merged[[c for c in tree_cols if c in merged_cols_set]]
                    .values.astype(np.float32), 0.0)

                preds_all = np.zeros((len(cell_ids), N_CLASSES),
                                     dtype=np.float32)
                for fold_id in range(N_FOLDS):
                    preds_all += predict_with_tree(X_tree, fold_id)
                preds_all /= N_FOLDS
                # Normalize
                row_sums = preds_all.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
                preds_all = preds_all / row_sums

                tree_df = pd.DataFrame({"cell_id": cell_ids})
                for ci, cn in enumerate(CLASS_NAMES):
                    tree_df[f"{cn}_pred"] = preds_all[:, ci]
                tree_df["prev_year"] = prev_year
                tree_df["curr_year"] = curr_year
                tree_df.to_parquet(tree_path, index=False)

            # MLP predictions (best architecture)
            if not os.path.exists(mlp_path):
                X_mlp = np.nan_to_num(
                    merged[[c for c in mlp_cols if c in merged_cols_set]]
                    .values.astype(np.float32), 0.0)

                preds_all = np.zeros((len(cell_ids), N_CLASSES),
                                     dtype=np.float32)
                for fold_id in range(N_FOLDS):
                    preds_all += predict_with_mlp_ensemble(
                        X_mlp, fold_id, n_mlp, device,
                        best_arch_dir, best_arch)
                preds_all /= N_FOLDS

                mlp_df = pd.DataFrame({"cell_id": cell_ids})
                for ci, cn in enumerate(CLASS_NAMES):
                    mlp_df[f"{cn}_pred"] = preds_all[:, ci]
                mlp_df["prev_year"] = prev_year
                mlp_df["curr_year"] = curr_year
                mlp_df["arch"] = best_arch["name"]
                mlp_df.to_parquet(mlp_path, index=False)

            print(f"  [{city.name}/{tag}] Done")

    print(f"\n[{ts()}] Predict stage complete.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-city pipeline: download -> extract -> train -> predict")
    parser.add_argument("--skip-download", action="store_true")
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
    print(f"[{ts()}] Multi-City Pipeline starting")
    print(f"  Cities: {[c.name for c in CITIES]}")
    print(f"  Training: {[c.name for c in TRAIN_CITIES]}")
    print(f"  Test: {[c.name for c in TEST_CITIES]}")
    print(f"  Years: {ALL_YEARS}")
    print(f"  MLP seeds: {N_SEEDS}")

    ensure_all_dirs()

    # Stage 0: Anchors
    stage_anchors()

    # Stage 1: Download
    if not args.skip_download:
        stage_download()
    else:
        print("\n  DOWNLOAD skipped")

    # Stage 2: Labels
    stage_labels()

    # Stage 3: Extract
    if not args.skip_extract:
        stage_extract()
    else:
        print("\n  EXTRACT skipped")

    # Stage 4: Train
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
    print(f"MULTI-CITY PIPELINE COMPLETE in {hours}h {mins}m")
    print(f"  Data:        {CITIES_DIR}")
    print(f"  Models:      {MODELS_DIR}")
    for c in CITIES:
        n_pred = len([f for f in os.listdir(city_predictions_dir(c))
                      if f.endswith(".parquet")])
        print(f"  {c.name:<15} {n_pred} prediction files")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
