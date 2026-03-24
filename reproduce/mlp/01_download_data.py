#!/usr/bin/env python3
"""
Step 1: Download Sentinel-2/S1 satellite composites and ESA WorldCover labels.

Uses the terrapulse Rust binary (must be built first: cargo build --release).
Downloads 3 seasons (spring/summer/autumn) x 2 years (2020, 2021) per city
plus WorldCover labels.

Usage:
    python 01_download_data.py                    # all cities
    python 01_download_data.py --cities munich nuremberg berlin  # specific
    python 01_download_data.py --list-cities      # list available cities
"""

import argparse, math, os, subprocess, sys, time, urllib.request
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                  "B11", "B12"]
SEASONS = ["spring", "summer", "autumn"]
YEARS = [2020, 2021]
WC_YEARS = [2020, 2021]
SEED = 42
N_CLASSES = 7
GRID_PX = 10
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
WC_CLASS_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4,
                60: 5, 70: 5, 100: 5, 80: 6}

CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
WC_TILES_DIR = os.path.join(CITIES_DIR, "worldcover_tiles")
TERRAPULSE_BIN = os.path.join(PROJECT_ROOT, "terrapulse", "target", "release",
                              "terrapulse" + (".exe" if sys.platform == "win32" else ""))

# ---------------------------------------------------------------------------
# City definitions (complete list for reproducibility)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import List

@dataclass
class CityConfig:
    name: str
    bbox: List[float]
    epsg: int
    wc_tile: str
    is_test: bool = False

# fmt: off
CITIES = [
    CityConfig("bremen",      [8.65, 53.00, 8.90, 53.14], 32632, "N51E006"),
    CityConfig("hamburg",     [9.80, 53.40, 10.15, 53.58], 32632, "N51E009"),
    CityConfig("duesseldorf", [6.70, 51.15, 6.90, 51.28], 32632, "N51E006"),
    CityConfig("leipzig",     [12.25, 51.27, 12.50, 51.40], 32633, "N51E012"),
    CityConfig("rostock",     [12.00, 54.05, 12.20, 54.18], 32633, "N54E012"),
    CityConfig("amsterdam",   [4.75, 52.30, 4.95, 52.40], 32631, "N51E003"),
    CityConfig("hambach_mine",[6.40, 50.85, 6.60, 50.98], 32632, "N48E006"),
    CityConfig("welzow_mine", [14.10, 51.50, 14.35, 51.65], 32633, "N51E012"),
    CityConfig("salzburg",    [12.95, 47.73, 13.15, 47.87], 32633, "N45E012"),
    CityConfig("malmo",       [12.90, 55.53, 13.15, 55.68], 32633, "N54E012"),
    CityConfig("london",      [-0.25, 51.42, 0.05, 51.56], 32631, "N51W003"),
    CityConfig("brussels",    [4.30, 50.80, 4.50, 50.92], 32631, "N48E003"),
    CityConfig("paris_south", [2.25, 48.75, 2.50, 48.89], 32631, "N48E000"),
    CityConfig("berlin",      [13.30, 52.45, 13.55, 52.58], 32633, "N51E012"),
    CityConfig("vienna",      [16.30, 48.15, 16.55, 48.28], 32633, "N48E015"),
    CityConfig("zurich",      [8.45, 47.32, 8.65, 47.44], 32632, "N45E006"),
    CityConfig("munich_north",[11.45, 48.22, 11.70, 48.36], 32632, "N48E009"),
    CityConfig("stuttgart",   [9.10, 48.72, 9.30, 48.84], 32632, "N48E009"),
    CityConfig("innsbruck",   [11.30, 47.22, 11.55, 47.35], 32632, "N45E009"),
    CityConfig("krakow",      [19.85, 50.02, 20.10, 50.14], 32634, "N48E018"),
    CityConfig("budapest",    [19.00, 47.42, 19.20, 47.55], 32634, "N45E018"),
    CityConfig("bratislava",  [17.05, 48.10, 17.25, 48.22], 32633, "N48E015"),
    CityConfig("helsinki",    [24.85, 60.13, 25.10, 60.27], 32635, "N60E024"),
    CityConfig("copenhagen",  [12.48, 55.62, 12.68, 55.74], 32633, "N54E012"),
    CityConfig("gothenburg",  [11.90, 57.65, 12.10, 57.78], 32632, "N57E009"),
    CityConfig("barcelona",   [2.05, 41.32, 2.30, 41.45], 32631, "N39E000"),
    CityConfig("madrid",      [-3.80, 40.35, -3.55, 40.48], 32630, "N39W006"),
    CityConfig("lisbon",      [-9.20, 38.68, -8.95, 38.82], 32629, "N36W012"),
    CityConfig("rome",        [12.40, 41.82, 12.60, 41.95], 32633, "N39E012"),
    CityConfig("milan",       [9.10, 45.42, 9.35, 45.55], 32632, "N45E009"),
    CityConfig("lyon",        [4.75, 45.70, 4.95, 45.82], 32631, "N45E003"),
    CityConfig("toulouse",    [1.35, 43.55, 1.55, 43.68], 32631, "N42E000"),
    CityConfig("athens",      [23.65, 37.92, 23.85, 38.05], 32635, "N36E021"),
    CityConfig("almeria_coast",[-2.50, 36.78, -2.30, 36.90], 32630, "N36W003"),
    CityConfig("central_hungary",[19.50, 47.10, 19.75, 47.24], 32634, "N45E018"),
    CityConfig("finnish_lakeland", [27.50, 61.80, 27.80, 61.95], 32636, "N60E027"),
    CityConfig("swedish_forest",   [15.30, 57.30, 15.55, 57.45], 32633, "N57E015"),
    CityConfig("scottish_highlands",[-5.20, 57.05, -4.95, 57.18], 32630, "N57W006"),
    CityConfig("sicily_interior",  [14.10, 37.40, 14.35, 37.53], 32633, "N36E012"),
    CityConfig("alentejo_portugal",[-7.90, 38.10, -7.65, 38.25], 32629, "N36W009"),
    CityConfig("peloponnese_rural",[22.00, 37.40, 22.25, 37.55], 32634, "N36E021"),
    CityConfig("carpathian_romania",[24.60, 45.50, 24.85, 45.65], 32635, "N45E024"),
    CityConfig("po_valley_rural",  [10.80, 44.90, 11.05, 45.05], 32632, "N42E009"),
    CityConfig("dutch_polders",    [5.10, 52.55, 5.35, 52.70], 32631, "N51E003"),
    CityConfig("danish_farmland",  [9.80, 55.30, 10.05, 55.45], 32632, "N54E009"),
    CityConfig("dublin",      [-6.35, 53.30, -6.15, 53.42], 32629, "N51W009"),
    CityConfig("marseille",   [5.30, 43.25, 5.50, 43.38], 32631, "N42E003"),
    CityConfig("naples",      [14.18, 40.80, 14.38, 40.93], 32633, "N39E012"),
    CityConfig("valencia",    [-0.45, 39.42, -0.25, 39.55], 32630, "N39W003"),
    CityConfig("bordeaux",    [-0.65, 44.80, -0.45, 44.93], 32630, "N42W003"),
    CityConfig("oslo",        [10.65, 59.87, 10.85, 60.00], 32632, "N57E009"),
    CityConfig("gdansk",      [18.55, 54.32, 18.75, 54.45], 32634, "N54E018"),
    CityConfig("castilla_meseta",     [-3.10, 39.20, -2.85, 39.35], 32630, "N39W006"),
    CityConfig("extremadura_dehesa",  [-6.15, 39.10, -5.90, 39.25], 32629, "N39W009"),
    CityConfig("aragon_steppe",       [-0.70, 41.05, -0.45, 41.20], 32630, "N39W003"),
    CityConfig("murcia_drylands",     [-1.60, 38.00, -1.35, 38.15], 32630, "N36W003"),
    CityConfig("tabernas_desert",     [-2.40, 37.00, -2.15, 37.15], 32630, "N36W003"),
    CityConfig("bardenas_reales",     [-1.57, 42.13, -1.32, 42.27], 32630, "N42W003"),
    CityConfig("sardinia_maquis",     [9.00, 39.80, 9.25, 39.95], 32632, "N39E009"),
    CityConfig("crete_phrygana",      [24.80, 35.20, 25.05, 35.35], 32635, "N33E024"),
    CityConfig("corsica_interior",    [9.10, 41.85, 9.35, 42.00], 32632, "N39E009"),
    CityConfig("thessaly_scrubland",  [22.30, 39.50, 22.55, 39.65], 32634, "N39E021"),
    CityConfig("thrace_steppe",       [26.40, 41.00, 26.65, 41.15], 32635, "N39E024"),
    CityConfig("el_ejido_greenhouses",[-2.94, 36.70, -2.69, 36.84], 32630, "N36W003"),
    CityConfig("skane_fields",        [13.40, 55.70, 13.65, 55.85], 32633, "N54E012"),
    CityConfig("trondelag_farmland",  [10.30, 63.35, 10.55, 63.50], 32632, "N63E009"),
    CityConfig("estonian_plains",     [25.50, 58.50, 25.75, 58.65], 32635, "N57E024"),
    CityConfig("latvian_farmland",    [24.00, 56.85, 24.25, 57.00], 32635, "N54E021"),
    CityConfig("lithuanian_lowland",  [23.80, 55.60, 24.05, 55.75], 32635, "N54E021"),
    CityConfig("finnish_coastal_farm",[24.00, 60.40, 24.25, 60.55], 32635, "N60E021"),
    CityConfig("iceland_highlands",   [-19.48, 64.13, -19.23, 64.27], 32627, "N63W021"),
    CityConfig("lapland_tundra",      [27.07, 68.28, 27.32, 68.42], 32635, "N66E027"),
    CityConfig("galicia_pastures",    [-8.60, 42.80, -8.35, 42.95], 32629, "N42W009"),
    CityConfig("brittany_bocage",     [-3.40, 48.10, -3.15, 48.25], 32630, "N48W006"),
    CityConfig("ireland_bog_pasture", [-7.80, 53.20, -7.55, 53.35], 32629, "N51W009"),
    CityConfig("wales_upland",        [-3.90, 52.00, -3.65, 52.15], 32630, "N51W006"),
    CityConfig("les_landes_forest",   [-0.93, 44.08, -0.68, 44.22], 32630, "N42W003"),
    CityConfig("hortobagy_puszta",    [21.02, 47.53, 21.27, 47.67], 32634, "N45E021"),
    CityConfig("wallachian_steppe",   [25.50, 44.20, 25.75, 44.35], 32635, "N42E024"),
    CityConfig("thracian_farmland",   [25.00, 42.10, 25.25, 42.25], 32635, "N42E024"),
    CityConfig("vojvodina_cropland",  [20.20, 45.30, 20.45, 45.45], 32634, "N45E018"),
    CityConfig("jaen_olives",         [-3.92, 37.78, -3.67, 37.92], 32630, "N36W006"),
    CityConfig("camargue_wetland",    [4.40, 43.40, 4.65, 43.55], 32631, "N42E003"),
    CityConfig("ebro_delta",          [0.65, 40.60, 0.90, 40.75], 32631, "N39E000"),
    CityConfig("wadden_tidal",        [8.00, 53.55, 8.25, 53.70], 32632, "N51E006"),
    CityConfig("danube_delta",        [29.32, 44.98, 29.57, 45.12], 32635, "N45E027"),
    CityConfig("pyrenees_meadows",    [0.40, 42.60, 0.65, 42.75], 32631, "N42E000"),
    CityConfig("norwegian_fjord",     [7.00, 61.50, 7.25, 61.65], 32632, "N60E006"),
    CityConfig("carpathian_alpine",   [24.50, 47.50, 24.75, 47.65], 32635, "N45E024"),
    CityConfig("swiss_alps_high",     [7.62, 45.93, 7.88, 46.07], 32632, "N45E006"),
    CityConfig("foggia_wheat",        [15.43, 41.38, 15.68, 41.52], 32633, "N39E015"),
    CityConfig("jutland_farmland",    [8.82, 56.18, 9.07, 56.32], 32632, "N54E006"),
    CityConfig("donana_marshes",      [-6.50, 36.90, -6.25, 37.05], 32630, "N36W009"),
    CityConfig("andalusia_olives",    [-4.20, 37.50, -3.95, 37.65], 32630, "N36W006"),
    CityConfig("central_spain_plateau",[-3.50, 40.60, -3.25, 40.75], 32630, "N39W006"),
    CityConfig("uppland_farmland",    [17.60, 59.60, 17.85, 59.75], 32633, "N57E015"),
    CityConfig("finnish_bog",         [26.00, 63.50, 26.25, 63.65], 32636, "N63E024"),
    CityConfig("mecklenburg_lakes",   [12.60, 53.40, 12.85, 53.55], 32633, "N51E012"),
    CityConfig("danube_floodplain",   [18.80, 47.80, 19.05, 47.95], 32634, "N45E018"),
    CityConfig("cretan_coast",        [25.10, 35.30, 25.35, 35.45], 32635, "N33E024"),
    CityConfig("cyprus_troodos",       [32.80, 34.85, 33.05, 35.00], 32636, "N33E030"),
    CityConfig("dalmatian_coast",      [15.40, 43.85, 15.65, 44.00], 32633, "N42E015"),
    CityConfig("greek_maquis",        [22.10, 37.50, 22.35, 37.65], 32634, "N36E021"),
    CityConfig("schwarzwald_edge",    [7.80, 47.90, 8.05, 48.05],   32632, "N45E006"),
    CityConfig("algarve_coast",       [-8.20, 37.00, -7.95, 37.15], 32629, "N36W009"),
    CityConfig("central_finland_bog", [25.30, 62.40, 25.55, 62.55], 32635, "N60E024"),
    CityConfig("northern_sweden",     [19.40, 68.30, 19.65, 68.45], 32634, "N66E018"),
    CityConfig("sw_ireland_heath",    [-10.00, 51.70, -9.75, 51.85],32629, "N51W012"),
    CityConfig("dresden",             [13.65, 50.98, 13.90, 51.12], 32633, "N48E012"),
    CityConfig("andalusia_sierra",    [-3.60, 36.90, -3.35, 37.05], 32630, "N36W006"),
    # Test cities
    CityConfig("munich",      [11.45, 48.08, 11.70, 48.22], 32632, "N48E009", is_test=True),
    CityConfig("nuremberg",   [10.95, 49.38, 11.20, 49.52], 32632, "N48E009", is_test=True),
    CityConfig("warsaw",      [20.90, 52.15, 21.15, 52.30], 32634, "N51E018", is_test=True),
    CityConfig("prague",      [14.35, 50.02, 14.55, 50.15], 32633, "N48E012", is_test=True),
    CityConfig("seville",     [-6.05, 37.32, -5.85, 37.45], 32630, "N36W009", is_test=True),
    CityConfig("stockholm",   [17.95, 59.28, 18.20, 59.42], 32633, "N57E015", is_test=True),
    # Extra test cities (held out for V10 evaluation)
    CityConfig("ankara_test",    [32.73, 39.87, 32.98, 40.01], 32636, "N39E030", is_test=True),
    CityConfig("sofia_test",     [23.25, 42.62, 23.50, 42.76], 32634, "N42E021", is_test=True),
    CityConfig("riga_test",      [24.00, 56.90, 24.25, 57.04], 32635, "N54E021", is_test=True),
    CityConfig("edinburgh_test", [-3.30, 55.90, -3.05, 56.04], 32630, "N54W006", is_test=True),
    CityConfig("palermo_test",   [13.30, 38.07, 13.55, 38.21], 32633, "N36E012", is_test=True),
]
# fmt: on

CITY_MAP = {c.name: c for c in CITIES}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ts():
    return time.strftime("%H:%M:%S")

def city_dir(city):
    return os.path.join(CITIES_DIR, city.name)

def city_raw_dir(city):
    return os.path.join(city_dir(city), "raw")

def city_anchor_path(city):
    raw = city_raw_dir(city)
    for season in SEASONS:
        for year in YEARS:
            p = os.path.join(raw, f"sentinel2_{city.name}_{year}_{season}.tif")
            if os.path.exists(p):
                return p
    return os.path.join(raw, f"sentinel2_{city.name}_{YEARS[0]}_{SEASONS[0]}.tif")

def city_labels_path(city, year=2021):
    return os.path.join(city_dir(city), f"labels_{year}.parquet")


# ---------------------------------------------------------------------------
# Download satellite data via terrapulse
# ---------------------------------------------------------------------------
def download_city(city):
    """Download S2 + S1 composites for a single city using terrapulse."""
    raw = city_raw_dir(city)
    os.makedirs(raw, exist_ok=True)

    # Check if already downloaded
    all_done = True
    for year in YEARS:
        for season in SEASONS:
            s2 = os.path.join(raw, f"sentinel2_{city.name}_{year}_{season}.tif")
            if not os.path.exists(s2):
                all_done = False
                break
        if not all_done:
            break
    if all_done:
        print(f"  [{city.name}] Already downloaded — skip")
        return

    bbox_str = ",".join(str(x) for x in city.bbox)
    for year in YEARS:
        print(f"  [{ts()}] [{city.name}] Downloading {year}...")
        cmd = [
            TERRAPULSE_BIN, "download",
            "--bbox", bbox_str,
            "--epsg", str(city.epsg),
            "--year", str(year),
            "--output-dir", raw,
            "--region", city.name,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                print(f"    ERROR: {result.stderr[-200:]}")
            else:
                print(f"    OK")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after 30 min")
        except FileNotFoundError:
            print(f"    ERROR: terrapulse binary not found at {TERRAPULSE_BIN}")
            print(f"    Build with: cd terrapulse && cargo build --release")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Download WorldCover labels
# ---------------------------------------------------------------------------
def wc_tiles_for_bbox(bbox):
    """Compute all ESA WorldCover 3x3 degree tile IDs covering a bbox."""
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


def download_worldcover_tile(tile, year):
    """Download a single WorldCover tile from ESA S3."""
    os.makedirs(WC_TILES_DIR, exist_ok=True)
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
        return path
    except Exception as e:
        print(f"  ERROR downloading {filename}: {e}")
        if os.path.exists(path):
            os.remove(path)
        return ""


def create_labels(city):
    """Create WorldCover label parquets if they don't exist."""
    import pandas as pd
    import rasterio
    from rasterio.warp import Resampling, reproject

    for year in WC_YEARS:
        label_path = city_labels_path(city, year)
        if os.path.exists(label_path):
            print(f"  [{city.name}/{year}] Labels exist — skip")
            continue

        # Download tiles
        tiles = wc_tiles_for_bbox(city.bbox)
        wc_paths = []
        for tile in tiles:
            p = download_worldcover_tile(tile, year)
            if p:
                wc_paths.append(p)

        if not wc_paths:
            print(f"  [{city.name}/{year}] No WorldCover tiles available")
            continue

        # Reproject to city anchor grid
        anchor_path = city_anchor_path(city)
        if not os.path.exists(anchor_path):
            print(f"  [{city.name}/{year}] No anchor TIF — skip labels")
            continue

        with rasterio.open(anchor_path) as ref:
            anchor_crs = ref.crs
            anchor_transform = ref.transform
            anchor_w = ref.width
            anchor_h = ref.height

        dst_array = np.zeros((anchor_h, anchor_w), dtype=np.uint8)
        for wc_path in wc_paths:
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

        # Aggregate to 100m grid cells
        n_cols = anchor_w // GRID_PX
        n_rows = anchor_h // GRID_PX
        records = []
        cell_id = 0
        total_px = GRID_PX * GRID_PX
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                r0, c0 = row_idx * GRID_PX, col_idx * GRID_PX
                patch = dst_array[r0:r0 + GRID_PX, c0:c0 + GRID_PX]
                proportions = np.zeros(N_CLASSES, dtype=np.float32)
                mapped = 0
                for wc_code, our_class in WC_CLASS_MAP.items():
                    count = int(np.sum(patch == wc_code))
                    proportions[our_class] += count
                    mapped += count
                if total_px > 0:
                    proportions /= total_px
                row_sum = proportions.sum()
                if row_sum > 0:
                    proportions /= row_sum
                record = {"cell_id": cell_id}
                for i, name in enumerate(CLASS_NAMES):
                    record[name] = float(proportions[i])
                records.append(record)
                cell_id += 1

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        df.to_parquet(label_path, index=False)
        print(f"  [{city.name}/{year}] Labels saved ({len(df)} cells)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download satellite data for MLP training")
    parser.add_argument("--cities", nargs="*", default=None,
                        help="Specific city names to download (default: all)")
    parser.add_argument("--list-cities", action="store_true",
                        help="List all available city names and exit")
    args = parser.parse_args()

    if args.list_cities:
        for c in CITIES:
            role = "test" if c.is_test else "train"
            print(f"  {c.name:30s} [{role}] bbox={c.bbox}")
        return

    if args.cities:
        cities = [CITY_MAP[name] for name in args.cities if name in CITY_MAP]
        if not cities:
            print(f"ERROR: No matching cities. Use --list-cities to see options.")
            return
    else:
        cities = CITIES

    print(f"\n{'='*70}")
    print(f"  Step 1: Download Satellite Data")
    print(f"  Cities: {len(cities)}")
    print(f"  Terrapulse binary: {TERRAPULSE_BIN}")
    print(f"{'='*70}\n")

    if not os.path.exists(TERRAPULSE_BIN):
        print(f"ERROR: Rust binary not found: {TERRAPULSE_BIN}")
        print(f"Build with: cd terrapulse && cargo build --release")
        sys.exit(1)

    # 1. Download S2+S1 composites
    print(f"[{ts()}] Downloading satellite composites...")
    for city in cities:
        download_city(city)

    # 2. Download WorldCover labels
    print(f"\n[{ts()}] Creating WorldCover labels...")
    for city in cities:
        create_labels(city)

    print(f"\n[{ts()}] Download complete!")


if __name__ == "__main__":
    main()
