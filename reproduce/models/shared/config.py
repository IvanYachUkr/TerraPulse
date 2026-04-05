"""
Shared configuration for the raw-band model comparison study.

Defines city list, train/val/test splits, band names, and class definitions.
Uses the exact same splits as the deployed MLP (reproduce/mlp/) for fair comparison.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
WC_TILES_DIR = os.path.join(CITIES_DIR, "worldcover_tiles")

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
N_CLASSES = 7
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]

# ESA WorldCover code → our 7-class index
WC_CLASS_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4,
                60: 5, 70: 5, 100: 5, 80: 6}

# Sentinel-2 bands (10 spectral, no SCL)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_NODATA = -9999

# Sentinel-1 bands
S1_BANDS = ["VV", "VH"]
S1_NODATA = -9999

# Temporal structure
YEARS = [2020, 2021]
SEASONS = ["spring", "summer", "autumn"]
N_TEMPORAL_SLOTS = len(YEARS) * len(SEASONS)  # 6

# Raw features per pixel
N_S2_FEATURES = len(S2_BANDS) * N_TEMPORAL_SLOTS        # 60
N_S1_FEATURES = len(S1_BANDS) * N_TEMPORAL_SLOTS        # 12
N_RAW_FEATURES = N_S2_FEATURES + N_S1_FEATURES          # 72

# Grid cell size (pixels)
GRID_PX = 10

# ── City definition ──────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import List

@dataclass
class CityConfig:
    name: str
    bbox: List[float]
    epsg: int
    wc_tile: str
    is_test: bool = False


# Import the full city list from reproduce/mlp (single source of truth)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "mlp"))
from importlib import import_module
_step1 = import_module("01_download_data")
CITIES = _step1.CITIES

CITY_MAP = {c.name: c for c in CITIES}

# ── Splits (identical to V10 BOHB / reproduce/mlp) ──────────────────────────
# Test cities: completely held out, never seen during training or validation
TEST_CITY_NAMES = {
    "nuremberg",
    "ankara_test", "sofia_test", "riga_test", "edinburgh_test", "palermo_test",
}

# Validation cities: 23 cities, label-balanced selection (from V10 BOHB)
VAL_CITY_NAMES = {
    "alentejo_portugal", "andalusia_olives", "berlin", "bordeaux",
    "central_spain_plateau", "corsica_interior", "dresden", "dutch_polders",
    "ebro_delta", "estonian_plains", "helsinki", "iceland_highlands",
    "ireland_bog_pasture", "jaen_olives", "madrid", "marseille",
    "northern_sweden", "paris_south", "peloponnese_rural", "po_valley_rural",
    "rostock", "uppland_farmland", "vojvodina_cropland",
}


def get_train_cities():
    return [c for c in CITIES
            if c.name not in VAL_CITY_NAMES
            and c.name not in TEST_CITY_NAMES]


def get_val_cities():
    return [c for c in CITIES if c.name in VAL_CITY_NAMES]


def get_test_cities():
    return [c for c in CITIES if c.name in TEST_CITY_NAMES]


def city_dir(city):
    return os.path.join(CITIES_DIR, city.name)


def raw_dir(city):
    """Return the raw TIF directory. Prefers raw_v7/, falls back to raw/."""
    d = os.path.join(city_dir(city), "raw_v7")
    if os.path.isdir(d) and os.listdir(d):
        return d
    d = os.path.join(city_dir(city), "raw")
    if os.path.isdir(d) and os.listdir(d):
        return d
    return None


def city_has_raw_tifs(city):
    """Check if a city has raw TIF files available."""
    d = raw_dir(city)
    if d is None:
        return False
    import glob
    return len(glob.glob(os.path.join(d, "sentinel2_*.tif"))) >= 6


# ── Feature column ordering ─────────────────────────────────────────────────

def raw_feature_names():
    """Return ordered list of 72 raw feature names."""
    names = []
    for year in YEARS:
        for season in SEASONS:
            for band in S2_BANDS:
                names.append(f"{band}_{year}_{season}")
            for band in S1_BANDS:
                names.append(f"SAR_{band}_{year}_{season}")
    return names
