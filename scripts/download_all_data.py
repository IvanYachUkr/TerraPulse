"""
Download all project data for 'Mapping Urban Change in Nuremberg with ML'.

This script downloads ALL required data in one go:
  1. ESA WorldCover 2020 & 2021 (ground-truth labels)
  2. Sentinel-2 composites 2020-2025 (satellite imagery features)
  3. OpenStreetMap features (buildings, roads, land-use, water, natural)

Usage:
    pip install -r requirements.txt
    python scripts/download_all_data.py

Runtime: ~10-15 minutes (depends on internet speed and OSM API load)
Disk space required: ~650 MB
"""

import os
import sys
import urllib.request
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")
OSM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "osm")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

NUREMBERG_BBOX = [10.95, 49.38, 11.20, 49.52]  # [west, south, east, north]

SENTINEL_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
SENTINEL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def setup_dirs():
    """Create data directory structure."""
    for d in [RAW_DIR, LABELS_DIR, OSM_DIR, PROCESSED_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Directory structure created.")


def download_worldcover():
    """Download ESA WorldCover 2020 (v100) and 2021 (v200) tiles for Nuremberg."""
    print("\n" + "=" * 60)
    print("STEP 1/3: ESA WorldCover Labels")
    print("=" * 60)

    tiles = {
        2020: {
            "url": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_N48E009_Map.tif",
            "file": "ESA_WorldCover_2020_N48E009.tif",
        },
        2021: {
            "url": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E009_Map.tif",
            "file": "ESA_WorldCover_2021_N48E009.tif",
        },
    }

    for year, info in tiles.items():
        path = os.path.join(LABELS_DIR, info["file"])
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [{year}] Already exists: {info['file']} ({size_mb:.1f} MB) -- skipping")
            continue
        print(f"  [{year}] Downloading {info['file']}...")
        urllib.request.urlretrieve(info["url"], path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{year}] Saved: {info['file']} ({size_mb:.1f} MB)")


def download_sentinel2():
    """Download Sentinel-2 L2A median composites via Microsoft Planetary Computer."""
    import planetary_computer
    import pystac_client
    import rasterio
    import stackstac
    from rasterio.crs import CRS

    print("\n" + "=" * 60)
    print("STEP 2/3: Sentinel-2 Imagery (via Planetary Computer)")
    print("=" * 60)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    print("  Connected to Planetary Computer STAC API")

    for year in SENTINEL_YEARS:
        path = os.path.join(RAW_DIR, f"sentinel2_nuremberg_{year}.tif")
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [{year}] Already exists ({size_mb:.1f} MB) -- skipping")
            continue

        print(f"  [{year}] Searching for scenes (Jun-Aug, <20% cloud)...")
        items = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=NUREMBERG_BBOX,
            datetime=f"{year}-06-01/{year}-08-31",
            query={"eo:cloud_cover": {"lt": 20}},
        ).item_collection()
        print(f"  [{year}] Found {len(items)} scenes")

        if len(items) == 0:
            print(f"  [{year}] WARNING: No images found -- skipping")
            continue

        print(f"  [{year}] Computing median composite...")
        comp = (
            stackstac.stack(
                items,
                assets=SENTINEL_BANDS,
                bounds_latlon=NUREMBERG_BBOX,
                resolution=10,
                epsg=32632,
            )
            .median(dim="time")
            .compute()
        )

        transform = rasterio.transform.from_bounds(
            comp.coords["x"].values.min(),
            comp.coords["y"].values.min(),
            comp.coords["x"].values.max(),
            comp.coords["y"].values.max(),
            comp.shape[2],
            comp.shape[1],
        )
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=comp.shape[1],
            width=comp.shape[2],
            count=len(SENTINEL_BANDS),
            dtype="float32",
            crs=CRS.from_epsg(32632),
            transform=transform,
            compress="lzw",
        ) as dst:
            for i in range(len(SENTINEL_BANDS)):
                band_data = np.nan_to_num(comp[i].values.astype(np.float32), nan=0.0)
                dst.write(band_data, i + 1)
                dst.set_band_description(i + 1, SENTINEL_BANDS[i])

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{year}] Saved: sentinel2_nuremberg_{year}.tif ({size_mb:.1f} MB)")


def download_osm():
    """Download OpenStreetMap features for Nuremberg."""
    import geopandas as gpd
    import osmnx as ox

    print("\n" + "=" * 60)
    print("STEP 3/3: OpenStreetMap Features")
    print("=" * 60)

    PLACE = "Nuremberg, Germany"

    # Buildings
    bpath = os.path.join(OSM_DIR, "buildings.gpkg")
    if os.path.exists(bpath) and os.path.getsize(bpath) > 1_000_000:
        print(f"  [buildings] Already exists -- skipping")
    else:
        print("  [buildings] Downloading footprints...")
        buildings = ox.features_from_place(PLACE, tags={"building": True})
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
        cols = [
            c for c in ["geometry", "building", "building:levels", "name"] if c in buildings.columns
        ]
        buildings[cols].to_file(bpath, driver="GPKG")
        print(f"  [buildings] Saved {len(buildings)} buildings")

    # Roads
    rpath = os.path.join(OSM_DIR, "roads.gpkg")
    npath = os.path.join(OSM_DIR, "road_nodes.gpkg")
    if os.path.exists(rpath):
        print(f"  [roads] Already exists -- skipping")
    else:
        print("  [roads] Downloading road network...")
        G = ox.graph_from_place(PLACE, network_type="drive")
        nodes, edges = ox.graph_to_gdfs(G)
        edges.to_file(rpath, driver="GPKG")
        nodes.to_file(npath, driver="GPKG")
        print(f"  [roads] Saved {len(edges)} segments, {len(nodes)} intersections")

    # Land use
    lpath = os.path.join(OSM_DIR, "landuse.gpkg")
    if os.path.exists(lpath):
        print(f"  [landuse] Already exists -- skipping")
    else:
        print("  [landuse] Downloading land-use zones...")
        landuse = ox.features_from_place(PLACE, tags={"landuse": True})
        landuse = landuse[landuse.geometry.type.isin(["Polygon", "MultiPolygon"])]
        landuse.to_file(lpath, driver="GPKG")
        print(f"  [landuse] Saved {len(landuse)} polygons")

    # Natural features
    napath = os.path.join(OSM_DIR, "natural.gpkg")
    if os.path.exists(napath):
        print(f"  [natural] Already exists -- skipping")
    else:
        print("  [natural] Downloading natural features...")
        natural = ox.features_from_place(PLACE, tags={"natural": True})
        natural = natural[natural.geometry.type.isin(["Polygon", "MultiPolygon"])]
        natural.to_file(napath, driver="GPKG")
        print(f"  [natural] Saved {len(natural)} features")

    # Water
    wpath = os.path.join(OSM_DIR, "water.gpkg")
    if os.path.exists(wpath):
        print(f"  [water] Already exists -- skipping")
    else:
        print("  [water] Downloading water bodies...")
        water = ox.features_from_place(PLACE, tags={"water": True})
        water = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
        water.to_file(wpath, driver="GPKG")
        print(f"  [water] Saved {len(water)} features")


def print_summary():
    """Print summary of all downloaded data."""
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE -- DATA SUMMARY")
    print("=" * 60)

    for label, directory in [
        ("Labels", LABELS_DIR),
        ("Raw Imagery", RAW_DIR),
        ("OSM Features", OSM_DIR),
    ]:
        print(f"\n  {label} ({directory}):")
        if os.path.exists(directory):
            for f in sorted(os.listdir(directory)):
                fp = os.path.join(directory, f)
                if os.path.isfile(fp):
                    print(f"    {f}: {os.path.getsize(fp) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    setup_dirs()
    download_worldcover()
    download_sentinel2()
    download_osm()
    print_summary()
