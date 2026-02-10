"""
Phase 2D: Spatial Alignment & Grid Creation

Creates the tabular dataset from raw rasters:
  1. Reproject WorldCover labels from EPSG:4326 -> EPSG:32632 (UTM 32N)
  2. Create 100m × 100m grid over Nuremberg (aligned with Sentinel-2)
  3. Aggregate WorldCover labels per grid cell -> class proportions
  4. Save grid as GeoParquet + labels as Parquet

Usage:
    python src/data/build_grid.py

Outputs:
    data/processed/grid.gpkg          — 100m grid geometries
    data/processed/labels_2020.parquet — class proportions per cell, 2020
    data/processed/labels_2021.parquet — class proportions per cell, 2021
"""

import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

GRID_SIZE_M = 100  # meters
TARGET_CRS = CRS.from_epsg(32632)  # UTM 32N

# Class mapping: WorldCover code -> our 6 classes
CLASS_MAP = {10: 0, 30: 1, 90: 1, 40: 2, 50: 3, 60: 4, 80: 5}  # wetland→grassland
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
N_CLASSES = 6

# Use Sentinel-2 bounds as our AOI (already clipped to Nuremberg bbox)
SENTINEL_REF = os.path.join(RAW_DIR, "sentinel2_nuremberg_2020.tif")


def get_aligned_bounds(sentinel_path: str, grid_size: int) -> tuple:
    """Get Sentinel-2 bounds, snapped to grid_size multiples."""
    with rasterio.open(sentinel_path) as ds:
        b = ds.bounds
    # Snap bounds outward to grid_size multiples
    left = np.floor(b.left / grid_size) * grid_size
    bottom = np.floor(b.bottom / grid_size) * grid_size
    right = np.ceil(b.right / grid_size) * grid_size
    top = np.ceil(b.top / grid_size) * grid_size
    return left, bottom, right, top


def create_grid(bounds: tuple, grid_size: int) -> gpd.GeoDataFrame:
    """Create a regular grid of square cells."""
    left, bottom, right, top = bounds
    xs = np.arange(left, right, grid_size)
    ys = np.arange(bottom, top, grid_size)

    cells = []
    cell_ids = []
    cell_id = 0
    for y in ys:
        for x in xs:
            cells.append(box(x, y, x + grid_size, y + grid_size))
            cell_ids.append(cell_id)
            cell_id += 1

    grid = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=cells, crs=TARGET_CRS)
    print(f"  Grid: {len(xs)} cols × {len(ys)} rows = {len(grid)} cells")
    print(f"  Bounds: [{left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f}] (EPSG:32632)")
    return grid


def reproject_worldcover(wc_path: str, bounds: tuple, pixel_size: float = 10.0) -> np.ndarray:
    """Reproject WorldCover from EPSG:4326 to EPSG:32632, clipped to bounds."""
    left, bottom, right, top = bounds
    dst_width = int((right - left) / pixel_size)
    dst_height = int((top - bottom) / pixel_size)

    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, dst_width, dst_height)
    dst_array = np.zeros((dst_height, dst_width), dtype=np.uint8)

    with rasterio.open(wc_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.nearest,  # categorical data -> nearest neighbor
        )

    print(f"  Reprojected to {dst_width}×{dst_height} pixels at {pixel_size}m")
    return dst_array, dst_transform


def aggregate_labels(
    wc_array: np.ndarray, wc_transform: rasterio.Affine, grid: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Compute class proportions per grid cell from WorldCover raster."""
    pixel_size = abs(wc_transform.a)
    grid_px = int(GRID_SIZE_M / pixel_size)  # pixels per grid cell side

    # Origin of the raster (top-left corner in UTM)
    origin_x = wc_transform.c
    origin_y = wc_transform.f  # top of raster

    records = []
    for _, row in grid.iterrows():
        geom = row.geometry
        # Convert cell bounds to pixel indices
        col_start = int((geom.bounds[0] - origin_x) / pixel_size)
        row_start = int((origin_y - geom.bounds[3]) / pixel_size)  # y is flipped

        # Extract patch
        patch = wc_array[row_start : row_start + grid_px, col_start : col_start + grid_px]

        # Remap to our 6 classes and compute proportions
        proportions = np.zeros(N_CLASSES, dtype=np.float32)
        total_valid = 0
        for wc_code, our_class in CLASS_MAP.items():
            count = np.sum(patch == wc_code)
            proportions[our_class] += count
            total_valid += count

        if total_valid > 0:
            proportions /= total_valid

        record = {"cell_id": row.cell_id, "valid_pixels": int(total_valid)}
        for i, name in enumerate(CLASS_NAMES):
            record[name] = proportions[i]
        records.append(record)

    df = pd.DataFrame(records)
    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Get aligned bounds from Sentinel-2
    print("Step 1: Computing grid-aligned bounds from Sentinel-2...")
    bounds = get_aligned_bounds(SENTINEL_REF, GRID_SIZE_M)
    print(f"  Aligned bounds: {bounds}")

    # Step 2: Create grid
    print("\nStep 2: Creating 100m × 100m grid...")
    grid = create_grid(bounds, GRID_SIZE_M)

    # Save grid
    grid_path = os.path.join(PROCESSED_DIR, "grid.gpkg")
    grid.to_file(grid_path, driver="GPKG")
    print(f"  Saved grid -> {grid_path}")

    # Step 3: Reproject and aggregate labels for each year
    for year, filename in [
        (2020, "ESA_WorldCover_2020_N48E009.tif"),
        (2021, "ESA_WorldCover_2021_N48E009.tif"),
    ]:
        print(f"\nStep 3: Processing WorldCover {year}...")
        wc_path = os.path.join(LABELS_DIR, filename)

        # Reproject
        print(f"  Reprojecting {filename} to EPSG:32632...")
        wc_array, wc_transform = reproject_worldcover(wc_path, bounds)

        # Aggregate
        print(f"  Aggregating labels per grid cell...")
        labels_df = aggregate_labels(wc_array, wc_transform, grid)

        # Summary statistics
        print(f"\n  Label summary ({year}):")
        for name in CLASS_NAMES:
            col = labels_df[name]
            print(
                f"    {name:<15} mean={col.mean():.3f}  std={col.std():.3f}  "
                f"min={col.min():.3f}  max={col.max():.3f}"
            )
        print(f"    Valid cells: {(labels_df['valid_pixels'] > 0).sum()} / {len(labels_df)}")
        print(f"    Avg valid pixels/cell: {labels_df['valid_pixels'].mean():.1f}")

        # Save
        labels_path = os.path.join(PROCESSED_DIR, f"labels_{year}.parquet")
        labels_df.to_parquet(labels_path, index=False)
        print(f"  Saved labels -> {labels_path}")

    # Step 4: Compute change labels
    print("\nStep 4: Computing change labels (Δ = 2021 − 2020)...")
    labels_2020 = pd.read_parquet(os.path.join(PROCESSED_DIR, "labels_2020.parquet"))
    labels_2021 = pd.read_parquet(os.path.join(PROCESSED_DIR, "labels_2021.parquet"))

    change_df = pd.DataFrame({"cell_id": labels_2020["cell_id"]})
    for name in CLASS_NAMES:
        change_df[f"delta_{name}"] = labels_2021[name] - labels_2020[name]

    change_path = os.path.join(PROCESSED_DIR, "labels_change.parquet")
    change_df.to_parquet(change_path, index=False)

    print(f"  Change summary:")
    for name in CLASS_NAMES:
        col = change_df[f"delta_{name}"]
        print(
            f"    Δ {name:<15} mean={col.mean():+.4f}  std={col.std():.4f}  "
            f"[{col.min():+.3f}, {col.max():+.3f}]"
        )
    print(f"  Saved change labels -> {change_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("DONE! Processed files in data/processed/:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fp = os.path.join(PROCESSED_DIR, f)
        if os.path.isfile(fp):
            print(f"  {f}: {os.path.getsize(fp) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
