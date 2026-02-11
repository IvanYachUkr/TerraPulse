"""
Create the canonical spatial anchor for the TerraPulse v2 pipeline.

This script produces a deterministic reference grid that all v2 rasters
must match exactly. Every Sentinel-2 composite, WorldCover layer, and
100m analysis cell is aligned to this anchor.

Outputs:
    data/grid/anchor_utm32632_10m.tif   – 1-band float32 dummy (nodata)
    data/grid/anchor.json               – full geometry spec

Rules:
    1. Project AOI bbox from EPSG:4326 → 32632 (densify_pts=21)
    2. Snap projected bounds to 10m grid lines (floor/ceil)
    3. Pad width/height to multiples of block (grid_size_m / pixel_size)
    4. Recompute right/bottom so transform + shape are coherent
    5. Assert grid_size_m % pixel_size == 0

Usage:
    python scripts/create_anchor.py
"""

import json
import os
import sys
from math import ceil, floor

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

# -- Load config ---------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG, PROJECT_ROOT

# -- Parameters from config ----------------------------------------------------
bbox_wgs84 = CFG["aoi"]["bbox"]  # [west, south, east, north] in EPSG:4326
target_epsg = CFG["aoi"]["epsg"]  # 32632
pixel_size = float(CFG["sentinel2"]["resolution"])  # 10.0
grid_size_m = float(CFG["grid"]["size_m"])  # 100.0
nodata = CFG["sentinel2"]["nodata"]  # -9999

# Block = number of pixels per grid cell side
block = int(grid_size_m / pixel_size)  # 10
assert grid_size_m % pixel_size == 0, (
    f"grid_size_m ({grid_size_m}) must be divisible by pixel_size ({pixel_size})"
)
assert CFG["grid"]["pixel_size"] == CFG["sentinel2"]["resolution"], (
    f"grid.pixel_size ({CFG['grid']['pixel_size']}) != sentinel2.resolution ({CFG['sentinel2']['resolution']})"
)

# -- Output paths --------------------------------------------------------------
grid_dir = os.path.join(PROJECT_ROOT, "data", "grid")
os.makedirs(grid_dir, exist_ok=True)

anchor_tif = os.path.join(grid_dir, "anchor_utm32632_10m.tif")
anchor_json = os.path.join(grid_dir, "anchor.json")


def main():
    print("=" * 60)
    print("Creating canonical spatial anchor")
    print("=" * 60)

    # ── Step 1: Project bbox from EPSG:4326 → target CRS ──────────────────
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(target_epsg)

    left, bottom, right, top = transform_bounds(
        src_crs, dst_crs,
        bbox_wgs84[0], bbox_wgs84[1], bbox_wgs84[2], bbox_wgs84[3],
        densify_pts=21,
    )
    print(f"  WGS84 bbox:     {bbox_wgs84}")
    print(f"  Projected bbox:  left={left:.2f} bottom={bottom:.2f} right={right:.2f} top={top:.2f}")

    # ── Step 2: Snap bounds to pixel_size grid lines ──────────────────────
    left_snapped = floor(left / pixel_size) * pixel_size
    bottom_snapped = floor(bottom / pixel_size) * pixel_size
    right_snapped = ceil(right / pixel_size) * pixel_size
    top_snapped = ceil(top / pixel_size) * pixel_size

    print(f"  Snapped bounds:  left={left_snapped:.2f} bottom={bottom_snapped:.2f} "
          f"right={right_snapped:.2f} top={top_snapped:.2f}")

    # ── Step 3: Compute initial width/height, then pad to block multiples ─
    width0 = round((right_snapped - left_snapped) / pixel_size)
    height0 = round((top_snapped - bottom_snapped) / pixel_size)

    width = ceil(width0 / block) * block
    height = ceil(height0 / block) * block

    print(f"  Initial pixels:  {width0} x {height0}")
    print(f"  Padded pixels:   {width} x {height}  (block={block})")
    print(f"  Grid cells:      {width // block} cols x {height // block} rows = "
          f"{(width // block) * (height // block)} cells")

    # ── Step 4: Recompute right/bottom after padding ──────────────────────
    # For a north-up raster: x goes left→right, y goes top→bottom
    right_final = left_snapped + width * pixel_size
    bottom_final = top_snapped - height * pixel_size

    print(f"  Final bounds:    left={left_snapped:.2f} bottom={bottom_final:.2f} "
          f"right={right_final:.2f} top={top_snapped:.2f}")

    # ── Step 5: Build affine transform ────────────────────────────────────
    transform = Affine(pixel_size, 0.0, left_snapped,
                       0.0, -pixel_size, top_snapped)

    # Sanity: verify transform maps pixel corners to expected bounds
    assert abs(transform.c - left_snapped) < 1e-6
    assert abs(transform.f - top_snapped) < 1e-6
    x_right = transform.c + width * transform.a
    y_bottom = transform.f + height * transform.e
    assert abs(x_right - right_final) < 1e-6, f"{x_right} != {right_final}"
    assert abs(y_bottom - bottom_final) < 1e-6, f"{y_bottom} != {bottom_final}"

    print(f"  Transform:       {transform}")

    # ── Step 6: Write anchor GeoTIFF ──────────────────────────────────────
    data = np.full((1, height, width), nodata, dtype=np.float32)

    with rasterio.open(
        anchor_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=dst_crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "ANCHOR_DUMMY")
        dst.update_tags(
            DESCRIPTION="Canonical spatial anchor for TerraPulse v2 pipeline",
            PIXEL_SIZE=str(pixel_size),
            GRID_SIZE_M=str(grid_size_m),
            BLOCK_PX=str(block),
            N_CELLS=str((width // block) * (height // block)),
        )

    size_kb = os.path.getsize(anchor_tif) / 1024
    print(f"  Wrote: {anchor_tif} ({size_kb:.1f} KB)")

    # ── Step 7: Write anchor.json ─────────────────────────────────────────
    anchor_meta = {
        "description": "Canonical spatial anchor for TerraPulse v2 pipeline",
        "epsg": target_epsg,
        "crs": f"EPSG:{target_epsg}",
        "pixel_size": pixel_size,
        "grid_size_m": grid_size_m,
        "block_px": block,
        "bounds_wgs84": {
            "west": bbox_wgs84[0],
            "south": bbox_wgs84[1],
            "east": bbox_wgs84[2],
            "north": bbox_wgs84[3],
        },
        "bounds_projected": {
            "left": left_snapped,
            "bottom": bottom_final,
            "right": right_final,
            "top": top_snapped,
        },
        "width": width,
        "height": height,
        "n_cols": width // block,
        "n_rows": height // block,
        "n_cells": (width // block) * (height // block),
        "transform": list(transform)[:6],
    }

    with open(anchor_json, "w") as f:
        json.dump(anchor_meta, f, indent=2)

    print(f"  Wrote: {anchor_json}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n  [OK] Anchor created successfully.")
    print(f"    {width}×{height} pixels @ {pixel_size}m = "
          f"{width // block}×{height // block} = "
          f"{(width // block) * (height // block)} cells of {grid_size_m}m")


if __name__ == "__main__":
    main()
