"""
Precompute binary label maps for the Nuremberg pixel dashboard.

For each year (2020, 2021) and resolution (1-10), creates:
  - nuremberg_labels_{year}_res{N}.bin  (uint8 flat array, 255=outside)
  - nuremberg_meta.json                 (bounds, dims per resolution)

Pixels outside the Nuremberg GeoJSON boundary → 255 (transparent).
Shrubland (class 1) → remapped to grassland (class 2) for Nuremberg.

Usage:
    .venv/Scripts/python.exe scripts/precompute_nuremberg_pixel.py
"""
import json
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import CITIES
from scripts.pixel_classifier_v2 import load_worldcover_pixels

OUT_DIR = os.path.join(PROJECT_ROOT, "src", "dashboard", "data", "nuremberg")
BOUNDARY_PATH = os.path.join(PROJECT_ROOT, "src", "dashboard", "data",
                             "nuremberg_boundary.geojson")

NUREMBERG = [c for c in CITIES if c.name == "nuremberg"][0]

# Classes for Nuremberg (no shrubland — remapped to grassland)
NUREMBERG_CLASSES = ["tree_cover", "grassland", "cropland",
                     "built_up", "bare_sparse", "water"]

# Remap: shrubland (1) → grassland (2), then shift classes 2+ down by 1
# Original: 0=tree, 1=shrub, 2=grass, 3=crop, 4=built, 5=bare, 6=water
# After remap: 0=tree, 1=grass, 2=crop, 3=built, 4=bare, 5=water
REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def ts():
    return time.strftime("%H:%M:%S")


def load_boundary_mask(anchor_path):
    """Rasterize the Nuremberg GeoJSON boundary onto the anchor grid."""
    import rasterio
    from rasterio.features import rasterize
    import fiona
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform

    with rasterio.open(anchor_path) as ref:
        anchor_crs = ref.crs
        anchor_transform = ref.transform
        H, W = ref.height, ref.width

    # Load GeoJSON polygons
    with fiona.open(BOUNDARY_PATH) as src:
        geojson_crs = src.crs
        geometries = [shape(f["geometry"]) for f in src]

    # Reproject from WGS84 to anchor CRS (EPSG:32632)
    transformer = Transformer.from_crs("EPSG:4326", anchor_crs, always_xy=True)
    reprojected = []
    for geom in geometries:
        reprojected.append(
            shapely_transform(transformer.transform, geom)
        )

    # Rasterize: 1 inside boundary, 0 outside
    mask = rasterize(
        [(geom, 1) for geom in reprojected],
        out_shape=(H, W),
        transform=anchor_transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def aggregate_labels(labels, mask, resolution):
    """Aggregate labels at a given resolution (NxN blocks).

    Returns (agg_labels, agg_H, agg_W) where agg_labels uses dominant class.
    """
    H, W = labels.shape
    N = resolution

    # Pad to multiple of N
    pad_H = (N - H % N) % N
    pad_W = (N - W % N) % N
    if pad_H or pad_W:
        labels = np.pad(labels, ((0, pad_H), (0, pad_W)),
                        constant_values=255)
        mask = np.pad(mask, ((0, pad_H), (0, pad_W)),
                      constant_values=0)

    new_H = labels.shape[0] // N
    new_W = labels.shape[1] // N

    agg = np.full((new_H, new_W), 255, dtype=np.uint8)

    for r in range(new_H):
        for c in range(new_W):
            block_labels = labels[r*N:(r+1)*N, c*N:(c+1)*N]
            block_mask = mask[r*N:(r+1)*N, c*N:(c+1)*N]

            # Only consider pixels inside boundary
            valid = block_labels[(block_mask > 0) & (block_labels < 255)]
            if len(valid) == 0:
                continue

            # Dominant class
            counts = np.bincount(valid, minlength=len(NUREMBERG_CLASSES))
            agg[r, c] = np.argmax(counts)

    return agg, new_H, new_W


def main():
    print(f"\n{'='*60}")
    print("  Precompute Nuremberg Pixel Dashboard Data")
    print(f"{'='*60}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load anchor info for bounds
    import rasterio
    anchor_path = os.path.join(PROJECT_ROOT, "data", "cities", "nuremberg",
                               "anchor_nuremberg.tif")
    with rasterio.open(anchor_path) as ref:
        anchor_transform = ref.transform
        H, W = ref.height, ref.width
        bounds = ref.bounds  # in EPSG:32632

    # Convert bounds to WGS84 for DeckGL
    from pyproj import Transformer
    transformer = Transformer.from_crs(ref.crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(bounds.left, bounds.bottom)
    east, north = transformer.transform(bounds.right, bounds.top)
    wgs84_bounds = [west, south, east, north]
    print(f"  Anchor: {W}x{H} pixels, WGS84 bounds: {wgs84_bounds}")

    # Load boundary mask
    print(f"[{ts()}] Rasterizing Nuremberg boundary...")
    boundary_mask = load_boundary_mask(anchor_path)
    inside_count = np.sum(boundary_mask > 0)
    print(f"  Pixels inside boundary: {inside_count:,} / {H*W:,} "
          f"({100*inside_count/(H*W):.1f}%)")

    meta = {
        "wgs84_bounds": wgs84_bounds,
        "classes": NUREMBERG_CLASSES,
        "years": [2020, 2021],
        "resolutions": {},
    }

    for year in [2020, 2021]:
        print(f"\n[{ts()}] Loading WorldCover {year} for Nuremberg...")
        raw_labels = load_worldcover_pixels(NUREMBERG, year=year)
        if raw_labels is None:
            print(f"  ERROR: No WorldCover data for {year}")
            continue

        # Remap shrubland → grassland and compact class IDs
        labels = np.full_like(raw_labels, 255)
        for old_cls, new_cls in REMAP.items():
            labels[raw_labels == old_cls] = new_cls

        # Apply boundary mask (outside → 255)
        labels[boundary_mask == 0] = 255

        valid = np.sum(labels < 255)
        print(f"  Valid pixels inside boundary: {valid:,}")

        # Class distribution
        for i, name in enumerate(NUREMBERG_CLASSES):
            count = np.sum(labels == i)
            pct = 100 * count / valid if valid > 0 else 0
            print(f"    {name:>15}: {count:>8,} ({pct:5.1f}%)")

        for res in range(1, 11):
            if res == 1:
                out_labels = labels.copy()
                out_labels[boundary_mask == 0] = 255
                out_H, out_W = H, W
            else:
                out_labels, out_H, out_W = aggregate_labels(
                    labels, boundary_mask, res)

            # Save binary
            fname = f"nuremberg_labels_{year}_res{res}.bin"
            fpath = os.path.join(OUT_DIR, fname)
            out_labels.tofile(fpath)
            fsize = os.path.getsize(fpath)
            print(f"  res={res:>2}: {out_W}x{out_H} -> {fname} "
                  f"({fsize/1024:.1f} KB)")

            # Store meta for this resolution
            res_key = f"res{res}"
            if res_key not in meta["resolutions"]:
                meta["resolutions"][res_key] = {
                    "width": out_W,
                    "height": out_H,
                }

    # Save metadata
    meta_path = os.path.join(OUT_DIR, "nuremberg_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[{ts()}] Saved metadata to {meta_path}")
    print(f"[{ts()}] Done!")


if __name__ == "__main__":
    main()
