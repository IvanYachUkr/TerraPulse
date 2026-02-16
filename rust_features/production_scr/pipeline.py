#!/usr/bin/env python3
"""
Production Pipeline: Download -> Extract -> Train -> Predict (2020-2025).

Fully self-contained orchestrator.  Only external dependency beyond pip
packages is the compiled terrapulse_features Rust extension (see lib.rs).

All config values are inlined — no project imports required.
Intermediate results are checkpointed so crashed runs can resume.

Usage:
    # Full Nuremberg pipeline (train + predict)
    python pipeline.py [--skip-download] [--skip-extract] [--skip-train] [--skip-predict]

    # Inference for arbitrary region (uses pre-trained Nuremberg models)
    python pipeline.py infer --bbox 11.4 48.0 11.7 48.2 --name munich
"""

import argparse
import dataclasses
import json
import math
import os
import pickle
import re
import sys
import time
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

# =====================================================================
# Inlined Config (from config/data_config.yml)
# =====================================================================

AOI_BBOX = [10.95, 49.38, 11.20, 49.52]
AOI_EPSG = 32632

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SENTINEL_RES = 10
SENTINEL_NODATA = -9999
MIN_SCENES = 8

SEASON_DATES = {
    "spring": ("04-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "autumn": ("09-01", "10-31"),
}
CLOUD_COVER_MAX = {"spring": 40, "summer": 20, "autumn": 40}
SCL_EXCLUDE = [0, 1, 2, 3, 8, 9, 10, 11]

GRID_PX = 10  # pixels per cell side
GRID_SIZE_M = GRID_PX * SENTINEL_RES  # 100m

WC_CLASS_MAP = {10: 0, 30: 1, 90: 1, 40: 2, 50: 3, 60: 4, 80: 5}
WC_TILE = "N48E009"
WC_YEARS = [2020, 2021]

SPLIT_BLOCK_ROWS = 10   # cells per tile vertically (10 x 100m = 1km)
SPLIT_BLOCK_COLS = 10   # cells per tile horizontally
BUFFER_TILES = 1        # Chebyshev tile buffer around test folds

ALL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LABELED_YEARS = [2020, 2021]
PREDICT_YEARS = [2022, 2023, 2024, 2025]
SEASONS = ["spring", "summer", "autumn"]

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)
SEED = 42
N_FOLDS = 5

MIN_VALID_FRAC = 0.3

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}

# Feature group prefixes
BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
}


# =====================================================================
# Path Setup
# =====================================================================

def find_project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "data")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot find project root (no 'data' dir found)")


PROJECT_ROOT = find_project_root()
RAW_V2_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "v2")
PROCESSED_V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
LABELS_DIR_PATH = os.path.join(PROJECT_ROOT, "data", "labels")
GRID_DIR = os.path.join(PROJECT_ROOT, "data", "grid")
GRID_REF_PATH = os.path.join(GRID_DIR, "anchor_utm32632_10m.tif")
GRID_JSON_PATH = os.path.join(GRID_DIR, "anchor.json")

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "pipeline_output")
FEATURES_DIR = os.path.join(OUT_DIR, "features")
MODELS_DIR = os.path.join(OUT_DIR, "models")
PREDICTIONS_DIR = os.path.join(OUT_DIR, "predictions")
REGIONS_DIR = os.path.join(OUT_DIR, "regions")


def ts():
    return time.strftime("%H:%M:%S")


def ensure_dirs():
    for d in [OUT_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR,
              RAW_V2_DIR, GRID_DIR, PROCESSED_V2_DIR]:
        os.makedirs(d, exist_ok=True)


# =====================================================================
# Multi-Region Context
# =====================================================================

@dataclasses.dataclass
class RegionCtx:
    """Runtime context for a non-default region (inference mode)."""
    name: str
    bbox: list             # [west, south, east, north]
    epsg: int              # UTM EPSG code
    wc_tile: str           # ESA WorldCover tile ID
    years: list            # years to process
    grid_dir: str = ""
    raw_dir: str = ""
    features_dir: str = ""
    predictions_dir: str = ""
    grid_ref_path: str = ""
    grid_json_path: str = ""


def _auto_epsg(bbox):
    """UTM zone EPSG from bbox center longitude."""
    lon = (bbox[0] + bbox[2]) / 2
    zone = int((lon + 180) / 6) + 1
    lat = (bbox[1] + bbox[3]) / 2
    return 32600 + zone if lat >= 0 else 32700 + zone


def _auto_wc_tile(bbox):
    """ESA WorldCover 3-degree tile ID from bbox center."""
    lat = (bbox[1] + bbox[3]) / 2
    lon = (bbox[0] + bbox[2]) / 2
    # Tile anchors at multiples of 3 degrees
    lat_anchor = int(lat // 3 * 3)
    lon_anchor = int(lon // 3 * 3)
    ns = "N" if lat_anchor >= 0 else "S"
    ew = "E" if lon_anchor >= 0 else "W"
    return f"{ns}{abs(lat_anchor):02d}{ew}{abs(lon_anchor):03d}"


def build_region_ctx(name, bbox, years=None):
    """Build a RegionCtx with auto-detected EPSG, WC tile, and paths."""
    region_root = os.path.join(REGIONS_DIR, name)
    grid_dir = os.path.join(region_root, "grid")
    return RegionCtx(
        name=name,
        bbox=bbox,
        epsg=_auto_epsg(bbox),
        wc_tile=_auto_wc_tile(bbox),
        years=years or PREDICT_YEARS,
        grid_dir=grid_dir,
        raw_dir=os.path.join(region_root, "raw"),
        features_dir=os.path.join(region_root, "features"),
        predictions_dir=os.path.join(region_root, "predictions"),
        grid_ref_path=os.path.join(grid_dir, f"anchor_utm{_auto_epsg(bbox)}_10m.tif"),
        grid_json_path=os.path.join(grid_dir, "anchor.json"),
    )


# =====================================================================
# Spatial Splitting (inlined)
# =====================================================================

@lru_cache(maxsize=8)
def _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles):
    neighbors = {}
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tid = tr * n_tile_cols + tc
            nbrs = set()
            for dr in range(-buffer_tiles, buffer_tiles + 1):
                for dc in range(-buffer_tiles, buffer_tiles + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                        nbrs.add(nr * n_tile_cols + nc)
            neighbors[tid] = frozenset(nbrs)
    return neighbors


def get_fold_indices(groups, fold_assignments, fold_idx,
                     n_tile_cols, n_tile_rows, buffer_tiles=1):
    test_mask = fold_assignments == fold_idx
    if buffer_tiles > 0:
        test_tiles = set(np.unique(groups[test_mask]))
        nbr_map = _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles)
        buf = set()
        for tt in test_tiles:
            for n in nbr_map.get(tt, set()):
                if n not in test_tiles:
                    buf.add(n)
        train_mask = (~test_mask) & (~np.isin(groups, list(buf)))
    else:
        train_mask = ~test_mask
    return np.where(train_mask)[0], np.where(test_mask)[0]


# =====================================================================
# Evaluation (inlined)
# =====================================================================

def evaluate_predictions(y_true, y_pred, per_class=False):
    """Compute R2 (uniform) and MAE (pp), optionally with per-class breakdown."""
    from sklearn.metrics import mean_absolute_error, r2_score
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    mae = float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")) * 100
    if not per_class:
        return r2, mae
    pc = {}
    for i, cn in enumerate(CLASS_NAMES):
        pc[cn] = {
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
            "mae_pp": float(mean_absolute_error(y_true[:, i], y_pred[:, i])) * 100,
        }
    return r2, mae, pc


# =====================================================================
# STAGE 0: ANCHOR GRID
# =====================================================================

def create_anchor(ctx=None):
    """Create the canonical spatial anchor grid if it doesn't exist.

    Produces a deterministic reference GeoTIFF that all downloads are
    warped to.  Algorithm: project AOI bbox from EPSG:4326 -> target EPSG,
    snap to pixel_size grid lines, pad to grid_size_m block multiples.
    """
    ref_path = ctx.grid_ref_path if ctx else GRID_REF_PATH
    json_path = ctx.grid_json_path if ctx else GRID_JSON_PATH
    g_dir = ctx.grid_dir if ctx else GRID_DIR
    bbox = ctx.bbox if ctx else AOI_BBOX
    epsg = ctx.epsg if ctx else AOI_EPSG

    if os.path.exists(ref_path):
        return  # already exists

    from math import ceil, floor
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    region_label = f" [{ctx.name}]" if ctx else ""
    print(f"\n{'='*70}")
    print(f"STAGE 0: CREATE ANCHOR GRID{region_label}")
    print(f"{'='*70}")

    pixel_size = float(SENTINEL_RES)
    block = GRID_PX

    # Step 1: Project bbox
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(epsg)
    left, bottom, right, top = transform_bounds(
        src_crs, dst_crs,
        bbox[0], bbox[1], bbox[2], bbox[3],
        densify_pts=21,
    )

    # Step 2: Snap to pixel grid
    left_s = floor(left / pixel_size) * pixel_size
    bottom_s = floor(bottom / pixel_size) * pixel_size
    right_s = ceil(right / pixel_size) * pixel_size
    top_s = ceil(top / pixel_size) * pixel_size

    # Step 3: Pad to block multiples
    width = ceil(round((right_s - left_s) / pixel_size) / block) * block
    height = ceil(round((top_s - bottom_s) / pixel_size) / block) * block
    right_f = left_s + width * pixel_size
    bottom_f = top_s - height * pixel_size

    n_cols = width // block
    n_rows = height // block
    n_cells = n_cols * n_rows
    print(f"  Grid: {width}x{height} px @ {pixel_size}m = {n_cols}x{n_rows} = {n_cells} cells")

    # Step 4: Build transform
    transform = Affine(pixel_size, 0.0, left_s, 0.0, -pixel_size, top_s)

    # Step 5: Write anchor GeoTIFF
    import rasterio
    os.makedirs(g_dir, exist_ok=True)

    data = np.full((1, height, width), SENTINEL_NODATA, dtype=np.float32)
    with rasterio.open(
        ref_path, "w", driver="GTiff",
        height=height, width=width, count=1, dtype="float32",
        crs=dst_crs, transform=transform, nodata=SENTINEL_NODATA,
        compress="lzw",
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "ANCHOR_DUMMY")
        dst.update_tags(
            DESCRIPTION="Canonical spatial anchor for TerraPulse pipeline",
            PIXEL_SIZE=str(pixel_size), GRID_SIZE_M=str(GRID_SIZE_M),
            BLOCK_PX=str(block), N_CELLS=str(n_cells),
        )

    # Step 6: Write anchor.json
    anchor_meta = {
        "description": "Canonical spatial anchor for TerraPulse pipeline",
        "epsg": epsg, "crs": f"EPSG:{epsg}",
        "pixel_size": pixel_size, "grid_size_m": GRID_SIZE_M, "block_px": block,
        "bounds_wgs84": {"west": bbox[0], "south": bbox[1],
                         "east": bbox[2], "north": bbox[3]},
        "bounds_projected": {"left": left_s, "bottom": bottom_f,
                             "right": right_f, "top": top_s},
        "width": width, "height": height,
        "n_cols": n_cols, "n_rows": n_rows, "n_cells": n_cells,
        "transform": list(transform)[:6],
    }
    if ctx:
        anchor_meta["region"] = ctx.name
    with open(json_path, "w") as f:
        json.dump(anchor_meta, f, indent=2)

    kb = os.path.getsize(ref_path) / 1024
    print(f"  Wrote: {ref_path} ({kb:.1f} KB)")
    print(f"  Wrote: {json_path}")


# =====================================================================
# STAGE 0b: PREPARE LABELS  (WorldCover → per-cell class proportions)
# =====================================================================

def _reproject_worldcover_to_anchor(wc_path, anchor):
    """Reproject WorldCover from EPSG:4326 to anchor grid (nearest)."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    dst_array = np.zeros((anchor["height"], anchor["width"]), dtype=np.uint8)
    with rasterio.open(wc_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst_array,
            src_transform=src.transform, src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=anchor["transform"], dst_crs=anchor["crs"],
            dst_nodata=0, resampling=Resampling.nearest,
        )
    return dst_array


def _aggregate_labels(wc_array, n_cols_cells, n_rows_cells):
    """Compute class proportions per cell from reprojected WorldCover."""
    block = GRID_PX
    total_px = block * block
    records = []
    cell_id = 0
    for row_idx in range(n_rows_cells):
        for col_idx in range(n_cols_cells):
            r0, c0 = row_idx * block, col_idx * block
            patch = wc_array[r0:r0 + block, c0:c0 + block]
            proportions = np.zeros(N_CLASSES, dtype=np.float32)
            mapped = 0
            for wc_code, our_class in WC_CLASS_MAP.items():
                count = int(np.sum(patch == wc_code))
                proportions[our_class] += count
                mapped += count
            coverage = mapped / total_px if total_px > 0 else 0.0
            if total_px > 0:
                proportions /= total_px
            record = {
                "cell_id": cell_id,
                "mapped_pixels": mapped,
                "unmapped_pixels": total_px - mapped,
                "coverage": float(coverage),
            }
            for i, name in enumerate(CLASS_NAMES):
                record[name] = float(proportions[i])
            records.append(record)
            cell_id += 1
    return pd.DataFrame(records)


def prepare_labels():
    """Create labels_20xx.parquet from ESA WorldCover if not present.

    Reads the anchor GeoTIFF for grid geometry, reprojects WorldCover
    to the anchor CRS, and computes per-cell class proportions.
    """
    # Check if labels already exist
    all_exist = all(
        os.path.exists(os.path.join(PROCESSED_V2_DIR, f"labels_{y}.parquet"))
        for y in WC_YEARS
    )
    if all_exist:
        return

    import rasterio

    print(f"\n{'='*70}")
    print(f"STAGE 0b: PREPARE LABELS (WorldCover -> class proportions)")
    print(f"{'='*70}")

    # Read anchor geometry
    assert os.path.exists(GRID_REF_PATH), (
        f"Missing anchor: {GRID_REF_PATH}. Run pipeline with download stage first."
    )
    with rasterio.open(GRID_REF_PATH) as src:
        anchor = {
            "crs": src.crs, "transform": src.transform,
            "width": src.width, "height": src.height,
        }

    block = GRID_PX
    assert anchor["width"] % block == 0 and anchor["height"] % block == 0
    n_cols_cells = anchor["width"] // block
    n_rows_cells = anchor["height"] // block
    print(f"  Anchor: {anchor['width']}x{anchor['height']} px -> "
          f"{n_cols_cells}x{n_rows_cells} = {n_cols_cells * n_rows_cells} cells")

    for year in WC_YEARS:
        labels_path = os.path.join(PROCESSED_V2_DIR, f"labels_{year}.parquet")
        if os.path.exists(labels_path):
            print(f"  [{year}] Already exists -- skip")
            continue

        # Find WorldCover file (two naming conventions)
        wc_path = os.path.join(LABELS_DIR_PATH, f"ESA_WorldCover_{year}_{WC_TILE}_Map.tif")
        if not os.path.exists(wc_path):
            wc_path = os.path.join(LABELS_DIR_PATH, f"ESA_WorldCover_{year}_{WC_TILE}.tif")
        if not os.path.exists(wc_path):
            # Auto-download from ESA public S3 (no auth required)
            import urllib.request
            os.makedirs(LABELS_DIR_PATH, exist_ok=True)
            wc_version = "v100" if year == 2020 else "v200"
            url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                   f"{wc_version}/{year}/map/"
                   f"ESA_WorldCover_10m_{year}_{wc_version}_{WC_TILE}_Map.tif")
            wc_path = os.path.join(LABELS_DIR_PATH, f"ESA_WorldCover_{year}_{WC_TILE}_Map.tif")
            print(f"  [{year}] Downloading WorldCover from ESA S3...")
            print(f"    URL: {url}")
            try:
                urllib.request.urlretrieve(url, wc_path)
                size_mb = os.path.getsize(wc_path) / (1024 * 1024)
                print(f"  [{year}] Downloaded: {os.path.basename(wc_path)} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  [{year}] ERROR: Download failed ({e})")
                if os.path.exists(wc_path):
                    os.remove(wc_path)
                continue

        print(f"  [{year}] Reprojecting {os.path.basename(wc_path)}...")
        wc_array = _reproject_worldcover_to_anchor(wc_path, anchor)
        print(f"  [{year}] Aggregating labels per cell...")
        labels_df = _aggregate_labels(wc_array, n_cols_cells, n_rows_cells)

        # Summary
        for name in CLASS_NAMES:
            col = labels_df[name]
            print(f"    {name:<15} mean={col.mean():.3f}  std={col.std():.3f}")
        labels_df.to_parquet(labels_path, index=False)
        print(f"  [{year}] Saved: {labels_path}")


# =====================================================================
# STAGE 0c: PREPARE SPATIAL SPLIT  (region growing on tile graph)
# =====================================================================

def _build_tile_adjacency(unique_tiles, n_tile_cols, n_tile_rows):
    """Build 4-connected (rook) adjacency dict on tile grid."""
    tile_set = set(unique_tiles.tolist())
    tile_adj = {t: [] for t in unique_tiles}
    for t in unique_tiles:
        tr, tc = t // n_tile_cols, t % n_tile_cols
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                nbr = nr * n_tile_cols + nc
                if nbr in tile_set:
                    tile_adj[t].append(nbr)
    return tile_adj


def _tile_manhattan(t1, t2, n_tile_cols):
    """Manhattan distance between two tiles."""
    r1, c1 = t1 // n_tile_cols, t1 % n_tile_cols
    r2, c2 = t2 // n_tile_cols, t2 % n_tile_cols
    return abs(r1 - r2) + abs(c1 - c2)


def _farthest_point_seeds(unique_tiles, n_folds, n_tile_cols, start_tile):
    """Pick K seed tiles maximally spread via greedy farthest-point."""
    seeds = [start_tile]
    for _ in range(1, n_folds):
        best_tile, best_min_dist = None, -1
        for t in unique_tiles:
            if t in seeds:
                continue
            min_d = min(_tile_manhattan(t, s, n_tile_cols) for s in seeds)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_tile = t
        seeds.append(best_tile)
    return seeds


def _grow_regions(seeds, tile_adj, tile_weight, n_folds, n_tiles,
                  unique_tiles, n_tile_cols, rng):
    """Greedy BFS region growing from seeds (lightest-first, compact)."""
    assigned = {}
    region_weight = [0] * n_folds
    frontier = [set() for _ in range(n_folds)]
    for fold_idx, s in enumerate(seeds):
        assigned[s] = fold_idx
        region_weight[fold_idx] += tile_weight[s]
        for nbr in tile_adj[s]:
            if nbr not in assigned:
                frontier[fold_idx].add(nbr)

    while len(assigned) < n_tiles:
        order = sorted(range(n_folds), key=lambda i: region_weight[i])
        grown = False
        for fold_idx in order:
            if not frontier[fold_idx]:
                continue
            candidates = list(frontier[fold_idx] - set(assigned.keys()))
            if not candidates:
                frontier[fold_idx].clear()
                continue
            seed_t = seeds[fold_idx]
            candidates.sort(key=lambda t: (
                _tile_manhattan(t, seed_t, n_tile_cols), rng.random()
            ))
            chosen = candidates[0]
            assigned[chosen] = fold_idx
            region_weight[fold_idx] += tile_weight[chosen]
            frontier[fold_idx].discard(chosen)
            for nbr in tile_adj[chosen]:
                if nbr not in assigned:
                    frontier[fold_idx].add(nbr)
            grown = True
            break
        if not grown:
            for t in unique_tiles:
                if t not in assigned:
                    dists = [_tile_manhattan(t, seeds[i], n_tile_cols)
                             for i in range(n_folds)]
                    assigned[t] = int(np.argmin(dists))
                    region_weight[assigned[t]] += tile_weight[t]
    return assigned, region_weight


def _score_partition(region_weight, assigned, tile_adj, n_folds, unique_tiles):
    """Score partition by balance + contiguity (lower is better)."""
    total = sum(region_weight)
    target = total / n_folds
    balance_penalty = max(abs(w - target) / target for w in region_weight)
    contiguity_penalty = 0
    for fold_idx in range(n_folds):
        fold_tiles = {t for t, f in assigned.items() if f == fold_idx}
        if not fold_tiles:
            contiguity_penalty += 1
            continue
        visited = set()
        n_components = 0
        for start in fold_tiles:
            if start in visited:
                continue
            n_components += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for nbr in tile_adj[node]:
                    if nbr in fold_tiles and nbr not in visited:
                        stack.append(nbr)
        contiguity_penalty += n_components - 1
    return balance_penalty + 10.0 * contiguity_penalty


def _build_region_growing_folds(groups, n_folds, n_tile_cols, n_tile_rows,
                                seed=42, n_starts=10):
    """Multi-start region growing fold assignment on tile graph."""
    rng = np.random.RandomState(seed)
    unique_tiles = np.unique(groups)
    n_tiles = len(unique_tiles)
    tile_ids, counts = np.unique(groups, return_counts=True)
    tile_weight = dict(zip(tile_ids.tolist(), counts.tolist()))
    tile_adj = _build_tile_adjacency(unique_tiles, n_tile_cols, n_tile_rows)

    best_score = float('inf')
    best_assigned = None
    n_starts = max(1, int(n_starts))
    n_extra = min(n_starts - 1, len(unique_tiles) - 1)
    start_tiles = [unique_tiles[0]]
    if n_extra > 0:
        start_tiles.extend(rng.choice(unique_tiles[1:], size=n_extra, replace=False))

    for start_tile in start_tiles:
        seeds = _farthest_point_seeds(unique_tiles, n_folds, n_tile_cols, start_tile)
        assigned, region_weight = _grow_regions(
            seeds, tile_adj, tile_weight, n_folds, n_tiles,
            unique_tiles, n_tile_cols, rng,
        )
        score = _score_partition(region_weight, assigned, tile_adj, n_folds, unique_tiles)
        if score < best_score:
            best_score = score
            best_assigned = assigned

    fold_assignments = np.array([best_assigned[g] for g in groups], dtype=int)
    return fold_assignments


def prepare_spatial_split():
    """Create split_spatial.parquet + metadata if not present.

    Assigns cells to 10x10 tile groups, then uses multi-start region
    growing (10 restarts) to produce 5 spatially contiguous folds.
    """
    split_path = os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet")
    meta_path = os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")
    if os.path.exists(split_path) and os.path.exists(meta_path):
        return

    from datetime import datetime, timezone

    print(f"\n{'='*70}")
    print(f"STAGE 0c: PREPARE SPATIAL SPLIT (region growing)")
    print(f"{'='*70}")

    # Need labels to know how many cells
    labels_path = os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet")
    assert os.path.exists(labels_path), (
        f"Missing {labels_path}. Run prepare_labels() first."
    )
    labels_df = pd.read_parquet(labels_path, columns=["cell_id"])
    cell_ids = labels_df["cell_id"].values
    n = len(cell_ids)

    # Get grid geometry from anchor
    anchor_json = os.path.join(GRID_DIR, "anchor.json")
    if os.path.exists(anchor_json):
        with open(anchor_json) as f:
            ameta = json.load(f)
        n_gcols = ameta["n_cols"]
        n_grows = ameta["n_rows"]
    else:
        # Fallback: derive from anchor GeoTIFF
        import rasterio
        with rasterio.open(GRID_REF_PATH) as src:
            n_gcols = src.width // GRID_PX
            n_grows = src.height // GRID_PX

    assert n_gcols * n_grows == n, (
        f"Grid mismatch: {n_grows}x{n_gcols}={n_grows*n_gcols} != {n}"
    )
    print(f"  Grid: {n_grows} rows x {n_gcols} cols = {n} cells")

    # Tile assignment
    row_idx = cell_ids // n_gcols
    col_idx = cell_ids % n_gcols
    tile_row = row_idx // SPLIT_BLOCK_ROWS
    tile_col = col_idx // SPLIT_BLOCK_COLS
    n_tile_cols = int(tile_col.max()) + 1
    n_tile_rows = int(tile_row.max()) + 1
    groups = tile_row * n_tile_cols + tile_col
    n_groups = len(np.unique(groups))
    print(f"  Tiles: {n_tile_rows}x{n_tile_cols} = {n_groups} groups "
          f"({SPLIT_BLOCK_ROWS}x{SPLIT_BLOCK_COLS} cells each)")

    # Region growing fold assignment
    print(f"  Running multi-start region growing ({N_FOLDS} folds, 10 starts)...")
    fold_assignments = _build_region_growing_folds(
        groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED, n_starts=10
    )

    # Print fold sizes
    for fi in range(N_FOLDS):
        cnt = int((fold_assignments == fi).sum())
        print(f"    Fold {fi}: {cnt} cells ({cnt/n:.1%})")

    # Save
    split_df = pd.DataFrame({
        "cell_id": cell_ids,
        "fold_region_growing": fold_assignments,
        "tile_group": groups,
    })
    split_df.to_parquet(split_path, index=False)
    print(f"  Saved: {split_path}")

    # Metadata
    meta = {
        "primary_fold_column": "fold_region_growing",
        "block_rows": SPLIT_BLOCK_ROWS, "block_cols": SPLIT_BLOCK_COLS,
        "cell_size_m": GRID_SIZE_M,
        "block_size_m": f"{SPLIT_BLOCK_ROWS * GRID_SIZE_M}x{SPLIT_BLOCK_COLS * GRID_SIZE_M}",
        "n_folds": N_FOLDS, "seed": SEED, "buffer_tiles": BUFFER_TILES,
        "buffer_metric": "chebyshev",
        "region_growing_n_starts": 10,
        "n_cells": n, "n_groups": n_groups,
        "grid_rows": n_grows, "grid_cols": n_gcols,
        "tile_rows": n_tile_rows, "tile_cols": n_tile_cols,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")


# =====================================================================
# STAGE 1: DOWNLOAD
# =====================================================================

def download_season(year, season, ctx=None):
    """Download one Sentinel-2 v2 composite via Planetary Computer."""
    import planetary_computer
    import pystac_client
    import rasterio
    import stackstac
    import xarray as xr
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    raw_dir = ctx.raw_dir if ctx else RAW_V2_DIR
    ref_path = ctx.grid_ref_path if ctx else GRID_REF_PATH
    bbox = ctx.bbox if ctx else AOI_BBOX
    epsg = ctx.epsg if ctx else AOI_EPSG
    region_name = ctx.name if ctx else "nuremberg"

    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"sentinel2_{region_name}_{year}_{season}.tif")
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{year}/{season}] Already exists ({mb:.1f} MB) -- skip")
        return

    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    s_date = f"{year}-{SEASON_DATES[season][0]}"
    e_date = f"{year}-{SEASON_DATES[season][1]}"

    for cloud_max in [40, 50, 60]:
        items = catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{s_date}/{e_date}",
            query={"eo:cloud_cover": {"lt": cloud_max}},
        ).item_collection()
        if len(items) >= MIN_SCENES:
            break

    if len(items) < MIN_SCENES:
        from datetime import datetime, timedelta
        s = datetime.strptime(s_date, "%Y-%m-%d")
        e = datetime.strptime(e_date, "%Y-%m-%d")
        s_date = (s - timedelta(days=14)).strftime("%Y-%m-%d")
        e_date = (e + timedelta(days=14)).strftime("%Y-%m-%d")
        items = catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{s_date}/{e_date}",
            query={"eo:cloud_cover": {"lt": 60}},
        ).item_collection()

    n_scenes = len(items)
    if n_scenes == 0:
        print(f"  [{year}/{season}] WARNING: No scenes found -- skipping!")
        return
    print(f"  [{year}/{season}] {n_scenes} scenes, compositing...")

    warnings.filterwarnings("ignore", module="stackstac")
    spectral = stackstac.stack(
        items, assets=SENTINEL_BANDS, bounds_latlon=bbox,
        resolution=SENTINEL_RES, epsg=epsg, dtype="float64",
        fill_value=np.nan, resampling=Resampling.bilinear, chunksize=1024,
        rescale=False,
    )
    scl = stackstac.stack(
        items, assets=["SCL"], bounds_latlon=bbox,
        resolution=SENTINEL_RES, epsg=epsg, dtype="float64",
        fill_value=np.nan, resampling=Resampling.nearest, chunksize=1024,
        rescale=False,
    ).sel(band="SCL")

    spectral, scl = xr.align(spectral, scl, join="exact")
    spectral = spectral.sel(band=SENTINEL_BANDS)

    import dask.array as da
    scl_vals = scl.data
    valid = xr.DataArray(da.isfinite(scl_vals), coords=scl.coords, dims=scl.dims)
    for cls in SCL_EXCLUDE:
        valid = valid & (scl != cls)

    valid_frac_xr = valid.mean(dim="time").astype("float32")
    composite_xr = (spectral.where(valid).median(dim="time", skipna=True)
                    .astype("float32"))

    print(f"  [{year}/{season}] Computing median composite...")
    composite = composite_xr.compute().values
    valid_fraction = valid_frac_xr.compute().values

    # PB 04.00 offset correction (Jan 2022+): subtract 1000 DN from
    # spectral bands.  stackstac(rescale=False) stores raw DN, but since
    # PB 04.00 Sentinel-2 L2A includes BOA_ADD_OFFSET = -1000.
    # Correcting here harmonizes all years to the same DN / 10000 scale.
    if year >= 2022:
        composite = composite - 1000.0
        composite = np.maximum(composite, 0.0)
        print(f"  [{year}/{season}] Applied PB 04.00 offset correction (-1000 DN)")

    xs = np.asarray(composite_xr.coords["x"].values)
    ys = np.asarray(composite_xr.coords["y"].values)
    rx = float(np.abs(xs[1] - xs[0]))
    ry = float(np.abs(ys[1] - ys[0]))
    src_transform = rasterio.transform.from_bounds(
        float(xs.min()) - rx / 2, float(ys.min()) - ry / 2,
        float(xs.max()) + rx / 2, float(ys.max()) + ry / 2,
        len(xs), len(ys))
    src_crs = CRS.from_epsg(epsg)

    nodata = SENTINEL_NODATA
    comp_clean = np.where(np.isnan(composite), nodata, composite).astype(np.float32)
    vf_clean = np.where(np.isnan(valid_fraction), nodata, valid_fraction).astype(np.float32)

    n_spectral = len(SENTINEL_BANDS)
    warped = np.full((n_spectral, dst_height, dst_width), nodata, dtype=np.float32)
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
        tags = dict(YEAR=str(year), SEASON=season, N_SCENES_TOTAL=str(n_scenes))
        if year >= 2022:
            tags["PB0400_OFFSET_CORRECTED"] = "true"
        dst.update_tags(**tags)

    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [{year}/{season}] Saved ({mb:.1f} MB)")


def stage_download(ctx=None):
    create_anchor(ctx)  # ensure anchor grid exists
    region_label = f" [{ctx.name}]" if ctx else ""
    print(f"\n{'='*70}")
    print(f"STAGE 1: DOWNLOAD (Sentinel-2 composites){region_label}")
    print(f"{'='*70}")
    years = ctx.years if ctx else ALL_YEARS
    for year in years:
        for season in SEASONS:
            download_season(year, season, ctx)
    print(f"\n[{ts()}] Download stage complete.")


# =====================================================================
# STAGE 2: EXTRACT FEATURES (Rust)
# =====================================================================

def load_sentinel_raster(year, season, ctx=None):
    """Load Sentinel-2 raster, return (spectral, valid_fraction)."""
    import rasterio

    raw_dir = ctx.raw_dir if ctx else RAW_V2_DIR
    region_name = ctx.name if ctx else "nuremberg"
    path = os.path.join(raw_dir, f"sentinel2_{region_name}_{year}_{season}.tif")
    with rasterio.open(path) as ds:
        data = ds.read()
        nodata = ds.nodata

    n_bands = len(SENTINEL_BANDS)
    spectral = data[:n_bands].astype(np.float32)
    if nodata is not None:
        spectral = np.where(spectral == nodata, np.nan, spectral)

    # Valid fraction is the last band (if present)
    vf = None
    if data.shape[0] > n_bands:
        vf = data[n_bands].astype(np.float32)
        if nodata is not None:
            vf = np.where(vf == nodata, np.nan, vf)

    return spectral, vf


def detect_scale(spectral):
    """Auto-detect if reflectance is 0..10000 or 0..1."""
    nir = spectral[6]  # B08
    finite = nir[np.isfinite(nir)]
    if len(finite) == 0:
        return 1.0
    return 10000.0 if np.percentile(finite, 95) > 2.0 else 1.0


def extract_year_pair(prev_year, curr_year, ctx=None):
    """Extract features for a year-pair using Rust.

    Loads 6 season rasters (3 per year), runs terrapulse_features in one
    shot, saves parquet with columns suffixed {model_year}_{season}.

    Returns path to parquet, or None if rasters missing.
    """
    import terrapulse_features as tf

    features_dir = ctx.features_dir if ctx else FEATURES_DIR
    raw_dir = ctx.raw_dir if ctx else RAW_V2_DIR
    region_name = ctx.name if ctx else "nuremberg"

    os.makedirs(features_dir, exist_ok=True)
    tag = f"{prev_year}_{curr_year}"
    out_path = os.path.join(features_dir, f"features_rust_{tag}.parquet")
    if os.path.exists(out_path):
        print(f"  [{tag}] Already extracted -- skip")
        return out_path

    # Map actual years to model-expected year tags (always 2020/2021)
    year_map = {prev_year: 2020, curr_year: 2021}

    # Check all TIFs exist
    jobs = []
    for actual_year in [prev_year, curr_year]:
        for season in SEASONS:
            tif = os.path.join(raw_dir,
                               f"sentinel2_{region_name}_{actual_year}_{season}.tif")
            if not os.path.exists(tif):
                print(f"  [{tag}] WARNING: Missing {tif} -- skip")
                return None
            jobs.append((actual_year, season))

    # Load rasters
    spectral_list = []
    suffixes = []
    nr, nc = None, None
    vf_first = None

    t0 = time.time()
    for actual_year, season in jobs:
        model_year = year_map[actual_year]
        spectral, vf = load_sentinel_raster(actual_year, season, ctx)
        scale = detect_scale(spectral)
        if nr is None:
            _, H, W = spectral.shape
            nr, nc = H // GRID_PX, W // GRID_PX
            vf_first = vf
        ref = spectral.astype(np.float32)
        if scale != 1.0:
            ref = ref / scale
        spectral_list.append(np.ascontiguousarray(ref))
        suffixes.append(f"{model_year}_{season}")
        print(f"    Loaded {actual_year}_{season} -> {model_year}_{season}")

    n_cells = nr * nc

    # Rust extraction (batch) — v2 expects pre-normalized data, no scale param
    t1 = time.time()
    n_feat = tf.n_features_per_cell()
    flat = tf.extract_all_seasons(spectral_list, nr, nc)
    dt_rust = time.time() - t1
    print(f"    Rust extraction: {dt_rust:.1f}s for {len(jobs)} seasons")
    del spectral_list

    # Build DataFrame
    result_2d = flat.reshape(n_cells, len(suffixes) * n_feat)
    columns = tf.feature_names_suffixed(suffixes)

    data = {"cell_id": np.arange(n_cells, dtype=np.int32)}
    for i, col in enumerate(columns):
        data[col] = result_2d[:, i]

    # Valid fraction from first raster
    if vf_first is not None:
        vf_cells = np.where(np.isfinite(vf_first), vf_first, 0.0).astype(np.float32)
        vf_cells = (vf_cells
                    .reshape(nr, GRID_PX, nc, GRID_PX)
                    .transpose(0, 2, 1, 3)
                    .reshape(n_cells, GRID_PX * GRID_PX)
                    .mean(axis=1))
        data["valid_fraction"] = vf_cells
        data["low_valid_fraction"] = (vf_cells < MIN_VALID_FRAC).astype(np.float32)

    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute NaN with column medians
    feat_cols = [c for c in df.columns if c not in CONTROL_COLS]
    nan_count = 0
    for c in feat_cols:
        n = df[c].isna().sum()
        if n > 0:
            nan_count += n
            med = df[c].median()
            df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)
    if nan_count > 0:
        print(f"    Imputed {nan_count} NaN values")

    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  [{tag}] Done: {df.shape[1]} cols, {mb:.1f} MB, {elapsed:.0f}s")
    return out_path


def stage_extract(ctx=None):
    region_label = f" [{ctx.name}]" if ctx else ""
    print(f"\n{'='*70}")
    print(f"STAGE 2: EXTRACT FEATURES (Rust){region_label}")
    print(f"{'='*70}")
    if ctx:
        # For inference: only need consecutive year pairs from ctx.years
        years = sorted(ctx.years)
        year_pairs = [(years[i], years[i + 1]) for i in range(len(years) - 1)]
    else:
        year_pairs = [(y, y + 1) for y in range(2020, 2025)]
    for prev_year, curr_year in year_pairs:
        extract_year_pair(prev_year, curr_year, ctx)
    print(f"\n[{ts()}] Extract stage complete.")


# =====================================================================
# STAGE 3: TRAIN
# =====================================================================

def build_bi_lbp(feature_cols):
    """Select bands + indices + TC + all LBP columns."""
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in BAND_PREFIXES or prefix in INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
    return sorted(set(selected))


def build_tree_features(feature_cols):
    """Select VegIdx + RedEdge + TC + NDTI + IRECI + CRI1."""
    selected = []
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    novel = ["NDTI", "IRECI", "CRI1"]
    for c in feature_cols:
        # VegIdx
        if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"]):
            if not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr"):
                selected.append(c)
                continue
        # RedEdge bands
        if band_pat.match(c):
            selected.append(c)
            continue
        # TC
        if c.startswith("TC_"):
            selected.append(c)
            continue
        # Novel indices
        for idx in novel:
            if c.startswith(f"{idx}_"):
                selected.append(c)
                break
    return selected


def swap_lbp_cols_for_mlp(feature_cols):
    """Swap NIR LBP -> NDTI LBP (best band per sweep)."""
    swapped = []
    for c in feature_cols:
        if c.startswith("LBP_u8_") or c.startswith("LBP_entropy_"):
            swapped.append("LBP_NDTI_" + c[len("LBP_"):])
        else:
            swapped.append(c)
    return swapped


# -- MLP model (inlined from train_mlp.py) --

def _build_mlp():
    """Late import to avoid loading torch at module level."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PlainBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout=0.15):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.norm = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.dropout(self.norm(F.silu(self.linear(x))))

    class PlainMLP(nn.Module):
        def __init__(self, in_features, n_classes=N_CLASSES, hidden=1024,
                     n_layers=5, dropout=0.15):
            super().__init__()
            layers = [PlainBlock(in_features, hidden, dropout)]
            for _ in range(n_layers - 1):
                layers.append(PlainBlock(hidden, hidden, dropout))
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x):
            return F.log_softmax(self.head(self.backbone(x)), dim=-1)

        def predict(self, x):
            self.eval()
            with torch.no_grad():
                return self.forward(x).exp()

    return PlainMLP


def normalize_targets(y):
    y = np.clip(y, 0, None).astype(np.float32)
    s = y.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    y = y / s + 1e-7
    y = y / y.sum(axis=1, keepdims=True)
    return y


def train_mlp_fold(X_trn, y_trn, X_val, y_val, n_features, device, fold_id):
    """Train one MLP fold with AMP + fused AdamW (speed-optimized)."""
    import torch

    PlainMLP = _build_mlp()
    use_amp = device == "cuda"

    torch.manual_seed(SEED + fold_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold_id)

    net = PlainMLP(n_features).to(device)

    # Optional torch.compile (PyTorch 2.x) for faster training
    try:
        net = torch.compile(net, mode="reduce-overhead")
        print(f"    torch.compile enabled (reduce-overhead)")
    except Exception:
        pass

    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=1e-3, weight_decay=1e-4, fused=use_amp)
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=1e-3, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n = X_trn.size(0)
    batch_size = 2048
    steps_per_ep = math.ceil(n / batch_size)
    max_epochs = 2000
    total_steps = max_epochs * steps_per_ep

    # Cosine warmup (3 epochs)
    warmup_steps = steps_per_ep * 3

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.001, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    patience_epochs = max(math.ceil(5000 / steps_per_ep), 5)
    min_epochs = max(math.ceil(2000 / steps_per_ep), 3)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if idx.size(0) < 2:
                continue
            xb, yb = X_trn[idx], y_trn[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logp = net(xb)
                loss = -(yb * logp).sum(dim=-1).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        net.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            val_loss = -(y_val * net(X_val)).sum(dim=-1).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if (epoch + 1) >= min_epochs and wait >= patience_epochs:
                break

    if best_state:
        net.load_state_dict(best_state)
    return net, epoch + 1, best_val


def predict_mlp_batched(net, X_cpu, device, batch_size=65536):
    import torch
    net.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, X_cpu.size(0), batch_size):
            xb = X_cpu[i:i + batch_size].to(device, non_blocking=True)
            parts.append(net.predict(xb).cpu())
    return torch.cat(parts, dim=0).numpy()


def stage_train():
    # Ensure labels and splits exist (auto-generate if missing)
    create_anchor()
    prepare_labels()
    prepare_spatial_split()

    import torch
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor

    print(f"\n{'='*70}")
    print(f"STAGE 3: TRAIN MODELS")
    print(f"{'='*70}")

    tree_done = os.path.exists(os.path.join(MODELS_DIR, "tree_meta.json"))
    mlp_done = os.path.exists(os.path.join(MODELS_DIR, "mlp_meta.json"))
    if tree_done and mlp_done:
        print("  Both models already trained -- skip")
        return

    # cell_ids needed for OOF parquets
    cell_ids = None  # populated below after feature load

    # Load features for labeled pair (2020+2021)
    print(f"  [{ts()}] Loading features for 2020+2021...")
    feat_path = os.path.join(FEATURES_DIR, "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        # Fallback to existing processed data
        feat_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    merged = pd.read_parquet(feat_path)
    cell_ids = merged["cell_id"].values if "cell_id" in merged.columns else np.arange(len(merged))
    print(f"  Shape: {merged.shape}")

    from pandas.api.types import is_numeric_dtype
    full_cols = [c for c in merged.columns
                 if c not in CONTROL_COLS and is_numeric_dtype(merged[c])]

    # Build feature sets
    mlp_idx = build_bi_lbp(full_cols)
    mlp_cols = [full_cols[i] for i in mlp_idx]
    tree_col_names = build_tree_features(full_cols)
    n_mlp = len(mlp_cols)
    n_tree = len(tree_col_names)
    print(f"  MLP features: {n_mlp}, Tree features: {n_tree}")

    X_mlp = np.nan_to_num(merged[mlp_cols].values.astype(np.float32), 0.0)
    X_tree = np.nan_to_num(merged[tree_col_names].values.astype(np.float32), 0.0)

    # Labels
    labels_path = os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet")
    y = pd.read_parquet(labels_path)[CLASS_NAMES].values.astype(np.float32)

    # Spatial splits
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        split_meta = json.load(f)
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    del split_df

    # -- LightGBM --
    tree_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, reg_lambda=0.1,
        subsample=0.85, colsample_bytree=0.85, verbosity=-1,
        n_jobs=-1,
    )
    if not tree_done:
        print(f"\n  [{ts()}] Training LightGBM (5 folds)...")
        tree_fold_metrics = []
        tree_oof = np.zeros_like(y)
        tree_oof_mask = np.zeros(len(y), dtype=bool)
        for fold_id in range(N_FOLDS):
            train_idx, test_idx = get_fold_indices(
                tiles, folds_arr, fold_id,
                split_meta["tile_cols"], split_meta["tile_rows"], buffer_tiles=1)

            t0 = time.time()
            run_params = {**tree_params, "random_state": SEED + fold_id}
            model = MultiOutputRegressor(lgb.LGBMRegressor(**run_params))
            model.fit(X_tree[train_idx], y[train_idx])
            y_pred = np.clip(model.predict(X_tree[test_idx]), 0, 100)
            elapsed = time.time() - t0

            tree_oof[test_idx] = y_pred
            tree_oof_mask[test_idx] = True

            r2, mae, pc = evaluate_predictions(y[test_idx], y_pred, per_class=True)
            tree_fold_metrics.append({
                "fold": fold_id, "r2": r2, "mae": mae,
                "per_class": pc, "time_s": round(elapsed, 1),
            })
            print(f"    Fold {fold_id}: R2={r2:.4f}  MAE={mae:.2f}pp  ({elapsed:.0f}s)")

            with open(os.path.join(MODELS_DIR, f"tree_fold_{fold_id}.pkl"), "wb") as f:
                pickle.dump(model, f)

        # Save OOF predictions
        tree_oof_df = pd.DataFrame({"cell_id": cell_ids})
        for ci, cn in enumerate(CLASS_NAMES):
            tree_oof_df[f"{cn}_pred"] = tree_oof[:, ci]
        tree_oof_df.to_parquet(
            os.path.join(MODELS_DIR, "tree_oof_predictions.parquet"), index=False)

        # Save rich metadata
        r2_vals = [m["r2"] for m in tree_fold_metrics]
        tree_r2 = np.mean(r2_vals)
        with open(os.path.join(MODELS_DIR, "tree_meta.json"), "w") as f:
            json.dump({
                "model": "LightGBM",
                "config": "VegIdx+RedEdge+TC+NDTI+IRECI+CRI1",
                "feature_cols": tree_col_names,
                "n_features": n_tree,
                "seed": SEED,
                "r2_mean": round(float(tree_r2), 4),
                "r2_std": round(float(np.std(r2_vals)), 4),
                "mae_mean_pp": round(float(np.mean([m["mae"] for m in tree_fold_metrics])), 2),
                "fold_metrics": tree_fold_metrics,
                "hyperparameters": tree_params,
            }, f, indent=2)
        print(f"  Tree mean R2: {tree_r2:.4f}")

    # -- MLP --
    mlp_arch_config = {
        "arch": "plain", "activation": "silu",
        "n_layers": 5, "d_model": 1024, "norm": "batchnorm",
        "dropout": 0.15,
    }
    mlp_train_config = {
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 2048,
        "max_epochs": 2000, "patience_steps": 5000, "min_steps": 2000,
        "scheduler": "cosine_warmup_3ep",
    }
    if not mlp_done:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        print(f"\n  [{ts()}] Training MLP on {device} (5 folds)...")
        mlp_fold_metrics = []
        mlp_oof = np.full((len(y), N_CLASSES), np.nan, dtype=np.float32)

        for fold_id in range(N_FOLDS):
            print(f"\n  --- MLP Fold {fold_id} ---")
            train_idx, test_idx = get_fold_indices(
                tiles, folds_arr, fold_id,
                split_meta["tile_cols"], split_meta["tile_rows"], buffer_tiles=1)

            rng = np.random.RandomState(SEED + fold_id)
            perm = rng.permutation(len(train_idx))
            n_val = max(int(len(train_idx) * 0.15), 100)
            val_idx = train_idx[perm[:n_val]]
            trn_idx = train_idx[perm[n_val:]]

            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X_mlp[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X_mlp[val_idx]).astype(np.float32)

            X_trn_t = torch.tensor(X_trn_s).to(device, non_blocking=True)
            X_val_t = torch.tensor(X_val_s).to(device, non_blocking=True)
            y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device, non_blocking=True)
            y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device, non_blocking=True)

            t0 = time.time()
            trained_net, n_epochs, best_val = train_mlp_fold(
                X_trn_t, y_trn_t, X_val_t, y_val_t, n_mlp, device, fold_id)
            elapsed = time.time() - t0

            # Save
            save_net = trained_net._orig_mod if hasattr(trained_net, "_orig_mod") else trained_net
            torch.save(save_net.state_dict(),
                       os.path.join(MODELS_DIR, f"mlp_fold_{fold_id}.pt"))
            with open(os.path.join(MODELS_DIR, f"mlp_scaler_{fold_id}.pkl"), "wb") as f:
                pickle.dump(scaler, f)

            # OOF predictions
            X_tst_s = scaler.transform(X_mlp[test_idx]).astype(np.float32)
            preds = predict_mlp_batched(
                trained_net, torch.tensor(X_tst_s), device)
            mlp_oof[test_idx] = preds

            r2, mae, pc = evaluate_predictions(y[test_idx], preds, per_class=True)
            mlp_fold_metrics.append({
                "fold": fold_id, "r2": r2, "mae": mae,
                "per_class": pc,
                "epochs": n_epochs, "val_loss": best_val,
                "time_s": round(elapsed, 1),
            })
            print(f"    R2={r2:.4f}  MAE={mae:.2f}pp  epochs={n_epochs}  ({elapsed:.0f}s)")

            del trained_net, X_trn_t, X_val_t, y_trn_t, y_val_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save OOF predictions
        mlp_oof_df = pd.DataFrame(mlp_oof, columns=[c + "_pred" for c in CLASS_NAMES])
        mlp_oof_df.insert(0, "cell_id", cell_ids)
        mlp_oof_df.to_parquet(
            os.path.join(MODELS_DIR, "mlp_oof_predictions.parquet"), index=False)

        # Save rich metadata
        r2_vals = [m["r2"] for m in mlp_fold_metrics]
        mlp_r2 = np.mean(r2_vals)
        with open(os.path.join(MODELS_DIR, "mlp_meta.json"), "w") as f:
            json.dump({
                "model": "PlainMLP",
                "config": "bi_LBP_plain_silu_L5_d1024_bn",
                "feature_set": "bi_LBP",
                "feature_cols": mlp_cols,
                "n_features": n_mlp,
                "seed": SEED,
                "r2_mean": round(float(mlp_r2), 4),
                "r2_std": round(float(np.std(r2_vals)), 4),
                "mae_mean_pp": round(float(np.mean([m["mae"] for m in mlp_fold_metrics])), 2),
                "fold_metrics": mlp_fold_metrics,
                "architecture": mlp_arch_config,
                "training": mlp_train_config,
                "speed_optimizations": [
                    "AMP_fp16", "fused_AdamW", "TF32_matmul",
                    "torch_compile", "set_to_none_zero_grad",
                ],
            }, f, indent=2)
        print(f"  MLP mean R2: {mlp_r2:.4f}")

    print(f"\n[{ts()}] Train stage complete.")


# =====================================================================
# STAGE 4: PREDICT
# =====================================================================

def stage_predict(ctx=None):
    region_label = f" [{ctx.name}]" if ctx else ""
    features_dir = ctx.features_dir if ctx else FEATURES_DIR
    predictions_dir = ctx.predictions_dir if ctx else PREDICTIONS_DIR

    os.makedirs(predictions_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"STAGE 4: PREDICT MAPS{region_label}")
    print(f"{'='*70}")

    with open(os.path.join(MODELS_DIR, "tree_meta.json")) as f:
        tree_cols = json.load(f)["feature_cols"]
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

    PlainMLP = _build_mlp()

    if ctx:
        years = sorted(ctx.years)
        year_pairs = [(years[i], years[i + 1]) for i in range(len(years) - 1)]
    else:
        year_pairs = [(y, y + 1) for y in range(2020, 2025)]

    for prev_year, curr_year in year_pairs:
        tag = f"{prev_year}_{curr_year}"
        tree_path = os.path.join(predictions_dir, f"predictions_tree_{tag}.parquet")
        mlp_path = os.path.join(predictions_dir, f"predictions_mlp_{tag}.parquet")

        if os.path.exists(tree_path) and os.path.exists(mlp_path):
            print(f"  [{tag}] Already predicted -- skip")
            continue

        print(f"\n  [{ts()}] Loading features for {tag}...")
        feat_path = os.path.join(features_dir, f"features_rust_{tag}.parquet")
        if not os.path.exists(feat_path):
            if not ctx and prev_year == 2020 and curr_year == 2021:
                feat_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
            if not os.path.exists(feat_path):
                print(f"  [{tag}] WARNING: No features found -- skip")
                continue
        merged = pd.read_parquet(feat_path)
        cell_ids = merged["cell_id"].values
        merged_cols_set = set(merged.columns)

        # Tree predictions (ensemble of 5 folds)
        if not os.path.exists(tree_path):
            print(f"  [{ts()}] Tree predictions for {tag}...")
            avail_tree = [c for c in tree_cols if c in merged_cols_set]
            X_tree = np.nan_to_num(merged[avail_tree].values.astype(np.float32), 0.0)
            preds_all = np.zeros((len(cell_ids), N_CLASSES), dtype=np.float32)
            for fold_id in range(N_FOLDS):
                with open(os.path.join(MODELS_DIR, f"tree_fold_{fold_id}.pkl"), "rb") as f:
                    model = pickle.load(f)
                preds_all += np.clip(model.predict(X_tree), 0, 100).astype(np.float32)
            preds_all /= N_FOLDS

            tree_df = pd.DataFrame({"cell_id": cell_ids})
            for ci, cn in enumerate(CLASS_NAMES):
                tree_df[f"{cn}_pred"] = preds_all[:, ci]
            tree_df["prev_year"] = prev_year
            tree_df["curr_year"] = curr_year
            tree_df.to_parquet(tree_path, index=False)
            print(f"    Saved: {tree_path}")

        # MLP predictions (ensemble of 5 folds)
        if not os.path.exists(mlp_path):
            import torch
            print(f"  [{ts()}] MLP predictions for {tag}...")
            avail_mlp = [c for c in mlp_cols if c in merged_cols_set]
            X_mlp = np.nan_to_num(merged[avail_mlp].values.astype(np.float32), 0.0)
            preds_all = np.zeros((len(cell_ids), N_CLASSES), dtype=np.float32)

            for fold_id in range(N_FOLDS):
                with open(os.path.join(MODELS_DIR, f"mlp_scaler_{fold_id}.pkl"), "rb") as f:
                    scaler = pickle.load(f)
                X_scaled = scaler.transform(X_mlp).astype(np.float32)

                net = PlainMLP(n_mlp).to(device)
                net.load_state_dict(torch.load(
                    os.path.join(MODELS_DIR, f"mlp_fold_{fold_id}.pt"),
                    map_location=device, weights_only=True))

                preds_all += predict_mlp_batched(net, torch.tensor(X_scaled), device)
                del net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            preds_all /= N_FOLDS

            mlp_df = pd.DataFrame({"cell_id": cell_ids})
            for ci, cn in enumerate(CLASS_NAMES):
                mlp_df[f"{cn}_pred"] = preds_all[:, ci]
            mlp_df["prev_year"] = prev_year
            mlp_df["curr_year"] = curr_year
            mlp_df.to_parquet(mlp_path, index=False)
            print(f"    Saved: {mlp_path}")

        del merged

    print(f"\n[{ts()}] Predict stage complete.")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production pipeline: download -> extract -> train -> predict")

    # Top-level skip flags (backward compatible with old usage)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # Default: full Nuremberg pipeline (no extra args needed)
    sub.add_parser("full", help="Full pipeline (default for Nuremberg)")

    # Inference for arbitrary region
    inf = sub.add_parser("infer",
        help="Inference-only for an arbitrary bounding box")
    inf.add_argument("--bbox", nargs=4, type=float, required=True,
                     metavar=("W", "S", "E", "N"),
                     help="Bounding box: west south east north (WGS84)")
    inf.add_argument("--name", required=True,
                     help="Region name (used for output directory)")
    inf.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                     help="Years to process (default: 2023 2024 2025)")

    args = parser.parse_args()

    # Default to 'full' if no subcommand given
    if args.command is None:
        args.command = "full"

    t_total = time.time()

    if args.command == "infer":
        # --- Inference mode for arbitrary region ---
        ctx = build_region_ctx(args.name, args.bbox, args.years)
        print(f"[{ts()}] Inference Pipeline starting")
        print(f"  Region: {ctx.name}")
        print(f"  Bbox: {ctx.bbox}")
        print(f"  EPSG: {ctx.epsg} (auto-detected)")
        print(f"  WorldCover tile: {ctx.wc_tile}")
        print(f"  Years: {ctx.years}")
        print(f"  Output: {os.path.join(REGIONS_DIR, ctx.name)}")

        if not args.skip_download:
            stage_download(ctx)
        else:
            print("\n  DOWNLOAD skipped (--skip-download)")

        if not args.skip_extract:
            stage_extract(ctx)
        else:
            print("\n  EXTRACT skipped (--skip-extract)")

        # Always predict (that's the point of inference mode)
        stage_predict(ctx)

        total = time.time() - t_total
        hours = int(total // 3600)
        mins = int((total % 3600) // 60)
        print(f"\n{'='*70}")
        print(f"INFERENCE COMPLETE [{ctx.name}] in {hours}h {mins}m")
        print(f"  Predictions: {ctx.predictions_dir}")
        print(f"{'='*70}")

    else:
        # --- Full Nuremberg pipeline ---
        print(f"[{ts()}] Production Pipeline starting")
        print(f"  Years: {ALL_YEARS}")
        print(f"  Seasons: {SEASONS}")
        print(f"  Labels available: {LABELED_YEARS}")
        print(f"  Predict: {PREDICT_YEARS}")

        ensure_dirs()

        if not args.skip_download:
            stage_download()
        else:
            print("\n  DOWNLOAD skipped (--skip-download)")

        if not args.skip_extract:
            stage_extract()
        else:
            print("\n  EXTRACT skipped (--skip-extract)")

        if not args.skip_train:
            stage_train()
        else:
            print("\n  TRAIN skipped (--skip-train)")

        if not args.skip_predict:
            stage_predict()
        else:
            print("\n  PREDICT skipped (--skip-predict)")

        total = time.time() - t_total
        hours = int(total // 3600)
        mins = int((total % 3600) // 60)
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE in {hours}h {mins}m")
        print(f"  Output: {OUT_DIR}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
