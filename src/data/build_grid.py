"""
Phase 3: Build Grid & Labels from Canonical Anchor

Derives everything from the anchor reference created in Phase 1:
  1. Create 100m grid from anchor pixel blocks (row-major cell_id)
  2. Reproject WorldCover Map to anchor grid (nearest)
  3. Compute class proportions + coverage per cell
  4. Compute change labels (delta = 2021 - 2020)

Usage:
    python src/data/build_grid.py

Outputs:
    data/processed/v2/grid.gpkg
    data/processed/v2/labels_2020.parquet
    data/processed/v2/labels_2021.parquet
    data/processed/v2/labels_change.parquet
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

# -- Load config ---------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CFG, GRID_REF_PATH, PROJECT_ROOT  # noqa: E402

LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")
V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")

# Config-driven constants
GRID_SIZE_M = int(CFG["grid"]["size_m"])
PIXEL_SIZE = float(CFG["grid"]["pixel_size"])
BLOCK = int(GRID_SIZE_M / PIXEL_SIZE)  # 10 pixels per 100m cell

CLASS_MAP = {int(k): int(v) for k, v in CFG["worldcover"]["class_map"].items()}
CLASS_NAMES = CFG["worldcover"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
WC_YEARS = CFG["worldcover"]["years"]
WC_TILE = CFG["worldcover"]["tile"]


def read_anchor():
    """Read canonical anchor geometry."""
    assert os.path.exists(GRID_REF_PATH), (
        f"Missing anchor: {GRID_REF_PATH}. Run scripts/create_anchor.py first."
    )
    with rasterio.open(GRID_REF_PATH) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }


def create_grid_from_anchor(anchor):
    """Create grid cells from anchor pixel blocks.

    Each cell is a BLOCK x BLOCK pixel window (100m x 100m).
    cell_id is row-major: row 0 left-to-right, then row 1, etc.
    """
    transform = anchor["transform"]
    n_cols_cells = anchor["width"] // BLOCK
    n_rows_cells = anchor["height"] // BLOCK

    cells = []
    cell_ids = []
    cell_id = 0

    for row_idx in range(n_rows_cells):
        for col_idx in range(n_cols_cells):
            # Top-left pixel of this cell
            px_col = col_idx * BLOCK
            px_row = row_idx * BLOCK
            # Transform pixel coords to map coords
            x_ul, y_ul = transform * (px_col, px_row)
            x_lr, y_lr = transform * (px_col + BLOCK, px_row + BLOCK)
            cells.append(box(x_ul, y_lr, x_lr, y_ul))
            cell_ids.append(cell_id)
            cell_id += 1

    import geopandas as gpd
    grid = gpd.GeoDataFrame(
        {"cell_id": cell_ids},
        geometry=cells,
        crs=anchor["crs"],
    )
    print(f"  Grid: {n_cols_cells} cols x {n_rows_cells} rows = {len(grid)} cells")
    print(f"  Cell size: {GRID_SIZE_M}m ({BLOCK}x{BLOCK} pixels)")
    return grid, n_cols_cells, n_rows_cells


def reproject_worldcover_to_anchor(wc_path, anchor):
    """Reproject WorldCover from EPSG:4326 to anchor grid (nearest)."""
    dst_array = np.zeros((anchor["height"], anchor["width"]), dtype=np.uint8)

    with rasterio.open(wc_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=anchor["transform"],
            dst_crs=anchor["crs"],
            resampling=Resampling.nearest,
        )

    print(f"  Reprojected to anchor grid: {anchor['width']}x{anchor['height']} px")
    return dst_array


def aggregate_labels(wc_array, n_cols_cells, n_rows_cells):
    """Compute class proportions + coverage per cell using pixel windows.

    Returns DataFrame with columns:
      cell_id, mapped_pixels, unmapped_pixels, coverage, <class_names...>
    """
    total_px = BLOCK * BLOCK
    records = []
    cell_id = 0

    for row_idx in range(n_rows_cells):
        for col_idx in range(n_cols_cells):
            r0 = row_idx * BLOCK
            c0 = col_idx * BLOCK
            patch = wc_array[r0 : r0 + BLOCK, c0 : c0 + BLOCK]

            # Count mapped pixels (those that appear in CLASS_MAP)
            proportions = np.zeros(N_CLASSES, dtype=np.float32)
            mapped = 0
            for wc_code, our_class in CLASS_MAP.items():
                count = int(np.sum(patch == wc_code))
                proportions[our_class] += count
                mapped += count

            unmapped = total_px - mapped
            coverage = mapped / total_px if total_px > 0 else 0.0

            # Proportions over total pixels (not just mapped)
            if total_px > 0:
                proportions /= total_px

            record = {
                "cell_id": cell_id,
                "mapped_pixels": mapped,
                "unmapped_pixels": unmapped,
                "coverage": float(coverage),
            }
            for i, name in enumerate(CLASS_NAMES):
                record[name] = float(proportions[i])
            records.append(record)
            cell_id += 1

    return pd.DataFrame(records)


def main():
    os.makedirs(V2_DIR, exist_ok=True)

    # -- Step 1: Read anchor geometry --
    print("Step 1: Reading anchor geometry...")
    anchor = read_anchor()
    print(f"  Anchor: {anchor['width']}x{anchor['height']} px, CRS={anchor['crs']}")
    print(f"  Transform: {anchor['transform']}")

    # -- Step 2: Create grid from anchor blocks --
    print("\nStep 2: Creating grid from anchor pixel blocks...")
    grid, n_cols_cells, n_rows_cells = create_grid_from_anchor(anchor)

    grid_path = os.path.join(V2_DIR, "grid.gpkg")
    grid.to_file(grid_path, driver="GPKG")
    print(f"  Saved -> {grid_path}")

    # -- Step 3: Labels for each year --
    for year in WC_YEARS:
        # Find the Map file (try both naming conventions)
        wc_filename_map = f"ESA_WorldCover_{year}_{WC_TILE}_Map.tif"
        wc_filename_plain = f"ESA_WorldCover_{year}_{WC_TILE}.tif"
        wc_path = os.path.join(LABELS_DIR, wc_filename_map)
        if not os.path.exists(wc_path):
            wc_path = os.path.join(LABELS_DIR, wc_filename_plain)
        if not os.path.exists(wc_path):
            print(f"\n  [{year}] WARNING: WorldCover Map not found, skipping")
            continue

        print(f"\nStep 3: Processing WorldCover {year}...")
        print(f"  Source: {os.path.basename(wc_path)}")

        # Reproject to anchor grid
        wc_array = reproject_worldcover_to_anchor(wc_path, anchor)

        # Aggregate labels per cell
        print(f"  Aggregating labels per grid cell...")
        labels_df = aggregate_labels(wc_array, n_cols_cells, n_rows_cells)

        # Summary
        print(f"\n  Label summary ({year}):")
        for name in CLASS_NAMES:
            col = labels_df[name]
            print(
                f"    {name:<15} mean={col.mean():.3f}  std={col.std():.3f}  "
                f"min={col.min():.3f}  max={col.max():.3f}"
            )
        n_valid = (labels_df["coverage"] > 0).sum()
        print(f"    Cells with data: {n_valid} / {len(labels_df)}")
        print(f"    Mean coverage:   {labels_df['coverage'].mean():.3f}")

        labels_path = os.path.join(V2_DIR, f"labels_{year}.parquet")
        labels_df.to_parquet(labels_path, index=False)
        print(f"  Saved -> {labels_path}")

    # -- Step 4: Change labels --
    print("\nStep 4: Computing change labels (delta = 2021 - 2020)...")
    l2020_path = os.path.join(V2_DIR, "labels_2020.parquet")
    l2021_path = os.path.join(V2_DIR, "labels_2021.parquet")

    if os.path.exists(l2020_path) and os.path.exists(l2021_path):
        labels_2020 = pd.read_parquet(l2020_path)
        labels_2021 = pd.read_parquet(l2021_path)

        assert len(labels_2020) == len(labels_2021), "Label row count mismatch!"
        assert (labels_2020["cell_id"] == labels_2021["cell_id"]).all(), "cell_id mismatch!"

        change_df = pd.DataFrame({"cell_id": labels_2020["cell_id"]})
        for name in CLASS_NAMES:
            change_df[f"delta_{name}"] = labels_2021[name] - labels_2020[name]

        print("  Change summary:")
        for name in CLASS_NAMES:
            col = change_df[f"delta_{name}"]
            print(
                f"    delta_{name:<15} mean={col.mean():+.4f}  std={col.std():.4f}  "
                f"[{col.min():+.3f}, {col.max():+.3f}]"
            )

        change_path = os.path.join(V2_DIR, "labels_change.parquet")
        change_df.to_parquet(change_path, index=False)
        print(f"  Saved -> {change_path}")
    else:
        print("  WARNING: Missing labels for one or both years, skipping change computation")

    # -- Final summary --
    print(f"\n{'='*60}")
    print("DONE! Files in data/processed/v2/:")
    for f in sorted(os.listdir(V2_DIR)):
        fp = os.path.join(V2_DIR, f)
        if os.path.isfile(fp):
            size_kb = os.path.getsize(fp) / 1024
            if size_kb > 1024:
                print(f"  {f}: {size_kb/1024:.1f} MB")
            else:
                print(f"  {f}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
