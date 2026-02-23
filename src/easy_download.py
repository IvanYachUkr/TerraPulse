#!/usr/bin/env python3
"""
TerraPulse Easy Download — download satellite imagery with one command.

Usage (inside the Docker container):
    python /app/easy_download.py --bbox 10.95 49.38 11.20 49.52 --years 2021 --output /data/nuremberg

This script:
  1. Auto-detects the UTM EPSG code from the bbox
  2. Creates an anchor reference GeoTIFF (no manual anchor needed)
  3. Runs `terrapulse download` for the specified years
  4. Copies the output GeoTIFFs to your chosen output directory
  5. Cleans up temporary files

From Docker:
    docker run --rm -v C:\\Users\\me\\satellite:/output \\
      ghcr.io/ivanyachukr/terrapulse:latest \\
      python /app/easy_download.py --bbox 10.95 49.38 11.20 49.52 --output /output
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from math import ceil, floor

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTINEL_RES = 10        # metres per pixel
SENTINEL_NODATA = -9999
GRID_PX = 10             # pixels per grid cell (100 m)
TERRAPULSE_BIN = os.environ.get("TERRAPULSE_BIN", "terrapulse")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def auto_epsg(bbox):
    """Determine UTM EPSG from bbox center."""
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) / 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def create_anchor(bbox, epsg, out_path):
    """Create a minimal anchor GeoTIFF that defines the output grid."""
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(epsg)
    left, bottom, right, top = transform_bounds(
        src_crs, dst_crs, *bbox, densify_pts=21
    )

    ps = float(SENTINEL_RES)
    left_s  = floor(left / ps) * ps
    top_s   = ceil(top / ps) * ps
    right_s = ceil(right / ps) * ps
    bottom_s = floor(bottom / ps) * ps

    w0 = round((right_s - left_s) / ps)
    h0 = round((top_s - bottom_s) / ps)
    width  = ceil(w0 / GRID_PX) * GRID_PX
    height = ceil(h0 / GRID_PX) * GRID_PX

    transform = Affine(ps, 0.0, left_s, 0.0, -ps, top_s)
    data = np.full((1, height, width), SENTINEL_NODATA, dtype=np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="float32", crs=dst_crs, transform=transform,
        nodata=SENTINEL_NODATA, compress="lzw",
    ) as dst:
        dst.write(data)

    nc, nr = width // GRID_PX, height // GRID_PX
    print(f"  Created anchor: {width}x{height} px, {nc}x{nr} = {nc*nr} grid cells")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download satellite imagery for any region on Earth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Nuremberg, 2021
  python easy_download.py --bbox 10.95 49.38 11.20 49.52 --years 2021 --output ./nuremberg

  # Paris, 2023-2024
  python easy_download.py --bbox 2.25 48.81 2.42 48.90 --years 2023 2024 --output ./paris

  # Custom region name
  python easy_download.py --bbox 10.95 49.38 11.20 49.52 --years 2021 --output ./data --region nuremberg
        """,
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84 degrees [west south east north]",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2021],
        help="Years to download (default: 2021)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory — satellite GeoTIFFs will be saved here",
    )
    parser.add_argument(
        "--region", type=str, default="region",
        help="Region name used in filenames (default: region)",
    )
    args = parser.parse_args()

    bbox = args.bbox
    years = args.years
    output_dir = os.path.abspath(args.output)
    region = args.region

    # Auto-detect EPSG
    epsg = auto_epsg(bbox)
    print(f"TerraPulse Easy Download")
    print(f"  Bbox:   [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    print(f"  EPSG:   {epsg} (auto-detected)")
    print(f"  Years:  {years}")
    print(f"  Region: {region}")
    print(f"  Output: {output_dir}")
    print()

    # Work in a temp directory
    with tempfile.TemporaryDirectory(prefix="terrapulse_") as tmp_dir:
        raw_dir = os.path.join(tmp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Step 1: Create anchor
        print("Step 1/3: Creating anchor reference...")
        anchor_path = os.path.join(tmp_dir, "anchor.tif")
        create_anchor(bbox, epsg, anchor_path)

        # Step 2: Run download
        print("\nStep 2/3: Downloading satellite imagery...")
        years_str = " ".join(str(y) for y in years)
        cmd = [
            TERRAPULSE_BIN, "download",
            "--bbox", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
            "--epsg", str(epsg),
            "--years", years_str,
            "--region", region,
            "--raw-dir", raw_dir,
            "--anchor-ref", anchor_path,
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"\nERROR: Download failed (exit code {result.returncode})")
            sys.exit(1)

        # Step 3: Copy results to output
        print(f"\nStep 3/3: Copying results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        tif_files = [f for f in os.listdir(raw_dir) if f.endswith(".tif")]
        if not tif_files:
            print("  WARNING: No GeoTIFF files were produced!")
            sys.exit(1)

        for f in sorted(tif_files):
            src = os.path.join(raw_dir, f)
            dst = os.path.join(output_dir, f)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.1f} MB)")

        print(f"\nDone! {len(tif_files)} files saved to {output_dir}")
        # Temp dir automatically cleaned up


if __name__ == "__main__":
    main()
