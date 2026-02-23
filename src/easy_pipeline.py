#!/usr/bin/env python3
"""
TerraPulse Pipeline — download, extract, and predict with one command.

Runs the full pipeline: downloads satellite imagery, extracts 1764 spectral/SAR
features, and produces land cover predictions using the bundled ONNX model.

Usage (inside the Docker container):
    python /app/easy_pipeline.py --bbox 10.95 49.38 11.20 49.52 --output /data

Output:
    predictions_{year}.json  — per-cell class probabilities
    grid.json                — cell geometries (GeoJSON)
    features_*.parquet       — extracted feature vectors
    sentinel2_*.tif          — seasonal composites (if --keep-raw)
"""
import argparse
import json
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
SENTINEL_RES = 10
SENTINEL_NODATA = -9999
GRID_PX = 10
TERRAPULSE_BIN = os.environ.get("TERRAPULSE_BIN", "terrapulse")
ONNX_MODELS_DIR = os.environ.get("ONNX_MODELS_DIR", "/app/models/onnx")


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
        description="Run the full TerraPulse pipeline: download + extract + predict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Nuremberg, 2021 — full pipeline
  python easy_pipeline.py --bbox 10.95 49.38 11.20 49.52 --output ./nuremberg

  # Paris, 2023 — keep raw GeoTIFFs
  python easy_pipeline.py --bbox 2.25 48.81 2.42 48.90 --output ./paris --years 2023 --keep-raw

  # Download + extract only (no prediction)
  python easy_pipeline.py --bbox 10.95 49.38 11.20 49.52 --output ./data --no-predict
        """,
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float, required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84 degrees",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2021],
        help="Years to process (default: 2021)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--region", type=str, default="region",
        help="Region name used in filenames (default: region)",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep raw GeoTIFF files in output (default: only features + predictions)",
    )
    parser.add_argument(
        "--no-predict", action="store_true",
        help="Skip ONNX prediction — only download + extract features",
    )
    args = parser.parse_args()

    bbox = args.bbox
    years = args.years
    output_dir = os.path.abspath(args.output)
    region = args.region
    os.makedirs(output_dir, exist_ok=True)

    epsg = auto_epsg(bbox)
    print()
    print("=" * 55)
    print("  TerraPulse Pipeline")
    print("=" * 55)
    print(f"  Bbox:    [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    print(f"  EPSG:    {epsg} (auto-detected)")
    print(f"  Years:   {years}")
    print(f"  Region:  {region}")
    print(f"  Output:  {output_dir}")
    print(f"  Predict: {'no' if args.no_predict else 'yes'}")
    print()

    # Step 1: Create anchor
    print("Step 1/4: Creating anchor reference...")
    anchor_path = os.path.join(output_dir, "anchor.tif")
    create_anchor(bbox, epsg, anchor_path)

    # Step 2: Download
    print("\nStep 2/4: Downloading satellite imagery...")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    years_str = " ".join(str(y) for y in years)
    cmd_dl = [
        TERRAPULSE_BIN, "download",
        "--bbox", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
        "--epsg", str(epsg),
        "--years", years_str,
        "--region", region,
        "--raw-dir", raw_dir,
        "--anchor-ref", anchor_path,
    ]
    print(f"  Running: {' '.join(cmd_dl)}")
    result = subprocess.run(cmd_dl)
    if result.returncode != 0:
        print(f"\nERROR: Download failed (exit code {result.returncode})")
        sys.exit(1)

    # Step 3: Extract features
    print("\nStep 3/4: Extracting features...")
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    cmd_ext = [
        TERRAPULSE_BIN, "extract",
        "--raw-dir", raw_dir,
        "--features-dir", features_dir,
        "--anchor-ref", anchor_path,
        "--region", region,
    ]
    print(f"  Running: {' '.join(cmd_ext)}")
    result = subprocess.run(cmd_ext)
    if result.returncode != 0:
        print(f"\nERROR: Feature extraction failed (exit code {result.returncode})")
        sys.exit(1)

    # Step 4: Predict (optional)
    if not args.no_predict:
        print("\nStep 4/4: Running ONNX prediction...")
        if not os.path.isdir(ONNX_MODELS_DIR):
            print(f"  WARNING: ONNX models dir not found: {ONNX_MODELS_DIR}")
            print("  Skipping prediction. Use --no-predict to suppress this warning.")
        else:
            cmd_pred = [
                TERRAPULSE_BIN, "predict",
                "--features-dir", features_dir,
                "--models-dir", ONNX_MODELS_DIR,
                "--output-dir", output_dir,
                "--region", region,
            ]
            print(f"  Running: {' '.join(cmd_pred)}")
            result = subprocess.run(cmd_pred)
            if result.returncode != 0:
                print(f"\nWARNING: Prediction failed (exit code {result.returncode})")
                print("  Features were extracted successfully — you can run prediction later.")
    else:
        print("\nStep 4/4: Skipping prediction (--no-predict)")

    # Clean up raw files if not keeping
    if not args.keep_raw and os.path.isdir(raw_dir):
        shutil.rmtree(raw_dir)
        print("\n  Cleaned up raw GeoTIFFs (use --keep-raw to keep them)")

    # Summary
    print()
    print("=" * 55)
    print("  Pipeline complete!")
    print("=" * 55)
    outputs = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, output_dir)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > 0.01:
                outputs.append((rel, size_mb))
    for rel, size in sorted(outputs):
        print(f"  {rel:45s}  {size:.1f} MB")
    print(f"\n  Saved to: {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
