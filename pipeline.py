#!/usr/bin/env python3
"""
TerraPulse Pipeline — download satellite imagery and extract features with one command.

Runs the full TerraPulse pipeline via Docker:
  1. Downloads Sentinel-2 + Sentinel-1 seasonal composites
  2. Extracts 1764 spectral/SAR features per 100m grid cell
  3. Produces land cover predictions using the bundled ONNX model

Requires: Python 3.7+ and Docker installed.

Usage:
    python pipeline.py --bbox 10.95 49.38 11.20 49.52 --output ./results
    python pipeline.py --bbox 2.25 48.81 2.42 48.90 --output ./paris --years 2023 --keep-raw
    python pipeline.py --bbox 10.95 49.38 11.20 49.52 --output ./data --no-predict
"""
import argparse
import os
import subprocess
import sys

IMAGE = "ghcr.io/ivanyachukr/terrapulse:latest"


def check_docker():
    """Verify Docker is installed and running."""
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10,
        )
        if r.returncode != 0:
            print("ERROR: Docker is installed but not running.")
            print("       Please start Docker Desktop and try again.")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Docker is not installed.")
        print("       Install Docker from https://www.docker.com/get-started")
        sys.exit(1)


def ensure_image():
    """Pull the image if not already present."""
    r = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"Pulling {IMAGE} (first time only, ~400 MB)...")
        subprocess.run(["docker", "pull", IMAGE], check=True)
    else:
        print(f"Using cached image: {IMAGE}")


def main():
    parser = argparse.ArgumentParser(
        description="Download satellite imagery and extract features for any location on Earth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pipeline.py --bbox 10.95 49.38 11.20 49.52 --output ./nuremberg
  python pipeline.py --bbox 10.95 49.38 11.20 49.52 --years 2023 2024 --output ./data
  python pipeline.py --bbox 2.25 48.81 2.42 48.90 --output ./paris --region paris --keep-raw
  python pipeline.py --bbox -73.99 40.75 -73.95 40.78 --output ./nyc --region nyc --no-predict

Bounding box format: west south east north (WGS84 degrees)
  Tip: use Google Maps to find coordinates — right-click any point to copy lat/lon.
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
        help="Local folder to save results",
    )
    parser.add_argument(
        "--region", type=str, default="region",
        help="Region name used in filenames (default: region)",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep raw satellite GeoTIFF files in output",
    )
    parser.add_argument(
        "--no-predict", action="store_true",
        help="Only download + extract features, skip ONNX prediction",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    bbox = args.bbox
    years = args.years
    region = args.region

    print()
    print("=" * 55)
    print("  TerraPulse Pipeline")
    print("=" * 55)
    print(f"  Bbox:    {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}")
    print(f"  Years:   {years}")
    print(f"  Region:  {region}")
    print(f"  Output:  {output_dir}")
    print(f"  Predict: {'no' if args.no_predict else 'yes'}")
    print(f"  Raw:     {'keep' if args.keep_raw else 'clean up'}")
    print()

    # Pre-checks
    check_docker()
    ensure_image()

    # Build the docker run command
    bbox_str = [str(b) for b in bbox]
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{output_dir}:/output",
        IMAGE,
        "python", "/app/easy_pipeline.py",
        "--bbox", *bbox_str,
        "--years", *[str(y) for y in years],
        "--output", "/output",
        "--region", region,
    ]
    if args.keep_raw:
        cmd.append("--keep-raw")
    if args.no_predict:
        cmd.append("--no-predict")

    print("Running pipeline...\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Pipeline failed (exit code {result.returncode})")
        sys.exit(1)

    # Summary
    print()
    print("=" * 55)
    print("  Done!")
    print("=" * 55)
    for root, dirs, files in os.walk(output_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            rel = os.path.relpath(path, output_dir)
            size = os.path.getsize(path) / (1024 * 1024)
            if size > 0.01:
                print(f"  {rel:45s}  {size:.1f} MB")
    print(f"\n  Saved to: {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
