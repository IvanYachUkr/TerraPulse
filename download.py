#!/usr/bin/env python3
"""
TerraPulse Satellite Download — one command, no Docker knowledge needed.

Download Sentinel-2 satellite imagery for any region on Earth.
Requires: Python 3.7+ and Docker installed.

Usage:
    python download.py --bbox 10.95 49.38 11.20 49.52 --output ./satellite_data
    python download.py --bbox 10.95 49.38 11.20 49.52 --years 2023 2024 --output C:\\Users\\me\\data
    python download.py --bbox 2.25 48.81 2.42 48.90 --output ./paris --region paris
"""
import argparse
import os
import shutil
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
        description="Download satellite imagery for any location on Earth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download.py --bbox 10.95 49.38 11.20 49.52 --output ./nuremberg
  python download.py --bbox 10.95 49.38 11.20 49.52 --years 2023 2024 --output ./data
  python download.py --bbox 2.25 48.81 2.42 48.90 --output ./paris --region paris
  python download.py --bbox -73.99 40.75 -73.95 40.78 --output ./nyc --region nyc

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
        help="Years to download (default: 2021)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Local folder to save satellite GeoTIFFs",
    )
    parser.add_argument(
        "--region", type=str, default="region",
        help="Region name used in filenames (default: region)",
    )
    args = parser.parse_args()

    # Resolve to absolute path (works on all OS)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    bbox = args.bbox
    years = args.years
    region = args.region

    print()
    print("=" * 55)
    print("  TerraPulse Satellite Download")
    print("=" * 55)
    print(f"  Bbox:   {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}")
    print(f"  Years:  {years}")
    print(f"  Region: {region}")
    print(f"  Output: {output_dir}")
    print()

    # Pre-checks
    check_docker()
    ensure_image()

    # Build the docker run command
    # Mount the output dir into the container at /output
    years_str = " ".join(str(y) for y in years)
    bbox_str = [str(b) for b in bbox]

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{output_dir}:/output",
        IMAGE,
        "python", "/app/easy_download.py",
        "--bbox", *bbox_str,
        "--years", *[str(y) for y in years],
        "--output", "/output",
        "--region", region,
    ]

    print("Downloading satellite imagery...\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Download failed (exit code {result.returncode})")
        sys.exit(1)

    # List what was saved
    tifs = [f for f in os.listdir(output_dir) if f.endswith(".tif")]
    if tifs:
        print()
        print("=" * 55)
        total_mb = 0
        for f in sorted(tifs):
            size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
            total_mb += size
            print(f"  ✓ {f}  ({size:.1f} MB)")
        print(f"\n  {len(tifs)} files, {total_mb:.0f} MB total")
        print(f"  Saved to: {output_dir}")
        print("=" * 55)
    else:
        print(f"\nFiles saved to: {output_dir}")


if __name__ == "__main__":
    main()
