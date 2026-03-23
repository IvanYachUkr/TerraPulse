#!/usr/bin/env python3
"""
Step 2: Extract features using the terrapulse Rust binary.

Runs `terrapulse extract` for each city to produce
features_v7/features_rust_2020_2021.parquet files.

Usage:
    python 02_extract_features.py                       # all cities
    python 02_extract_features.py --cities munich berlin  # specific
"""

import argparse, os, subprocess, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
TERRAPULSE_BIN = os.path.join(PROJECT_ROOT, "terrapulse", "target", "release",
                              "terrapulse" + (".exe" if sys.platform == "win32" else ""))

# Import city list from step 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
step1 = import_module("01_download_data")
CITIES = step1.CITIES
CITY_MAP = step1.CITY_MAP

def ts():
    return time.strftime("%H:%M:%S")


def extract_city(city):
    """Extract features for a single city using terrapulse."""
    raw = os.path.join(CITIES_DIR, city.name, "raw")
    feat = os.path.join(CITIES_DIR, city.name, "features_v7")

    # Check if already extracted
    target = os.path.join(feat, "features_rust_2020_2021.parquet")
    if os.path.exists(target):
        print(f"  [{city.name}] Already extracted — skip")
        return

    if not os.path.isdir(raw):
        print(f"  [{city.name}] No raw data — skip")
        return

    print(f"  [{ts()}] [{city.name}] Extracting features...")
    cmd = [
        TERRAPULSE_BIN, "extract",
        "--year-pairs", "2020_2021",
        "--region", city.name,
        "--raw-dir", raw,
        "--features-dir", feat,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-300:]}")
    else:
        print(f"    Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Extract features via terrapulse")
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    if args.cities:
        cities = [CITY_MAP[n] for n in args.cities if n in CITY_MAP]
    else:
        cities = CITIES

    print(f"\n{'='*70}")
    print(f"  Step 2: Extract Features (Rust)")
    print(f"  Cities: {len(cities)}")
    print(f"{'='*70}\n")

    if not os.path.exists(TERRAPULSE_BIN):
        print(f"ERROR: Binary not found: {TERRAPULSE_BIN}")
        print(f"Build: cd terrapulse && cargo build --release")
        sys.exit(1)

    for city in cities:
        extract_city(city)

    print(f"\n[{ts()}] Feature extraction complete!")


if __name__ == "__main__":
    main()
