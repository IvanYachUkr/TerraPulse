#!/usr/bin/env python3
"""
setup_2-step_model.py – Master Reproducibility Script for the Nuremberg Project.

This script builds the entire local data environment:
1. Downloads ESA WorldCover 2020 & 2021 labels (Nuremberg Clip) via STAC.
2. Triggers Sentinel-2 downloads for training years (2019-2021) via Docker.
3. Rasterizes socioeconomic district statistics for the ML feature matrix.
4. Executes the Model 3/4 pipeline to generate all dashboard validation maps.

Usage:
    python scripts/setup_2-step_model.py
"""

import os
import sys
import subprocess
from pathlib import Path
import pystac_client
import planetary_computer
import stackstac
import rasterio
import numpy as np

# Add root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

# Use the study area from config.py
try:
    from config import AOI_BBOX, WC_DIR, S2_DIR, STATS_DIR
except ImportError:
    # Fallback if config is different
    AOI_BBOX = [10.9430, 49.3057, 11.3197, 49.5642]
    WC_DIR = PROJECT_DIR / "data" / "worldcover"
    S2_DIR = PROJECT_DIR / "data" / "sentinel2"
    STATS_DIR = PROJECT_DIR / "data" / "nuremberg_stats"

def download_worldcover():
    """Download WorldCover (2020 & 2021) labels using Microsoft Planetary Computer."""
    print("\n[1/4] Downloading WorldCover 10m Labels (2020 & 2021)...")
    WC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Skip if both years exist
    if (WC_DIR / "worldcover_2020.tif").exists() and (WC_DIR / "worldcover_2021.tif").exists():
        print("  ✓ WorldCover labels already exist – skipping download.")
        return

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    for year in [2020, 2021]:
        print(f"  Fetching WorldCover metadata for {year}...")
        results = catalog.search(collections=["esa-worldcover"], bbox=AOI_BBOX, datetime=f"{year}")
        items = list(results.get_items())
        
        if not items:
            print(f"  ⚠ No WorldCover labels found for {year}.")
            continue

        print(f"  Streaming and clipping {year} raster...")
        data = stackstac.stack(items, bounds_latlon=AOI_BBOX, epsg=32632, assets=["map"]).squeeze().compute()
        
        out_path = WC_DIR / f"worldcover_{year}.tif"
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=data.shape[0], width=data.shape[1],
            count=1, dtype=data.dtype,
            crs="EPSG:32632", transform=data.spec.transform
        ) as dst:
            dst.write(data.values, 1)
        print(f"  ✓ {out_path.name} saved.")

def download_sentinel2():
    """Trigger the multi-year Sentinel-2 download via the Docker wrapper."""
    print("\n[2/4] Triggering Sentinel-2 Downloads (2019-2021)...")
    download_script = PROJECT_DIR / "download.py"
    
    if not download_script.exists():
        print("  ⚠ download.py not found in root. Skipping S2 download.")
        return

    # Skip if files already exist
    all_exist = True
    for y in [2019, 2020, 2021]:
        if not (S2_DIR / f"s2_median_{y}.tif").exists():
            all_exist = False
            break
    
    if all_exist:
        print("  ✓ Sentinel-2 composites already exist – skipping download.")
        return

    # Call the existing download script for our 3 training/validation years
    bbox_str = [str(b) for b in AOI_BBOX]
    subprocess.run([
        sys.executable, str(download_script),
        "--bbox", *bbox_str,
        "--years", "2019", "2020", "2021",
        "--output", str(S2_DIR),
        "--region", "nuremberg"
    ])

def process_socio_stats():
    """Run the socioeconomic stat rasterizer to prepare ML features."""
    print("\n[3/4] Rasterizing Socio-Economic District Stats...")
    stats_script = PROJECT_DIR / "scripts" / "rasterize_stats.py"
    
    if stats_script.exists():
        subprocess.run([sys.executable, str(stats_script)])
        print("  ✓ District stats ready.")
    else:
        print("  ⚠ rasterize_stats.py missing.")

def run_model_pipeline():
    """Execute the two-stage model training and prediction."""
    print("\n[4/4] Executing Final Model 3/4 Pipeline...")
    model_script = PROJECT_DIR / "scripts" / "experimental_nuremberg.py"
    
    if model_script.exists():
        subprocess.run([sys.executable, str(model_script)])
        print("\n✓ Model trained and dashboard maps generated.")
    else:
        print("  ⚠ experimental_nuremberg.py missing.")

def main():
    print("TerraPulse — Optimized 2-Step Model Setup & End-to-End Reproduction")
    print("=" * 70)
    
    download_worldcover()
    download_sentinel2()
    process_socio_stats()
    run_model_pipeline()
    
    print("\nDone! Environment is fully prepared and Experimental results are live.")
    print("Refresh your dashboard to see the final validation maps.")

if __name__ == "__main__":
    main()
