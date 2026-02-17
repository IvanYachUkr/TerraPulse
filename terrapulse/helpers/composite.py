#!/usr/bin/env python3
"""
Composite helper for terrapulse Rust pipeline.

Called by the Rust download module to handle reprojection + median compositing
using rasterio, since Rust doesn't have mature reprojection libraries.

Optimizations:
  - GDAL HTTP timeouts set at OS-level BEFORE rasterio import
  - Parallel band downloads (ThreadPoolExecutor)
  - Hard per-scene timeout to prevent infinite hangs
  - Retries on failed band reads

Usage:
    python composite.py \\
        --scenes-json scenes.json \\
        --anchor-ref anchor_utm32632_10m.tif \\
        --output sentinel2_nuremberg_2024_summer.tif \\
        --year 2024
"""

import os

# ── MUST be set before importing rasterio/GDAL ──
os.environ["GDAL_HTTP_TIMEOUT"] = "60"
os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = "15"
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.TIF"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["CPL_CURL_VERBOSE"] = "NO"

import argparse
import json
import sys
import signal
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SCL_EXCLUDE = {0, 1, 2, 3, 8, 9, 10, 11}
NODATA = -9999
MAX_BAND_WORKERS = 6     # parallel band downloads per scene
SCENE_TIMEOUT = 120      # hard timeout per scene (seconds)


def read_band_warped(href, dst_crs, dst_transform, dst_width, dst_height, is_scl=False):
    """Read a single band via WarpedVRT, reprojecting on-the-fly."""
    with rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=Resampling.nearest if is_scl else Resampling.bilinear,
            dst_nodata=0 if is_scl else np.nan,
        ) as vrt:
            return vrt.read(1)


def download_scene_inner(scene, dst_crs, dst_transform, dst_width, dst_height):
    """Download all bands for one scene in parallel."""
    bands = scene["bands"]
    futures = {}

    with ThreadPoolExecutor(max_workers=MAX_BAND_WORKERS) as executor:
        for b in SENTINEL_BANDS:
            futures[executor.submit(
                read_band_warped, bands[b], dst_crs, dst_transform, dst_width, dst_height
            )] = ("spectral", b)
        futures[executor.submit(
            read_band_warped, bands["SCL"], dst_crs, dst_transform, dst_width, dst_height, True
        )] = ("scl", "SCL")

        spectral_dict = {}
        scl = None
        for future in as_completed(futures, timeout=SCENE_TIMEOUT):
            band_type, band_name = futures[future]
            data = future.result()
            if band_type == "scl":
                scl = data
            else:
                spectral_dict[band_name] = data

    spectral_stack = np.stack([spectral_dict[b] for b in SENTINEL_BANDS])
    return spectral_stack, scl


def process_scenes(scenes, dst_crs, dst_transform, dst_width, dst_height, year):
    """Download, reproject, mask, and composite all scenes."""
    all_spectral = []
    all_scl = []

    for i, scene in enumerate(scenes):
        try:
            spectral_stack, scl = download_scene_inner(
                scene, dst_crs, dst_transform, dst_width, dst_height)
            all_spectral.append(spectral_stack)
            all_scl.append(scl)
            print(f"    Scene {i+1}/{len(scenes)}: OK", file=sys.stderr)
        except TimeoutError:
            print(f"    Scene {i+1}/{len(scenes)}: TIMEOUT ({SCENE_TIMEOUT}s) - skipping", file=sys.stderr)
            continue
        except Exception as e:
            print(f"    Scene {i+1}/{len(scenes)}: FAILED ({e})", file=sys.stderr)
            continue

    if not all_spectral:
        return None, None

    # Cloud masking
    scl_stack = np.stack(all_scl)
    valid_mask = np.ones_like(scl_stack, dtype=bool)
    valid_mask &= (scl_stack > 0)
    for cls in SCL_EXCLUDE:
        valid_mask &= (scl_stack != cls)

    valid_frac = valid_mask.mean(axis=0).astype(np.float32)

    # Mask invalid pixels
    spectral_4d = np.stack(all_spectral).astype(np.float32)
    for s in range(spectral_4d.shape[0]):
        spectral_4d[s, :, ~valid_mask[s]] = np.nan

    # Median composite
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(spectral_4d, axis=0)

    # PB 04.00 offset correction (Jan 2022+)
    if year >= 2022:
        composite = np.maximum(composite - 1000.0, 0.0)

    return composite, valid_frac


def write_composite_tif(composite, valid_frac, dst_crs, dst_transform, dst_height, dst_width, output_path):
    """Write composite + valid_fraction to a multi-band GeoTIFF."""
    n_bands = composite.shape[0]
    comp_clean = np.where(np.isnan(composite), NODATA, composite).astype(np.float32)
    vf_clean = np.where(np.isnan(valid_frac), NODATA, valid_frac).astype(np.float32)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=dst_height, width=dst_width,
        count=n_bands + 1, dtype="float32",
        crs=dst_crs, transform=dst_transform,
        compress="lzw", nodata=NODATA,
    ) as dst:
        for i, band_name in enumerate(SENTINEL_BANDS):
            dst.write(comp_clean[i], i + 1)
            dst.set_band_description(i + 1, band_name)
        dst.write(vf_clean, n_bands + 1)
        dst.set_band_description(n_bands + 1, "VALID_FRACTION")


def main():
    parser = argparse.ArgumentParser(description="Composite helper for terrapulse")
    parser.add_argument("--scenes-json", required=True)
    parser.add_argument("--anchor-ref", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    with open(args.scenes_json) as f:
        scenes = json.load(f)

    with rasterio.open(args.anchor_ref) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    print(f"  Compositing {len(scenes)} scenes -> {dst_width}x{dst_height} ...", file=sys.stderr)

    composite, valid_frac = process_scenes(
        scenes, dst_crs, dst_transform, dst_width, dst_height, args.year)

    if composite is None:
        print("  ERROR: No valid scenes!", file=sys.stderr)
        sys.exit(1)

    write_composite_tif(composite, valid_frac, dst_crs, dst_transform, dst_height, dst_width, args.output)
    print(f"  Done: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
