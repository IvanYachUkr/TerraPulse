#!/usr/bin/env python3
"""
Composite helper for terrapulse Rust pipeline.

Called by the Rust download module to handle reprojection + median compositing
using rasterio, since Rust doesn't have mature reprojection libraries.

Optimizations (v2):
  - GDAL HTTP tuning set BEFORE rasterio import
  - Parallel SCENE downloads (ThreadPoolExecutor at scene level)
  - Parallel band downloads within each scene
  - Hard per-scene timeout to prevent infinite hangs
  - Retries on failed band reads
  - Vectorized cloud masking via np.isin()

Usage:
    python composite.py \\
        --scenes-json scenes.json \\
        --anchor-ref anchor_utm32632_10m.tif \\
        --output sentinel2_nuremberg_2024_summer.tif \\
        --year 2024
"""

import os

# ── MUST be set before importing rasterio/GDAL ──
os.environ["GDAL_HTTP_TIMEOUT"] = "15"           # was 60 — fail fast, retry instead
os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = "8"       # was 15
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "67108864"  # 64 MB VSIL cache
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.TIF"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_VERSION"] = "2"       # force HTTP/2 for multiplexing
os.environ["CPL_CURL_VERBOSE"] = "NO"

import argparse
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# SCL classes to exclude: 0=nodata, 1=defective, 2=dark, 3=shadow,
# 8=cloud_med, 9=cloud_high, 10=cirrus, 11=snow
SCL_EXCLUDE = frozenset({0, 1, 2, 3, 8, 9, 10, 11})
SCL_EXCLUDE_ARR = np.array(sorted(SCL_EXCLUDE), dtype=np.uint8)
NODATA = -9999
MAX_BAND_WORKERS = 10    # parallel band downloads per scene
MAX_SCENE_WORKERS = 4    # parallel scene downloads
SCENE_TIMEOUT = 30       # hard timeout per scene (seconds) — was 120/180, caused 2min hangs
BAND_RETRIES = 2         # retries per band on failure


def read_band_warped(href, dst_crs, dst_transform, dst_width, dst_height, is_scl=False):
    """Read a single band via WarpedVRT, reprojecting on-the-fly."""
    last_err = None
    for attempt in range(1 + BAND_RETRIES):
        try:
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
        except Exception as e:
            last_err = e
            if attempt < BAND_RETRIES:
                import time
                time.sleep(1.0 * (attempt + 1))
    raise last_err


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


def _download_one_scene(args):
    """Wrapper for scene-level parallelism. Returns (index, result_or_None)."""
    idx, scene, dst_crs, dst_transform, dst_width, dst_height = args
    try:
        spectral_stack, scl = download_scene_inner(
            scene, dst_crs, dst_transform, dst_width, dst_height)
        print(f"    Scene {idx+1}: OK", file=sys.stderr)
        return idx, spectral_stack, scl
    except Exception as e:
        print(f"    Scene {idx+1}: FAILED ({e})", file=sys.stderr)
        return idx, None, None


def process_scenes(scenes, dst_crs, dst_transform, dst_width, dst_height, year):
    """Download, reproject, mask, and composite all scenes — parallelized."""
    n_scenes = len(scenes)
    n_bands = len(SENTINEL_BANDS)

    print(f"    Downloading {n_scenes} scenes ({MAX_SCENE_WORKERS} parallel)...",
          file=sys.stderr, flush=True)

    # ── Parallel scene downloads ──
    args_list = [
        (i, scene, dst_crs, dst_transform, dst_width, dst_height)
        for i, scene in enumerate(scenes)
    ]

    all_spectral = []
    all_scl = []

    with ThreadPoolExecutor(max_workers=MAX_SCENE_WORKERS) as pool:
        futures = {pool.submit(_download_one_scene, a): a[0] for a in args_list}
        for future in as_completed(futures):
            idx, spectral, scl = future.result()
            if spectral is not None:
                all_spectral.append(spectral)
                all_scl.append(scl)

    n_ok = len(all_spectral)
    print(f"    {n_ok}/{n_scenes} scenes OK", file=sys.stderr, flush=True)

    if not all_spectral:
        return None, None

    # ── Cloud masking (vectorized) ──
    scl_stack = np.stack(all_scl)                       # (n_ok, H, W)
    # Single vectorized call instead of per-class loop
    valid_mask = ~np.isin(scl_stack, SCL_EXCLUDE_ARR)   # (n_ok, H, W)
    valid_mask &= (scl_stack > 0)

    valid_frac = valid_mask.mean(axis=0).astype(np.float32)

    # ── Mask invalid pixels + median composite ──
    spectral_4d = np.stack(all_spectral, dtype=np.float32)   # (n_ok, bands, H, W)
    # Mask invalid pixels: valid_mask is (n_ok, H, W), broadcast across bands
    # Use np.where for correct broadcasting over the band dimension
    invalid = ~valid_mask[:, np.newaxis, :, :]               # (n_ok, 1, H, W)
    spectral_4d = np.where(invalid, np.nan, spectral_4d)     # broadcasts correctly

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(spectral_4d, axis=0)    # (bands, H, W)

    # Free large arrays immediately
    del spectral_4d, scl_stack, valid_mask, invalid

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
