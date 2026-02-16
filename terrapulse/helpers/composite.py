#!/usr/bin/env python3
"""
Composite helper for terrapulse Rust pipeline.

Called by the Rust download module to handle reprojection + median compositing
using rasterio, since Rust doesn't have mature reprojection libraries.

Usage:
    python composite.py \\
        --scenes-json scenes.json \\
        --anchor-ref anchor_utm32632_10m.tif \\
        --output sentinel2_nuremberg_2024_summer.tif \\
        --year 2024
"""

import argparse
import json
import sys
import warnings

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SCL_EXCLUDE = {0, 1, 2, 3, 8, 9, 10, 11}
NODATA = -9999


def read_band_warped(href, dst_crs, dst_transform, dst_width, dst_height, is_scl=False):
    """Read a single band via WarpedVRT, reprojecting on-the-fly."""
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True):
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


def process_scenes(scenes, dst_crs, dst_transform, dst_width, dst_height, year):
    """Download, reproject, mask, and composite all scenes."""
    all_spectral = []
    all_scl = []

    for i, scene in enumerate(scenes):
        bands = scene["bands"]
        try:
            # Read spectral bands
            spectral_stack = np.stack([
                read_band_warped(bands[b], dst_crs, dst_transform, dst_width, dst_height)
                for b in SENTINEL_BANDS
            ])
            # Read SCL
            scl = read_band_warped(
                bands["SCL"], dst_crs, dst_transform, dst_width, dst_height, is_scl=True
            )
            all_spectral.append(spectral_stack)
            all_scl.append(scl)
            print(f"    Scene {i+1}/{len(scenes)}: {scene['id'][:30]}...", file=sys.stderr)
        except Exception as e:
            print(f"    Scene {i+1}/{len(scenes)}: FAILED ({e})", file=sys.stderr)
            continue

    if not all_spectral:
        return None, None

    # Cloud masking
    scl_stack = np.stack(all_scl)  # [n_scenes, H, W]
    valid_mask = np.ones_like(scl_stack, dtype=bool)
    valid_mask &= (scl_stack > 0)
    for cls in SCL_EXCLUDE:
        valid_mask &= (scl_stack != cls)

    valid_frac = valid_mask.mean(axis=0).astype(np.float32)

    # Mask invalid pixels
    spectral_4d = np.stack(all_spectral).astype(np.float32)  # [n_scenes, n_bands, H, W]
    for s in range(spectral_4d.shape[0]):
        spectral_4d[s, :, ~valid_mask[s]] = np.nan

    # Median composite
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(spectral_4d, axis=0)  # [n_bands, H, W]

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
    parser.add_argument("--scenes-json", required=True, help="Path to JSON with signed scene URLs")
    parser.add_argument("--anchor-ref", required=True, help="Path to anchor reference GeoTIFF")
    parser.add_argument("--output", required=True, help="Output GeoTIFF path")
    parser.add_argument("--year", type=int, required=True, help="Year (for PB 04.00 correction)")
    args = parser.parse_args()

    # Load scenes
    with open(args.scenes_json) as f:
        scenes = json.load(f)

    # Read anchor reference for target CRS/transform/dimensions
    with rasterio.open(args.anchor_ref) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    print(f"  Compositing {len(scenes)} scenes -> {dst_width}x{dst_height}...", file=sys.stderr)

    composite, valid_frac = process_scenes(
        scenes, dst_crs, dst_transform, dst_width, dst_height, args.year
    )

    if composite is None:
        print("  ERROR: No valid scenes!", file=sys.stderr)
        sys.exit(1)

    write_composite_tif(composite, valid_frac, dst_crs, dst_transform, dst_height, dst_width, args.output)
    print(f"  Done: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
