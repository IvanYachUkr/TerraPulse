"""
Fix Sentinel-2 Processing Baseline 04.00 offset in saved rasters.

Since PB 04.00 (Jan 25, 2022), Sentinel-2 L2A products include a
BOA_ADD_OFFSET of -1000 DN for all spectral bands. Our download pipeline
used `stackstac.stack(..., rescale=False)` which stores raw DN values.

For years >= 2022 the correct reflectance is:
    reflectance = (DN - 1000) / 10000

For years < 2022:
    reflectance = DN / 10000

This script subtracts 1000 from all spectral bands in the post-2022
rasters so that all years use the same `DN / 10000` convention.
"""
import os
import sys
import numpy as np
from pathlib import Path

try:
    import rasterio
except ImportError:
    print("Need rasterio: pip install rasterio")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_V2 = PROJECT_ROOT / "data" / "raw" / "v2"

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# Years that need the offset correction (processed after PB 04.00)
OFFSET_YEARS = [2022, 2023, 2024, 2025]
SEASONS = ["spring", "summer", "autumn"]
OFFSET = 1000  # DN offset to subtract

def needs_correction(tif_path):
    """Check if a raster has already been corrected by reading B02 stats."""
    with rasterio.open(tif_path) as ds:
        nodata = ds.nodata
        b02 = ds.read(1).astype(np.float64)
        if nodata is not None:
            b02[b02 == nodata] = np.nan
        finite = b02[np.isfinite(b02)]
        if len(finite) == 0:
            return False
        p5 = np.percentile(finite, 5)
        # Pre-correction: B02 p5 is ~1200+; post-correction: ~200
        return p5 > 800

def correct_raster(tif_path):
    """Subtract the PB04.00 offset from spectral bands, rewrite in-place."""
    import tempfile, shutil

    with rasterio.open(tif_path) as ds:
        profile = ds.profile.copy()
        nodata = ds.nodata
        n_bands = ds.count
        data = ds.read()
        tags = ds.tags()
        band_descs = [ds.descriptions[i] for i in range(n_bands)]

    n_spectral = len(BANDS)
    for i in range(n_spectral):
        band = data[i].astype(np.float32)
        mask = band != nodata if nodata is not None else np.ones_like(band, dtype=bool)
        band[mask] = band[mask] - OFFSET
        # Clamp negative to 0 (very dark surfaces)
        band[mask] = np.maximum(band[mask], 0)
        data[i] = band

    # Write to temp file, then replace
    tmp_path = str(tif_path) + ".tmp"
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        for i in range(n_bands):
            if band_descs[i]:
                dst.set_band_description(i + 1, band_descs[i])
        tags["PB0400_OFFSET_CORRECTED"] = "true"
        dst.update_tags(**tags)

    # Atomic replace
    shutil.move(tmp_path, str(tif_path))


def main():
    print("Sentinel-2 PB 04.00 Offset Correction")
    print("=" * 50)
    print(f"Offset to subtract: {OFFSET} DN (for years >= 2022)")
    print()

    for year in OFFSET_YEARS:
        for season in SEASONS:
            tif = RAW_V2 / f"sentinel2_nuremberg_{year}_{season}.tif"
            if not tif.exists():
                print(f"  {year}_{season}: MISSING")
                continue

            if not needs_correction(str(tif)):
                print(f"  {year}_{season}: Already corrected (B02 p5 < 800)")
                continue

            # Read before stats
            with rasterio.open(str(tif)) as ds:
                b02 = ds.read(1).astype(np.float64)
                nodata = ds.nodata
                if nodata is not None:
                    b02[b02 == nodata] = np.nan
                before = np.nanmean(b02)

            print(f"  {year}_{season}: B02 mean={before:.0f} -> ", end="")
            correct_raster(str(tif))

            # Verify
            with rasterio.open(str(tif)) as ds:
                b02 = ds.read(1).astype(np.float64)
                if nodata is not None:
                    b02[b02 == nodata] = np.nan
                after = np.nanmean(b02)
            print(f"{after:.0f} (delta={before-after:.0f})")

    print("\nDone! Re-run the feature extraction to get corrected features.")


if __name__ == "__main__":
    main()
