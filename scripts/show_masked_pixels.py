"""
Render Autumn 2022 TIF as RGB but highlight totally masked (no-data) pixels as bright red.
"""
import os
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
TIF_PATH = PROJECT / "data" / "fresh_download" / "sentinel2_nuremberg_2022_autumn.tif"
OUT_PATH = PROJECT / "data" / "fresh_download" / "rgb_preview" / "autumn2022_masked_red.png"

NODATA = -9999.0

def stretch(band, mask, lo=2, hi=98):
    # Only use valid pixels for calculating stats to stretch
    valid_data = band[mask]
    if len(valid_data) == 0:
        return np.zeros_like(band, dtype=np.uint8)
        
    vmin = np.percentile(valid_data, lo)
    vmax = np.percentile(valid_data, hi)
    
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
        
    s = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return (s * 255).astype(np.uint8)

def main():
    print(f"Reading {TIF_PATH.name}...")
    with rasterio.open(TIF_PATH) as ds:
        blue = ds.read(1).astype(np.float32)   # B02
        green = ds.read(2).astype(np.float32)  # B03
        red = ds.read(3).astype(np.float32)    # B04

    # Determine which pixels are entirely masked (no valid data)
    # The Rust code writes NODATA for pixels that had no valid observations
    is_nodata = (red == NODATA) | np.isnan(red)
    valid_mask = ~is_nodata

    print(f"Total pixels: {red.size}")
    print(f"Masked pixels: {is_nodata.sum()} ({100 * is_nodata.sum() / red.size:.2f}%)")

    # Stretch the valid pixels
    r = stretch(red, valid_mask)
    g = stretch(green, valid_mask)
    b = stretch(blue, valid_mask)

    # Apply bright red to the masked pixels
    r[is_nodata] = 255
    g[is_nodata] = 0
    b[is_nodata] = 0

    # Stack to RGB
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb)
    
    os.makedirs(OUT_PATH.parent, exist_ok=True)
    img.save(OUT_PATH)
    print(f"Saved image with red masked pixels to {OUT_PATH}")

if __name__ == "__main__":
    main()
