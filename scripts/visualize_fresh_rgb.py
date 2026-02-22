"""
Visualize freshly downloaded Sentinel-2 seasonal composites as true-color RGB PNGs.

Reads the 11-band seasonal TIFs from data/fresh_download/ (produced by the current
Rust binary using 25th-percentile compositing with cloud masking), extracts
B04 (Red), B03 (Green), B02 (Blue), and renders them as natural-color images.

Outputs:
  - Individual PNGs per year/season  → data/fresh_download/rgb_preview/
  - Grid overview (6 years × 3 seasons) → data/fresh_download/rgb_preview/overview_grid.png

Band order in TIFs: B02(0), B03(1), B04(2), B05(3), B06(4), B07(5),
                    B08(6), B8A(7), B11(8), B12(9), VALID_FRAC(10)
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    import rasterio
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Need rasterio and Pillow: pip install rasterio Pillow")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT / "data" / "fresh_download"
OUT_DIR = RAW_DIR / "rgb_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
SEASONS = ["spring", "summer", "autumn"]

# Band indices (0-based) — TIF band numbers are 1-based
B02_IDX = 0  # Blue
B03_IDX = 1  # Green
B04_IDX = 2  # Red

NODATA = -9999.0


def percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch a single band to [0, 255] using percentile clipping."""
    finite = band[np.isfinite(band) & (band != NODATA)]
    if len(finite) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin = np.percentile(finite, lo)
    vmax = np.percentile(finite, hi)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    stretched = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    stretched = np.where(np.isfinite(band) & (band != NODATA), stretched, 0)
    return (stretched * 255).astype(np.uint8)


def tif_to_rgb(tif_path: str) -> tuple:
    """Read a seasonal TIF and return (H, W, 3) uint8 RGB + band stats."""
    with rasterio.open(tif_path) as ds:
        # Read RGB bands (1-indexed in rasterio)
        red = ds.read(B04_IDX + 1).astype(np.float32)
        green = ds.read(B03_IDX + 1).astype(np.float32)
        blue = ds.read(B02_IDX + 1).astype(np.float32)

    # Compute stats on valid pixels
    stats = {}
    for name, band in [("B04_red", red), ("B03_green", green), ("B02_blue", blue)]:
        valid = band[np.isfinite(band) & (band != NODATA) & (band > 0)]
        if len(valid) > 0:
            stats[name] = {
                "mean": float(np.mean(valid)),
                "p5": float(np.percentile(valid, 5)),
                "p50": float(np.percentile(valid, 50)),
                "p95": float(np.percentile(valid, 95)),
                "nan_pct": float(100 * (1 - len(valid) / band.size)),
            }
        else:
            stats[name] = {"mean": 0, "p5": 0, "p50": 0, "p95": 0, "nan_pct": 100}

    # Per-band percentile stretch
    r = percentile_stretch(red)
    g = percentile_stretch(green)
    b = percentile_stretch(blue)

    rgb = np.stack([r, g, b], axis=-1)  # H x W x 3
    return rgb, stats


def make_grid(images: dict, cell_max_w: int = 400) -> Image.Image:
    """
    Arrange images into a years×seasons grid with labels.
    images: {(year, season): PIL.Image}
    """
    sample = next(iter(images.values()))
    orig_w, orig_h = sample.size
    scale = cell_max_w / orig_w
    cell_w = cell_max_w
    cell_h = int(orig_h * scale)

    label_h = 28
    pad = 4

    grid_w = pad + len(SEASONS) * (cell_w + pad)
    grid_h = pad + label_h + len(YEARS) * (cell_h + label_h + pad)

    canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Column headers (seasons)
    for si, season in enumerate(SEASONS):
        x = pad + si * (cell_w + pad) + cell_w // 2
        draw.text((x, pad), season.capitalize(), fill=(220, 220, 220),
                  font=font, anchor="mt")

    # Row labels + images
    for yi, year in enumerate(YEARS):
        y_off = pad + label_h + yi * (cell_h + label_h + pad)
        for si, season in enumerate(SEASONS):
            x_off = pad + si * (cell_w + pad)
            key = (year, season)
            if key in images:
                thumb = images[key].resize((cell_w, cell_h), Image.LANCZOS)
                canvas.paste(thumb, (x_off, y_off))
            # Year label on left of first column
            if si == 0:
                draw.text((x_off + 4, y_off + 4), str(year),
                          fill=(255, 255, 100), font=font)

    return canvas


def main():
    print("=" * 60)
    print("Visualizing Fresh Seasonal Composites (Rust 25th-percentile)")
    print(f"  Source: {RAW_DIR}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    images = {}

    for year in YEARS:
        for season in SEASONS:
            tif = RAW_DIR / f"sentinel2_nuremberg_{year}_{season}.tif"
            if not tif.exists():
                print(f"  MISSING: {tif.name}")
                continue

            print(f"\n  {year} {season}:")
            rgb, stats = tif_to_rgb(str(tif))

            # Print compact stats
            for bname, s in stats.items():
                print(f"    {bname}: mean={s['mean']:.1f}  "
                      f"p5={s['p5']:.1f}  p50={s['p50']:.1f}  "
                      f"p95={s['p95']:.1f}  NaN={s['nan_pct']:.1f}%")

            # Save individual PNG
            img = Image.fromarray(rgb)
            out_path = OUT_DIR / f"rgb_{year}_{season}.png"
            img.save(out_path)
            print(f"    -> {out_path.name}  ({img.size[0]}x{img.size[1]})")

            images[(year, season)] = img

    # Generate overview grid
    if images:
        print(f"\n  Generating overview grid ({len(images)} images)...")
        grid = make_grid(images)
        grid_path = OUT_DIR / "overview_grid.png"
        grid.save(grid_path, quality=95)
        print(f"  -> {grid_path.name}  ({grid.size[0]}x{grid.size[1]})")

    print(f"\n{'='*60}")
    print(f"Done! {len(images)} images saved to {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
