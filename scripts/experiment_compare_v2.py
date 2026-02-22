"""
Quick comparison: Render the new autumn 2022 TIF (from the updated Rust binary
with ref SCL mask [1,3,7,8,9,10]) and compare with the old experiment results.
"""
import numpy as np
from pathlib import Path

import rasterio
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
FRESH_DIR = PROJECT / "data" / "fresh_download"
EXPERIMENT_DIR = FRESH_DIR / "experiment"
OUT_DIR = EXPERIMENT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODATA = -9999.0


def stretch(band, lo=2, hi=98):
    finite = band[np.isfinite(band) & (band != NODATA) & (band > 0)]
    if len(finite) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin = np.percentile(finite, lo)
    vmax = np.percentile(finite, hi)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    s = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    s = np.where(np.isfinite(band) & (band != NODATA), s, 0)
    return (s * 255).astype(np.uint8)


def tif_to_rgb(tif_path):
    with rasterio.open(tif_path) as ds:
        red = ds.read(3).astype(np.float32)   # B04 (idx 2, 1-based = 3)
        green = ds.read(2).astype(np.float32) # B03 (idx 1, 1-based = 2)
        blue = ds.read(1).astype(np.float32)  # B02 (idx 0, 1-based = 1)

    # Stats
    for name, band in [("Red", red), ("Green", green), ("Blue", blue)]:
        valid = band[np.isfinite(band) & (band != NODATA) & (band > 0)]
        nan_pct = 100 * (1 - len(valid) / band.size) if band.size > 0 else 100
        print(f"  {name}: mean={np.mean(valid):.0f} "
              f"p5={np.percentile(valid, 5):.0f} "
              f"p50={np.percentile(valid, 50):.0f} "
              f"p95={np.percentile(valid, 95):.0f} "
              f"NaN={nan_pct:.1f}%")

    r = stretch(red)
    g = stretch(green)
    b = stretch(blue)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def make_comparison(images, titles, out_path, cell_max_w=600):
    n = len(images)
    orig_w, orig_h = images[0].size
    scale = cell_max_w / orig_w
    cell_w = cell_max_w
    cell_h = int(orig_h * scale)

    title_h = 50
    pad = 8
    grid_w = pad + n * (cell_w + pad)
    grid_h = pad + title_h + cell_h + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, title) in enumerate(zip(images, titles)):
        x = pad + i * (cell_w + pad)
        lines = title.split("\n")
        for li, line in enumerate(lines):
            draw.text((x + cell_w // 2, pad + li * 18), line,
                      fill=(220, 220, 220), font=font, anchor="mt")
        thumb = img.resize((cell_w, cell_h), Image.LANCZOS)
        canvas.paste(thumb, (x, pad + title_h))

    canvas.save(out_path, quality=95)
    print(f"\nSaved: {out_path} ({canvas.size[0]}x{canvas.size[1]})")


def main():
    print("=" * 60)
    print("Experiment V2: Comparing old vs new Rust SCL mask")
    print("=" * 60)

    # New Rust result (ref mask [1,3,7,8,9,10] + Q1)
    new_tif = FRESH_DIR / "sentinel2_nuremberg_2022_autumn.tif"
    print(f"\n[NEW] Rust ref mask + Q1: {new_tif.name}")
    img_new = tif_to_rgb(str(new_tif))
    img_new.save(OUT_DIR / "autumn2022_v2_new_rust.png")

    # Load old experiment images for comparison
    img_old_a = Image.open(OUT_DIR / "autumn2022_A.png")
    img_old_b = Image.open(OUT_DIR / "autumn2022_B.png")
    img_old_c = Image.open(OUT_DIR / "autumn2022_C.png")

    # Make comparison grid: Old Rust (A) vs New Rust vs Python Ref+Median (C)
    make_comparison(
        [img_old_a, img_new, img_old_c],
        [
            "OLD Rust mask [0,1,3,8,9,10,11]\nQ1 · 8 scenes",
            "NEW Rust ref mask [1,3,7,8,9,10]\nQ1 · 8 scenes",
            "Python ref mask [1,3,7,8,9,10]\nMedian · 4 scenes",
        ],
        OUT_DIR / "comparison_v2.png",
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
