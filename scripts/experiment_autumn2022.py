"""
Experiment: Download 8 raw autumn-2022 scenes for Nuremberg, apply the
reference algorithm's SCL mask (exclude 1,3,7,8,9,10), compute the
first-quartile composite, and render as RGB.

Compares THREE compositing strategies side-by-side:
  A) Current Rust mask  : exclude [0,1,3,8,9,10,11]      + Q1 (25th percentile)
  B) Reference algorithm: exclude [1,3,7,8,9,10]          + Q1 (25th percentile)
  C) Reference algorithm: exclude [1,3,7,8,9,10]          + Median (50th percentile)

Outputs:
  data/fresh_download/experiment/comparison_grid.png
"""

import os
import sys
import warnings
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "data" / "fresh_download" / "experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nuremberg config
BBOX = [10.95, 49.38, 11.20, 49.52]
EPSG = 32632
YEAR = 2022
SEASON_START = f"{YEAR}-09-01"
SEASON_END   = f"{YEAR}-10-31"
ANCHOR_TIF   = PROJECT / "data" / "grid" / "anchor_utm32632_10m.tif"

# Bands to download (we only need RGB + SCL for this experiment)
RGB_BANDS = ["B02", "B03", "B04"]  # Blue, Green, Red
NODATA = -9999.0

# Three SCL mask strategies
MASK_RUST    = {0, 1, 3, 8, 9, 10, 11}       # Current Rust code
MASK_REF     = {1, 3, 7, 8, 9, 10}            # Reference algorithm (from image)


def search_scenes():
    """Search STAC for autumn 2022 scenes over Nuremberg."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=f"{SEASON_START}/{SEASON_END}",
        query={"eo:cloud_cover": {"lt": 60}},
    ).item_collection()
    print(f"Found {len(items)} scenes for autumn 2022")
    return items


def load_stack(items):
    """Stack scenes using stackstac and return spectral + SCL arrays."""
    import rasterio
    import stackstac
    from rasterio.enums import Resampling

    warnings.filterwarnings("ignore", module="stackstac")

    # Read anchor grid for target dimensions
    with rasterio.open(ANCHOR_TIF) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    print(f"Anchor grid: {dst_width}x{dst_height}, CRS={dst_crs}")

    # Stack RGB bands
    print("Stacking RGB bands...")
    spectral = stackstac.stack(
        items, assets=RGB_BANDS, bounds_latlon=BBOX,
        resolution=10, epsg=EPSG, dtype="float64",
        fill_value=np.nan, resampling=Resampling.bilinear, chunksize=1024,
        rescale=False,
    )

    # Stack SCL
    print("Stacking SCL band...")
    scl = stackstac.stack(
        items, assets=["SCL"], bounds_latlon=BBOX,
        resolution=10, epsg=EPSG, dtype="float64",
        fill_value=np.nan, resampling=Resampling.nearest, chunksize=1024,
        rescale=False,
    ).sel(band="SCL")

    import xarray as xr
    spectral, scl = xr.align(spectral, scl, join="exact")
    spectral = spectral.sel(band=RGB_BANDS)

    return spectral, scl, dst_width, dst_height, dst_transform, dst_crs


def apply_boa_offset(spectral, scl):
    """Subtract BOA_ADD_OFFSET (+1000) for 2022+ scenes where median B02 > 900."""
    import xarray as xr
    import dask.array as da

    # Check each scene's B02 median
    b02 = spectral.sel(band="B02")
    n_scenes = spectral.sizes["time"]
    print(f"Checking BOA_ADD_OFFSET for {n_scenes} scenes...")

    # Compute the spectral data (we need it in memory anyway)
    spectral_arr = spectral.compute()
    scl_arr = scl.compute()

    # For each scene, check B02 median and subtract 1000 if needed
    n_corrected = 0
    for t in range(n_scenes):
        b02_scene = spectral_arr.isel(time=t).sel(band="B02").values
        valid = b02_scene[np.isfinite(b02_scene) & (b02_scene > 0)]
        if len(valid) > 0:
            median = np.median(valid)
            if median > 900:
                # Subtract 1000 from all bands for this scene
                for band in RGB_BANDS:
                    vals = spectral_arr.isel(time=t).sel(band=band).values
                    mask = np.isfinite(vals) & (vals > 0)
                    vals[mask] = np.maximum(vals[mask] - 1000.0, 0.0)
                n_corrected += 1

    print(f"  BOA corrected: {n_corrected}/{n_scenes} scenes")
    return spectral_arr, scl_arr


def make_composite(spectral, scl, exclude_set, method="q1"):
    """
    Composite using given SCL exclude set and aggregation method.

    method: 'q1' (25th percentile) or 'median' (50th percentile)
    """
    import dask.array as da

    n_scenes = spectral.sizes["time"]
    print(f"  Compositing with mask={sorted(exclude_set)}, method={method}, "
          f"{n_scenes} scenes...")

    # Build valid mask: True where SCL is not in exclude_set and is finite
    scl_vals = scl.values  # [time, y, x]
    valid = np.isfinite(scl_vals)
    for cls in exclude_set:
        valid = valid & (scl_vals != cls)

    # Count valid observations per pixel
    n_valid = valid.sum(axis=0)
    total_pixels = n_valid.size
    no_data_pixels = (n_valid == 0).sum()
    print(f"    Valid obs per pixel: min={n_valid.min()}, "
          f"median={np.median(n_valid):.0f}, max={n_valid.max()}, "
          f"no-data pixels={no_data_pixels} ({100*no_data_pixels/total_pixels:.1f}%)")

    # For each band, compute composite
    composites = {}
    for band in RGB_BANDS:
        band_data = spectral.sel(band=band).values  # [time, y, x]
        # Mask invalid pixels
        masked = np.where(valid & np.isfinite(band_data), band_data, np.nan)

        if method == "q1":
            # 25th percentile along time axis
            comp = np.nanpercentile(masked, 25, axis=0)
        else:
            # median
            comp = np.nanmedian(masked, axis=0)

        composites[band] = comp.astype(np.float32)

    return composites, n_valid


def warp_to_anchor(composites, src_xs, src_ys, dst_width, dst_height,
                   dst_transform, dst_crs):
    """Reproject composite to anchor grid."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    # Source transform from stackstac coordinates
    xs = np.asarray(src_xs)
    ys = np.asarray(src_ys)
    rx = float(np.abs(xs[1] - xs[0]))
    ry = float(np.abs(ys[1] - ys[0]))
    left = float(xs.min()) - rx / 2
    right = float(xs.max()) + rx / 2
    bottom = float(ys.min()) - ry / 2
    top = float(ys.max()) + ry / 2
    src_transform = rasterio.transform.from_bounds(
        left, bottom, right, top, len(xs), len(ys))
    src_crs = CRS.from_epsg(EPSG)

    warped = {}
    for band, data in composites.items():
        clean = np.where(np.isnan(data), NODATA, data).astype(np.float32)
        dst = np.full((dst_height, dst_width), NODATA, dtype=np.float32)
        reproject(
            source=clean, destination=dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=NODATA, dst_nodata=NODATA,
        )
        warped[band] = dst

    return warped


def render_rgb(composites, title=""):
    """Convert band composites to an RGB image with percentile stretch."""
    from PIL import Image

    red = composites["B04"]
    green = composites["B03"]
    blue = composites["B02"]

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

    r = stretch(red)
    g = stretch(green)
    b = stretch(blue)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb)


def make_comparison_grid(images, titles, out_path):
    """Create a side-by-side comparison grid."""
    from PIL import Image, ImageDraw, ImageFont

    n = len(images)
    max_w = 600
    sample = images[0]
    orig_w, orig_h = sample.size
    scale = max_w / orig_w
    cell_w = max_w
    cell_h = int(orig_h * scale)

    title_h = 60
    pad = 8

    grid_w = pad + n * (cell_w + pad)
    grid_h = pad + title_h + cell_h + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_big = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_big = font

    for i, (img, title) in enumerate(zip(images, titles)):
        x = pad + i * (cell_w + pad)
        y = pad + title_h

        # Draw title (multiline)
        lines = title.split("\n")
        for li, line in enumerate(lines):
            draw.text((x + cell_w // 2, pad + li * 20), line,
                      fill=(220, 220, 220), font=font if li > 0 else font_big,
                      anchor="mt")

        thumb = img.resize((cell_w, cell_h), Image.LANCZOS)
        canvas.paste(thumb, (x, y))

    canvas.save(out_path, quality=95)
    print(f"\nSaved comparison grid: {out_path} ({canvas.size[0]}x{canvas.size[1]})")
    return canvas


def main():
    print("=" * 70)
    print("Autumn 2022 Compositing Experiment")
    print("=" * 70)

    # 1. Search scenes
    items = search_scenes()

    # 2. Load stack
    spectral, scl, dst_w, dst_h, dst_t, dst_crs = load_stack(items)

    # Get coordinate arrays for reprojection
    src_xs = spectral.coords["x"].values
    src_ys = spectral.coords["y"].values

    # 3. Apply BOA offset correction
    spectral, scl = apply_boa_offset(spectral, scl)

    # 4. Three compositing strategies
    strategies = [
        ("A: Rust mask + Q1\nexclude [0,1,3,8,9,10,11]",
         MASK_RUST, "q1"),
        ("B: Ref mask + Q1\nexclude [1,3,7,8,9,10]",
         MASK_REF, "q1"),
        ("C: Ref mask + Median\nexclude [1,3,7,8,9,10]",
         MASK_REF, "median"),
    ]

    images = []
    titles = []
    for title, mask, method in strategies:
        print(f"\n--- {title.split(chr(10))[0]} ---")
        comp, n_valid = make_composite(spectral, scl, mask, method)
        warped = warp_to_anchor(comp, src_xs, src_ys, dst_w, dst_h, dst_t,
                                dst_crs)
        img = render_rgb(warped, title)

        # Save individual image
        label = title.split(":")[0].strip()
        img.save(OUT_DIR / f"autumn2022_{label}.png")
        print(f"  Saved autumn2022_{label}.png")

        images.append(img)
        titles.append(title)

    # 5. Comparison grid
    grid_path = OUT_DIR / "comparison_grid.png"
    make_comparison_grid(images, titles, grid_path)

    print(f"\n{'='*70}")
    print("Done! Check data/fresh_download/experiment/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
