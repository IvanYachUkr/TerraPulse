"""
Download Sentinel-2 + Sentinel-1 SAR for Nuremberg timeseries (2017-2025).

Uses stackstac + planetary_computer for correct S2 compositing.
Uses rasterio WarpedVRT for SAR (handles LERC compression that Rust can't).
Outputs to nuremberg_timeseries/raw/ for Rust extract/predict pipeline.

Usage:
    python scripts/download_timeseries_s2.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", module="stackstac")
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG, GRID_REF_PATH, PROJECT_ROOT

# ---------- Config ----------
NUREMBERG_BBOX = CFG["aoi"]["bbox"]
SENTINEL_BANDS = CFG["sentinel2"]["bands"]
SEASONS_CFG = CFG["sentinel2"]["seasons"]
SEASON_ORDER = CFG["sentinel2"]["season_order"]
CLOUD_MAX = CFG["sentinel2"]["cloud_cover_max"]
MIN_SCENES = CFG["sentinel2"]["min_scenes"]
FALLBACK_CLOUD_MAX = CFG["sentinel2"]["fallback_cloud_max"]
FALLBACK_EXPAND_DAYS = CFG["sentinel2"]["fallback_window_expand_days"]
SCL_EXCLUDE = CFG["scl_mask"]["exclude_classes"]
NODATA = CFG["sentinel2"]["nodata"]

# SAR constants (match run_multi_city_pipeline_v5.py)
SAR_BANDS = ["vv", "vh"]
SAR_NODATA = -9999
SAR_SEASON_DATES = {
    "spring": ("04-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "autumn": ("09-01", "10-31"),
}

# Override: download ALL years
ALL_YEARS = list(range(2017, 2026))

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "nuremberg_timeseries", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

ts = lambda: time.strftime("%H:%M:%S")


def transform_from_xy_centers(xs, ys, expected_res):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    assert xs.size > 1 and ys.size > 1
    rx = float(np.abs(xs[1] - xs[0]))
    ry = float(np.abs(ys[1] - ys[0]))
    assert abs(rx - expected_res) < 1e-3
    assert abs(ry - expected_res) < 1e-3
    import rasterio
    left = float(xs.min()) - rx / 2
    top = float(ys.max()) + ry / 2
    return rasterio.transform.from_bounds(
        left, float(ys.min()) - ry / 2,
        float(xs.max()) + rx / 2, top,
        len(xs), len(ys),
    )


# ============================================================
# S2 DOWNLOAD
# ============================================================
def download_s2():
    import planetary_computer
    import pystac_client
    import rasterio
    import stackstac
    import xarray as xr
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    print("=" * 60)
    print(f"[{ts()}] Sentinel-2 Download (stackstac)")
    print("=" * 60)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    target_epsg = CFG["aoi"]["epsg"]
    expected_res = float(CFG["sentinel2"]["resolution"])

    with rasterio.open(GRID_REF_PATH) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
    print(f"  Anchor: {dst_width}x{dst_height}, CRS={dst_crs}")

    for year in ALL_YEARS:
        for season in SEASON_ORDER:
            path = os.path.join(OUT_DIR, f"sentinel2_nuremberg_{year}_{season}.tif")
            if os.path.exists(path):
                mb = os.path.getsize(path) / 1e6
                print(f"  [{year}/{season}] Exists ({mb:.1f} MB) -- skip")
                continue

            start_date = f"{year}-{SEASONS_CFG[season][0]}"
            end_date = f"{year}-{SEASONS_CFG[season][1]}"
            cloud_max = CLOUD_MAX[season]

            print(f"\n  [{ts()}] [{year}/{season}] Searching...")
            items = catalog.search(
                collections=["sentinel-2-l2a"], bbox=NUREMBERG_BBOX,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": cloud_max}},
            ).item_collection()
            n_scenes = len(items)
            threshold_used = cloud_max

            if n_scenes < MIN_SCENES:
                items = catalog.search(
                    collections=["sentinel-2-l2a"], bbox=NUREMBERG_BBOX,
                    datetime=f"{start_date}/{end_date}",
                    query={"eo:cloud_cover": {"lt": FALLBACK_CLOUD_MAX}},
                ).item_collection()
                n_scenes = len(items)
                threshold_used = FALLBACK_CLOUD_MAX

            if n_scenes < MIN_SCENES:
                from datetime import datetime, timedelta
                s = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=FALLBACK_EXPAND_DAYS)).strftime("%Y-%m-%d")
                e = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=FALLBACK_EXPAND_DAYS)).strftime("%Y-%m-%d")
                items = catalog.search(
                    collections=["sentinel-2-l2a"], bbox=NUREMBERG_BBOX,
                    datetime=f"{s}/{e}",
                    query={"eo:cloud_cover": {"lt": FALLBACK_CLOUD_MAX}},
                ).item_collection()
                n_scenes = len(items)

            if n_scenes == 0:
                print(f"  [{year}/{season}] WARNING: No scenes -- skip")
                continue

            print(f"  [{year}/{season}] {n_scenes} scenes (cloud<{threshold_used}%), stacking...")
            spectral = stackstac.stack(
                items, assets=SENTINEL_BANDS, bounds_latlon=NUREMBERG_BBOX,
                resolution=expected_res, epsg=target_epsg, dtype="float64",
                fill_value=np.nan, resampling=Resampling.bilinear,
                chunksize=1024, rescale=False,
            )
            scl = stackstac.stack(
                items, assets=["SCL"], bounds_latlon=NUREMBERG_BBOX,
                resolution=expected_res, epsg=target_epsg, dtype="float64",
                fill_value=np.nan, resampling=Resampling.nearest,
                chunksize=1024, rescale=False,
            ).sel(band="SCL")
            spectral, scl = xr.align(spectral, scl, join="exact")
            spectral = spectral.sel(band=SENTINEL_BANDS)

            import dask.array as da
            is_finite = da.isfinite(scl.data)
            valid = xr.DataArray(is_finite, coords=scl.coords, dims=scl.dims)
            for cls in SCL_EXCLUDE:
                valid = valid & (scl != cls)

            valid_fraction_xr = valid.mean(dim="time").astype("float32")
            composite_xr = spectral.where(valid).median(dim="time", skipna=True).astype("float32")

            print(f"  [{year}/{season}] Computing median...")
            composite = composite_xr.compute().values
            valid_fraction = valid_fraction_xr.compute().values

            src_transform = transform_from_xy_centers(
                composite_xr.coords["x"].values,
                composite_xr.coords["y"].values,
                expected_res=expected_res,
            )
            src_crs = CRS.from_epsg(target_epsg)

            n_spectral = len(SENTINEL_BANDS)
            comp_clean = np.where(np.isnan(composite), NODATA, composite).astype(np.float32)
            vf_clean = np.where(np.isnan(valid_fraction), NODATA, valid_fraction).astype(np.float32)

            warped = np.full((n_spectral, dst_height, dst_width), NODATA, dtype=np.float32)
            for i in range(n_spectral):
                reproject(
                    source=comp_clean[i], destination=warped[i],
                    src_transform=src_transform, src_crs=src_crs,
                    dst_transform=dst_transform, dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=NODATA, dst_nodata=NODATA,
                )

            vf_warped = np.full((dst_height, dst_width), NODATA, dtype=np.float32)
            reproject(
                source=vf_clean, destination=vf_warped,
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=NODATA, dst_nodata=NODATA,
            )
            vf_mask = vf_warped != NODATA
            vf_warped[vf_mask] = np.clip(vf_warped[vf_mask], 0.0, 1.0)

            n_output_bands = n_spectral + 1
            band_names = SENTINEL_BANDS + ["VALID_FRACTION"]
            with rasterio.open(
                path, "w", driver="GTiff", height=dst_height, width=dst_width,
                count=n_output_bands, dtype="float32", crs=dst_crs,
                transform=dst_transform, compress="lzw", nodata=NODATA,
            ) as dst:
                for i in range(n_spectral):
                    dst.write(warped[i], i + 1)
                    dst.set_band_description(i + 1, band_names[i])
                dst.write(vf_warped, n_output_bands)
                dst.set_band_description(n_output_bands, "VALID_FRACTION")

            with rasterio.open(path) as out, rasterio.open(GRID_REF_PATH) as ref:
                assert out.crs == ref.crs and out.width == ref.width and out.height == ref.height

            mb = os.path.getsize(path) / 1e6
            with rasterio.open(path) as ds:
                b1 = ds.read(1)
                v = b1[(b1 > -9000) & (b1 > 0)]
                p50 = np.median(v) if len(v) > 0 else -1
            print(f"  [{ts()}] [{year}/{season}] Saved ({mb:.1f} MB, B1_p50={p50:.0f})")


# ============================================================
# SAR DOWNLOAD (from run_multi_city_pipeline_v5.py)
# ============================================================
def download_sar():
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    print("\n" + "=" * 60)
    print(f"[{ts()}] Sentinel-1 SAR Download")
    print("=" * 60)

    with rasterio.open(GRID_REF_PATH) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="1",
        VSI_CACHE=True,
        VSI_CACHE_SIZE=5000000,
    )

    for year in ALL_YEARS:
        for season in SEASON_ORDER:
            path = os.path.join(OUT_DIR, f"sentinel1_nuremberg_{year}_{season}.tif")
            if os.path.exists(path):
                mb = os.path.getsize(path) / 1e6
                print(f"  [SAR {year}/{season}] Exists ({mb:.1f} MB) -- skip")
                continue

            start_date = f"{year}-{SAR_SEASON_DATES[season][0]}"
            end_date = f"{year}-{SAR_SEASON_DATES[season][1]}"

            items = catalog.search(
                collections=["sentinel-1-grd"],
                bbox=NUREMBERG_BBOX,
                datetime=f"{start_date}/{end_date}",
                query={
                    "sar:instrument_mode": {"eq": "IW"},
                    "sat:orbit_state": {"eq": "ascending"},
                },
            ).item_collection()

            if len(items) < 3:
                items_both = catalog.search(
                    collections=["sentinel-1-grd"],
                    bbox=NUREMBERG_BBOX,
                    datetime=f"{start_date}/{end_date}",
                    query={"sar:instrument_mode": {"eq": "IW"}},
                ).item_collection()
                if len(items_both) > len(items):
                    items = items_both

            n_scenes = len(items)
            if n_scenes == 0:
                print(f"  [SAR {year}/{season}] WARNING: No scenes!")
                continue

            print(f"  [{ts()}] [SAR {year}/{season}] {n_scenes} scenes, reading...")

            all_vv, all_vh = [], []
            with env:
                for i, item in enumerate(items):
                    vv_asset = item.assets.get("vv")
                    vh_asset = item.assets.get("vh")
                    if not vv_asset or not vh_asset:
                        continue
                    try:
                        with rasterio.open(vv_asset.href) as src:
                            gcp_crs = src.gcps[1] if src.gcps[0] else src.crs
                            with WarpedVRT(src, src_crs=gcp_crs, crs=gcp_crs) as vrt_4326:
                                with WarpedVRT(
                                    vrt_4326, crs=dst_crs, transform=dst_transform,
                                    width=dst_width, height=dst_height,
                                    resampling=Resampling.bilinear,
                                ) as vrt:
                                    vv_data = vrt.read(1).astype(np.float32)

                        with rasterio.open(vh_asset.href) as src:
                            gcp_crs = src.gcps[1] if src.gcps[0] else src.crs
                            with WarpedVRT(src, src_crs=gcp_crs, crs=gcp_crs) as vrt_4326:
                                with WarpedVRT(
                                    vrt_4326, crs=dst_crs, transform=dst_transform,
                                    width=dst_width, height=dst_height,
                                    resampling=Resampling.bilinear,
                                ) as vrt:
                                    vh_data = vrt.read(1).astype(np.float32)

                        vv_data[vv_data == 0] = np.nan
                        vh_data[vh_data == 0] = np.nan
                        all_vv.append(vv_data)
                        all_vh.append(vh_data)

                        if (i + 1) % 5 == 0 or i == n_scenes - 1:
                            print(f"    Read {i+1}/{n_scenes}")
                    except Exception as e:
                        print(f"    Scene {i} failed: {e}")

            if not all_vv:
                print(f"  [SAR {year}/{season}] ERROR: No readable scenes!")
                continue

            print(f"  [SAR {year}/{season}] Median from {len(all_vv)} scenes...")
            vv_med = np.nanmedian(np.stack(all_vv), axis=0)
            vh_med = np.nanmedian(np.stack(all_vh), axis=0)
            del all_vv, all_vh

            composite = np.stack([vv_med, vh_med], axis=0)
            MAX_DN = 2000.0
            valid_mask = np.isfinite(composite) & (composite > 0)
            scaled = np.full_like(composite, SAR_NODATA)
            scaled[valid_mask] = np.clip(composite[valid_mask] / MAX_DN, 0.0, 1.0)

            with rasterio.open(
                path, "w", driver="GTiff", height=dst_height, width=dst_width,
                count=2, dtype="float32", crs=dst_crs,
                transform=dst_transform, compress="lzw", nodata=SAR_NODATA,
            ) as dst:
                dst.write(scaled[0], 1)
                dst.write(scaled[1], 2)

            mb = os.path.getsize(path) / 1e6
            print(f"  [{ts()}] [SAR {year}/{season}] Saved ({mb:.1f} MB)")


# ============================================================
# BOA OFFSET FIX (for 2022+ S2)
# ============================================================
def apply_offset_fix():
    import rasterio
    import shutil

    print("\n" + "=" * 60)
    print(f"[{ts()}] BOA_ADD_OFFSET correction (2022+)")
    print("=" * 60)

    OFFSET = 1000
    n_spectral = len(SENTINEL_BANDS)

    for year in range(2022, 2026):
        for season in SEASON_ORDER:
            path = os.path.join(OUT_DIR, f"sentinel2_nuremberg_{year}_{season}.tif")
            if not os.path.exists(path):
                continue

            with rasterio.open(path) as ds:
                tags = ds.tags()
                if tags.get("PB0400_OFFSET_CORRECTED") == "true":
                    print(f"  {year}/{season}: Already corrected -- skip")
                    continue
                nodata = ds.nodata
                b02 = ds.read(1).astype(np.float64)
                if nodata is not None:
                    b02[b02 == nodata] = np.nan
                finite = b02[np.isfinite(b02)]
                if len(finite) == 0:
                    continue
                p5 = np.percentile(finite, 5)

            if p5 <= 800:
                print(f"  {year}/{season}: p5={p5:.0f} <= 800, OK")
                continue

            with rasterio.open(path) as ds:
                profile = ds.profile.copy()
                data = ds.read()
                tags = ds.tags()
                band_descs = [ds.descriptions[i] for i in range(ds.count)]
                nodata = ds.nodata

            before = np.nanmedian(data[0][data[0] != nodata]) if nodata else np.nanmedian(data[0])
            for i in range(n_spectral):
                band = data[i].astype(np.float32)
                mask = band != nodata if nodata is not None else np.ones_like(band, dtype=bool)
                band[mask] = np.maximum(band[mask] - OFFSET, 0)
                data[i] = band

            tmp = path + ".tmp"
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(data)
                for i in range(ds.count):
                    if band_descs[i]:
                        dst.set_band_description(i + 1, band_descs[i])
                tags["PB0400_OFFSET_CORRECTED"] = "true"
                dst.update_tags(**tags)
            shutil.move(tmp, path)

            with rasterio.open(path) as ds2:
                b02 = ds2.read(1)
                after = np.nanmedian(b02[b02 != ds2.nodata]) if ds2.nodata else np.nanmedian(b02)
            print(f"  {year}/{season}: B02 p50 {before:.0f} -> {after:.0f}")


if __name__ == "__main__":
    t0 = time.time()
    download_s2()
    download_sar()
    apply_offset_fix()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"Output: {OUT_DIR}")
    print("Next: terrapulse pipeline --skip-download")
