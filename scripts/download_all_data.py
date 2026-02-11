"""
Download all project data for 'Mapping Urban Change in Nuremberg with ML'.

This script downloads ALL required data in one go:
  1. ESA WorldCover 2020 & 2021 (Map + InputQuality layers)
  2. Sentinel-2 L2A seasonal composites (spring/summer/autumn x 2020-2021)
     with SCL cloud/shadow masking and valid_fraction quality band
  3. OpenStreetMap features (buildings, roads, land-use, water, natural)

Usage:
    pip install -r requirements.txt
    python scripts/download_all_data.py

Runtime: ~30-45 minutes (depends on internet speed and API load)
Disk space required: ~1.2 GB
"""

import os
import sys
import urllib.request
import warnings

import numpy as np

# Suppress noisy stackstac/dask warnings only
warnings.filterwarnings("ignore", module="stackstac")
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

# -- Load config ---------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CFG,
    GRID_REF_PATH,
    LABELS_DIR,
    OSM_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RAW_V2_DIR,
)

NUREMBERG_BBOX = CFG["aoi"]["bbox"]
SENTINEL_BANDS = CFG["sentinel2"]["bands"]
SENTINEL_YEARS = CFG["sentinel2"]["years"]
SEASON_ORDER = CFG["sentinel2"]["season_order"]
SEASONS = CFG["sentinel2"]["seasons"]
CLOUD_MAX = CFG["sentinel2"]["cloud_cover_max"]
MIN_SCENES = CFG["sentinel2"]["min_scenes"]
FALLBACK_CLOUD_MAX = CFG["sentinel2"]["fallback_cloud_max"]
FALLBACK_EXPAND_DAYS = CFG["sentinel2"]["fallback_window_expand_days"]
SCL_EXCLUDE = CFG["scl_mask"]["exclude_classes"]
NODATA = CFG["sentinel2"]["nodata"]
WC_TILE = CFG["worldcover"]["tile"]
WC_YEARS = CFG["worldcover"]["years"]
WC_LAYERS = CFG["worldcover"]["layers"]


def setup_dirs():
    """Create data directory structure."""
    for d in [RAW_DIR, RAW_V2_DIR, LABELS_DIR, OSM_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Directory structure created.")


def download_worldcover():
    """Download ESA WorldCover Map + InputQuality tiles for Nuremberg."""
    print("\n" + "=" * 60)
    print("STEP 1/3: ESA WorldCover Labels")
    print("=" * 60)

    url_templates = {
        2020: {
            "Map": f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{WC_TILE}_Map.tif",
            "InputQuality": f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{WC_TILE}_InputQuality.tif",
        },
        2021: {
            "Map": f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{WC_TILE}_Map.tif",
            "InputQuality": f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{WC_TILE}_InputQuality.tif",
        },
    }

    for year in WC_YEARS:
        for layer in WC_LAYERS:
            filename = f"ESA_WorldCover_{year}_{WC_TILE}_{layer}.tif"
            path = os.path.join(LABELS_DIR, filename)

            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  [{year}/{layer}] Already exists: {filename} ({size_mb:.1f} MB) -- skipping")
                continue

            url = url_templates[year][layer]
            print(f"  [{year}/{layer}] Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, path)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  [{year}/{layer}] Saved: {filename} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  [{year}/{layer}] WARNING: Download failed ({e})")
                print(f"  [{year}/{layer}] TODO: manual download required from https://esa-worldcover.org/en/data-access")
                if os.path.exists(path):
                    os.remove(path)


def transform_from_xy_centers(xs, ys, expected_res):
    """Compute affine transform from xarray center coordinates.

    xarray/stackstac coords are pixel CENTERS, not edges. This helper
    applies the half-pixel correction needed to derive the correct
    upper-left origin and validates that pixel spacing matches the
    expected resolution.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    assert xs.size > 1 and ys.size > 1, "Need at least 2 coords in each axis"

    rx = float(np.abs(xs[1] - xs[0]))
    ry = float(np.abs(ys[1] - ys[0]))
    assert abs(rx - expected_res) < 1e-3, f"x spacing {rx} != {expected_res}"
    assert abs(ry - expected_res) < 1e-3, f"y spacing {ry} != {expected_res}"

    import rasterio
    left = float(xs.min()) - rx / 2
    right = float(xs.max()) + rx / 2
    bottom = float(ys.min()) - ry / 2
    top = float(ys.max()) + ry / 2
    return rasterio.transform.from_bounds(left, bottom, right, top, len(xs), len(ys))


def download_sentinel2():
    """Download Sentinel-2 L2A seasonal composites with SCL masking.

    All composites are warped to the canonical anchor grid before writing,
    guaranteeing identical CRS/transform/shape across all seasons and years.

    Implementation:
    - Two separate stacks: spectral (bilinear) + SCL (nearest)
    - SCL NaN → invalid (np.isfinite gate)
    - Lazy xarray until final .compute()
    - Warp to anchor grid via rasterio.warp.reproject
    - 11-band output: bands 1-10 spectral, band 11 = VALID_FRACTION
    - Hard assertion: output geometry == anchor geometry
    """
    import planetary_computer
    import pystac_client
    import rasterio
    import stackstac
    import xarray as xr
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    print("\n" + "=" * 60)
    print("STEP 2/3: Sentinel-2 Seasonal Composites (via Planetary Computer)")
    print("=" * 60)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    print("  Connected to Planetary Computer STAC API")

    target_epsg = CFG["aoi"]["epsg"]
    expected_res = float(CFG["sentinel2"]["resolution"])
    spatial_chunksize = 1024  # spatial chunk size for dask

    # -- Read anchor spec (C1) -------------------------------------------------
    assert os.path.exists(GRID_REF_PATH), (
        f"Missing anchor: {GRID_REF_PATH}. Run scripts/create_anchor.py first."
    )
    with rasterio.open(GRID_REF_PATH) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
    print(f"  Anchor: {dst_width}x{dst_height} @ {dst_transform.a}m, CRS={dst_crs}")

    for year in SENTINEL_YEARS:
        for season in SEASON_ORDER:
            path = os.path.join(RAW_V2_DIR, f"sentinel2_nuremberg_{year}_{season}.tif")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  [{year}/{season}] Already exists ({size_mb:.1f} MB) -- skipping")
                continue

            # -- Determine date range and cloud threshold --
            start_date = f"{year}-{SEASONS[season][0]}"
            end_date = f"{year}-{SEASONS[season][1]}"
            cloud_max = CLOUD_MAX[season]

            print(f"\n  [{year}/{season}] Searching (cloud < {cloud_max}%, {start_date} to {end_date})...")
            items = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=NUREMBERG_BBOX,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": cloud_max}},
            ).item_collection()

            n_scenes = len(items)
            threshold_used = cloud_max
            date_range_used = f"{start_date}/{end_date}"

            # -- Fallback: relax cloud threshold --
            if n_scenes < MIN_SCENES:
                print(f"  [{year}/{season}] Only {n_scenes} scenes (< {MIN_SCENES}), relaxing to cloud < {FALLBACK_CLOUD_MAX}%...")
                items = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=NUREMBERG_BBOX,
                    datetime=f"{start_date}/{end_date}",
                    query={"eo:cloud_cover": {"lt": FALLBACK_CLOUD_MAX}},
                ).item_collection()
                n_scenes = len(items)
                threshold_used = FALLBACK_CLOUD_MAX

            # -- Fallback: widen date window --
            if n_scenes < MIN_SCENES:
                from datetime import datetime, timedelta

                orig_start = datetime.strptime(start_date, "%Y-%m-%d")
                orig_end = datetime.strptime(end_date, "%Y-%m-%d")
                expanded_start = (orig_start - timedelta(days=FALLBACK_EXPAND_DAYS)).strftime("%Y-%m-%d")
                expanded_end = (orig_end + timedelta(days=FALLBACK_EXPAND_DAYS)).strftime("%Y-%m-%d")

                print(f"  [{year}/{season}] Still only {n_scenes} scenes, expanding window to {expanded_start} .. {expanded_end}...")
                items = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=NUREMBERG_BBOX,
                    datetime=f"{expanded_start}/{expanded_end}",
                    query={"eo:cloud_cover": {"lt": FALLBACK_CLOUD_MAX}},
                ).item_collection()
                n_scenes = len(items)
                date_range_used = f"{expanded_start}/{expanded_end}"

            if n_scenes == 0:
                print(f"  [{year}/{season}] WARNING: No scenes found -- skipping")
                continue

            scenes_below_min = n_scenes < MIN_SCENES
            if scenes_below_min:
                print(f"  [{year}/{season}] WARNING: Only {n_scenes} scenes (< MIN_SCENES={MIN_SCENES}) after all fallbacks!")

            print(f"  [{year}/{season}] Found {n_scenes} scenes (threshold={threshold_used}%, range={date_range_used})")
            print(f"  [{year}/{season}] Stacking...")

            # -- Two separate stacks: bilinear for spectral, nearest for SCL --
            spectral = stackstac.stack(
                items,
                assets=SENTINEL_BANDS,
                bounds_latlon=NUREMBERG_BBOX,
                resolution=expected_res,
                epsg=target_epsg,
                dtype="float64",
                fill_value=np.nan,
                resampling=Resampling.bilinear,
                chunksize=spatial_chunksize,
                rescale=False,
            )

            scl = stackstac.stack(
                items,
                assets=["SCL"],
                bounds_latlon=NUREMBERG_BBOX,
                resolution=expected_res,
                epsg=target_epsg,
                dtype="float64",
                fill_value=np.nan,
                resampling=Resampling.nearest,
                chunksize=spatial_chunksize,
                rescale=False,
            ).sel(band="SCL")

            # A1: Align exact — never shrink silently
            spectral, scl = xr.align(spectral, scl, join="exact")
            assert spectral.sizes["x"] == scl.sizes["x"], "x dimension mismatch after align"
            assert spectral.sizes["y"] == scl.sizes["y"], "y dimension mismatch after align"
            spectral = spectral.sel(band=SENTINEL_BANDS)

            n_total = int(scl.sizes["time"])

            # -- SCL validity mask (lazy, dask-aware) --
            import dask.array as da
            scl_vals = scl.data  # dask array
            is_finite = da.isfinite(scl_vals)
            valid = xr.DataArray(is_finite, coords=scl.coords, dims=scl.dims)
            for cls in SCL_EXCLUDE:
                valid = valid & (scl != cls)

            # -- Lazy composite + valid_fraction --
            valid_fraction_xr = valid.mean(dim="time").astype("float32")
            composite_xr = spectral.where(valid).median(dim="time", skipna=True).astype("float32")

            print(f"  [{year}/{season}] Computing masked median (this may take a few minutes)...")
            composite = composite_xr.compute().values       # (band, y, x)
            valid_fraction = valid_fraction_xr.compute().values  # (y, x)

            # -- A2: Compute source transform with half-pixel correction --
            src_transform = transform_from_xy_centers(
                composite_xr.coords["x"].values,
                composite_xr.coords["y"].values,
                expected_res=expected_res,
            )
            src_crs = CRS.from_epsg(target_epsg)

            # -- C: Warp composite + valid_fraction to anchor grid --
            # Convert NaN -> NODATA before reprojection (deterministic GDAL behavior)
            print(f"  [{year}/{season}] Warping to anchor grid ({dst_width}x{dst_height})...")
            n_spectral = len(SENTINEL_BANDS)

            comp_clean = np.where(np.isnan(composite), NODATA, composite).astype(np.float32)
            vf_clean = np.where(np.isnan(valid_fraction), NODATA, valid_fraction).astype(np.float32)

            warped = np.full((n_spectral, dst_height, dst_width), NODATA, dtype=np.float32)
            for i in range(n_spectral):
                reproject(
                    source=comp_clean[i],
                    destination=warped[i],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=NODATA,
                    dst_nodata=NODATA,
                )

            vf_warped = np.full((dst_height, dst_width), NODATA, dtype=np.float32)
            reproject(
                source=vf_clean,
                destination=vf_warped,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=NODATA,
                dst_nodata=NODATA,
            )
            # Clamp valid_fraction to [0,1] (bilinear can overshoot near edges)
            vf_mask = vf_warped != NODATA
            vf_warped[vf_mask] = np.clip(vf_warped[vf_mask], 0.0, 1.0)

            # -- C4: Write GeoTIFF with EXACT anchor geometry --
            n_output_bands = n_spectral + 1
            band_names = SENTINEL_BANDS + ["VALID_FRACTION"]

            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=dst_height,
                width=dst_width,
                count=n_output_bands,
                dtype="float32",
                crs=dst_crs,
                transform=dst_transform,
                compress="lzw",
                nodata=NODATA,
            ) as dst:
                for i in range(n_spectral):
                    dst.write(warped[i], i + 1)
                    dst.set_band_description(i + 1, band_names[i])

                dst.write(vf_warped, n_output_bands)
                dst.set_band_description(n_output_bands, "VALID_FRACTION")

                dst.update_tags(
                    N_SCENES_TOTAL=str(n_total),
                    SCENES_BELOW_MIN="1" if scenes_below_min else "0",
                    DATE_RANGE=date_range_used,
                    CLOUD_THRESHOLD=str(threshold_used),
                    SEASON=season,
                    YEAR=str(year),
                    MIN_SCENES=str(MIN_SCENES),
                    FALLBACK_CLOUD_MAX=str(FALLBACK_CLOUD_MAX),
                    FALLBACK_EXPAND_DAYS=str(FALLBACK_EXPAND_DAYS),
                    SCL_EXCLUDE=str(SCL_EXCLUDE),
                    RESAMPLING_SPECTRAL="bilinear",
                    RESAMPLING_SCL="nearest",
                )

            # -- C5: Hard assertion — output must match anchor exactly --
            with rasterio.open(path) as out, rasterio.open(GRID_REF_PATH) as ref:
                assert out.crs == ref.crs, f"CRS mismatch: {out.crs} != {ref.crs}"
                assert out.transform == ref.transform, f"Transform mismatch: {out.transform} != {ref.transform}"
                assert out.width == ref.width, f"Width mismatch: {out.width} != {ref.width}"
                assert out.height == ref.height, f"Height mismatch: {out.height} != {ref.height}"

            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [{year}/{season}] Saved: {os.path.basename(path)} ({size_mb:.1f} MB)")
            print(f"    n_scenes_total={n_total}, threshold={threshold_used}%, range={date_range_used}")
            print(f"    Geometry verified: matches anchor exactly")


def download_osm():
    """Download OpenStreetMap features for Nuremberg."""
    import geopandas as gpd
    import osmnx as ox

    print("\n" + "=" * 60)
    print("STEP 3/3: OpenStreetMap Features")
    print("=" * 60)

    PLACE = "Nuremberg, Germany"

    # Buildings
    bpath = os.path.join(OSM_DIR, "buildings.gpkg")
    if os.path.exists(bpath) and os.path.getsize(bpath) > 1_000_000:
        print("  [buildings] Already exists -- skipping")
    else:
        print("  [buildings] Downloading footprints...")
        buildings = ox.features_from_place(PLACE, tags={"building": True})
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
        cols = [
            c for c in ["geometry", "building", "building:levels", "name"] if c in buildings.columns
        ]
        buildings[cols].to_file(bpath, driver="GPKG")
        print(f"  [buildings] Saved {len(buildings)} buildings")

    # Roads
    rpath = os.path.join(OSM_DIR, "roads.gpkg")
    npath = os.path.join(OSM_DIR, "road_nodes.gpkg")
    if os.path.exists(rpath):
        print("  [roads] Already exists -- skipping")
    else:
        print("  [roads] Downloading road network...")
        G = ox.graph_from_place(PLACE, network_type="drive")
        nodes, edges = ox.graph_to_gdfs(G)
        edges.to_file(rpath, driver="GPKG")
        nodes.to_file(npath, driver="GPKG")
        print(f"  [roads] Saved {len(edges)} segments, {len(nodes)} intersections")

    # Land use
    lpath = os.path.join(OSM_DIR, "landuse.gpkg")
    if os.path.exists(lpath):
        print("  [landuse] Already exists -- skipping")
    else:
        print("  [landuse] Downloading land-use zones...")
        landuse = ox.features_from_place(PLACE, tags={"landuse": True})
        landuse = landuse[landuse.geometry.type.isin(["Polygon", "MultiPolygon"])]
        landuse.to_file(lpath, driver="GPKG")
        print(f"  [landuse] Saved {len(landuse)} polygons")

    # Natural features
    napath = os.path.join(OSM_DIR, "natural.gpkg")
    if os.path.exists(napath):
        print("  [natural] Already exists -- skipping")
    else:
        print("  [natural] Downloading natural features...")
        natural = ox.features_from_place(PLACE, tags={"natural": True})
        natural = natural[natural.geometry.type.isin(["Polygon", "MultiPolygon"])]
        natural.to_file(napath, driver="GPKG")
        print(f"  [natural] Saved {len(natural)} features")

    # Water
    wpath = os.path.join(OSM_DIR, "water.gpkg")
    if os.path.exists(wpath):
        print("  [water] Already exists -- skipping")
    else:
        print("  [water] Downloading water bodies...")
        water = ox.features_from_place(PLACE, tags={"water": True})
        water = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
        water.to_file(wpath, driver="GPKG")
        print(f"  [water] Saved {len(water)} features")


def print_summary():
    """Print summary of all downloaded data."""
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE -- DATA SUMMARY")
    print("=" * 60)

    for label, directory in [
        ("Labels", LABELS_DIR),
        ("Raw Imagery (v2)", RAW_V2_DIR),
        ("OSM Features", OSM_DIR),
    ]:
        print(f"\n  {label} ({directory}):")
        if os.path.exists(directory):
            for f in sorted(os.listdir(directory)):
                fp = os.path.join(directory, f)
                if os.path.isfile(fp):
                    print(f"    {f}: {os.path.getsize(fp) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    setup_dirs()
    download_worldcover()
    download_sentinel2()
    download_osm()
    print_summary()
