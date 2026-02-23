#!/usr/bin/env python3
"""Analyze LUCAS polygon/geometry data."""
import os, sys, zipfile
import geopandas as gpd

LUCAS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lucas")

# === 1) Copernicus 2022 Polygons ===
print("=" * 70)
print("1) COPERNICUS 2022 POLYGONS (gpkg)")
print("=" * 70)

gpkg = os.path.join(LUCAS_DIR, "l2022_copernicus_polygons.gpkg")
fsize = os.path.getsize(gpkg) / 1e6
print("  File: {:.0f} MB".format(fsize))

try:
    import pyogrio
    info = pyogrio.read_info(gpkg)
    print("  Total features: {:,}".format(info["features"]))
except Exception:
    pass

gdf = gpd.read_file(gpkg, rows=10)
print("  Columns: {}".format(len(gdf.columns)))
print("  CRS: {}".format(gdf.crs))
print("  Geom types: {}".format(gdf.geometry.geom_type.unique()))

# Key columns
key_cols = [c for c in gdf.columns if any(k in c.upper()
            for k in ["LC1", "LC2", "LU1", "AREA", "PERC", "RADIUS", "POINT_ID"])]
print("  Key columns: {}".format(key_cols))

# Polygon sizes
gdf_m = gdf.to_crs(epsg=3035)
areas = gdf_m.geometry.area
print("  Polygon areas (m2): min={:.0f}, mean={:.0f}, max={:.0f}".format(
    areas.min(), areas.mean(), areas.max()))
import math
print("  Approx radii: min={:.0f}m, mean={:.0f}m, max={:.0f}m".format(
    math.sqrt(areas.min()/3.14), math.sqrt(areas.mean()/3.14),
    math.sqrt(areas.max()/3.14)))

# Show sample
print("\n  Sample record:")
for c in ["POINT_ID", "SURVEY_LC1", "SURVEY_LC1_PERC", "SURVEY_LC2",
           "SURVEY_PARCEL_AREA_HA"]:
    if c in gdf.columns:
        print("    {}: {}".format(c, gdf[c].iloc[0]))

# === 2) Harmonised GPS Geometry ===
print("\n" + "=" * 70)
print("2) LUCAS HARMONISED GPS GEOMETRY")
print("=" * 70)

gps_zip = os.path.join(LUCAS_DIR, "LUCAS_gps_geom.zip")
if os.path.exists(gps_zip):
    zf = zipfile.ZipFile(gps_zip)
    files = zf.namelist()
    print("  Files in zip ({})".format(len(files)))
    for f in files[:10]:
        print("    {}".format(f))

    # Try reading directly from zip
    shps = [f for f in files if f.endswith(".shp")]
    gpkgs = [f for f in files if f.endswith(".gpkg")]
    csvs = [f for f in files if f.endswith(".csv")]
    print("  Shapefiles: {}, GPKGs: {}, CSVs: {}".format(len(shps), len(gpkgs), len(csvs)))

    if gpkgs:
        target = gpkgs[0]
    elif shps:
        target = shps[0]
    else:
        target = None

    if target:
        print("  Reading: {}".format(target))
        gdf2 = gpd.read_file("zip://" + os.path.abspath(gps_zip) + "!" + target, rows=5)
        print("  Columns: {}".format(list(gdf2.columns)[:15]))
        print("  CRS: {}".format(gdf2.crs))
        print("  Geom types: {}".format(gdf2.geometry.geom_type.unique()))
        print("  Sample:")
        print(gdf2.head(2).to_string())

# === 3) Theoretical Geometry ===
print("\n" + "=" * 70)
print("3) LUCAS THEORETICAL GEOMETRY")
print("=" * 70)

th_zip = os.path.join(LUCAS_DIR, "LUCAS_th_geom.zip")
if os.path.exists(th_zip):
    zf = zipfile.ZipFile(th_zip)
    files = zf.namelist()
    print("  Files in zip ({})".format(len(files)))
    for f in files[:10]:
        print("    {}".format(f))

    gpkgs = [f for f in files if f.endswith(".gpkg")]
    shps = [f for f in files if f.endswith(".shp")]
    print("  Shapefiles: {}, GPKGs: {}".format(len(shps), len(gpkgs)))

    if gpkgs:
        target = gpkgs[0]
    elif shps:
        target = shps[0]
    else:
        target = None

    if target:
        print("  Reading: {}".format(target))
        gdf3 = gpd.read_file("zip://" + os.path.abspath(th_zip) + "!" + target, rows=5)
        print("  Columns: {}".format(list(gdf3.columns)[:15]))
        print("  CRS: {}".format(gdf3.crs))
        print("  Geom types: {}".format(gdf3.geometry.geom_type.unique()))

print("\nDone.")
