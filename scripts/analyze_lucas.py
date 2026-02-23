#!/usr/bin/env python3
"""Quick analysis of all three LUCAS data sources."""
import os, sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUCAS_DIR = os.path.join(PROJECT_ROOT, "data", "lucas")

# ===== 1) EU LUCAS 2022 Survey Points =====
print("=" * 70)
print("1) EU LUCAS 2022 SURVEY POINTS")
print("=" * 70)

csv_path = os.path.join(LUCAS_DIR, "EU_LUCAS_2022.csv")
df = pd.read_csv(csv_path, low_memory=False)
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Countries: {df['POINT_NUTS0'].nunique()}")

# Surveyed vs not
surveyed = df["SURVEY_LC1"].notna()
gps = df["SURVEY_GPS_LAT"].notna() & df["SURVEY_GPS_LONG"].notna()
has_point = df["POINT_LAT"].notna()
print(f"  Surveyed (have LC1): {surveyed.sum():,} ({surveyed.mean()*100:.1f}%)")
print(f"  With GPS coords: {gps.sum():,}")
print(f"  With POINT_LAT/LONG: {has_point.sum():,}")
print(f"  Surveyed + any coords: {(surveyed & (gps | has_point)).sum():,}")

# Country breakdown
print("\n  Surveyed points by country:")
s = df[surveyed].groupby("POINT_NUTS0").size().sort_values(ascending=False)
for c, n in s.items():
    print(f"    {c}: {n:>6,}")

# LC1 codes
print("\n  Top 20 SURVEY_LC1 codes:")
lc1 = df[surveyed]["SURVEY_LC1"].value_counts().head(20)
# LUCAS code mapping
code_map = {
    "A": "built_up", "B": "cropland", "C": "tree_cover",
    "D": "shrubland", "E": "grassland", "F": "bare_sparse",
    "G": "water", "H": "wetland"
}
for code, n in lc1.items():
    letter = code[0] if isinstance(code, str) else "?"
    wc_class = code_map.get(letter, "unknown")
    print(f"    {code:8s} -> {wc_class:15s}  {n:>6,}")

# Map to our 7 classes
print("\n  Mapped to our 7 WorldCover classes:")
df_s = df[surveyed].copy()
df_s["wc_class"] = df_s["SURVEY_LC1"].apply(
    lambda x: code_map.get(str(x)[0], "unknown") if pd.notna(x) else "unknown"
)
class_counts = df_s["wc_class"].value_counts()
for cls, n in class_counts.items():
    print(f"    {cls:20s} {n:>6,}")

# How many fall in our training tile bboxes?
print("\n  === Points falling in our training tile bboxes ===")
sys.path.insert(0, PROJECT_ROOT)
from scripts.run_multi_city_pipeline_v5 import CITIES

# Use POINT_LAT/LONG (always available) with fallback to GPS
lats = df_s["SURVEY_GPS_LAT"].fillna(df_s["POINT_LAT"])
lons = df_s["SURVEY_GPS_LONG"].fillna(df_s["POINT_LONG"])

total_hits = 0
city_hits = {}
for city in CITIES:
    w, s, e, n = city.bbox
    mask = (lons >= w) & (lons <= e) & (lats >= s) & (lats <= n)
    hits = mask.sum()
    if hits > 0:
        city_hits[city.name] = hits
        total_hits += hits

print(f"  Total LUCAS points in our tiles: {total_hits:,}")
print(f"  Cities with LUCAS points:")
for name, n in sorted(city_hits.items(), key=lambda x: -x[1]):
    print(f"    {name:30s} {n:>4}")

# Test cities specifically
print("\n  === LUCAS points in VALIDATION cities ===")
test_cities = [c for c in CITIES if c.is_test]
for city in test_cities:
    w, s, e, n = city.bbox
    mask = (lons >= w) & (lons <= e) & (lats >= s) & (lats <= n)
    hits = mask.sum()
    print(f"    {city.name:20s} {hits:>4} points")

# DE data comparison
print("\n  === DE subset comparison ===")
de_csv = os.path.join(PROJECT_ROOT, "DE_LUCAS_2022.csv")
if os.path.exists(de_csv):
    de = pd.read_csv(de_csv, low_memory=False)
    eu_de = df[df["POINT_NUTS0"] == "DE"]
    print(f"  DE file rows: {len(de):,}")
    print(f"  EU file DE rows: {len(eu_de):,}")
    print(f"  Same data: {'YES' if len(de) == len(eu_de) else 'DIFFERENT'}")

print("\nDone.")
