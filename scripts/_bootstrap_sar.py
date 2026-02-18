#!/usr/bin/env python3
"""
Step 1: Download SAR TIFs for all cities
Step 2: Delete old feature parquets (force re-extraction with SAR)
Step 3: Re-extract features using updated Rust binary
Step 4: Verify SAR columns exist in parquets
"""
import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    CITIES, SEASONS, ALL_YEARS,
    stage_anchors, stage_download_sar, stage_extract,
    city_features_dir, ts,
)

print(f"[{ts()}] === V5 SAR BOOTSTRAP ===")
print(f"  Cities: {len(CITIES)}")
print(f"  Years: {ALL_YEARS}")
print(f"  Seasons: {SEASONS}")

# Step 0: Ensure anchors exist
stage_anchors()

# Step 1: Download all SAR TIFs
print(f"\n[{ts()}] STEP 1: Downloading SAR TIFs...")
stage_download_sar()

# Step 2: Delete old feature parquets to force re-extraction
print(f"\n[{ts()}] STEP 2: Deleting old feature parquets...")
deleted = 0
for city in CITIES:
    feat_dir = city_features_dir(city)
    if not os.path.exists(feat_dir):
        continue
    for f in os.listdir(feat_dir):
        if f.startswith("features_rust_") and f.endswith(".parquet"):
            path = os.path.join(feat_dir, f)
            os.remove(path)
            deleted += 1
            print(f"  Deleted: {city.name}/{f}")
print(f"  Deleted {deleted} old parquets")

# Step 3: Re-extract with SAR-aware Rust binary
print(f"\n[{ts()}] STEP 3: Re-extracting features (with SAR)...")
stage_extract()

# Step 4: Verify SAR columns
print(f"\n[{ts()}] STEP 4: Verifying SAR columns in parquets...")
import pyarrow.parquet as pq
for city in CITIES:
    feat_dir = city_features_dir(city)
    pq_path = os.path.join(feat_dir, "features_rust_2020_2021.parquet")
    if not os.path.exists(pq_path):
        print(f"  [{city.name}] MISSING parquet!")
        continue
    schema = pq.read_schema(pq_path)
    all_cols = [f.name for f in schema]
    sar_cols = [c for c in all_cols if c.startswith("SAR")]
    print(f"  [{city.name}] {len(all_cols)} total cols, {len(sar_cols)} SAR cols")

print(f"\n[{ts()}] === DONE ===")
