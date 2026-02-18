#!/usr/bin/env python3
"""
V5 Pan-European Overnight Pipeline
===================================
Full pipeline for 56 cities: anchors → S2 download → SAR download (parallel)
→ WorldCover labels → delete stale parquets → feature extraction → MLP sweep.

Usage:
    .venv/Scripts/python.exe -u scripts/run_overnight_v5_sar.py
"""
import os, sys, time, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    CITIES, TRAIN_CITIES, TEST_CITIES, SEASONS, ALL_YEARS,
    stage_anchors, stage_download, stage_download_sar, stage_extract,
    stage_labels,
    download_sar_season, ensure_all_dirs,
    city_features_dir, city_raw_dir, ts,
)

LOG_PATH = os.path.join(PROJECT_ROOT, "data", "cities", "overnight_v5_sar.log")

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# =====================================================================
# STEP 0: Setup
# =====================================================================
log("=" * 70)
log("OVERNIGHT V5 PAN-EU PIPELINE START")
log("=" * 70)
log(f"Total cities: {len(CITIES)}")
log(f"Training: {len(TRAIN_CITIES)} ({[c.name for c in TRAIN_CITIES]})")
log(f"Test: {len(TEST_CITIES)} ({[c.name for c in TEST_CITIES]})")

ensure_all_dirs()
stage_anchors()

# =====================================================================
# STEP 1a: Download S2 optical — PARALLELIZED (2 workers)
# Lower parallelism than SAR because stackstac/dask uses more RAM
# =====================================================================
log("")
log("STEP 1/6: DOWNLOADING SENTINEL-2 OPTICAL (2 parallel workers)")
log("-" * 40)

from scripts.run_multi_city_pipeline_v5 import download_season

s2_jobs = []
for city in CITIES:
    for year in ALL_YEARS:
        for season in SEASONS:
            s2_jobs.append((city, year, season))

# Count cached
s2_cached = 0
for city, year, season in s2_jobs:
    path = os.path.join(city_raw_dir(city),
                        f"sentinel2_{city.name}_{year}_{season}.tif")
    if os.path.exists(path):
        s2_cached += 1
log(f"Total S2 jobs: {len(s2_jobs)}, cached: {s2_cached}, "
    f"remaining: {len(s2_jobs) - s2_cached}")

s2_done = 0
s2_fail = 0

def _download_s2_one(args):
    city, year, season = args
    for attempt in range(3):
        try:
            download_season(city, year, season)
            return (city.name, year, season, True, "")
        except Exception as e:
            if attempt < 2:
                time.sleep(30)
            else:
                return (city.name, year, season, False, str(e))

S2_WORKERS = 2
with ThreadPoolExecutor(max_workers=S2_WORKERS) as pool:
    futures = {pool.submit(_download_s2_one, job): job for job in s2_jobs}
    for future in as_completed(futures):
        name, year, season, ok, err = future.result()
        if ok:
            s2_done += 1
        else:
            s2_fail += 1
            log(f"  S2 FAILED: {name}/{year}/{season}: {err}")
        total_s2_done = s2_done + s2_fail
        if total_s2_done % 20 == 0 or total_s2_done == len(s2_jobs):
            log(f"  S2 Progress: {total_s2_done}/{len(s2_jobs)} "
                f"({s2_done} ok, {s2_fail} fail)")

log(f"S2 download complete: {s2_done} ok, {s2_fail} failed")

# =====================================================================
# STEP 1b: Download SAR TIFs — PARALLELIZED (4 workers)
# =====================================================================
log("")
log("STEP 2/6: DOWNLOADING SAR TIFs (4 parallel workers)")
log("-" * 40)

# Build job list
sar_jobs = []
for city in CITIES:
    for year in ALL_YEARS:
        for season in SEASONS:
            sar_jobs.append((city, year, season))

log(f"Total SAR jobs: {len(sar_jobs)}")

# Count already cached
cached = 0
for city, year, season in sar_jobs:
    path = os.path.join(city_raw_dir(city),
                        f"sentinel1_{city.name}_{year}_{season}.tif")
    if os.path.exists(path):
        cached += 1
log(f"Already cached: {cached}, remaining: {len(sar_jobs) - cached}")

done_count = 0
fail_count = 0

def _download_one(args):
    city, year, season = args
    try:
        download_sar_season(city, year, season)
        return (city.name, year, season, True, "")
    except Exception as e:
        return (city.name, year, season, False, str(e))

N_WORKERS = 4
with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(_download_one, job): job for job in sar_jobs}
    for future in as_completed(futures):
        name, year, season, ok, err = future.result()
        if ok:
            done_count += 1
        else:
            fail_count += 1
            log(f"  FAILED: {name}/{year}/{season}: {err}")
        # Progress every 20 completions
        total_done = done_count + fail_count
        if total_done % 20 == 0 or total_done == len(sar_jobs):
            log(f"  SAR Progress: {total_done}/{len(sar_jobs)} "
                f"({done_count} ok, {fail_count} fail)")

log(f"SAR download complete: {done_count} ok, {fail_count} failed "
    f"out of {len(sar_jobs)}")

# =====================================================================
# STEP 2: WorldCover labels (needed for training)
# =====================================================================
log("")
log("STEP 3/6: DOWNLOADING WORLDCOVER LABELS")
log("-" * 40)
stage_labels()

# =====================================================================
# STEP 3: Delete old feature parquets (force re-extraction with SAR)
# =====================================================================
log("")
log("STEP 4/6: DELETING OLD FEATURE PARQUETS")
log("-" * 40)
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
log(f"Deleted {deleted} old parquets")

# =====================================================================
# STEP 4: Re-extract features with SAR-aware Rust binary
# =====================================================================
log("")
log("STEP 5/6: RE-EXTRACTING FEATURES (WITH SAR)")
log("-" * 40)
stage_extract()

# Verify SAR columns exist
log("")
log("VERIFYING SAR COLUMNS...")
import pyarrow.parquet as pq
for city in CITIES:
    feat_dir = city_features_dir(city)
    pq_path = os.path.join(feat_dir, "features_rust_2020_2021.parquet")
    if not os.path.exists(pq_path):
        log(f"  [{city.name}] MISSING parquet!")
        continue
    schema = pq.read_schema(pq_path)
    all_cols = [f.name for f in schema]
    sar_cols = [c for c in all_cols if c.startswith("SAR")]
    log(f"  [{city.name}] {len(all_cols)} total cols, {len(sar_cols)} SAR cols")

# =====================================================================
# STEP 5: Run full MLP sweep with all features (including SAR)
# =====================================================================
log("")
log("STEP 6/6: RUNNING MLP ARCHITECTURE SWEEP")
log("-" * 40)

sweep_script = os.path.join(PROJECT_ROOT, "scripts", "sweep_mlp_v5.py")
python_exe = sys.executable

result = subprocess.run(
    [python_exe, "-u", sweep_script],
    cwd=PROJECT_ROOT,
    stdout=sys.stdout,
    stderr=sys.stderr,
)

log("")
log("=" * 70)
if result.returncode == 0:
    log("OVERNIGHT V5 PAN-EU PIPELINE COMPLETED SUCCESSFULLY")
else:
    log(f"SWEEP EXITED WITH CODE {result.returncode}")
log("=" * 70)
