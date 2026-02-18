#!/usr/bin/env python3
"""Re-download SAR for cities that are missing SAR files.
Runs with 2 workers (lower than usual to leave bandwidth for sweep).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.run_multi_city_pipeline_v5 import (
    CITIES, ALL_YEARS, SEASONS, city_raw_dir, download_sar_season,
)

def ts():
    return time.strftime("%H:%M:%S")

# Find cities missing SAR
missing_cities = []
for city in CITIES:
    raw = city_raw_dir(city)
    if not os.path.exists(raw):
        missing_cities.append(city)
        continue
    sar = [f for f in os.listdir(raw) if f.startswith("sentinel1_")]
    if len(sar) < 18:
        missing_cities.append(city)

print(f"[{ts()}] SAR re-download for {len(missing_cities)} cities")

# Build job list (only missing files)
jobs = []
for city in missing_cities:
    raw = city_raw_dir(city)
    for year in ALL_YEARS:
        for season in SEASONS:
            path = os.path.join(raw, f"sentinel1_{city.name}_{year}_{season}.tif")
            if not os.path.exists(path):
                jobs.append((city, year, season))

print(f"  Total missing SAR files: {len(jobs)}")

done = 0
fail = 0

def _dl(args):
    city, year, season = args
    for attempt in range(3):
        try:
            download_sar_season(city, year, season)
            return (city.name, year, season, True, "")
        except Exception as e:
            if attempt < 2:
                time.sleep(30)
            else:
                return (city.name, year, season, False, str(e))

N_WORKERS = 2  # lower to not compete with GPU sweep for resources
with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(_dl, job): job for job in jobs}
    for future in as_completed(futures):
        name, year, season, ok, err = future.result()
        if ok:
            done += 1
        else:
            fail += 1
            print(f"  FAILED: {name}/{year}/{season}: {err}")
        total = done + fail
        if total % 20 == 0 or total == len(jobs):
            print(f"[{ts()}] SAR Progress: {total}/{len(jobs)} "
                  f"({done} ok, {fail} fail)")

print(f"\n[{ts()}] SAR re-download complete: {done} ok, {fail} failed")
