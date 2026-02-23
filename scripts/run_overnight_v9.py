#!/usr/bin/env python3
"""
V9 Overnight Pipeline: Download cities + extract features + MLP sweep.

Reuses raw_v7/ and features_v7/ directories (no duplication).
Calls sweep_mlp_v9.py for training with label threshold sweep.

Usage:
    python scripts/run_overnight_v9.py                   # full run
    python scripts/run_overnight_v9.py --skip-download    # extract + train only
    python scripts/run_overnight_v9.py --skip-extract     # train only
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    CITIES,
    CityConfig,
    city_dir, city_anchor_path,
    create_anchor, create_labels,
    ensure_all_dirs,
    TERRAPULSE_BIN, GRID_PX,
)

YEARS = [2020, 2021]
YEAR_PAIRS = ["2020_2021"]


def city_raw(city: CityConfig) -> str:
    return os.path.join(city_dir(city), "raw_v7")

def city_features(city: CityConfig) -> str:
    return os.path.join(city_dir(city), "features_v7")

def ts():
    return time.strftime("%H:%M:%S")


def ensure_dirs(cities):
    ensure_all_dirs()
    for c in cities:
        os.makedirs(city_raw(c), exist_ok=True)
        os.makedirs(city_features(c), exist_ok=True)


def stage_anchors(cities):
    print(f"\n{'='*70}")
    print("STAGE 0: CREATE ANCHORS")
    print(f"{'='*70}")
    for city in cities:
        create_anchor(city)
    print(f"  [OK] All anchors ready.")


def download_city(city: CityConfig):
    raw = city_raw(city)
    anchor = city_anchor_path(city)

    all_exist = True
    for year in YEARS:
        for season in ["spring", "summer", "autumn"]:
            s2_path = os.path.join(raw, f"sentinel2_{city.name}_{year}_{season}.tif")
            s1_path = os.path.join(raw, f"sentinel1_{city.name}_{year}_{season}.tif")
            if not os.path.exists(s2_path) or not os.path.exists(s1_path):
                all_exist = False
                break
        if not all_exist:
            break
    if all_exist:
        print(f"  [{city.name}] All S2+SAR TIFs exist — skip")
        return True

    years_str = " ".join(str(y) for y in YEARS)
    bbox_str = [str(x) for x in city.bbox]

    cmd = [
        TERRAPULSE_BIN, "download",
        "--bbox", *bbox_str,
        "--epsg", str(city.epsg),
        "--years", years_str,
        "--region", city.name,
        "--raw-dir", raw,
        "--anchor-ref", anchor,
    ]

    print(f"  [{city.name}] Downloading S2+SAR via Rust...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [{city.name}] ERROR ({elapsed:.0f}s): {result.stderr[-500:]}")
        return False
    else:
        print(f"  [{city.name}] Done in {elapsed:.0f}s")
        for line in result.stdout.strip().split("\n")[-3:]:
            print(f"    {line}")
        return True


def stage_download(cities, max_workers=2):
    print(f"\n{'='*70}")
    print("STAGE 1: DOWNLOAD S2 + SAR (Rust) — reusing raw_v7/")
    print(f"{'='*70}")

    if not os.path.exists(TERRAPULSE_BIN):
        print(f"  ERROR: Rust binary not found: {TERRAPULSE_BIN}")
        print(f"  Build with: cargo build --release -p terrapulse")
        return False

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_city, c): c for c in cities}
        for future in as_completed(futures):
            city = futures[future]
            try:
                if not future.result():
                    failed.append(city.name)
            except Exception as e:
                print(f"  [{city.name}] EXCEPTION: {e}")
                failed.append(city.name)

    if failed:
        print(f"\n  WARNING: {len(failed)} cities failed: {failed}")
    print(f"\n[{ts()}] Download stage complete.")
    return len(failed) == 0


def stage_labels(cities):
    print(f"\n{'='*70}")
    print("STAGE 2: WORLDCOVER LABELS")
    print(f"{'='*70}")
    for city in cities:
        create_labels(city)
    print(f"  [OK] All labels ready.")


def stage_extract(cities):
    print(f"\n{'='*70}")
    print("STAGE 3: EXTRACT FEATURES (Rust CLI) — reusing features_v7/")
    print(f"{'='*70}")

    if not os.path.exists(TERRAPULSE_BIN):
        print(f"  ERROR: Rust binary not found: {TERRAPULSE_BIN}")
        return False

    failed = []
    for city in cities:
        year_pairs_str = " ".join(YEAR_PAIRS)
        raw = city_raw(city)
        feat = city_features(city)

        all_done = all(
            os.path.exists(os.path.join(feat, f"features_rust_{yp}.parquet"))
            for yp in YEAR_PAIRS
        )
        if all_done:
            print(f"  [{city.name}] All features already extracted — skip")
            continue

        print(f"  [{city.name}] Extracting features...")
        cmd = [
            TERRAPULSE_BIN, "extract",
            "--year-pairs", year_pairs_str,
            "--region", city.name,
            "--raw-dir", raw,
            "--features-dir", feat,
        ]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [{city.name}] ERROR: {result.stderr[-500:]}")
            failed.append(city.name)
        else:
            print(f"  [{city.name}] Done in {elapsed:.1f}s")
            for line in result.stdout.strip().split("\n")[-5:]:
                print(f"    {line}")

    if failed:
        print(f"\n  WARNING: {len(failed)} cities failed extraction: {failed}")
    print(f"\n[{ts()}] Extract stage complete.")
    return len(failed) == 0


def stage_train():
    print(f"\n{'='*70}")
    print("STAGE 4: MLP TRAINING (sweep_mlp_v9)")
    print(f"{'='*70}")

    sweep_script = os.path.join(SCRIPT_DIR, "sweep_mlp_v9.py")
    if not os.path.exists(sweep_script):
        print(f"  ERROR: {sweep_script} not found")
        return

    print(f"  Running: {sweep_script}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, sweep_script],
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - t0
    print(f"\n[{ts()}] Training complete in {elapsed/3600:.1f}h (exit={result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="V9 Overnight Pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cities = CITIES
    print(f"V9 Pipeline — {len(cities)} cities, years {YEARS}")
    print(f"  Raw data:   raw_v7/  (reused)")
    print(f"  Features:   features_v7/  (reused)")
    print(f"  Rust binary: {TERRAPULSE_BIN} (exists={os.path.exists(TERRAPULSE_BIN)})")

    t_total = time.time()

    ensure_dirs(cities)
    stage_anchors(cities)

    if not args.skip_download:
        for attempt in range(1, 4):
            ok = stage_download(cities)
            if ok:
                break
            if attempt < 3:
                print(f"\n  Retry {attempt}/3...")
            else:
                print("\n  WARNING: Download had failures, continuing anyway...")

    if not args.skip_labels:
        stage_labels(cities)

    if not args.skip_extract:
        ok = stage_extract(cities)
        if not ok:
            print("\n  WARNING: Extract had failures, continuing anyway...")

    if not args.skip_train:
        stage_train()

    total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"V9 PIPELINE COMPLETE — {total/3600:.1f}h total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
