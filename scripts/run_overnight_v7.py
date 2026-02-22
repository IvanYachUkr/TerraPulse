#!/usr/bin/env python3
"""
V7 Overnight Pipeline: Full Rust download + extract + MLP sweep.

All S2 and SAR downloads go through the Rust binary (new cloud mask + Q1
aggregation + DEFLATE compression). Feature extraction with spatial NN
NaN fill (no median imputation). Labels via Python WorldCover.

Usage:
    python scripts/run_overnight_v7.py                  # full run
    python scripts/run_overnight_v7.py --test-cities 3  # test on N cities first
    python scripts/run_overnight_v7.py --skip-download   # extract + train only
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import city configs and label logic from V5 pipeline
from scripts.run_multi_city_pipeline_v5 import (
    CITIES, TRAIN_CITIES, TEST_CITIES,
    CityConfig,
    city_dir, city_anchor_path,
    create_anchor, create_labels,
    ensure_all_dirs,
    TERRAPULSE_BIN, GRID_PX,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v7_sweep")


# V7-specific directories — separate from V5 to allow comparison
def city_raw_v7(city: CityConfig) -> str:
    return os.path.join(city_dir(city), "raw_v7")

def city_features_v7(city: CityConfig) -> str:
    return os.path.join(city_dir(city), "features_v7")

YEARS = [2020, 2021]
YEAR_PAIRS = ["2020_2021"]


def ts():
    return time.strftime("%H:%M:%S")


# ===========================================================================
# STAGE -1: ENSURE DIRECTORIES EXIST
# ===========================================================================

def ensure_v7_dirs(cities):
    """Create all required directories before anything runs."""
    ensure_all_dirs()  # V5 base dirs: city dirs, worldcover_tiles, etc.
    os.makedirs(MODELS_DIR, exist_ok=True)
    for c in cities:
        os.makedirs(city_raw_v7(c), exist_ok=True)
        os.makedirs(city_features_v7(c), exist_ok=True)


# ===========================================================================
# STAGE 0: ANCHORS
# ===========================================================================

def stage_anchors(cities):
    print(f"\n{'='*70}")
    print("STAGE 0: CREATE ANCHORS")
    print(f"{'='*70}")
    for city in cities:
        create_anchor(city)
    print(f"  [OK] All anchors ready.")


# ===========================================================================
# STAGE 1: DOWNLOAD VIA RUST (S2 + SAR)
# ===========================================================================

def download_city(city: CityConfig):
    """Download S2 + SAR composites for a city using the Rust binary."""
    raw = city_raw_v7(city)
    anchor = city_anchor_path(city)

    # Check if already downloaded — both S2 AND SAR
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


def stage_download(cities, max_workers=1):
    print(f"\n{'='*70}")
    print("STAGE 1: DOWNLOAD S2 + SAR (Rust)")
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


# ===========================================================================
# STAGE 2: LABELS (Python WorldCover — reuse from V5)
# ===========================================================================

def stage_labels(cities):
    print(f"\n{'='*70}")
    print("STAGE 2: WORLDCOVER LABELS")
    print(f"{'='*70}")
    for city in cities:
        create_labels(city)
    print(f"  [OK] All labels ready.")


# ===========================================================================
# STAGE 3: EXTRACT FEATURES (Rust CLI)
# ===========================================================================

def stage_extract(cities):
    print(f"\n{'='*70}")
    print("STAGE 3: EXTRACT FEATURES (Rust CLI — spatial NN fill)")
    print(f"{'='*70}")

    if not os.path.exists(TERRAPULSE_BIN):
        print(f"  ERROR: Rust binary not found: {TERRAPULSE_BIN}")
        return False

    failed = []
    for city in cities:
        year_pairs_str = " ".join(YEAR_PAIRS)
        raw = city_raw_v7(city)
        feat = city_features_v7(city)

        # Check if all already extracted
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


# ===========================================================================
# STAGE 3.5: CLEANUP OLD FEATURES (free ~100GB)
# ===========================================================================

def stage_cleanup(cities):
    print(f"\n{'='*70}")
    print("STAGE 3.5: CLEANUP OLD FEATURES")
    print(f"{'='*70}")

    freed = 0
    cleaned = 0
    for city in cities:
        old_dir = os.path.join(city_dir(city), "features")
        new_parquet = os.path.join(city_features_v7(city),
                                   "features_rust_2020_2021.parquet")

        if not os.path.isdir(old_dir):
            continue

        if not os.path.exists(new_parquet):
            print(f"  [{city.name}] No V7 features yet — keep old")
            continue

        # Safety check: verify V7 parquet is readable and has data
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(new_parquet)
            if meta.num_rows < 10:
                print(f"  [{city.name}] V7 parquet too small ({meta.num_rows} rows) — keep old")
                continue
        except Exception as e:
            print(f"  [{city.name}] V7 parquet unreadable ({e}) — keep old")
            continue

        size = sum(
            os.path.getsize(os.path.join(old_dir, f))
            for f in os.listdir(old_dir)
            if os.path.isfile(os.path.join(old_dir, f))
        )
        shutil.rmtree(old_dir)
        freed += size
        cleaned += 1
        print(f"  [{city.name}] Deleted features/ ({size/1e6:.0f} MB)")

    print(f"  Cleaned {cleaned} cities, freed {freed/1e9:.1f} GB")


# ===========================================================================
# STAGE 4: MLP TRAINING
# ===========================================================================

def stage_train():
    print(f"\n{'='*70}")
    print("STAGE 4: MLP TRAINING (sweep_mlp_v7)")
    print(f"{'='*70}")

    sweep_script = os.path.join(SCRIPT_DIR, "sweep_mlp_v7.py")
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


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="V7 Overnight Pipeline")
    parser.add_argument("--test-cities", type=int, default=0,
                        help="Run on first N cities only (for testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing TIFs")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip feature extraction")
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip WorldCover label creation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip MLP training")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Keep old features/ dirs")
    args = parser.parse_args()

    cities = CITIES
    if args.test_cities > 0:
        cities = CITIES[:args.test_cities]
        print(f"=== TEST MODE: {args.test_cities} cities ===")

    print(f"V7 Pipeline — {len(cities)} cities, years {YEARS}")
    print(f"  Training: {len([c for c in cities if not c.is_test])}")
    print(f"  Test:     {len([c for c in cities if c.is_test])}")
    print(f"  Rust binary: {TERRAPULSE_BIN}")
    print(f"  Exists: {os.path.exists(TERRAPULSE_BIN)}")

    t_total = time.time()

    # Create all directories upfront
    ensure_v7_dirs(cities)

    # Stage 0: Anchors
    stage_anchors(cities)

    # Stage 1: Download S2 + SAR via Rust (with retry for stragglers)
    if not args.skip_download:
        for attempt in range(1, 4):  # up to 3 attempts
            ok = stage_download(cities)
            if ok:
                break
            if attempt < 3:
                print(f"\n  Retry {attempt}/3: re-checking incomplete cities...")
            else:
                print("\n  FATAL: Download stage still has failures after 3 attempts — stopping.")
                print("  Re-run with same command to retry failed cities.")
                sys.exit(1)

    # Stage 2: Labels
    if not args.skip_labels:
        stage_labels(cities)

    # Stage 3: Extract features via Rust
    if not args.skip_extract:
        ok = stage_extract(cities)
        if not ok:
            print("\n  FATAL: Extract stage had failures — stopping.")
            sys.exit(1)

    # Stage 3.5: Cleanup old features
    if not args.skip_cleanup:
        stage_cleanup(cities)

    # Stage 4: Train
    if not args.skip_train:
        stage_train()

    total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"V7 PIPELINE COMPLETE — {total/3600:.1f}h total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
