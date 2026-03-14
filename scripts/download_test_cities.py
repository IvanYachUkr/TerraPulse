#!/usr/bin/env python3
"""Download and extract features for 5 new test cities.

Uses the existing pipeline infrastructure:
  1. Create anchor TIFs
  2. Download S2 + S1 via terrapulse Rust binary
  3. Create WorldCover labels
  4. Extract features via terrapulse Rust binary
"""
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    CityConfig,
    city_dir, city_anchor_path,
    create_anchor, create_labels,
    ensure_all_dirs,
    TERRAPULSE_BIN, CITIES_DIR,
)

YEARS = [2020, 2021]
YEAR_PAIRS = ["2020_2021"]

# =====================================================================
# 5 NEW TEST CITIES
# =====================================================================
# EPSG zones:  UTM zone from longitude
#   Ankara (32.8E)   -> UTM 36N -> EPSG:32636
#   Sofia  (23.3E)   -> UTM 34N -> EPSG:32634
#   Riga   (24.1E)   -> UTM 34N -> EPSG:32634
#   Edinburgh (3.2W) -> UTM 30N -> EPSG:32630
#   Palermo (13.4E)  -> UTM 33N -> EPSG:32633

NEW_TEST_CITIES = [
    CityConfig(
        name="ankara_test",
        bbox=[32.65, 39.85, 32.95, 40.05],  # ~30x22 km
        epsg=32636,
        wc_tile="N39E032",
        is_test=True,
    ),
    CityConfig(
        name="sofia_test",
        bbox=[23.20, 42.62, 23.45, 42.77],  # ~20x17 km
        epsg=32634,
        wc_tile="N42E023",
        is_test=True,
    ),
    CityConfig(
        name="riga_test",
        bbox=[23.95, 56.88, 24.25, 57.05],  # ~18x19 km
        epsg=32634,
        wc_tile="N56E023",
        is_test=True,
    ),
    CityConfig(
        name="edinburgh_test",
        bbox=[-3.35, 55.88, -3.05, 56.02],  # ~18x16 km
        epsg=32630,
        wc_tile="N55W004",
        is_test=True,
    ),
    CityConfig(
        name="palermo_test",
        bbox=[13.25, 38.05, 13.45, 38.20],  # ~16x17 km
        epsg=32633,
        wc_tile="N38E013",
        is_test=True,
    ),
]


def ts():
    return time.strftime("%H:%M:%S")


def city_raw(city):
    return os.path.join(city_dir(city), "raw_v7")


def city_features(city):
    return os.path.join(city_dir(city), "features_v7")


def main():
    print(f"\n{'='*70}")
    print(f"  DOWNLOADING 5 NEW TEST CITIES")
    print(f"{'='*70}")

    # Check Rust binary
    if not os.path.exists(TERRAPULSE_BIN):
        print(f"  ERROR: Rust binary not found: {TERRAPULSE_BIN}")
        print(f"  Build with: cargo build --release -p terrapulse")
        sys.exit(1)
    print(f"  Rust binary: {TERRAPULSE_BIN}")

    t_total = time.time()

    for ci, city in enumerate(NEW_TEST_CITIES, 1):
        t_city = time.time()
        print(f"\n{'='*70}")
        print(f"  [{ci}/5] {city.name} — bbox={city.bbox} epsg={city.epsg}")
        print(f"{'='*70}")

        # Create dirs
        os.makedirs(city_raw(city), exist_ok=True)
        os.makedirs(city_features(city), exist_ok=True)

        # Step 1: Anchor
        print(f"  [{ts()}] Creating anchor...")
        create_anchor(city)

        # Step 2: Download S2+S1
        anchor = city_anchor_path(city)
        raw = city_raw(city)

        # Check if already downloaded
        all_exist = all(
            os.path.exists(os.path.join(raw, f"sentinel{sat}_{city.name}_{year}_{season}.tif"))
            for sat in ["2", "1"]
            for year in YEARS
            for season in ["spring", "summer", "autumn"]
        )
        if all_exist:
            print(f"  [{ts()}] All S2+SAR TIFs exist — skip download")
        else:
            print(f"  [{ts()}] Downloading S2+SAR via Rust...")
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                print(f"  [{ts()}] DOWNLOAD ERROR: {result.stderr[-500:]}")
                continue
            else:
                print(f"  [{ts()}] Download complete")
                for line in result.stdout.strip().split("\n")[-3:]:
                    print(f"    {line}")

        # Step 3: Labels
        print(f"  [{ts()}] Creating WorldCover labels...")
        create_labels(city)

        # Step 4: Extract features
        feat = city_features(city)
        feat_done = all(
            os.path.exists(os.path.join(feat, f"features_rust_{yp}.parquet"))
            for yp in YEAR_PAIRS
        )
        if feat_done:
            print(f"  [{ts()}] Features already extracted — skip")
        else:
            print(f"  [{ts()}] Extracting features via Rust...")
            cmd = [
                TERRAPULSE_BIN, "extract",
                "--year-pairs", " ".join(YEAR_PAIRS),
                "--region", city.name,
                "--raw-dir", raw,
                "--features-dir", feat,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                print(f"  [{ts()}] EXTRACT ERROR: {result.stderr[-500:]}")
                continue
            else:
                print(f"  [{ts()}] Extraction complete")
                for line in result.stdout.strip().split("\n")[-5:]:
                    print(f"    {line}")

        elapsed = time.time() - t_city
        print(f"  [{ts()}] {city.name} complete in {elapsed:.0f}s")

    total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {total/60:.1f}min total")
    print(f"{'='*70}")

    # Summary
    print(f"\n  City data status:")
    for city in NEW_TEST_CITIES:
        feat = os.path.join(city_features(city), "features_rust_2020_2021.parquet")
        label = os.path.join(city_dir(city), "labels_2021.parquet")
        has_feat = os.path.exists(feat)
        has_label = os.path.exists(label)
        status = "READY" if (has_feat and has_label) else f"feat={has_feat} label={has_label}"
        print(f"    {city.name:25s}: {status}")


if __name__ == "__main__":
    main()
