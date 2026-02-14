#!/usr/bin/env python3
"""
Feature Extraction v3 — Rust-accelerated.

Minimal Python wrapper: loads GeoTIFFs, passes to Rust, writes parquet.
All computation happens in Rust (terrapulse_features extension).

Usage:
    python scripts/extract_features_v3_rust.py
    python scripts/extract_features_v3_rust.py --dry-run
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import PROCESSED_V2_DIR
from src.features.extract_features import (
    load_sentinel_v2,
    detect_reflectance_scale,
    compute_grid_shape,
    SENTINEL_YEARS,
    SEASON_ORDER,
    MIN_VALID_FRAC,
)
import terrapulse_features as tf


def main():
    parser = argparse.ArgumentParser(
        description="Feature Extraction v3 — Rust-accelerated")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract first season only")
    args = parser.parse_args()

    n_feat = tf.n_features_per_cell()
    print("=" * 60)
    print(f"Feature Extraction v3 (Rust) — {n_feat} features/season")
    print("=" * 60)

    jobs = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]
    if args.dry_run:
        jobs = jobs[:1]
        print(f"  DRY RUN: only {jobs[0]}")

    nr, nc = None, None
    spectral_arrays = []
    suffixes = []
    vf_first = None

    # Phase 1: Load all rasters (Python — rasterio I/O)
    t_io_total = 0
    for year, season in jobs:
        t0 = time.perf_counter()
        spectral, vf = load_sentinel_v2(year, season)
        dt = time.perf_counter() - t0
        t_io_total += dt

        scale = detect_reflectance_scale(spectral)
        if nr is None:
            nr, nc = compute_grid_shape(spectral)
            vf_first = vf

        # Normalise and ensure contiguous f32
        ref = spectral.astype(np.float32)
        if scale != 1.0:
            ref = ref / scale
        spectral_arrays.append(np.ascontiguousarray(ref))
        suffixes.append(f"{year}_{season}")
        print(f"  Loaded {year}_{season} in {dt:.2f}s", flush=True)

    n_cells = nr * nc

    # Phase 2: Rust computation (all seasons at once)
    t1 = time.perf_counter()
    flat = tf.extract_all_seasons(spectral_arrays, nr, nc)
    dt_rust = time.perf_counter() - t1
    print(f"\n  Rust: {dt_rust:.3f}s for {len(jobs)} seasons "
          f"({dt_rust/len(jobs):.3f}s/season)", flush=True)

    # Free spectral arrays (large)
    del spectral_arrays

    # Phase 3: Build DataFrame directly (no merge!)
    t2 = time.perf_counter()
    result_2d = flat.reshape(n_cells, len(suffixes) * n_feat)

    # Column names from Rust
    columns = tf.feature_names_suffixed(suffixes)

    # Build DataFrame in one shot
    data = {"cell_id": np.arange(n_cells, dtype=np.int32)}
    for i, col in enumerate(columns):
        data[col] = result_2d[:, i]

    # Valid fraction (from first season)
    from src.features.extract_features import GRID_PX
    vf_cells = np.where(np.isfinite(vf_first), vf_first, 0.0).astype(np.float32)
    vf_cells = (vf_cells
                .reshape(nr, GRID_PX, nc, GRID_PX)
                .transpose(0, 2, 1, 3)
                .reshape(n_cells, GRID_PX * GRID_PX)
                .mean(axis=1))
    data["valid_fraction"] = vf_cells
    data["low_valid_fraction"] = (vf_cells < MIN_VALID_FRAC).astype(np.float32)

    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute NaN
    feat_cols = [c for c in df.columns
                 if c not in ("cell_id", "valid_fraction", "low_valid_fraction")]
    nan_count = 0
    for c in feat_cols:
        n = df[c].isna().sum()
        if n > 0:
            nan_count += n
            med = df[c].median()
            df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)
    if nan_count > 0:
        print(f"  Imputed {nan_count} NaN values", flush=True)

    dt_df = time.perf_counter() - t2

    # Phase 4: Write parquet
    t3 = time.perf_counter()
    out_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    df.to_parquet(out_path, index=False)
    dt_write = time.perf_counter() - t3

    total = time.perf_counter() - (t1 - t_io_total)
    size_mb = os.path.getsize(out_path) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  I/O:      {t_io_total:.1f}s")
    print(f"  Rust:     {dt_rust:.1f}s")
    print(f"  DataFrame:{dt_df:.1f}s")
    print(f"  Parquet:  {dt_write:.1f}s")
    print(f"  Total:    {t_io_total + dt_rust + dt_df + dt_write:.1f}s")
    print(f"  Shape:    {df.shape[0]} cells x {df.shape[1]} columns")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Size:     {size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
