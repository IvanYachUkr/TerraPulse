"""
Phase 5: Compute delta features (year-over-year + seasonal contrasts).

Reads feature parquet files produced by extract_features.py (Phase 4),
computes deltas, and writes a single merged dataset with all per-cell features.

Delta types:
  1. YoY (year-over-year): feat_2021_season - feat_2020_season
     → Columns: delta_yoy_{season}_{feature}
  2. Seasonal contrast (within a year): feat_season_B - feat_season_A
     → Columns: delta_{year}_{seasonB}_vs_{seasonA}_{feature}

Output:
  data/processed/v2/features_merged_{feature_set}.parquet
    - One row per cell_id (29,946 cells)
    - All 6 composites' features as columns (suffixed by year_season)
    - All delta columns
    - Control columns (cell_id, valid_fraction, low_valid_fraction per composite)

Usage:
  python src/features/compute_deltas.py --feature-set core
  python src/features/compute_deltas.py --feature-set full
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CFG, PROJECT_ROOT  # noqa: E402

V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")

SENTINEL_YEARS = CFG["sentinel2"]["years"]
SEASON_ORDER = CFG["sentinel2"]["season_order"]

# Columns that are per-composite metadata, NOT features for deltas
CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                "reflectance_scale", "full_features_computed"}


def load_features(year: int, season: str, feature_set: str) -> pd.DataFrame:
    """Load one composite's feature parquet."""
    path = os.path.join(V2_DIR, f"features_{year}_{season}_{feature_set}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_parquet(path)
    df = df.sort_values("cell_id").reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return numeric feature columns (exclude control/metadata)."""
    return [c for c in df.columns
            if c not in CONTROL_COLS and pd.api.types.is_numeric_dtype(df[c])]


def suffix_columns(df: pd.DataFrame, suffix: str, feature_cols: list) -> pd.DataFrame:
    """Rename feature columns with a suffix, keep cell_id as-is."""
    rename = {}
    for c in df.columns:
        if c == "cell_id":
            continue
        elif c in feature_cols:
            rename[c] = f"{c}_{suffix}"
        else:
            # Control columns get suffixed too to avoid collision
            rename[c] = f"{c}_{suffix}"
    return df.rename(columns=rename)


def compute_yoy_deltas(merged: pd.DataFrame, season: str,
                       feature_cols: list) -> pd.DataFrame:
    """Compute year-over-year deltas: 2021 - 2020 for a given season."""
    deltas = {}
    y0, y1 = sorted(SENTINEL_YEARS)[:2]
    for feat in feature_cols:
        col_new = f"{feat}_{y1}_{season}"
        col_old = f"{feat}_{y0}_{season}"
        if col_new in merged.columns and col_old in merged.columns:
            deltas[f"delta_yoy_{season}_{feat}"] = merged[col_new] - merged[col_old]
    return pd.DataFrame(deltas, index=merged.index)


def compute_seasonal_deltas(merged: pd.DataFrame, year: int,
                            feature_cols: list) -> pd.DataFrame:
    """Compute seasonal contrasts (pairwise) within a year."""
    deltas = {}
    for i in range(len(SEASON_ORDER)):
        for j in range(i + 1, len(SEASON_ORDER)):
            s_a = SEASON_ORDER[i]
            s_b = SEASON_ORDER[j]
            for feat in feature_cols:
                col_b = f"{feat}_{year}_{s_b}"
                col_a = f"{feat}_{year}_{s_a}"
                if col_b in merged.columns and col_a in merged.columns:
                    deltas[f"delta_{year}_{s_b}_vs_{s_a}_{feat}"] = (
                        merged[col_b] - merged[col_a]
                    )
    return pd.DataFrame(deltas, index=merged.index)


def main():
    parser = argparse.ArgumentParser(description="Compute delta features (v2)")
    parser.add_argument("--feature-set", choices=["core", "full"], default="core")
    args = parser.parse_args()

    feature_set = args.feature_set
    print(f"Computing deltas for feature-set: {feature_set}")
    print(f"Years: {SENTINEL_YEARS}, Seasons: {SEASON_ORDER}")

    # ── Load all composites ──────────────────────────────────────────────
    dfs = {}
    n_cells = None
    feature_cols = None

    for year in SENTINEL_YEARS:
        for season in SEASON_ORDER:
            tag = f"{year}_{season}"
            print(f"\nLoading {tag}...")
            df = load_features(year, season, feature_set)
            print(f"  {len(df)} cells, {len(df.columns)} columns")

            # Verify identical cell_id sets
            if n_cells is None:
                n_cells = len(df)
                expected_ids = df["cell_id"].values.copy()
                feature_cols = set(get_feature_cols(df))
                print(f"  Reference: {n_cells} cells, {len(feature_cols)} feature columns")
            else:
                assert len(df) == n_cells, f"Cell count mismatch: {len(df)} vs {n_cells}"
                assert (df["cell_id"].values == expected_ids).all(), (
                    f"cell_id mismatch in {tag}"
                )
                this_cols = set(get_feature_cols(df))
                if this_cols != feature_cols:
                    extra = this_cols - feature_cols
                    missing = feature_cols - this_cols
                    if extra:
                        print(f"  WARNING: {tag} has {len(extra)} extra columns: {sorted(extra)[:5]}...")
                    if missing:
                        print(f"  WARNING: {tag} missing {len(missing)} columns: {sorted(missing)[:5]}...")
                    feature_cols = feature_cols & this_cols  # use intersection

            dfs[tag] = df

    feature_cols = sorted(feature_cols)  # convert set -> sorted list
    print(f"\n{'='*60}")
    print(f"All {len(dfs)} composites loaded: {n_cells} cells, {len(feature_cols)} features each")

    # ── Build wide table (one row per cell_id) ───────────────────────────
    print("\nBuilding wide table...")
    merged = pd.DataFrame({"cell_id": expected_ids})

    for tag, df in dfs.items():
        suffixed = suffix_columns(df, tag, feature_cols)
        merged = merged.merge(suffixed, on="cell_id", how="left")

    print(f"  Wide table: {merged.shape[0]} rows x {merged.shape[1]} columns")

    # ── Compute YoY deltas ───────────────────────────────────────────────
    print("\nComputing year-over-year deltas...")
    yoy_dfs = []
    for season in SEASON_ORDER:
        yoy = compute_yoy_deltas(merged, season, feature_cols)
        yoy_dfs.append(yoy)
        print(f"  {season}: {yoy.shape[1]} delta columns")

    # ── Compute seasonal deltas ──────────────────────────────────────────
    print("\nComputing seasonal contrasts...")
    seasonal_dfs = []
    for year in SENTINEL_YEARS:
        s_delta = compute_seasonal_deltas(merged, year, feature_cols)
        seasonal_dfs.append(s_delta)
        n_contrasts = len(SEASON_ORDER) * (len(SEASON_ORDER) - 1) // 2
        print(f"  {year}: {s_delta.shape[1]} delta columns ({n_contrasts} contrasts)")

    # ── Concatenate everything ───────────────────────────────────────────
    all_parts = [merged] + yoy_dfs + seasonal_dfs
    result = pd.concat(all_parts, axis=1)

    # Clean infinities
    result = result.replace([np.inf, -np.inf], np.nan)

    # Summary
    n_base = merged.shape[1]
    n_yoy = sum(df.shape[1] for df in yoy_dfs)
    n_seasonal = sum(df.shape[1] for df in seasonal_dfs)
    n_total = result.shape[1]
    nan_pct = 100.0 * result.isna().sum().sum() / (result.shape[0] * result.shape[1])

    print(f"\n{'='*60}")
    print(f"Result shape: {result.shape[0]} cells x {result.shape[1]} columns")
    print(f"  Base features (6 composites): {n_base} columns")
    print(f"  YoY deltas: {n_yoy} columns")
    print(f"  Seasonal deltas: {n_seasonal} columns")
    print(f"  Total: {n_total} columns")
    print(f"  Overall NaN%: {nan_pct:.2f}%")

    # Spot-check key indices
    for idx_name in ["NDVI_mean", "NDBI_mean", "NBR_mean"]:
        cols = [c for c in result.columns if c.startswith(idx_name + "_")]
        if cols:
            vals = result[cols].mean().values
            print(f"  {idx_name} means across composites: "
                  f"[{', '.join(f'{v:.4f}' for v in vals[:6])}]")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(V2_DIR, f"features_merged_{feature_set}.parquet")
    result.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
