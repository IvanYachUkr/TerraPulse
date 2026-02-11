"""
Phase 7: Run spatial train/test split and leakage comparison.

Outputs:
  data/processed/v2/split_spatial.parquet        -- fold assignments per cell
  data/processed/v2/split_spatial_meta.json       -- reproducibility metadata
  reports/phase7/tables/leakage_comparison.csv
  reports/phase7/figures/fold_map_grouped.png
  reports/phase7/figures/fold_map_contiguous.png
  reports/phase7/figures/leakage_barplot.png

Usage:
  python scripts/run_split.py
  python scripts/run_split.py --feature-set full
"""

import argparse
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from pandas.api.types import is_numeric_dtype

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CFG, PROCESSED_V2_DIR, PROJECT_ROOT
from src.splitting import (
    assign_tile_groups,
    build_buffered_folds_from_assignments,
    build_contiguous_band_folds,
    build_random_folds,
    build_spatial_folds,
    leakage_comparison,
    save_split_metadata,
)


# =====================================================================
# Config
# =====================================================================

SPLIT_CFG = CFG["split"]
BLOCK_ROWS = SPLIT_CFG["block_rows"]
BLOCK_COLS = SPLIT_CFG["block_cols"]
N_FOLDS = SPLIT_CFG["n_folds"]
SEED = SPLIT_CFG["seed"]
BUFFER_TILES = SPLIT_CFG["buffer_tiles"]

GRID_CFG = CFG["grid"]
N_GROWS = GRID_CFG["n_rows"]
N_GCOLS = GRID_CFG["n_cols"]
CELL_SIZE = GRID_CFG["size_m"]

CLASS_NAMES = CFG["worldcover"]["class_names"]


def save_fig(fig, name, fig_dir, dpi=150):
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_table(df, name, tbl_dir):
    path = os.path.join(tbl_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def plot_fold_map(fold_assignments, cell_ids, n_folds, n_grows, n_gcols,
                  block_rows, block_cols, n_tile_rows, n_tile_cols, n_groups,
                  title_prefix, fig_name, fig_dir):
    """Reusable fold map plotter."""
    row_idx = cell_ids // n_gcols
    col_idx = cell_ids % n_gcols

    fold_grid = np.full((n_grows, n_gcols), np.nan)
    fold_grid[row_idx, col_idx] = fold_assignments
    fold_grid = np.ma.masked_invalid(fold_grid)

    cmap = ListedColormap(plt.get_cmap("tab10", n_folds).colors)
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(fold_grid, cmap=cmap, vmin=-0.5, vmax=n_folds - 0.5,
                   origin="upper", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_folds), shrink=0.7)
    cbar.set_label("Fold", fontsize=12)
    cbar.set_ticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.set_title(
        f"{title_prefix} -- {block_rows}x{block_cols} tiles "
        f"({n_tile_rows}x{n_tile_cols} = {n_groups} groups, {n_folds} folds)",
        fontweight="bold", fontsize=13,
    )
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")

    for r in range(0, n_grows + 1, block_rows):
        ax.axhline(r - 0.5, color="white", linewidth=0.3, alpha=0.5)
    for c in range(0, n_gcols + 1, block_cols):
        ax.axvline(c - 0.5, color="white", linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    save_fig(fig, fig_name, fig_dir)


# =====================================================================
# Main
# =====================================================================

def main(feature_set="core"):
    print("=" * 70)
    print("Phase 7: Spatial Train/Test Split Design")
    print("=" * 70)

    # -- Output dirs --------------------------------------------------
    report_dir = os.path.join(PROJECT_ROOT, "reports", "phase7")
    fig_dir = os.path.join(report_dir, "figures")
    tbl_dir = os.path.join(report_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # -- Load data ----------------------------------------------------
    print("\n[1/8] Loading data...")
    feat_path = os.path.join(
        PROCESSED_V2_DIR, f"features_merged_{feature_set}.parquet"
    )
    merged = pd.read_parquet(feat_path)
    l20 = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2020.parquet"))
    grid = gpd.read_file(os.path.join(PROCESSED_V2_DIR, "grid.gpkg"))

    # Sort everything by cell_id for alignment
    merged = merged.sort_values("cell_id").reset_index(drop=True)
    l20 = l20.sort_values("cell_id").reset_index(drop=True)
    grid = grid.sort_values("cell_id").reset_index(drop=True)

    assert len(merged) == len(l20) == len(grid), \
        f"Row count mismatch: merged={len(merged)}, l20={len(l20)}, grid={len(grid)}"
    assert (merged["cell_id"].values == l20["cell_id"].values).all(), \
        "cell_id mismatch between features and labels"
    assert (merged["cell_id"].values == grid["cell_id"].values).all(), \
        "cell_id mismatch between features and grid"

    n = len(merged)
    cell_ids = merged["cell_id"].values

    # Assert cell_id contiguity
    expected_ids = np.arange(n, dtype=cell_ids.dtype)
    assert (cell_ids == expected_ids).all(), \
        "cell_id must be contiguous row-major 0..N-1"

    print(f"  {n} cells loaded ({feature_set} feature set)")

    # -- Grid geometry (from config) ----------------------------------
    assert N_GCOLS * N_GROWS == n, \
        f"Grid geometry mismatch: {N_GROWS}x{N_GCOLS}={N_GCOLS*N_GROWS} != {n}"
    print(f"  Grid: {N_GROWS} rows x {N_GCOLS} cols (from config)")

    # -- Tile assignment ----------------------------------------------
    print(f"\n[2/8] Assigning tile groups ({BLOCK_ROWS}x{BLOCK_COLS} cells per tile)...")
    groups, n_tile_cols, n_tile_rows = assign_tile_groups(
        cell_ids, N_GCOLS, BLOCK_ROWS, BLOCK_COLS
    )
    n_groups = len(np.unique(groups))
    print(f"  {n_tile_rows} x {n_tile_cols} = {n_groups} tile groups")
    assert n_groups >= N_FOLDS, \
        f"Too few tile groups ({n_groups}) for {N_FOLDS}-fold CV"

    # -- Build all fold strategies ------------------------------------
    print(f"\n[3/8] Building {N_FOLDS}-fold splits (4 strategies)...")

    # 1) Scattered spatial (GroupKFold)
    grouped_folds, grouped_assignments = build_spatial_folds(groups, N_FOLDS)

    # 2) Contiguous row-band spatial
    contiguous_folds, contiguous_assignments = build_contiguous_band_folds(
        groups, N_FOLDS, n_tile_cols, n_tile_rows
    )

    # 3) Contiguous + Chebyshev buffer
    contiguous_buffered_folds, n_excluded_buf = \
        build_buffered_folds_from_assignments(
            groups, contiguous_assignments, N_FOLDS,
            n_tile_cols, n_tile_rows, buffer_tiles=BUFFER_TILES,
        )

    # 4) Random baseline
    random_folds = build_random_folds(n, N_FOLDS, SEED)

    # Validate grouped folds
    for i, (tr, te) in enumerate(grouped_folds):
        assert len(tr) + len(te) == n, f"Grouped fold {i}: train+test != n"
        assert len(np.intersect1d(tr, te)) == 0, f"Grouped fold {i}: overlap"

    # Validate contiguous folds
    for i, (tr, te) in enumerate(contiguous_folds):
        assert len(tr) + len(te) == n, f"Contiguous fold {i}: train+test != n"
        assert len(np.intersect1d(tr, te)) == 0, f"Contiguous fold {i}: overlap"

    # Validate buffered folds (train+test+buffer = n, no overlap)
    MIN_TRAIN_FRACTION = 0.1
    for i, (tr, te) in enumerate(contiguous_buffered_folds):
        assert len(np.intersect1d(tr, te)) == 0, \
            f"Buffered fold {i}: train/test overlap"
        assert len(tr) > 0, f"Buffered fold {i}: empty training set"
        assert len(tr) >= MIN_TRAIN_FRACTION * n, \
            f"Buffered fold {i}: training set too small ({len(tr)}/{n} = " \
            f"{len(tr)/n:.1%}, need >= {MIN_TRAIN_FRACTION:.0%})"

    # Verify tile integrity for grouped folds
    group_fold = pd.DataFrame({"group": groups, "fold": grouped_assignments})
    folds_per_group = group_fold.groupby("group")["fold"].nunique()
    assert (folds_per_group == 1).all(), \
        "Some tile groups split across multiple folds -- GroupKFold violated"

    print(f"  Grouped fold sizes:     {[len(te) for _, te in grouped_folds]}")
    print(f"  Contiguous fold sizes:  {[len(te) for _, te in contiguous_folds]}")
    print(f"  Contiguous+buf train:   {[len(tr) for tr, _ in contiguous_buffered_folds]}"
          f"  (excluded: {n_excluded_buf})")
    print(f"  Random fold sizes:      {[len(te) for _, te in random_folds]}")
    print("  [OK] All folds validated (partition, no overlap, min train size)")

    # -- Save split (contiguous = primary) ----------------------------
    print("\n[4/8] Saving fold assignments...")

    # Save both assignment types
    split_df = pd.DataFrame({
        "cell_id": cell_ids,
        "fold_grouped": grouped_assignments,
        "fold_contiguous": contiguous_assignments,
        "tile_group": groups,
    })
    split_path = os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet")
    split_df.to_parquet(split_path, index=False)
    print(f"  Saved: {split_path}")
    save_table(split_df, "split_assignments", tbl_dir)

    # Save metadata
    meta_path = os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")
    save_split_metadata(
        meta_path,
        block_rows=BLOCK_ROWS, block_cols=BLOCK_COLS,
        cell_size_m=CELL_SIZE,
        n_folds=N_FOLDS, seed=SEED, buffer_tiles=BUFFER_TILES,
        n_cells=n, n_groups=n_groups, n_gcols=N_GCOLS, n_grows=N_GROWS,
        n_tile_cols=n_tile_cols, n_tile_rows=n_tile_rows,
    )

    # -- Fold maps ----------------------------------------------------
    print("\n[5/8] Plotting fold maps...")

    common_kwargs = dict(
        cell_ids=cell_ids, n_folds=N_FOLDS, n_grows=N_GROWS, n_gcols=N_GCOLS,
        block_rows=BLOCK_ROWS, block_cols=BLOCK_COLS,
        n_tile_rows=n_tile_rows, n_tile_cols=n_tile_cols, n_groups=n_groups,
        fig_dir=fig_dir,
    )
    plot_fold_map(
        grouped_assignments,
        title_prefix="Scattered GroupKFold",
        fig_name="fold_map_grouped",
        **common_kwargs,
    )
    plot_fold_map(
        contiguous_assignments,
        title_prefix="Contiguous Row Bands",
        fig_name="fold_map_contiguous",
        **common_kwargs,
    )

    # -- Leakage comparison -------------------------------------------
    print("\n[6/8] Running leakage comparison (Ridge on 6 label outputs)...")

    # Build X: numeric feature columns only
    control_cols = {"cell_id", "row_idx", "col_idx"}
    feat_cols = [c for c in merged.columns
                 if c not in control_cols and is_numeric_dtype(merged[c])]
    X = merged[feat_cols].values.astype(np.float64)

    # Build y: 6 label proportions
    label_cols = [c for c in CLASS_NAMES if c in l20.columns]
    assert len(label_cols) == 6, \
        f"Expected 6 label cols, got {len(label_cols)}: {label_cols}"
    y = l20[label_cols].values.astype(np.float64)

    assert np.isfinite(X).all(), "Features contain NaN/Inf"
    assert np.isfinite(y).all(), "Labels contain NaN/Inf"

    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  Feature columns used: {len(feat_cols)}")

    # 4-way comparison
    fold_configs = [
        ("random", random_folds),
        ("grouped", grouped_folds),
        ("contiguous", contiguous_folds),
        ("contiguous_buffered", contiguous_buffered_folds),
    ]
    results = leakage_comparison(X, y, fold_configs)
    save_table(results, "leakage_comparison", tbl_dir)

    # Summary
    print("\n  Leakage comparison (R2 uniform average):")
    for stype in ["random", "grouped", "contiguous", "contiguous_buffered"]:
        subset = results[results["split_type"] == stype]
        r2 = subset["r2_uniform"]
        n_nan = int(r2.isna().sum())
        nan_note = f"  ({n_nan} NaN folds)" if n_nan > 0 else ""
        print(f"    {stype:24s}: R2 = {np.nanmean(r2):.4f} +/- {np.nanstd(r2):.4f}"
              f"  (train: {subset['train_size'].mean():.0f},"
              f" test: {subset['test_size'].mean():.0f}){nan_note}")

    random_r2 = results[results["split_type"] == "random"]["r2_uniform"]
    grouped_r2 = results[results["split_type"] == "grouped"]["r2_uniform"]
    contig_r2 = results[results["split_type"] == "contiguous"]["r2_uniform"]
    contig_buf_r2 = results[results["split_type"] == "contiguous_buffered"]["r2_uniform"]

    gap_g = np.nanmean(random_r2) - np.nanmean(grouped_r2)
    gap_c = np.nanmean(random_r2) - np.nanmean(contig_r2)
    gap_cb = np.nanmean(random_r2) - np.nanmean(contig_buf_r2)

    print(f"\n  Leakage gaps (random - X):")
    print(f"    grouped:              {gap_g:+.4f}")
    print(f"    contiguous:           {gap_c:+.4f}")
    print(f"    contiguous+buffered:  {gap_cb:+.4f}")

    if gap_g > 0.01:
        print("  [!] Random split is optimistic -- spatial leakage confirmed!")

    # -- Leakage barplot ----------------------------------------------
    print("\n[7/8] Plotting leakage barplot...")
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(N_FOLDS)
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
    labels = ["Random", "Grouped", "Contiguous", "Contiguous + buffer"]
    r2_arrays = [random_r2, grouped_r2, contig_r2, contig_buf_r2]

    for offset, color, label, r2_vals in zip(offsets, colors, labels, r2_arrays):
        vals = r2_vals.values.copy()
        finite_mask = np.isfinite(vals)
        # Only plot finite bars; leave NaN positions empty
        plot_vals = np.where(finite_mask, vals, 0)
        bar_alpha = np.where(finite_mask, 0.8, 0.0)
        bars = ax.bar(x_pos + offset, plot_vals, width, label=label,
                      color=color)
        for bar, alpha in zip(bars, bar_alpha):
            bar.set_alpha(alpha)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Held-out R2 (uniform avg, 6 outputs)")
    ax.set_title(
        f"Leakage Comparison: Ridge Regression ({BLOCK_ROWS}x{BLOCK_COLS} tiles)\n"
        f"Gaps: grouped {gap_g:+.3f}, contiguous {gap_c:+.3f}, "
        f"contiguous+buf {gap_cb:+.3f}",
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Fold {i}" for i in range(N_FOLDS)])
    ax.legend(loc="lower left", fontsize=9)

    for r2_vals, color in zip(r2_arrays, colors):
        ax.axhline(np.nanmean(r2_vals), color=color, linestyle="--",
                   linewidth=0.7, alpha=0.4)

    fig.tight_layout()
    save_fig(fig, "leakage_barplot", fig_dir)

    # -- Summary table plot -------------------------------------------
    print("\n[8/8] Summary...")
    print("\n" + "=" * 70)
    print("Phase 7 complete.")
    print(f"  Split:    {split_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Figures:  {fig_dir}")
    print(f"  Tables:   {tbl_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: Spatial Split")
    parser.add_argument("--feature-set", default="core",
                        choices=["core", "full"],
                        help="Which merged feature set to use (default: core)")
    args = parser.parse_args()
    main(feature_set=args.feature_set)
