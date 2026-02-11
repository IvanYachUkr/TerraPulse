"""
Phase 7: Run spatial train/test split and leakage comparison.

Outputs:
  data/processed/v2/split_spatial.parquet          -- fold assignments per cell
  data/processed/v2/split_spatial_meta.json         -- reproducibility metadata
  reports/phase7/tables/leakage_comparison.csv
  reports/phase7/tables/buffer_sweep.csv
  reports/phase7/tables/fold_contiguity.csv
  reports/phase7/figures/fold_map_grouped.png
  reports/phase7/figures/fold_map_contiguous.png
  reports/phase7/figures/fold_map_morton.png
  reports/phase7/figures/fold_map_region_growing.png
  reports/phase7/figures/leakage_barplot.png
  reports/phase7/figures/buffer_sweep.png

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
    build_morton_folds,
    build_random_folds,
    build_region_growing_folds,
    build_spatial_folds,
    compute_fold_metrics,
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

# Strategy display config
STRATEGY_ORDER = [
    "random", "grouped", "contiguous", "morton",
    "region_growing", "region_growing_buf",
]
STRATEGY_LABELS = {
    "random": "Random",
    "grouped": "Grouped (scattered)",
    "contiguous": "Contiguous (row bands)",
    "morton": "Morton Z-curve",
    "region_growing": "Region growing",
    "region_growing_buf": "Region growing + buffer",
}
STRATEGY_COLORS = {
    "random": "#E74C3C",
    "grouped": "#F39C12",
    "contiguous": "#3498DB",
    "morton": "#9B59B6",
    "region_growing": "#2ECC71",
    "region_growing_buf": "#1ABC9C",
}


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
    print("\n[1/9] Loading data...")
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
    print(f"\n[2/9] Assigning tile groups ({BLOCK_ROWS}x{BLOCK_COLS} cells per tile)...")
    groups, n_tile_cols, n_tile_rows = assign_tile_groups(
        cell_ids, N_GCOLS, BLOCK_ROWS, BLOCK_COLS
    )
    n_groups = len(np.unique(groups))
    print(f"  {n_tile_rows} x {n_tile_cols} = {n_groups} tile groups")
    assert n_groups >= N_FOLDS, \
        f"Too few tile groups ({n_groups}) for {N_FOLDS}-fold CV"

    # -- Build all fold strategies ------------------------------------
    print(f"\n[3/9] Building {N_FOLDS}-fold splits (6 strategies)...")

    # 1) Random baseline
    random_folds = build_random_folds(n, N_FOLDS, SEED)

    # 2) Scattered spatial (GroupKFold)
    grouped_folds, grouped_assignments = build_spatial_folds(groups, N_FOLDS)

    # 3) Contiguous row bands
    contiguous_folds, contiguous_assignments = build_contiguous_band_folds(
        groups, N_FOLDS, n_tile_cols, n_tile_rows
    )

    # 4) Morton Z-curve
    morton_folds, morton_assignments = build_morton_folds(
        groups, N_FOLDS, n_tile_cols, n_tile_rows
    )

    # 5) Region growing (graph partition)
    rg_folds, rg_assignments = build_region_growing_folds(
        groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED
    )

    # 6) Region growing + Chebyshev buffer
    rg_buffered_folds, rg_n_excluded = build_buffered_folds_from_assignments(
        groups, rg_assignments, N_FOLDS,
        n_tile_cols, n_tile_rows, buffer_tiles=BUFFER_TILES,
    )

    # -- Validate all folds -------------------------------------------
    MIN_TRAIN_FRACTION = 0.1

    all_strategies = [
        ("random", random_folds),
        ("grouped", grouped_folds),
        ("contiguous", contiguous_folds),
        ("morton", morton_folds),
        ("region_growing", rg_folds),
    ]
    for name, folds in all_strategies:
        for i, (tr, te) in enumerate(folds):
            assert len(tr) + len(te) == n, f"{name} fold {i}: train+test != n"
            assert len(np.intersect1d(tr, te)) == 0, f"{name} fold {i}: overlap"

    for i, (tr, te) in enumerate(rg_buffered_folds):
        assert len(np.intersect1d(tr, te)) == 0, \
            f"region_growing_buf fold {i}: train/test overlap"
        assert len(tr) >= MIN_TRAIN_FRACTION * n, \
            f"region_growing_buf fold {i}: training set too small " \
            f"({len(tr)}/{n} = {len(tr)/n:.1%})"

    # Tile integrity for grouped folds
    group_fold = pd.DataFrame({"group": groups, "fold": grouped_assignments})
    assert (group_fold.groupby("group")["fold"].nunique() == 1).all(), \
        "GroupKFold tile integrity violated"

    # Print summary
    for name, folds in all_strategies:
        sizes = [len(te) for _, te in folds]
        print(f"  {name:20s}: test sizes = {sizes}")
    print(f"  {'region_growing_buf':20s}: train sizes = "
          f"{[len(tr) for tr, _ in rg_buffered_folds]}"
          f"  (excluded: {rg_n_excluded})")
    print("  [OK] All folds validated")

    # -- Contiguity + balance metrics ---------------------------------
    print("\n[3b/9] Computing contiguity & balance metrics...")

    TILE_HEIGHT_M = BLOCK_ROWS * CELL_SIZE
    TILE_WIDTH_M = BLOCK_COLS * CELL_SIZE

    metrics_configs = [
        ("contiguous", contiguous_assignments),
        ("morton", morton_assignments),
        ("region_growing", rg_assignments),
    ]
    all_metrics = []
    for strat_name, assignments in metrics_configs:
        m = compute_fold_metrics(assignments, groups, n_tile_cols, n_tile_rows)
        m.insert(0, "strategy", strat_name)
        all_metrics.append(m)
        total_comp = m["n_components"].sum()
        max_dev = m["weight_deviation_pct"].max()
        connected = "YES" if (m["n_components"] == 1).all() else "NO"
        mean_compact = m["compactness"].mean()
        worst_compact = m["compactness"].max()  # lower = more compact
        print(f"  {strat_name:20s}: components={list(m['n_components'])} "
              f"connected={connected}  max_dev={max_dev:.1f}%  "
              f"compactness={mean_compact:.3f} (worst={worst_compact:.3f})")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    save_table(metrics_df, "fold_contiguity", tbl_dir)

    # -- Save split ---------------------------------------------------
    print("\n[4/9] Saving fold assignments...")

    split_df = pd.DataFrame({
        "cell_id": cell_ids,
        "fold_grouped": grouped_assignments,
        "fold_contiguous": contiguous_assignments,
        "fold_morton": morton_assignments,
        "fold_region_growing": rg_assignments,
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
    print("\n[5/9] Plotting fold maps...")

    common_kwargs = dict(
        cell_ids=cell_ids, n_folds=N_FOLDS, n_grows=N_GROWS, n_gcols=N_GCOLS,
        block_rows=BLOCK_ROWS, block_cols=BLOCK_COLS,
        n_tile_rows=n_tile_rows, n_tile_cols=n_tile_cols, n_groups=n_groups,
        fig_dir=fig_dir,
    )
    maps = [
        (grouped_assignments, "Scattered GroupKFold", "fold_map_grouped"),
        (contiguous_assignments, "Contiguous Row Bands", "fold_map_contiguous"),
        (morton_assignments, "Morton Z-curve", "fold_map_morton"),
        (rg_assignments, "Region Growing", "fold_map_region_growing"),
    ]
    for assignments, title, fname in maps:
        plot_fold_map(assignments, title_prefix=title, fig_name=fname,
                      **common_kwargs)

    # -- Prepare X, y for leakage comparison --------------------------
    print("\n[6/9] Preparing features and labels...")

    control_cols = {"cell_id", "row_idx", "col_idx"}
    feat_cols = [c for c in merged.columns
                 if c not in control_cols and is_numeric_dtype(merged[c])]
    X = merged[feat_cols].values.astype(np.float64)

    label_cols = [c for c in CLASS_NAMES if c in l20.columns]
    assert len(label_cols) == 6, \
        f"Expected 6 label cols, got {len(label_cols)}: {label_cols}"
    y = l20[label_cols].values.astype(np.float64)

    assert np.isfinite(X).all(), "Features contain NaN/Inf"
    assert np.isfinite(y).all(), "Labels contain NaN/Inf"
    print(f"  X: {X.shape}, y: {y.shape}, features: {len(feat_cols)}")

    # -- 6-way leakage comparison -------------------------------------
    print("\n[7/9] Running 6-way leakage comparison (Ridge regression)...")

    fold_configs = [
        ("random", random_folds),
        ("grouped", grouped_folds),
        ("contiguous", contiguous_folds),
        ("morton", morton_folds),
        ("region_growing", rg_folds),
        ("region_growing_buf", rg_buffered_folds),
    ]
    results = leakage_comparison(X, y, fold_configs)
    save_table(results, "leakage_comparison", tbl_dir)

    # Summary
    print("\n  Leakage comparison (R2 uniform average):")
    random_r2_mean = np.nanmean(
        results[results["split_type"] == "random"]["r2_uniform"]
    )
    for stype in STRATEGY_ORDER:
        subset = results[results["split_type"] == stype]
        r2 = subset["r2_uniform"]
        n_nan = int(r2.isna().sum())
        mean_r2 = np.nanmean(r2)
        gap = random_r2_mean - mean_r2
        nan_note = f"  ({n_nan} NaN)" if n_nan > 0 else ""
        print(f"    {stype:24s}: R2={mean_r2:.4f} +/- {np.nanstd(r2):.4f}"
              f"  gap={gap:+.4f}"
              f"  (train:{subset['train_size'].mean():.0f},"
              f" test:{subset['test_size'].mean():.0f}){nan_note}")

    # -- Leakage barplot ----------------------------------------------
    print("\n[8/9] Plotting leakage barplot...")
    fig, ax = plt.subplots(figsize=(14, 5))
    x_pos = np.arange(N_FOLDS)
    n_strats = len(STRATEGY_ORDER)
    width = 0.8 / n_strats

    for i, stype in enumerate(STRATEGY_ORDER):
        r2_vals = results[results["split_type"] == stype]["r2_uniform"].values
        offset = (i - (n_strats - 1) / 2) * width
        finite_mask = np.isfinite(r2_vals)
        plot_vals = np.where(finite_mask, r2_vals, 0)
        bars = ax.bar(x_pos + offset, plot_vals, width,
                      label=STRATEGY_LABELS[stype],
                      color=STRATEGY_COLORS[stype])
        for bar, is_fin in zip(bars, finite_mask):
            bar.set_alpha(0.8 if is_fin else 0.0)
        # Mean line
        ax.axhline(np.nanmean(r2_vals), color=STRATEGY_COLORS[stype],
                   linestyle="--", linewidth=0.6, alpha=0.3)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Held-out R2 (uniform avg, 6 outputs)")
    ax.set_title(
        f"6-Way Leakage Comparison: Ridge Regression "
        f"({BLOCK_ROWS}x{BLOCK_COLS} tiles, {n_groups} groups)",
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Fold {i}" for i in range(N_FOLDS)])
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, "leakage_barplot", fig_dir)

    # -- Buffer sweep -------------------------------------------------
    print("\n[9/9] Running buffer sweep (region growing, buffer 0-2)...")

    sweep_rows = []
    for buf in range(3):  # 0, 1, 2
        if buf == 0:
            buf_folds = rg_folds
        else:
            buf_folds, _ = build_buffered_folds_from_assignments(
                groups, rg_assignments, N_FOLDS,
                n_tile_cols, n_tile_rows, buffer_tiles=buf,
            )
        buf_results = leakage_comparison(X, y, [(f"buf_{buf}", buf_folds)])
        for _, row in buf_results.iterrows():
            sweep_rows.append({
                "buffer_tiles": buf,
                "buffer_m": buf * max(TILE_HEIGHT_M, TILE_WIDTH_M),
                "fold": row["fold"],
                "r2_uniform": row["r2_uniform"],
                "r2_weighted": row["r2_weighted"],
                "train_size": row["train_size"],
                "test_size": row["test_size"],
                "train_fraction": row["train_size"] / n,
            })

    sweep_df = pd.DataFrame(sweep_rows)
    save_table(sweep_df, "buffer_sweep", tbl_dir)

    # Buffer sweep figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    buf_vals = sorted(sweep_df["buffer_tiles"].unique())
    buf_m_vals = [b * max(TILE_HEIGHT_M, TILE_WIDTH_M) for b in buf_vals]
    mean_r2 = [np.nanmean(
        sweep_df[sweep_df["buffer_tiles"] == b]["r2_uniform"]
    ) for b in buf_vals]
    std_r2 = [np.nanstd(
        sweep_df[sweep_df["buffer_tiles"] == b]["r2_uniform"]
    ) for b in buf_vals]
    mean_frac = [sweep_df[sweep_df["buffer_tiles"] == b][
        "train_fraction"].mean() for b in buf_vals]

    # R2 vs buffer distance
    ax1.errorbar(buf_m_vals, mean_r2, yerr=std_r2, marker="o",
                 capsize=4, color="#2ECC71", linewidth=2, markersize=8)
    ax1.axhline(random_r2_mean, color="#E74C3C", linestyle="--",
                linewidth=1, alpha=0.7, label=f"Random R2={random_r2_mean:.3f}")
    ax1.set_xlabel("Buffer distance (m)")
    ax1.set_ylabel("Mean R2 (uniform avg)")
    ax1.set_title("Performance vs Spatial Separation", fontweight="bold")
    ax1.legend(fontsize=9)
    for i, (bm, r2, s) in enumerate(zip(buf_m_vals, mean_r2, std_r2)):
        ax1.annotate(f"{r2:.3f}", (bm, r2 + s + 0.005),
                     ha="center", fontsize=9)

    # Train fraction vs buffer distance
    ax2.bar(buf_m_vals, mean_frac, width=300, color="#3498DB", alpha=0.7)
    ax2.set_xlabel("Buffer distance (m)")
    ax2.set_ylabel("Training fraction retained")
    ax2.set_title("Cost of Spatial Separation", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    for bm, frac in zip(buf_m_vals, mean_frac):
        ax2.text(bm, frac + 0.02, f"{frac:.1%}", ha="center", fontsize=10)

    fig.suptitle("Buffer Sweep: Region Growing Folds",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "buffer_sweep", fig_dir)

    # -- Done ---------------------------------------------------------
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
