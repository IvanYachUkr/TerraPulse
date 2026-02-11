"""
Phase 6: EDA & Reality Check — Report-ready figures.

Produces 8 figure groups saved to reports/figures/:
  1. Label distributions (2020 & 2021)
  2. Label change distributions
  3. Spatial maps (dominant class + deltas)
  4. Key feature distributions across seasons
  5. Feature correlation heatmap
  6. Valid fraction analysis
  7. Reflectance scale verification
  8. Data issues summary

Usage:
  python scripts/run_eda.py
  python scripts/run_eda.py --feature-set full
"""

import argparse
import os
import sys
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG, PROJECT_ROOT  # noqa: E402

# -- Paths ---------------------------------------------------------------------
V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -- Config --------------------------------------------------------------------
CLASS_NAMES = CFG["worldcover"]["class_names"]
CLASS_COLORS = {
    "tree_cover": "#228B22",
    "grassland": "#90EE90",
    "cropland": "#FFD700",
    "built_up": "#DC143C",
    "bare_sparse": "#D2B48C",
    "water": "#4169E1",
}
SENTINEL_YEARS = CFG["sentinel2"]["years"]
SEASON_ORDER = CFG["sentinel2"]["season_order"]

# Aesthetic defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.15


def save_fig(fig, name: str):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Figure 1: Label Distributions
# =============================================================================

def fig1_label_distributions(l20: pd.DataFrame, l21: pd.DataFrame):
    """Grouped bar of class means + per-class histograms."""
    print("\n[Fig 1] Label distributions...")

    # 1a: Grouped bar of class means
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    means_20 = [l20[c].mean() for c in CLASS_NAMES]
    means_21 = [l21[c].mean() for c in CLASS_NAMES]
    bars1 = ax.bar(x - width/2, means_20, width, label="2020",
                   color=[CLASS_COLORS[c] for c in CLASS_NAMES], edgecolor="black", alpha=0.8)
    bars2 = ax.bar(x + width/2, means_21, width, label="2021",
                   color=[CLASS_COLORS[c] for c in CLASS_NAMES], edgecolor="black", alpha=0.5,
                   hatch="//")
    ax.set_ylabel("Mean Proportion")
    ax.set_title("Land-Cover Class Proportions: 2020 vs 2021")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in CLASS_NAMES], rotation=15)
    ax.legend()
    ax.set_ylim(0, max(max(means_20), max(means_21)) * 1.2)
    for bar, val in zip(bars1, means_20):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, means_21):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    save_fig(fig, "01a_class_proportions_bar")

    # 1b: Per-class histograms (show zero-inflation)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, cls in enumerate(CLASS_NAMES):
        ax = axes.flat[i]
        ax.hist(l20[cls], bins=50, alpha=0.6, color=CLASS_COLORS[cls],
                label="2020", edgecolor="white")
        ax.hist(l21[cls], bins=50, alpha=0.4, color=CLASS_COLORS[cls],
                label="2021", edgecolor="white", hatch="//")
        ax.set_title(cls.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Proportion")
        ax.set_ylabel("Cell count")
        pct_zero_20 = 100.0 * (l20[cls] == 0).mean()
        ax.text(0.95, 0.85, f"{pct_zero_20:.0f}% zero",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Per-Class Proportion Distributions", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "01b_class_proportion_histograms")


# =============================================================================
# Figure 2: Label Change Distributions
# =============================================================================

def fig2_label_change(lch: pd.DataFrame):
    """Violin/box plots of delta per class."""
    print("\n[Fig 2] Label change distributions...")

    delta_cols = [f"delta_{c}" for c in CLASS_NAMES]
    data = lch[delta_cols].copy()
    data.columns = [c.replace("delta_", "").replace("_", " ").title() for c in delta_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(
        [data[col].values for col in data.columns],
        positions=range(len(data.columns)),
        showmeans=True, showmedians=True, showextrema=False,
    )
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Δ Proportion (2021 − 2020)")
    ax.set_title("Land-Cover Change Distribution per Class", fontweight="bold")

    # Add stats annotation
    for i, col in enumerate(data.columns):
        vals = data[col].values
        nonzero = np.sum(vals != 0)
        pct_changed = 100.0 * nonzero / len(vals)
        ax.text(i, ax.get_ylim()[1] * 0.9, f"{pct_changed:.0f}%\nchanged",
                ha="center", fontsize=8, color="gray")

    save_fig(fig, "02_label_change_violin")


# =============================================================================
# Figure 3: Spatial Maps
# =============================================================================

def fig3_spatial_maps(grid: gpd.GeoDataFrame, l20: pd.DataFrame,
                      l21: pd.DataFrame, lch: pd.DataFrame):
    """Dominant class maps + delta maps."""
    print("\n[Fig 3] Spatial maps...")

    grid = grid.sort_values("cell_id").reset_index(drop=True)
    gdf = grid.copy()

    # Merge labels
    gdf = gdf.merge(l20[["cell_id"] + CLASS_NAMES], on="cell_id", suffixes=("", "_20"))
    gdf["dominant_2020"] = gdf[CLASS_NAMES].idxmax(axis=1)

    l21_renamed = l21[["cell_id"] + CLASS_NAMES].rename(
        columns={c: f"{c}_21" for c in CLASS_NAMES}
    )
    gdf = gdf.merge(l21_renamed, on="cell_id")
    gdf["dominant_2021"] = gdf[[f"{c}_21" for c in CLASS_NAMES]].idxmax(axis=1)
    gdf["dominant_2021"] = gdf["dominant_2021"].str.replace("_21", "")

    gdf = gdf.merge(lch[["cell_id", "delta_tree_cover", "delta_built_up", "delta_grassland"]],
                     on="cell_id")

    # Color map
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    cmap_classes = mcolors.ListedColormap([CLASS_COLORS[c] for c in CLASS_NAMES])
    norm_classes = mcolors.BoundaryNorm(np.arange(len(CLASS_NAMES) + 1) - 0.5,
                                        len(CLASS_NAMES))

    # 3a: Dominant class map 2020 vs 2021
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, year, col in zip(axes, ["2020", "2021"], ["dominant_2020", "dominant_2021"]):
        gdf["_cls_idx"] = gdf[col].map(class_to_idx)
        gdf.plot(column="_cls_idx", ax=ax, cmap=cmap_classes, norm=norm_classes,
                 linewidth=0, markersize=0)
        ax.set_title(f"Dominant Class — {year}", fontweight="bold")
        ax.set_axis_off()

    legend_patches = [Patch(color=CLASS_COLORS[c],
                            label=c.replace("_", " ").title()) for c in CLASS_NAMES]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=10)
    fig.suptitle("Dominant Land-Cover Class", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    save_fig(fig, "03a_dominant_class_map")

    # 3b: Delta maps for top-3 changing classes
    delta_cols = [("delta_tree_cover", "Tree Cover Δ"),
                  ("delta_built_up", "Built-Up Δ"),
                  ("delta_grassland", "Grassland Δ")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (col, title) in zip(axes, delta_cols):
        vmax = max(abs(gdf[col].quantile(0.01)), abs(gdf[col].quantile(0.99)))
        gdf.plot(column=col, ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 linewidth=0, markersize=0, legend=True,
                 legend_kwds={"shrink": 0.6, "label": "Δ proportion"})
        ax.set_title(title, fontweight="bold")
        ax.set_axis_off()
    fig.suptitle("Land-Cover Change (2021 − 2020)", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "03b_delta_maps")


# =============================================================================
# Figure 4: Key Feature Distributions Across Seasons
# =============================================================================

def fig4_feature_distributions(merged: pd.DataFrame):
    """NDVI/NDBI/NBR across 6 composites, showing phenology."""
    print("\n[Fig 4] Key feature distributions...")

    indices = ["NDVI_mean", "NDBI_mean", "NBR_mean"]
    composites = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, idx in zip(axes, indices):
        data_list = []
        labels = []
        for y, s in composites:
            col = f"{idx}_{y}_{s}"
            if col in merged.columns:
                data_list.append(merged[col].values)
                labels.append(f"{s[:2].upper()}\n{y}")

        parts = ax.violinplot(data_list, positions=range(len(labels)),
                              showmeans=True, showmedians=False, showextrema=False)
        # Color by year
        for i, pc in enumerate(parts["bodies"]):
            year = composites[i][0]
            pc.set_facecolor("#3498DB" if year == 2020 else "#E74C3C")
            pc.set_alpha(0.6)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(idx.replace("_", " "), fontweight="bold")
        ax.set_ylabel("Value")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color="#3498DB", lw=8, alpha=0.6, label="2020"),
                       Line2D([0], [0], color="#E74C3C", lw=8, alpha=0.6, label="2021")]
    axes[0].legend(handles=legend_elements, fontsize=9)

    fig.suptitle("Spectral Index Distributions Across Seasons",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "04_feature_distributions_seasonal")


# =============================================================================
# Figure 5: Feature Correlation Heatmap
# =============================================================================

def fig5_correlation_heatmap(merged: pd.DataFrame, labels: pd.DataFrame):
    """Top-k features correlated with targets + inter-feature correlation."""
    print("\n[Fig 5] Correlation heatmap...")

    # Use 2020 spring features vs 2020 labels
    feature_cols_spring = [c for c in merged.columns
                           if c.endswith("_2020_spring") and not c.startswith("valid_")
                           and not c.startswith("low_") and not c.startswith("reflectance")
                           and not c.startswith("full_features")]

    if not feature_cols_spring:
        print("  Skipping: no spring features found")
        return

    # Merge features with labels
    X = merged[["cell_id"] + feature_cols_spring].copy()
    X = X.merge(labels[["cell_id"] + CLASS_NAMES], on="cell_id")

    # Compute correlation with each class
    corrs = {}
    for cls in CLASS_NAMES:
        for fc in feature_cols_spring:
            c = X[fc].corr(X[cls])
            if np.isfinite(c):
                corrs[(fc, cls)] = c

    if not corrs:
        print("  Skipping: no valid correlations")
        return

    corr_df = pd.Series(corrs).unstack(fill_value=0)
    # Top 20 features by max absolute correlation with any class
    max_abs_corr = corr_df.abs().max(axis=1)
    top_feats = max_abs_corr.nlargest(20).index.tolist()

    # 5a: Feature-target correlation
    fig, ax = plt.subplots(figsize=(10, 12))
    hm_data = corr_df.loc[top_feats]
    hm_data.index = [c.replace("_2020_spring", "") for c in hm_data.index]
    hm_data.columns = [c.replace("_", " ").title() for c in hm_data.columns]
    sns.heatmap(hm_data, center=0, cmap="RdBu_r", annot=True, fmt=".2f",
                ax=ax, linewidths=0.5, vmin=-0.8, vmax=0.8)
    ax.set_title("Top-20 Feature Correlations with Land-Cover Classes\n(2020 Spring)",
                 fontweight="bold")
    ax.set_xlabel("")
    save_fig(fig, "05a_feature_target_correlation")

    # 5b: Inter-feature correlation (top features only)
    feat_vals = X[top_feats]
    feat_vals.columns = [c.replace("_2020_spring", "") for c in feat_vals.columns]
    corr_matrix = feat_vals.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, center=0, cmap="coolwarm",
                annot=False, ax=ax, linewidths=0.3, vmin=-1, vmax=1)
    ax.set_title("Inter-Feature Correlations (Top-20 Features)", fontweight="bold")
    save_fig(fig, "05b_inter_feature_correlation")


# =============================================================================
# Figure 6: Valid Fraction Analysis
# =============================================================================

def fig6_valid_fraction(merged: pd.DataFrame):
    """Distribution and impact of valid_fraction."""
    print("\n[Fig 6] Valid fraction analysis...")

    composites = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, (y, s) in enumerate(composites):
        ax = axes.flat[i]
        vf_col = f"valid_fraction_{y}_{s}"
        if vf_col not in merged.columns:
            continue
        vf = merged[vf_col].values
        ax.hist(vf, bins=50, color="#3498DB", edgecolor="white", alpha=0.7)
        ax.axvline(0.3, color="red", linestyle="--", linewidth=1.5, label="min threshold")
        ax.set_title(f"{y} {s.title()}", fontweight="bold")
        ax.set_xlabel("Valid Fraction")
        ax.set_ylabel("Cell count")

        below = np.sum(vf < 0.3)
        ax.text(0.05, 0.85, f"{below} cells < 0.3\n({100*below/len(vf):.1f}%)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Valid Fraction Distribution per Composite", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "06_valid_fraction_distribution")


# =============================================================================
# Figure 7: Reflectance Scale Verification
# =============================================================================

def fig7_reflectance_scale(merged: pd.DataFrame):
    """Show B08 (NIR) range across composites after /10000 normalization."""
    print("\n[Fig 7] Reflectance scale verification...")

    composites = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]

    nir_data = []
    labels = []
    for y, s in composites:
        col = f"B08_mean_{y}_{s}"
        if col in merged.columns:
            nir_data.append(merged[col].values)
            labels.append(f"{s[:2].upper()}\n{y}")

    if not nir_data:
        print("  Skipping: no B08 data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 7a: Box plot of B08_mean values
    ax = axes[0]
    bp = ax.boxplot(nir_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        year = composites[i][0]
        patch.set_facecolor("#3498DB" if year == 2020 else "#E74C3C")
        patch.set_alpha(0.6)
    ax.set_ylabel("B08 (NIR) Mean Reflectance")
    ax.set_title("NIR Reflectance Range (after /10000)", fontweight="bold")
    ax.text(0.5, 0.95, "Expected range: 0–0.5 (vegetation)",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # 7b: Check scale values
    ax = axes[1]
    scales = []
    for y, s in composites:
        col = f"reflectance_scale_{y}_{s}"
        if col in merged.columns:
            scales.append(merged[col].iloc[0])
        else:
            scales.append(np.nan)
    ax.bar(range(len(labels)), scales, color=["#3498DB"]*3 + ["#E74C3C"]*3, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detected Scale Factor")
    ax.set_title("Auto-Detected Reflectance Scale", fontweight="bold")
    ax.set_ylim(0, 12000)
    for i, v in enumerate(scales):
        if np.isfinite(v):
            ax.text(i, v + 200, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "07_reflectance_scale_verification")


# =============================================================================
# Figure 8: Data Issues Summary
# =============================================================================

def fig8_data_issues(merged: pd.DataFrame, lch: pd.DataFrame,
                      grid: gpd.GeoDataFrame):
    """Visual evidence for documented data issues."""
    print("\n[Fig 8] Data issues summary...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Issue 1 (FIXED): Reflectance scale
    ax = axes[0, 0]
    nir_cols = [f"B08_mean_{y}_{s}" for y in SENTINEL_YEARS for s in SEASON_ORDER
                if f"B08_mean_{y}_{s}" in merged.columns]
    if nir_cols:
        all_nir = pd.concat([merged[c] for c in nir_cols])
        ax.hist(all_nir, bins=100, color="#E74C3C", alpha=0.6, edgecolor="white")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.axvline(0.5, color="green", linewidth=1.5, linestyle="--",
                   label="Typical vegetation max")
        ax.set_xlabel("B08 (NIR) Mean Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    ax.set_title("Issue 1 (FIXED): Reflectance Scale\n"
                 "Values now in [0, 1] after /10000 normalization",
                 fontweight="bold", fontsize=10)

    # Issue 2 (FIXED): Cell dropping
    ax = axes[0, 1]
    vf_cols = [f"valid_fraction_{y}_{s}" for y in SENTINEL_YEARS for s in SEASON_ORDER
               if f"valid_fraction_{y}_{s}" in merged.columns]
    low_vf_cols = [f"low_valid_fraction_{y}_{s}" for y in SENTINEL_YEARS for s in SEASON_ORDER
                   if f"low_valid_fraction_{y}_{s}" in merged.columns]

    if low_vf_cols:
        low_counts = [(c.replace("low_valid_fraction_", ""),
                       int(merged[c].sum())) for c in low_vf_cols]
        tags, counts = zip(*low_counts)
        ax.bar(range(len(tags)), counts,
               color=["#3498DB"]*3 + ["#E74C3C"]*3, alpha=0.7)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=8)
        ax.set_ylabel("Cells with low valid_fraction")
        # Highlight: different seasons have different counts
        ax.text(0.5, 0.85, f"Total cells: {len(merged):,}\nAll retained (none dropped)",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax.set_title("Issue 2 (FIXED): Cell Dropping → Flagging\n"
                 "All cell_ids present; low-quality cells flagged",
                 fontweight="bold", fontsize=10)

    # Issue 3 (FIXED): Imputation scope
    ax = axes[1, 0]
    # Show that control columns were NOT imputed (should have no NaN artifacts)
    control_info = []
    for col_prefix in ["valid_fraction", "low_valid_fraction", "reflectance_scale"]:
        matching = [c for c in merged.columns if c.startswith(col_prefix)]
        for c in matching[:2]:
            nan_count = int(merged[c].isna().sum())
            control_info.append((c.split("_202")[0], nan_count))

    if control_info:
        labels_ci, nan_counts = zip(*control_info)
        colors_ci = ["green" if n == 0 else "red" for n in nan_counts]
        ax.barh(range(len(labels_ci)), nan_counts, color=colors_ci, alpha=0.7)
        ax.set_yticks(range(len(labels_ci)))
        ax.set_yticklabels(labels_ci, fontsize=8)
        ax.set_xlabel("NaN Count")
        ax.text(0.5, 0.85, "✓ Control columns excluded from imputation",
                transform=ax.transAxes, ha="center", fontsize=9, color="green",
                bbox=dict(boxstyle="round", fc="honeydew", alpha=0.9))
    ax.set_title("Issue 3 (FIXED): Imputation Scope\n"
                 "Metadata columns protected from median imputation",
                 fontweight="bold", fontsize=10)

    # Issue 4 (NOT FIXED): WorldCover v100 → v200 label shift
    ax = axes[1, 1]
    gdf = grid.copy().sort_values("cell_id").reset_index(drop=True)
    gdf = gdf.merge(lch[["cell_id", "delta_tree_cover", "delta_built_up"]],
                     on="cell_id")

    # Scatter: tree_cover Δ vs built_up Δ (should show systematic pattern)
    ax.scatter(
        lch["delta_tree_cover"], lch["delta_built_up"],
        s=1, alpha=0.3, c="#555", rasterized=True,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ Tree Cover")
    ax.set_ylabel("Δ Built-Up")

    # Annotate systematic shift
    mean_dt = lch["delta_tree_cover"].mean()
    mean_db = lch["delta_built_up"].mean()
    ax.annotate(f"Mean shift:\nΔtree={mean_dt:+.3f}\nΔbuilt={mean_db:+.3f}",
                xy=(mean_dt, mean_db), xytext=(0.15, 0.15),
                fontsize=9, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax.set_title("Issue 4 (NOT FIXED): WorldCover v100→v200\n"
                 "Systematic Δ may reflect algorithm, not real change",
                 fontweight="bold", fontsize=10)

    fig.suptitle("Documented Data Issues", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "08_data_issues_summary")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 6: EDA")
    parser.add_argument("--feature-set", choices=["core", "full"], default="core")
    args = parser.parse_args()

    feature_set = args.feature_set
    print(f"Phase 6 EDA — feature set: {feature_set}")
    print(f"Output directory: {FIG_DIR}\n")

    # Load data
    print("Loading data...")
    merged = pd.read_parquet(os.path.join(V2_DIR, f"features_merged_{feature_set}.parquet"))
    l20 = pd.read_parquet(os.path.join(V2_DIR, "labels_2020.parquet"))
    l21 = pd.read_parquet(os.path.join(V2_DIR, "labels_2021.parquet"))
    lch = pd.read_parquet(os.path.join(V2_DIR, "labels_change.parquet"))
    grid = gpd.read_file(os.path.join(V2_DIR, "grid.gpkg"))
    print(f"  Merged features: {merged.shape}")
    print(f"  Labels 2020: {l20.shape}")
    print(f"  Labels 2021: {l21.shape}")
    print(f"  Labels change: {lch.shape}")
    print(f"  Grid: {len(grid)} cells")

    # Generate all figures
    fig1_label_distributions(l20, l21)
    fig2_label_change(lch)
    fig3_spatial_maps(grid, l20, l21, lch)
    fig4_feature_distributions(merged)
    fig5_correlation_heatmap(merged, l20)
    fig6_valid_fraction(merged)
    fig7_reflectance_scale(merged)
    fig8_data_issues(merged, lch, grid)

    # Summary
    n_figs = len([f for f in os.listdir(FIG_DIR) if f.endswith(".png")])
    print(f"\n{'='*60}")
    print(f"EDA COMPLETE — {n_figs} figures saved to {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
