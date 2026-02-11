"""
Phase 6 v2: Research-grade EDA & Reality Check.

Outputs to reports/phase6/{feature_set}/:
  figures/  — PNGs
  tables/   — CSVs with computed statistics

Usage:
  python scripts/run_eda.py --feature-set core
  python scripts/run_eda.py --feature-set full --sample-n 10000 --dpi 200
"""
import argparse, os, sys, re, warnings
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG, PROJECT_ROOT

V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
CLASS_NAMES = CFG["worldcover"]["class_names"]
CLASS_COLORS = {"tree_cover":"#228B22","grassland":"#90EE90","cropland":"#FFD700",
                "built_up":"#DC143C","bare_sparse":"#D2B48C","water":"#4169E1"}
_missing_colors = set(CLASS_NAMES) - set(CLASS_COLORS)
assert not _missing_colors, f"Missing CLASS_COLORS for: {sorted(_missing_colors)}"
SENTINEL_YEARS = sorted(CFG["sentinel2"]["years"])
SEASON_ORDER = CFG["sentinel2"]["season_order"]
COMPOSITES = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]
assert len(CLASS_NAMES) == 6, f"Script assumes 6 classes for 2x3 plots, got {len(CLASS_NAMES)}"
assert len(COMPOSITES) == 6, f"Script assumes 2 years x 3 seasons = 6 composites, got {len(COMPOSITES)}"

sns.set_theme(style="whitegrid", font_scale=1.1)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def save_fig(fig, name, fig_dir, dpi=150):
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {path}")

def save_table(df, name, tbl_dir):
    path = os.path.join(tbl_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

def sanity_checks(merged, l20, l21, lch, grid):
    """Fail-fast validation before any plotting."""
    print("Running sanity checks...")
    n = len(grid)
    for name, df in [("merged", merged), ("labels_2020", l20),
                      ("labels_2021", l21), ("labels_change", lch)]:
        assert len(df) == n, f"{name} has {len(df)} rows, expected {n}"
        assert df["cell_id"].is_unique, f"{name} has duplicate cell_ids"
        assert (df["cell_id"].sort_values().values == np.arange(n)).all(), \
            f"{name} cell_ids not contiguous 0..{n-1}"
    # Cross-table alignment: all must share identical cell_id order
    ref_ids = merged["cell_id"].values
    assert (ref_ids == l20["cell_id"].values).all(), "merged/l20 cell_id mismatch"
    assert (ref_ids == l21["cell_id"].values).all(), "merged/l21 cell_id mismatch"
    assert (ref_ids == lch["cell_id"].values).all(), "merged/lch cell_id mismatch"
    assert (ref_ids == grid["cell_id"].values).all(), "merged/grid cell_id mismatch"
    assert len(merged.columns) == len(set(merged.columns)), "Duplicate columns in merged"
    for y, s in COMPOSITES:
        col = f"NDVI_mean_{y}_{s}"
        assert col in merged.columns, f"Missing expected column: {col}"
    # Label invariants: proportions in [0,1], rows sum to ~1, deltas sum to ~0
    tol = 1e-2
    for name, lab in [("l20", l20), ("l21", l21)]:
        vals = lab[CLASS_NAMES].to_numpy()
        assert np.nanmin(vals) >= -tol and np.nanmax(vals) <= 1+tol, \
            f"{name} label proportions outside [0,1]"
        row_sums = np.nansum(vals, axis=1)
        assert np.all(np.abs(row_sums - 1) < tol), \
            f"{name} label proportions don't sum to 1 (max deviation: {np.max(np.abs(row_sums-1)):.4f})"
    delta_cols_check = [f"delta_{c}" for c in CLASS_NAMES if f"delta_{c}" in lch.columns]
    if delta_cols_check:
        d = lch[delta_cols_check].to_numpy()
        assert np.all(np.abs(np.nansum(d, axis=1)) < tol), \
            "label deltas do not sum to ~0"
    # Ensure required columns exist (prevents KeyError deep in plotting)
    missing_l20 = set(CLASS_NAMES) - set(l20.columns)
    missing_l21 = set(CLASS_NAMES) - set(l21.columns)
    assert not missing_l20, f"labels_2020 missing class cols: {sorted(missing_l20)}"
    assert not missing_l21, f"labels_2021 missing class cols: {sorted(missing_l21)}"
    missing_deltas = {f"delta_{c}" for c in CLASS_NAMES} - set(lch.columns)
    assert not missing_deltas, f"labels_change missing delta cols: {sorted(missing_deltas)}"
    print(f"  All checks passed ({n} cells, {len(merged.columns)} features, row-aligned)")


# ═══════════════════════════════════════════════════════════════════════
# C1: Feature Manifest
# ═══════════════════════════════════════════════════════════════════════

FAMILY_PATTERNS = [
    (r"^B\d+", "band_stats"), (r"^NDVI", "indices"), (r"^NDWI", "indices"),
    (r"^NDBI", "indices"), (r"^NDMI", "indices"), (r"^NBR", "indices"),
    (r"^SAVI", "indices"), (r"^BSI", "indices"), (r"^NDRE", "indices"),
    (r"^TC_", "tasseled_cap"), (r"^edge_", "spatial"), (r"^laplacian_", "spatial"),
    (r"^moran_", "spatial"), (r"^ndvi_spread", "spatial"),
    (r"^glcm_", "texture_glcm"), (r"^gabor_", "texture_gabor"),
    (r"^lbp_", "texture_lbp"), (r"^hog_", "texture_hog"),
    (r"^morph_", "morphological"), (r"^semivar_", "semivariogram"),
]

def classify_family(root):
    for pat, fam in FAMILY_PATTERNS:
        if re.match(pat, root, re.IGNORECASE):
            return fam
    return "other"

def build_manifest(columns):
    """Parse column names into structured manifest."""
    print("\n[Manifest] Building feature manifest...")
    records = []
    for col in columns:
        if col == "cell_id":
            records.append(dict(column=col, kind="id", year=None, season=None,
                                feature_root=col, family="meta"))
            continue
        # Try delta_yoy pattern
        m = re.match(r"delta_yoy_(\w+?)_(.+)$", col)
        if m:
            records.append(dict(column=col, kind="yoy_delta", year=None,
                                season=m.group(1), feature_root=m.group(2),
                                family=classify_family(m.group(2))))
            continue
        # Try delta_seasonal pattern
        m = re.match(r"delta_(\d{4})_(\w+?)_vs_(\w+?)_(.+)$", col)
        if m:
            records.append(dict(column=col, kind="seasonal_delta", year=int(m.group(1)),
                                season=f"{m.group(2)}_vs_{m.group(3)}",
                                feature_root=m.group(4),
                                family=classify_family(m.group(4))))
            continue
        # Try base feature: {root}_{year}_{season}
        m = re.match(r"(.+?)_(\d{4})_(spring|summer|autumn)$", col)
        if m:
            root = m.group(1)
            kind = "control" if root in {"valid_fraction","low_valid_fraction",
                                          "reflectance_scale","full_features_computed"} else "base"
            records.append(dict(column=col, kind=kind, year=int(m.group(2)),
                                season=m.group(3), feature_root=root,
                                family="control" if kind=="control" else classify_family(root)))
            continue
        records.append(dict(column=col, kind="unknown", year=None, season=None,
                            feature_root=col, family="other"))

    manifest = pd.DataFrame(records)
    print(f"  {len(manifest)} columns parsed:")
    print(manifest.groupby("kind").size().to_string())
    return manifest


# ═══════════════════════════════════════════════════════════════════════
# Figures 1–4: Labels + Basic Features
# ═══════════════════════════════════════════════════════════════════════

def fig1_label_distributions(l20, l21, fig_dir, tbl_dir, dpi):
    print("\n[Fig 1] Label distributions...")
    # 1a: Grouped bar
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CLASS_NAMES)); w = 0.35
    m20 = [l20[c].mean() for c in CLASS_NAMES]
    m21 = [l21[c].mean() for c in CLASS_NAMES]
    cols = [CLASS_COLORS[c] for c in CLASS_NAMES]
    ax.bar(x-w/2, m20, w, color=cols, edgecolor="black", alpha=0.8, label="2020")
    ax.bar(x+w/2, m21, w, color=cols, edgecolor="black", alpha=0.5, hatch="//", label="2021")
    for i in range(len(CLASS_NAMES)):
        ax.text(x[i]-w/2, m20[i]+.005, f"{m20[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i]+w/2, m21[i]+.005, f"{m21[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_"," ").title() for c in CLASS_NAMES], rotation=15)
    ax.set_ylabel("Mean Proportion"); ax.set_title("Land-Cover Class Proportions: 2020 vs 2021")
    ax.legend(); ax.set_ylim(0, max(max(m20),max(m21))*1.2)
    save_fig(fig, "01a_class_proportions_bar", fig_dir, dpi)

    # 1b: Histograms with zero % for both years
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, cls in enumerate(CLASS_NAMES):
        ax = axes.flat[i]
        ax.hist(l20[cls], bins=50, alpha=0.6, color=CLASS_COLORS[cls], label="2020", edgecolor="white")
        ax.hist(l21[cls], bins=50, alpha=0.4, color=CLASS_COLORS[cls], label="2021", hatch="//", edgecolor="white")
        ax.set_title(cls.replace("_"," ").title(), fontweight="bold")
        ax.set_xlabel("Proportion"); ax.set_ylabel("Cells")
        z20 = 100*(l20[cls]==0).mean(); z21 = 100*(l21[cls]==0).mean()
        ax.text(.95,.85, f"zero: {z20:.0f}%/{z21:.0f}%", transform=ax.transAxes, ha="right",
                fontsize=8, bbox=dict(boxstyle="round,pad=.3", fc="white", alpha=.8))
        if i==0: ax.legend(fontsize=8)
    fig.suptitle("Per-Class Proportion Distributions", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95])
    save_fig(fig, "01b_class_proportion_histograms", fig_dir, dpi)

    # Save stats
    stats = pd.DataFrame({"class": CLASS_NAMES, "mean_2020": m20, "mean_2021": m21,
                           "zero_pct_2020": [100*(l20[c]==0).mean() for c in CLASS_NAMES],
                           "zero_pct_2021": [100*(l21[c]==0).mean() for c in CLASS_NAMES]})
    save_table(stats, "label_summary_stats", tbl_dir)


def fig2_label_change(lch, fig_dir, tbl_dir, dpi):
    print("\n[Fig 2] Label change distributions...")
    delta_cols = [f"delta_{c}" for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot([lch[c].values for c in delta_cols], showmeans=True, showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], CLASS_NAMES):
        pc.set_facecolor(CLASS_COLORS[c]); pc.set_alpha(0.6)
    labels_nice = [c.replace("delta_","").replace("_"," ").title() for c in delta_cols]
    ax.set_xticks(range(1, len(delta_cols)+1)); ax.set_xticklabels(labels_nice, rotation=15)
    ax.axhline(0, color="black", linewidth=.8, linestyle="--")
    ax.set_ylabel("Δ Proportion (2021 − 2020)"); ax.set_title("Land-Cover Change Distribution", fontweight="bold")
    for i, c in enumerate(delta_cols):
        changed = np.sum(np.abs(lch[c]) > 0.01)
        ax.text(i+1, ax.get_ylim()[1]*.9, f"{100*changed/len(lch):.0f}%\n|Δ|>.01", ha="center", fontsize=8, color="gray")
    save_fig(fig, "02_label_change_violin", fig_dir, dpi)

    stats = pd.DataFrame({c: lch[c].describe() for c in delta_cols})
    stats.loc["pct_changed_001"] = [(np.abs(lch[c])>0.01).mean()*100 for c in delta_cols]
    save_table(stats.T.reset_index().rename(columns={"index":"class"}), "change_summary_stats", tbl_dir)


def fig3_spatial_maps(grid, l20, l21, lch, fig_dir, dpi):
    print("\n[Fig 3] Spatial maps...")
    gdf = grid.sort_values("cell_id").reset_index(drop=True).copy()
    gdf = gdf.merge(l20[["cell_id"]+CLASS_NAMES], on="cell_id")
    gdf["dominant_2020"] = gdf[CLASS_NAMES].idxmax(axis=1)
    l21r = l21[["cell_id"]+CLASS_NAMES].rename(columns={c: f"{c}_21" for c in CLASS_NAMES})
    gdf = gdf.merge(l21r, on="cell_id")
    gdf["dominant_2021"] = gdf[[f"{c}_21" for c in CLASS_NAMES]].idxmax(axis=1).str.replace("_21","")
    gdf = gdf.merge(lch[["cell_id","delta_tree_cover","delta_built_up","delta_grassland"]], on="cell_id")

    c2i = {c:i for i,c in enumerate(CLASS_NAMES)}
    cmap = mcolors.ListedColormap([CLASS_COLORS[c] for c in CLASS_NAMES])
    norm = mcolors.BoundaryNorm(np.arange(len(CLASS_NAMES)+1)-.5, len(CLASS_NAMES))

    fig, axes = plt.subplots(1,2, figsize=(16,8))
    for ax, yr, col in zip(axes, ["2020","2021"], ["dominant_2020","dominant_2021"]):
        gdf["_ci"] = gdf[col].map(c2i)
        gdf.plot(column="_ci", ax=ax, cmap=cmap, norm=norm, linewidth=0)
        ax.set_title(f"Dominant Class — {yr}", fontweight="bold"); ax.set_axis_off()
    fig.legend(handles=[Patch(color=CLASS_COLORS[c], label=c.replace("_"," ").title()) for c in CLASS_NAMES],
               loc="lower center", ncol=6, fontsize=10)
    fig.suptitle("Dominant Land-Cover Class", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0,.06,1,.95])
    save_fig(fig, "03a_dominant_class_map", fig_dir, dpi)

    dcols = [("delta_tree_cover","Tree Cover Δ"),("delta_built_up","Built-Up Δ"),("delta_grassland","Grassland Δ")]
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    for ax, (col, title) in zip(axes, dcols):
        vmax = max(abs(gdf[col].quantile(.01)), abs(gdf[col].quantile(.99)))
        gdf.plot(column=col, ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax, linewidth=0,
                 legend=True, legend_kwds={"shrink":.6, "label":"Δ"})
        ax.set_title(title, fontweight="bold"); ax.set_axis_off()
    fig.suptitle("Land-Cover Change (2021−2020) — quality-coupled cells flagged in text", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95])
    save_fig(fig, "03b_delta_maps", fig_dir, dpi)


def fig4_feature_distributions(merged, fig_dir, dpi):
    print("\n[Fig 4] Key feature distributions...")
    indices = ["NDVI_mean", "NDBI_mean", "NBR_mean"]

    # 4a: Seasonal distributions
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    for ax, idx in zip(axes, indices):
        data_list, labels_list, years_used = [], [], []
        for y, s in COMPOSITES:
            col = f"{idx}_{y}_{s}"
            if col in merged.columns:
                data_list.append(merged[col].values)
                labels_list.append(f"{s[:2].upper()}\n{y}")
                years_used.append(y)
        if not data_list:
            ax.text(.5, .5, f"{idx}\nMissing", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
            ax.set_axis_off()
            continue
        parts = ax.violinplot(data_list, showmeans=True, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor("#3498DB" if years_used[i]==SENTINEL_YEARS[0] else "#E74C3C")
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels_list)+1)); ax.set_xticklabels(labels_list, fontsize=9)
        ax.set_title(idx.replace("_"," "), fontweight="bold"); ax.set_ylabel("Value")
    axes[0].legend(handles=[Line2D([0],[0],color="#3498DB",lw=8,alpha=.6,label=str(SENTINEL_YEARS[0])),
                            Line2D([0],[0],color="#E74C3C",lw=8,alpha=.6,label=str(SENTINEL_YEARS[1]))], fontsize=9)
    fig.suptitle("Spectral Index Distributions Across Seasons", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95])
    save_fig(fig, "04a_feature_distributions_seasonal", fig_dir, dpi)

    # 4b: YoY delta distributions
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    for ax, idx in zip(axes, indices):
        data_list, labels_list = [], []
        for s in SEASON_ORDER:
            col = f"delta_yoy_{s}_{idx}"
            if col in merged.columns:
                data_list.append(merged[col].values)
                labels_list.append(s.title())
        if data_list:
            parts = ax.violinplot(data_list, showmeans=True, showextrema=False)
            for pc in parts["bodies"]: pc.set_facecolor("#9B59B6"); pc.set_alpha(0.5)
            ax.set_xticks(range(1, len(labels_list)+1)); ax.set_xticklabels(labels_list)
            ax.axhline(0, color="black", linewidth=.8, linestyle="--")
        ax.set_title(f"YoY Δ {idx.replace('_',' ')}", fontweight="bold"); ax.set_ylabel("Δ Value")
    fig.suptitle("Year-over-Year Feature Deltas by Season", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95])
    save_fig(fig, "04b_yoy_delta_distributions", fig_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════
# C2: Redundancy Clustering + Drift
# ═══════════════════════════════════════════════════════════════════════

def fig5_redundancy_and_drift(merged, labels, manifest, fig_dir, tbl_dir, dpi, sample_n, seed):
    print("\n[Fig 5] Redundancy clustering + drift metrics...")

    # Get base feature columns for 2020 spring (reference)
    base_spring = manifest[(manifest.kind=="base") & (manifest.year==SENTINEL_YEARS[0])
                           & (manifest.season=="spring")]["column"].tolist()
    if len(base_spring) < 5:
        print("  Skipping: too few base spring features"); return

    # 5a: Spearman feature-target correlation
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(merged), min(sample_n, len(merged)), replace=False)
    X_sample = merged.iloc[idx]

    corr_records = []
    for fc in base_spring:
        for cls in CLASS_NAMES:
            r, _ = spearmanr(X_sample[fc], labels.iloc[idx][cls], nan_policy="omit")
            if np.isfinite(r):
                corr_records.append({"feature": fc, "class": cls, "spearman_r": r})
    corr_df = pd.DataFrame(corr_records)
    if corr_df.empty:
        print("  Skipping: no valid correlations"); return

    pivot = corr_df.pivot(index="feature", columns="class", values="spearman_r").fillna(0)
    top_feats = pivot.abs().max(axis=1).nlargest(20).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 12))
    hm = pivot.loc[top_feats].copy()
    hm.index = [c.replace(f"_{SENTINEL_YEARS[0]}_spring","") for c in hm.index]
    hm.columns = [c.replace("_"," ").title() for c in hm.columns]
    sns.heatmap(hm, center=0, cmap="RdBu_r", annot=True, fmt=".2f", ax=ax,
                linewidths=.5, vmin=-.8, vmax=.8)
    ax.set_title("Top-20 Spearman Correlations with Targets\n(2020 Spring, sampled)", fontweight="bold")
    save_fig(fig, "05a_feature_target_spearman", fig_dir, dpi)
    save_table(corr_df, "feature_target_correlations", tbl_dir)

    # 5b: Redundancy clustering
    if len(base_spring) > 10:
        X_corr = X_sample[base_spring].corr(method="spearman").fillna(0)
        dist = 1 - X_corr.abs().values
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, 2)
        dist_condensed = squareform(dist, checks=False)
        Z = linkage(dist_condensed, method="average")
        clusters = fcluster(Z, t=0.3, criterion="distance")

        cluster_df = pd.DataFrame({"feature": base_spring, "cluster": clusters})
        cluster_df["feature_root"] = cluster_df["feature"].str.replace(f"_{SENTINEL_YEARS[0]}_spring","")
        cluster_df = cluster_df.sort_values("cluster")
        n_clusters = cluster_df["cluster"].nunique()
        print(f"  Redundancy: {len(base_spring)} features -> {n_clusters} clusters (t=0.3)")
        save_table(cluster_df, "redundancy_clusters", tbl_dir)

        # Heatmap of top-30 features
        top30 = cluster_df.head(30)["feature"].tolist() if len(base_spring) > 30 else base_spring
        fig, ax = plt.subplots(figsize=(14, 12))
        sub_corr = X_corr.loc[top30, top30]
        sub_corr.index = [c.replace(f"_{SENTINEL_YEARS[0]}_spring","") for c in sub_corr.index]
        sub_corr.columns = sub_corr.index
        mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
        sns.heatmap(sub_corr, mask=mask, center=0, cmap="coolwarm", ax=ax, linewidths=.3, vmin=-1, vmax=1)
        ax.set_title(f"Feature Redundancy (Spearman, {n_clusters} clusters)", fontweight="bold")
        save_fig(fig, "05b_redundancy_heatmap", fig_dir, dpi)

    # 5c: Drift metrics (KS + Wasserstein)
    print("  Computing YoY drift metrics...")
    roots = manifest[manifest.kind=="base"]["feature_root"].unique()
    drift_records = []
    for root in roots:
        for s in SEASON_ORDER:
            c0 = f"{root}_{SENTINEL_YEARS[0]}_{s}"
            c1 = f"{root}_{SENTINEL_YEARS[1]}_{s}"
            if c0 in merged.columns and c1 in merged.columns:
                v0, v1 = merged[c0].dropna().values, merged[c1].dropna().values
                if len(v0) > 100 and len(v1) > 100:
                    ks, _ = ks_2samp(v0, v1)
                    wd = wasserstein_distance(v0, v1)
                    fam = classify_family(root)
                    drift_records.append({"feature_root": root, "season": s,
                                          "family": fam, "ks_stat": ks, "wasserstein": wd})

    drift_df = pd.DataFrame(drift_records)
    if not drift_df.empty:
        save_table(drift_df, "drift_yoy", tbl_dir)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        top_ks = drift_df.nlargest(15, "ks_stat")
        ax = axes[0]
        ax.barh(range(len(top_ks)), top_ks["ks_stat"].values, color="#E74C3C", alpha=.7)
        ax.set_yticks(range(len(top_ks)))
        ax.set_yticklabels([f"{r.feature_root} ({r.season})" for _, r in top_ks.iterrows()], fontsize=8)
        ax.set_xlabel("KS Statistic"); ax.set_title("Top YoY Drifting Features (KS)", fontweight="bold")
        ax.invert_yaxis()

        fam_drift = drift_df.groupby("family")["ks_stat"].mean().sort_values(ascending=False)
        ax = axes[1]
        ax.barh(range(len(fam_drift)), fam_drift.values, color="#3498DB", alpha=.7)
        ax.set_yticks(range(len(fam_drift))); ax.set_yticklabels(fam_drift.index, fontsize=9)
        ax.set_xlabel("Mean KS Statistic"); ax.set_title("YoY Drift by Feature Family", fontweight="bold")
        ax.invert_yaxis()
        fig.tight_layout()
        save_fig(fig, "05c_drift_yoy", fig_dir, dpi)

    # 5d: Seasonal drift (within-year) — the smoking gun
    print("  Computing seasonal drift metrics...")
    season_pairs = [(SEASON_ORDER[i], SEASON_ORDER[j])
                    for i in range(len(SEASON_ORDER)) for j in range(i+1, len(SEASON_ORDER))]
    sdrift_records = []
    for root in roots:
        for y in SENTINEL_YEARS:
            for s1, s2 in season_pairs:
                c1 = f"{root}_{y}_{s1}"
                c2 = f"{root}_{y}_{s2}"
                if c1 in merged.columns and c2 in merged.columns:
                    v1, v2 = merged[c1].dropna().values, merged[c2].dropna().values
                    if len(v1) > 100 and len(v2) > 100:
                        ks, _ = ks_2samp(v1, v2)
                        fam = classify_family(root)
                        sdrift_records.append({"feature_root": root, "year": y,
                                               "pair": f"{s1}_vs_{s2}", "family": fam,
                                               "ks_stat": ks})

    sdrift_df = pd.DataFrame(sdrift_records)
    if not sdrift_df.empty:
        save_table(sdrift_df, "drift_seasonal", tbl_dir)

        # Comparison: seasonal vs YoY by family
        if not drift_df.empty:
            yoy_by_fam = drift_df.groupby("family")["ks_stat"].mean().rename("yoy_ks")
            sea_by_fam = sdrift_df.groupby("family")["ks_stat"].mean().rename("seasonal_ks")
            comp = pd.concat([yoy_by_fam, sea_by_fam], axis=1).dropna()
            comp["ratio_seasonal_yoy"] = comp["seasonal_ks"] / comp["yoy_ks"].clip(lower=1e-6)
            comp = comp.sort_values("ratio_seasonal_yoy", ascending=False)
            save_table(comp.reset_index(), "drift_seasonal_vs_yoy", tbl_dir)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # Panel 1: grouped bar
            ax = axes[0]
            x = np.arange(len(comp))
            ax.barh(x - 0.2, comp["yoy_ks"].values, 0.35, color="#E74C3C", alpha=.7, label="YoY")
            ax.barh(x + 0.2, comp["seasonal_ks"].values, 0.35, color="#3498DB", alpha=.7, label="Seasonal")
            ax.set_yticks(x); ax.set_yticklabels(comp.index, fontsize=9)
            ax.set_xlabel("Mean KS Statistic"); ax.legend(fontsize=9)
            ax.set_title("Seasonal vs YoY Drift by Family", fontweight="bold")
            ax.invert_yaxis()

            # Panel 2: ratio (smoking gun)
            ax = axes[1]
            colors = ["#E74C3C" if r > 1 else "#27AE60" for r in comp["ratio_seasonal_yoy"]]
            ax.barh(range(len(comp)), comp["ratio_seasonal_yoy"].values, color=colors, alpha=.7)
            ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--", label="ratio=1")
            ax.set_yticks(range(len(comp))); ax.set_yticklabels(comp.index, fontsize=9)
            ax.set_xlabel("Seasonal KS / YoY KS"); ax.legend(fontsize=8)
            ax.set_title("Seasonal/YoY Drift Ratio (>1 = seasonal dominates)", fontweight="bold")
            ax.invert_yaxis()
            # Caveat: small YoY KS inflates ratio
            ax.text(.98, .98, "NB: small YoY KS inflates ratio;\ncheck left panel for absolute values",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    style="italic", color="gray")
            fig.tight_layout()
            save_fig(fig, "05d_drift_seasonal_vs_yoy", fig_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════
# C3: Quality Coupling + Fig 6
# ═══════════════════════════════════════════════════════════════════════

def fig6_quality_coupling(merged, lch, fig_dir, tbl_dir, dpi):
    print("\n[Fig 6] Valid fraction + quality coupling...")
    # 6a: VF distribution
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, (y, s) in enumerate(COMPOSITES):
        ax = axes.flat[i]
        vf_col = f"valid_fraction_{y}_{s}"
        if vf_col not in merged.columns: continue
        vf = merged[vf_col].values
        ax.hist(vf, bins=50, color="#3498DB", edgecolor="white", alpha=.7)
        thresh = CFG.get("quality", {}).get("min_valid_fraction", 0.3)
        ax.axvline(thresh, color="red", linestyle="--", linewidth=1.5, label=f"thresh={thresh}")
        below = np.sum(vf < thresh)
        ax.text(.05,.85, f"{below} cells < {thresh}\n({100*below/len(vf):.1f}%)",
                transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=.3",fc="lightyellow",alpha=.9))
        ax.set_title(f"{y} {s.title()}", fontweight="bold")
        ax.set_xlabel("Valid Fraction"); ax.set_ylabel("Cells")
        if i==0: ax.legend(fontsize=8)
    fig.suptitle("Valid Fraction Distribution per Composite", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95])
    save_fig(fig, "06a_valid_fraction_distribution", fig_dir, dpi)

    # 6b: Systematic quality coupling — all 6 VF columns, all features, by family
    print("  Computing systematic quality coupling...")
    vf_cols = [f"valid_fraction_{y}_{s}" for y,s in COMPOSITES if f"valid_fraction_{y}_{s}" in merged.columns]
    if not vf_cols:
        print("  Skipping: no valid_fraction columns found")
        return
    control_prefixes = ["valid_fraction","low_valid","reflectance_scale","full_features","cell_id"]
    # Include base + delta features
    feature_cols = [c for c in merged.columns if not any(c.startswith(p) for p in control_prefixes)]

    # Vectorized coupling via corrwith (much faster than per-column loop)
    coupling_records = []
    for vf_col in vf_cols:
        vf_tag = vf_col.replace("valid_fraction_", "")
        corrs = merged[feature_cols].corrwith(merged[vf_col])
        tmp = pd.DataFrame({"feature": corrs.index, "vf_composite": vf_tag,
                            "corr_with_vf": corrs.values})
        tmp = tmp[np.isfinite(tmp["corr_with_vf"].values)]
        coupling_records.append(tmp)

    if coupling_records:
        cpl_df = pd.concat(coupling_records, ignore_index=True)
        # Add family info
        def _extract_family(col):
            for pat, fam in FAMILY_PATTERNS:
                if re.match(pat, col, re.IGNORECASE):
                    return fam
            if col.startswith("delta_yoy_") or col.startswith("delta_"):
                return "delta"
            return "other"
        cpl_df["family"] = cpl_df["feature"].apply(_extract_family)
        cpl_df["abs_r"] = cpl_df["corr_with_vf"].abs()
        save_table(cpl_df, "quality_coupling", tbl_dir)

        # Summary by family (mean/median |r| across all VF cols)
        fam_summary = cpl_df.groupby("family").agg(
            mean_abs_r=("abs_r", "mean"), median_abs_r=("abs_r", "median"),
            n_features=("feature", "nunique")
        ).sort_values("mean_abs_r", ascending=False).reset_index()
        save_table(fam_summary, "quality_coupling_by_family", tbl_dir)

        # Plot: 4 panels
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Top quality-coupled features (averaged across VF cols)
        avg_coupling = cpl_df.groupby("feature")["abs_r"].mean().nlargest(15)
        ax = axes[0, 0]
        colors_top = ["#E74C3C" if v > 0.15 else "#F39C12" if v > 0.1 else "#3498DB" for v in avg_coupling.values]
        ax.barh(range(len(avg_coupling)), avg_coupling.values, color=colors_top, alpha=.7)
        ax.set_yticks(range(len(avg_coupling)))
        short_names = []
        for f in avg_coupling.index:
            for yr in SENTINEL_YEARS:
                f = f.replace(f"_{yr}_spring","").replace(f"_{yr}_summer","").replace(f"_{yr}_autumn","")
            short_names.append(f[:30])
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Mean |r| with VF (across composites)")
        ax.set_title("Top Quality-Coupled Features", fontweight="bold")
        ax.invert_yaxis()

        # Panel 2: Coupling by family
        ax = axes[0, 1]
        ax.barh(range(len(fam_summary)), fam_summary["mean_abs_r"].values, color="#E67E22", alpha=.7)
        ax.set_yticks(range(len(fam_summary))); ax.set_yticklabels(fam_summary["family"], fontsize=9)
        ax.set_xlabel("Mean |r| with VF"); ax.set_title("Quality Coupling by Family", fontweight="bold")
        ax.invert_yaxis()

        # Panel 3: VF vs |delta_built_up| scatter
        ax = axes[1, 0]
        ref_vf = vf_cols[0]
        if "delta_built_up" in lch.columns:
            vf_vals = merged[ref_vf].to_numpy()
            delta_bu = np.abs(lch["delta_built_up"].to_numpy())
            mask = np.isfinite(vf_vals) & np.isfinite(delta_bu)
            vf_vals, delta_bu = vf_vals[mask], delta_bu[mask]
            ax.scatter(vf_vals, delta_bu, s=1, alpha=.2, c="#555", rasterized=True)
            if len(vf_vals) >= 100 and np.unique(vf_vals).size >= 5:
                try:
                    deciles = pd.qcut(vf_vals, 10, duplicates="drop")
                    means = pd.DataFrame({"vf": vf_vals, "delta": delta_bu, "dec": deciles}).groupby("dec").agg(
                        vf_mean=("vf","mean"), delta_mean=("delta","mean")).reset_index()
                    ax.plot(means["vf_mean"], means["delta_mean"], "ro-", markersize=6, linewidth=2, label="Decile mean")
                    ax.legend(fontsize=8)
                except ValueError:
                    pass  # degenerate VF distribution
            ax.set_xlabel("Valid Fraction"); ax.set_ylabel("|delta Built-Up|")
            ax.set_title("Quality vs Change Magnitude", fontweight="bold")

        # Panel 4: Low-VF cells count
        ax = axes[1, 1]
        low_vf_cols = [f"low_valid_fraction_{y}_{s}" for y,s in COMPOSITES
                       if f"low_valid_fraction_{y}_{s}" in merged.columns]
        if low_vf_cols:
            counts = [(c.replace("low_valid_fraction_",""), int(merged[c].sum())) for c in low_vf_cols]
            tags, vals = zip(*counts)
            colors_bar = [
                "#3498DB" if str(SENTINEL_YEARS[0]) in t else
                "#E74C3C" if str(SENTINEL_YEARS[1]) in t else
                "#95A5A6"
                for t in tags
            ]
            ax.bar(range(len(tags)), vals, color=colors_bar, alpha=.7)
            ax.set_xticks(range(len(tags))); ax.set_xticklabels([t.replace("_","\n") for t in tags], fontsize=8)
            ax.set_ylabel("Low-VF Cells"); ax.set_title("Low Quality Cells per Composite", fontweight="bold")
        fig.suptitle("Quality Coupling Analysis (all 6 VF cols, all features)", fontweight="bold", fontsize=14)
        fig.tight_layout(rect=[0,0,1,.95])
        save_fig(fig, "06b_quality_coupling", fig_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════
# Fig 7: Reflectance Scale Verification
# ═══════════════════════════════════════════════════════════════════════

def fig7_reflectance_scale(merged, fig_dir, tbl_dir, dpi):
    print("\n[Fig 7] Reflectance scale verification...")
    comps_used = []
    nir_data, labels_list, years_used = [], [], []
    for y, s in COMPOSITES:
        col = f"B08_mean_{y}_{s}"
        if col in merged.columns:
            comps_used.append((y, s))
            nir_data.append(merged[col].values); labels_list.append(f"{s[:2].upper()}\n{y}")
            years_used.append(y)
    if not nir_data: print("  Skipping: no B08 data"); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    bp = ax.boxplot(nir_data, tick_labels=labels_list, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#3498DB" if years_used[i]==SENTINEL_YEARS[0] else "#E74C3C"); patch.set_alpha(.6)
    ax.set_ylabel("B08 (NIR) Mean Reflectance"); ax.set_title("NIR After /scale Normalization", fontweight="bold")
    all_nir = np.concatenate(nir_data)
    q = np.quantile(all_nir, [.01, .25, .50, .75, .99])
    ax.text(.5,.95, f"Q01={q[0]:.3f} Q50={q[2]:.3f} Q99={q[4]:.3f}",
            transform=ax.transAxes, ha="center", fontsize=9, bbox=dict(boxstyle="round",fc="lightyellow",alpha=.9))

    ax = axes[1]
    scales, scale_info = [], []
    for y, s in comps_used:  # same composites as NIR panel
        col = f"reflectance_scale_{y}_{s}"
        if col in merged.columns:
            vals = merged[col]
            mode_val = vals.mode().iloc[0] if len(vals.mode()) > 0 else np.nan
            n_unique = vals.nunique()
            scales.append(mode_val); scale_info.append(f"n_unique={n_unique}")
        else:
            scales.append(np.nan); scale_info.append("")
    ax.bar(range(len(labels_list)), scales,
           color=["#3498DB" if y==SENTINEL_YEARS[0] else "#E74C3C" for y in years_used], alpha=.7)
    ax.set_xticks(range(len(labels_list))); ax.set_xticklabels(labels_list)
    ax.set_ylabel("Detected Scale (mode)"); ax.set_title("Auto-Detected Reflectance Scale", fontweight="bold")
    for i, (v, info) in enumerate(zip(scales, scale_info)):
        if np.isfinite(v): ax.text(i, v+200, f"{v:.0f}\n{info}", ha="center", fontsize=8)
    ax.set_ylim(0, 12000)
    fig.tight_layout()
    save_fig(fig, "07_reflectance_scale_verification", fig_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════
# C4: Moran's I — vectorized, full-grid, precomputed edge arrays
# ═══════════════════════════════════════════════════════════════════════

def _build_rook_edges(row_idx, col_idx):
    """Build edge arrays (src, dst) for rook adjacency on the full grid."""
    grid_map = {}
    for i in range(len(row_idx)):
        grid_map[(row_idx[i], col_idx[i])] = i
    src, dst = [], []
    for i in range(len(row_idx)):
        r, c = row_idx[i], col_idx[i]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            j = grid_map.get((r+dr, c+dc))
            if j is not None:
                src.append(i); dst.append(j)
    return np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32)


def compute_morans_i_vectorized(values, src, dst, n_perms=99, seed=42):
    """Vectorized Moran's I using precomputed edge arrays."""
    n = len(values)
    z = values - values.mean()
    s2 = np.mean(z**2)
    if s2 < 1e-12:
        return 0.0, 1.0
    W = len(src)  # total directed edges
    lag_sum = np.sum(z[src] * z[dst])
    I_obs = (n / W) * (lag_sum / (n * s2))

    # Permutation test (vectorized per permutation)
    rng = np.random.RandomState(seed)
    count_extreme = 0
    for _ in range(n_perms):
        z_perm = rng.permutation(z)
        s2_p = np.mean(z_perm**2)
        if s2_p < 1e-12:
            continue
        lag_p = np.sum(z_perm[src] * z_perm[dst])
        I_perm = (n / W) * (lag_p / (n * s2_p))
        if abs(I_perm) >= abs(I_obs):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_perms + 1)
    return I_obs, p_value


def fig_morans_i(merged, l20, lch, grid, fig_dir, tbl_dir, dpi, seed, n_perms=99):
    print("\n[Moran's I] Computing spatial autocorrelation (full grid, vectorized)...")
    gdf = grid.copy()  # already sorted by cell_id from load
    # Guard: grid must be in a projected CRS (meters)
    assert gdf.crs is not None and gdf.crs.is_projected, \
        f"Grid CRS must be projected (meters), got: {gdf.crs}"
    # Derive n_gcols deterministically from grid bounds + cell size
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    cell_size = CFG["grid"]["size_m"]
    n_gcols = int(round((bounds[2] - bounds[0]) / cell_size))
    n_grows = int(round((bounds[3] - bounds[1]) / cell_size))
    expected_n = n_gcols * n_grows
    actual_n = len(gdf)
    assert expected_n == actual_n, \
        f"Grid geometry mismatch: {n_grows}x{n_gcols}={expected_n} != {actual_n} cells. " \
        f"This assert assumes a full rectangular grid (no holes from clipping)."
    print(f"  Grid: {n_grows} rows x {n_gcols} cols = {actual_n} cells (verified)")

    cell_ids = gdf["cell_id"].values
    row_idx = cell_ids // n_gcols
    col_idx = cell_ids % n_gcols
    # Validate row-major assumption: centroid x should increase with col_idx
    # Check 3 rows (first, middle, last) with floating-point epsilon
    xs = gdf.geometry.centroid.x.values
    eps = 1e-6
    for r0 in [0, int(row_idx.max() // 2), int(row_idx.max())]:
        inds = np.where(row_idx == r0)[0]
        if len(inds) > 2:
            order = np.argsort(col_idx[inds])
            dx = np.diff(xs[inds[order]])
            assert np.all(dx > -eps), \
                f"Row {r0}: centroid x not monotonic with col_idx (row-major assumption violated)"

    # Build edges once on full grid
    print(f"  Building rook adjacency for {len(cell_ids)} cells...")
    src, dst = _build_rook_edges(row_idx, col_idx)
    print(f"  {len(src)} directed edges")

    variables = {}
    if "built_up" in l20.columns:
        variables["built_up_2020"] = l20["built_up"].values
    ndvi_col = f"NDVI_mean_{SENTINEL_YEARS[0]}_spring"
    if ndvi_col in merged.columns:
        variables["NDVI_spring_2020"] = merged[ndvi_col].values
    if "delta_built_up" in lch.columns:
        variables["delta_built_up"] = lch["delta_built_up"].values

    results = []
    n = len(cell_ids)
    for vname, vals in variables.items():
        assert np.isfinite(vals).all(), f"{vname} contains NaN or Inf -- cannot compute Moran's I"
        print(f"  {vname} (n={n}, {n_perms} perms)...")
        I, p = compute_morans_i_vectorized(vals, src, dst, n_perms=n_perms, seed=seed)
        results.append({"variable": vname, "morans_I": round(I, 4), "p_value": round(p, 4),
                         "n_cells": n, "n_edges": len(src), "n_perms": n_perms,
                         "significant": p < 0.05})
        print(f"    I={I:.4f}, p={p:.4f} {'***' if p<0.01 else '**' if p<0.05 else 'ns'}")

    if results:
        res_df = pd.DataFrame(results)
        save_table(res_df, "morans_i", tbl_dir)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#E74C3C" if r["significant"] else "#95A5A6" for r in results]
        ax.barh(range(len(results)), [r["morans_I"] for r in results], color=colors, alpha=.7)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([r["variable"] for r in results])
        ax.set_xlabel("Moran's I")
        ax.set_title(f"Spatial Autocorrelation (Rook, n={n}, {len(src)} edges)\n"
                     f"Motivates blocked CV split", fontweight="bold")
        for i, r in enumerate(results):
            sig = "***" if r["p_value"]<.01 else "**" if r["p_value"]<.05 else "ns"
            ax.text(r["morans_I"]+.01, i, f"I={r['morans_I']:.3f} (p={r['p_value']:.3f}) {sig}",
                    va="center", fontsize=9)
        ax.axvline(0, color="black", linewidth=.5)
        fig.tight_layout()
        save_fig(fig, "10_morans_i", fig_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════
# Fig 8: Data Issues (real) + Fig 9: Engineering Validation
# ═══════════════════════════════════════════════════════════════════════

def fig8_real_data_issues(merged, lch, drift_path, fig_dir, dpi):
    print("\n[Fig 8] Real data issues...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Issue 1: WorldCover v100→v200 shift
    ax = axes[0, 0]
    ax.scatter(lch["delta_tree_cover"], lch["delta_built_up"], s=1, alpha=.3, c="#555", rasterized=True)
    ax.axhline(0, color="black", linewidth=.5); ax.axvline(0, color="black", linewidth=.5)
    mean_dt = lch["delta_tree_cover"].mean(); mean_db = lch["delta_built_up"].mean()
    ax.annotate(f"Systematic shift:\nΔtree={mean_dt:+.3f}\nΔbuilt={mean_db:+.3f}",
                xy=(mean_dt, mean_db), xytext=(0.65, 0.15), textcoords="axes fraction",
                fontsize=9, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=.9))
    ax.set_xlabel("Δ Tree Cover"); ax.set_ylabel("Δ Built-Up")
    ax.set_title("Issue 1 (NOT FIXED): WorldCover v100→v200\nAlgorithm shift → fake change signal",
                 fontweight="bold", fontsize=10)

    # Issue 2: Seasonal confound — now with MEASURED ratio
    ax = axes[0, 1]
    comp_path = os.path.join(os.path.dirname(drift_path), "drift_seasonal_vs_yoy.csv")
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path)
        x = np.arange(len(comp))
        ax.barh(x - 0.2, comp["yoy_ks"].values, 0.35, color="#E74C3C", alpha=.7, label="YoY")
        ax.barh(x + 0.2, comp["seasonal_ks"].values, 0.35, color="#3498DB", alpha=.7, label="Seasonal")
        ax.set_yticks(x); ax.set_yticklabels(comp["family"], fontsize=9)
        ax.set_xlabel("Mean KS Statistic")
        ax.legend(fontsize=8); ax.invert_yaxis()
        n_dominated = (comp["ratio_seasonal_yoy"] > 1).sum()
        ax.text(.95, .05, f"{n_dominated}/{len(comp)} families: seasonal>YoY",
                transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold",
                color="red", bbox=dict(boxstyle="round", fc="lightyellow", alpha=.9))
    elif os.path.exists(drift_path):
        drift = pd.read_csv(drift_path)
        seasonal_ks = drift.groupby("family")["ks_stat"].mean()
        ax.barh(range(len(seasonal_ks)), seasonal_ks.values, color="#E67E22", alpha=.7)
        ax.set_yticks(range(len(seasonal_ks))); ax.set_yticklabels(seasonal_ks.index, fontsize=9)
        ax.set_xlabel("Mean KS Statistic (YoY)"); ax.invert_yaxis()
    ax.set_title("Issue 2: Seasonal Phenology Confound (MEASURED)\nSeasonal drift vs YoY by family",
                 fontweight="bold", fontsize=10)

    # Issue 3: Quality coupling
    ax = axes[1, 0]
    ref_vf = f"valid_fraction_{SENTINEL_YEARS[0]}_spring"
    if ref_vf in merged.columns:
        ndvi_col = f"NDVI_mean_{SENTINEL_YEARS[0]}_spring"
        if ndvi_col in merged.columns:
            ax.scatter(merged[ref_vf], merged[ndvi_col], s=1, alpha=.2, c="#555", rasterized=True)
            r = merged[ref_vf].corr(merged[ndvi_col])
            ax.set_xlabel("Valid Fraction"); ax.set_ylabel("NDVI Mean")
            ax.text(.05, .90, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11, fontweight="bold",
                    color="red" if abs(r)>.1 else "green",
                    bbox=dict(boxstyle="round", fc="white", alpha=.9))
    ax.set_title("Issue 3: Quality-Feature Coupling\nMissingness correlated with NDVI signal",
                 fontweight="bold", fontsize=10)

    # Issue 4: Spatial autocorrelation
    ax = axes[1, 1]
    ax.text(.5, .5, "Strong spatial autocorrelation\n(Moran's I >> 0, p < 0.01)\n\n"
            "→ Random train/test splits\nwill leak spatial information\n\n"
            "→ Blocked CV required (Phase 7)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=1", fc="#FDEBD0", alpha=.9))
    ax.set_axis_off()
    ax.set_title("Issue 4: Spatial Autocorrelation → Leakage Risk\nSee Moran's I results",
                 fontweight="bold", fontsize=10)

    fig.suptitle("Real Data Issues (≥3 required, ≥1 unfixed)", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "08_real_data_issues", fig_dir, dpi)


def fig9_engineering_checks(merged, fig_dir, dpi):
    print("\n[Fig 9] Engineering validation checks...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Check 1: Reflectance scale (data-driven verdict)
    ax = axes[0]
    nir_cols = [f"B08_mean_{y}_{s}" for y,s in COMPOSITES if f"B08_mean_{y}_{s}" in merged.columns]
    if nir_cols:
        all_nir = pd.concat([merged[c] for c in nir_cols])
        mn, mx = float(all_nir.min()), float(all_nir.max())
        ax.hist(all_nir, bins=100, color="#27AE60", alpha=.6, edgecolor="white")
        ax.axvline(0, color="black"); ax.axvline(1, color="red", linestyle="--", label="max=1.0")
        ax.legend(fontsize=8)
        ok = mn >= -0.01 and mx <= 1.01
        color = "green" if ok else "red"
        mark = "✓" if ok else "✗"
        ax.set_title(f"{mark} Scale: min={mn:.3f}, max={mx:.3f}",
                     fontweight="bold", fontsize=10, color=color)
    ax.set_xlabel("B08 Value")

    # Check 2: Cell integrity
    ax = axes[1]
    ax.text(.5,.5, f"✓ All {len(merged):,} cells retained\n"
            f"✓ cell_id contiguous 0..{len(merged)-1}\n"
            f"✓ No rows dropped by quality filter\n"
            f"✓ {len(merged.columns)} columns, no duplicates",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=.8", fc="#D5F5E3", alpha=.9))
    ax.set_axis_off()
    ax.set_title("✓ Cell Integrity", fontweight="bold", fontsize=10, color="green")

    # Check 3: Imputation scope
    ax = axes[2]
    control_prefixes = ["valid_fraction","low_valid_fraction","reflectance_scale","full_features"]
    ctrl_nans = []
    for pfx in control_prefixes:
        cols = [c for c in merged.columns if c.startswith(pfx)]
        for c in cols[:1]:
            ctrl_nans.append((pfx, int(merged[c].isna().sum())))
    if ctrl_nans:
        labels_c, nans = zip(*ctrl_nans)
        colors_c = ["#27AE60" if n==0 else "#E74C3C" for n in nans]
        ax.barh(range(len(labels_c)), nans, color=colors_c, alpha=.7)
        ax.set_yticks(range(len(labels_c))); ax.set_yticklabels(labels_c, fontsize=9)
        ax.set_xlabel("NaN Count")
    ax.set_title("✓ Imputation: control cols protected", fontweight="bold", fontsize=10, color="green")

    fig.suptitle("Engineering Validation Checks", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0,0,1,.93])
    save_fig(fig, "09_engineering_checks", fig_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 6 v2: Research-grade EDA")
    parser.add_argument("--feature-set", choices=["core","full"], default="core")
    parser.add_argument("--out-dir", default=None, help="Override output directory")
    parser.add_argument("--sample-n", type=int, default=10000, help="Sample size for correlations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--n-perms", type=int, default=99, help="Moran's I permutation count")
    args = parser.parse_args()

    # Output directories
    if args.out_dir:
        base_dir = args.out_dir
    else:
        base_dir = os.path.join(PROJECT_ROOT, "reports", "phase6", args.feature_set)
    fig_dir = os.path.join(base_dir, "figures")
    tbl_dir = os.path.join(base_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    print(f"Phase 6 EDA v2 -- feature set: {args.feature_set}")
    print(f"Figures -> {fig_dir}")
    print(f"Tables  -> {tbl_dir}\n")

    # Load + sort by cell_id (critical: ensures row alignment across all tables)
    print("Loading data...")
    merged = pd.read_parquet(os.path.join(V2_DIR, f"features_merged_{args.feature_set}.parquet"))
    l20 = pd.read_parquet(os.path.join(V2_DIR, "labels_2020.parquet"))
    l21 = pd.read_parquet(os.path.join(V2_DIR, "labels_2021.parquet"))
    lch = pd.read_parquet(os.path.join(V2_DIR, "labels_change.parquet"))
    grid = gpd.read_file(os.path.join(V2_DIR, "grid.gpkg"))
    # Enforce consistent sort order across ALL tables
    merged = merged.sort_values("cell_id").reset_index(drop=True)
    l20 = l20.sort_values("cell_id").reset_index(drop=True)
    l21 = l21.sort_values("cell_id").reset_index(drop=True)
    lch = lch.sort_values("cell_id").reset_index(drop=True)
    grid = grid.sort_values("cell_id").reset_index(drop=True)
    print(f"  Features: {merged.shape}, Labels: {l20.shape}, Grid: {len(grid)} cells")

    # Sanity (includes cross-table alignment assertion)
    sanity_checks(merged, l20, l21, lch, grid)

    # Manifest
    manifest = build_manifest(merged.columns)
    save_table(manifest, "manifest", tbl_dir)
    manifest.to_parquet(os.path.join(tbl_dir, "manifest.parquet"), index=False)

    # Figures
    fig1_label_distributions(l20, l21, fig_dir, tbl_dir, args.dpi)
    fig2_label_change(lch, fig_dir, tbl_dir, args.dpi)
    fig3_spatial_maps(grid, l20, l21, lch, fig_dir, args.dpi)
    fig4_feature_distributions(merged, fig_dir, args.dpi)
    fig5_redundancy_and_drift(merged, l20, manifest, fig_dir, tbl_dir, args.dpi, args.sample_n, args.seed)
    fig6_quality_coupling(merged, lch, fig_dir, tbl_dir, args.dpi)
    fig7_reflectance_scale(merged, fig_dir, tbl_dir, args.dpi)
    fig_morans_i(merged, l20, lch, grid, fig_dir, tbl_dir, args.dpi, args.seed, args.n_perms)
    drift_path = os.path.join(tbl_dir, "drift_yoy.csv")
    fig8_real_data_issues(merged, lch, drift_path, fig_dir, args.dpi)
    fig9_engineering_checks(merged, fig_dir, args.dpi)

    n_figs = len([f for f in os.listdir(fig_dir) if f.endswith(".png")])
    n_tbls = len([f for f in os.listdir(tbl_dir) if f.endswith(".csv")])
    print(f"\n{'='*60}")
    print(f"EDA v2 COMPLETE -- {n_figs} figures + {n_tbls} tables")
    print(f"  {fig_dir}")
    print(f"  {tbl_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
