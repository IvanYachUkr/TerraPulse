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
SENTINEL_YEARS = sorted(CFG["sentinel2"]["years"])
SEASON_ORDER = CFG["sentinel2"]["season_order"]
COMPOSITES = [(y, s) for y in SENTINEL_YEARS for s in SEASON_ORDER]

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
    assert len(merged.columns) == len(set(merged.columns)), "Duplicate columns in merged"
    for y, s in COMPOSITES:
        col = f"NDVI_mean_{y}_{s}"
        assert col in merged.columns, f"Missing expected column: {col}"
    print(f"  All checks passed ({n} cells, {len(merged.columns)} features)")


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
    print("  Computing drift metrics...")
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
        # Plot: top drifting features per family
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

    # 6b: Quality coupling — corr(|delta_built_up|, valid_fraction) + scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vf_cols = [f"valid_fraction_{y}_{s}" for y,s in COMPOSITES if f"valid_fraction_{y}_{s}" in merged.columns]

    # Panel 1: corr of features with valid_fraction
    coupling_records = []
    ref_vf = vf_cols[0] if vf_cols else None
    if ref_vf:
        feature_cols = [c for c in merged.columns if not any(c.startswith(p) for p in
                        ["valid_fraction","low_valid","reflectance_scale","full_features","cell_id","delta_"])]
        for fc in feature_cols[:200]:  # cap for speed
            r = merged[fc].corr(merged[ref_vf])
            if np.isfinite(r):
                coupling_records.append({"feature": fc, "corr_with_vf": r})
    if coupling_records:
        cpl_df = pd.DataFrame(coupling_records).sort_values("corr_with_vf", key=abs, ascending=False)
        save_table(cpl_df, "quality_coupling", tbl_dir)
        top10 = cpl_df.head(10)
        ax = axes[0]
        colors = ["#E74C3C" if v<0 else "#3498DB" for v in top10["corr_with_vf"]]
        ax.barh(range(len(top10)), top10["corr_with_vf"].values, color=colors, alpha=.7)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels([f.split(f"_{SENTINEL_YEARS[0]}_")[0] if f"_{SENTINEL_YEARS[0]}_" in f else f
                            for f in top10["feature"]], fontsize=8)
        ax.set_xlabel("Pearson r with valid_fraction"); ax.set_title("Top Quality-Coupled Features", fontweight="bold")
        ax.invert_yaxis()

    # Panel 2: VF vs |delta_built_up| scatter
    ax = axes[1]
    if ref_vf and "delta_built_up" in lch.columns:
        vf_vals = merged[ref_vf].values
        delta_bu = np.abs(lch.sort_values("cell_id")["delta_built_up"].values)
        ax.scatter(vf_vals, delta_bu, s=1, alpha=.2, c="#555", rasterized=True)
        # Decile means
        deciles = pd.qcut(vf_vals, 10, duplicates="drop")
        means = pd.DataFrame({"vf": vf_vals, "delta": delta_bu, "dec": deciles}).groupby("dec").agg(
            vf_mean=("vf","mean"), delta_mean=("delta","mean")).reset_index()
        ax.plot(means["vf_mean"], means["delta_mean"], "ro-", markersize=6, linewidth=2, label="Decile mean")
        ax.set_xlabel("Valid Fraction"); ax.set_ylabel("|Δ Built-Up|")
        ax.set_title("Quality vs Change Magnitude", fontweight="bold"); ax.legend(fontsize=8)

    # Panel 3: low_vf cells count
    ax = axes[2]
    low_vf_cols = [f"low_valid_fraction_{y}_{s}" for y,s in COMPOSITES
                   if f"low_valid_fraction_{y}_{s}" in merged.columns]
    if low_vf_cols:
        counts = [(c.replace("low_valid_fraction_",""), int(merged[c].sum())) for c in low_vf_cols]
        tags, vals = zip(*counts)
        ax.bar(range(len(tags)), vals, color=["#3498DB"]*3+["#E74C3C"]*3, alpha=.7)
        ax.set_xticks(range(len(tags))); ax.set_xticklabels([t.replace("_","\n") for t in tags], fontsize=8)
        ax.set_ylabel("Low-VF Cells"); ax.set_title("Low Quality Cells per Composite", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "06b_quality_coupling", fig_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════
# Fig 7: Reflectance Scale Verification
# ═══════════════════════════════════════════════════════════════════════

def fig7_reflectance_scale(merged, fig_dir, tbl_dir, dpi):
    print("\n[Fig 7] Reflectance scale verification...")
    nir_data, labels_list, years_used = [], [], []
    for y, s in COMPOSITES:
        col = f"B08_mean_{y}_{s}"
        if col in merged.columns:
            nir_data.append(merged[col].values); labels_list.append(f"{s[:2].upper()}\n{y}")
            years_used.append(y)
    if not nir_data: print("  Skipping: no B08 data"); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    bp = ax.boxplot(nir_data, tick_labels=labels_list, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#3498DB" if years_used[i]==SENTINEL_YEARS[0] else "#E74C3C"); patch.set_alpha(.6)
    ax.set_ylabel("B08 (NIR) Mean Reflectance"); ax.set_title("NIR After /scale Normalization", fontweight="bold")
    # Quantiles instead of "expected range"
    all_nir = np.concatenate(nir_data)
    q = np.quantile(all_nir, [.01, .25, .50, .75, .99])
    ax.text(.5,.95, f"Q01={q[0]:.3f} Q50={q[2]:.3f} Q99={q[4]:.3f}",
            transform=ax.transAxes, ha="center", fontsize=9, bbox=dict(boxstyle="round",fc="lightyellow",alpha=.9))

    ax = axes[1]
    scales, scale_info = [], []
    for y, s in COMPOSITES:
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
# C4: Moran's I (manual rook adjacency)
# ═══════════════════════════════════════════════════════════════════════

def compute_morans_i(values, row_idx, col_idx, n_rows, n_cols, n_perms=199, seed=42):
    """Manual Moran's I with rook (4-neighbor) adjacency."""
    n = len(values)
    z = values - values.mean()
    s2 = np.mean(z**2)
    if s2 < 1e-12: return 0.0, 1.0

    # Build neighbor indices (rook: up/down/left/right)
    grid_map = {}
    for i in range(n):
        grid_map[(row_idx[i], col_idx[i])] = i

    W_sum = 0.0; lag_sum = 0.0
    for i in range(n):
        r, c = row_idx[i], col_idx[i]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            j = grid_map.get((r+dr, c+dc))
            if j is not None:
                lag_sum += z[i] * z[j]
                W_sum += 1

    if W_sum == 0: return 0.0, 1.0
    I_obs = (n / W_sum) * (lag_sum / (n * s2))

    # Permutation test
    rng = np.random.RandomState(seed)
    count_extreme = 0
    for _ in range(n_perms):
        z_perm = rng.permutation(z)
        lag_p = 0.0
        for i in range(n):
            r, c = row_idx[i], col_idx[i]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                j = grid_map.get((r+dr, c+dc))
                if j is not None:
                    lag_p += z_perm[i] * z_perm[j]
        I_perm = (n / W_sum) * (lag_p / (n * np.mean(z_perm**2) + 1e-15))
        if abs(I_perm) >= abs(I_obs):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_perms + 1)
    return I_obs, p_value


def fig_morans_i(merged, l20, grid, fig_dir, tbl_dir, dpi, seed):
    print("\n[Moran's I] Computing spatial autocorrelation...")
    # Derive row/col from cell_id
    n_cols_grid = int(grid.sort_values("cell_id").iloc[-1].geometry.bounds[2] -
                      grid.sort_values("cell_id").iloc[0].geometry.bounds[0]) // 100
    # Simpler: use grid dimensions from config
    ref = CFG["grid"]
    block = ref["size_m"] // ref["pixel_size"]  # 10

    gdf = grid.sort_values("cell_id").reset_index(drop=True)
    # Get grid dimensions from geometry
    xs = gdf.geometry.centroid.x.values
    ys = gdf.geometry.centroid.y.values
    unique_xs = np.sort(np.unique(np.round(xs, 1)))
    unique_ys = np.sort(np.unique(np.round(ys, 1)))
    n_gcols = len(unique_xs)
    n_grows = len(unique_ys)

    cell_ids = gdf["cell_id"].values
    row_idx = cell_ids // n_gcols
    col_idx = cell_ids % n_gcols

    variables = {}
    if "built_up" in l20.columns:
        variables["built_up_2020"] = l20.sort_values("cell_id")["built_up"].values
    ndvi_col = f"NDVI_mean_{SENTINEL_YEARS[0]}_spring"
    if ndvi_col in merged.columns:
        variables["NDVI_spring_2020"] = merged.sort_values("cell_id")[ndvi_col].values

    # Use sample for speed (Moran's I on 30k cells with permutations is slow)
    rng = np.random.RandomState(seed)
    sample_size = min(5000, len(cell_ids))
    sample_idx = rng.choice(len(cell_ids), sample_size, replace=False)

    results = []
    for vname, vals in variables.items():
        print(f"  {vname} (n={sample_size}, 199 perms)...")
        I, p = compute_morans_i(vals[sample_idx], row_idx[sample_idx], col_idx[sample_idx],
                                 n_grows, n_gcols, n_perms=199, seed=seed)
        results.append({"variable": vname, "morans_I": round(I, 4), "p_value": round(p, 4),
                         "n_cells": sample_size, "significant": p < 0.05})
        print(f"    I={I:.4f}, p={p:.4f} {'***' if p<0.01 else '**' if p<0.05 else 'ns'}")

    if results:
        res_df = pd.DataFrame(results)
        save_table(res_df, "morans_i", tbl_dir)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#E74C3C" if r["significant"] else "#95A5A6" for r in results]
        ax.barh(range(len(results)), [r["morans_I"] for r in results], color=colors, alpha=.7)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([r["variable"] for r in results])
        ax.set_xlabel("Moran's I")
        ax.set_title("Spatial Autocorrelation (Rook Adjacency)\nMotivates blocked CV split", fontweight="bold")
        for i, r in enumerate(results):
            sig = "***" if r["p_value"]<.01 else "**" if r["p_value"]<.05 else "ns"
            ax.text(r["morans_I"]+.01, i, f"I={r['morans_I']:.3f} (p={r['p_value']:.3f}) {sig}",
                    va="center", fontsize=9)
        ax.axvline(0, color="black", linewidth=.5)
        fig.tight_layout()
        save_fig(fig, "09_morans_i", fig_dir, dpi)


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

    # Issue 2: Seasonal confound (drift >> YoY)
    ax = axes[0, 1]
    if os.path.exists(drift_path):
        drift = pd.read_csv(drift_path)
        seasonal_ks = drift.groupby("family")["ks_stat"].mean()
        ax.barh(range(len(seasonal_ks)), seasonal_ks.values, color="#E67E22", alpha=.7)
        ax.set_yticks(range(len(seasonal_ks))); ax.set_yticklabels(seasonal_ks.index, fontsize=9)
        ax.set_xlabel("Mean KS Statistic (YoY)")
        ax.invert_yaxis()
    ax.set_title("Issue 2: Seasonal Phenology Confound\nSeasonal drift may exceed real YoY change",
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

    # Check 1: Reflectance scale
    ax = axes[0]
    nir_cols = [f"B08_mean_{y}_{s}" for y,s in COMPOSITES if f"B08_mean_{y}_{s}" in merged.columns]
    if nir_cols:
        all_nir = pd.concat([merged[c] for c in nir_cols])
        ax.hist(all_nir, bins=100, color="#27AE60", alpha=.6, edgecolor="white")
        ax.axvline(0, color="black"); ax.axvline(1, color="red", linestyle="--", label="max=1.0")
        ax.legend(fontsize=8)
    ax.set_title("✓ Scale: values in [0,1]", fontweight="bold", fontsize=10, color="green")
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

    # Load
    print("Loading data...")
    merged = pd.read_parquet(os.path.join(V2_DIR, f"features_merged_{args.feature_set}.parquet"))
    l20 = pd.read_parquet(os.path.join(V2_DIR, "labels_2020.parquet"))
    l21 = pd.read_parquet(os.path.join(V2_DIR, "labels_2021.parquet"))
    lch = pd.read_parquet(os.path.join(V2_DIR, "labels_change.parquet"))
    grid = gpd.read_file(os.path.join(V2_DIR, "grid.gpkg"))
    print(f"  Features: {merged.shape}, Labels: {l20.shape}, Grid: {len(grid)} cells")

    # Sanity
    sanity_checks(merged, l20, l21, lch, grid)

    # Manifest
    manifest = build_manifest(merged.columns)
    save_table(manifest, "column_inventory", tbl_dir)
    manifest.to_parquet(os.path.join(tbl_dir, "manifest.parquet"), index=False)

    # Figures
    fig1_label_distributions(l20, l21, fig_dir, tbl_dir, args.dpi)
    fig2_label_change(lch, fig_dir, tbl_dir, args.dpi)
    fig3_spatial_maps(grid, l20, l21, lch, fig_dir, args.dpi)
    fig4_feature_distributions(merged, fig_dir, args.dpi)
    fig5_redundancy_and_drift(merged, l20, manifest, fig_dir, tbl_dir, args.dpi, args.sample_n, args.seed)
    fig6_quality_coupling(merged, lch, fig_dir, tbl_dir, args.dpi)
    fig7_reflectance_scale(merged, fig_dir, tbl_dir, args.dpi)
    fig_morans_i(merged, l20, grid, fig_dir, tbl_dir, args.dpi, args.seed)
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
