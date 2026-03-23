#!/usr/bin/env python3
"""
eda_labels_and_features.py -- Exploratory Data Analysis for the TerraPulse report.

Part 1: Compare ESA WorldCover 2021 labels (from dashboard bins) against
         LUCAS 2022 field-surveyed land cover points within Nuremberg.

Part 2: Per-class distribution analysis of key S2/S1 features from the
         grid-level training data.

Outputs diagrams to: mlp/diagrams/
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_dashboard"
META_PATH = BIN_DIR / "nuremberg_dashboard_meta.json"
LUCAS_PATH = PROJECT_DIR / "DE_LUCAS_2022.csv"
DIAGRAM_DIR = PROJECT_DIR / "mlp" / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
CITIES_DIR = PROJECT_DIR / "data" / "cities"

# Dashboard 6-class scheme (7-class ESA with shrubland merged into grassland)
DASH_CLASSES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
DASH_LABELS  = ["Tree Cover", "Grassland", "Cropland", "Built-up", "Bare/Sparse", "Water"]
DASH_COLORS  = ["#2d6a4f", "#6a994e", "#f4a261", "#e76f51", "#d4a373", "#0096c7"]

# ESA WorldCover 7-class scheme (used in training data)
ESA_CLASSES = ["tree_cover", "shrubland", "grassland", "cropland", "built_up", "bare_sparse", "water"]
ESA_LABELS  = ["Tree Cover", "Shrubland", "Grassland", "Cropland", "Built-up", "Bare/Sparse", "Water"]
ESA_COLORS  = ["#2d6a4f", "#a3b18a", "#6a994e", "#f4a261", "#e76f51", "#d4a373", "#0096c7"]

# LUCAS SURVEY_LC1 codes -> ESA 7-class index
LUCAS_TO_ESA = {}
# A = Artificial land -> 4 (built_up)
for c in ["A10", "A11", "A12", "A13", "A21", "A22", "A30"]:
    LUCAS_TO_ESA[c] = 4
# B = Cropland/agriculture -> 3 (cropland)
for p in ["B1","B2","B3","B4","B5","B7","B8","Bx"]:
    for s in range(10):
        LUCAS_TO_ESA[f"{p}{s}"] = 3
        for sub in ["k","n","j"]:
            LUCAS_TO_ESA[f"{p}{s}{sub}"] = 3
# C = Woodland -> 0 (tree_cover)
for p in ["C1","C2","C3"]:
    for s in range(10):
        LUCAS_TO_ESA[f"{p}{s}"] = 0
# D = Shrubland -> 1
for p in ["D1","D2"]:
    for s in range(10):
        LUCAS_TO_ESA[f"{p}{s}"] = 1
# E = Grassland -> 2
for p in ["E1","E2","E3"]:
    for s in range(10):
        LUCAS_TO_ESA[f"{p}{s}"] = 2
# F = Bare land -> 5
for p in ["F1","F2","F3","F4"]:
    for s in range(10):
        LUCAS_TO_ESA[f"{p}{s}"] = 5
# G/H = Water/Wetlands -> 6
for p in ["G1","G2","G3","H1","H2"]:
    for s in range(10):
        LUCAS_TO_ESA[f"{p}{s}"] = 6

# Dashboard bin mapping (7-class ESA -> 6-class dashboard)
# ESA: 0=tree, 1=shrub, 2=grass, 3=crop, 4=built, 5=bare, 6=water
# Dash: 0=tree, 1=grass(+shrub), 2=crop, 3=built, 4=bare, 5=water
ESA7_TO_DASH6 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def part1_lucas_vs_esa():
    """Compare LUCAS ground truth with ESA WorldCover labels in Nuremberg."""
    print("\n" + "=" * 60)
    print("Part 1: LUCAS 2022 vs ESA WorldCover 2021")
    print("=" * 60)

    meta = json.load(open(META_PATH))
    west, south, east, north = meta["wgs84_bounds"]
    H, W = 2850, 2550

    # Load ESA labels (dashboard 6-class)
    label_bin = np.frombuffer(
        (BIN_DIR / "nuremberg_labels_2021_res1.bin").read_bytes(), dtype=np.uint8
    ).reshape(H, W)

    print("  Loading LUCAS CSV...")
    lucas = pd.read_csv(LUCAS_PATH, usecols=[
        "POINT_LAT", "POINT_LONG", "SURVEY_LC1", "SURVEY_OBS_TYPE"
    ], dtype={"SURVEY_LC1": str})

    # Filter to Nuremberg bbox
    nbg = lucas[
        (lucas.POINT_LAT >= south) & (lucas.POINT_LAT <= north) &
        (lucas.POINT_LONG >= west) & (lucas.POINT_LONG <= east)
    ].copy()
    print(f"  LUCAS points in Nuremberg bbox: {len(nbg)}")

    # Map LUCAS code -> dashboard 6-class
    nbg["lucas_esa7"] = nbg.SURVEY_LC1.map(LUCAS_TO_ESA)
    nbg = nbg.dropna(subset=["lucas_esa7"])
    nbg["lucas_dash6"] = nbg.lucas_esa7.astype(int).map(ESA7_TO_DASH6)
    nbg = nbg.dropna(subset=["lucas_dash6"])
    nbg["lucas_dash6"] = nbg.lucas_dash6.astype(int)
    print(f"  Mapped LUCAS points: {len(nbg)}")

    # Convert to pixel coordinates
    nbg["px_x"] = ((nbg.POINT_LONG - west) / (east - west) * W).astype(int).clip(0, W-1)
    nbg["px_y"] = ((north - nbg.POINT_LAT) / (north - south) * H).astype(int).clip(0, H-1)

    # Look up ESA label
    nbg["esa_label"] = label_bin[nbg.px_y.values, nbg.px_x.values]
    valid = nbg[nbg.esa_label < len(DASH_CLASSES)].copy()
    print(f"  Valid (non-masked) comparisons: {len(valid)}")

    if len(valid) < 3:
        print("  Not enough valid points. Skipping.")
        return

    n_classes = len(DASH_CLASSES)
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for _, row in valid.iterrows():
        conf[int(row.lucas_dash6), int(row.esa_label)] += 1

    # Per-class agreement
    print("\n  Per-class agreement (LUCAS -> ESA):")
    total_agree, total_count = 0, 0
    per_class = {}
    for i, name in enumerate(DASH_LABELS):
        n = conf[i, :].sum()
        agree = conf[i, i]
        rate = agree / max(n, 1) * 100
        per_class[DASH_CLASSES[i]] = {"count": int(n), "agree": int(agree), "rate": round(rate, 1)}
        total_agree += agree
        total_count += n
        if n > 0:
            print(f"    {name:15s}: {agree:3d}/{n:3d} ({rate:5.1f}%)")

    overall = total_agree / max(total_count, 1) * 100
    print(f"    {'OVERALL':15s}: {total_agree:3d}/{total_count:3d} ({overall:5.1f}%)")

    # -- Confusion matrix heatmap --
    fig, ax = plt.subplots(figsize=(8, 6.5))
    conf_pct = conf.astype(float)
    row_sums = conf_pct.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_pct = conf_pct / row_sums * 100

    im = ax.imshow(conf_pct, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(DASH_LABELS, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(DASH_LABELS, fontsize=9)
    ax.set_xlabel("ESA WorldCover 2021 (satellite-derived)", fontweight='bold', fontsize=10)
    ax.set_ylabel("LUCAS 2022 (field survey ground truth)", fontweight='bold', fontsize=10)
    ax.set_title(
        f"Label Agreement: LUCAS 2022 vs ESA WorldCover 2021\n"
        f"Nuremberg  |  n={total_count}  |  Overall agreement: {overall:.1f}%",
        fontsize=12, fontweight='bold')

    for i in range(n_classes):
        for j in range(n_classes):
            val = conf_pct[i, j]
            count = conf[i, j]
            if count > 0:
                color = 'white' if val > 55 else 'black'
                ax.text(j, i, f"{val:.0f}%\n({count})", ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold' if i == j else 'normal')

    plt.colorbar(im, ax=ax, label='Row %', shrink=0.8)
    plt.tight_layout()
    out = DIAGRAM_DIR / "lucas_vs_esa_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out}")

    # Save JSON
    result = {
        "total_points": int(total_count),
        "overall_agreement": round(overall, 1),
        "per_class": per_class,
        "note": "Low n due to most LUCAS points falling outside the Nuremberg city mask (255)."
    }
    json_out = DIAGRAM_DIR / "lucas_vs_esa_agreement.json"
    with open(json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_out}")


### ---------------------------------------------------------------
### Part 2: Feature EDA
### ---------------------------------------------------------------

# Select German cities for feature EDA (closest to Nuremberg)
GERMAN_CITIES = [
    "nuremberg", "munich", "berlin", "dresden", "duesseldorf", "bremen",
    "hamburg", "frankfurt", "hannover", "leipzig", "stuttgart", "koeln",
]


def _dominant_class(row, classes):
    """Return the class index with the highest fraction."""
    vals = [row.get(c, 0) for c in classes]
    return int(np.argmax(vals))


def part2_feature_eda():
    """Per-class feature distributions from training grids."""
    print("\n" + "=" * 60)
    print("Part 2: Feature EDA (Per-Class Distributions)")
    print("=" * 60)

    # Load features and labels for a few cities
    feat_dfs = []
    skipped = []
    loaded_cities = []

    city_dirs = sorted(CITIES_DIR.iterdir()) if CITIES_DIR.exists() else []
    for city_dir in city_dirs:
        feat_path = city_dir / "features_v7" / "features_rust_2020_2021.parquet"
        label_path = city_dir / "labels_2021.parquet"
        if not feat_path.exists() or not label_path.exists():
            continue

        try:
            feat = pd.read_parquet(feat_path)
            lab = pd.read_parquet(label_path)
            # Merge on cell_id
            merged = feat.merge(lab[["cell_id"] + ESA_CLASSES], on="cell_id", how="inner")
            # Dominant class
            merged["label"] = merged.apply(lambda r: _dominant_class(r, ESA_CLASSES), axis=1)
            feat_dfs.append(merged)
            loaded_cities.append(city_dir.name)
        except Exception as e:
            skipped.append((city_dir.name, str(e)))

        if len(feat_dfs) >= 8:  # Limit to 8 cities for memory
            break

    if not feat_dfs:
        print("  No city data loaded. Aborting.")
        return

    df = pd.concat(feat_dfs, ignore_index=True)
    print(f"  Loaded {len(loaded_cities)} cities: {loaded_cities}")
    print(f"  Total samples: {len(df)}")
    print(f"  Feature columns: {df.shape[1]}")

    # Key features to plot (representative of major feature types)
    feature_groups = {
        "Vegetation Indices\n(Summer 2021)": [
            "NDVI_mean_2021_summer",
            "NDWI_mean_2021_summer",
            "NDBI_mean_2021_summer",
            "EVI2_mean_2021_summer",
        ],
        "Vegetation Indices\n(Seasonal, 2021)": [
            "NDVI_mean_2021_spring",
            "NDVI_mean_2021_summer",
            "NDVI_mean_2021_autumn",
        ],
        "SAR Backscatter\n(Summer 2021)": [
            "SAR_VV_mean_2021_summer",
            "SAR_VH_mean_2021_summer",
            "SAR_CR_mean_2021_summer",
        ],
        "Spectral Bands\n(Summer 2021)": [
            "B04_mean_2021_summer",
            "B08_mean_2021_summer",
            "B11_mean_2021_summer",
            "B12_mean_2021_summer",
        ],
    }

    # Check which features actually exist
    avail_cols = set(df.columns)
    for grp, feats in list(feature_groups.items()):
        existing = [f for f in feats if f in avail_cols]
        if not existing:
            print(f"  WARNING: none of {feats[:2]} found for '{grp}'")
            del feature_groups[grp]
        else:
            feature_groups[grp] = existing

    if not feature_groups:
        # List actual columns to debug
        sample_cols = [c for c in df.columns if "NDVI" in c or "ndvi" in c][:10]
        print(f"  Available NDVI cols: {sample_cols}")
        sample_cols = [c for c in df.columns if "SAR" in c][:10]
        print(f"  Available SAR cols: {sample_cols}")
        sample_cols = [c for c in df.columns if "B04" in c][:10]
        print(f"  Available B04 cols: {sample_cols}")
        return

    # Plot each group as a multi-panel figure
    for grp_name, features in feature_groups.items():
        n = len(features)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5.5))
        if n == 1:
            axes = [axes]

        classes = sorted(df.label.unique())
        class_names = [ESA_LABELS[c] if c < len(ESA_LABELS) else f"C{c}" for c in classes]
        colors = [ESA_COLORS[c] if c < len(ESA_COLORS) else "#999" for c in classes]

        for ax, feat in zip(axes, features):
            data = []
            labs = []
            cols = []
            for c, nm, co in zip(classes, class_names, colors):
                vals = df.loc[df.label == c, feat].dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    labs.append(nm)
                    cols.append(co)

            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.65)
            for patch, co in zip(bp['boxes'], cols):
                patch.set_facecolor(co)
                patch.set_alpha(0.75)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)

            ax.set_xticklabels(labs, rotation=45, ha='right', fontsize=7)
            # Prettify feature name
            title = feat.replace("_mean_", " ").replace("2021_", "").replace("2020_", "y1 ")
            title = title.replace("_", " ").title()
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.25)

        fig.suptitle(f"Per-Class Feature Distributions: {grp_name}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        safe = grp_name.lower().replace("\n", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        out = DIAGRAM_DIR / f"eda_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")

    # Extra: Overall class distribution in training data
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df.label.value_counts().sort_index()
    bars = ax.bar(range(len(counts)), counts.values,
                  color=[ESA_COLORS[i] for i in counts.index])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([ESA_LABELS[i] for i in counts.index], rotation=30, ha='right')
    ax.set_ylabel("Number of Grid Cells")
    ax.set_title(f"Training Data Class Distribution ({len(loaded_cities)} cities, {len(df):,} cells)",
                 fontsize=12, fontweight='bold')
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{cnt:,}", ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    out = DIAGRAM_DIR / "eda_class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    part1_lucas_vs_esa()
    part2_feature_eda()
    print("\n--- All EDA complete ---")
