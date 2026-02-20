"""Consolidate multi-year Nuremberg predictions into shareable datasets + visualizations.

Outputs:
  data/nuremberg_timeseries/nuremberg_landcover_2018_2025.parquet
  data/nuremberg_timeseries/nuremberg_landcover_2018_2025.csv
  data/nuremberg_timeseries/nuremberg_summary_by_year.csv
  data/nuremberg_timeseries/viz_stacked_area.png
  data/nuremberg_timeseries/viz_per_class_trends.png
  data/nuremberg_timeseries/viz_dominant_class_change.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = r"C:\Users\vanya\Documents\ML_1_sem\final"
TS_DIR = os.path.join(PROJECT, "data", "nuremberg_timeseries")
PRED_DIR = os.path.join(TS_DIR, "predictions")
LABELS_DIR = os.path.join(PROJECT, "data", "cities", "nuremberg")

CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland", "built_up", "bare_sparse", "water"]
CLASS_COLORS = {
    "tree_cover": "#228B22",
    "shrubland": "#8B6914",
    "grassland": "#90EE90",
    "cropland": "#FFD700",
    "built_up": "#DC143C",
    "bare_sparse": "#D2B48C",
    "water": "#4169E1",
}
CLASS_LABELS = {
    "tree_cover": "Tree Cover",
    "shrubland": "Shrubland",
    "grassland": "Grassland",
    "cropland": "Cropland",
    "built_up": "Built-up",
    "bare_sparse": "Bare/Sparse",
    "water": "Water",
}

# Year pairs → prediction year (second year of pair)
YEAR_PAIRS = [
    ("2017_2018", 2018),
    ("2018_2019", 2019),
    ("2019_2020", 2020),
    ("2020_2021", 2021),
    ("2021_2022", 2022),
    ("2022_2023", 2023),
    ("2023_2024", 2024),
    ("2024_2025", 2025),
]

# ── Step 1: Load all predictions ─────────────────────────────────────
print("=" * 60)
print("Loading predictions...")
print("=" * 60)

all_rows = []
for yp, pred_year in YEAR_PAIRS:
    path = os.path.join(PRED_DIR, f"pred_mlp_{yp}.parquet")
    if not os.path.exists(path):
        print(f"  SKIP {yp} (not found)")
        continue
    df = pd.read_parquet(path)
    print(f"  {yp} -> year {pred_year}: {len(df)} cells, {df.shape[1]} cols")

    mlp_cols = [f"{cls}_mlp" for cls in CLASS_NAMES]
    for col in mlp_cols:
        if col not in df.columns:
            df[col] = 0.0

    for _, row in df.iterrows():
        entry = {
            "cell_id": int(row["cell_id"]),
            "year": pred_year,
            "source": "prediction",
        }
        for cls in CLASS_NAMES:
            entry[cls] = round(float(row[f"{cls}_mlp"]), 4)
        all_rows.append(entry)

# ── Step 2: Load ground-truth labels (2020, 2021) ────────────────────
print("\nLoading ground-truth labels...")
for lbl_year in [2020, 2021]:
    path = os.path.join(LABELS_DIR, f"labels_{lbl_year}.parquet")
    if not os.path.exists(path):
        print(f"  SKIP labels_{lbl_year} (not found)")
        continue
    df = pd.read_parquet(path)
    print(f"  Labels {lbl_year}: {len(df)} cells")
    for _, row in df.iterrows():
        entry = {
            "cell_id": int(row["cell_id"]),
            "year": lbl_year,
            "source": "worldcover_label",
        }
        for cls in CLASS_NAMES:
            entry[cls] = round(float(row.get(cls, 0)), 4)
        all_rows.append(entry)

# ── Step 3: Build consolidated dataframe ──────────────────────────────
print(f"\nConsolidated: {len(all_rows)} rows")
result_df = pd.DataFrame(all_rows)
result_df = result_df.sort_values(["year", "cell_id"]).reset_index(drop=True)

# Save parquet
pq_path = os.path.join(TS_DIR, "nuremberg_landcover_2018_2025.parquet")
result_df.to_parquet(pq_path, index=False)
print(f"  Saved: {pq_path} ({os.path.getsize(pq_path)/1e6:.1f} MB)")

# Save CSV
csv_path = os.path.join(TS_DIR, "nuremberg_landcover_2018_2025.csv")
result_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path} ({os.path.getsize(csv_path)/1e6:.1f} MB)")

# ── Step 4: Summary table ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary by year (predictions only)")
print("=" * 60)

pred_df = result_df[result_df["source"] == "prediction"]
summary = pred_df.groupby("year")[CLASS_NAMES].mean()
print(summary.round(4).to_string())

# Add labels for comparison
lbl_df = result_df[result_df["source"] == "worldcover_label"]
if len(lbl_df) > 0:
    lbl_summary = lbl_df.groupby("year")[CLASS_NAMES].mean()
    print("\nGround truth labels (WorldCover):")
    print(lbl_summary.round(4).to_string())

summary_path = os.path.join(TS_DIR, "nuremberg_summary_by_year.csv")
summary.round(4).to_csv(summary_path)
print(f"\n  Saved: {summary_path}")

# ── Step 5: Visualizations ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Generating visualizations...")
print("=" * 60)

years = sorted(pred_df["year"].unique())
means = pred_df.groupby("year")[CLASS_NAMES].mean()

# 5a. Stacked area chart
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(years))
for cls in CLASS_NAMES:
    vals = means[cls].values
    ax.fill_between(years, bottom, bottom + vals,
                    color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], alpha=0.85)
    bottom += vals
ax.set_xlim(years[0], years[-1])
ax.set_ylim(0, 1)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Mean Land Cover Fraction", fontsize=12)
ax.set_title("Nuremberg Land Cover Composition (2018-2025)", fontsize=14, fontweight="bold")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
ax.set_xticks(years)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(TS_DIR, "viz_stacked_area.png"), dpi=150, bbox_inches="tight")
print("  Saved viz_stacked_area.png")
plt.close(fig)

# 5b. Per-class trend lines
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, cls in enumerate(CLASS_NAMES):
    ax = axes[i]
    pred_vals = means[cls].values

    ax.plot(years, pred_vals, "o-", color=CLASS_COLORS[cls], linewidth=2, markersize=6,
            label="Prediction")

    # Add labels if available
    for lbl_year in [2020, 2021]:
        if lbl_year in lbl_summary.index:
            lbl_val = lbl_summary.loc[lbl_year, cls]
            ax.plot(lbl_year, lbl_val, "D", color="black", markersize=8, zorder=5)

    ax.set_title(CLASS_LABELS[cls], fontsize=11, fontweight="bold", color=CLASS_COLORS[cls])
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=8, rotation=45)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

# Hide extra subplot
axes[-1].axis("off")
axes[-1].text(0.5, 0.5, "Black diamonds = WorldCover labels",
              ha="center", va="center", fontsize=10, transform=axes[-1].transAxes)

fig.suptitle("Nuremberg Per-Class Land Cover Trends (2018-2025)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(TS_DIR, "viz_per_class_trends.png"), dpi=150, bbox_inches="tight")
print("  Saved viz_per_class_trends.png")
plt.close(fig)

# 5c. Dominant class change map (2018 vs 2022)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (year, ax) in enumerate(zip([2018, 2025, None], axes)):
    if year is not None:
        yr_df = pred_df[pred_df["year"] == year].sort_values("cell_id")
        dom_class = yr_df[CLASS_NAMES].values.argmax(axis=1)

        # Reconstruct grid
        n_cells = len(yr_df)
        # Estimate grid dimensions from anchor (1860/10 x 1610/10 = 186 x 161)
        nc = 186
        nr = n_cells // nc
        if nr * nc == n_cells:
            grid = dom_class.reshape(nr, nc)
        else:
            # Fallback: square-ish grid
            nc = int(np.ceil(np.sqrt(n_cells)))
            nr = int(np.ceil(n_cells / nc))
            padded = np.full(nr * nc, -1, dtype=int)
            padded[:n_cells] = dom_class
            grid = padded.reshape(nr, nc)

        # Custom colormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([CLASS_COLORS[c] for c in CLASS_NAMES])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES)-1, interpolation="nearest")
        ax.set_title(f"Dominant Class {year}", fontsize=12, fontweight="bold")
        ax.axis("off")
    else:
        # Change map
        yr1 = pred_df[pred_df["year"] == 2018].sort_values("cell_id")
        yr2 = pred_df[pred_df["year"] == 2025].sort_values("cell_id")
        dom1 = yr1[CLASS_NAMES].values.argmax(axis=1)
        dom2 = yr2[CLASS_NAMES].values.argmax(axis=1)
        changed = (dom1 != dom2).astype(int)
        n_changed = changed.sum()
        pct_changed = 100 * n_changed / len(changed)

        nc = 186
        nr = len(changed) // nc
        if nr * nc == len(changed):
            grid = changed.reshape(nr, nc)
        else:
            nc = int(np.ceil(np.sqrt(len(changed))))
            nr = int(np.ceil(len(changed) / nc))
            padded = np.zeros(nr * nc, dtype=int)
            padded[:len(changed)] = changed
            grid = padded.reshape(nr, nc)

        from matplotlib.colors import ListedColormap
        cmap_change = ListedColormap(["#EEEEEE", "#FF4444"])
        ax.imshow(grid, cmap=cmap_change, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Changed Cells: {n_changed} ({pct_changed:.1f}%)", fontsize=12, fontweight="bold")
        ax.axis("off")

plt.suptitle("Nuremberg Dominant Land Cover: 2018 vs 2025", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(TS_DIR, "viz_dominant_class_change.png"), dpi=150, bbox_inches="tight")
print("  Saved viz_dominant_class_change.png")
plt.close(fig)

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"\nOutputs in: {TS_DIR}")
print(f"  nuremberg_landcover_2018_2025.parquet  ({result_df.shape[0]} rows)")
print(f"  nuremberg_landcover_2018_2025.csv")
print(f"  nuremberg_summary_by_year.csv")
print(f"  viz_stacked_area.png")
print(f"  viz_per_class_trends.png")
print(f"  viz_dominant_class_change.png")
