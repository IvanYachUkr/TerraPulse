#!/usr/bin/env python3
"""
Generate label-composition charts for train / val / test splits.

Outputs:
  mlp/diagrams/label_composition_by_split.png   (line chart)
  mlp/diagrams/label_composition_bars.png       (grouped bar chart)
"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from mlp.config import (
    CITIES, ALL_TRAIN, ALL_TEST,
    VAL_CITY_NAMES, EXCLUDED_CITY_NAMES,
    CLASS_NAMES, N_CLASSES,
    city_labels_path, city_features_path,
)

# ── 1. Reproduce the exact split logic from run.py ──────────────────────

def city_has_data(city):
    """Check if a city has both features and labels."""
    return os.path.exists(city_features_path(city)) and \
           os.path.exists(city_labels_path(city))

all_usable = [c for c in CITIES if city_has_data(c)]

train_cities = [c for c in all_usable
                if not c.is_test
                and c.name not in VAL_CITY_NAMES
                and c.name not in EXCLUDED_CITY_NAMES]
val_cities   = [c for c in all_usable if c.name in VAL_CITY_NAMES]
test_cities  = [c for c in all_usable if c.is_test and c.name not in EXCLUDED_CITY_NAMES]

print(f"Train: {len(train_cities)} cities")
print(f"Val:   {len(val_cities)} cities")
print(f"Test:  {len(test_cities)} cities")
print(f"Total: {len(train_cities) + len(val_cities) + len(test_cities)}")

# ── 2. Aggregate label fractions per split ───────────────────────────────

def aggregate_labels(cities):
    """Return mean class-fraction across all cells in `cities`."""
    total_cells = 0
    weighted_sum = np.zeros(N_CLASSES, dtype=np.float64)
    for city in cities:
        lp = city_labels_path(city)
        if not os.path.exists(lp):
            continue
        df = pd.read_parquet(lp)
        y = df[CLASS_NAMES].values.astype(np.float64)
        row_sums = y.sum(axis=1, keepdims=True)
        valid = row_sums.ravel() > 0
        y = y[valid]
        row_sums = row_sums[valid]
        y = y / np.maximum(row_sums, 1e-8)
        weighted_sum += y.sum(axis=0)
        total_cells += y.shape[0]
    if total_cells == 0:
        return np.zeros(N_CLASSES), 0
    return weighted_sum / total_cells, total_cells

train_fracs, train_n = aggregate_labels(train_cities)
val_fracs,   val_n   = aggregate_labels(val_cities)
test_fracs,  test_n  = aggregate_labels(test_cities)

print(f"\nCells  –  train: {train_n:,}  val: {val_n:,}  test: {test_n:,}")

# ── 3. Pretty names and colours ─────────────────────────────────────────

pretty_names = {
    "tree_cover":  "Tree Cover",
    "shrubland":   "Shrubland",
    "grassland":   "Grassland",
    "cropland":    "Cropland",
    "built_up":    "Built-Up",
    "bare_sparse": "Bare / Sparse",
    "water":       "Water",
}

# ESA WorldCover-inspired palette
class_colours = {
    "tree_cover":  "#006400",
    "shrubland":   "#ffbb22",
    "grassland":   "#a3de00",
    "cropland":    "#c77d29",
    "built_up":    "#e30613",
    "bare_sparse": "#b4b4b4",
    "water":       "#0064c8",
}

DIAG_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# CHART 1 — LINE CHART  (the main one the user asked for)
# ==========================================================================

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

x_labels = ["Train", "Validation", "Test"]
x = np.arange(len(x_labels))
all_fracs = np.array([train_fracs, val_fracs, test_fracs])  # (3, 7)

markers = ["o", "s", "D", "^", "P", "X", "v"]

for cls_i, cls_name in enumerate(CLASS_NAMES):
    col = class_colours[cls_name]
    vals = all_fracs[:, cls_i] * 100
    ax.plot(x, vals,
            marker=markers[cls_i], markersize=9, linewidth=2.4,
            color=col, label=pretty_names[cls_name],
            zorder=3)
    # annotate each point
    for xi, vi in zip(x, vals):
        offset_y = 1.0 if vi > 2 else 0.5
        ax.annotate(f"{vi:.1f}%",
                    (xi, vi), textcoords="offset points",
                    xytext=(0, 8 + offset_y), ha="center",
                    fontsize=7, fontweight="bold", color=col)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"Train\n({len(train_cities)} cities · {train_n:,} cells)",
     f"Validation\n({len(val_cities)} cities · {val_n:,} cells)",
     f"Test\n({len(test_cities)} cities · {test_n:,} cells)"],
    fontsize=9.5, fontweight="semibold",
)

ax.set_ylabel("Mean class fraction  (%)", fontsize=11, fontweight="semibold")
ax.set_title("Label Composition across Data Splits",
             fontsize=14, fontweight="bold", pad=14)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylim(0, max(all_fracs.max() * 100 * 1.22, 5))
ax.set_xlim(-0.3, 2.3)
ax.grid(axis="y", linewidth=0.3, alpha=0.5)
ax.grid(axis="x", linewidth=0.3, alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92,
                edgecolor="#ccc", ncol=2, title="Land-Cover Class",
                title_fontsize=9.5)

plt.tight_layout()
out1 = os.path.join(DIAG_DIR, "label_composition_by_split.png")
fig.savefig(out1, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()

# ==========================================================================
# CHART 2 — GROUPED BAR CHART (complementary view)
# ==========================================================================

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

x = np.arange(len(CLASS_NAMES))
bar_width = 0.24
offsets = [-bar_width, 0, bar_width]
split_edge = ["#1a73e8", "#34a853", "#ea4335"]
hatches = ["", "//", ".."]
splits = ["Train", "Validation", "Test"]

for i, (split, fracs) in enumerate(zip(splits, all_fracs)):
    bars = ax.bar(
        x + offsets[i], fracs * 100, bar_width,
        label=f"{split}  ({[train_n, val_n, test_n][i]:,} cells)",
        color=[class_colours[c] for c in CLASS_NAMES],
        edgecolor=split_edge[i],
        linewidth=1.6,
        alpha=0.72 + 0.12 * i,
        hatch=hatches[i],
    )
    for bar, v in zip(bars, fracs):
        if v * 100 > 1.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v*100:.1f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", color="#333")

ax.set_xticks(x)
ax.set_xticklabels([pretty_names[c] for c in CLASS_NAMES],
                   fontsize=10, fontweight="semibold")
ax.set_ylabel("Mean class fraction  (%)", fontsize=11, fontweight="semibold")
ax.set_title("Label Composition by Data Split",
             fontsize=14, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylim(0, max(all_fracs.max() * 100 * 1.15, 5))
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
                edgecolor="#ccc", title="Split", title_fontsize=10)

info = (f"Cities — Train: {len(train_cities)}  |  "
        f"Val: {len(val_cities)}  |  Test: {len(test_cities)}")
ax.text(0.5, -0.11, info, transform=ax.transAxes,
        ha="center", fontsize=9, color="#555", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
out2 = os.path.join(DIAG_DIR, "label_composition_bars.png")
fig.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()
