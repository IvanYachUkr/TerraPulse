#!/usr/bin/env python3
"""
Compare tile-level MLP predictions vs WorldCover ground truth.

Generates ground truth labels at 10x10 cell resolution from WorldCover,
loads MLP predictions from the Rust pipeline, and computes:
  - Top-1, Top-2, Top-3 accuracy (dominant class)
  - Distribution MAE (mean absolute error per cell)
  - Per-class MAE
  - Jensen-Shannon divergence
  - Pearson correlation of proportions
"""

import json
import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

# ── paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "nuremberg_dashboard")
ANCHOR_PATH = os.path.join(DATA_DIR, "anchor_nuremberg_dashboard.tif")
BOUNDARY_PATH = os.path.join(PROJECT_ROOT, "nuremberg_stat_bezirke_wgs84.geojson")

# ── class mapping ────────────────────────────────────────────────────────
# 7 dashboard classes (with shrubland)
CLASS_NAMES_7 = ["tree_cover", "shrubland", "grassland", "cropland",
                 "built_up", "bare_sparse", "water"]
# WorldCover ESA codes → 7-class index
WC_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4, 60: 5, 80: 6}

GRID_PX = 10          # 10×10 pixel cells
WC_NODATA = 0
WC_YEARS = {2020: "v100", 2021: "v200"}


def load_anchor_meta():
    """Read anchor CRS, transform and dimensions."""
    with rasterio.open(ANCHOR_PATH) as src:
        return src.crs, src.transform, src.width, src.height


def rasterize_boundary(crs, transform, w, h):
    """Return boolean mask of pixels inside the Nuremberg boundary."""
    import geopandas as gpd
    from rasterio.features import geometry_mask
    gdf = gpd.read_file(BOUNDARY_PATH).to_crs(crs)
    mask = geometry_mask(gdf.geometry, transform=transform,
                         out_shape=(h, w), invert=True)
    return mask


def download_worldcover(year):
    """Download the WC tile covering Nuremberg (N48E009)."""
    ver = WC_YEARS[year]
    tile_id = "N48E009"
    fname = f"ESA_WorldCover_10m_{year}_{ver}_{tile_id}_Map.tif"
    wc_dir = os.path.join(PROJECT_ROOT, "data", "worldcover")
    os.makedirs(wc_dir, exist_ok=True)
    local = os.path.join(wc_dir, fname)
    if os.path.exists(local):
        return local
    url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com"
           f"/{ver}/2021/map/{fname}")
    if year == 2020:
        url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com"
               f"/{ver}/2020/map/{fname}")
    print(f"  Downloading {fname}...")
    import urllib.request
    urllib.request.urlretrieve(url, local)
    return local


def reproject_wc_to_anchor(wc_path, crs, transform, w, h):
    """Reproject WorldCover raster to anchor grid."""
    with rasterio.open(wc_path) as src:
        dst = np.zeros((1, h, w), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst[0],
            dst_crs=crs,
            dst_transform=transform,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return dst[0]


def aggregate_labels(wc_arr, mask, w, h, n_classes=7):
    """Compute 7-class proportions per 10×10 cell from WorldCover pixels."""
    n_cols = w // GRID_PX
    n_rows = h // GRID_PX
    n_cells = n_cols * n_rows

    # Remap WC codes to class indices
    remapped = np.full_like(wc_arr, 255, dtype=np.uint8)
    for code, idx in WC_MAP.items():
        remapped[wc_arr == code] = idx

    # Track which cells are inside boundary
    cell_inside = np.zeros(n_cells, dtype=bool)
    proportions = np.zeros((n_cells, n_classes), dtype=np.float32)

    for row in range(n_rows):
        for col in range(n_cols):
            cell_id = row * n_cols + col
            r0, r1 = row * GRID_PX, (row + 1) * GRID_PX
            c0, c1 = col * GRID_PX, (col + 1) * GRID_PX

            tile_mask = mask[r0:r1, c0:c1]
            if not tile_mask.any():
                continue
            cell_inside[cell_id] = True

            tile_wc = remapped[r0:r1, c0:c1]
            valid = tile_wc[tile_mask & (tile_wc < n_classes)]
            if len(valid) == 0:
                continue

            counts = np.bincount(valid, minlength=n_classes)[:n_classes]
            proportions[cell_id] = counts / counts.sum()

    return proportions, cell_inside, n_cols, n_rows


def load_predictions(year):
    """Load MLP predictions from the Rust pipeline output."""
    path = os.path.join(DATA_DIR, f"predictions_{year}.json")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return None
    with open(path) as f:
        data = json.load(f)
    n_cells = len(data)
    preds = np.zeros((n_cells, 7), dtype=np.float32)
    for cell_id_str, vals in data.items():
        i = int(cell_id_str)
        if i < n_cells:
            for j, cls in enumerate(CLASS_NAMES_7):
                preds[i, j] = vals.get(cls, 0.0)
    return preds


def compute_metrics(labels, preds, cell_inside):
    """Compute accuracy and distribution metrics."""
    # Filter to cells inside boundary
    mask = cell_inside & (labels.sum(axis=1) > 0) & (preds.sum(axis=1) > 0)
    L = labels[mask]
    P = preds[mask]
    n = len(L)

    print(f"\n  Cells inside boundary: {n}")

    # ── Top-K accuracy (dominant class) ──────────────────────────────────
    gt_dom = np.argmax(L, axis=1)
    pred_ranked = np.argsort(-P, axis=1)  # descending

    top1 = np.mean(pred_ranked[:, 0] == gt_dom)
    top2 = np.mean([gt_dom[i] in pred_ranked[i, :2] for i in range(n)])
    top3 = np.mean([gt_dom[i] in pred_ranked[i, :3] for i in range(n)])

    print(f"\n  +--- Dominant-Class Accuracy ---+")
    print(f"  |  Top-1:  {top1:6.1%}              |")
    print(f"  |  Top-2:  {top2:6.1%}              |")
    print(f"  |  Top-3:  {top3:6.1%}              |")
    print(f"  +--------------------------------+")

    # ── Distribution MAE ─────────────────────────────────────────────────
    abs_err = np.abs(L - P)
    mae_overall = abs_err.mean()
    mae_per_class = abs_err.mean(axis=0)

    print(f"\n  Distribution MAE (per cell, averaged): {mae_overall:.4f}")
    print(f"  +--- Per-Class MAE --------------------+")
    for j, cls in enumerate(CLASS_NAMES_7):
        bar = "#" * int(mae_per_class[j] * 100)
        print(f"  |  {cls:12s}  {mae_per_class[j]:.4f}  {bar}")
    print(f"  +----------------------------------------+")

    # ── Jensen-Shannon divergence ────────────────────────────────────────
    eps = 1e-10
    M = 0.5 * (L + P)
    kl_lm = np.sum(L * np.log((L + eps) / (M + eps)), axis=1)
    kl_pm = np.sum(P * np.log((P + eps) / (M + eps)), axis=1)
    jsd = 0.5 * (kl_lm + kl_pm)
    print(f"\n  Jensen-Shannon divergence: {jsd.mean():.4f} (mean), {np.median(jsd):.4f} (median)")

    # ── Pearson correlation of proportions ────────────────────────────────
    from scipy.stats import pearsonr
    correlations = []
    for i in range(n):
        if L[i].std() > 0 and P[i].std() > 0:
            r, _ = pearsonr(L[i], P[i])
            correlations.append(r)
    corr_mean = np.mean(correlations)
    corr_med = np.median(correlations)
    print(f"  Pearson correlation: {corr_mean:.4f} (mean), {corr_med:.4f} (median)")

    # Confusion-like: GT dominant -> Predicted dominant
    print(f"\n  +--- Confusion: GT dom -> Pred dom ------------------------+")
    pred_dom = pred_ranked[:, 0]
    for j, cls in enumerate(CLASS_NAMES_7):
        gt_mask = gt_dom == j
        if gt_mask.sum() == 0:
            continue
        correct = (pred_dom[gt_mask] == j).mean()
        n_cells_cls = gt_mask.sum()
        print(f"  |  GT={cls:12s} ({n_cells_cls:5d} cells) -> "
              f"correct: {correct:5.1%}")
    print(f"  +---------------------------------------------------------+")

    return {
        "n_cells": n,
        "top1": top1, "top2": top2, "top3": top3,
        "mae_overall": mae_overall,
        "mae_per_class": dict(zip(CLASS_NAMES_7, mae_per_class.tolist())),
        "jsd_mean": jsd.mean(), "jsd_median": np.median(jsd),
        "corr_mean": corr_mean, "corr_median": corr_med,
    }


def main():
    print("=" * 70)
    print("  Tile-Level MLP Accuracy: Predictions vs WorldCover Ground Truth")
    print("=" * 70)

    crs, transform, w, h = load_anchor_meta()
    n_cols, n_rows = w // GRID_PX, h // GRID_PX
    n_cells = n_cols * n_rows
    print(f"\n  Anchor: {w}x{h} px -> {n_cols}x{n_rows} = {n_cells} cells")

    print("\n  Rasterizing boundary...")
    boundary_mask = rasterize_boundary(crs, transform, w, h)

    results = {}
    for year in [2020, 2021]:
        print(f"\n{'=' * 70}")
        print(f"  YEAR {year}")
        print(f"{'=' * 70}")

        # Labels
        print(f"\n  Loading WorldCover {year}...")
        wc_path = download_worldcover(year)
        wc_arr = reproject_wc_to_anchor(wc_path, crs, transform, w, h)
        labels, cell_inside, _, _ = aggregate_labels(
            wc_arr, boundary_mask, w, h)
        n_inside = cell_inside.sum()
        print(f"  Aggregated {n_inside} cells inside boundary")

        # Predictions
        print(f"  Loading MLP predictions {year}...")
        preds = load_predictions(year)
        if preds is None:
            continue
        # Trim/pad to match cell count
        if len(preds) != n_cells:
            print(f"  WARNING: prediction count ({len(preds)}) != grid cells ({n_cells})")
            min_n = min(len(preds), n_cells)
            preds = preds[:min_n]
            labels = labels[:min_n]
            cell_inside = cell_inside[:min_n]

        results[year] = compute_metrics(labels, preds, cell_inside)

    # ── Summary comparison ───────────────────────────────────────────────
    if len(results) == 2:
        print(f"\n{'=' * 70}")
        print("  SUMMARY: 2020 vs 2021")
        print(f"{'=' * 70}")
        print(f"\n  {'Metric':<30s}  {'2020':>8s}  {'2021':>8s}")
        print(f"  {'-' * 50}")
        for metric in ["top1", "top2", "top3", "mae_overall",
                        "jsd_mean", "corr_mean"]:
            v20 = results[2020][metric]
            v21 = results[2021][metric]
            if "top" in metric:
                print(f"  {metric:<30s}  {v20:7.1%}  {v21:7.1%}")
            else:
                print(f"  {metric:<30s}  {v20:8.4f}  {v21:8.4f}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
