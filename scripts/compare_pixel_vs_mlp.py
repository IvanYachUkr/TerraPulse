#!/usr/bin/env python3
"""
Head-to-head: Pixel CatBoost (aggregated) vs MLP tile model.

Loads:
  1. WorldCover ground truth -> aggregated to 10x10 tiles (7 classes)
  2. MLP tile predictions (7 classes, from Rust ONNX pipeline)
  3. CatBoost pixel predictions -> aggregated to 10x10 tiles (6 classes, no shrubland)

Computes top-1/2/3 accuracy, distribution MAE, JSD, Pearson correlation
for both models and prints a side-by-side comparison.
"""

import json, os, sys, time
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "nuremberg_dashboard")
ANCHOR_PATH = os.path.join(DATA_DIR, "anchor_nuremberg_dashboard.tif")
BOUNDARY_PATH = os.path.join(PROJECT_ROOT, "nuremberg_stat_bezirke_wgs84.geojson")
PIXEL_PRED_DIR = os.path.join(PROJECT_ROOT, "src", "dashboard", "data", "nuremberg_dashboard")

# MLP 7 classes (with shrubland)
CLS7 = ["tree_cover", "shrubland", "grassland", "cropland", "built_up", "bare_sparse", "water"]
# Pixel 6 classes (shrubland merged into grassland)
CLS6 = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]

# WorldCover -> 7-class index
WC_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4, 60: 5, 80: 6}
# 7-class -> 6-class (merge shrubland into grassland)
REMAP_7_TO_6 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

GRID_PX = 10
WC_YEARS = {2020: "v100", 2021: "v200"}


def load_anchor():
    with rasterio.open(ANCHOR_PATH) as src:
        return src.crs, src.transform, src.width, src.height


def rasterize_boundary(crs, transform, w, h):
    import geopandas as gpd
    from rasterio.features import geometry_mask
    gdf = gpd.read_file(BOUNDARY_PATH).to_crs(crs)
    return geometry_mask(gdf.geometry, transform=transform,
                         out_shape=(h, w), invert=True)


def download_worldcover(year):
    ver = WC_YEARS[year]
    fname = f"ESA_WorldCover_10m_{year}_{ver}_N48E009_Map.tif"
    wc_dir = os.path.join(PROJECT_ROOT, "data", "worldcover")
    os.makedirs(wc_dir, exist_ok=True)
    local = os.path.join(wc_dir, fname)
    if os.path.exists(local):
        return local
    url_year = year if year == 2020 else 2021
    url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{ver}/{url_year}/map/{fname}"
    print(f"  Downloading {fname}...")
    import urllib.request
    urllib.request.urlretrieve(url, local)
    return local


def reproject_wc(wc_path, crs, transform, w, h):
    with rasterio.open(wc_path) as src:
        dst = np.zeros((h, w), dtype=np.uint8)
        reproject(source=rasterio.band(src, 1), destination=dst,
                  dst_crs=crs, dst_transform=transform, dst_nodata=0,
                  resampling=Resampling.nearest)
    return dst


def aggregate_to_tiles(class_map_2d, boundary_mask, w, h, n_classes):
    """Aggregate a 2D class map to 10x10 tile proportions.
    class_map_2d: uint8 array (H, W) with class indices, 255=nodata
    Returns: (proportions[n_cells, n_classes], cell_mask[n_cells])
    """
    nc, nr = w // GRID_PX, h // GRID_PX
    n_cells = nc * nr
    props = np.zeros((n_cells, n_classes), dtype=np.float32)
    cell_ok = np.zeros(n_cells, dtype=bool)

    for row in range(nr):
        for col in range(nc):
            cid = row * nc + col
            r0, r1 = row * GRID_PX, (row + 1) * GRID_PX
            c0, c1 = col * GRID_PX, (col + 1) * GRID_PX
            bm = boundary_mask[r0:r1, c0:c1]
            if not bm.any():
                continue
            cell_ok[cid] = True
            tile = class_map_2d[r0:r1, c0:c1]
            valid = tile[bm & (tile < n_classes)]
            if len(valid) == 0:
                continue
            counts = np.bincount(valid, minlength=n_classes)[:n_classes]
            props[cid] = counts / counts.sum()
    return props, cell_ok


def load_pixel_predictions(year, h, w):
    """Load CatBoost pixel predictions from binary file (6 classes)."""
    path = os.path.join(PIXEL_PRED_DIR, f"nuremberg_pred_{year}_res1.bin")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return None
    data = np.fromfile(path, dtype=np.uint8).reshape(h, w)
    return data


def load_mlp_predictions(year, n_cells):
    """Load MLP tile predictions from JSON (7 classes)."""
    path = os.path.join(DATA_DIR, f"predictions_{year}.json")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return None
    with open(path) as f:
        data = json.load(f)
    preds = np.zeros((n_cells, 7), dtype=np.float32)
    for cid_str, vals in data.items():
        i = int(cid_str)
        if i < n_cells:
            for j, cls in enumerate(CLS7):
                preds[i, j] = vals.get(cls, 0.0)
    return preds


def compute_metrics(gt, pred, mask, label="Model"):
    """Compute all accuracy metrics. Both gt and pred are (n_cells, n_classes)."""
    L = gt[mask]
    P = pred[mask]
    n = len(L)
    nc = L.shape[1]

    gt_dom = np.argmax(L, axis=1)
    pred_ranked = np.argsort(-P, axis=1)

    top1 = float(np.mean(pred_ranked[:, 0] == gt_dom))
    top2 = float(np.mean([gt_dom[i] in pred_ranked[i, :2] for i in range(n)]))
    top3 = float(np.mean([gt_dom[i] in pred_ranked[i, :3] for i in range(n)]))

    abs_err = np.abs(L - P)
    mae = float(abs_err.mean())
    mae_per_cls = abs_err.mean(axis=0)

    eps = 1e-10
    M = 0.5 * (L + P)
    kl_lm = np.sum(L * np.log((L + eps) / (M + eps)), axis=1)
    kl_pm = np.sum(P * np.log((P + eps) / (M + eps)), axis=1)
    jsd = float(np.mean(0.5 * (kl_lm + kl_pm)))

    from scipy.stats import pearsonr
    corrs = []
    for i in range(n):
        if L[i].std() > 0 and P[i].std() > 0:
            r, _ = pearsonr(L[i], P[i])
            corrs.append(r)
    corr = float(np.mean(corrs)) if corrs else 0.0

    return {
        "n_cells": n, "top1": top1, "top2": top2, "top3": top3,
        "mae": mae, "jsd": jsd, "corr": corr,
        "mae_per_cls": mae_per_cls,
    }


def remap_7_to_6(props_7):
    """Remap 7-class proportions to 6-class (merge shrubland into grassland)."""
    props_6 = np.zeros((props_7.shape[0], 6), dtype=np.float32)
    props_6[:, 0] = props_7[:, 0]                 # tree_cover
    props_6[:, 1] = props_7[:, 1] + props_7[:, 2]  # grassland + shrubland
    props_6[:, 2] = props_7[:, 3]                  # cropland
    props_6[:, 3] = props_7[:, 4]                  # built_up
    props_6[:, 4] = props_7[:, 5]                  # bare_sparse
    props_6[:, 5] = props_7[:, 6]                  # water
    return props_6


def main():
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Head-to-Head: Pixel CatBoost (aggregated) vs MLP Tile Model")
    print(f"{sep}\n")

    crs, transform, w, h = load_anchor()
    nc, nr = w // GRID_PX, h // GRID_PX
    n_cells = nc * nr
    print(f"  Anchor: {w}x{h} px -> {nc}x{nr} = {n_cells} cells")

    print("  Rasterizing boundary...")
    bmask = rasterize_boundary(crs, transform, w, h)

    for year in [2020, 2021]:
        print(f"\n{sep}")
        print(f"  YEAR {year}")
        print(f"{sep}")

        # --- Ground truth (7 classes) ---
        print(f"\n  [GT] Loading WorldCover {year}...")
        wc_path = download_worldcover(year)
        wc_arr = reproject_wc(wc_path, crs, transform, w, h)

        # Remap WC to 7-class indices
        gt7_2d = np.full((h, w), 255, dtype=np.uint8)
        for code, idx in WC_MAP.items():
            gt7_2d[wc_arr == code] = idx

        gt7_props, gt7_mask = aggregate_to_tiles(gt7_2d, bmask, w, h, 7)

        # Also create 6-class GT (merge shrubland->grassland)
        gt6_2d = np.full((h, w), 255, dtype=np.uint8)
        for code, idx in WC_MAP.items():
            gt6_2d[wc_arr == code] = REMAP_7_TO_6[idx]
        gt6_props, gt6_mask = aggregate_to_tiles(gt6_2d, bmask, w, h, 6)

        n_inside = gt7_mask.sum()
        print(f"  [GT] {n_inside} cells inside boundary")

        # --- MLP tile model (7 classes -> remap to 6 for fair comparison) ---
        print(f"\n  [MLP] Loading tile predictions {year}...")
        mlp7 = load_mlp_predictions(year, n_cells)
        if mlp7 is None:
            continue
        # Remap to 6 classes for fair comparison
        mlp6 = remap_7_to_6(mlp7)

        # --- Pixel CatBoost predictions (6 classes) ---
        print(f"  [Pixel] Loading CatBoost pixel predictions {year}...")
        pixel_2d = load_pixel_predictions(year, h, w)
        if pixel_2d is None:
            continue
        # Aggregate to tiles
        pixel6_props, pixel_mask = aggregate_to_tiles(pixel_2d, bmask, w, h, 6)

        # --- Compute metrics (all in 6-class space for fair comparison) ---
        mask = gt6_mask & pixel_mask & (gt6_props.sum(axis=1) > 0)

        print(f"\n  Computing metrics on {mask.sum()} valid cells...")
        r_mlp = compute_metrics(gt6_props, mlp6, mask, "MLP")
        r_pix = compute_metrics(gt6_props, pixel6_props, mask, "Pixel")

        # --- Print results ---
        print(f"\n  +{'='*62}+")
        print(f"  |{'RESULTS ' + str(year):^62}|")
        print(f"  +{'='*62}+")
        print(f"  | {'Metric':<25} | {'MLP Tile':>12} | {'Pixel Agg':>12} | {'Winner':>6} |")
        print(f"  +{'-'*62}+")

        rows = [
            ("Top-1 Accuracy",  "top1",  True),
            ("Top-2 Accuracy",  "top2",  True),
            ("Top-3 Accuracy",  "top3",  True),
            ("Distribution MAE", "mae",  False),
            ("Jensen-Shannon Div", "jsd", False),
            ("Pearson Corr",    "corr",  True),
        ]
        for name, key, higher_better in rows:
            vm = r_mlp[key]
            vp = r_pix[key]
            if higher_better:
                winner = "MLP" if vm > vp else "Pixel" if vp > vm else "Tie"
            else:
                winner = "MLP" if vm < vp else "Pixel" if vp < vm else "Tie"

            if key in ("top1", "top2", "top3"):
                print(f"  | {name:<25} | {vm:11.1%} | {vp:11.1%} | {winner:>6} |")
            else:
                print(f"  | {name:<25} | {vm:12.4f} | {vp:12.4f} | {winner:>6} |")

        print(f"  +{'-'*62}+")

        # Per-class MAE
        print(f"\n  Per-Class Distribution MAE:")
        print(f"  {'Class':<15} {'MLP':>8} {'Pixel':>8} {'Winner':>8}")
        print(f"  {'-'*42}")
        for j, cls in enumerate(CLS6):
            vm = r_mlp["mae_per_cls"][j]
            vp = r_pix["mae_per_cls"][j]
            winner = "MLP" if vm < vp else "Pixel" if vp < vm else "Tie"
            print(f"  {cls:<15} {vm:8.4f} {vp:8.4f} {winner:>8}")

        # Dominant-class confusion
        L = gt6_props[mask]
        P_mlp = mlp6[mask]
        P_pix = pixel6_props[mask]
        gt_dom = np.argmax(L, axis=1)

        print(f"\n  Per-Class Dominant Accuracy:")
        print(f"  {'GT Class':<15} {'Count':>6} {'MLP':>8} {'Pixel':>8} {'Winner':>8}")
        print(f"  {'-'*50}")
        for j, cls in enumerate(CLS6):
            sel = gt_dom == j
            if sel.sum() == 0:
                continue
            mlp_acc = (np.argmax(P_mlp[sel], axis=1) == j).mean()
            pix_acc = (np.argmax(P_pix[sel], axis=1) == j).mean()
            winner = "MLP" if mlp_acc > pix_acc else "Pixel" if pix_acc > mlp_acc else "Tie"
            print(f"  {cls:<15} {sel.sum():>6} {mlp_acc:7.1%} {pix_acc:7.1%} {winner:>8}")

    print(f"\n{sep}")
    print("  Done!")
    print(f"{sep}")


if __name__ == "__main__":
    main()
