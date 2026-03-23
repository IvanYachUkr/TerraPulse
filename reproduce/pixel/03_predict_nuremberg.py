#!/usr/bin/env python3
"""
Step 3: Run CatBoost pixel model on Nuremberg for dashboard predictions.

This reproduces the Nuremberg dashboard predictions (10 resolutions × 8 years).
It generates .bin files that the dashboard API serves.

Usage:
    python 03_predict_nuremberg.py
    python 03_predict_nuremberg.py --catboost-dir data/cities/models_pixel_v5
"""

import argparse, gc, math, os, sys, time
import numpy as np
import rasterio
from catboost import CatBoostClassifier

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "mlp"))
from importlib import import_module
step1 = import_module("01_download_data")
CITY_MAP = step1.CITY_MAP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
step2 = import_module("02_train_catboost")

NUREMBERG = CITY_MAP["nuremberg"]
N_CLASSES = 7
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
SENTINEL_NODATA = -9999
SAR_NODATA = -9999
RESOLUTIONS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 100]
PREDICT_YEARS = [(y, y + 1) for y in range(2017, 2025)]

DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "src", "dashboard", "data",
                              "nuremberg_dashboard")

def ts():
    return time.strftime("%H:%M:%S")


def _raw_dir_nuremberg():
    """Find Nuremberg raw dir (may be raw_v7 or raw)."""
    d = os.path.join(CITIES_DIR, "nuremberg", "raw_v7")
    if os.path.isdir(d): return d
    return os.path.join(CITIES_DIR, "nuremberg", "raw")


def build_prediction_features(raw_dir, year_pair, anchor_path, H, W):
    """Build per-pixel features for a specific year pair."""
    y1, y2 = year_pair
    seasons = ["spring", "summer", "autumn"]
    years = [y1, y2]

    all_bands, band_names = [], []
    indices_by_tag, sar_by_tag = {}, {}
    INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDMI", "NBR", "BSI", "EVI2",
                   "NDRE1", "NDRE2"]

    for year in years:
        for season in seasons:
            tag = f"{year}_{season}"
            s2_path = os.path.join(raw_dir, f"sentinel2_nuremberg_{year}_{season}.tif")
            s1_path = os.path.join(raw_dir, f"sentinel1_nuremberg_{year}_{season}.tif")

            s2 = step2._load_tif(s2_path, SENTINEL_NODATA) if os.path.exists(s2_path) else None
            if s2 is not None and s2.shape[0] >= 10:
                for bi, bn in enumerate(step2.SENTINEL_BANDS):
                    all_bands.append(s2[bi]); band_names.append(f"{bn}_{tag}")
                idx = step2._compute_indices(s2[:10])
                indices_by_tag[tag] = idx
                for n in INDEX_NAMES:
                    all_bands.append(idx[n]); band_names.append(f"{n}_{tag}")
            else:
                for bn in step2.SENTINEL_BANDS:
                    all_bands.append(np.full((H,W), np.nan, np.float32))
                    band_names.append(f"{bn}_{tag}")
                for n in INDEX_NAMES:
                    all_bands.append(np.full((H,W), np.nan, np.float32))
                    band_names.append(f"{n}_{tag}")

            s1 = step2._load_tif(s1_path, SAR_NODATA) if os.path.exists(s1_path) else None
            if s1 is not None:
                for bi, sn in enumerate(["vv","vh"]):
                    all_bands.append(s1[bi]); band_names.append(f"SAR_{sn.upper()}_{tag}")
                vvvh = np.where(np.abs(s1[1]) > 1e-10, s1[0]/s1[1], np.nan).astype(np.float32)
                all_bands.append(vvvh); band_names.append(f"SAR_VVVH_{tag}")
                sar_by_tag[tag] = {"vv": s1[0], "vh": s1[1]}
            else:
                for sn in ["vv","vh"]:
                    all_bands.append(np.full((H,W), np.nan, np.float32))
                    band_names.append(f"SAR_{sn.upper()}_{tag}")
                all_bands.append(np.full((H,W), np.nan, np.float32))
                band_names.append(f"SAR_VVVH_{tag}")

    # Temporal diffs
    for year in years:
        for sf, st in [("spring","summer"),("summer","autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in indices_by_tag and tt in indices_by_tag:
                for n in INDEX_NAMES:
                    all_bands.append((indices_by_tag[tt][n] - indices_by_tag[tf][n]).astype(np.float32))
                    band_names.append(f"{n}_diff_{st}_{sf}_{year}")
    for season in seasons:
        t0, t1 = f"{years[0]}_{season}", f"{years[1]}_{season}"
        if t0 in indices_by_tag and t1 in indices_by_tag:
            for n in INDEX_NAMES:
                all_bands.append((indices_by_tag[t1][n] - indices_by_tag[t0][n]).astype(np.float32))
                band_names.append(f"{n}_interannual_{season}")
    for year in years:
        ts_s, ts_a = f"{year}_spring", f"{year}_autumn"
        if ts_s in indices_by_tag and ts_a in indices_by_tag:
            for n in ["NDVI","NDWI","EVI2","BSI"]:
                all_bands.append((indices_by_tag[ts_a][n] - indices_by_tag[ts_s][n]).astype(np.float32))
                band_names.append(f"{n}_range_{year}")
    for year in years:
        for sf, st in [("spring","summer"),("summer","autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in sar_by_tag and tt in sar_by_tag:
                for b in ["vv","vh"]:
                    all_bands.append((sar_by_tag[tt][b] - sar_by_tag[tf][b]).astype(np.float32))
                    band_names.append(f"SAR_{b.upper()}_diff_{st}_{sf}_{year}")
    for season in seasons:
        t0, t1 = f"{years[0]}_{season}", f"{years[1]}_{season}"
        if t0 in sar_by_tag and t1 in sar_by_tag:
            for b in ["vv","vh"]:
                all_bands.append((sar_by_tag[t1][b] - sar_by_tag[t0][b]).astype(np.float32))
                band_names.append(f"SAR_{b.upper()}_interannual_{season}")

    cube = np.stack(all_bands, axis=-1)
    n_f = cube.shape[-1]
    valid = np.isnan(cube).sum(axis=-1) < n_f * 0.5
    np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    del all_bands, indices_by_tag, sar_by_tag; gc.collect()
    return cube, valid, band_names


def downsample_predictions(pred_full, H, W, res):
    """Downsample pixel predictions by majority vote in res×res blocks."""
    bh, bw = H // res, W // res
    out = np.full((bh, bw), 255, dtype=np.uint8)
    for r in range(bh):
        for c in range(bw):
            block = pred_full[r*res:(r+1)*res, c*res:(c+1)*res]
            valid = block[block < 255]
            if len(valid) == 0: continue
            classes, counts = np.unique(valid, return_counts=True)
            out[r, c] = classes[counts.argmax()]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catboost-dir", default=os.path.join(CITIES_DIR, "models_pixel_v5"))
    parser.add_argument("--model-name", default="catboost_pixel_v5_deep_unweighted.cbm")
    args = parser.parse_args()

    model_path = os.path.join(args.catboost_dir, args.model_name)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        print("Run 02_train_catboost.py first.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Nuremberg Dashboard Predictions")
    print(f"  Model: {args.model_name}")
    print(f"{'='*70}\n")

    model = CatBoostClassifier()
    model.load_model(model_path)
    print(f"  Model loaded ({model.tree_count_} trees)")

    raw_dir = _raw_dir_nuremberg()
    anchor = os.path.join(raw_dir, "sentinel2_nuremberg_2020_spring.tif")
    if not os.path.exists(anchor):
        print(f"ERROR: Anchor TIF not found. Run 01_download_data.py --cities nuremberg")
        sys.exit(1)

    with rasterio.open(anchor) as src:
        H, W = src.height, src.width
    print(f"  Nuremberg grid: {H}×{W} = {H*W:,} pixels")

    # Load city mask from labels
    mask_path = os.path.join(DASHBOARD_DIR, "nuremberg_labels_2021_res1.bin")
    if os.path.exists(mask_path):
        city_mask = np.fromfile(mask_path, dtype=np.uint8).reshape(H//10, W//10)
        city_mask_px = np.repeat(np.repeat(city_mask, 10, axis=0), 10, axis=1)
        inside_city = city_mask_px != 255
        print(f"  City mask: {inside_city.sum():,} inside pixels")
    else:
        inside_city = None
        print("  WARNING: No city mask — predicting all pixels")

    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    for year_pair in PREDICT_YEARS:
        y1, y2 = year_pair
        tag = f"{y1}_{y2}"
        print(f"\n  [{ts()}] Year pair {tag}...")

        cube, valid, feat_names = build_prediction_features(
            raw_dir, year_pair, anchor, H, W)

        # Predict
        flat_X = cube.reshape(-1, cube.shape[-1])
        flat_valid = valid.reshape(-1)
        if inside_city is not None:
            flat_inside = inside_city[:H, :W].reshape(-1)
            predict_mask = flat_valid & flat_inside
        else:
            predict_mask = flat_valid

        n_predict = predict_mask.sum()
        print(f"    Predicting {n_predict:,} pixels...")

        pred_full = np.full(H * W, 255, dtype=np.uint8)
        if n_predict > 0:
            X_predict = flat_X[predict_mask]
            CHUNK = 500_000
            preds = []
            for i in range(0, len(X_predict), CHUNK):
                preds.append(model.predict(X_predict[i:i+CHUNK]).flatten().astype(np.uint8))
            pred_full[predict_mask] = np.concatenate(preds)

        pred_2d = pred_full.reshape(H, W)
        del cube, flat_X; gc.collect()

        # Generate all resolutions
        for res in RESOLUTIONS:
            out = downsample_predictions(pred_2d, H, W, res) if res > 1 else pred_2d[::1, ::1]
            if res == 1:
                bh, bw = H, W
            else:
                bh, bw = H // res, W // res
                out = downsample_predictions(pred_2d, H, W, res)

            # Apply city mask for coarse resolutions too
            fname = f"nuremberg_pred_{tag}_res{res}.bin"
            path = os.path.join(DASHBOARD_DIR, fname)
            out.tofile(path)
            print(f"    {fname} ({bh}×{bw})")

    print(f"\n[{ts()}] All predictions saved to: {DASHBOARD_DIR}")
    print(f"  Total: {len(PREDICT_YEARS) * len(RESOLUTIONS)} bin files")


if __name__ == "__main__":
    main()
