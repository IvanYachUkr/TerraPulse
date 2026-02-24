#!/usr/bin/env python3
"""
Predict land cover for Nuremberg Dashboard (expanded bbox).

For each prediction year Y, loads features from year-pair (Y-1, Y) using V2-style
feature extraction, then runs CatBoost V5 inference. Outputs binary label maps
at 10 resolutions, plus regenerated ground-truth labels from WorldCover.

Usage:
    python scripts/predict_nuremberg_dashboard.py
"""
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    SEASONS, SENTINEL_BANDS, SENTINEL_NODATA, WC_CLASS_MAP, CLASS_NAMES,
    N_CLASSES, _wc_tiles_for_bbox,
)
from scripts.pixel_classifier_v2 import (
    SAR_BANDS, SAR_NODATA, INDEX_NAMES, _compute_indices, _safe_ratio,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CITY_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "nuremberg_dashboard")
RAW_DIR = os.path.join(CITY_DIR, "raw")
ANCHOR_PATH = os.path.join(CITY_DIR, "anchor_nuremberg_dashboard.tif")
WC_TILES_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "worldcover_tiles")

OUT_DIR = os.path.join(PROJECT_ROOT, "src", "dashboard", "data", "nuremberg_dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

GEOJSON_PATH = os.path.join(PROJECT_ROOT, "src", "dashboard", "data",
                            "nuremberg_boundary.geojson")

MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "cities", "models_pixel_v5",
                          "catboost_pixel_v5_deep_unweighted.cbm")

# Expanded bbox (must match anchor creation)
BBOX = [10.96, 49.31, 11.30, 49.56]
PREDICTION_YEARS = list(range(2018, 2026))  # 2018-2025
LABEL_YEARS = [2020, 2021]
MAX_RES = 10

# Shrubland (class 1) -> grassland (class 2) remap
# After remap: 0=tree_cover, 1=grassland, 2=cropland, 3=built_up, 4=bare_sparse, 5=water
REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
DASH_CLASSES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]


def ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Load TIFs
# ---------------------------------------------------------------------------
def _load_tif(path, nodata_val):
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    data[data == nodata_val] = np.nan
    return data


def load_tifs_for_years(year0, year1):
    """Load S2+S1 TIFs for a pair of years."""
    results = {}
    for year in [year0, year1]:
        for season in SEASONS:
            tag = f"{year}_{season}"
            s2_path = os.path.join(RAW_DIR,
                f"sentinel2_nuremberg_dashboard_{year}_{season}.tif")
            s1_path = os.path.join(RAW_DIR,
                f"sentinel1_nuremberg_dashboard_{year}_{season}.tif")
            results[f"s2_{tag}"] = _load_tif(s2_path, SENTINEL_NODATA)
            results[f"s1_{tag}"] = _load_tif(s1_path, SAR_NODATA)
    return results


# ---------------------------------------------------------------------------
# Feature builder for arbitrary year-pair
# ---------------------------------------------------------------------------
def build_features_for_years(year0, year1, H, W):
    """Build V2-compatible feature matrix for a year-pair (year0, year1)."""
    tifs = load_tifs_for_years(year0, year1)
    years = [year0, year1]

    all_bands = []
    band_names = []
    indices_by_tag = {}
    sar_by_tag = {}
    has_any_s2 = False

    for year in years:
        for season in SEASONS:
            tag = f"{year}_{season}"

            # Sentinel-2
            s2_data = tifs.get(f"s2_{tag}")
            if s2_data is not None and s2_data.shape[0] >= 11:
                has_any_s2 = True
                s2 = s2_data[:10]

                for bi, bname in enumerate(SENTINEL_BANDS):
                    all_bands.append(s2[bi])
                    band_names.append(f"{bname}_{tag}")

                idx_dict = _compute_indices(s2)
                indices_by_tag[tag] = idx_dict
                for idx_name in INDEX_NAMES:
                    all_bands.append(idx_dict[idx_name])
                    band_names.append(f"{idx_name}_{tag}")
            else:
                for bname in SENTINEL_BANDS:
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                    band_names.append(f"{bname}_{tag}")
                for idx_name in INDEX_NAMES:
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                    band_names.append(f"{idx_name}_{tag}")

            # Sentinel-1
            s1_data = tifs.get(f"s1_{tag}")
            if s1_data is not None:
                for bi, sname in enumerate(SAR_BANDS):
                    all_bands.append(s1_data[bi])
                    band_names.append(f"SAR_{sname.upper()}_{tag}")
                vv_vh = np.where(np.abs(s1_data[1]) > 1e-10,
                                 s1_data[0] / s1_data[1], np.nan).astype(np.float32)
                all_bands.append(vv_vh)
                band_names.append(f"SAR_VVVH_{tag}")
                sar_by_tag[tag] = {"vv": s1_data[0], "vh": s1_data[1]}
            else:
                for sname in SAR_BANDS:
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                    band_names.append(f"SAR_{sname.upper()}_{tag}")
                all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                band_names.append(f"SAR_VVVH_{tag}")

    if not has_any_s2:
        print("  WARNING: No S2 data found!")
        return None, []

    del tifs; gc.collect()

    # Temporal diffs (intra-annual)
    for year in years:
        for s_from, s_to in [("spring", "summer"), ("summer", "autumn")]:
            tf = f"{year}_{s_from}"
            tt = f"{year}_{s_to}"
            if tf in indices_by_tag and tt in indices_by_tag:
                for idx_name in INDEX_NAMES:
                    diff = indices_by_tag[tt][idx_name] - indices_by_tag[tf][idx_name]
                    all_bands.append(diff.astype(np.float32))
                    band_names.append(f"{idx_name}_diff_{s_to}_{s_from}_{year}")

    # Inter-annual diffs
    for season in SEASONS:
        t0 = f"{year0}_{season}"
        t1 = f"{year1}_{season}"
        if t0 in indices_by_tag and t1 in indices_by_tag:
            for idx_name in INDEX_NAMES:
                diff = indices_by_tag[t1][idx_name] - indices_by_tag[t0][idx_name]
                all_bands.append(diff.astype(np.float32))
                band_names.append(f"{idx_name}_interannual_{season}")

    # Spring-to-autumn range
    for year in years:
        ts_spring = f"{year}_spring"
        ts_autumn = f"{year}_autumn"
        if ts_spring in indices_by_tag and ts_autumn in indices_by_tag:
            for idx_name in ["NDVI", "NDWI", "EVI2", "BSI"]:
                rng = indices_by_tag[ts_autumn][idx_name] - indices_by_tag[ts_spring][idx_name]
                all_bands.append(rng.astype(np.float32))
                band_names.append(f"{idx_name}_range_{year}")

    # SAR temporal diffs
    for year in years:
        for s_from, s_to in [("spring", "summer"), ("summer", "autumn")]:
            tf = f"{year}_{s_from}"
            tt = f"{year}_{s_to}"
            if tf in sar_by_tag and tt in sar_by_tag:
                for band in ["vv", "vh"]:
                    diff = sar_by_tag[tt][band] - sar_by_tag[tf][band]
                    all_bands.append(diff.astype(np.float32))
                    band_names.append(f"SAR_{band.upper()}_diff_{s_to}_{s_from}_{year}")

    # SAR inter-annual
    for season in SEASONS:
        t0 = f"{year0}_{season}"
        t1 = f"{year1}_{season}"
        if t0 in sar_by_tag and t1 in sar_by_tag:
            for band in ["vv", "vh"]:
                diff = sar_by_tag[t1][band] - sar_by_tag[t0][band]
                all_bands.append(diff.astype(np.float32))
                band_names.append(f"SAR_{band.upper()}_interannual_{season}")

    n_features = len(all_bands)
    feature_cube = np.stack(all_bands, axis=-1)  # (H, W, F)
    del all_bands, indices_by_tag, sar_by_tag; gc.collect()

    return feature_cube, band_names


# ---------------------------------------------------------------------------
# WorldCover labels
# ---------------------------------------------------------------------------
def load_worldcover_expanded(year=2021):
    """Load WorldCover labels for the expanded anchor."""
    with rasterio.open(ANCHOR_PATH) as ref:
        anchor_crs = ref.crs
        anchor_transform = ref.transform
        anchor_width = ref.width
        anchor_height = ref.height

    tiles = _wc_tiles_for_bbox(BBOX)
    version = "v100" if year == 2020 else "v200"

    dst_array = np.zeros((anchor_height, anchor_width), dtype=np.uint8)
    found_any = False
    for tile in tiles:
        filename = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        wc_path = os.path.join(WC_TILES_DIR, filename)
        if not os.path.exists(wc_path):
            continue
        found_any = True
        tmp = np.zeros_like(dst_array)
        with rasterio.open(wc_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=tmp,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=anchor_transform, dst_crs=anchor_crs,
                dst_nodata=0, resampling=Resampling.nearest,
            )
        mask = (dst_array == 0) & (tmp > 0)
        dst_array[mask] = tmp[mask]

    if not found_any:
        return None

    label_array = np.full((anchor_height, anchor_width), 255, dtype=np.uint8)
    for wc_code, our_class in WC_CLASS_MAP.items():
        label_array[dst_array == wc_code] = our_class
    return label_array


# ---------------------------------------------------------------------------
# Rasterize boundary mask
# ---------------------------------------------------------------------------
def make_boundary_mask(H, W):
    """Rasterize Nuremberg GeoJSON boundary to mask."""
    import geopandas as gpd
    from rasterio.features import rasterize

    with rasterio.open(ANCHOR_PATH) as ref:
        transform = ref.transform
        crs = ref.crs

    gdf = gpd.read_file(GEOJSON_PATH).to_crs(crs)
    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = rasterize(shapes, out_shape=(H, W), transform=transform,
                     fill=0, dtype=np.uint8)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Save binary maps at multiple resolutions
# ---------------------------------------------------------------------------
def save_binary_maps(label_2d, boundary_mask, tag, H, W, meta_resolutions):
    """Save label maps at resolutions 1-10. Returns file list."""
    files = []
    for res in range(1, MAX_RES + 1):
        if res == 1:
            out_labels = label_2d.copy()
            out_labels[~boundary_mask] = 255
        else:
            out_H = H // res
            out_W = W // res
            out_labels = np.full((out_H, out_W), 255, dtype=np.uint8)
            cropped_H = out_H * res
            cropped_W = out_W * res
            block = label_2d[:cropped_H, :cropped_W].reshape(out_H, res, out_W, res)
            bmask = boundary_mask[:cropped_H, :cropped_W].reshape(out_H, res, out_W, res)
            any_inside = bmask.any(axis=(1, 3))

            for r in range(out_H):
                for c in range(out_W):
                    if not any_inside[r, c]:
                        continue
                    patch = block[r, :, c, :]
                    valid = patch[patch < 255]
                    if len(valid) == 0:
                        continue
                    vals, counts = np.unique(valid, return_counts=True)
                    out_labels[r, c] = vals[counts.argmax()]

        fname = f"nuremberg_{tag}_res{res}.bin"
        fpath = os.path.join(OUT_DIR, fname)
        out_labels.tofile(fpath)
        fsize = os.path.getsize(fpath)
        out_H_r = H // res if res > 1 else H
        out_W_r = W // res if res > 1 else W
        files.append(fname)

        reskey = f"res{res}"
        meta_resolutions[reskey] = {"width": out_W_r, "height": out_H_r}

        print(f"    res={res:>2}: {out_W_r}x{out_H_r} -> {fname} ({fsize/1024:.1f} KB)")

    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from catboost import CatBoostClassifier

    print(f"\n{'='*70}")
    print(f"  Nuremberg Dashboard: Predict + Label Generation")
    print(f"{'='*70}\n")

    # Load anchor dims
    with rasterio.open(ANCHOR_PATH) as ref:
        H, W = ref.height, ref.width
        anchor_transform = ref.transform
        anchor_crs = ref.crs
        bounds = ref.bounds

    from rasterio.warp import transform_bounds
    from rasterio.crs import CRS
    wgs84 = CRS.from_epsg(4326)
    w, s, e, n = transform_bounds(anchor_crs, wgs84, *bounds, densify_pts=21)
    print(f"  Anchor: {W}x{H} px, WGS84 bounds: [{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}]")

    # Boundary mask
    print(f"\n[{ts()}] Rasterizing boundary...")
    boundary_mask = make_boundary_mask(H, W)
    n_inside = boundary_mask.sum()
    print(f"  Pixels inside boundary: {n_inside:,} / {H*W:,} ({100*n_inside/(H*W):.1f}%)")

    # Load model
    print(f"\n[{ts()}] Loading CatBoost model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(f"  Model: {MODEL_PATH}")
    n_model_features = len(model.feature_names_)
    print(f"  Features expected: {n_model_features}")

    meta = {
        "wgs84_bounds": [w, s, e, n],
        "classes": DASH_CLASSES,
        "label_years": LABEL_YEARS,
        "prediction_years": PREDICTION_YEARS,
        "resolutions": {},
    }

    # --- Ground truth labels ---
    for year in LABEL_YEARS:
        print(f"\n[{ts()}] Loading WorldCover {year}...")
        raw_labels = load_worldcover_expanded(year)
        if raw_labels is None:
            print(f"  ERROR: No WorldCover data for {year}")
            continue

        # Remap to dash classes
        label_2d = np.full_like(raw_labels, 255)
        for src_cls, dst_cls in REMAP.items():
            label_2d[raw_labels == src_cls] = dst_cls

        valid_inside = (label_2d < 255) & boundary_mask
        print(f"  Valid pixels inside boundary: {valid_inside.sum():,}")
        for i, cname in enumerate(DASH_CLASSES):
            cnt = ((label_2d == i) & boundary_mask).sum()
            pct = 100 * cnt / max(valid_inside.sum(), 1)
            print(f"    {cname:>15}: {cnt:>8,} ({pct:5.1f}%)")

        print(f"  Saving label maps...")
        save_binary_maps(label_2d, boundary_mask, f"labels_{year}", H, W,
                        meta["resolutions"])
        del raw_labels, label_2d; gc.collect()

    # --- Predictions ---
    for pred_year in PREDICTION_YEARS:
        year0 = pred_year - 1
        year1 = pred_year
        print(f"\n{'='*70}")
        print(f"  [{ts()}] Predicting {pred_year} (features from {year0}+{year1})")
        print(f"{'='*70}")

        feature_cube, band_names = build_features_for_years(year0, year1, H, W)
        if feature_cube is None:
            print(f"  SKIPPED: no S2 data")
            continue

        n_features = feature_cube.shape[-1]
        print(f"  Features: {n_features} bands")

        if n_features != n_model_features:
            print(f"  WARNING: Feature count mismatch! Got {n_features}, model expects {n_model_features}")
            if n_features < n_model_features:
                # Pad with NaN columns
                pad = n_model_features - n_features
                print(f"  Padding with {pad} NaN columns...")
                pad_arr = np.full((H, W, pad), np.nan, dtype=np.float32)
                feature_cube = np.concatenate([feature_cube, pad_arr], axis=-1)
            else:
                # Truncate
                print(f"  Truncating to {n_model_features} features...")
                feature_cube = feature_cube[:, :, :n_model_features]

        # Flatten valid pixels only (inside boundary)
        flat = feature_cube[boundary_mask]  # (N, F)
        np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        print(f"  Predicting {flat.shape[0]:,} pixels...")

        # Predict in chunks
        CHUNK = 500_000
        pred_classes = np.zeros(flat.shape[0], dtype=np.uint8)
        for i in range(0, flat.shape[0], CHUNK):
            j = min(i + CHUNK, flat.shape[0])
            pred = model.predict(flat[i:j]).flatten().astype(np.uint8)
            pred_classes[i:j] = pred
            print(f"    Chunk {i//CHUNK+1}: {i:,}-{j:,} done")

        del flat, feature_cube; gc.collect()

        # Unravel to 2D
        pred_2d = np.full((H, W), 255, dtype=np.uint8)
        pred_remapped = np.array([REMAP.get(c, 255) for c in pred_classes], dtype=np.uint8)
        pred_2d[boundary_mask] = pred_remapped

        # Stats
        for i, cname in enumerate(DASH_CLASSES):
            cnt = (pred_2d[boundary_mask] == i).sum()
            pct = 100 * cnt / max(n_inside, 1)
            print(f"    {cname:>15}: {cnt:>8,} ({pct:5.1f}%)")

        print(f"  Saving prediction maps...")
        save_binary_maps(pred_2d, boundary_mask, f"pred_{pred_year}", H, W,
                        meta["resolutions"])
        del pred_2d, pred_classes, pred_remapped; gc.collect()

    # Save metadata
    meta_path = os.path.join(OUT_DIR, "nuremberg_dashboard_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[{ts()}] Metadata saved to {meta_path}")
    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
