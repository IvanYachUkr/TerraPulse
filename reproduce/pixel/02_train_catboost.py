#!/usr/bin/env python3
"""
Step 2: Train CatBoost V5 pixel-wise land-cover classifier.

Trains the deployed CatBoost configuration (deep_unweighted) on per-pixel
features (spectral bands, indices, SAR, temporal diffs).
Uses RANDOM sampling (150K max per city).

Config: depth=8, trees=3000, lr=0.03, l2=3.0, no class weights.

Split:
  - Train: ~100 cities (excluding nuremberg + 15 val cities)
  - Val:   15 geographically diverse cities
  - Test:  6 held-out cities (nuremberg, ankara_test, sofia_test, riga_test, edinburgh_test, palermo_test)

Usage:
    python 02_train_catboost.py
    python 02_train_catboost.py --cities munich   # quick single-city test
"""

import argparse, gc, json, math, os, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
WC_TILES_DIR = os.path.join(CITIES_DIR, "worldcover_tiles")

# Import city list from MLP step 1
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "mlp"))
from importlib import import_module
step1 = import_module("01_download_data")
ALL_CITIES = step1.CITIES
CITY_MAP = step1.CITY_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_CLASSES = 7
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                  "B11", "B12"]
SENTINEL_NODATA = -9999
SAR_BANDS = ["vv", "vh"]
SAR_NODATA = -9999
SEASONS = ["spring", "summer", "autumn"]
YEARS = [2020, 2021]
WC_CLASS_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4,
                60: 5, 70: 5, 100: 5, 80: 6}
INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDMI", "NBR", "BSI", "EVI2",
               "NDRE1", "NDRE2"]
MAX_PX = 150000  # per city

# Deployed model hyperparameters
CATBOOST_DEPTH = 8
CATBOOST_TREES = 3000
CATBOOST_LR = 0.03
CATBOOST_L2 = 3.0

EXCLUDED = {
    "nuremberg",
    "ankara_test", "sofia_test", "riga_test", "edinburgh_test", "palermo_test",
}
VAL_CITIES = {
    "finnish_lakeland", "danish_farmland", "tabernas_desert",
    "sardinia_maquis", "crete_phrygana", "iceland_highlands",
    "lapland_tundra", "ireland_bog_pasture", "hortobagy_puszta",
    "vojvodina_cropland", "camargue_wetland", "pyrenees_meadows",
    "munich", "seville", "stockholm",
}


def ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _safe_ratio(a, b, eps=1e-10):
    denom = a + b
    mask = np.abs(denom) > eps
    result = np.full_like(a, np.nan, dtype=np.float32)
    result[mask] = (a[mask] - b[mask]) / denom[mask]
    return result


def _compute_indices(s2):
    B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 = [s2[i] for i in range(10)]
    denom_evi = B08 + 2.4 * B04 + 1.0
    return {
        "NDVI": _safe_ratio(B08, B04), "NDWI": _safe_ratio(B03, B08),
        "NDBI": _safe_ratio(B11, B08), "NDMI": _safe_ratio(B08, B11),
        "NBR": _safe_ratio(B08, B12), "BSI": _safe_ratio(B11 + B04, B08 + B02),
        "EVI2": np.where(np.abs(denom_evi) > 1e-10,
                         2.5 * (B08 - B04) / denom_evi, np.nan).astype(np.float32),
        "NDRE1": _safe_ratio(B08, B05), "NDRE2": _safe_ratio(B08, B06),
    }


def _load_tif(path, nodata_val):
    if not os.path.exists(path): return None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    data[data == nodata_val] = np.nan
    return data


def _raw_dir(city):
    d = os.path.join(CITIES_DIR, city.name, "raw_v7")
    if os.path.isdir(d): return d
    return os.path.join(CITIES_DIR, city.name, "raw")


def load_worldcover_pixels(city, year=2021):
    anchor_path = None
    raw = _raw_dir(city)
    for season in SEASONS:
        for y in YEARS:
            p = os.path.join(raw, f"sentinel2_{city.name}_{y}_{season}.tif")
            if os.path.exists(p): anchor_path = p; break
        if anchor_path: break
    if anchor_path is None or not os.path.exists(anchor_path): return None

    with rasterio.open(anchor_path) as ref:
        acrs, atr, aw, ah = ref.crs, ref.transform, ref.width, ref.height

    bbox = city.bbox
    west, south, east, north = bbox
    lat_lo = int(math.floor(south / 3.0)) * 3
    lat_hi = int(math.floor(north / 3.0)) * 3
    lon_lo = int(math.floor(west / 3.0)) * 3
    lon_hi = int(math.floor(east / 3.0)) * 3
    tiles = []
    for lat in range(lat_lo, lat_hi + 1, 3):
        for lon in range(lon_lo, lon_hi + 1, 3):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")

    version = "v100" if year == 2020 else "v200"
    dst = np.zeros((ah, aw), dtype=np.uint8)
    for tile in tiles:
        fn = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        wp = os.path.join(WC_TILES_DIR, fn)
        if not os.path.exists(wp): continue
        tmp = np.zeros_like(dst)
        with rasterio.open(wp) as src:
            reproject(source=rasterio.band(src, 1), destination=tmp,
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_transform=atr, dst_crs=acrs,
                      dst_nodata=0, resampling=Resampling.nearest)
        mask = (dst == 0) & (tmp > 0)
        dst[mask] = tmp[mask]

    labels = np.full((ah, aw), 255, dtype=np.uint8)
    for wc_code, cls in WC_CLASS_MAP.items():
        labels[dst == wc_code] = cls
    return labels


def build_pixel_features(city):
    """Build per-pixel feature matrix and labels."""
    labels = load_worldcover_pixels(city)
    if labels is None: return None, None, 0, 0, []
    H, W = labels.shape
    raw = _raw_dir(city)

    # Load TIFs in parallel
    tasks = {}
    for year in YEARS:
        for season in SEASONS:
            tag = f"{year}_{season}"
            tasks[f"s2_{tag}"] = (os.path.join(raw, f"sentinel2_{city.name}_{tag}.tif"), SENTINEL_NODATA)
            tasks[f"s1_{tag}"] = (os.path.join(raw, f"sentinel1_{city.name}_{tag}.tif"), SAR_NODATA)

    tifs = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_load_tif, p, nd): k for k, (p, nd) in tasks.items()}
        for f in as_completed(futures): tifs[futures[f]] = f.result()

    all_bands, band_names = [], []
    indices_by_tag, sar_by_tag = {}, {}
    has_any_s2 = False

    for year in YEARS:
        for season in SEASONS:
            tag = f"{year}_{season}"
            s2 = tifs.get(f"s2_{tag}")
            if s2 is not None and s2.shape[0] >= 11:
                has_any_s2 = True
                for bi, bn in enumerate(SENTINEL_BANDS):
                    all_bands.append(s2[bi]); band_names.append(f"{bn}_{tag}")
                idx = _compute_indices(s2[:10])
                indices_by_tag[tag] = idx
                for n in INDEX_NAMES:
                    all_bands.append(idx[n]); band_names.append(f"{n}_{tag}")
            else:
                for bn in SENTINEL_BANDS:
                    all_bands.append(np.full((H,W), np.nan, dtype=np.float32))
                    band_names.append(f"{bn}_{tag}")
                for n in INDEX_NAMES:
                    all_bands.append(np.full((H,W), np.nan, dtype=np.float32))
                    band_names.append(f"{n}_{tag}")

            s1 = tifs.get(f"s1_{tag}")
            if s1 is not None:
                for bi, sn in enumerate(SAR_BANDS):
                    all_bands.append(s1[bi]); band_names.append(f"SAR_{sn.upper()}_{tag}")
                vvvh = np.where(np.abs(s1[1]) > 1e-10, s1[0]/s1[1], np.nan).astype(np.float32)
                all_bands.append(vvvh); band_names.append(f"SAR_VVVH_{tag}")
                sar_by_tag[tag] = {"vv": s1[0], "vh": s1[1]}
            else:
                for sn in SAR_BANDS:
                    all_bands.append(np.full((H,W), np.nan, dtype=np.float32))
                    band_names.append(f"SAR_{sn.upper()}_{tag}")
                all_bands.append(np.full((H,W), np.nan, dtype=np.float32))
                band_names.append(f"SAR_VVVH_{tag}")

    if not has_any_s2: return None, None, 0, 0, []
    del tifs; gc.collect()

    # Temporal diffs
    for year in YEARS:
        for sf, st in [("spring","summer"), ("summer","autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in indices_by_tag and tt in indices_by_tag:
                for n in INDEX_NAMES:
                    all_bands.append((indices_by_tag[tt][n] - indices_by_tag[tf][n]).astype(np.float32))
                    band_names.append(f"{n}_diff_{st}_{sf}_{year}")
    for season in SEASONS:
        t0, t1 = f"2020_{season}", f"2021_{season}"
        if t0 in indices_by_tag and t1 in indices_by_tag:
            for n in INDEX_NAMES:
                all_bands.append((indices_by_tag[t1][n] - indices_by_tag[t0][n]).astype(np.float32))
                band_names.append(f"{n}_interannual_{season}")
    for year in YEARS:
        ts_s, ts_a = f"{year}_spring", f"{year}_autumn"
        if ts_s in indices_by_tag and ts_a in indices_by_tag:
            for n in ["NDVI", "NDWI", "EVI2", "BSI"]:
                all_bands.append((indices_by_tag[ts_a][n] - indices_by_tag[ts_s][n]).astype(np.float32))
                band_names.append(f"{n}_range_{year}")
    for year in YEARS:
        for sf, st in [("spring","summer"), ("summer","autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in sar_by_tag and tt in sar_by_tag:
                for b in ["vv","vh"]:
                    all_bands.append((sar_by_tag[tt][b] - sar_by_tag[tf][b]).astype(np.float32))
                    band_names.append(f"SAR_{b.upper()}_diff_{st}_{sf}_{year}")
    for season in SEASONS:
        t0, t1 = f"2020_{season}", f"2021_{season}"
        if t0 in sar_by_tag and t1 in sar_by_tag:
            for b in ["vv","vh"]:
                all_bands.append((sar_by_tag[t1][b] - sar_by_tag[t0][b]).astype(np.float32))
                band_names.append(f"SAR_{b.upper()}_interannual_{season}")

    n_f = len(all_bands)
    cube = np.stack(all_bands, axis=-1)
    del all_bands, indices_by_tag, sar_by_tag; gc.collect()

    valid_label = labels < N_CLASSES
    nan_count = np.isnan(cube).sum(axis=-1)
    valid = valid_label & (nan_count < n_f * 0.5)

    X = cube[valid]
    y = labels[valid].astype(np.int32)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    del cube, labels; gc.collect()
    return X, y, H, W, band_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CatBoost V5 pixel classifier")
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    np.random.seed(SEED)
    out_dir = os.path.join(CITIES_DIR, "models_pixel_v5")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  CatBoost V5 Pixel Classifier (deep_unweighted)")
    print(f"  depth={CATBOOST_DEPTH}, trees={CATBOOST_TREES}, lr={CATBOOST_LR}")
    print(f"  Sampling: RANDOM {MAX_PX//1000}K per city")
    print(f"{'='*70}\n")

    # City splits
    if args.cities:
        train_cities = [CITY_MAP[n] for n in args.cities if n in CITY_MAP]
        val_cities = []
    else:
        train_cities = [c for c in ALL_CITIES
                        if c.name not in EXCLUDED and c.name not in VAL_CITIES]
        val_cities = [c for c in ALL_CITIES if c.name in VAL_CITIES]

    print(f"  Train: {len(train_cities)} cities, Val: {len(val_cities)} cities\n")

    # Load data
    rng = np.random.RandomState(SEED)
    all_X_train, all_y_train = [], []
    feat_names = None

    print(f"[{ts()}] Loading TRAIN data...")
    for i, city in enumerate(train_cities):
        print(f"  [{ts()}] [{i+1}/{len(train_cities)}] {city.name}...")
        result = build_pixel_features(city)
        if result[0] is None: continue
        X, y, H, W, names = result
        if feat_names is None: feat_names = names
        if len(X) > MAX_PX:
            idx = rng.choice(len(X), MAX_PX, replace=False)
            X, y = X[idx], y[idx]
        all_X_train.append(X.astype(np.float16))
        all_y_train.append(y)
        del X, y; gc.collect()

    X_train = np.concatenate(all_X_train).astype(np.float32)
    y_train = np.concatenate(all_y_train)
    del all_X_train, all_y_train; gc.collect()

    if val_cities:
        print(f"\n[{ts()}] Loading VAL data...")
        all_X_val, all_y_val = [], []
        for i, city in enumerate(val_cities):
            print(f"  [{ts()}] [{i+1}/{len(val_cities)}] {city.name}...")
            result = build_pixel_features(city)
            if result[0] is None: continue
            X, y, H, W, names = result
            if len(X) > MAX_PX:
                idx = rng.choice(len(X), MAX_PX, replace=False)
                X, y = X[idx], y[idx]
            all_X_val.append(X.astype(np.float16))
            all_y_val.append(y)
            del X, y; gc.collect()
        X_val = np.concatenate(all_X_val).astype(np.float32)
        y_val = np.concatenate(all_y_val)
        del all_X_val, all_y_val; gc.collect()
    else:
        # Single-city mode: 80/20 split
        n = len(X_train)
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        X_val = X_train[perm[split:]]
        y_val = y_train[perm[split:]]
        X_train = X_train[perm[:split]]
        y_train = y_train[perm[:split]]

    print(f"\n  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    print(f"  Val:   {X_val.shape[0]:,} x {X_val.shape[1]}")

    # Train single deployed config
    print(f"\n{'='*70}")
    print(f"  Training CatBoost (deep_unweighted)")
    print(f"{'='*70}")

    for dev in ['GPU', 'CPU']:
        try:
            params = {
                'iterations': CATBOOST_TREES, 'depth': CATBOOST_DEPTH,
                'learning_rate': CATBOOST_LR, 'l2_leaf_reg': CATBOOST_L2,
                'random_seed': SEED, 'task_type': dev,
                'loss_function': 'MultiClass', 'eval_metric': 'MultiClass',
                'verbose': 100, 'early_stopping_rounds': 80,
                'use_best_model': True,
            }
            if dev == 'GPU':
                params['devices'] = '0'
                params['gpu_ram_part'] = 0.95

            print(f"\n  [{ts()}] Training on {dev}...")
            model = CatBoostClassifier(**params)
            model.fit(Pool(X_train, y_train, feature_names=feat_names),
                      eval_set=Pool(X_val, y_val, feature_names=feat_names))
            print(f"  Best iteration: {model.get_best_iteration()}")
            break
        except Exception as e:
            print(f"  {dev} failed: {e}")
            if dev == 'CPU': raise

    pred = model.predict(X_val).flatten().astype(int)
    acc = accuracy_score(y_val, pred)
    report = classification_report(y_val, pred, target_names=CLASS_NAMES, digits=4)
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(report)

    path = os.path.join(out_dir, "catboost_pixel_v5_deep_unweighted.cbm")
    model.save_model(path)
    print(f"  Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    with open(os.path.join(out_dir, "metrics_pixel_v5.json"), 'w') as f:
        json.dump({'accuracy': float(acc)}, f, indent=2)

    print(f"\n[{ts()}] Done!")
