#!/usr/bin/env python3
"""
Per-Pixel Land Cover Classification V2.

Improvements over V1:
  - Stratified sampling per class
  - GPU-accelerated LightGBM
  - Richer phenology features (temporal diffs for ALL indices + SAR)
  - Multi-config sweeping (bigger trees, more estimators)
  - Faster within-city TIF loading (parallel band reading)
  - Memory-safe: sequential city processing, float16 accumulated storage

Usage:
    # Quick sanity check
    python scripts/pixel_classifier_v2.py --max-pixels-per-city 10000 --cities munich --model lgbm

    # Full multi-city GPU training with big trees
    python scripts/pixel_classifier_v2.py --max-pixels-per-city 50000 --model lgbm --device gpu

    # Sweep multiple configs
    python scripts/pixel_classifier_v2.py --max-pixels-per-city 50000 --model lgbm --device gpu --sweep
"""

import argparse
import gc
import json
import math
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    CITIES, SEASONS, SENTINEL_BANDS, SENTINEL_NODATA,
    WC_CLASS_MAP, CLASS_NAMES, N_CLASSES, GRID_PX,
    city_dir, city_anchor_path,
    _wc_tiles_for_bbox,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAR_BANDS = ["vv", "vh"]
SAR_NODATA = -9999
YEARS = [2020, 2021]
SEED = 42

VAL_CITY_NAMES = {
    'munich', 'vojvodina_cropland', 'hortobagy_puszta',
    'seville', 'crete_phrygana', 'tabernas_desert', 'sardinia_maquis',
    'camargue_wetland', 'pyrenees_meadows',
    'ireland_bog_pasture', 'danish_farmland',
    'stockholm', 'finnish_lakeland', 'iceland_highlands', 'lapland_tundra',
}
EXCLUDED_CITY_NAMES = {'nuremberg'}

CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
WC_TILES_DIR = os.path.join(CITIES_DIR, "worldcover_tiles")
MIN_PER_CLASS = 500

# Index names we compute
INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDMI", "NBR", "BSI", "EVI2",
               "NDRE1", "NDRE2"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts():
    return time.strftime("%H:%M:%S")


def _raw_dir(city):
    d = os.path.join(city_dir(city), "raw_v7")
    if os.path.isdir(d):
        return d
    return os.path.join(city_dir(city), "raw")


def _safe_ratio(a, b, eps=1e-10):
    denom = a + b
    mask = np.abs(denom) > eps
    result = np.full_like(a, np.nan, dtype=np.float32)
    result[mask] = (a[mask] - b[mask]) / denom[mask]
    return result


# ---------------------------------------------------------------------------
# Fast data loading (parallel TIF reading within city)
# ---------------------------------------------------------------------------

def _load_tif(path, nodata_val):
    """Load a single TIF, return data array or None."""
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    data[data == nodata_val] = np.nan
    return data


def load_all_tifs_parallel(city):
    """Load all S2+S1 TIFs for a city in parallel. Returns dict of arrays."""
    raw = _raw_dir(city)
    tasks = {}

    for year in YEARS:
        for season in SEASONS:
            tag = f"{year}_{season}"
            s2_path = os.path.join(raw, f"sentinel2_{city.name}_{year}_{season}.tif")
            s1_path = os.path.join(raw, f"sentinel1_{city.name}_{year}_{season}.tif")
            tasks[f"s2_{tag}"] = (s2_path, SENTINEL_NODATA)
            tasks[f"s1_{tag}"] = (s1_path, SAR_NODATA)

    results = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_load_tif, path, nd): key
                   for key, (path, nd) in tasks.items()}
        for f in as_completed(futures):
            key = futures[f]
            results[key] = f.result()

    return results


def load_worldcover_pixels(city, year=2021):
    anchor_path = city_anchor_path(city)
    if not os.path.exists(anchor_path):
        return None

    with rasterio.open(anchor_path) as ref:
        anchor_crs = ref.crs
        anchor_transform = ref.transform
        anchor_width = ref.width
        anchor_height = ref.height

    tiles = _wc_tiles_for_bbox(city.bbox)
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
# Feature engineering (richer phenology)
# ---------------------------------------------------------------------------

def _compute_indices(s2):
    """Compute spectral indices from S2 (10, H, W) -> dict of (H, W) arrays."""
    B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 = [s2[i] for i in range(10)]

    ndvi = _safe_ratio(B08, B04)
    ndwi = _safe_ratio(B03, B08)
    ndbi = _safe_ratio(B11, B08)
    ndmi = _safe_ratio(B08, B11)
    nbr  = _safe_ratio(B08, B12)
    bsi  = _safe_ratio(B11 + B04, B08 + B02)
    denom_evi = B08 + 2.4 * B04 + 1.0
    evi2 = np.where(np.abs(denom_evi) > 1e-10,
                    2.5 * (B08 - B04) / denom_evi, np.nan).astype(np.float32)
    ndre1 = _safe_ratio(B08, B05)
    ndre2 = _safe_ratio(B08, B06)

    return {
        "NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "NDMI": ndmi,
        "NBR": nbr, "BSI": bsi, "EVI2": evi2, "NDRE1": ndre1, "NDRE2": ndre2,
    }


def build_pixel_features(city):
    """
    Build per-pixel feature matrix and labels.
    V2: parallel TIF loading + richer phenology features.
    Returns: X (N, F), y (N,), H, W, feature_names
    """
    labels = load_worldcover_pixels(city)
    if labels is None:
        return None, None, 0, 0, []

    H, W = labels.shape

    # Load all TIFs in parallel
    tifs = load_all_tifs_parallel(city)

    all_bands = []
    band_names = []
    # Store indices per tag for temporal diffs
    indices_by_tag = {}  # tag -> {index_name: array}
    sar_by_tag = {}      # tag -> {vv: array, vh: array}
    has_any_s2 = False

    for year in YEARS:
        for season in SEASONS:
            tag = f"{year}_{season}"

            # --- Sentinel-2 ---
            s2_data = tifs.get(f"s2_{tag}")
            if s2_data is not None and s2_data.shape[0] >= 11:
                has_any_s2 = True
                s2 = s2_data[:10]  # spectral bands

                # Raw bands
                for bi, bname in enumerate(SENTINEL_BANDS):
                    all_bands.append(s2[bi])
                    band_names.append(f"{bname}_{tag}")

                # Spectral indices
                idx_dict = _compute_indices(s2)
                indices_by_tag[tag] = idx_dict

                for idx_name in INDEX_NAMES:
                    all_bands.append(idx_dict[idx_name])
                    band_names.append(f"{idx_name}_{tag}")
            else:
                # NaN placeholders
                for bname in SENTINEL_BANDS:
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                    band_names.append(f"{bname}_{tag}")
                for idx_name in INDEX_NAMES:
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                    band_names.append(f"{idx_name}_{tag}")

            # --- Sentinel-1 ---
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
        return None, None, 0, 0, []

    # Free raw TIF data
    del tifs
    gc.collect()

    # --- RICH TEMPORAL DIFFS (the main V2 improvement for features) ---
    # Intra-annual diffs for ALL indices (not just NDVI)
    for year in YEARS:
        for s_from, s_to in [("spring", "summer"), ("summer", "autumn")]:
            tf = f"{year}_{s_from}"
            tt = f"{year}_{s_to}"
            if tf in indices_by_tag and tt in indices_by_tag:
                for idx_name in INDEX_NAMES:
                    diff = indices_by_tag[tt][idx_name] - indices_by_tag[tf][idx_name]
                    all_bands.append(diff.astype(np.float32))
                    band_names.append(f"{idx_name}_diff_{s_to}_{s_from}_{year}")

    # Inter-annual diffs for ALL indices
    for season in SEASONS:
        t0 = f"2020_{season}"
        t1 = f"2021_{season}"
        if t0 in indices_by_tag and t1 in indices_by_tag:
            for idx_name in INDEX_NAMES:
                diff = indices_by_tag[t1][idx_name] - indices_by_tag[t0][idx_name]
                all_bands.append(diff.astype(np.float32))
                band_names.append(f"{idx_name}_interannual_{season}")

    # Spring-to-autumn range (full growing season amplitude) for key indices
    for year in YEARS:
        ts_spring = f"{year}_spring"
        ts_autumn = f"{year}_autumn"
        if ts_spring in indices_by_tag and ts_autumn in indices_by_tag:
            for idx_name in ["NDVI", "NDWI", "EVI2", "BSI"]:
                rng = indices_by_tag[ts_autumn][idx_name] - indices_by_tag[ts_spring][idx_name]
                all_bands.append(rng.astype(np.float32))
                band_names.append(f"{idx_name}_range_{year}")

    # SAR temporal diffs (VV and VH)
    for year in YEARS:
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
        t0 = f"2020_{season}"
        t1 = f"2021_{season}"
        if t0 in sar_by_tag and t1 in sar_by_tag:
            for band in ["vv", "vh"]:
                diff = sar_by_tag[t1][band] - sar_by_tag[t0][band]
                all_bands.append(diff.astype(np.float32))
                band_names.append(f"SAR_{band.upper()}_interannual_{season}")

    n_features = len(all_bands)

    feature_cube = np.stack(all_bands, axis=-1)
    del all_bands, indices_by_tag, sar_by_tag
    gc.collect()

    valid_label = labels < N_CLASSES
    nan_count = np.isnan(feature_cube).sum(axis=-1)
    valid_features = nan_count < (n_features * 0.5)
    valid = valid_label & valid_features

    n_valid = int(valid.sum())

    X = feature_cube[valid]
    y = labels[valid].astype(np.int32)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    del feature_cube, labels
    gc.collect()

    return X, y, H, W, band_names


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_subsample(X, y, max_pixels, min_per_class=MIN_PER_CLASS, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED)
    n_total = len(X)
    if n_total <= max_pixels:
        return X, y

    classes = np.unique(y)
    selected_indices = []
    budget_used = 0

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n_take = min(min_per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, n_take, replace=False)
        selected_indices.append(chosen)
        budget_used += n_take

    remaining = max_pixels - budget_used
    if remaining > 0:
        already_selected = set(np.concatenate(selected_indices))
        pool = np.array([i for i in range(n_total) if i not in already_selected])
        if len(pool) > 0:
            n_fill = min(remaining, len(pool))
            fill_idx = rng.choice(pool, n_fill, replace=False)
            selected_indices.append(fill_idx)

    all_idx = np.concatenate(selected_indices)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


# ---------------------------------------------------------------------------
# Sequential city loading (memory-safe: only 1 full raster at a time)
# ---------------------------------------------------------------------------

def load_cities_sequential(cities, max_px, min_per_class):
    """
    Load cities ONE AT A TIME to avoid OOM.
    Each city's full raster (~2.6 GB) is loaded, subsampled to 50k pixels,
    stored as float16 (~21 KB), then the full raster is freed.
    Peak RAM: ~3 GB (one city) + accumulated float16 results.
    """
    all_X = []
    all_y = []
    feature_names = []

    for i, city in enumerate(cities):
        rng = np.random.RandomState(SEED + i * 7)
        try:
            X, y, H, W, names = build_pixel_features(city)
            if X is None or len(X) == 0:
                continue

            if not feature_names:
                feature_names = names

            X, y = stratified_subsample(X, y, max_px, min_per_class, rng)

            # Store as float16 to halve memory (50k x 217 x 2 = 21 KB vs 43 KB)
            all_X.append(X.astype(np.float16))
            all_y.append(y)

            classes, counts = np.unique(y, return_counts=True)
            cls_str = " ".join(f"{CLASS_NAMES[c][:4]}={cnt}"
                               for c, cnt in zip(classes, counts))
            print(f"  [{ts()}] [{city.name}] {len(y):,}px {X.shape[1]}f -> {cls_str}")

            del X, y
            gc.collect()
        except Exception as e:
            print(f"  [{city.name}] ERROR: {e}")

    return all_X, all_y, feature_names


# ---------------------------------------------------------------------------
# Feature data caching (skip 27-min TIF loading on reruns)
# ---------------------------------------------------------------------------

DATA_CACHE_DIR = os.path.join(CITIES_DIR, "pixel_cache_v2")


def _cache_path(tag, max_px, min_pc):
    """Cache file path based on params."""
    return os.path.join(DATA_CACHE_DIR, f"{tag}_px{max_px}_mc{min_pc}.npz")


def save_data_cache(X_train, y_train, X_val, y_val, feat_names, max_px, min_pc):
    """Save loaded data to disk as compressed float16."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    path = _cache_path("data", max_px, min_pc)
    print(f"  [{ts()}] Saving feature cache to {path}...")
    np.savez_compressed(
        path,
        X_train=X_train.astype(np.float16),
        y_train=y_train,
        X_val=X_val.astype(np.float16),
        y_val=y_val,
    )
    # Save feature names separately
    names_path = _cache_path("names", max_px, min_pc).replace('.npz', '.json')
    with open(names_path, 'w') as f:
        json.dump(feat_names, f)
    sz = os.path.getsize(path) / (1024**3)
    print(f"  [{ts()}] Cache saved ({sz:.2f} GB)")


def load_data_cache(max_px, min_pc):
    """Load cached data if exists. Returns tuple or None."""
    path = _cache_path("data", max_px, min_pc)
    names_path = _cache_path("names", max_px, min_pc).replace('.npz', '.json')
    if not os.path.exists(path) or not os.path.exists(names_path):
        return None
    print(f"  [{ts()}] Loading from cache: {path}")
    d = np.load(path)
    X_train = d['X_train'].astype(np.float32)
    y_train = d['y_train']
    X_val = d['X_val'].astype(np.float32)
    y_val = d['y_val']
    with open(names_path) as f:
        feat_names = json.load(f)
    sz = os.path.getsize(path) / (1024**3)
    print(f"  [{ts()}] Cache loaded ({sz:.2f} GB)")
    return X_train, y_train, X_val, y_val, feat_names


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgbm(X_train, y_train, X_val, y_val, config, feature_names=None):
    """Train LightGBM with given config dict. Auto-falls back CPU on GPU crash."""
    import lightgbm as lgb

    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weight_dict = {c: total / (len(classes) * cnt) for c, cnt in
                   zip(classes, counts)}
    sample_weights = np.array([weight_dict[yi] for yi in y_train],
                              dtype=np.float32)

    cname = config.get('name', 'default')
    device = config.get('device', 'cpu')

    def _build_params(dev):
        params = {
            'objective': 'multiclass',
            'num_class': N_CLASSES,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 255),
            'max_depth': config.get('max_depth', 15),
            'learning_rate': config.get('lr', 0.05),
            'n_estimators': config.get('n_estimators', 2000),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': config.get('min_child_samples', 50),
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': SEED,
            'verbose': -1,
            'n_jobs': -1,
        }
        if dev in ('cuda', 'gpu'):
            params['device'] = dev
            if dev == 'gpu':
                params['gpu_use_dp'] = False
        return params

    # Device priority: cuda -> gpu -> cpu
    devices_to_try = []
    if device == 'cuda':
        devices_to_try = ['cuda', 'cpu']
    elif device == 'gpu':
        devices_to_try = ['gpu', 'cpu']
    else:
        devices_to_try = ['cpu']

    for dev in devices_to_try:
        try:
            print(f"\n  [{ts()}] Training LightGBM [{cname}] on {dev.upper()}...")
            print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
            print(f"  Val:   {X_val.shape[0]:,}")

            params = _build_params(dev)
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(80, verbose=True),
                    lgb.log_evaluation(100),
                ],
            )
            print(f"  Best iteration: {model.best_iteration_}")
            return model

        except Exception as e:
            print(f"\n  [!] {dev.upper()} training FAILED: {e}")
            if dev != devices_to_try[-1]:
                next_dev = devices_to_try[devices_to_try.index(dev) + 1]
                print(f"  [!] Falling back to {next_dev.upper()}...")
            else:
                raise  # no more fallbacks


def train_mlp(X_train, y_train, X_val, y_val, n_features, device='cuda'):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weights = torch.tensor([total / (len(classes) * counts[i])
                            for i in range(len(classes))],
                           dtype=torch.float32, device=device)

    class PixelMLP(nn.Module):
        def __init__(self, in_f, n_cls, widths=[512, 256, 128]):
            super().__init__()
            layers = []
            prev = in_f
            for w in widths:
                layers.extend([nn.Linear(prev, w), nn.BatchNorm1d(w),
                               nn.SiLU(), nn.Dropout(0.15)])
                prev = w
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(prev, n_cls)
        def forward(self, x):
            return self.head(self.backbone(x))

    net = PixelMLP(n_features, N_CLASSES).to(device)
    print(f"\n  [{ts()}] MLP ({sum(p.numel() for p in net.parameters()):,} params) on {device}")

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights)
    X_t = torch.tensor(X_train_s, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_v = torch.tensor(X_val_s, device=device)
    y_v = torch.tensor(y_val, dtype=torch.long, device=device)

    BATCH = 4096
    best_vl, best_st, patience, wait = float('inf'), None, 20, 0
    n_train = X_t.shape[0]

    for epoch in range(200):
        net.train()
        perm = torch.randperm(n_train, device=device)
        eloss = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i+BATCH]; xb, yb = X_t[idx], y_t[idx]
            opt.zero_grad(); pred = net(xb); loss = crit(pred, yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); eloss += loss.item()

        net.eval()
        with torch.no_grad():
            vl_parts = [net(X_v[i:i+8192]) for i in range(0, X_v.shape[0], 8192)]
            vl = torch.cat(vl_parts)
            vl_loss = crit(vl, y_v).item()
            vl_acc = (vl.argmax(1) == y_v).float().mean().item()

        if epoch % 10 == 0 or epoch < 5:
            print(f"  Ep {epoch:3d} | train={eloss/((n_train+BATCH-1)//BATCH):.4f} "
                  f"val={vl_loss:.4f} acc={vl_acc:.3f}")
        if vl_loss < best_vl:
            best_vl = vl_loss
            best_st = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop ep {epoch} (best={best_vl:.4f})")
                break

    net.load_state_dict(best_st); net.eval()
    return {'model': net, 'scaler': scaler}


def evaluate_model(model, X_test, y_test, model_type='lgbm'):
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    if model_type == 'lgbm':
        y_pred = model.predict(X_test)
    else:
        import torch
        net = model['model']; scaler = model['scaler']
        X_s = scaler.transform(X_test).astype(np.float32)
        device = next(net.parameters()).device
        preds = []
        net.eval()
        with torch.no_grad():
            for i in range(0, len(X_s), 8192):
                preds.append(net(torch.tensor(X_s[i:i+8192], device=device)).argmax(1).cpu().numpy())
        y_pred = np.concatenate(preds)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    present = sorted(set(y_test) | set(y_pred))
    names = [CLASS_NAMES[i] for i in present]
    report = classification_report(y_test, y_pred, labels=present,
                                   target_names=names, digits=4, zero_division=0)
    print(f"\n{report}")

    cm = confusion_matrix(y_test, y_pred, labels=present)
    print("  Confusion Matrix:")
    header = "          " + " ".join(f"{n[:6]:>6}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {names[i]:>8} " + " ".join(f"{v:6d}" for v in row))

    return {'accuracy': float(acc), 'report': report, 'cm': cm.tolist()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-pixel classifier V2")
    parser.add_argument('--model', choices=['lgbm', 'mlp', 'both'],
                        default='lgbm')
    parser.add_argument('--max-pixels-per-city', type=int, default=50000)
    parser.add_argument('--min-per-class', type=int, default=500)
    parser.add_argument('--cities', nargs='*', default=None)
    parser.add_argument('--device', default='gpu',
                        help='cpu or gpu for LightGBM, cuda for MLP')
    parser.add_argument('--sweep', action='store_true',
                        help='Try multiple LightGBM configs')
    args = parser.parse_args()

    np.random.seed(SEED)
    out_dir = os.path.join(CITIES_DIR, "models_pixel_v2")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Per-Pixel Land Cover Classification V2")
    print(f"  Model: {args.model} | Max px/city: {args.max_pixels_per_city:,}")
    print(f"  Stratified: min {args.min_per_class}/class | Sequential loading")
    print(f"  Device: {args.device} | Sweep: {args.sweep}")
    print(f"{'='*70}\n")

    # --- City splits ---
    if args.cities:
        selected = [c for c in CITIES if c.name in args.cities]
        if not selected:
            print(f"ERROR: No cities matched {args.cities}"); return
        train_cities = selected; val_cities = []; single_city_mode = True
        print(f"  Single-city mode: {[c.name for c in selected]}")
    else:
        train_cities = [c for c in CITIES
                        if c.name not in EXCLUDED_CITY_NAMES
                        and c.name not in VAL_CITY_NAMES]
        val_cities = [c for c in CITIES if c.name in VAL_CITY_NAMES]
        single_city_mode = False
        print(f"  Train: {len(train_cities)} cities | Val: {len(val_cities)} cities")

    # --- Load data (with cache support) ---
    t_load = time.time()

    # Try loading from cache first
    cached = None
    if not single_city_mode:
        cached = load_data_cache(args.max_pixels_per_city, args.min_per_class)

    if cached is not None:
        X_train, y_train, X_val, y_val, feat_names = cached
    else:
        if single_city_mode:
            print(f"\n[{ts()}] Loading (single-city)...")
            all_X_train, all_y_train = [], []
            all_X_val, all_y_val = [], []
            feat_names = []

            for city in train_cities:
                X, y, H, W, feat_names = build_pixel_features(city)
                if X is None: continue
                X, y = stratified_subsample(X, y, args.max_pixels_per_city,
                                            args.min_per_class)
                classes, counts = np.unique(y, return_counts=True)
                cls_str = " ".join(f"{CLASS_NAMES[c][:4]}={cnt}"
                                   for c, cnt in zip(classes, counts))
                print(f"  [{city.name}] {len(y):,}px {X.shape[1]}f -> {cls_str}")
                n = len(X); perm = np.random.permutation(n); split = int(0.8*n)
                all_X_train.append(X[perm[:split]].astype(np.float16))
                all_y_train.append(y[perm[:split]])
                all_X_val.append(X[perm[split:]].astype(np.float16))
                all_y_val.append(y[perm[split:]])
                del X, y; gc.collect()
        else:
            # Sequential: only 1 city raster in memory at a time
            print(f"\n[{ts()}] Loading TRAIN ({len(train_cities)} cities, sequential)...")
            all_X_train, all_y_train, feat_names = load_cities_sequential(
                train_cities, args.max_pixels_per_city, args.min_per_class)

            print(f"\n[{ts()}] Loading VAL ({len(val_cities)} cities, sequential)...")
            all_X_val, all_y_val, _ = load_cities_sequential(
                val_cities, args.max_pixels_per_city, args.min_per_class)

        if not all_X_train or not all_X_val:
            print("ERROR: No data!"); return

        # Concatenate and convert back to float32 for training
        X_train = np.concatenate(all_X_train).astype(np.float32)
        y_train = np.concatenate(all_y_train)
        X_val = np.concatenate(all_X_val).astype(np.float32)
        y_val = np.concatenate(all_y_val)
        del all_X_train, all_y_train, all_X_val, all_y_val; gc.collect()

        # Save cache for future runs
        if not single_city_mode:
            save_data_cache(X_train, y_train, X_val, y_val, feat_names,
                            args.max_pixels_per_city, args.min_per_class)

    load_time = time.time() - t_load
    print(f"\n[{ts()}] Loading: {load_time:.1f}s")

    n_features = X_train.shape[1]
    print(f"\n[{ts()}] Data: Train {X_train.shape[0]:,} x {n_features} | "
          f"Val {X_val.shape[0]:,} x {n_features}")
    if feat_names:
        print(f"  Features: {feat_names[:3]}...{feat_names[-3:]}")

    for sn, sy in [("Train", y_train), ("Val", y_val)]:
        cls, cnt = np.unique(sy, return_counts=True); tot = cnt.sum()
        print(f"\n  {sn} distribution:")
        for c, n in zip(cls, cnt):
            print(f"    {CLASS_NAMES[c]:>15}: {n:>8,} ({100*n/tot:5.1f}%)")

    # --- Train ---
    results = {}

    if args.model in ('lgbm', 'both'):
        if args.sweep:
            configs = [
                {'name': 'baseline', 'num_leaves': 255, 'max_depth': 15,
                 'n_estimators': 2000, 'lr': 0.05, 'device': args.device},
                {'name': 'big_tree', 'num_leaves': 511, 'max_depth': 18,
                 'n_estimators': 2000, 'lr': 0.03, 'device': args.device},
                {'name': 'deep_slow', 'num_leaves': 255, 'max_depth': 20,
                 'n_estimators': 3000, 'lr': 0.02, 'min_child_samples': 30,
                 'device': args.device},
            ]
        else:
            configs = [
                {'name': 'default', 'num_leaves': 255, 'max_depth': 15,
                 'n_estimators': 2000, 'lr': 0.05, 'device': args.device},
            ]

        for config in configs:
            cname = config['name']
            print(f"\n{'='*70}")
            print(f"  LightGBM [{cname}]")
            print(f"{'='*70}")
            t0 = time.time()
            model = train_lgbm(X_train, y_train, X_val, y_val, config,
                               feat_names)
            elapsed = time.time() - t0
            print(f"\n  [{ts()}] Trained in {elapsed:.1f}s")

            print(f"\n  --- [{cname}] Validation ---")
            metrics = evaluate_model(model, X_val, y_val, 'lgbm')
            results[f'lgbm_{cname}'] = metrics

            # Feature importance top 20
            if hasattr(model, 'feature_importances_') and feat_names:
                imp = model.feature_importances_
                top_idx = np.argsort(imp)[::-1][:20]
                print(f"\n  Top 20 Features:")
                for rank, idx in enumerate(top_idx):
                    fname = feat_names[idx] if idx < len(feat_names) else f"f_{idx}"
                    print(f"    {rank+1:2d}. {fname:<35s}: {imp[idx]:>6.0f}")

            # Save best
            path = os.path.join(out_dir, f"lgbm_pixel_v2_{cname}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"\n  Saved: {path}")

    if args.model in ('mlp', 'both'):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n{'='*70}\n  MLP Training\n{'='*70}")
        t0 = time.time()
        mlp = train_mlp(X_train, y_train, X_val, y_val, n_features, device)
        print(f"\n  [{ts()}] MLP trained in {time.time()-t0:.1f}s")
        metrics = evaluate_model(mlp, X_val, y_val, 'mlp')
        results['mlp'] = metrics
        torch.save({'state': mlp['model'].state_dict(),
                     'scaler_mean': mlp['scaler'].mean_,
                     'scaler_scale': mlp['scaler'].scale_,
                     'n_features': n_features},
                    os.path.join(out_dir, "mlp_pixel_v2.pt"))

    # Save metrics
    def jsonify(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(os.path.join(out_dir, "metrics_pixel_v2.json"), 'w') as f:
        json.dump(results, f, indent=2, default=jsonify)

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
