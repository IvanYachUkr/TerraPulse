#!/usr/bin/env python3
"""
Per-Pixel Land Cover Classification.

Reads Sentinel-2/1 TIFFs directly, extracts per-pixel spectral/SAR features,
pairs with WorldCover per-pixel labels, and trains a classifier (LightGBM or MLP).

Usage:
    # Quick sanity check on Munich
    .venv/Scripts/python.exe scripts/pixel_classifier.py --max-pixels-per-city 10000 --cities munich --model lgbm

    # Full multi-city LightGBM training
    .venv/Scripts/python.exe scripts/pixel_classifier.py --max-pixels-per-city 50000 --model lgbm

    # With 3x3 neighborhood context
    .venv/Scripts/python.exe scripts/pixel_classifier.py --max-pixels-per-city 50000 --model lgbm --use-neighbors
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
from datetime import datetime

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

# Same val cities as v9
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts():
    return time.strftime("%H:%M:%S")


def _raw_dir(city):
    """Prefer raw_v7/ (has S1+S2), fall back to raw/."""
    d = os.path.join(city_dir(city), "raw_v7")
    if os.path.isdir(d):
        return d
    return os.path.join(city_dir(city), "raw")


def _safe_ratio(a, b, eps=1e-10):
    """Element-wise (a-b)/(a+b) with NaN-safe division."""
    denom = a + b
    mask = np.abs(denom) > eps
    result = np.full_like(a, np.nan, dtype=np.float32)
    result[mask] = (a[mask] - b[mask]) / denom[mask]
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sentinel2(city, year, season):
    """Load S2 TIF → (10, H, W) float32 + valid_fraction (H, W)."""
    raw = _raw_dir(city)
    path = os.path.join(raw, f"sentinel2_{city.name}_{year}_{season}.tif")
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # (11, H, W) — 10 spectral + valid_fraction
    spectral = data[:10]  # (10, H, W)
    vf = data[10]          # (H, W)
    # Replace nodata with NaN
    spectral[spectral == SENTINEL_NODATA] = np.nan
    vf[vf == SENTINEL_NODATA] = np.nan
    return spectral, vf


def load_sentinel1(city, year, season):
    """Load S1 TIF → (2, H, W) float32."""
    raw = _raw_dir(city)
    path = os.path.join(raw, f"sentinel1_{city.name}_{year}_{season}.tif")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # (2, H, W)
    data[data == SAR_NODATA] = np.nan
    return data


def load_worldcover_pixels(city, year=2021):
    """Load WorldCover and reproject to city anchor grid → (H, W) class IDs 0-6."""
    anchor_path = city_anchor_path(city)
    if not os.path.exists(anchor_path):
        return None

    with rasterio.open(anchor_path) as ref:
        anchor_crs = ref.crs
        anchor_transform = ref.transform
        anchor_width = ref.width
        anchor_height = ref.height

    # Find WC tiles
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

    # Map WC codes to our 0-6 class IDs
    label_array = np.full((anchor_height, anchor_width), 255, dtype=np.uint8)
    for wc_code, our_class in WC_CLASS_MAP.items():
        label_array[dst_array == wc_code] = our_class

    return label_array


# ---------------------------------------------------------------------------
# Feature engineering (per-pixel, working on flat arrays)
# ---------------------------------------------------------------------------

def build_pixel_features(city, return_mask=False):
    """
    Build per-pixel feature matrix and labels for a city.

    Returns:
        X: (N, F) float32 feature matrix
        y: (N,) int32 class labels 0-6
        H, W: spatial dimensions of the raster
    """
    print(f"  [{ts()}] [{city.name}] Loading WorldCover labels...")
    labels = load_worldcover_pixels(city)
    if labels is None:
        print(f"  [{city.name}] WARNING: No WorldCover labels — skip")
        return None, None, 0, 0

    H, W = labels.shape
    print(f"  [{city.name}] Raster: {H}×{W} = {H*W:,} pixels")

    # Collect all bands across seasons/years
    all_bands = []       # list of (H, W) arrays
    band_names = []      # feature names
    ndvi_list = []       # NDVI per season for temporal diffs

    has_any_s2 = False
    has_any_s1 = False

    for year in YEARS:
        for season in SEASONS:
            tag = f"{year}_{season}"

            # --- Sentinel-2 ---
            s2, vf = load_sentinel2(city, year, season)
            if s2 is not None:
                has_any_s2 = True
                # Band order: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
                for bi, bname in enumerate(SENTINEL_BANDS):
                    all_bands.append(s2[bi])
                    band_names.append(f"{bname}_{tag}")

                # Spectral indices
                B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 = [
                    s2[i] for i in range(10)]

                ndvi = _safe_ratio(B08, B04)
                ndwi = _safe_ratio(B03, B08)
                ndbi = _safe_ratio(B11, B08)
                ndmi = _safe_ratio(B08, B11)
                nbr = _safe_ratio(B08, B12)
                # BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
                bsi = _safe_ratio(B11 + B04, B08 + B02)
                # EVI2 = 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1)
                denom_evi = B08 + 2.4 * B04 + 1.0
                evi2 = np.where(np.abs(denom_evi) > 1e-10,
                                2.5 * (B08 - B04) / denom_evi, np.nan).astype(np.float32)
                ndre1 = _safe_ratio(B08, B05)
                ndre2 = _safe_ratio(B08, B06)

                for idx_arr, idx_name in [
                    (ndvi, "NDVI"), (ndwi, "NDWI"), (ndbi, "NDBI"),
                    (ndmi, "NDMI"), (nbr, "NBR"), (bsi, "BSI"),
                    (evi2, "EVI2"), (ndre1, "NDRE1"), (ndre2, "NDRE2"),
                ]:
                    all_bands.append(idx_arr)
                    band_names.append(f"{idx_name}_{tag}")

                ndvi_list.append((tag, ndvi))
                del s2, vf
            else:
                # Append NaN placeholders
                for bname in SENTINEL_BANDS:
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                    band_names.append(f"{bname}_{tag}")
                for idx_name in ["NDVI", "NDWI", "NDBI", "NDMI", "NBR",
                                 "BSI", "EVI2", "NDRE1", "NDRE2"]:
                    nan_arr = np.full((H, W), np.nan, dtype=np.float32)
                    all_bands.append(nan_arr)
                    band_names.append(f"{idx_name}_{tag}")
                    if idx_name == "NDVI":
                        ndvi_list.append((tag, nan_arr))

            # --- Sentinel-1 ---
            s1 = load_sentinel1(city, year, season)
            if s1 is not None:
                has_any_s1 = True
                for bi, sname in enumerate(SAR_BANDS):
                    all_bands.append(s1[bi])
                    band_names.append(f"SAR_{sname.upper()}_{tag}")

                # VV/VH ratio
                vv_vh = np.where(np.abs(s1[1]) > 1e-10,
                                 s1[0] / s1[1], np.nan).astype(np.float32)
                all_bands.append(vv_vh)
                band_names.append(f"SAR_VVVH_{tag}")
                del s1
            else:
                for sname in SAR_BANDS:
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                    band_names.append(f"SAR_{sname.upper()}_{tag}")
                all_bands.append(np.full((H, W), np.nan, dtype=np.float32))
                band_names.append(f"SAR_VVVH_{tag}")

    if not has_any_s2:
        print(f"  [{city.name}] WARNING: No S2 data at all — skip")
        return None, None, 0, 0

    # --- Temporal difference features (NDVI changes) ---
    # Group NDVI by year-season for computing diffs
    ndvi_by_tag = {tag: arr for tag, arr in ndvi_list}
    for year in YEARS:
        for s_from, s_to in [("spring", "summer"), ("summer", "autumn")]:
            tag_from = f"{year}_{s_from}"
            tag_to = f"{year}_{s_to}"
            if tag_from in ndvi_by_tag and tag_to in ndvi_by_tag:
                diff = ndvi_by_tag[tag_to] - ndvi_by_tag[tag_from]
                all_bands.append(diff.astype(np.float32))
                band_names.append(f"NDVI_diff_{s_to}_{s_from}_{year}")

    # Inter-annual diffs (2021 summer - 2020 summer)
    for season in SEASONS:
        t0 = f"2020_{season}"
        t1 = f"2021_{season}"
        if t0 in ndvi_by_tag and t1 in ndvi_by_tag:
            diff = ndvi_by_tag[t1] - ndvi_by_tag[t0]
            all_bands.append(diff.astype(np.float32))
            band_names.append(f"NDVI_interannual_{season}")

    n_features = len(all_bands)
    print(f"  [{city.name}] {n_features} features: {band_names[:5]}...{band_names[-3:]}")

    # Stack into (H, W, F)
    feature_cube = np.stack(all_bands, axis=-1)  # (H, W, F)
    del all_bands
    gc.collect()

    # Create valid pixel mask:
    # - Label must be 0-6 (not 255 = unmapped)
    # - At least *some* non-NaN features
    valid_label = labels < N_CLASSES
    nan_count = np.isnan(feature_cube).sum(axis=-1)
    valid_features = nan_count < (n_features * 0.5)  # tolerate up to 50% NaN
    valid = valid_label & valid_features

    n_valid = valid.sum()
    print(f"  [{city.name}] Valid pixels: {n_valid:,} / {H*W:,} "
          f"({100*n_valid/(H*W):.1f}%)")

    if return_mask:
        return valid

    # Extract valid pixels
    X = feature_cube[valid]   # (N, F)
    y = labels[valid].astype(np.int32)  # (N,)

    # Replace remaining NaN with 0 (for tree models) or will be scaled for MLP
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    del feature_cube, labels
    gc.collect()

    return X, y, H, W


def add_neighbor_features(X, y, H, W, valid_mask):
    """
    For each valid pixel, add features from 8 neighbors.
    Returns X_aug (N, F*9), y unchanged.

    This is memory-intensive — only for small subsets.
    """
    F = X.shape[1]
    N = X.shape[0]

    # Reconstruct the full feature cube with zeros for invalid pixels
    cube = np.zeros((H, W, F), dtype=np.float32)
    cube[valid_mask] = X

    # Pad with zeros (1 pixel border)
    padded = np.pad(cube, ((1, 1), (1, 1), (0, 0)), mode='constant',
                    constant_values=0)

    # Get valid pixel coordinates
    rows, cols = np.where(valid_mask)

    # Extract 3x3 neighborhoods
    X_aug = np.zeros((N, F * 9), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        # padded coords are (r+1, c+1) for the center
        patch = padded[r:r+3, c:c+3, :]  # (3, 3, F)
        X_aug[i] = patch.reshape(-1)

    del cube, padded
    gc.collect()
    return X_aug


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgbm(X_train, y_train, X_val, y_val, n_features, feature_names=None):
    """Train LightGBM classifier."""
    import lightgbm as lgb

    # Compute class weights (inverse frequency)
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weight_dict = {c: total / (len(classes) * cnt) for c, cnt in
                   zip(classes, counts)}
    sample_weights = np.array([weight_dict[yi] for yi in y_train],
                              dtype=np.float32)

    print(f"\n  [{ts()}] Training LightGBM...")
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print(f"  Classes: {dict(zip(classes, counts))}")

    params = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 12,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(50),
        ],
    )

    print(f"  Best iteration: {model.best_iteration_}")
    return model


def train_mlp(X_train, y_train, X_val, y_val, n_features, device='cuda'):
    """Train a simple MLP classifier."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    # Class weights
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weights = torch.tensor([total / (len(classes) * counts[i])
                            for i in range(len(classes))],
                           dtype=torch.float32, device=device)

    # Simple MLP
    class PixelMLP(nn.Module):
        def __init__(self, in_features, n_classes, widths=[512, 256, 128]):
            super().__init__()
            layers = []
            prev = in_features
            for w in widths:
                layers.extend([
                    nn.Linear(prev, w),
                    nn.BatchNorm1d(w),
                    nn.SiLU(),
                    nn.Dropout(0.15),
                ])
                prev = w
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(prev, n_classes)

        def forward(self, x):
            return self.head(self.backbone(x))

    net = PixelMLP(n_features, N_CLASSES).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n  [{ts()}] Training MLP ({n_params:,} params) on {device}...")

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # To tensors
    X_t = torch.tensor(X_train_s, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_v = torch.tensor(X_val_s, device=device)
    y_v = torch.tensor(y_val, dtype=torch.long, device=device)

    BATCH = 4096
    best_val_loss = float('inf')
    best_state = None
    patience = 20
    wait = 0

    n_train = X_t.shape[0]
    steps_per_epoch = (n_train + BATCH - 1) // BATCH

    for epoch in range(200):
        net.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / steps_per_epoch

        # Validate
        net.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(0, X_v.shape[0], 8192):
                val_logits.append(net(X_v[i:i+8192]))
            val_logits = torch.cat(val_logits)
            val_loss = criterion(val_logits, y_v).item()
            val_acc = (val_logits.argmax(1) == y_v).float().mean().item()

        if epoch % 10 == 0 or epoch < 5:
            print(f"  Epoch {epoch:3d} | train_loss={avg_train:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    net.load_state_dict(best_state)
    net.eval()

    return {'model': net, 'scaler': scaler}


def evaluate_model(model, X_test, y_test, model_type='lgbm'):
    """Evaluate and print classification report."""
    from sklearn.metrics import (classification_report, accuracy_score,
                                  confusion_matrix)

    if model_type == 'lgbm':
        y_pred = model.predict(X_test)
    else:
        import torch
        net = model['model']
        scaler = model['scaler']
        X_s = scaler.transform(X_test).astype(np.float32)
        device = next(net.parameters()).device
        preds = []
        net.eval()
        with torch.no_grad():
            for i in range(0, len(X_s), 8192):
                xb = torch.tensor(X_s[i:i+8192], device=device)
                preds.append(net(xb).argmax(1).cpu().numpy())
        y_pred = np.concatenate(preds)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Per-class report
    # Only include classes that appear in the test set
    present_classes = sorted(set(y_test) | set(y_pred))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    report = classification_report(
        y_test, y_pred, labels=present_classes,
        target_names=target_names, digits=4, zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    print("  Confusion Matrix:")
    header = "          " + " ".join(f"{n[:6]:>6}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:6d}" for v in row)
        print(f"  {target_names[i]:>8} {row_str}")

    return {
        'accuracy': float(acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-pixel land cover classifier")
    parser.add_argument('--model', choices=['lgbm', 'mlp', 'both'],
                        default='lgbm', help='Model type')
    parser.add_argument('--max-pixels-per-city', type=int, default=50000,
                        help='Max pixels to sample per city (default 50k)')
    parser.add_argument('--cities', nargs='*', default=None,
                        help='Train on specific cities only (for testing)')
    parser.add_argument('--use-neighbors', action='store_true',
                        help='Include 3x3 neighborhood features (9x features)')
    parser.add_argument('--device', default='cuda',
                        help='Device for MLP training')
    args = parser.parse_args()

    np.random.seed(SEED)
    out_dir = os.path.join(CITIES_DIR, "models_pixel_v1")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Per-Pixel Land Cover Classification")
    print(f"  Model: {args.model} | Max pixels/city: {args.max_pixels_per_city:,}")
    print(f"  Neighbors: {args.use_neighbors}")
    print(f"{'='*70}\n")

    # --- Determine city splits ---
    if args.cities:
        # Single/specific city mode — split data within city 80/20
        selected = [c for c in CITIES if c.name in args.cities]
        if not selected:
            print(f"ERROR: No cities matched {args.cities}")
            return
        train_cities = selected
        val_cities = []
        single_city_mode = True
        print(f"  Single-city mode: {[c.name for c in selected]}")
    else:
        train_cities = [c for c in CITIES
                        if c.name not in EXCLUDED_CITY_NAMES
                        and c.name not in VAL_CITY_NAMES]
        val_cities = [c for c in CITIES if c.name in VAL_CITY_NAMES]
        single_city_mode = False
        print(f"  Train cities: {len(train_cities)}")
        print(f"  Val cities:   {len(val_cities)}")

    # --- Load data ---
    print(f"\n[{ts()}] Loading pixel data...")

    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []
    feature_names = None

    def process_city(city, max_px):
        """Load & subsample pixels for one city."""
        X, y, H, W = build_pixel_features(city)
        if X is None or len(X) == 0:
            return None, None
        # Subsample
        if len(X) > max_px:
            idx = np.random.choice(len(X), max_px, replace=False)
            X = X[idx]
            y = y[idx]
            print(f"  [{city.name}] Subsampled to {max_px:,} pixels")
        return X, y

    # Train cities
    for city in train_cities:
        X, y = process_city(city, args.max_pixels_per_city)
        if X is None:
            continue
        if single_city_mode:
            # 80/20 split within city
            n = len(X)
            perm = np.random.permutation(n)
            split = int(0.8 * n)
            all_X_train.append(X[perm[:split]])
            all_y_train.append(y[perm[:split]])
            all_X_val.append(X[perm[split:]])
            all_y_val.append(y[perm[split:]])
        else:
            all_X_train.append(X)
            all_y_train.append(y)
        del X, y
        gc.collect()

    # Val cities
    for city in val_cities:
        X, y = process_city(city, args.max_pixels_per_city)
        if X is None:
            continue
        all_X_val.append(X)
        all_y_val.append(y)
        del X, y
        gc.collect()

    if not all_X_train:
        print("ERROR: No training data loaded!")
        return
    if not all_X_val:
        print("ERROR: No validation data loaded!")
        return

    X_train = np.concatenate(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_val = np.concatenate(all_X_val)
    y_val = np.concatenate(all_y_val)
    del all_X_train, all_y_train, all_X_val, all_y_val
    gc.collect()

    n_features = X_train.shape[1]
    print(f"\n[{ts()}] Data assembled:")
    print(f"  Train: {X_train.shape[0]:,} × {n_features}")
    print(f"  Val:   {X_val.shape[0]:,} × {n_features}")

    # Class distribution
    for split_name, split_y in [("Train", y_train), ("Val", y_val)]:
        classes, counts = np.unique(split_y, return_counts=True)
        total = counts.sum()
        print(f"\n  {split_name} class distribution:")
        for c, cnt in zip(classes, counts):
            print(f"    {CLASS_NAMES[c]:>15}: {cnt:>8,} ({100*cnt/total:5.1f}%)")

    # --- Train models ---
    results = {}

    if args.model in ('lgbm', 'both'):
        print(f"\n{'='*70}")
        print(f"  LightGBM Training")
        print(f"{'='*70}")
        t0 = time.time()
        lgbm_model = train_lgbm(X_train, y_train, X_val, y_val, n_features)
        elapsed = time.time() - t0
        print(f"\n  [{ts()}] LightGBM trained in {elapsed:.1f}s")

        # Evaluate
        print(f"\n  --- LightGBM Validation Results ---")
        metrics = evaluate_model(lgbm_model, X_val, y_val, 'lgbm')
        results['lgbm'] = metrics

        # Feature importance (top 20)
        if hasattr(lgbm_model, 'feature_importances_'):
            imp = lgbm_model.feature_importances_
            top_idx = np.argsort(imp)[::-1][:20]
            print(f"\n  Top 20 Feature Importances:")
            for rank, idx in enumerate(top_idx):
                print(f"    {rank+1:2d}. feature_{idx:<4d}: {imp[idx]:>6.0f}")

        # Save
        model_path = os.path.join(out_dir, "lgbm_pixel_v1.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(lgbm_model, f)
        print(f"\n  Saved: {model_path}")

    if args.model in ('mlp', 'both'):
        print(f"\n{'='*70}")
        print(f"  MLP Training")
        print(f"{'='*70}")
        import torch
        device = args.device if torch.cuda.is_available() else 'cpu'
        t0 = time.time()
        mlp_model = train_mlp(X_train, y_train, X_val, y_val, n_features,
                              device=device)
        elapsed = time.time() - t0
        print(f"\n  [{ts()}] MLP trained in {elapsed:.1f}s")

        print(f"\n  --- MLP Validation Results ---")
        metrics = evaluate_model(mlp_model, X_val, y_val, 'mlp')
        results['mlp'] = metrics

        # Save
        model_path = os.path.join(out_dir, "mlp_pixel_v1.pt")
        torch.save({
            'model_state': mlp_model['model'].state_dict(),
            'scaler_mean': mlp_model['scaler'].mean_,
            'scaler_scale': mlp_model['scaler'].scale_,
            'n_features': n_features,
        }, model_path)
        print(f"\n  Saved: {model_path}")

    # Save metrics
    # Convert numpy types to JSON-serializable
    def jsonify(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_path = os.path.join(out_dir, "metrics_pixel_v1.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=jsonify)
    print(f"\n  Metrics saved: {metrics_path}")

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
