#!/usr/bin/env python3
"""
Production Pipeline: Download -> Extract -> Train -> Predict (2020-2025).

Standalone orchestrator that uses only sibling modules:
  - extract_features.py  (Rust-accelerated feature extraction)
  - train_mlp.py         (MLP champion model)
  - train_tree.py        (LightGBM champion model)

All config values are inlined — no project imports required.
Intermediate results are checkpointed so crashed runs can resume.

Usage:
    python pipeline.py [--skip-download] [--skip-extract] [--skip-train] [--skip-predict]
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

# =====================================================================
# Inlined Config (from config/data_config.yml)
# =====================================================================

AOI_BBOX = [10.95, 49.38, 11.20, 49.52]
AOI_EPSG = 32632

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SENTINEL_RES = 10
SENTINEL_NODATA = -9999
MIN_SCENES = 8

SEASON_DATES = {
    "spring": ("04-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "autumn": ("09-01", "10-31"),
}
CLOUD_COVER_MAX = {"spring": 40, "summer": 20, "autumn": 40}
SCL_EXCLUDE = [0, 1, 2, 3, 8, 9, 10, 11]

GRID_PX = 10  # pixels per cell side

ALL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LABELED_YEARS = [2020, 2021]
PREDICT_YEARS = [2022, 2023, 2024, 2025]
SEASONS = ["spring", "summer", "autumn"]

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)
SEED = 42
N_FOLDS = 5

MIN_VALID_FRAC = 0.3

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}

# Feature group prefixes
BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
}


# =====================================================================
# Path Setup
# =====================================================================

def find_project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "data")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot find project root (no 'data' dir found)")


PROJECT_ROOT = find_project_root()
RAW_V2_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "v2")
PROCESSED_V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
GRID_REF_PATH = os.path.join(PROJECT_ROOT, "data", "grid", "anchor_utm32632_10m.tif")

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "pipeline_output")
FEATURES_DIR = os.path.join(OUT_DIR, "features")
MODELS_DIR = os.path.join(OUT_DIR, "models")
PREDICTIONS_DIR = os.path.join(OUT_DIR, "predictions")


def ts():
    return time.strftime("%H:%M:%S")


def ensure_dirs():
    for d in [OUT_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, RAW_V2_DIR]:
        os.makedirs(d, exist_ok=True)


# =====================================================================
# Spatial Splitting (inlined)
# =====================================================================

@lru_cache(maxsize=8)
def _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles):
    neighbors = {}
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tid = tr * n_tile_cols + tc
            nbrs = set()
            for dr in range(-buffer_tiles, buffer_tiles + 1):
                for dc in range(-buffer_tiles, buffer_tiles + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                        nbrs.add(nr * n_tile_cols + nc)
            neighbors[tid] = frozenset(nbrs)
    return neighbors


def get_fold_indices(groups, fold_assignments, fold_idx,
                     n_tile_cols, n_tile_rows, buffer_tiles=1):
    test_mask = fold_assignments == fold_idx
    if buffer_tiles > 0:
        test_tiles = set(np.unique(groups[test_mask]))
        nbr_map = _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles)
        buf = set()
        for tt in test_tiles:
            for n in nbr_map.get(tt, set()):
                if n not in test_tiles:
                    buf.add(n)
        train_mask = (~test_mask) & (~np.isin(groups, list(buf)))
    else:
        train_mask = ~test_mask
    return np.where(train_mask)[0], np.where(test_mask)[0]


# =====================================================================
# Evaluation (inlined)
# =====================================================================

def evaluate_predictions(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, r2_score
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    mae = float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")) * 100
    return r2, mae


# =====================================================================
# STAGE 1: DOWNLOAD
# =====================================================================

def download_season(year, season):
    """Download one Sentinel-2 v2 composite via Planetary Computer."""
    import planetary_computer
    import pystac_client
    import rasterio
    import stackstac
    import xarray as xr
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    path = os.path.join(RAW_V2_DIR, f"sentinel2_nuremberg_{year}_{season}.tif")
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{year}/{season}] Already exists ({mb:.1f} MB) -- skip")
        return

    with rasterio.open(GRID_REF_PATH) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    s_date = f"{year}-{SEASON_DATES[season][0]}"
    e_date = f"{year}-{SEASON_DATES[season][1]}"

    for cloud_max in [40, 50, 60]:
        items = catalog.search(
            collections=["sentinel-2-l2a"], bbox=AOI_BBOX,
            datetime=f"{s_date}/{e_date}",
            query={"eo:cloud_cover": {"lt": cloud_max}},
        ).item_collection()
        if len(items) >= MIN_SCENES:
            break

    if len(items) < MIN_SCENES:
        from datetime import datetime, timedelta
        s = datetime.strptime(s_date, "%Y-%m-%d")
        e = datetime.strptime(e_date, "%Y-%m-%d")
        s_date = (s - timedelta(days=14)).strftime("%Y-%m-%d")
        e_date = (e + timedelta(days=14)).strftime("%Y-%m-%d")
        items = catalog.search(
            collections=["sentinel-2-l2a"], bbox=AOI_BBOX,
            datetime=f"{s_date}/{e_date}",
            query={"eo:cloud_cover": {"lt": 60}},
        ).item_collection()

    n_scenes = len(items)
    if n_scenes == 0:
        print(f"  [{year}/{season}] WARNING: No scenes found -- skipping!")
        return
    print(f"  [{year}/{season}] {n_scenes} scenes, compositing...")

    warnings.filterwarnings("ignore", module="stackstac")
    spectral = stackstac.stack(
        items, assets=SENTINEL_BANDS, bounds_latlon=AOI_BBOX,
        resolution=SENTINEL_RES, epsg=AOI_EPSG, dtype="float64",
        fill_value=np.nan, resampling=Resampling.bilinear, chunksize=1024,
        rescale=False,
    )
    scl = stackstac.stack(
        items, assets=["SCL"], bounds_latlon=AOI_BBOX,
        resolution=SENTINEL_RES, epsg=AOI_EPSG, dtype="float64",
        fill_value=np.nan, resampling=Resampling.nearest, chunksize=1024,
        rescale=False,
    ).sel(band="SCL")

    spectral, scl = xr.align(spectral, scl, join="exact")
    spectral = spectral.sel(band=SENTINEL_BANDS)

    import dask.array as da
    scl_vals = scl.data
    valid = xr.DataArray(da.isfinite(scl_vals), coords=scl.coords, dims=scl.dims)
    for cls in SCL_EXCLUDE:
        valid = valid & (scl != cls)

    valid_frac_xr = valid.mean(dim="time").astype("float32")
    composite_xr = (spectral.where(valid).median(dim="time", skipna=True)
                    .astype("float32"))

    print(f"  [{year}/{season}] Computing median composite...")
    composite = composite_xr.compute().values
    valid_fraction = valid_frac_xr.compute().values

    xs = np.asarray(composite_xr.coords["x"].values)
    ys = np.asarray(composite_xr.coords["y"].values)
    rx = float(np.abs(xs[1] - xs[0]))
    ry = float(np.abs(ys[1] - ys[0]))
    src_transform = rasterio.transform.from_bounds(
        float(xs.min()) - rx / 2, float(ys.min()) - ry / 2,
        float(xs.max()) + rx / 2, float(ys.max()) + ry / 2,
        len(xs), len(ys))
    src_crs = CRS.from_epsg(AOI_EPSG)

    nodata = SENTINEL_NODATA
    comp_clean = np.where(np.isnan(composite), nodata, composite).astype(np.float32)
    vf_clean = np.where(np.isnan(valid_fraction), nodata, valid_fraction).astype(np.float32)

    n_spectral = len(SENTINEL_BANDS)
    warped = np.full((n_spectral, dst_height, dst_width), nodata, dtype=np.float32)
    for i in range(n_spectral):
        reproject(
            source=comp_clean[i], destination=warped[i],
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata, dst_nodata=nodata,
        )

    vf_warped = np.full((dst_height, dst_width), nodata, dtype=np.float32)
    reproject(
        source=vf_clean, destination=vf_warped,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata, dst_nodata=nodata,
    )
    vf_mask = vf_warped != nodata
    vf_warped[vf_mask] = np.clip(vf_warped[vf_mask], 0.0, 1.0)

    with rasterio.open(
        path, "w", driver="GTiff", height=dst_height, width=dst_width,
        count=n_spectral + 1, dtype="float32", crs=dst_crs,
        transform=dst_transform, compress="lzw", nodata=nodata,
    ) as dst:
        for i in range(n_spectral):
            dst.write(warped[i], i + 1)
            dst.set_band_description(i + 1, SENTINEL_BANDS[i])
        dst.write(vf_warped, n_spectral + 1)
        dst.set_band_description(n_spectral + 1, "VALID_FRACTION")
        dst.update_tags(YEAR=str(year), SEASON=season, N_SCENES_TOTAL=str(n_scenes))

    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [{year}/{season}] Saved ({mb:.1f} MB)")


def stage_download():
    print(f"\n{'='*70}")
    print(f"STAGE 1: DOWNLOAD (Sentinel-2 composites)")
    print(f"{'='*70}")
    for year in ALL_YEARS:
        for season in SEASONS:
            download_season(year, season)
    print(f"\n[{ts()}] Download stage complete.")


# =====================================================================
# STAGE 2: EXTRACT FEATURES (Rust)
# =====================================================================

def load_sentinel_raster(year, season):
    """Load Sentinel-2 raster, return (spectral, valid_fraction)."""
    import rasterio

    path = os.path.join(RAW_V2_DIR, f"sentinel2_nuremberg_{year}_{season}.tif")
    with rasterio.open(path) as ds:
        data = ds.read()
        nodata = ds.nodata

    n_bands = len(SENTINEL_BANDS)
    spectral = data[:n_bands].astype(np.float32)
    if nodata is not None:
        spectral = np.where(spectral == nodata, np.nan, spectral)

    # Valid fraction is the last band (if present)
    vf = None
    if data.shape[0] > n_bands:
        vf = data[n_bands].astype(np.float32)
        if nodata is not None:
            vf = np.where(vf == nodata, np.nan, vf)

    return spectral, vf


def detect_scale(spectral):
    """Auto-detect if reflectance is 0..10000 or 0..1."""
    nir = spectral[6]  # B08
    finite = nir[np.isfinite(nir)]
    if len(finite) == 0:
        return 1.0
    return 10000.0 if np.percentile(finite, 95) > 2.0 else 1.0


def extract_year_pair(prev_year, curr_year):
    """Extract features for a year-pair using Rust.

    Loads 6 season rasters (3 per year), runs terrapulse_features in one
    shot, saves parquet with columns suffixed {model_year}_{season}.

    Returns path to parquet, or None if rasters missing.
    """
    import terrapulse_features as tf

    tag = f"{prev_year}_{curr_year}"
    out_path = os.path.join(FEATURES_DIR, f"features_rust_{tag}.parquet")
    if os.path.exists(out_path):
        print(f"  [{tag}] Already extracted -- skip")
        return out_path

    # Map actual years to model-expected year tags (always 2020/2021)
    year_map = {prev_year: 2020, curr_year: 2021}

    # Check all TIFs exist
    jobs = []
    for actual_year in [prev_year, curr_year]:
        for season in SEASONS:
            tif = os.path.join(RAW_V2_DIR,
                               f"sentinel2_nuremberg_{actual_year}_{season}.tif")
            if not os.path.exists(tif):
                print(f"  [{tag}] WARNING: Missing {tif} -- skip")
                return None
            jobs.append((actual_year, season))

    # Load rasters
    spectral_list = []
    suffixes = []
    nr, nc = None, None
    vf_first = None

    t0 = time.time()
    for actual_year, season in jobs:
        model_year = year_map[actual_year]
        spectral, vf = load_sentinel_raster(actual_year, season)
        scale = detect_scale(spectral)
        if nr is None:
            _, H, W = spectral.shape
            nr, nc = H // GRID_PX, W // GRID_PX
            vf_first = vf
        ref = spectral.astype(np.float32)
        if scale != 1.0:
            ref = ref / scale
        spectral_list.append(np.ascontiguousarray(ref))
        suffixes.append(f"{model_year}_{season}")
        print(f"    Loaded {actual_year}_{season} -> {model_year}_{season}")

    n_cells = nr * nc

    # Rust extraction (batch) — v2 expects pre-normalized data, no scale param
    t1 = time.time()
    n_feat = tf.n_features_per_cell()
    flat = tf.extract_all_seasons_v2(spectral_list, nr, nc)
    dt_rust = time.time() - t1
    print(f"    Rust extraction: {dt_rust:.1f}s for {len(jobs)} seasons")
    del spectral_list

    # Build DataFrame
    result_2d = flat.reshape(n_cells, len(suffixes) * n_feat)
    columns = tf.feature_names_suffixed(suffixes)

    data = {"cell_id": np.arange(n_cells, dtype=np.int32)}
    for i, col in enumerate(columns):
        data[col] = result_2d[:, i]

    # Valid fraction from first raster
    if vf_first is not None:
        vf_cells = np.where(np.isfinite(vf_first), vf_first, 0.0).astype(np.float32)
        vf_cells = (vf_cells
                    .reshape(nr, GRID_PX, nc, GRID_PX)
                    .transpose(0, 2, 1, 3)
                    .reshape(n_cells, GRID_PX * GRID_PX)
                    .mean(axis=1))
        data["valid_fraction"] = vf_cells
        data["low_valid_fraction"] = (vf_cells < MIN_VALID_FRAC).astype(np.float32)

    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute NaN with column medians
    feat_cols = [c for c in df.columns if c not in CONTROL_COLS]
    nan_count = 0
    for c in feat_cols:
        n = df[c].isna().sum()
        if n > 0:
            nan_count += n
            med = df[c].median()
            df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)
    if nan_count > 0:
        print(f"    Imputed {nan_count} NaN values")

    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  [{tag}] Done: {df.shape[1]} cols, {mb:.1f} MB, {elapsed:.0f}s")
    return out_path


def stage_extract():
    print(f"\n{'='*70}")
    print(f"STAGE 2: EXTRACT FEATURES (Rust)")
    print(f"{'='*70}")
    year_pairs = [(y, y + 1) for y in range(2020, 2025)]
    for prev_year, curr_year in year_pairs:
        extract_year_pair(prev_year, curr_year)
    print(f"\n[{ts()}] Extract stage complete.")


# =====================================================================
# STAGE 3: TRAIN
# =====================================================================

def build_bi_lbp(feature_cols):
    """Select bands + indices + TC + all LBP columns."""
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in BAND_PREFIXES or prefix in INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
    return sorted(set(selected))


def build_tree_features(feature_cols):
    """Select VegIdx + RedEdge + TC + NDTI + IRECI + CRI1."""
    selected = []
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    novel = ["NDTI", "IRECI", "CRI1"]
    for c in feature_cols:
        # VegIdx
        if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"]):
            if not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr"):
                selected.append(c)
                continue
        # RedEdge bands
        if band_pat.match(c):
            selected.append(c)
            continue
        # TC
        if c.startswith("TC_"):
            selected.append(c)
            continue
        # Novel indices
        for idx in novel:
            if c.startswith(f"{idx}_"):
                selected.append(c)
                break
    return selected


def swap_lbp_cols_for_mlp(feature_cols):
    """Swap NIR LBP -> NDTI LBP (best band per sweep)."""
    swapped = []
    for c in feature_cols:
        if c.startswith("LBP_u8_") or c.startswith("LBP_entropy_"):
            swapped.append("LBP_NDTI_" + c[len("LBP_"):])
        else:
            swapped.append(c)
    return swapped


# -- MLP model (inlined from train_mlp.py) --

def _build_mlp():
    """Late import to avoid loading torch at module level."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PlainBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout=0.15):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.norm = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.dropout(self.norm(F.silu(self.linear(x))))

    class PlainMLP(nn.Module):
        def __init__(self, in_features, n_classes=N_CLASSES, hidden=1024,
                     n_layers=5, dropout=0.15):
            super().__init__()
            layers = [PlainBlock(in_features, hidden, dropout)]
            for _ in range(n_layers - 1):
                layers.append(PlainBlock(hidden, hidden, dropout))
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x):
            return F.log_softmax(self.head(self.backbone(x)), dim=-1)

        def predict(self, x):
            self.eval()
            with torch.no_grad():
                return self.forward(x).exp()

    return PlainMLP


def normalize_targets(y):
    y = np.clip(y, 0, None).astype(np.float32)
    s = y.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    y = y / s + 1e-7
    y = y / y.sum(axis=1, keepdims=True)
    return y


def train_mlp_fold(X_trn, y_trn, X_val, y_val, n_features, device, fold_id):
    """Train one MLP fold with AMP + fused AdamW (speed-optimized)."""
    import torch

    PlainMLP = _build_mlp()
    use_amp = device == "cuda"

    torch.manual_seed(SEED + fold_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold_id)

    net = PlainMLP(n_features).to(device)

    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=1e-3, weight_decay=1e-4, fused=use_amp)
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=1e-3, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n = X_trn.size(0)
    batch_size = 2048
    steps_per_ep = math.ceil(n / batch_size)
    max_epochs = 2000
    total_steps = max_epochs * steps_per_ep

    # Cosine warmup (3 epochs)
    warmup_steps = steps_per_ep * 3

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.001, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    patience_epochs = max(math.ceil(5000 / steps_per_ep), 5)
    min_epochs = max(math.ceil(2000 / steps_per_ep), 3)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if idx.size(0) < 2:
                continue
            xb, yb = X_trn[idx], y_trn[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logp = net(xb)
                loss = -(yb * logp).sum(dim=-1).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        net.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            val_loss = -(y_val * net(X_val)).sum(dim=-1).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if (epoch + 1) >= min_epochs and wait >= patience_epochs:
                break

    if best_state:
        net.load_state_dict(best_state)
    return net, epoch + 1, best_val


def predict_mlp_batched(net, X_cpu, device, batch_size=65536):
    import torch
    net.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, X_cpu.size(0), batch_size):
            xb = X_cpu[i:i + batch_size].to(device, non_blocking=True)
            parts.append(net.predict(xb).cpu())
    return torch.cat(parts, dim=0).numpy()


def stage_train():
    import torch
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor

    print(f"\n{'='*70}")
    print(f"STAGE 3: TRAIN MODELS")
    print(f"{'='*70}")

    tree_done = os.path.exists(os.path.join(MODELS_DIR, "tree_meta.json"))
    mlp_done = os.path.exists(os.path.join(MODELS_DIR, "mlp_meta.json"))
    if tree_done and mlp_done:
        print("  Both models already trained -- skip")
        return

    # Load features for labeled pair (2020+2021)
    print(f"  [{ts()}] Loading features for 2020+2021...")
    feat_path = os.path.join(FEATURES_DIR, "features_rust_2020_2021.parquet")
    if not os.path.exists(feat_path):
        # Fallback to existing processed data
        feat_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
    merged = pd.read_parquet(feat_path)
    print(f"  Shape: {merged.shape}")

    from pandas.api.types import is_numeric_dtype
    full_cols = [c for c in merged.columns
                 if c not in CONTROL_COLS and is_numeric_dtype(merged[c])]

    # Build feature sets
    mlp_idx = build_bi_lbp(full_cols)
    mlp_cols = [full_cols[i] for i in mlp_idx]
    tree_col_names = build_tree_features(full_cols)
    n_mlp = len(mlp_cols)
    n_tree = len(tree_col_names)
    print(f"  MLP features: {n_mlp}, Tree features: {n_tree}")

    X_mlp = np.nan_to_num(merged[mlp_cols].values.astype(np.float32), 0.0)
    X_tree = np.nan_to_num(merged[tree_col_names].values.astype(np.float32), 0.0)

    # Labels
    labels_path = os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet")
    y = pd.read_parquet(labels_path)[CLASS_NAMES].values.astype(np.float32)

    # Spatial splits
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        split_meta = json.load(f)
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    del split_df

    # -- LightGBM --
    if not tree_done:
        print(f"\n  [{ts()}] Training LightGBM (5 folds)...")
        tree_fold_metrics = []
        for fold_id in range(N_FOLDS):
            train_idx, test_idx = get_fold_indices(
                tiles, folds_arr, fold_id,
                split_meta["tile_cols"], split_meta["tile_rows"], buffer_tiles=1)

            t0 = time.time()
            params = dict(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20, reg_lambda=0.1,
                subsample=0.85, colsample_bytree=0.85, verbosity=-1,
                random_state=SEED + fold_id, n_jobs=-1,
            )
            model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
            model.fit(X_tree[train_idx], y[train_idx])
            y_pred = np.clip(model.predict(X_tree[test_idx]), 0, 100)
            elapsed = time.time() - t0

            r2, mae = evaluate_predictions(y[test_idx], y_pred)
            tree_fold_metrics.append({
                "fold": fold_id, "r2": r2, "mae": mae, "time_s": round(elapsed, 1),
            })
            print(f"    Fold {fold_id}: R2={r2:.4f}  MAE={mae:.2f}pp  ({elapsed:.0f}s)")

            with open(os.path.join(MODELS_DIR, f"tree_fold_{fold_id}.pkl"), "wb") as f:
                pickle.dump(model, f)

        tree_r2 = np.mean([m["r2"] for m in tree_fold_metrics])
        with open(os.path.join(MODELS_DIR, "tree_meta.json"), "w") as f:
            json.dump({
                "model": "LightGBM", "feature_cols": tree_col_names,
                "n_features": n_tree, "r2_mean": float(tree_r2),
                "fold_metrics": tree_fold_metrics,
            }, f, indent=2)
        print(f"  Tree mean R2: {tree_r2:.4f}")

    # -- MLP --
    if not mlp_done:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        print(f"\n  [{ts()}] Training MLP on {device} (5 folds)...")
        mlp_fold_metrics = []

        for fold_id in range(N_FOLDS):
            print(f"\n  --- MLP Fold {fold_id} ---")
            train_idx, test_idx = get_fold_indices(
                tiles, folds_arr, fold_id,
                split_meta["tile_cols"], split_meta["tile_rows"], buffer_tiles=1)

            rng = np.random.RandomState(SEED + fold_id)
            perm = rng.permutation(len(train_idx))
            n_val = max(int(len(train_idx) * 0.15), 100)
            val_idx = train_idx[perm[:n_val]]
            trn_idx = train_idx[perm[n_val:]]

            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X_mlp[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X_mlp[val_idx]).astype(np.float32)

            X_trn_t = torch.tensor(X_trn_s).to(device, non_blocking=True)
            X_val_t = torch.tensor(X_val_s).to(device, non_blocking=True)
            y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device, non_blocking=True)
            y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device, non_blocking=True)

            t0 = time.time()
            trained_net, n_epochs, best_val = train_mlp_fold(
                X_trn_t, y_trn_t, X_val_t, y_val_t, n_mlp, device, fold_id)
            elapsed = time.time() - t0

            # Save
            save_net = trained_net._orig_mod if hasattr(trained_net, "_orig_mod") else trained_net
            torch.save(save_net.state_dict(),
                       os.path.join(MODELS_DIR, f"mlp_fold_{fold_id}.pt"))
            with open(os.path.join(MODELS_DIR, f"mlp_scaler_{fold_id}.pkl"), "wb") as f:
                pickle.dump(scaler, f)

            # OOF eval
            X_tst_s = scaler.transform(X_mlp[test_idx]).astype(np.float32)
            preds = predict_mlp_batched(
                trained_net, torch.tensor(X_tst_s), device)

            r2, mae = evaluate_predictions(y[test_idx], preds)
            mlp_fold_metrics.append({
                "fold": fold_id, "r2": r2, "mae": mae,
                "epochs": n_epochs, "time_s": round(elapsed, 1),
            })
            print(f"    R2={r2:.4f}  MAE={mae:.2f}pp  epochs={n_epochs}  ({elapsed:.0f}s)")

            del trained_net, X_trn_t, X_val_t, y_trn_t, y_val_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mlp_r2 = np.mean([m["r2"] for m in mlp_fold_metrics])
        with open(os.path.join(MODELS_DIR, "mlp_meta.json"), "w") as f:
            json.dump({
                "model": "MLP", "feature_cols": mlp_cols,
                "n_features": n_mlp, "r2_mean": float(mlp_r2),
                "fold_metrics": mlp_fold_metrics,
            }, f, indent=2)
        print(f"  MLP mean R2: {mlp_r2:.4f}")

    print(f"\n[{ts()}] Train stage complete.")


# =====================================================================
# STAGE 4: PREDICT
# =====================================================================

def stage_predict():
    print(f"\n{'='*70}")
    print(f"STAGE 4: PREDICT MAPS")
    print(f"{'='*70}")

    with open(os.path.join(MODELS_DIR, "tree_meta.json")) as f:
        tree_cols = json.load(f)["feature_cols"]
    with open(os.path.join(MODELS_DIR, "mlp_meta.json")) as f:
        mlp_meta = json.load(f)
    mlp_cols = mlp_meta["feature_cols"]
    n_mlp = mlp_meta["n_features"]

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    print(f"  Device: {device}")

    PlainMLP = _build_mlp()

    year_pairs = [(y, y + 1) for y in range(2020, 2025)]

    for prev_year, curr_year in year_pairs:
        tag = f"{prev_year}_{curr_year}"
        tree_path = os.path.join(PREDICTIONS_DIR, f"predictions_tree_{tag}.parquet")
        mlp_path = os.path.join(PREDICTIONS_DIR, f"predictions_mlp_{tag}.parquet")

        if os.path.exists(tree_path) and os.path.exists(mlp_path):
            print(f"  [{tag}] Already predicted -- skip")
            continue

        print(f"\n  [{ts()}] Loading features for {tag}...")
        feat_path = os.path.join(FEATURES_DIR, f"features_rust_{tag}.parquet")
        if not os.path.exists(feat_path):
            if prev_year == 2020 and curr_year == 2021:
                feat_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
            if not os.path.exists(feat_path):
                print(f"  [{tag}] WARNING: No features found -- skip")
                continue
        merged = pd.read_parquet(feat_path)
        cell_ids = merged["cell_id"].values
        merged_cols_set = set(merged.columns)

        # Tree predictions (ensemble of 5 folds)
        if not os.path.exists(tree_path):
            print(f"  [{ts()}] Tree predictions for {tag}...")
            avail_tree = [c for c in tree_cols if c in merged_cols_set]
            X_tree = np.nan_to_num(merged[avail_tree].values.astype(np.float32), 0.0)
            preds_all = np.zeros((len(cell_ids), N_CLASSES), dtype=np.float32)
            for fold_id in range(N_FOLDS):
                with open(os.path.join(MODELS_DIR, f"tree_fold_{fold_id}.pkl"), "rb") as f:
                    model = pickle.load(f)
                preds_all += np.clip(model.predict(X_tree), 0, 100).astype(np.float32)
            preds_all /= N_FOLDS

            tree_df = pd.DataFrame({"cell_id": cell_ids})
            for ci, cn in enumerate(CLASS_NAMES):
                tree_df[f"{cn}_pred"] = preds_all[:, ci]
            tree_df["prev_year"] = prev_year
            tree_df["curr_year"] = curr_year
            tree_df.to_parquet(tree_path, index=False)
            print(f"    Saved: {tree_path}")

        # MLP predictions (ensemble of 5 folds)
        if not os.path.exists(mlp_path):
            import torch
            print(f"  [{ts()}] MLP predictions for {tag}...")
            avail_mlp = [c for c in mlp_cols if c in merged_cols_set]
            X_mlp = np.nan_to_num(merged[avail_mlp].values.astype(np.float32), 0.0)
            preds_all = np.zeros((len(cell_ids), N_CLASSES), dtype=np.float32)

            for fold_id in range(N_FOLDS):
                with open(os.path.join(MODELS_DIR, f"mlp_scaler_{fold_id}.pkl"), "rb") as f:
                    scaler = pickle.load(f)
                X_scaled = scaler.transform(X_mlp).astype(np.float32)

                net = PlainMLP(n_mlp).to(device)
                net.load_state_dict(torch.load(
                    os.path.join(MODELS_DIR, f"mlp_fold_{fold_id}.pt"),
                    map_location=device, weights_only=True))

                preds_all += predict_mlp_batched(net, torch.tensor(X_scaled), device)
                del net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            preds_all /= N_FOLDS

            mlp_df = pd.DataFrame({"cell_id": cell_ids})
            for ci, cn in enumerate(CLASS_NAMES):
                mlp_df[f"{cn}_pred"] = preds_all[:, ci]
            mlp_df["prev_year"] = prev_year
            mlp_df["curr_year"] = curr_year
            mlp_df.to_parquet(mlp_path, index=False)
            print(f"    Saved: {mlp_path}")

        del merged

    print(f"\n[{ts()}] Predict stage complete.")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production pipeline: download -> extract -> train -> predict")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    t_total = time.time()
    print(f"[{ts()}] Production Pipeline starting")
    print(f"  Years: {ALL_YEARS}")
    print(f"  Seasons: {SEASONS}")
    print(f"  Labels available: {LABELED_YEARS}")
    print(f"  Predict: {PREDICT_YEARS}")

    ensure_dirs()

    if not args.skip_download:
        stage_download()
    else:
        print("\n  DOWNLOAD skipped (--skip-download)")

    if not args.skip_extract:
        stage_extract()
    else:
        print("\n  EXTRACT skipped (--skip-extract)")

    if not args.skip_train:
        stage_train()
    else:
        print("\n  TRAIN skipped (--skip-train)")

    if not args.skip_predict:
        stage_predict()
    else:
        print("\n  PREDICT skipped (--skip-predict)")

    total = time.time() - t_total
    hours = int(total // 3600)
    mins = int((total % 3600) // 60)
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE in {hours}h {mins}m")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
