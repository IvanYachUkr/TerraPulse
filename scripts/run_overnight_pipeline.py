#!/usr/bin/env python3
"""
Overnight Pipeline: Download -> Extract -> Train -> Predict for 2020-2025.

Stages:
  1. DOWNLOAD   - Sentinel-2 v2 composites for all missing year-seasons
  2. EXTRACT    - Feature extraction using original Python pipeline
  3. TRAIN      - Train best MLP + best LightGBM on 2020+2021 labeled data
  4. PREDICT    - Generate land-cover maps for all unlabeled year-seasons

All intermediate results are checkpointed so crashed runs can resume.

Usage:
    python scripts/run_overnight_pipeline.py [--skip-download] [--skip-extract]

Runtime: ~6-8 hours (depends on internet + CPU)
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root / config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import CFG, PROCESSED_V2_DIR, RAW_V2_DIR

GRID_PATH = os.path.join(PROCESSED_V2_DIR, "grid.gpkg")
GRID_REF_PATH = os.path.join(PROJECT_ROOT, CFG["grid"]["reference_file"])

# Output directory for this pipeline
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "pipeline_output")
FEATURES_DIR = os.path.join(OUT_DIR, "features")
MODELS_DIR = os.path.join(OUT_DIR, "models")
PREDICTIONS_DIR = os.path.join(OUT_DIR, "predictions")

# Config
ALL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LABELED_YEARS = [2020, 2021]
PREDICT_YEARS = [2022, 2023, 2024, 2025]
SEASONS = ["spring", "summer", "autumn"]
SEASON_DATES = CFG["sentinel2"]["seasons"]
SENTINEL_BANDS = CFG["sentinel2"]["bands"]

# Model configs
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ts():
    """Timestamp string for logging."""
    return time.strftime("%H:%M:%S")


def ensure_dirs():
    """Create all output directories."""
    for d in [OUT_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR,
              RAW_V2_DIR]:
        os.makedirs(d, exist_ok=True)


# ===========================================================================
# STAGE 1: DOWNLOAD
# ===========================================================================

def download_season(year, season):
    """Download one Sentinel-2 v2 composite via Planetary Computer.

    Reuses the same logic as download_all_data.py: STAC search -> stackstac ->
    SCL masking -> median composite -> warp to anchor grid -> write GeoTIFF.
    """
    import planetary_computer
    import pystac_client
    import rasterio
    import stackstac
    import xarray as xr
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    path = os.path.join(RAW_V2_DIR,
                        f"sentinel2_nuremberg_{year}_{season}.tif")
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{year}/{season}] Already exists ({mb:.1f} MB) -- skip")
        return

    bbox = CFG["aoi"]["bbox"]
    target_epsg = CFG["aoi"]["epsg"]
    expected_res = float(CFG["sentinel2"]["resolution"])
    nodata = CFG["sentinel2"]["nodata"]
    scl_exclude = CFG["scl_mask"]["exclude_classes"]
    min_scenes = CFG["sentinel2"]["min_scenes"]

    # Read anchor grid spec
    with rasterio.open(GRID_REF_PATH) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    start_date = f"{year}-{SEASON_DATES[season][0]}"
    end_date = f"{year}-{SEASON_DATES[season][1]}"

    # Try progressively relaxed thresholds
    for cloud_max in [40, 50, 60]:
        items = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": cloud_max}},
        ).item_collection()
        if len(items) >= min_scenes:
            break

    # Widen date window if still too few
    if len(items) < min_scenes:
        from datetime import datetime, timedelta
        orig_s = datetime.strptime(start_date, "%Y-%m-%d")
        orig_e = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (orig_s - timedelta(days=14)).strftime("%Y-%m-%d")
        end_date = (orig_e + timedelta(days=14)).strftime("%Y-%m-%d")
        items = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": 60}},
        ).item_collection()

    n_scenes = len(items)
    if n_scenes == 0:
        print(f"  [{year}/{season}] WARNING: No scenes found -- skipping!")
        return
    print(f"  [{year}/{season}] {n_scenes} scenes, compositing...")

    # Stack spectral + SCL
    warnings.filterwarnings("ignore", module="stackstac")
    spectral = stackstac.stack(
        items, assets=SENTINEL_BANDS, bounds_latlon=bbox,
        resolution=expected_res, epsg=target_epsg, dtype="float64",
        fill_value=np.nan, resampling=Resampling.bilinear, chunksize=1024,
        rescale=False,
    )
    scl = stackstac.stack(
        items, assets=["SCL"], bounds_latlon=bbox,
        resolution=expected_res, epsg=target_epsg, dtype="float64",
        fill_value=np.nan, resampling=Resampling.nearest, chunksize=1024,
        rescale=False,
    ).sel(band="SCL")

    spectral, scl = xr.align(spectral, scl, join="exact")
    spectral = spectral.sel(band=SENTINEL_BANDS)

    # SCL mask
    import dask.array as da
    scl_vals = scl.data
    valid = xr.DataArray(da.isfinite(scl_vals),
                         coords=scl.coords, dims=scl.dims)
    for cls in scl_exclude:
        valid = valid & (scl != cls)

    valid_fraction_xr = valid.mean(dim="time").astype("float32")
    composite_xr = (spectral.where(valid)
                    .median(dim="time", skipna=True)
                    .astype("float32"))

    print(f"  [{year}/{season}] Computing median composite...")
    composite = composite_xr.compute().values
    valid_fraction = valid_fraction_xr.compute().values

    # Source transform (inlined from download_all_data.py)
    xs = np.asarray(composite_xr.coords["x"].values)
    ys = np.asarray(composite_xr.coords["y"].values)
    rx = float(np.abs(xs[1] - xs[0]))
    ry = float(np.abs(ys[1] - ys[0]))
    left = float(xs.min()) - rx / 2
    right = float(xs.max()) + rx / 2
    bottom = float(ys.min()) - ry / 2
    top = float(ys.max()) + ry / 2
    src_transform = rasterio.transform.from_bounds(
        left, bottom, right, top, len(xs), len(ys))
    src_crs = CRS.from_epsg(target_epsg)

    # Warp to anchor
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

    # Write
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
        dst.update_tags(YEAR=str(year), SEASON=season,
                        N_SCENES_TOTAL=str(n_scenes))

    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [{year}/{season}] Saved ({mb:.1f} MB)")


def stage_download():
    """Download all missing year-season composites."""
    print(f"\n{'='*70}")
    print(f"STAGE 1: DOWNLOAD (Sentinel-2 v2 composites)")
    print(f"{'='*70}")

    for year in ALL_YEARS:
        for season in SEASONS:
            download_season(year, season)

    print(f"\n[{ts()}] Download stage complete.")


# ===========================================================================
# STAGE 2: EXTRACT FEATURES (Rust)
# ===========================================================================

def extract_year_pair_rust(prev_year, curr_year):
    """Run the Rust feature extractor for a consecutive year-pair.

    Loads 6 season rasters (3 per year), runs terrapulse_features in one
    shot, and saves the result as a single wide parquet with columns
    already suffixed with the model's expected year tags (2020_*/2021_*).

    Returns:
        Path to the saved parquet, or None if any rasters are missing.
    """
    import terrapulse_features as tf
    from src.features.extract_features import (
        load_sentinel_v2, detect_reflectance_scale, compute_grid_shape,
        GRID_PX,
    )

    tag = f"{prev_year}_{curr_year}"
    out_path = os.path.join(FEATURES_DIR, f"features_rust_{tag}.parquet")
    if os.path.exists(out_path):
        print(f"  [{tag}] Already extracted -- skip")
        return out_path

    # Map actual years to model-expected year tags
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

    # Phase 1: Load rasters
    spectral_arrays = []
    suffixes = []
    nr, nc = None, None
    vf_first = None

    t0 = time.time()
    for actual_year, season in jobs:
        model_year = year_map[actual_year]
        spectral, vf = load_sentinel_v2(actual_year, season)
        scale = detect_reflectance_scale(spectral)
        if nr is None:
            nr, nc = compute_grid_shape(spectral)
            vf_first = vf
        ref = spectral.astype(np.float32)
        if scale != 1.0:
            ref = ref / scale
        spectral_arrays.append(np.ascontiguousarray(ref))
        suffixes.append(f"{model_year}_{season}")
        print(f"    Loaded {actual_year}_{season} -> {model_year}_{season}")

    n_cells = nr * nc

    # Phase 2: Rust extraction
    t1 = time.time()
    n_feat = tf.n_features_per_cell()
    flat = tf.extract_all_seasons_v2(spectral_arrays, nr, nc)
    dt_rust = time.time() - t1
    print(f"    Rust extraction: {dt_rust:.1f}s for {len(jobs)} seasons")

    del spectral_arrays

    # Phase 3: Build DataFrame
    result_2d = flat.reshape(n_cells, len(suffixes) * n_feat)
    columns = tf.feature_names_suffixed(suffixes)

    data = {"cell_id": np.arange(n_cells, dtype=np.int32)}
    for i, col in enumerate(columns):
        data[col] = result_2d[:, i]

    # Valid fraction from first raster
    MIN_VALID_FRAC = CFG["quality"]["min_valid_fraction"]
    vf_cells = np.where(np.isfinite(vf_first), vf_first, 0.0).astype(np.float32)
    from src.features.extract_features import GRID_PX
    vf_cells = (vf_cells
                .reshape(nr, GRID_PX, nc, GRID_PX)
                .transpose(0, 2, 1, 3)
                .reshape(n_cells, GRID_PX * GRID_PX)
                .mean(axis=1))
    data["valid_fraction"] = vf_cells
    data["low_valid_fraction"] = (vf_cells < MIN_VALID_FRAC).astype(np.float32)

    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute NaNs with column medians
    feat_cols = [c for c in df.columns
                 if c not in ("cell_id", "valid_fraction", "low_valid_fraction")]
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
    """Extract features for all year-pairs using the Rust pipeline."""
    print(f"\n{'='*70}")
    print(f"STAGE 2: EXTRACT FEATURES (Rust pipeline)")
    print(f"{'='*70}")

    # Year pairs: (2020,2021), (2021,2022), ..., (2024,2025)
    year_pairs = [(y, y + 1) for y in range(2020, 2025)]
    for prev_year, curr_year in year_pairs:
        extract_year_pair_rust(prev_year, curr_year)

    print(f"\n[{ts()}] Extract stage complete.")


# ===========================================================================
# STAGE 3: BUILD FEATURE MATRICES & TRAIN
# ===========================================================================

# Control columns to exclude when building feature matrices
CONTROL_COLS = {"cell_id", "valid_fraction", "low_valid_fraction",
                "reflectance_scale", "full_features_computed"}


def load_rust_features(prev_year, curr_year):
    """Load the Rust-extracted feature parquet for a year-pair.

    Falls back to the existing v3 parquet (from PROCESSED_V2_DIR) for
    the labeled year-pair 2020+2021 if no pipeline parquet exists.
    """
    tag = f"{prev_year}_{curr_year}"
    path = os.path.join(FEATURES_DIR, f"features_rust_{tag}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    # Fallback: existing v3 Rust parquet (only for 2020+2021)
    if prev_year == 2020 and curr_year == 2021:
        v3_path = os.path.join(PROCESSED_V2_DIR, "features_v3.parquet")
        if os.path.exists(v3_path):
            return pd.read_parquet(v3_path)
    raise FileNotFoundError(
        f"No Rust features found for {tag}. Checked: {path}")


# Feature group prefixes for dynamic feature selection
_BAND_PREFIXES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
_INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
}


def build_bi_lbp(feature_cols):
    """Select bands + indices + TC + all LBP columns from available features.

    Dynamically discovers features from the Rust parquet columns instead of
    loading a hardcoded column list from old model metadata.
    """
    selected = []
    for i, col in enumerate(feature_cols):
        if col.startswith("delta"):
            continue
        prefix = col.split("_")[0]
        if prefix in _BAND_PREFIXES or prefix in _INDEX_PREFIXES:
            selected.append(i)
        elif prefix == "LBP":
            selected.append(i)
    return sorted(set(selected))


def build_tree_features(feature_cols):
    """Select VegIdx + RedEdge + TC + NDTI + IRECI + CRI1 from available features."""
    band_pat = re.compile(r'^B(05|06|07|8A)_')
    novel = ["NDTI", "IRECI", "CRI1"]
    selected = []
    for c in feature_cols:
        if any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"]):
            if not c.startswith("NDVI_range") and not c.startswith("NDVI_iqr"):
                selected.append(c)
                continue
        if band_pat.match(c):
            selected.append(c)
            continue
        if c.startswith("TC_"):
            selected.append(c)
            continue
        for idx in novel:
            if c.startswith(f"{idx}_"):
                selected.append(c)
                break
    return selected


def load_labels(year):
    """Load labels for a year, checking multiple directories."""
    for base in [OUT_DIR, PROCESSED_V2_DIR,
                 os.path.join(PROJECT_ROOT, "data", "processed")]:
        path = os.path.join(base, f"labels_{year}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No labels found for {year}")


def train_tree_model(X_train, y_train, fold_id):
    """Train a LightGBM MultiOutputRegressor with the best config."""
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor

    params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, reg_lambda=0.1,
        subsample=0.85, colsample_bytree=0.85, verbosity=-1,
        random_state=SEED + fold_id, n_jobs=-1,
    )
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    model.fit(X_train, y_train)
    return model


def train_mlp_model(X_train, y_train, X_val, y_val, fold_id, n_features,
                    device):
    """Train the best MLP config (from V10)."""
    import torch

    # Import training utilities from existing codebase
    from scripts.run_mlp_overnight_v4 import (
        build_model, _cfg, train_model, normalize_targets,
    )

    cfg = _cfg(0, "bi_LBP", "plain", "silu", 5, 1024, "batchnorm")

    torch.manual_seed(SEED + fold_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold_id)

    net = build_model(cfg, n_features, device)

    X_trn_t = torch.tensor(X_train).to(device)
    y_trn_t = torch.tensor(normalize_targets(y_train)).to(device)
    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(normalize_targets(y_val)).to(device)

    n_epochs, best_val, trained_net = train_model(
        net, X_trn_t, y_trn_t, X_val_t, y_val_t,
        lr=cfg["lr"], weight_decay=1e-4,
        batch_size=2048, max_epochs=2000, patience_steps=5000,
        min_steps=2000, mixup_alpha=0, use_swa=False, use_cosine=True,
    )

    del X_trn_t, y_trn_t, X_val_t, y_val_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trained_net, n_epochs, best_val


def stage_train():
    """Train MLP and LightGBM on labeled years (2020+2021)."""
    print(f"\n{'='*70}")
    print(f"STAGE 3: TRAIN MODELS")
    print(f"{'='*70}")

    from sklearn.preprocessing import StandardScaler
    from src.splitting import get_fold_indices
    from src.models.evaluation import evaluate_model

    # Check if already trained
    tree_done = os.path.exists(
        os.path.join(MODELS_DIR, "tree_meta.json"))
    mlp_done = os.path.exists(
        os.path.join(MODELS_DIR, "mlp_meta.json"))
    if tree_done and mlp_done:
        print("  Both models already trained -- skip")
        return

    # Load Rust-extracted features for 2020+2021
    print(f"  [{ts()}] Loading Rust features for 2020+2021...")
    merged = load_rust_features(2020, 2021)
    print(f"  Merged shape: {merged.shape}")

    # Dynamic feature selection from ACTUAL Rust parquet columns
    # (not from old Python-extraction model metadata)
    from pandas.api.types import is_numeric_dtype
    full_feature_cols = [
        c for c in merged.columns
        if c not in CONTROL_COLS and is_numeric_dtype(merged[c])
    ]

    # MLP: bands + indices + TC + all LBP (including multi-band LBP)
    mlp_idx = build_bi_lbp(full_feature_cols)
    mlp_cols = [full_feature_cols[i] for i in mlp_idx]

    # Tree: VegIdx + RedEdge + TC + NDTI + IRECI + CRI1
    tree_cols = build_tree_features(full_feature_cols)

    n_mlp = len(mlp_cols)
    n_tree = len(tree_cols)
    print(f"  MLP features: {n_mlp} (bi_LBP, dynamic from Rust parquet)")
    print(f"  Tree features: {n_tree} (VegIdx+RedEdge+TC+novel, dynamic)")

    X_mlp = np.nan_to_num(
        merged[mlp_cols].values.astype(np.float32), 0.0)
    X_tree = np.nan_to_num(
        merged[tree_cols].values.astype(np.float32), 0.0)

    # Labels: we use 2021 labels (the "current year" in our setup)
    labels_df = load_labels(2021)
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    # Spatial split info
    split_df = pd.read_parquet(
        os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR,
              "split_spatial_meta.json")) as f:
        split_meta = json.load(f)
    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values

    print(f"  Labels: {y.shape}")

    # -- Train LightGBM --
    if not tree_done:
        print(f"\n  [{ts()}] Training LightGBM (5 folds)...")
        tree_models = {}
        tree_fold_metrics = []

        for fold_id in range(N_FOLDS):
            train_idx, test_idx = get_fold_indices(
                tiles, folds_arr, fold_id,
                split_meta["tile_cols"], split_meta["tile_rows"],
                buffer_tiles=1)

            t0 = time.time()
            model = train_tree_model(
                X_tree[train_idx], y[train_idx], fold_id)
            y_pred = np.clip(model.predict(X_tree[test_idx]), 0, 100)

            summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
            elapsed = time.time() - t0
            tree_fold_metrics.append({
                "fold": fold_id,
                "r2": summary["r2_uniform"],
                "mae": summary["mae_mean_pp"],
                "time_s": round(elapsed, 1),
            })
            tree_models[fold_id] = model
            print(f"    Fold {fold_id}: R2={summary['r2_uniform']:.4f}  "
                  f"MAE={summary['mae_mean_pp']:.2f}pp  ({elapsed:.0f}s)")

        # Save tree models
        for fold_id, model in tree_models.items():
            with open(os.path.join(MODELS_DIR,
                      f"tree_fold_{fold_id}.pkl"), "wb") as f:
                pickle.dump(model, f)

        tree_r2 = np.mean([m["r2"] for m in tree_fold_metrics])
        tree_meta_out = {
            "model": "LightGBM", "feature_cols": tree_cols,
            "n_features": n_tree, "r2_mean": float(tree_r2),
            "fold_metrics": tree_fold_metrics,
        }
        with open(os.path.join(MODELS_DIR, "tree_meta.json"), "w") as f:
            json.dump(tree_meta_out, f, indent=2)
        print(f"  Tree mean R2: {tree_r2:.4f}")

    # -- Train MLP --
    if not mlp_done:
        import torch
        from scripts.run_mlp_overnight_v4 import (
            _predict_batched, normalize_targets,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  [{ts()}] Training MLP on {device} (5 folds)...")

        mlp_fold_metrics = []

        for fold_id in range(N_FOLDS):
            print(f"\n  --- MLP Fold {fold_id} ---")
            train_idx, test_idx = get_fold_indices(
                tiles, folds_arr, fold_id,
                split_meta["tile_cols"], split_meta["tile_rows"],
                buffer_tiles=1)

            rng = np.random.RandomState(SEED + fold_id)
            perm = rng.permutation(len(train_idx))
            n_val = max(int(len(train_idx) * 0.15), 100)
            val_idx = train_idx[perm[:n_val]]
            trn_idx = train_idx[perm[n_val:]]

            scaler = StandardScaler()
            X_trn = scaler.fit_transform(X_mlp[trn_idx]).astype(np.float32)
            X_val = scaler.transform(X_mlp[val_idx]).astype(np.float32)
            X_tst = scaler.transform(X_mlp[test_idx]).astype(np.float32)

            t0 = time.time()
            trained_net, n_epochs, best_val = train_mlp_model(
                X_trn, y[trn_idx], X_val, y[val_idx],
                fold_id, n_mlp, device)
            elapsed = time.time() - t0

            # Save model + scaler
            torch.save(trained_net.state_dict(),
                       os.path.join(MODELS_DIR,
                                    f"mlp_fold_{fold_id}.pt"))
            with open(os.path.join(MODELS_DIR,
                      f"mlp_scaler_{fold_id}.pkl"), "wb") as f:
                pickle.dump(scaler, f)

            # Evaluate
            preds = _predict_batched(
                trained_net, torch.tensor(X_tst), device)

            summary, _ = evaluate_model(y[test_idx], preds, CLASS_NAMES)
            mlp_fold_metrics.append({
                "fold": fold_id,
                "r2": summary["r2_uniform"],
                "mae": summary["mae_mean_pp"],
                "epochs": n_epochs,
                "time_s": round(elapsed, 1),
            })
            print(f"    R2={summary['r2_uniform']:.4f}  "
                  f"MAE={summary['mae_mean_pp']:.2f}pp  "
                  f"epochs={n_epochs}  ({elapsed:.0f}s)")

            del trained_net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mlp_r2 = np.mean([m["r2"] for m in mlp_fold_metrics])
        mlp_meta_out = {
            "model": "MLP", "lbp_band": "all_rust",
            "feature_cols": mlp_cols,
            "n_features": n_mlp, "r2_mean": float(mlp_r2),
            "fold_metrics": mlp_fold_metrics,
        }
        with open(os.path.join(MODELS_DIR, "mlp_meta.json"), "w") as f:
            json.dump(mlp_meta_out, f, indent=2)
        print(f"  MLP mean R2: {mlp_r2:.4f}")

    print(f"\n[{ts()}] Train stage complete.")


# ===========================================================================
# STAGE 4: PREDICT
# ===========================================================================

def predict_with_tree(X, fold_id):
    """Load tree model for fold and predict."""
    path = os.path.join(MODELS_DIR, f"tree_fold_{fold_id}.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return np.clip(model.predict(X), 0, 100).astype(np.float32)


def predict_with_mlp(X, fold_id, n_features, device):
    """Load MLP model for fold and predict."""
    import torch
    from scripts.run_mlp_overnight_v4 import build_model, _cfg, _predict_batched

    cfg = _cfg(0, "bi_LBP", "plain", "silu", 5, 1024, "batchnorm")
    net = build_model(cfg, n_features, device)
    net.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, f"mlp_fold_{fold_id}.pt"),
        map_location=device, weights_only=True))
    net.eval()

    with open(os.path.join(MODELS_DIR,
              f"mlp_scaler_{fold_id}.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X).astype(np.float32)
    preds = _predict_batched(net, torch.tensor(X_scaled), device)
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def stage_predict():
    """Predict land-cover maps for all year-pairs."""
    print(f"\n{'='*70}")
    print(f"STAGE 4: PREDICT MAPS")
    print(f"{'='*70}")

    # Load feature column lists from our trained models
    with open(os.path.join(MODELS_DIR, "tree_meta.json")) as f:
        tree_cols = json.load(f)["feature_cols"]
    with open(os.path.join(MODELS_DIR, "mlp_meta.json")) as f:
        mlp_meta = json.load(f)
    mlp_cols = mlp_meta["feature_cols"]  # Already NDTI LBP
    n_mlp = mlp_meta["n_features"]

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    print(f"  Device: {device}")

    # For each consecutive year-pair, load Rust features and predict
    year_pairs = [(y, y + 1) for y in range(2020, 2025)]

    for prev_year, curr_year in year_pairs:
        tag = f"{prev_year}_{curr_year}"
        tree_path = os.path.join(PREDICTIONS_DIR,
                                 f"predictions_tree_{tag}.parquet")
        mlp_path = os.path.join(PREDICTIONS_DIR,
                                f"predictions_mlp_{tag}.parquet")

        if os.path.exists(tree_path) and os.path.exists(mlp_path):
            print(f"  [{tag}] Already predicted -- skip")
            continue

        print(f"\n  [{ts()}] Loading Rust features for {prev_year}+{curr_year}...")
        try:
            merged = load_rust_features(prev_year, curr_year)
        except FileNotFoundError as e:
            print(f"  [{tag}] WARNING: {e} -- skipping")
            continue

        cell_ids = merged["cell_id"].values
        merged_cols_set = set(merged.columns)

        # Tree predictions (ensemble of 5 folds)
        if not os.path.exists(tree_path):
            print(f"  [{ts()}] Tree predictions for {tag}...")
            X_tree = np.nan_to_num(
                merged[[c for c in tree_cols if c in merged_cols_set]]
                .values.astype(np.float32), 0.0)

            preds_all = np.zeros((len(cell_ids), len(CLASS_NAMES)),
                                 dtype=np.float32)
            for fold_id in range(N_FOLDS):
                preds_all += predict_with_tree(X_tree, fold_id)
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
            print(f"  [{ts()}] MLP predictions for {tag}...")
            X_mlp = np.nan_to_num(
                merged[[c for c in mlp_cols if c in merged_cols_set]]
                .values.astype(np.float32), 0.0)

            preds_all = np.zeros((len(cell_ids), len(CLASS_NAMES)),
                                 dtype=np.float32)
            for fold_id in range(N_FOLDS):
                preds_all += predict_with_mlp(
                    X_mlp, fold_id, n_mlp, device)
            preds_all /= N_FOLDS

            mlp_df = pd.DataFrame({"cell_id": cell_ids})
            for ci, cn in enumerate(CLASS_NAMES):
                mlp_df[f"{cn}_pred"] = preds_all[:, ci]
            mlp_df["prev_year"] = prev_year
            mlp_df["curr_year"] = curr_year
            mlp_df.to_parquet(mlp_path, index=False)
            print(f"    Saved: {mlp_path}")


    print(f"\n[{ts()}] Predict stage complete.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overnight pipeline: download -> extract -> train -> predict")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download stage")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction stage")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training stage")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip prediction stage")
    args = parser.parse_args()

    t_total = time.time()
    print(f"[{ts()}] Overnight Pipeline starting")
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
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Features:    {FEATURES_DIR}")
    print(f"  Models:      {MODELS_DIR}")
    print(f"  Predictions: {PREDICTIONS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
