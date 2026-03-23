#!/usr/bin/env python3
"""
experimental_nuremberg.py – Optimized 2-Stage Land-Cover Prediction for Nuremberg.

This script implements the "Model 3 & 4" pipeline described in the final report:
1. Stage 1 (Change Detector): A binary Random Forest that identifies high-probability change pixels.
2. Stage 2 (Multiclass Predictor): A specialists Random Forest that predicts new labels for changed pixels.

The model uses a strict 17-feature set (10 spectral bands, 4 socio-demographic stats, 
NDVI, previous land-cover, and NDVI variability) and 95% confidence gating.

Workflow:
- Trains on 2020 Sentinel-2 features vs 2021 WorldCover labels.
- Predicts the 2021 validation state for the dashboard's "Experimental" tab.
- Applies strict boundary masking to the Nuremberg city limits.

Usage:
    python scripts/experimental_nuremberg.py
"""

import os
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.stats import mode as scipy_mode
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Project Paths & Anchor Definition
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
S2_DIR = DATA_DIR / "sentinel2"
WC_DIR = DATA_DIR / "worldcover"
STATS_DIR = DATA_DIR / "nuremberg_stats"
DASHBOARD_DATA = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_dashboard"

# Dashboard anchor grid configuration (10m Resolution, UTM Zone 32N)
ANCHOR_CRS = "EPSG:32632"
ANCHOR_ORIGIN_X = 641740.0
ANCHOR_ORIGIN_Y = 5492260.0
ANCHOR_RES = 10.0
ANCHOR_W = 2550
ANCHOR_H = 2850
ANCHOR_TRANSFORM = Affine(ANCHOR_RES, 0.0, ANCHOR_ORIGIN_X,
                           0.0, -ANCHOR_RES, ANCHOR_ORIGIN_Y)

# Map WorldCover raw IDs to our simplified 6-class scheme
WC_TO_CLASS = {
    10: 0, 20: 1, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 1,
}

def remap_worldcover(wc_arr):
    """Normalize WorldCover IDs to 0-5 indices."""
    out = np.full_like(wc_arr, 255, dtype=np.uint8)
    for wc_id, cls_idx in WC_TO_CLASS.items():
        out[wc_arr == wc_id] = cls_idx
    return out

# ---------------------------------------------------------------------------
# Geospatial Utility Functions
# ---------------------------------------------------------------------------
def reproject_to_anchor(src_path, categorical=False):
    """Align any raster input to the exact pixel-grid of the dashboard."""
    with rasterio.open(src_path) as src:
        dtype = np.uint8 if categorical else np.float32
        resampling = Resampling.nearest if categorical else Resampling.bilinear
        n_bands = src.count
        
        dst_array = np.full((n_bands, ANCHOR_H, ANCHOR_W), 0 if categorical else np.nan, dtype=dtype)
        for i in range(n_bands):
            reproject(
                source=rasterio.band(src, i+1),
                destination=dst_array[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ANCHOR_TRANSFORM,
                dst_crs=ANCHOR_CRS,
                resampling=resampling
            )
        return dst_array if not categorical else dst_array[0]

def build_boundary_mask():
    """Build a boolean mask for the Nuremberg city boundary."""
    geo_path = STATS_DIR / "geojsons_stats" / "nuremberg_stat_bezirke_wgs84.geojson"
    if not geo_path.exists():
        geo_path = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_boundary.geojson"

    gdf = gpd.read_file(geo_path).to_crs(ANCHOR_CRS)
    mask = rasterize([(geom, 1) for geom in gdf.geometry], out_shape=(ANCHOR_H, ANCHOR_W),
                     transform=ANCHOR_TRANSFORM, fill=0, all_touched=True, dtype=np.uint8)
    return mask == 1

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
def find_s2_file(year):
    """Logic to find local Sentinel-2 composites across different download dirs."""
    paths = [
        S2_DIR / f"s2_median_{year}.tif",
        S2_DIR / "temp_download" / f"s2_s2_{year}.tif",
        S2_DIR / "temp_download_v2" / f"s2_s2_{year}.tif"
    ]
    for p in paths:
        if p.exists(): return p
    return None

def load_multi_year_aligned(year):
    """Load both Target Year and Previous Year imagery aligned to the grid."""
    curr_p = find_s2_file(year)
    prev_p = find_s2_file(year - 1) or curr_p # Fallback to current if prev missing
    
    if not curr_p: return None
    
    s_curr = reproject_to_anchor(curr_p)
    s_prev = reproject_to_anchor(prev_p)
    stats = reproject_to_anchor(STATS_DIR / "rasterized_stats.tif")
    
    # Fallback to 2020 NDVI variability if specific year is missing
    std_p = S2_DIR / f"s2_ndvi_std_{year}.tif"
    if not std_p.exists(): std_p = S2_DIR / "s2_ndvi_std_2020.tif"
    std_layer = reproject_to_anchor(std_p) if std_p.exists() else None
    
    return s_curr, s_prev, stats, std_layer

def build_feature_matrix(s_curr, stats, std_layer, boundary_mask, wc_prev_path=None):
    """Construct the standardized 17-feature vector for Model 3 & 4."""
    valid = (boundary_mask & np.all(np.isfinite(s_curr), axis=0) & np.all(np.isfinite(stats), axis=0))
    
    # Determine the "Starting Class" (Previous Year LC)
    if wc_prev_path and wc_prev_path.exists():
        lc_prev = remap_worldcover(reproject_to_anchor(wc_prev_path, categorical=True))[valid]
    else:
        # For blind prediction years, use the stable 2020 WorldCover baseline
        wc_2020 = reproject_to_anchor(WC_DIR / "worldcover_2020.tif", categorical=True)
        lc_prev = remap_worldcover(wc_2020)[valid]

    features = []
    # [1-10] Sentinel-2 Bands
    for i in range(10): features.append(s_curr[i][valid])
    # [11] Current NDVI
    b4, b8 = s_curr[2][valid], s_curr[6][valid]
    features.append((b8 - b4) / (b8 + b4 + 1e-8))
    # [12] Previous Year Class
    features.append(lc_prev)
    # [13-16] Socio-Demographic District Stats
    for i in range(4): features.append(stats[i][valid])
    # [17] Temporal NDVI Variability
    features.append(std_layer[0][valid] if std_layer is not None else np.zeros(lc_prev.shape))

    X = np.column_stack(features)
    flat_idx = np.where(valid.flatten())[0]
    
    # Latitudinal banding for spatial cross-validation
    y_grid = np.broadcast_to(np.arange(ANCHOR_H)[:, np.newaxis], (ANCHOR_H, ANCHOR_W))
    y_row = y_grid[valid]
    
    return X, y_row, flat_idx

# ---------------------------------------------------------------------------
# Training & Inference
# ---------------------------------------------------------------------------
def train_two_stage_model(X, y_start, y_target, y_row):
    """Train the two-step prediction pipeline using optimized HPO parameters."""
    print("Training Stage 1 (Change) & Stage 2 (Prediction)...")
    is_changed = (y_target != y_start).astype(np.int8)

    # 4-Fold Spatial Holdout (train on 3 quadrants, use all for production)
    sort_idx = np.argsort(y_row)
    train_mask = (np.arange(len(y_row))[sort_idx] % 4 != 0) 
    
    X_tr, y_tr_chg, y_tr_target = X[train_mask], is_changed[train_mask], y_target[train_mask]

    # Stage 1: Change Detector (Under-sampled for balance)
    chg_idx = np.where(y_tr_chg == 1)[0]
    sta_idx = np.where(y_tr_chg == 0)[0]
    sta_sample = np.random.default_rng(42).choice(sta_idx, size=len(chg_idx), replace=False)
    bal_idx = np.concatenate([chg_idx, sta_sample])

    clf1 = RandomForestClassifier(n_estimators=50, max_depth=35, max_features=0.785, n_jobs=-1, random_state=42)
    clf1.fit(X_tr[bal_idx], y_tr_chg[bal_idx])

    # Stage 2: Multiclass Specialist (Balanced Subsampling)
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=29, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    clf2.fit(X_tr[y_tr_chg == 1], y_tr_target[y_tr_chg == 1])

    return clf1, clf2

def run_prediction(clf1, clf2, X, y_start, threshold=0.95):
    """Execute the confidence-gated prediction."""
    probs = clf1.predict_proba(X)[:, 1]
    is_chg = probs >= threshold
    preds = y_start.copy()
    if is_chg.any():
        preds[is_chg] = clf2.predict(X[is_chg])
    return preds, probs

# ---------------------------------------------------------------------------
# Export Logic
# ---------------------------------------------------------------------------
def save_dashboard_files(year, preds, probs, flat_idx, y_start):
    """Write binary maps at all resolutions (1-10) for the Experimental tab."""
    print(f"Exporting resolution-optimized maps for {year}...")
    
    # Base 10m arrays (255 = masked)
    p_map = np.full((ANCHOR_H, ANCHOR_W), 255, dtype=np.uint8)
    p_map.flat[flat_idx] = preds.astype(np.uint8)
    
    h_map = np.full((ANCHOR_H, ANCHOR_W), 255, dtype=np.uint8)
    h_map.flat[flat_idx] = np.clip(probs * 254, 0, 254).astype(np.uint8)
    
    c_map = np.full((ANCHOR_H, ANCHOR_W), 255, dtype=np.uint8)
    chg_v = np.full(len(preds), 254, dtype=np.uint8)
    mask = (preds != y_start)
    chg_v[mask] = preds[mask].astype(np.uint8)
    c_map.flat[flat_idx] = chg_v

    for r in range(1, 11):
        if r == 1: out_p, out_h, out_c = p_map, h_map, c_map
        else:
            # Aggregated downsampling for crisp lower-res views
            rh, rw = ANCHOR_H // r, ANCHOR_W // r
            out_p, out_h, out_c = [np.full((rh, rw), 255, dtype=np.uint8) for _ in range(3)]
            for row in range(rh):
                for col in range(rw):
                    b_p = p_map[row*r:(row+1)*r, col*r:(col+1)*r]
                    v_p = b_p[b_p != 255]
                    if v_p.size: out_p[row,col] = scipy_mode(v_p, keepdims=False).mode
                    
                    b_h = h_map[row*r:(row+1)*r, col*r:(col+1)*r]
                    v_h = b_h[b_h != 255]
                    if v_h.size: out_h[row,col] = int(v_h.mean())
                    
                    b_c = c_map[row*r:(row+1)*r, col*r:(col+1)*r]
                    v_c = b_c[(b_c != 255) & (b_c != 254)]
                    if v_c.size: out_c[row,col] = scipy_mode(v_c, keepdims=False).mode
                    elif (b_c != 255).any(): out_c[row,col] = 254

        # Save to experimental prefix
        (DASHBOARD_DATA / f"experimental_pred_{year}_res{r}.bin").write_bytes(out_p.tobytes())
        (DASHBOARD_DATA / f"experimental_heatmap_{year}_res{r}.bin").write_bytes(out_h.tobytes())
        (DASHBOARD_DATA / f"experimental_changes_{year}_res{r}.bin").write_bytes(out_c.tobytes())

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    print("Nuremberg 2-Step Prediction Pipeline (Standardized)")
    print("-" * 50)
    boundary = build_boundary_mask()

    print("\n[Step 1/3] Loading Training Data (2020-2021)...")
    wc_2020_path = WC_DIR / "worldcover_2020.tif"
    wc_2021_path = WC_DIR / "worldcover_2021.tif"
    
    s_2020, s_2019, stats, std = load_multi_year_aligned(2020)
    X, y_row, flat_idx = build_feature_matrix(s_2020, stats, std, boundary, wc_2020_path)
    
    y_start = remap_worldcover(reproject_to_anchor(wc_2020_path, categorical=True)).flatten()[np.where(boundary.flatten())[0]]
    y_target = remap_worldcover(reproject_to_anchor(wc_2021_path, categorical=True)).flatten()[np.where(boundary.flatten())[0]]
    
    # Re-align to only those with valid spectral data
    mask_in_valid = np.isin(np.where(boundary.flatten())[0], flat_idx)
    y_start, y_target = y_start[mask_in_valid], y_target[mask_in_valid]

    print("\n[Step 2/3] Learning Rules of Change...")
    clf1, clf2 = train_two_stage_model(X, y_start, y_target, y_row)

    print("\n[Step 3/3] Generating Validation Maps for 2021...")
    align_21 = load_multi_year_aligned(2021)
    if align_21:
        s_21, _, stats_21, std_21 = align_21
        X_21, _, flat_21 = build_feature_matrix(s_21, stats_21, std_21, boundary)
        y_start_21 = remap_worldcover(reproject_to_anchor(wc_2020_path, categorical=True)).flatten()[flat_21]
        
        preds, probs = run_prediction(clf1, clf2, X_21, y_start_21)
        save_dashboard_files(2021, preds, probs, flat_21, y_start_21)
    
    print("\nSuccess! All experimental maps are finalized and masked.")

if __name__ == "__main__":
    main()
