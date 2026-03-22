#!/usr/bin/env python3
"""Run CatBoost V5 pixel predictions on freshly downloaded nuremberg_dashboard_v2 data.

Matches Rust pipeline behavior:
  - Temporal NaN fill (same season, other year)
  - Zero-fill remaining NaN
  - Predict on ALL pixels (no hard intersection mask)

Usage:
    python scripts/_predict_nuremberg_v2.py
    python scripts/_predict_nuremberg_v2.py 2020 2021   # single pair
"""
import gc, os, sys, time
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "cities" / "nuremberg_dashboard_v2" / "raw"
MODEL_PATH = PROJECT_DIR / "data" / "cities" / "models_pixel_v5" / "catboost_pixel_v5_deep_unweighted.cbm"
OUTPUT_DIR = PROJECT_DIR / "data" / "cities" / "nuremberg_bins_v2"

ANCHOR_W = 2550
ANCHOR_H = 2850
NODATA = -9999.0

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEASONS = ["spring", "summer", "autumn"]
INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDMI", "NBR", "BSI", "EVI2", "NDRE1", "NDRE2"]

MODEL_TO_DASHBOARD = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

def ts():
    return time.strftime("%H:%M:%S")

def load_raster(path, bands=None):
    import rasterio
    with rasterio.open(path) as src:
        if bands is not None:
            return src.read(bands).astype(np.float32)
        return src.read().astype(np.float32)

def compute_indices(s2):
    B04 = s2[2]; B05 = s2[3]; B06 = s2[4]; B08 = s2[6]; B8A = s2[7]
    B11 = s2[8]; B12 = s2[9]; B03 = s2[1]
    eps = 1e-10
    return {
        "NDVI": (B08 - B04) / (B08 + B04 + eps),
        "NDWI": (B03 - B08) / (B03 + B08 + eps),
        "NDBI": (B11 - B08) / (B11 + B08 + eps),
        "NDMI": (B08 - B11) / (B08 + B11 + eps),
        "NBR":  (B08 - B12) / (B08 + B12 + eps),
        "BSI":  ((B11 + B04) - (B08 + B03)) / ((B11 + B04) + (B08 + B03) + eps),
        "EVI2": 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1.0 + eps),
        "NDRE1": (B08 - B05) / (B08 + B05 + eps),
        "NDRE2": (B08 - B06) / (B08 + B06 + eps),
    }


def temporal_nan_fill(data, year1, year2):
    """Fill NaN from same season in the other year (matches Rust temporal fill)."""
    filled = 0
    for season in SEASONS:
        d1 = data[(year1, season)]
        d2 = data[(year2, season)]
        for key in ["s2", "sar"]:
            arr1 = d1[key]
            arr2 = d2[key]
            for band_idx in range(arr1.shape[0]):
                nan1 = ~np.isfinite(arr1[band_idx])
                ok2 = np.isfinite(arr2[band_idx])
                fill_mask = nan1 & ok2
                count = fill_mask.sum()
                if count > 0:
                    arr1[band_idx][fill_mask] = arr2[band_idx][fill_mask]
                    filled += count

                nan2 = ~np.isfinite(arr2[band_idx])
                ok1 = np.isfinite(arr1[band_idx])
                fill_mask2 = nan2 & ok1
                count2 = fill_mask2.sum()
                if count2 > 0:
                    arr2[band_idx][fill_mask2] = arr1[band_idx][fill_mask2]
                    filled += count2
    return filled


def build_features_for_year_pair(year1, year2):
    """Build the 217-feature pixel matrix with temporal fill + zero-fill.

    Matches Rust pipeline: no hard intersection mask, temporal fill, zero-fill.
    """
    print(f"[{ts()}] Building features for {year1}+{year2}...")

    data = {}
    for year in [year1, year2]:
        for season in SEASONS:
            s2_path = RAW_DIR / f"sentinel2_nuremberg_dashboard_{year}_{season}.tif"
            sar_path = RAW_DIR / f"sentinel1_nuremberg_dashboard_{year}_{season}.tif"

            if not s2_path.exists() or not sar_path.exists():
                print(f"  WARNING: Missing data for {year}/{season}")
                return None, None

            s2 = load_raster(s2_path, bands=list(range(1, 11)))
            sar = load_raster(sar_path)

            # Replace nodata with NaN
            s2[s2 == NODATA] = np.nan
            sar[sar == NODATA] = np.nan

            # Zero values from footprint edges -> NaN (matches Rust footprint masking)
            all_zero = np.all(s2 <= 1.0, axis=0)
            n_zero = all_zero.sum()
            if n_zero > 0:
                s2[:, all_zero] = np.nan

            data[(year, season)] = {"s2": s2, "sar": sar}

    # --- Temporal NaN fill (same season, other year) ---
    temporal_filled = temporal_nan_fill(data, year1, year2)
    print(f"  Temporal fill: {temporal_filled:,} values filled")

    # Compute indices AFTER temporal fill (so filled bands produce valid indices)
    for key in data:
        data[key]["indices"] = compute_indices(data[key]["s2"])

    # --- Build pixel mask: at least one valid band anywhere (not hard intersection) ---
    # A pixel is "totally dead" only if ALL bands in ALL seasons are NaN
    any_valid = np.zeros((ANCHOR_H, ANCHOR_W), dtype=bool)
    for key, d in data.items():
        any_valid |= np.any(np.isfinite(d["s2"]), axis=0)

    n_valid = any_valid.sum()
    total = ANCHOR_H * ANCHOR_W
    print(f"  Pixels with any valid data: {n_valid:,} / {total:,} ({100*n_valid/total:.1f}%)")

    # Build features for ALL pixels with any valid data
    features = []

    # --- Per season per year: 22 features ---
    for year in [year1, year2]:
        for season in SEASONS:
            d = data[(year, season)]
            for i in range(10):
                features.append(d["s2"][i][any_valid])
            for idx_name in INDEX_NAMES:
                features.append(d["indices"][idx_name][any_valid])
            vv = d["sar"][0][any_valid]
            vh = d["sar"][1][any_valid]
            features.append(vv)
            features.append(vh)
            features.append(vv / (vh + 1e-10))

    # --- Intra-annual index diffs ---
    for year in [year1, year2]:
        for idx_name in INDEX_NAMES:
            sp = data[(year, "spring")]["indices"][idx_name][any_valid]
            sm = data[(year, "summer")]["indices"][idx_name][any_valid]
            features.append(sm - sp)
        for idx_name in INDEX_NAMES:
            sm = data[(year, "summer")]["indices"][idx_name][any_valid]
            au = data[(year, "autumn")]["indices"][idx_name][any_valid]
            features.append(au - sm)

    # --- Interannual index diffs ---
    for season in SEASONS:
        for idx_name in INDEX_NAMES:
            v1 = data[(year1, season)]["indices"][idx_name][any_valid]
            v2 = data[(year2, season)]["indices"][idx_name][any_valid]
            features.append(v2 - v1)

    # --- Index ranges ---
    for year in [year1, year2]:
        for idx_name in ["NDVI", "NDWI", "EVI2", "BSI"]:
            vals = np.stack([
                data[(year, s)]["indices"][idx_name][any_valid] for s in SEASONS
            ])
            features.append(vals.max(axis=0) - vals.min(axis=0))

    # --- SAR diffs (intra-annual) ---
    for year in [year1, year2]:
        for sar_idx in [0, 1]:  # VV, VH
            sp = data[(year, "spring")]["sar"][sar_idx][any_valid]
            sm = data[(year, "summer")]["sar"][sar_idx][any_valid]
            au = data[(year, "autumn")]["sar"][sar_idx][any_valid]
            features.append(sm - sp)
            features.append(au - sm)

    # --- SAR interannual diffs ---
    for season in SEASONS:
        for sar_idx in [0, 1]:
            v1 = data[(year1, season)]["sar"][sar_idx][any_valid]
            v2 = data[(year2, season)]["sar"][sar_idx][any_valid]
            features.append(v2 - v1)

    X = np.column_stack(features).astype(np.float32)

    # --- Zero-fill remaining NaN/Inf in-place (no temp alloc, avoids OOM) ---
    nan_before = 0
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        n_bad = bad.sum()
        if n_bad > 0:
            X[bad, col] = 0.0
            nan_before += n_bad
    print(f"  Zero-filled {nan_before:,} remaining NaN/Inf values")
    print(f"  Feature matrix: {X.shape}")

    del data, features
    gc.collect()

    return X, any_valid


def generate_bins(pred_map, year_label, n_classes=6):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for res in range(1, 11):
        if res == 1:
            out = pred_map
        else:
            rh = ANCHOR_H // res
            rw = ANCHOR_W // res
            cropped = pred_map[:rh * res, :rw * res]
            blocks = cropped.reshape(rh, res, rw, res)
            out = np.full((rh, rw), 255, dtype=np.uint8)
            for cls in range(n_classes):
                counts = np.sum(blocks == cls, axis=(1, 3))
                if cls == 0:
                    best_counts = counts.copy()
                    out[:] = 0
                else:
                    better = counts > best_counts
                    out[better] = cls
                    best_counts = np.maximum(best_counts, counts)
            valid_counts = np.sum(blocks != 255, axis=(1, 3))
            out[valid_counts == 0] = 255

        fname = f"nuremberg_pred_{year_label}_res{res}.bin"
        (OUTPUT_DIR / fname).write_bytes(out.tobytes())
        print(f"  Wrote {fname} ({out.shape[0]}x{out.shape[1]})")


def main():
    from catboost import CatBoostClassifier

    print("=" * 60)
    print("Predictions on nuremberg_dashboard_v2 (with temporal fill)")
    print("=" * 60)
    print(f"  RAW_DIR:  {RAW_DIR}")
    print(f"  OUTPUT:   {OUTPUT_DIR}")

    print(f"\n[{ts()}] Loading CatBoost model (GPU)...")
    model = CatBoostClassifier(task_type='GPU')
    model.load_model(str(MODEL_PATH))
    print(f"  Features: {len(model.feature_names_)}")

    all_pairs = [
        (2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021),
        (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025),
    ]

    if len(sys.argv) == 3:
        y1, y2 = int(sys.argv[1]), int(sys.argv[2])
        year_pairs = [(y1, y2)]
    else:
        year_pairs = all_pairs

    for y1, y2 in year_pairs:
        print(f"\n{'='*60}")
        print(f"  [{ts()}] Year pair: {y1} + {y2}")
        print(f"{'='*60}")

        X, valid = build_features_for_year_pair(y1, y2)
        if X is None:
            print(f"  SKIPPED (missing data)")
            continue

        print(f"[{ts()}] Running CatBoost GPU on {X.shape[0]:,} pixels...")
        start = time.time()
        preds_raw = model.predict(X).flatten().astype(int)
        elapsed = time.time() - start
        print(f"  Predicted in {elapsed:.1f}s")

        preds_dashboard = np.array([MODEL_TO_DASHBOARD[p] for p in preds_raw], dtype=np.uint8)

        pred_map = np.full((ANCHOR_H, ANCHOR_W), 255, dtype=np.uint8)
        pred_map[valid] = preds_dashboard

        generate_bins(pred_map, y2)

        del X, valid, preds_raw, preds_dashboard, pred_map
        gc.collect()

    print(f"\n[{ts()}] Done! Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
