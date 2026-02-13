"""
Spectral index/band ablation across ALL 5 folds.
Tests individual groups and meaningful combinations (<500 features each).
Outputs CSV for report generation.
"""

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_V2_DIR, PROJECT_ROOT  # noqa: E402
from src.splitting import get_fold_indices  # noqa: E402
from src.models.evaluation import evaluate_model  # noqa: E402

SEED = 42
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
PARQUET_PATH = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")
OUT_CSV = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables",
                       "ablation_spectral_all_folds.csv")

# Texture + control patterns (to exclude from base)
TEXTURE = ["GLCM_", "LBP_", "Gabor_", "HOG_", "MP_", "SV_"]
CONTROL = ["valid_fraction_", "low_valid_fraction_", "reflectance_scale_",
           "full_features_computed_"]


def classify_columns(feat_cols):
    """Classify columns into feature groups."""
    groups = {}
    band_pat = re.compile(r'^B(02|03|04|05|06|07|08|8A|11|12)_')

    for c in feat_cols:
        if any(c.startswith(p) for p in CONTROL):
            continue
        if any(p in c for p in TEXTURE):
            continue
        if "delta_" in c:
            groups.setdefault("deltas", []).append(c)
            continue

        # NDVI_range/iqr = spatial, check before veg indices
        if c.startswith("NDVI_range") or c.startswith("NDVI_iqr"):
            groups.setdefault("spatial", []).append(c)
        elif any(c.startswith(p) for p in ["edge_", "lap_", "morans_"]):
            groups.setdefault("spatial", []).append(c)
        elif any(c.startswith(p) for p in ["TC_"]):
            groups.setdefault("tc", []).append(c)
        elif any(c.startswith(p) for p in ["NDVI_", "SAVI_", "NDRE"]):
            groups.setdefault("veg_idx", []).append(c)
        elif any(c.startswith(p) for p in ["NDBI_", "BSI_"]):
            groups.setdefault("urban_idx", []).append(c)
        elif any(c.startswith(p) for p in ["NDWI_", "NDMI_", "NBR_"]):
            groups.setdefault("water_idx", []).append(c)
        elif band_pat.match(c):
            band = band_pat.match(c).group(0).rstrip("_")
            if band in ("B02", "B03", "B04"):
                groups.setdefault("vis_bands", []).append(c)
            elif band == "B08":
                groups.setdefault("nir_band", []).append(c)
            elif band in ("B05", "B06", "B07", "B8A"):
                groups.setdefault("rededge_bands", []).append(c)
            elif band in ("B11", "B12"):
                groups.setdefault("swir_bands", []).append(c)
        else:
            groups.setdefault("other", []).append(c)

    return groups


def main():
    t0_total = time.time()

    # Read schema
    schema = pq.read_schema(PARQUET_PATH)
    feat_cols = [c for c in schema.names if c != "cell_id"]
    groups = classify_columns(feat_cols)

    print("Feature categories:", flush=True)
    for cat in sorted(groups.keys()):
        print(f"  {cat:<20s}: {len(groups[cat]):>5d}", flush=True)

    # Build experiments
    g = groups  # shorthand
    experiments = [
        # -- Individual groups --
        ("Vis bands (B02-04)", g.get("vis_bands", [])),
        ("NIR band (B08)", g.get("nir_band", [])),
        ("Red-edge (B05-07,8A)", g.get("rededge_bands", [])),
        ("SWIR (B11-12)", g.get("swir_bands", [])),
        ("Veg idx", g.get("veg_idx", [])),
        ("Urban idx", g.get("urban_idx", [])),
        ("Water idx", g.get("water_idx", [])),
        ("Tasseled Cap", g.get("tc", [])),
        ("Spatial", g.get("spatial", [])),

        # -- Meaningful combos (from fold 4: top performers combined) --
        ("RedEdge+WaterIdx", g.get("rededge_bands", []) + g.get("water_idx", [])),
        ("RedEdge+TC", g.get("rededge_bands", []) + g.get("tc", [])),
        ("RedEdge+WaterIdx+TC", g.get("rededge_bands", []) + g.get("water_idx", [])
         + g.get("tc", [])),
        ("RedEdge+VegIdx+TC", g.get("rededge_bands", []) + g.get("veg_idx", [])
         + g.get("tc", [])),

        # -- Band group combos --
        ("Vis+SWIR", g.get("vis_bands", []) + g.get("swir_bands", [])),
        ("Vis+NIR+RedEdge", g.get("vis_bands", []) + g.get("nir_band", [])
         + g.get("rededge_bands", [])),

        # -- Standard sets --
        ("All bands", g.get("vis_bands", []) + g.get("nir_band", [])
         + g.get("rededge_bands", []) + g.get("swir_bands", [])),
        ("All indices", g.get("veg_idx", []) + g.get("urban_idx", [])
         + g.get("water_idx", [])),
        ("Bands+indices", g.get("vis_bands", []) + g.get("nir_band", [])
         + g.get("rededge_bands", []) + g.get("swir_bands", [])
         + g.get("veg_idx", []) + g.get("urban_idx", [])
         + g.get("water_idx", [])),
        ("Bands+idx+TC", g.get("vis_bands", []) + g.get("nir_band", [])
         + g.get("rededge_bands", []) + g.get("swir_bands", [])
         + g.get("veg_idx", []) + g.get("urban_idx", [])
         + g.get("water_idx", []) + g.get("tc", [])),
    ]

    # Load splits + labels
    print("\nLoading splits + labels...", flush=True)
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    # Run all folds × all experiments
    print(f"\n{'='*80}", flush=True)
    print(f"{'Fold':>4s} {'Feature Set':<30s} {'N feat':>7s} {'R2':>8s} {'MAE':>8s} {'Time':>6s}",
          flush=True)
    print(f"{'='*80}", flush=True)

    results = []
    for fold_id in range(5):
        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )
        print(f"\n--- Fold {fold_id} (train={len(train_idx)}, test={len(test_idx)}) ---",
              flush=True)

        for name, cols in experiments:
            if not cols:
                continue

            t0 = time.time()
            df = pd.read_parquet(PARQUET_PATH, columns=cols)
            X = np.nan_to_num(df.values.astype(np.float32), 0.0)

            model = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=300, max_depth=6, learning_rate=0.05,
                    min_samples_leaf=20, l2_regularization=0.1,
                    random_state=SEED, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30,
                )
            )
            model.fit(X[train_idx], y[train_idx])
            y_pred = np.clip(model.predict(X[test_idx]), 0, 100)

            summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
            r2 = summary["r2_uniform"]
            mae = summary["mae_mean_pp"]
            elapsed = time.time() - t0

            print(f"  F{fold_id} {name:<30s} {len(cols):>7d} {r2:>8.4f} {mae:>8.3f} {elapsed:>5.1f}s",
                  flush=True)
            results.append({
                "fold": fold_id, "group": name, "n_features": len(cols),
                "r2": r2, "mae": mae, "time": elapsed,
            })

            del df, X, model

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}", flush=True)
    print(f"Total time: {time.time() - t0_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
