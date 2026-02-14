"""
Spectral index ablation: test different feature subsets on fold 4.
Each subset <500 features. Isolates which groups carry signal.
"""

import json
import os
import sys
import time
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_V2_DIR  # noqa: E402
from src.splitting import get_fold_indices  # noqa: E402
from src.models.evaluation import evaluate_model  # noqa: E402

SEED = 42
FOLD_ID = 4
CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
PARQUET_PATH = os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet")

# Texture patterns (to exclude)
TEXTURE = ["GLCM_", "LBP_", "Gabor_", "HOG_", "MP_", "SV_"]
# Control columns (to exclude)
CONTROL = ["valid_fraction_", "low_valid_fraction_", "reflectance_scale_",
           "full_features_computed_"]


def classify_col(c):
    """Classify a column into its feature category."""
    if any(c.startswith(p) for p in CONTROL):
        return "control"
    if any(p in c for p in TEXTURE):
        return "texture"
    if "delta_" in c:
        return "delta"

    # Individual index groups
    veg_idx = ["NDVI_", "SAVI_", "NDRE1_", "NDRE2_"]
    urban_idx = ["NDBI_", "BSI_"]
    water_idx = ["NDWI_", "NDMI_", "NBR_"]
    tc = ["TC_"]
    spatial = ["edge_", "lap_", "morans_"]
    ndvi_sp = ["NDVI_range", "NDVI_iqr"]

    # Check NDVI_range/iqr BEFORE general NDVI_
    if any(c.startswith(p) for p in ndvi_sp):
        return "spatial"
    if any(c.startswith(p) for p in spatial):
        return "spatial"
    if any(c.startswith(p) for p in tc):
        return "tc"
    if any(c.startswith(p) for p in veg_idx):
        return "veg_idx"
    if any(c.startswith(p) for p in urban_idx):
        return "urban_idx"
    if any(c.startswith(p) for p in water_idx):
        return "water_idx"

    # Band stats
    band_pat = re.compile(r'^B(02|03|04|05|06|07|08|8A|11|12)_')
    if band_pat.match(c):
        # Visible (B02-B04), NIR (B08), RedEdge (B05-B07,B8A), SWIR (B11-B12)
        band = band_pat.match(c).group(0).rstrip("_")
        if band in ("B02", "B03", "B04"):
            return "vis_bands"
        elif band in ("B08",):
            return "nir_band"
        elif band in ("B05", "B06", "B07", "B8A"):
            return "rededge_bands"
        elif band in ("B11", "B12"):
            return "swir_bands"
    return "other"


def main():
    t0_total = time.time()

    # Read schema
    schema = pq.read_schema(PARQUET_PATH)
    all_cols = [c for c in schema.names if c != "cell_id"]

    # Classify all columns
    groups = {}
    for c in all_cols:
        cat = classify_col(c)
        groups.setdefault(cat, []).append(c)

    print("Feature categories:", flush=True)
    for cat in sorted(groups.keys()):
        print(f"  {cat:<20s}: {len(groups[cat]):>5d}", flush=True)

    # Load splits + labels
    print("\nLoading splits + labels...", flush=True)
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    y = labels_df[CLASS_NAMES].values.astype(np.float32)

    train_idx, test_idx = get_fold_indices(
        tiles, folds_arr, FOLD_ID, meta["tile_cols"], meta["tile_rows"],
        buffer_tiles=1,
    )
    print(f"Fold {FOLD_ID}: train={len(train_idx)}, test={len(test_idx)}", flush=True)

    # Define experiments (each under 500 features)
    experiments = [
        # Individual groups
        ("Vis bands (B02-04)", groups.get("vis_bands", [])),
        ("NIR band (B08)", groups.get("nir_band", [])),
        ("Red-edge (B05-07,8A)", groups.get("rededge_bands", [])),
        ("SWIR bands (B11-12)", groups.get("swir_bands", [])),
        ("Veg indices (NDVI,SAVI,NDRE)", groups.get("veg_idx", [])),
        ("Urban idx (NDBI,BSI)", groups.get("urban_idx", [])),
        ("Water idx (NDWI,NDMI,NBR)", groups.get("water_idx", [])),
        ("Tasseled Cap", groups.get("tc", [])),
        ("Spatial feats", groups.get("spatial", [])),

        # Combos
        ("All bands only", groups.get("vis_bands", []) + groups.get("nir_band", [])
         + groups.get("rededge_bands", []) + groups.get("swir_bands", [])),
        ("All indices only", groups.get("veg_idx", []) + groups.get("urban_idx", [])
         + groups.get("water_idx", [])),
        ("Bands + indices", groups.get("vis_bands", []) + groups.get("nir_band", [])
         + groups.get("rededge_bands", []) + groups.get("swir_bands", [])
         + groups.get("veg_idx", []) + groups.get("urban_idx", [])
         + groups.get("water_idx", [])),
        ("Bands+idx+TC+spatial", groups.get("vis_bands", []) + groups.get("nir_band", [])
         + groups.get("rededge_bands", []) + groups.get("swir_bands", [])
         + groups.get("veg_idx", []) + groups.get("urban_idx", [])
         + groups.get("water_idx", []) + groups.get("tc", [])
         + groups.get("spatial", [])),
    ]

    # Run
    print(f"\n{'='*65}", flush=True)
    print(f"{'Feature Set':<30s} {'N feat':>7s} {'R2':>8s} {'MAE':>8s} {'Time':>6s}",
          flush=True)
    print(f"{'='*65}", flush=True)

    for name, cols in experiments:
        if not cols:
            print(f"{name:<30s}       0     skip", flush=True)
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

        print(f"{name:<30s} {len(cols):>7d} {r2:>8.4f} {mae:>8.3f} {elapsed:>5.1f}s",
              flush=True)
        del df, X, model

    print(f"\nTotal time: {time.time() - t0_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
