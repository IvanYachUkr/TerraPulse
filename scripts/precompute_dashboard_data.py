"""
Precompute dashboard JSON data from parquet/gpkg sources.

Converts the heavy geospatial + tabular data into lightweight JSON files
that the FastAPI dashboard backend can serve directly without runtime
geopandas or parquet dependencies.

Usage:
    python scripts/precompute_dashboard_data.py

Outputs -> src/dashboard/data/
"""

import json
import os
import sys

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
REPORTS = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
OUT_DIR = os.path.join(PROJECT_ROOT, "src", "dashboard", "data")

MODELS = ["ridge", "elasticnet", "extratrees", "rf", "catboost", "mlp"]
CLASSES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]


def _round_dict(d, decimals=4):
    """Round all float values in a dict."""
    return {k: round(v, decimals) if isinstance(v, float) else v for k, v in d.items()}


# ---------------------------------------------------------------------------
# 1. Grid -> GeoJSON (EPSG:4326)
# ---------------------------------------------------------------------------
def export_grid(out_dir):
    print("  Loading grid.gpkg ...")
    gdf = gpd.read_file(os.path.join(PROCESSED, "grid.gpkg"))
    print(f"  Reprojecting {len(gdf)} cells EPSG:32632 -> 4326 ...")
    gdf = gdf.to_crs(4326)

    # Simplify coordinates to 6 decimal places (~0.1m precision)
    features = []
    for _, row in gdf.iterrows():
        coords = row.geometry.__geo_interface__["coordinates"]
        # Round coordinate tuples
        rounded = [
            [tuple(round(c, 6) for c in pt) for pt in ring]
            for ring in coords
        ]
        features.append({
            "type": "Feature",
            "properties": {"cell_id": int(row.cell_id)},
            "geometry": {"type": "Polygon", "coordinates": rounded},
        })

    geojson = {"type": "FeatureCollection", "features": features}
    path = os.path.join(out_dir, "grid.json")
    with open(path, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [ok] grid.json -- {len(features)} features, {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------
def export_labels(out_dir):
    for year in [2020, 2021]:
        df = pd.read_parquet(os.path.join(PROCESSED, f"labels_{year}.parquet"))
        data = {}
        for _, row in df.iterrows():
            data[int(row.cell_id)] = _round_dict(
                {c: float(row[c]) for c in CLASSES}
            )
        path = os.path.join(out_dir, f"labels_{year}.json")
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"  [ok] labels_{year}.json -- {len(data)} cells")


# ---------------------------------------------------------------------------
# 3. Change (Delta labels)
# ---------------------------------------------------------------------------
def export_change(out_dir):
    df = pd.read_parquet(os.path.join(PROCESSED, "labels_change.parquet"))
    delta_cols = [f"delta_{c}" for c in CLASSES]
    data = {}
    for _, row in df.iterrows():
        data[int(row.cell_id)] = _round_dict(
            {c: float(row[c]) for c in delta_cols}
        )
    path = os.path.join(out_dir, "labels_change.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  [ok] labels_change.json -- {len(data)} cells")


# ---------------------------------------------------------------------------
# 4. Predictions (holdout fold only)
# ---------------------------------------------------------------------------
def export_predictions(out_dir):
    for model in MODELS:
        fname = f"predictions_{model}_core.parquet"
        path_in = os.path.join(PROCESSED, fname)
        if not os.path.exists(path_in):
            print(f"  [!] {fname} not found, skipping")
            continue
        df = pd.read_parquet(path_in)
        data = {}
        for _, row in df.iterrows():
            data[int(row.cell_id)] = _round_dict(
                {c: float(row[c]) for c in CLASSES}
            )
        path = os.path.join(out_dir, f"predictions_{model}.json")
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"  [ok] predictions_{model}.json -- {len(data)} cells")


# ---------------------------------------------------------------------------
# 5. Model benchmark
# ---------------------------------------------------------------------------
def export_benchmark(out_dir):
    df = pd.read_csv(os.path.join(REPORTS, "model_benchmark.csv"))
    records = []
    for _, row in df.iterrows():
        rec = {
            "model": row["model"],
            "r2_uniform": round(float(row["r2_uniform"]), 4),
            "r2_weighted": round(float(row["r2_weighted"]), 4),
            "mae_mean_pp": round(float(row["mae_mean_pp"]), 2),
            "rmse_mean_pp": round(float(row["rmse_mean_pp"]), 2),
            "aitchison_mean": round(float(row["aitchison_mean"]), 3),
        }
        # Per-class R2
        for c in CLASSES:
            key = f"r2_{c}"
            if key in row:
                rec[key] = round(float(row[key]), 4)
        records.append(rec)
    path = os.path.join(out_dir, "model_benchmark.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  [ok] model_benchmark.json -- {len(records)} models")


# ---------------------------------------------------------------------------
# 6. Conformal coverage
# ---------------------------------------------------------------------------
def export_conformal(out_dir):
    df = pd.read_csv(os.path.join(REPORTS, "conformal_coverage.csv"))
    data = {}
    for model in MODELS:
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue
        data[model] = {}
        for _, row in mdf.iterrows():
            data[model][row["class"]] = {
                "coverage_pct": round(float(row["coverage_pct"]), 1),
                "mean_width_pp": round(float(row["mean_width_pp"]), 2),
                "median_width_pp": round(float(row["median_width_pp"]), 2),
            }
    path = os.path.join(out_dir, "conformal.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [ok] conformal.json -- {len(data)} models")


# ---------------------------------------------------------------------------
# 7. Split info (which cells are holdout)
# ---------------------------------------------------------------------------
def export_split(out_dir):
    df = pd.read_parquet(os.path.join(PROCESSED, "split_spatial.parquet"))
    # Export fold_grouped (the split used for modeling) and tile_group
    data = {}
    for _, row in df.iterrows():
        data[int(row.cell_id)] = {
            "fold": int(row.fold_grouped),
            "tile_group": int(row.tile_group),
        }
    path = os.path.join(out_dir, "split.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  [ok] split.json -- {len(data)} cells")


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    print("[1/7] Grid (GeoJSON) ...")
    export_grid(OUT_DIR)

    print("[2/7] Labels ...")
    export_labels(OUT_DIR)

    print("[3/7] Change labels ...")
    export_change(OUT_DIR)

    print("[4/7] Predictions ...")
    export_predictions(OUT_DIR)

    print("[5/7] Model benchmark ...")
    export_benchmark(OUT_DIR)

    print("[6/7] Conformal coverage ...")
    export_conformal(OUT_DIR)

    print("[7/7] Split info ...")
    export_split(OUT_DIR)

    print(f"\n[OK] Done -- {len(os.listdir(OUT_DIR))} files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
