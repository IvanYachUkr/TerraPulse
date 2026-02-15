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
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "src", "dashboard", "data")

MODELS = ["mlp", "tree", "ridge"]
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
# 4. Predictions (full OOF from final models)
# ---------------------------------------------------------------------------
def export_predictions(out_dir):
    for model in MODELS:
        model_dir = os.path.join(MODELS_DIR, f"final_{model}")
        oof_path = os.path.join(model_dir, "oof_predictions.parquet")
        if not os.path.exists(oof_path):
            print(f"  [!] {oof_path} not found, skipping")
            continue
        df = pd.read_parquet(oof_path)
        # OOF columns are {class}_pred, map to {class}
        pred_cols = [f"{c}_pred" for c in CLASSES]
        # Fall back to plain class names if _pred suffix not present
        if pred_cols[0] not in df.columns:
            pred_cols = CLASSES
        data = {}
        for _, row in df.iterrows():
            data[int(row.cell_id)] = _round_dict(
                {c: float(row[pc]) for c, pc in zip(CLASSES, pred_cols)}
            )
        path = os.path.join(out_dir, f"predictions_{model}.json")
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"  [ok] predictions_{model}.json -- {len(data)} cells (full OOF)")


# ---------------------------------------------------------------------------
# 5. Model benchmark (from final model meta.json)
# ---------------------------------------------------------------------------
def export_benchmark(out_dir):
    records = []
    for model in MODELS:
        meta_path = os.path.join(MODELS_DIR, f"final_{model}", "meta.json")
        if not os.path.exists(meta_path):
            print(f"  [!] {meta_path} not found, skipping")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        rec = {
            "model": model,
            "r2_uniform": round(float(meta.get("r2_mean", 0)), 4),
            "mae_mean_pp": round(float(meta.get("mae_mean_pp", 0)), 2),
            "n_features": int(meta.get("n_features", 0)),
            "feature_set": meta.get("feature_set", ""),
        }
        # Per-fold metrics (key names vary: r2 or r2_uniform, mae or mae_mean_pp)
        fold_metrics = meta.get("fold_metrics", [])
        if fold_metrics:
            r2_key = "r2" if "r2" in fold_metrics[0] else "r2_uniform"
            mae_key = "mae" if "mae" in fold_metrics[0] else "mae_mean_pp"
            rec["fold_r2"] = [round(fm[r2_key], 4) for fm in fold_metrics]
            rec["fold_mae"] = [round(fm[mae_key], 2) for fm in fold_metrics]
        records.append(rec)
    path = os.path.join(out_dir, "model_benchmark.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  [ok] model_benchmark.json -- {len(records)} final models")


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


# ---------------------------------------------------------------------------
# 8. Evaluation metrics (Phase 9)
# ---------------------------------------------------------------------------
def export_evaluation(out_dir):
    phase9_tables = os.path.join(PROJECT_ROOT, "reports", "phase9", "tables")

    # Per-class metrics
    pc_path = os.path.join(phase9_tables, "per_class_metrics.csv")
    if not os.path.exists(pc_path):
        print("  [!] per_class_metrics.csv not found, skipping evaluation")
        return
    pc_df = pd.read_csv(pc_path)
    per_class = []
    for _, row in pc_df.iterrows():
        per_class.append({
            "model": row["model"],
            "class": row["class"],
            "r2": round(float(row["r2"]), 4),
            "mae_pp": round(float(row["mae_pp"]), 2),
            "rmse_pp": round(float(row["rmse_pp"]), 2),
        })

    # Aggregate metrics
    agg_path = os.path.join(phase9_tables, "aggregate_metrics.csv")
    aggregate = []
    if os.path.exists(agg_path):
        agg_df = pd.read_csv(agg_path)
        for _, row in agg_df.iterrows():
            aggregate.append({
                "model": row["model"],
                "r2_uniform": round(float(row["r2_uniform"]), 4),
                "mae_mean_pp": round(float(row["mae_mean_pp"]), 2),
                "aitchison_mean": round(float(row["aitchison_mean"]), 3),
                "aitchison_median": round(float(row["aitchison_median"]), 3),
                "kl_mean": round(float(row["kl_mean"]), 4),
            })

    # Change metrics
    change_path = os.path.join(phase9_tables, "change_metrics.csv")
    change_metrics = []
    if os.path.exists(change_path):
        ch_df = pd.read_csv(change_path)
        for _, row in ch_df.iterrows():
            change_metrics.append({
                "model": row["model"],
                "threshold": round(float(row["threshold"]), 2),
                "n_stable": int(row["n_stable"]),
                "n_changed": int(row["n_changed"]),
                "false_change_pct": round(float(row["false_change_rate_pct"]), 1),
                "missed_change_pct": round(float(row["missed_change_rate_pct"]), 1),
                "stability_mae_pp": round(float(row["stability_mae_pp"]), 2),
            })

    data = {
        "per_class": per_class,
        "aggregate": aggregate,
        "change_detection": change_metrics,
    }
    path = os.path.join(out_dir, "evaluation.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [ok] evaluation.json -- {len(per_class)} per-class rows, "
          f"{len(aggregate)} aggregate rows, {len(change_metrics)} change rows")


# ---------------------------------------------------------------------------
# 9. Stress tests (Phase 9)
# ---------------------------------------------------------------------------
def export_stress_tests(out_dir):
    phase9_tables = os.path.join(PROJECT_ROOT, "reports", "phase9", "tables")
    data = {}

    # Noise injection
    noise_path = os.path.join(phase9_tables, "stress_noise.csv")
    if os.path.exists(noise_path):
        df = pd.read_csv(noise_path)
        data["noise"] = [
            {"noise_sigma": round(float(r["noise_sigma"]), 2),
             "r2": round(float(r["r2"]), 4),
             "mae_pp": round(float(r["mae_pp"]), 2)}
            for _, r in df.iterrows()
        ]

    # Season dropout
    season_path = os.path.join(phase9_tables, "stress_season_dropout.csv")
    if os.path.exists(season_path):
        df = pd.read_csv(season_path)
        data["season_dropout"] = [
            {"season_dropped": r["season_dropped"],
             "r2": round(float(r["r2"]), 4),
             "mae_pp": round(float(r["mae_pp"]), 2),
             "n_zeroed": int(r["n_zeroed"])}
            for _, r in df.iterrows()
        ]

    # Feature group ablation
    ablation_path = os.path.join(phase9_tables, "stress_feature_ablation.csv")
    if os.path.exists(ablation_path):
        df = pd.read_csv(ablation_path)
        data["feature_ablation"] = [
            {"group_dropped": r["group_dropped"],
             "r2": round(float(r["r2"]), 4),
             "mae_pp": round(float(r["mae_pp"]), 2),
             "n_zeroed": int(r["n_zeroed"]),
             "n_remaining": int(r["n_remaining"])}
            for _, r in df.iterrows()
        ]

    if not data:
        print("  [!] No stress test CSVs found, skipping")
        return

    path = os.path.join(out_dir, "stress_tests.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [ok] stress_tests.json -- {len(data)} test types")


# ---------------------------------------------------------------------------
# 10. Failure analysis (Phase 9)
# ---------------------------------------------------------------------------
def export_failure_analysis(out_dir):
    phase9_tables = os.path.join(PROJECT_ROOT, "reports", "phase9", "tables")
    failure_path = os.path.join(phase9_tables, "failure_by_landcover.csv")
    if not os.path.exists(failure_path):
        print("  [!] failure_by_landcover.csv not found, skipping")
        return

    df = pd.read_csv(failure_path)
    data = []
    for _, row in df.iterrows():
        data.append({
            "dominant_class": row["dominant_class"],
            "n_cells": int(row["n_cells"]),
            "mae_pp": round(float(row["mae_pp"]), 2),
            "mae_std_pp": round(float(row["mae_std_pp"]), 2),
            "aitchison_mean": round(float(row["aitchison_mean"]), 3),
            "r2_uniform": round(float(row["r2_uniform"]), 4),
        })

    path = os.path.join(out_dir, "failure_analysis.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [ok] failure_analysis.json -- {len(data)} land-cover classes")


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    print("[1/10] Grid (GeoJSON) ...")
    export_grid(OUT_DIR)

    print("[2/10] Labels ...")
    export_labels(OUT_DIR)

    print("[3/10] Change labels ...")
    export_change(OUT_DIR)

    print("[4/10] Predictions ...")
    export_predictions(OUT_DIR)

    print("[5/10] Model benchmark ...")
    export_benchmark(OUT_DIR)

    print("[6/10] Conformal coverage ...")
    export_conformal(OUT_DIR)

    print("[7/10] Split info ...")
    export_split(OUT_DIR)

    print("[8/10] Evaluation metrics (Phase 9) ...")
    export_evaluation(OUT_DIR)

    print("[9/10] Stress tests (Phase 9) ...")
    export_stress_tests(OUT_DIR)

    print("[10/10] Failure analysis (Phase 9) ...")
    export_failure_analysis(OUT_DIR)

    print(f"\n[OK] Done -- {len(os.listdir(OUT_DIR))} files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
