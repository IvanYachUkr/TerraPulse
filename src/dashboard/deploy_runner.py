"""
Deploy runner — run the TerraPulse pipeline for an arbitrary bbox.

Uses the Rust `terrapulse pipeline` binary for download, feature extraction,
and ONNX-based inference. Python handles anchor creation, output conversion,
WorldCover labels, and grid GeoJSON.

API:
    submit(bbox, years)  →  job_id
    status(job_id)       →  progress dict
    results(job_id)      →  per-year prediction dicts
"""

import json
import math
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from math import ceil, floor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

ONNX_MODELS_DIR = os.path.join(
    PROJECT_ROOT, "data", "pipeline_output", "models", "onnx")
DEPLOY_DIR = os.path.join(PROJECT_ROOT, "data", "deploy_jobs")
TERRAPULSE_BIN = os.path.join(
    PROJECT_ROOT, "terrapulse", "target", "release", "terrapulse.exe"
)

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up",
               "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)
GRID_PX = 10
SENTINEL_RES = 10
SENTINEL_NODATA = -9999

WC_CLASS_MAP = {10: 0, 30: 1, 90: 1, 40: 2, 50: 3, 60: 4, 80: 5}


# ---------------------------------------------------------------------------
# Job state management
# ---------------------------------------------------------------------------
@dataclass
class DeployJob:
    job_id: str
    bbox: List[float]     # [west, south, east, north] WGS84
    epsg: int
    years: List[int]
    status: str = "pending"      # pending | running | complete | error
    progress: float = 0.0        # 0-100
    stage: str = ""              # current stage name
    messages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    grid_cells: int = 0
    result_years: List[int] = field(default_factory=list)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.messages.append(f"[{ts}] {msg}")
        print(f"  [deploy:{self.job_id[:8]}] {msg}")


# Global job store (in-memory; survives until API restart)
_JOBS: Dict[str, DeployJob] = {}


def _job_dir(job: DeployJob) -> str:
    d = os.path.join(DEPLOY_DIR, job.job_id)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Automatic UTM zone detection
# ---------------------------------------------------------------------------
def _auto_epsg(bbox):
    """Determine UTM EPSG from bbox center."""
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone  # North
    else:
        return 32700 + zone  # South


# ---------------------------------------------------------------------------
# STAGE 0: Create anchor GeoTIFF
# ---------------------------------------------------------------------------
def _create_anchor(job: DeployJob, out_dir: str) -> str:
    """Create an anchor GeoTIFF for the given bbox."""
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    path = os.path.join(out_dir, "anchor.tif")
    if os.path.exists(path):
        with rasterio.open(path) as src:
            nc = src.width // GRID_PX
            nr = src.height // GRID_PX
            job.grid_cells = nc * nr
            job.log(f"Anchor exists ({nc}x{nr}={nc*nr} cells)")
        return path

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(job.epsg)
    west, south, east, north = job.bbox
    left, bottom, right, top = transform_bounds(
        src_crs, dst_crs, west, south, east, north, densify_pts=21)

    ps = float(SENTINEL_RES)
    left_s = floor(left / ps) * ps
    top_s = ceil(top / ps) * ps
    right_s = ceil(right / ps) * ps
    bottom_s = floor(bottom / ps) * ps

    w0 = round((right_s - left_s) / ps)
    h0 = round((top_s - bottom_s) / ps)
    width = ceil(w0 / GRID_PX) * GRID_PX
    height = ceil(h0 / GRID_PX) * GRID_PX

    transform = Affine(ps, 0.0, left_s, 0.0, -ps, top_s)
    data = np.full((1, height, width), SENTINEL_NODATA, dtype=np.float32)

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="float32", crs=dst_crs, transform=transform,
        nodata=SENTINEL_NODATA, compress="lzw",
    ) as dst:
        dst.write(data)

    nc = width // GRID_PX
    nr = height // GRID_PX
    job.grid_cells = nc * nr
    job.log(f"Created anchor: {width}x{height} px, {nc}x{nr}={nc*nr} cells")
    return path


# ---------------------------------------------------------------------------
# STAGE 1: Run Rust pipeline (download + extract + predict)
# ---------------------------------------------------------------------------
def _run_rust_pipeline(job: DeployJob, out_dir: str, anchor_path: str):
    """Call `terrapulse pipeline` for download → extract → predict."""
    cmd = [
        TERRAPULSE_BIN, "pipeline",
        "--bbox",
        str(job.bbox[0]), str(job.bbox[1]),
        str(job.bbox[2]), str(job.bbox[3]),
        "--epsg", str(job.epsg),
        "--years", " ".join(str(y) for y in job.years),
        "--region", "deploy",
        "--data-dir", out_dir,
        "--anchor-ref", anchor_path,
        "--models-dir", ONNX_MODELS_DIR,
    ]

    job.log(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)

    # Stream stdout for progress (stderr merged in via STDOUT)
    for line in iter(proc.stdout.readline, ''):
        line = line.rstrip()
        if not line:
            continue

        # Update stage from Rust output
        if "STAGE 1: DOWNLOAD" in line:
            job.stage = "Downloading Sentinel-2 (Rust)"
            job.progress = 20
        elif "STAGE 2: EXTRACT" in line:
            job.stage = "Extracting features (Rust)"
            job.progress = 50
        elif "STAGE 3: PREDICT" in line:
            job.stage = "Running ONNX inference (Rust)"
            job.progress = 70
        elif "Pipeline complete" in line:
            job.progress = 80

        # Log interesting lines
        if any(kw in line for kw in [
            "scenes", "Written", "Loaded", "done:", "Done",
            "Wrote", "WARNING", "ERROR", "Pipeline", "cells",
            "Year", "Region", "BBOX", "Helper", "TerraPulse",
            "Scene", "Compositing", "TIMEOUT", "FAILED",
        ]):
            job.log(line.strip())

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Rust pipeline failed (exit code {proc.returncode})")


# ---------------------------------------------------------------------------
# STAGE 2: Convert Parquet predictions → JSON
# ---------------------------------------------------------------------------
def _convert_predictions(job: DeployJob, out_dir: str):
    """Read pred_mlp_*.parquet from Rust output and convert to JSON."""
    preds_dir = os.path.join(out_dir, "predictions")
    if not os.path.isdir(preds_dir):
        job.log("No predictions directory found — skipping conversion")
        return

    years = sorted(job.years)
    year_pairs = [f"{years[i]}_{years[i+1]}" for i in range(len(years) - 1)]

    for yp in year_pairs:
        # Derive the "current" year (second in pair)
        curr_year = int(yp.split("_")[1])

        # Read MLP predictions parquet
        parquet_path = os.path.join(preds_dir, f"pred_mlp_{yp}.parquet")
        if not os.path.exists(parquet_path):
            job.log(f"No MLP predictions for {yp}")
            continue

        out_path = os.path.join(out_dir, f"predictions_{curr_year}.json")
        if os.path.exists(out_path):
            job.log(f"Predictions {curr_year}: cached")
            if curr_year not in job.result_years:
                job.result_years.append(curr_year)
            continue

        df = pd.read_parquet(parquet_path)
        job.log(f"Converting predictions for {curr_year}: {len(df)} cells")

        # Rust parquet columns: cell_id, tree_cover_mlp, grassland_mlp, ...
        result = {}
        for _, row in df.iterrows():
            cid = str(int(row["cell_id"]))
            cell_data = {}
            for cls in CLASS_NAMES:
                col = f"{cls}_mlp"
                val = float(row[col]) if col in df.columns else 0.0
                cell_data[cls] = round(val, 4)
            result[cid] = cell_data

        with open(out_path, "w") as f:
            json.dump(result, f)

        if curr_year not in job.result_years:
            job.result_years.append(curr_year)
        job.log(f"Predictions {curr_year}: {len(result)} cells saved")


# ---------------------------------------------------------------------------
# STAGE 3: WorldCover labels (for 2020, 2021 ground truth)
# ---------------------------------------------------------------------------
def _download_worldcover(job: DeployJob, out_dir: str, anchor_path: str):
    """Download and reproject WorldCover labels for validation years."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    import urllib.request

    label_years = [y for y in job.years if y <= 2021]
    if not label_years:
        return

    # Determine WC tile from bbox center
    lat = (job.bbox[1] + job.bbox[3]) / 2
    lon = (job.bbox[0] + job.bbox[2]) / 2
    lat_tile = int(math.floor(lat / 3.0) * 3)
    lon_tile = int(math.floor(lon / 3.0) * 3)
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    tile = f"{ns}{abs(lat_tile):02d}{ew}{abs(lon_tile):03d}"

    tiles_dir = os.path.join(out_dir, "wc_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    for year in label_years:
        labels_path = os.path.join(out_dir, f"labels_{year}.json")
        if os.path.exists(labels_path):
            job.log(f"Labels {year}: cached")
            continue

        version = "v100" if year == 2020 else "v200"
        filename = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        wc_path = os.path.join(tiles_dir, filename)

        if not os.path.exists(wc_path):
            url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                   f"{version}/{year}/map/{filename}")
            job.log(f"Downloading WorldCover {year} ({tile})...")
            try:
                urllib.request.urlretrieve(url, wc_path)
            except Exception as e:
                job.log(f"WARNING: WorldCover download failed: {e}")
                continue

        # Reproject to anchor grid
        with rasterio.open(anchor_path) as ref:
            anchor_meta = {
                "crs": ref.crs, "transform": ref.transform,
                "width": ref.width, "height": ref.height,
            }

        dst_array = np.zeros(
            (anchor_meta["height"], anchor_meta["width"]), dtype=np.uint8)
        with rasterio.open(wc_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=dst_array,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=anchor_meta["transform"],
                dst_crs=anchor_meta["crs"],
                dst_nodata=0, resampling=Resampling.nearest,
            )

        # Aggregate per cell
        nc = anchor_meta["width"] // GRID_PX
        nr = anchor_meta["height"] // GRID_PX
        total_px = GRID_PX * GRID_PX
        result = {}
        cell_id = 0
        for row_idx in range(nr):
            for col_idx in range(nc):
                r0 = row_idx * GRID_PX
                c0 = col_idx * GRID_PX
                patch = dst_array[r0:r0+GRID_PX, c0:c0+GRID_PX]
                proportions = np.zeros(N_CLASSES, dtype=np.float32)
                for wc_code, our_class in WC_CLASS_MAP.items():
                    count = int(np.sum(patch == wc_code))
                    proportions[our_class] += count
                if total_px > 0:
                    proportions /= total_px
                result[str(cell_id)] = {
                    CLASS_NAMES[i]: round(float(proportions[i]), 4)
                    for i in range(N_CLASSES)
                }
                cell_id += 1

        with open(labels_path, "w") as f:
            json.dump(result, f)
        job.log(f"Labels {year}: {len(result)} cells")


# ---------------------------------------------------------------------------
# STAGE 4: Build grid GeoJSON (for map overlay)
# ---------------------------------------------------------------------------
def _build_grid_geojson(job: DeployJob, out_dir: str, anchor_path: str):
    """Create a GeoJSON FeatureCollection with cell polygons in WGS84."""
    import rasterio
    from rasterio.warp import transform

    grid_path = os.path.join(out_dir, "grid.json")
    if os.path.exists(grid_path):
        return

    with rasterio.open(anchor_path) as src:
        crs = src.crs
        t = src.transform
        width = src.width
        height = src.height

    nc = width // GRID_PX
    nr = height // GRID_PX
    ps = SENTINEL_RES

    features = []
    cell_id = 0
    for row_idx in range(nr):
        for col_idx in range(nc):
            # Cell corners in projected CRS
            x0 = t.c + col_idx * GRID_PX * ps
            y0 = t.f - row_idx * GRID_PX * ps
            x1 = x0 + GRID_PX * ps
            y1 = y0 - GRID_PX * ps

            # Transform to WGS84
            xs, ys = transform(crs, "EPSG:4326", [x0, x1, x1, x0, x0],
                               [y0, y0, y1, y1, y0])
            coords = list(zip(xs, ys))

            features.append({
                "type": "Feature",
                "properties": {"cell_id": cell_id},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })
            cell_id += 1

    geojson = {"type": "FeatureCollection", "features": features}
    with open(grid_path, "w") as f:
        json.dump(geojson, f)

    job.log(f"Grid GeoJSON: {cell_id} cells")


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------
def _run_pipeline(job: DeployJob):
    """Run the full deploy pipeline in a background thread."""
    try:
        job.status = "running"
        out_dir = _job_dir(job)

        # Stage 0: Anchor
        job.stage = "Creating anchor grid"
        job.progress = 5
        anchor_path = _create_anchor(job, out_dir)

        # Stage 1: Rust pipeline (download + extract + predict)
        job.stage = "Running Rust pipeline"
        job.progress = 10
        _run_rust_pipeline(job, out_dir, anchor_path)

        # Stage 2: Convert Parquet → JSON
        job.stage = "Converting predictions"
        job.progress = 82
        _convert_predictions(job, out_dir)

        # Stage 3: WorldCover labels
        job.stage = "Downloading labels"
        job.progress = 88
        _download_worldcover(job, out_dir, anchor_path)

        # Stage 4: Grid GeoJSON
        job.stage = "Building grid"
        job.progress = 95
        _build_grid_geojson(job, out_dir, anchor_path)

        job.status = "complete"
        job.progress = 100
        job.stage = "Complete"
        job.log("Pipeline complete!")

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def submit_job(bbox: List[float], years: List[int]) -> str:
    """Submit a new deploy job. Returns job_id."""
    job_id = uuid.uuid4().hex[:12]
    epsg = _auto_epsg(bbox)
    years = sorted(set(years))

    job = DeployJob(
        job_id=job_id,
        bbox=bbox,
        epsg=epsg,
        years=years,
    )
    _JOBS[job_id] = job

    # Run in background thread
    t = threading.Thread(target=_run_pipeline, args=(job,), daemon=True)
    t.start()

    job.log(f"Job submitted: bbox={bbox}, years={years}, EPSG={epsg}")
    return job_id


def get_job(job_id: str) -> Optional[DeployJob]:
    return _JOBS.get(job_id)


def get_results(job_id: str, year: int) -> Optional[dict]:
    """Load prediction results for a year."""
    job = _JOBS.get(job_id)
    if not job:
        return None
    path = os.path.join(_job_dir(job), f"predictions_{year}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Try labels
    path = os.path.join(_job_dir(job), f"labels_{year}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_grid(job_id: str) -> Optional[dict]:
    """Load grid GeoJSON."""
    job = _JOBS.get(job_id)
    if not job:
        return None
    path = os.path.join(_job_dir(job), "grid.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_labels(job_id: str, year: int) -> Optional[dict]:
    """Load ground-truth labels for a year."""
    job = _JOBS.get(job_id)
    if not job:
        return None
    path = os.path.join(_job_dir(job), f"labels_{year}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
