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

ONNX_MODELS_DIR = os.environ.get("ONNX_MODELS_DIR", os.path.join(
    PROJECT_ROOT, "data", "pipeline_output", "models", "onnx"))
DEPLOY_DIR = os.environ.get("DEPLOY_DIR", os.path.join(
    PROJECT_ROOT, "data", "deploy_jobs"))
_default_bin = "terrapulse.exe" if os.name == "nt" else "terrapulse"
TERRAPULSE_BIN = os.environ.get("TERRAPULSE_BIN", os.path.join(
    PROJECT_ROOT, "terrapulse", "target", "release", _default_bin
))

CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
N_CLASSES = len(CLASS_NAMES)
GRID_PX = 10
SENTINEL_RES = 10
SENTINEL_NODATA = -9999

# WorldCover ESA class codes → our class indices
# 10=Tree, 20=Shrubland, 30=Grassland, 40=Cropland, 50=Built-up,
# 60=Bare/sparse, 80=Water, 90=Herbaceous wetland → Grassland
WC_CLASS_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4, 60: 5, 80: 6}


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
        elif "STAGE 4: LABELS" in line:
            job.stage = "Downloading labels (Rust)"
            job.progress = 85
        elif "STAGE 5: GRID" in line:
            job.stage = "Building grid (Rust)"
            job.progress = 95
        elif "Pipeline complete" in line:
            job.progress = 100

        # Sync result years immediately if predictions are emitted
        if "Wrote json" in line and "predictions_" in line:
            try:
                yr_str = line.split("predictions_")[1].split(".json")[0]
                curr_year = int(yr_str)
                if curr_year not in job.result_years:
                    job.result_years.append(curr_year)
            except Exception:
                pass


        # Log interesting lines
        if any(kw in line for kw in [
            "scenes", "Written", "Loaded", "done:", "Done",
            "Wrote", "WARNING", "ERROR", "Pipeline", "cells",
            "Year", "Region", "BBOX", "Helper", "TerraPulse",
            "Scene", "Compositing", "TIMEOUT", "FAILED", "Labels", "Fetching"
        ]):
            job.log(line.strip())

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Rust pipeline failed (exit code {proc.returncode})")


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

        # Stage 1-5: Rust pipeline handles everything now
        job.stage = "Running Rust pipeline"
        job.progress = 10
        _run_rust_pipeline(job, out_dir, anchor_path)

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
