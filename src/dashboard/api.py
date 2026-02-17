"""
Dashboard API server for TerraPulse.

Serves precomputed JSON data (grid, labels, predictions, uncertainty)
to the React frontend. All data is loaded into memory at startup for
sub-millisecond response times.

Usage:
    python -m uvicorn src.dashboard.api:app --port 8000 --reload
"""

import json
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

MODELS = ["mlp", "tree", "ridge"]
CLASSES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]

# ---------------------------------------------------------------------------
# Data loading (cached at startup)
# ---------------------------------------------------------------------------

def _load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    with open(path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def get_grid():
    return _load_json("grid.json")


@lru_cache(maxsize=None)
def get_labels(year: int):
    return _load_json(f"labels_{year}.json")


@lru_cache(maxsize=None)
def get_change():
    return _load_json("labels_change.json")


@lru_cache(maxsize=None)
def get_predictions(model: str):
    return _load_json(f"predictions_{model}.json")


@lru_cache(maxsize=None)
def get_predictions_year(model: str, year: int):
    """Load predictions for a specific model and year.
    For 2021 falls back to the OOF predictions file.
    For 2022-2025 loads the pipeline prediction file.
    """
    name = f"predictions_{model}_{year}.json"
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # Fall back to the OOF file for 2021
    if year == 2021:
        return get_predictions(model)
    raise FileNotFoundError(f"No predictions for {model}/{year}")


def get_available_prediction_years():
    """Scan data directory for available prediction years."""
    years = {2021}  # Always available (OOF)
    for fname in os.listdir(DATA_DIR):
        # predictions_mlp_2023.json
        if fname.startswith("predictions_") and fname.endswith(".json"):
            parts = fname.replace(".json", "").split("_")
            if len(parts) == 3 and parts[2].isdigit():
                years.add(int(parts[2]))
    return sorted(years)


@lru_cache(maxsize=None)
def get_benchmark():
    return _load_json("model_benchmark.json")


@lru_cache(maxsize=None)
def get_conformal():
    return _load_json("conformal.json")


@lru_cache(maxsize=None)
def get_split():
    return _load_json("split.json")


@lru_cache(maxsize=None)
def get_evaluation():
    return _load_json("evaluation.json")


@lru_cache(maxsize=None)
def get_stress_tests():
    return _load_json("stress_tests.json")


@lru_cache(maxsize=None)
def get_failure_analysis():
    return _load_json("failure_analysis.json")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TerraPulse Dashboard API",
    version="1.0.0",
    description="Serves precomputed land-cover prediction data for the interactive dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/grid")
def grid():
    """GeoJSON FeatureCollection of all 29,946 grid cells (EPSG:4326)."""
    return JSONResponse(content=get_grid(), media_type="application/geo+json")


@app.get("/api/labels/{year}")
def labels(year: int):
    """Per-cell land-cover proportions for a given year."""
    if year not in (2020, 2021):
        raise HTTPException(status_code=404, detail="Year must be 2020 or 2021")
    return get_labels(year)


@app.get("/api/change")
def change():
    """Per-cell delta (2021 - 2020) for each land-cover class."""
    return get_change()


@app.get("/api/predictions/{model}")
def predictions(model: str):
    """OOF predicted proportions for all cells from a given final model."""
    if model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model}'. Available: {MODELS}",
        )
    return get_predictions(model)


@app.get("/api/predictions/{model}/{year}")
def predictions_year(model: str, year: int):
    """Predicted proportions for a model and year (2021=OOF, 2022-2025=pipeline)."""
    if model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model}'")
    try:
        return get_predictions_year(model, year)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No predictions for {model}/{year}")


@app.get("/api/prediction-years")
def prediction_years():
    """List available prediction years."""
    return get_available_prediction_years()


@app.get("/api/models")
def models():
    """Benchmark metrics for all models."""
    return get_benchmark()



@app.get("/api/conformal")
def conformal():
    """Conformal prediction coverage and interval widths per model per class."""
    return get_conformal()


@app.get("/api/split")
def split():
    """Per-cell spatial CV fold and tile group assignments."""
    return get_split()


@app.get("/api/evaluation")
def evaluation():
    """Phase 9 evaluation: per-class metrics, aggregate metrics, change detection."""
    return get_evaluation()


@app.get("/api/stress-tests")
def stress_tests():
    """Phase 9 stress tests: noise injection, season dropout, feature ablation."""
    return get_stress_tests()


@app.get("/api/failure-analysis")
def failure_analysis():
    """Phase 9 failure analysis: error breakdown by dominant land-cover class."""
    return get_failure_analysis()


@lru_cache(maxsize=None)
def get_explainability():
    return _load_json("explainability.json")


@app.get("/api/explainability")
def explainability():
    """Phase 10 explainability: feature importance, SHAP, explanations."""
    return get_explainability()


@app.get("/api/cell/{cell_id}")
def cell_detail(cell_id: int):
    """
    Full detail for a single cell: labels (both years), change,
    predictions from all models, split info.
    """
    cell_key = str(cell_id)

    labels_2020 = get_labels(2020).get(cell_key)
    labels_2021 = get_labels(2021).get(cell_key)
    change_data = get_change().get(cell_key)
    split_data = get_split().get(cell_key)

    if labels_2020 is None:
        raise HTTPException(status_code=404, detail=f"cell_id {cell_id} not found")

    # Gather predictions from all models (only for holdout cells)
    preds = {}
    for m in MODELS:
        model_preds = get_predictions(m)
        if cell_key in model_preds:
            preds[m] = model_preds[cell_key]

    return {
        "cell_id": cell_id,
        "labels_2020": labels_2020,
        "labels_2021": labels_2021,
        "change": change_data,
        "predictions": preds,
        "split": split_data,
    }


@app.get("/api/meta")
def meta():
    """Static metadata about the dataset."""
    return {
        "classes": CLASSES,
        "models": MODELS,
        "grid_size": 29946,
        "holdout_fold": 0,
        "cell_size_m": 100,
        "crs": "EPSG:4326",
        "aoi": "Nuremberg, Germany",
        "class_colors": {
            "tree_cover": "#2d6a4f",
            "grassland": "#95d5b2",
            "cropland": "#f4a261",
            "built_up": "#e76f51",
            "bare_sparse": "#d4a373",
            "water": "#0096c7",
        },
    }


# ---------------------------------------------------------------------------
# Deploy endpoints
# ---------------------------------------------------------------------------
from src.dashboard import deploy_runner


class DeployRequest(BaseModel):
    bbox: List[float]   # [west, south, east, north] WGS84
    years: List[int]    # e.g. [2020, 2021, 2022, 2023, 2024, 2025]


@app.post("/api/deploy")
def deploy_submit(req: DeployRequest):
    """Submit a new deploy job."""
    if len(req.bbox) != 4:
        raise HTTPException(400, "bbox must have 4 elements")
    if not req.years:
        raise HTTPException(400, "years must not be empty")
    job_id = deploy_runner.submit_job(req.bbox, req.years)
    return {"job_id": job_id}


@app.get("/api/deploy/status/{job_id}")
def deploy_status(job_id: str):
    """Get deploy job status."""
    job = deploy_runner.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "messages": job.messages[-20:],
        "error": job.error,
        "grid_cells": job.grid_cells,
        "result_years": sorted(job.result_years),
        "bbox": job.bbox,
        "epsg": job.epsg,
    }


@app.get("/api/deploy/results/{job_id}/{year}")
def deploy_results(job_id: str, year: int):
    """Get prediction results for a deployed region and year."""
    result = deploy_runner.get_results(job_id, year)
    if result is None:
        raise HTTPException(404, "Results not found")
    return result


@app.get("/api/deploy/grid/{job_id}")
def deploy_grid(job_id: str):
    """Get grid GeoJSON for a deployed region."""
    grid_data = deploy_runner.get_grid(job_id)
    if grid_data is None:
        raise HTTPException(404, "Grid not found")
    return JSONResponse(content=grid_data, media_type="application/geo+json")


@app.get("/api/deploy/labels/{job_id}/{year}")
def deploy_labels(job_id: str, year: int):
    """Get ground-truth labels for a deployed region and year."""
    labels_data = deploy_runner.get_labels(job_id, year)
    if labels_data is None:
        raise HTTPException(404, "Labels not found")
    return labels_data
