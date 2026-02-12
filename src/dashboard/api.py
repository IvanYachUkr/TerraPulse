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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

MODELS = ["ridge", "elasticnet", "extratrees", "rf", "catboost", "mlp"]
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
def get_benchmark():
    return _load_json("model_benchmark.json")


@lru_cache(maxsize=None)
def get_conformal():
    return _load_json("conformal.json")


@lru_cache(maxsize=None)
def get_split():
    return _load_json("split.json")


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
    """Predicted proportions for holdout cells (fold 0) from a given model."""
    if model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model}'. Available: {MODELS}",
        )
    return get_predictions(model)


@app.get("/api/models")
def models():
    """Benchmark metrics for all models."""
    return get_benchmark()


@app.get("/api/conformal")
def conformal():
    """Conformal prediction coverage and interval widths per model per class."""
    return get_conformal()


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
