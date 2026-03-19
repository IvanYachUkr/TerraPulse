"""
Feature selection and column discovery for MLP training.

Improvements over legacy build_bi_lbp:
  - Includes valid_fraction / low_valid_fraction as quality signals
"""

import os
from typing import List, Optional, Set

import pyarrow.parquet as pq

from .config import CityConfig, city_features_path

# Columns that are NEVER model features (IDs and pipeline internals).
# Note: valid_fraction and low_valid_fraction are deliberately NOT here —
# they carry useful data-quality signal for the model.
CONTROL_COLS = {"cell_id", "reflectance_scale", "full_features_computed"}

# Accepted first-token prefixes for spectral bands
_BAND_PREFIXES = {
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
}

# Accepted first-token prefixes for spectral indices
_INDEX_PREFIXES = {
    "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
    "NDRE1", "NDRE2", "EVI2", "CRI1", "MNDWI", "GNDVI", "NDTI", "IRECI", "TC",
}

# Spatial texture features (edge, lap, morans) are NOT included:
# empirically they don't discriminate between classes at 100m resolution
# and add 36 noisy columns. LBP already captures useful texture.


def select_features(column_names: List[str]) -> List[str]:
    """Select model-relevant feature columns from a full column list.

    Includes:
      - Spectral bands (B02..B12) per season
      - Spectral indices (NDVI, NDWI, ...) per season
      - LBP texture features
      - SAR features (VV, VH, CR, RVI, LBP, pheno, temporal)
      - Phenological features (*_pheno_*)
      - Data quality (valid_fraction, low_valid_fraction)

    Excludes:
      - Delta/change features (prefix 'delta')
      - Spatial texture (edge, lap, morans) — no class discrimination at 100m
      - Control columns (cell_id, reflectance_scale, ...)
    """
    selected = []
    for col in column_names:
        if col in CONTROL_COLS:
            continue
        if col.startswith("delta"):
            continue

        # Quality columns — explicitly include
        if col in ("valid_fraction", "low_valid_fraction"):
            selected.append(col)
            continue

        prefix = col.split("_")[0]

        if prefix in _BAND_PREFIXES or prefix in _INDEX_PREFIXES:
            selected.append(col)
        elif prefix == "LBP":
            selected.append(col)
        elif prefix == "SAR":
            selected.append(col)
        elif "_pheno_" in col:
            selected.append(col)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique = []
    for c in selected:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def get_common_columns(cities: List[CityConfig]) -> List[str]:
    """Find the intersection of numeric feature columns across all cities.

    Returns columns in the order they appear in the first city's schema.
    """
    numeric_types = {"float", "double", "int32", "int64", "float32", "float64"}
    all_col_sets: List[Set[str]] = []

    for city in cities:
        feat_path = city_features_path(city)
        if not os.path.exists(feat_path):
            continue
        schema = pq.read_schema(feat_path)
        cols = set()
        for field in schema:
            type_str = str(field.type).lower()
            if any(t in type_str for t in numeric_types):
                if field.name not in CONTROL_COLS:
                    cols.add(field.name)
        all_col_sets.append(cols)

    if not all_col_sets:
        raise RuntimeError("No parquet files found for any city")

    common = set.intersection(*all_col_sets)

    # Preserve schema order from first available city
    first_city = next(
        c for c in cities if os.path.exists(city_features_path(c))
    )
    schema = pq.read_schema(city_features_path(first_city))
    ordered = [f.name for f in schema if f.name in common]
    return ordered


def city_has_sar(city: CityConfig) -> bool:
    """Check if a city's parquet has SAR features."""
    feat_path = city_features_path(city)
    if not os.path.exists(feat_path):
        return False
    schema = pq.read_schema(feat_path)
    return any(f.name.startswith("SAR_") for f in schema)
