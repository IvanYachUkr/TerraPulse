"""
Data loading, scaling, and memory-mapped dataset construction.
"""

import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    CityConfig, CLASS_NAMES, N_CLASSES,
    city_features_path, city_labels_path,
)


def _ts():
    return time.strftime("%H:%M:%S")


def load_city_features(city: CityConfig, columns: List[str]) -> Optional[np.ndarray]:
    """Load specified feature columns from a city's parquet as float32 array.

    Returns None if the parquet file does not exist.
    """
    feat_path = city_features_path(city)
    if not os.path.exists(feat_path):
        return None
    # Only read needed columns (fast columnar read)
    available = set(c for c in columns)  # validated by caller
    cols_to_read = [c for c in columns if c != "cell_id"]
    df = pd.read_parquet(feat_path, columns=cols_to_read)
    arr = np.nan_to_num(df.values.astype(np.float32), nan=0.0)
    del df
    return arr


def load_city_labels(city: CityConfig, year: int = 2021) -> Optional[np.ndarray]:
    """Load and normalise land-cover labels to class fractions.

    Returns (n_cells, N_CLASSES) float32 array, or None if missing.
    Rows with zero total are dropped.
    """
    label_path = city_labels_path(city, year)
    if not os.path.exists(label_path):
        return None
    labels = pd.read_parquet(label_path)
    y = labels[CLASS_NAMES].values.astype(np.float32)
    del labels
    row_sums = y.sum(axis=1, keepdims=True)
    valid = (row_sums.ravel() > 0)
    y = y[valid]
    row_sums = row_sums[valid]
    y = y / np.maximum(row_sums, 1e-8)
    return y, valid


def build_memmap_dataset(
    cities: List[CityConfig],
    columns: List[str],
    out_dir: str,
) -> Tuple[str, str, int]:
    """Load all training cities into memory-mapped files.

    Returns (X_path, y_path, n_samples).
    """
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    X_path = os.path.join(out_dir, "X_train.dat")
    y_path = os.path.join(out_dir, "y_train.dat")
    n_features = len(columns)

    # Pre-count rows  
    city_counts = []
    valid_cities = []
    for city in cities:
        feat_path = city_features_path(city)
        if not os.path.exists(feat_path):
            continue
        n = pq.read_metadata(feat_path).num_rows
        city_counts.append(n)
        valid_cities.append(city)

    total_est = sum(city_counts)
    print(f"[{_ts()}] Pre-allocating memmap ({total_est:,} x {n_features})...")

    X_mm = np.memmap(X_path, dtype=np.float32, mode="w+",
                     shape=(total_est, n_features))
    y_mm = np.memmap(y_path, dtype=np.float32, mode="w+",
                     shape=(total_est, N_CLASSES))

    offset = 0
    for city, n in zip(valid_cities, city_counts):
        X_city = load_city_features(city, columns)
        if X_city is None:
            continue
        label_result = load_city_labels(city)
        if label_result is None:
            del X_city
            continue
        y_city, valid_mask = label_result

        # Apply same validity filter to features
        if not valid_mask.all():
            X_city = X_city[valid_mask]

        actual_n = min(X_city.shape[0], y_city.shape[0])
        X_mm[offset:offset + actual_n] = X_city[:actual_n]
        y_mm[offset:offset + actual_n] = y_city[:actual_n]
        del X_city, y_city
        offset += actual_n
        gc.collect()
        print(f"  [{city.name}] {actual_n:,} cells")

    # Trim if overestimated
    if offset < total_est:
        print(f"  Trimming from {total_est:,} to {offset:,}")
    X_mm.flush()
    y_mm.flush()

    # Re-open at actual size
    X_mm = np.memmap(X_path, dtype=np.float32, mode="r+",
                     shape=(offset, n_features))
    y_mm = np.memmap(y_path, dtype=np.float32, mode="r+",
                     shape=(offset, N_CLASSES))

    print(f"  Total: {offset:,} x {n_features}")
    return X_path, y_path, offset


def fit_scaler(X_memmap: np.ndarray, chunk_size: int = 200_000) -> StandardScaler:
    """Fit a StandardScaler on memmap data using partial_fit (low memory)."""
    scaler = StandardScaler()
    n = X_memmap.shape[0]
    for start in range(0, n, chunk_size):
        scaler.partial_fit(X_memmap[start:start + chunk_size])
    return scaler


def apply_scaler_inplace(X_memmap: np.ndarray, scaler: StandardScaler):
    """Apply scaler to memmap data in-place."""
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    X_memmap -= mean
    X_memmap /= scale


def load_val_to_gpu(
    cities: List[CityConfig],
    columns: List[str],
    scaler: StandardScaler,
    device: str,
) -> Dict[str, dict]:
    """Load validation cities into GPU tensors.

    Returns dict: city_name → {"X": tensor, "y_norm": tensor, "y_raw": ndarray}
    """
    import torch
    from .train import normalize_targets

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    val_tensors = {}
    for city in cities:
        X_v = load_city_features(city, columns)
        if X_v is None:
            continue
        label_result = load_city_labels(city)
        if label_result is None:
            del X_v
            continue
        y_v, valid_mask = label_result

        if not valid_mask.all():
            X_v = X_v[valid_mask]

        # Scale features
        X_v = (X_v - mean) / scale
        y_v_norm = normalize_targets(y_v)

        val_tensors[city.name] = {
            "X": torch.from_numpy(X_v).to(device),
            "y_norm": torch.from_numpy(y_v_norm).to(device),
            "y_raw": y_v,   # keep on CPU for R² eval
        }
        del X_v, y_v_norm
        print(f"  [{city.name}] {val_tensors[city.name]['X'].shape[0]:,} cells -> VRAM")
        gc.collect()

    return val_tensors
