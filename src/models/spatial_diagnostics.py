"""
Phase 8: Spatial diagnostics — Moran's I on model residuals.

Tests whether model residuals exhibit spatial autocorrelation,
validating that the spatial split successfully decorrelates errors.
"""

import numpy as np
from collections import defaultdict


def _build_rook_adjacency(unique_tiles, n_tile_cols, n_tile_rows):
    """
    Build 4-connected (rook) adjacency dict on tile grid.

    Replicates the logic from src/splitting._build_tile_adjacency
    to avoid circular imports.
    """
    adj = defaultdict(set)
    tile_set = set(unique_tiles)
    for t in unique_tiles:
        r, c = divmod(t, n_tile_cols)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                nbr = nr * n_tile_cols + nc
                if nbr in tile_set:
                    adj[t].add(nbr)
    return adj


def morans_i(values, adjacency):
    """
    Compute Moran's I for a 1D array of values on a graph.

    Parameters
    ----------
    values : dict or ndarray
        If dict: node_id → value.
        If ndarray: indexed by node position.
    adjacency : dict
        node_id → set of neighbor node_ids.

    Returns
    -------
    I : float
        Moran's I statistic. Range roughly [-1, 1].
        > 0: positive spatial autocorrelation (similar neighbors)
        ≈ 0: no spatial pattern
        < 0: negative autocorrelation (dissimilar neighbors)
    """
    if isinstance(values, dict):
        nodes = sorted(values.keys())
        vals = np.array([values[n] for n in nodes])
    else:
        nodes = list(range(len(values)))
        vals = np.asarray(values)

    N = len(vals)
    if N < 2:
        return 0.0

    mean_val = vals.mean()
    devs = vals - mean_val
    ss = (devs ** 2).sum()

    if ss < 1e-15:
        return 0.0

    # Weighted sum and total weight
    W = 0.0
    weighted_sum = 0.0
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    for i_node, node in enumerate(nodes):
        for nbr in adjacency.get(node, []):
            if nbr in node_to_idx:
                j = node_to_idx[nbr]
                weighted_sum += devs[i_node] * devs[j]
                W += 1.0

    if W < 1e-15:
        return 0.0

    return (N / W) * (weighted_sum / ss)


def compute_tile_residuals(y_true, y_pred, tile_groups):
    """
    Compute per-tile mean residual for each class.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (N, C)
    tile_groups : ndarray, shape (N,)

    Returns
    -------
    tile_residuals : dict
        class_idx → dict(tile_id → mean_residual)
    unique_tiles : ndarray
    """
    residuals = y_true - y_pred  # (N, C)
    unique_tiles = np.unique(tile_groups)
    C = residuals.shape[1]

    tile_residuals = {}
    for c in range(C):
        tile_vals = {}
        for t in unique_tiles:
            mask = tile_groups == t
            tile_vals[int(t)] = float(residuals[mask, c].mean())
        tile_residuals[c] = tile_vals

    return tile_residuals, unique_tiles


def compute_residual_morans_i(y_true, y_pred, tile_groups,
                               n_tile_cols, n_tile_rows,
                               class_names=None):
    """
    Compute Moran's I on model residuals per class.

    Parameters
    ----------
    y_true, y_pred : ndarray (N, C)
    tile_groups : ndarray (N,)
    n_tile_cols, n_tile_rows : int
    class_names : list of str, optional

    Returns
    -------
    result : dict
        class_name → Moran's I value
    mean_morans_i : float
        Mean across classes.
    """
    C = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    unique_tiles = np.unique(tile_groups)
    adj = _build_rook_adjacency(unique_tiles, n_tile_cols, n_tile_rows)
    tile_res, _ = compute_tile_residuals(y_true, y_pred, tile_groups)

    result = {}
    for c in range(C):
        mi = morans_i(tile_res[c], adj)
        result[class_names[c]] = round(mi, 4)

    mean_mi = float(np.mean(list(result.values())))
    return result, mean_mi
