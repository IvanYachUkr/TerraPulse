"""
Phase 7: Spatial train/test split utilities.

Provides tile-based splitting with multiple fold assignment strategies
(scattered GroupKFold, contiguous row-bands, Morton Z-curve,
multi-start region growing), optional Chebyshev buffer exclusion,
contiguity/balance/compactness metrics, and leakage comparison.

Usage (from scripts/run_split.py):
    from src.splitting import (
        assign_tile_groups,
        build_spatial_folds,
        build_contiguous_band_folds,
        build_morton_folds,
        build_region_growing_folds,
        build_buffered_folds_from_assignments,
        build_random_folds,
        get_fold_indices,
        leakage_comparison,
        save_split_metadata,
    )
"""

import json
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler


# =====================================================================
# Tile assignment
# =====================================================================

def assign_tile_groups(cell_ids, n_gcols, block_rows, block_cols):
    """
    Assign each cell to a rectangular tile group based on its
    row/col position in the grid.

    Parameters
    ----------
    cell_ids : ndarray of int, shape (N,)
        Row-major cell identifiers (0 .. N-1).
    n_gcols : int
        Number of columns in the full grid (186 for Nuremberg).
    block_rows : int
        Number of cell rows per tile (e.g. 10 -> 1 km).
    block_cols : int
        Number of cell columns per tile (e.g. 10 -> 1 km).

    Returns
    -------
    groups : ndarray of int, shape (N,)
        Tile group ID for each cell.
    n_tile_cols : int
        Number of tile columns.
    n_tile_rows : int
        Number of tile rows.
    """
    row_idx = cell_ids // n_gcols
    col_idx = cell_ids % n_gcols

    tile_row = row_idx // block_rows
    tile_col = col_idx // block_cols

    n_tile_cols = int(tile_col.max()) + 1
    n_tile_rows = int(tile_row.max()) + 1

    groups = tile_row * n_tile_cols + tile_col
    return groups, n_tile_cols, n_tile_rows


# =====================================================================
# Fold builders
# =====================================================================

def build_spatial_folds(groups, n_folds):
    """
    Build spatial CV folds using sklearn GroupKFold.

    Each tile group is kept intact -- all cells in a tile go to the
    same fold. GroupKFold distributes tiles for balanced sample counts,
    which means test tiles are SCATTERED across the grid.

    Returns
    -------
    folds : list of (train_idx, test_idx) tuples
    fold_assignments : ndarray of int, shape (N,)
        Fold number (0..n_folds-1) for each sample.
    """
    gkf = GroupKFold(n_splits=n_folds)
    dummy_X = np.zeros((len(groups), 1))
    dummy_y = np.zeros(len(groups))

    folds = list(gkf.split(dummy_X, dummy_y, groups=groups))

    fold_assignments = np.full(len(groups), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(folds):
        fold_assignments[test_idx] = fold_idx

    assert (fold_assignments >= 0).all(), "Some cells not assigned to any fold"
    return folds, fold_assignments


def build_contiguous_band_folds(groups, n_folds, n_tile_cols, n_tile_rows):
    """
    Build spatial CV folds where each fold is a contiguous horizontal
    band of tile rows.

    For 17 tile rows and 5 folds, produces chunks like [4, 4, 3, 3, 3].
    Test regions are spatially contiguous, making buffered CV meaningful.

    Parameters
    ----------
    groups : ndarray of int, shape (N,)
        Tile group IDs from assign_tile_groups.
    n_folds : int
    n_tile_cols, n_tile_rows : int
        Tile grid dimensions.

    Returns
    -------
    folds : list of (train_idx, test_idx) tuples
    fold_assignments : ndarray of int, shape (N,)
        Fold number (0..n_folds-1) for each sample.
    """
    assert n_tile_rows >= n_folds, \
        f"Need at least one tile row per fold ({n_tile_rows} rows < {n_folds} folds)"
    # Compute tile_row for each cell from its tile group
    tile_rows = groups // n_tile_cols

    # Split tile rows into n_folds contiguous chunks
    # np.array_split handles uneven division (e.g. 17 rows -> [4,4,3,3,3])
    all_tile_rows = np.arange(n_tile_rows)
    chunks = np.array_split(all_tile_rows, n_folds)

    # Map tile_row -> fold
    tile_row_to_fold = np.full(n_tile_rows, -1, dtype=int)
    for fold_idx, chunk in enumerate(chunks):
        for tr in chunk:
            tile_row_to_fold[tr] = fold_idx

    fold_assignments = tile_row_to_fold[tile_rows]
    assert (fold_assignments >= 0).all(), "Some cells not assigned to any fold"

    # Build (train_idx, test_idx) tuples
    folds = []
    for fold_idx in range(n_folds):
        test_mask = fold_assignments == fold_idx
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        folds.append((train_idx, test_idx))

    return folds, fold_assignments


def _morton_interleave(x, y):
    """
    Compute Morton (Z-order) index by bit-interleaving x and y.

    Bit i of x -> position 2i, bit i of y -> position 2i+1.
    Uses explicit uint64 to avoid signed overflow on large grids.
    Supports tile grids up to 65535 x 65535.
    """
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    z = np.zeros_like(x, dtype=np.uint64)
    for i in range(16):
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z


def build_morton_folds(groups, n_folds, n_tile_cols, n_tile_rows):
    """
    Build spatial CV folds using Morton (Z-order) space-filling curve.

    Tiles are sorted along a Z-curve that preserves spatial locality,
    then split into K contiguous chunks. This produces mostly-contiguous
    folds with perfect balance.

    Parameters
    ----------
    groups : ndarray of int, shape (N,)
        Tile group IDs from assign_tile_groups.
    n_folds : int
    n_tile_cols, n_tile_rows : int
        Tile grid dimensions.

    Returns
    -------
    folds : list of (train_idx, test_idx) tuples
    fold_assignments : ndarray of int, shape (N,)
    """
    # Compute Morton index for each unique tile
    unique_tiles = np.unique(groups)
    tile_tr = unique_tiles // n_tile_cols
    tile_tc = unique_tiles % n_tile_cols
    morton_idx = _morton_interleave(tile_tr, tile_tc)

    # Sort tiles by Morton index, then split into K chunks
    sort_order = np.argsort(morton_idx)
    sorted_tiles = unique_tiles[sort_order]
    chunks = np.array_split(sorted_tiles, n_folds)

    # Map tile -> fold
    tile_to_fold = {}
    for fold_idx, chunk in enumerate(chunks):
        for t in chunk:
            tile_to_fold[t] = fold_idx

    fold_assignments = np.array([tile_to_fold[g] for g in groups], dtype=int)

    folds = []
    for fold_idx in range(n_folds):
        test_mask = fold_assignments == fold_idx
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        folds.append((train_idx, test_idx))

    return folds, fold_assignments


def _build_tile_adjacency(unique_tiles, n_tile_cols, n_tile_rows):
    """Build 4-connected (rook) adjacency dict on tile grid."""
    tile_set = set(unique_tiles.tolist())
    tile_adj = {t: [] for t in unique_tiles}
    for t in unique_tiles:
        tr, tc = t // n_tile_cols, t % n_tile_cols
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                nbr = nr * n_tile_cols + nc
                if nbr in tile_set:
                    tile_adj[t].append(nbr)
    return tile_adj


def _tile_manhattan(t1, t2, n_tile_cols):
    """Manhattan distance between two tiles on the grid."""
    r1, c1 = t1 // n_tile_cols, t1 % n_tile_cols
    r2, c2 = t2 // n_tile_cols, t2 % n_tile_cols
    return abs(r1 - r2) + abs(c1 - c2)


def _farthest_point_seeds(unique_tiles, n_folds, n_tile_cols, start_tile):
    """Pick K seed tiles maximally spread via greedy farthest-point."""
    seeds = [start_tile]
    for _ in range(1, n_folds):
        best_tile = None
        best_min_dist = -1
        for t in unique_tiles:
            if t in seeds:
                continue
            min_d = min(_tile_manhattan(t, s, n_tile_cols) for s in seeds)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_tile = t
        seeds.append(best_tile)
    return seeds


def _grow_regions(seeds, tile_adj, tile_weight, n_folds, n_tiles,
                  unique_tiles, n_tile_cols, rng):
    """
    Greedy BFS region growing from seeds.

    Always expands the lightest region. Candidate selection prefers
    tiles closest to the region's seed (compactness).

    Returns
    -------
    assigned : dict, tile_id -> fold_idx
    region_weight : list of int
    """
    assigned = {}
    region_weight = [0] * n_folds
    frontier = [set() for _ in range(n_folds)]

    for fold_idx, s in enumerate(seeds):
        assigned[s] = fold_idx
        region_weight[fold_idx] += tile_weight[s]
        for nbr in tile_adj[s]:
            if nbr not in assigned:
                frontier[fold_idx].add(nbr)

    while len(assigned) < n_tiles:
        order = sorted(range(n_folds), key=lambda i: region_weight[i])
        grown = False
        for fold_idx in order:
            if not frontier[fold_idx]:
                continue
            candidates = list(frontier[fold_idx] - set(assigned.keys()))
            if not candidates:
                frontier[fold_idx].clear()
                continue
            seed_t = seeds[fold_idx]
            candidates.sort(key=lambda t: (
                _tile_manhattan(t, seed_t, n_tile_cols), rng.random()
            ))
            chosen = candidates[0]
            assigned[chosen] = fold_idx
            region_weight[fold_idx] += tile_weight[chosen]
            frontier[fold_idx].discard(chosen)
            for nbr in tile_adj[chosen]:
                if nbr not in assigned:
                    frontier[fold_idx].add(nbr)
            grown = True
            break
        if not grown:
            for t in unique_tiles:
                if t not in assigned:
                    dists = [_tile_manhattan(t, seeds[i], n_tile_cols)
                             for i in range(n_folds)]
                    assigned[t] = int(np.argmin(dists))
                    region_weight[assigned[t]] += tile_weight[t]

    return assigned, region_weight


def _score_partition(region_weight, assigned, tile_adj, n_folds, unique_tiles):
    """
    Score a partition by balance + contiguity.

    Lower is better.
    balance_penalty = max relative deviation from target weight.
    contiguity_penalty = total number of connected components - n_folds.
    """
    total = sum(region_weight)
    target = total / n_folds
    balance_penalty = max(abs(w - target) / target for w in region_weight)

    # Count connected components per fold (BFS)
    contiguity_penalty = 0
    for fold_idx in range(n_folds):
        fold_tiles = {t for t, f in assigned.items() if f == fold_idx}
        if not fold_tiles:
            contiguity_penalty += 1
            continue
        visited = set()
        n_components = 0
        for start in fold_tiles:
            if start in visited:
                continue
            n_components += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for nbr in tile_adj[node]:
                    if nbr in fold_tiles and nbr not in visited:
                        stack.append(nbr)
        contiguity_penalty += n_components - 1  # 0 if connected

    return balance_penalty + 10.0 * contiguity_penalty


def build_region_growing_folds(groups, n_folds, n_tile_cols, n_tile_rows,
                                seed=42, n_starts=10):
    """
    Build spatial CV folds via multi-start region growing on tile graph.

    Algorithm:
    1. Try N_STARTS random starting tiles for farthest-point seeding.
    2. For each start, grow K regions via greedy BFS (lightest-first).
    3. Score each partition by balance + contiguity.
    4. Keep the best.

    Parameters
    ----------
    groups : ndarray of int, shape (N,)
        Tile group IDs from assign_tile_groups.
    n_folds : int (K)
    n_tile_cols, n_tile_rows : int
    seed : int
        RNG seed.
    n_starts : int
        Number of random restarts (default 10).

    Returns
    -------
    folds : list of (train_idx, test_idx) tuples
    fold_assignments : ndarray of int, shape (N,)
    """
    rng = np.random.RandomState(seed)
    unique_tiles = np.unique(groups)
    n_tiles = len(unique_tiles)

    tile_ids, counts = np.unique(groups, return_counts=True)
    tile_weight = dict(zip(tile_ids.tolist(), counts.tolist()))

    tile_adj = _build_tile_adjacency(unique_tiles, n_tile_cols, n_tile_rows)

    # Multi-start search
    best_score = float('inf')
    best_assigned = None

    start_tiles = [unique_tiles[0]]  # always try corner
    start_tiles.extend(rng.choice(unique_tiles, size=n_starts - 1,
                                   replace=False))

    for start_tile in start_tiles:
        seeds = _farthest_point_seeds(unique_tiles, n_folds,
                                      n_tile_cols, start_tile)
        assigned, region_weight = _grow_regions(
            seeds, tile_adj, tile_weight, n_folds, n_tiles,
            unique_tiles, n_tile_cols, rng,
        )
        score = _score_partition(region_weight, assigned, tile_adj,
                                 n_folds, unique_tiles)
        if score < best_score:
            best_score = score
            best_assigned = assigned

    fold_assignments = np.array([best_assigned[g] for g in groups], dtype=int)

    folds = []
    for fold_idx in range(n_folds):
        test_mask = fold_assignments == fold_idx
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        folds.append((train_idx, test_idx))

    return folds, fold_assignments


def compute_fold_metrics(fold_assignments, groups, n_tile_cols, n_tile_rows):
    """
    Compute contiguity, balance and compactness metrics for each fold.

    For each fold, performs BFS on the tile adjacency graph to count
    connected components, reports weight balance, and counts cut edges
    (adjacent tile pairs belonging to different folds) as a compactness
    proxy.

    Parameters
    ----------
    fold_assignments : ndarray of int, shape (N,)
    groups : ndarray of int, shape (N,)
    n_tile_cols, n_tile_rows : int

    Returns
    -------
    pd.DataFrame with columns:
        fold, n_cells, n_tiles, n_components, weight_deviation_pct,
        cut_edges, compactness_ratio
    """
    unique_tiles = np.unique(groups)
    tile_adj = _build_tile_adjacency(unique_tiles, n_tile_cols, n_tile_rows)

    # Map tile -> fold assignment (majority vote per tile)
    tile_ids, counts = np.unique(groups, return_counts=True)
    tile_to_fold = {}
    for t in tile_ids:
        mask = groups == t
        tile_to_fold[t] = int(fold_assignments[mask][0])

    n_folds = int(fold_assignments.max()) + 1
    total_cells = len(fold_assignments)
    target_weight = total_cells / n_folds

    # Count total edges and cut edges per fold
    # A cut edge: tile t in fold A, neighbor nbr in fold B (A != B)
    fold_cut_edges = [0] * n_folds
    fold_total_edges = [0] * n_folds
    for t in unique_tiles:
        t_fold = tile_to_fold[t]
        for nbr in tile_adj[t]:
            fold_total_edges[t_fold] += 1
            if tile_to_fold.get(nbr, t_fold) != t_fold:
                fold_cut_edges[t_fold] += 1

    rows = []
    for fold_idx in range(n_folds):
        fold_mask = fold_assignments == fold_idx
        fold_cells = int(fold_mask.sum())
        fold_tiles = set(np.unique(groups[fold_mask]).tolist())
        n_tiles_in_fold = len(fold_tiles)

        # BFS to count connected components
        visited = set()
        n_components = 0
        for start in fold_tiles:
            if start in visited:
                continue
            n_components += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for nbr in tile_adj[node]:
                    if nbr in fold_tiles and nbr not in visited:
                        stack.append(nbr)

        weight_dev = abs(fold_cells - target_weight) / target_weight * 100

        # Compactness: fraction of edges that are NOT cut (internal)
        # 1.0 = perfectly compact (no boundary), lower = more fragmented
        total_e = fold_total_edges[fold_idx]
        cut_e = fold_cut_edges[fold_idx]
        compactness = 1.0 - (cut_e / total_e) if total_e > 0 else 0.0

        rows.append({
            "fold": fold_idx,
            "n_cells": fold_cells,
            "n_tiles": n_tiles_in_fold,
            "n_components": n_components,
            "weight_deviation_pct": round(weight_dev, 1),
            "cut_edges": cut_e,
            "compactness_ratio": round(compactness, 3),
        })

    return pd.DataFrame(rows)



@lru_cache(maxsize=8)
def _precompute_tile_neighbors(n_tile_cols, n_tile_rows, buffer_tiles):
    """
    Precompute Chebyshev (square) neighborhood for each tile.

    Cached so repeated calls from get_fold_indices / buffered builders
    don't recompute. All args are hashable ints.

    Returns dict: tile_id -> frozenset of neighbor tile_ids (excluding self).
    """
    neighbors = {}
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile_id = tr * n_tile_cols + tc
            nbrs = set()
            for dr in range(-buffer_tiles, buffer_tiles + 1):
                for dc in range(-buffer_tiles, buffer_tiles + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < n_tile_rows and 0 <= nc < n_tile_cols:
                        nbrs.add(nr * n_tile_cols + nc)
            neighbors[tile_id] = frozenset(nbrs)
    return neighbors


def build_buffered_folds_from_assignments(groups, fold_assignments, n_folds,
                                           n_tile_cols, n_tile_rows,
                                           buffer_tiles=1):
    """
    Apply a tile-level Chebyshev (square) buffer to any fold assignment.

    For each fold, training tiles that are within `buffer_tiles`
    Chebyshev distance of any test tile are excluded from the
    training set. This prevents near-boundary leakage.

    Works with both scattered (GroupKFold) and contiguous (band)
    fold assignments.

    Parameters
    ----------
    groups : ndarray of int, shape (N,)
        Tile group IDs.
    fold_assignments : ndarray of int, shape (N,)
        Fold number per cell (from any fold builder).
    n_folds : int
    n_tile_cols, n_tile_rows : int
    buffer_tiles : int
        Chebyshev distance in tiles (default: 1 = 8-neighborhood).

    Returns
    -------
    folds : list of (train_idx, test_idx) tuples
        train_idx excludes cells in the buffer zone.
    n_excluded_per_fold : list of int
        Number of cells dropped from training per fold.
    """
    # Precompute neighbors once
    tile_neighbors = _precompute_tile_neighbors(
        n_tile_cols, n_tile_rows, buffer_tiles
    )

    buffered_folds = []
    n_excluded_per_fold = []

    for fold_idx in range(n_folds):
        test_mask = fold_assignments == fold_idx
        test_tiles = set(np.unique(groups[test_mask]))

        # Collect buffer tiles: neighbors of test tiles that aren't test
        buffer_tiles_set = set()
        for tt in test_tiles:
            for nbr in tile_neighbors.get(tt, set()):
                if nbr not in test_tiles:
                    buffer_tiles_set.add(nbr)

        buffer_cell_mask = np.isin(groups, list(buffer_tiles_set))
        train_mask = (~test_mask) & (~buffer_cell_mask)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        n_excluded = int(buffer_cell_mask.sum())

        buffered_folds.append((train_idx, test_idx))
        n_excluded_per_fold.append(n_excluded)

    return buffered_folds, n_excluded_per_fold


def build_random_folds(n_samples, n_folds, seed):
    """
    Build standard random CV folds (shuffled) for leakage comparison.

    Returns
    -------
    folds : list of (train_idx, test_idx) tuples
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dummy_X = np.zeros((n_samples, 1))
    return list(kf.split(dummy_X))


# =====================================================================
# Downstream helper
# =====================================================================

def get_fold_indices(groups, fold_assignments, fold_idx,
                     n_tile_cols, n_tile_rows, buffer_tiles=0):
    """
    Get train/test indices for a single fold, with optional buffer.

    This is the recommended entry point for Phase 8+ scripts.
    Load split_spatial.parquet, then call this to get indices.

    Parameters
    ----------
    groups : ndarray of int
        Tile group IDs (from split parquet's tile_group column).
    fold_assignments : ndarray of int
        Fold IDs (from split parquet's fold column).
    fold_idx : int
        Which fold to hold out as test.
    n_tile_cols, n_tile_rows : int
        Tile grid dimensions (from config or metadata JSON).
    buffer_tiles : int
        Chebyshev buffer in tiles (0 = no buffer).

    Returns
    -------
    train_idx, test_idx : ndarrays of int
    """
    test_mask = fold_assignments == fold_idx

    if buffer_tiles > 0:
        test_tiles = set(np.unique(groups[test_mask]))
        tile_neighbors = _precompute_tile_neighbors(
            n_tile_cols, n_tile_rows, buffer_tiles
        )
        buffer_tiles_set = set()
        for tt in test_tiles:
            for nbr in tile_neighbors.get(tt, set()):
                if nbr not in test_tiles:
                    buffer_tiles_set.add(nbr)
        buffer_mask = np.isin(groups, list(buffer_tiles_set))
        train_mask = (~test_mask) & (~buffer_mask)
    else:
        train_mask = ~test_mask

    return np.where(train_mask)[0], np.where(test_mask)[0]


# =====================================================================
# Leakage comparison
# =====================================================================

def leakage_comparison(X, y, fold_configs):
    """
    Train Ridge regression on each fold for multiple split strategies.
    Compare held-out R2 to quantify spatial leakage.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Feature matrix (already numeric, no NaNs).
    y : ndarray, shape (N, C)
        Multi-output labels (e.g. 6 land-cover proportions).
    fold_configs : list of (split_name, folds_list)
        Each entry is (str_name, list_of_(train_idx, test_idx)).

    Returns
    -------
    results : pd.DataFrame
        Columns: split_type, fold, r2_uniform, r2_weighted,
                 train_size, test_size.
    """
    rows = []

    for split_name, folds in fold_configs:
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # Guard against degenerate folds
            if len(train_idx) < 2 or len(test_idx) < 2:
                rows.append({
                    "split_type": split_name, "fold": fold_idx,
                    "r2_uniform": np.nan, "r2_weighted": np.nan,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                })
                continue

            # Scale inside the CV loop to prevent leakage
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            y_train = y[train_idx]
            y_test = y[test_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute R2 with NaN safety
            try:
                r2_uniform = float(r2_score(
                    y_test, y_pred, multioutput="uniform_average"))
            except ValueError:
                r2_uniform = np.nan
            try:
                r2_weighted = float(r2_score(
                    y_test, y_pred, multioutput="variance_weighted"))
            except ValueError:
                r2_weighted = np.nan

            if not np.isfinite(r2_uniform):
                r2_uniform = np.nan
            if not np.isfinite(r2_weighted):
                r2_weighted = np.nan

            rows.append({
                "split_type": split_name,
                "fold": fold_idx,
                "r2_uniform": round(r2_uniform, 4) if np.isfinite(r2_uniform) else np.nan,
                "r2_weighted": round(r2_weighted, 4) if np.isfinite(r2_weighted) else np.nan,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            })

    return pd.DataFrame(rows)


# =====================================================================
# Metadata
# =====================================================================

def save_split_metadata(path, *, block_rows, block_cols, cell_size_m,
                         n_folds, seed, buffer_tiles,
                         n_cells, n_groups, n_gcols, n_grows,
                         n_tile_cols, n_tile_rows):
    """Save split configuration as JSON for reproducibility."""
    meta = {
        "strategies": [
            "grouped_groupkfold",
            "contiguous_row_bands",
            "morton_z_curve",
            "region_growing",
        ],
        "primary_fold_column": "fold_region_growing",
        "block_rows": block_rows,
        "block_cols": block_cols,
        "cell_size_m": cell_size_m,
        "block_size_m": f"{block_rows * cell_size_m}x{block_cols * cell_size_m}",
        "n_folds": n_folds,
        "seed": seed,
        "buffer_tiles": buffer_tiles,
        "buffer_metric": "chebyshev",
        "n_cells": n_cells,
        "n_groups": n_groups,
        "grid_rows": n_grows,
        "grid_cols": n_gcols,
        "tile_rows": n_tile_rows,
        "tile_cols": n_tile_cols,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {path}")
