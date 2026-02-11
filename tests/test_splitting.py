"""
Unit tests for src/splitting.py invariants.

Tests verify structural correctness of all fold assignment strategies,
not model performance. Run with: python -m pytest tests/test_splitting.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.splitting import (
    assign_tile_groups,
    build_contiguous_band_folds,
    build_morton_folds,
    build_random_folds,
    build_region_growing_folds,
    build_spatial_folds,
    build_buffered_folds_from_assignments,
    compute_fold_metrics,
)


# -- Fixtures ---------------------------------------------------------------

N_ROWS, N_COLS = 50, 60       # small grid for fast tests
BLOCK_ROWS, BLOCK_COLS = 5, 5
N_FOLDS = 5
SEED = 42


@pytest.fixture
def grid_setup():
    """Create a small grid with tile groups."""
    n = N_ROWS * N_COLS
    cell_ids = np.arange(n)
    groups, n_tile_cols, n_tile_rows = assign_tile_groups(
        cell_ids, N_COLS, BLOCK_ROWS, BLOCK_COLS
    )
    return groups, n, n_tile_cols, n_tile_rows


# -- Helpers -----------------------------------------------------------------

def _check_fold_invariants(folds, n, strategy_name):
    """Assert basic fold invariants: partition, no overlap, coverage."""
    all_test = []
    for i, (train_idx, test_idx) in enumerate(folds):
        # No overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0, \
            f"{strategy_name} fold {i}: train/test overlap"
        # Full coverage
        assert len(train_idx) + len(test_idx) == n, \
            f"{strategy_name} fold {i}: train+test != n " \
            f"({len(train_idx)}+{len(test_idx)} != {n})"
        # Non-empty
        assert len(train_idx) > 0, f"{strategy_name} fold {i}: empty train"
        assert len(test_idx) > 0, f"{strategy_name} fold {i}: empty test"
        all_test.append(test_idx)

    # Every sample appears in exactly one test fold
    combined = np.concatenate(all_test)
    assert len(combined) == n, f"{strategy_name}: test folds don't cover all samples"
    assert len(np.unique(combined)) == n, f"{strategy_name}: duplicate test samples"


def _check_tile_integrity(groups, fold_assignments, strategy_name):
    """Assert each tile maps to exactly one fold."""
    unique_tiles = np.unique(groups)
    for t in unique_tiles:
        mask = groups == t
        folds_in_tile = np.unique(fold_assignments[mask])
        assert len(folds_in_tile) == 1, \
            f"{strategy_name}: tile {t} spans folds {folds_in_tile}"


# -- Tests -------------------------------------------------------------------

class TestRandomFolds:
    def test_invariants(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        folds = build_random_folds(n, N_FOLDS, SEED)
        _check_fold_invariants(folds, n, "random")


class TestGroupedFolds:
    def test_invariants(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        folds, assignments = build_spatial_folds(groups, N_FOLDS)
        _check_fold_invariants(folds, n, "grouped")
        _check_tile_integrity(groups, assignments, "grouped")

    def test_tile_integrity(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, assignments = build_spatial_folds(groups, N_FOLDS)
        _check_tile_integrity(groups, assignments, "grouped")


class TestContiguousFolds:
    def test_invariants(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        folds, assignments = build_contiguous_band_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows
        )
        _check_fold_invariants(folds, n, "contiguous")
        _check_tile_integrity(groups, assignments, "contiguous")

    def test_connected(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, assignments = build_contiguous_band_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows
        )
        metrics = compute_fold_metrics(assignments, groups,
                                        n_tile_cols, n_tile_rows)
        assert (metrics["n_components"] == 1).all(), \
            "Contiguous folds should all be connected"


class TestMortonFolds:
    def test_invariants(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        folds, assignments = build_morton_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows
        )
        _check_fold_invariants(folds, n, "morton")
        _check_tile_integrity(groups, assignments, "morton")


class TestRegionGrowingFolds:
    def test_invariants(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        folds, assignments = build_region_growing_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED
        )
        _check_fold_invariants(folds, n, "region_growing")
        _check_tile_integrity(groups, assignments, "region_growing")

    def test_connected(self, grid_setup):
        """Region growing folds should be connected (single component)."""
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, assignments = build_region_growing_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED
        )
        metrics = compute_fold_metrics(assignments, groups,
                                        n_tile_cols, n_tile_rows)
        assert (metrics["n_components"] == 1).all(), \
            f"Region growing folds not connected: {list(metrics['n_components'])}"

    def test_balance(self, grid_setup):
        """Region growing should achieve reasonable weight balance."""
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, assignments = build_region_growing_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED
        )
        metrics = compute_fold_metrics(assignments, groups,
                                        n_tile_cols, n_tile_rows)
        # Max deviation should be under 20% for a regular grid
        assert metrics["weight_deviation_pct"].max() < 20.0, \
            f"Weight deviation too high: {metrics['weight_deviation_pct'].max():.1f}%"

    def test_n_starts_larger_than_tiles(self, grid_setup):
        """n_starts > n_tiles should not crash."""
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        folds, assignments = build_region_growing_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows,
            seed=SEED, n_starts=9999
        )
        _check_fold_invariants(folds, n, "region_growing_big_nstarts")

    def test_deterministic(self, grid_setup):
        """Same seed should produce same folds."""
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, a1 = build_region_growing_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED
        )
        _, a2 = build_region_growing_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED
        )
        np.testing.assert_array_equal(a1, a2)


class TestBufferedFolds:
    def test_no_overlap(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, rg_assignments = build_region_growing_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows, seed=SEED
        )
        buf_folds, n_excluded = build_buffered_folds_from_assignments(
            groups, rg_assignments, N_FOLDS,
            n_tile_cols, n_tile_rows, buffer_tiles=1,
        )
        for i, (tr, te) in enumerate(buf_folds):
            assert len(np.intersect1d(tr, te)) == 0, \
                f"Buffer fold {i}: train/test overlap"
            # Train + test + excluded = n
            assert len(tr) + len(te) + n_excluded[i] == n


class TestComputeFoldMetrics:
    def test_columns(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, assignments = build_contiguous_band_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows
        )
        metrics = compute_fold_metrics(assignments, groups,
                                        n_tile_cols, n_tile_rows)
        expected_cols = {"fold", "n_cells", "n_tiles", "n_components",
                         "weight_deviation_pct", "boundary_edges",
                         "compactness"}
        assert set(metrics.columns) == expected_cols

    def test_total_cells(self, grid_setup):
        groups, n, n_tile_cols, n_tile_rows = grid_setup
        _, assignments = build_contiguous_band_folds(
            groups, N_FOLDS, n_tile_cols, n_tile_rows
        )
        metrics = compute_fold_metrics(assignments, groups,
                                        n_tile_cols, n_tile_rows)
        assert metrics["n_cells"].sum() == n
