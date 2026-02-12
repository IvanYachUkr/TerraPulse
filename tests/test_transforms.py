"""
Unit tests for src/transforms.py — compositional data transform invariants.

Run with:  python -m pytest tests/test_transforms.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transforms import (
    aitchison_distance,
    alr_forward,
    alr_inverse,
    clr_forward,
    clr_inverse,
    closure,
    helmert_basis,
    ilr_forward,
    ilr_inverse,
    pivot_basis,
)


# -- Fixtures ---------------------------------------------------------------

D = 6  # number of land-cover classes
N = 100
RNG = np.random.RandomState(42)


@pytest.fixture
def random_compositions():
    """Generate random compositions on the 6-simplex."""
    raw = RNG.dirichlet(np.ones(D), size=N)
    return raw


@pytest.fixture
def edge_compositions():
    """Compositions with edge cases: uniform, near-zero, dominant class."""
    return np.array([
        [1 / 6] * 6,                            # uniform
        [0.98, 0.004, 0.004, 0.004, 0.004, 0.004],  # single dominant
        [0.0, 0.2, 0.3, 0.1, 0.4, 0.0],         # zeros present
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],         # sparse
    ])


# -- Basis tests ------------------------------------------------------------

class TestBasisConstruction:
    def test_helmert_shape(self):
        psi = helmert_basis(D)
        assert psi.shape == (D - 1, D)

    def test_pivot_shape(self):
        psi = pivot_basis(D)
        assert psi.shape == (D - 1, D)

    def test_helmert_orthonormal(self):
        psi = helmert_basis(D)
        eye = psi @ psi.T
        np.testing.assert_allclose(eye, np.eye(D - 1), atol=1e-12)

    def test_pivot_orthonormal(self):
        psi = pivot_basis(D)
        eye = psi @ psi.T
        np.testing.assert_allclose(eye, np.eye(D - 1), atol=1e-12)

    @pytest.mark.parametrize("d", [3, 4, 5, 6, 10])
    def test_helmert_various_D(self, d):
        psi = helmert_basis(d)
        eye = psi @ psi.T
        np.testing.assert_allclose(eye, np.eye(d - 1), atol=1e-12)

    @pytest.mark.parametrize("d", [3, 4, 5, 6, 10])
    def test_pivot_various_D(self, d):
        psi = pivot_basis(d)
        eye = psi @ psi.T
        np.testing.assert_allclose(eye, np.eye(d - 1), atol=1e-12)


# -- ILR roundtrip tests ----------------------------------------------------

class TestILRRoundtrip:
    def test_roundtrip_helmert(self, random_compositions):
        z = ilr_forward(random_compositions, basis=helmert_basis(D))
        y_rec = ilr_inverse(z, basis=helmert_basis(D))
        np.testing.assert_allclose(y_rec, random_compositions, atol=1e-5)

    def test_roundtrip_pivot(self, random_compositions):
        z = ilr_forward(random_compositions, basis=pivot_basis(D))
        y_rec = ilr_inverse(z, basis=pivot_basis(D))
        np.testing.assert_allclose(y_rec, random_compositions, atol=1e-5)

    def test_roundtrip_default_basis(self, random_compositions):
        """Default basis (Helmert) should also roundtrip."""
        z = ilr_forward(random_compositions)
        y_rec = ilr_inverse(z)
        np.testing.assert_allclose(y_rec, random_compositions, atol=1e-5)

    def test_roundtrip_edge_cases(self, edge_compositions):
        z = ilr_forward(edge_compositions)
        y_rec = ilr_inverse(z)
        # Edge cases use epsilon smoothing so won't match exactly
        # but should still be valid simplex
        assert y_rec.shape == edge_compositions.shape
        np.testing.assert_allclose(y_rec.sum(axis=1), 1.0, atol=1e-10)
        assert (y_rec > 0).all()

    def test_dimensionality(self, random_compositions):
        z = ilr_forward(random_compositions)
        assert z.shape == (N, D - 1)
        y_rec = ilr_inverse(z)
        assert y_rec.shape == (N, D)


# -- Simplex validity -------------------------------------------------------

class TestSimplexValidity:
    def test_ilr_inverse_on_simplex(self, random_compositions):
        z = ilr_forward(random_compositions)
        y_rec = ilr_inverse(z)
        np.testing.assert_allclose(y_rec.sum(axis=1), 1.0, atol=1e-10)
        assert (y_rec > 0).all()

    def test_ilr_inverse_random_z(self):
        """Random z vectors should always map to valid simplex."""
        z_rand = RNG.randn(50, D - 1) * 3  # large range
        y = ilr_inverse(z_rand)
        np.testing.assert_allclose(y.sum(axis=1), 1.0, atol=1e-10)
        assert (y > 0).all()

    def test_closure(self):
        raw = np.array([[2.0, 3.0, 5.0]])
        c = closure(raw)
        np.testing.assert_allclose(c, [[0.2, 0.3, 0.5]])


# -- CLR tests ---------------------------------------------------------------

class TestCLR:
    def test_roundtrip(self, random_compositions):
        c = clr_forward(random_compositions)
        y_rec = clr_inverse(c)
        np.testing.assert_allclose(y_rec, random_compositions, atol=1e-5)

    def test_rows_sum_to_zero(self, random_compositions):
        c = clr_forward(random_compositions)
        np.testing.assert_allclose(c.sum(axis=1), 0.0, atol=1e-10)


# -- ALR tests ---------------------------------------------------------------

class TestALR:
    def test_roundtrip(self, random_compositions):
        z = alr_forward(random_compositions, ref=-1)
        y_rec = alr_inverse(z, ref=-1)
        np.testing.assert_allclose(y_rec, random_compositions, atol=1e-5)

    def test_dimensionality(self, random_compositions):
        z = alr_forward(random_compositions)
        assert z.shape == (N, D - 1)


# -- Aitchison distance ------------------------------------------------------

class TestAitchisonDistance:
    def test_self_distance_zero(self, random_compositions):
        d = aitchison_distance(random_compositions, random_compositions)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_positive(self, random_compositions):
        y2 = random_compositions[np.roll(np.arange(N), 1)]
        d = aitchison_distance(random_compositions, y2)
        assert (d >= 0).all()

    def test_triangle_inequality(self, random_compositions):
        """d(a,c) <= d(a,b) + d(b,c) for random triplets."""
        a = random_compositions[:30]
        b = random_compositions[30:60]
        c = random_compositions[60:90]
        d_ac = aitchison_distance(a, c)
        d_ab = aitchison_distance(a, b)
        d_bc = aitchison_distance(b, c)
        assert (d_ac <= d_ab + d_bc + 1e-10).all()

    def test_basis_invariance(self, random_compositions):
        """Aitchison distance should be the same regardless of ILR basis."""
        y1 = random_compositions[:50]
        y2 = random_compositions[50:]
        d_helmert = aitchison_distance(y1, y2)
        # Also compute via ILR Euclidean distance with pivot basis
        z1_p = ilr_forward(y1, basis=pivot_basis(D))
        z2_p = ilr_forward(y2, basis=pivot_basis(D))
        d_pivot = np.sqrt(((z1_p - z2_p) ** 2).sum(axis=1))
        np.testing.assert_allclose(d_helmert, d_pivot, atol=1e-10)
