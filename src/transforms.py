"""
Compositional data transforms for land-cover proportions.

Implements ILR (Isometric Log-Ratio), CLR (Centred Log-Ratio), and
ALR (Additive Log-Ratio) transforms for mapping proportions on the
simplex to/from unconstrained Euclidean space.

Usage:
    from src.transforms import ilr_forward, ilr_inverse, aitchison_distance

    z = ilr_forward(y)            # (N, 6) → (N, 5)
    y_hat = ilr_inverse(z_pred)   # (N, 5) → (N, 6)
    d = aitchison_distance(y1, y2)

References:
    Aitchison (1986). The Statistical Analysis of Compositional Data.
    Egozcue et al. (2003). Isometric logratio transformations for
        compositional data analysis. Mathematical Geology 35(3).
"""

import numpy as np


# =====================================================================
# Basis construction
# =====================================================================

def helmert_basis(D):
    """
    Helmert subcomposition contrast matrix for D-part compositions.

    Returns an orthonormal (D-1, D) matrix Ψ such that Ψ @ Ψᵀ = I.
    Row i contrasts the geometric mean of parts 0..i against part i+1.

    Parameters
    ----------
    D : int
        Number of parts (e.g. 6 for 6 land-cover classes).

    Returns
    -------
    psi : ndarray, shape (D-1, D)
    """
    psi = np.zeros((D - 1, D))
    for i in range(D - 1):
        k = i + 1  # number of parts in the "left" group
        coeff = 1.0 / np.sqrt(k * (k + 1))
        psi[i, :k] = coeff
        psi[i, k] = -k * coeff
    return psi


def pivot_basis(D):
    """
    Pivot (sequential binary partition) contrast matrix for D-part
    compositions.

    Row i contrasts part i against the geometric mean of parts i+1..D-1.
    Also orthonormal.

    Parameters
    ----------
    D : int
        Number of parts.

    Returns
    -------
    psi : ndarray, shape (D-1, D)
    """
    psi = np.zeros((D - 1, D))
    for i in range(D - 1):
        r = D - i - 1  # number of parts in the "right" group
        coeff = 1.0 / np.sqrt(r * (r + 1))
        psi[i, i] = r * coeff
        psi[i, i + 1:] = -coeff
    return psi


def _validate_basis(psi, D):
    """Check that a contrast matrix is orthonormal and correctly shaped."""
    assert psi.shape == (D - 1, D), f"Basis shape {psi.shape}, expected ({D-1}, {D})"
    eye = psi @ psi.T
    np.testing.assert_allclose(
        eye, np.eye(D - 1), atol=1e-12,
        err_msg="Basis is not orthonormal: Ψ @ Ψᵀ ≠ I"
    )


# =====================================================================
# Epsilon smoothing (closure)
# =====================================================================

def _smooth_and_close(y, eps=1e-6):
    """
    Replace zeros with eps and renormalize each row to sum to 1.

    Parameters
    ----------
    y : ndarray, shape (N, D)
        Compositions (proportions). May contain exact zeros.
    eps : float
        Small constant to replace zeros.

    Returns
    -------
    y_smooth : ndarray, shape (N, D)
        Strictly positive compositions summing to 1.
    """
    y = np.asarray(y, dtype=np.float64)
    y_s = np.clip(y, eps, None)
    row_sums = y_s.sum(axis=1, keepdims=True)
    return y_s / row_sums


def closure(x):
    """
    Project rows onto the simplex by dividing by row sums.

    Parameters
    ----------
    x : ndarray, shape (N, D)
        Non-negative values (e.g. exp of log-ratios).

    Returns
    -------
    y : ndarray, shape (N, D)
        Rows summing to 1.
    """
    x = np.asarray(x, dtype=np.float64)
    return x / x.sum(axis=1, keepdims=True)


# =====================================================================
# ILR transform
# =====================================================================

def ilr_forward(y, basis=None, eps=1e-6):
    """
    ILR (Isometric Log-Ratio) forward transform.

    Maps compositions on the D-simplex to (D-1)-dimensional Euclidean
    space using an orthonormal contrast matrix (default: Helmert).

    Parameters
    ----------
    y : ndarray, shape (N, D)
        Compositions (proportions summing to ~1).
    basis : ndarray, shape (D-1, D), optional
        Contrast matrix. If None, uses Helmert basis.
    eps : float
        Smoothing constant for zero replacement.

    Returns
    -------
    z : ndarray, shape (N, D-1)
        ILR coordinates in Euclidean space.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    D = y.shape[1]
    if basis is None:
        basis = helmert_basis(D)
    _validate_basis(basis, D)

    y_s = _smooth_and_close(y, eps)
    return np.log(y_s) @ basis.T


def ilr_inverse(z, basis=None, D=None):
    """
    ILR inverse transform.

    Maps (D-1)-dimensional Euclidean coordinates back to the D-simplex.

    Parameters
    ----------
    z : ndarray, shape (N, D-1)
        ILR coordinates.
    basis : ndarray, shape (D-1, D), optional
        Contrast matrix (must match the one used in forward).
        If None, uses Helmert basis of dimension D.
    D : int, optional
        Number of parts. Inferred as z.shape[1]+1 if None.

    Returns
    -------
    y : ndarray, shape (N, D)
        Compositions on the simplex (all > 0, rows sum to 1).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    if D is None:
        D = z.shape[1] + 1
    if basis is None:
        basis = helmert_basis(D)
    _validate_basis(basis, D)

    # z @ Ψ gives CLR coordinates; exponentiate and close
    clr_vals = z @ basis
    return closure(np.exp(clr_vals))


# =====================================================================
# CLR transform
# =====================================================================

def clr_forward(y, eps=1e-6):
    """
    CLR (Centred Log-Ratio) forward transform.

    Parameters
    ----------
    y : ndarray, shape (N, D)
    eps : float

    Returns
    -------
    c : ndarray, shape (N, D)
        CLR coordinates (rows sum to 0).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    y_s = _smooth_and_close(y, eps)
    log_y = np.log(y_s)
    return log_y - log_y.mean(axis=1, keepdims=True)


def clr_inverse(c):
    """
    CLR inverse transform.

    Parameters
    ----------
    c : ndarray, shape (N, D)

    Returns
    -------
    y : ndarray, shape (N, D)
    """
    return closure(np.exp(np.asarray(c, dtype=np.float64)))


# =====================================================================
# ALR transform
# =====================================================================

def alr_forward(y, ref=-1, eps=1e-6):
    """
    ALR (Additive Log-Ratio) forward transform.

    Parameters
    ----------
    y : ndarray, shape (N, D)
    ref : int
        Index of the reference part (denominator). Default: last.
    eps : float

    Returns
    -------
    z : ndarray, shape (N, D-1)
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    y_s = _smooth_and_close(y, eps)
    D = y_s.shape[1]
    idx = [i for i in range(D) if i != (ref % D)]
    return np.log(y_s[:, idx] / y_s[:, ref % D : (ref % D) + 1])


def alr_inverse(z, ref=-1):
    """
    ALR inverse transform.

    Parameters
    ----------
    z : ndarray, shape (N, D-1)
    ref : int
        Index of the reference part in the output.

    Returns
    -------
    y : ndarray, shape (N, D)
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    N, Dm1 = z.shape
    D = Dm1 + 1
    ref_idx = ref % D

    exp_z = np.exp(z)
    denom = 1.0 + exp_z.sum(axis=1, keepdims=True)

    y = np.empty((N, D), dtype=np.float64)
    j = 0
    for i in range(D):
        if i == ref_idx:
            y[:, i] = 1.0 / denom.ravel()
        else:
            y[:, i] = exp_z[:, j] / denom.ravel()
            j += 1

    return y


# =====================================================================
# Aitchison distance
# =====================================================================

def aitchison_distance(y1, y2, eps=1e-6):
    """
    Aitchison distance between two compositions (or arrays of compositions).

    Defined as the Euclidean distance in CLR (or equivalently ILR) space.
    Using CLR is simpler since it avoids basis construction.

    Parameters
    ----------
    y1, y2 : ndarray, shape (N, D) or (D,)
    eps : float

    Returns
    -------
    d : ndarray, shape (N,)
        Per-row Aitchison distance.
    """
    c1 = clr_forward(y1, eps)
    c2 = clr_forward(y2, eps)
    return np.sqrt(((c1 - c2) ** 2).sum(axis=1))
