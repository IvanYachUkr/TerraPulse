"""
Phase 8: Linear model wrappers (Ridge, ElasticNet).

Both models train on ILR-transformed targets and invert predictions
back to the proportion simplex.
"""

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.transforms import ilr_inverse


class RidgeModel:
    """
    Multi-output Ridge regression in ILR space.

    sklearn Ridge natively supports multi-output, so no wrapper needed.
    """

    def __init__(self, alpha=1.0, basis=None):
        self.alpha = alpha
        self.basis = basis
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, z_train):
        """Fit on ILR-transformed targets z_train (N, D-1)."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, z_train)
        return self

    def predict_ilr(self, X):
        """Predict in ILR space."""
        return self.model.predict(self.scaler.transform(X))

    def predict_proportions(self, X):
        """Predict and invert to simplex proportions."""
        z_pred = self.predict_ilr(X)
        return ilr_inverse(z_pred, basis=self.basis)

    def get_params_dict(self):
        return {"model": "Ridge", "alpha": self.alpha}

    @property
    def coef_(self):
        """Coefficient matrix (D-1, n_features)."""
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_


class ElasticNetModel:
    """
    Multi-output ElasticNet in ILR space.

    Uses MultiOutputRegressor since sklearn ElasticNet does not
    natively support multi-output.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=5000, basis=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.basis = basis
        self.scaler = StandardScaler()
        self.model = MultiOutputRegressor(
            ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                random_state=42,
            )
        )

    def fit(self, X_train, z_train):
        """Fit on ILR-transformed targets z_train (N, D-1)."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, z_train)
        return self

    def predict_ilr(self, X):
        """Predict in ILR space."""
        return self.model.predict(self.scaler.transform(X))

    def predict_proportions(self, X):
        """Predict and invert to simplex proportions."""
        z_pred = self.predict_ilr(X)
        return ilr_inverse(z_pred, basis=self.basis)

    def get_params_dict(self):
        return {
            "model": "ElasticNet",
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
        }

    @property
    def coef_(self):
        """Coefficient matrix (D-1, n_features) stacked from per-output models."""
        return np.array([est.coef_ for est in self.model.estimators_])

    @property
    def n_nonzero(self):
        """Number of non-zero coefficients per ILR coordinate."""
        return [(np.abs(est.coef_) > 1e-10).sum() for est in self.model.estimators_]
