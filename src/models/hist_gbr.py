"""
Phase 8: HistGradientBoosting model wrapper.

Pure-sklearn boosting fallback using HistGradientBoostingRegressor +
MultiOutputRegressor. Stays entirely within the sklearn ecosystem,
avoiding external dependencies like CatBoost.

Trains on ILR-transformed targets, same as other tree models.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.transforms import ilr_inverse


class HistGBRModel:
    """
    Multi-output HistGradientBoosting regression in ILR space.

    Uses sklearn's HistGradientBoostingRegressor (very fast for
    medium-sized tabular data) wrapped in MultiOutputRegressor
    to handle the multi-output ILR targets.
    """

    def __init__(self, max_iter=500, max_depth=6, learning_rate=0.1,
                 min_samples_leaf=20, max_leaf_nodes=31,
                 l2_regularization=0.0, random_state=42,
                 n_jobs=-1, basis=None):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.basis = basis
        self.scaler = StandardScaler()

        base_est = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.15,
        )
        self.model = MultiOutputRegressor(base_est, n_jobs=n_jobs)

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
            "model": "HistGBR",
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
        }
