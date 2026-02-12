"""
Phase 8: CatBoost model wrapper.

Trains on ILR-transformed targets using MultiRMSE loss for true
multi-output gradient boosting.
"""

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler

from src.transforms import ilr_inverse


class CatBoostModel:
    """
    Multi-output CatBoost regression in ILR space.

    Uses MultiRMSE loss for joint prediction of all ILR coordinates.
    """

    def __init__(self, iterations=1000, depth=6, learning_rate=0.1,
                 l2_leaf_reg=3.0, random_strength=1.0,
                 random_seed=42, verbose=0, basis=None):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.random_seed = random_seed
        self.verbose = verbose
        self.basis = basis
        self.scaler = StandardScaler()
        self.model = CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            random_seed=random_seed,
            verbose=verbose,
        )

    def fit(self, X_train, z_train, X_val=None, z_val=None,
            early_stopping_rounds=50):
        """
        Fit on ILR-transformed targets.

        Parameters
        ----------
        X_train : ndarray (N, D)
        z_train : ndarray (N, C-1)
        X_val : ndarray, optional
            Validation features for early stopping.
        z_val : ndarray, optional
            Validation ILR targets for early stopping.
        early_stopping_rounds : int
        """
        X_scaled = self.scaler.fit_transform(X_train)
        train_pool = Pool(X_scaled, z_train)

        fit_kwargs = {}
        if X_val is not None and z_val is not None:
            eval_pool = Pool(self.scaler.transform(X_val), z_val)
            fit_kwargs["eval_set"] = eval_pool
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(train_pool, **fit_kwargs)
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
            "model": "CatBoost",
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_strength": self.random_strength,
            "best_iteration": getattr(self.model, "best_iteration_", None),
        }

    @property
    def feature_importances_(self):
        """Feature importance from CatBoost."""
        return self.model.get_feature_importance()
