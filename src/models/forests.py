"""
Phase 8: Tree ensemble model wrappers (ExtraTrees, RandomForest).

Both models train on ILR-transformed targets and invert predictions
back to the proportion simplex.
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.transforms import ilr_inverse


class ExtraTreesModel:
    """
    Multi-output Extra Trees regression in ILR space.

    ExtraTrees natively supports multi-output regression.
    """

    def __init__(self, n_estimators=500, max_features="sqrt",
                 min_samples_leaf=1, n_jobs=-1, random_state=42,
                 basis=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.basis = basis
        self.scaler = StandardScaler()
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
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
            "model": "ExtraTrees",
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
        }

    @property
    def feature_importances_(self):
        """Feature importance (mean decrease in impurity), shape (n_features,)."""
        return self.model.feature_importances_


class RandomForestModel:
    """
    Multi-output Random Forest regression in ILR space.
    """

    def __init__(self, n_estimators=500, max_features="sqrt",
                 min_samples_leaf=1, n_jobs=-1, random_state=42,
                 basis=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.basis = basis
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
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
            "model": "RandomForest",
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
        }

    @property
    def feature_importances_(self):
        """Feature importance (mean decrease in impurity), shape (n_features,)."""
        return self.model.feature_importances_
