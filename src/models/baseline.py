"""
Phase 8: Baseline model (predict training-set mean).

Provides a "dumb" baseline that always predicts the training mean
proportions. Essential for contextualizing all other scores — if a
model can't beat mean prediction, it's useless.
"""

import numpy as np

from src.transforms import ilr_inverse


class DummyBaseline:
    """
    Always predicts the training-set mean proportions.

    Works in ILR space for interface compatibility, but `predict_proportions`
    returns the constant mean vector for every input.
    """

    def __init__(self, basis=None):
        self.basis = basis
        self.mean_z = None
        self.mean_y = None

    def fit(self, X_train, z_train):
        """Store the training-set mean of ILR targets."""
        self.mean_z = z_train.mean(axis=0)
        self.mean_y = ilr_inverse(
            self.mean_z.reshape(1, -1), basis=self.basis
        ).squeeze(0)
        return self

    def predict_ilr(self, X):
        """Return constant mean ILR vector for all inputs."""
        return np.tile(self.mean_z, (X.shape[0], 1))

    def predict_proportions(self, X):
        """Return constant mean proportions for all inputs."""
        return np.tile(self.mean_y, (X.shape[0], 1))

    def get_params_dict(self):
        return {"model": "DummyBaseline"}
