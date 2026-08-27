# Copyright (c) 2026 PoseDeformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Linear pose-to-delta regressor with ridge regularization.

Maps the feature vector (F,) to per-vertex deltas (3V,) via a linear map
trained with the closed-form normal equations solution.
"""

from __future__ import annotations

import numpy as np


class LinearRegressor:
    """Feature vector (F,) -> per-vertex delta (3V,) linear map."""

    def __init__(self):
        self.matrix = None  # (F, 3V)

    def fit(self, X, D, regularization=1e-4):
        """X: (F, N) feature matrix, D: (3V, N) delta matrix."""
        X = np.asarray(X, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        n_features = X.shape[0]
        gram = X @ X.T
        gram.flat[:: n_features + 1] += float(regularization)
        self.matrix = np.linalg.solve(gram, X @ D.T)  # (F, 3V)
        return self.loss(X, D)

    def loss(self, X, D):
        return float(np.mean((self.predict(X) - D) ** 2))

    def predict(self, X):
        """X: (F, N) -> deltas (3V, N)."""
        if self.matrix is None:
            raise RuntimeError("LinearRegressor is not trained yet")
        return self.matrix.T @ np.asarray(X, dtype=np.float64)
