"""Logistic regression baseline with class-weighted training."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticModel:
    """Thin wrapper around sklearn LogisticRegression with balanced class weights.

    Trained on single LOB snapshots (40 raw features). No temporal structure.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self._model = LogisticRegression(
            class_weight="balanced",
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on snapshot features and integer class labels (0/1/2)."""
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix of shape (N, 3)."""
        return self._model.predict_proba(X)
