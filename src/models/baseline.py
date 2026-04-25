"""Majority-class baseline — always predicts the most frequent class."""

import numpy as np


class MajorityClassifier:
    """Predicts the majority class seen during training at every test point.

    Serves as the absolute performance floor: zero parameters, zero learning.
    """

    def __init__(self):
        self.majority_class = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store the most frequent class label."""
        self.majority_class = int(np.bincount(y).argmax())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return an array filled with the majority class."""
        return np.full(len(X), self.majority_class, dtype=np.int64)
