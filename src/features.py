"""Engineered features derived from the 40 raw LOB columns."""

import numpy as np


# Horizon index → k value mapping (for documentation)
HORIZONS = [1, 2, 3, 5, 10]

# Column layout within X (40 features):
# Level i occupies columns 4i, 4i+1, 4i+2, 4i+3 = [p_ask, v_ask, p_bid, v_bid]
_N_LEVELS = 10


def add_engineered_features(X: np.ndarray) -> np.ndarray:
    """Append 13 engineered features to the 40 raw LOB features.

    Appended features (columns 40–52):
        cols 40–49: order-book imbalance at each of the 10 levels
        col  50:    bid-ask spread (best ask − best bid)
        col  51:    imbalance-weighted mid-price at level 1
        col  52:    arithmetic mid-price at level 1

    Args:
        X: float32 array of shape (N, 40)

    Returns:
        float32 array of shape (N, 53)
    """
    N = len(X)
    extras = np.empty((N, 13), dtype=np.float32)

    for i in range(_N_LEVELS):
        p_ask = X[:, 4 * i]
        v_ask = X[:, 4 * i + 1]
        p_bid = X[:, 4 * i + 2]
        v_bid = X[:, 4 * i + 3]
        extras[:, i] = (v_bid - v_ask) / (v_bid + v_ask + 1e-8)

    p_ask1, v_ask1 = X[:, 0], X[:, 1]
    p_bid1, v_bid1 = X[:, 2], X[:, 3]

    extras[:, 10] = p_ask1 - p_bid1                              # spread

    imbalance = v_bid1 / (v_ask1 + v_bid1 + 1e-8)
    extras[:, 11] = imbalance * p_ask1 + (1.0 - imbalance) * p_bid1  # weighted mid

    extras[:, 12] = (p_ask1 + p_bid1) / 2.0                     # arithmetic mid

    return np.concatenate([X, extras], axis=1)
