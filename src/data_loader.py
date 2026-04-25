"""Data loading, splitting, and PyTorch Dataset/DataLoader utilities for FI-2010."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_raw(path: str) -> pd.DataFrame:
    """Read a FI-2010 CSV and return a DataFrame with 149 feature/label columns.

    The CSV has an unnamed index column as column 0; we drop it via index_col=0.
    Resulting DataFrame has columns named 0–148 (149 columns total).
    """
    df = pd.read_csv(path, index_col=0)
    assert df.shape[1] == 149, f"Expected 149 columns after dropping index, got {df.shape[1]}"
    return df


def extract_features_labels(df: pd.DataFrame):
    """Extract the 40 LOB feature columns and 5 label columns from a raw DataFrame.

    Features: df.iloc[:, 0:40]  — 40 z-score normalised LOB prices/volumes
    Labels:   df.iloc[:, -5:]   — 5 horizons k=1,2,3,5,10; raw values 1/2/3 shifted to 0/1/2

    Returns:
        X: float32 ndarray of shape (N, 40)
        Y: int64 ndarray of shape (N, 5)
    """
    X = df.iloc[:, 0:40].values.astype(np.float32)
    Y = (df.iloc[:, -5:].values - 1).astype(np.int64)
    return X, Y


def temporal_split(X: np.ndarray, Y: np.ndarray, val_ratio: float = 0.15):
    """Split arrays temporally (no shuffling) into train and validation portions.

    Returns:
        (X_train, Y_train, X_val, Y_val)
    """
    cut = int(len(X) * (1.0 - val_ratio))
    return X[:cut], Y[:cut], X[cut:], Y[cut:]


def find_stock_boundaries(X: np.ndarray, n_stocks: int = 5) -> np.ndarray:
    """Detect stock boundaries in a concatenated FI-2010 array via mid-price jump detection.

    Finds the n_stocks-1 largest absolute differences between consecutive mid-prices and
    returns their row indices (where each new stock begins), sorted ascending.
    """
    mid = (X[:, 0] + X[:, 2]) / 2
    diffs = np.abs(np.diff(mid))
    top_idx = np.argpartition(diffs, -(n_stocks - 1))[-(n_stocks - 1):]
    return np.sort(top_idx + 1).astype(np.int64)


def make_windows(
    X: np.ndarray,
    Y: np.ndarray,
    horizon_idx: int,
    window_size: int = 100,
    boundaries: np.ndarray = None,
) -> tuple:
    """Build sliding windows of LOB snapshots, skipping any window that crosses a stock boundary.

    Returns (windows, labels): float32 array of shape (N, window_size, 40) and int64 array
    of shape (N,).
    """
    boundary_set = set(boundaries.tolist()) if boundaries is not None else set()
    windows, labels = [], []
    for i in range(len(X) - window_size + 1):
        if any(i <= b < i + window_size for b in boundary_set):
            continue
        windows.append(X[i : i + window_size])
        labels.append(Y[i + window_size - 1, horizon_idx])
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


class SlidingWindowDataset(Dataset):
    """Yields (window, label) pairs for sequence models (LSTM, DeepLOB).

    Each sample is a window of `window` consecutive LOB snapshots and the label
    at the last position of the window for a chosen prediction horizon.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, window: int = 100, horizon_idx: int = 0):
        """
        Args:
            X: float32 array of shape (N, 40)
            Y: int64 array of shape (N, 5)
            window: number of consecutive snapshots per sample
            horizon_idx: which of the 5 horizons (0–4 → k=1,2,3,5,10) to use as label
        """
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y[:, horizon_idx])
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, i):
        return self.X[i : i + self.window], self.Y[i + self.window - 1]


class SnapshotDataset(Dataset):
    """Yields (snapshot, label) pairs for single-step models (LR, XGBoost placeholder).

    Each sample is one LOB snapshot and the label for a chosen prediction horizon.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, horizon_idx: int = 0):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y[:, horizon_idx])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def make_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    """Wrap a Dataset in a DataLoader with sensible defaults."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)
