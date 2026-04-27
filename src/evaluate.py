"""Evaluation metrics, throughput measurement, and results-table builder.

AI attribution: this file contains AI-assisted implementation reviewed, debugged,
and validated by the author; see ATTRIBUTION.md.
"""

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

HORIZONS = [1, 2, 3, 5, 10]
CLASS_NAMES = ["down", "stationary", "up"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a full set of classification metrics.

    Returns a dict with keys:
        weighted_f1, accuracy, cohen_kappa,
        per_class_precision, per_class_recall, per_class_f1  (each a list of 3)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": float(np.mean(y_true == y_pred)),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }


def measure_throughput(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    batch_size: int = 512,
    n_warmup: int = 3,
    n_timed: int = 10,
) -> float:
    """Return predictions-per-second for a callable predict_fn.

    Args:
        predict_fn: accepts an ndarray batch and returns predictions
        X: dataset to sample batches from
        batch_size: rows per batch
        n_warmup: warm-up passes (not timed)
        n_timed: timed passes to average over
    """
    batch = X[:batch_size]
    for _ in range(n_warmup):
        predict_fn(batch)

    t0 = time.perf_counter()
    for _ in range(n_timed):
        predict_fn(batch)
    elapsed = time.perf_counter() - t0

    return (batch_size * n_timed) / elapsed


def build_results_table(
    model_results: dict[str, list[dict]],
    horizons: list[int] = HORIZONS,
    metric: str = "weighted_f1",
) -> pd.DataFrame:
    """Build a models × horizons DataFrame of a single scalar metric.

    Args:
        model_results: {model_name: [metrics_dict_for_k1, ..., metrics_dict_for_k10]}
                       The list must align with `horizons` (same length).
        horizons: prediction horizons, used as column names.
        metric: key to extract from each metrics dict.

    Returns:
        DataFrame with model names as index and horizon values as columns.
    """
    rows = {}
    for model_name, results_list in model_results.items():
        rows[model_name] = {f"k={k}": r[metric] for k, r in zip(horizons, results_list)}
    return pd.DataFrame(rows).T


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = ""):
    """Print sklearn classification report with class names."""
    header = f"=== {model_name} ===" if model_name else "=== Results ==="
    print(header)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return confusion matrix with rows=true, cols=predicted."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
