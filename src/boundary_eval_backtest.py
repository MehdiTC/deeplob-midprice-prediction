"""Boundary-filtered checkpoint evaluation and robust proxy backtests.

This script uses the saved LSTM and DeepLOB checkpoints; it does not retrain models.
It removes sequence windows that cross inferred FI-2010 stock boundaries, evaluates
classification metrics on the remaining windows, and runs backtest sensitivity checks
for DeepLOB predictions.

AI attribution: this file contains AI-assisted implementation reviewed, debugged,
and validated by the author; see ATTRIBUTION.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.backtest import compute_tstat, extract_mid_prices, run_backtest
from src.data_loader import (
    extract_features_labels,
    find_stock_boundaries,
    load_raw,
    valid_window_starts,
)
from src.evaluate import HORIZONS, compute_metrics
from src.models.deeplob import DeepLOB
from src.models.lstm_baseline import SimpleLSTM
from src.train import DEVICE, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
WINDOW = 100
BATCH_SIZE = 512


def filtered_boundaries_from_starts(starts: np.ndarray, row_boundaries: np.ndarray) -> list[int]:
    """Convert raw row boundaries into indices for a filtered window/mid-price series."""
    boundaries = [0]
    for row_boundary in row_boundaries:
        idx = int(np.searchsorted(starts, row_boundary, side="left"))
        if idx != boundaries[-1]:
            boundaries.append(idx)
    if boundaries[-1] != len(starts):
        boundaries.append(len(starts))
    return boundaries


def predict_logits(
    model: torch.nn.Module,
    X: np.ndarray,
    starts: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run batched checkpoint inference for a sequence model."""
    model.eval().to(device)
    logits = []
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch = np.stack([X[i : i + WINDOW] for i in batch_starts]).astype(np.float32)
            X_batch = torch.from_numpy(batch).to(device)
            logits.append(model(X_batch).cpu().numpy())
    return np.concatenate(logits, axis=0)


def apply_confidence_threshold(preds: np.ndarray, probs: np.ndarray, threshold: float | None) -> np.ndarray:
    """Map low-confidence directional signals to stationary for robustness checks."""
    if threshold is None:
        return preds
    thresholded = preds.copy()
    directional = thresholded != 1
    low_conf = probs.max(axis=1) < threshold
    thresholded[directional & low_conf] = 1
    return thresholded


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df_test = load_raw(str(DATA / "FI2010_test.csv"))
    X_test, Y_test = extract_features_labels(df_test)
    row_boundaries = find_stock_boundaries(X_test)
    starts_filtered = valid_window_starts(
        len(X_test),
        window_size=WINDOW,
        boundaries=row_boundaries,
    )
    starts_unfiltered = valid_window_starts(len(X_test), window_size=WINDOW)

    mid_all = extract_mid_prices(X_test)
    mid_filtered = mid_all[starts_filtered + WINDOW - 1]
    bt_boundaries = filtered_boundaries_from_starts(starts_filtered, row_boundaries)

    summary = pd.DataFrame(
        [
            {
                "split": "test",
                "rows": len(X_test),
                "window_size": WINDOW,
                "inferred_row_boundaries": row_boundaries.tolist(),
                "unfiltered_windows": len(starts_unfiltered),
                "boundary_filtered_windows": len(starts_filtered),
                "removed_cross_boundary_windows": len(starts_unfiltered) - len(starts_filtered),
                "filtered_series_boundaries": bt_boundaries,
            }
        ]
    )
    summary.to_csv(RESULTS / "boundary_filter_summary.csv", index=False)

    metric_rows = []
    backtest_rows = []
    cumulative_base = {}

    model_specs = {
        "SimpleLSTM": (SimpleLSTM, "lstm"),
        "DeepLOB": (DeepLOB, "deeplob"),
    }
    scenarios = [
        {
            "scenario": "paper_proxy",
            "entry_slippage": 5,
            "exit_slippage": 0,
            "transaction_cost": 0.0,
            "confidence_threshold": None,
        },
        {
            "scenario": "delayed_exit_5",
            "entry_slippage": 5,
            "exit_slippage": 5,
            "transaction_cost": 0.0,
            "confidence_threshold": None,
        },
        {
            "scenario": "cost_1e-5",
            "entry_slippage": 5,
            "exit_slippage": 0,
            "transaction_cost": 1e-5,
            "confidence_threshold": None,
        },
        {
            "scenario": "cost_5e-5",
            "entry_slippage": 5,
            "exit_slippage": 0,
            "transaction_cost": 5e-5,
            "confidence_threshold": None,
        },
        {
            "scenario": "conf_threshold_0.50",
            "entry_slippage": 5,
            "exit_slippage": 0,
            "transaction_cost": 0.0,
            "confidence_threshold": 0.50,
        },
    ]

    for model_name, (model_cls, checkpoint_prefix) in model_specs.items():
        for h_idx, horizon in enumerate(HORIZONS):
            checkpoint = MODELS / f"{checkpoint_prefix}_k{horizon}.pt"
            model = load_checkpoint(model_cls(), str(checkpoint))
            logits = predict_logits(model, X_test, starts_filtered, DEVICE)
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            preds = probs.argmax(axis=1)
            y_true = Y_test[starts_filtered + WINDOW - 1, h_idx]

            metrics = compute_metrics(y_true, preds)
            metric_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "n_windows": len(y_true),
                    "weighted_f1": metrics["weighted_f1"],
                    "accuracy": metrics["accuracy"],
                    "cohen_kappa": metrics["cohen_kappa"],
                    "down_f1": metrics["per_class_f1"][0],
                    "stationary_f1": metrics["per_class_f1"][1],
                    "up_f1": metrics["per_class_f1"][2],
                }
            )

            if model_name != "DeepLOB":
                continue

            for scenario in scenarios:
                bt_preds = apply_confidence_threshold(
                    preds,
                    probs,
                    scenario["confidence_threshold"],
                )
                bt = run_backtest(
                    bt_preds,
                    mid_filtered,
                    boundaries=bt_boundaries,
                    slippage_steps=scenario["entry_slippage"],
                    exit_slippage_steps=scenario["exit_slippage"],
                    transaction_cost=scenario["transaction_cost"],
                )
                tstat = compute_tstat(bt["segment_pnl"])
                backtest_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "scenario": scenario["scenario"],
                        "entry_slippage": scenario["entry_slippage"],
                        "exit_slippage": scenario["exit_slippage"],
                        "transaction_cost": scenario["transaction_cost"],
                        "confidence_threshold": scenario["confidence_threshold"],
                        "segment_pnl_mean": float(np.mean(bt["segment_pnl"])),
                        "total_pnl": float(bt["cumulative_pnl"][-1]),
                        "t_statistic": tstat,
                        "n_trades": bt["n_trades"],
                        "mean_trade_pnl": float(np.mean(bt["trade_pnl"])) if bt["trade_pnl"] else 0.0,
                    }
                )
                if scenario["scenario"] == "paper_proxy":
                    cumulative_base[horizon] = {
                        "cumulative_pnl": bt["cumulative_pnl"],
                        "tstat": tstat,
                    }

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS / "boundary_filtered_metrics.csv", index=False)

    backtest_df = pd.DataFrame(backtest_rows)
    backtest_df.to_csv(RESULTS / "backtest_robustness.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(HORIZONS)))
    for horizon, color in zip(HORIZONS, colors):
        bt = cumulative_base[horizon]
        axes[0].plot(
            bt["cumulative_pnl"],
            label=f"k={horizon} (t={bt['tstat']:.2f})",
            color=color,
            lw=1.5,
        )
    for boundary in bt_boundaries[1:-1]:
        axes[0].axvline(boundary, color="0.85", lw=0.7, linestyle=":")
    axes[0].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[0].set_title("DeepLOB boundary-filtered proxy P&L")
    axes[0].set_xlabel("Filtered window index")
    axes[0].set_ylabel("Cumulative proxy P&L")
    axes[0].legend(fontsize=8)

    pivot = backtest_df.pivot(index="horizon", columns="scenario", values="total_pnl")
    pivot = pivot[[scenario["scenario"] for scenario in scenarios]]
    x = np.arange(len(pivot.index))
    width = 0.15
    for i, scenario in enumerate(pivot.columns):
        axes[1].bar(x + (i - 2) * width, pivot[scenario], width=width, label=scenario)
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1].set_title("DeepLOB backtest robustness")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"k={h}" for h in pivot.index])
    axes[1].set_ylabel("Total proxy P&L")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "backtest_robustness.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    print("Boundary filter summary")
    print(summary.to_string(index=False))
    print("\nBoundary-filtered weighted F1")
    print(
        metrics_df.pivot(index="model", columns="horizon", values="weighted_f1")
        .round(4)
        .to_string()
    )
    print("\nDeepLOB backtest robustness: total proxy P&L")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
