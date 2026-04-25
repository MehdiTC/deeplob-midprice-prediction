"""Trading backtest simulation matching the setup in Zhang et al. (2019) Section V.D.

Rules:
- Signal 2 (up)   → buy 1 share at t+5 (slippage), hold until signal 0 (down) appears → sell
- Signal 0 (down) → short 1 share at t+5, hold until signal 2 (up) appears → cover
- Signal 1 (stat) → do nothing
- All open positions are closed at the end of each trading day at the prevailing mid-price.
- No transaction costs; mid-price execution throughout.
"""

import numpy as np
from scipy import stats


# Test set: 31,837 windows across 3 trading days (equal split: ~10,612 per day)
DEFAULT_DAY_BOUNDARIES = [0, 10612, 21224, 31837]


def extract_mid_prices(X_raw: np.ndarray) -> np.ndarray:
    """Compute mid-price from LOB snapshot matrix.

    Column 0 = best ask price (p_ask_1), column 2 = best bid price (p_bid_1).

    Args:
        X_raw: float32 array of shape (N, 40)

    Returns:
        float32 array of shape (N,)
    """
    return (X_raw[:, 0] + X_raw[:, 2]) / 2.0


def run_backtest(
    predictions: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: list = DEFAULT_DAY_BOUNDARIES,
    slippage_steps: int = 5,
) -> dict:
    """Simulate the paper's trading strategy on model predictions.

    Args:
        predictions:     integer array of shape (N,) with values 0/1/2
        mid_prices:      float array of shape (N,) — mid-price per window
        day_boundaries:  list of window indices marking day start/end (inclusive)
                         e.g. [0, 10612, 21224, 31837] for 3 days
        slippage_steps:  buy/sell executed this many steps after signal (default 5)

    Returns dict with:
        daily_pnl:       list of PnL per day
        cumulative_pnl:  ndarray of shape (N,) cumulative PnL per step
        n_trades:        total number of completed round-trips
    """
    N = len(predictions)
    pnl_per_step = np.zeros(N, dtype=np.float64)
    n_trades = 0

    for day_idx in range(len(day_boundaries) - 1):
        start, end = day_boundaries[day_idx], day_boundaries[day_idx + 1]
        position = 0          # +1 = long, -1 = short, 0 = flat
        entry_price = 0.0
        entry_step = -1

        i = start
        while i < end:
            sig = predictions[i]

            if position == 0:
                if sig == 2:                          # up signal → go long
                    exec_i = min(i + slippage_steps, end - 1)
                    entry_price = mid_prices[exec_i]
                    position = 1
                    entry_step = exec_i
                    i = exec_i + 1
                    continue
                elif sig == 0:                        # down signal → go short
                    exec_i = min(i + slippage_steps, end - 1)
                    entry_price = mid_prices[exec_i]
                    position = -1
                    entry_step = exec_i
                    i = exec_i + 1
                    continue
            elif position == 1 and sig == 0:         # close long on down signal
                exit_price = mid_prices[i]
                trade_pnl = exit_price - entry_price
                pnl_per_step[i] += trade_pnl
                position = 0
                n_trades += 1
            elif position == -1 and sig == 2:        # close short on up signal
                exit_price = mid_prices[i]
                trade_pnl = entry_price - exit_price
                pnl_per_step[i] += trade_pnl
                position = 0
                n_trades += 1

            i += 1

        # Force-close any open position at end of day
        if position != 0:
            exit_price = mid_prices[end - 1]
            if position == 1:
                trade_pnl = exit_price - entry_price
            else:
                trade_pnl = entry_price - exit_price
            pnl_per_step[end - 1] += trade_pnl
            position = 0
            n_trades += 1

    cumulative_pnl = np.cumsum(pnl_per_step)
    daily_pnl = [
        float(pnl_per_step[day_boundaries[d] : day_boundaries[d + 1]].sum())
        for d in range(len(day_boundaries) - 1)
    ]

    return {
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "n_trades": n_trades,
    }


def compute_tstat(daily_pnl: list) -> float:
    """Return t-statistic testing whether mean daily PnL is significantly > 0."""
    result = stats.ttest_1samp(daily_pnl, 0.0)
    return float(result.statistic)
