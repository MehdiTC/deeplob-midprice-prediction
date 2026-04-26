"""Backtest-style proxy simulation for FI-2010 DeepLOB predictions.

Rules:
- Signal 2 (up)   → buy 1 share at t+5 (slippage), hold until signal 0 (down) appears → sell
- Signal 0 (down) → short 1 share at t+5, hold until signal 2 (up) appears → cover
- Signal 1 (stat) → do nothing
- Open positions are force-closed at supplied segment boundaries.
- Transaction costs default to zero; mid-price execution throughout.

FI-2010 features are z-score normalized, so the resulting P&L is in normalized mid-price
units. The public CSV does not provide reliable day identifiers after concatenation, so
the default helper detects contiguous stock/segment boundaries and avoids carrying a
position across them.
"""

import numpy as np
from scipy import stats


DEFAULT_N_SEGMENTS = 5


def extract_mid_prices(X_raw: np.ndarray) -> np.ndarray:
    """Compute mid-price from LOB snapshot matrix.

    Column 0 = best ask price (p_ask_1), column 2 = best bid price (p_bid_1).

    Args:
        X_raw: float32 array of shape (N, 40)

    Returns:
        float32 array of shape (N,)
    """
    return (X_raw[:, 0] + X_raw[:, 2]) / 2.0


def infer_segment_boundaries(mid_prices: np.ndarray, n_segments: int = DEFAULT_N_SEGMENTS) -> list[int]:
    """Infer contiguous stock/segment boundaries from jumps in normalized mid-price."""
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1")
    if len(mid_prices) == 0:
        return [0]
    if n_segments == 1 or len(mid_prices) == 1:
        return [0, len(mid_prices)]

    n_cuts = min(n_segments - 1, len(mid_prices) - 1)
    diffs = np.abs(np.diff(mid_prices))
    cuts = np.argpartition(diffs, -n_cuts)[-n_cuts:] + 1
    return [0, *np.sort(cuts).astype(int).tolist(), len(mid_prices)]


def run_backtest(
    predictions: np.ndarray,
    mid_prices: np.ndarray,
    boundaries: list[int] | None = None,
    slippage_steps: int = 5,
    exit_slippage_steps: int = 0,
    transaction_cost: float = 0.0,
) -> dict:
    """Simulate the trading rule on model predictions.

    Args:
        predictions:     integer array of shape (N,) with values 0/1/2
        mid_prices:      float array of shape (N,) — normalized mid-price per window
        boundaries:      segment start/end indices; inferred from mid-price jumps if omitted
        slippage_steps:  buy/sell executed this many steps after signal (default 5)
        exit_slippage_steps: close/cover executed this many steps after reversal signal
        transaction_cost: normalized-price cost per side, subtracted on entry and exit

    Returns dict with:
        segment_pnl:     list of proxy P&L per contiguous segment
        cumulative_pnl:  ndarray of shape (N,) cumulative PnL per step
        trade_pnl:       list of completed round-trip P&L values after costs
        n_trades:        total number of completed round-trips
        boundaries:      segment boundaries used by the simulation
    """
    N = len(predictions)
    if len(mid_prices) != N:
        raise ValueError(f"predictions and mid_prices must have same length, got {N} and {len(mid_prices)}")

    if boundaries is None:
        boundaries = infer_segment_boundaries(mid_prices)
    if boundaries[0] != 0 or boundaries[-1] != N:
        raise ValueError("boundaries must start at 0 and end at len(predictions)")

    pnl_per_step = np.zeros(N, dtype=np.float64)
    trade_pnl = []
    n_trades = 0
    round_trip_cost = 2.0 * transaction_cost

    for segment_idx in range(len(boundaries) - 1):
        start, end = boundaries[segment_idx], boundaries[segment_idx + 1]
        position = 0          # +1 = long, -1 = short, 0 = flat
        entry_price = 0.0

        i = start
        while i < end:
            sig = predictions[i]

            if position == 0:
                if sig == 2:                          # up signal → go long
                    exec_i = min(i + slippage_steps, end - 1)
                    entry_price = mid_prices[exec_i]
                    position = 1
                    i = exec_i + 1
                    continue
                elif sig == 0:                        # down signal → go short
                    exec_i = min(i + slippage_steps, end - 1)
                    entry_price = mid_prices[exec_i]
                    position = -1
                    i = exec_i + 1
                    continue
            elif position == 1 and sig == 0:         # close long on down signal
                exec_i = min(i + exit_slippage_steps, end - 1)
                exit_price = mid_prices[exec_i]
                pnl = exit_price - entry_price - round_trip_cost
                pnl_per_step[exec_i] += pnl
                trade_pnl.append(float(pnl))
                position = 0
                n_trades += 1
                i = exec_i + 1
                continue
            elif position == -1 and sig == 2:        # close short on up signal
                exec_i = min(i + exit_slippage_steps, end - 1)
                exit_price = mid_prices[exec_i]
                pnl = entry_price - exit_price - round_trip_cost
                pnl_per_step[exec_i] += pnl
                trade_pnl.append(float(pnl))
                position = 0
                n_trades += 1
                i = exec_i + 1
                continue

            i += 1

        # Force-close any open position at the segment boundary.
        if position != 0:
            exit_price = mid_prices[end - 1]
            if position == 1:
                pnl = exit_price - entry_price - round_trip_cost
            else:
                pnl = entry_price - exit_price - round_trip_cost
            pnl_per_step[end - 1] += pnl
            trade_pnl.append(float(pnl))
            position = 0
            n_trades += 1

    cumulative_pnl = np.cumsum(pnl_per_step)
    segment_pnl = [
        float(pnl_per_step[boundaries[d] : boundaries[d + 1]].sum())
        for d in range(len(boundaries) - 1)
    ]

    return {
        "segment_pnl": segment_pnl,
        "cumulative_pnl": cumulative_pnl,
        "trade_pnl": trade_pnl,
        "n_trades": n_trades,
        "boundaries": boundaries,
    }


def compute_tstat(segment_pnl: list) -> float:
    """Return t-statistic testing whether mean segment P&L is significantly > 0."""
    result = stats.ttest_1samp(segment_pnl, 0.0)
    return float(result.statistic)
