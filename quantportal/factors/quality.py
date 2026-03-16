"""
Quality Factor — price-based quality/stability signals

★ What is "Quality" without fundamentals?
  Without P/E ratios and balance sheets, we can still measure quality
  using price-based proxies:

  1. Return stability (Sharpe-like): consistent positive returns
  2. Drawdown resilience: how quickly does the asset recover?
  3. Tail risk: how fat are the tails (excess kurtosis)?
  4. Autocorrelation: predictable momentum vs random walk

  These are "statistical quality" signals — they capture assets that
  exhibit stable, reliable behavior vs erratic, risky behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualitySignal:
    """Cross-sectional statistical quality signals"""

    rolling_sharpe: pd.DataFrame     # Rolling Sharpe ratio
    drawdown_score: pd.DataFrame     # Inverse of max drawdown (higher = better)
    tail_risk: pd.DataFrame          # Excess kurtosis (lower = safer)
    stability_score: pd.DataFrame    # Combined quality z-score


def compute_quality_signals(
    returns: pd.DataFrame,
    window: int = 63,
) -> QualitySignal:
    """Compute cross-sectional quality signals.

    Parameters
    ----------
    window : int
        Rolling window (63 ≈ 3 months).
    """
    # Rolling Sharpe
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan)

    # Drawdown score: inverse of current drawdown depth
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    rolling_dd = drawdown.rolling(window).min()  # worst DD in window
    drawdown_score = -rolling_dd  # negate so higher = better

    # Tail risk: rolling excess kurtosis
    tail_risk = returns.rolling(window).kurt()

    # Combined stability score: z-score of (sharpe + dd_score - tail_risk)
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)

    stability = (
        _cs_zscore(rolling_sharpe).fillna(0)
        + _cs_zscore(drawdown_score).fillna(0)
        - _cs_zscore(tail_risk).fillna(0)
    ) / 3.0

    return QualitySignal(
        rolling_sharpe=rolling_sharpe,
        drawdown_score=drawdown_score,
        tail_risk=tail_risk,
        stability_score=stability,
    )
