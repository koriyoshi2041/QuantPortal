"""
Volatility Factor — realized vol, vol-of-vol, and low-vol anomaly signals

★ The Low-Volatility Anomaly:
  Low-vol stocks consistently outperform high-vol stocks on a risk-adjusted basis.
  This contradicts CAPM (higher risk ≠ higher return in practice).

  Why? Lottery preference: retail investors overpay for high-vol "lottery tickets",
  pushing up their prices and lowering future returns.

★ Signals provided:
  1. Realized volatility (rolling 20-day annualized)
  2. Vol-of-vol (instability of volatility itself)
  3. Low-vol rank score (cross-sectional: lower vol → higher score)
  4. Vol trend (is vol expanding or contracting?)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilitySignal:
    """Cross-sectional volatility signals"""

    realized_vol: pd.DataFrame       # Rolling annualized vol
    vol_of_vol: pd.DataFrame         # Std of rolling vol
    low_vol_score: pd.DataFrame      # CS rank: low vol → high score
    vol_trend: pd.DataFrame          # Vol expanding (+) or contracting (-)


def compute_volatility_signals(
    returns: pd.DataFrame,
    vol_window: int = 21,
    vov_window: int = 63,
) -> VolatilitySignal:
    """Compute cross-sectional volatility signals.

    Parameters
    ----------
    vol_window : int
        Rolling window for realized vol (21 ≈ 1 month).
    vov_window : int
        Rolling window for vol-of-vol (63 ≈ 3 months).
    """
    # Realized vol (annualized)
    realized_vol = returns.rolling(vol_window).std() * np.sqrt(252)

    # Vol-of-vol
    vol_of_vol = realized_vol.rolling(vov_window).std()

    # Low-vol score: cross-sectional rank (invert: low vol = high score)
    low_vol_score = realized_vol.rank(axis=1, pct=True, ascending=False)

    # Vol trend: current vol vs 3-month avg (positive = expanding)
    vol_ma = realized_vol.rolling(vov_window).mean()
    vol_trend = realized_vol - vol_ma

    return VolatilitySignal(
        realized_vol=realized_vol,
        vol_of_vol=vol_of_vol,
        low_vol_score=low_vol_score,
        vol_trend=vol_trend,
    )
