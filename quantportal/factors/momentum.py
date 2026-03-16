"""
Momentum Factor — cross-sectional momentum signals

★ What is momentum?
  Assets that performed well recently tend to continue performing well,
  and vice versa. This is one of the most robust anomalies in finance,
  documented by Jegadeesh & Titman (1993).

  Classic momentum: rank assets by past 12-month return (skip last month),
  go long top quintile, short bottom quintile.

★ Why does it work?
  Behavioral: investors underreact to new information (anchoring, slow diffusion)
  Risk-based: momentum stocks are riskier during crashes ("momentum crashes")

★ Implementation details:
  - Lookback: 12 months return, skip most recent month (to avoid reversal)
  - Z-score normalization: rank-based to avoid outlier sensitivity
  - Time-series momentum: also supported (asset's own past return)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumSignal:
    """Cross-sectional momentum signals"""

    # Signal matrices (date × ticker)
    raw_momentum: pd.DataFrame      # Raw past returns
    cs_zscore: pd.DataFrame         # Cross-sectional z-score (rank-based)
    ts_momentum: pd.DataFrame       # Time-series momentum (own past return)

    lookback: int
    skip: int


def compute_momentum(
    returns: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> MomentumSignal:
    """Compute cross-sectional and time-series momentum.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (date × ticker).
    lookback : int
        Lookback window in trading days (252 ≈ 12 months).
    skip : int
        Skip most recent N days (21 ≈ 1 month, avoids short-term reversal).
    """
    # Raw momentum: cumulative return over [t-lookback, t-skip]
    cum_ret = returns.rolling(lookback).sum()
    skip_ret = returns.rolling(skip).sum()
    raw_mom = cum_ret - skip_ret  # exclude recent skip days

    # Cross-sectional z-score (rank-based for robustness)
    def _rank_zscore(row: pd.Series) -> pd.Series:
        ranked = row.rank(pct=True)
        # Transform to approximate z-score: Φ^{-1}(rank)
        from scipy.stats import norm
        clipped = ranked.clip(0.01, 0.99)
        return pd.Series(norm.ppf(clipped), index=row.index)

    cs_z = raw_mom.apply(_rank_zscore, axis=1)

    # Time-series momentum (binary: own past return > 0)
    ts_mom = (raw_mom > 0).astype(float) * 2 - 1  # +1 or -1

    return MomentumSignal(
        raw_momentum=raw_mom,
        cs_zscore=cs_z,
        ts_momentum=ts_mom,
        lookback=lookback,
        skip=skip,
    )


def compute_short_term_reversal(
    returns: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """Short-term reversal: negative of past-week return.

    Assets that went up last week tend to reverse — this captures
    mean-reversion at the weekly frequency (contrarian signal).
    """
    past_ret = returns.rolling(lookback).sum()
    return -past_ret  # negate: high past return → negative signal
