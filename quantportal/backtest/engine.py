"""
Backtesting Engine — walk-forward portfolio backtesting

★ How it works:
  1. At each rebalance date (monthly):
     a. Compute factor signals from past data only
     b. Optimize portfolio weights
     c. Hold until next rebalance
  2. Track portfolio value, turnover, and transaction costs
  3. Report: Sharpe, max drawdown, turnover, sector exposure

★ Anti-overfitting measures:
  - Walk-forward: only use data available at each point in time
  - Transaction costs: 10 bps per leg (realistic for institutional)
  - Turnover tracking: excessive turnover destroys real-world returns
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Portfolio backtest results"""

    portfolio_value: pd.Series      # daily portfolio NAV
    daily_returns: pd.Series        # daily returns
    weights_history: pd.DataFrame   # date × ticker: weights at each rebalance
    turnover: pd.Series             # per-rebalance turnover

    # Performance metrics
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    calmar_ratio: float             # annual return / max drawdown
    avg_turnover: float             # average per-rebalance turnover
    n_rebalances: int


def _max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def backtest_portfolio(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    rebalance_freq: int = 21,
    top_n: int | None = None,
    long_only: bool = True,
    transaction_cost: float = 0.001,
    max_weight: float = 0.20,
) -> BacktestResult:
    """Run walk-forward portfolio backtest.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (date × ticker).
    signal : pd.DataFrame
        Signal scores (date × ticker). Higher = more desirable.
    rebalance_freq : int
        Rebalance every N trading days (21 ≈ monthly).
    top_n : int or None
        If set, only hold top N assets by signal strength.
    long_only : bool
        If True, no shorting.
    transaction_cost : float
        Round-trip transaction cost per unit turnover.
    max_weight : float
        Maximum weight per asset.
    """
    common = returns.index.intersection(signal.index)
    returns_aligned = returns.loc[common]
    signal_aligned = signal.loc[common]

    dates = returns_aligned.index
    tickers = returns_aligned.columns
    n_assets = len(tickers)

    nav = 1.0
    nav_series = []
    daily_ret_series = []
    weights_history = {}
    turnover_series = {}
    current_weights = pd.Series(0.0, index=tickers)

    for i, date in enumerate(dates):
        if i % rebalance_freq == 0 and i > 0:
            # Rebalance
            sig = signal_aligned.iloc[i]
            sig = sig.dropna()

            if len(sig) == 0:
                new_weights = pd.Series(0.0, index=tickers)
            else:
                if top_n is not None:
                    # Top-N equal weight
                    top_assets = sig.nlargest(min(top_n, len(sig))).index
                    new_weights = pd.Series(0.0, index=tickers)
                    w = 1.0 / len(top_assets)
                    for t in top_assets:
                        new_weights[t] = min(w, max_weight)
                    new_weights = new_weights / new_weights.sum()
                else:
                    # Signal-weighted (with constraints)
                    if long_only:
                        sig_pos = sig.clip(lower=0)
                    else:
                        sig_pos = sig

                    if sig_pos.abs().sum() > 0:
                        new_weights = pd.Series(0.0, index=tickers)
                        for t in sig.index:
                            if t in tickers:
                                new_weights[t] = sig_pos.get(t, 0.0)
                        total = new_weights.abs().sum()
                        if total > 0:
                            new_weights = new_weights / total
                        # Apply max weight constraint
                        new_weights = new_weights.clip(upper=max_weight)
                        total = new_weights.sum()
                        if total > 0:
                            new_weights = new_weights / total
                    else:
                        new_weights = pd.Series(0.0, index=tickers)

            # Transaction cost
            turnover = float((new_weights - current_weights).abs().sum())
            tc = turnover * transaction_cost
            nav *= (1 - tc)

            turnover_series[date] = turnover
            weights_history[date] = new_weights.copy()
            current_weights = new_weights.copy()

        # Daily return
        daily_ret = float((current_weights * returns_aligned.iloc[i]).sum())
        nav *= (1 + daily_ret)
        nav_series.append((date, nav))
        daily_ret_series.append((date, daily_ret))

    # Build result series
    nav_s = pd.Series(dict(nav_series), name="NAV")
    ret_s = pd.Series(dict(daily_ret_series), name="return")

    if len(weights_history) > 0:
        weights_df = pd.DataFrame(weights_history).T
    else:
        weights_df = pd.DataFrame()

    turnover_s = pd.Series(turnover_series, name="turnover")

    # Metrics
    total_ret = nav_s.iloc[-1] / nav_s.iloc[0] - 1 if len(nav_s) > 1 else 0.0
    n_years = len(ret_s) / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    annual_vol = float(ret_s.std() * np.sqrt(252))
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
    max_dd = _max_drawdown(nav_s)
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0.0
    avg_turnover = float(turnover_s.mean()) if len(turnover_s) > 0 else 0.0

    return BacktestResult(
        portfolio_value=nav_s,
        daily_returns=ret_s,
        weights_history=weights_df,
        turnover=turnover_s,
        total_return=total_ret,
        annual_return=annual_ret,
        annual_vol=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        avg_turnover=avg_turnover,
        n_rebalances=len(weights_history),
    )


def print_backtest(bt: BacktestResult) -> None:
    """Print backtest results"""

    print(f"\n{'='*55}")
    print(f"  Portfolio Backtest Results")
    print(f"{'='*55}")
    print()
    print(f"  Total Return:     {bt.total_return:+.1%}")
    print(f"  Annual Return:    {bt.annual_return:+.1%}")
    print(f"  Annual Volatility:{bt.annual_vol:>7.1%}")
    print(f"  Sharpe Ratio:     {bt.sharpe:.3f}")
    print(f"  Max Drawdown:     {bt.max_drawdown:.1%}")
    print(f"  Calmar Ratio:     {bt.calmar_ratio:.2f}")
    print()
    print(f"  Rebalances:       {bt.n_rebalances}")
    print(f"  Avg Turnover:     {bt.avg_turnover:.1%} per rebalance")
    print(f"{'='*55}\n")
