"""
QuantPortal Scanner — main entry points

Usage:
    from quantportal.scan import quick_scan, full_scan

    # Quick scan: factor signals + equal-weight backtest
    result = quick_scan(["AAPL", "MSFT", "GOOG", "AMZN", "META"])

    # Full scan: factors + ML combination + optimized portfolio + backtest
    result = full_scan(["AAPL", "MSFT", "GOOG", ...], ml=True)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantportal.data.universe import Universe, fetch_universe
from quantportal.backtest.engine import BacktestResult


@dataclass(frozen=True)
class ScanReport:
    """Quick scan report"""

    universe: Universe
    signal: pd.DataFrame
    backtest: BacktestResult


@dataclass(frozen=True)
class FullReport:
    """Full scan report with ML and optimization"""

    universe: Universe
    factor_signals: dict[str, pd.DataFrame]
    combined_signal: pd.DataFrame
    backtest_equal: BacktestResult
    backtest_ml: BacktestResult | None
    ml_importance: pd.Series | None


def quick_scan(
    tickers: list[str],
    start: str = "2015-01-01",
    top_n: int = 5,
) -> ScanReport:
    """Quick factor scan + equal-weight backtest.

    Computes momentum, volatility, and quality signals,
    combines them equally, and runs a top-N backtest.
    """
    from quantportal.factors.momentum import compute_momentum
    from quantportal.factors.volatility import compute_volatility_signals
    from quantportal.factors.quality import compute_quality_signals
    from quantportal.ml.signal_combiner import combine_signals_equal
    from quantportal.backtest.engine import backtest_portfolio, print_backtest

    universe = fetch_universe(tickers, start=start)

    print(f"\n  Universe: {universe.n_assets} assets, {universe.n_days} days")
    print(f"  Period: {universe.start_date} ~ {universe.end_date}")

    mom = compute_momentum(universe.returns)
    vol = compute_volatility_signals(universe.returns)
    qual = compute_quality_signals(universe.returns)

    features = {
        "momentum": mom.cs_zscore,
        "low_vol": vol.low_vol_score,
        "quality": qual.stability_score,
    }

    combined = combine_signals_equal(features)

    bt = backtest_portfolio(
        universe.returns, combined, top_n=top_n, rebalance_freq=21,
    )
    print_backtest(bt)

    return ScanReport(universe=universe, signal=combined, backtest=bt)


def full_scan(
    tickers: list[str],
    start: str = "2015-01-01",
    top_n: int = 5,
    use_ml: bool = True,
) -> FullReport:
    """Full scan with ML signal combination and optimized portfolio.

    Steps:
    1. Fetch universe data
    2. Compute all factor signals (momentum, vol, quality)
    3. Equal-weight baseline backtest
    4. ML-combined signal backtest (if use_ml=True)
    5. Feature importance analysis
    """
    from quantportal.factors.momentum import compute_momentum
    from quantportal.factors.volatility import compute_volatility_signals
    from quantportal.factors.quality import compute_quality_signals
    from quantportal.ml.signal_combiner import (
        combine_signals_equal, combine_signals_ml, print_ml_signal,
    )
    from quantportal.backtest.engine import backtest_portfolio, print_backtest

    universe = fetch_universe(tickers, start=start)

    print(f"\n  Universe: {universe.n_assets} assets, {universe.n_days} days")
    print(f"  Period: {universe.start_date} ~ {universe.end_date}")

    # Compute factors
    mom = compute_momentum(universe.returns)
    vol = compute_volatility_signals(universe.returns)
    qual = compute_quality_signals(universe.returns)

    features = {
        "momentum": mom.cs_zscore,
        "ts_momentum": mom.ts_momentum,
        "low_vol": vol.low_vol_score,
        "vol_trend": vol.vol_trend,
        "quality": qual.stability_score,
        "rolling_sharpe": qual.rolling_sharpe,
    }

    # Equal-weight baseline
    combined_equal = combine_signals_equal(features)
    bt_equal = backtest_portfolio(
        universe.returns, combined_equal, top_n=top_n, rebalance_freq=21,
    )
    print(f"\n  === Equal-Weight Factor Combination ===")
    print_backtest(bt_equal)

    # ML combination
    bt_ml = None
    ml_importance = None

    if use_ml:
        forward_ret = universe.returns.shift(-1)  # next-day return as target
        ml_signal = combine_signals_ml(features, forward_ret)
        print_ml_signal(ml_signal)

        bt_ml = backtest_portfolio(
            universe.returns, ml_signal.predictions, top_n=top_n, rebalance_freq=21,
        )
        print(f"\n  === ML-Combined Signal ===")
        print_backtest(bt_ml)

        ml_importance = ml_signal.feature_importance

    return FullReport(
        universe=universe,
        factor_signals=features,
        combined_signal=combined_equal,
        backtest_equal=bt_equal,
        backtest_ml=bt_ml,
        ml_importance=ml_importance,
    )
