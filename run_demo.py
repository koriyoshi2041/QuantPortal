#!/usr/bin/env python3
"""
QuantPortal Demo — ML-Driven Multi-Signal Portfolio Optimizer

Run:
    cd quantportal
    source .venv/bin/activate
    python run_demo.py

Demonstrates:
  1. Multi-factor signal generation (momentum, volatility, quality)
  2. Cross-sectional pair scanning
  3. Equal-weight factor combination
  4. ML (LightGBM) signal combination
  5. Portfolio optimization
  6. Walk-forward backtesting
  7. Publication-quality visualizations
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_factor_signals():
    """Demo 1: Multi-factor signal generation"""
    print("\n" + "=" * 70)
    print("  DEMO 1: Multi-Factor Signal Generation")
    print("=" * 70)

    from quantportal.data.universe import fetch_universe, SP500_TECH
    from quantportal.factors.momentum import compute_momentum
    from quantportal.factors.volatility import compute_volatility_signals
    from quantportal.factors.quality import compute_quality_signals

    universe = fetch_universe(SP500_TECH[:10], start="2018-01-01")
    print(f"\n  Universe: {universe.n_assets} assets, {universe.n_days} days")
    print(f"  Period: {universe.start_date} ~ {universe.end_date}")

    mom = compute_momentum(universe.returns)
    vol = compute_volatility_signals(universe.returns)
    qual = compute_quality_signals(universe.returns)

    print("\n  Momentum signal (latest cross-section):")
    latest_mom = mom.cs_zscore.iloc[-1].sort_values(ascending=False)
    for ticker, score in latest_mom.head(5).items():
        print(f"    {ticker:<6} {score:+.2f}")

    print("\n  Low-vol ranking (latest):")
    latest_vol = vol.low_vol_score.iloc[-1].sort_values(ascending=False)
    for ticker, score in latest_vol.head(5).items():
        print(f"    {ticker:<6} {score:.2f}")

    print("\n  Quality score (latest):")
    latest_qual = qual.stability_score.iloc[-1].sort_values(ascending=False)
    for ticker, score in latest_qual.head(5).items():
        print(f"    {ticker:<6} {score:+.2f}")

    # Visualize signal heatmap
    from quantportal.viz.plots import plot_signal_heatmap
    plot_signal_heatmap(
        mom.cs_zscore.dropna(), "Momentum Signal (12M-1M)",
        save_path=os.path.join(OUTPUT_DIR, "momentum_heatmap.png"),
    )
    print(f"\n  Signal heatmap saved to {OUTPUT_DIR}/")

    return universe


def demo_pair_scanning():
    """Demo 2: Automated pair discovery"""
    print("\n" + "=" * 70)
    print("  DEMO 2: Pair Scanner — Bond ETF Universe")
    print("=" * 70)

    from quantportal.data.universe import fetch_universe, BOND_PAIRS
    from quantportal.scanner.pair_scanner import scan_pairs, print_scan_result

    universe = fetch_universe(BOND_PAIRS, start="2015-01-01")
    print(f"\n  Scanning {universe.n_assets} bond ETFs for cointegrated pairs...")

    result = scan_pairs(universe.prices, universe.returns)
    print_scan_result(result)

    return result


def demo_backtest(universe):
    """Demo 3: Factor backtest"""
    print("\n" + "=" * 70)
    print("  DEMO 3: Multi-Factor Portfolio Backtest")
    print("=" * 70)

    from quantportal.factors.momentum import compute_momentum
    from quantportal.factors.volatility import compute_volatility_signals
    from quantportal.factors.quality import compute_quality_signals
    from quantportal.ml.signal_combiner import combine_signals_equal
    from quantportal.backtest.engine import backtest_portfolio, print_backtest
    from quantportal.viz.plots import plot_equity_curve, plot_weights_timeline

    mom = compute_momentum(universe.returns)
    vol = compute_volatility_signals(universe.returns)
    qual = compute_quality_signals(universe.returns)

    features = {
        "momentum": mom.cs_zscore,
        "low_vol": vol.low_vol_score,
        "quality": qual.stability_score,
    }
    combined = combine_signals_equal(features)

    # Top-5 portfolio
    bt = backtest_portfolio(
        universe.returns, combined, top_n=5, rebalance_freq=21,
    )
    print_backtest(bt)

    # Benchmark: equal weight all assets
    equal_sig = pd.DataFrame(1.0, index=universe.returns.index, columns=universe.returns.columns)
    bt_bench = backtest_portfolio(universe.returns, equal_sig, rebalance_freq=63)

    # Plot equity curve
    plot_equity_curve(
        bt.portfolio_value,
        bt_bench.portfolio_value,
        title="Multi-Factor Top-5 vs Equal-Weight Benchmark",
        save_path=os.path.join(OUTPUT_DIR, "equity_curve.png"),
    )

    # Plot weight allocation
    plot_weights_timeline(
        bt.weights_history,
        title="Portfolio Weight Allocation Over Time",
        save_path=os.path.join(OUTPUT_DIR, "weights_timeline.png"),
    )

    print(f"\n  Charts saved to {OUTPUT_DIR}/")
    return bt


def demo_ml_combination(universe):
    """Demo 4: ML signal combination"""
    print("\n" + "=" * 70)
    print("  DEMO 4: LightGBM Signal Combination")
    print("=" * 70)

    from quantportal.factors.momentum import compute_momentum
    from quantportal.factors.volatility import compute_volatility_signals
    from quantportal.factors.quality import compute_quality_signals
    from quantportal.ml.signal_combiner import combine_signals_ml, print_ml_signal
    from quantportal.backtest.engine import backtest_portfolio, print_backtest
    from quantportal.viz.plots import plot_feature_importance

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

    forward_ret = universe.returns.shift(-1)
    ml_signal = combine_signals_ml(features, forward_ret)
    print_ml_signal(ml_signal)

    # ML backtest
    bt = backtest_portfolio(
        universe.returns, ml_signal.predictions, top_n=5, rebalance_freq=21,
    )
    print_backtest(bt)

    # Feature importance
    plot_feature_importance(
        ml_signal.feature_importance,
        title="LightGBM Feature Importance",
        save_path=os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )

    print(f"\n  Feature importance chart saved")
    return ml_signal


def demo_portfolio_optimization(universe):
    """Demo 5: Portfolio optimization"""
    print("\n" + "=" * 70)
    print("  DEMO 5: Portfolio Optimization")
    print("=" * 70)

    from quantportal.optimizer.portfolio import (
        optimize_portfolio, OptMethod, print_portfolio,
    )

    # Expected returns = historical mean
    exp_ret = universe.returns.mean() * 252
    cov = universe.returns.cov() * 252

    for method in [OptMethod.MAX_SHARPE, OptMethod.MIN_VARIANCE, OptMethod.RISK_PARITY]:
        pw = optimize_portfolio(exp_ret, cov, method=method)
        print_portfolio(pw)


def main():
    print("\n" + "#" * 70)
    print("  QuantPortal — ML-Driven Multi-Signal Portfolio Optimizer")
    print("  Complete Demo")
    print("#" * 70)

    # Demo 1: Factor signals
    universe = demo_factor_signals()

    # Demo 2: Pair scanning
    demo_pair_scanning()

    # Demo 3: Factor backtest
    demo_backtest(universe)

    # Demo 4: ML combination
    demo_ml_combination(universe)

    # Demo 5: Portfolio optimization
    demo_portfolio_optimization(universe)

    print("\n" + "#" * 70)
    print("  All demos complete!")
    print(f"  Charts saved to: {OUTPUT_DIR}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
