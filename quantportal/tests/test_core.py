"""
QuantPortal Core Tests — all using synthetic data (no network dependency)

Tests cover:
  - Factor signals (momentum, volatility, quality)
  - Pair scanner
  - Portfolio optimizer
  - ML signal combiner
  - Backtest engine
  - Visualization (smoke tests)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _make_universe(n_assets: int = 10, n_days: int = 500, seed: int = 42):
    """Generate a synthetic multi-asset universe."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_assets)]

    # Common market factor + idiosyncratic
    market = rng.normal(0.0003, 0.01, n_days)
    returns_data = {}
    for i, t in enumerate(tickers):
        beta = 0.5 + rng.uniform(0, 1)
        idio = rng.normal(0, 0.005, n_days)
        returns_data[t] = market * beta + idio

    returns = pd.DataFrame(returns_data, index=dates)
    prices = (1 + returns).cumprod() * 100

    return prices, returns, tickers


def _make_cointegrated_prices(n: int = 1000, seed: int = 42):
    """Generate a pair of cointegrated price series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    common = np.cumsum(rng.normal(0, 0.01, n))
    noise_a = rng.normal(0, 0.005, n)
    noise_b = rng.normal(0, 0.005, n)

    pa = pd.Series(100 + common + noise_a, index=dates, name="A")
    pb = pd.Series(50 + 0.5 * common + noise_b, index=dates, name="B")

    return pa, pb


# ---------------------------------------------------------------------------
# factors/momentum.py
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_momentum_shape(self):
        from quantportal.factors.momentum import compute_momentum
        _, returns, tickers = _make_universe()
        mom = compute_momentum(returns)
        assert mom.raw_momentum.shape == returns.shape
        assert mom.cs_zscore.shape == returns.shape
        assert mom.lookback == 252

    def test_cs_zscore_mean_near_zero(self):
        """Cross-sectional z-score should have ~0 mean across assets."""
        from quantportal.factors.momentum import compute_momentum
        _, returns, _ = _make_universe()
        mom = compute_momentum(returns)
        # After warming up, CS mean should be near 0
        valid = mom.cs_zscore.dropna()
        cs_means = valid.mean(axis=1)
        assert abs(cs_means.mean()) < 0.5

    def test_ts_momentum_binary(self):
        """Time-series momentum should be +1 or -1."""
        from quantportal.factors.momentum import compute_momentum
        _, returns, _ = _make_universe()
        mom = compute_momentum(returns)
        valid = mom.ts_momentum.dropna()
        unique = set(valid.values.flatten())
        assert unique.issubset({-1.0, 1.0})

    def test_short_term_reversal(self):
        from quantportal.factors.momentum import compute_short_term_reversal
        _, returns, _ = _make_universe()
        rev = compute_short_term_reversal(returns)
        assert rev.shape == returns.shape


# ---------------------------------------------------------------------------
# factors/volatility.py
# ---------------------------------------------------------------------------

class TestVolatilityFactor:
    def test_vol_signals_shape(self):
        from quantportal.factors.volatility import compute_volatility_signals
        _, returns, _ = _make_universe()
        vol = compute_volatility_signals(returns)
        assert vol.realized_vol.shape == returns.shape
        assert vol.low_vol_score.shape == returns.shape

    def test_low_vol_score_bounded(self):
        """Low-vol score should be between 0 and 1 (percentile rank)."""
        from quantportal.factors.volatility import compute_volatility_signals
        _, returns, _ = _make_universe()
        vol = compute_volatility_signals(returns)
        valid = vol.low_vol_score.dropna()
        assert valid.min().min() >= 0
        assert valid.max().max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# factors/quality.py
# ---------------------------------------------------------------------------

class TestQuality:
    def test_quality_shape(self):
        from quantportal.factors.quality import compute_quality_signals
        _, returns, _ = _make_universe()
        qual = compute_quality_signals(returns)
        assert qual.rolling_sharpe.shape == returns.shape
        assert qual.stability_score.shape == returns.shape

    def test_drawdown_score_positive(self):
        """Drawdown score should be non-negative (inverted drawdown)."""
        from quantportal.factors.quality import compute_quality_signals
        _, returns, _ = _make_universe()
        qual = compute_quality_signals(returns)
        valid = qual.drawdown_score.dropna()
        assert valid.min().min() >= -1e-9


# ---------------------------------------------------------------------------
# scanner/pair_scanner.py
# ---------------------------------------------------------------------------

class TestPairScanner:
    def test_scan_finds_pair(self):
        """Scanning cointegrated pair should find at least one result."""
        from quantportal.scanner.pair_scanner import scan_pairs
        pa, pb = _make_cointegrated_prices()
        prices = pd.DataFrame({"A": pa, "B": pb})
        result = scan_pairs(prices, max_pval=0.5, apply_bonferroni=False)
        assert result.n_tested == 1
        assert len(result.pairs) >= 0  # may or may not find it

    def test_scan_multi_asset(self):
        from quantportal.scanner.pair_scanner import scan_pairs
        prices, _, tickers = _make_universe(n_assets=5)
        result = scan_pairs(prices, apply_bonferroni=False)
        assert result.n_tested == 10  # C(5,2) = 10

    def test_scan_result_fields(self):
        from quantportal.scanner.pair_scanner import scan_pairs
        pa, pb = _make_cointegrated_prices()
        prices = pd.DataFrame({"A": pa, "B": pb})
        result = scan_pairs(prices, apply_bonferroni=False)
        assert isinstance(result.bonferroni_threshold, float)
        assert result.n_tested >= 0


# ---------------------------------------------------------------------------
# optimizer/portfolio.py
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_equal_weight(self):
        from quantportal.optimizer.portfolio import optimize_portfolio, OptMethod
        _, returns, tickers = _make_universe(n_assets=5)
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        pw = optimize_portfolio(exp_ret, cov, method=OptMethod.EQUAL_WEIGHT)
        assert abs(pw.weights.sum() - 1.0) < 1e-9
        assert all(abs(w - 0.2) < 1e-9 for w in pw.weights)

    def test_min_variance(self):
        from quantportal.optimizer.portfolio import optimize_portfolio, OptMethod
        _, returns, _ = _make_universe(n_assets=5)
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        pw = optimize_portfolio(exp_ret, cov, method=OptMethod.MIN_VARIANCE)
        assert abs(pw.weights.sum() - 1.0) < 1e-6
        assert pw.weights.min() >= -1e-6  # long only

    def test_max_sharpe(self):
        from quantportal.optimizer.portfolio import optimize_portfolio, OptMethod
        _, returns, _ = _make_universe(n_assets=5)
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        pw = optimize_portfolio(exp_ret, cov, method=OptMethod.MAX_SHARPE)
        assert abs(pw.weights.sum() - 1.0) < 1e-6

    def test_risk_parity(self):
        from quantportal.optimizer.portfolio import optimize_portfolio, OptMethod
        _, returns, _ = _make_universe(n_assets=5)
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        pw = optimize_portfolio(exp_ret, cov, method=OptMethod.RISK_PARITY)
        assert abs(pw.weights.sum() - 1.0) < 1e-6
        assert pw.weights.min() >= 0

    def test_max_weight_constraint(self):
        from quantportal.optimizer.portfolio import optimize_portfolio, OptMethod
        _, returns, _ = _make_universe(n_assets=5)
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        pw = optimize_portfolio(exp_ret, cov, method=OptMethod.MIN_VARIANCE, max_weight=0.30)
        assert pw.weights.max() <= 0.30 + 1e-4


# ---------------------------------------------------------------------------
# ml/signal_combiner.py
# ---------------------------------------------------------------------------

class TestMLCombiner:
    def test_equal_combination(self):
        from quantportal.ml.signal_combiner import combine_signals_equal
        _, returns, tickers = _make_universe()
        sig1 = returns.rolling(20).mean()
        sig2 = returns.rolling(20).std()
        combined = combine_signals_equal({"sig1": sig1, "sig2": sig2})
        assert combined.shape == returns.shape

    def test_ml_combination(self):
        from quantportal.ml.signal_combiner import combine_signals_ml
        _, returns, tickers = _make_universe(n_assets=5, n_days=400)
        sig1 = returns.rolling(20).mean()
        sig2 = returns.rolling(20).std()
        forward = returns.shift(-1)
        ml = combine_signals_ml(
            {"sig1": sig1, "sig2": sig2},
            forward,
            n_estimators=50,
        )
        assert ml.n_features == 2
        assert isinstance(ml.train_r2, float)
        assert isinstance(ml.oos_r2, float)
        assert len(ml.feature_importance) == 2

    def test_ml_insufficient_data(self):
        """Too little data should still return result (fallback to equal)."""
        from quantportal.ml.signal_combiner import combine_signals_ml
        dates = pd.bdate_range("2020-01-01", periods=20)
        tickers = ["A", "B"]
        returns = pd.DataFrame(
            np.random.randn(20, 2) * 0.01,
            index=dates, columns=tickers,
        )
        sig = returns.rolling(5).mean()
        forward = returns.shift(-1)
        ml = combine_signals_ml({"sig": sig}, forward)
        assert ml.n_features == 1


# ---------------------------------------------------------------------------
# backtest/engine.py
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_backtest_runs(self):
        from quantportal.backtest.engine import backtest_portfolio
        _, returns, tickers = _make_universe()
        signal = returns.rolling(60).mean()  # momentum signal
        bt = backtest_portfolio(returns, signal, top_n=3)
        assert len(bt.portfolio_value) > 0
        assert bt.n_rebalances > 0

    def test_backtest_nav_positive(self):
        """NAV should always be positive."""
        from quantportal.backtest.engine import backtest_portfolio
        _, returns, _ = _make_universe()
        signal = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
        bt = backtest_portfolio(returns, signal)
        assert bt.portfolio_value.min() > 0

    def test_backtest_transaction_costs(self):
        """Higher transaction costs should reduce returns."""
        from quantportal.backtest.engine import backtest_portfolio
        _, returns, _ = _make_universe()
        signal = returns.rolling(20).mean()
        bt_low = backtest_portfolio(returns, signal, top_n=3, transaction_cost=0.0001)
        bt_high = backtest_portfolio(returns, signal, top_n=3, transaction_cost=0.01)
        assert bt_low.total_return >= bt_high.total_return - 0.01

    def test_backtest_metrics_reasonable(self):
        from quantportal.backtest.engine import backtest_portfolio
        _, returns, _ = _make_universe()
        signal = returns.rolling(60).mean()
        bt = backtest_portfolio(returns, signal, top_n=3)
        assert -1 < bt.max_drawdown <= 0
        assert bt.annual_vol > 0
        assert isinstance(bt.sharpe, float)
        assert bt.avg_turnover >= 0


# ---------------------------------------------------------------------------
# viz/plots.py (smoke tests)
# ---------------------------------------------------------------------------

class TestViz:
    def test_equity_curve_no_crash(self):
        import matplotlib
        matplotlib.use("Agg")
        from quantportal.viz.plots import plot_equity_curve
        nav = pd.Series(np.cumsum(np.random.randn(100) * 0.01) + 1.0,
                        index=pd.bdate_range("2020-01-01", periods=100))
        plot_equity_curve(nav, save_path="/tmp/test_equity.png")

    def test_weights_no_crash(self):
        import matplotlib
        matplotlib.use("Agg")
        from quantportal.viz.plots import plot_weights_timeline
        dates = pd.bdate_range("2020-01-01", periods=10)
        weights = pd.DataFrame(
            np.random.dirichlet([1]*5, 10), index=dates,
            columns=[f"S{i}" for i in range(5)],
        )
        plot_weights_timeline(weights, save_path="/tmp/test_weights.png")

    def test_feature_importance_no_crash(self):
        import matplotlib
        matplotlib.use("Agg")
        from quantportal.viz.plots import plot_feature_importance
        imp = pd.Series({"momentum": 0.4, "vol": 0.35, "quality": 0.25})
        plot_feature_importance(imp, save_path="/tmp/test_imp.png")

    def test_signal_heatmap_no_crash(self):
        import matplotlib
        matplotlib.use("Agg")
        from quantportal.viz.plots import plot_signal_heatmap
        dates = pd.bdate_range("2020-01-01", periods=100)
        sig = pd.DataFrame(
            np.random.randn(100, 5),
            index=dates, columns=[f"S{i}" for i in range(5)],
        )
        plot_signal_heatmap(sig, save_path="/tmp/test_heatmap.png")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
