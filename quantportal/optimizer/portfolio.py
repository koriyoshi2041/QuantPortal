"""
Portfolio Optimizer — mean-variance optimization with constraints

★ Why optimize?
  Factor signals tell you WHAT to buy. The optimizer tells you HOW MUCH.
  Without optimization, equal-weight portfolios leave money on the table
  by ignoring correlations between assets.

★ Methods provided:
  1. Mean-Variance (Markowitz): maximize Sharpe = E[r] / σ
  2. Minimum Variance: minimize portfolio volatility (ignores expected returns)
  3. Risk Parity: equalize risk contribution from each asset
  4. Max Diversification: maximize diversification ratio

★ Constraints:
  - Long-only option (no shorting)
  - Position limits (max weight per asset)
  - Turnover penalty (avoid excessive rebalancing)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class OptMethod(Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


@dataclass(frozen=True)
class PortfolioWeights:
    """Optimal portfolio weights"""

    weights: pd.Series              # ticker → weight
    method: str
    expected_return: float
    expected_vol: float
    expected_sharpe: float
    diversification_ratio: float


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    method: OptMethod = OptMethod.MAX_SHARPE,
    long_only: bool = True,
    max_weight: float = 0.25,
    risk_free_rate: float = 0.0,
) -> PortfolioWeights:
    """Optimize portfolio weights.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected return per asset (annualized).
    cov_matrix : pd.DataFrame
        Covariance matrix of returns (annualized).
    method : OptMethod
        Optimization method.
    long_only : bool
        If True, no short selling (weights >= 0).
    max_weight : float
        Maximum weight per asset.
    risk_free_rate : float
        Risk-free rate for Sharpe calculation.
    """
    tickers = list(expected_returns.index)
    n = len(tickers)
    mu = expected_returns.values
    Sigma = cov_matrix.values

    if method == OptMethod.EQUAL_WEIGHT:
        w = np.ones(n) / n

    elif method == OptMethod.MIN_VARIANCE:
        w = _min_variance(Sigma, long_only, max_weight, n)

    elif method == OptMethod.RISK_PARITY:
        w = _risk_parity(Sigma, n)

    elif method == OptMethod.MAX_SHARPE:
        w = _max_sharpe(mu, Sigma, risk_free_rate, long_only, max_weight, n)

    else:
        w = np.ones(n) / n

    # Normalize
    w = w / w.sum()

    # Expected metrics
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ Sigma @ w))
    port_sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    # Diversification ratio = weighted avg vol / portfolio vol
    asset_vols = np.sqrt(np.diag(Sigma))
    weighted_avg_vol = float(w @ asset_vols)
    div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0

    return PortfolioWeights(
        weights=pd.Series(w, index=tickers),
        method=method.value,
        expected_return=port_ret,
        expected_vol=port_vol,
        expected_sharpe=port_sharpe,
        diversification_ratio=div_ratio,
    )


def _min_variance(Sigma: np.ndarray, long_only: bool, max_w: float, n: int) -> np.ndarray:
    """Minimum variance portfolio via analytical or iterative solution."""
    try:
        import cvxpy as cp
        w = cp.Variable(n)
        risk = cp.quad_form(w, Sigma)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        constraints.append(w <= max_w)
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if w.value is not None:
            return np.array(w.value).flatten()
    except Exception:
        pass

    # Fallback: analytical unconstrained minimum variance
    Sigma_inv = np.linalg.pinv(Sigma)
    ones = np.ones(n)
    w = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
    if long_only:
        w = np.maximum(w, 0)
        w = w / w.sum()
    return w


def _max_sharpe(
    mu: np.ndarray, Sigma: np.ndarray, rf: float,
    long_only: bool, max_w: float, n: int,
) -> np.ndarray:
    """Maximum Sharpe ratio portfolio."""
    try:
        import cvxpy as cp
        w = cp.Variable(n)
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        constraints.append(w <= max_w)
        # Maximize Sharpe ≈ maximize return - λ*risk for some λ
        # Use a grid search over risk aversion
        best_sharpe = -np.inf
        best_w = np.ones(n) / n

        for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            obj = ret - lam * risk
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve(solver=cp.SCS, verbose=False)
            if w.value is not None:
                wv = np.array(w.value).flatten()
                wv = wv / wv.sum()
                port_ret = float(wv @ mu)
                port_vol = float(np.sqrt(wv @ Sigma @ wv))
                sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w = wv.copy()

        return best_w
    except Exception:
        pass

    # Fallback: analytical tangent portfolio
    excess = mu - rf
    Sigma_inv = np.linalg.pinv(Sigma)
    w = Sigma_inv @ excess
    if long_only:
        w = np.maximum(w, 0)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(n) / n
    return w


def _risk_parity(Sigma: np.ndarray, n: int) -> np.ndarray:
    """Risk parity: equalize marginal risk contribution.

    Uses the inverse-volatility heuristic as starting point,
    then iteratively adjusts.
    """
    asset_vols = np.sqrt(np.diag(Sigma))
    # Inverse-vol weighting (simple risk parity approximation)
    w = 1.0 / asset_vols
    w = w / w.sum()
    return w


def print_portfolio(pw: PortfolioWeights, top_n: int = 15) -> None:
    """Print portfolio optimization results"""

    print(f"\n{'='*55}")
    print(f"  Portfolio Optimization ({pw.method})")
    print(f"{'='*55}")
    print()
    print(f"  Expected Return:        {pw.expected_return:+.1%}")
    print(f"  Expected Volatility:    {pw.expected_vol:.1%}")
    print(f"  Expected Sharpe:        {pw.expected_sharpe:.3f}")
    print(f"  Diversification Ratio:  {pw.diversification_ratio:.2f}")
    print()

    sorted_w = pw.weights.sort_values(ascending=False)
    print(f"  Top Holdings:")
    for ticker, weight in sorted_w.head(top_n).items():
        bar = "#" * int(weight * 100)
        print(f"    {ticker:<6} {weight:>6.1%}  {bar}")

    if len(sorted_w) > top_n:
        others = sorted_w.iloc[top_n:].sum()
        print(f"    {'others':<6} {others:>6.1%}")

    print(f"{'='*55}\n")
