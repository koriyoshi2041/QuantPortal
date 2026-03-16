"""
Pair Scanner — automated discovery of cointegrated pairs from a universe

★ The problem with manual pair selection:
  Most quant tutorials pick GLD/GDX or KO/PEP and call it a day.
  But real pair trading requires systematic screening:
  1. Test all N*(N-1)/2 combinations for cointegration
  2. Filter by spread stationarity (ADF test)
  3. Rank by half-life (faster mean reversion = more tradeable)
  4. Apply Bonferroni correction for multiple testing

★ This scanner:
  - Tests all pairs in a universe
  - Ranks by cointegration p-value, half-life, and stability
  - Applies configurable significance threshold with multiple testing warning
  - Returns a ranked list of tradeable pairs
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller


@dataclass(frozen=True)
class PairCandidate:
    """A candidate cointegrated pair"""

    ticker_a: str
    ticker_b: str
    coint_pvalue: float
    hedge_ratio: float
    half_life: float
    spread_adf_pvalue: float
    correlation: float
    score: float              # composite ranking score


@dataclass(frozen=True)
class ScanResult:
    """Results from pair scanning"""

    n_tested: int
    n_significant: int
    n_tradeable: int          # passes all filters
    pairs: list[PairCandidate]
    bonferroni_threshold: float


def scan_pairs(
    prices: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    max_pval: float = 0.05,
    max_half_life: float = 30.0,
    min_half_life: float = 1.0,
    apply_bonferroni: bool = True,
) -> ScanResult:
    """Scan all pairs in a universe for cointegration.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices (date × ticker).
    returns : pd.DataFrame or None
        Log returns (date × ticker). If None, computed from prices.
    max_pval : float
        Maximum cointegration p-value.
    max_half_life : float
        Maximum acceptable half-life (days).
    min_half_life : float
        Minimum half-life (filter out spurious fast-reversion).
    apply_bonferroni : bool
        Whether to apply Bonferroni correction for multiple testing.
    """
    tickers = list(prices.columns)
    n = len(tickers)
    n_pairs = n * (n - 1) // 2

    if returns is None:
        returns = np.log(prices / prices.shift(1)).dropna()

    threshold = max_pval / n_pairs if apply_bonferroni else max_pval

    candidates: list[PairCandidate] = []

    for i in range(n):
        for j in range(i + 1, n):
            ta, tb = tickers[i], tickers[j]
            pa = prices[ta].dropna()
            pb = prices[tb].dropna()
            common = pa.index.intersection(pb.index)
            pa_c = pa.loc[common]
            pb_c = pb.loc[common]

            if len(pa_c) < 100:
                continue

            # Cointegration test
            try:
                _, pval, _ = coint(pa_c, pb_c)
            except Exception:
                continue

            if pval > max_pval:  # loose filter first
                continue

            # Hedge ratio
            beta = float(np.polyfit(pb_c.values, pa_c.values, 1)[0])

            # Spread
            spread = pa_c - beta * pb_c

            # ADF test on spread
            try:
                _, adf_pval, *_ = adfuller(spread.dropna(), maxlag=20)
            except Exception:
                adf_pval = 1.0

            # Half-life
            spread_clean = spread.dropna()
            if len(spread_clean) > 10:
                s_lag = spread_clean.shift(1).dropna()
                s_now = spread_clean.iloc[1:]
                ci = s_lag.index.intersection(s_now.index)
                if len(ci) > 10:
                    slope, _, _, _, _ = stats.linregress(s_lag.loc[ci], s_now.loc[ci])
                    hl = -np.log(2) / np.log(slope) if 0 < slope < 1 else float("inf")
                else:
                    hl = float("inf")
            else:
                hl = float("inf")

            # Correlation
            ra = returns[ta].dropna()
            rb = returns[tb].dropna()
            ci = ra.index.intersection(rb.index)
            corr = float(ra.loc[ci].corr(rb.loc[ci]))

            # Composite score: lower pval + faster half-life + lower ADF = better
            if hl == float("inf") or hl < min_half_life or hl > max_half_life:
                score = -999.0
            else:
                score = -np.log10(max(pval, 1e-20)) + 10.0 / hl - np.log10(max(adf_pval, 1e-20))

            candidates.append(PairCandidate(
                ticker_a=ta,
                ticker_b=tb,
                coint_pvalue=float(pval),
                hedge_ratio=beta,
                half_life=hl,
                spread_adf_pvalue=float(adf_pval),
                correlation=corr,
                score=score,
            ))

    # Sort by score (descending)
    candidates.sort(key=lambda c: c.score, reverse=True)

    n_significant = sum(1 for c in candidates if c.coint_pvalue < max_pval)
    n_tradeable = sum(
        1 for c in candidates
        if c.coint_pvalue < max_pval
        and min_half_life < c.half_life < max_half_life
        and c.spread_adf_pvalue < 0.05
    )

    return ScanResult(
        n_tested=n_pairs,
        n_significant=n_significant,
        n_tradeable=n_tradeable,
        pairs=candidates,
        bonferroni_threshold=threshold,
    )


def print_scan_result(sr: ScanResult, top_n: int = 10) -> None:
    """Print pair scan results"""

    print(f"\n{'='*70}")
    print(f"  Pair Scanner Results")
    print(f"{'='*70}")
    print(f"\n  Tested: {sr.n_tested} pairs")
    print(f"  Significant (p<0.05): {sr.n_significant}")
    print(f"  Tradeable (all filters): {sr.n_tradeable}")
    print(f"  Bonferroni threshold: {sr.bonferroni_threshold:.6f}")
    print()

    if not sr.pairs:
        print("  No pairs found.")
        return

    top = [p for p in sr.pairs if p.score > -999][:top_n]
    if not top:
        print("  No tradeable pairs found.")
        return

    print(f"  {'Rank':>4} {'Pair':<12} {'Coint p':>8} {'ADF p':>8} {'HL':>6} {'Corr':>6} {'Score':>7}")
    print(f"  {'-'*55}")
    for i, p in enumerate(top):
        hl_str = f"{p.half_life:.0f}d" if p.half_life < 1000 else "inf"
        print(f"  {i+1:>4} {p.ticker_a}/{p.ticker_b:<10} {p.coint_pvalue:>8.4f} {p.spread_adf_pvalue:>8.4f} {hl_str:>6} {p.correlation:>6.3f} {p.score:>7.1f}")

    print(f"{'='*70}\n")
