"""
Universe Manager — fetch and manage a universe of stocks

Provides a clean interface to download price/return data for multiple
tickers and align them into a single DataFrame for cross-sectional analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Universe:
    """A universe of aligned stock data"""

    tickers: list[str]
    prices: pd.DataFrame        # Adj Close prices, tickers as columns
    returns: pd.DataFrame       # Log returns, tickers as columns
    n_assets: int
    n_days: int
    start_date: str
    end_date: str


def fetch_universe(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
) -> Universe:
    """Fetch aligned price data for a universe of tickers.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance ticker symbols.
    start : str
        Start date (YYYY-MM-DD).
    end : str or None
        End date (YYYY-MM-DD). Defaults to today.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    # Drop tickers with too much missing data (>20%)
    valid_cols = prices.columns[prices.isna().mean() < 0.2]
    prices = prices[valid_cols].dropna()

    returns = np.log(prices / prices.shift(1)).dropna()

    actual_tickers = list(prices.columns)

    return Universe(
        tickers=actual_tickers,
        prices=prices,
        returns=returns,
        n_assets=len(actual_tickers),
        n_days=len(returns),
        start_date=str(returns.index[0].date()),
        end_date=str(returns.index[-1].date()),
    )


# Pre-defined universes
SP500_TECH = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ADBE", "NFLX", "PYPL", "QCOM", "AVGO",
]

ETF_SECTORS = [
    "SPY", "QQQ", "XLF", "XLE", "XLV", "XLK", "XLI", "XLU", "XLP", "XLY",
    "GLD", "TLT", "IEF", "HYG", "LQD",
]

BOND_PAIRS = [
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "TIP",
]
