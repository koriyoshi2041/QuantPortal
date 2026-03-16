"""
QuantPortal Visualization — dark-theme publication-quality charts

Charts:
1. Portfolio equity curve vs benchmark
2. Factor signal heatmap (cross-sectional)
3. Weight allocation over time
4. Drawdown chart
5. Feature importance bar chart
6. Pair scanner result visualization
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dark theme colors
COLORS = {
    "bg": "#1a1a2e",
    "panel": "#16213e",
    "text": "#e0e0e0",
    "grid": "#2a2a4a",
    "primary": "#00d2ff",
    "secondary": "#ff6b6b",
    "accent": "#ffd93d",
    "success": "#6bcb77",
    "purple": "#b088f9",
}


def _setup_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def plot_equity_curve(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    title: str = "Portfolio Equity Curve",
    save_path: str | None = None,
) -> None:
    """Plot portfolio NAV vs benchmark."""
    _setup_dark_theme()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1],
                              gridspec_kw={"hspace": 0.3})

    # Equity curve
    ax = axes[0]
    ax.plot(portfolio_nav.index, portfolio_nav.values,
            color=COLORS["primary"], linewidth=1.5, label="Portfolio")
    if benchmark_nav is not None:
        ax.plot(benchmark_nav.index, benchmark_nav.values,
                color=COLORS["secondary"], linewidth=1, alpha=0.7, label="Benchmark")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2)

    # Drawdown
    ax2 = axes[1]
    peak = portfolio_nav.cummax()
    dd = (portfolio_nav - peak) / peak
    ax2.fill_between(dd.index, dd.values, 0, color=COLORS["secondary"], alpha=0.4)
    ax2.set_title("Drawdown", fontsize=11)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)


def plot_weights_timeline(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Weights Over Time",
    save_path: str | None = None,
) -> None:
    """Stacked area chart of portfolio weights."""
    _setup_dark_theme()
    fig, ax = plt.subplots(figsize=(12, 5))

    if weights_df.empty:
        ax.text(0.5, 0.5, "No weight data available",
                ha="center", va="center", color=COLORS["text"])
    else:
        # Keep top N assets for readability
        avg_weight = weights_df.mean().sort_values(ascending=False)
        top_n = min(8, len(avg_weight))
        top_cols = avg_weight.head(top_n).index.tolist()
        plot_df = weights_df[top_cols]

        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  COLORS["success"], COLORS["purple"], "#ff9f43", "#ee5a24", "#a29bfe"]

        ax.stackplot(plot_df.index, plot_df.T.values,
                     labels=top_cols, colors=colors[:top_n], alpha=0.7)
        ax.legend(loc="upper left", framealpha=0.3, fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)


def plot_feature_importance(
    importance: pd.Series,
    title: str = "Feature Importance",
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of feature importances."""
    _setup_dark_theme()
    fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.4)))

    sorted_imp = importance.sort_values(ascending=True)
    bars = ax.barh(sorted_imp.index, sorted_imp.values, color=COLORS["primary"], alpha=0.8)

    for bar, val in zip(bars, sorted_imp.values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9, color=COLORS["text"])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)


def plot_signal_heatmap(
    signal: pd.DataFrame,
    title: str = "Cross-Sectional Signal",
    save_path: str | None = None,
) -> None:
    """Heatmap of signal values over time."""
    _setup_dark_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Subsample for readability
    sample_dates = signal.index[::max(1, len(signal)//60)]
    signal_sub = signal.loc[sample_dates]

    im = ax.imshow(signal_sub.T.values, aspect="auto", cmap="RdYlGn",
                    vmin=-2, vmax=2)
    ax.set_yticks(range(len(signal_sub.columns)))
    ax.set_yticklabels(signal_sub.columns, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Date labels
    n_labels = min(8, len(sample_dates))
    step = max(1, len(sample_dates) // n_labels)
    ax.set_xticks(range(0, len(sample_dates), step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in sample_dates[::step]],
        rotation=45, fontsize=8,
    )

    fig.colorbar(im, ax=ax, label="Signal Score", shrink=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
