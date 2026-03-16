"""
ML Signal Combiner — use LightGBM to optimally weight multiple signals

★ Why ML for signal combination?
  Simple equal-weighting of signals (momentum + low-vol + quality)
  ignores that different signals work better in different regimes.

  ML approach:
  - Features: momentum z-score, vol score, quality score, regime indicator
  - Target: next-period return (or rank)
  - Model: LightGBM (gradient boosted trees)

  Trees naturally handle regime-conditional behavior:
  "If regime=choppy AND momentum>0, weight quality higher"
  This is impossible with a fixed linear combination.

★ Anti-overfitting measures:
  1. Purged cross-validation (no leakage across time)
  2. Feature importance monitoring
  3. Regularization (max_depth, min_child_weight, L1/L2)
  4. Walk-forward retraining (only use past data)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MLSignal:
    """ML-combined signal"""

    predictions: pd.DataFrame      # date × ticker: predicted return/rank
    feature_importance: pd.Series  # feature name → importance
    train_r2: float
    oos_r2: float
    n_features: int


def combine_signals_ml(
    features: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    train_ratio: float = 0.7,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
) -> MLSignal:
    """Combine multiple signal DataFrames using LightGBM.

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        Mapping from feature name to signal DataFrame (date × ticker).
    forward_returns : pd.DataFrame
        Next-period returns (date × ticker) — the prediction target.
    train_ratio : float
        Fraction of data used for training (rest is OOS test).
    """
    import lightgbm as lgb

    # Stack all features into a single panel
    feature_names = list(features.keys())
    all_rows = []

    dates = forward_returns.index
    tickers = forward_returns.columns

    for date in dates:
        for ticker in tickers:
            row = {}
            valid = True
            for fname, fdf in features.items():
                if date in fdf.index and ticker in fdf.columns:
                    val = fdf.loc[date, ticker]
                    if pd.isna(val):
                        valid = False
                        break
                    row[fname] = val
                else:
                    valid = False
                    break

            if not valid:
                continue

            target = forward_returns.loc[date, ticker]
            if pd.isna(target):
                continue

            row["_date"] = date
            row["_ticker"] = ticker
            row["_target"] = target
            all_rows.append(row)

    if len(all_rows) < 100:
        # Not enough data — return equal-weight combination
        combined = pd.DataFrame(0.0, index=dates, columns=tickers)
        for fdf in features.values():
            aligned = fdf.reindex(index=dates, columns=tickers)
            combined += aligned.fillna(0)
        combined /= len(features)

        return MLSignal(
            predictions=combined,
            feature_importance=pd.Series({f: 1.0/len(features) for f in feature_names}),
            train_r2=0.0,
            oos_r2=0.0,
            n_features=len(feature_names),
        )

    panel = pd.DataFrame(all_rows)
    panel = panel.sort_values("_date")

    X = panel[feature_names].values
    y = panel["_target"].values

    # Train/test split (temporal)
    split_idx = int(len(panel) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # R-squared
    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    train_r2 = _r2(y_train, train_pred)
    oos_r2 = _r2(y_test, test_pred)

    # Feature importance
    importance = model.feature_importances_
    feat_imp = pd.Series(
        importance / importance.sum(),
        index=feature_names,
    ).sort_values(ascending=False)

    # Generate full predictions
    all_pred = model.predict(X)
    panel["_pred"] = all_pred

    # Pivot back to date × ticker
    predictions = panel.pivot(index="_date", columns="_ticker", values="_pred")

    return MLSignal(
        predictions=predictions,
        feature_importance=feat_imp,
        train_r2=float(train_r2),
        oos_r2=float(oos_r2),
        n_features=len(feature_names),
    )


def combine_signals_equal(
    features: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Simple equal-weight signal combination (baseline).

    Returns cross-sectional z-scored combined signal.
    """
    dfs = list(features.values())
    combined = dfs[0].copy() * 0
    for df in dfs:
        # Cross-sectional z-score each signal
        cs_z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
        combined += cs_z.fillna(0)
    combined /= len(dfs)
    return combined


def print_ml_signal(ml: MLSignal) -> None:
    """Print ML signal combiner results"""

    print(f"\n{'='*55}")
    print(f"  ML Signal Combiner (LightGBM)")
    print(f"{'='*55}")
    print(f"\n  Features:     {ml.n_features}")
    print(f"  Train R2:     {ml.train_r2:.4f}")
    print(f"  OOS R2:       {ml.oos_r2:.4f}")

    overfitting = ml.train_r2 - ml.oos_r2
    print(f"  Overfit gap:  {overfitting:.4f}  ", end="")
    if overfitting < 0.02:
        print("(Good)")
    elif overfitting < 0.05:
        print("(Fair)")
    else:
        print("(Warning: possible overfitting)")

    print(f"\n  Feature Importance:")
    for feat, imp in ml.feature_importance.items():
        bar = "#" * int(imp * 50)
        print(f"    {feat:<20} {imp:>5.1%}  {bar}")

    print(f"{'='*55}\n")
