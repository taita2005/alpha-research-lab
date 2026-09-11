"""Forward labels and cross-sectional rank IC evaluation."""

import numpy as np
import pandas as pd

from .signals import _check_panel, _check_window


def forward_returns(adj_close, horizon=21, entry_lag=1):
    """Return from close[t+entry_lag] to close[t+entry_lag+horizon]."""
    _check_panel(adj_close, "adj_close")
    _check_window(horizon, "horizon")
    _check_window(entry_lag, "entry_lag")

    entry = adj_close.shift(-entry_lag)
    exit_price = adj_close.shift(-(entry_lag + horizon))

    return exit_price / entry - 1.0


def rank_ic(
    scores,
    adj_close,
    split_start,
    split_end,
    horizon=21,
    entry_lag=1,
    min_assets=5,
):
    """Daily Spearman IC; signal, entry and exit must stay inside split."""
    _check_panel(adj_close, "adj_close")
    _check_window(min_assets, "min_assets")

    if min_assets < 2:
        raise ValueError("min_assets must be at least 2")

    if (
        not scores.index.equals(adj_close.index)
        or not scores.columns.equals(adj_close.columns)
    ):
        raise ValueError("Scores and prices must have identical axes")

    # NaN scores are allowed: warmup or ineligible assets.
    if np.isinf(scores.to_numpy(dtype=float)).any():
        raise ValueError("Scores contain infinite values")

    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)

    if start > end:
        raise ValueError("Invalid split boundaries")

    labels = forward_returns(adj_close, horizon, entry_lag)
    dates = pd.Series(adj_close.index, index=adj_close.index)

    entry_dates = dates.shift(-entry_lag)
    exit_dates = dates.shift(-(entry_lag + horizon))

    eligible_dates = (
        (dates >= start)
        & (dates <= end)
        & (entry_dates >= start)
        & (entry_dates <= end)
        & (exit_dates >= start)
        & (exit_dates <= end)
    )

    rows = []

    for date in adj_close.index[eligible_dates]:
        score = scores.loc[date]
        label = labels.loc[date]

        valid = score.notna() & label.notna()
        n_valid = int(valid.sum())

        ic = np.nan
        status = "insufficient_assets"

        if n_valid >= min_assets:
            x = score[valid]
            y = label[valid]

            if x.nunique() < 2 or y.nunique() < 2:
                status = "constant_cross_section"
            else:
                # Spearman = Pearson correlation of average ranks.
                ic = float(x.rank().corr(y.rank()))
                status = "ok"

        rows.append({
            "date": date,
            "entry_date": entry_dates.loc[date],
            "exit_date": exit_dates.loc[date],
            "n_assets": n_valid,
            "coverage": n_valid / len(scores.columns),
            "ic": ic,
            "status": status,
        })

    if not rows:
        raise ValueError("No valid label dates inside this split")

    return pd.DataFrame(rows).set_index("date")


def summarize_ic(daily):
    values = daily["ic"].dropna()

    return {
        "label_dates": len(daily),
        "ic_dates": len(values),
        "ic_date_fraction": len(values) / len(daily),
        "mean_asset_coverage": float(daily["coverage"].mean()),
        "mean_ic": float(values.mean()),
        "median_ic": float(values.median()),
        "std_ic": float(values.std(ddof=1)),
        "positive_ic_fraction": (
            float((values > 0).mean()) if len(values) else np.nan
        ),
    }
