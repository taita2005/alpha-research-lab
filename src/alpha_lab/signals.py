"""Causal daily signals. Larger scores indicate higher preference."""

import numpy as np
import pandas as pd


def _check_window(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_panel(frame, name, allow_zero=False):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")

    if frame.index.hasnans:
        raise ValueError(f"{name} contains missing dates")

    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and sorted")

    if frame.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate asset columns")

    values = frame.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or infinite values")

    invalid = values < 0 if allow_zero else values <= 0
    if invalid.any():
        raise ValueError(f"{name} contains invalid values")


def momentum(adj_close, lookback=126):
    """Close-to-close return over lookback sessions, including date t."""
    _check_window(lookback, "lookback")
    _check_panel(adj_close, "adj_close")

    return adj_close / adj_close.shift(lookback) - 1.0


def reversal(adj_close, lookback=5):
    """Negative trailing return: recent losers receive higher scores."""
    return -momentum(adj_close, lookback=lookback)


def volume_filtered_momentum(
    adj_close,
    close,
    volume,
    lookback=126,
    short_window=20,
    long_window=120,
    min_ratio=1.0,
):
    """Momentum eligible only when short/long dollar-volume ratio exceeds threshold."""
    _check_panel(adj_close, "adj_close")
    _check_panel(close, "close")
    _check_panel(volume, "volume", allow_zero=True)

    _check_window(short_window, "short_window")
    _check_window(long_window, "long_window")

    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    if not np.isfinite(min_ratio) or min_ratio <= 0:
        raise ValueError("min_ratio must be finite and positive")

    for frame in (close, volume):
        if (
            not frame.index.equals(adj_close.index)
            or not frame.columns.equals(adj_close.columns)
        ):
            raise ValueError("All panels must have identical dates and assets")

    score = momentum(adj_close, lookback=lookback)

    dollar_volume = close * volume
    short_mean = dollar_volume.rolling(
        short_window, min_periods=short_window
    ).mean()
    long_mean = dollar_volume.rolling(
        long_window, min_periods=long_window
    ).mean()

    ratio = short_mean / long_mean.replace(0.0, np.nan)

    return score.where(ratio > min_ratio)
