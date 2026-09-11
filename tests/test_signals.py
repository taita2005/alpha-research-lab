import numpy as np
import pandas as pd
import pytest

from alpha_lab.signals import (
    momentum,
    reversal,
    volume_filtered_momentum,
)


def example_panels():
    dates = pd.bdate_range("2020-01-01", periods=12)
    steps = np.arange(12)

    adj = pd.DataFrame({
        "A": 100 * 1.02 ** steps,
        "B": 100 * 0.98 ** steps,
    }, index=dates)

    close = adj * 1.1
    volume = pd.DataFrame({
        "A": 1000 + steps * 100,
        "B": 2000 - steps * 50,
    }, index=dates, dtype=float)

    return adj, close, volume


def calculate(kind, adj, close, volume):
    if kind == "momentum":
        return momentum(adj, lookback=3)
    if kind == "reversal":
        return reversal(adj, lookback=3)
    return volume_filtered_momentum(
        adj, close, volume,
        lookback=3,
        short_window=2,
        long_window=4,
        min_ratio=1.0,
    )


def test_known_returns_and_warmup():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({
        "A": [100.0, 110.0, 121.0],
        "B": [100.0, 90.0, 81.0],
    }, index=dates)

    result = momentum(prices, lookback=2)

    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result.iloc[2], [0.21, -0.19])
    np.testing.assert_allclose(
        reversal(prices, lookback=2).iloc[2],
        [-0.21, 0.19],
    )


def test_volume_filter_uses_dollar_volume_and_strict_threshold():
    dates = pd.date_range("2020-01-01", periods=3)
    adj = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)

    # Dollar volume = [100, 100, 400].
    close = pd.DataFrame({"A": [10.0, 10.0, 20.0]}, index=dates)
    volume = pd.DataFrame({"A": [10.0, 10.0, 20.0]}, index=dates)

    # 400 / mean(100, 100, 400) = 2.
    accepted = volume_filtered_momentum(
        adj, close, volume,
        lookback=2, short_window=1, long_window=3,
        min_ratio=1.5,
    )
    np.testing.assert_allclose(accepted.iloc[-1, 0], 0.21)

    rejected = volume_filtered_momentum(
        adj, close, volume,
        lookback=2, short_window=1, long_window=3,
        min_ratio=2.0,
    )
    assert pd.isna(rejected.iloc[-1, 0])


@pytest.mark.parametrize("kind", [
    "momentum", "reversal", "volume_filtered_momentum",
])
def test_future_changes_do_not_change_past(kind):
    adj, close, volume = example_panels()
    expected = calculate(kind, adj, close, volume)
    cutoff = 7

    changed_adj = adj.copy()
    changed_close = close.copy()
    changed_volume = volume.copy()

    # Chỉ sửa dữ liệu SAU cutoff.
    changed_adj.iloc[cutoff + 1:] *= 5
    changed_close.iloc[cutoff + 1:] *= 3
    changed_volume.iloc[cutoff + 1:] *= 10

    actual = calculate(
        kind, changed_adj, changed_close, changed_volume
    )

    pd.testing.assert_frame_equal(
        expected.iloc[:cutoff + 1],
        actual.iloc[:cutoff + 1],
    )

    # Kết quả từ lịch sử cắt ngắn cũng phải trùng.
    prefix_result = calculate(
        kind,
        adj.iloc[:cutoff + 1],
        close.iloc[:cutoff + 1],
        volume.iloc[:cutoff + 1],
    )

    pd.testing.assert_frame_equal(
        expected.iloc[:cutoff + 1],
        prefix_result,
    )


@pytest.mark.parametrize("bad_window", [0, -1, 1.5, True])
def test_invalid_lookback_is_rejected(bad_window):
    adj, _, _ = example_panels()
    with pytest.raises(ValueError):
        momentum(adj, lookback=bad_window)


def test_misaligned_panels_are_rejected():
    adj, close, volume = example_panels()
    with pytest.raises(ValueError, match="identical"):
        volume_filtered_momentum(adj, close, volume.iloc[1:])


def test_missing_prices_are_rejected():
    adj, _, _ = example_panels()
    adj.iloc[3, 0] = np.nan

    with pytest.raises(ValueError, match="missing"):
        momentum(adj, lookback=3)


def test_zero_volume_produces_no_eligible_signal():
    adj, close, volume = example_panels()
    volume[:] = 0.0

    result = volume_filtered_momentum(
        adj, close, volume,
        lookback=3, short_window=2, long_window=4,
    )
    assert result.isna().all().all()
