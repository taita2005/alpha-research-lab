import numpy as np
import pandas as pd

from alpha_lab.evaluation import (
    forward_returns,
    rank_ic,
    summarize_ic,
)


def fixture_panels():
    dates = pd.bdate_range("2020-01-01", periods=10)

    prices = pd.DataFrame({
        "A": 100 * 1.10 ** np.arange(10),
        "B": 100 * 1.05 ** np.arange(10),
        "C": np.full(10, 100.0),
    }, index=dates)

    scores = pd.DataFrame(
        [[3.0, 2.0, 1.0]] * 10,
        index=dates,
        columns=prices.columns,
    )
    return prices, scores


def test_return_starts_at_execution_price():
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 200.0, 220.0, 242.0]},
        index=dates,
    )

    labels = forward_returns(prices, horizon=1, entry_lag=1)

    # Entry at 200, exit at 220: 10%, not 100% or 120%.
    np.testing.assert_allclose(labels.iloc[0, 0], 0.10)


def test_inverse_ranking():
    prices, scores = fixture_panels()

    daily = rank_ic(
        -scores, prices,
        prices.index[0], prices.index[-1],
        horizon=2, entry_lag=1, min_assets=3,
    )

    np.testing.assert_allclose(daily["ic"], -1.0)


def test_labels_never_cross_split_end():
    prices, scores = fixture_panels()
    end = prices.index[5]

    daily = rank_ic(
        scores, prices,
        prices.index[0], end,
        horizon=2, entry_lag=1, min_assets=3,
    )

    assert (daily["exit_date"] <= end).all()
    assert daily.index.max() == prices.index[2]
    assert len(daily) == 3


def test_constant_scores_are_undefined():
    prices, scores = fixture_panels()
    scores[:] = 1.0

    daily = rank_ic(
        scores, prices,
        prices.index[0], prices.index[-1],
        horizon=2, entry_lag=1, min_assets=3,
    )

    assert daily["ic"].isna().all()
    assert (daily["status"] == "constant_cross_section").all()


def test_missing_assets_are_reported():
    prices, scores = fixture_panels()
    scores["C"] = np.nan

    daily = rank_ic(
        scores, prices,
        prices.index[0], prices.index[-1],
        horizon=2, entry_lag=1, min_assets=3,
    )

    assert daily["ic"].isna().all()
    np.testing.assert_allclose(daily["coverage"], 2 / 3)
    assert (daily["status"] == "insufficient_assets").all()


def test_summary_counts_only_defined_ic():
    daily = pd.DataFrame({
        "ic": [1.0, -1.0, np.nan],
        "coverage": [1.0, 1.0, 0.5],
    })

    result = summarize_ic(daily)

    assert result["label_dates"] == 3
    assert result["ic_dates"] == 2
    assert result["mean_ic"] == 0.0
    assert result["positive_ic_fraction"] == 0.5
