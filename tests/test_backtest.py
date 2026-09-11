import numpy as np
import pandas as pd

from alpha_lab.backtest import (
    target_weights,
    rebalance_values,
    run_backtest,
    performance_metrics,
)


def test_tie_breaking_and_positive_filter():
    scores = pd.Series({
        "B": 2.0, "A": 2.0, "C": -1.0, "D": np.nan,
    })

    weights = target_weights(scores, top_n=1, invested_fraction=0.95)

    assert weights["A"] == 0.95
    assert weights.drop("A").sum() == 0.0


def test_no_positive_scores_means_cash():
    scores = pd.Series({"A": 0.0, "B": -1.0, "C": np.nan})
    assert target_weights(scores).sum() == 0.0


def test_sell_before_buy_without_starting_cash():
    values = pd.Series({"A": 0.0, "B": 100.0})
    weights = pd.Series({"A": 0.95, "B": 0.0})

    holdings, cash, trades = rebalance_values(
        values, 0.0, weights, cost_rate=0.001
    )

    assert [t["side"] for t in trades] == ["SELL", "BUY"]

    fees = sum(t["fee"] for t in trades)
    equity = holdings.sum() + cash

    np.testing.assert_allclose(equity, 100.0 - fees)
    np.testing.assert_allclose(holdings["A"] / equity, 0.95)
    np.testing.assert_allclose(cash / equity, 0.05)


def test_initial_purchase_fee_has_known_solution():
    values = pd.Series({"A": 0.0})
    weights = pd.Series({"A": 0.95})

    holdings, cash, trades = rebalance_values(
        values, 100.0, weights, cost_rate=0.001
    )

    expected_equity = 100.0 / (1 + 0.95 * 0.001)

    np.testing.assert_allclose(holdings.sum() + cash, expected_equity)
    np.testing.assert_allclose(holdings["A"], 0.95 * expected_equity)


def test_next_close_execution():
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 200.0, 220.0, 242.0]}, index=dates
    )
    scores = pd.DataFrame({"A": [1.0] * 4}, index=dates)

    history, trades = run_backtest(
        prices, scores, dates[0], dates[-1],
        initial_capital=100.0, invested_fraction=1.0, cost_bps=0,
    )

    np.testing.assert_allclose(
        history["equity"], [100.0, 100.0, 110.0, 121.0]
    )
    assert trades.iloc[0]["signal_date"] == dates[0]
    assert trades.iloc[0]["execution_date"] == dates[1]


def test_weights_drift_without_unscheduled_trades():
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({
        "A": [100.0, 100.0, 200.0, 200.0],
        "B": [100.0, 100.0, 100.0, 100.0],
    }, index=dates)
    scores = pd.DataFrame(1.0, index=dates, columns=prices.columns)

    history, trades = run_backtest(
        prices, scores, dates[0], dates[-1],
        initial_capital=100.0, invested_fraction=1.0,
        top_n=2, cost_bps=0, rebalance_every=21,
    )

    assert len(trades) == 2  # Initial A and B purchases only.
    np.testing.assert_allclose(history.iloc[2]["weight_A"], 2 / 3)
    assert history.iloc[2:]["turnover"].sum() == 0


def test_rebalance_schedule_and_split_reset():
    dates = pd.bdate_range("2020-01-01", periods=8)
    prices = pd.DataFrame({"A": [100.0] * 8}, index=dates)
    scores = pd.DataFrame({"A": [1.0] * 8}, index=dates)

    history, _ = run_backtest(
        prices, scores, dates[2], dates[-1],
        initial_capital=100.0, cost_bps=0, rebalance_every=2,
    )

    assert history.iloc[0]["cash"] == 100.0

    actual_dates = history.index[history["executed"]].tolist()
    assert actual_dates == [dates[3], dates[5], dates[7]]


def test_buy_hold_never_rebalances():
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame({
        "SPY": [100, 100, 120, 110, 130, 140],
        "B": [100, 100, 100, 100, 100, 100],
    }, index=dates, dtype=float)

    history, trades = run_backtest(
        prices, None, dates[0], dates[-1],
        rebalance_every=1,
        benchmark="equal_weight_buy_hold",
    )

    assert len(trades) == 2
    assert history["executed"].sum() == 1


def test_drawdown_uses_previous_peak():
    history = pd.DataFrame({
        "equity": [100.0, 120.0, 90.0],
        "net_return": [np.nan, 0.2, -0.25],
        "turnover": [0.0, 0.0, 0.0],
        "fee": [0.0, 0.0, 0.0],
        "cash": [0.0, 0.0, 0.0],
    })

    result = performance_metrics(history)

    np.testing.assert_allclose(result["total_return"], -0.10)
    np.testing.assert_allclose(result["max_drawdown"], -0.25)
