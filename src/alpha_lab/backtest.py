"""Long-only adjusted-return portfolio simulator."""

import numpy as np
import pandas as pd

from .signals import _check_panel, _check_window


def target_weights(scores, top_n=3, invested_fraction=0.95):
    """Positive scores only; deterministic ticker tie-breaking."""
    _check_window(top_n, "top_n")

    if not 0 <= invested_fraction <= 1:
        raise ValueError("invested_fraction must be between 0 and 1")

    if scores.index.has_duplicates:
        raise ValueError("Duplicate asset names")

    values = scores.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite scores")

    weights = pd.Series(0.0, index=scores.index)

    # Sort tickers first; stable score sort preserves ticker order on ties.
    eligible = scores.dropna()
    eligible = eligible[eligible > 0].sort_index()
    selected = eligible.sort_values(
        ascending=False, kind="mergesort"
    ).head(top_n)

    if len(selected):
        weights.loc[selected.index] = invested_fraction / len(selected)

    return weights


def rebalance_values(values, cash, weights, cost_rate):
    """Execute sells before buys, targeting weights of post-cost equity."""
    if not values.index.equals(weights.index):
        raise ValueError("Asset axes must match")

    if not np.isfinite(values.to_numpy()).all() or (values < 0).any():
        raise ValueError("Invalid position values")

    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Invalid cash")

    if (
        not np.isfinite(weights.to_numpy()).all()
        or (weights < 0).any()
        or weights.sum() > 1 + 1e-12
    ):
        raise ValueError("Invalid weights")

    if not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("Invalid cost rate")

    before = float(values.sum() + cash)
    if before <= 0:
        raise ValueError("Portfolio equity must be positive")

    # Solve E_after + fee * sum(abs(w * E_after - old_values)) = E_before.
    lo, hi = 0.0, before

    for _ in range(70):
        mid = (lo + hi) / 2
        required = mid + cost_rate * (weights * mid - values).abs().sum()

        if required > before:
            hi = mid
        else:
            lo = mid

    after = (lo + hi) / 2
    desired = weights * after
    orders = desired - values

    holdings = values.copy()
    trades = []
    tolerance = max(1e-8, before * 1e-10)

    for side in ("SELL", "BUY"):
        for asset, amount in orders.items():
            if side == "SELL" and amount < 0:
                notional = float(-amount)
                fee = notional * cost_rate

                holdings[asset] = desired[asset]
                cash += notional - fee

            elif side == "BUY" and amount > 0:
                notional = float(amount)
                fee = notional * cost_rate

                if notional + fee > cash + tolerance:
                    raise RuntimeError("Insufficient cash")

                holdings[asset] = desired[asset]
                cash -= notional + fee
            else:
                continue

            trades.append({
                "asset": asset,
                "side": side,
                "notional": notional,
                "fee": fee,
            })

    total_fee = sum(t["fee"] for t in trades)
    actual = float(holdings.sum() + cash)

    if not np.isclose(actual, before - total_fee, rtol=1e-10, atol=1e-8):
        raise RuntimeError("Portfolio accounting mismatch")

    if cash < -tolerance:
        raise RuntimeError("Negative cash")

    return holdings, float(cash), trades


def run_backtest(
    adj_close,
    scores,
    split_start,
    split_end,
    initial_capital=100_000.0,
    top_n=3,
    invested_fraction=0.95,
    rebalance_every=21,
    cost_bps=10.0,
    benchmark=None,
):
    """Execute signals at next session close; reset to cash per split."""
    _check_panel(adj_close, "adj_close")
    _check_window(rebalance_every, "rebalance_every")
    _check_window(top_n, "top_n")

    if benchmark not in (None, "spy_buy_hold", "equal_weight_buy_hold"):
        raise ValueError("Unknown benchmark")

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Invalid initial capital")

    if not 0 <= invested_fraction <= 1:
        raise ValueError("Invalid invested fraction")

    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("Invalid cost")

    if benchmark is None:
        if (
            scores is None
            or not scores.index.equals(adj_close.index)
            or not scores.columns.equals(adj_close.columns)
        ):
            raise ValueError("Scores and prices must have identical axes")

    prices = adj_close.loc[split_start:split_end]
    if len(prices) < 2:
        raise ValueError("Need at least two sessions")

    returns = prices / prices.shift(1) - 1
    returns.iloc[0] = 0.0

    values = pd.Series(0.0, index=prices.columns)
    cash = float(initial_capital)
    pending = None
    pending_date = None
    history, trade_log = [], []

    for step, date in enumerate(prices.index):
        # Existing holdings earn today's close-to-close adjusted return.
        values = values * (1 + returns.loc[date])
        equity_before_trade = float(values.sum() + cash)
        fee = 0.0
        turnover = 0.0
        executed = pending is not None

        if executed:
            values, cash, trades = rebalance_values(
                values, cash, pending, cost_bps / 10000
            )

            fee = sum(t["fee"] for t in trades)
            turnover = (
                sum(t["notional"] for t in trades) / equity_before_trade
            )

            for trade in trades:
                trade_log.append({
                    "signal_date": pending_date,
                    "execution_date": date,
                    **trade,
                })

            pending = None
            pending_date = None

        equity = float(values.sum() + cash)

        row = {
            "date": date,
            "equity": equity,
            "cash": cash,
            "fee": fee,
            "turnover": turnover,
            "executed": executed,
        }

        for asset in prices.columns:
            row[f"weight_{asset}"] = values[asset] / equity

        history.append(row)

        # Benchmarks buy once; candidates rebalance on a fixed session schedule.
        decision_due = (
            step == 0 if benchmark is not None
            else step % rebalance_every == 0
        )

        if decision_due and step < len(prices) - 1:
            if benchmark == "spy_buy_hold":
                if "SPY" not in prices.columns:
                    raise ValueError("SPY is missing")

                pending = pd.Series(0.0, index=prices.columns)
                pending["SPY"] = invested_fraction

            elif benchmark == "equal_weight_buy_hold":
                pending = pd.Series(
                    invested_fraction / len(prices.columns),
                    index=prices.columns,
                )

            else:
                pending = target_weights(
                    scores.loc[date], top_n, invested_fraction
                )

            pending_date = date

    history = pd.DataFrame(history).set_index("date")
    history["net_return"] = history["equity"].pct_change(fill_method=None)

    trades = pd.DataFrame(
        trade_log,
        columns=[
            "signal_date", "execution_date", "asset",
            "side", "notional", "fee",
        ],
    )

    return history, trades


def performance_metrics(history, annualization=252):
    """Zero risk-free rate; net daily returns; no forced final liquidation."""
    returns = history["net_return"].dropna()
    if len(returns) < 1:
        raise ValueError("Not enough returns")

    equity = history["equity"]
    years = len(returns) / annualization
    growth = float(equity.iloc[-1] / equity.iloc[0])
    volatility = float(returns.std(ddof=1))
    drawdown = equity / equity.cummax() - 1

    sharpe = (
        float(returns.mean() / volatility * np.sqrt(annualization))
        if np.isfinite(volatility) and volatility > 1e-12
        else np.nan
    )

    return {
        "total_return": growth - 1,
        "cagr": growth ** (1 / years) - 1,
        "net_sharpe": sharpe,
        "annual_volatility": volatility * np.sqrt(annualization),
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(history["turnover"].sum() / years),
        "total_fees": float(history["fee"].sum()),
        "final_equity": float(equity.iloc[-1]),
        "mean_cash_fraction": float(
            (history["cash"] / history["equity"]).mean()
        ),
    }
