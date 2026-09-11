"""Deterministic exploratory screening; not a significance test."""

import math


def decide(candidate, baseline, rules):
    required = ("net_sharpe", "max_drawdown", "annual_turnover")

    valid = all(
        isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
        and math.isfinite(row[key])
        for row in (candidate, baseline)
        for key in required
    )

    if not valid:
        return {
            "keep": False,
            "status": "invalid_metrics",
            "gates": {},
            "failed": ["finite_metrics"],
        }

    gates = {
        "sharpe": (
            candidate["net_sharpe"]
            >= baseline["net_sharpe"] + rules["min_sharpe_improvement"]
        ),
        "drawdown": (
            candidate["max_drawdown"]
            >= baseline["max_drawdown"]
            - rules["max_drawdown_deterioration"]
        ),
        "turnover": (
            candidate["annual_turnover"]
            <= baseline["annual_turnover"] * rules["max_turnover_ratio"]
        ),
    }

    failed = [name for name, passed in gates.items() if not passed]

    return {
        "keep": not failed,
        "status": "keep_for_research" if not failed else "reject",
        "gates": gates,
        "failed": failed,
        "interpretation": (
            "Exploratory validation screening only; "
            "not independent test evidence or deployment approval."
        ),
    }
