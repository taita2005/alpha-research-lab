"""Deterministic candidate selection using base-cost validation results."""

import numpy as np
import pandas as pd


def select_candidate(results, config):
    rules = config["selection"]
    base_cost = config["execution"]["cost_bps_per_side"]
    names = [item["name"] for item in config["candidates"]]
    metric = rules["metric"]

    if rules["split"] != "val":
        raise ValueError("Selection must use validation")

    rows = results.loc[
        results["strategy"].isin(names)
        & (results["cost_bps"] == base_cost)
    ].copy()

    if (
        len(rows) != len(names)
        or rows["strategy"].duplicated().any()
        or set(rows["strategy"]) != set(names)
    ):
        raise ValueError("Need exactly one base-cost row per candidate")

    finite = np.isfinite(rows[metric].to_numpy(dtype=float))
    valid = rows.loc[finite]

    if valid.empty:
        raise ValueError("No candidate has a defined selection metric")

    best = float(valid[metric].max())
    tolerance = rules["tie_tolerance"]

    tied = set(
        valid.loc[
            best - valid[metric] <= tolerance,
            "strategy",
        ]
    )

    for name in rules["tie_break_order"]:
        if name in tied:
            return name

    raise ValueError("Tie-break order does not cover candidates")
