"""Evaluate one bounded alpha proposal on development and validation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .alpha_schema import (
    validate_proposal, validate_expression, expression_hash, DSL_VERSION,
)
from .alpha_dsl import evaluate_expression, format_expression
from .research_data import load_research_data
from .research_decision import decide
from .backtest import run_backtest, performance_metrics
from .evaluation import rank_ic, summarize_ic


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _clean_record(record):
    # Convert undefined numeric metrics to JSON null.
    return json.loads(
        pd.DataFrame([record]).to_json(
            orient="records", double_precision=15
        )
    )[0]


def run_experiment(root, proposal):
    root = Path(root)
    info = validate_proposal(proposal)

    # Detach caller-owned objects.
    proposal = json.loads(json.dumps(proposal, allow_nan=False))
    config_path = root / "configs/interactive_research.json"
    config = json.loads(config_path.read_text())

    if config["data_cutoff"] != "2021-12-31":
        raise ValueError("Unsupported data cutoff")
    if set(config["splits"]) != {"dev", "val"}:
        raise ValueError("Only dev and val are allowed")

    # Copy registered rules at experiment start.
    panels, data_hashes = load_research_data(root)
    prices = panels["adj_close"]
    execution = config["execution"]
    base_cost = execution["base_cost_bps"]
    costs = [base_cost, execution["stress_cost_bps"]]

    if costs != [10.0, 20.0]:
        raise ValueError("Unexpected cost settings")

    baseline_expression = config["baseline"]["expression"]
    baseline_info = validate_expression(baseline_expression)
    needed_history = max(
        info["required_history_sessions"],
        baseline_info["required_history_sessions"],
    )

    for split, (start, end) in config["splits"].items():
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError("Invalid split")
        if pd.Timestamp(end) > pd.Timestamp(config["data_cutoff"]):
            raise ValueError("Split exceeds cutoff")
        if int((prices.index < pd.Timestamp(start)).sum()) < needed_history:
            raise ValueError(
                f"Insufficient warmup for {split}: "
                f"requires {needed_history} earlier sessions"
            )

    run_id = uuid4().hex
    run_dir = root / "artifacts/v2_experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    code_names = [
        "alpha_schema.py", "alpha_dsl.py", "research_data.py",
        "research_decision.py", "experiment_runner.py",
        "backtest.py", "evaluation.py", "signals.py",
    ]
    code_hashes = {
        f"src/alpha_lab/{name}": _hash(root / "src/alpha_lab" / name)
        for name in code_names
    }

    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal": proposal,
        "formula": format_expression(proposal["expression"]),
        "expression_hash": expression_hash(proposal["expression"]),
        "dsl_version": DSL_VERSION,
        "config": config,
        "config_sha256": _hash(config_path),
        "data_hashes": data_hashes,
        "code_hashes": code_hashes,
        "holdout_used": False,
        "study_type": config["study_type"],
    }
    _write_json(run_dir / "experiment.json", metadata)

    try:
        metric_rows = []
        ic_rows = []

        for split, (start, end) in config["splits"].items():
            # Compute dev scores without even passing validation rows.
            split_panels = {
                name: frame.loc[:end]
                for name, frame in panels.items()
            }
            split_prices = split_panels["adj_close"]

            candidate_scores = evaluate_expression(
                proposal["expression"], split_panels
            )
            baseline_scores = evaluate_expression(
                baseline_expression, split_panels
            )

            for label, scores in [
                ("candidate", candidate_scores),
                ("baseline", baseline_scores),
            ]:
                daily_ic = rank_ic(
                    scores, split_prices, start, end,
                    horizon=config["ic"]["horizon"],
                    entry_lag=config["ic"]["entry_lag"],
                    min_assets=config["ic"]["min_assets"],
                )
                daily_ic.to_csv(run_dir / f"{split}_{label}_ic.csv")
                ic_rows.append({
                    "split": split,
                    "strategy": label,
                    **summarize_ic(daily_ic),
                })

            for cost in costs:
                for label in (
                    "candidate", "baseline",
                    "spy_buy_hold", "equal_weight_buy_hold",
                ):
                    benchmark = (
                        label if label.endswith("buy_hold") else None
                    )
                    scores = (
                        candidate_scores if label == "candidate"
                        else baseline_scores
                    )

                    history, trades = run_backtest(
                        adj_close=split_prices,
                        scores=None if benchmark else scores,
                        split_start=start,
                        split_end=end,
                        initial_capital=execution["initial_capital"],
                        top_n=execution["top_n"],
                        invested_fraction=execution["invested_fraction"],
                        rebalance_every=execution["rebalance_every"],
                        cost_bps=cost,
                        benchmark=benchmark,
                    )

                    prefix = f"{split}_{label}_{cost:g}bps"
                    history.to_csv(run_dir / f"{prefix}_history.csv")
                    trades.to_csv(
                        run_dir / f"{prefix}_trades.csv", index=False
                    )
                    metric_rows.append({
                        "split": split,
                        "strategy": label,
                        "cost_bps": cost,
                        **performance_metrics(history),
                    })

        metrics = pd.DataFrame(metric_rows)
        ic_summary = pd.DataFrame(ic_rows)

        metrics.to_csv(run_dir / "metrics.csv", index=False)
        ic_summary.to_csv(run_dir / "ic_summary.csv", index=False)

        validation = metrics.loc[
            (metrics["split"] == "val")
            & (metrics["cost_bps"] == base_cost)
        ].set_index("strategy")

        decision = decide(
            validation.loc["candidate"].to_dict(),
            validation.loc["baseline"].to_dict(),
            config["decision"],
        )
        _write_json(run_dir / "decision.json", decision)

        result = {
            "run_id": run_id,
            "formula": metadata["formula"],
            "proposal": proposal,
            "decision": decision,
            "metrics": [_clean_record(row) for row in metric_rows],
            "ic": [_clean_record(row) for row in ic_rows],
            "holdout_used": False,
            "interpretation": (
                "Exploratory dev/validation results. "
                "The v1 holdout has already been observed by the researcher."
            ),
        }
        _write_json(run_dir / "result.json", result)

        # Detect changes during this run.
        for relative, expected in {**data_hashes, **code_hashes}.items():
            if _hash(root / relative) != expected:
                raise RuntimeError(f"Input changed during run: {relative}")
        if _hash(config_path) != metadata["config_sha256"]:
            raise RuntimeError("Config changed during run")

        metadata["status"] = "completed"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["result_sha256"] = _hash(run_dir / "result.json")
        _write_json(run_dir / "experiment.json", metadata)
        return result

    except Exception as error:
        metadata["status"] = "failed"
        metadata["error_type"] = type(error).__name__
        _write_json(run_dir / "experiment.json", metadata)
        raise
