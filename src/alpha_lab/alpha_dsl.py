"""Deterministic interpreter for bounded JSON alpha expressions."""

import numpy as np
import pandas as pd

from .alpha_schema import (
    FIELDS,
    DIVISION_EPSILON,
    validate_expression,
)


def _validated_panels(panels):
    if not isinstance(panels, dict) or set(panels) != set(FIELDS):
        raise ValueError(f"Expected exactly these panels: {FIELDS}")

    cleaned = {}
    reference = None

    for name in FIELDS:
        frame = panels[name]

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name}: expected a non-empty DataFrame")

        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans
            or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing
        ):
            raise ValueError(f"{name}: dates must be sorted and unique")

        if frame.columns.has_duplicates:
            raise ValueError(f"{name}: duplicate assets")

        if not all(isinstance(column, str) for column in frame.columns):
            raise ValueError(f"{name}: asset names must be strings")

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_bool_dtype(dtype)
            and not pd.api.types.is_complex_dtype(dtype)
            for dtype in frame.dtypes
        ):
            raise ValueError(f"{name}: expected real numeric columns")

        values = frame.to_numpy(dtype=float, na_value=np.nan)

        if np.isinf(values).any():
            raise ValueError(f"{name}: infinite input")

        finite = values[np.isfinite(values)]
        if name == "volume":
            if (finite < 0).any():
                raise ValueError("volume cannot be negative")
        elif (finite <= 0).any():
            raise ValueError("Observed prices must be positive")

        if reference is not None:
            if (
                not frame.index.equals(reference.index)
                or not frame.columns.equals(reference.columns)
            ):
                raise ValueError("All panels must have identical axes")
        else:
            reference = frame

        cleaned[name] = pd.DataFrame(
            values.copy(),
            index=frame.index.copy(),
            columns=frame.columns.copy(),
        )

    return cleaned


def _finite_result(frame):
    # Numerical overflow is an error, not a silently eligible score.
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Non-finite result from arithmetic overflow")
    return frame


def _safe_divide(left, right):
    denominator = right.where(right.abs() > DIVISION_EPSILON)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = left / denominator
    return _finite_result(result)


def evaluate_expression(expression, panels):
    """Return causal numeric scores; do not perform execution-time shifting."""
    validate_expression(expression)
    data = _validated_panels(panels)
    template = data["adj_close"]

    def visit(node):
        op = node["op"]

        if op == "field":
            return data[node["name"]].copy()

        if op == "constant":
            return pd.DataFrame(
                float(node["value"]),
                index=template.index,
                columns=template.columns,
            )

        if op == "return":
            price = data[node["field"]]
            return _safe_divide(price, price.shift(node["window"])) - 1.0

        if op == "filter":
            score = visit(node["input"])
            condition = visit(node["condition"])
            return score.where(condition.fillna(False).astype(bool))

        if op in ("add", "subtract", "safe_divide", "greater_than"):
            left = visit(node["left"])
            right = visit(node["right"])

            if op == "safe_divide":
                return _safe_divide(left, right)

            if op == "greater_than":
                # Keep "unknown" distinct from False until filter is applied.
                result = left.gt(right).astype("boolean")
                return result.mask(left.isna() | right.isna(), pd.NA)

            with np.errstate(over="ignore", invalid="ignore"):
                result = left + right if op == "add" else left - right
            return _finite_result(result)

        value = visit(node["input"])

        if op == "rank":
            return value.rank(
                axis=1,
                method="average",
                ascending=True,
                na_option="keep",
                pct=True,
            )

        if op == "negate":
            return -value

        if op == "multiply":
            with np.errstate(over="ignore", invalid="ignore"):
                result = value * float(node["value"])
            return _finite_result(result)

        if op == "lag":
            return value.shift(node["window"])

        window = node["window"]
        rolling = value.rolling(
            window=window,
            min_periods=window,
            center=False,
        )

        if op == "rolling_mean":
            return _finite_result(rolling.mean())

        if op == "rolling_std":
            return _finite_result(rolling.std(ddof=0))

        raise ValueError("Unreachable operator after schema validation")

    result = visit(expression).astype(float)

    if (
        not result.index.equals(template.index)
        or not result.columns.equals(template.columns)
    ):
        raise RuntimeError("Interpreter changed output axes")

    return _finite_result(result)


def format_expression(expression):
    """Display-only formula renderer. Output must never be executed."""
    validate_expression(expression)

    def render(node):
        op = node["op"]

        if op == "field":
            return node["name"]

        if op == "constant":
            return f"{float(node['value']):g}"

        if op == "return":
            return f"return({node['field']}, {node['window']})"

        if op == "filter":
            return (
                f"filter({render(node['input'])}, "
                f"{render(node['condition'])})"
            )

        if op in ("add", "subtract", "greater_than"):
            symbol = {"add": "+", "subtract": "-", "greater_than": ">"}[op]
            return f"({render(node['left'])} {symbol} {render(node['right'])})"

        if op == "safe_divide":
            return (
                f"safe_divide({render(node['left'])}, "
                f"{render(node['right'])})"
            )

        if op == "multiply":
            return f"({float(node['value']):g} * {render(node['input'])})"

        if op == "negate":
            return f"(-{render(node['input'])})"

        if op == "rank":
            return f"rank({render(node['input'])})"

        return f"{op}({render(node['input'])}, {node['window']})"

    return render(expression)
