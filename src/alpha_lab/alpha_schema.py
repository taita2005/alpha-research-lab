"""Versioned, bounded schema for JSON alpha expressions."""

import hashlib
import json
import math
import re


DSL_VERSION = "1.0"

FIELDS = ("adj_close", "close", "volume")

MAX_DEPTH = 8
MAX_NODES = 63
MAX_WINDOW = 252
MAX_HISTORY = 504
MAX_ABS_CONSTANT = 10.0
DIVISION_EPSILON = 1e-12

# Exact keys required for each node.
NODE_KEYS = {
    "field": {"op", "name"},
    "constant": {"op", "value"},
    "return": {"op", "field", "window"},
    "rolling_mean": {"op", "input", "window"},
    "rolling_std": {"op", "input", "window"},
    "lag": {"op", "input", "window"},
    "rank": {"op", "input"},
    "negate": {"op", "input"},
    "multiply": {"op", "input", "value"},
    "add": {"op", "left", "right"},
    "subtract": {"op", "left", "right"},
    "safe_divide": {"op", "left", "right"},
    "greater_than": {"op", "left", "right"},
    "filter": {"op", "input", "condition"},
}


class AlphaValidationError(ValueError):
    """Invalid or unsupported alpha proposal."""


def _fail(message):
    raise AlphaValidationError(message)


def _window(value):
    if type(value) is not int or not 1 <= value <= MAX_WINDOW:
        _fail(f"window must be an integer in [1, {MAX_WINDOW}]")


def _number(value):
    if type(value) not in (int, float):
        _fail("value must be a JSON number, not bool or string")
    if not math.isfinite(value) or abs(value) > MAX_ABS_CONSTANT:
        _fail(f"value must be finite and within +/-{MAX_ABS_CONSTANT}")


def validate_expression(expression):
    """Return structural metadata; the root must be numeric."""
    node_count = 0
    deepest = 0
    fields = set()

    def visit(node, depth):
        nonlocal node_count, deepest

        if depth > MAX_DEPTH:
            _fail("Expression exceeds maximum depth")

        node_count += 1
        deepest = max(deepest, depth)

        if node_count > MAX_NODES:
            _fail("Expression exceeds maximum node count")

        if type(node) is not dict:
            _fail("Each expression node must be an object")

        op = node.get("op")
        if not isinstance(op, str) or op not in NODE_KEYS:
            _fail("Unknown operator")

        if set(node) != NODE_KEYS[op]:
            _fail(f"Unexpected or missing keys for operator {op}")

        if "window" in node:
            _window(node["window"])

        if "value" in node:
            _number(node["value"])

        if op in ("field", "return"):
            name = node["name"] if op == "field" else node["field"]

            if not isinstance(name, str) or name not in FIELDS:
                _fail("Unknown input field")

            fields.add(name)
            history = node["window"] if op == "return" else 0
            return "numeric", history

        if op == "constant":
            return "numeric", 0

        if op == "filter":
            input_type, input_history = visit(node["input"], depth + 1)
            condition_type, condition_history = visit(
                node["condition"], depth + 1
            )

            if input_type != "numeric" or condition_type != "boolean":
                _fail("filter requires numeric input and boolean condition")

            return "numeric", max(input_history, condition_history)

        if op in ("add", "subtract", "safe_divide", "greater_than"):
            left_type, left_history = visit(node["left"], depth + 1)
            right_type, right_history = visit(node["right"], depth + 1)

            if left_type != "numeric" or right_type != "numeric":
                _fail(f"{op} requires numeric operands")

            result_type = "boolean" if op == "greater_than" else "numeric"
            return result_type, max(left_history, right_history)

        input_type, history = visit(node["input"], depth + 1)

        if input_type != "numeric":
            _fail(f"{op} requires numeric input")

        if op == "lag":
            history += node["window"]
        elif op in ("rolling_mean", "rolling_std"):
            history += node["window"] - 1

        return "numeric", history

    result_type, history = visit(expression, 1)

    if result_type != "numeric":
        _fail("Root expression must produce numeric scores")

    if not fields:
        _fail("Expression must depend on at least one market-data field")

    if history > MAX_HISTORY:
        _fail("Expression requires too much historical data")

    return {
        "dsl_version": DSL_VERSION,
        "node_count": node_count,
        "depth": deepest,
        "required_history_sessions": history,
        "fields": sorted(fields),
        "output_type": result_type,
    }


def canonical_expression(expression):
    """Stable serialization, independent of dictionary key order."""
    validate_expression(expression)
    return json.dumps(
        expression,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def expression_hash(expression):
    """Version-aware structural hash, not algebraic equivalence detection."""
    payload = DSL_VERSION + "\n" + canonical_expression(expression)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_proposal(proposal):
    """Validate the full object an agent will submit."""
    expected = {"name", "hypothesis", "expression", "overfit_risk"}

    if type(proposal) is not dict or set(proposal) != expected:
        _fail("Proposal must contain exactly name, hypothesis, expression, overfit_risk")

    name = proposal["name"]
    if not isinstance(name, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]{2,63}", name
    ):
        _fail("name must be 3-64 lowercase letters/digits/underscores")

    for key in ("hypothesis", "overfit_risk"):
        text = proposal[key]
        if not isinstance(text, str) or not 10 <= len(text.strip()) <= 2000:
            _fail(f"{key} must contain 10-2000 characters")

    return validate_expression(proposal["expression"])
