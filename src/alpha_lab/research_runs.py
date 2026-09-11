
"""Read persisted research runs without calling models or experiments."""

import hashlib
import json
import re
from pathlib import Path


class RunIntegrityError(RuntimeError):
    pass


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checked_output(path, expected_hash):
    path = Path(path)
    if not isinstance(expected_hash, str) or not expected_hash:
        raise RunIntegrityError("Missing output hash.")

    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_hash:
        raise RunIntegrityError(f"Output hash mismatch: {path.name}")

    return json.loads(payload)


def _validate_result(root, request_dir, request_id, output):
    if output.get("request_id") != request_id:
        raise RunIntegrityError("Request ID mismatch.")

    result = _read_json(request_dir / "experiment_result.json")
    if output.get("result") != result:
        raise RunIntegrityError("Output differs from saved experiment result.")

    experiment_id = result.get("run_id", "")
    if not re.fullmatch(r"[a-f0-9]{32}", experiment_id):
        raise RunIntegrityError("Invalid experiment ID.")

    if output.get("experiment_id") != experiment_id:
        raise RunIntegrityError("Experiment ID mismatch.")

    engine_result = _read_json(
        root / "artifacts" / "v2_experiments"
        / experiment_id / "result.json"
    )
    if result != engine_result:
        raise RunIntegrityError("Request and engine results differ.")

    proposal = _read_json(request_dir / "proposal.json")
    if result.get("proposal") != proposal:
        raise RunIntegrityError("Saved proposal mismatch.")
    if output.get("proposal") != proposal:
        raise RunIntegrityError("Output proposal mismatch.")
    if output.get("formula") != result.get("formula"):
        raise RunIntegrityError("Formula mismatch.")

    if result.get("holdout_used") is not False:
        raise RunIntegrityError("Expected dev/validation research only.")

    if not isinstance(output.get("critique"), str):
        raise RunIntegrityError("Missing critique.")
    if not output["critique"].strip():
        raise RunIntegrityError("Empty critique.")

    # Current workflow has no human-verification mechanism.
    if output.get("critic_human_verified") is not False:
        raise RunIntegrityError("Unexpected human-verification status.")

    return result


def load_run(root, request_id):
    """Return a UI-friendly view; do not modify the original registry."""
    root = Path(root)

    if not isinstance(request_id, str) or not re.fullmatch(
        r"[a-f0-9]{32}", request_id
    ):
        raise ValueError("Invalid request ID.")

    request_dir = root / "artifacts" / "v2_requests" / request_id
    state = _read_json(request_dir / "state.json")

    view = {
        "request_id": request_id,
        "original_status": state["status"],
        "stage": state.get("stage"),
        "error_type": state.get("error_type"),
        "status": state["status"],
        "review_available": False,
        "output": None,
        "result": None,
        "source": None,
    }

    # Prefer the original completed output if one exists.
    if state["status"] == "completed":
        output_path = request_dir / "output.json"
        output = _checked_output(
            output_path, state.get("output_sha256")
        )
        result = _validate_result(
            root, request_dir, request_id, output
        )
        view.update(
            status="completed",
            review_available=True,
            output=output,
            result=result,
            source=str(output_path.relative_to(root)),
        )
        return view

    # A failed original request may have a completed Critic recovery.
    recovery_dir = request_dir / "critic_recovery"
    progress_path = recovery_dir / "progress.json"

    if progress_path.exists():
        progress = _read_json(progress_path)
        if progress.get("status") == "completed":
            output_path = recovery_dir / "output.json"
            output = _checked_output(
                output_path, progress.get("output_sha256")
            )
            result = _validate_result(
                root, request_dir, request_id, output
            )

            binding = progress.get("binding", {})
            if (
                binding.get("request_id") != request_id
                or binding.get("experiment_id") != result["run_id"]
            ):
                raise RunIntegrityError("Recovery binding mismatch.")

            view.update(
                status="completed_with_recovery",
                review_available=True,
                output=output,
                result=result,
                source=str(output_path.relative_to(root)),
            )
            return view

    # Preserve access to experiment results even if Critic is unavailable.
    result_path = request_dir / "experiment_result.json"
    if result_path.exists():
        view["result"] = _read_json(result_path)

    return view


def list_runs(root):
    """Return summaries; expose corrupt runs as integrity errors."""
    root = Path(root)
    base = root / "artifacts" / "v2_requests"
    if not base.exists():
        return []

    rows = []
    for directory in sorted(base.iterdir(), key=lambda p: p.name):
        if not directory.is_dir():
            continue
        if not re.fullmatch(r"[a-f0-9]{32}", directory.name):
            continue

        try:
            view = load_run(root, directory.name)
            result = view.get("result") or {}
            rows.append({
                "request_id": directory.name,
                "status": view["status"],
                "original_status": view["original_status"],
                "stage": view["stage"],
                "experiment_id": result.get("run_id"),
                "review_available": view["review_available"],
                "decision": result.get("decision", {}).get("status"),
                "error_type": view["error_type"],
            })
        except (
            OSError, ValueError, KeyError, TypeError,
            RunIntegrityError,
        ) as error:
            rows.append({
                "request_id": directory.name,
                "status": "integrity_error",
                "review_available": False,
                "error_type": type(error).__name__,
            })

    return rows
