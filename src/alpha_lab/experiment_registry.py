"""Persistent request registry with bounded admission and replay."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4


MAX_REQUESTS = 10


def digest(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RequestRegistry:
    def __init__(self, root):
        self.base = Path(root) / "artifacts/v2_requests"
        self.base.mkdir(parents=True, exist_ok=True)

    def directory(self, request_id):
        if not isinstance(request_id, str) or not re.fullmatch(
            r"[a-f0-9]{32}", request_id
        ):
            raise ValueError("request_id must be a UUID hex string")
        return self.base / request_id

    @contextmanager
    def admission_lock(self):
        lock = self.base / ".admission_lock"
        try:
            lock.mkdir()
        except FileExistsError:
            raise RuntimeError(
                "Registry busy or interrupted lock remains; inspect before retry."
            ) from None
        try:
            yield
        finally:
            lock.rmdir()

    def reserve(self, request_id, signature):
        """Return cached output or reserve a new request."""
        directory = self.directory(request_id)

        with self.admission_lock():
            if directory.exists():
                state = json.loads(
                    (directory / "state.json").read_text(encoding="utf-8")
                )

                if state["signature"] != signature:
                    raise ValueError(
                        "request_id reused with different input or code/data/config"
                    )

                if state["status"] == "completed":
                    output_path = directory / "output.json"
                    payload = output_path.read_bytes()

                    if hashlib.sha256(payload).hexdigest() != state["output_sha256"]:
                        raise RuntimeError("Cached output changed")

                    return json.loads(payload)

                raise RuntimeError(
                    "Request already exists with status: " + state["status"]
                    + ". Automatic rerun is disabled."
                )

            count = sum(
                1 for path in self.base.iterdir()
                if path.is_dir() and re.fullmatch(r"[a-f0-9]{32}", path.name)
            )
            if count >= MAX_REQUESTS:
                raise RuntimeError("Workspace request budget exhausted")

            directory.mkdir()
            atomic_json(directory / "state.json", {
                "request_id": request_id,
                "signature": signature,
                "status": "running",
                "stage": "reserved",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            return None

    def update(self, request_id, stage, **extra):
        path = self.directory(request_id) / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["status"] != "running":
            raise RuntimeError("Cannot update a non-running request")
        state.update(extra)
        state["stage"] = stage
        atomic_json(path, state)

    def complete(self, request_id, output):
        directory = self.directory(request_id)
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))

        if state["status"] != "running":
            raise RuntimeError("Cannot complete a non-running request")

        output_path = directory / "output.json"
        atomic_json(output_path, output)

        state.update({
            "status": "completed",
            "stage": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        })
        atomic_json(state_path, state)

    def fail(self, request_id, error):
        path = self.directory(request_id) / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update({
            "status": "failed",
            "error_type": type(error).__name__,
        })
        atomic_json(path, state)
