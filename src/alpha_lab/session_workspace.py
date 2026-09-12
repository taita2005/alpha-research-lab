"""Create an isolated workspace from the deployment research snapshot."""

from pathlib import Path
import shutil
import tempfile

from alpha_lab.research_data import load_research_data


def create_workspace(base_root):
    base_root = Path(base_root)

    # Verify the bundled data against the frozen hashes.
    load_research_data(base_root)

    session_root = Path(tempfile.mkdtemp(prefix="alpha_session_"))

    try:
        for directory in ("src", "configs"):
            shutil.copytree(
                base_root / directory,
                session_root / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        artifacts = session_root / "artifacts"
        artifacts.mkdir()

        shutil.copy2(
            base_root / "artifacts/frozen_selection.json",
            artifacts / "frozen_selection.json",
        )

        # Include the recorded examples.
        for directory in ("v2_requests", "v2_experiments"):
            source = base_root / "artifacts" / directory
            if source.exists():
                shutil.copytree(source, artifacts / directory)

        # Install only pre-holdout data.
        for split in ("warmup", "dev", "val"):
            for field in ("adj_close", "close", "volume"):
                relative = (
                    Path("data/processed")
                    / split
                    / f"{field}.parquet"
                )
                destination = session_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(base_root / relative, destination)

        load_research_data(session_root)
        return session_root

    except Exception:
        shutil.rmtree(session_root, ignore_errors=True)
        raise
