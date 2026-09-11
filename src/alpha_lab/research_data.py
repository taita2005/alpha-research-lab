"""Load only pre-holdout data for exploratory v2 experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_CUTOFF = pd.Timestamp("2021-12-31")
FILES = ("adj_close.parquet", "close.parquet", "volume.parquet")


def check_panels(panels, cutoff=DATA_CUTOFF):
    required = {"adj_close", "close", "volume"}
    if set(panels) != required:
        raise ValueError("Expected adj_close, close and volume")

    reference = panels["adj_close"]

    for name, frame in panels.items():
        if frame.empty:
            raise ValueError(f"Empty panel: {name}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex required")
        if (
            frame.index.hasnans
            or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing
        ):
            raise ValueError("Dates must be valid, unique and sorted")
        if frame.index.max() > pd.Timestamp(cutoff):
            raise ValueError("Data extends beyond the v2 cutoff")
        if (
            not frame.index.equals(reference.index)
            or not frame.columns.equals(reference.columns)
        ):
            raise ValueError("Panel axes differ")
        if frame.columns.has_duplicates:
            raise ValueError("Duplicate assets")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Missing or infinite input data")
        if name == "volume":
            if (values < 0).any():
                raise ValueError("Negative volume")
        elif (values <= 0).any():
            raise ValueError("Nonpositive prices")


def load_research_data(root):
    root = Path(root)
    frozen = json.loads(
        (root / "artifacts/frozen_selection.json").read_text()
    )
    frames = {name.replace(".parquet", ""): [] for name in FILES}
    source_hashes = {}

    # No raw/full-history/holdout price files are opened here.
    for split in ("warmup", "dev", "val"):
        for filename in FILES:
            relative = f"data/processed/{split}/{filename}"
            path = root / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            if frozen["file_hashes"].get(relative) != digest:
                raise ValueError(f"Unverified data file: {relative}")

            frame = pd.read_parquet(path)
            frames[filename.replace(".parquet", "")].append(frame)
            source_hashes[relative] = digest

    panels = {
        name: pd.concat(blocks).sort_index()
        for name, blocks in frames.items()
    }
    check_panels(panels)
    return panels, source_hashes
