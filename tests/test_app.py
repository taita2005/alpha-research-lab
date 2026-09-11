from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/app.py"


@pytest.fixture
def app():
    instance = AppTest.from_file(
        str(APP), default_timeout=60
    ).run()

    assert len(instance.exception) == 0
    assert len(instance.error) == 0
    return instance


def assert_clean(app):
    assert len(app.exception) == 0
    assert len(app.error) == 0


def test_initial_page(app):
    assert app.title[0].value == "Alpha Research Lab"
    assert len(app.tabs) == 4

    table = pd.read_csv(
        ROOT / "artifacts/holdout/summary.csv"
    ).set_index("strategy")

    import json
    manifest = json.loads(
        (ROOT / "artifacts/demo_manifest.json").read_text()
    )
    selected = manifest["selected_strategy"]

    assert app.metric[0].value == f"{table.loc[selected, 'cagr']:.2%}"


def test_validation_cost_filter(app):
    app.selectbox(key="period").select("Validation").run()
    app.selectbox(key="cost_bps").select(20.0).run()
    assert_clean(app)

    # Overview table is first; research-period table is second.
    displayed = app.dataframe[1].value

    assert len(displayed) == 5
    assert (displayed["cost_bps"] == 20.0).all()


def test_holdout_contains_only_frozen_candidate_and_benchmarks(app):
    app.selectbox(key="period").select("Holdout").run()
    assert_clean(app)

    displayed = app.dataframe[1].value
    assert len(displayed) == 3


def test_review_version_switch(app):
    app.radio(key="review_version").set_value("Original LLM").run()
    assert_clean(app)
    assert app.radio(key="review_version").value == "Original LLM"

    app.radio(key="review_version").set_value("Edited").run()
    assert_clean(app)
    assert any(
        "Editorial status:" in element.value
        for element in app.markdown
    )


def test_tool_evidence_selection(app):
    app.selectbox(key="tool_index").select(4).run()
    assert_clean(app)
    assert app.selectbox(key="tool_index").value == 4
