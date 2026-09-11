
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from alpha_lab import workspace_service


ROOT = Path(os.environ["ALPHA_PROJECT_ROOT"])
APP = ROOT / "app/research_workspace.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("ALPHA_ENABLE_LIVE_RESEARCH", "1")
    instance = AppTest.from_file(str(APP), default_timeout=30)
    instance.run()
    assert not instance.exception
    return instance


def run_button(app):
    return next(
        button for button in app.button
        if button.label == "Run research"
    )


def test_saved_recovery_is_visible(app):
    app.selectbox(key="selected_research_run").select(
        "2a61aa8a65b448219249785ff87b9e19"
    ).run()

    assert not app.exception
    assert any(
        "completed_with_recovery" in element.value
        for element in app.caption
    )
    assert any(
        "safe_divide" in element.value
        for element in app.code
    )


def test_filters_do_not_call_backend(app):
    with patch.object(workspace_service, "execute_request") as execute:
        app.selectbox(key="research_split").select("val").run()
        app.selectbox(key="research_cost").select(20.0).run()
        app.button(key="refresh_results").click().run()

    assert not app.exception
    execute.assert_not_called()


def test_missing_key_does_not_submit(app):
    with patch.object(workspace_service, "execute_request") as execute:
        run_button(app).click().run()

    assert not app.exception
    execute.assert_not_called()
    assert any(
        "Enter your Gemini API key" in element.value
        for element in app.error
    )


def test_submit_once_and_rerun_without_duplicate(app):
    app.text_input(key="workspace_api_key").input("fake-test-key")

    with patch.object(
        workspace_service,
        "execute_request",
        return_value={"software_test": True},
    ) as execute:
        run_button(app).click().run()
        assert execute.call_count == 1
        first_id = execute.call_args.kwargs["request_id"]

        app.button(key="refresh_results").click().run()
        assert execute.call_count == 1
        assert run_button(app).disabled

        app.button(key="prepare_request").click().run()
        assert app.session_state["workspace_request_id"] != first_id
        assert not run_button(app).disabled

    assert not app.exception


def test_backend_failure_does_not_auto_retry(app):
    app.text_input(key="workspace_api_key").input("fake-test-key")

    with patch.object(
        workspace_service,
        "execute_request",
        side_effect=RuntimeError("Simulated failure"),
    ) as execute:
        run_button(app).click().run()
        app.button(key="refresh_results").click().run()
        assert execute.call_count == 1

    assert not app.exception
    assert app.session_state["workspace_submitted"] is True
