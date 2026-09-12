import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(
    os.environ.get(
        "ALPHA_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_lab import research_runs, workspace_service

st.set_page_config(
    page_title="Alpha Research Workspace",
    page_icon="🔬",
    layout="wide",
)

st.title("Alpha Research Workspace")
st.caption(
    "Proposal → validated formula → fixed experiment engine → "
    "Python decision → Critic"
)
st.warning(
    "Exploratory research using development and validation data. "
    "Results are not independent test evidence. "
    "LLM reviews require human verification."
)

# LIVE_WORKSPACE_V1
# An explicit environment setting takes precedence over Secrets.
live_setting = os.environ.get("ALPHA_ENABLE_LIVE_RESEARCH")

if live_setting is None:
    try:
        live_setting = st.secrets.get(
            "ALPHA_ENABLE_LIVE_RESEARCH", False
        )
    except FileNotFoundError:
        live_setting = False

live_enabled = str(live_setting).lower() in ("1", "true")

if live_enabled:
    from alpha_lab.session_workspace import create_workspace

    if "live_workspace_path" not in st.session_state:
        try:
            with st.spinner("Preparing your research workspace..."):
                session_root = create_workspace(ROOT)
            st.session_state.live_workspace_path = str(session_root)
        except Exception as error:
            st.error(
                "Research data preparation failed: "
                + type(error).__name__
            )
            st.stop()

    ROOT = Path(st.session_state.live_workspace_path)

    st.caption(
        "Live research enabled. New results belong to this browser "
        "session. Download results before leaving."
    )

if "workspace_request_id" not in st.session_state:
    st.session_state.workspace_request_id = uuid.uuid4().hex
if "workspace_submitted" not in st.session_state:
    st.session_state.workspace_submitted = False

st.subheader("Create a research request")

if not live_enabled:
    st.info(
        "Recorded-results mode. Live research is disabled on this server."
    )

if st.button(
    "Prepare another request",
    key="prepare_request",
    disabled=not st.session_state.workspace_submitted,
):
    st.session_state.workspace_request_id = uuid.uuid4().hex
    st.session_state.workspace_submitted = False

st.caption("Request ID: " + st.session_state.workspace_request_id)


if st.button("Clear API key", key="clear_workspace_key"):
    st.session_state["workspace_api_key"] = ""

with st.form("research_form"):
    user_request = st.text_area(
        "Research question",
        value=(
            "Propose one simple structural improvement to 126-session "
            "momentum. Explain the hypothesis and overfitting risk. "
            "Use only the allowed DSL."
        ),
        max_chars=2000,
        key="research_question",
    )
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        key="workspace_api_key",
        help="Used for this session. Do not include it in the research question.",
    )

    submitted = st.form_submit_button(
        "Run research",
        disabled=(
            not live_enabled
            or st.session_state.workspace_submitted
        ),
    )

if submitted:
    question = user_request.strip()
    key = api_key.strip()

    if len(question) < 10:
        st.error("Enter a research question with at least 10 characters.")
    elif not key:
        st.error("Enter your Gemini API key.")
    elif key in question:
        st.error("Remove the API key from the research question.")
    else:
        # Mark before calling the backend: a later rerun must not resubmit.
        st.session_state.workspace_submitted = True
        request_id = st.session_state.workspace_request_id

        try:
            with st.spinner(
                "Generating a proposal, running dev/validation, "
                "and requesting the Critic..."
            ):
                workspace_service.execute_request(
                    root=ROOT,
                    request_id=request_id,
                    user_request=question,
                    api_key=key,
                )
            st.success("Research completed. Results are available below.")
        except Exception as error:
            # Do not render raw SDK errors that could contain request details.
            st.error(
                f"Research stopped: {type(error).__name__}. "
                f"Request ID: {request_id}. "
                "Inspect saved results before starting another request."
            )

if st.session_state.workspace_submitted:
    st.info(
        "This request has already been submitted. Changing filters or "
        "refreshing widgets will not run it again."
    )

st.divider()
st.subheader("Research history")

# Button causes a rerun; no API call is attached to it.
st.button("Refresh saved results", key="refresh_results")

rows = research_runs.list_runs(ROOT)

if not rows:
    st.info("No saved research requests yet.")
    st.stop()

st.dataframe(pd.DataFrame(rows), hide_index=True)

valid_ids = [
    row["request_id"]
    for row in rows
    if row["status"] != "integrity_error"
]
if not valid_ids:
    st.error("No readable requests. Inspect artifact integrity errors.")
    st.stop()

current_id = st.session_state.workspace_request_id
default_index = (
    valid_ids.index(current_id)
    if current_id in valid_ids
    else 0
)

selected_id = st.selectbox(
    "Open request",
    valid_ids,
    index=default_index,
    key="selected_research_run",
)

try:
    view = research_runs.load_run(ROOT, selected_id)
except Exception as error:
    st.error(f"Unable to verify this request: {type(error).__name__}")
    st.stop()

st.caption(
    f"Effective status: {view['status']} | "
    f"Original status: {view['original_status']}"
)

if view["status"] == "completed_with_recovery":
    st.info(
        "The original request failed. Its Critic review was subsequently "
        "recovered using the saved experiment."
    )

result = view["result"]
if result is None:
    st.info("This request has no saved experiment result.")
    st.stop()

proposal = result.get("proposal", {})
st.subheader(proposal.get("name", "Candidate"))
st.code(result.get("formula", ""), language="text")
st.write("**Hypothesis:**", proposal.get("hypothesis", ""))
st.write("**Overfitting risk:**", proposal.get("overfit_risk", ""))

with st.expander("Structured proposal"):
    st.json(proposal)

decision = result.get("decision", {})
st.subheader("Deterministic decision")
st.write("Status:", decision.get("status"))
st.json(decision.get("gates", {}))
st.caption(result.get("interpretation", ""))

metrics = pd.DataFrame(result.get("metrics", []))
if not metrics.empty:
    st.subheader("Performance comparison")
    left, right = st.columns(2)

    with left:
        split = st.selectbox(
            "Split",
            ["dev", "val"],
            key="research_split",
        )
    with right:
        costs = sorted(metrics["cost_bps"].dropna().unique().tolist())
        cost = st.selectbox(
            "Transaction cost (bps)",
            costs,
            key="research_cost",
        )

    table = metrics.loc[
        metrics["split"].eq(split) & metrics["cost_bps"].eq(cost)
    ].copy()

    for column in (
        "total_return", "cagr", "annual_volatility",
        "max_drawdown", "mean_cash_fraction",
    ):
        if column in table:
            table[column + "_pct"] = table.pop(column) * 100

    preferred = [
        "strategy", "cagr_pct", "net_sharpe",
        "annual_volatility_pct", "max_drawdown_pct",
        "annual_turnover", "total_fees",
    ]
    st.dataframe(
        table[[c for c in preferred if c in table]].round(4),
        hide_index=True,
    )

with st.expander("Information coefficient results"):
    st.dataframe(pd.DataFrame(result.get("ic", [])), hide_index=True)

st.subheader("Critic review")
if view["review_available"]:
    st.warning(
        "AI-generated interpretation; not yet factually verified by a human."
    )
    st.markdown(view["output"]["critique"])

    st.download_button(
        "Download research result and review",
        data=json.dumps(
            view["output"],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        file_name=f"research_{selected_id}.json",
        mime="application/json",
    )
else:
    st.warning(
        "The experiment result is available, but the Critic review is "
        "incomplete. Recover the Critic from the saved result."
    )
