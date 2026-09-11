"""Artifact-driven research demo. No live model calls or backtests."""

from pathlib import Path
import hashlib
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Alpha Research Lab",
    page_icon="📊",
    layout="wide",
)


def load_bundle():
    manifest_path = ROOT / "artifacts/demo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = {}

    for name, entry in manifest["sources"].items():
        path = (ROOT / entry["path"]).resolve()

        if not path.is_relative_to(ROOT):
            raise ValueError("Artifact path is outside the project")

        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()

        if actual_hash != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")

        files[name] = payload.decode("utf-8")

    return manifest, files


try:
    manifest, files = load_bundle()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Unable to load the registered demo artifacts: {error}")
    st.stop()


def read_table(name):
    from io import StringIO
    return pd.read_csv(StringIO(files[name]))


def metric_table(frame):
    """Convert decimal fractions to percentages for display only."""
    output = frame.copy()

    for column in [
        "total_return", "cagr", "annual_volatility",
        "max_drawdown", "mean_cash_fraction",
    ]:
        if column in output.columns:
            output[column] = output[column] * 100
            output = output.rename(columns={column: column + "_pct"})

    return output.round(3)


holdout = read_table("holdout_backtest").set_index("strategy")
config = json.loads(files["research_config"])
data_config = json.loads(files["data_config"])
agent_metadata = json.loads(files["agent_metadata"])

trace = [
    json.loads(line)
    for line in files["agent_trace"].splitlines()
    if line.strip()
]
tool_events = [event for event in trace if event.get("event") == "tool"]
request_events = [
    event for event in trace if event.get("event") == "request"
]

selected = manifest["selected_strategy"]
selected_metrics = holdout.loc[selected]

st.title("Alpha Research Lab")
st.caption("Reproducible signal research with an LLM-assisted evidence review")

st.sidebar.header("Study snapshot")
st.sidebar.write("Universe: 9 US equity ETFs")
st.sidebar.write(f"Selected strategy: `{selected}`")
st.sidebar.write(f"Reviewer model: `{agent_metadata['model']}`")
st.sidebar.caption(
    "Recorded experiment and agent run. "
    "This interface does not place trades or call an LLM."
)

overview, results, review_tab, trace_tab = st.tabs([
    "Overview",
    "Research Results",
    "Agent Review",
    "Agent Trace",
])

with overview:
    st.subheader("Research question")
    st.write(
        "Do simple momentum, reversal and volume-conditioned signals "
        "produce useful rankings and portfolio performance after modeled costs?"
    )

    st.caption(
        "Three candidates were compared on validation net Sharpe. "
        "The selected candidate was frozen before the final holdout evaluation."
    )

    holdout_start, holdout_end = data_config["splits"]["holdout"]
    st.subheader(f"Holdout: {holdout_start} to {holdout_end}")

    columns = st.columns(4)
    columns[0].metric(
        "Selected strategy CAGR",
        f"{selected_metrics['cagr']:.2%}",
    )
    columns[1].metric(
        "Net Sharpe",
        f"{selected_metrics['net_sharpe']:.3f}",
    )
    columns[2].metric(
        "Maximum drawdown",
        f"{selected_metrics['max_drawdown']:.2%}",
    )
    columns[3].metric(
        "Annual turnover",
        f"{selected_metrics['annual_turnover']:.2f}×",
    )

    histories = {}
    for strategy in holdout.index:
        frame = read_table(f"holdout_history:{strategy}")
        frame["date"] = pd.to_datetime(frame["date"])
        histories[strategy] = frame.set_index("date")["equity"]

    equity = pd.concat(histories, axis=1)
    growth = equity / equity.iloc[0]
    drawdown_pct = (equity / equity.cummax() - 1) * 100

    st.write("**Growth of one accounting unit**")
    st.line_chart(growth)

    st.write("**Drawdown (%)**")
    st.line_chart(drawdown_pct)

    st.dataframe(metric_table(holdout))

    st.info(
        "Holdout results do not establish statistically significant alpha. "
        "The selected strategy underperformed the passive benchmarks on "
        "Sharpe in development and validation."
    )

    with st.expander("Method and limitations"):
        st.write(
            "Adjusted-return portfolio simulation; next-session-close "
            "execution; proportional trading costs; zero cash interest; "
            "each split starts in cash. No spread, market impact, partial "
            "fills or factor attribution are modeled. ETF exposures overlap "
            "and the fixed surviving universe introduces selection concerns."
        )
        st.json(config)

with results:
    st.subheader("Backtest results")

    period = st.selectbox(
        "Evaluation period",
        ["Development", "Validation", "Holdout"],
        key="period",
    )

    source_key = {
        "Development": "dev_backtest",
        "Validation": "val_backtest",
        "Holdout": "holdout_backtest",
    }[period]

    table = read_table(source_key)

    if period == "Validation":
        costs = sorted(table["cost_bps"].unique().tolist())
        chosen_cost = st.selectbox(
            "Cost per side (bps)", costs, key="cost_bps"
        )
        table = table.loc[table["cost_bps"] == chosen_cost]

    st.dataframe(metric_table(table.set_index("strategy")))

    st.caption(
        "Percentage columns are labeled _pct. Sharpe is unitless. "
        "Turnover sums purchases plus sales. "
        "Only the frozen candidate and two benchmarks were evaluated on holdout."
    )

    st.subheader("Development signal evidence")
    ic_table = read_table("dev_ic")
    st.dataframe(ic_table.set_index("candidate").round(4))

    st.caption(
        "Rank IC is cross-sectional Spearman correlation with forward returns. "
        "Filtered IC uses different asset/date coverage. "
        "Forward labels overlap; no naive independent-sample p-values are reported."
    )

with review_tab:
    st.subheader("Post-experiment LLM review")
    st.write(
        "The agent retrieved registered evidence and wrote a retrospective "
        "critique. It did not create the strategy or its returns."
    )

    st.info(
        "Editorial corrections have been applied. "
        "Human approval is still pending."
    )

    version = st.radio(
        "Review version",
        ["Edited", "Original LLM"],
        key="review_version",
        horizontal=True,
    )

    review_key = (
        "agent_review_edited"
        if version == "Edited"
        else "agent_review_original"
    )
    st.markdown(files[review_key])

    with st.expander("Editorial change record"):
        st.json(json.loads(files["agent_editorial_record"]))

    st.caption(
        "Evidence tags refer to registered methodology and result tables. "
        "Presence of a tag does not automatically validate the associated claim."
    )

with trace_tab:
    st.subheader("Recorded agent execution")

    columns = st.columns(3)
    columns[0].metric("Model requests", str(len(request_events)))
    columns[1].metric("Tool calls", str(len(tool_events)))
    columns[2].metric(
        "Topics retrieved",
        str(len({
            event["args"]["topic"]
            for event in tool_events
            if "result" in event.get("response", {})
        })),
    )

    st.caption(f"Recorded run: {manifest['agent_run_id']}")

    event_rows = []
    for index, event in enumerate(trace):
        event_rows.append({
            "step": index + 1,
            "timestamp_utc": event.get("timestamp_utc"),
            "event": event.get("event"),
            "tool": event.get("name", ""),
            "topic": event.get("args", {}).get("topic", ""),
            "finish_reason": event.get("finish_reason", ""),
        })
    st.dataframe(pd.DataFrame(event_rows), hide_index=True)

    if tool_events:
        tool_index = st.selectbox(
            "Inspect tool evidence",
            list(range(len(tool_events))),
            format_func=lambda index: (
                f"{index + 1}. "
                f"{tool_events[index]['args'].get('topic', 'unknown')}"
            ),
            key="tool_index",
        )
        st.json(tool_events[tool_index]["response"])

    with st.expander("Token usage reported by the provider"):
        usages = [
            {"request": event.get("number"), "usage": event.get("usage")}
            for event in trace if event.get("event") == "usage"
        ]
        st.json(usages)

    with st.expander("Artifact manifest"):
        st.json(manifest)
        st.caption(
            "Checksums verify consistency with this manifest, not the truth "
            "of market data or the correctness of every LLM statement."
        )

st.divider()
st.caption(
    "Research prototype · Recorded results · "
    "No live trading · No API key required to view"
)
