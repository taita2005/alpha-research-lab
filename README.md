# Alpha Research Lab

A research prototype for evaluating simple ETF signals and auditing
the results with a tool-calling LLM reviewer.

## What it does

- Evaluates momentum, reversal and volume-conditioned momentum.
- Measures cross-sectional rank IC and signal coverage.
- Simulates long-only portfolios with delayed execution and transaction costs.
- Selects a candidate on validation and records a frozen holdout evaluation.
- Runs a read-only Gemini reviewer against registered evidence.
- Presents saved results, reviews and tool traces in a Streamlit app.

## Main result

The frozen `momentum_126` strategy achieved holdout CAGR
**12.00%** and net Sharpe **0.680**.

It underperformed passive benchmarks on CAGR and Sharpe in development
and validation. Its holdout result does not establish statistically
significant or persistent alpha.

See [the research report](reports/research_report.md).

## Run the demo

Use a Python environment compatible with the recorded dependencies.
The original research ran on Python 3.13.

```bash
python -m venv .venv
```

Activate the environment using your operating system's usual command, then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

The demo reads saved CSV, JSON and Markdown artifacts.
No API key, raw price download or live LLM call is needed to view it.

## Run tests

Install both the demo and research dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-research.txt
python -m pytest tests -q
```

The tests included in this package cover the research modules and the UI.
Additional agent and selection checks were executed in the development notebook.

## Agent

The recorded reviewer used `gemini-3.6-flash` with an explicit
function-calling loop.

Its only exposed tool is `get_evidence(topic)`. It can read five registered
evidence topics. It cannot modify strategies or run arbitrary code through
the provided tool interface.

The agent was added after the research experiment. It did not generate the
strategy or its performance.

To develop or rerun the reviewer, install `requirements-agent.txt` and
provide your own Gemini credentials securely. Provider access and quota
are separate from the demo.

## Structure

- `src/alpha_lab/`: signals, evaluation, backtest, selection and reviewer.
- `configs/`: data and research rules.
- `tests/`: research and application tests.
- `app/app.py`: Streamlit entrypoint.
- `artifacts/`: saved metrics, histories, review and audit records.
- `reports/research_report.md`: research findings and limitations.

## Reproducibility

This package supports artifact replay and software tests.
It does not yet provide a one-command reproduction of the entire study.
Raw market data are excluded. Registered historical hashes may refer to
files retained in the original research workspace rather than this package.

The demo verifies its own listed artifact checksums against
`artifacts/demo_manifest.json`.

## Limitations

This is a nine-ETF research prototype, not a trading service.
It does not model realistic order-book execution or establish novel alpha.
Original LLM output remains available alongside editorial corrections;
human approval is pending.

## Data and publication

Raw Yahoo Finance price data are not included.
Review applicable data-use terms before publicly distributing derived
artifacts. This repository package is prepared locally; packaging itself
does not authorize public redistribution.
