# Alpha Research Lab

LLM-assisted alpha research with a restricted expression DSL,
a deterministic backtest engine, and an evidence-based Critic.

## Live demo

Entry point: app/research_workspace.py

Enable live research in Streamlit app settings / Secrets:

    ALPHA_ENABLE_LIVE_RESEARCH = true

Visitors enter their own Gemini API key in the password field.
The application does not save that key in research artifacts.

The deployment includes the verified warmup, development and
validation snapshot needed by the current engine.
Holdout prices are not included.

## Workflow

1. Enter a research question.
2. Generate and validate a structured alpha proposal.
3. Run dev/validation experiments with fixed execution rules.
4. Apply deterministic selection gates.
5. Request a Critic review and download results.

Each browser session receives its own temporary workspace.
Download new results before leaving the session.
This is a personal research demo, not a durable multi-user service.

A request may fail if the external model is unavailable.
The engine result is retained if failure occurs at the Critic stage.
Automatic Critic recovery is not yet integrated into the web interface.

## Research limitations

Development and validation are exploratory datasets.
The researcher has already observed the v1 holdout.
Results do not establish statistical significance or deployment readiness.
LLM reviews require human factual verification.

The recorded volatility-adjusted momentum candidate improved validation
Sharpe versus raw momentum, but remained below the passive benchmarks.

## Local usage

Use Python 3.13 and install requirements.txt.

Recorded mode:

    streamlit run app/research_workspace.py

Live mode on Linux/macOS:

    ALPHA_ENABLE_LIVE_RESEARCH=1 streamlit run app/research_workspace.py

Request IDs prevent replaying the same request.
Equivalent formulas under different request IDs are not yet deduplicated.
