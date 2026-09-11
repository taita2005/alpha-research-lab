# Alpha Research Lab — Research Report

## Research question

Do simple momentum, short-term reversal and volume-conditioned momentum
signals provide useful cross-sectional rankings and portfolio performance
after modeled transaction costs?

This is a small empirical research prototype, not evidence of a novel alpha
or a production trading system.

## Data and protocol

The universe comprises SPY, QQQ, IWM, XLE, XLF, XLK, XLV, XLP and XLY.
Daily data were downloaded from Yahoo Finance using yfinance.

- Warmup: 2009.
- Development: 2010–2017.
- Validation: 2018–2021.
- Holdout: 2022–2025.
- The data snapshot and registered artifacts have recorded SHA-256 hashes.
- Hashes support consistency checks; they do not establish market-data truth.
- The fixed surviving universe introduces potential selection bias.

## Signals and portfolio construction

Momentum uses 126-session adjusted-price returns. Reversal negates
5-session returns. Volume-conditioned momentum retains momentum scores
when 20-session average dollar volume exceeds its 120-session average.

At each decision date, the portfolio selects at most three assets with
positive scores and allocates 95% equally among them. If no asset qualifies,
it remains in cash. Score ties use ascending ticker order.

Decisions occur every 21 sessions, anchored at the first evaluation session
of each split. Targets execute at the next session close. Every split starts
in cash. Passive benchmarks invest 95% once and do not rebalance.

The simulator updates position values using adjusted-price returns.
It is not a simulator of actual shares and separate corporate-action cash
flows. Cash earns zero interest. Base costs are 10 bps per side.
Turnover sums buy and sell notionals divided by pre-trade equity.
No terminal liquidation is forced.

## Signal evidence

Rank IC is the cross-sectional Spearman correlation between scores at t
and adjusted returns from t+1 to t+22. Entry and exit must lie inside the
evaluation split. At least five eligible assets are required.

| Signal | Mean IC | IC dates | Mean asset coverage |
|---|---:|---:|---:|
| momentum_126 | 0.0305 | 1991 | 100.00% |
| reversal_5 | -0.0118 | 1991 | 100.00% |
| volume_momentum_126 | 0.0506 | 944 | 47.27% |

Volume-filtered IC uses a different sample of assets and dates. Its larger
mean IC therefore does not establish a superior signal. The daily forward
labels overlap, so naive independent-observation significance tests are
not reported.

## Development

| Strategy | CAGR | Net Sharpe | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|
| momentum_126 | 7.68% | 0.600 | -20.90% | 7.71 |
| reversal_5 | 7.17% | 0.555 | -28.71% | 12.47 |
| volume_momentum_126 | 4.99% | 0.440 | -30.67% | 11.35 |
| spy_buy_hold | 13.10% | 0.946 | -17.84% | 0.12 |
| equal_weight_buy_hold | 13.56% | 0.953 | -18.01% | 0.12 |

All three candidates underperformed both passive benchmarks on net Sharpe
and CAGR in development.

## Validation and selection

| Strategy | CAGR | Net Sharpe | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|
| momentum_126 | 13.71% | 0.711 | -29.07% | 7.46 |
| reversal_5 | 10.18% | 0.530 | -42.45% | 15.18 |
| volume_momentum_126 | 4.51% | 0.322 | -29.07% | 11.68 |
| spy_buy_hold | 16.43% | 0.866 | -32.40% | 0.24 |
| equal_weight_buy_hold | 15.86% | 0.827 | -32.72% | 0.24 |

The registered rule selected `momentum_126` using the highest validation
net Sharpe among the three candidates at base costs. Benchmarks were
references, not candidates in this selection rule. Selecting a candidate
does not imply that it beat the benchmarks.

The 20-bps sensitivity results are retained in
`artifacts/val_backtest/summary.csv`; they did not change the selection rule.
The selected strategy and registered files were frozen before holdout evaluation.

## Holdout

| Strategy | CAGR | Net Sharpe | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|
| momentum_126 | 12.00% | 0.680 | -21.45% | 6.67 |
| spy_buy_hold | 10.41% | 0.665 | -23.32% | 0.24 |
| equal_weight_buy_hold | 8.51% | 0.588 | -17.71% | 0.24 |

The selected strategy achieved CAGR 12.00% and net Sharpe
0.680. Its CAGR exceeded SPY by
1.58 percentage points and the
equal-weight benchmark by
3.49 percentage points.

Its Sharpe difference versus SPY was
0.016.
This small observed difference does not demonstrate statistical significance.
The selected strategy had a smaller maximum drawdown than SPY but a larger
maximum drawdown than equal-weight buy-and-hold.

These results must be read alongside its weaker development and validation
performance. They do not establish persistent outperformance.

## LLM-assisted review

A Gemini reviewer was added after the experiment. It used one allowlisted,
read-only tool to retrieve five evidence topics: methodology, development IC,
development backtests, validation and holdout.

The recorded successful run used 2 model requests
and 5 tool calls. Tool inputs and outputs were
logged. The agent had no exposed tool for code execution, strategy modification
or rerunning the holdout.

The LLM did not discover the strategy or produce its returns. Its role was
retrospective critique. The original review, an editorially corrected version,
and a correction record are retained separately. Human approval remains pending.

## Validation of implementation

The research modules passed 27 tests covering signals, forward-label timing,
portfolio construction and backtest accounting.
The demo passed 5 Streamlit AppTest tests. Additional notebook checks covered
candidate selection, artifact hashes and the agent tool protocol.

Passing these tests does not establish that every assumption or calculation
is free from error.

## Limitations and future work

- Small, fixed, surviving ETF universe with overlapping exposures.
- No point-in-time universe reconstruction or delisted-fund study.
- No spread, market impact, partial fills, taxes or currency conversion.
- Zero risk-free rate and cash interest.
- No factor attribution or formal outperformance significance test.
- No empirical comparison demonstrating that LLM review improves research outcomes.
- Hosted-model availability and reproducibility are not guaranteed.
- The observed holdout cannot serve as an untouched test for further tuning.

Future work could add broader historical universes, realistic execution
assumptions, dependence-aware statistical analysis and evaluated agent review
quality. These extensions were not executed in this experiment.

## Reproduction scope

The packaged repository can replay the demo from saved artifacts and run
software tests. Full research reproduction additionally requires the original
data and the experiment notebook. A one-command end-to-end research runner
is not included in this MVP.

## References

- [Handbook inspiration](https://www.freecodecamp.org/news/build-a-multi-agent-trading-research-system-with-langchain-deep-agents-handbook)
- [yfinance documentation](https://ranaroussi.github.io/yfinance/)
- [Gemini SDK](https://googleapis.github.io/python-genai/)
- [Streamlit documentation](https://docs.streamlit.io/)

This implementation uses a single Gemini review agent, not LangChain Deep Agents.
