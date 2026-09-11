## Executive Assessment

This post-experiment audit evaluates a systematic trading research study across three candidate signals (`momentum_126`, `reversal_5`, and `volume_momentum_126`) applied to a portfolio of nine surviving US equity ETFs [methodology]. Performance was tracked across three chronological evaluation splits: development, validation, and holdout [methodology]. `momentum_126` was selected based on validation net Sharpe ratio [methodology]. In the holdout period, `momentum_126` achieved a CAGR of 0.119978 and a net Sharpe of 0.680390, compared to `spy_buy_hold` (CAGR 0.104129, net Sharpe 0.664887) and `equal_weight_buy_hold` (CAGR 0.085116, net Sharpe 0.588197) [holdout]. SHA-256 hash checks verify artifact file integrity across all outputs [dev_ic], [dev], [val], [holdout], but establish data consistency rather than underlying market-data truth [methodology].

## Signal Evidence and Coverage

Signal predictive power was evaluated in development via cross-sectional Spearman IC using 21-session forward returns [methodology]. 

* `momentum_126` demonstrated a mean IC of 0.030462 over 1,991 evaluation dates, achieving a 1.000000 date and mean asset coverage fraction [dev_ic].
* `reversal_5` recorded a mean IC of -0.011803 over 1,991 dates [dev_ic].
* `volume_momentum_126` exhibited a higher mean IC of 0.050562, but its date coverage fraction was reduced to 0.474134 (944 dates) and mean asset coverage fell to 0.472683 [dev_ic].

The volume filter in `volume_momentum_126` excludes assets and dates that fail the liquidity criteria. As a result, its IC is measured on a distinct, filtered subset of assets and dates, making direct IC comparisons against full-coverage signals improper without controlling for sample differences [methodology].

## Development and Validation

During development, passive benchmarks outperformed all active strategies on a risk-adjusted basis [dev]. `equal_weight_buy_hold` attained a CAGR of 0.135611 and a net Sharpe of 0.953044, while `spy_buy_hold` reached a CAGR of 0.130993 and net Sharpe of 0.945698 [dev]. `momentum_126` led active candidates with a CAGR of 0.076776 and net Sharpe of 0.600176, whereas `reversal_5` (CAGR 0.071674, net Sharpe 0.555002) and `volume_momentum_126` (CAGR 0.049944, net Sharpe 0.440292) lagged [dev].

In validation at baseline 10 bps costs, benchmarks maintained superior net Sharpe ratios (`spy_buy_hold` CAGR 0.164256, net Sharpe 0.866104; `equal_weight_buy_hold` CAGR 0.158599, net Sharpe 0.827317) [val]. `momentum_126` registered a CAGR of 0.137056 and net Sharpe of 0.710927 [val]. Under 20 bps cost stress, `momentum_126` net Sharpe dropped to 0.676084 due to trading friction from an annual turnover of 7.455258 [val]. `momentum_126` was frozen for holdout evaluation as the top candidate [methodology].

## Holdout Performance and Trade-Offs

In the out-of-sample holdout split (10 bps cost), `momentum_126` produced a CAGR of 0.119978, net Sharpe of 0.680390, annual volatility of 0.194262, and max drawdown of -0.214530 [holdout]. Benchmark comparisons were:
* `spy_buy_hold`: CAGR 0.104129, net Sharpe 0.664887, annual volatility 0.170902, max drawdown -0.233204 [holdout].
* `equal_weight_buy_hold`: CAGR 0.085116, net Sharpe 0.588197, annual volatility 0.160880, max drawdown -0.177144 [holdout].

Although `momentum_126` delivered higher nominal return than both benchmarks in holdout, the net Sharpe difference relative to SPY (0.680390 vs 0.664887) is small and does not demonstrate statistical significance [methodology]. Higher turnover (6.666952 per year vs benchmark 0.238695) accumulated 3,228.26 in total fees [holdout].

## Limitations

1. **Universe & Survivorship**: The nine-ETF universe lacks non-surviving historical funds, introducing potential selection bias [methodology].
2. **Asset Independence**: ETF holdings overlap significantly, violating asset independence assumptions [methodology].
3. **Execution Realism**: Simulation uses adjusted returns and fixed proportional fees without modeling bid-ask spread, market impact, or partial fills [methodology].
4. **Rate Assumptions**: Risk-free rate and cash interest are fixed at zero [methodology].
5. **Overlapping Labels**: 21-session forward returns introduce serial correlation in IC series [methodology].
6. **Missing Attribution**: No risk factor decomposition or benchmark beta attribution was performed [methodology].

## Future Research — Not Executed

To preserve research protocol integrity, no hyperparameter tuning or model adjustments were executed after observing holdout data [methodology]. Recommended future research steps include:
1. Testing on an expanded historical universe including delisted ETFs.
2. Incorporating market impact and bid-ask spread models into execution mechanics.
3. Conducting multi-factor exposure attribution (e.g., Fama-French/Carhart).
4. Applying statistical significance tests for risk-adjusted performance across market regimes.