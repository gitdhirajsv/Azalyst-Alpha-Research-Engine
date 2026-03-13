# Research Workflow (Branch3 flavor)

## Step 1 — Data Loading
- **Technically:** Parquet ingestion runs in parallel via `ProcessPoolExecutor`. Polars builds wide OHLCV panels while DuckDB provides fast, cross-sectional SQL scans; timestamps are normalized and resampled (5m → 1h) as needed.
- **In Plain English:** Reads every Binance 5-minute candle across 400+ symbols for 3 years and organizes the history into a massive, queryable table.

## Step 2 — Factor Research (35 Signals)
- **Technically:** `FactorEngineV2` produces 35 cross-sectional factors. `CrossSectionalAnalyser` calculates Spearman ICs (1h–1w), ICIR, Newey-West t-stats, and decay curves to rank signal durability.
- **In Plain English:** Tests momentum, reversal, volatility, liquidity, microstructure, and technical hypotheses to find signals that consistently beat randomness.

## Step 3 — Institutional Validation
- **Technically:** `FactorValidator` neutralizes BTC beta, size, and liquidity; runs Fama-MacBeth regressions and applies the Benjamini-Hochberg false discovery rate correction.
- **In Plain English:** Strips out market noise and statistical flukes so that only truly independent alpha survives.

## Step 4 — Statistical Arbitrage (Pairs Trading)
- **Technically:** Runs Engle-Granger cointegration across symbol pairs, validating with Hurst and half-life metrics; live z-scores track mean reversion triggers.
- **In Plain English:** Finds coin pairs that normally move together and bets on the gap getting fixed—market neutral trades ideal for risk control.

## Step 5 — Machine Learning v4.0
- **Technically:** LightGBM with CUDA trains via purged time-series CV. PumpDumpDetector spots pump patterns, ReturnPredictor forecasts the next 4h direction, RegimeDetector uses a 4-state GMM on BTC/breadth.
- **In Plain English:** Trade signals learn over time and adapt to Bull/Bear/High-Vol/Quiet regimes, all while purging lookahead bias.

## Step 6 — Walk-Forward Simulation
- **Technically:** Rolling-window simulation retrains every 30 days; scalers fit only on train data; trades execute at the next bar’s open with 0.1% taker fees.
- **In Plain English:** The engine replays history like a time machine—train on a year, test the next month, slide forward, repeat—mirroring live trading.
