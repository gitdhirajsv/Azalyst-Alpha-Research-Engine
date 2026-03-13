# Research Workflow

## Step 1 - Data Loading
- **Technically:** Parallel parquet ingestion runs via `ProcessPoolExecutor`. Polars builds wide OHLCV panels while DuckDB handles fast cross-sectional scans and timestamp normalization.
- **In Plain English:** The engine turns raw Binance candles into a research-grade market panel.

## Step 2 - Factor Research (35 Signals)
- **Technically:** `FactorEngineV2` produces 35 cross-sectional factors. `CrossSectionalAnalyser` evaluates Spearman IC, ICIR, Newey-West t-stats, and decay from 1 hour to 1 week.
- **In Plain English:** The platform tests whether momentum, reversal, volatility, liquidity, and microstructure effects actually persist.

## Step 3 - Institutional Validation
- **Technically:** `FactorValidator` neutralizes BTC beta, size, and liquidity effects, then applies Fama-MacBeth regressions with Benjamini-Hochberg false discovery rate control.
- **In Plain English:** Signals must survive after the obvious market effects are stripped away.

## Step 4 - Statistical Arbitrage
- **Technically:** Engle-Granger cointegration testing, Hurst validation, half-life estimation, and live spread z-score monitoring.
- **In Plain English:** The engine finds pairs that should move together and watches for temporary dislocations.

## Step 5 - Machine Learning Layer
- **Technically:** LightGBM with optional CUDA acceleration trains with purged time-series CV. Core models include `PumpDumpDetector`, `ReturnPredictor`, and a four-state `RegimeDetector`.
- **In Plain English:** The research stack learns changing market structure without leaking future information.

## Step 6 - Walk-Forward Simulation
- **Technically:** Rolling train-test windows retrain every 30 days, fit scalers only on training data, and simulate entries at the next bar open with fees and checkpoints.
- **In Plain English:** The platform replays history in a live-like sequence rather than relying on a static backtest.
