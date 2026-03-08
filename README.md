# Azalyst Alpha Research Engine

> **An institutional-style quantitative research platform for crypto markets — built as a personal project.**
> Not a hedge fund. Not a financial product. Just a passion for systematic research.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Data](https://img.shields.io/badge/Data-Binance%205min%20OHLCV-yellow?style=flat-square)
![Symbols](https://img.shields.io/badge/Symbols-400%2B%20USDT%20Pairs-orange?style=flat-square)
![Factors](https://img.shields.io/badge/Factors-35%20Cross--Sectional-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)

---

## What Is This?

Most people look at crypto charts and guess. This project tries to do something different — **study the market like a professional quant researcher would.**

It takes 3 years of Binance 5-minute OHLCV data across 400+ coins and runs it through a full institutional-style research pipeline:

- **Cross-sectional factor research** — testing 35 quantitative signals to find which ones genuinely predict future returns using IC/ICIR methodology
- **Statistical arbitrage** — identifying cointegrated coin pairs to trade the mean-reverting spread between them
- **Machine learning** — training pump/dump detectors, return predictors, and regime classifiers on historical microstructure data
- **Walk-forward simulation** — replaying 3 years of history with rolling model retraining to produce out-of-sample paper trade results
- **Regime-adaptive signal fusion** — combining all four alpha sources into a single ranked signal per coin, with weights that shift automatically based on the current market state

The answers come out as a ranked signal table (`signals.csv`) that scores every coin from **STRONG BUY to STRONG SELL** based on what the data says — not gut feeling.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AZALYST PIPELINE                             │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │  DATA LAYER  │───▶│  FACTOR ENGINE  │───▶│  IC RESEARCH  │  │
│  │  Polars +    │    │  35 factors     │    │  IC/ICIR/     │  │
│  │  DuckDB      │    │  cross-section  │    │  t-stat/decay │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│          │                                         │            │
│          ▼                                         ▼            │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │   STATARB    │    │  REGIME DETECT  │    │  ML SCORING   │  │
│  │ Cointegration│    │  4-state GMM    │    │  Pump/Return  │  │
│  │ Pairs Engine │    │  BTC + breadth  │    │  Predictor    │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│          │                   │                      │           │
│          └───────────────────┴──────────────────────┘           │
│                              ▼                                  │
│                  ┌─────────────────────┐                        │
│                  │   SIGNAL COMBINER   │                        │
│                  │  Regime-adaptive    │                        │
│                  │  weighted fusion    │                        │
│                  └─────────────────────┘                        │
│                              ▼                                  │
│                  ┌─────────────────────┐                        │
│                  │     signals.csv     │                        │
│                  │  Ranked per symbol  │                        │
│                  └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works — Step by Step

### Step 1 — Data Loading
**What it does technically:** Parallel parquet ingestion via `ProcessPoolExecutor` across all symbols. Builds wide close/volume panels (T × N matrix) using Polars lazy scanning and DuckDB for cross-sectional SQL queries. Handles timestamp normalization, schema validation, and optional OHLCV resampling (5min → 1H → 4H).

**In plain English:** It reads all your Binance price files (one per coin) and organises them into a giant table — every coin, every 5-minute candle, for 3 years. It does this in parallel so it doesn't take forever.

---

### Step 2 — Factor Research (35 Signals)
**What it does technically:** `FactorEngineV2` computes 35 cross-sectional quantitative factors across the full universe panel. All factors are output as cross-sectional percentile ranks (0→1) at each timestamp. The `CrossSectionalAnalyser` then computes Spearman rank IC between each factor and forward returns at horizons of 1H, 4H, 1D, 3D, 1W — along with ICIR, Newey-West corrected t-statistics, factor decay curves, and Q1→Q5 quantile spread returns.

**In plain English:** It tests 35 different signals — things like "did coins that went up strongly yesterday keep going up?", "did unusually high volume predict a big move?", "which coins are closest to their yearly high?" For each signal it asks: *does this actually predict the next candle, or is it random?* Only the ones that pass a statistical significance test are trusted.

#### Factor Categories

| Category | Factors | What It Looks For |
|---|---|---|
| **Momentum** | MOM_1H → MOM_30D, OVERNIGHT, CLOSE_TO_OPEN | Coins continuing in the same direction |
| **Reversal** | REV_1H, REV_4H, REV_1D | Coins bouncing back after sharp moves |
| **Volatility** | RVOL_1D, RVOL_1W, VOL_OF_VOL, DOWNVOL_1W | How wildly a coin has been moving |
| **Liquidity** | AMIHUD, CORWIN_SCHULTZ, TURNOVER, VOL_RATIO | How easy/hard it is to trade a coin |
| **Microstructure** | MAX_RET, SKEW_1W, KURT_1W, VWAP_DEV, BTC_BETA, IDIO_MOM | Hidden patterns in price behaviour |
| **Technical** | TREND_48, BB_POS, RSI_RANK, MA_SLOPE, WEEK52_HIGH | Classic chart patterns, ranked cross-sectionally |

---

### Step 3 — Statistical Arbitrage
**What it does technically:** Engle-Granger two-step cointegration test across all symbol pairs. Pairs that pass (p-value < 0.05) are further validated with Hurst exponent (< 0.5 confirms mean-reversion) and half-life of mean reversion via OLS on the spread AR(1) process. Live z-scores computed as `z = (spread - rolling_mean) / rolling_std` with entry at ±2σ and exit at ±0.5σ.

**In plain English:** Some coins are statistically linked — when one goes up, the other tends to follow. When they diverge, the gap usually closes back. This module finds those pairs automatically by testing every possible combination. Instead of betting on one coin going up, you trade the *gap* between two related coins — which is far less risky because you don't care about market direction, only whether the gap closes.

---

### Step 4 — Machine Learning
**What it does technically:** Three models trained with strict lookahead-bias controls using `TimeSeriesSplit` cross-validation and purged embargo periods.

**In plain English:** Three separate models trained on the historical data:

| Model | Technical Detail | Plain English |
|---|---|---|
| **PumpDumpDetector** | LightGBM binary classifier on 28 microstructure features. Target: price spike >15% within next 48 bars. Evaluated by AUC. | Learns the fingerprint a coin shows *before* it gets pumped — unusual volume, candle shape, wick ratios. Flags coins to avoid. |
| **ReturnPredictor** | RandomForest with TimeSeriesCV. Target: sign of 4H forward return. Features pre-shifted +1 bar to prevent lookahead. | Learns which combinations of signals tend to be followed by price going up vs down in the next 4 hours. |
| **RegimeDetector** | 4-component Gaussian Mixture Model on BTC log-returns, realized volatility, and market breadth (fraction of coins above 48-bar MA). | Looks at BTC behaviour and classifies the whole market as Bull, Bear, High Volatility, or Quiet. The same signal works very differently in each regime. |

---

### Step 5 — Walk-Forward Simulation (The Time Machine Test)
**What it does technically:** Rolling window walk-forward with `TRAIN_DAYS=365` and `PREDICT_DAYS=30`. Scaler fitted exclusively on training rows. Features pre-shifted +1 bar in `build_feature_cache.py`. Entry simulated at next bar's open. Binance taker fee of 0.1% per leg applied on round-trip. Checkpoint saved to `checkpoint.json` after every prediction window.

**In plain English:** This is the most important part — and the most commonly misunderstood.

Instead of testing the model on the same data it trained on (which would be cheating), the engine runs a time machine simulation:

```
Train on 2023-03-07 → 2024-03-06  (1 full year of learning)
→ Predict + paper trade: 2024-03-06 → 2024-04-05 (next 30 days)
→ Save every trade to paper_trades.csv
→ Slide forward 30 days, retrain on fresh data
→ Predict next 30 days
→ Repeat ~24 times until present day
```

**Important:** When it says "predicting" on a 30-day window, it already has those candles saved in your data folder. It is **replaying history** — not predicting the future in real time. This is a backtest, not a live trading system. The paper trades you get are as close to real historical performance as possible without risking any money.

---

### Step 6 — Regime-Adaptive Signal Fusion
**What it does technically:** `SignalCombiner` computes a weighted composite score per symbol: `composite = w_factor × factor_rank + w_ml × return_proba + w_pump × (1 - pump_proba) + w_statarb × statarb_score`. Weights sourced from `REGIME_WEIGHT_TABLE` keyed on current `RegimeDetector` output. Missing sources receive neutral score 0.5 with weight redistributed proportionally.

**In plain English:** The final step takes all four sources and blends them into one score per coin. The blend shifts automatically based on the market:

| Source | Bull Market | Bear Market | High Volatility | Quiet Market |
|---|---|---|---|---|
| Factor signals | 45% | 25% | 15% | 30% |
| ML return prediction | 35% | 20% | 15% | 30% |
| Pump risk filter | 10% | 20% | 35% | 15% |
| StatArb pair signal | 10% | 35% | 35% | 25% |

Every coin gets a final grade:

```
≥ 0.75  →  STRONG BUY
≥ 0.60  →  BUY
  0.50  →  HOLD
≤ 0.40  →  SELL
≤ 0.25  →  STRONG SELL
```

---

## Critical — Research Mode vs Live Mode

This is the most important thing to understand before using this project:

| Research Mode (this repo) | Live Mode (not built yet) |
|---|---|
| Runs on historical saved parquet files | Would run on live Binance API feed |
| Already has all candles — replays history | Sees only the current new candle |
| Measures how well the model *would have* worked | Would fire signals on real new 5-min bars |
| Outputs paper trades — zero real money | Would require real execution logic |
| **This is what the pipeline does right now** | Built only after research validates the model |

**The correct order:**
```
1. ✅ Train and test on historical data  ← this repo
2.    Check results: is Sharpe > 1? Win rate > 55%? Drawdown acceptable?
3.    Only if results justify it → build live runner on top of trained models
```

---

## What You Get When It Finishes

```
azalyst_output/
  signals.csv              ← start here — every coin scored and graded
  factor_ic_results.csv    ← which of the 35 signals actually worked
  statarb_pairs.csv        ← coin pairs to trade as spreads
  paper_trades.csv         ← every simulated trade across 3 years
  performance_metrics.csv  ← Sharpe ratio, win rate, max drawdown per cycle
  models/                  ← trained ML model files (reusable for live runner)
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/gitdhirajsv/azalyst-alpha-research-engine.git
cd azalyst-alpha-research-engine
```

### 2. Install dependencies
```bash
pip install pandas numpy scipy scikit-learn lightgbm statsmodels polars duckdb pyarrow
```

### 3. Add your Binance data
```
data/
  BTCUSDT.parquet
  ETHUSDT.parquet
  ...
```
Schema: `timestamp | open | high | low | close | volume`

### 4. Run the full pipeline
**Windows:**
```
double-click  RUN_AZALYST.bat
```
**Command line:**
```bash
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```

Safe to stop and restart anytime — each of the 8 steps saves a checkpoint.

---

## How Long Does It Take?

| Step | Time (400+ symbols, 3 years) |
|---|---|
| Feature cache build | 30 min – 2 hours |
| Walk-forward simulation | 2 – 8 hours |
| Factor research (IC/ICIR) | 30 – 60 min |
| StatArb cointegration scan | 1 – 3 hours |
| ML model training | 15 – 45 min |
| Signal fusion + reports | 5 – 15 min |

Start it, let it run, come back later.

---

## Module Reference

| File | Description |
|---|---|
| `azalyst_orchestrator.py` | Master pipeline — chains all 8 stages end to end |
| `azalyst_signal_combiner.py` | Regime-adaptive signal fusion engine |
| `azalyst_engine.py` | DataLoader, FactorEngine (20 factors), CrossSectionalAnalyser, BacktestEngine |
| `azalyst_factors_v2.py` | Extended 35-factor library with full OHLCV support |
| `azalyst_data.py` | Polars lazy engine + DuckDB OLAP analytics layer |
| `azalyst_ml.py` | PumpDumpDetector, ReturnPredictor, RegimeDetector, AnomalyDetector |
| `azalyst_statarb.py` | Engle-Granger cointegration scanner + pairs trading engine |
| `azalyst_risk.py` | MVO, HRP, Black-Litterman portfolio optimization + VaR/CVaR |
| `azalyst_execution.py` | LOB simulator, VWAP/TWAP algos, square-root market impact model |
| `azalyst_benchmark.py` | BTC buy-hold, equal-weight, volume-weighted baselines |
| `azalyst_report.py` | Research report generator + LiveAlphaScanner |
| `azalyst_tearsheet.py` | Factor tearsheet + quantile spread analyser |
| `azalyst_alphaopt.py` | Alpha signal combination and optimisation utilities |
| `azalyst_auditor.py` | Binance copy-trader reverse-engineering tool |
| `build_feature_cache.py` | Pre-computes and caches 28 ML features per symbol |
| `walkforward_simulator.py` | Rolling retrain walk-forward paper trading simulator |

---

## Research Basis

| Paper | What It Contributed |
|---|---|
| Liu & Tsyvinski (2021) | Crypto momentum factors — strongest documented alpha |
| Fang & Li / Cambridge (2024) | CTREND factor — t-stat 4.22 on 5-min crypto data |
| Amihud (2002) | Illiquidity premium — illiquid coins earn more |
| Corwin & Schultz (2012) | Bid-ask spread estimation from OHLC data |
| Bali, Cakici & Whitelaw (2011) | MAX effect — lottery coins underperform |
| Harvey & Siddique (2000) | Skewness premium in returns |
| Hong & Stein (1999) | Information diffusion — why momentum works |
| Adrian, Etula & Muir (2019) | Downside volatility risk premium |
| Lopez de Prado (2016) | Hierarchical Risk Parity |
| Black & Litterman (1992) | Portfolio optimization with investor views |

---

## Requirements

```
Python 3.10+
pandas, numpy, scipy, scikit-learn
lightgbm, statsmodels
polars, duckdb, pyarrow
```

---

## Disclaimer

> This is a **personal research and learning project.**
>
> Azalyst is not a hedge fund, trading firm, or financial service of any kind.
> Nothing here is financial advice. Past simulation results say nothing about future performance.
> **Use entirely at your own risk.**

---

## License

MIT — free to use, modify, and build on with attribution.

---

<div align="center">

**Built by Azalyst ([gitdhirajsv](https://github.com/gitdhirajsv)) &nbsp;|&nbsp; Azalyst Quant Research**

</div>
