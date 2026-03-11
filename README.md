# Azalyst Alpha Research Engine

> **An institutional-style quantitative research platform for crypto markets — built as a personal project.**
> Not a hedge fund. Not a financial product. Just a passion for systematic research.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)
![Factors](https://img.shields.io/badge/Factors-35%20Crypto--Native-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-LightGBM%20%2B%20CUDA%20v4.0-blueviolet?style=flat-square)

---

## What Is This?

Most people look at crypto charts and guess. This project tries to do something different — **study the market like a professional quant researcher would.**

The Azalyst Alpha Research Engine takes 3 years of Binance 5-minute OHLCV data across 400+ coins and runs it through a full institutional-style research pipeline. It identifies persistent signals, validates them against systematic risk, and fuses them into a single ranked alpha signal per coin.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    AZALYST ALPHA RESEARCH ENGINE                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐   │
│  │  DATA LAYER  │───▶│  FACTOR ENGINE v2│───▶│  VALIDATION   │   │
│  │  Polars +    │    │  35 factors      │    │  Style Neut.  │   │
│  │  DuckDB      │    │  Cross-section   │    │  Fama-MacBeth │   │
│  └──────────────┘    └──────────────────┘    └───────────────┘   │
│          │                                          │            │
│          ▼                                          ▼            │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐   │
│  │   STATARB    │    │   REGIME DETECT  │    │   ML SCORING  │   │
│  │ Cointegration│    │   GMM + Breadth  │    │   LGBM + CUDA │   │
│  │ Pairs Engine │    │   BTC Microstr.  │    │   Pump/Return │   │
│  └──────────────┘    └──────────────────┘    └───────────────┘   │
│          │                    │                       │          │
│          └────────────────────┴───────────────────────┘          │
│                               ▼                                  │
│                   ┌───────────────────────┐                      │
│                   │    SIGNAL COMBINER    │                      │
│                   │   Regime-adaptive     │                      │
│                   │   weighted fusion     │                      │
│                   └───────────────────────┘                      │
│                               ▼                                  │
│                   ┌───────────────────────┐                      │
│                   │      signals.csv      │                      │
│                   │   Ranked per symbol   │                      │
│                   └───────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## How It Works — Step by Step

### Step 1 — Data Loading
**Technically:** Parallel parquet ingestion via `ProcessPoolExecutor`. Builds wide close/volume panels using Polars lazy scanning and DuckDB for cross-sectional SQL queries. Handles timestamp normalization and optional resampling (5min → 1H).

**In Plain English:** It reads all your Binance price files and organizes them into a giant table — every coin, every 5-minute candle, for 3 years. It does this in parallel for speed.

### Step 2 — Factor Research (The v2 Library: 35 Signals)
**Technically:** `FactorEngineV2` computes 35 cross-sectional quantitative factors. The `CrossSectionalAnalyser` computes Spearman rank IC between each factor and forward returns (1H, 4H, 1D, 3D, 1W) with ICIR, Newey-West corrected t-stats, and decay curves.

**In Plain English:** It tests 35 different signals — things like "did coins that went up strongly yesterday keep going up?" or "did unusual volume predict a big move?" Only signals that pass statistical significance tests are trusted.

| Category | Factors | What It Looks For |
|---|---|---|
| **Momentum** | MOM_1H → MOM_30D, OVERNIGHT | Coins continuing in the same direction |
| **Reversal** | REV_1H, REV_4H, REV_1D | Coins bouncing back after sharp moves |
| **Volatility** | DOWNVOL_1W, RVOL_1D, VOL_OF_VOL | Risk premiums and downside deviation |
| **Liquidity** | AMIHUD, CORWIN_SCHULTZ, TURNOVER | Ease of trading and taker pressure |
| **Microstructure**| MAX_RET, SKEW_1W, KURT_1W, BTC_BETA | Hidden patterns and systematic risk |
| **Technical** | TREND_48, BB_POS, RSI_RANK, MA_SLOPE | Classic patterns, ranked cross-sectionally |

### Step 3 — Institutional Validation
**Technically:** `FactorValidator` performs Style Neutralization to remove BTC-beta, Size (Market Cap), and Liquidity tiers from returns. It then runs Fama-MacBeth regressions and applies Benjamini-Hochberg correction to control the False Discovery Rate.

**In Plain English:** Raw signals are often just BTC in disguise. This step strips away the market "noise" to find the TRUE unique alpha of a coin. It's the standard used by top quant hedge funds to avoid overfitting.

### Step 4 — Statistical Arbitrage (Pairs Trading)
**Technically:** Engle-Granger two-step cointegration test across all symbol pairs. Validated with Hurst exponent and half-life of mean reversion. Live z-scores are computed for mean-reverting spreads.

**In Plain English:** It finds coins that are linked — when one goes up, the other usually follows. When they diverge, the engine flags a trade for the gap to close, which is a market-neutral strategy.

### Step 5 — Machine Learning v4.0 (Fast Training)
**Technically:** Migrated to LightGBM with NVIDIA CUDA support. Features purged time-series cross-validation (Purged CV) to eliminate lookahead bias.
*   **PumpDumpDetector**: AUC-optimized model to flag coins with pre-pump signatures.
*   **ReturnPredictor**: Predicts 4H forward return direction.
*   **RegimeDetector**: 4-component GMM on BTC and market breadth to classify Bull, Bear, High Vol, or Quiet markets.

**In Plain English:** High-speed models (trained in ~5-15 mins) learn the "fingerprints" of price moves. The engine automatically shifts its strategy based on the market regime it detects.

### Step 6 — Walk-Forward Simulation (The Time Machine Test)
**Technically:** Rolling window walk-forward with retraining every 30 days. Scaler fitted exclusively on training rows. Entries simulated at next bar's open with 0.1% taker fees applied.

**In Plain English:** This is a backtest that replays history. It learns on one year of data, predicts the next 30 days, and then "slides" forward to repeat the process, just as it would in real life.

---

## Module Reference

| File | Description |
|---|---|
| `azalyst_orchestrator.py` | Master pipeline — chains all 8 stages end to end. |
| `azalyst_validator.py` | **Institutional Validation** (Style Neut, Fama-MacBeth, BH Correction). |
| `azalyst_factors_v2.py` | **Factor Library v2** with 35 crypto-native alpha signals. |
| `azalyst_ml.py` | **Fast ML Module** (LightGBM + CUDA) and Regime Detection. |
| `azalyst_engine.py` | DataLoader, IC Research, and core Backtest Engine. |
| `azalyst_data.py` | Polars/DuckDB analytics layer for high-performance data processing. |
| `azalyst_statarb.py` | Engle-Granger Cointegration scanner for statistical arbitrage. |
| `azalyst_risk.py` | MVO, HRP, and Black-Litterman portfolio optimization. |
| `azalyst_output/` | Signals, IC results, paper trades, and performance reports. |

---

## Quickstart

### 1. Install Dependencies
```bash
pip install pandas numpy scipy scikit-learn lightgbm statsmodels polars duckdb pyarrow
```
*Note: For maximum speed, ensure your LightGBM installation has CUDA/GPU support.*

### 2. Add Your Data
Place your Binance 5m parquet files in the `data/` folder.
Schema: `timestamp | open | high | low | close | volume`

### 3. Run the Pipeline
**Windows:** Double-click `RUN_AZALYST.bat`
**Command Line:**
```bash
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```

---

## Disclaimer
This is a **personal research and learning project.** Azalyst is not a financial service. Nothing here is financial advice. Use entirely at your own risk.

---
<div align="center">
Built by [Azalyst](https://github.com/gitdhirajsv)
</div>
