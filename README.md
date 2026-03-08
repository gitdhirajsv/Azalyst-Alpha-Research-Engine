# Azalyst Alpha Research Engine

> **Institutional-grade quantitative crypto research pipeline — built as a personal project.**
> Not a hedge fund. Not a financial product. Just a passion for systematic research.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Data](https://img.shields.io/badge/Data-Binance%205min%20OHLCV-yellow?style=flat-square)
![Factors](https://img.shields.io/badge/Factors-35%20Cross--Sectional-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)

---

## What Is This?

**Azalyst Alpha Research Engine** is a personal quantitative research platform built to study crypto markets the same way institutional quant teams do — using factor models, statistical arbitrage, machine learning, and rigorous backtesting.

The methodology is drawn from academic research (Liu & Tsyvinski, Amihud, Lopez de Prado, and others) and the architecture mirrors what you would find at a systematic hedge fund — but this is entirely a personal research project built for learning, exploration, and open sharing.

**What it does:**
- Discovers which quantitative signals (factors) actually predict crypto returns
- Runs cointegration-based pairs trading across 400+ Binance symbols
- Detects pump/dump patterns using ML before they happen
- Identifies the current market regime and adapts strategy accordingly
- Fuses everything into a single ranked signal table per symbol

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

## Features

### 35-Factor Cross-Sectional Library
All factors output cross-sectional percentile ranks (0→1) so they are directly comparable across assets.

| Category | Factors |
|---|---|
| **Momentum** | MOM_1H, MOM_4H, MOM_1D, MOM_3D, MOM_1W, MOM_2W, MOM_30D, OVERNIGHT, CLOSE_TO_OPEN |
| **Reversal** | REV_1H, REV_4H, REV_1D |
| **Volatility** | RVOL_1D, RVOL_1W, VOL_OF_VOL, DOWNVOL_1W |
| **Liquidity** | AMIHUD, CORWIN_SCHULTZ, TURNOVER, VOL_RATIO, VOL_MOM_1D |
| **Microstructure** | MAX_RET, SKEW_1W, KURT_1W, PRICE_ACCEL, VOLUME_SURPRISE, VWAP_DEV, BTC_BETA, IDIO_MOM |
| **Technical** | TREND_48, BB_POS, RSI_RANK, MA_SLOPE, WEEK52_HIGH, WEEK52_LOW |

### IC / ICIR Research Framework
- Spearman rank IC between factor and forward returns
- ICIR (IC Information Ratio) for consistency measurement
- Newey-West corrected t-statistics
- Factor decay curves across 1H / 4H / 1D / 3D / 1W horizons
- Quantile portfolio spread analysis (Q1 vs Q5)

### Statistical Arbitrage Engine
- Engle-Granger cointegration test across all symbol pairs
- Johansen test for multivariate cointegration
- Hurst exponent confirmation (< 0.5 = mean-reverting)
- Half-life of mean reversion per pair
- Live z-score spread signals

### Machine Learning Layer

| Model | Purpose | Algorithm |
|---|---|---|
| PumpDumpDetector | Flags manipulation risk before it happens | LightGBM / GBM |
| ReturnPredictor | 4H direction probability per symbol | RandomForest + TimeSeriesCV |
| RegimeDetector | Current market state classification | Gaussian Mixture Model |
| AnomalyDetector | Unusual bar filtering | IsolationForest |

### Regime-Adaptive Signal Fusion
Four signal sources fused with weights that automatically shift based on detected market regime:

| Source | BULL | BEAR | HIGH_VOL | LOW_VOL |
|---|---|---|---|---|
| Factor composite | 45% | 25% | 15% | 30% |
| ML return prob | 35% | 20% | 15% | 30% |
| Pump risk (inverted) | 10% | 20% | 35% | 15% |
| StatArb z-score | 10% | 35% | 35% | 25% |

Final signal grades per symbol: **STRONG BUY / BUY / HOLD / SELL / STRONG SELL**

### Walk-Forward Simulator
3-year historical replay with rolling retrain — never looks at future data:
- Features pre-shifted +1 bar (no lookahead bias)
- Scaler fitted only on training window
- Entry executed at next bar's open
- Auto-checkpoint + resume if interrupted mid-run

---

## Module Reference

| File | Description |
|---|---|
| `azalyst_orchestrator.py` | Master pipeline — chains all modules end to end |
| `azalyst_signal_combiner.py` | Regime-adaptive signal fusion engine |
| `azalyst_engine.py` | DataLoader, FactorEngine, CrossSectionalAnalyser, BacktestEngine |
| `azalyst_factors_v2.py` | Full 35-factor library |
| `azalyst_data.py` | Polars lazy engine + DuckDB analytics layer |
| `azalyst_ml.py` | All four ML models |
| `azalyst_statarb.py` | Cointegration scanner + pairs trading engine |
| `azalyst_risk.py` | MVO, HRP, Black-Litterman, VaR/CVaR |
| `azalyst_execution.py` | LOB simulator, VWAP/TWAP algos, market impact model |
| `azalyst_benchmark.py` | BTC buy-hold, equal-weight, volume-weighted baselines |
| `azalyst_report.py` | Research report generator + LiveAlphaScanner |
| `azalyst_tearsheet.py` | Factor tearsheet + quantile spread analyser |
| `azalyst_alphaopt.py` | Alpha combination and optimisation utilities |
| `azalyst_auditor.py` | Binance copy-trader reverse-engineering tool |
| `build_feature_cache.py` | Pre-computes ML features per symbol |
| `walkforward_simulator.py` | 3-year walk-forward paper trading simulator |

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

### 3. Add your data
Place Binance OHLCV parquet files in `./data/`
```
data/
  BTCUSDT.parquet
  ETHUSDT.parquet
  SOLUSDT.parquet
  ...
```
Expected schema: `timestamp | open | high | low | close | volume`

### 4. Run the full pipeline

**Windows (easiest):**
```
double-click  RUN_AZALYST.bat
```

**Command line:**
```bash
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```

### 5. Read your results
```
azalyst_output/
  signals.csv              ← start here — ranked signal table per symbol
  factor_ic_results.csv    ← which of the 35 factors work on your data
  statarb_pairs.csv        ← best cointegrated pairs to trade
  paper_trades.csv         ← 3-year simulated trade log
  performance_metrics.csv  ← Sharpe, drawdown, win rate per cycle
  models/                  ← trained ML model snapshots
```

---

## CLI Options

```bash
# Full pipeline on all symbols
python azalyst_orchestrator.py --data-dir ./data

# Quick test — limit to 30 symbols
python azalyst_orchestrator.py --data-dir ./data --max-symbols 30

# Run specific stages only
python azalyst_orchestrator.py --data-dir ./data --stages data factors ic signals

# Use 4H candles instead of 1H
python azalyst_orchestrator.py --data-dir ./data --resample 4H

# StatArb scan only
python azalyst_statarb.py --data-dir ./data --out-dir ./azalyst_output

# ML training only
python azalyst_ml.py --data-dir ./data --out-dir ./models --model all
```

---

## Resume / Checkpoint System

The pipeline is **safe to stop and restart at any time.**

- Each of the 8 steps writes a `.done` flag to `pipeline_checkpoints\`
- On re-run, completed steps are automatically skipped
- Walk-forward simulation has its own internal `checkpoint.json` for mid-run resume
- To force re-run a specific step — delete its `.done` file and run again

---

## Research Basis

Grounded in published academic research:

| Paper | Application |
|---|---|
| Liu & Tsyvinski (2021) | Crypto momentum factors |
| Fang & Li / Cambridge (2024) | CTREND factor — t-stat 4.22 |
| Amihud (2002) | Illiquidity premium |
| Corwin & Schultz (2012) | Bid-ask spread from OHLC |
| Bali, Cakici & Whitelaw (2011) | MAX effect in crypto |
| Harvey & Siddique (2000) | Skewness premium |
| Hong & Stein (1999) | Information diffusion / momentum |
| Adrian, Etula & Muir (2019) | Downside volatility premium |
| Lopez de Prado (2016) | Hierarchical Risk Parity |
| Black & Litterman (1992) | Portfolio optimization |

---

## Requirements

```
Python        3.10+
pandas        2.0+
numpy         1.24+
scipy         1.11+
scikit-learn  1.3+
lightgbm      4.0+
statsmodels   0.14+
polars        0.20+
duckdb        0.9+
pyarrow       14.0+
```

---

## Disclaimer

> This is a **personal research and learning project.**
>
> Azalyst is not a hedge fund, not a trading firm, and not a financial product of any kind.
> Nothing in this repository constitutes financial advice or a recommendation to trade.
> Past simulation results do not guarantee future performance.
> **Use entirely at your own risk.**

---

## License

MIT — free to use, modify, and build on with attribution.

---

<div align="center">

**Built by Azalyst ([gitdhirajsv](https://github.com/gitdhirajsv)) &nbsp;|&nbsp; Azalyst Quant Research**

</div>
