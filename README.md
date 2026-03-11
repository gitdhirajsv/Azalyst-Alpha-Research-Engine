# Azalyst Alpha Research Engine

> **An institutional-style quantitative research platform for crypto markets — built as a personal project.**
> Not a hedge fund. Not a financial product. Just a passion for systematic research.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)
![ML](https://img.shields.io/badge/ML-LightGBM%20%2B%20CUDA-blueviolet?style=flat-square)
![Factors](https://img.shields.io/badge/Factors-35%20Crypto--Native-red?style=flat-square)

---

## What Is This?

The Azalyst Alpha Research Engine is a high-performance quantitative pipeline designed to bridge the gap between retail trading and institutional-grade research. It processes 3+ years of Binance 5-minute OHLCV data across 400+ coins through a rigorous, multi-stage alpha discovery process.

### Core Pillars
- **Extended Factor Research (v2)**: A library of 35 cross-sectional quantitative signals tested for predictive power using Spearman rank IC/ICIR methodology.
- **Institutional Validation**: Advanced statistical verification including Style Neutralization (partialing out BTC beta, size, and liquidity) and Fama-MacBeth regressions with Newey-West corrected t-statistics.
- **High-Performance ML**: Optimized LightGBM models with NVIDIA CUDA support, featuring purged time-series cross-validation to prevent lookahead bias.
- **Regime-Adaptive Fusion**: A dynamic signal combiner that adjusts alpha weights based on detected market regimes (Bull, Bear, High Volatility, Quiet).

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

## Key Modules — Updated Logic

### 1. Extended Factor Library (v35)
Now computes 35 crypto-native factors across 7 categories. All signals are cross-sectionally ranked (0→1) at each timestamp.
*   **Momentum**: 1H to 30D (with skip-1D academic standard), Overnight gaps, Close-to-Open.
*   **Reversal**: 1H, 4H, and 1D (liquidity provider alpha).
*   **Volatility**: Downside semi-deviation (Adrian et al. 2019), Vol-of-Vol.
*   **Microstructure**: Cambridge CTREND (t-stat 4.22), VWAP Deviation, Idiosyncratic Momentum.

### 2. Statistical Validation (Citadel/Two Sigma-grade)
Raw IC is noise. This module provides the proof:
*   **Style Neutralization**: Removes BTC-beta, Market-Cap (Size), and Liquidity tiers to find TRUE idiosyncratic alpha.
*   **Fama-MacBeth**: Industry-standard cross-sectional regressions.
*   **Newey-West Correction**: HAC t-stats to account for autocorrelation in multi-day horizons.
*   **Benjamini-Hochberg**: Multiple-testing correction to control False Discovery Rate.

### 3. Machine Learning v4.0
*   **LightGBM + CUDA**: Migrated from RandomForest to LightGBM. Training time reduced from 4 hours to ~15 minutes (or 5 minutes on GPU).
*   **Purged CV**: Strict embargo periods between training and validation folds to eliminate leakage.
*   **Pump/Dump Detector**: Detects pre-pump signatures (+25% rise with 50% retrace labels).

---

## Module Reference

| File | Role |
|---|---|
| `azalyst_orchestrator.py` | Master entry point. Chains Stage 1–8 into a single pipeline. |
| `azalyst_validator.py` | **[NEW]** Statistical validation framework (Fama-MacBeth, Style Neut). |
| `azalyst_factors_v2.py` | **[UPDATED]** The core 35-factor library with OHLCV support. |
| `azalyst_ml.py` | **[FAST]** LightGBM models with CUDA/GPU acceleration. |
| `azalyst_engine.py` | Data ingestion, IC research, and core backtesting logic. |
| `azalyst_data.py` | Polars and DuckDB analytics layer for massive OHLCV panels. |
| `azalyst_statarb.py` | Engle-Granger cointegration scanner for pairs trading. |
| `walkforward_simulator.py`| Rolling retrain simulator (historical time-machine test). |

---

## Quickstart

### 1. Install Dependencies
```bash
pip install pandas numpy scipy scikit-learn lightgbm statsmodels polars duckdb pyarrow
```
*Note: For GPU acceleration, ensure `lightgbm` is built with CUDA support.*

### 2. Prepare Data
Place your Binance 5m parquet files in `./data/`.
Schema: `timestamp | open | high | low | close | volume`

### 3. Run Pipeline
```bash
# Windows
Double-click RUN_AZALYST.bat

# CLI
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```

---

## Disclaimer
This is a **personal research project.** It is not financial advice. Past performance in simulation is not indicative of future results. Use at your own risk.

---
<div align="center">
Built by [gitdhirajsv](https://github.com/gitdhirajsv) | Azalyst Quant Research
</div>
