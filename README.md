# Azalyst Alpha Research Engine

An institutional-style quantitative research platform built as a personal project. Not a hedge fund. Not a financial product. Just a passion for systematic research.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)
![Features](https://img.shields.io/badge/Features-56%20Cross--Sectional-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20CUDA-blueviolet?style=flat-square)
![CV](https://img.shields.io/badge/CV-Purged%20K--Fold-orange?style=flat-square)

</div>

---

## Overview

Azalyst Alpha Research Engine is a research infrastructure project for discovering and validating systematic alpha signals in cryptocurrency markets. It is designed as a rigorous quantitative research system — not a trading bot, not a signal service, not a financial product.

At a high level, the engine processes 3+ years of 5-minute OHLCV data across 444 Binance pairs, engineers 56 cross-sectional features (including WorldQuant-inspired alphas, microstructure signals, and fractionally differentiated price), trains a two-stage XGBoost model using purged K-Fold cross-validation, and validates it strictly out-of-sample on a full year of data it never saw during training. Every signal, every metric, and every trade simulation is logged and explainable.

The project exists for a single reason: most open-source crypto research is toy-level — fit a moving average, overfit on in-sample, declare victory. Azalyst is the antithesis. These methods mirror how the top systematic funds structure research, not because we are a fund, but because they are the only honest way to know if your signal is real.

Core capabilities:

- 56 cross-sectional features across 9 categories — returns, volume, volatility, technical, microstructure, price structure, WorldQuant alphas, regime, and fractional differentiation.
- Two-stage XGBoost pipeline — primary model (cross-sectional outperformance) + meta-labeling model (confidence-weighted position sizing).
- Purged K-Fold cross-validation with 48-bar embargo — no information leakage from autocorrelated features.
- IC-weighted regime-adaptive signal fusion across 4 alpha sources.
- Walk-forward Year 3 validation — the test set is never touched during training, ever.
- GPU-accelerated training on NVIDIA CUDA (RTX 2050 / T4 / any CUDA-capable device).
- Windows one-click launcher with auto-install, GPU detection, and Spyder integration.

Research controls:

- Strictly out-of-sample test set: Year 3 only (never seen during training)
- Quarterly retrain cadence: every 13 weeks (primary + meta together)
- Meta-labeling position sizing: high-confidence signals get more capital automatically
- Position-tracked fee model: 0.2% round-trip only on new entries (not on held positions)
- IC/ICIR weekly tracking throughout Year 3 walk-forward

Primary dependencies:

- Python 3.10+
- xgboost (CUDA)  scikit-learn  polars  pyarrow  duckdb
- pandas  numpy  scipy  statsmodels
- matplotlib  seaborn

---

## Architecture

```
                     
                              AZALYST RESEARCH ENGINE          
                     
                                    
       
                                                               
              
  DATA LAYER              FEATURE ENGINE             SIGNAL SOURCES   
                                                                      
 Polars+DuckDB    56 cross-sectional  Factor scores     
 444 coins               features, TF-aware         ML return prob    
 26M+ rows               Frac. diff (AFML)          Pump/dump filter  
 3-year 5min             Hurst + FFT                StatArb z-scores  
              
                                                                
                                         
                            SIGNAL COMBINER   
                                              
                           Regime-adaptive    
                           IC-weighted fusion 
                           4-state detector   
                          
                                   
       
                                                           
         
  PRIMARY MODEL         META-LABELING           WALK-FORWARD     
                        (AFML Ch. 3)                             
 XGBoost CUDA     2nd-stage XGBoost   Y1+Y2 train       
 Purged K-Fold         P(primary correct)      Y3 strict OOS     
 48-bar embargo        Confidence sizing       Quarterly retrain 
 RobustScaler                                  Weekly IC + ICIR  
         
```

---

## The Three Pillars

### 1. Fractional Differentiation (Lopez de Prado, AFML Ch. 5)

Standard returns destroy all memory of price levels. Raw prices preserve memory but are non-stationary and break tree-based models. Fractional differentiation with `d=0.4` using the Fixed-Width Window (FFD) method gives the model access to **where the price actually is** while maintaining stationarity.

```
d=0.0     raw price (non-stationary, max memory)
d=0.4     Azalyst default (stationary, retains memory)   HERE
d=1.0     standard returns (stationary, zero memory)
```

### 2. Meta-Labeling (Lopez de Prado, AFML Ch. 3)

The primary model says "BUY this coin." But how confident should we be? A second-stage XGBoost model is trained on a meta-question: **"When the primary model predicted UP, was it actually correct?"** The output confidence probability directly scales position size:

- **High confidence (0.85)**  full position, maximum capital allocation
- **Low confidence (0.45)**  reduced position, capital preservation

Wrong signals get less money. Right signals get more. The meta-model trains on honest out-of-sample predictions — no information leakage.

### 3. IC-Weighted Signal Fusion (Grinold & Kahn)

The signal combiner fuses 4 alpha sources using regime-adaptive weights, but instead of static weights IC-weighted fusion tracks the rolling Information Coefficient of each source over the last 13 weeks and dynamically reweights — signals that are currently working get more weight, decaying signals get less.

```
Base regime weights    IC multiplier    Normalized adaptive weights
                         
              max(0.1, min(3.0, 1 + 10mean_IC))
```

---

## Feature Engineering — 56 Features, 9 Categories

| Category | Count | Features |
|---|---|---|
| Returns | 7 | `ret_1bar`  `ret_1h`  `ret_4h`  `ret_1d`  `ret_2d`  `ret_3d`  `ret_1w` |
| Volume | 6 | `vol_ratio`  `vol_ret_1h`  `vol_ret_1d`  `obv_change`  `vpt_change`  `vol_momentum` |
| Volatility | 7 | `rvol_1h`  `rvol_4h`  `rvol_1d`  `vol_ratio_1h_1d`  `atr_norm`  `parkinson_vol`  `garman_klass` |
| Technical | 10 | `rsi_14`  `rsi_6`  `macd_hist`  `bb_pos`  `bb_width`  `stoch_k`  `stoch_d`  `cci_14`  `adx_14`  `dmi_diff` |
| Microstructure | 6 | `vwap_dev`  `amihud`  `kyle_lambda`  `spread_proxy`  `body_ratio`  `candle_dir` |
| Price Structure | 6 | `wick_top`  `wick_bot`  `price_accel`  `skew_1d`  `kurt_1d`  `max_ret_4h` |
| WorldQuant Alphas | 8 | `wq_alpha001`  `wq_alpha012`  `wq_alpha031`  `wq_alpha098`  `cs_momentum`  `cs_reversal`  `vol_adjusted_mom`  `trend_consistency` |
| Regime | 5 | `vol_regime`  `trend_strength`  `corr_btc_proxy`  `hurst_exp`  `fft_strength` |
| Memory-Preserving | 1 | `frac_diff_close` — fractional differentiation d=0.4 (AFML Ch. 5) |

---

## ML Pipeline

### Training Label — Cross-Sectional Alpha

The model predicts whether a coin will **outperform the cross-sectional median** return at the next 4H horizon. Direction-agnostic — works in bull and bear markets equally.

$$\text{alpha\_label}_i = \mathbb{1}\left[ r_{i,t+48} > \text{median}(r_{j,t+48}) \;\forall\; j \in \text{universe} \right]$$

### Purged K-Fold Cross-Validation

48-bar embargo gap between train and validation prevents information leakage from autocorrelated features:

```
| TRAIN | 48-bar gap | VAL |
                  (4 hours)
```

5 purged folds. RobustScaler for fat-tailed crypto distributions.

### Meta-Labeling (Second-Stage Model)

```
Primary Model predictions (OOS from purged CV)
    
Meta-label: did primary model get this row correct? (binary)
    
Second XGBoost: features + primary_prob  P(correct)
    
Output: confidence score per symbol per week  position sizing
```

### Walk-Forward Architecture

```
Year 1 + Year 2 (730 days)
    
[BASE MODEL] + [META MODEL]
XGBoost CUDA  Purged K-Fold (5 splits, gap=48)
RobustScaler  IC + ICIR + AUC
    
Year 3 only (never seen during training)
    

  Each week:                                                  
    1. Predict   — rank all 444 symbols by outperformance prob
    2. Meta-size — scale positions by meta-model confidence   
    3. Trade     — long top 15%, short bottom 15%             
    4. Fees      — position-tracked (only new entries pay)    
    5. Evaluate  — weekly IC + confidence-weighted return     
    6. Retrain   — every 13 weeks (primary + meta together)   
    7. Save      — weekly summary + all trades                

    
performance_year3.json + performance_year3.png
```

### Signal Fusion — 4 Sources, IC-Weighted

```
REGIME DETECTOR (4-state)
    
     BULL_TREND        Factor: 0.45  ML: 0.35  Pump: 0.10  StatArb: 0.10
     BEAR_TREND        Factor: 0.25  ML: 0.20  Pump: 0.20  StatArb: 0.35
     HIGH_VOL_LATERAL  Factor: 0.15  ML: 0.15  Pump: 0.35  StatArb: 0.35
     LOW_VOL_GRIND     Factor: 0.30  ML: 0.30  Pump: 0.15  StatArb: 0.25
                              
                     IC multiplier per source (rolling 13-week IC)
                              
                    Renormalized adaptive weights  composite score
```

---

## Execution Simulation

### Position-Tracked Fee Model

The simulation charges transaction fees **only when a symbol enters the portfolio**. Held positions pay zero fees. This accurately models real-world turnover costs:

```
Fee per new entry:  0.1% per leg  2 = 0.2% round-trip
Held positions:     0% (no fee)
Turnover tracked:   % of portfolio that's new each week
```

### Meta-Labeling Position Sizing

```
pnl_i = (raw_return_i  fee_i)  meta_confidence_i  100
weekly_return = weighted_average(pnl, weights=meta_confidence)
```

High-conviction trades dominate the portfolio return. Low-conviction trades are automatically down-weighted.

---

## Running the Engine

### Option 1 — Windows One-Click (recommended for local)

Double-click **`RUN_AZALYST.bat`** — guides through 3 prompts then runs fully unattended:

1. **Select compute device** — `[1] GPU` (RTX 2050, ~4x faster) or `[2] CPU`
2. **Select output mode** — `[1] Terminal only` or `[2] Terminal + Spyder` (live charts)
3. **Confirm start** — `Y` to launch

The batch file auto-detects Python, GPU availability, and auto-installs all missing packages on first run.

### Option 2 — Kaggle + Google Drive (recommended for full 444-coin run)

The Kaggle pipeline is split into two notebooks. Feature files are stored in **Google Drive** as the intermediate cache — this sidesteps Kaggle's 20GB disk/output limit entirely (444 coins × ~20MB compressed = ~9GB total). Run them in order.

#### One-time Google Drive setup (before first run)
1. Go to [console.cloud.google.com](https://console.cloud.google.com) → create a project → enable **Google Drive API**
2. Create a **Service Account** → Keys → Add Key → JSON → download the key file
3. In Google Drive, create a folder named `azalyst-feature-cache` — copy the folder ID from the URL
4. Share the folder with the service account email (ends in `@...iam.gserviceaccount.com`) — **Editor** access
5. In both Kaggle notebooks → **Add-ons → Secrets** → add secret `GDRIVE_SERVICE_KEY` (paste full JSON key contents)
6. Set `GDRIVE_FOLDER_ID` in Cell 1 of both notebooks to your folder ID

**Step 1 — Build Feature Cache** (`notebooks/azalyst_1_feature_cache.ipynb`)

Upload to Kaggle. Attach the [Binance 5-min dataset](https://www.kaggle.com/datasets/dhirajsuryavanshi/binance-data-5min-300-coins-3years) as input. Enable **Internet**.

- Loops all 444 symbols automatically across 9 batches of 50 — no manual re-runs
- Builds 56 features per coin with `zstd` compression (~20MB per file vs ~80MB uncompressed)
- Uploads each `.parquet` directly to Google Drive, then deletes the local copy — disk never exceeds ~400MB
- Resume-safe: files already in Drive are automatically skipped on re-run
- **Runtime**: ~3–4 hours | **Output**: 444 `.parquet` files in your Google Drive folder

**Step 2 — Train + Backtest** (`notebooks/azalyst_2_train.ipynb`)

Upload to Kaggle. Enable **GPU T4**. Add `GDRIVE_SERVICE_KEY` secret (same as above).

- Downloads the full feature cache from Google Drive into `/kaggle/working/feature_cache/` (~9GB)
- Trains XGBoost base model + Meta-labeling model on Years 1+2
- Walk-forward backtest on Year 3 — cross-sectional ranking (top/bottom 15%) uses the full 444-coin universe
- Quarterly retrain + meta-labeling confidence sizing throughout Year 3
- **VRAM cap**: 4M rows (T4 16GB) | **RAM guard**: stride-sampling keeps usage under 30GB
- **Output**: `/kaggle/working/azalyst_output/azalyst_results.zip`

> **Run order**: Notebook 1 must fully complete before running Notebook 2. After Notebook 1 finishes, confirm all ~444 files appear in your Google Drive folder, then run Notebook 2.

### Option 3 — Local Jupyter (RTX 2050 / any GPU)

Open **`notebooks/azalyst_jupyter.ipynb`** in VSCode or Jupyter. Run Cell 0 to install dependencies, then run cells in order.

- **Data**: `../data/` (relative to notebooks folder)
- **Output**: `C:/Users/Administrator/Music/azalyst_jupyter_output/`
- **VRAM cap**: 2M rows (RTX 2050 4GB)
- Auto-detects CUDA with dual API support (`device='cuda'` or `tree_method='gpu_hist'`)
- Falls back to CPU if no GPU detected

### Option 4 — Core research pipeline (CLI)

```bash
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```

---

## Installation

**Easiest:** Double-click `RUN_AZALYST.bat` — auto-installs all missing packages on first run.

**Manual:**

```bash
pip install -r requirements.txt
```

For local GPU:

```bash
pip install xgboost --upgrade
python azalyst_local_gpu.py   # verify GPU
```

---

## Data Requirements

Place Binance 5-minute parquet files in `data/`:

```
timestamp | open | high | low | close | volume
```

444 symbols  3 years  5-min bars  26M+ rows.

---

## Repository Map

### Core Pipeline

| File | Purpose |
|---|---|
| `azalyst_factors_v2.py` | 56 cross-sectional features — returns, volume, microstructure, WorldQuant alphas, Hurst, FFT, fractional differentiation |
| `azalyst_train.py` | Primary + Meta model training — XGBoost CUDA, Purged K-Fold, IC+ICIR, meta-labeling |
| `azalyst_weekly_loop.py` | Walk-forward Year 3 — quarterly retrain, meta-labeling sizing, position-tracked fees |
| `azalyst_signal_combiner.py` | IC-weighted regime-adaptive signal fusion — 4 sources, dynamic reweighting |
| `azalyst_alpha_metrics.py` | IC, ICIR, Sharpe, drawdown, retrain trigger |
| `build_feature_cache.py` | Precompute features  parquet cache (5–20x speedup) |

### Research Modules

| File | Purpose |
|---|---|
| `azalyst_engine.py` | Data loading, IC research, backtest engine |
| `azalyst_ml.py` | Regime detection, pump/dump detector |
| `azalyst_statarb.py` | Cointegration scanner |
| `azalyst_risk.py` | Portfolio optimization — MVO, HRP, Black-Litterman |
| `azalyst_alphaopt.py` | Ridge/ElasticNet optimal factor combination |
| `azalyst_validator.py` | Fama-MacBeth, Newey-West, BH correction |

### Infrastructure

| File | Purpose |
|---|---|
| `azalyst_orchestrator.py` | End-to-end research pipeline |
| `azalyst_local_gpu.py` | GPU test + standalone walk-forward runner (RTX 2050) |
| `azalyst_data.py` | Polars + DuckDB high-performance data layer |
| `azalyst_execution.py` | Order book simulation, smart order routing, VWAP/TWAP |
| `azalyst_tf_utils.py` | Timeframe-aware bar count utilities |
| `azalyst_benchmark.py` | BTC buy-and-hold + equal-weight benchmarks |

### Reports & Monitoring

| File | Purpose |
|---|---|
| `azalyst_tearsheet.py` | Factor tear sheet generator |
| `azalyst_report.py` | Research report + live signal scanner |
| `azalyst_auditor.py` | Binance copy-trader strategy auditor |
| `monitor_dashboard.py` | Browser-based live monitor (`http://127.0.0.1:8080`) |

### Notebooks

| File | Purpose |
|---|---|
| `notebooks/azalyst_jupyter.ipynb` | Local GPU notebook — RTX 2050 optimised, 2M row VRAM cap, full pipeline in one run |
| `notebooks/azalyst_1_feature_cache.ipynb` | Kaggle Step 1 — builds 56 features for all 444 coins with zstd compression, uploads each file to Google Drive immediately, deletes local copy to stay within disk limits |
| `notebooks/azalyst_2_train.ipynb` | Kaggle Step 2 — downloads feature cache from Google Drive, trains model, runs Year 3 walk-forward on full 444-coin universe |

---

## Primary Outputs

| File | Description |
|---|---|
| `results/weekly_summary_year3.csv` | Week-by-week return, IC, turnover — Year 3 out-of-sample |
| `results/all_trades_year3.csv` | Every simulated trade with meta-labeling confidence |
| `results/performance_year3.json` | Annual return, Sharpe, IC, ICIR, win rate |
| `results/performance_year3.png` | 4-panel chart: cumulative return, distribution, IC series, trade P&L |
| `results/feature_importance_base.csv` | Feature importance from base model |
| `results/models/model_base_y1y2.*` | Base XGBoost + meta model (Year 1+2) |
| `results/models/model_y3_week*.*` | Quarterly retrained primary + meta models |

---

## How to Interpret Results

| Metric | Acceptable | Good | Strong |
|---|---|---|---|
| IC | > 0.01 | > 0.03 | > 0.05 |
| ICIR | > 0.2 | > 0.5 | > 1.0 |
| Sharpe | > 0.3 | > 0.7 | > 1.5 |
| IC % positive weeks | > 52% | > 58% | > 65% |

IC > 0.05 with ICIR > 1.0 is institutional-quality signal strength. See Grinold & Kahn's *Active Portfolio Management* for why these numbers matter.

---

## Technical Specifications

| Parameter | Value |
|---|---|
| XGBoost trees | 1,000 (primary)  500 (meta) |
| Learning rate | 0.02 |
| Max depth | 6 (primary)  4 (meta) |
| Min child weight | 30 (primary)  50 (meta) |
| Subsample | 0.8 |
| Column sample | 0.7 (tree)  0.7 (level) |
| Regularisation | alpha=0.1, lambda=1.0 |
| CV splits | 5, purged (48-bar gap) |
| VRAM guard | 2M rows (RTX 2050)  4M rows (T4) |
| Train/test | Year 1+2 / Year 3 (strict OOS) |
| Retrain | Every 13 weeks (quarterly) |
| Universe | 444 coins, cross-sectional pooling |
| Horizon | 4H (48  5-min bars) |
| Portfolio | Long top 15%, short bottom 15% |
| Fees | 0.2% round-trip, position-tracked |
| Frac. diff. d | 0.4 (FFD method, threshold 1e-5) |
| IC lookback | 13 weeks rolling (signal reweighting) |

---

## Theoretical Foundations

| Domain | Source | What Azalyst Uses |
|---|---|---|
| Feature engineering | **Lopez de Prado** — *Advances in Financial Machine Learning* | Fractional differentiation, meta-labeling, purged K-Fold CV |
| Signal combination | **Grinold & Kahn** — *Active Portfolio Management* | IC-weighted signal fusion, information ratio targeting |
| Statistical learning | **Hastie, Tibshirani, Friedman** — *Elements of Statistical Learning* | Regularization, cross-validation methodology |
| Robust estimation | **Huber** — *Robust Statistics* (via RobustScaler) | Median/IQR scaling for fat-tailed crypto distributions |
| Factor models | **Fama & French**, **Barra** | Cross-sectional alpha label, factor decomposition |
| Microstructure | **Kyle (1985)**, **Amihud (2002)** | Kyle lambda, Amihud illiquidity ratio |
| Volatility | **Garman & Klass (1980)**, **Parkinson (1980)** | Range-based volatility estimators |
| Time series | **Hurst (1951)**, **FFT** | Regime detection, cyclical pattern identification |

---

## Research Principles

- **Strict train/test split** — Year 3 never touched during training, ever
- **Transparency over mystique** — every decision documented, every metric shown
- **Repeatable pipelines** — same code, same data, same results
- **Evidence over claims** — results are observations, not promises
- **Position-aware costs** — fee simulation reflects real-world execution

---

## Troubleshooting

**No GPU detected:** run `python -c "import xgboost as xgb; print(xgb.__version__)"` and verify CUDA build.
**Feature cache stale:** delete `data/feature_cache/` and re-run — the engine rebuilds automatically.
**OOM / freeze during loading:** reduce `MAX_TRAIN_ROWS` in config or use `LOAD_STRIDE` guard (auto-activates at >8M rows).
**Spyder not found:** the BAT auto-installs Spyder via pip if not detected — check `azalyst.log` for install output.
**Pipeline closes immediately:** confirm Python path has no spaces; use `RUN_AZALYST.bat` which handles quoted paths automatically.
**Google Drive shows fewer files than expected after Notebook 1:** re-run Notebook 1 — the skip logic checks Drive on each batch, so it will resume from where it stopped without re-uploading existing files.

---

## Disclaimer

This is a research and educational project. Not financial advice. Past performance does not indicate future results. Use at your own risk. Always do your own research.

---

<div align="center">

Built by [Azalyst](https://github.com/gitdhirajsv) | Azalyst Quant Research

</div>
