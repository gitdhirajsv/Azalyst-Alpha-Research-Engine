# Azalyst Alpha Research Engine

An institutional-style quantitative research platform built as a personal project. Not a hedge fund. Not a financial product. Just a passion for systematic research.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)
![Features](https://img.shields.io/badge/Features-56%20Cross--Sectional-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20CUDA-blueviolet?style=flat-square)
![CV](https://img.shields.io/badge/CV-Purged%20K--Fold-orange?style=flat-square)
![Version](https://img.shields.io/badge/Engine-v4.0-gold?style=flat-square)

</div>

---

## Overview

Azalyst Alpha Research Engine is a research infrastructure project for discovering and validating systematic alpha signals in cryptocurrency markets. It is designed as a rigorous quantitative research system — not a trading bot, not a signal service, not a financial product.

At a high level, the engine processes 3+ years of 5-minute OHLCV data across 444 Binance pairs, engineers 56 cross-sectional features (including WorldQuant-inspired alphas, microstructure signals, and fractionally differentiated price), trains a two-stage XGBoost model using purged K-Fold cross-validation with an **expanding training window**, and validates strictly out-of-sample across **2 full years** (Y2+Y3) it never saw during initial training. Every signal, every metric, and every trade simulation is logged, explainable via SHAP, and persisted in SQLite.

The project exists for a single reason: most open-source crypto research is toy-level — fit a moving average, overfit on in-sample, declare victory. Azalyst is the antithesis. These methods mirror how the top systematic funds structure research, not because we are a fund, but because they are the only honest way to know if your signal is real.

---

## v4 Architecture

```
                     AZALYST v4.0 RESEARCH ENGINE

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
 XGBoost CUDA     2nd-stage XGBoost   Expanding window
 Purged K-Fold         P(primary correct)      Walk Y2+Y3 (2yr)
 48-bar embargo        Confidence sizing       Quarterly retrain
 RobustScaler                                  IC feature select

  RISK INTEGRATION      KILL-SWITCH             PERSISTENCE
 VaR / CVaR            -15% max DD             SQLite (azalyst.db)
 Position risk cap     4-week pause            SHAP per cycle
 HRP weighting         Auto-resume             Full run history
```

### Core Capabilities

- **56 cross-sectional features** across 9 categories — returns, volume, volatility, technical, microstructure, price structure, WorldQuant alphas, regime, and fractional differentiation
- **Expanding training window** — train on Y1, then Y1+Y2, then Y1+Y2+Y3 (not fixed window)
- **2-year out-of-sample** — walk-forward on Y2+Y3 (104 weeks, never seen during initial training)
- **Regime-aware feature selection** — every 2 weeks compute per-feature IC, drop consistently negative IC features at retrain
- **Risk integration** — VaR/CVaR scaled position sizing, 3% per-position risk cap
- **Drawdown kill-switch** — halt all trading if max DD exceeds -15%, resume after 4 weeks
- **SHAP explainability** — TreeExplainer after every training cycle, stored in SQLite + CSV
- **SQLite persistence** — all trades, metrics, SHAP, models, feature IC in `results/azalyst.db`
- **GPU-accelerated** — NVIDIA CUDA via XGBoost (RTX 2050/T4/any CUDA device)
- **Meta-labeling** — 2nd-stage XGBoost for confidence-weighted position sizing (AFML Ch. 3)

---

## The Three Pillars

### 1. Fractional Differentiation (Lopez de Prado, AFML Ch. 5)

Standard returns destroy all memory of price levels. Raw prices preserve memory but are non-stationary and break tree-based models. Fractional differentiation with `d=0.4` using the Fixed-Width Window (FFD) method gives the model access to **where the price actually is** while maintaining stationarity.

```
d=0.0     raw price (non-stationary, max memory)
d=0.4     Azalyst default (stationary, retains memory)   ◄ HERE
d=1.0     standard returns (stationary, zero memory)
```

### 2. Meta-Labeling (Lopez de Prado, AFML Ch. 3)

The primary model says "BUY this coin." But how confident should we be? A second-stage XGBoost model is trained on a meta-question: **"When the primary model predicted UP, was it actually correct?"** The output confidence probability directly scales position size:

- **High confidence (0.85)** → full position, maximum capital allocation
- **Low confidence (0.45)** → reduced position, capital preservation

### 3. Regime-Aware IC Selection (Grinold & Kahn + v4)

Instead of static signal weights, the v4 engine tracks rolling Information Coefficient per feature over the last 8 weeks. Features with consistently negative IC (< -0.02) are dropped at the next quarterly retrain. This prevents the model from training on features that have become adversarial in the current regime.

```
56 features → compute per-feature IC biweekly → rolling 8-week mean
→ drop features below -0.02 IC threshold (min 20 retained)
→ retrain with cleaner feature set every 13 weeks
```

---

## Feature Engineering — 56 Features, 9 Categories

| Category | Count | Features |
|---|---|---|
| Returns | 7 | `ret_1bar` `ret_1h` `ret_4h` `ret_1d` `ret_2d` `ret_3d` `ret_1w` |
| Volume | 6 | `vol_ratio` `vol_ret_1h` `vol_ret_1d` `obv_change` `vpt_change` `vol_momentum` |
| Volatility | 7 | `rvol_1h` `rvol_4h` `rvol_1d` `vol_ratio_1h_1d` `atr_norm` `parkinson_vol` `garman_klass` |
| Technical | 10 | `rsi_14` `rsi_6` `macd_hist` `bb_pos` `bb_width` `stoch_k` `stoch_d` `cci_14` `adx_14` `dmi_diff` |
| Microstructure | 6 | `vwap_dev` `amihud` `kyle_lambda` `spread_proxy` `body_ratio` `candle_dir` |
| Price Structure | 6 | `wick_top` `wick_bot` `price_accel` `skew_1d` `kurt_1d` `max_ret_4h` |
| WorldQuant Alphas | 8 | `wq_alpha001` `wq_alpha012` `wq_alpha031` `wq_alpha098` `cs_momentum` `cs_reversal` `vol_adjusted_mom` `trend_consistency` |
| Regime | 5 | `vol_regime` `trend_strength` `corr_btc_proxy` `hurst_exp` `fft_strength` |
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

### Expanding Window Walk-Forward

Instead of fixed training, the model expands its training window over time:

```
Walk Week 1:   Train on Y1                    → Predict Y2 Week 1
Walk Week 13:  Train on Y1 + Y2[:13wk]        → Predict Y2 Week 14
Walk Week 26:  Train on Y1 + Y2[:26wk]        → Predict Y2 Week 27
...
Walk Week 52:  Train on Y1 + Y2               → Predict Y3 Week 1
Walk Week 65:  Train on Y1 + Y2 + Y3[:13wk]   → Predict Y3 Week 14
```

### Kill-Switch & Risk Integration

```
If cumulative DD > -15%  →  HALT all trading for 4 weeks
VaR exceeds risk cap     →  Scale down position sizes (min 0.3x)
Auto-resume after pause  →  Continue walk-forward normally
```

### Signal Fusion — 4 Sources, IC-Weighted

```
REGIME DETECTOR (4-state)
     BULL_TREND        Factor: 0.45  ML: 0.35  Pump: 0.10  StatArb: 0.10
     BEAR_TREND        Factor: 0.25  ML: 0.20  Pump: 0.20  StatArb: 0.35
     HIGH_VOL_LATERAL  Factor: 0.15  ML: 0.15  Pump: 0.35  StatArb: 0.35
     LOW_VOL_GRIND     Factor: 0.30  ML: 0.30  Pump: 0.15  StatArb: 0.25
```

---

## Running the Engine

### Option 1 — Windows One-Click (recommended for local)

Double-click **`RUN_AZALYST.bat`** — guides through 2 prompts then runs fully unattended:

1. **Select compute** — `[1] GPU` (RTX 2050, ~4x faster) or `[2] CPU`
2. **Confirm start** — `Y` to launch

The batch file auto-detects Python, GPU availability, and auto-installs all missing packages on first run. It runs `azalyst_v4_engine.py` with full logging.

**What you'll see during training:**
```
  Week  4 [Y2] | ret=+0.32%  IC=+0.0312  cum=+1.2%  DD=-0.4%  n=42  BULL_TREND
  Week  8 [Y2] | ret=-0.15%  IC=-0.0103  cum=+0.8%  DD=-0.6%  n=38  LOW_VOL_GRIND
  Week 13 [Y2]: EXPANDING RETRAIN (data up to 2024-06-15)...
    IC filter: 56 -> 48 features
    AUC=0.5234  IC=0.0156  ICIR=0.4821  (42.3s)
  ...
```

### Option 2 — Kaggle + Google Drive (recommended for full 444-coin run)

The Kaggle pipeline uses two notebooks with **Google Drive** as intermediate cache (sidesteps Kaggle's 20GB limit).

#### One-time Google Drive setup
1. [console.cloud.google.com](https://console.cloud.google.com) → create project → enable **Google Drive API**
2. Create **Service Account** → Keys → JSON → download key file
3. Google Drive → create folder `azalyst-feature-cache` → copy folder ID from URL
4. Share folder with service account email → **Editor** access
5. Kaggle notebooks → **Add-ons → Secrets** → add `GDRIVE_SERVICE_KEY` (paste full JSON)
6. Set `GDRIVE_FOLDER_ID` in Cell 1 of both notebooks

**Step 1 — Build Feature Cache** (`Notebooks/azalyst_1_feature_cache.ipynb`)
- Build 56 features for all 444 coins → upload to Google Drive → auto-cleanup
- Runtime: ~3-4 hours | Resume-safe

**Step 2 — Train + Backtest** (`Notebooks/azalyst_2_train.ipynb`)
- Downloads cache from Drive → v4 expanding window training → walk-forward Y2+Y3
- All v4 logic inlined (Kaggle can't import local modules)
- VRAM cap: 4M rows (T4 16GB)

### Option 3 — Local Jupyter (RTX 2050 / any GPU)

Open **`Notebooks/azalyst_jupyter.ipynb`** in VSCode or Jupyter. Run cells in order.
- Imports from local modules (`azalyst_v4_engine`, `azalyst_db`, `azalyst_risk`)
- VRAM cap: 2M rows (RTX 2050 4GB)
- SQLite persistence to `results/azalyst.db`

### Option 4 — CLI

```bash
# GPU run
python azalyst_v4_engine.py --gpu

# CPU run with custom drawdown limit
python azalyst_v4_engine.py --max-dd -0.10

# Skip SHAP for faster iteration
python azalyst_v4_engine.py --gpu --no-shap

# Custom run ID
python azalyst_v4_engine.py --gpu --run-id experiment_01
```

---

## Outputs

| File | Description |
|---|---|
| `results/weekly_summary_v4.csv` | Week-by-week IC, returns, regime, drawdown |
| `results/all_trades_v4.csv` | All simulated trades with meta-sizing |
| `results/performance_v4.json` | Final metrics incl. Y2 vs Y3 split, VaR/CVaR |
| `results/performance_v4.png` | 4-panel chart: cumulative return, distribution, IC series, trade P&L |
| `results/azalyst.db` | SQLite database with full run history |
| `results/shap/shap_importance_v4_*.csv` | SHAP feature importance per training cycle |
| `results/models/model_v4_*.json` | XGBoost models (base + quarterly retrains) |

---

## Repository Map

### v4 Core Pipeline

| File | Purpose |
|---|---|
| `azalyst_v4_engine.py` | **v4 engine** — expanding window, regime-aware IC selection, risk integration, kill-switch, SHAP, SQLite |
| `azalyst_db.py` | SQLite persistence — trades, metrics, SHAP, model artifacts (7 tables, WAL mode) |
| `azalyst_factors_v2.py` | 56 cross-sectional features — returns, volume, microstructure, WorldQuant alphas, Hurst, FFT, frac. diff |
| `azalyst_risk.py` | Portfolio risk — MVO, HRP, Black-Litterman, VaR/CVaR, position constraints |
| `azalyst_signal_combiner.py` | IC-weighted regime-adaptive signal fusion — 4 sources, 4-state detector |
| `azalyst_tf_utils.py` | Timeframe-aware bar count utilities |
| `build_feature_cache.py` | Precompute features → parquet cache (5–20x speedup) |
| `RUN_AZALYST.bat` | Windows one-click launcher — GPU detection, auto-install, runs v4 engine |

### Research Modules

| File | Purpose |
|---|---|
| `azalyst_train.py` | Training module with PurgedTimeSeriesCV (pre-v4 architecture, reference) |
| `azalyst_alpha_metrics.py` | Performance evaluation — IC, ICIR, Sharpe, drawdown |
| `azalyst_weekly_loop.py` | Pre-v4 walk-forward loop (reference) |
| `azalyst_ml.py` | ML module v2 — regime detection, pump/dump detector |
| `azalyst_local_gpu.py` | Standalone RTX 2050 GPU runner |
| `azalyst_statarb.py` | Cointegration / pairs trading scanner |
| `azalyst_validator.py` | Fama-MacBeth, Newey-West, Benjamini-Hochberg correction |
| `azalyst_benchmark.py` | BTC buy-hold + equal-weight benchmarks |
| `azalyst_tearsheet.py` | Factor tear sheet generator |
| `azalyst_execution.py` | Order book simulation, VWAP/TWAP execution algos |
| `azalyst_auditor.py` | Binance copy-trader strategy auditor |
| `azalyst_report.py` | Research report + live signal scanner |
| `VIEW_TRAINING.py` | Live 4-panel training dashboard — win rate, PnL, Sharpe, log tail (refreshes every 5s) |
| `monitor_dashboard.py` | Browser-based live monitor |

### Notebooks

| File | Purpose |
|---|---|
| `Notebooks/azalyst_jupyter.ipynb` | Local GPU notebook — v4 pipeline, imports local modules, 2M row VRAM cap |
| `Notebooks/azalyst_1_feature_cache.ipynb` | Kaggle Step 1 — build 56 features, upload to Google Drive |
| `Notebooks/azalyst_2_train.ipynb` | Kaggle Step 2 — v4 training + walk-forward, all logic inlined for Kaggle |

### Tests

```bash
pytest -v tests/test_azalyst.py   # 34 tests
```

---

## Testing

```bash
pytest -v tests/test_azalyst.py
```

Tests cover: SQLite persistence, feature engineering (frac. diff), risk module (VaR/CVaR/HRP), v4 engine components (PurgedTimeSeriesCV, training matrix, regime detection, feature IC selection, drawdown), signal combiner, and integration roundtrip.

---

## How to Interpret Results

| Metric | Acceptable | Good | Strong |
|---|---|---|---|
| IC | > 0.01 | > 0.03 | > 0.05 |
| ICIR | > 0.2 | > 0.5 | > 1.0 |
| Sharpe | > 0.3 | > 0.7 | > 1.5 |
| IC % positive weeks | > 52% | > 58% | > 65% |

IC > 0.05 with ICIR > 1.0 is institutional-quality signal strength.

---

## Technical Specifications

| Parameter | Value |
|---|---|
| XGBoost trees | 1,000 (primary) / 500 (meta) |
| Learning rate | 0.02 |
| Max depth | 6 (primary) / 4 (meta) |
| Min child weight | 30 (primary) / 50 (meta) |
| Subsample | 0.8 |
| Column sample | 0.7 (tree) / 0.7 (level) |
| Regularisation | alpha=0.1, lambda=1.0 |
| CV splits | 5, purged (48-bar gap) |
| VRAM guard | 2M rows (RTX 2050) / 4M rows (T4) |
| Training | Expanding window (Y1 → Y1+Y2 → Y1+Y2+Y3) |
| Walk-forward | Y2 + Y3 (2-year strict OOS) |
| Retrain | Every 13 weeks (quarterly, expanding window) |
| Feature selection | Rolling 8-week IC, threshold -0.02, min 20 features |
| Kill-switch | -15% max drawdown, 4-week pause |
| Risk cap | 3% portfolio risk per position (VaR-based) |
| Universe | 444 coins, cross-sectional pooling |
| Horizon | 4H (48 × 5-min bars) |
| Portfolio | Long top 15%, short bottom 15% |
| Fees | 0.2% round-trip, position-tracked |
| Frac. diff. d | 0.4 (FFD method, threshold 1e-5) |

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

## Installation

**Easiest:** Double-click `RUN_AZALYST.bat` — auto-installs everything on first run.

**Manual:**

```bash
pip install -r requirements.txt
```

For local GPU:

```bash
pip install xgboost --upgrade
python -c "import xgboost; print(xgboost.__version__)"
```

---

## Data Requirements

Place Binance 5-minute parquet files in `data/`:

```
timestamp | open | high | low | close | volume
```

444 symbols × 3 years × 5-min bars = 26M+ rows.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No GPU detected | `python -c "import xgboost as xgb; print(xgb.__version__)"` — verify CUDA build |
| GPU found but CPU-only menu appears | Fixed in current `RUN_AZALYST.bat` — CMD variable scoping bug resolved |
| Feature cache stale | Delete `feature_cache/` and re-run — rebuilds automatically |
| OOM / freeze | Reduce `MAX_TRAIN_ROWS` in config (2M for RTX 2050, 4M for T4) |
| Pipeline closes immediately | Confirm Python path has no spaces; use `RUN_AZALYST.bat` |
| BAT says "Pipeline completed" but no results | Check `results/` for v4 output files. If empty, check data folder has `.parquet` files |
| Kaggle notebook 1 shows fewer files | Re-run — skip logic resumes from where it stopped |
| Live dashboard shows no data | Normal until engine completes its first week — charts populate automatically |

---

## Research Principles

- **Strict OOS** — Y2+Y3 walk-forward never touches training data
- **Transparency** — every decision documented, every metric logged
- **Repeatable** — same code, same data, same results
- **Evidence over claims** — results are observations, not promises
- **Position-aware costs** — fee simulation reflects real-world execution

---

## Disclaimer

This is a research and educational project. Not financial advice. Past performance does not indicate future results. Use at your own risk. Always do your own research.

---

<div align="center">

Built by [Azalyst](https://github.com/gitdhirajsv) | Azalyst Quant Research

</div>
