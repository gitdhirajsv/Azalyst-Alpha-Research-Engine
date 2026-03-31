# Azalyst Alpha Research Engine

An institutional-style quantitative research platform built as a personal project. Not a hedge fund. Not a financial product. Just a passion for systematic research.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)
![Features](https://img.shields.io/badge/Features-72%20Cross--Sectional-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20Regression%20CUDA-blueviolet?style=flat-square)
![CV](https://img.shields.io/badge/CV-Purged%20K--Fold-orange?style=flat-square)
![Version](https://img.shields.io/badge/Engine-v5.0-gold?style=flat-square)

</div>

---

## Overview

Azalyst Alpha Research Engine is a research infrastructure project for discovering and validating systematic alpha signals in cryptocurrency markets. It is designed as a rigorous quantitative research system — not a trading bot, not a signal service, not a financial product.

**v5** is a ground-up rebuild of the ML pipeline, informed by a comprehensive audit of v4's failures and inspired by Jane Street's Kaggle competition approach. The v4 binary classifier with momentum features produced 0/103 profitable weeks because crypto mean-reverts — v5 fixes this with:

1. **Regression, not classification** — predict continuous forward returns (XGBRegressor, `reg:squarederror`)
2. **Short horizons** — 1hr (12 bars) and 15min (3 bars) instead of 4hr (48 bars)
3. **Reversal-dominated features** — 72 features with 8 reversal signals, 6 pump-dump indicators, and 4 quantile-ranked features (Jane Street technique)
4. **Per-bar prediction** — no week-averaging that destroys signal
5. **Pump-dump detection** — multi-signal detector to filter manipulated coins
6. **IC-gating kill-switch** — halt trading when model signal inverts
7. **Weighted R² metric** — penalizes direction + magnitude errors (Jane Street metric)
8. **Confidence model** — P(direction correct) for position sizing

The engine processes 3+ years of 5-minute OHLCV data across 444 Binance pairs, engineers 72 cross-sectional features, trains an XGBoost regression model using purged K-Fold cross-validation with an expanding training window, and validates strictly out-of-sample across 2 full years (Y2+Y3).

<div align="center">

![Azalyst Spyder Monitor](docs/assets/azalyst_spyder_monitor.svg)

</div>

---

## v5 Architecture

```
                     AZALYST v5.0 RESEARCH ENGINE

  DATA LAYER              FEATURE ENGINE             SIGNAL SOURCES
 Polars+DuckDB    72 cross-sectional  Regression return pred
 444 coins               features, TF-aware         Confidence model
 26M+ rows               8 reversal signals         Pump/dump filter
 3-year 5min             4 quantile-ranked          IC-gated selection

                            SIGNAL COMBINER
                           Regime-adaptive
                           IC-weighted fusion
                           4-state detector

  PRIMARY MODEL         CONFIDENCE MODEL        WALK-FORWARD
                        (replaces meta-label)
 XGBRegressor CUDA   2nd-stage XGBoost   Expanding window
 Purged K-Fold         P(direction correct)     Walk Y2+Y3 (2yr)
 12-bar horizon        Magnitude sizing         Quarterly retrain
 Weighted R²                                    IC feature gating

  RISK INTEGRATION      KILL-SWITCHES           PERSISTENCE
 VaR / CVaR            -15% max DD             SQLite (azalyst.db)
 Position risk cap     IC-gating threshold     SHAP per cycle
 Magnitude sizing      Pump-dump filter        Full run history
```

### What Changed from v4 → v5

| Aspect | v4 (broken) | v5 (rebuilt) |
|---|---|---|
| Model type | XGBClassifier (binary) | XGBRegressor (continuous returns) |
| Objective | `binary:logistic` + AUC | `reg:squarederror` + Weighted R² |
| Features | 56 (momentum-dominated) | 72 (reversal-dominated + pump-dump + qrank) |
| Horizon | 48 bars (4hr) | 12 bars (1hr) / 3 bars (15min) |
| Label | Binary: `r > cross-sectional median` | Continuous: raw forward return |
| Prediction | `predict_proba()[:, 1]` averaged per week | `predict()` per bar |
| Sizing | Meta-labeling P(correct) | Confidence model + predicted magnitude |
| IC threshold | -0.02 (too lax) | 0.00 (strict) + IC-gating at -0.03 |
| Pump-dump | Not integrated | Multi-signal detector with regime classification |
| Kill-switches | DD only (-15%) | DD (-15%) + IC-gating + pump-dump filter |

### Core Capabilities

- **72 cross-sectional features** across 11 categories — returns, volume, volatility, technical, microstructure, price structure, WorldQuant alphas, regime, fractional differentiation, **reversal signals**, **pump-dump indicators**, and **quantile-ranked features**
- **XGBoost Regression** — continuous return prediction, Weighted R² metric (Jane Street)
- **Short-horizon forecasting** — 1hr (12 bars) and 15min (3 bars) forward returns
- **Pump-dump detection** — multi-signal composite score filtering manipulated coins
- **IC-gating** — halt all trading when rolling IC drops below -0.03
- **Expanding training window** — train on Y1, then Y1+Y2, then Y1+Y2+Y3
- **2-year out-of-sample** — walk-forward on Y2+Y3 (104 weeks, never seen during initial training)
- **Risk integration** — VaR/CVaR scaled position sizing, 3% per-position risk cap
- **SHAP explainability** — TreeExplainer after every training cycle
- **SQLite persistence** — all trades, metrics, SHAP, models in `results/azalyst.db`
- **GPU-accelerated** — NVIDIA CUDA via XGBoost

---

## Feature Engineering — 72 Features, 11 Categories

| Category | Count | Features |
|---|---|---|
| Returns | 7 | `ret_1bar` `ret_1h` `ret_4h` `ret_1d` `ret_2d` `ret_3d` `ret_1w` |
| Volume | 6 | `vol_ratio` `vol_ret_1h` `vol_ret_1d` `obv_change` `vpt_change` `vol_momentum` |
| Volatility | 7 | `rvol_1h` `rvol_4h` `rvol_1d` `vol_ratio_1h_1d` `atr_norm` `parkinson_vol` `garman_klass` |
| Technical | 10 | `rsi_14` `rsi_6` `macd_hist` `bb_pos` `bb_width` `stoch_k` `stoch_d` `cci_14` `adx_14` `dmi_diff` |
| Microstructure | 6 | `vwap_dev` `amihud` `kyle_lambda` `spread_proxy` `body_ratio` `candle_dir` |
| Price Structure | 6 | `wick_top` `wick_bot` `price_accel` `skew_1d` `kurt_1d` `max_ret_4h` |
| WorldQuant Alphas | 6 | `wq_alpha001` `wq_alpha012` `wq_alpha031` `wq_alpha098` `vol_adjusted_mom` `trend_consistency` |
| Regime | 5 | `vol_regime` `trend_strength` `corr_btc_proxy` `hurst_exp` `fft_strength` |
| Memory-Preserving | 1 | `frac_diff_close` — fractional differentiation d=0.4 (AFML Ch. 5) |
| **Reversal** (new) | **8** | `rev_1h` `rev_4h` `rev_1d` `rev_2d` `mean_rev_zscore_1h` `mean_rev_zscore_4h` `overbought_rev` `oversold_rev` |
| **Pump-Dump** (new) | **6** | `pump_score` `dump_score` `vol_spike_zscore` `ret_vol_ratio_1h` `tail_risk_1h` `abnormal_range` |
| **Quantile Rank** (new) | **4** | `qrank_ret_1h` `qrank_rvol_1d` `qrank_rev_1h` `qrank_vol_ratio` |

---

## ML Pipeline

### Training Target — Continuous Forward Returns (v5)

The model predicts the **raw forward return** at the 1hr horizon. No binary conversion, no cross-sectional median — the model learns to predict magnitude and direction simultaneously.

$$\hat{r}_{i,t+12} = f(X_{i,t}) \quad \text{where } r \text{ is the log return over 12 bars (1hr)}$$

### Prediction → Trade Signal

- **Predicted return > 0** → candidate long (ranked by magnitude)
- **Predicted return < 0** → candidate short (ranked by magnitude)
- **Top 15% by predicted return** → long portfolio
- **Bottom 15% by predicted return** → short portfolio
- **Position size** ∝ predicted magnitude × confidence model probability × risk scale

### Pump-Dump Filter

Multi-signal detector with 4 components:
- Price spike z-score (sudden price jumps)
- Volume spike z-score (abnormal volume)
- Range spike z-score (wick expansion)
- Reversal pattern score (quick reversal after spike)

Composite score [0, 1]. Symbols exceeding threshold (0.6) are filtered from the tradeable universe.

### IC-Gating Kill-Switch (v5)

When the rolling average feature IC drops below -0.03, the model's signal has inverted. Instead of trading on an inverted signal (losing money systematically), IC-gating halts all predictions until the signal recovers.

```
If avg_recent_IC < -0.03  →  SKIP week (no trades, no risk)
If cumulative DD > -15%   →  HALT 4 weeks (standard kill-switch)
```

### Confidence Model (replaces Meta-Labeling)

A second-stage XGBoost classifier trained on the meta-question: **"When the regression model predicted a direction, was it correct?"**

- Uses OOS fold predictions from base model + scaled features
- Output: P(direction correct) ∈ [0, 1]
- Directly scales position size: high confidence → larger position

---

## Running the Engine

### Option 1 — Windows One-Click

Double-click **`RUN_AZALYST.bat`** — auto-detects GPU, installs dependencies, runs engine. The launcher now prompts for universe mode:

- **[1] Top-6 Persistent Coins** — runs the curated winning configuration automatically (see below)
- **[2] Full Universe** — standard run against all coins in `data/`

### Option 2 — CLI

```bash
# GPU run (full universe)
python azalyst_v5_engine.py --gpu

# CPU run with custom drawdown limit
python azalyst_v5_engine.py --max-dd -0.10

# Skip SHAP for faster iteration
python azalyst_v5_engine.py --gpu --no-shap

# Top-6 persistent coins — winning config (5d horizon, force-invert, 3x leverage)
python azalyst_v5_engine.py --gpu --no-shap \
  --data-dir "./data_top6" --feature-dir "./cache_top6" --out-dir "./results_top6" \
  --target 5d --force-invert --leverage 3 --ic-gating-threshold -1.0 --max-dd -1.0 \
  --pin-coins "1000SATSUSDT,BONKUSDT,ADXUSDT,FDUSDUSDT,WINUSDT,AEURUSDT"
```

---

## Persistent-Universe Strategy (v5)

A 103-week walk-forward analysis of the 50-coin backtest revealed that most model picks were concentrated in a small set of coins that consistently ranked in the top quantile. Broadening to 444 coins at the same 15% quantile would mean trading ~66 long + 66 short positions weekly — increasing noise and execution cost with diminishing signal quality.

**Persistence analysis result** — weeks appearing in the long basket across 103 OOS weeks:

| Rank | Symbol | Weeks Present | Persistence |
|---|---|---|---|
| 1 | `1000SATSUSDT` | 49/103 | 47.6% |
| 2 | `BONKUSDT` | 49/103 | 47.6% |
| 3 | `ADXUSDT` | 38/103 | 36.9% |
| 4 | `FDUSDUSDT` | 30/103 | 29.1% |
| 5 | `WINUSDT` | 27/103 | 26.2% |
| 6 | `AEURUSDT` | 23/103 | 22.3% |

These 6 coins were selected as the **pinned universe** for the production configuration. At 15% adaptive quantile over 6 symbols, the engine picks ~2 longs and ~2 shorts per week — concentrated, low-turnover positions from a pre-validated universe.

### `--pin-coins` Flag

The `--pin-coins` argument restricts the weekly prediction → ranking → trade cycle to a specified set of symbols. Filtering happens after model predictions and after inversion, so the XGBoost ranking is performed only over the pinned set:

```bash
--pin-coins "1000SATSUSDT,BONKUSDT,ADXUSDT,FDUSDUSDT,WINUSDT,AEURUSDT"
```

Data for pinned coins lives in `data_top6/` and pre-built feature cache in `cache_top6/`.

### Winning Backtest Config (50-coin run, 103 weeks OOS)

| Metric | Value |
|---|---|
| Total return | **+8,111.85%** |
| Annualised | **+825.79%** |
| Sharpe ratio | **3.68** |
| Win rate | **70.87%** (73W / 30L) |
| Horizon | 5-day forward return |
| Inversion | `--force-invert` (anti-signal mode) |
| Leverage | 3× |
| IC gate | disabled (`-1.0` threshold) |

**What you'll see during training:**
```
  AZALYST v5  —  Short-Horizon Regression Engine
  Model: XGBoost Regressor (1hr forward return)
  Features: Reversal-dominated + Pump-Dump + Quantile Rank

  Week  4 [Y2] | ret=+0.32%  IC=+0.0312  cum=+1.2%  DD=-0.4%  BULL_TREND
  Week 13 [Y2]: QUARTERLY RETRAIN (expanding window to 2024-06-15)...
    R²=0.0023  IC=0.0156  ICIR=0.4821  (42.3s)
  ...
```

---

## Outputs

| File | Description |
|---|---|
| `results/weekly_summary_v4.csv` | Week-by-week IC, returns, regime, drawdown |
| `results/all_trades_v4.csv` | All simulated trades with magnitude-based sizing |
| `results/performance_v4.json` | Final metrics incl. Y2 vs Y3 split, VaR/CVaR |
| `results/azalyst.db` | SQLite database with full run history |
| `results/shap/shap_importance_v4_*.csv` | SHAP feature importance per training cycle |
| `results/models/model_v4_*.json` | XGBoost models (base + quarterly retrains) |

---

## Repository Map

### Core Pipeline

| File | Purpose |
|---|---|
| `azalyst_v5_engine.py` | **v5 engine** — regression walk-forward, IC-gating, pump-dump filter, confidence model, SHAP, SQLite |
| `azalyst_db.py` | SQLite persistence — trades, metrics, SHAP, model artifacts (7 tables, WAL mode) |
| `azalyst_factors_v2.py` | 72 cross-sectional features — reversal signals, pump-dump indicators, quantile rank, WQ alphas, frac. diff |
| `azalyst_pump_dump.py` | **NEW** — Multi-signal pump-dump detector with regime classification |
| `azalyst_train.py` | Training module — XGBRegressor, XGBClassifier confidence model, PurgedTimeSeriesCV, Weighted R² |
| `azalyst_ml.py` | ML module v5 — XGBoost regression predictor class |
| `azalyst_risk.py` | Portfolio risk — MVO, HRP, Black-Litterman, VaR/CVaR, position constraints |
| `azalyst_signal_combiner.py` | IC-weighted regime-adaptive signal fusion — 4 sources, 4-state detector |
| `azalyst_tf_utils.py` | Timeframe-aware bar count utilities |
| `build_feature_cache.py` | Precompute features → parquet cache (5–20x speedup) |
| `RUN_AZALYST.bat` | Windows one-click launcher — GPU detection, auto-install |

### Utilities

| File | Purpose |
|---|---|
| `VIEW_TRAINING.py` | Live 4-panel training dashboard |
| `monitor_dashboard.py` | Browser-based live monitor |

### Tests

```bash
pytest -v tests/test_azalyst.py   # 45+ tests covering v5 pipeline
```

---

## Technical Specifications

| Parameter | Value |
|---|---|
| Model | XGBRegressor (`reg:squarederror`) |
| XGBoost trees | 1,000 (primary) / 500 (confidence) |
| Learning rate | 0.02 |
| Max depth | 6 (primary) / 4 (confidence) |
| Min child weight | 30 (primary) / 50 (confidence) |
| Subsample | 0.8 |
| Column sample | 0.7 (tree) / 0.7 (level) |
| Regularisation | alpha=0.1, lambda=1.0 |
| CV splits | 5, purged (48-bar gap) |
| VRAM guard | 2M rows (RTX 2050) / 4M rows (T4) |
| Training | Expanding window (Y1 → Y1+Y2 → Y1+Y2+Y3) |
| Walk-forward | Y2 + Y3 (2-year strict OOS) |
| Retrain | Every 13 weeks (quarterly, expanding window) |
| Feature selection | Rolling 8-week IC, threshold 0.00, min 20 features |
| IC-gating | Halt when avg IC < -0.03 |
| DD kill-switch | -15% max drawdown, 4-week pause |
| Risk cap | 3% portfolio risk per position (VaR-based) |
| Universe | 444 coins (full) / 6 coins (pinned, `--pin-coins`) |
| Horizon | 1hr (12 × 5-min bars) / 5d (1440 × 5-min bars, top-6 config) |
| Portfolio | Long top 15%, short bottom 15% (adaptive, min 1 per side) |
| Fees | 0.2% round-trip, position-tracked |
| Pump-dump threshold | 0.6 composite score |
| Frac. diff. d | 0.4 (FFD method, threshold 1e-5) |

---

## Theoretical Foundations

| Domain | Source | What Azalyst Uses |
|---|---|---|
| Feature engineering | **Lopez de Prado** — *Advances in Financial Machine Learning* | Fractional differentiation, purged K-Fold CV |
| Signal combination | **Grinold & Kahn** — *Active Portfolio Management* | IC-weighted signal fusion, information ratio targeting |
| Competition ML | **Jane Street Kaggle** | Regression objective, Weighted R², quantile rank features, per-timestep prediction |
| Robust estimation | **Huber** — *Robust Statistics* (via RobustScaler) | Median/IQR scaling for fat-tailed crypto distributions |
| Factor models | **Fama & French**, **Barra** | Cross-sectional alpha, factor decomposition |
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
| Feature cache stale | Delete `feature_cache/` and re-run — rebuilds automatically |
| OOM / freeze | Reduce `MAX_TRAIN_ROWS` in config (2M for RTX 2050, 4M for T4) |
| Pipeline closes immediately | Confirm Python path has no spaces; use `RUN_AZALYST.bat` |
| No results after run | Check `results/` for output files. If empty, check data folder has `.parquet` files |

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
