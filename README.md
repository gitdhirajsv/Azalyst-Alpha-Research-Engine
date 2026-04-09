# Azalyst Alpha Research Engine

An institutional-style quantitative research platform built as a personal project. Not a hedge fund. Not a financial product. Just a passion for systematic research.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)
![Features](https://img.shields.io/badge/Features-10%20Stable-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-ElasticNet%20%2B%20XGBoost%20Challenger-blueviolet?style=flat-square)
![CV](https://img.shields.io/badge/CV-Purged%20K--Fold-orange?style=flat-square)
![Version](https://img.shields.io/badge/Engine-v6.0-gold?style=flat-square)

</div>

---

## Overview

Azalyst Alpha Research Engine is a research infrastructure project for discovering and validating systematic alpha signals in cryptocurrency markets. It is designed as a rigorous quantitative research system — not a trading bot, not a signal service, not a financial product.

The engine is a consensus rebuild synthesized from recommendations by 7 independent AI models (GPT 5.4, Gemini 3.1 Pro, Claude Opus 4.6, Qwen, DeepSeek, GLM5, Mistral) after a comprehensive audit. All 7 recommendations were evaluated, ranked, and merged into a single consensus engine.

The engine processes 3+ years of 5-minute OHLCV data across 444 Binance pairs, engineers 72 cross-sectional features (selecting 10 stable ones for training), trains an Elastic Net model with optional XGBoost challenger using purged K-Fold cross-validation with a rolling 13-week training window, and validates strictly out-of-sample across 2 full years (Y2+Y3).

---

## v6 Architecture

```
                     AZALYST v6.0 CONSENSUS REBUILD ENGINE

  DATA LAYER              FEATURE ENGINE            SIGNAL SOURCES
 Polars+DuckDB    72 computed,          Beta-neutral return pred
 444 coins               10 stable selected         Regime-gated portfolio
 26M+ rows               Stability-tracked          Falsification campaign
 3-year 5min             Turnover cap ≤3            4-gate kill criteria

                          REGIME GATING
                         BULL_TREND → long-only, half size
                         HIGH_VOL   → half position size
                         BEAR/LOW   → full long-short

  PRIMARY MODEL         CHALLENGER MODEL        WALK-FORWARD
 ElasticNetCV           XGBoost (optional)      Rolling 13-week window
 Purged K-Fold          Must beat EN by IC      Walk Y2+Y3 (2yr)
 12-bar horizon         IC margin = 0.005       Quarterly retrain
 Beta-neutral target                            IC-gated adoption

  RISK INTEGRATION      KILL CRITERIA           PERSISTENCE
 Vol-scaled longs       4-gate evaluation       SQLite (azalyst_v6.db)
 1× leverage            OOS IC positive         Feature stability log
 Top-5 per side         Feature Jaccard >0.5    Long/short PnL decomp
 0.2% round-trip fee    Regime survival ≥2      Full run history
                        ML beats baseline
```

### Core Capabilities

- **10 stable features** — 3 core (`ret_1w`, `ret_3d`, `vol_regime`) + 7 stable, with turnover cap ≤3 per retrain
- **Elastic Net regression** — interpretable linear model with built-in alpha/l1_ratio cross-validation
- **XGBoost challenger** — optional second model that must beat Elastic Net by IC margin to be adopted
- **Beta-neutral target** — daily cross-sectional demeaned forward returns, eliminates the need for `--force-invert`
- **Rolling 13-week training window** — uses only recent data, adapts to changing market conditions
- **Regime-gated portfolio** — no shorts in BULL_TREND, vol-scaled long sizing, HIGH_VOL_LATERAL skip option
- **Feature stability tracking** — Jaccard overlap across retrains, positive IC in ≥2 periods to add, negative IC in ≥3 to drop
- **Built-in falsification campaign** — single-factor baselines vs ML to prove signal exists
- **4-gate kill criteria** — OOS IC + feature stability + regime survival + beat baseline
- **IC-gated retraining** — model updates rejected if OOS IC ≤ 0
- **Long/short PnL decomposition** — separate tracking of each portfolio leg
- **2-year out-of-sample** — walk-forward on Y2+Y3 (104 weeks, never seen during initial training)
- **SQLite persistence** — all trades, metrics, models in `results_v6/azalyst_v6.db`

---

## Feature Engineering — 10 Stable Features

72 cross-sectional features are computed from raw OHLCV data, but only 10 are selected for the model based on stability and economic interpretability.

### Core Features (never dropped)

| Feature | Description |
|---|---|
| `ret_1w` | 1-week return — momentum/reversal signal |
| `ret_3d` | 3-day return — short-term momentum |
| `vol_regime` | Volatility regime — state variable |

### Stable Features (default set)

| Feature | Description |
|---|---|
| `rvol_1d` | Daily realized volatility |
| `rsi_14` | Mean reversion indicator |
| `skew_1d` | Distribution asymmetry / tail risk |
| `adx_14` | Trend strength |
| `kyle_lambda` | Price impact / liquidity |
| `mean_rev_zscore_1h` | Z-score of 1hr mean reversion |
| `vol_ratio_1h_1d` | Intraday vs daily volatility ratio |

### Feature Stability Rules

- **Turnover cap**: max 3 features added/removed per retrain
- **Add rule**: positive IC in ≥2 recent periods
- **Drop rule**: negative IC in ≥3 recent periods
- **Candidates**: `ret_1d`, `ret_2d`, `rev_1h`, `rev_1d`, `rvol_4h`, `atr_norm`, `cci_14`, `bb_pos`, `vwap_dev`, `amihud`, `trend_strength`, `frac_diff_close`, `vol_ret_1d`

---

## ML Pipeline

### Training Target — Beta-Neutral Forward Returns

The model predicts the **beta-neutral forward return** — the raw 1hr forward log return minus the daily cross-sectional mean. This removes market-wide moves and focuses on relative outperformance.

$$\hat{r}^{*}_{i,t+12} = f(X_{i,t}) \quad \text{where } r^{*} = r_{i} - \bar{r}_{\text{day}} \text{ (demeaned)}$$

### Prediction → Trade Signal

- **Predicted return > 0** → candidate long (ranked by magnitude)
- **Predicted return < 0** → candidate short (ranked by magnitude)
- **Long filter** — only enter longs where predicted return > 0 (prevents low-beta fake signals)
- **Top-N mode** (default) — top 5 longs + bottom 5 shorts per week
- **Vol-scaled long sizing** — long positions sized inversely to `rvol_1d`; shorts equal-weight
- **PnL cap** — no single position can lose more than 100%

### Regime-Gated Portfolio

The portfolio is regime-aware with concrete position rules:

| Regime | Longs | Shorts | Position Scale |
|---|---|---|---|
| BULL_TREND | Top-N (pred > 0 only) | **None** | 0.5×, vol-scaled by rvol_1d |
| HIGH_VOL_LATERAL | Top-N (pred > 0 only) | Bottom-N | 0.5× (or skip with `--no-trade-high-vol`) |
| BEAR_TREND | Top-N (pred > 0 only) | Bottom-N | 1.0× |
| LOW_VOL_GRIND | Top-N (pred > 0 only) | Bottom-N | 1.0× |

### IC-Gated Retraining

Model retraining occurs every 13 weeks (quarterly) using a rolling 13-week window. New models are only adopted if OOS IC > 0 — otherwise the previous model is kept.

```
If retrained model IC > 0        →  ADOPT new model
If retrained model IC ≤ 0        →  REJECT, keep previous
If cumulative DD < -20%          →  PAUSE until DD recovers above -10%
```

### Falsification Campaign

Before trusting ML predictions, the engine runs single-factor baselines:
- `ret_1w` rank alone
- `ret_3d` rank alone
- `vol_regime` rank alone
- Composite rank (average of core features)
- Random predictions (null hypothesis)

ML must beat the best baseline IC to justify its complexity.

### 4-Gate Kill Criteria

All 4 gates must pass for the strategy to continue:
1. **OOS IC positive** — mean IC > 0, positive in >50% of weeks
2. **Feature stability** — Jaccard overlap > 0.5 across retrains
3. **Regime survival** — positive returns in ≥2 regimes
4. **ML beats baseline** — ML IC ≥ best single-factor baseline IC

---

## Running the Engine

### Option 1 — Windows One-Click

Double-click **`RUN_AZALYST.bat`** — auto-detects GPU, installs dependencies, runs the v6 engine with Elastic Net, beta-neutral target, regime-gated portfolio, rolling 13-week window, top-5 per side.

### Option 2 — CLI

```bash
# Default — Elastic Net, CPU, top-5 per side
python azalyst_v6_engine.py --no-gpu --top-n 5

# With XGBoost challenger + GPU
python azalyst_v6_engine.py --gpu --xgb-challenger --top-n 5

# Custom top-N (e.g. top 10 longs + 10 shorts)
python azalyst_v6_engine.py --gpu --top-n 10

# Custom rolling window (e.g. 52 weeks instead of 26)
python azalyst_v6_engine.py --no-gpu --rolling-window 52

# Skip falsification campaign
python azalyst_v6_engine.py --no-gpu --no-falsify
```

**What you'll see during training:**
```
  AZALYST v6  —  Consensus Rebuild Engine
  Model        : Elastic Net (linear)
  Target       : future_ret_1h (beta-neutral)
  Window       : Rolling 26 weeks
  Features     : 10 stable + turnover cap 3
  Portfolio    : top-5 per side, regime-gated

  Week  4 [Y2]:   5 trades (5L/0S) | ret=+0.32% (L=+0.32% S=+0.00%) | IC=+0.0312 | cum=+1.2% | DD=0.4% | BULL_TREND
  Week 13 [Y2]: QUARTERLY RETRAIN (rolling 26wk to 2024-06-15)...
    [v6_w013] ElasticNet: alpha=0.000312  l1_ratio=0.50  nonzero=8/10  R²=0.0023  IC=0.0156  ICIR=0.4821
    Adopted new model (IC=+0.0156 > 0)  (2.3s)
  ...
```

---

## Outputs

All output files are written to `results_v6/` (default) or the directory passed via `--out-dir`.

| File | Description |
|------|-------------|
| `results_v6/checkpoint_v6_latest.json` | Live checkpoint — weekly summary, trades, run state |
| `results_v6/run_log_v6.txt` | Full pipeline log |
| `results_v6/train_summary_v6.json` | Training metrics (model type, IC, R², beta-neutral flag) |
| `results_v6/feature_importance_v6_*.csv` | Feature importances per retrain cycle |
| `results_v6/all_trades_v6.csv` | All trades with long/short PnL decomposition |
| `results_v6/weekly_summary_v6.csv` | Weekly metrics (return, IC, regime, long/short PnL) |
| `results_v6/performance_v6.json` | Final performance report + 4-gate kill criteria |
| `results_v6/azalyst_v6.db` | SQLite database with full run history |
| `results_v6/models/` | Elastic Net / XGBoost models (base + quarterly retrains) |

---

## Live Dashboard (Spyder Monitor)

`VIEW_TRAINING.py` is a 4-panel live dashboard that displays real-time training progress. It is auto-launched by `RUN_AZALYST.bat` when you choose **Monitor = 1 (Terminal + Spyder)**.

**Panels:**

| Panel | What it shows |
|---|---|
| Training Quality by Cycle | Win rate % and rolling 4-week Sharpe per week |
| PnL and Drawdown | Cumulative return curve + max drawdown curve |
| Current Status | Run state, week, trade count, win rate, Sharpe, drawdown, profit factor |
| Recent Log Tail | Last 18 lines from `run_log.txt` (live engine output) |

**How to use:**
- **Auto:** Select `Monitor: 1` in `RUN_AZALYST.bat` — dashboard opens automatically
- **Manual:** Run `python VIEW_TRAINING.py` from a terminal, or press **F5** in Spyder
- Refreshes every **5 seconds** — reads `results_v6/checkpoint_v6_latest.json` and `results_v6/run_log_v6.txt`
- Close the window or press `Ctrl+C` to exit

---

## Repository Map

### Core Pipeline

| File | Purpose |
|---|---|
| `azalyst_v6_engine.py` | **v6 engine** — Elastic Net consensus rebuild, beta-neutral target, regime-gated portfolio, rolling window, falsification campaign, 4-gate kill criteria |
| `azalyst_v5_engine.py` | Shared infrastructure — `LazySymbolStore`, `detect_regime`, `build_feature_store`, `PurgedTimeSeriesCV`, checkpoint utilities |
| `azalyst_db.py` | SQLite persistence — trades, metrics, model artifacts (7 tables, WAL mode) |
| `azalyst_factors_v2.py` | 72 cross-sectional features — reversal signals, pump-dump indicators, quantile rank, WQ alphas, frac. diff |
| `azalyst_pump_dump.py` | Multi-signal pump-dump detector with regime classification |
| `azalyst_train.py` | Training utilities — `compute_ic`, PurgedTimeSeriesCV |
| `azalyst_risk.py` | Portfolio risk — MVO, HRP, Black-Litterman, VaR/CVaR, position constraints |
| `build_feature_cache.py` | Precompute features → parquet cache (5–20x speedup) |
| `validate_startup.py` | Pre-flight checks — directories, modules, engine config |
| `VIEW_TRAINING.py` | **Live 4-panel Spyder Monitor** — reads `results_v6/` every 5 s, shows PnL, win rate, Sharpe, log tail |
| `RUN_AZALYST.bat` | Windows one-click launcher — GPU detection, auto-install, optional dashboard launch |

---

## Technical Specifications

| Parameter | Value |
|---|---|
| Primary model | ElasticNetCV (`l1_ratio` ∈ [0.1, 0.3, 0.5, 0.7, 0.9], 50 alphas) |
| Challenger model | XGBoost (`--xgb-challenger`, max_depth=3, n_estimators=500) |
| Target | Beta-neutral 1hr forward log return (daily cross-sectional demeaned) |
| CV splits | 5, purged (48-bar gap) |
| VRAM guard | 2M rows max training matrix |
| Training | Rolling 13-week window (configurable via `--rolling-window`) |
| Walk-forward | Y2 + Y3 (2-year strict OOS) |
| Retrain | Every 13 weeks (quarterly), IC-gated adoption (reject if IC ≤ 0) |
| Features | 10 stable (3 core + 7 stable), turnover cap ≤3, regime-conditional IC tracking |
| Regime gating | BULL_TREND → long-only, vol-scaled longs; HIGH_VOL_LATERAL → optional skip |
| Kill criteria | 4-gate: OOS IC + feature stability + regime survival + beat baseline |
| DD kill-switch | -20% max drawdown, resume when DD recovers above -10% |
| Universe | 444 coins cross-sectional pooling |
| Horizon | 1hr (12 × 5-min bars) default |
| Portfolio | Top-5 per side (default), vol-scaled longs, 1× leverage |
| Fees | 0.2% round-trip, position-tracked |
| Falsification | Single-factor baselines (ret_1w, ret_3d, vol_regime, composite, random) |

---

## Theoretical Foundations

| Domain | Source | What Azalyst Uses |
|---|---|---|
| Feature engineering | **Lopez de Prado** — *Advances in Financial Machine Learning* | Fractional differentiation, purged K-Fold CV |
| Signal combination | **Grinold & Kahn** — *Active Portfolio Management* | IC-weighted signal fusion, information ratio targeting |
| Model selection | **Consensus of 7 AI models** | Elastic Net default, XGBoost challenger, falsification campaign |
| Robust estimation | **Huber** — *Robust Statistics* (via RobustScaler) | Median/IQR scaling for fat-tailed crypto distributions |
| Factor models | **Fama & French**, **Barra** | Cross-sectional alpha, beta-neutral target |
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
| No results after run | Check `results_v6/` for output files. If empty, check data folder has `.parquet` files |
| Dashboard shows "IDLE" | Engine hasn't started yet — launch `VIEW_TRAINING.py` after starting the engine, not before |
| Dashboard not found | `VIEW_TRAINING.py` must be in the engine root folder — restore it with `git checkout HEAD -- VIEW_TRAINING.py` |

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
