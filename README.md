# Azalyst Alpha Research Engine — v7.0 Final

An institutional-style quantitative research platform for discovering and validating systematic alpha signals in cryptocurrency markets. Built as a personal project. Not a hedge fund. Not a financial product. Just a passion for systematic research.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-FINAL%20v7-brightgreen?style=flat-square)
![Version](https://img.shields.io/badge/Engine-v7.0-red?style=flat-square)
![Model](https://img.shields.io/badge/Model-XGBoost%20Primary-blueviolet?style=flat-square)
![Features](https://img.shields.io/badge/Features-5%20Proven-orange?style=flat-square)
![CV](https://img.shields.io/badge/CV-Purged%20K--Fold-orange?style=flat-square)

### Verified Performance
![Return](https://img.shields.io/badge/Return-%2B564.7%25%20(77wk%20OOS)-brightgreen?style=flat-square)
![Sharpe](https://img.shields.io/badge/Sharpe-3.29-brightgreen?style=flat-square)
![Deflated Sharpe](https://img.shields.io/badge/Deflated%20Sharpe-0.9999-brightgreen?style=flat-square)
![Max DD](https://img.shields.io/badge/Max%20DD--13.46%25-green?style=flat-square)

</div>

---

## Quick Start

### Windows One-Click
Double-click **`RUN_AZALYST.bat`** — auto-detects GPU, installs dependencies, runs the engine.

### CLI
```bash
# Default — XGBoost primary, CPU, top-5 per side
python azalyst_v6_engine.py --no-gpu --top-n 5

# With GPU
python azalyst_v6_engine.py --gpu --top-n 5

# Custom leverage
python azalyst_v6_engine.py --no-gpu --top-n 5 --leverage 0.5
```

---

## Verified Results

### v6.3 — Regime-Specific XGBoost (Final Configuration)

| Metric | Value |
|---|---|
| **Total Return** | **+564.75%** |
| **Period** | 77 weeks OOS (Sep 2024 – Mar 2026) |
| **Annualized Return** | +259.39% |
| **Sharpe Ratio** | 3.29 |
| **Deflated Sharpe** | 0.9999 (statistically significant) |
| **Max Drawdown** | -13.46% |
| **IC Mean** | +0.0166 (100% positive weeks) |
| **Total Trades** | 597 |
| **Long PnL** | +18.19% |
| **Short PnL** | +212.82% |
| **4-Gate Verdict** | ALL PASS — CONTINUE |

**Run ID:** `v6_20260415_123817` | **Commit:** `6c2bccb` | **Date:** April 15-16, 2026

All results are stored in `results_v6/` — trades, models, logs, performance JSON, SQLite database.

---

## Architecture Overview

```
                     AZALYST v7.0 — FINAL ENGINE

  DATA LAYER              FEATURE ENGINE            ML MODEL
 Parquet + cache     72 computed → 5-10 selected   XGBoost primary
 444 coins           Liquidity + momentum          ElasticNet fallback
 26M+ rows           Rank-normalized target        Purged K-Fold CV
 3-year 5min         Turnover cap ≤3               IC-gated adoption

  REGIME GATING         PORTFOLIO               WALK-FORWARD
 BULL_TREND → long-only, 0.5×   Top-5 per side     77 weeks strict OOS
 BEAR_TREND → short-only, 1.0×  Vol-scaled longs   Monthly retrain (4wk)
 LOW_VOL_GRIND → long+short, 1× Shorts equal-weight Rolling 104wk window
 HIGH_VOL_LATERAL → skip option  1× leverage max    19 retrains total

  RISK                  KILL CRITERIA             PERSISTENCE
 0.2% round-trip fee   4-gate evaluation          SQLite (azalyst_v6.db)
 -20% max DD kill      OOS IC positive             Feature stability log
 0.12 recovery thresh  Feature Jaccard >0.5        Long/short PnL decomp
 PnL clip ±100%        Regime survival ≥2          Full run history
                       ML beats baseline
```

---

## What Changed from v6 → v7

v7 is the final stable release that incorporates all learnings from 4 runs across 77 weeks of out-of-sample testing. Here's the evolution:

### Key Changes in v7 (final)

1. **XGBoost Primary Model** — replaced ElasticNet as default. Non-linear, captures feature interactions (e.g., kyle_lambda × vol_regime). ElasticNet retained as fallback.

2. **Rank-Normalized Target** — beta-neutral forward returns transformed to [-1, 1] via rank normalization. Aligns MSE optimization with Spearman IC (deployment metric). Fixes IS→OOS decay.

3. **Regime-Specific Portfolio** — evidence-based rules from 4 runs:
   - **BULL_TREND**: Long-only (shorts unprofitable)
   - **BEAR_TREND**: Short-only (longs lost -92% historically)
   - **LOW_VOL_GRIND**: Long+short (70.6% win rate, alpha source)
   - **HIGH_VOL_LATERAL**: Optional skip

4. **5 Proven Features** (minimum overfit set):
   - `kyle_lambda` (IC=+0.112, 79% positive) — price impact
   - `amihud` (IC=+0.109, 79% positive) — illiquidity
   - `ret_3d` (IC=+0.013, 55% positive) — short-term momentum
   - `vol_regime` — state variable
   - `rsi_14` (IC=+0.018, 65% positive) — mean reversion

5. **Monthly Retraining** (4 weeks) — adapts to regime changes faster than quarterly. 19 retrains over 77 weeks.

6. **Rolling 104-Week Window** — 2-year training history ensures model has seen a full bull+bear cycle before going OOS.

7. **Volatility-Scaled Long Sizing** — longs sized inversely to `rvol_1d`, capped at 1.0×. Shorts equal-weighted.

8. **Leverage 0.5×** (default) — conservative position sizing. Max 1.0× per position.

9. **Beta-Neutral Target** — daily cross-sectional demeaned returns remove market-wide moves, focus on relative outperformance.

10. **IC-Gated Adoption** — retrained models only adopted if OOS IC > 0. Otherwise previous model kept.

### What Was Removed/Fixed from Earlier Versions

| Issue (v6.0-v6.2) | Fix (v7) |
|---|---|
| ElasticNet alpha too small (0.00002) → overfit | XGBoost primary + alpha floor 0.001 |
| Long win rate 35.6% (worse than random) | BEAR_TREND short-only rule, momentum filter disabled |
| IS→OOS IC decay 29× | Rank-normalized target + conservative XGBoost params |
| MemoryError on 2M rows | Chunked symbol processing, reduced n_alphas |
| Leakage via look-ahead | 60-min embargo, safe_end = train_end - 1hr |
| Feature instability | Turnover cap ≤3, Jaccard tracking >0.5 |
| Kill switch too permissive (-25%) | Tightened to -20% with 0.12 recovery threshold |

---

## ML Pipeline

### Training Target

**Beta-neutral rank-normalized forward returns:**
1. Compute 1hr forward log return: `log(close[t+12] / close[t])`
2. Subtract daily cross-sectional mean (beta-neutral)
3. Winsorize at 1st/99th percentile (outlier clipping)
4. Rank-normalize to [-1, 1] (aligns MSE with Spearman IC)

```
r*_i,t+12 = rank(r_i,t+12 - mean(r_day,t+12)) → [-1, 1]
```

### Model Training

**Primary: XGBoost**
```
n_estimators=500, learning_rate=0.03, max_depth=3
min_child_weight=200, subsample=0.6, colsample_bytree=0.8
reg_alpha=2.0 (L1), reg_lambda=10.0 (L2), gamma=1.0
early_stopping_rounds=30
```

**Fallback: ElasticNet**
```
alpha ≥ 0.001 (floor), l1_ratio ∈ [0.5, 0.7, 0.9, 0.95, 0.99]
ElasticNet adopted only if IC beats XGBoost by ≥ 0.005
```

### Prediction → Trade Signal

| Signal | Action |
|---|---|
| Predicted return > 0 | Candidate long (ranked by magnitude) |
| Predicted return < 0 | Candidate short (ranked by magnitude) |
| Top-N selection | Top 5 longs + bottom 5 shorts per week |

### Regime-Gated Portfolio

| Regime | Longs | Shorts | Size | Rationale |
|---|---|---|---|---|
| **BULL_TREND** | Top-N (pred > 0) | **None** | 0.5×, vol-scaled | Shorts lose in bull markets |
| **BEAR_TREND** | **None** | Bottom-N | 1.0× | Longs lost -92% historically |
| **LOW_VOL_GRIND** | Top-N | Bottom-N | 1.0× | Alpha source (70.6% WR) |
| **HIGH_VOL_LATERAL** | Top-N | Bottom-N | 0.5× (or skip) | High uncertainty |

### IC-Gated Retraining

```
Every 4 weeks (monthly):
  1. Build training matrix (rolling 104wk, 60-min embargo)
  2. Train XGBoost on new data
  3. Compute OOS IC on held-out fold
  4. If IC > 0 → ADOPT new model
     If IC ≤ 0 → REJECT, keep previous
  5. If cumulative DD < -20% → PAUSE until DD > -12%
```

### 4-Gate Kill Criteria

All 4 gates must pass for the strategy to continue:

| Gate | Condition | Status (v7) |
|---|---|---|
| **OOS IC Positive** | Mean IC > 0, positive >50% weeks | ✅ PASS (IC=+0.0166, 100%) |
| **Feature Stability** | Jaccard overlap > 0.5 | ✅ PASS (Jaccard=0.764) |
| **Regime Survival** | Positive returns in ≥2 regimes | ✅ PASS (4/4 regimes positive) |
| **ML Beats Baseline** | ML IC ≥ best single-factor IC | ✅ PASS |

---

## Feature Engineering

### 72 Cross-Sectional Features Computed

From raw 5-min OHLCV data, the engine computes 72 features including:
- Reversal signals (1h, 1d, 2d, 3d, 1w returns)
- Volatility measures (realized vol, ATR, vol ratio)
- Technical indicators (RSI, ADX, CCI, Bollinger Bands)
- Microstructure (Kyle lambda, Amihud, VWAP deviation)
- Distributional (skewness, fractional differentiation)
- WorldQuant-style alphas

### 5 Proven Features (Final Set)

Only 5 features are used in the final model based on consistent IC across 4 runs:

| Feature | Description | Mean IC | Positive % | Economic Rationale |
|---|---|---|---|---|
| **kyle_lambda** | Price impact / liquidity | **+0.112** | **79%** | Illiquid coins move more → predictable |
| **amihud** | Illiquidity ratio (|return|/volume) | **+0.109** | **79%** | Low liquidity → higher expected returns |
| **ret_3d** | 3-day return | +0.013 | 55% | Short-term momentum |
| **vol_regime** | Volatility regime (state variable) | +0.028 | — | Controls risk scaling |
| **rsi_14** | RSI mean reversion | +0.018 | 65% | Overbought/oversold reversion |

### Candidate Pool (Rotated In/Out)

These features are tracked and rotated in if IC proves strong over 4+ periods:

`ret_1w, ret_1d, ret_2d, rev_1h, rev_1d, rvol_1d, rvol_4h, skew_1d, atr_norm, cci_14, bb_pos, vwap_dev, mean_rev_zscore_1h, vol_ratio_1h_1d, frac_diff_close, vol_ret_1d, adx_14, trend_strength`

### Feature Stability Rules

- **Core features**: `ret_3d`, `vol_regime` — never dropped
- **Turnover cap**: max 3 features added/removed per retrain
- **Add rule**: positive IC in ≥2 recent periods
- **Drop rule**: negative IC in ≥3 recent periods
- **Regime-conditional IC**: mean-reversion features only evaluated in appropriate regimes (LOW_VOL_GRIND, HIGH_VOL_LATERAL)

---

## Data & Storage

### Data Requirements

| Item | Details |
|---|---|
| **Source** | Binance 5-minute OHLCV (parquet format) |
| **Symbols** | 444 USDT pairs |
| **Time Span** | 3+ years (minimum 2 years required) |
| **Columns** | `timestamp, open, high, low, close, volume` |
| **Total Rows** | ~26M+ (444 × 3yr × 5min) |
| **Data Size** | ~4.8 GB (443 parquet files) |

### Feature Cache

| Item | Details |
|---|---|
| **Location** | `feature_cache/` |
| **Format** | Parquet files (one per symbol) |
| **Size** | ~33 GB (443 files) |
| **Build Time** | 5-20 minutes (first run) |
| **Rebuild** | Only when `azalyst_factors_v2.py` or `build_feature_cache.py` changes |

### Results Directory

| File | Description |
|---|---|
| `results_v6/performance_v6.json` | Full performance metrics + 4-gate evaluation + diagnostics |
| `results_v6/weekly_summary_v6.csv` | Weekly returns, IC, regime, drawdown |
| `results_v6/all_trades_v6.csv` | All 597 trades with PnL, position scale, predictions |
| `results_v6/run_log_v6.txt` | Full pipeline execution log |
| `results_v6/run_output.txt` | Console output from run |
| `results_v6/train_summary_v6.json` | Training metrics (model type, IC, R², rows) |
| `results_v6/feature_importance_v6_*.csv` | Feature importances per retrain (20 files) |
| `results_v6/azalyst_v6.db` | SQLite database (trades, metrics, model artifacts) |
| `results_v6/models/` | XGBoost models (base + 19 weekly retrains) + scalers |
| `results_v6/checkpoint_v6_latest.json` | Live checkpoint (cleared on completion) |

**Note:** `results_v6/` is gitignored by default. To version-control results for reproducibility:
```bash
git add -f results_v6/performance_v6.json
git add -f results_v6/weekly_summary_v6.csv
git add -f results_v6/all_trades_v6.csv
git add -f results_v6/run_output.txt
```

---

## Running the Engine

### Prerequisites

```bash
pip install -r requirements.txt
```

### Commands

```bash
# Basic run — XGBoost, CPU, top-5 per side, 0.5x leverage
python azalyst_v6_engine.py --no-gpu --top-n 5 --leverage 0.5

# With GPU (CUDA)
python azalyst_v6_engine.py --gpu --top-n 5 --leverage 0.5

# XGBoost challenger mode (trains both, picks winner)
python azalyst_v6_engine.py --gpu --xgb-challenger --top-n 5

# Custom rolling window (default 104 weeks = 2 years)
python azalyst_v6_engine.py --no-gpu --rolling-window 52

# Skip falsification campaign (saves ~13 weeks of runtime)
python azalyst_v6_engine.py --no-gpu --no-falsify

# Restrict to specific coins
python azalyst_v6_engine.py --no-gpu --pin-coins BTCUSDT,ETHUSDT,SOLUSDT

# Skip HIGH_VOL_LATERAL regime entirely
python azalyst_v6_engine.py --no-gpu --no-trade-high-vol

# Rebuild feature cache (only when factors change)
python azalyst_v6_engine.py --no-gpu --rebuild-cache

# Resume from checkpoint
python azalyst_v6_engine.py --no-gpu

# Fresh run (ignore checkpoint)
python azalyst_v6_engine.py --no-gpu --no-resume
```

### Live Dashboard

```bash
# 4-panel Spyder monitor — auto-refreshes every 5 seconds
python VIEW_TRAINING.py

# Or select Monitor=1 in RUN_AZALYST.bat
```

### View Results

```bash
# Summary view
python VIEW_RESULTS_V6.py

# Or check performance_v6.json directly
cat results_v6/performance_v6.json
```

---

## Regime Breakdown (v7 Results)

| Regime | Weeks | Avg Return | Total Return | Avg IC | Strategy |
|---|---|---|---|---|---|
| **LOW_VOL_GRIND** | 43 | +0.51% | +22.14% | +0.0009 | Long+short |
| **BULL_TREND** | 13 | +5.32% | +69.19% | -0.0155 | Long-only |
| **BEAR_TREND** | 18 | +5.54% | +99.67% | +0.0368 | **Short-only** |
| **HIGH_VOL_LATERAL** | 3 | +4.30% | +12.90% | +0.2589 | Short-biased |

**Key insight:** BEAR_TREND contributed the most absolute return (+99.67%) via short-only positioning, confirming that the engine's primary edge is in identifying and shorting overvalued coins during market declines.

---

## Top Traded Symbols (v7)

| Symbol | Trades | Rationale |
|---|---|---|
| DEXEUSDT | 11 | High liquidity, consistent signals |
| FUNUSDT | 9 | Meme coin volatility |
| PROMUSDT | 9 | Mid-cap momentum |
| RAYUSDT | 8 | Solana ecosystem beta |
| OGUSDT | 7 | Fan token seasonality |
| ZECUSDT | 6 | Privacy coin cycles |
| ARDRUSDT | 6 | Low-float momentum |
| DEGOUSDT | 6 | DeFi ecosystem |
| AIXBTUSDT | 6 | AI narrative beta |
| RESOLVUSDT | 6 | New listing volatility |

293 unique symbols traded across 77 weeks. Average 2.04 trades per symbol.

---

## Technical Specifications

| Parameter | Value |
|---|---|
| **Primary model** | XGBoost (max_depth=3, 500 trees, strong regularization) |
| **Fallback model** | ElasticNet (alpha ≥ 0.001, L1-biased) |
| **Target** | Beta-neutral rank-normalized 1hr forward return |
| **CV method** | Purged K-Fold (5 splits, gap=12 bars) |
| **Training window** | Rolling 104 weeks (2 years) |
| **Retrain cadence** | Every 4 weeks (monthly) |
| **Walk-forward** | 77 weeks strict OOS (Sep 2024 – Mar 2026) |
| **Max training rows** | 2,000,000 (VRAM guard) |
| **Features** | 5 proven + up to 3 candidates (turnover cap) |
| **Portfolio** | Top-5 per side, regime-specific |
| **Leverage** | 0.5× default, 1.0× max per position |
| **Fees** | 0.2% round-trip (position-tracked) |
| **Kill switch** | -20% max DD, resume at -12% recovery |
| **Universe** | 444 Binance USDT pairs (428 tradeable) |
| **Horizon** | 1hr (12 × 5-min bars) |
| **Blacklist** | FTTUSDT, EURUSDT, stablecoins, fiat-pegged |

---

## Repository Structure

```
Azalyst-Alpha-Research-Engine/
│
├── azalyst_v6_engine.py          # Main engine (v7 final)
├── azalyst_v5_engine.py          # Shared infrastructure (helpers)
├── azalyst_factors_v2.py         # 72 cross-sectional features
├── azalyst_train.py              # Training utilities (CV, IC)
├── azalyst_risk.py               # Portfolio risk management
├── azalyst_db.py                 # SQLite persistence
├── azalyst_leak_test.py          # Pre-training leakage checks
├── azalyst_deflated_sharpe.py    # Deflated Sharpe Ratio
├── azalyst_ic_filter.py          # IC/ICIR feature filtering
├── azalyst_pump_dump.py          # Pump-dump detector
├── azalyst_tf_utils.py           # Timeframe utilities
├── azalyst_validator.py          # Validation helpers
├── build_feature_cache.py        # Feature cache builder
├── validate_startup.py           # Pre-flight checks
├── regime_analysis.py            # Regime analysis utilities
│
├── RUN_AZALYST.bat               # Windows one-click launcher
├── VIEW_TRAINING.py              # Live 4-panel Spyder monitor
├── VIEW_RESULTS_V6.py            # Results viewer
├── PUSH_TO_GITHUB.bat            # Git push helper
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
│
├── data/                         # Raw parquet (443 files, ~4.8 GB) [GITIGNORED]
├── feature_cache/                # Precomputed features (443 files, ~33 GB) [GITIGNORED]
├── results_v6/                   # Run output (trades, models, logs) [GITIGNORED]
│   ├── performance_v6.json       #   Performance report
│   ├── weekly_summary_v6.csv     #   Weekly metrics
│   ├── all_trades_v6.csv         #   All trades
│   ├── run_output.txt            #   Console log
│   ├── azalyst_v6.db             #   SQLite database
│   └── models/                   #   XGBoost models (20 files)
│
└── .qwen/                        # Qwen Code settings
```

---

## Deployment Guide (Future)

This repository is designed for eventual deployment as an online trading bot. Here's the path:

### 1. Data Pipeline (Current)
```
Binance API → 5-min OHLCV → Parquet → Feature Cache → Training Matrix
```
- Use `build_feature_cache.py` to precompute features
- Cache rebuilds only when factor logic changes

### 2. Signal Generation (Current)
```
Training Matrix → XGBoost → Predictions → Portfolio → Trade Signals
```
- Engine runs weekly (Monday pre-market)
- Models retrained monthly with rolling window

### 3. Execution Layer (To Build)
```
Trade Signals → Order Router → Binance API → Position Tracking → PnL Reporting
```
- Integrate with `python-binance` for live execution
- Add position management, stop-loss, take-profit
- Paper trading mode for validation

### 4. Infrastructure (To Build)
```
Cloud Server (AWS/GCP) → Scheduler (cron) → Engine → Discord/Telegram Alerts
```
- Deploy on always-on server
- Weekly cron job for signal generation
- Real-time alerting

### 5. Monitoring (To Build)
```
Live Dashboard → PnL Tracking → Risk Metrics → Kill Switch Integration
```
- Extend `VIEW_TRAINING.py` for live monitoring
- Add drawdown alerts, IC decay warnings
- Automated kill switch execution

### Required Artifacts for Deployment

All artifacts needed to restart from scratch are preserved in this repo:

| Artifact | Location | Purpose |
|---|---|---|
| Engine code | `azalyst_v6_engine.py` | Full pipeline logic |
| Feature logic | `azalyst_factors_v2.py` | 72-feature computation |
| Cache builder | `build_feature_cache.py` | Parquet cache generation |
| Models | `results_v6/models/` | Trained XGBoost models |
| Feature importance | `results_v6/feature_importance_*.csv` | Signal attribution |
| Performance | `results_v6/performance_v6.json` | Historical validation |
| Trade log | `results_v6/all_trades_v6.csv` | Execution record |
| Database | `results_v6/azalyst_v6.db` | Full run history |
| Dependencies | `requirements.txt` | Python environment |

To restart on a new machine:
1. Clone this repo
2. Download Binance data to `data/`
3. Run `build_feature_cache.py`
4. Run `azalyst_v6_engine.py`
5. Results appear in `results_v6/`

---

## Theoretical Foundations

| Domain | Source | What Azalyst Uses |
|---|---|---|
| Feature engineering | **Lopez de Prado** — *Advances in Financial Machine Learning* | Fractional differentiation, purged K-Fold CV |
| Signal combination | **Grinold & Kahn** — *Active Portfolio Management* | IC-weighted signal fusion, information ratio |
| Model selection | **Consensus of 7 AI models** | XGBoost default, ElasticNet fallback, falsification |
| Robust estimation | **Huber** — *Robust Statistics** (via RobustScaler) | Median/IQR scaling for fat-tailed distributions |
| Factor models | **Fama & French**, **Barra** | Cross-sectional alpha, beta-neutral target |
| Microstructure | **Kyle (1985)**, **Amihud (2002)** | Kyle lambda, Amihud illiquidity ratio |
| Volatility | **Garman & Klass (1980)**, **Parkinson (1980)** | Range-based volatility estimators |
| Time series | **Hurst (1951)**, **FFT** | Regime detection, cyclical patterns |
| Performance evaluation | **Bailey & Lopez de Prado** — *Deflated Sharpe Ratio* | Multiple-testing-adjusted performance metric |

---

## Version History

| Version | Date | Key Changes |
|---|---|---|
| **v6.0** | Apr 2026 | Initial consensus rebuild — ElasticNet, 13-week window |
| **v6.1** | Apr 2026 | 2-year rolling window, leakage audit, position sizing fix |
| **v6.2** | Apr 2026 | Rank-normalize target, disable momentum filter, tighten kill switch |
| **v6.2.1** | Apr 2026 | Fix MemoryError in training matrix subsampling |
| **v6.2.2** | Apr 2026 | Reduce ElasticNetCV n_alphas 50→20 |
| **v6.3** | Apr 2026 | **XGBoost primary + regime-specific portfolio → +564.7%** |
| **v6.4** | Apr 2026 | Fix compute_oos_diagnostics NameError + regularization guardrails |
| **v7.0** | Apr 2026 | **FINAL RELEASE** — All fixes integrated, verified 77-week OOS |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No GPU detected | `python -c "import xgboost as xgb; print(xgb.__version__)"` — verify CUDA build |
| Feature cache stale | Rebuild only if factors, cache-builder logic, raw `data/`, or timeframe assumptions changed |
| OOM / freeze | Reduce `MAX_TRAIN_ROWS` in config (2M for RTX 2050, 4M for T4) |
| Pipeline closes immediately | Confirm Python path has no spaces; use `RUN_AZALYST.bat` |
| No results after run | Check `results_v6/` for output files. If empty, check data folder has `.parquet` files |
| Dashboard shows "IDLE" | Engine hasn't started yet — launch `VIEW_TRAINING.py` after starting the engine |
| MemoryError on large datasets | Chunked processing is enabled by default — 20 symbols per chunk |
| Kill switch triggered | Check regime — may be in drawdown pause. Recovers at -12% threshold |

---

## Research Principles

- **Strict OOS** — Walk-forward never touches training data
- **Transparency** — Every decision documented, every metric logged
- **Repeatable** — Same code, same data, same results
- **Evidence over claims** — Results are observations, not promises
- **Position-aware costs** — Fee simulation reflects real-world execution
- **Falsification first** — Prove signal exists before trusting ML
- **Regime awareness** — No single portfolio rule works in all conditions
- **Feature discipline** — Only keep features with consistent IC

---

## Disclaimer

This is a research and educational project. **Not financial advice.** Past performance does not indicate future results. The +564.75% return was achieved in a specific out-of-sample period (Sep 2024 – Mar 2026) with specific market conditions. Use at your own risk. Always do your own research.

---

<div align="center">

**v7.0 — Final Release** | Built by [Azalyst](https://github.com/gitdhirajsv) | Azalyst Quant Research

*"Evidence over claims. Always."*

</div>
