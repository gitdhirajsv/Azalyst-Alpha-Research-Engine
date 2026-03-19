# Azalyst Alpha Research Engine

> **Autonomous quantitative research infrastructure for crypto alpha generation** — built by Azalyst Research. Cross-sectional signal discovery, GPU-accelerated machine learning, and institutional walk-forward validation across 300+ digital assets.

<p align="center">
  <em>Where systematic research meets execution discipline.</em>
</p>

---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)
![Features](https://img.shields.io/badge/Features-56%20Cross--Sectional-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20CUDA-blueviolet?style=flat-square)
![CV](https://img.shields.io/badge/CV-Purged%20K--Fold-orange?style=flat-square)
![Meta](https://img.shields.io/badge/Meta--Labeling-AFML%20Ch.3-yellow?style=flat-square)
![Signals](https://img.shields.io/badge/Signals-IC--Weighted%20Fusion-teal?style=flat-square)

---

## Why Azalyst Exists

Most open-source crypto "bots" are toy systems: fit a moving average on BTC, overfit on in-sample, declare victory. Azalyst is the antithesis.

This engine was built to answer a single question with rigour: **can a systematic cross-sectional model generate persistent alpha in crypto, and can you prove it out-of-sample?**

The architecture mirrors how the top systematic funds (Citadel, Two Sigma, Renaissance) structure research — not because we're a fund, but because these methods are the only honest way to know if your signal is real.

---

## v2 Changelog

| Area | v1 | v2 |
|---|---|---|
| Features | 27 generic TA | **56 features** — WorldQuant alphas, Garman-Klass, ADX, Kyle lambda, Hurst, FFT, **Fractional Differentiation** |
| Training data | Year 1 only | **Year 1 + Year 2 combined** |
| Test set | Year 2+3 rolling | **Year 3 only** — strict out-of-sample |
| Retrain | Every week (OOM) | **Quarterly** — every 13 weeks, stable |
| Cross-validation | TimeSeriesSplit (leakage) | **Purged K-Fold** — 48-bar embargo |
| Scaler | StandardScaler | **RobustScaler** — handles fat tails |
| GPU backend | LightGBM CUDA (broken) | **XGBoost CUDA** — confirmed T4 + RTX 2050 |
| Metrics | AUC only | **AUC + IC + ICIR** |
| Position sizing | Equal weight | **Meta-Labeling** — AFML Ch. 3, confidence-weighted |
| Signal fusion | Static weights | **IC-Weighted** — rolling IC reweights per-source |
| Fee simulation | Per-bar (broken) | **Position-tracked** — only new entries pay fees |
| Output | CSV | CSV + charts + **JSON summary** |

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │         AZALYST RESEARCH ENGINE          │
                         │         Built by Azalyst Research        │
                         └──────────────┬───────────────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  ┌─────────────────┐       ┌────────────────────┐       ┌───────────────────┐
  │  DATA LAYER     │       │  FEATURE ENGINE    │       │  SIGNAL SOURCES   │
  │                 │       │                    │       │                   │
  │ Polars+DuckDB   │──────▶│ 56 cross-sectional │──────▶│ Factor scores     │
  │ 300+ coins      │       │ features, TF-aware │       │ ML return prob    │
  │ 26M+ rows       │       │ Frac. diff (AFML)  │       │ Pump/dump filter  │
  │ 3-year 5min     │       │ Hurst + FFT        │       │ StatArb z-scores  │
  └─────────────────┘       └────────────────────┘       └────────┬──────────┘
                                                                  │
                            ┌────────────────────┐                │
                            │  SIGNAL COMBINER   │◀───────────────┘
                            │                    │
                            │ Regime-adaptive    │
                            │ IC-weighted fusion │
                            │ 4-state detector   │
                            └────────┬───────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
  ┌─────────────────┐     ┌────────────────────┐    ┌───────────────────┐
  │  PRIMARY MODEL  │     │  META-LABELING     │    │  WALK-FORWARD     │
  │                 │     │  (AFML Ch. 3)      │    │                   │
  │ XGBoost CUDA    │────▶│ 2nd-stage XGBoost  │───▶│ Y1+Y2 train      │
  │ Purged K-Fold   │     │ P(primary correct) │    │ Y3 strict OOS    │
  │ 48-bar embargo  │     │ Position sizing    │    │ Quarterly retrain │
  │ RobustScaler    │     │ Confidence weights │    │ Weekly IC + ICIR  │
  └─────────────────┘     └────────────────────┘    └───────────────────┘
```

---

## The Three Pillars of v2

### 1. Fractional Differentiation (Lopez de Prado, AFML Ch. 5)

Standard returns (`log(P_t / P_{t-1})`) are stationary but **destroy all memory of price levels**. Raw prices preserve memory but are non-stationary — they break tree-based models.

Fractional differentiation with `d=0.4` using the Fixed-Width Window (FFD) method gives the model access to **where the price actually is** while maintaining stationarity. It's the sweet spot between returns and prices that most quant systems miss.

```
d=0.0   →  raw price (non-stationary, max memory)
d=0.4   →  Azalyst default (stationary, retains memory)  ← HERE
d=1.0   →  standard returns (stationary, zero memory)
```

### 2. Meta-Labeling (Lopez de Prado, AFML Ch. 3)

The primary model says "BUY this coin." But how confident should we be?

A second-stage XGBoost model is trained on a meta-question: **"When the primary model predicted UP, was it actually correct?"** The meta-model outputs a confidence probability that directly scales position size:

- **High confidence (0.85)** → full position, maximum capital allocation
- **Low confidence (0.45)** → reduced position, capital preservation
- **Wrong signals get less money. Right signals get more.** That's meta-labeling.

The meta-model trains on honest out-of-sample predictions from purged cross-validation — no information leakage. It retrains alongside the primary model every 13 weeks.

### 3. IC-Weighted Signal Fusion (Grinold & Kahn)

The signal combiner fuses 4 alpha sources (factor scores, ML return probability, pump filter, stat-arb z-scores) using regime-adaptive weights. But static weights assume every signal performs equally forever.

IC-weighted fusion tracks the rolling Information Coefficient of each signal source over the last 13 weeks and **dynamically reweights** — signals that are currently working get more weight, decaying signals get less.

```
Base regime weights  ×  IC multiplier  →  Normalized adaptive weights
                         ↑
              max(0.1, min(3.0, 1 + 10·mean_IC))
```

This is the Grinold & Kahn principle of **scaling signal weight by its demonstrated predictive power**.

---

## Feature Engineering — 56 Features, 9 Categories

### 1. Returns (7)
`ret_1bar` · `ret_1h` · `ret_4h` · `ret_1d` · `ret_2d` · `ret_3d` · `ret_1w`

### 2. Volume (6)
`vol_ratio` · `vol_ret_1h` · `vol_ret_1d` · `obv_change` · `vpt_change` · `vol_momentum`

### 3. Volatility (7)
`rvol_1h` · `rvol_4h` · `rvol_1d` · `vol_ratio_1h_1d` · `atr_norm` · `parkinson_vol` · `garman_klass`

Parkinson and Garman-Klass use High/Low range — less noisy than close-to-close volatility.

### 4. Technical (10)
`rsi_14` · `rsi_6` · `macd_hist` · `bb_pos` · `bb_width` · `stoch_k` · `stoch_d` · `cci_14` · `adx_14` · `dmi_diff`

ADX measures trend strength. DMI diff quantifies directional bias.

### 5. Microstructure (6)
`vwap_dev` · `amihud` · `kyle_lambda` · `spread_proxy` · `body_ratio` · `candle_dir`

Kyle lambda estimates price impact per unit volume — a genuine microstructure signal rarely seen in open-source crypto research.

### 6. Price Structure (6)
`wick_top` · `wick_bot` · `price_accel` · `skew_1d` · `kurt_1d` · `max_ret_4h`

### 7. WorldQuant-Inspired Alphas (8)
`wq_alpha001` · `wq_alpha012` · `wq_alpha031` · `wq_alpha098` · `cs_momentum` · `cs_reversal` · `vol_adjusted_mom` · `trend_consistency`

Cross-sectional signals inspired by the WorldQuant 101 Alphas paper.

### 8. Regime Features (5)
`vol_regime` · `trend_strength` · `corr_btc_proxy` · `hurst_exp` · `fft_strength`

Hurst exponent identifies trending vs mean-reverting states. FFT captures dominant price cycles.

### 9. Memory-Preserving (1) — NEW in v2
`frac_diff_close`

Fractional differentiation of log-price (d=0.4). Retains price level memory while achieving stationarity. Based on AFML Ch. 5.

---

## ML Pipeline

### Training Label — Cross-Sectional Alpha

The model predicts whether a coin will **outperform the cross-sectional median** return at the next 4H horizon. This is direction-agnostic — works in bull and bear markets equally. It's the standard label construction at institutional quant funds.

$$
\text{alpha\_label}_i = \mathbb{1}\left[ r_{i,t+48} > \text{median}(r_{j,t+48}) \;\forall\; j \in \text{universe} \right]
$$

IC (Information Coefficient) = Spearman rank correlation between predicted probabilities and actual returns. ICIR = IC / std(IC). Both tracked weekly throughout Year 3.

### Purged K-Fold Cross-Validation

48-bar embargo gap between train and validation prevents information leakage from autocorrelated features:

```
|──── TRAIN ────| 48-bar gap |── VAL ──|
                  (4 hours)
```

5 purged folds. RobustScaler for fat-tailed crypto distributions.

### Meta-Labeling (Second-Stage Model)

```
Primary Model predictions (OOS from purged CV)
    ↓
Meta-label: did primary model get this row correct? (binary)
    ↓
Second XGBoost: features + primary_prob → P(correct)
    ↓
Output: confidence score per symbol per week → position sizing
```

The meta-model is shallower (depth=4, 500 trees, min_child_weight=50) to avoid overfitting to noise in the correctness signal.

### Walk-Forward Architecture

```
Year 1 + Year 2 (730 days)
    ↓
[BASE MODEL] + [META MODEL]
XGBoost CUDA · Purged K-Fold (5 splits, gap=48)
RobustScaler · IC + ICIR + AUC
    ↓
Year 3 only (never seen during training)
    ↓
┌──────────────────────────────────────────────────────────────┐
│  Each week:                                                  │
│    1. Predict   — rank all symbols by outperformance prob    │
│    2. Meta-size — scale positions by meta-model confidence   │
│    3. Trade     — long top 15%, short bottom 15%             │
│    4. Fees      — position-tracked (only new entries pay)    │
│    5. Evaluate  — weekly IC + confidence-weighted return     │
│    6. Retrain   — every 13 weeks (primary + meta together)   │
│    7. Save      — weekly summary + all trades                │
└──────────────────────────────────────────────────────────────┘
    ↓
performance_year3.json + performance_year3.png
```

### Signal Fusion — 4 Sources, IC-Weighted

```
REGIME DETECTOR (4-state)
    │
    ├── BULL_TREND       → Factor: 0.45  ML: 0.35  Pump: 0.10  StatArb: 0.10
    ├── BEAR_TREND       → Factor: 0.25  ML: 0.20  Pump: 0.20  StatArb: 0.35
    ├── HIGH_VOL_LATERAL → Factor: 0.15  ML: 0.15  Pump: 0.35  StatArb: 0.35
    └── LOW_VOL_GRIND    → Factor: 0.30  ML: 0.30  Pump: 0.15  StatArb: 0.25
                              ↓
                    × IC multiplier per source (rolling 13-week IC)
                              ↓
                    Renormalized adaptive weights → composite score
```

---

## Execution Simulation

### Position-Tracked Fee Model

The simulation charges transaction fees **only when a symbol enters the portfolio**. Held positions (same side as prior week) pay zero fees. This accurately models real-world turnover costs:

```
Fee per new entry:  0.1% per leg × 2 = 0.2% round-trip
Held positions:     0% (no fee)
Turnover tracked:   % of portfolio that's new each week
```

### Meta-Labeling Position Sizing

Each trade's P&L is scaled by the meta-model's confidence output:

```
pnl_i = (raw_return_i − fee_i) × meta_confidence_i × 100
weekly_return = weighted_average(pnl, weights=meta_confidence)
```

High-conviction trades dominate the portfolio return. Low-conviction trades are automatically down-weighted.

---

## Running the Engine

### Option 1 — Windows One-Click (recommended for local)

Just double-click **`RUN_AZALYST.bat`** — it guides you through 3 quick prompts then runs the full pipeline automatically:

1. **Select compute device** — `[1] GPU` (RTX 2050, ~4x faster) or `[2] CPU`
2. **Select output mode** — `[1] Terminal only` or `[2] Terminal + Spyder` (live charts)
3. **Confirm start** — `Y` to launch

After confirmation it runs fully unattended. The batch file also:

- Detects Python installation and GPU availability
- **Auto-installs all missing packages** on first run (no manual `pip install` needed)
- Validates data files in `./data/`
- Sets UTF-8 encoding to prevent Windows console crashes
- Launches `azalyst_local_gpu.py --gpu` for GPU mode or `azalyst_engine.py` for CPU mode

### Option 2 — VSCode Jupyter (Local GPU — RTX 2050 / any NVIDIA)

Open `azalyst-alpha-research-engine.ipynb` directly in **VSCode Jupyter** for an interactive,
cell-by-cell experience on your local machine:

1. Install the **Jupyter** extension in VSCode
2. Place your `.parquet` data files in `./data/`
3. Open `azalyst-alpha-research-engine.ipynb` in VSCode
4. Select your Python environment (must have `requirements.txt` packages installed)
5. Run **Cell 0** first to install all dependencies (`pip install -r requirements.txt`)
6. Run remaining cells in order — the notebook auto-detects CUDA and caps VRAM at 2 M rows

> The notebook is pre-configured for **NVIDIA RTX 2050 (4 GB GDDR6) + Intel i5-11260H**.
> It supports both the new `device='cuda'` XGBoost API and the legacy `tree_method='gpu_hist'`
> API, with automatic CPU fallback.  See `SETUP_LOCAL_GPU.md` for full setup details.

### Option 3 — CPU only

Same as Option 2 — the notebook auto-detects hardware and falls back to CPU if no NVIDIA GPU is available.

### Option 4 — Core research pipeline

```bash
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```

---

## Bug Fixes

### v2.3 — RUN_AZALYST.bat: Spyder Not Detected + Pipeline Closes Immediately (19 Mar 2026)

**Problem 1 — Spyder not found even when installed.**
The PATH-boost loop added `Python311\Scripts` to `PATH` only when `python.exe` existed inside `Scripts\` — it doesn't (it lives in the base dir). So `where spyder` failed and the `[OK] Spyder found` branch was never reached.

**Fix:** PATH loop now checks for `python.exe` in the **base dir** (`%%~d\python.exe`) and adds **both** `%%~d` *and* `%%~d\Scripts` simultaneously. `spyder.exe`, `pip.exe`, etc. are immediately visible.

**Problem 2 — Spyder auto-install.**
When Spyder was missing the BAT printed "To install: pip install spyder" but never installed it, leaving the user to do it manually.

**Fix:** Added a step 3d — if Spyder is not found via PATH, module import, or any of 15 explicit paths (including `.venv\Scripts\spyder.exe`), the BAT now **automatically runs `pip install spyder`** into global Python. If that succeeds, Spyder is launched normally. If it fails, execution continues in terminal-only mode (no crash).

**Problem 3 — Pipeline using wrong Python / packages.**
There was no distinction between the Python used for `pip install` vs the Python that runs the pipeline. On machines with a `.venv`, the venv Python (which has xgboost, pyarrow, etc.) should run the pipeline, while global Python handles installs.

**Fix:** Introduced two variables:
- `PYTHON_CMD` — global Python (for `pip install`)
- `RUN_PYTHON` — `.venv` Python when a `.venv` exists, otherwise falls back to `PYTHON_CMD`

All pipeline scripts now launch via `!RUN_PYTHON!`. Package install retries in the run-env if global pip fails.

**Problem 4 — GPU mode ignoring explicit data path.**
`azalyst_local_gpu.py` had no `--data-dir` / `--out-dir` arguments. GPU mode in the BAT launched `python azalyst_local_gpu.py --gpu` with no path, relying on `./data` relative to CWD — which fails if the terminal isn't in the project root.

**Fix:** Added `--data-dir` and `--out-dir` to `azalyst_local_gpu.py` argparse. The BAT now passes `--data-dir "%~dp0data" --out-dir "%~dp0results"` (absolute paths) for both GPU and CPU mode, matching the CPU launch that already worked correctly.

**Status:** ✅ Verified — PATH boost finds `spyder.exe` at `Python311\Scripts\spyder.exe`, Spyder auto-installs to global pip on first run, `.venv` Python runs pipeline, GPU mode receives correct data path.

---

### v2.3b — RUN_AZALYST.bat: Quoted RUN_PYTHON Path + pip --upgrade RECORD Error (19 Mar 2026)

**Problem 1 — `can't open file 'D:\\Azalyst Alpha Research Engine\\Alpha'`.**
`!RUN_PYTHON!` was used unquoted as an executable. Because the path contains spaces (`D:\Azalyst Alpha Research Engine\...`), `cmd.exe` split it at every space — so Python received `Alpha` as the script name instead of the actual file.

**Fix:** All `!RUN_PYTHON!` usages as an executable are now wrapped in double-quotes: `"!RUN_PYTHON!"`.

**Problem 2 — `error: uninstall-no-record-file / Cannot uninstall psutil None`.**
The package install step used `pip install ... --upgrade`, which forces pip to uninstall the existing version first. `psutil` was installed without a `RECORD` metadata file (system-managed or pre-existing install), so pip crashed trying to remove it even though it was perfectly functional.

**Fix:** Removed `--upgrade` from both pip install calls. Packages are only installed if missing; existing working installs are left untouched.

**Problem 3 — Spyder launch mode comparison broken for long paths.**
`if "!SPYDER_CMD!"=="!RUN_PYTHON! -m spyder"` — comparing a variable that contains a long path with spaces against a string embedding another variable is unreliable in cmd's delayed-expansion mode.

**Fix:** Replaced with a clean `SPYDER_MODE` flag (`PATH` / `MODULE` / `EXE`) set at detection time. Launch block checks this flag instead.

**Status:** ✅ Tested in real `cmd` context — `[OK] All packages present` with quoted `.venv` path confirmed.

---

### v2.3c — RUN_AZALYST.bat: Launch Summary Always Showed "Terminal only" (19 Mar 2026)

**Problem:** User selects `[2] Terminal + Spyder` but Launch Summary displayed `Output : Terminal only`.

**Root cause:** Windows batch `set USE_SPYDER=1 & echo...` silently captures the space before `&` as part of the value — `USE_SPYDER` becomes `"1 "` (trailing space), not `"1"`. The check `if "!USE_SPYDER!"=="1"` in the Launch Summary never matched, so it always fell through to `else → Terminal only`. The same bug affected `COMPUTE_CHOICE` and `COMPUTE_LABEL` on the GPU/CPU selection lines.

**Fix:** All inline `set VAR=value` commands changed to the quoted form `set "VAR=value"` which correctly trims surrounding whitespace. Tested: `USE_SPYDER=[1]` and `SUMMARY: Terminal + Spyder` confirmed in cmd.

**Status:** ✅ Verified via isolated test BAT — quoted `set` form resolves trailing-space correctly.

---

### v2.2 — Notebook: Stale Feature Cache Detection (Cell 6)

**Problem:** After `frac_diff_close` was added in v2.1, the feature cache contained 443 files with 56 columns. Cell 6 only checked whether enough *files* existed (≥ 90% threshold) — it never validated *column* presence. So the rebuild was silently skipped and the pipeline continued with a cache missing `frac_diff_close`.

**Fix:** Cell 6 now reads a sample cached file and checks every column listed in `FEATURE_COLS`. If any column is missing it prints `[REBUILD] Cache stale -- missing cols: [...]`, deletes all stale files, and triggers a full rebuild before continuing.

### v2.2 — Notebook: OOM / Freeze When Loading Feature Store (Cell 7)

**Problem:** Cell 7 loaded all 443 symbol files simultaneously — 443 × ~47,779 rows × 57 columns ≈ 21 million rows. This exceeded available RAM on most machines (8–16 GB), causing the notebook to freeze silently after printing `Found 443 cached symbol files`.

**Fix:** Cell 7 now probes the first file to estimate total rows, computes `LOAD_STRIDE = max(1, total_est // 4_000_000)`, and applies `df.iloc[::LOAD_STRIDE]` per symbol during loading. This caps the pooled dataset at ~4 million rows (~1 GB). If the guard activates you will see `[MEM GUARD] ~21M est. rows -> LOAD_STRIDE=5 (load ~4.2M rows)`. Training quality is preserved because the 4M-row cap still feeds into the 2M-row `MAX_TRAIN_ROWS` VRAM guard.

### v2.1 — Position-Tracked Fee Simulation

**Problem:** Old simulation charged 0.2% round-trip fee on every position at every 5-min bar. With ~340 symbols and 288 bars/day, this created ~50,000 phantom "trades" per week — massive fee drag that turned a positive-IC model into -5% annual return.

**Fix:** Rewrote `simulate_weekly_trades()` to track `prev_longs` / `prev_shorts` sets. Fees are charged only on new portfolio entries. Turnover percentage is tracked per week.

### v1.1 — Timeframe-Aware Feature Engineering

**Problem:** Rolling windows hardcoded to 5-min math (`BARS_PER_DAY = 288`). Scoring daily/weekly candles caused NaN flooding.

**Fix:** `azalyst_tf_utils.py` — `get_tf_constants(resample_str)` derives all window sizes dynamically.

---

## Repository Map

### Core Pipeline
| File | Purpose |
|---|---|
| `azalyst_factors_v2.py` | **56 cross-sectional features** — returns, volume, microstructure, WorldQuant alphas, Hurst, FFT, fractional differentiation |
| `azalyst_train.py` | **Primary + Meta model training** — XGBoost CUDA, Purged K-Fold, IC+ICIR, meta-labeling (AFML Ch. 3) |
| `azalyst_weekly_loop.py` | **Walk-forward Year 3** — quarterly retrain, meta-labeling sizing, position-tracked fees, IC weekly |
| `azalyst_signal_combiner.py` | **IC-weighted regime-adaptive signal fusion** — 4 sources, dynamic reweighting by rolling IC |
| `azalyst_alpha_metrics.py` | IC, ICIR, Sharpe, drawdown, retrain trigger |
| `build_feature_cache.py` | Precompute features → parquet cache (5-20x speedup) |

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
| `azalyst-alpha-research-engine.ipynb` | VSCode Jupyter notebook — local RTX 2050 GPU |

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

IC > 0.05 with ICIR > 1.0 is institutional-quality signal strength. Read Grinold & Kahn's *Active Portfolio Management* for why these numbers matter.

---

## Theoretical Foundations

This engine draws from the quantitative finance canon — the same reading list found on Citadel, Two Sigma, and Renaissance Technologies job descriptions:

| Domain | Source | What Azalyst Uses |
|---|---|---|
| Feature engineering | **Lopez de Prado** — *Advances in Financial Machine Learning* | Fractional differentiation, meta-labeling, purged K-Fold CV |
| Signal combination | **Grinold & Kahn** — *Active Portfolio Management* | IC-weighted signal fusion, information ratio targeting |
| Statistical learning | **Hastie, Tibshirani, Friedman** — *Elements of Statistical Learning* | Regularization (alpha/lambda), cross-validation methodology |
| Robust estimation | **Huber** — *Robust Statistics* (via RobustScaler) | Median/IQR scaling for fat-tailed crypto distributions |
| Factor models | **Fama & French**, **Barra** | Cross-sectional alpha label, factor decomposition |
| Microstructure | **Kyle (1985)**, **Amihud (2002)** | Kyle lambda, Amihud illiquidity ratio |
| Volatility | **Garman & Klass (1980)**, **Parkinson (1980)** | Range-based volatility estimators |
| Time series | **Hurst (1951)**, **FFT** | Regime detection, cyclical pattern identification |

---

## Technical Specifications

| Parameter | Value |
|---|---|
| XGBoost trees | 1,000 (primary) · 500 (meta) |
| Learning rate | 0.02 |
| Max depth | 6 (primary) · 4 (meta) |
| Min child weight | 30 (primary) · 50 (meta) |
| Subsample | 0.8 |
| Column sample | 0.7 (tree) · 0.7 (level) |
| Regularisation | alpha=0.1, lambda=1.0 |
| CV splits | 5, purged (48-bar gap) |
| VRAM guard | 2M rows (RTX 2050) · 4M rows (T4) |
| Train/test | Year 1+2 / Year 3 (strict OOS) |
| Retrain | Every 13 weeks (quarterly) |
| Universe | 300+ coins, cross-sectional pooling |
| Horizon | 4H (48 × 5-min bars) |
| Portfolio | Long top 15%, short bottom 15% |
| Fees | 0.2% round-trip, position-tracked |
| Frac. diff. d | 0.4 (FFD method, threshold 1e-5) |
| IC lookback | 13 weeks rolling (signal reweighting) |

---

## Data Requirements

Place Binance 5-minute parquet files in `data/`:

```
timestamp | open | high | low | close | volume
```

300+ symbols × 3 years × 5-min bars = ~26M rows.

## Installation

**Easiest:** Just double-click `RUN_AZALYST.bat` — it auto-installs all missing packages on first run.

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

## Research Principles

- **Transparency over mystique** — every decision documented, every metric shown
- **Strict train/test split** — Year 3 never touched during training, ever 
- **Repeatable pipelines** — same code, same data, same results
- **Evidence over claims** — results are observations, not promises
- **No LLM in the training loop** — pure quantitative self-improvement
- **Position-aware costs** — fee simulation reflects real-world execution

---

## About Azalyst Research

Azalyst is a personal quantitative research project — not a hedge fund, not a financial product. It exists because systematic research is a craft worth pursuing with the same rigour the best firms apply, even as an independent researcher.

The goal is simple: build something you'd be proud to show at a quant research desk, and make it open-source so others can learn from it.

## Disclaimer

This is a research and educational project. Not financial advice. Past performance does not indicate future results. Use at your own risk. Always do your own research.

---

<div align="center">
Built by <a href="https://github.com/gitdhirajsv">Azalyst</a>
</div>

