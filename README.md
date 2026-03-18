# Azalyst Alpha Research Engine

Quantitative crypto research engine — 20 cross-sectional factors, IC analysis, vectorized backtest on 5-min Binance OHLCV data.

## Quick Start (Windows)

### Prerequisites
- Python 3.10+ ([python.org](https://python.org)) — check "Add Python to PATH"
- Your `.parquet` data files in a `data/` folder (5-min OHLCV, Binance format)
- *(Optional)* NVIDIA GPU with CUDA drivers for ~4x faster runs
- *(Optional)* [Spyder IDE](https://www.spyder-ide.org/) for live charts

### Run
1. Download or clone this repo
2. Put your `.parquet` data files into `data/` folder inside the repo
3. Double-click **`RUN_AZALYST.bat`**
4. Answer 2 questions:
   - **Compute:** GPU / CPU / Auto
   - **Output:** Terminal only / Terminal + Spyder
5. Results save to `results/`

### What the bat file does
- Checks Python, detects GPU, detects Spyder
- Installs missing packages automatically (one-time, ~2 min)
- Asks 2 questions then runs `azalyst_engine.py`
- If you pick Terminal + Spyder: Spyder opens in background, closing it does NOT stop the pipeline

---

## Output Files

| File | Description |
|---|---|
| `results/ic_analysis.csv` | Factor IC / ICIR / t-stat across all horizons |
| `results/backtest_pnl.csv` | Daily PnL with gross/net/fees/drawdown |
| `results/performance_summary.csv` | Sharpe, Sortino, Calmar, win rate, etc. |
| `results/hurst_exponents.csv` | Mean-reversion scores per symbol (if --hurst) |

---

## Direct CLI Usage

```bash
# Full pipeline, daily rebalancing, reversal composite
python azalyst_engine.py --data-dir ./data --out-dir ./results

# Momentum composite, 4H rebalancing, first 50 symbols
python azalyst_engine.py --data-dir ./data --composite momentum --rebal 4H --max-symbols 50

# Long-only, IC-weighted
python azalyst_engine.py --data-dir ./data --composite ic_weighted --long-only

# Skip IC analysis (faster backtest only)
python azalyst_engine.py --data-dir ./data --skip-ic

# Compute Hurst exponents for mean-reversion candidates
python azalyst_engine.py --data-dir ./data --hurst
```

---

## Factor Universe (20 factors)

| Group | Factors |
|---|---|
| Momentum | MOM_1H, MOM_4H, MOM_1D, MOM_3D, MOM_1W, MOM_2W |
| Reversal | REV_1H, REV_4H |
| Volume | VOL_RATIO, VOL_MOM_1D |
| Volatility | RVOL_1D, RVOL_1W, VOL_OF_VOL |
| Microstructure | AMIHUD, MAX_RET, SKEW_1W, PRICE_ACCEL |
| Structural | TREND_48, BB_POS, RSI_RANK |

---

## Data Format

Parquet files in `data/` folder, one file per symbol (e.g. `BTCUSDT.parquet`).

Required columns: `timestamp` (ms int or datetime), `open`, `high`, `low`, `close`, `volume`

Compatible with Binance OHLCV downloaders.

---

## Requirements

Auto-installed by `RUN_AZALYST.bat`, or manually:
```bash
pip install xgboost>=2.0.3 numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels
```

---

## Architecture

```
azalyst_engine.py          ← Main research pipeline (run this)
azalyst_execution.py       ← Market impact model
azalyst_local_gpu.py       ← XGBoost ML training pipeline (GPU/CPU)
azalyst_spyder_monitor.py  ← Open in Spyder for live charts
RUN_AZALYST.bat            ← Windows launcher (double-click to run)
```

---

*Research basis: Liu & Tsyvinski (2021), Fieberg et al. (2024), Baybutt (2024), Borri et al. (2024), Cambridge CTREND (2024)*
