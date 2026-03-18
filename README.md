# Azalyst Alpha Research Engine

Quantitative crypto research engine — 20 cross-sectional factors, IC analysis, vectorized long/short backtest on 5-min Binance OHLCV data.

## Quick Start (Windows)

### Prerequisites
- Python 3.10+ from [python.org](https://python.org) — check "Add Python to PATH"
- Your `.parquet` data files in a `data/` folder next to `RUN_AZALYST.bat`
  - Format: 5-min OHLCV Binance data, columns: `timestamp, open, high, low, close, volume`
- *(Optional)* NVIDIA GPU + CUDA drivers for ~4× faster runs
- *(Optional)* [Spyder IDE](https://www.spyder-ide.org/) for live charts

### Run
1. Place `.parquet` files in `data\`
2. Double-click `RUN_AZALYST.bat`
3. Answer 2 questions (compute device, output mode)
4. Results saved to `results\`

### Output Files
| File | Contents |
|------|----------|
| `ic_analysis.csv` | Factor IC / ICIR scores per horizon |
| `backtest_pnl.csv` | Daily PnL with fee breakdown |
| `performance_summary.csv` | Sharpe, Sortino, Calmar, Max DD |
| `factor_weights.csv` | Optimised factor combination weights |

## CLI Usage
```bash
python azalyst_engine.py --data-dir ./data --out-dir ./results
python azalyst_engine.py --data-dir ./data --out-dir ./results --skip-ic
python azalyst_engine.py --data-dir ./data --out-dir ./results --long-only
```

## Factor Universe (20 factors)
| Category | Factors |
|----------|---------|
| Momentum | MOM_1H, MOM_4H, MOM_1D, MOM_3D, MOM_1W, MOM_2W |
| Reversal | REV_1H, REV_4H |
| Volume | VOL_RATIO, VOL_MOM_1D |
| Volatility | RVOL_1D, RVOL_1W, VOL_OF_VOL |
| Microstructure | AMIHUD, MAX_RET, SKEW_1W, PRICE_ACCEL |
| Structural | TREND_48, BB_POS, RSI_RANK |

## Architecture
```
azalyst_engine.py        — Core: DataLoader, FactorEngine, CrossSectionalAnalyser, BacktestEngine
azalyst_data.py          — Binance data downloader
azalyst_execution.py     — ImpactModel, order execution simulation
azalyst_risk.py          — Risk management
azalyst_signal_combiner.py — Multi-factor signal combination
azalyst_alphaopt.py      — Alpha optimisation
azalyst_benchmark.py     — Benchmark comparison
azalyst_report.py        — Report generation
azalyst_tearsheet.py     — Performance tearsheet
```

## Data Format
Parquet files, one file per symbol (e.g. `BTCUSDT.parquet`):
- Column `timestamp`: Unix ms integer or datetime
- Columns: `open`, `high`, `low`, `close`, `volume` (float)
- Minimum history: 2 weeks of 5-min bars (~4,032 rows)

## Research Basis
Liu & Tsyvinski (2021), Fieberg et al. (2024), Baybutt (2024),
Borri et al. (2024), Kakushadze (2019), Dobrynskaya (2024), Cambridge CTREND (2024)

