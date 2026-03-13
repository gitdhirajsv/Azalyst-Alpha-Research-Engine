# Module Reference (Branch3 snapshot)

| File | Purpose |
|---|---|
| `azalyst_orchestrator.py` | Master pipeline that chains together all 8 stages end-to-end. |
| `azalyst_validator.py` | Institutional validation (style neutralization, Fama-MacBeth, BH correction). |
| `azalyst_factors_v2.py` | Factor library v2 with 35 crypto-native alpha signals. |
| `azalyst_ml.py` | Fast ML module (LightGBM + CUDA) plus regime detection and pump detection. |
| `azalyst_engine.py` | Data loader, IC research, and core backtest engine. |
| `azalyst_data.py` | Polars + DuckDB analytics helper layer for high-performance data processing. |
| `azalyst_statarb.py` | Engle-Granger cointegration scanner for statistical arbitrage pairs. |
| `azalyst_risk.py` | Portfolio optimization (MVO, HRP, Black-Litterman). |
| `azalyst_output/` | Signals, IC results, paper trades, and performance reports. |
