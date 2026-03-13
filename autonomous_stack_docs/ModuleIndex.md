# Module Reference

| File | Purpose |
| --- | --- |
| `azalyst_orchestrator.py` | Master pipeline that chains the core research stages end to end. |
| `azalyst_validator.py` | Institutional validation: style neutralization, Fama-MacBeth, and BH correction. |
| `azalyst_factors_v2.py` | Factor library with 35 crypto-native alpha signals. |
| `azalyst_ml.py` | ML layer with LightGBM, regime detection, and predictive models. |
| `azalyst_engine.py` | Data loading, IC research, and core backtest utilities. |
| `azalyst_data.py` | Polars and DuckDB analytics layer for high-performance processing. |
| `azalyst_statarb.py` | Cointegration scanner for statistical arbitrage research. |
| `azalyst_risk.py` | Portfolio optimization modules including MVO, HRP, and Black-Litterman. |
| `azalyst_autonomous_team.py` | Local multi-agent autonomous runner. |
| `walkforward_simulator.py` | Rolling walk-forward simulation with checkpoint support. |
| `monitor_dashboard.py` | Browser-based live monitor. |
| `Azalyst_Live_Monitor.ipynb` | Jupyter notebook monitor. |
| `azalyst_output/` | Signals, IC results, paper trades, and performance reports. |
