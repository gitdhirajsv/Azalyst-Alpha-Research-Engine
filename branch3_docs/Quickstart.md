# Quickstart (branch3 edition)

1. **Install Dependencies**

```bash
pip install pandas numpy scipy scikit-learn lightgbm statsmodels polars duckdb pyarrow
```

> For maximum throughput, build LightGBM with CUDA support.

2. **Add Your Data**

Drop Binance 5-minute parquet files into `data/`.
Required schema: `timestamp | open | high | low | close | volume`.

3. **Run the Pipeline**

- Windows: double-click `RUN_AZALYST.bat`.
- Command line:

```bash
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```
