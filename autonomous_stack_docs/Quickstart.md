# Quickstart (Autonomous Monitor Edition)

## 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 2. Optional: enable notebook monitoring

```bash
pip install notebook ipykernel
```

## 3. Install and prepare Ollama

```bash
ollama pull deepseek-r1:14b
```

## 4. Add market data

Drop Binance 5-minute parquet files into `data/`.

Required schema:

```text
timestamp | open | high | low | close | volume
```

## 5. Launch the platform

### Windows one-click launcher

Run:

```text
RUN_SHIFT_MONITOR.bat
```

This launcher:
- starts the local dashboard,
- opens the Jupyter notebook monitor when available,
- reconnects to Ollama,
- warms the model, and
- starts the autonomous research team.

### Manual pipeline

```bash
python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output
```
