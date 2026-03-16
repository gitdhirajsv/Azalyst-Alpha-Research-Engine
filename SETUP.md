# Azalyst — Setup Guide

## What was removed
- `azalyst_autonomous_team.py` — entire LLM loop (Olivia/Marcus)
- `azalyst_agent.py` — v2 agent
- All Ollama API calls
- `RUN_SHIFT_MONITOR.bat` / `Azalyst_Spyder.bat` launcher scripts

## New files
| File | Purpose |
|---|---|
| `azalyst_alpha_metrics.py` | Alpha calculator — target 1000% annual |
| `azalyst_train.py` | Year 1 training (pure ML) |
| `azalyst_weekly_loop.py` | Year 2+3 weekly predict→evaluate→retrain |
| `kaggle_pipeline.py` | Full pipeline for Kaggle GPU |
| `.github/workflows/azalyst_training.yml` | GitHub Actions CI/CD |

---

## Option A — Kaggle (faster, GPU)

### 1. Upload your data to Kaggle
You already have this done.

### 2. Create a new Kaggle notebook
- Go to kaggle.com → Code → New Notebook
- Title: `Azalyst ML Pipeline`
- Settings → Accelerator → **GPU T4 x2**
- Settings → Internet → On (needed to install packages)

### 3. Upload these files to the notebook session
Upload all 5 Python files:
```
build_feature_cache.py
azalyst_alpha_metrics.py
azalyst_train.py
azalyst_weekly_loop.py
kaggle_pipeline.py
```

In Kaggle notebook, add a cell at the top:
```python
import subprocess, sys

# Upload your files via the + Add Data button
# OR copy-paste each file using:
# %%writefile azalyst_train.py
# <paste file content>
```

### 4. Set the dataset path
In `kaggle_pipeline.py`, the data directory is auto-detected:
```python
input_dirs = list(Path("/kaggle/input").iterdir())
DATA_DIR = str(input_dirs[0])   # auto-finds your dataset
```
If you have multiple datasets, change to:
```python
DATA_DIR = "/kaggle/input/YOUR-DATASET-NAME"
```

### 5. Run the notebook
Click **Run All**. Expected time with T4 GPU:
- Feature cache: ~15-30 min
- Year 1 training: ~10-20 min
- Year 2 loop: ~60-120 min
- Year 3 loop: ~60-120 min
- Total: ~3-5 hours

### 6. Download results
When done, go to Output tab → Download `results.zip`

---

## Option B — GitHub Actions (automated, runs on every push)

### 1. Push these files to your GitHub repo
```
azalyst_alpha_metrics.py
azalyst_train.py
azalyst_weekly_loop.py
azalyst_weekly_loop.py
build_feature_cache.py
.github/workflows/azalyst_training.yml
```

### 2. Set up GitHub Secrets
Go to your repo → Settings → Secrets and variables → Actions → New secret

Add these 3 secrets:

| Secret name | Value |
|---|---|
| `KAGGLE_USERNAME` | Your Kaggle username |
| `KAGGLE_KEY` | Your Kaggle API key (from kaggle.com → Account → Create New Token) |
| `KAGGLE_DATASET` | `your-kaggle-username/your-dataset-name` |

To find your dataset name: go to your Kaggle dataset page, the name is in the URL.
Example: `kaggle.com/datasets/johndoe/binance-5min-crypto` → use `johndoe/binance-5min-crypto`

### 3. Trigger the workflow
Either:
- Push any change to `main` branch
- Go to Actions tab → `Azalyst ML Training Pipeline` → Run workflow

### 4. Download results
When the workflow finishes (3-8 hours):
- Go to Actions tab → click the completed run
- Scroll to Artifacts section
- Download `azalyst-results-{run_number}.zip`
- Also download `azalyst-key-files-{run_number}` for the key CSVs

---

## Results files explained

| File | What to send to Claude |
|---|---|
| `alpha_report.json` | ✅ Always send — overall summary |
| `weekly_summary_all.csv` | ✅ Always send — week-by-week performance |
| `all_trades_all.csv` | ✅ Always send — every trade |
| `feature_importance_year1.csv` | ✅ Send — which features mattered at start |
| `feature_importance_year*_week*.csv` | ✅ Send — how features changed after retraining |
| `train_summary.json` | ✅ Send — training metadata |
| `models/*.pkl` | ❌ Don't send to Claude — too large |

---

## Running locally (for testing)

```bash
# Step 1: Build feature cache (run once)
python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache

# Step 2: Train Year 1
python azalyst_train.py --feature-dir ./feature_cache --out-dir ./results

# Step 3: Run weekly loop
python azalyst_weekly_loop.py --feature-dir ./feature_cache --results-dir ./results

# With GPU (if you have NVIDIA CUDA)
python azalyst_train.py --feature-dir ./feature_cache --out-dir ./results --gpu
python azalyst_weekly_loop.py --feature-dir ./feature_cache --results-dir ./results --gpu

# Year 2 only (faster test)
python azalyst_weekly_loop.py --feature-dir ./feature_cache --results-dir ./results --year2-only
```

---

## Alpha target

The system targets **1000% annual return (10x capital)**.

- Weekly equivalent: ~4.65% per week (compounded)
- Retrain trigger: rolling 4-week annualised return < 1000%
- Catastrophic loss guard: single week < -15% → immediate retrain

This means the model will retrain frequently at first (it's learning).
Over time, with more data, it should retrain less often as it finds stable signals.

---

## How to interpret results for Claude

When you send results to Claude, include:

1. **alpha_report.json** — did it achieve 1000%? How many retrains?
2. **Top/bottom weeks from weekly_summary_all.csv** — when did it work/fail?
3. **Feature importances before vs after retraining** — what changed?
4. **Trade distribution** — which signals (BUY/SELL), which symbols?

Claude will then suggest:
- Feature engineering improvements
- Model architecture changes
- Signal generation improvements
- Position sizing adjustments
