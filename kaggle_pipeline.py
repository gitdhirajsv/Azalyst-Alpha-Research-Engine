"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  KAGGLE PIPELINE  (GPU-optimised)
║                                                                             ║
║  Run this in a Kaggle notebook with GPU accelerator.                       ║
║  Input: your Kaggle dataset containing the parquet OHLCV files.            ║
║  Output: /kaggle/working/results.zip  (download after run)                 ║
║                                                                             ║
║  In Kaggle:                                                                 ║
║    1. New Notebook → Upload this file                                      ║
║    2. Add your dataset as input  (parquet files)                           ║
║    3. Settings → Accelerator → GPU T4 x2                                  ║
║    4. Run All                                                               ║
║    5. Download results.zip from /kaggle/working/                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# ── Cell 1: Environment detection and setup ──────────────────────────────────
import os
import sys
import subprocess
import time
from pathlib import Path

IS_KAGGLE  = os.path.exists("/kaggle/working")
IS_GITHUB  = os.environ.get("GITHUB_ACTIONS") == "true"
IS_COLAB   = os.path.exists("/content")

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'GitHub Actions' if IS_GITHUB else 'Local'}")

# Paths
if IS_KAGGLE:
    # Find the dataset input directory
    input_dirs = list(Path("/kaggle/input").iterdir()) if Path("/kaggle/input").exists() else []
    DATA_DIR   = str(input_dirs[0]) if input_dirs else "/kaggle/input/azalyst-data"
    WORK_DIR   = "/kaggle/working"
else:
    DATA_DIR   = "./data"
    WORK_DIR   = "."

FEATURE_DIR  = os.path.join(WORK_DIR, "feature_cache")
RESULTS_DIR  = os.path.join(WORK_DIR, "results")
USE_GPU      = True  # Kaggle T4 GPU

for d in [FEATURE_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Data dir    : {DATA_DIR}")
print(f"Feature dir : {FEATURE_DIR}")
print(f"Results dir : {RESULTS_DIR}")

# ── Cell 2: Install / verify dependencies ────────────────────────────────────
def install_if_needed(packages):
    for pkg in packages:
        try:
            __import__(pkg.split("==")[0].replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   pkg, "-q"])

install_if_needed(["lightgbm", "pyarrow", "polars"])

# Detect GPU
import subprocess as _sp
HAS_NVIDIA = False
try:
    r = _sp.run(["nvidia-smi"], capture_output=True, timeout=5)
    HAS_NVIDIA = r.returncode == 0
except Exception:
    pass
print(f"GPU available: {HAS_NVIDIA}")

# ── Cell 3: Build feature cache ───────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: Building feature cache")
print("="*60)

t0 = time.time()

# Run build_feature_cache.py
result = subprocess.run(
    [sys.executable, "build_feature_cache.py",
     "--data-dir",    DATA_DIR,
     "--out-dir",     FEATURE_DIR,
     "--workers",     "4"],
    capture_output=False,
    text=True,
)
if result.returncode != 0:
    print(f"[ERROR] Feature cache build failed")
    if hasattr(result, "stderr") and result.stderr:
        print(result.stderr[-2000:])
else:
    cached = list(Path(FEATURE_DIR).glob("*.parquet"))
    print(f"  Feature cache: {len(cached)} symbols in "
          f"{(time.time()-t0)/60:.1f} min")

# ── Cell 4: Year 1 training ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: Year 1 Training")
print("="*60)

t1 = time.time()
gpu_flag = ["--gpu"] if HAS_NVIDIA else []

result = subprocess.run(
    [sys.executable, "azalyst_train.py",
     "--feature-dir", FEATURE_DIR,
     "--out-dir",     RESULTS_DIR,
     "--year1-days",  "365",
     *gpu_flag],
    capture_output=False,
    text=True,
)

if result.returncode != 0:
    print("[ERROR] Training failed")
else:
    print(f"  Year 1 training complete in {(time.time()-t1)/60:.1f} min")

# ── Cell 5: Weekly loop Year 2 + Year 3 ──────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: Weekly Self-Improving Loop (Year 2 + Year 3)")
print("="*60)

t2 = time.time()

result = subprocess.run(
    [sys.executable, "azalyst_weekly_loop.py",
     "--feature-dir", FEATURE_DIR,
     "--results-dir", RESULTS_DIR,
     *gpu_flag],
    capture_output=False,
    text=True,
)

if result.returncode != 0:
    print("[ERROR] Weekly loop failed")
else:
    print(f"  Weekly loop complete in {(time.time()-t2)/60:.1f} min")

# ── Cell 6: Verify outputs ────────────────────────────────────────────────────
print("\n" + "="*60)
print("  OUTPUT FILES")
print("="*60)

results_path = Path(RESULTS_DIR)
important_files = [
    "weekly_summary_all.csv",
    "all_trades_all.csv",
    "all_trades_year2.csv",
    "all_trades_year3.csv",
    "weekly_summary_Year2.csv",
    "weekly_summary_Year3.csv",
    "alpha_report.json",
    "feature_importance_year1.csv",
    "train_summary.json",
    "date_config.json",
]

found = []
for fname in important_files:
    p = results_path / fname
    if p.exists():
        size_kb = p.stat().st_size // 1024
        print(f"  ✓ {fname:<40} {size_kb:>6} KB")
        found.append(str(p))
    else:
        print(f"  ✗ {fname:<40}  MISSING")

# All model files
model_files = list((results_path / "models").glob("*.pkl"))
print(f"  ✓ models/ — {len(model_files)} model checkpoints")

# Feature importance files
imp_files = list(results_path.glob("feature_importance_*.csv"))
print(f"  ✓ feature_importance_*.csv — {len(imp_files)} files")

# ── Cell 7: Print alpha summary ───────────────────────────────────────────────
import json, pandas as pd

alpha_json = results_path / "alpha_report.json"
if alpha_json.exists():
    with open(alpha_json) as fh:
        rpt = json.load(fh)
    print("\n" + "="*60)
    print("  ALPHA REPORT SUMMARY")
    print("="*60)
    for k, v in rpt.items():
        print(f"  {k:<30} {v}")

# Quick peek at weekly performance
wk_all = results_path / "weekly_summary_all.csv"
if wk_all.exists():
    wk = pd.read_csv(wk_all)
    print(f"\n  Weekly returns summary:")
    print(f"  Mean weekly return   : {wk['week_return_pct'].mean():.2f}%")
    print(f"  Weeks on track       : "
          f"{wk['on_track'].sum()} / {len(wk)}")
    print(f"  Total retrains       : {wk['retrained'].sum()}")
    if "annualised_pct" in wk.columns:
        print(f"  Avg annualised proj  : {wk['annualised_pct'].mean():.0f}%")

# ── Cell 8: Package everything into results.zip ───────────────────────────────
print("\n" + "="*60)
print("  PACKAGING results.zip")
print("="*60)

zip_path = os.path.join(WORK_DIR, "results.zip")
subprocess.run(
    ["zip", "-r", zip_path, RESULTS_DIR, "-x", "*/feature_cache/*"],
    cwd=WORK_DIR,
)

if os.path.exists(zip_path):
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  results.zip created: {size_mb:.1f} MB")
    print(f"  Download from Kaggle output tab: results.zip")
else:
    # Fallback: list the results dir
    print("  zip not available — files are in:")
    print(f"  {RESULTS_DIR}")
    for f in sorted(results_path.rglob("*.csv")):
        print(f"    {f.relative_to(results_path)}")

total_time = (time.time() - t0) / 3600
print(f"\n  Total pipeline time: {total_time:.2f} hours")
print(f"\n  Files to send to Claude for review:")
print(f"    1. alpha_report.json")
print(f"    2. weekly_summary_all.csv")
print(f"    3. all_trades_all.csv")
print(f"    4. feature_importance_year1.csv")
print(f"    5. feature_importance_* (retrain checkpoints)")
