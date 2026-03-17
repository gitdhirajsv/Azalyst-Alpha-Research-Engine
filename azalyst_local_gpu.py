"""
╔══════════════════════════════════════════════════════════════════════════════╗
     AZALYST — LOCAL GPU CONFIG  |  RTX 2050 4GB VRAM  |  i5-11260H
╔══════════════════════════════════════════════════════════════════════════════╗

Key constraints on your hardware:
  RTX 2050    : 4GB VRAM, 64-bit bus — must cap training rows
  i5-11260H   : 6 cores — parallel data loading fine
  16GB RAM    : enough for feature cache + model
  Virtual mem : 48GB — good safety net

STEP 1: Install XGBoost with CUDA (if not done yet):
    pip install xgboost --upgrade
    
STEP 2: Run this file to verify GPU works:
    python azalyst_local_gpu.py

STEP 3: Then run pipeline with GPU flag:
    python azalyst_train.py --feature-dir ./feature_cache --out-dir ./results --gpu
    python azalyst_weekly_loop.py --feature-dir ./feature_cache --results-dir ./results --gpu
"""

from __future__ import annotations
import numpy as np
import time

# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE CONSTANTS  (tuned for RTX 2050 4GB VRAM)
# ─────────────────────────────────────────────────────────────────────────────

# Max training rows that fit in 4GB VRAM at float32, 65 features
# 4GB VRAM / (65 features × 4 bytes) = ~15.4M rows theoretical max
# XGBoost histogram buffers add overhead → safe cap is ~4M rows
MAX_TRAIN_ROWS_GPU = 4_000_000   # 4M rows — safe for 4GB VRAM

# XGBoost params tuned for RTX 2050
#   max_bin=128 : less memory than 256, minimal accuracy loss
#   tree_method='hist' + device='cuda' : histogram GPU mode
#   subsample=0.7 : reduces GPU memory per tree
#   max_depth=5   : shallower trees use less VRAM
XGBOOST_GPU_PARAMS = {
    "tree_method":        "hist",
    "device":             "cuda",
    "max_bin":            128,          # RTX 2050 safe (256 needs more VRAM)
    "n_estimators":       500,
    "learning_rate":      0.05,
    "max_depth":          5,            # shallower = less VRAM per tree
    "min_child_weight":   30,           # avoid overfitting
    "subsample":          0.7,          # stochastic gradient — saves VRAM
    "colsample_bytree":   0.7,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "objective":          "binary:logistic",
    "eval_metric":        "auc",
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             1,            # GPU mode: CPU threads not used for trees
}

# Fallback CPU params if GPU fails
XGBOOST_CPU_PARAMS = {
    "tree_method":        "hist",
    "device":             "cpu",
    "max_bin":            256,
    "n_estimators":       300,
    "learning_rate":      0.05,
    "max_depth":          6,
    "min_child_weight":   30,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "objective":          "binary:logistic",
    "eval_metric":        "auc",
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             6,            # i5-11260H has 6 cores
}


# ─────────────────────────────────────────────────────────────────────────────
#  GPU DETECTION + WARM-UP TEST
# ─────────────────────────────────────────────────────────────────────────────

def detect_gpu() -> tuple[bool, str]:
    """
    Test XGBoost CUDA on RTX 2050.
    Returns (gpu_available, device_string).
    """
    try:
        import xgboost as xgb
        print(f"  XGBoost version: {xgb.__version__}")
    except ImportError:
        return False, "cpu"

    # Test with small data
    X_test = np.random.rand(1000, 10).astype(np.float32)
    y_test = np.random.randint(0, 2, 1000).astype(float)
    dtrain = xgb.DMatrix(X_test, label=y_test)

    try:
        t0 = time.time()
        params = {"tree_method": "hist", "device": "cuda",
                  "max_bin": 128, "max_depth": 4, "verbosity": 0,
                  "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        elapsed = time.time() - t0
        print(f"  RTX 2050 CUDA warm-up: OK ({elapsed:.2f}s)")
        return True, "cuda"
    except Exception as e:
        print(f"  CUDA failed: {e}")
        print("  Falling back to CPU")
        return False, "cpu"


def get_vram_mb() -> int:
    """Approximate VRAM available via nvidia-smi."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0


def safe_max_rows() -> int:
    """
    Dynamically compute max safe training rows based on available VRAM.
    Formula: (free_vram_bytes × 0.6) / (65 features × 4 bytes)
    """
    free_mb = get_vram_mb()
    if free_mb < 100:
        # nvidia-smi unavailable — use conservative default
        return MAX_TRAIN_ROWS_GPU
    free_bytes = free_mb * 1024 * 1024
    usable     = free_bytes * 0.60         # use 60% of free VRAM
    n_features = 65
    bytes_per_row = n_features * 4         # float32
    max_rows = int(usable / bytes_per_row)
    max_rows = min(max_rows, MAX_TRAIN_ROWS_GPU)
    max_rows = max(max_rows, 500_000)      # never less than 500K
    return max_rows


# ─────────────────────────────────────────────────────────────────────────────
#  STRATIFIED TIME-SERIES SUBSAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def subsample_for_vram(X: np.ndarray, y: np.ndarray,
                       max_rows: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample a training dataset to fit in 4GB VRAM.
    Uses evenly-spaced time sampling to preserve temporal distribution.
    Stratified by label to preserve class balance.
    """
    if max_rows is None:
        max_rows = safe_max_rows()

    n = len(X)
    if n <= max_rows:
        return X, y

    print(f"  [VRAM guard] {n:,} rows → {max_rows:,} rows "
          f"(RTX 2050 4GB limit)")

    # Stratified: sample each class proportionally
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_keep  = max(10, int(max_rows * len(cls_idx) / n))
        if n_keep >= len(cls_idx):
            indices.append(cls_idx)
        else:
            # Evenly spaced — preserves temporal distribution
            step   = len(cls_idx) / n_keep
            chosen = cls_idx[np.round(np.arange(0, len(cls_idx), step))
                               .astype(int)[:n_keep]]
            indices.append(chosen)

    idx = np.concatenate(indices)
    idx.sort()   # restore time order
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  — run this to verify GPU setup on your RTX 2050
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AZALYST  —  RTX 2050 GPU SETUP CHECK")
    print("=" * 60)

    # 1. Detect GPU
    print("\n[1] Testing XGBoost CUDA on RTX 2050...")
    gpu_ok, device = detect_gpu()
    print(f"  Result: {'GPU READY ✓' if gpu_ok else 'CPU fallback'}")

    # 2. Check VRAM
    free_mb = get_vram_mb()
    if free_mb > 0:
        print(f"\n[2] Free VRAM: {free_mb} MB")
    else:
        print(f"\n[2] VRAM check: assuming 3500 MB free (nvidia-smi unavailable)")

    # 3. Show safe training limit
    max_rows = safe_max_rows()
    print(f"\n[3] Safe max training rows for 4GB VRAM: {max_rows:,}")

    # 4. Benchmark GPU vs CPU
    if gpu_ok:
        print("\n[4] Benchmark: 1M rows × 65 features...")
        import xgboost as xgb

        X_bench = np.random.rand(1_000_000, 65).astype(np.float32)
        y_bench = np.random.randint(0, 2, 1_000_000).astype(float)
        d = xgb.DMatrix(X_bench, label=y_bench)

        # GPU
        t0 = time.time()
        params_gpu = {**XGBOOST_GPU_PARAMS, "n_estimators": None}
        params_gpu.pop("n_estimators", None)
        params_gpu.pop("random_state", None)
        params_gpu.pop("n_jobs", None)
        params_gpu.pop("eval_metric", None)
        xgb.train(params_gpu, d, num_boost_round=50, verbose_eval=False)
        gpu_time = time.time() - t0

        # CPU
        t0 = time.time()
        params_cpu = {**XGBOOST_CPU_PARAMS}
        params_cpu.pop("n_estimators", None)
        params_cpu.pop("random_state", None)
        params_cpu.pop("eval_metric", None)
        params_cpu["n_jobs"] = 6
        xgb.train(params_cpu, d, num_boost_round=50, verbose_eval=False)
        cpu_time = time.time() - t0

        speedup = cpu_time / gpu_time
        print(f"  GPU: {gpu_time:.1f}s  |  CPU (6 cores): {cpu_time:.1f}s  |  "
              f"Speedup: {speedup:.1f}x")

    print("\n" + "=" * 60)
    if gpu_ok:
        print("  ✓ RTX 2050 CUDA working — run pipeline with --gpu flag")
        print("  ✓ Training capped at", f"{max_rows:,}", "rows (VRAM safe)")
        print("\n  Commands:")
        print("    python azalyst_train.py --feature-dir ./feature_cache \\")
        print("        --out-dir ./results --gpu")
        print("    python azalyst_weekly_loop.py --feature-dir ./feature_cache \\")
        print("        --results-dir ./results --gpu")
    else:
        print("  ⚠ GPU not working — running on CPU (i5-11260H 6 cores)")
        print("  Run without --gpu flag. Will be slower but still works.")
    print("=" * 60)
