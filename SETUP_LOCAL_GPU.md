# Azalyst — Local Setup Guide (ASUS FX506HF, RTX 2050)

## Your Hardware

| Component | Spec | Notes |
|---|---|---|
| CPU | i5-11260H | 6 cores / 12 threads — good |
| GPU | RTX 2050 4GB GDDR6 | CUDA working ✅ — BUT only 4GB VRAM |
| RAM | 16GB DDR4 dual channel | Enough |
| Virtual Memory | 48GB (3×16GB) | Good safety net |
| VRAM Bus | 64-bit | Narrow — slower than RTX 3050 |

**Key constraint: 4GB VRAM.** Training on 26M rows won't fit. The scripts below cap training at 4M rows automatically.

---

## Step 1: Install CUDA drivers + XGBoost

Open terminal as admin:

```bash
# Verify CUDA is installed
nvidia-smi

# You should see: NVIDIA GeForce RTX 2050, Driver Version, CUDA Version

# Install XGBoost with CUDA support
pip install xgboost --upgrade

# Also install other dependencies if needed
pip install numpy pandas scikit-learn scipy pyarrow lightgbm
```

---

## Step 2: Verify GPU works

```bash
# Run this first — tests XGBoost CUDA on RTX 2050
python azalyst_local_gpu.py
```

Expected output:
```
[1] Testing XGBoost CUDA on RTX 2050...
    RTX 2050 CUDA warm-up: OK (0.8s)
    Result: GPU READY ✓

[2] Free VRAM: ~3500 MB

[3] Safe max training rows for 4GB VRAM: 4,000,000

[4] Benchmark: 1M rows × 65 features...
    GPU: 45s  |  CPU (6 cores): 180s  |  Speedup: 4.0x
```

If GPU fails → still works on CPU (just slower).

---

## Step 3: Build feature cache (run once)

```bash
python build_feature_cache.py ^
    --data-dir ./data ^
    --out-dir ./feature_cache ^
    --workers 4
```

Expected: ~30-60 min. Creates one `.parquet` per symbol.

---

## Step 4: Train Year 1+2

```bash
# With GPU (recommended)
python azalyst_train_local.py ^
    --feature-dir ./feature_cache ^
    --out-dir ./results ^
    --gpu

# Without GPU (CPU fallback)
python azalyst_train_local.py ^
    --feature-dir ./feature_cache ^
    --out-dir ./results
```

Expected time:
- **GPU:** ~25-40 min
- **CPU:** ~90-120 min

---

## Step 5: Walk-forward test (Year 3)

```bash
python azalyst_weekly_loop.py ^
    --feature-dir ./feature_cache ^
    --results-dir ./results ^
    --gpu
```

Expected: ~2-4 hours total.

---

## Memory Management Tips

Your laptop has 16GB RAM + 48GB virtual memory. These settings help:

### Prevent RAM spikes during training:
The scripts automatically:
- Load features as `float32` (half the RAM of float64)
- Cap training at 4M rows (fits in 4GB VRAM)
- Run garbage collection between steps

### If you get OOM errors:
```bash
# Add --resample 1D flag (fewer rows, less RAM)
python azalyst_train_local.py --feature-dir ./feature_cache --out-dir ./results --gpu --resample 1D
```

### Close these before running:
- Chrome / Edge browser tabs
- Other heavy applications
- Spyder IDE (run from terminal instead)

---

## Expected Performance vs Kaggle T4

| | RTX 2050 (your laptop) | Kaggle T4 |
|---|---|---|
| VRAM | 4GB | 15GB |
| VRAM bus | 64-bit | 256-bit |
| Feature cache build | ~45 min | ~20 min |
| Year 1+2 training | ~35 min | ~15 min |
| Year 3 walk-forward | ~3 hours | ~1.5 hours |
| Total | ~4-5 hours | ~2-3 hours |

The RTX 2050 will be ~2-3x slower than the T4 mainly due to the 64-bit bus and 4GB VRAM limit forcing row capping. Results should be similar quality.

---

## File Summary

| File | What it does |
|---|---|
| `azalyst_local_gpu.py` | **Run first** — tests RTX 2050 CUDA |
| `azalyst_train_local.py` | Year 1+2 training with 4GB VRAM guard |
| `build_feature_cache.py` | Build feature cache (run once) |
| `azalyst_weekly_loop.py` | Walk-forward Year 3 |

---

## Troubleshooting

**"CUDA out of memory"**
→ The VRAM guard should prevent this. If it still happens, lower the cap:
```python
# In azalyst_local_gpu.py, change:
MAX_TRAIN_ROWS_GPU = 2_000_000  # try 2M instead of 4M
```

**"XGBoost CUDA failed"**
→ Try reinstalling: `pip install xgboost==2.1.0`
→ Check driver: `nvidia-smi` should show CUDA Version ≥ 11.0

**Training very slow**
→ Make sure Windows is not in battery saver mode
→ Right-click desktop → NVIDIA Control Panel → Manage 3D settings → Power management mode → **Prefer maximum performance**
→ ASUS Armoury Crate → Performance mode

**Virtual memory tip**
→ Your 48GB virtual memory is already set — good. Leave it at 3× RAM.
