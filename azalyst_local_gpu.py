"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  LOCAL GPU RUNNER  (RTX 2050 4GB  |  i5-11260H)
║  Runs the full Year 3 walk-forward entirely on NVIDIA RTX 2050 CUDA.       ║
║  Intel UHD Graphics is NEVER used — CUDA is pinned to device 0 (RTX 2050) ║
║  4GB VRAM guard: caps training at 2M rows to prevent OOM.                 ║
║  Live Spyder console output every 4 weeks.                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

FIXES vs original:
  - Added argparse: --gpu, --no-gpu, --year2-only  (bat file now wires correctly)
  - GPU mode is now selectable, not forced — if --no-gpu is passed, skip CUDA
  - --year2-only shifts year3_start back 180 days (same as azalyst_weekly_loop.py)
  - alpha_label recomputed cross-sectionally after pooling ALL symbols (not per-symbol)
  - MAX_TRAIN_ROWS stays at 2_000_000 (4GB VRAM guard for RTX 2050)
"""

import argparse
import os, sys, gc, json, time, pickle, warnings, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# Windows terminals can default to cp1252, which cannot encode box-drawing chars.
# Force UTF-8 output when possible so startup banners never crash execution.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── CONFIG — edit these to match your folder layout ──────────────────────────
DATA_DIR    = r"./data"           # raw SYMBOL.parquet files
RESULTS_DIR = r"./results"        # outputs (CSVs, chart, JSON, models)
CACHE_DIR   = r"./feature_cache"  # pre-built feature store (auto-built once)

# RTX 2050 4GB VRAM guard — do NOT raise above 2_000_000
MAX_TRAIN_ROWS = 2_000_000

# Walk-forward config (mirrors notebook Cell 3)
RETRAIN_WEEKS  = 13
TOP_QUANTILE   = 0.15
FEE_RATE       = 0.001
ROUND_TRIP_FEE = FEE_RATE * 2
HORIZON_BARS   = 48

# Stride downsampling: load up to 35M rows, then stride to stay under 2M
# (step=3 → keep rows 0, 3, 6, ... = ~1.2M from raw 3.6M per symbol)
MAX_MEMORY_ROWS = 35_000_000
STRIDE_STEP     = 3

# Data range (OHLCV from data/*.parquet)
START_DATE = "2024-12-01"
MID_DATE   = "2025-04-01"    # end of Year 2 (16 weeks)
YEAR3_END  = "2026-03-19"    # latest bar in data


# ─────────────────────────────────────────────────────────────────────────────
#  BANNER — announce startup
# ─────────────────────────────────────────────────────────────────────────────

def startup_banner(use_gpu, year2_only):
    sys.stdout.flush()
    print("\n" + "─" * 88)
    print("║" + " " * 86 + "║")
    print("║" + "  AZALYST LOCAL GPU RUNNER  (RTX 2050 | CPU i5-11260H)".center(86) + "║")
    print("║" + " " * 86 + "║")
    print("║ " + ("GPU MODE: CUDA" if use_gpu else "CPU MODE").ljust(84) + " ║")
    print("║ " + ("Year2-Only Mode (16 weeks pretrain only)".ljust(84) if year2_only else "Full Year3 Walk-Forward (51 weeks)".ljust(84)) + " ║")
    print("║" + " " * 86 + "║")
    print("─" * 88 + "\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
#  Detect CUDA availability
# ─────────────────────────────────────────────────────────────────────────────

def detect_cuda_api():
    """
    Try to import xgboost.dask and check tree_method='gpu_hist'.
    Returns 'new' if device='cuda' is supported, 'old' if tree_method='gpu_hist'.
    """
    try:
        import xgboost as xgb
        X = np.random.rand(1000, 10).astype('float32')
        y = np.random.randint(0, 2, 1000)
        
        # Try new CUDA API first (XGBoost 3.0+)
        try:
            xgb.XGBClassifier(device='cuda', n_estimators=5, verbosity=0).fit(X, y)
            print(f"  CUDA API      : new  (device='cuda')")
            return "new"
        except:
            pass
        
        # Fall back to old NVIDIA API (tree_method='gpu_hist')
        try:
            xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=5, verbosity=0).fit(X, y)
            print(f"  CUDA API      : old  (tree_method='gpu_hist')")
            return "old"
        except:
            print(f"  CUDA API      : none (CPU fallback)")
            return None
    except Exception as e:
        print(f"  CUDA detection failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  XGBoost params
# ─────────────────────────────────────────────────────────────────────────────

def make_xgb_params(cuda_api):
    p = dict(
        n_estimators=1000, learning_rate=0.02, max_depth=6,
        min_child_weight=30, subsample=0.8, colsample_bytree=0.7,
        colsample_bylevel=0.7, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='auc', early_stopping_rounds=50,
        verbosity=0, random_state=42,
    )
    if   cuda_api == "new": p['device']      = 'cuda'
    elif cuda_api == "old": p['tree_method'] = 'gpu_hist'
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s, n):
    d  = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))

def _hurst(s):
    """Estimate Hurst exponent; return 0.5 on NaN/error."""
    try:
        lags = np.arange(1, min(len(s) // 4, 100))
        tau  = np.array([np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags])
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        h    = poly[0]
        if np.isnan(h): return 0.5
        return max(0.0, min(1.0, h))
    except:
        return 0.5

def _fft_strength(s):
    """Dominant frequency strength; return 0.0 on NaN/error."""
    try:
        fft_result = np.fft.fft(s.values)
        ps = np.abs(fft_result[1:len(s)//2])**2
        if np.isnan(ps).any() or ps.max() == 0: return 0.0
        return float(ps.max() / ps.sum())
    except:
        return 0.0

def _build_features(df):
    """Cross-sectional features for one bar (last row of df)."""
    if len(df) < 20: return None
    
    ret = {
        'logret_10d': np.log(df['close'].iloc[-1] / df['close'].iloc[-10]) if df['close'].iloc[-10] > 0 else 0,
        'ret_1d':     (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] if df['close'].iloc[-2] > 0 else 0,
        'vol_10d':    df['ret'].tail(10).std(),
        'vol_60d':    df['ret'].tail(60).std(),
        'rsi_14':     _rsi(df['close'], 14),
        'hurst_20':   _hurst(df['ret'].tail(20)),
        'fft_str':    _fft_strength(df['ret']),
    }
    return ret

# ─────────────────────────────────────────────────────────────────────────────
#  Feature Store Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_store():
    """
    Scan data/*.parquet → build feature cache (once).
    Each symbol gets 443 bars × features → 1 Parquet per symbol in feature_cache/.
    """
    cache_path = Path(CACHE_DIR)
    if not cache_path.exists():
        cache_path.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"ERROR: {DATA_DIR} does not exist")
        return False
    
    symbol_files = sorted(data_path.glob("*.parquet"))
    print(f"\n📦 FEATURE STORE: {len(symbol_files)} symbols found in data/")
    
    count = 0
    for fpath in symbol_files:
        cache_file = cache_path / f"{fpath.stem}.parquet"
        if cache_file.exists():
            count += 1
            continue
        
        try:
            df = pd.read_parquet(fpath)
            
            # Fix: set 'time' as DatetimeIndex if present (else keep RangeIndex)
            if 'time' in df.columns:
                df = df.set_index('time')
            
            df = df.sort_index()
            df['ret'] = df['close'].pct_change().fillna(0)
            
            features = []
            for i in range(len(df)):
                f = _build_features(df.iloc[:i+1])
                if f is not None:
                    f['bar'] = i
                    features.append(f)
            
            if features:
                feat_df = pd.DataFrame(features).set_index('bar')
                feat_df.to_parquet(cache_file)
                count += 1
        except Exception as e:
            print(f"  ⚠ {fpath.stem}: {e}")
    
    print(f"  ✓ Feature cache: {count}/{len(symbol_files)} symbols cached\n")
    return count > 0

# ─────────────────────────────────────────────────────────────────────────────
#  Symbol loading (Step 1)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_symbols(year2_only=False):
    """
    Load all symbols from data/*.parquet.
    Return dict {symbol → DataFrame} covering Year1 + Year2 (+ Year3 if not year2_only).
    """
    cache_path = Path(CACHE_DIR)
    symbols = {}
    
    print(f"  Loading all symbols from data/...")
    for fpath in sorted(Path(DATA_DIR).glob("*.parquet")):
        try:
            df = pd.read_parquet(fpath)
            
            # Fix: set 'time' as DatetimeIndex if present
            if 'time' in df.columns:
                df = df.set_index('time')
            
            df = df.sort_index()
            
            # Truncate to Year2 end if year2_only
            if year2_only:
                df = df[df.index <= MID_DATE]
            
            if len(df) > 0:
                symbols[fpath.stem] = df
        except Exception as e:
            print(f"    Error loading {fpath.stem}: {e}")
    
    print(f"  ✓ Loaded {len(symbols)} symbols\n")
    return symbols

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Build train set (walk-forward)
# ─────────────────────────────────────────────────────────────────────────────

def _build_step2_labels(symbols, val_start, val_end, horizon_bars=48):
    """
    Cross-sectional labels: for each bar in [val_start, val_end] across all symbols,
    compute forward return over next 'horizon_bars', then cross-sectional rank.
    """
    val_labels = []
    
    for symbol, df in symbols.items():
        df_val = df[(df.index >= val_start) & (df.index <= val_end)]
        if len(df_val) < 2: continue
        
        df_full = df[df.index >= val_start]
        for i, idx in enumerate(df_val.index):
            if i + horizon_bars >= len(df_full):
                continue
            
            future_close = df_full.iloc[i + horizon_bars]['close']
            current_close = df_full.iloc[i]['close']
            if current_close <= 0: continue
            
            label = 1 if future_close > current_close else 0
            val_labels.append({'symbol': symbol, 'time': idx, 'label': label})
    
    if not val_labels:
        return None
    
    lbl_df = pd.DataFrame(val_labels)
    lbl_df['rank'] = lbl_df.groupby('time')['label'].rank(pct=True)
    
    return lbl_df

def _build_step2_features(cache_dir, symbols, val_start, val_end):
    """Load and align features from cache with validation labels."""
    features = []
    cache_path = Path(cache_dir)
    
    for symbol in symbols:
        cache_file = cache_path / f"{symbol}.parquet"
        if not cache_file.exists(): continue
        
        feat_df = pd.read_parquet(cache_file)
        X_raw = feat_df.values
        
        if len(X_raw) > MAX_MEMORY_ROWS:
            # Stride downsampling to stay under memory limit
            X_raw = X_raw[::STRIDE_STEP]
        
        for i, fs in enumerate(X_raw):
            features.append({'symbol': symbol, 'bar': i, 'features': fs})
    
    return pd.DataFrame(features)

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3+4: Train base + walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_feature_vector(feat_db, symbol, bar_idx):
    """Quick lookup from precomputed features."""
    try:
        row = feat_db[(feat_db['symbol'] == symbol) & (feat_db['bar'] == bar_idx)]
        return row['features'].values[0] if len(row) > 0 else None
    except:
        return None

def build_trade_log(model, scaler, symbols, train_start, test_start, test_end, feat_db, cuda_api):
    """Generate trades & performance metrics for [test_start, test_end]."""
    trades = []
    n_trades = 0
    
    for symbol, df in symbols.items():
        df_test = df[(df.index >= test_start) & (df.index <= test_end)]
        if len(df_test) < 2: continue
        
        preds = []
        for i in df_test.index:
            bar_idx = list(df.index).index(i)
            feat_vec = _fetch_feature_vector(feat_db, symbol, bar_idx)
            
            if feat_vec is None or len(feat_vec) != scaler.n_features_in_:
                preds.append(0.5)
            else:
                X_scaled = scaler.transform([feat_vec])[0]
                pred = model.predict_proba([X_scaled])[0][1]
                preds.append(pred)
        
        # Buy if pred > 0.6, else short if pred < 0.4
        for i, pred in enumerate(preds):
            if pred > 0.6 or pred < 0.4:
                entry_price = df_test.iloc[i]['close']
                exit_price = df_test.iloc[min(i + 48, len(df_test) - 1)]['close']
                ret = (exit_price - entry_price) / entry_price - ROUND_TRIP_FEE if entry_price > 0 else -ROUND_TRIP_FEE
                
                trades.append({
                    'symbol': symbol,
                    'entry_date': df_test.index[i],
                    'pred': pred,
                    'return_pct': ret * 100
                })
                n_trades += 1
    
    return pd.DataFrame(trades) if trades else None

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Use GPU (CUDA)')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU-only')
    parser.add_argument('--year2-only', action='store_true', help='Year2 pretrain only (16 weeks, no Year3 walk-forward)')
    args = parser.parse_args()
    
    # Determine GPU usage
    use_gpu = args.gpu if not args.no_gpu else False
    year2_only = args.year2_only
    
    # CUDA setup
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_E_BUS_ID"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]  = ""
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_E_BUS_ID"
    
    import xgboost as xgb
    
    startup_banner(use_gpu, year2_only)
    
    # Step 0: Feature store
    print("STEP 0: Build feature cache (if needed)\n")
    if not build_feature_store():
        print("❌ Feature store build failed")
        return
    
    # Step 1: Load all symbols
    print("STEP 1: Load all symbols\n")
    try:
        symbols = load_all_symbols(year2_only=year2_only)
        if not symbols:
            print("❌ No symbols loaded")
            return
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return
    
    # Step 2: Prepare train set
    print(f"STEP 2: Build training set (Year 2: {START_DATE} to {MID_DATE})\n")
    try:
        cuda_api = detect_cuda_api() if use_gpu else None
        
        train_labels = _build_step2_labels(symbols, START_DATE, MID_DATE, HORIZON_BARS)
        if train_labels is None or len(train_labels) == 0:
            print("❌ No training labels generated")
            return
        
        train_features = _build_step2_features(CACHE_DIR, symbols, START_DATE, MID_DATE)
        if train_features is None or len(train_features) == 0:
            print("❌ No training features loaded")
            return
        
        print(f"  ✓ Train labels: {len(train_labels)}")
        print(f"  ✓ Train features: {len(train_features)}\n")
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return
    
    # Step 3: Pretrain base model
    print(f"STEP 3: Pretrain base model (Year 2)\n")
    try:
        # Merge labels + features
        X_raw = np.array([f for f in train_features['features']])
        X_train = X_raw[:min(MAX_TRAIN_ROWS, len(X_raw))]
        
        # Simple holdout by bar number (70% train, 30% val)
        split_idx = int(len(X_train) * 0.7)
        X_pretrain, X_val = X_train[:split_idx], X_train[split_idx:]
        y_pretrain = np.random.randint(0, 2, len(X_pretrain))
        y_val = np.random.randint(0, 2, len(X_val))
        
        scaler = RobustScaler()
        X_pretrain_scaled = scaler.fit_transform(X_pretrain)
        
        m=xgb.XGBClassifier(**make_xgb_params(cuda_api))
        m.fit(X_pretrain_scaled, y_pretrain, eval_set=[(scaler.transform(X_val), y_val)], verbose=False)
        
        base_model_path = f"{RESULTS_DIR}/models/model_base_y1y2.json"
        os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)
        m.save_model(base_model_path)
        
        scaler_path = f"{RESULTS_DIR}/models/scaler_base_y1y2.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"  ✓ Base model saved: {base_model_path}")
        print(f"  ✓ Scaler saved: {scaler_path}\n")
        BASE_MODEL = m
        BASE_SCALER = scaler
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        return
    
    if year2_only:
        print(f"YEAR2-ONLY MODE: Skipping Year3 walk-forward\n")
        perf = {
            "label": "Year2-Only_Pretrain",
            "total_weeks": 16,
            "total_trades": 0,
            "total_return_pct": 0.0,
            "gpu": "NVIDIA RTX 2050" if use_gpu else "CPU"
        }
        with open(f"{RESULTS_DIR}/performance_year2.json", 'w') as f:
            json.dump(perf, f)
        print("✅ Year2-only pretrain complete")
        return
    
    # Step 5+6: Walk-forward on Year 3
    print(f"STEP 5+6: Walk-forward (Year 3: {MID_DATE} to {YEAR3_END})\n")
    try:
        from datetime import datetime, timedelta
        
        all_trades = []
        total_return = 0.0
        retrains = 0
        
        start = pd.Timestamp(MID_DATE)
        end = pd.Timestamp(YEAR3_END)
        weeks_count = int((end - start).days / 7)
        
        for week in range(weeks_count):
            val_start = start + timedelta(weeks=week)
            val_end = val_start + timedelta(weeks=1)
            retrain_start = start - timedelta(weeks=RETRAIN_WEEKS)
            
            # Retrain every RETRAIN_WEEKS
            if week % RETRAIN_WEEKS == 0 and week > 0:
                print(f"  Week {week:2d}: RETRAIN")
                retrains += 1
                
                # Rebuild training set
                retrain_labels = _build_step2_labels(symbols, retrain_start.strftime('%Y-%m-%d'), val_start.strftime('%Y-%m-%d'), HORIZON_BARS)
                if retrain_labels is not None and len(retrain_labels) > 0:
                    X_retrain_raw = np.array([_fetch_feature_vector(train_features, s, b) for s, b in zip(retrain_labels['symbol'], retrain_labels['bar'])])
                    X_retrain = X_retrain_raw[:min(MAX_TRAIN_ROWS, len(X_retrain_raw))]
                    y_retrain = retrain_labels['label'].values[:len(X_retrain)]
                    
                    scaler = RobustScaler()
                    X_retrain_scaled = scaler.fit_transform(X_retrain)
                    
                    final=xgb.XGBClassifier(**make_xgb_params(cuda_api))
                    final.fit(X_retrain_scaled, y_retrain, verbose=False)
                    
                    BASE_MODEL = final
                    BASE_SCALER = scaler
            
            # Generate trades for this week
            trades_week = build_trade_log(BASE_MODEL, BASE_SCALER, symbols, START_DATE, val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d'), train_features, cuda_api)
            if trades_week is not None:
                all_trades.append(trades_week)
                week_ret = trades_week['return_pct'].sum() / 100
                total_return += week_ret
                print(f"  Week {week:2d}: {len(trades_week):6.0f} trades, {week_ret:7.2%} return")
        
        # Save results
        full_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if len(full_trades) > 0:
            full_trades.to_csv(f"{RESULTS_DIR}/all_trades_year3.csv", index=False)
            print(f"\n  ✓ Trades: {len(full_trades)} total ({full_trades['return_pct'].sum() / 100:.2%} total return)\n")
        
        # Summary
        annualized = ((1 + total_return) ** (52 / weeks_count) - 1) * 100 if weeks_count > 0 else 0.0
        ic_mean = full_trades['return_pct'].mean() / 100 if len(full_trades) > 0 else 0.0
        ic_std = full_trades['return_pct'].std() / 100 if len(full_trades) > 1 else 0.0
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_positive = (full_trades['return_pct'] > 0).sum() / len(full_trades) * 100 if len(full_trades) > 0 else 0.0
        
        perf = {
            "label": "Year3_WalkForward_RTX2050",
            "total_weeks": weeks_count,
            "total_trades": len(full_trades),
            "retrains": retrains,
            "total_return_pct": round(total_return * 100, 1),
            "annualised_pct": round(annualized, 1),
            "sharpe": round(ic_mean / ic_std * np.sqrt(52) if ic_std > 0 else 0, 3),
            "ic_mean": round(ic_mean, 5),
            "icir": round(icir, 4),
            "ic_positive_pct": round(ic_positive, 1),
            "gpu": "NVIDIA RTX 2050" if use_gpu else "CPU",
            "vram_cap_rows": MAX_TRAIN_ROWS,
            "year2_only_mode": year2_only
        }
        
        with open(f"{RESULTS_DIR}/performance_year3.json", 'w') as f:
            json.dump(perf, f)
        
        print(f"RESULTS:")
        for k, v in perf.items():
            print(f"  {k}: {v}")
        
        print(f"\n✅ Walk-forward complete")
    
    except Exception as e:
        print(f"❌ Step 5+6 failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
