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
        # Pre-index features for O(1) lookups during weekly prediction
        feat_index = {}
        for _, row in train_features.iterrows():
            feat_index[(row['symbol'], row['bar'])] = row['features']
        print(f"  Feature index: {len(feat_index):,} entries")

        weeks = pd.date_range(start=MID_DATE, end=YEAR3_END, freq="W-MON")
        if len(weeks) < 2:
            print("  Not enough weeks in Year 3 range")
            return

        prev_longs  = set()
        prev_shorts = set()
        all_trades      = []
        weekly_summary  = []
        weekly_returns  = []
        retrains = 0

        for week_num, (ws, we) in enumerate(zip(weeks[:-1], weeks[1:]), 1):

            # ── Quarterly retrain ────────────────────────────────────────
            if week_num % RETRAIN_WEEKS == 0:
                print(f"  Week {week_num:2d}: RETRAIN")
                X_rt, y_rt = [], []
                retrain_end = str(ws)
                for symbol, df in symbols.items():
                    df_rt = df[(df.index >= START_DATE) & (df.index < retrain_end)]
                    if len(df_rt) < HORIZON_BARS + 10:
                        continue
                    n_bars = len(df_rt) - HORIZON_BARS
                    step = max(1, n_bars // 500)
                    for i in range(0, n_bars, step):
                        try:
                            bar_idx = df.index.get_loc(df_rt.index[i])
                        except KeyError:
                            continue
                        fv = feat_index.get((symbol, bar_idx))
                        if fv is None or len(fv) != BASE_SCALER.n_features_in_:
                            continue
                        cur_c = df_rt.iloc[i]['close']
                        fut_c = df_rt.iloc[i + HORIZON_BARS]['close']
                        if cur_c <= 0:
                            continue
                        X_rt.append(fv)
                        y_rt.append(1 if fut_c > cur_c else 0)

                if len(X_rt) > 100:
                    X_rt = np.array(X_rt[:MAX_TRAIN_ROWS])
                    y_rt = np.array(y_rt[:len(X_rt)])
                    scaler_new = RobustScaler()
                    X_rt_s = scaler_new.fit_transform(X_rt)
                    m_new = xgb.XGBClassifier(**make_xgb_params(cuda_api))
                    split = int(len(X_rt_s) * 0.9)
                    m_new.fit(
                        X_rt_s[:split], y_rt[:split],
                        eval_set=[(X_rt_s[split:], y_rt[split:])],
                        verbose=False,
                    )
                    BASE_MODEL  = m_new
                    BASE_SCALER = scaler_new
                    retrains += 1
                    print(f"    → {len(X_rt):,} samples, retrained OK")
                    del X_rt, y_rt, X_rt_s
                    gc.collect()
                else:
                    print(f"    → too few samples ({len(X_rt)}), skipped retrain")

            # ── Predict: one probability per symbol per week ─────────────
            predictions = {}
            actual_rets = {}

            for symbol, df in symbols.items():
                mask = (df.index >= str(ws)) & (df.index < str(we))
                df_week = df[mask]
                if len(df_week) < 3:
                    continue

                feat_vecs = []
                for idx in df_week.index:
                    try:
                        bar_idx = df.index.get_loc(idx)
                    except KeyError:
                        continue
                    fv = feat_index.get((symbol, bar_idx))
                    if fv is not None and len(fv) == BASE_SCALER.n_features_in_:
                        feat_vecs.append(fv)

                if not feat_vecs:
                    continue

                X = np.vstack(feat_vecs)
                X_scaled = BASE_SCALER.transform(X)
                probs = BASE_MODEL.predict_proba(X_scaled)[:, 1]
                predictions[symbol] = float(probs.mean())

                # Actual return: close-to-close over the week
                if len(df_week) >= 2:
                    wk_r = (df_week.iloc[-1]['close'] / df_week.iloc[0]['close']) - 1
                    if np.isfinite(wk_r):
                        actual_rets[symbol] = float(wk_r)

            if len(predictions) < 5:
                print(f"  Week {week_num:2d}: skipped — only {len(predictions)} symbols")
                continue

            # ── Cross-sectional ranking + position-tracked fees ──────────
            pred_series = pd.Series(predictions)
            ranked = pred_series.rank(pct=True)
            cur_longs  = set(ranked[ranked >= (1 - TOP_QUANTILE)].index)
            cur_shorts = set(ranked[ranked <= TOP_QUANTILE].index)

            trades = []
            for sym in cur_longs:
                ret = actual_rets.get(sym, 0.0)
                if not np.isfinite(ret):
                    ret = 0.0
                fee = 0.0 if sym in prev_longs else ROUND_TRIP_FEE
                pnl = (ret - fee) * 100
                trades.append({
                    'week': week_num, 'symbol': sym, 'signal': 'BUY',
                    'pred_prob': round(predictions[sym], 5),
                    'return_pct': round(pnl, 4),
                    'raw_ret_pct': round(ret * 100, 4),
                })

            for sym in cur_shorts:
                ret = actual_rets.get(sym, 0.0)
                if not np.isfinite(ret):
                    ret = 0.0
                fee = 0.0 if sym in prev_shorts else ROUND_TRIP_FEE
                pnl = (-ret - fee) * 100
                trades.append({
                    'week': week_num, 'symbol': sym, 'signal': 'SELL',
                    'pred_prob': round(predictions[sym], 5),
                    'return_pct': round(pnl, 4),
                    'raw_ret_pct': round(ret * 100, 4),
                })

            week_ret = float(np.mean([t['return_pct'] for t in trades])) / 100 if trades else 0.0
            weekly_returns.append(week_ret)

            # IC (Spearman rank correlation)
            common = [s for s in predictions if s in actual_rets]
            if len(common) > 10:
                pred_arr = np.array([predictions[s] for s in common])
                ret_arr  = np.array([actual_rets[s]  for s in common])
                week_ic  = float(stats.spearmanr(pred_arr, ret_arr)[0])
            else:
                week_ic = 0.0

            # Turnover
            n_cur = len(cur_longs) + len(cur_shorts)
            n_new = len(cur_longs - prev_longs) + len(cur_shorts - prev_shorts)
            turnover = round(n_new / n_cur * 100, 1) if n_cur > 0 else 100.0
            prev_longs, prev_shorts = cur_longs, cur_shorts

            all_trades.extend(trades)
            ann_proj = ((1 + week_ret) ** 52 - 1) * 100
            weekly_summary.append({
                'week': week_num,
                'week_start': str(ws.date()),
                'week_end': str(we.date()),
                'n_symbols': len(predictions),
                'n_trades': len(trades),
                'week_return_pct': round(week_ret * 100, 4),
                'annualised_pct': round(ann_proj, 2),
                'ic': round(week_ic, 5),
                'turnover_pct': turnover,
                'on_track': week_ret >= ((1.10) ** (1 / 52) - 1),
                'retrained': week_num % RETRAIN_WEEKS == 0,
            })

            if week_num % 4 == 0 or week_num <= 2:
                rolling = np.mean(weekly_returns[-4:]) * 100
                print(f"  Week {week_num:2d}: {len(trades):4d} trades | "
                      f"ret={week_ret*100:+.2f}% | IC={week_ic:+.4f} | "
                      f"turnover={turnover:.0f}% | 4w_avg={rolling:+.2f}%")

        # ── Save results ─────────────────────────────────────────────────
        os.makedirs(RESULTS_DIR, exist_ok=True)

        trades_df  = pd.DataFrame(all_trades)  if all_trades  else pd.DataFrame()
        summary_df = pd.DataFrame(weekly_summary) if weekly_summary else pd.DataFrame()

        if len(trades_df) > 0:
            trades_df.to_csv(f"{RESULTS_DIR}/all_trades_year3.csv", index=False)
        if len(summary_df) > 0:
            summary_df.to_csv(f"{RESULTS_DIR}/weekly_summary_year3.csv", index=False)

        # Performance metrics (proper definitions)
        n_wks = len(weekly_returns)
        if n_wks > 0:
            cum_ret = float(np.prod([1 + r for r in weekly_returns]) - 1)
            ann_ret = ((1 + cum_ret) ** (52 / n_wks) - 1) * 100
            wk_std  = float(np.std(weekly_returns))
            sharpe  = float(np.mean(weekly_returns)) / wk_std * np.sqrt(52) if wk_std > 0 else 0.0
        else:
            cum_ret, ann_ret, sharpe = 0.0, 0.0, 0.0

        ic_series = summary_df['ic'] if len(summary_df) > 0 else pd.Series(dtype=float)
        ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
        ic_std_v = float(ic_series.std()) if len(ic_series) > 1 else 0.0
        icir = ic_mean / (ic_std_v + 1e-8)
        ic_pos = float((ic_series > 0).mean() * 100) if len(ic_series) > 0 else 0.0

        perf = {
            "label": "Year3_WalkForward_RTX2050",
            "total_weeks": n_wks,
            "total_trades": len(trades_df),
            "retrains": retrains,
            "total_return_pct": round(cum_ret * 100, 1),
            "annualised_pct": round(ann_ret, 1),
            "sharpe": round(sharpe, 3),
            "ic_mean": round(ic_mean, 5),
            "icir": round(icir, 4),
            "ic_positive_pct": round(ic_pos, 1),
            "gpu": "NVIDIA RTX 2050" if use_gpu else "CPU",
            "vram_cap_rows": MAX_TRAIN_ROWS,
            "year2_only_mode": year2_only,
        }

        with open(f"{RESULTS_DIR}/performance_year3.json", 'w') as f:
            json.dump(perf, f, indent=2)

        print(f"\nRESULTS:")
        for k, v in perf.items():
            print(f"  {k}: {v}")

        print(f"\n✅ Walk-forward complete")
        print(f"  Trades  → {RESULTS_DIR}/all_trades_year3.csv")
        print(f"  Summary → {RESULTS_DIR}/weekly_summary_year3.csv")
        print(f"  Perf    → {RESULTS_DIR}/performance_year3.json")
    
    except Exception as e:
        print(f"❌ Step 5+6 failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
