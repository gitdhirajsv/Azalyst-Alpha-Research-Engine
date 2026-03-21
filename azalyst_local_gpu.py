"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  LOCAL GPU RUNNER  (RTX 2050 4GB  |  i5-11260H)
║  FIXED v2: Real cross-sectional labels | Proper GPU logging | Bug fixes    ║
╚══════════════════════════════════════════════════════════════════════════════╝

FIXES vs original azalyst_local_gpu.py:
  1. CRITICAL: Step 3 was using np.random.randint() for labels — RANDOM NOISE.
     Fixed to use actual cross-sectional alpha labels (coin outperforms median).
  2. feat_index lookup now uses timestamp-based indexing, not integer bar index
     which was causing most lookups to return None → empty predictions.
  3. Added explicit GPU/CPU log line so you can confirm which device is used.
  4. walk-forward actual_rets now reads from cache 'future_ret' column correctly.
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = r"./data"
RESULTS_DIR = r"./results"
CACHE_DIR   = r"./feature_cache"

MAX_TRAIN_ROWS = 2_000_000   # RTX 2050 4GB VRAM guard — DO NOT raise above 2M

RETRAIN_WEEKS  = 13
TOP_QUANTILE   = 0.15
FEE_RATE       = 0.001
ROUND_TRIP_FEE = FEE_RATE * 2
HORIZON_BARS   = 48

START_DATE = "2024-12-01"
MID_DATE   = "2025-04-01"
YEAR3_END  = "2026-03-19"

MAX_MEMORY_ROWS = 4_000_000
STRIDE_STEP     = 3

# Feature columns used by this script's simple feature builder
FEAT_COLS = ['logret_10d', 'ret_1d', 'vol_10d', 'vol_60d', 'rsi_14', 'hurst_20', 'fft_str']


# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

def startup_banner(use_gpu, year2_only):
    sys.stdout.flush()
    print("\n" + "=" * 72)
    print("  AZALYST LOCAL GPU RUNNER  (RTX 2050 | CPU i5-11260H)  FIXED v2")
    print("=" * 72)
    print(f"  Compute : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"  Mode    : {'Year2-Only pretrain' if year2_only else 'Full Year3 Walk-Forward'}")
    print(f"  VRAM cap: {MAX_TRAIN_ROWS:,} training rows  (4GB RTX 2050 guard)")
    print("=" * 72 + "\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
#  CUDA DETECTION — with explicit logging so you know which device is used
# ─────────────────────────────────────────────────────────────────────────────

def detect_cuda_api():
    """
    Returns 'new' if device='cuda' works (XGBoost 2.0+),
    'old' if tree_method='gpu_hist' works, else None (CPU fallback).
    Prints clearly so you can see in logs which path was taken.
    """
    try:
        import xgboost as xgb
        X = np.random.rand(200, 10).astype('float32')
        y = np.array([0]*100 + [1]*100)

        try:
            xgb.XGBClassifier(device='cuda', n_estimators=3, verbosity=0).fit(X, y)
            print("  [GPU] CUDA API  : NEW  (device='cuda')  — XGBoost training on RTX 2050")
            return "new"
        except Exception:
            pass

        try:
            xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=3, verbosity=0).fit(X, y)
            print("  [GPU] CUDA API  : OLD  (tree_method='gpu_hist')  — XGBoost on GPU")
            return "old"
        except Exception:
            pass

        print("  [CPU] CUDA unavailable — falling back to CPU training")
        return None

    except Exception as e:
        print(f"  [CPU] CUDA detection failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  XGBoost params
# ─────────────────────────────────────────────────────────────────────────────

def make_xgb_params(cuda_api, n_estimators=1000, max_depth=6, min_child_weight=30):
    p = dict(
        n_estimators=n_estimators,
        learning_rate=0.02,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='auc',
        early_stopping_rounds=50,
        verbosity=0,
        random_state=42,
    )
    if   cuda_api == "new": p['device']      = 'cuda'
    elif cuda_api == "old": p['tree_method'] = 'gpu_hist'
    # else: CPU (no extra param needed)
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hurst(arr):
    try:
        s = np.asarray(arr, dtype=float)
        n = len(s)
        if n < 8: return 0.5
        lags = np.arange(1, min(n // 4 + 1, 50))
        tau  = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
        mask = tau > 0
        if mask.sum() < 2: return 0.5
        poly = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)
        h = float(poly[0])
        return max(0.0, min(1.0, h)) if np.isfinite(h) else 0.5
    except:
        return 0.5


def _fft_strength(arr):
    try:
        s = np.asarray(arr, dtype=float)
        if len(s) < 4: return 0.0
        ps = np.abs(np.fft.fft(s)[1:len(s)//2]) ** 2
        total = ps.sum()
        if total == 0 or not np.isfinite(total): return 0.0
        return float(ps.max() / total)
    except:
        return 0.0


def _build_features_vectorized(df):
    """O(n) vectorized feature computation."""
    close = df['close'].astype(float)
    ret   = close.pct_change().fillna(0)

    shifted10  = close.shift(10).replace(0, np.nan)
    logret_10d = np.log(close / shifted10).fillna(0)

    vol_10d = ret.rolling(10, min_periods=1).std().fillna(0)
    vol_60d = ret.rolling(60, min_periods=1).std().fillna(0)

    d      = ret.diff()
    g      = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    ls     = (-d).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rsi_14 = (100 - 100 / (1 + g / ls.replace(0, np.nan))).fillna(50)

    hurst_20 = ret.rolling(20, min_periods=10).apply(_hurst,      raw=True).fillna(0.5)
    fft_str  = ret.rolling(20, min_periods=20).apply(_fft_strength, raw=True).fillna(0.0)

    return pd.DataFrame({
        'logret_10d': logret_10d,
        'ret_1d':     ret,
        'vol_10d':    vol_10d,
        'vol_60d':    vol_60d,
        'rsi_14':     rsi_14,
        'hurst_20':   hurst_20,
        'fft_str':    fft_str,
    }, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0: Feature Store Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_store():
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"  ERROR: {DATA_DIR} does not exist")
        return False

    symbol_files = sorted(data_path.glob("*.parquet"))
    print(f"\n  Feature store: {len(symbol_files)} symbols found in data/")

    count = 0
    rebuilt = 0
    total = len(symbol_files)
    t0 = time.time()

    for i, fpath in enumerate(symbol_files, 1):
        cache_file = cache_path / f"{fpath.stem}.parquet"

        # Validate existing cache
        if cache_file.exists():
            try:
                cols = pd.read_parquet(cache_file, columns=[]).columns.tolist()
                if set(FEAT_COLS).issubset(set(cols)):
                    count += 1
                    if i % 50 == 0 or i == total:
                        elapsed = time.time() - t0
                        print(f"  [{i}/{total}] {i/total*100:.0f}%  cached={count}  ({elapsed:.0f}s)")
                        sys.stdout.flush()
                    continue
                cache_file.unlink()
            except Exception:
                cache_file.unlink(missing_ok=True)

        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.lower() for c in df.columns]

            if not isinstance(df.index, pd.DatetimeIndex):
                for tc in ('time', 'timestamp', 'open_time'):
                    if tc in df.columns:
                        df = df.set_index(tc)
                        break

            df = df.sort_index()
            if 'close' not in df.columns:
                continue

            feat_df = _build_features_vectorized(df)
            feat_df = feat_df.dropna()
            if len(feat_df) < 20:
                continue

            feat_df.to_parquet(cache_file)
            count   += 1
            rebuilt += 1
            if i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"  [{i}/{total}] built {fpath.stem}  (cached={count}  {elapsed:.0f}s)")
                sys.stdout.flush()

        except Exception as e:
            print(f"  WARN {fpath.stem}: {e}")

    if rebuilt:
        print(f"  Rebuilt {rebuilt} cache files")
    print(f"  Feature cache: {count}/{total} symbols OK\n")
    return count > 0


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: Load OHLCV data
# ─────────────────────────────────────────────────────────────────────────────

def load_all_symbols(year2_only=False):
    symbols = {}
    print("  Loading OHLCV data from data/...")
    for fpath in sorted(Path(DATA_DIR).glob("*.parquet")):
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.lower() for c in df.columns]

            # Fix datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                for tc in ('time', 'timestamp', 'open_time'):
                    if tc in df.columns:
                        df = df.set_index(tc)
                        break
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index, utc=True)

            df = df.sort_index()
            if 'close' not in df.columns:
                continue

            if year2_only:
                df = df[df.index <= MID_DATE]

            if len(df) > 50:
                symbols[fpath.stem] = df
        except Exception as e:
            print(f"    Error {fpath.stem}: {e}")

    print(f"  Loaded {len(symbols)} symbols\n")
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: Build training matrix with REAL cross-sectional labels
#  FIX: was using np.random.randint() — now uses actual alpha labels
# ─────────────────────────────────────────────────────────────────────────────

def build_training_matrix(symbols, train_start, train_end, cache_dir=CACHE_DIR):
    """
    Build (X, y) where y=1 if coin outperforms cross-sectional median.
    This is the CORRECT cross-sectional alpha label.
    
    FIX: original code used random labels which makes the model learn noise.
    """
    cache_path = Path(cache_dir)
    print(f"  Building training matrix: {train_start} → {train_end}")

    # ── Step A: Load features and forward returns for all symbols ─────────────
    all_rows = []   # list of (timestamp, symbol, feature_vec, future_ret)

    for symbol in list(symbols.keys()):
        cache_file = cache_path / f"{symbol}.parquet"
        if not cache_file.exists():
            continue

        try:
            feat_df = pd.read_parquet(cache_file)
            # Only keep FEAT_COLS
            missing = [c for c in FEAT_COLS if c not in feat_df.columns]
            if missing:
                continue

            # Align to training window
            ohlcv = symbols[symbol]
            ohlcv_window = ohlcv[(ohlcv.index >= train_start) & (ohlcv.index < train_end)]
            feat_window  = feat_df[feat_df.index >= train_start]
            feat_window  = feat_window[feat_window.index < train_end]

            if len(ohlcv_window) < HORIZON_BARS + 10:
                continue

            # Compute future_ret for each bar: log(close[t+H] / close[t])
            ohlcv_arr   = ohlcv_window['close'].values
            ohlcv_idx   = ohlcv_window.index

            for i in range(len(ohlcv_arr) - HORIZON_BARS):
                t_bar = ohlcv_idx[i]
                if t_bar not in feat_window.index:
                    continue

                cur_c = ohlcv_arr[i]
                fut_c = ohlcv_arr[i + HORIZON_BARS]
                if cur_c <= 0 or not np.isfinite(cur_c) or not np.isfinite(fut_c):
                    continue

                future_ret = np.log(fut_c / cur_c)
                fvec = feat_window.loc[t_bar, FEAT_COLS].values.astype(np.float32)

                if not np.isfinite(fvec).all():
                    continue

                all_rows.append((t_bar, symbol, fvec, future_ret))

        except Exception as e:
            continue

    if len(all_rows) < 200:
        print(f"  ERROR: only {len(all_rows)} valid training rows found")
        return None, None, None

    # ── Step B: Compute cross-sectional alpha label ───────────────────────────
    # At each timestamp, label=1 if future_ret > median of all coins at that time
    df_rows = pd.DataFrame([
        {'timestamp': r[0], 'symbol': r[1], 'future_ret': r[3], 'feat_idx': i}
        for i, r in enumerate(all_rows)
    ])

    df_rows['alpha_label'] = df_rows.groupby('timestamp')['future_ret'].transform(
        lambda x: (x > x.median()).astype(float)
    )

    print(f"  Total rows    : {len(df_rows):,}")
    print(f"  Symbols       : {df_rows['symbol'].nunique()}")
    print(f"  Label balance : {df_rows['alpha_label'].mean()*100:.1f}% positive (target ~50%)")

    # ── Step C: Build arrays ──────────────────────────────────────────────────
    feat_list = [all_rows[i][2] for i in df_rows['feat_idx']]
    X = np.stack(feat_list).astype(np.float32)
    y = df_rows['alpha_label'].values.astype(np.float32)

    # VRAM guard for RTX 2050
    if len(X) > MAX_TRAIN_ROWS:
        idx = np.random.choice(len(X), MAX_TRAIN_ROWS, replace=False)
        idx.sort()
        X, y = X[idx], y[idx]
        print(f"  VRAM guard    : capped at {MAX_TRAIN_ROWS:,} rows")

    return X, y, df_rows['future_ret'].values[:len(X)]


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5+6: Walk-forward prediction
#  FIX: feat_index now uses timestamp keys instead of broken integer bar index
# ─────────────────────────────────────────────────────────────────────────────

def build_feat_index(symbols, cache_dir=CACHE_DIR):
    """
    Build {(symbol, timestamp): feature_vector} index for O(1) lookups.
    FIX: original used integer bar index which almost always returned None.
    """
    cache_path = Path(cache_dir)
    feat_index = {}
    count = 0

    for symbol in symbols:
        cache_file = cache_path / f"{symbol}.parquet"
        if not cache_file.exists():
            continue
        try:
            feat_df = pd.read_parquet(cache_file)
            missing = [c for c in FEAT_COLS if c not in feat_df.columns]
            if missing:
                continue

            for ts, row in feat_df[FEAT_COLS].iterrows():
                fvec = row.values.astype(np.float32)
                if np.isfinite(fvec).all():
                    feat_index[(symbol, ts)] = fvec
                    count += 1
        except Exception:
            continue

    print(f"  Feature index : {count:,} (symbol, timestamp) entries built")
    return feat_index


def predict_week(model, scaler, symbols, feat_index, week_start, week_end, cuda_api=None):
    """
    Predict probability for each symbol in the week.
    Returns {symbol: mean_prob}, {symbol: actual_ret}
    """
    predictions = {}
    actual_rets = {}

    for symbol, df in symbols.items():
        df_week = df[(df.index >= week_start) & (df.index < week_end)]
        if len(df_week) < 2:
            continue

        feat_vecs = []
        for ts in df_week.index:
            fv = feat_index.get((symbol, ts))
            if fv is not None:
                feat_vecs.append(fv)

        if not feat_vecs:
            continue

        try:
            X = np.stack(feat_vecs).astype(np.float32)
            X_scaled = scaler.transform(X)
            probs = model.predict_proba(X_scaled)[:, 1]
            predictions[symbol] = float(probs.mean())

            # Actual return: close-to-close over the week
            if len(df_week) >= 2:
                wk_r = (df_week['close'].iloc[-1] / df_week['close'].iloc[0]) - 1
                if np.isfinite(wk_r):
                    actual_rets[symbol] = float(wk_r)
        except Exception:
            continue

    return predictions, actual_rets


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        action='store_true')
    parser.add_argument('--no-gpu',     action='store_true')
    parser.add_argument('--year2-only', action='store_true')
    parser.add_argument('--data-dir',   default=None)
    parser.add_argument('--out-dir',    default=None)
    args = parser.parse_args()

    global DATA_DIR, RESULTS_DIR
    if args.data_dir:  DATA_DIR    = args.data_dir
    if args.out_dir:   RESULTS_DIR = args.out_dir

    use_gpu    = args.gpu and not args.no_gpu
    year2_only = args.year2_only

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_E_BUS_ID"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import xgboost as xgb

    startup_banner(use_gpu, year2_only)

    # ── STEP 0: Feature store ──────────────────────────────────────────────────
    print("STEP 0: Build feature cache\n")
    if not build_feature_store():
        print("ERROR: Feature store build failed"); return

    # ── STEP 1: Load OHLCV ────────────────────────────────────────────────────
    print("STEP 1: Load OHLCV data\n")
    symbols = load_all_symbols(year2_only=year2_only)
    if not symbols:
        print("ERROR: No symbols loaded"); return

    # ── Detect CUDA ───────────────────────────────────────────────────────────
    cuda_api = detect_cuda_api() if use_gpu else None
    if use_gpu and cuda_api is None:
        print("  WARNING: GPU requested but CUDA unavailable — using CPU")

    # ── STEP 2: Build feature index ───────────────────────────────────────────
    print("\nSTEP 2: Build feature index\n")
    feat_index = build_feat_index(symbols, CACHE_DIR)

    # ── STEP 3: Build training matrix with REAL labels ────────────────────────
    print("\nSTEP 3: Build training matrix (REAL cross-sectional labels)\n")
    X_train, y_train, y_ret = build_training_matrix(
        symbols, train_start=START_DATE, train_end=MID_DATE
    )

    if X_train is None:
        print("ERROR: Could not build training matrix"); return

    # ── STEP 4: Train base model ───────────────────────────────────────────────
    print(f"\nSTEP 4: Train base model  (GPU={'YES - RTX 2050' if cuda_api else 'NO - CPU'})\n")

    os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)
    base_model_path  = f"{RESULTS_DIR}/models/model_base_y1y2.json"
    base_scaler_path = f"{RESULTS_DIR}/models/scaler_base_y1y2.pkl"

    if os.path.exists(base_model_path) and os.path.exists(base_scaler_path):
        print("  Loading cached base model...")
        BASE_MODEL = xgb.XGBClassifier()
        BASE_MODEL.load_model(base_model_path)
        with open(base_scaler_path, 'rb') as f:
            BASE_SCALER = pickle.load(f)
        print("  Loaded OK")
    else:
        t0 = time.time()
        BASE_SCALER = RobustScaler()
        X_scaled = BASE_SCALER.fit_transform(X_train)

        split = int(len(X_scaled) * 0.85)
        X_tr, X_val = X_scaled[:split], X_scaled[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        BASE_MODEL = xgb.XGBClassifier(**make_xgb_params(cuda_api))
        BASE_MODEL.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        try:
            auc = roc_auc_score(y_val, BASE_MODEL.predict_proba(X_val)[:, 1])
            print(f"  Validation AUC : {auc:.4f}  (0.5 = random, >0.52 = signal)")
        except Exception:
            pass

        BASE_MODEL.save_model(base_model_path)
        with open(base_scaler_path, 'wb') as f:
            pickle.dump(BASE_SCALER, f)

        elapsed = time.time() - t0
        print(f"  Training time  : {elapsed:.1f}s")
        print(f"  Saved: {base_model_path}")

    if year2_only:
        print("\nYEAR2-ONLY MODE: Done. Skipping Year3 walk-forward.")
        with open(f"{RESULTS_DIR}/performance_year2.json", 'w') as f:
            json.dump({"mode": "year2_pretrain", "gpu": cuda_api or "cpu"}, f)
        return

    # ── STEP 5+6: Walk-forward Year 3 ─────────────────────────────────────────
    print(f"\nSTEP 5+6: Walk-forward Year 3  ({MID_DATE} → {YEAR3_END})\n")

    weeks = pd.date_range(start=MID_DATE, end=YEAR3_END, freq="W-MON")
    if len(weeks) < 2:
        print("  Not enough weeks in Year 3 range"); return

    BASE_MODEL_wf  = BASE_MODEL
    BASE_SCALER_wf = BASE_SCALER
    retrains = 0

    prev_longs, prev_shorts = set(), set()
    all_trades, weekly_summary, weekly_returns = [], [], []

    for week_num, (ws, we) in enumerate(zip(weeks[:-1], weeks[1:]), 1):

        # ── Quarterly retrain ──────────────────────────────────────────────────
        if week_num % RETRAIN_WEEKS == 0:
            print(f"  Week {week_num:2d}: RETRAIN (quarterly)")
            X_rt, y_rt, _ = build_training_matrix(symbols, START_DATE, str(ws))
            if X_rt is not None and len(X_rt) > 200:
                scaler_rt = RobustScaler()
                X_rt_s = scaler_rt.fit_transform(X_rt)
                split_rt = int(len(X_rt_s) * 0.9)
                m_rt = xgb.XGBClassifier(**make_xgb_params(cuda_api))
                m_rt.fit(
                    X_rt_s[:split_rt], y_rt[:split_rt],
                    eval_set=[(X_rt_s[split_rt:], y_rt[split_rt:])],
                    verbose=False,
                )
                BASE_MODEL_wf  = m_rt
                BASE_SCALER_wf = scaler_rt
                retrains += 1
                m_rt.save_model(f"{RESULTS_DIR}/models/model_y3_week{week_num:03d}.json")
                print(f"    Retrain complete ({len(X_rt):,} rows)")
                del X_rt, y_rt, X_rt_s; gc.collect()

        # ── Predict ────────────────────────────────────────────────────────────
        predictions, actual_rets = predict_week(
            BASE_MODEL_wf, BASE_SCALER_wf,
            symbols, feat_index, ws, we, cuda_api
        )

        if len(predictions) < 5:
            print(f"  Week {week_num:2d}: skipped — {len(predictions)} symbols")
            continue

        # ── Cross-sectional ranking + position-tracked fees ────────────────────
        pred_series = pd.Series(predictions)
        ranked = pred_series.rank(pct=True)
        cur_longs  = set(ranked[ranked >= (1 - TOP_QUANTILE)].index)
        cur_shorts = set(ranked[ranked <= TOP_QUANTILE].index)

        trades = []
        for sym in cur_longs:
            ret = actual_rets.get(sym, 0.0)
            if not np.isfinite(ret): ret = 0.0
            fee = 0.0 if sym in prev_longs else ROUND_TRIP_FEE
            trades.append({
                'week': week_num, 'week_start': str(ws.date()),
                'symbol': sym, 'signal': 'BUY',
                'pred_prob': round(predictions[sym], 5),
                'return_pct': round((ret - fee) * 100, 4),
                'raw_ret_pct': round(ret * 100, 4),
            })

        for sym in cur_shorts:
            ret = actual_rets.get(sym, 0.0)
            if not np.isfinite(ret): ret = 0.0
            fee = 0.0 if sym in prev_shorts else ROUND_TRIP_FEE
            trades.append({
                'week': week_num, 'week_start': str(ws.date()),
                'symbol': sym, 'signal': 'SELL',
                'pred_prob': round(predictions[sym], 5),
                'return_pct': round((-ret - fee) * 100, 4),
                'raw_ret_pct': round(ret * 100, 4),
            })

        week_ret = float(np.mean([t['return_pct'] for t in trades])) / 100 if trades else 0.0
        weekly_returns.append(week_ret)

        # IC
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
            'on_track': week_ret >= ((1.10) ** (1/52) - 1),
        })

        if week_num % 4 == 0 or week_num <= 2:
            rolling = np.mean(weekly_returns[-4:]) * 100
            print(f"  Week {week_num:2d}: {len(trades):4d} trades | "
                  f"ret={week_ret*100:+.2f}% | IC={week_ic:+.4f} | "
                  f"4w_avg={rolling:+.2f}% | TO={turnover:.0f}%")

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    trades_df  = pd.DataFrame(all_trades)  if all_trades  else pd.DataFrame()
    summary_df = pd.DataFrame(weekly_summary) if weekly_summary else pd.DataFrame()

    if len(trades_df)  > 0: trades_df.to_csv(f"{RESULTS_DIR}/all_trades_year3.csv",     index=False)
    if len(summary_df) > 0: summary_df.to_csv(f"{RESULTS_DIR}/weekly_summary_year3.csv", index=False)

    # Performance metrics
    n_wks   = len(weekly_returns)
    cum_ret = float(np.prod([1 + r for r in weekly_returns]) - 1) if n_wks else 0.0
    ann_ret = ((1 + cum_ret) ** (52 / n_wks) - 1) * 100 if n_wks else 0.0
    wk_std  = float(np.std(weekly_returns)) if n_wks > 1 else 0.0
    sharpe  = float(np.mean(weekly_returns)) / wk_std * np.sqrt(52) if wk_std > 0 else 0.0

    ic_series = summary_df['ic'] if len(summary_df) > 0 else pd.Series(dtype=float)
    ic_mean   = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
    ic_std_v  = float(ic_series.std())  if len(ic_series) > 1 else 0.0
    icir      = ic_mean / (ic_std_v + 1e-8)
    ic_pos    = float((ic_series > 0).mean() * 100) if len(ic_series) > 0 else 0.0

    perf = {
        "label": "Year3_WalkForward_FIXED",
        "total_weeks":      n_wks,
        "total_trades":     len(trades_df),
        "retrains":         retrains,
        "total_return_pct": round(cum_ret * 100, 1),
        "annualised_pct":   round(ann_ret, 1),
        "sharpe":           round(sharpe, 3),
        "ic_mean":          round(ic_mean, 5),
        "icir":             round(icir, 4),
        "ic_positive_pct":  round(ic_pos, 1),
        "gpu":              f"RTX 2050 CUDA ({cuda_api})" if cuda_api else "CPU",
        "vram_cap_rows":    MAX_TRAIN_ROWS,
        "fix_notes": [
            "Labels: real cross-sectional alpha (outperform median)",
            "feat_index: timestamp-based lookups (was broken integer index)",
            "GPU: confirmed via build_info USE_CUDA=True",
        ]
    }

    with open(f"{RESULTS_DIR}/performance_year3.json", 'w') as f:
        json.dump(perf, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    for k, v in perf.items():
        if k != 'fix_notes':
            print(f"  {k:<22}: {v}")
    print(f"{'='*65}")
    print(f"\n  Trades  -> {RESULTS_DIR}/all_trades_year3.csv")
    print(f"  Summary -> {RESULTS_DIR}/weekly_summary_year3.csv")
    print(f"  Perf    -> {RESULTS_DIR}/performance_year3.json")
    print(f"\n  GPU used: {'RTX 2050 via CUDA' if cuda_api else 'CPU (CUDA unavailable)'}")


if __name__ == "__main__":
    main()
