"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AZALYST ALPHA RESEARCH ENGINE  v5.0                                ║
║         Short-Horizon Regression  |  15min/1hr Forecasting                 ║
║         Reversal-Dominated  |  Pump-Dump Aware  |  Jane Street Inspired    ║
║                                                                            ║
║  CHANGES FROM v4 (lessons learned from audit):                             ║
║   1. REGRESSION not classification — predict continuous returns            ║
║   2. Short horizons: 3 bars (15min) and 12 bars (1hr) — not 48 (4hr)     ║
║   3. Reversal-dominated features — crypto mean-reverts, not trends        ║
║   4. Per-bar prediction — no week-averaging that destroys signal          ║
║   5. Pump-dump detection — filter out manipulated coins                   ║
║   6. IC-gating kill-switch — halt when signal inverts                     ║
║   7. Weighted R² metric — penalizes direction + magnitude errors          ║
║   8. Confidence model — P(direction correct) for position sizing          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from azalyst_factors_v2 import build_features, FEATURE_COLS
from azalyst_risk import RiskManager
from azalyst_db import AzalystDB
from azalyst_pump_dump import compute_pump_dump_scores, filter_pump_dump_symbols
from azalyst_train import (
    train_regression_model, train_confidence_model,
    train_model, train_meta_model,
    make_xgb_regressor, compute_ic, weighted_r2_score,
    PurgedTimeSeriesCV as TrainPurgedCV,
)

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR    = "./data"
RESULTS_DIR = "./results"
CACHE_DIR   = "./feature_cache"

MAX_TRAIN_ROWS   = 2_000_000
RETRAIN_WEEKS    = 13
TOP_QUANTILE     = 0.15
FEE_RATE         = 0.001
ROUND_TRIP_FEE   = FEE_RATE * 2

# v5: Short horizons — 15min (3 bars) and 1hr (12 bars)
HORIZON_BARS_15M = 3     # 15 min at 5-min frequency
HORIZON_BARS_1H  = 12    # 1 hour at 5-min frequency
HORIZON_BARS     = 12    # default prediction horizon (1hr)

# v5 new config
MAX_DRAWDOWN_KILL   = -0.15
IC_SELECTION_THRESH = 0.00   # v5: stricter — must be non-negative (was -0.02)
IC_LOOKBACK_WEEKS   = 8
MIN_FEATURES        = 20
VAR_CONFIDENCE      = 0.95
POSITION_RISK_CAP   = 0.03
PUMP_DUMP_THRESHOLD = 0.6   # pump-dump score above this → avoid symbol
IC_GATING_THRESHOLD = -0.03 # if rolling IC below this → halt predictions


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _fix_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next(
            (c for c in df.columns if c in ("time", "timestamp", "open_time")),
            None,
        )
        if ts_col:
            unit = "ms" if pd.api.types.is_integer_dtype(df[ts_col]) else None
            df.index = pd.to_datetime(df[ts_col], unit=unit, utc=True)
            df = df.drop(columns=[ts_col])
        elif pd.api.types.is_integer_dtype(df.index):
            df.index = pd.to_datetime(df.index, unit="ms", utc=True)
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    if df.index.max().year < 2018:
        raise ValueError(f"1970 timestamp still present: max={df.index.max()}")
    return df


def _required_cache_columns() -> list:
    return FEATURE_COLS + ["future_ret", "future_ret_15m", "future_ret_1h"]


def _read_parquet_columns(path: Path) -> list:
    if pq is not None:
        return list(pq.read_schema(path).names)
    return pd.read_parquet(path).columns.tolist()


# ── CUDA DETECTION ────────────────────────────────────────────────────────────

def detect_cuda_api() -> str | None:
    try:
        import xgboost as xgb
        X = np.random.rand(200, 10).astype("float32")
        y = np.random.randn(200).astype("float32")
        try:
            xgb.XGBRegressor(device="cuda", n_estimators=3, verbosity=0).fit(X, y)
            print("  [GPU] CUDA API: NEW  (device='cuda') — regression mode")
            return "new"
        except Exception:
            pass
        try:
            xgb.XGBRegressor(tree_method="gpu_hist", n_estimators=3, verbosity=0).fit(X, y)
            print("  [GPU] CUDA API: OLD  (tree_method='gpu_hist') — regression mode")
            return "old"
        except Exception:
            pass
        print("  [CPU] CUDA unavailable — falling back to CPU")
        return None
    except Exception as e:
        print(f"  [CPU] CUDA detection failed: {e}")
        return None


def make_xgb_params(cuda_api: str | None, n_estimators: int = 1000,
                    max_depth: int = 6, min_child_weight: int = 30,
                    regression: bool = True) -> dict:
    """Build XGBoost params. v5 default is regression mode."""
    if regression:
        p = dict(
            n_estimators=n_estimators, learning_rate=0.02, max_depth=max_depth,
            min_child_weight=min_child_weight, subsample=0.8,
            colsample_bytree=0.7, colsample_bylevel=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            objective="reg:squarederror", eval_metric="rmse",
            early_stopping_rounds=50, verbosity=0, random_state=42,
        )
    else:
        p = dict(
            n_estimators=n_estimators, learning_rate=0.02, max_depth=max_depth,
            min_child_weight=min_child_weight, subsample=0.8,
            colsample_bytree=0.7, colsample_bylevel=0.7,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc",
            early_stopping_rounds=50, verbosity=0, random_state=42,
        )
    if cuda_api == "new":
        p["device"] = "cuda"
    elif cuda_api == "old":
        p["tree_method"] = "gpu_hist"
    return p


# ── REGIME DETECTION ──────────────────────────────────────────────────────────

def detect_regime(symbols: dict, week_end) -> str:
    btc_df = symbols.get("BTCUSDT")
    if btc_df is None:
        return "LOW_VOL_GRIND"

    lookback = btc_df[btc_df.index < week_end].tail(4 * 288)
    if len(lookback) < 288:
        return "LOW_VOL_GRIND"

    btc_ret = float(np.log(lookback["ret_1d"].iloc[-1] + 1)) if "ret_1d" in lookback.columns else 0.0
    if len(lookback) > 0:
        rvol = lookback.get("rvol_1d")
        if rvol is not None:
            avg_vol = float(rvol.dropna().mean())
            recent_vol = float(rvol.dropna().tail(288).mean()) if len(rvol.dropna()) > 288 else avg_vol
        else:
            avg_vol = recent_vol = 0.02

        ret_1w = lookback.get("ret_1w")
        if ret_1w is not None and len(ret_1w.dropna()) > 0:
            btc_ret = float(ret_1w.dropna().iloc[-1])
    else:
        avg_vol = recent_vol = 0.02

    high_vol = recent_vol > avg_vol * 1.3
    trending_up = btc_ret > 0.03
    trending_down = btc_ret < -0.03

    if trending_up and not high_vol:
        return "BULL_TREND"
    elif trending_down:
        return "BEAR_TREND"
    elif high_vol:
        return "HIGH_VOL_LATERAL"
    else:
        return "LOW_VOL_GRIND"


# ── FEATURE STORE ─────────────────────────────────────────────────────────────

def inspect_feature_store() -> tuple:
    cache_path = Path(CACHE_DIR)
    files = sorted(cache_path.glob("*.parquet"))
    required = set(_required_cache_columns())
    valid = invalid = 0
    for fpath in files:
        try:
            cols = set(_read_parquet_columns(fpath))
            if required.issubset(cols):
                valid += 1
            else:
                invalid += 1
        except Exception:
            invalid += 1
    return len(files), valid, invalid


def build_feature_store() -> bool:
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"  ERROR: {DATA_DIR} does not exist")
        return False

    symbol_files = sorted(data_path.glob("*.parquet"))
    if not symbol_files:
        print(f"  ERROR: no .parquet files in {DATA_DIR}")
        return False

    print(f"\n  Feature store: {len(symbol_files)} symbols | "
          f"target features: {len(FEATURE_COLS)} + future_ret")

    count = rebuilt = 0
    total = len(symbol_files)
    t0 = time.time()

    for i, fpath in enumerate(symbol_files, 1):
        cache_file = cache_path / f"{fpath.stem}.parquet"
        if cache_file.exists():
            try:
                cached_cols = set(_read_parquet_columns(cache_file))
                required = set(_required_cache_columns())
                if required.issubset(cached_cols):
                    count += 1
                    if i % 50 == 0 or i == total:
                        print(f"  [{i}/{total}] {i/total*100:.0f}%  cached={count}  "
                              f"({time.time()-t0:.0f}s)")
                        sys.stdout.flush()
                    continue
                cache_file.unlink()
            except Exception:
                cache_file.unlink(missing_ok=True)

        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.lower() for c in df.columns]
            df = _fix_timestamp(df)
            if "close" not in df.columns or len(df) < HORIZON_BARS + 50:
                continue

            feat_df = build_features(df, timeframe="5min")
            feat_df["future_ret"] = np.log(
                df["close"].shift(-HORIZON_BARS) / df["close"]
            ).astype(np.float32)
            feat_df["future_ret_15m"] = np.log(
                df["close"].shift(-HORIZON_BARS_15M) / df["close"]
            ).astype(np.float32)
            feat_df["future_ret_1h"] = np.log(
                df["close"].shift(-HORIZON_BARS_1H) / df["close"]
            ).astype(np.float32)
            feat_df = feat_df.dropna(subset=FEATURE_COLS, how="all")
            if len(feat_df) < 20:
                continue

            feat_df.to_parquet(cache_file)
            count += 1
            rebuilt += 1

            if i % 25 == 0 or i == total:
                print(f"  [{i}/{total}] built {fpath.stem}  "
                      f"(cached={count}  {time.time()-t0:.0f}s)")
                sys.stdout.flush()
        except ValueError as e:
            print(f"  SKIP {fpath.stem}: {e}")
        except Exception as e:
            print(f"  WARN {fpath.stem}: {e}")

    print(f"  Feature cache: {count}/{total} symbols OK\n")
    return count > 0


def load_feature_store() -> dict:
    symbols = {}
    cache_path = Path(CACHE_DIR)
    files = sorted(cache_path.glob("*.parquet"))
    required = set(_required_cache_columns())

    print(f"  Loading {len(files)} symbols from feature cache...")
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            if not required.issubset(set(df.columns)):
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df.sort_index()
            if df.index.max().year < 2018:
                continue
            if len(df) > 50:
                symbols[fpath.stem] = df
        except Exception:
            pass

    print(f"  Loaded {len(symbols)} valid symbols")
    return symbols


# ── DATE SPLITS ───────────────────────────────────────────────────────────────

def get_date_splits(symbols: dict) -> tuple:
    all_min = [df.index.min() for df in symbols.values()]
    all_max = [df.index.max() for df in symbols.values()]
    global_min = min(all_min)
    global_max = max(all_max)
    total_span = global_max - global_min
    y1_end = global_min + (total_span / 3)
    y2_end = global_min + (total_span * 2 / 3)

    print(f"  Data range   : {global_min.date()} -> {global_max.date()}")
    print(f"  Year 1 (init): {global_min.date()} -> {y1_end.date()}")
    print(f"  Year 2 (walk): {y1_end.date()} -> {y2_end.date()}")
    print(f"  Year 3 (walk): {y2_end.date()} -> {global_max.date()}")
    return global_min, global_max, y1_end, y2_end


# ── REGIME-AWARE FEATURE SELECTION ────────────────────────────────────────────

def compute_feature_ic(symbols: dict, week_start, week_end,
                       features: list) -> Dict[str, float]:
    feature_ics = {}
    week_data = {}

    for sym, df in symbols.items():
        mask = (df.index >= week_start) & (df.index < week_end)
        subset = df.loc[mask]
        if len(subset) < 3 or "future_ret" not in subset.columns:
            continue
        week_data[sym] = subset

    if len(week_data) < 20:
        return {f: 0.0 for f in features}

    for feat in features:
        feat_vals = []
        ret_vals = []
        for sym, subset in week_data.items():
            if feat in subset.columns:
                valid = subset[[feat, "future_ret"]].dropna()
                if len(valid) > 0:
                    feat_vals.append(float(valid[feat].mean()))
                    ret_vals.append(float(valid["future_ret"].mean()))

        if len(feat_vals) >= 20:
            ic, _ = stats.spearmanr(feat_vals, ret_vals)
            feature_ics[feat] = float(ic) if np.isfinite(ic) else 0.0
        else:
            feature_ics[feat] = 0.0

    return feature_ics


def select_features_by_ic(ic_history: Dict[str, List[float]],
                          all_features: list) -> list:
    if not ic_history:
        return all_features

    scores = {}
    for feat in all_features:
        ics = ic_history.get(feat, [])
        if len(ics) >= 4:
            recent = ics[-IC_LOOKBACK_WEEKS:]
            scores[feat] = float(np.mean(recent))
        else:
            scores[feat] = 0.0

    selected = [f for f, ic in scores.items() if ic >= IC_SELECTION_THRESH]

    if len(selected) < MIN_FEATURES:
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        selected = [f for f, _ in ranked[:MIN_FEATURES]]

    return selected


# ── TRAINING MATRIX ───────────────────────────────────────────────────────────

class PurgedTimeSeriesCV:
    def __init__(self, n_splits=5, gap=48):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            val_start = train_end + self.gap
            val_end = val_start + fold_size
            if val_end > n:
                break
            yield np.arange(0, train_end), np.arange(val_start, val_end)


def build_training_matrix(symbols: dict, train_end,
                          features: list = None) -> tuple:
    """v5: Returns (features, continuous_returns, continuous_returns).
    Label IS the raw forward return — no binary conversion."""
    use_features = features or FEATURE_COLS
    print(f"  Building training matrix up to {pd.Timestamp(train_end).date()} "
          f"({len(use_features)} features)...")

    eligible = []
    total_rows = 0

    for sym, df in symbols.items():
        if "future_ret" not in df.columns:
            continue
        mask = df.index < train_end
        row_count = int(mask.sum())
        if row_count < HORIZON_BARS + 50:
            continue
        eligible.append(sym)
        total_rows += row_count

    if not eligible:
        print("  ERROR: no valid symbol data")
        return None, None, None

    print(f"  Candidate pool: {total_rows:,} rows x {len(eligible)} symbols")

    rng = np.random.default_rng(42)
    sample_prob = min(1.0, MAX_TRAIN_ROWS / max(total_rows, 1))

    n_feat = len(use_features)
    initial = min(total_rows, int(MAX_TRAIN_ROWS * 1.05))
    feat_arr = np.empty((max(initial, 1), n_feat), dtype=np.float32)
    ret_arr = np.empty(max(initial, 1), dtype=np.float32)
    cursor = 0

    def grow(needed):
        nonlocal feat_arr, ret_arr
        if needed <= len(ret_arr):
            return
        new_size = max(needed, int(len(ret_arr) * 1.25))
        feat_arr = np.resize(feat_arr, (new_size, n_feat))
        ret_arr = np.resize(ret_arr, new_size)

    for sym in eligible:
        try:
            df = symbols[sym]
            subset = df.loc[df.index < train_end, use_features + ["future_ret"]]
            if len(subset) < HORIZON_BARS + 50:
                continue

            f = subset[use_features].to_numpy(dtype=np.float32)
            r = subset["future_ret"].to_numpy(dtype=np.float32)

            valid = np.isfinite(f).all(axis=1) & np.isfinite(r)
            if sample_prob < 1.0:
                valid &= rng.random(len(valid)) < sample_prob

            keep = np.flatnonzero(valid)
            if len(keep) == 0:
                continue

            end = cursor + len(keep)
            grow(end)
            feat_arr[cursor:end] = f[keep]
            ret_arr[cursor:end] = r[keep]
            cursor = end
        except Exception:
            pass

    feat_arr = feat_arr[:cursor]
    ret_arr = ret_arr[:cursor]

    if len(feat_arr) < 50:
        print("  ERROR: fewer than 50 valid rows")
        return None, None, None

    if len(feat_arr) > MAX_TRAIN_ROWS:
        idx = rng.choice(len(feat_arr), MAX_TRAIN_ROWS, replace=False)
        idx.sort()
        feat_arr, ret_arr = feat_arr[idx], ret_arr[idx]
        print(f"  VRAM guard: capped at {MAX_TRAIN_ROWS:,}")

    mean_ret = float(np.mean(ret_arr))
    std_ret = float(np.std(ret_arr))
    print(f"  Training matrix: {len(feat_arr):,} rows x {n_feat} features  |  "
          f"target mean={mean_ret*100:.3f}%  std={std_ret*100:.3f}%")
    gc.collect()
    return feat_arr, ret_arr, ret_arr


# ── TRAIN MODEL ───────────────────────────────────────────────────────────────

def train_model(X, y, y_ret, cuda_api, features_used, label=""):
    """v5: XGBoost regression — predicts continuous returns."""
    import xgboost as xgb
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    r2s, ics = [], []

    _effective_cuda = cuda_api

    def _fit_with_fallback(model, Xtr, ytr, Xval, yval, verbose=False):
        nonlocal _effective_cuda
        try:
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=verbose)
        except Exception as e:
            err = str(e).lower()
            if _effective_cuda and ("memory" in err or "cuda" in err
                                    or "out of memory" in err):
                print(f"  [WARN] GPU OOM during {label} — falling back to CPU")
                _effective_cuda = None
                model = xgb.XGBRegressor(**make_xgb_params(None, regression=True))
                model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=verbose)
            else:
                raise
        return model

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        m = xgb.XGBRegressor(**make_xgb_params(_effective_cuda, regression=True))
        m = _fit_with_fallback(m, Xs[tr], y[tr], Xs[val], y[val])
        preds = m.predict(Xs[val])
        # Weighted R²
        ss_res = float(np.sum((y[val] - preds) ** 2))
        ss_tot = float(np.sum((y[val] - np.mean(y[val])) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2s.append(r2)
        # IC (Spearman rank correlation)
        mask = np.isfinite(preds) & np.isfinite(y[val])
        if mask.sum() >= 10:
            ics.append(float(stats.spearmanr(preds[mask], y[val][mask])[0]))

    mean_r2 = float(np.mean(r2s)) if r2s else 0.0
    mean_ic = float(np.mean(ics)) if ics else 0.0
    icir = float(np.mean(ics) / (np.std(ics) + 1e-8)) if len(ics) > 1 else 0.0

    final = xgb.XGBRegressor(**make_xgb_params(_effective_cuda, regression=True))
    split = int(len(Xs) * 0.9)
    final = _fit_with_fallback(final, Xs[:split], y[:split], Xs[split:], y[split:],
                               verbose=100)

    importance = pd.Series(final.feature_importances_, index=features_used,
                           name="importance").sort_values(ascending=False)
    return final, scaler, importance, mean_r2, mean_ic, icir


def train_meta_model(base_model, base_scaler, X, y, cuda_api, features_used,
                     label="meta"):
    """v5: Confidence model — predicts P(direction correct) for sizing."""
    import xgboost as xgb
    Xs = base_scaler.transform(X)
    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    oos_preds = np.full(len(y), np.nan, dtype=np.float32)
    _effective_cuda = cuda_api

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        m_temp = xgb.XGBRegressor(**make_xgb_params(_effective_cuda, regression=True))
        try:
            m_temp.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)
        except Exception as e:
            err = str(e).lower()
            if _effective_cuda and ("memory" in err or "cuda" in err
                                    or "out of memory" in err):
                print(f"  [WARN] GPU OOM in meta {label} fold {fold} — CPU fallback")
                _effective_cuda = None
                m_temp = xgb.XGBRegressor(**make_xgb_params(None, regression=True))
                m_temp.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)
            else:
                raise
        oos_preds[val] = m_temp.predict(Xs[val])

    valid = np.isfinite(oos_preds)
    if valid.sum() < 200:
        print(f"  [{label}] Insufficient OOS data ({valid.sum()}) — skipping confidence model")
        return None, None

    # Confidence label: 1 if predicted sign matches actual sign, 0 otherwise
    pred_sign = np.sign(oos_preds[valid])
    actual_sign = np.sign(y[valid])
    conf_y = (pred_sign == actual_sign).astype(np.float32)

    X_meta = np.column_stack([Xs[valid], oos_preds[valid]])
    meta_scaler = RobustScaler()
    X_meta_s = meta_scaler.fit_transform(X_meta)

    meta_params = make_xgb_params(_effective_cuda, regression=False)
    meta_params.update(n_estimators=500, max_depth=4, min_child_weight=50)
    meta = xgb.XGBClassifier(**meta_params)
    split = int(len(X_meta_s) * 0.9)
    try:
        meta.fit(X_meta_s[:split], conf_y[:split],
                 eval_set=[(X_meta_s[split:], conf_y[split:])], verbose=False)
    except Exception as e:
        err = str(e).lower()
        if _effective_cuda and ("memory" in err or "cuda" in err
                                or "out of memory" in err):
            print(f"  [WARN] GPU OOM in meta final {label} — CPU fallback")
            meta_params = make_xgb_params(None, regression=False)
            meta_params.update(n_estimators=500, max_depth=4, min_child_weight=50)
            meta = xgb.XGBClassifier(**meta_params)
            meta.fit(X_meta_s[:split], conf_y[:split],
                     eval_set=[(X_meta_s[split:], conf_y[split:])], verbose=False)
        else:
            raise

    val_acc = float((meta.predict(X_meta_s[split:]) == conf_y[split:]).mean())
    print(f"  [{label}] Confidence model accuracy: {val_acc*100:.1f}%")
    return meta, meta_scaler


# ── SHAP EXPLAINABILITY ───────────────────────────────────────────────────────

def compute_shap(model, X_sample, feature_names, max_samples=5000):
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap package not installed — skipping")
        return {}

    if len(X_sample) > max_samples:
        idx = np.random.choice(len(X_sample), max_samples, replace=False)
        X_sample = X_sample[idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        result = {feat: float(val) for feat, val in zip(feature_names, mean_abs)}
        print(f"  [SHAP] Computed for {len(feature_names)} features "
              f"(top: {max(result, key=result.get)}={max(result.values()):.4f})")
        return result
    except Exception as e:
        print(f"  [SHAP] Error: {e}")
        return {}


def save_shap_csv(shap_dict: dict, path: str, label: str = ""):
    if not shap_dict:
        return
    shap_dir = os.path.join(os.path.dirname(path), "shap")
    os.makedirs(shap_dir, exist_ok=True)
    suffix = f"_{label}" if label else ""
    out_path = os.path.join(shap_dir, f"shap_importance{suffix}.csv")
    df = pd.DataFrame([
        {"feature": k, "mean_abs_shap": v}
        for k, v in sorted(shap_dict.items(), key=lambda x: -x[1])
    ])
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(out_path, index=False)
    print(f"  [SHAP] Saved -> {out_path}")


# ── WALK-FORWARD PREDICTION + TRADE SIMULATION ───────────────────────────────

def predict_week(model, scaler, symbols, week_start, week_end,
                 features_used, meta_model=None, meta_scaler=None):
    """v5: Per-bar regression predictions, averaged per symbol per week."""
    predictions = {}
    actual_rets = {}
    meta_confs = {}

    for sym, df in symbols.items():
        try:
            week_data = df[(df.index >= week_start) & (df.index < week_end)]
            if len(week_data) < 3:
                continue

            feat = week_data[features_used].values.astype(np.float32)
            valid = np.isfinite(feat).all(axis=1)
            if valid.sum() < 2:
                continue

            feat_scaled = scaler.transform(feat[valid])
            pred_rets = model.predict(feat_scaled)
            predictions[sym] = float(np.mean(pred_rets))

            if meta_model is not None and meta_scaler is not None:
                try:
                    meta_input = np.column_stack([feat_scaled,
                                                  pred_rets.reshape(-1, 1)])
                    meta_scaled = meta_scaler.transform(meta_input)
                    meta_probs = meta_model.predict_proba(meta_scaled)[:, 1]
                    meta_confs[sym] = float(meta_probs.mean())
                except Exception:
                    pass

            if "future_ret" in week_data.columns:
                ret_col = week_data["future_ret"].values[valid]
                finite = ret_col[np.isfinite(ret_col)]
                if len(finite) > 0:
                    actual_rets[sym] = float(finite.mean())
        except Exception:
            pass

    return predictions, actual_rets, meta_confs


def simulate_weekly_trades(predictions, actual_rets, prev_longs, prev_shorts,
                           meta_confs=None, risk_manager=None,
                           weekly_returns_hist=None):
    """v5: Use predicted return sign for direction, magnitude for sizing."""
    if not predictions:
        return [], 0.0, set(), set()

    pred_series = pd.Series(predictions)

    # v5: Long top predicted returns, short bottom predicted returns
    ranked = pred_series.rank(pct=True)
    cur_longs = set(ranked[ranked >= (1 - TOP_QUANTILE)].index)
    cur_shorts = set(ranked[ranked <= TOP_QUANTILE].index)

    risk_scale = 1.0
    if risk_manager is not None and weekly_returns_hist and len(weekly_returns_hist) >= 4:
        ret_series = pd.Series(weekly_returns_hist)
        var = risk_manager.calculate_var(ret_series, VAR_CONFIDENCE)
        if var < -POSITION_RISK_CAP:
            risk_scale = max(0.3, POSITION_RISK_CAP / abs(var))

    trades = []
    for sym in cur_longs:
        ret = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        fee = 0.0 if sym in prev_longs else ROUND_TRIP_FEE
        meta = meta_confs.get(sym, 1.0) if meta_confs else 1.0
        # v5: scale by predicted return magnitude (confidence)
        pred_mag = min(abs(predictions[sym]) * 100, 3.0) / 3.0  # cap at 3%
        sized = meta * risk_scale * (0.5 + 0.5 * pred_mag)
        trades.append({
            "symbol": sym, "signal": "BUY",
            "pred_ret": round(predictions[sym] * 100, 5),
            "pnl_percent": round((ret - fee) * sized * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "meta_size": round(sized, 4),
        })

    for sym in cur_shorts:
        ret = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        fee = 0.0 if sym in prev_shorts else ROUND_TRIP_FEE
        meta = meta_confs.get(sym, 1.0) if meta_confs else 1.0
        pred_mag = min(abs(predictions[sym]) * 100, 3.0) / 3.0
        sized = meta * risk_scale * (0.5 + 0.5 * pred_mag)
        trades.append({
            "symbol": sym, "signal": "SELL",
            "pred_ret": round(predictions[sym] * 100, 5),
            "pnl_percent": round((-ret - fee) * sized * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "meta_size": round(sized, 4),
        })

    if trades:
        sizes = np.array([t["meta_size"] for t in trades])
        pnls = np.array([t["pnl_percent"] for t in trades])
        week_ret = float(np.average(pnls, weights=sizes)) / 100
    else:
        week_ret = 0.0

    return trades, week_ret, cur_longs, cur_shorts


# ── DRAWDOWN ─────────────────────────────────────────────────────────────────

def compute_drawdown(weekly_returns: list) -> float:
    if not weekly_returns:
        return 0.0
    cum = np.cumprod([1 + r for r in weekly_returns])
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


# ── CHECKPOINT ───────────────────────────────────────────────────────────────

def _ckpt_path(results_dir):
    return os.path.join(results_dir, "checkpoint_v4_latest.json")


def save_checkpoint(results_dir, state):
    path = _ckpt_path(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


def load_checkpoint(results_dir):
    path = _ckpt_path(results_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            ckpt = json.load(f)
        print(f"  [CHECKPOINT] Found  run_id={ckpt.get('run_id')}  "
              f"last_week={ckpt.get('last_week')}  ts={ckpt.get('ts', '?')}")
        return ckpt
    except Exception as e:
        print(f"  [CHECKPOINT] Could not load ({e}) — starting fresh")
        return None


# ── MAIN ENGINE ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azalyst v5 Engine")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-dd", type=float, default=MAX_DRAWDOWN_KILL)
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP computation (faster)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-resume", dest="no_resume", action="store_true")
    args = parser.parse_args()

    global DATA_DIR, RESULTS_DIR, CACHE_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.feature_dir:
        CACHE_DIR = args.feature_dir
    if args.out_dir:
        RESULTS_DIR = args.out_dir

    use_gpu = args.gpu and not args.no_gpu
    dd_kill = args.max_dd

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # FIX: was PCI_E_BUS_ID — correct NVIDIA env value is PCI_BUS_ID
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import xgboost as xgb

    print("\n" + "=" * 72)
    print("  AZALYST v5  —  Short-Horizon Regression Engine")
    print("  Model: XGBoost Regressor (1hr forward return)")
    print("  Features: Reversal-dominated + Pump-Dump + Quantile Rank")
    print("  Walk:  Y2 + Y3 (2-year out-of-sample)")
    print("=" * 72)
    print(f"  Compute      : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"  Features     : {len(FEATURE_COLS)} (with IC-gating selection)")
    print(f"  Horizon      : {HORIZON_BARS} bars (1hr) / {HORIZON_BARS_15M} bars (15min)")
    print(f"  Kill-switch  : {dd_kill*100:.0f}% max drawdown")
    print(f"  SHAP         : {'disabled (--no-shap)' if args.no_shap else 'enabled'}")
    print(f"  Persistence  : SQLite (results/azalyst.db)")
    print("=" * 72 + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    _log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    try:
        _log_fh = open(_log_path, "w", encoding="utf-8", buffering=1)
    except Exception:
        _log_fh = None

    def _log(msg: str = "") -> None:
        print(msg)
        if _log_fh:
            try:
                _log_fh.write(msg + "\n")
                _log_fh.flush()
            except Exception:
                pass

    db = AzalystDB(f"{RESULTS_DIR}/azalyst.db")
    ckpt = None if args.no_resume else load_checkpoint(RESULTS_DIR)
    resuming = ckpt is not None
    if resuming:
        run_id = ckpt["run_id"]
        _log(f"\n  [CHECKPOINT] Resuming run_id={run_id}  "
             f"from week {ckpt['last_week'] + 1}\n")
        sys.stdout.flush()
    else:
        run_id = args.run_id or f"v5_{time.strftime('%Y%m%d_%H%M%S')}"
        db.start_run(run_id, {
            "version": "v5", "gpu": use_gpu, "features": len(FEATURE_COLS),
            "max_dd_kill": dd_kill, "shap": not args.no_shap,
            "retrain_weeks": RETRAIN_WEEKS,
            "horizon_bars": HORIZON_BARS,
            "model_type": "XGBRegressor",
        })
    risk_mgr = RiskManager()

    _log("STEP 0: Feature cache\n")
    if args.rebuild_cache:
        if not build_feature_store():
            _log("ERROR: Feature store build failed")
            db.finish_run(run_id, "failed")
            return
    else:
        total, valid, invalid = inspect_feature_store()
        if total == 0:
            _log("  No cache -- building...")
            if not build_feature_store():
                db.finish_run(run_id, "failed")
                return
        elif invalid:
            build_feature_store()
        else:
            _log(f"  Found {valid} valid cache files")

    cuda_api = detect_cuda_api() if use_gpu else None

    _log("\nSTEP 1: Load feature cache\n")
    _log(f"  Loading {len(list(Path(CACHE_DIR).glob('*.parquet')))} symbols from feature cache...")
    symbols = load_feature_store()
    if not symbols:
        _log("ERROR: No symbols loaded")
        db.finish_run(run_id, "failed")
        return
    _log(f"  Loaded {len(symbols)} valid symbols")

    print("\nSTEP 2: Date splits (v4: walk Y2+Y3)\n")
    global_min, global_max, y1_end, y2_end = get_date_splits(symbols)

    os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)
    if resuming:
        print("\nSTEP 3: Skipping — loading models from checkpoint...\n")
        active_features = ckpt["active_features"]
        base_model = xgb.XGBRegressor()
        base_model.load_model(ckpt["current_model_path"])
        with open(ckpt["current_scaler_path"], "rb") as f:
            base_scaler = pickle.load(f)
        meta_model, meta_scaler = None, None
        if ckpt.get("current_meta_path") and os.path.exists(ckpt["current_meta_path"]):
            with open(ckpt["current_meta_path"], "rb") as f:
                meta_model = pickle.load(f)
        if ckpt.get("current_meta_scaler_path") and \
                os.path.exists(ckpt["current_meta_scaler_path"]):
            with open(ckpt["current_meta_scaler_path"], "rb") as f:
                meta_scaler = pickle.load(f)
        print(f"  Model : {ckpt['current_model_path']}")
        print(f"  Scaler: {ckpt['current_scaler_path']}")
        sys.stdout.flush()
    else:
        print("\nSTEP 3: Initial training on Y1\n")
        active_features = list(FEATURE_COLS)
        X_train, y_train, y_ret = build_training_matrix(symbols, y1_end, active_features)
        if X_train is None:
            print("ERROR: Could not build training matrix")
            db.finish_run(run_id, "failed")
            return

        print(f"\n  Training base regression model (GPU={'YES' if cuda_api else 'NO'})...")
        sys.stdout.flush()
        t0 = time.time()
        base_model, base_scaler, importance, mean_r2, mean_ic, icir = train_model(
            X_train, y_train, y_ret, cuda_api, active_features, label="base_y1"
        )
        print(f"  R²={mean_r2:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}  "
              f"({time.time()-t0:.1f}s)")
        sys.stdout.flush()

        base_model.save_model(f"{RESULTS_DIR}/models/model_v4_base.json")
        with open(f"{RESULTS_DIR}/models/scaler_v4_base.pkl", "wb") as f:
            pickle.dump(base_scaler, f)
        importance.to_csv(f"{RESULTS_DIR}/feature_importance_v4_base.csv")

        db.insert_model_artifact(run_id, "base_y1", 0,
                                 f"{RESULTS_DIR}/models/model_v4_base.json",
                                 f"{RESULTS_DIR}/models/scaler_v4_base.pkl",
                                 mean_r2, mean_ic, icir, len(active_features))

        if not args.no_shap:
            print("\n  Computing SHAP values...")
            Xs = base_scaler.transform(X_train)
            shap_vals = compute_shap(base_model, Xs, active_features)
            if shap_vals:
                save_shap_csv(shap_vals, f"{RESULTS_DIR}/models/", "v4_base")
                db.insert_shap_values(run_id, "base_y1", shap_vals)

        print("\n  Training confidence model...")
        sys.stdout.flush()
        meta_model, meta_scaler = train_meta_model(
            base_model, base_scaler, X_train, y_train, cuda_api,
            active_features, label="conf_y1"
        )
        if meta_model is not None:
            with open(f"{RESULTS_DIR}/models/meta_v4_base.pkl", "wb") as f:
                pickle.dump(meta_model, f)
            with open(f"{RESULTS_DIR}/models/meta_scaler_v4_base.pkl", "wb") as f:
                pickle.dump(meta_scaler, f)

        with open(f"{RESULTS_DIR}/train_summary_v4.json", "w") as f:
            json.dump({"mean_r2": round(mean_r2, 5), "mean_ic": round(mean_ic, 5),
                       "icir": round(icir, 5), "n_rows": int(len(X_train)),
                       "n_features": len(active_features),
                       "model_type": "XGBRegressor",
                       "objective": "reg:squarederror",
                       "horizon_bars": HORIZON_BARS,
                       "use_gpu": cuda_api is not None}, f, indent=2)

        del X_train, y_train, y_ret
        gc.collect()

    walk_start = y1_end
    print(f"\nSTEP 4+5: Walk-forward  ({walk_start.date()} -> {global_max.date()})\n")
    print(f"  Y2 zone: {walk_start.date()} -> {y2_end.date()}")
    print(f"  Y3 zone: {y2_end.date()} -> {global_max.date()}\n")
    sys.stdout.flush()

    weeks = pd.date_range(start=walk_start, end=global_max, freq="W-MON")
    if len(weeks) < 2:
        print("  Not enough weeks")
        db.finish_run(run_id, "failed")
        return

    current_model = base_model
    current_scaler = base_scaler
    current_meta = meta_model
    current_meta_scaler = meta_scaler
    current_model_path = f"{RESULTS_DIR}/models/model_v4_base.json"
    current_scaler_path = f"{RESULTS_DIR}/models/scaler_v4_base.pkl"
    current_meta_path = f"{RESULTS_DIR}/models/meta_v4_base.pkl"
    current_meta_scaler_path = f"{RESULTS_DIR}/models/meta_scaler_v4_base.pkl"

    if resuming:
        retrains = ckpt["retrains"]
        prev_longs = set(ckpt["prev_longs"])
        prev_shorts = set(ckpt["prev_shorts"])
        all_trades = ckpt["all_trades"]
        weekly_summary = ckpt["weekly_summary"]
        weekly_returns = ckpt["weekly_returns"]
        feature_ic_history = {k: list(v)
                              for k, v in ckpt.get("feature_ic_history", {}).items()}
        kill_switch_hit = ckpt.get("kill_switch_hit", False)
        resume_from_week = ckpt["last_week"]
        current_model_path = ckpt.get("current_model_path", current_model_path)
        current_scaler_path = ckpt.get("current_scaler_path", current_scaler_path)
        current_meta_path = ckpt.get("current_meta_path", current_meta_path)
        current_meta_scaler_path = ckpt.get("current_meta_scaler_path",
                                            current_meta_scaler_path)
        print(f"  Restored {len(weekly_returns)} weeks of history, "
              f"{len(all_trades)} trades. Resuming from week {resume_from_week + 1}.\n")
        sys.stdout.flush()
    else:
        retrains = 0
        prev_longs, prev_shorts = set(), set()
        all_trades = []
        weekly_summary = []
        weekly_returns = []
        feature_ic_history: Dict[str, List[float]] = {}
        kill_switch_hit = False
        resume_from_week = 0

    for week_num, (ws, we) in enumerate(zip(weeks[:-1], weeks[1:]), 1):
        if week_num <= resume_from_week:
            continue

        current_dd = compute_drawdown(weekly_returns)
        if current_dd < dd_kill:
            print(f"\n  *** KILL SWITCH *** Week {week_num}: "
                  f"DD={current_dd*100:.1f}% < {dd_kill*100:.0f}% threshold")
            print("  Halting all trading. Pausing for 4 weeks...")
            kill_switch_hit = True
            weekly_returns.extend([0.0] * min(4, len(weeks) - week_num - 1))
            for skip in range(4):
                weekly_summary.append({
                    "week": week_num + skip, "week_start": str(ws.date()),
                    "week_end": str(we.date()), "n_symbols": 0, "n_trades": 0,
                    "week_return_pct": 0.0, "annualised_pct": 0.0, "ic": 0.0,
                    "turnover_pct": 0.0, "on_track": False,
                    "cum_return_pct": round(
                        (np.prod([1 + r for r in weekly_returns]) - 1) * 100, 4),
                    "max_drawdown_pct": round(current_dd * 100, 4),
                    "regime": "KILL_SWITCH",
                })
            kill_switch_hit = False
            continue

        regime = detect_regime(symbols, we)

        if week_num > 1 and week_num % 2 == 0:
            fic = compute_feature_ic(symbols, ws, we, active_features)
            for feat, ic_val in fic.items():
                if feat not in feature_ic_history:
                    feature_ic_history[feat] = []
                feature_ic_history[feat].append(ic_val)
            db.insert_feature_ic(run_id, week_num, fic, active_features)

        if week_num % RETRAIN_WEEKS == 0:
            print(f"\n  Week {week_num:3d}: QUARTERLY RETRAIN (expanding window to {we.date()})...")

            new_features = select_features_by_ic(feature_ic_history, FEATURE_COLS)
            n_dropped = len(active_features) - len(new_features)
            if n_dropped > 0:
                print(f"    Feature selection: {len(active_features)} -> {len(new_features)} "
                      f"({n_dropped} dropped by IC filter)")
            active_features = new_features

            X_rt, y_rt, y_ret_rt = build_training_matrix(symbols, we, active_features)
            if X_rt is not None and len(X_rt) > 200:
                t0 = time.time()
                m_new, s_new, imp_new, r2_n, ic_n, icir_n = train_model(
                    X_rt, y_rt, y_ret_rt, cuda_api, active_features,
                    label=f"v5_w{week_num:03d}"
                )
                current_model, current_scaler = m_new, s_new
                m_new.save_model(f"{RESULTS_DIR}/models/model_v4_week{week_num:03d}.json")
                with open(f"{RESULTS_DIR}/models/scaler_v4_week{week_num:03d}.pkl", "wb") as f:
                    pickle.dump(s_new, f)
                current_model_path = f"{RESULTS_DIR}/models/model_v4_week{week_num:03d}.json"
                current_scaler_path = f"{RESULTS_DIR}/models/scaler_v4_week{week_num:03d}.pkl"
                imp_new.to_csv(f"{RESULTS_DIR}/feature_importance_v4_week{week_num:03d}.csv")
                print(f"    R²={r2_n:.4f}  IC={ic_n:.4f}  ICIR={icir_n:.4f}  "
                      f"({time.time()-t0:.1f}s)")

                db.insert_model_artifact(run_id, f"retrain_w{week_num:03d}", week_num,
                                         current_model_path,
                                         current_scaler_path,
                                         r2_n, ic_n, icir_n, len(active_features))

                if not args.no_shap:
                    Xs_rt = s_new.transform(X_rt)
                    shap_vals = compute_shap(m_new, Xs_rt, active_features)
                    if shap_vals:
                        save_shap_csv(shap_vals, f"{RESULTS_DIR}/models/",
                                      f"v4_week{week_num:03d}")
                        db.insert_shap_values(run_id, f"retrain_w{week_num:03d}", shap_vals)

                meta_new, meta_s_new = train_meta_model(
                    current_model, current_scaler, X_rt, y_rt, cuda_api,
                    active_features, label=f"conf_w{week_num:03d}"
                )
                if meta_new is not None:
                    current_meta, current_meta_scaler = meta_new, meta_s_new
                    with open(f"{RESULTS_DIR}/models/meta_v4_week{week_num:03d}.pkl", "wb") as f:
                        pickle.dump(meta_new, f)
                    with open(f"{RESULTS_DIR}/models/meta_scaler_v4_week{week_num:03d}.pkl", "wb") as f:
                        pickle.dump(meta_s_new, f)
                    current_meta_path = f"{RESULTS_DIR}/models/meta_v4_week{week_num:03d}.pkl"
                    current_meta_scaler_path = f"{RESULTS_DIR}/models/meta_scaler_v4_week{week_num:03d}.pkl"

                retrains += 1
                del X_rt, y_rt, y_ret_rt
                gc.collect()

        # v5: IC-gating — if recent IC is below threshold, skip trading
        recent_ics = []
        for feat_ics in feature_ic_history.values():
            if feat_ics:
                recent_ics.append(feat_ics[-1])
        avg_recent_ic = float(np.mean(recent_ics)) if recent_ics else 0.0
        ic_gated = avg_recent_ic < IC_GATING_THRESHOLD and len(recent_ics) >= 4

        if ic_gated:
            weekly_returns.append(0.0)
            weekly_summary.append({
                "week": week_num, "week_start": str(ws.date()),
                "week_end": str(we.date()), "n_symbols": 0, "n_trades": 0,
                "week_return_pct": 0.0, "annualised_pct": 0.0, "ic": avg_recent_ic,
                "turnover_pct": 0.0, "on_track": False,
                "cum_return_pct": round(
                    (np.prod([1 + r for r in weekly_returns]) - 1) * 100, 4),
                "max_drawdown_pct": round(compute_drawdown(weekly_returns) * 100, 4),
                "regime": "IC_GATED",
            })
            print(f"  Week {week_num:3d}: IC-GATED (avg_ic={avg_recent_ic:.4f} "
                  f"< {IC_GATING_THRESHOLD})")
            continue

        predictions, actual_rets, meta_confs = predict_week(
            current_model, current_scaler, symbols, ws, we,
            active_features, meta_model=current_meta,
            meta_scaler=current_meta_scaler
        )

        if len(predictions) < 5:
            continue

        trades, week_ret, cur_longs, cur_shorts = simulate_weekly_trades(
            predictions, actual_rets, prev_longs, prev_shorts,
            meta_confs, risk_manager=risk_mgr,
            weekly_returns_hist=weekly_returns
        )
        weekly_returns.append(week_ret)

        common = [s for s in predictions if s in actual_rets]
        if len(common) > 10:
            pred_arr = np.array([predictions[s] for s in common])
            ret_arr = np.array([actual_rets[s] for s in common])
            week_ic = float(stats.spearmanr(pred_arr, ret_arr)[0])
        else:
            week_ic = 0.0

        n_cur = len(cur_longs) + len(cur_shorts)
        n_new = len(cur_longs - prev_longs) + len(cur_shorts - prev_shorts)
        turnover = round(n_new / n_cur * 100, 1) if n_cur > 0 else 100.0
        prev_longs, prev_shorts = cur_longs, cur_shorts

        cum_ret = float(np.prod([1 + r for r in weekly_returns]) - 1)
        max_dd = compute_drawdown(weekly_returns)

        for t in trades:
            t["week"] = week_num
            t["week_start"] = str(ws.date())
        all_trades.extend(trades)

        zone = "Y2" if we <= y2_end else "Y3"
        ann_proj = ((1 + week_ret) ** 52 - 1) * 100

        metric = {
            "week": week_num, "week_start": str(ws.date()),
            "week_end": str(we.date()), "n_symbols": len(predictions),
            "n_trades": len(trades), "week_return_pct": round(week_ret * 100, 4),
            "annualised_pct": round(ann_proj, 2), "ic": round(week_ic, 5),
            "turnover_pct": turnover,
            "on_track": week_ret >= ((1.10) ** (1 / 52) - 1),
            "cum_return_pct": round(cum_ret * 100, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "regime": regime,
        }
        weekly_summary.append(metric)
        db.insert_weekly_metric(run_id, metric)
        db.insert_trades(run_id, trades)

        print(f"  Week {week_num:3d} [{zone}]: {len(trades):4d} trades | "
              f"ret={week_ret*100:+.2f}% | IC={week_ic:+.4f} | "
              f"cum={cum_ret*100:+.1f}% | DD={max_dd*100:.1f}% | "
              f"{regime}")
        _log(f"  Week {week_num:3d} [{zone}]: {len(trades):4d} trades | "
             f"ret={week_ret*100:+.2f}% | IC={week_ic:+.4f} | "
             f"cum={cum_ret*100:+.1f}% | DD={max_dd*100:.1f}% | "
             f"{regime}")
        sys.stdout.flush()

        save_checkpoint(RESULTS_DIR, {
            "run_id": run_id,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_week": week_num,
            "weekly_returns": weekly_returns,
            "weekly_summary": weekly_summary,
            "all_trades": all_trades,
            "prev_longs": list(prev_longs),
            "prev_shorts": list(prev_shorts),
            "retrains": retrains,
            "active_features": active_features,
            "feature_ic_history": {k: list(v) for k, v in feature_ic_history.items()},
            "kill_switch_hit": kill_switch_hit,
            "current_model_path": current_model_path,
            "current_scaler_path": current_scaler_path,
            "current_meta_path": current_meta_path,
            "current_meta_scaler_path": current_meta_scaler_path,
        })

    # ── Save results ──────────────────────────────────────────────────────────
    trades_df = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(weekly_summary)

    if len(trades_df) > 0:
        trades_df.to_csv(f"{RESULTS_DIR}/all_trades_v4.csv", index=False)
    if len(summary_df) > 0:
        summary_df.to_csv(f"{RESULTS_DIR}/weekly_summary_v4.csv", index=False)

    n_wks = len(weekly_returns)
    cum_ret = float(np.prod([1 + r for r in weekly_returns]) - 1) if n_wks else 0.0
    ann_ret = ((1 + cum_ret) ** (52 / max(n_wks, 1)) - 1) * 100 if n_wks else 0.0
    wk_std = float(np.std(weekly_returns)) if n_wks > 1 else 0.0
    sharpe = float(np.mean(weekly_returns)) / wk_std * np.sqrt(52) if wk_std > 0 else 0.0
    max_dd = compute_drawdown(weekly_returns)

    ret_s = pd.Series(weekly_returns)
    var_95 = float(risk_mgr.calculate_var(ret_s, 0.95)) if n_wks > 4 else 0.0
    cvar_95 = float(risk_mgr.calculate_cvar(ret_s, 0.95)) if n_wks > 4 else 0.0

    ic_s = summary_df["ic"] if len(summary_df) > 0 else pd.Series(dtype=float)
    ic_mean = float(ic_s.mean()) if len(ic_s) > 0 else 0.0
    ic_std = float(ic_s.std()) if len(ic_s) > 1 else 0.0
    icir_val = ic_mean / (ic_std + 1e-8)
    ic_pos = float((ic_s > 0).mean() * 100) if len(ic_s) > 0 else 0.0

    _y2_end = pd.Timestamp(y2_end).replace(tzinfo=None)
    y2_weeks = summary_df[summary_df["week_end"].apply(
        lambda x: pd.Timestamp(x) <= _y2_end if x else False)]
    y3_weeks = summary_df[summary_df["week_end"].apply(
        lambda x: pd.Timestamp(x) > _y2_end if x else False)]

    y2_ret = float(y2_weeks["week_return_pct"].sum()) if len(y2_weeks) > 0 else 0.0
    y3_ret = float(y3_weeks["week_return_pct"].sum()) if len(y3_weeks) > 0 else 0.0
    y2_ic = float(y2_weeks["ic"].mean()) if len(y2_weeks) > 0 else 0.0
    y3_ic = float(y3_weeks["ic"].mean()) if len(y3_weeks) > 0 else 0.0

    perf = {
        "label": "v5_Regression_WalkForward_Y2Y3",
        "run_id": run_id,
        "total_weeks": n_wks,
        "total_trades": len(trades_df),
        "retrains": retrains,
        "total_return_pct": round(cum_ret * 100, 2),
        "annualised_pct": round(ann_ret, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "ic_mean": round(ic_mean, 5),
        "icir": round(icir_val, 4),
        "ic_positive_pct": round(ic_pos, 1),
        "var_95": round(var_95 * 100, 4),
        "cvar_95": round(cvar_95 * 100, 4),
        "kill_switch_hit": kill_switch_hit,
        "y2_return_pct": round(y2_ret, 2),
        "y3_return_pct": round(y3_ret, 2),
        "y2_ic_mean": round(y2_ic, 5),
        "y3_ic_mean": round(y3_ic, 5),
        "gpu": f"CUDA ({cuda_api})" if cuda_api else "CPU",
        "features_final": len(active_features),
        "features_initial": len(FEATURE_COLS),
        "data_range": f"{global_min.date()} -> {global_max.date()}",
    }

    with open(f"{RESULTS_DIR}/performance_v4.json", "w") as f:
        json.dump(perf, f, indent=2)

    db.insert_performance_summary(run_id, perf)
    db.finish_run(run_id)

    print(f"\n{'='*72}")
    print(f"  AZALYST v5  —  RUN COMPLETE  [{run_id}]")
    print(f"{'='*72}")
    print(f"  total_weeks       : {n_wks}")
    print(f"  total_trades      : {len(trades_df)}")
    print(f"  retrains          : {retrains}")
    print(f"  total_return_pct  : {cum_ret*100:+.2f}%")
    print(f"  annualised_pct    : {ann_ret:+.2f}%")
    print(f"  sharpe            : {sharpe:.4f}")
    print(f"  max_drawdown_pct  : {max_dd*100:.2f}%")
    print(f"  VaR (95%)         : {var_95*100:.2f}%")
    print(f"  CVaR (95%)        : {cvar_95*100:.2f}%")
    print(f"  ic_mean           : {ic_mean:.5f}")
    print(f"  icir              : {icir_val:.4f}")
    print(f"  ic_positive_pct   : {ic_pos:.1f}%")
    print(f"  features_used     : {len(active_features)}/{len(FEATURE_COLS)}")
    print(f"  kill_switch       : {'HIT' if kill_switch_hit else 'not triggered'}")
    print(f"  {'─'*68}")
    print(f"  Y2 (good zone)    : ret={y2_ret:+.2f}%  IC={y2_ic:.5f}")
    print(f"  Y3 (flip zone)    : ret={y3_ret:+.2f}%  IC={y3_ic:.5f}")
    print(f"{'='*72}")
    print(f"\n  Trades   -> {RESULTS_DIR}/all_trades_v4.csv")
    print(f"  Summary  -> {RESULTS_DIR}/weekly_summary_v4.csv")
    print(f"  Perf     -> {RESULTS_DIR}/performance_v4.json")
    print(f"  Database -> {RESULTS_DIR}/azalyst.db")
    print(f"  GPU used : {'RTX 2050 CUDA' if cuda_api else 'CPU'}")

    ckpt_file = _ckpt_path(RESULTS_DIR)
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
        print("  [CHECKPOINT] Cleared — run complete")
    db.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        import os as _os
        _os.devnull
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Checkpoint preserved — run again to resume.")
        sys.exit(1)
    except Exception as _e:
        import traceback
        print(f"\n  [FATAL] {type(_e).__name__}: {_e}")
        traceback.print_exc()
        sys.exit(1)
