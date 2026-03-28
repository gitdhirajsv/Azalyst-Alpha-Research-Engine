"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  LOCAL GPU RUNNER  (RTX 2050 4GB  |  i5-11260H)
║  FIXED v3: Full 56-feature builder | Dynamic date splits | No hardcoded    ║
║            dates | Consistent timestamp handling matching notebooks         ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY FIXES vs v2 (azalyst_local_gpu.py):
  1. CRITICAL: Uses build_features() + FEATURE_COLS from azalyst_factors_v2.py
     (v2 used a private 7-feature builder — cache was incompatible with notebooks)
  2. CRITICAL: Date splits are now DYNAMIC from actual data range, not hardcoded.
     (v2 hardcoded START_DATE="2024-12-01" etc which broke on any other dataset)
  3. Timestamp fix is identical to Notebook 1 + build_feature_cache.py:
     integer ms → pd.to_datetime(unit='ms', utc=True), then year < 2018 skip.
  4. future_ret column name matches notebooks ('future_ret', not 'future_ret_4h').
  5. alpha_label computed cross-sectionally AFTER pooling (not per-symbol).
  6. HORIZON_BARS=48 (4H @ 5-min) consistent with all other scripts.
"""

import argparse
import os
import sys
import gc
import json
import time
import pickle
import warnings
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Import the REAL feature builder (56 features) ────────────────────────────
# This is the fix: v2 used a private 7-feature builder; we now use the same
# build_features() and FEATURE_COLS that the notebooks and build_feature_cache.py use.
from azalyst_factors_v2 import build_features, FEATURE_COLS

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "./data"
RESULTS_DIR = "./results"
CACHE_DIR   = "./feature_cache"

MAX_TRAIN_ROWS = 2_000_000   # RTX 2050 4GB VRAM guard — DO NOT raise above 2M
RETRAIN_WEEKS  = 13
TOP_QUANTILE   = 0.15
FEE_RATE       = 0.001
ROUND_TRIP_FEE = FEE_RATE * 2
HORIZON_BARS   = 48          # 4H @ 5-min bars — matches notebooks exactly


# ─────────────────────────────────────────────────────────────────────────────
#  TIMESTAMP FIX  — identical logic to Notebook 1 and build_feature_cache.py
# ─────────────────────────────────────────────────────────────────────────────

def _fix_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a DataFrame to a UTC DatetimeIndex.
    Handles: integer-ms index, named timestamp columns, string datetimes.
    Raises ValueError if max year is still < 2018 after conversion (1970 bug).
    """
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


def _required_cache_columns() -> list[str]:
    return FEATURE_COLS + ["future_ret"]


def _read_parquet_columns(path: Path) -> list[str]:
    """
    Read parquet column names from metadata when possible so cache validation
    stays fast even when each symbol file is large.
    """
    if pq is not None:
        return list(pq.read_schema(path).names)
    return pd.read_parquet(path).columns.tolist()


def inspect_feature_store() -> tuple[int, int, int]:
    """Return (total_files, valid_files, stale_or_corrupt_files)."""
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


# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

def startup_banner(use_gpu: bool, year2_only: bool) -> None:
    sys.stdout.flush()
    print("\n" + "=" * 72)
    print("  AZALYST LOCAL GPU RUNNER  v3  (RTX 2050 | CPU i5-11260H)")
    print("=" * 72)
    print(f"  Compute     : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"  Mode        : {'Year-2-Only pretrain' if year2_only else 'Full Year-3 Walk-Forward'}")
    print(f"  VRAM cap    : {MAX_TRAIN_ROWS:,} training rows  (4GB RTX 2050 guard)")
    print(f"  Features    : {len(FEATURE_COLS)}  (full azalyst_factors_v2 set)")
    print(f"  Date splits : DYNAMIC from actual data range (no hardcoded dates)")
    print("=" * 72 + "\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
#  CUDA DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_cuda_api() -> str | None:
    """
    Returns 'new' (device='cuda'), 'old' (tree_method='gpu_hist'), or None (CPU).
    Prints clearly which path was taken.
    """
    try:
        import xgboost as xgb
        X = np.random.rand(200, 10).astype("float32")
        y = np.array([0] * 100 + [1] * 100)
        try:
            xgb.XGBClassifier(device="cuda", n_estimators=3, verbosity=0).fit(X, y)
            print("  [GPU] CUDA API: NEW  (device='cuda')  — XGBoost on RTX 2050")
            return "new"
        except Exception:
            pass
        try:
            xgb.XGBClassifier(tree_method="gpu_hist", n_estimators=3, verbosity=0).fit(X, y)
            print("  [GPU] CUDA API: OLD  (tree_method='gpu_hist')  — XGBoost on GPU")
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

def make_xgb_params(cuda_api: str | None, n_estimators: int = 1000,
                    max_depth: int = 6, min_child_weight: int = 30) -> dict:
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
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
        random_state=42,
    )
    if cuda_api == "new":
        p["device"] = "cuda"
    elif cuda_api == "old":
        p["tree_method"] = "gpu_hist"
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0: Feature Store Builder  (uses full 56-feature build_features())
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_store() -> bool:
    """
    Build feature cache using the FULL 56-feature build_features() from
    azalyst_factors_v2.py.  Saves one parquet per symbol with:
      - All FEATURE_COLS (56 features)
      - 'future_ret' column  (log return HORIZON_BARS ahead)
      - UTC DatetimeIndex

    Skips symbols already cached if ALL FEATURE_COLS are present.
    Rebuilds stale cache entries that are missing any column.
    """
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

        # Validate existing cache — must have all 56 FEATURE_COLS + future_ret
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
                # Stale — missing columns — rebuild
                cache_file.unlink()
            except Exception:
                cache_file.unlink(missing_ok=True)

        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.lower() for c in df.columns]
            df = _fix_timestamp(df)                 # UTC + year>=2018 check

            if "close" not in df.columns:
                continue
            if len(df) < HORIZON_BARS + 50:
                continue

            # Build full 56-feature set (identical to notebook Cell 3/4)
            feat_df = build_features(df, timeframe="5min")

            # Forward return label — log(close[t+H] / close[t])
            feat_df["future_ret"] = np.log(
                df["close"].shift(-HORIZON_BARS) / df["close"]
            ).astype(np.float32)

            # Drop rows where ALL feature cols are NaN (warmup period)
            feat_df = feat_df.dropna(subset=FEATURE_COLS, how="all")
            if len(feat_df) < 20:
                continue

            feat_df.to_parquet(cache_file)
            count += 1
            rebuilt += 1

            if i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"  [{i}/{total}] built {fpath.stem}  "
                      f"(cached={count}  {elapsed:.0f}s)")
                sys.stdout.flush()

        except ValueError as e:
            # 1970 timestamp or similar — skip silently
            print(f"  SKIP {fpath.stem}: {e}")
        except Exception as e:
            print(f"  WARN {fpath.stem}: {e}")

    if rebuilt:
        print(f"  Rebuilt {rebuilt} cache files")
    print(f"  Feature cache: {count}/{total} symbols OK  ({len(FEATURE_COLS)} features each)\n")
    return count > 0


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: Load feature cache (all symbols)
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_store() -> dict[str, pd.DataFrame]:
    """Load all cached symbol DataFrames into memory."""
    symbols: dict[str, pd.DataFrame] = {}
    cache_path = Path(CACHE_DIR)
    files = sorted(cache_path.glob("*.parquet"))
    required = set(_required_cache_columns())

    print(f"  Loading {len(files)} symbols from feature cache...")
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            missing = sorted(required.difference(df.columns))
            if missing:
                print(f"  [WARN] {fpath.stem}: missing {len(missing)} required columns")
                continue
            # Ensure UTC DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df.sort_index()
            if df.index.max().year < 2018:
                continue
            if len(df) > 50:
                symbols[fpath.stem] = df
        except Exception as e:
            print(f"  [WARN] {fpath.stem}: {e}")

    print(f"  Loaded {len(symbols)} valid symbols")
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: Dynamic date splits (matches notebook logic exactly)
# ─────────────────────────────────────────────────────────────────────────────

def get_date_splits(symbols: dict[str, pd.DataFrame]) -> tuple:
    """
    Derive Year-3 start date DYNAMICALLY from the actual data range.
    Year 3 = final 1/3 of the total timeline.
    This is identical to Cell 5 in azalyst_2_train.ipynb.
    """
    all_min, all_max = [], []
    for df in symbols.values():
        all_min.append(df.index.min())
        all_max.append(df.index.max())

    global_min = min(all_min)
    global_max = max(all_max)
    total_span = global_max - global_min
    year2_end  = global_min + (total_span * 2 / 3)

    print(f"  Data range  : {global_min.date()} → {global_max.date()}")
    print(f"  Train (Y1+Y2): {global_min.date()} → {year2_end.date()}")
    print(f"  Test  (Y3)  : {year2_end.date()} → {global_max.date()}")
    return global_min, global_max, year2_end


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: Build training matrix with REAL cross-sectional labels
# ─────────────────────────────────────────────────────────────────────────────

def build_training_matrix(symbols: dict[str, pd.DataFrame],
                           train_end) -> tuple:
    """
    Build pooled (X, y, y_ret) from all symbols up to train_end.

    alpha_label is computed CROSS-SECTIONALLY after pooling — at each
    timestamp, label=1 if this coin's future_ret > median of ALL coins.
    This is the correct objective (matches notebook Cell 5).

    Caps at MAX_TRAIN_ROWS for RTX 2050 VRAM safety.
    """
    print(f"  Building training matrix up to {pd.Timestamp(train_end).date()}...")

    symbol_dfs = []
    for sym, df in symbols.items():
        try:
            subset = df[df.index < train_end].copy()
            if len(subset) < HORIZON_BARS + 50:
                continue
            if "future_ret" not in subset.columns:
                continue

            # Fill missing feature columns with 0
            for col in FEATURE_COLS:
                if col not in subset.columns:
                    subset[col] = 0.0

            subset["_symbol"] = sym
            symbol_dfs.append(subset[FEATURE_COLS + ["future_ret", "_symbol"]])
        except Exception:
            pass

    if not symbol_dfs:
        print("  ERROR: no valid symbol data found")
        return None, None, None

    pooled = pd.concat(symbol_dfs, axis=0).sort_index()
    print(f"  Pooled: {len(pooled):,} rows × {pooled['_symbol'].nunique()} symbols")

    # Cross-sectional alpha label: 1 if coin beats median at this timestamp
    pooled["alpha_label"] = (
        pooled.groupby(pooled.index)["future_ret"]
        .transform(lambda x: (x > x.median()).astype(float))
    )

    feat   = pooled[FEATURE_COLS].values.astype(np.float32)
    labels = pooled["alpha_label"].values.astype(np.float32)
    rets   = pooled["future_ret"].values.astype(np.float32)

    valid = np.isfinite(feat).all(axis=1) & np.isfinite(labels) & np.isfinite(rets)
    feat, labels, rets = feat[valid], labels[valid], rets[valid]

    if len(feat) < 50:
        print("  ERROR: fewer than 50 valid rows after cleaning")
        return None, None, None

    if len(feat) > MAX_TRAIN_ROWS:
        idx = np.random.choice(len(feat), MAX_TRAIN_ROWS, replace=False)
        idx.sort()
        feat, labels, rets = feat[idx], labels[idx], rets[idx]
        print(f"  VRAM guard: capped at {MAX_TRAIN_ROWS:,} rows")

    print(f"  Training matrix: {len(feat):,} rows × {len(FEATURE_COLS)} features  |  "
          f"Label balance: {labels.mean()*100:.1f}% positive (target ~50%)")

    del pooled, symbol_dfs
    gc.collect()
    return feat, labels, rets


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: Train model (Purged K-Fold + RobustScaler)
# ─────────────────────────────────────────────────────────────────────────────

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
            val_end   = val_start + fold_size
            if val_end > n:
                break
            yield np.arange(0, train_end), np.arange(val_start, val_end)


def train_model(X, y, y_ret, cuda_api, label=""):
    import xgboost as xgb
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    aucs, ics = [], []

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        if len(np.unique(y[val])) < 2:
            continue
        m = xgb.XGBClassifier(**make_xgb_params(cuda_api))
        m.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)
        probs = m.predict_proba(Xs[val])[:, 1]
        try:
            aucs.append(roc_auc_score(y[val], probs))
        except Exception:
            pass
        if y_ret is not None and np.isfinite(y_ret[val]).any():
            mask = np.isfinite(probs) & np.isfinite(y_ret[val])
            if mask.sum() >= 10:
                ics.append(float(stats.spearmanr(probs[mask], y_ret[val][mask])[0]))

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    mean_ic  = float(np.mean(ics))  if ics  else 0.0
    icir     = float(np.mean(ics) / (np.std(ics) + 1e-8)) if len(ics) > 1 else 0.0

    final = xgb.XGBClassifier(**make_xgb_params(cuda_api))
    split = int(len(Xs) * 0.9)
    final.fit(Xs[:split], y[:split], eval_set=[(Xs[split:], y[split:])], verbose=False)

    importance = pd.Series(final.feature_importances_, index=FEATURE_COLS,
                            name="importance").sort_values(ascending=False)
    return final, scaler, importance, mean_auc, mean_ic, icir


def train_meta_model(base_model, base_scaler, X, y, cuda_api, label="meta"):
    import xgboost as xgb
    Xs = base_scaler.transform(X)
    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    oos_preds = np.full(len(y), np.nan, dtype=np.float32)

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        if len(np.unique(y[val])) < 2:
            continue
        m_temp = xgb.XGBClassifier(**make_xgb_params(cuda_api))
        m_temp.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)
        oos_preds[val] = m_temp.predict_proba(Xs[val])[:, 1]

    valid = np.isfinite(oos_preds)
    if valid.sum() < 200:
        print(f"  [{label}] Insufficient OOS data ({valid.sum()}) — skipping meta")
        return None, None

    meta_y   = ((oos_preds[valid] >= 0.5).astype(float) == y[valid]).astype(float)
    X_meta   = np.column_stack([Xs[valid], oos_preds[valid]])
    meta_scaler = RobustScaler()
    X_meta_s = meta_scaler.fit_transform(X_meta)

    meta_params = make_xgb_params(cuda_api)
    meta_params.update(n_estimators=500, max_depth=4, min_child_weight=50)
    meta = xgb.XGBClassifier(**meta_params)
    split = int(len(X_meta_s) * 0.9)
    meta.fit(X_meta_s[:split], meta_y[:split],
             eval_set=[(X_meta_s[split:], meta_y[split:])], verbose=False)

    val_acc = float((meta.predict(X_meta_s[split:]) == meta_y[split:]).mean())
    print(f"  [{label}] Meta accuracy: {val_acc*100:.1f}% on {valid.sum():,} OOS rows")
    return meta, meta_scaler


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5+6: Walk-forward prediction + trade simulation
# ─────────────────────────────────────────────────────────────────────────────

def predict_week(model, scaler, symbols, week_start, week_end,
                 meta_model=None, meta_scaler=None):
    """Predict for all symbols within the week window using cached features."""
    predictions: dict[str, float] = {}
    actual_rets: dict[str, float] = {}
    meta_confs:  dict[str, float] = {}

    for sym, df in symbols.items():
        try:
            week_data = df[(df.index >= week_start) & (df.index < week_end)]
            if len(week_data) < 3:
                continue

            for col in FEATURE_COLS:
                if col not in week_data.columns:
                    week_data = week_data.copy()
                    week_data[col] = 0.0

            feat  = week_data[FEATURE_COLS].values.astype(np.float32)
            valid = np.isfinite(feat).all(axis=1)
            if valid.sum() < 2:
                continue

            feat_scaled = scaler.transform(feat[valid])
            probs = model.predict_proba(feat_scaled)[:, 1]
            predictions[sym] = float(probs.mean())

            # Meta-labeling confidence
            if meta_model is not None and meta_scaler is not None:
                try:
                    meta_input  = np.column_stack([feat_scaled, probs.reshape(-1, 1)])
                    meta_scaled = meta_scaler.transform(meta_input)
                    meta_probs  = meta_model.predict_proba(meta_scaled)[:, 1]
                    meta_confs[sym] = float(meta_probs.mean())
                except Exception:
                    pass

            if "future_ret" in week_data.columns:
                ret_col = week_data["future_ret"].values[valid]
                finite  = ret_col[np.isfinite(ret_col)]
                if len(finite) > 0:
                    actual_rets[sym] = float(finite.mean())

        except Exception:
            pass

    return predictions, actual_rets, meta_confs


def simulate_weekly_trades(predictions, actual_rets, prev_longs, prev_shorts,
                           meta_confs=None):
    """Position-tracked fees + meta-labeling confidence sizing."""
    if not predictions:
        return [], 0.0, set(), set()

    pred_series = pd.Series(predictions)
    ranked      = pred_series.rank(pct=True)
    cur_longs   = set(ranked[ranked >= (1 - TOP_QUANTILE)].index)
    cur_shorts  = set(ranked[ranked <= TOP_QUANTILE].index)

    trades = []
    for sym in cur_longs:
        ret  = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret): ret = 0.0
        fee  = 0.0 if sym in prev_longs else ROUND_TRIP_FEE
        meta = meta_confs.get(sym, 1.0) if meta_confs else 1.0
        trades.append({
            "symbol": sym, "signal": "BUY",
            "pred_prob": round(predictions[sym], 5),
            "pnl_percent": round((ret - fee) * meta * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "meta_size": round(meta, 4),
        })

    for sym in cur_shorts:
        ret  = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret): ret = 0.0
        fee  = 0.0 if sym in prev_shorts else ROUND_TRIP_FEE
        meta = meta_confs.get(sym, 1.0) if meta_confs else 1.0
        trades.append({
            "symbol": sym, "signal": "SELL",
            "pred_prob": round(predictions[sym], 5),
            "pnl_percent": round((-ret - fee) * meta * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "meta_size": round(meta, 4),
        })

    if trades:
        sizes    = np.array([t["meta_size"]   for t in trades])
        pnls     = np.array([t["pnl_percent"] for t in trades])
        week_ret = float(np.average(pnls, weights=sizes)) / 100
    else:
        week_ret = 0.0

    return trades, week_ret, cur_longs, cur_shorts


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        action="store_true")
    parser.add_argument("--no-gpu",     action="store_true")
    parser.add_argument("--year2-only", action="store_true",
                        help="Pretrain only (no Year 3 walk-forward)")
    parser.add_argument("--data-dir",   default=None)
    parser.add_argument("--feature-dir", default=None,
                        help="Directory containing cached feature parquet files")
    parser.add_argument("--out-dir",    default=None)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force a rebuild of the feature cache from raw data")
    parser.add_argument("--resample", default=None,
                        help="Compatibility flag. Rebuild the cache with build_feature_cache.py "
                             "if you need a different timeframe.")
    args = parser.parse_args()

    global DATA_DIR, RESULTS_DIR, CACHE_DIR
    if args.data_dir:  DATA_DIR    = args.data_dir
    if args.feature_dir: CACHE_DIR = args.feature_dir
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

    if args.resample:
        print(f"  NOTE: --resample={args.resample} is not applied by the trainer itself.")
        print("  Build a separate cache with build_feature_cache.py --resample and")
        print(f"  point --feature-dir at that cache if you need a different timeframe.\n")

    # STEP 0: Inspect / repair feature store
    print("STEP 0: Inspect feature cache\n")
    if args.rebuild_cache:
        print(f"  Rebuilding cache in {CACHE_DIR} from raw data in {DATA_DIR}...")
        if not build_feature_store():
            print("ERROR: Feature store build failed")
            return
    else:
        total_cache, valid_cache, invalid_cache = inspect_feature_store()
        if total_cache == 0:
            print(f"  No cache files found in {CACHE_DIR} - building now...")
            if not build_feature_store():
                print("ERROR: Feature store build failed")
                return
        elif invalid_cache:
            print(f"  Found {total_cache} cache files in {CACHE_DIR} "
                  f"({valid_cache} valid, {invalid_cache} stale/corrupt)")
            print("  Attempting to rebuild only the stale/corrupt cache entries...")
            if not build_feature_store():
                if valid_cache == 0:
                    print("ERROR: Feature store repair failed and no valid cache remains")
                    return
                print("  WARNING: Cache repair failed - continuing with valid cached symbols")
        else:
            print(f"  Found {valid_cache} valid cache files in {CACHE_DIR} - skipping rebuild")

    # Detect CUDA
    cuda_api = detect_cuda_api() if use_gpu else None
    if use_gpu and cuda_api is None:
        print("  WARNING: GPU requested but CUDA unavailable — using CPU")

    # STEP 1: Load feature cache
    print("\nSTEP 1: Load feature cache\n")
    symbols = load_feature_store()
    if not symbols:
        print("ERROR: No symbols loaded"); return

    # STEP 2: Dynamic date splits
    print("\nSTEP 2: Dynamic date splits\n")
    global_min, global_max, year2_end = get_date_splits(symbols)

    # For year2-only mode, still use the real year3 start but skip the loop
    year3_start = year2_end

    # STEP 3: Build training matrix
    print("\nSTEP 3: Build training matrix (cross-sectional labels)\n")
    X_train, y_train, y_ret = build_training_matrix(symbols, year3_start)
    if X_train is None:
        print("ERROR: Could not build training matrix"); return

    # STEP 4: Train base model
    os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)
    base_model_path  = f"{RESULTS_DIR}/models/model_base_y1y2.json"
    base_scaler_path = f"{RESULTS_DIR}/models/scaler_base_y1y2.pkl"

    print(f"\nSTEP 4: Train base model  "
          f"(GPU={'YES - RTX 2050 CUDA' if cuda_api else 'NO - CPU'})\n")

    if os.path.exists(base_model_path) and os.path.exists(base_scaler_path):
        print("  Loading cached base model...")
        BASE_MODEL = xgb.XGBClassifier()
        BASE_MODEL.load_model(base_model_path)
        with open(base_scaler_path, "rb") as f:
            BASE_SCALER = pickle.load(f)
        print("  Loaded OK")
    else:
        t0 = time.time()
        BASE_MODEL, BASE_SCALER, importance, mean_auc, mean_ic, icir = train_model(
            X_train, y_train, y_ret, cuda_api, label="base_y1y2"
        )
        BASE_MODEL.save_model(base_model_path)
        with open(base_scaler_path, "wb") as f:
            pickle.dump(BASE_SCALER, f)
        importance.to_csv(f"{RESULTS_DIR}/feature_importance_base.csv")
        print(f"  AUC={mean_auc:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}  "
              f"({time.time()-t0:.1f}s)")
        with open(f"{RESULTS_DIR}/train_summary.json", "w") as f:
            json.dump({"mean_auc": round(mean_auc, 5), "mean_ic": round(mean_ic, 5),
                       "icir": round(icir, 5), "n_rows": int(len(X_train)),
                       "use_gpu": cuda_api is not None}, f, indent=2)

    # Meta model
    meta_model_path  = f"{RESULTS_DIR}/models/meta_model_base.pkl"
    meta_scaler_path = f"{RESULTS_DIR}/models/meta_scaler_base.pkl"
    META_MODEL = META_SCALER = None

    if os.path.exists(meta_model_path) and os.path.exists(meta_scaler_path):
        print("  Loading cached meta model...")
        with open(meta_model_path, "rb") as f: META_MODEL = pickle.load(f)
        with open(meta_scaler_path, "rb") as f: META_SCALER = pickle.load(f)
    else:
        print("  Training meta-labeling model...")
        META_MODEL, META_SCALER = train_meta_model(
            BASE_MODEL, BASE_SCALER, X_train, y_train, cuda_api, label="meta_base"
        )
        if META_MODEL is not None:
            with open(meta_model_path, "wb") as f: pickle.dump(META_MODEL, f)
            with open(meta_scaler_path, "wb") as f: pickle.dump(META_SCALER, f)

    del X_train, y_train, y_ret
    gc.collect()

    if year2_only:
        print("\nYEAR2-ONLY MODE: Done. Skipping Year 3 walk-forward.")
        with open(f"{RESULTS_DIR}/performance_year2.json", "w") as f:
            json.dump({"mode": "year2_pretrain", "gpu": cuda_api or "cpu",
                       "train_end": str(year3_start.date()),
                       "features": len(FEATURE_COLS)}, f, indent=2)
        return

    # STEP 5+6: Walk-forward Year 3
    print(f"\nSTEP 5+6: Walk-forward Year 3  "
          f"({year3_start.date()} → {global_max.date()})\n")

    weeks = pd.date_range(start=year3_start, end=global_max, freq="W-MON")
    if len(weeks) < 2:
        print("  Not enough weeks in Year 3"); return

    current_model  = BASE_MODEL
    current_scaler = BASE_SCALER
    meta_model     = META_MODEL
    meta_scaler    = META_SCALER
    retrains       = 0

    prev_longs, prev_shorts = set(), set()
    all_trades, weekly_summary, weekly_returns = [], [], []

    for week_num, (ws, we) in enumerate(zip(weeks[:-1], weeks[1:]), 1):

        # Quarterly retrain
        if week_num % RETRAIN_WEEKS == 0:
            print(f"  Week {week_num:2d}: QUARTERLY RETRAIN...")
            X_rt, y_rt, y_ret_rt = build_training_matrix(symbols, we)
            if X_rt is not None and len(X_rt) > 200:
                m_new, s_new, imp_new, auc_n, ic_n, icir_n = train_model(
                    X_rt, y_rt, y_ret_rt, cuda_api, label=f"y3_w{week_num:03d}"
                )
                current_model, current_scaler = m_new, s_new
                m_new.save_model(f"{RESULTS_DIR}/models/model_y3_week{week_num:03d}.json")
                imp_new.to_csv(f"{RESULTS_DIR}/feature_importance_y3_week{week_num:03d}.csv")
                print(f"    AUC={auc_n:.4f}  IC={ic_n:.4f}  ICIR={icir_n:.4f}")
                meta_new, meta_s_new = train_meta_model(
                    current_model, current_scaler, X_rt, y_rt, cuda_api,
                    label=f"meta_w{week_num:03d}"
                )
                if meta_new is not None:
                    meta_model, meta_scaler = meta_new, meta_s_new
                retrains += 1
                del X_rt, y_rt, y_ret_rt; gc.collect()

        # Predict
        predictions, actual_rets, meta_confs = predict_week(
            current_model, current_scaler, symbols, ws, we,
            meta_model=meta_model, meta_scaler=meta_scaler
        )

        if len(predictions) < 5:
            continue

        # Trade simulation
        trades, week_ret, cur_longs, cur_shorts = simulate_weekly_trades(
            predictions, actual_rets, prev_longs, prev_shorts, meta_confs
        )
        weekly_returns.append(week_ret)

        # IC
        common   = [s for s in predictions if s in actual_rets]
        if len(common) > 10:
            pred_arr = np.array([predictions[s] for s in common])
            ret_arr  = np.array([actual_rets[s]  for s in common])
            week_ic  = float(stats.spearmanr(pred_arr, ret_arr)[0])
        else:
            week_ic = 0.0

        n_cur      = len(cur_longs) + len(cur_shorts)
        n_new      = len(cur_longs - prev_longs) + len(cur_shorts - prev_shorts)
        turnover   = round(n_new / n_cur * 100, 1) if n_cur > 0 else 100.0
        prev_longs, prev_shorts = cur_longs, cur_shorts

        for t in trades:
            t["week"] = week_num
            t["week_start"] = str(ws.date())
        all_trades.extend(trades)

        ann_proj = ((1 + week_ret) ** 52 - 1) * 100
        weekly_summary.append({
            "week": week_num, "week_start": str(ws.date()), "week_end": str(we.date()),
            "n_symbols": len(predictions), "n_trades": len(trades),
            "week_return_pct": round(week_ret * 100, 4),
            "annualised_pct": round(ann_proj, 2),
            "ic": round(week_ic, 5),
            "turnover_pct": turnover,
            "on_track": week_ret >= ((1.10) ** (1/52) - 1),
        })

        if week_num % 4 == 0 or week_num <= 2:
            rolling = np.mean(weekly_returns[-4:]) * 100
            print(f"  Week {week_num:2d}: {len(trades):4d} trades | "
                  f"ret={week_ret*100:+.2f}% | IC={week_ic:+.4f} | "
                  f"4w_avg={rolling:+.2f}% | TO={turnover:.0f}%")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trades_df  = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(weekly_summary)

    if len(trades_df)  > 0: trades_df.to_csv(f"{RESULTS_DIR}/all_trades_year3.csv",     index=False)
    if len(summary_df) > 0: summary_df.to_csv(f"{RESULTS_DIR}/weekly_summary_year3.csv", index=False)

    n_wks   = len(weekly_returns)
    cum_ret = float(np.prod([1 + r for r in weekly_returns]) - 1) if n_wks else 0.0
    ann_ret = ((1 + cum_ret) ** (52 / n_wks) - 1) * 100 if n_wks else 0.0
    wk_std  = float(np.std(weekly_returns)) if n_wks > 1 else 0.0
    sharpe  = float(np.mean(weekly_returns)) / wk_std * np.sqrt(52) if wk_std > 0 else 0.0

    ic_s    = summary_df["ic"] if len(summary_df) > 0 else pd.Series(dtype=float)
    ic_mean = float(ic_s.mean()) if len(ic_s) > 0 else 0.0
    ic_std  = float(ic_s.std())  if len(ic_s) > 1 else 0.0
    icir    = ic_mean / (ic_std + 1e-8)
    ic_pos  = float((ic_s > 0).mean() * 100) if len(ic_s) > 0 else 0.0

    perf = {
        "label": "Year3_WalkForward_v3",
        "total_weeks": n_wks, "total_trades": len(trades_df), "retrains": retrains,
        "total_return_pct": round(cum_ret * 100, 2),
        "annualised_pct": round(ann_ret, 2),
        "sharpe": round(sharpe, 4),
        "ic_mean": round(ic_mean, 5),
        "icir": round(icir, 4),
        "ic_positive_pct": round(ic_pos, 1),
        "gpu": f"CUDA ({cuda_api})" if cuda_api else "CPU",
        "features": len(FEATURE_COLS),
        "data_range": f"{global_min.date()} → {global_max.date()}",
        "train_end": str(year3_start.date()),
        "fix_notes": [
            "Full 56-feature build_features() from azalyst_factors_v2.py",
            "Dynamic date splits — no hardcoded dates",
            "Timestamp fix: integer ms → UTC, year<2018 skip",
            "alpha_label: cross-sectional after pooling (not per-symbol)",
            "future_ret column matches notebooks exactly",
        ],
    }

    with open(f"{RESULTS_DIR}/performance_year3.json", "w") as f:
        json.dump(perf, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  AZALYST v3  —  RUN COMPLETE")
    print(f"{'='*65}")
    for k, v in perf.items():
        if k not in ("fix_notes", "label"):
            print(f"  {k:<22}: {v}")
    print(f"{'='*65}")
    print(f"\n  Trades  → {RESULTS_DIR}/all_trades_year3.csv")
    print(f"  Summary → {RESULTS_DIR}/weekly_summary_year3.csv")
    print(f"  Perf    → {RESULTS_DIR}/performance_year3.json")
    print(f"  GPU used: {'RTX 2050 CUDA' if cuda_api else 'CPU'}")


if __name__ == "__main__":
    main()
