"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FEATURE CACHE BUILDER
║        Precompute ML features once  |  5-20x simulation speedup            ║
║        Parallel  |  Streaming  |  Lookahead-Safe  |  pyarrow parquet       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Purpose
───────
Pre-computes all ML features and future-return targets for every symbol in
data/, and saves them as individual parquet files in feature_cache/.

This means the walk-forward simulator can load pre-computed features rather
than recomputing indicators on every training window — 5-20x faster.

Lookahead-safe design
─────────────────────
• All input features are shifted +1 bar so they only use information from
  bars that have ALREADY CLOSED by the time the prediction is made.
• Future-return targets (future_ret_4h, future_ret_1d) use close.shift(-N)
  meaning they represent what happens AFTER the current bar — used only as
  training labels, never as model inputs.

Output schema (per symbol parquet)
───────────────────────────────────
timestamp | symbol | ret_1bar | ret_1h | … (27 features) | future_ret_4h | future_ret_1d

Usage
─────
  python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache
  python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache --workers 8
  python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache --max-symbols 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BARS_PER_HOUR = 12    # 5-min bars per hour
BARS_PER_DAY  = 288   # 5-min bars per day
BARS_PER_WEEK = 2016  # 5-min bars per week

HORIZON_4H  = BARS_PER_HOUR * 4   # 48 bars  — short-term prediction
HORIZON_1D  = BARS_PER_DAY        # 288 bars — medium-term prediction

MIN_ROWS_REQUIRED = BARS_PER_WEEK  # at least 1 week of data

# ─────────────────────────────────────────────────────────────────────────────
#  RSI HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE BUILDER  (27 features, all shifted +1 bar — no lookahead)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ret_1bar", "ret_1h", "ret_4h", "ret_1d",
    "vol_ratio", "vol_ret_1h", "vol_ret_1d",
    "body_ratio", "wick_top", "wick_bot", "candle_dir",
    "rvol_1h", "rvol_4h", "rvol_1d", "vol_ratio_1h_1d",
    "rsi_14", "rsi_6", "bb_pos", "bb_width",
    "vwap_dev", "ctrend_12", "ctrend_48", "price_accel",
    "skew_1d", "kurt_1d", "max_ret_4h", "amihud",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 27 ML features from an OHLCV DataFrame.

    CRITICAL: The returned feature DataFrame is shifted +1 bar so that
    at row t, all features reflect information up to and including bar t-1.
    This prevents any same-bar lookahead.
    """
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]    # noqa: E741
    v = df["volume"]

    f = pd.DataFrame(index=df.index)

    # ── Return features ────────────────────────────────────────────────────
    lr = np.log(c / c.shift(1))
    f["ret_1bar"] = lr
    f["ret_1h"]   = np.log(c / c.shift(BARS_PER_HOUR))
    f["ret_4h"]   = np.log(c / c.shift(BARS_PER_HOUR * 4))
    f["ret_1d"]   = np.log(c / c.shift(BARS_PER_DAY))

    # ── Volume features ────────────────────────────────────────────────────
    av = v.rolling(BARS_PER_DAY, min_periods=BARS_PER_HOUR).mean()
    f["vol_ratio"]   = v / av.replace(0, np.nan)
    f["vol_ret_1h"]  = np.log(v / v.shift(BARS_PER_HOUR).replace(0, np.nan))
    f["vol_ret_1d"]  = np.log(v / v.shift(BARS_PER_DAY).replace(0, np.nan))

    # ── Candle structure ───────────────────────────────────────────────────
    rng = (h - l).replace(0, np.nan)
    f["body_ratio"] = (c - o).abs() / rng
    f["wick_top"]   = (h - c.clip(lower=o)) / rng
    f["wick_bot"]   = (c.clip(upper=o) - l) / rng
    f["candle_dir"] = np.sign(c - o)

    # ── Volatility features ────────────────────────────────────────────────
    f["rvol_1h"]          = lr.rolling(BARS_PER_HOUR, min_periods=6).std()
    f["rvol_4h"]          = lr.rolling(BARS_PER_HOUR * 4, min_periods=12).std()
    f["rvol_1d"]          = lr.rolling(BARS_PER_DAY, min_periods=BARS_PER_HOUR).std()
    f["vol_ratio_1h_1d"]  = f["rvol_1h"] / f["rvol_1d"].replace(0, np.nan)

    # ── Oscillators ────────────────────────────────────────────────────────
    f["rsi_14"] = _rsi(c, 14) / 100.0
    f["rsi_6"]  = _rsi(c,  6) / 100.0

    # ── Bollinger Bands ────────────────────────────────────────────────────
    ma  = c.rolling(20, min_periods=10).mean()
    std = c.rolling(20, min_periods=10).std(ddof=0)
    bw  = (4 * std).replace(0, np.nan)
    f["bb_pos"]   = ((c - (ma - 2 * std)) / bw).clip(0, 1)
    f["bb_width"] = bw / ma.replace(0, np.nan)

    # ── VWAP deviation ─────────────────────────────────────────────────────
    tp   = (h + l + c) / 3
    vwap = (
        (tp * v).rolling(BARS_PER_DAY, min_periods=BARS_PER_HOUR).sum()
        / v.rolling(BARS_PER_DAY, min_periods=BARS_PER_HOUR).sum().replace(0, np.nan)
    )
    f["vwap_dev"] = (c - vwap) / c.replace(0, np.nan)

    # ── Trend signals ──────────────────────────────────────────────────────
    s = np.sign(lr)
    f["ctrend_12"] = s.rolling(12, min_periods=6).sum()
    f["ctrend_48"] = s.rolling(48, min_periods=24).sum()
    m1 = c.pct_change(BARS_PER_HOUR)
    f["price_accel"] = m1 - m1.shift(BARS_PER_HOUR)

    # ── Higher-moment features ─────────────────────────────────────────────
    f["skew_1d"]    = lr.rolling(BARS_PER_DAY, min_periods=BARS_PER_HOUR).skew()
    f["kurt_1d"]    = lr.rolling(BARS_PER_DAY, min_periods=BARS_PER_HOUR).kurt()
    f["max_ret_4h"] = lr.rolling(BARS_PER_HOUR * 4, min_periods=BARS_PER_HOUR).max()
    f["amihud"]     = (
        (lr.abs() / v.replace(0, np.nan))
        .rolling(BARS_PER_DAY, min_periods=BARS_PER_HOUR)
        .mean()
    )

    # ── Clean up ───────────────────────────────────────────────────────────
    f = f.replace([np.inf, -np.inf], np.nan)

    # ── SHIFT +1 BAR to prevent same-bar lookahead ─────────────────────────
    # At time t, features now reflect information through t-1 (past data only)
    f = f.shift(1)

    return f


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward return targets. These use FUTURE prices for labelling only.
    They must NEVER be used as model features — only as training targets.

    future_ret_4h  = log(close[t+48]  / close[t])
    future_ret_1d  = log(close[t+288] / close[t])
    """
    c  = df["close"]
    lr = np.log(c / c.shift(1))

    targets = pd.DataFrame(index=df.index)
    # Rolling sum of log returns over the horizon = cumulative log return
    targets["future_ret_4h"] = lr.shift(-HORIZON_4H).rolling(
        HORIZON_4H, min_periods=HORIZON_4H // 2
    ).sum().shift(-(HORIZON_4H - 1))
    targets["future_ret_1d"] = lr.shift(-HORIZON_1D).rolling(
        HORIZON_1D, min_periods=HORIZON_1D // 2
    ).sum().shift(-(HORIZON_1D - 1))

    # Simpler, more direct: just the N-bar forward return
    targets["future_ret_4h"] = np.log(c.shift(-HORIZON_4H) / c)
    targets["future_ret_1d"] = np.log(c.shift(-HORIZON_1D) / c)

    # Binary labels: 1 if positive return, 0 if negative
    targets["label_4h"] = (targets["future_ret_4h"] > 0).astype(float)
    targets["label_1d"] = (targets["future_ret_1d"] > 0).astype(float)

    return targets


# ─────────────────────────────────────────────────────────────────────────────
#  PER-SYMBOL WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _process_symbol(args: Tuple[str, str, str]) -> Tuple[str, bool, str]:
    """
    Worker function: load one symbol, compute features + targets, save parquet.
    Returns (symbol, success, message).
    """
    symbol, data_dir, out_dir = args
    out_path = Path(out_dir) / f"{symbol}.parquet"

    # Skip if already cached
    if out_path.exists():
        return symbol, True, "skipped (cached)"

    try:
        parquet_path = Path(data_dir) / f"{symbol}.parquet"
        if not parquet_path.exists():
            return symbol, False, "source parquet not found"

        df = pd.read_parquet(parquet_path)
        df.columns = [c.lower() for c in df.columns]

        # Normalise timestamp index
        ts_col = next(
            (c for c in df.columns if c in ("timestamp", "time", "open_time")), None
        )
        if ts_col:
            col = df[ts_col]
            if pd.api.types.is_integer_dtype(col):
                df["timestamp"] = pd.to_datetime(col, unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(col, utc=True)
            if ts_col != "timestamp":
                df = df.drop(columns=[ts_col])
            df = df.set_index("timestamp")
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = pd.to_datetime(df.index, utc=True)

        df = df.sort_index()

        # Validate schema
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return symbol, False, f"missing columns: {required - set(df.columns)}"

        df = df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()

        if len(df) < MIN_ROWS_REQUIRED:
            return symbol, False, f"too few rows ({len(df)})"

        # Compute features and targets
        feats   = compute_features(df)
        targets = compute_targets(df)

        # Combine
        result = feats.join(targets, how="inner")
        result.insert(0, "symbol", symbol)

        # Drop rows with all-NaN features (warmup period)
        result = result.dropna(subset=FEATURE_COLS, how="all")

        if len(result) < 100:
            return symbol, False, "too few valid rows after dropna"

        result.to_parquet(out_path, engine="pyarrow", compression="snappy")
        return symbol, True, f"{len(result):,} rows"

    except Exception as e:
        return symbol, False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Feature Cache Builder — precompute ML features once",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache
  python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache --workers 8
  python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache --max-symbols 10 --overwrite
        """,
    )
    parser.add_argument("--data-dir",     default="./data",          help="Source parquet directory")
    parser.add_argument("--out-dir",      default="./feature_cache", help="Output feature cache directory")
    parser.add_argument("--workers",      type=int, default=4,        help="Parallel worker count")
    parser.add_argument("--max-symbols",  type=int, default=None,     help="Cap symbol count (for testing)")
    parser.add_argument("--overwrite",    action="store_true",        help="Recompute even if cache exists")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover symbols
    parket_files = sorted(data_dir.glob("*.parquet"))
    if not parket_files:
        print(f"[ERROR] No parquet files found in {data_dir}"); sys.exit(1)

    symbols = [f.stem for f in parket_files]
    # Filter non-USDT or test symbols (keep only XXXUSDT)
    symbols = [s for s in symbols if s.endswith("USDT") and len(s) > 5]

    if args.max_symbols:
        symbols = symbols[:args.max_symbols]

    # Clear cache if overwrite requested
    if args.overwrite:
        for f in out_dir.glob("*.parquet"):
            f.unlink()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("         AZALYST  —  FEATURE CACHE BUILDER")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Source   : {data_dir.resolve()}")
    print(f"  Output   : {out_dir.resolve()}")
    print(f"  Symbols  : {len(symbols)}")
    print(f"  Workers  : {args.workers}")
    print(f"  Features : {len(FEATURE_COLS)} per bar + 4 target columns")
    print(f"  Lookahead: ALL features shifted +1 bar (no lookahead)")
    print()

    t0 = time.time()
    ok_count  = 0
    err_count = 0
    skip_count = 0

    work_args = [(sym, str(data_dir), str(out_dir)) for sym in symbols]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_symbol, a): a[0] for a in work_args}

        for i, fut in enumerate(as_completed(futures), 1):
            sym, success, msg = fut.result()

            if msg.startswith("skipped"):
                skip_count += 1
                status = "⏭"
            elif success:
                ok_count += 1
                status = "✓"
            else:
                err_count += 1
                status = "✗"

            # Print every symbol or every 10 for large universes
            if len(symbols) <= 30 or i % 10 == 0 or not success:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta  = (len(symbols) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i:>4}/{len(symbols)}] {status} {sym:<20} | {msg}"
                    f"  [ETA {eta/60:.1f}m]"
                )

    elapsed = time.time() - t0
    print()
    print("─" * 62)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Succeeded : {ok_count}")
    print(f"  Skipped   : {skip_count} (already cached)")
    print(f"  Failed    : {err_count}")
    cached_files = list(out_dir.glob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in cached_files) / 1e6
    print(f"  Cache size: {len(cached_files)} files, {total_mb:.1f} MB")
    print()
    print(f"  Feature cache ready → {out_dir.resolve()}")
    print()
    print("  Next step: python walkforward_simulator.py --data-dir ./data")


if __name__ == "__main__":
    main()
