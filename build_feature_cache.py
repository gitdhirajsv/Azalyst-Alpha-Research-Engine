"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FEATURE CACHE BUILDER
║        Precompute ML features once  |  5-20x simulation speedup            ║
║        Parallel  |  Streaming  |  Lookahead-Safe  |  pyarrow parquet       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FIX (timeframe-aware):                                                    ║
║    All rolling windows now derived via get_tf_constants(resample_str).     ║
║    Previously hardcoded to 5-min (bph=12, bpd=288) — any other candle     ║
║    size caused complete NaN flooding of features.                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

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
#  TIMEFRAME CONSTANTS  (FIX: derived dynamically, not hardcoded)
# ─────────────────────────────────────────────────────────────────────────────
# Default 5-min constants — kept as module-level globals for backwards compat
# but compute_features() now overrides them via get_tf_constants().
BARS_PER_HOUR = 12
BARS_PER_DAY  = 288
BARS_PER_WEEK = 2016

MIN_ROWS_REQUIRED = BARS_PER_WEEK   # at least 1 week of 5-min data

FEATURE_COLS = [
    "ret_1bar", "ret_1h", "ret_4h", "ret_1d",
    "vol_ratio", "vol_ret_1h", "vol_ret_1d",
    "body_ratio", "wick_top", "wick_bot", "candle_dir",
    "rvol_1h", "rvol_4h", "rvol_1d", "vol_ratio_1h_1d",
    "rsi_14", "rsi_6", "bb_pos", "bb_width",
    "vwap_dev", "ctrend_12", "ctrend_48", "price_accel",
    "skew_1d", "kurt_1d", "max_ret_4h", "amihud",
]


def get_tf_constants(resample_str: str) -> tuple[int, int, int]:
    """
    Convert pandas resample string → (bars_per_hour, bars_per_day, horizon_bars).
    horizon_bars = 4h equivalent, floored at 1.
    """
    import re
    s = resample_str.lower().strip()
    _map = {
        '1min': 1, '1t': 1, '3min': 3, '3t': 3, '5min': 5, '5t': 5,
        '15min': 15, '15t': 15, '30min': 30, '30t': 30,
        '1h': 60, '60min': 60, '60t': 60, '2h': 120, '4h': 240,
        '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '1b': 1440, 'd': 1440,
        '1w': 10080, 'w': 10080, 'w-mon': 10080, '1w-mon': 10080,
    }
    mins = _map.get(s)
    if mins is None:
        m = re.match(r'^(\d+)([a-z]+)', s)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mins = n * {'min': 1, 't': 1, 'h': 60, 'd': 1440, 'w': 10080}.get(unit, 1)
        else:
            mins = 5
    bph = max(1, 60   // mins)
    bpd = max(1, 1440 // mins)
    hor = max(1, 240  // mins)
    return bph, bpd, hor


# ─────────────────────────────────────────────────────────────────────────────
#  RSI HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s: pd.Series, n: int) -> pd.Series:
    d  = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE BUILDER  (FIX: resample param drives all window sizes)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, resample: str = '5min') -> pd.DataFrame:
    """
    Compute 27 ML features from an OHLCV DataFrame.

    FIX: All rolling windows are now derived from `resample` via
    get_tf_constants(). At 5-min (default) the output is identical to the
    original. At any other timeframe the windows scale correctly so features
    retain their semantic meaning (e.g. ret_1h really is 1-hour return).

    CRITICAL: The returned DataFrame is shifted +1 bar (no same-bar lookahead).
    """
    bph, bpd, hor = get_tf_constants(resample)

    c = df["close"]; o = df["open"]
    h = df["high"];  l = df["low"]   # noqa: E741
    v = df["volume"]

    f = pd.DataFrame(index=df.index)

    # ── Return features ────────────────────────────────────────────────────
    lr = np.log(c / c.shift(1))
    f["ret_1bar"] = lr
    f["ret_1h"]   = np.log(c / c.shift(bph))
    f["ret_4h"]   = np.log(c / c.shift(bph * 4))
    f["ret_1d"]   = np.log(c / c.shift(bpd))

    # ── Volume features ────────────────────────────────────────────────────
    av = v.rolling(bpd, min_periods=max(2, bph)).mean()
    f["vol_ratio"]  = v / av.replace(0, np.nan)
    f["vol_ret_1h"] = np.log(v / v.shift(bph).replace(0, np.nan))
    f["vol_ret_1d"] = np.log(v / v.shift(bpd).replace(0, np.nan))

    # ── Candle structure ───────────────────────────────────────────────────
    rng = (h - l).replace(0, np.nan)
    f["body_ratio"] = (c - o).abs() / rng
    f["wick_top"]   = (h - c.clip(lower=o)) / rng
    f["wick_bot"]   = (c.clip(upper=o) - l) / rng
    f["candle_dir"] = np.sign(c - o)

    # ── Volatility features ────────────────────────────────────────────────
    f["rvol_1h"]         = lr.rolling(bph,     min_periods=max(2, bph // 2)).std()
    f["rvol_4h"]         = lr.rolling(bph * 4, min_periods=max(2, bph)).std()
    f["rvol_1d"]         = lr.rolling(bpd,     min_periods=max(2, bph)).std()
    f["vol_ratio_1h_1d"] = f["rvol_1h"] / f["rvol_1d"].replace(0, np.nan)

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
        (tp * v).rolling(bpd, min_periods=max(2, bph)).sum()
        / v.rolling(bpd, min_periods=max(2, bph)).sum().replace(0, np.nan)
    )
    f["vwap_dev"] = (c - vwap) / c.replace(0, np.nan)

    # ── Trend signals ──────────────────────────────────────────────────────
    s = np.sign(lr)
    ct12 = max(2, bph)          # ~1h of bars
    ct48 = max(2, bph * 4)      # ~4h of bars
    f["ctrend_12"] = s.rolling(ct12, min_periods=max(2, ct12 // 2)).sum()
    f["ctrend_48"] = s.rolling(ct48, min_periods=max(2, ct48 // 2)).sum()

    m1 = c.pct_change(bph)
    f["price_accel"] = m1 - m1.shift(bph)

    # ── Higher-moment features ─────────────────────────────────────────────
    f["skew_1d"]    = lr.rolling(bpd, min_periods=max(4, bph)).skew()
    f["kurt_1d"]    = lr.rolling(bpd, min_periods=max(4, bph)).kurt()
    f["max_ret_4h"] = lr.rolling(bph * 4, min_periods=max(2, bph)).max()
    f["amihud"]     = (
        (lr.abs() / v.replace(0, np.nan))
        .rolling(bpd, min_periods=max(2, bph))
        .mean()
    )

    f = f.replace([np.inf, -np.inf], np.nan)
    # +1 bar shift — no same-bar lookahead
    return f.shift(1)


def compute_targets(df: pd.DataFrame, resample: str = '5min') -> pd.DataFrame:
    """
    Forward return targets. NEVER use as model features — training labels only.
    horizon scaled by resample so the target always represents ~4H forward return.
    """
    _, _, hor = get_tf_constants(resample)
    c  = df["close"]
    targets = pd.DataFrame(index=df.index)
    targets["future_ret_4h"] = np.log(c.shift(-hor) / c)
    # Keep a 1-day label too (scaled)
    _, bpd, _ = get_tf_constants(resample)
    targets["future_ret_1d"] = np.log(c.shift(-bpd) / c)
    targets["label_4h"] = (targets["future_ret_4h"] > 0).astype(float)
    targets["label_1d"] = (targets["future_ret_1d"] > 0).astype(float)
    return targets


# ─────────────────────────────────────────────────────────────────────────────
#  PER-SYMBOL WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _process_symbol(args: Tuple[str, str, str, str]) -> Tuple[str, bool, str]:
    """Worker: load one symbol, compute features + targets, save parquet."""
    symbol, data_dir, out_dir, resample = args
    out_path = Path(out_dir) / f"{symbol}.parquet"

    if out_path.exists():
        return symbol, True, "skipped (cached)"

    try:
        path = Path(data_dir) / f"{symbol}.parquet"
        if not path.exists():
            return symbol, False, "source parquet not found"

        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]

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

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return symbol, False, f"missing: {required - set(df.columns)}"

        df = df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()

        # If a non-5min resample is requested, resample raw data first
        if resample not in ('5min', '5t'):
            agg = {'open': 'first', 'high': 'max', 'low': 'min',
                   'close': 'last', 'volume': 'sum'}
            df = df.resample(resample, label='left', closed='left').agg(agg).dropna()

        if len(df) < 50:   # after resampling may have fewer rows
            return symbol, False, f"too few rows ({len(df)}) after resample"

        feats   = compute_features(df, resample=resample)
        targets = compute_targets(df, resample=resample)

        result = feats.join(targets, how="inner")
        result.insert(0, "symbol", symbol)
        result = result.dropna(subset=FEATURE_COLS, how="all")

        if len(result) < 50:
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
    )
    parser.add_argument("--data-dir",     default="./data")
    parser.add_argument("--out-dir",      default="./feature_cache")
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--max-symbols",  type=int, default=None)
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument(
        "--resample", default="5min",
        help="Candle timeframe for feature computation (default: 5min). "
             "Use '4h', '1D', '1W' etc for other timeframes."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[ERROR] No parquet files in {data_dir}"); sys.exit(1)

    symbols = [f.stem for f in parquet_files]
    symbols = [s for s in symbols if s.endswith("USDT") and len(s) > 5]

    if args.max_symbols:
        symbols = symbols[:args.max_symbols]

    if args.overwrite:
        for f in out_dir.glob("*.parquet"):
            f.unlink()

    bph, bpd, hor = get_tf_constants(args.resample)
    print("╔══════════════════════════════════════════════════════════════╗")
    print("         AZALYST  —  FEATURE CACHE BUILDER (FIXED)")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Source   : {data_dir.resolve()}")
    print(f"  Output   : {out_dir.resolve()}")
    print(f"  Resample : {args.resample}  (bph={bph}, bpd={bpd}, horizon={hor})")
    print(f"  Symbols  : {len(symbols)}")
    print(f"  Workers  : {args.workers}")
    print()

    t0 = time.time()
    ok_count = err_count = skip_count = 0

    work_args = [(sym, str(data_dir), str(out_dir), args.resample) for sym in symbols]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_symbol, a): a[0] for a in work_args}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, success, msg = fut.result()
            if msg.startswith("skipped"):
                skip_count += 1; status = "⏭"
            elif success:
                ok_count += 1;   status = "✓"
            else:
                err_count += 1;  status = "✗"

            if len(symbols) <= 30 or i % 10 == 0 or not success:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 1
                eta  = (len(symbols) - i) / rate
                print(f"  [{i:>4}/{len(symbols)}] {status} {sym:<20} | {msg}  [ETA {eta/60:.1f}m]")

    elapsed = time.time() - t0
    cached_files = list(out_dir.glob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in cached_files) / 1e6
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Succeeded : {ok_count}")
    print(f"  Skipped   : {skip_count}")
    print(f"  Failed    : {err_count}")
    print(f"  Cache size: {len(cached_files)} files, {total_mb:.1f} MB")
    print(f"\n  Feature cache ready → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
