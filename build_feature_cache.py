"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FEATURE CACHE BUILDER  v2
║   65 features across 8 categories  |  Timeframe-aware  |  Lookahead-safe  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v2 CHANGES vs v1:                                                         ║
║   - 65 features (was 27): +WorldQuant, +Garman-Klass, +ADX, +Kyle lambda  ║
║   - Returns: ret_2d, ret_3d, ret_1w added                                  ║
║   - Volume: OBV change, VPT change, volume momentum added                  ║
║   - Volatility: ATR norm, Parkinson, Garman-Klass added                    ║
║   - Technical: MACD hist, stoch_k/d, CCI, ADX, DMI diff added             ║
║   - Microstructure: Kyle lambda, spread proxy added                        ║
║   - Price: price_accel, skew, kurt, max_ret kept; wick_top/bot kept        ║
║   - WorldQuant-inspired: 8 new cross-sectional alpha signals               ║
║   - Regime: vol_regime, trend_strength, BTC corr, Hurst, FFT              ║
║  TF FIX (v1.1): all rolling windows derived from get_tf_constants()        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache
    python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache --workers 4
    python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache --max-symbols 10
"""

from __future__ import annotations

import argparse
import os
import re
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
BARS_PER_HOUR = 12
BARS_PER_DAY  = 288
BARS_PER_WEEK = 2016

# All 65 v2 feature columns
FEATURE_COLS = [
    # Returns (7)
    "ret_1bar", "ret_1h", "ret_4h", "ret_1d", "ret_2d", "ret_3d", "ret_1w",
    # Volume (6)
    "vol_ratio", "vol_ret_1h", "vol_ret_1d", "obv_change", "vpt_change", "vol_momentum",
    # Volatility (7)
    "rvol_1h", "rvol_4h", "rvol_1d", "vol_ratio_1h_1d",
    "atr_norm", "parkinson_vol", "garman_klass",
    # Technical (10)
    "rsi_14", "rsi_6", "macd_hist", "bb_pos", "bb_width",
    "stoch_k", "stoch_d", "cci_14", "adx_14", "dmi_diff",
    # Microstructure (6)
    "vwap_dev", "amihud", "kyle_lambda", "spread_proxy", "body_ratio", "candle_dir",
    # Price structure (6)
    "wick_top", "wick_bot", "price_accel", "skew_1d", "kurt_1d", "max_ret_4h",
    # WorldQuant-inspired alphas (8)
    "wq_alpha001", "wq_alpha012", "wq_alpha031", "wq_alpha098",
    "cs_momentum", "cs_reversal", "vol_adjusted_mom", "trend_consistency",
    # Regime (5)
    "vol_regime", "trend_strength", "corr_btc_proxy", "hurst_exp", "fft_strength",
]

MIN_ROWS_REQUIRED = BARS_PER_WEEK


# ─────────────────────────────────────────────────────────────────────────────
#  TIMEFRAME UTILITY  (TF fix — all windows derived dynamically)
# ─────────────────────────────────────────────────────────────────────────────

def get_tf_constants(resample_str: str) -> tuple:
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
#  RSI / EMA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s: pd.Series, n: int) -> pd.Series:
    d  = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _true_range(high, low, close) -> pd.Series:
    prev_c = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE BUILDER  (65 features, TF-aware)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, resample: str = '5min') -> pd.DataFrame:
    """
    Compute all 65 v2 ML features from OHLCV.
    All windows scale with `resample` via get_tf_constants().
    Shifted +1 bar — no same-bar lookahead.
    """
    bph, bpd, hor = get_tf_constants(resample)

    c = df["close"]; o = df["open"]
    h = df["high"];  l = df["low"]
    v = df["volume"]

    f  = pd.DataFrame(index=df.index)
    lr = np.log(c / c.shift(1))

    # ── 1. RETURNS (7) ────────────────────────────────────────────────────────
    f["ret_1bar"] = lr
    f["ret_1h"]   = np.log(c / c.shift(bph))
    f["ret_4h"]   = np.log(c / c.shift(bph * 4))
    f["ret_1d"]   = np.log(c / c.shift(bpd))
    f["ret_2d"]   = np.log(c / c.shift(bpd * 2))
    f["ret_3d"]   = np.log(c / c.shift(bpd * 3))
    f["ret_1w"]   = np.log(c / c.shift(bpd * 7))

    # ── 2. VOLUME (6) ─────────────────────────────────────────────────────────
    av = v.rolling(bpd, min_periods=max(2, bph)).mean()
    f["vol_ratio"]   = v / av.replace(0, np.nan)
    f["vol_ret_1h"]  = np.log(v / v.shift(bph).replace(0, np.nan))
    f["vol_ret_1d"]  = np.log(v / v.shift(bpd).replace(0, np.nan))
    # OBV change: cumulative buy/sell pressure
    obv = (np.sign(lr) * v).cumsum()
    f["obv_change"]  = obv.pct_change(bph)
    # VPT: price-weighted volume pressure
    vpt = ((lr) * v).cumsum()
    f["vpt_change"]  = vpt.pct_change(bph)
    # Volume momentum: acceleration of volume
    f["vol_momentum"] = v.rolling(bph, min_periods=2).mean().pct_change(bph)

    # ── 3. VOLATILITY (7) ─────────────────────────────────────────────────────
    f["rvol_1h"]         = lr.rolling(bph,     min_periods=max(2, bph // 2)).std()
    f["rvol_4h"]         = lr.rolling(bph * 4, min_periods=max(2, bph)).std()
    f["rvol_1d"]         = lr.rolling(bpd,     min_periods=max(2, bph)).std()
    f["vol_ratio_1h_1d"] = f["rvol_1h"] / f["rvol_1d"].replace(0, np.nan)
    # ATR normalised
    tr = _true_range(h, l, c)
    f["atr_norm"] = tr.rolling(bpd, min_periods=max(2, bph)).mean() / c.replace(0, np.nan)
    # Parkinson vol: uses High-Low range — less noisy than close-to-close
    hl_log = np.log(h / l.replace(0, np.nan))
    f["parkinson_vol"] = (hl_log ** 2 / (4 * np.log(2))).rolling(
        bpd, min_periods=max(2, bph)).mean().pow(0.5)
    # Garman-Klass vol: most efficient OHLC estimator
    co_log = np.log(c / o.replace(0, np.nan))
    hl_log2 = np.log(h / l.replace(0, np.nan))
    gk = 0.5 * hl_log2 ** 2 - (2 * np.log(2) - 1) * co_log ** 2
    f["garman_klass"] = gk.rolling(bpd, min_periods=max(2, bph)).mean().pow(0.5)

    # ── 4. TECHNICAL (10) ─────────────────────────────────────────────────────
    f["rsi_14"] = _rsi(c, 14) / 100.0
    f["rsi_6"]  = _rsi(c,  6) / 100.0
    # MACD histogram
    ema12 = _ema(c, 12); ema26 = _ema(c, 26)
    macd  = ema12 - ema26
    sig   = _ema(macd, 9)
    f["macd_hist"] = (macd - sig) / c.replace(0, np.nan)
    # Bollinger Bands
    ma  = c.rolling(20, min_periods=10).mean()
    std = c.rolling(20, min_periods=10).std(ddof=0)
    bw  = (4 * std).replace(0, np.nan)
    f["bb_pos"]   = ((c - (ma - 2 * std)) / bw).clip(0, 1)
    f["bb_width"] = bw / ma.replace(0, np.nan)
    # Stochastic %K and %D
    lo_n = l.rolling(bph * 2, min_periods=max(2, bph)).min()
    hi_n = h.rolling(bph * 2, min_periods=max(2, bph)).max()
    stoch_k = (c - lo_n) / (hi_n - lo_n).replace(0, np.nan)
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean()
    # CCI (Commodity Channel Index)
    tp  = (h + l + c) / 3
    tp_ma  = tp.rolling(bph, min_periods=max(2, bph // 2)).mean()
    tp_mad = tp.rolling(bph, min_periods=max(2, bph // 2)).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    f["cci_14"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    f["cci_14"] = f["cci_14"] / 200.0  # normalise to ~[-1, 1]
    # ADX (Average Directional Index) — trend strength
    plus_dm  = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14    = tr.ewm(span=bph, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=bph, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=bph, adjust=False).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    f["adx_14"]  = dx.ewm(span=bph, adjust=False).mean() / 100.0
    f["dmi_diff"] = (plus_di - minus_di) / 100.0

    # ── 5. MICROSTRUCTURE (6) ─────────────────────────────────────────────────
    # VWAP deviation
    tpv_sum = (tp * v).rolling(bpd, min_periods=max(2, bph)).sum()
    v_sum   = v.rolling(bpd, min_periods=max(2, bph)).sum().replace(0, np.nan)
    vwap    = tpv_sum / v_sum
    f["vwap_dev"]     = (c - vwap) / c.replace(0, np.nan)
    # Amihud illiquidity
    f["amihud"]       = (lr.abs() / v.replace(0, np.nan)).rolling(
        bpd, min_periods=max(2, bph)).mean()
    # Kyle lambda (price impact per unit volume)
    # λ = |Δp| / volume — rolling estimate
    dp = c.diff().abs()
    f["kyle_lambda"]  = (dp / v.replace(0, np.nan)).rolling(
        bph * 4, min_periods=max(2, bph)).mean()
    # Spread proxy from High-Low
    f["spread_proxy"] = (h - l) / ((h + l) / 2).replace(0, np.nan)
    # Candle structure
    rng = (h - l).replace(0, np.nan)
    f["body_ratio"]   = (c - o).abs() / rng
    f["candle_dir"]   = np.sign(c - o)

    # ── 6. PRICE STRUCTURE (6) ────────────────────────────────────────────────
    f["wick_top"]     = (h - c.clip(lower=o)) / rng
    f["wick_bot"]     = (c.clip(upper=o) - l) / rng
    m1 = c.pct_change(bph)
    f["price_accel"]  = m1 - m1.shift(bph)
    f["skew_1d"]      = lr.rolling(bpd, min_periods=max(4, bph)).skew()
    f["kurt_1d"]      = lr.rolling(bpd, min_periods=max(4, bph)).kurt()
    f["max_ret_4h"]   = lr.rolling(bph * 4, min_periods=max(2, bph)).max()

    # ── 7. WORLDQUANT-INSPIRED ALPHAS (8) ─────────────────────────────────────
    # Alpha001: vol-adjusted momentum (from 101 Formulaic Alphas)
    signed_ret = np.where(f["rvol_1d"] < f["rvol_1d"].shift(bph), -lr, lr)
    f["wq_alpha001"] = pd.Series(signed_ret, index=df.index).rolling(
        bpd, min_periods=max(2, bph)).sum()

    # Alpha012: volume × sign of 1-bar return (volume-price confirmation)
    f["wq_alpha012"] = np.sign(lr.diff()) * (-lr.diff())

    # Alpha031: negative correlation of price change and volume change
    pch = c.pct_change()
    vch = v.pct_change()
    alpha031 = -(pch.rolling(bph, min_periods=2).corr(vch))
    f["wq_alpha031"] = alpha031

    # Alpha098: Sharpe-like momentum ratio
    ret_sum = lr.rolling(bpd, min_periods=max(2, bph)).sum()
    ret_std = lr.rolling(bpd, min_periods=max(2, bph)).std().replace(0, np.nan)
    f["wq_alpha098"] = ret_sum / ret_std

    # CS momentum: ratio of recent to longer-term returns
    f["cs_momentum"] = f["ret_4h"] / f["ret_1d"].replace(0, np.nan)

    # CS reversal: negative 1H return (short-term mean reversion signal)
    f["cs_reversal"] = -f["ret_1h"]

    # Volume-adjusted momentum: momentum scaled by relative volume
    f["vol_adjusted_mom"] = f["ret_4h"] * (f["vol_ratio"].clip(0.1, 10))

    # Trend consistency: fraction of bars with positive return in rolling window
    s = np.sign(lr)
    ct48 = max(2, bph * 4)
    f["trend_consistency"] = s.rolling(ct48, min_periods=max(2, ct48 // 2)).mean()

    # ── 8. REGIME FEATURES (5) ────────────────────────────────────────────────
    # Volatility regime: current vol vs 30-day average vol
    rvol_long = lr.rolling(bpd * 30, min_periods=bpd).std()
    f["vol_regime"] = f["rvol_1d"] / rvol_long.replace(0, np.nan)

    # Trend strength: abs value of normalised return over window
    f["trend_strength"] = f["ret_1d"].abs() / f["rvol_1d"].replace(0, np.nan)

    # BTC correlation proxy: rolling correlation of price with its own 4H lag
    # (cross-sectional BTC proxy using auto-correlation when BTC not available)
    f["corr_btc_proxy"] = lr.rolling(bph * 4, min_periods=max(2, bph)).corr(
        lr.shift(bph * 4))

    # Hurst exponent proxy: ratio of range to std (R/S approximation)
    rolling_range = c.rolling(bpd, min_periods=max(2, bph)).max() - \
                    c.rolling(bpd, min_periods=max(2, bph)).min()
    rolling_std   = c.rolling(bpd, min_periods=max(2, bph)).std().replace(0, np.nan)
    rs_ratio      = rolling_range / rolling_std
    # H = log(R/S) / log(T); T = bpd
    import math
    f["hurst_exp"] = np.log(rs_ratio.replace(0, np.nan) + 1e-10) / math.log(bpd)

    # FFT strength: dominant frequency amplitude (cycle detection)
    # Rolling FFT — compute as ratio of max frequency component to DC
    def _fft_strength(arr):
        if len(arr) < 4 or np.all(np.isnan(arr)):
            return np.nan
        arr = arr[~np.isnan(arr)]
        if len(arr) < 4:
            return np.nan
        fft_vals = np.abs(np.fft.rfft(arr - arr.mean()))
        if len(fft_vals) < 2 or fft_vals[0] == 0:
            return np.nan
        return float(fft_vals[1:].max() / (fft_vals[0] + 1e-10))

    fft_win = max(bph * 4, 16)
    f["fft_strength"] = lr.rolling(fft_win, min_periods=max(4, fft_win // 2)).apply(
        _fft_strength, raw=True)

    # ── CLEANUP ───────────────────────────────────────────────────────────────
    f = f.replace([np.inf, -np.inf], np.nan)
    return f.shift(1)   # +1 bar shift — no same-bar lookahead


def compute_targets(df: pd.DataFrame, resample: str = '5min') -> pd.DataFrame:
    """Forward return targets. Training labels only — never use as features."""
    _, _, hor = get_tf_constants(resample)
    _, bpd, _ = get_tf_constants(resample)
    c = df["close"]
    t = pd.DataFrame(index=df.index)
    t["future_ret_4h"] = np.log(c.shift(-hor)  / c)
    t["future_ret_1d"] = np.log(c.shift(-bpd)  / c)
    t["label_4h"]      = (t["future_ret_4h"] > 0).astype(float)
    t["label_1d"]      = (t["future_ret_1d"] > 0).astype(float)
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  PER-SYMBOL WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _process_symbol(args: Tuple) -> Tuple[str, bool, str]:
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
            (c for c in df.columns if c in ("timestamp", "time", "open_time")), None)
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

        if resample not in ('5min', '5t'):
            agg = {'open': 'first', 'high': 'max', 'low': 'min',
                   'close': 'last', 'volume': 'sum'}
            df = df.resample(resample, label='left', closed='left').agg(agg).dropna()

        if len(df) < 50:
            return symbol, False, f"too few rows ({len(df)}) after resample"

        feats   = compute_features(df, resample=resample)
        targets = compute_targets(df, resample=resample)
        result  = feats.join(targets, how="inner")

        result.insert(0, "symbol", symbol)
        # Cast to float32 — halves file size
        float_cols = [c for c in result.columns if c != "symbol"]
        result[float_cols] = result[float_cols].astype("float32")

        avail = [c for c in FEATURE_COLS if c in result.columns]
        result = result.dropna(subset=avail, how="all")

        if len(result) < 50:
            return symbol, False, "too few valid rows after dropna"

        result.to_parquet(out_path, engine="pyarrow", compression="snappy")
        return symbol, True, f"{len(result):,} rows, {len(avail)} features"

    except Exception as e:
        return symbol, False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Feature Cache Builder v2 — 65 features"
    )
    parser.add_argument("--data-dir",    default="./data")
    parser.add_argument("--out-dir",     default="./feature_cache")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--resample",    default="5min",
                        help="Candle timeframe (default: 5min)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[ERROR] No parquet files in {data_dir}"); sys.exit(1)

    symbols = [f.stem for f in parquet_files]
    symbols = [s for s in symbols if s.endswith("USDT") and len(s) > 5]
    if not symbols:
        symbols = [f.stem for f in parquet_files]   # fallback: all files

    if args.max_symbols:
        symbols = symbols[:args.max_symbols]

    if args.overwrite:
        for f in out_dir.glob("*.parquet"):
            f.unlink()

    bph, bpd, hor = get_tf_constants(args.resample)
    print("╔══════════════════════════════════════════════════════════════╗")
    print("         AZALYST  —  FEATURE CACHE BUILDER v2  (65 features)")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Source   : {data_dir.resolve()}")
    print(f"  Output   : {out_dir.resolve()}")
    print(f"  Resample : {args.resample}  (bph={bph}, bpd={bpd}, horizon={hor})")
    print(f"  Symbols  : {len(symbols)}")
    print(f"  Workers  : {args.workers}")
    print(f"  Features : {len(FEATURE_COLS)}")
    print()

    t0 = time.time()
    ok_count = err_count = skip_count = 0

    work_args = [(sym, str(data_dir), str(out_dir), args.resample)
                 for sym in symbols]

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

            if len(symbols) <= 20 or i % 10 == 0 or not success:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 1
                eta  = (len(symbols) - i) / rate
                print(f"  [{i:>4}/{len(symbols)}] {status} {sym:<20} | "
                      f"{msg}  [ETA {eta/60:.1f}m]")

    elapsed = time.time() - t0
    cached  = list(out_dir.glob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in cached) / 1e6
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Succeeded : {ok_count}")
    print(f"  Skipped   : {skip_count}")
    print(f"  Failed    : {err_count}")
    print(f"  Cache     : {len(cached)} files, {total_mb:.1f} MB")
    print(f"  Features  : {len(FEATURE_COLS)} per symbol")
    print(f"\n  Cache ready → {out_dir.resolve()}")
    print(f"\n  Next step:")
    print(f"    python azalyst_train_local.py --feature-dir {out_dir} --out-dir ./results --gpu")


if __name__ == "__main__":
    main()
