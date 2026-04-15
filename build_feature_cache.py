"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FEATURE CACHE BUILDER v2
║   72 features  |  Vectorized Factor Engine  |  Multi-process Scaling        ║
╚══════════════════════════════════════════════════════════════════════════════╝

FIXES vs original:
  - Renamed future_ret_4h → future_ret  (aligns with the current V7 workflow)
  - Removed per-symbol alpha_label computation — WRONG to compute per symbol.
    Cross-sectional labels require ALL symbols pooled together at the same
    timestamp, so they are handled inside the training pipeline, not the cache.
"""

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from azalyst_factors_v2 import build_features, FEATURE_COLS
from azalyst_tf_utils import get_tf_constants

warnings.filterwarnings("ignore")


def _process_symbol(args: Tuple) -> Tuple[str, bool, str]:
    symbol, data_dir, out_dir, resample = args
    out_path = Path(out_dir) / f"{symbol}.parquet"

    try:
        path = Path(data_dir) / f"{symbol}.parquet"
        if not path.exists():
            return symbol, False, "source parquet not found"

        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]

        # ── Ensure datetime index ─────────────────────────────────────────────
        if not isinstance(df.index, pd.DatetimeIndex):
            ts_col = next(
                (c for c in df.columns if c in ("timestamp", "time", "open_time")), None
            )
            if ts_col:
                df.index = pd.to_datetime(
                    df[ts_col],
                    unit="ms" if pd.api.types.is_integer_dtype(df[ts_col]) else None,
                    utc=True,
                )
                df = df.drop(columns=[ts_col])
            elif pd.api.types.is_integer_dtype(df.index):
                df.index = pd.to_datetime(df.index, unit='ms', utc=True)
            else:
                df.index = pd.to_datetime(df.index, utc=True)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df = df.sort_index()

        # ── 1970 timestamp check ─────────────────────────────────────────────
        if df.index.max().year < 2018:
            return symbol, False, f"1970 timestamp bug: max={df.index.max()}"

        # ── Resample if needed ────────────────────────────────────────────────
        if resample not in ("5min", "5t"):
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            df = df.resample(resample).agg(agg).dropna()

        bph, bpd, hor = get_tf_constants(resample)

        feats = build_features(df, timeframe=resample)

        # Include raw close for weekly PnL computation in the engine
        feats["close"] = df["close"].astype(np.float32)

        # ── FIX: column is now 'future_ret' (not 'future_ret_4h') ─────────────
        # This aligns with the current V7 engine and notebook.
        feats["future_ret"] = np.log(df["close"].shift(-hor) / df["close"])

        # Horizon-in-bars for each forward-return target.
        # Use tf constants to handle all timeframes correctly.
        bars_per_min = bph / 60.0
        horizon_15m = max(1, int(round(15 * bars_per_min)))
        horizon_1h = max(1, int(round(60 * bars_per_min)))
        horizon_1d = max(1, bpd)
        horizon_5d = max(1, bpd * 5)
        feats["future_ret_15m"] = np.log(
            df["close"].shift(-horizon_15m) / df["close"]
        ).astype(np.float32)
        feats["future_ret_1h"] = np.log(
            df["close"].shift(-horizon_1h) / df["close"]
        ).astype(np.float32)
        feats["future_ret_1d"] = np.log(
            df["close"].shift(-horizon_1d) / df["close"]
        ).astype(np.float32)
        feats["future_ret_5d"] = np.log(
            df["close"].shift(-horizon_5d) / df["close"]
        ).astype(np.float32)

        # ── FIX: do NOT compute alpha_label here ──────────────────────────────
        # Cross-sectional labels require all symbols at the same timestamp.
        # Per-symbol labels would just answer "did this coin go up?" which is
        # not the current V7 relative-return objective.

        # ── Save: keep only valid feature rows ───────────────────────────────
        min_non_nan = int(0.80 * len(FEATURE_COLS))
        feats = feats.dropna(subset=FEATURE_COLS, thresh=min_non_nan).astype(np.float32)
        feats.to_parquet(out_path, engine="pyarrow", compression="snappy")

        return symbol, True, f"{len(feats):,} rows saved"

    except Exception as e:
        return symbol, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Feature Cache Builder v2 — pre-compute 72 features for all symbols"
    )
    parser.add_argument("--data-dir",  required=True,  help="Directory with SYMBOL.parquet files")
    parser.add_argument("--out-dir",   default="./feature_cache", help="Output directory")
    parser.add_argument("--workers",   type=int, default=4,   help="Parallel workers (default 4)")
    parser.add_argument("--resample",  default="5min",         help="Resample string (default 5min)")
    parser.add_argument("--overwrite", action="store_true",    help="Reprocess already-cached symbols")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[ERROR] No .parquet files found in {data_dir}")
        sys.exit(1)

    symbols = [f.stem for f in parquet_files]

    # Skip already-cached unless --overwrite
    if not args.overwrite:
        symbols = [s for s in symbols if not (out_dir / f"{s}.parquet").exists()]
        cached  = len(parquet_files) - len(symbols)
        if cached:
            print(f"[Cache] {cached} symbols already cached — skipping (use --overwrite to rebuild)")

    if not symbols:
        print("[Cache] All symbols already cached. Done.")
        return

    print(f"[FeatureCache] Building cache for {len(symbols)} symbols "
          f"({args.resample}, {args.workers} workers)...")
    t0 = time.time()

    ok_count  = 0
    err_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work    = [(s, str(data_dir), str(out_dir), args.resample) for s in symbols]
        futures = {pool.submit(_process_symbol, a): a[0] for a in work}

        for i, fut in enumerate(as_completed(futures), 1):
            sym, success, msg = fut.result()
            if success:
                ok_count += 1
            else:
                err_count += 1
                print(f"  [WARN] {sym}: {msg}")

            if i % 25 == 0 or i == len(symbols):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(symbols) - i)
                print(f"  [{i}/{len(symbols)}] ok={ok_count} err={err_count} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    cached_total = len(list(out_dir.glob("*.parquet")))
    print(f"\n[FeatureCache] Done in {elapsed:.1f}s")
    print(f"  Built : {ok_count}")
    print(f"  Errors: {err_count}")
    print(f"  Total cached: {cached_total} symbols in {out_dir}")
    print()
    print("  NOTE: alpha_label is NOT stored in the cache.")
    print("  It is computed cross-sectionally after pooling all symbols during training.")


if __name__ == "__main__":
    main()
