"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FEATURE CACHE BUILDER v2
║   65 features  |  Vectorized Factor Engine  |  Multi-process Scaling        ║
╚══════════════════════════════════════════════════════════════════════════════╝
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

from azalyst_factors_v2 import compute_v2_features, FEATURE_COLS
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
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            ts_col = next((c for c in df.columns if c in ("timestamp", "time", "open_time")), None)
            if ts_col:
                df.index = pd.to_datetime(df[ts_col], unit="ms" if df[ts_col].dtype == np.int64 else None, utc=True)
                df = df.drop(columns=[ts_col])
            else:
                df.index = pd.to_datetime(df.index, utc=True)
        
        df = df.sort_index()
        
        # Resample to desired timeframe if needed
        if resample not in ('5min', '5t'):
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df = df.resample(resample).agg(agg).dropna()

        bph, bpd, hor = get_tf_constants(resample)
        
        feats = compute_v2_features(df, bph=bph, bpd=bpd)
        
        # Future returns (target)
        feats['future_ret_4h'] = np.log(df['close'].shift(-hor) / df['close'])
        feats['alpha_label']   = (feats['future_ret_4h'] > 0).astype(np.float32)
        
        # Clean and save
        feats = feats.dropna(subset=FEATURE_COLS, how="all").astype(np.float32)
        feats.to_parquet(out_path, engine="pyarrow", compression="snappy")
        
        return symbol, True, f"{len(feats):,} rows saved"

    except Exception as e:
        return symbol, False, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="./feature_cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resample", default="5min")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    symbols = [f.stem for f in parquet_files]

    print(f"Building cache for {len(symbols)} symbols...")
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work = [(s, str(data_dir), str(out_dir), args.resample) for s in symbols]
        futures = {pool.submit(_process_symbol, a): a[0] for a in work}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, success, msg = fut.result()
            if i % 10 == 0:
                print(f"[{i}/{len(symbols)}] {sym}: {'✓' if success else '✗'} {msg}")

    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
