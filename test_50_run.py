"""
Quick 50-symbol test run for Azalyst v5.
Usage: python test_50_run.py [--target 1h|1d|5d] [--no-rebuild]
"""
import argparse
import json
import os
import random
import shutil
import sys

import pandas as pd
from pathlib import Path

SEED = 42
N = 50
SRC_DATA = Path("./data")
TEST_DATA = Path("./test_50_data")
TEST_CACHE = Path("./test_50_cache")
TEST_RESULTS = Path("./test_50_results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="1h", choices=["1h", "1d", "5d"])
    p.add_argument("--no-rebuild", action="store_true")
    args = p.parse_args()

    if not args.no_rebuild:
        syms = sorted(f.stem for f in SRC_DATA.glob("*.parquet"))
        must = [s for s in syms if s in ("BTCUSDT", "ETHUSDT")]
        rest = [s for s in syms if s not in must]
        random.seed(SEED)
        sample = must + random.sample(rest, N - len(must))
        print(f"Selected {len(sample)} symbols")

        for d in [TEST_DATA, TEST_CACHE, TEST_RESULTS]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        for sym in sample:
            src = SRC_DATA / f"{sym}.parquet"
            dst = TEST_DATA / f"{sym}.parquet"
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copy2(str(src), str(dst))

        data_count = len(list(TEST_DATA.glob("*.parquet")))
        print(f"Data: {data_count} files")

        rc = os.system(
            f'python build_feature_cache.py'
            f' --data-dir "{TEST_DATA}"'
            f' --out-dir "{TEST_CACHE}"'
            f' --workers 4'
        )
        if rc != 0:
            return 1

    if TEST_RESULTS.exists():
        shutil.rmtree(TEST_RESULTS)
    TEST_RESULTS.mkdir(parents=True)

    ret = os.system(
        f'python azalyst_v5_engine.py --gpu'
        f' --data-dir "{TEST_DATA}"'
        f' --feature-dir "{TEST_CACHE}"'
        f' --out-dir "{TEST_RESULTS}"'
        f' --no-resume --no-shap'
        f' --target {args.target}'
        f' --run-id test_50_{args.target}'
    )

    pl = TEST_RESULTS / "performance_v4.json"
    if pl.exists():
        perf = json.loads(pl.read_text())
        print("=" * 60)
        print(f"TEST RESULTS  target={args.target}")
        print("=" * 60)
        for k, v in perf.items():
            if isinstance(v, float):
                print(f"  {k:35s} = {v:.4f}")
            else:
                print(f"  {k:35s} = {v}")

    wl = TEST_RESULTS / "weekly_summary_v4.csv"
    if wl.exists():
        ws = pd.read_csv(wl)
        print(f"  {len(ws)} weeks")
        if "ic" in ws.columns:
            ic = pd.to_numeric(ws["ic"], errors="coerce").dropna()
            pos = (ic > 0).sum()
            print(f"  IC mean={ic.mean():.4f}  positive_weeks={pos}/{len(ic)}")
        if "week_return_pct" in ws.columns:
            r = pd.to_numeric(ws["week_return_pct"], errors="coerce").dropna()
            winrate = (r > 0).mean() * 100
            print(f"  Ret mean={r.mean():.4f}  win_rate={winrate:.1f}%")
        if "cum_return_pct" in ws.columns:
            c = pd.to_numeric(ws["cum_return_pct"], errors="coerce").dropna()
            if len(c):
                print(f"  Cumulative={c.iloc[-1]:.2f}%")
        if "regime" in ws.columns:
            print(ws["regime"].value_counts().to_string())
        if "n_trades" in ws.columns:
            t = pd.to_numeric(ws["n_trades"], errors="coerce")
            print(f"  Trades/wk avg={t.mean():.1f}  max={t.max():.0f}")

    return ret


if __name__ == "__main__":
    sys.exit(main() or 0)