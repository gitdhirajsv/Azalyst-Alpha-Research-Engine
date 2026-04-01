"""
audit_survivorship.py
=====================
AZALYST ALPHA RESEARCH ENGINE — Survivorship Bias Audit

STEP 08 of Session Plan — D.E. Shaw Standard:
  "Survivorship bias in crypto is severe; coins that went to zero between
   2023-2026 must be in the universe or the backtest overstates performance."

Compares data/ universe (source parquets) vs feature_cache/ (cached symbols).
For each symbol that is IN data/ but NOT in cache, classifies it as:
  - DELISTED     : last row date < cutoff_date AND price trended to ~0
  - LOW_VOLUME   : last row date < cutoff_date but price is non-zero
  - DATA_GAP     : ≤ min_rows rows (not enough history to cache)
  - CACHE_ERROR  : file exists in data but cache build failed

Survivorship score = (delisted symbols) / (total data symbols).
If > 5%, this is a material survivorship bias risk (D.E. Shaw threshold).

Usage:
    python audit_survivorship.py
        --data-dir     ./data
        --cache-dir    ./feature_cache
        --cutoff-date  2026-01-01    # symbols last active before this = suspect
        --min-rows     5000          # symbols with fewer rows than this = DATA_GAP

Output:
    survivorship_audit.csv  — per-symbol classification in data/ directory
    Prints summary table
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def classify_missing(symbol: str, data_dir: Path,
                     cutoff: pd.Timestamp, min_rows: int) -> dict:
    """Load symbol parquet and classify why it's missing from cache."""
    path = data_dir / f"{symbol}.parquet"
    result = {
        "symbol":      symbol,
        "status":      "UNKNOWN",
        "last_date":   None,
        "n_rows":      0,
        "last_close":  None,
        "notes":       "",
    }

    if not path.exists():
        result["status"] = "FILE_MISSING"
        return result

    try:
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]

        # Timestamp parse
        if not isinstance(df.index, pd.DatetimeIndex):
            ts_col = next(
                (c for c in df.columns if c in ("timestamp", "time", "open_time")),
                None,
            )
            if ts_col:
                df.index = pd.to_datetime(
                    df[ts_col],
                    unit="ms" if pd.api.types.is_integer_dtype(df[ts_col]) else None,
                    utc=True,
                )
                df = df.drop(columns=[ts_col])
            elif pd.api.types.is_integer_dtype(df.index):
                df.index = pd.to_datetime(df.index, unit="ms", utc=True)
            else:
                df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df = df.sort_index()

        n_rows = len(df)
        result["n_rows"] = n_rows
        last_date = df.index.max()
        result["last_date"] = str(last_date.date())

        last_close = float(df["close"].iloc[-1]) if "close" in df.columns else None
        result["last_close"] = last_close

        if n_rows < min_rows:
            result["status"] = "DATA_GAP"
            result["notes"] = f"only {n_rows} rows (min={min_rows})"
            return result

        if last_date < cutoff:
            # Symbol stopped trading before cutoff — likely delisted or merged
            if last_close is not None:
                # Check if price trended to near zero in the last 10% of rows
                last_10pct = df["close"].iloc[max(0, n_rows - n_rows//10):]
                min_price  = last_10pct.min()
                max_price  = df["close"].max()
                ratio = min_price / (max_price + 1e-10) if max_price > 0 else 0.0
                if ratio < 0.01:   # price lost >99% from peak
                    result["status"] = "DELISTED_ZERO"
                    result["notes"]  = f"last={last_date.date()} price→0 (ratio={ratio:.4f})"
                elif ratio < 0.10:
                    result["status"] = "DELISTED_DUMP"
                    result["notes"]  = f"last={last_date.date()} price↓90% (ratio={ratio:.4f})"
                else:
                    result["status"] = "LOW_VOLUME"
                    result["notes"]  = f"last={last_date.date()} price ok, stopped updating"
            else:
                result["status"] = "LOW_VOLUME"
                result["notes"] = f"last={last_date.date()} no close column"
        else:
            # Symbol has recent data but wasn't cached — build failure
            result["status"] = "CACHE_ERROR"
            result["notes"]  = f"data to {last_date.date()} but not in cache"

    except Exception as e:
        result["status"] = "READ_ERROR"
        result["notes"]  = str(e)[:120]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Survivorship bias audit (D.E. Shaw standard)")
    parser.add_argument("--data-dir",   default="./data",          help="Raw parquet directory")
    parser.add_argument("--cache-dir",  default="./feature_cache", help="Feature cache directory")
    parser.add_argument("--cutoff-date", default="2025-01-01",     help="Date before which a symbol is 'suspect' (YYYY-MM-DD)")
    parser.add_argument("--min-rows",   type=int, default=5000,    help="Min rows to be cache-eligible")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cutoff    = pd.Timestamp(args.cutoff_date, tz="UTC")

    data_syms  = {f.stem for f in data_dir.glob("*.parquet")}
    cache_syms = {f.stem for f in cache_dir.glob("*.parquet")}
    missing    = sorted(data_syms - cache_syms)
    present    = sorted(data_syms & cache_syms)

    print(f"\n{'='*72}")
    print(f"  AZALYST — SURVIVORSHIP BIAS AUDIT (D.E. Shaw Standard)")
    print(f"{'='*72}")
    print(f"  Data universe       : {len(data_syms)} symbols")
    print(f"  Cached symbols      : {len(cache_syms)} symbols")
    print(f"  Missing from cache  : {len(missing)} symbols ({len(missing)/len(data_syms)*100:.1f}%)")
    print(f"  Cutoff date         : {args.cutoff_date}")
    print(f"  Min rows threshold  : {args.min_rows:,}\n")

    if not missing:
        print("  ✓ All data symbols are cached — no survivorship risk.\n")
        return

    # Classify each missing symbol
    print(f"  Classifying {len(missing)} missing symbols (may take a moment)...\n")
    classifications = []
    for sym in missing:
        c = classify_missing(sym, data_dir, cutoff, args.min_rows)
        classifications.append(c)

    df_c = pd.DataFrame(classifications)

    # ── summary by status ─────────────────────────────────────────────────────
    status_counts = df_c["status"].value_counts()
    print(f"  {'Status':<22}  {'Count':>7}  {'% missing':>10}  {'% universe':>12}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*10}  {'─'*12}")
    for status, count in status_counts.items():
        pct_m = count / len(missing) * 100
        pct_u = count / len(data_syms) * 100
        print(f"  {status:<22}  {count:>7,}  {pct_m:>9.1f}%  {pct_u:>11.1f}%")

    # ── survivorship bias score ───────────────────────────────────────────────
    delisted = df_c[df_c["status"].str.startswith("DELISTED")]
    surv_pct = len(delisted) / len(data_syms) * 100
    print(f"\n  SURVIVORSHIP BIAS SCORE: {surv_pct:.2f}%  (D.E. Shaw threshold: <5%)")

    if surv_pct > 5.0:
        print(f"  ⚠️  MATERIAL SURVIVORSHIP BIAS — {len(delisted)} delisted coins excluded from backtest.")
        print(f"     These coins likely generated large NEGATIVE returns (went to zero).")
        print(f"     Excluding them OVERSTATES backtest performance. Investigate.")
    elif surv_pct > 1.0:
        print(f"  ⚠️  MINOR SURVIVORSHIP BIAS — {len(delisted)} delisted coins (<5%). Monitor.")
    else:
        print(f"  ✓  CLEAN — survivorship bias <1%.")

    # ── cache errors (fixable) ────────────────────────────────────────────────
    cache_errors = df_c[df_c["status"] == "CACHE_ERROR"]
    if len(cache_errors) > 0:
        print(f"\n  CACHE ERRORS (fixable — {len(cache_errors)} symbols have data but failed caching):")
        for _, row in cache_errors.head(10).iterrows():
            print(f"    {row['symbol']:<20}  last={row['last_date']}  {row['notes']}")
        if len(cache_errors) > 10:
            print(f"    ... and {len(cache_errors) - 10} more (see survivorship_audit.csv)")

    # ── data gap symbols ──────────────────────────────────────────────────────
    data_gaps = df_c[df_c["status"] == "DATA_GAP"]
    if len(data_gaps) > 0:
        print(f"\n  DATA GAP symbols ({len(data_gaps)} — too few rows to cache):")
        for _, row in data_gaps.head(5).iterrows():
            print(f"    {row['symbol']:<20}  rows={row['n_rows']}  {row['notes']}")
        if len(data_gaps) > 5:
            print(f"    ... and {len(data_gaps) - 5} more")

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = Path("survivorship_audit.csv")
    df_c.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*72}\n")

    # ── return code for CI integration ────────────────────────────────────────
    if surv_pct > 5.0:
        sys.exit(1)  # fail-fast if survivorship is material


if __name__ == "__main__":
    main()
