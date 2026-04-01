"""
audit_signal_decay.py
=====================
AZALYST ALPHA RESEARCH ENGINE — Signal Decay Analysis

STEP 07 of Session Plan — Citadel Standard:
  "Signal half-life analysis is mandatory before choosing prediction horizon.
   Crypto reversal signals decay in ~1-4 hours; momentum signals in ~1-5 days."

Measures the IC of the model's rank-predictions (from all_trades_v4.csv pred_ret)
against actual close-to-close returns at increasing holding horizons:
  t+1 bar (5min), t+12 bars (1hr), t+48 bars (4hr), t+288 bars (1day)

For a genuine reversal signal:
  IC should peak at t+3 to t+12 (15min to 1hr) and decay toward 0 by t+48.
  If IC is still elevated at t+288, this is evidence of survivorship bias or
  look-ahead contamination — NOT genuine short-horizon alpha.

Usage:
    python audit_signal_decay.py
        --trades results_top6/all_trades_v4.csv
        --cache  cache_top6/
        --horizons 1,12,48,288,1440

Requires:
    - all_trades_v4.csv with columns: symbol, week, week_start, pred_ret
    - Feature cache parquet files (to compute multi-horizon returns)

Output:
    signal_decay.csv — IC at each horizon, with decay half-life estimate
    Prints summary table
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ─── helpers ──────────────────────────────────────────────────────────────────

def compute_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 5:
        return 0.0
    ic, _ = stats.spearmanr(pred[mask], actual[mask])
    return float(ic) if np.isfinite(ic) else 0.0


def load_cache_symbol(cache_dir: Path, symbol: str) -> pd.DataFrame | None:
    f = cache_dir / f"{symbol}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()
        if "close" not in df.columns:
            return None
        return df
    except Exception:
        return None


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Signal decay analysis (Citadel standard)")
    parser.add_argument(
        "--trades",
        default="results_top6/all_trades_v4.csv",
        help="Path to all_trades_v4.csv",
    )
    parser.add_argument(
        "--cache",
        default="cache_top6",
        help="Path to feature cache directory (must have close column in parquet)",
    )
    parser.add_argument(
        "--horizons",
        default="1,3,12,48,144,288,1440",
        help="Comma-separated horizon list in bars (default: 1,3,12,48,144,288,1440)",
    )
    parser.add_argument(
        "--max-weeks",
        type=int,
        default=0,
        help="Limit to first N weeks (0 = all)",
    )
    args = parser.parse_args()

    trades_path = Path(args.trades)
    cache_dir   = Path(args.cache)
    horizons    = [int(h) for h in args.horizons.split(",")]

    if not trades_path.exists():
        print(f"[ERROR] Trades file not found: {trades_path}")
        sys.exit(1)
    if not cache_dir.exists():
        print(f"[ERROR] Cache directory not found: {cache_dir}")
        sys.exit(1)

    trades = pd.read_csv(trades_path)

    # Normalise pred direction: SELL = flip pred sign
    trades["pred_signed"] = trades.apply(
        lambda r: -float(r["pred_ret"]) if r["signal"] == "SELL" else float(r["pred_ret"]),
        axis=1,
    )

    weeks = sorted(trades["week"].astype(int).unique())
    if args.max_weeks > 0:
        weeks = weeks[: args.max_weeks]

    print(f"\n{'='*72}")
    print(f"  AZALYST — SIGNAL DECAY ANALYSIS (Citadel Standard)")
    print(f"  Source : {trades_path}")
    print(f"  Cache  : {cache_dir}")
    print(f"  Weeks  : {len(weeks)}   Horizons (bars): {horizons}")
    print(f"  At 5-min frequency: 1bar=5m, 12=1hr, 48=4hr, 288=1d, 1440=5d")
    print(f"{'='*72}\n")

    # Load all unique symbols' close price series
    symbols_needed = trades["symbol"].unique().tolist()
    price_cache: dict[str, pd.Series] = {}
    for sym in symbols_needed:
        df = load_cache_symbol(cache_dir, sym)
        if df is not None:
            price_cache[sym] = df["close"]

    missing = [s for s in symbols_needed if s not in price_cache]
    if missing:
        print(f"  [WARN] {len(missing)} symbols not in cache: {missing[:5]}{'...' if len(missing)>5 else ''}")

    # For each horizon, compute IC across all weeks
    horizon_ics: dict[int, list] = {h: [] for h in horizons}

    for wk in weeks:
        wk_df = trades[trades["week"].astype(int) == wk].copy()
        week_start_str = wk_df["week_start"].iloc[0]
        week_start = pd.Timestamp(week_start_str, tz="UTC")

        preds = {}
        for _, row in wk_df.iterrows():
            sym = row["symbol"]
            preds[sym] = float(row["pred_signed"])

        for horizon in horizons:
            hour_rets = {}
            for sym, pred in preds.items():
                if sym not in price_cache:
                    continue
                closes = price_cache[sym]
                # Find first bar at or after week_start
                idx_mask = closes.index >= week_start
                if idx_mask.sum() == 0:
                    continue
                first_idx = int(np.argmax(idx_mask.values))
                last_idx  = first_idx + horizon
                if last_idx >= len(closes):
                    continue
                c0 = float(closes.iloc[first_idx])
                c1 = float(closes.iloc[last_idx])
                if c0 > 0 and np.isfinite(c0) and np.isfinite(c1):
                    hour_rets[sym] = float(np.log(c1 / c0))

            if len(hour_rets) < 5:
                continue

            pred_arr = np.array([preds[s] for s in hour_rets])
            ret_arr  = np.array([hour_rets[s] for s in hour_rets])
            ic = compute_ic(pred_arr, ret_arr)
            horizon_ics[horizon].append(ic)

    # ── results table ─────────────────────────────────────────────────────────
    rows = []
    print(f"  {'Horizon':>12}  {'Bars':>6}  {'n_wks':>6}  {'IC mean':>10}  "
          f"{'IC std':>9}  {'ICIR':>8}  {'IC>0%':>7}  {'Verdict'}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*12}")

    bars_to_label = {1: "5min", 3: "15min", 12: "1hr", 48: "4hr",
                     144: "12hr", 288: "1day", 1440: "5day"}

    for h in horizons:
        ic_arr = np.array(horizon_ics[h])
        if len(ic_arr) < 3:
            print(f"  {'─':>12}  {h:>6}  {'<3':>6}  {'─':>10}  {'─':>9}  {'─':>8}  {'─':>7}  {'insufficient data'}")
            continue
        ic_m   = float(np.mean(ic_arr))
        ic_s   = float(np.std(ic_arr) + 1e-12)
        icir   = float(ic_m / ic_s)
        ic_pos = float((ic_arr > 0).mean() * 100)

        label  = bars_to_label.get(h, f"{h}b")
        if ic_m > 0.030 and icir > 0.5:
            verdict = "STRONG  ✓"
        elif ic_m > 0.010:
            verdict = "WEAK    ⚠"
        elif ic_m < -0.010:
            verdict = "INVERTED ✗"
        else:
            verdict = "NOISE   ─"

        print(f"  {label:>12}  {h:>6}  {len(ic_arr):>6}  {ic_m:>+10.5f}  "
              f"{ic_s:>9.5f}  {icir:>8.4f}  {ic_pos:>6.1f}%  {verdict}")

        rows.append({
            "horizon_bars":    h,
            "horizon_label":   label,
            "n_weeks":         len(ic_arr),
            "ic_mean":         round(ic_m, 6),
            "ic_std":          round(ic_s, 6),
            "icir":            round(icir, 6),
            "ic_positive_pct": round(ic_pos, 2),
        })

    print(f"\n  DECAY INTERPRETATION (Citadel standard):")
    ics = [r["ic_mean"] for r in rows if np.isfinite(r["ic_mean"])]
    if len(ics) >= 2:
        peak_idx = int(np.argmax(ics))
        peak_h   = rows[peak_idx]["horizon_label"]
        peak_ic  = rows[peak_idx]["ic_mean"]
        print(f"    Peak IC {peak_ic:+.5f} at horizon {peak_h}")

        # Find half-life: first horizon where IC < peak/2
        half = peak_ic / 2
        half_label = "beyond tested range"
        for r in rows[peak_idx + 1:]:
            if r["ic_mean"] < half:
                half_label = r["horizon_label"]
                break
        print(f"    Half-life (IC decays to {half:+.5f}): {half_label}")

        # Leakage check: if IC at 1day is still > 50% of peak, suspect look-ahead
        r_1d = next((r for r in rows if r["horizon_label"] == "1day"), None)
        if r_1d:
            ratio_1d = r_1d["ic_mean"] / (peak_ic + 1e-12)
            if ratio_1d > 0.5 and peak_ic > 0.02:
                print(f"    ⚠️  LEAKAGE SUSPECT: IC at 1day is {ratio_1d*100:.0f}% of peak — "
                      f"reversal signal should decay much faster")
            else:
                print(f"    ✓  IC at 1day = {ratio_1d*100:.0f}% of peak — "
                      f"decay profile consistent with short-horizon reversal")

    # ── save ──────────────────────────────────────────────────────────────────
    if rows:
        out_path = trades_path.parent / "signal_decay.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
