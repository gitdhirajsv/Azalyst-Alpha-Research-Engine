"""
audit_corrected_pnl.py
======================
AZALYST ALPHA RESEARCH ENGINE — Post-hoc corrected PnL audit.

STEP 05 of Session Plan — Two Sigma / BlackRock Standard:
  "OOS IC is the only metric that matters. The headline return must be
   decomposed into signal alpha vs sizing artifact."

Applies equal-weight (no meta_size leverage) to an existing all_trades_v4.csv
to produce the honest portfolio return. Also checks:
  - meta_size range (confirms sizing bug if > 1.0)
  - long vs short contribution split
  - IC (Spearman) between pred_ret and raw_ret_pct per week
  - Annualised Sharpe (52 weeks/year)
  - Max drawdown on equity curve

Usage:
    python audit_corrected_pnl.py --trades results_top6/all_trades_v4.csv

Outputs:
    Prints a corrected performance table to stdout.
    Writes corrected_performance.json next to the input file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ─── helpers ──────────────────────────────────────────────────────────────────

def compute_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank IC. Returns 0.0 if fewer than 5 valid pairs."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 5:
        return 0.0
    ic, _ = stats.spearmanr(pred[mask], actual[mask])
    return float(ic) if np.isfinite(ic) else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    """Max peak-to-trough drawdown on an equity curve (fractional)."""
    peak = np.maximum.accumulate(equity)
    dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    return float(dd.min())


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected equal-weight PnL audit")
    parser.add_argument(
        "--trades",
        default="results_top6/all_trades_v4.csv",
        help="Path to all_trades_v4.csv (default: results_top6/all_trades_v4.csv)",
    )
    parser.add_argument(
        "--weeks-per-year",
        type=int,
        default=52,
        help="Weeks per year for annualisation (crypto ≈ 52)",
    )
    args = parser.parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"[ERROR] File not found: {trades_path}")
        sys.exit(1)

    trades = pd.read_csv(trades_path)
    print(f"\n{'='*72}")
    print(f"  AZALYST — CORRECTED PnL AUDIT")
    print(f"  Source: {trades_path}")
    print(f"  Total trade rows: {len(trades):,}")
    print(f"{'='*72}\n")

    # ── sizing bug check ──────────────────────────────────────────────────────
    meta_max = trades["meta_size"].astype(float).max()
    meta_min = trades["meta_size"].astype(float).min()
    meta_mean = trades["meta_size"].astype(float).mean()
    has_bug = meta_max > 1.0

    print(f"  META_SIZE RANGE: min={meta_min:.4f}  mean={meta_mean:.4f}  max={meta_max:.4f}")
    if has_bug:
        print(f"  ⚠️  SIZING BUG DETECTED: meta_size > 1.0 in {(trades['meta_size'].astype(float) > 1.0).sum()} rows")
        print(f"     Original PnL was inflated by ~{meta_mean:.2f}x leverage per trade")
        print(f"     Corrected mode: equal-weight (meta_size ignored)\n")
    else:
        print(f"  ✓ meta_size ≤ 1.0 — no sizing inflation detected\n")

    # ── direction split ───────────────────────────────────────────────────────
    n_buy  = (trades["signal"] == "BUY").sum()
    n_sell = (trades["signal"] == "SELL").sum()
    print(f"  Signal split:  BUY={n_buy}  SELL={n_sell}  ratio={n_buy/(n_sell+1e-9):.2f}")

    # ── per-week corrected PnL ────────────────────────────────────────────────
    weeks = sorted(trades["week"].astype(int).unique())
    weekly_results = []

    for wk in weeks:
        wk_df = trades[trades["week"].astype(int) == wk].copy()
        wk_df["raw_ret"] = wk_df["raw_ret_pct"].astype(float) / 100.0
        wk_df["pred"] = wk_df["pred_ret"].astype(float)

        # Equal-weight long-short portfolio return
        long_df  = wk_df[wk_df["signal"] == "BUY"]
        short_df = wk_df[wk_df["signal"] == "SELL"]

        long_ret  = float(long_df["raw_ret"].mean())  if len(long_df)  > 0 else 0.0
        short_ret = float(-short_df["raw_ret"].mean()) if len(short_df) > 0 else 0.0

        n_sides = (1 if len(long_df) > 0 else 0) + (1 if len(short_df) > 0 else 0)
        if n_sides > 0:
            week_ret = (long_ret + short_ret) / n_sides
        else:
            week_ret = 0.0

        # IC: Spearman rank correlation between model prediction and actual price return.
        # pred_ret is already directional (negative for SELL, positive for BUY).
        # Do NOT flip — both pred and actual are negative for winning shorts.
        preds   = wk_df["pred"].values.astype(np.float64)
        actuals = wk_df["raw_ret"].values.astype(np.float64)
        ic = compute_ic(preds, actuals)

        weekly_results.append({
            "week":          wk,
            "n_trades":      len(wk_df),
            "week_ret_eq":   week_ret,
            "long_ret":      long_ret,
            "short_ret":     short_ret,
            "ic":            ic,
        })

    df_w = pd.DataFrame(weekly_results)

    # ── equity curve ─────────────────────────────────────────────────────────
    equity = np.cumprod(1.0 + df_w["week_ret_eq"].values)
    total_ret = float(equity[-1] - 1.0)
    ann_ret   = float((equity[-1] ** (args.weeks_per_year / len(weeks))) - 1.0)

    # Sharpe
    wr = df_w["week_ret_eq"].values
    sharpe = float(np.mean(wr) / (np.std(wr) + 1e-12)) * np.sqrt(args.weeks_per_year)

    mdd = max_drawdown(equity)

    # IC stats
    ic_vals = df_w["ic"].values
    ic_mean  = float(np.mean(ic_vals))
    ic_std   = float(np.std(ic_vals) + 1e-12)
    icir     = float(ic_mean / ic_std)
    ic_pos   = float((ic_vals > 0).mean() * 100)

    # Win rate
    win_rate  = float((wr > 0).mean() * 100)
    lose_rate = float((wr < 0).mean() * 100)

    # Long vs Short contribution
    long_contrib  = float(df_w["long_ret"].mean() * 100)
    short_contrib = float(df_w["short_ret"].mean() * 100)

    print(f"\n{'─'*72}")
    print(f"  CORRECTED PERFORMANCE (equal-weight, 1x leverage, no meta_size)")
    print(f"{'─'*72}")
    print(f"  Total weeks      : {len(weeks)}")
    print(f"  Total trades     : {len(trades)}")
    print(f"  Total return     : {total_ret*100:+.2f}%")
    print(f"  Annualised return: {ann_ret*100:+.2f}%")
    print(f"  Sharpe (annl.)   : {sharpe:.4f}")
    print(f"  Max drawdown     : {mdd*100:.2f}%")
    print(f"  Win rate (weeks) : {win_rate:.1f}%  |  Lose rate: {lose_rate:.1f}%")
    print(f"")
    print(f"  IC mean          : {ic_mean:.5f}  (target > 0.020)")
    print(f"  IC std           : {ic_std:.5f}")
    print(f"  ICIR             : {icir:.4f}     (target > 0.500)")
    print(f"  IC% positive     : {ic_pos:.1f}%  (target > 55%)")
    print(f"")
    print(f"  LONG contribution (avg/wk)  : {long_contrib:+.4f}%")
    print(f"  SHORT contribution (avg/wk) : {short_contrib:+.4f}%")

    # ── institutional standards check ────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  TWO SIGMA BENCHMARK CHECK")
    print(f"{'─'*72}")
    checks = [
        ("IC OOS > 0.010",          ic_mean > 0.010,    f"{ic_mean:.5f}"),
        ("IC OOS > 0.030 (good)",   ic_mean > 0.030,    f"{ic_mean:.5f}"),
        ("IC OOS > 0.050 (strong)", ic_mean > 0.050,    f"{ic_mean:.5f}"),
        ("ICIR > 0.500",            icir > 0.500,        f"{icir:.4f}"),
        ("ICIR > 1.000 (strong)",   icir > 1.000,        f"{icir:.4f}"),
        ("Sharpe > 0.700",          sharpe > 0.700,      f"{sharpe:.4f}"),
        ("Sharpe > 1.500 (strong)", sharpe > 1.500,      f"{sharpe:.4f}"),
        ("Max DD < -20%",           mdd > -0.20,         f"{mdd*100:.2f}%"),
        ("Max DD < -10% (good)",    mdd > -0.10,         f"{mdd*100:.2f}%"),
        ("IC% positive > 55%",      ic_pos > 55.0,       f"{ic_pos:.1f}%"),
        ("IC% positive > 65%",      ic_pos > 65.0,       f"{ic_pos:.1f}%"),
        ("Win rate > 55%",          win_rate > 55.0,     f"{win_rate:.1f}%"),
    ]
    for label, passed, val in checks:
        mark = "✓" if passed else "✗"
        print(f"  [{mark}] {label:<36} actual={val}")

    # ── save ──────────────────────────────────────────────────────────────────
    out = {
        "source_file":          str(trades_path),
        "sizing_bug_detected":  has_bug,
        "meta_size_max_orig":   round(float(meta_max), 6),
        "meta_size_mean_orig":  round(float(meta_mean), 6),
        "corrected_mode":       "equal-weight, meta_size ignored",
        "total_weeks":          int(len(weeks)),
        "total_trades":         int(len(trades)),
        "total_return_pct":     round(total_ret * 100, 4),
        "annualised_pct":       round(ann_ret * 100, 4),
        "sharpe":               round(sharpe, 6),
        "max_drawdown_pct":     round(mdd * 100, 4),
        "win_rate_pct":         round(win_rate, 2),
        "ic_mean":              round(ic_mean, 6),
        "ic_std":               round(ic_std, 6),
        "icir":                 round(icir, 6),
        "ic_positive_pct":      round(ic_pos, 2),
        "long_contrib_pct":     round(long_contrib, 4),
        "short_contrib_pct":    round(short_contrib, 4),
    }

    out_path = trades_path.parent / "corrected_performance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
