"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  WEEKLY SELF-IMPROVING LOOP  (Year 2 + Year 3)
║  NO LLM. Pure quant self-improvement.                                      ║
║                                                                             ║
║  PROCESS (for each week):                                                  ║
║    1. Predict  — generate signals on this week's data                      ║
║    2. Trade    — simulate paper trades with Binance fees                   ║
║    3. Evaluate — calculate alpha (target: 1000% annual)                    ║
║    4. Decide   — if NOT on track: expand training window, retrain model    ║
║    5. Save     — weekly summary + all trades CSV                           ║
║                                                                             ║
║  Expanding window: Year 1 base + each completed week added to training     ║
║  This means the model gets smarter every time it sees new data.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python azalyst_weekly_loop.py --feature-dir ./feature_cache --results-dir ./results
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGBM = True
except ImportError:
    _LGBM = False

# Import our modules
from azalyst_alpha_metrics import (
    calculate_weekly_alpha,
    should_retrain,
    session_report,
    ROUND_TRIP_FEE,
    FEE_RATE,
)
from azalyst_train import (
    load_data_for_window,
    build_alpha_labels,
    cross_sectional_rank,
    train_model,
    FEATURE_COLS,
    BARS_PER_DAY,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
STOP_LOSS_PCT    = -1.5    # % SL per trade
TAKE_PROFIT_PCT  =  4.0    # % TP per trade (skewed for 1000% target)
HORIZON_BARS     = 48      # 4H default exit horizon
TOP_QUANTILE     = 0.20    # long top 20% of ranked symbols
BOTTOM_QUANTILE  = 0.20    # short bottom 20%


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL GENERATION  (cross-sectional ranking)
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(
    model,
    scaler: StandardScaler,
    week_df: pd.DataFrame,
    feature_cols: List[str],
    top_pct: float = TOP_QUANTILE,
    bottom_pct: float = BOTTOM_QUANTILE,
) -> pd.DataFrame:
    """
    At each timestamp within the week, rank all symbols by predicted
    outperformance probability. Long top 20%, short bottom 20%.

    This is a proper cross-sectional strategy — we're always positioned
    in the RELATIVE winners vs losers, regardless of market direction.
    """
    week_df = week_df.copy()
    week_df["prob"]   = np.nan
    week_df["signal"] = "HOLD"

    avail = [c for c in feature_cols if c in week_df.columns]

    for ts, group in week_df.groupby(level=0):
        valid = group.dropna(subset=avail)
        if len(valid) < 5:
            continue

        X     = valid[avail].values.astype(np.float32)
        try:
            Xs    = scaler.transform(X)
            probs = model.predict_proba(Xs)[:, 1]
        except Exception:
            continue

        week_df.loc[valid.index, "prob"] = probs

        # Cross-sectional ranking within this timestamp
        n      = len(valid)
        n_long = max(1, int(n * top_pct))
        n_shrt = max(1, int(n * bottom_pct))

        sorted_idx = valid.index[np.argsort(probs)]
        # High prob = model predicts this coin outperforms → LONG
        week_df.loc[sorted_idx[-n_long:], "signal"] = "BUY"
        # Low prob = model predicts underperformance → SHORT
        week_df.loc[sorted_idx[:n_shrt],  "signal"] = "SELL"

    return week_df


# ─────────────────────────────────────────────────────────────────────────────
#  BTC RETURN CALCULATOR  (for excess return vs benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def get_btc_weekly_return(
    feature_dir: Path,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> Optional[float]:
    """Get BTC return for the week as the benchmark."""
    for name in ["BTCUSDT", "BTCUSDT.parquet"]:
        for f in [feature_dir / f"{name}.parquet",
                  feature_dir / f"{name}"]:
            if f.exists():
                try:
                    df = pd.read_parquet(f, columns=["ret_1bar"])
                    df.index = pd.to_datetime(df.index, utc=True)
                    week = df[(df.index >= week_start) & (df.index < week_end)]
                    if len(week) > 0:
                        return float((1 + week["ret_1bar"].dropna()).prod() - 1)
                except Exception:
                    pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER TRADE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trades(
    signals_df: pd.DataFrame,
    feature_dir: Path,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    horizon_bars: int = HORIZON_BARS,
) -> List[Dict]:
    """
    For each BUY/SELL signal, simulate trade entry and exit.

    Entry:  next bar open after signal
    Exit:   first of — SL hit, TP hit, or horizon_bars elapsed
    PnL:    (exit/entry - 1) - ROUND_TRIP_FEE, direction-adjusted
    """
    trades = []
    signal_rows = signals_df[signals_df["signal"].isin(["BUY", "SELL"])]

    for ts, row in signal_rows.iterrows():
        sym    = str(row.get("symbol", ""))
        signal = str(row["signal"])
        prob   = float(row.get("prob", 0.5))

        # Load raw OHLCV for this symbol (for trade price lookup)
        ohlcv_path = None
        for data_dir_name in ["../data", "./data", "data"]:
            p = feature_dir.parent / data_dir_name / f"{sym}.parquet"
            if p.exists():
                ohlcv_path = p
                break
            p2 = Path(data_dir_name) / f"{sym}.parquet"
            if p2.exists():
                ohlcv_path = p2
                break

        if ohlcv_path is None:
            continue

        try:
            ohlcv = pd.read_parquet(ohlcv_path)
            ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
            ohlcv.columns = [c.lower() for c in ohlcv.columns]
            # Normalise timestamp column if needed
            ts_col = next((c for c in ohlcv.columns
                           if c in ("timestamp", "time", "open_time")), None)
            if ts_col:
                ohlcv.index = pd.to_datetime(
                    ohlcv[ts_col],
                    unit="ms" if pd.api.types.is_integer_dtype(ohlcv[ts_col]) else None,
                    utc=True,
                )
                ohlcv = ohlcv.drop(columns=[ts_col])
            ohlcv = ohlcv.sort_index()
            ohlcv = ohlcv[["open", "high", "low", "close"]]
        except Exception:
            continue

        future = ohlcv[ohlcv.index > ts].head(horizon_bars + 10)
        if len(future) < 2:
            continue

        entry_bar   = future.iloc[0]
        entry_price = float(entry_bar["open"])
        if entry_price <= 0:
            continue

        # SL / TP levels
        sl_mult = (1 + STOP_LOSS_PCT   / 100) if signal == "BUY" else (1 - STOP_LOSS_PCT   / 100)
        tp_mult = (1 + TAKE_PROFIT_PCT / 100) if signal == "BUY" else (1 - TAKE_PROFIT_PCT / 100)
        sl_price = entry_price * sl_mult
        tp_price = entry_price * tp_mult

        exit_price  = None
        exit_time   = None
        exit_reason = "horizon"

        for _, bar in future.iloc[1:horizon_bars + 1].iterrows():
            lo = float(bar["low"])
            hi = float(bar["high"])
            if signal == "BUY":
                if lo <= sl_price:
                    exit_price, exit_time, exit_reason = sl_price, bar.name, "stop_loss"
                    break
                if hi >= tp_price:
                    exit_price, exit_time, exit_reason = tp_price, bar.name, "take_profit"
                    break
            else:
                if hi >= sl_price:
                    exit_price, exit_time, exit_reason = sl_price, bar.name, "stop_loss"
                    break
                if lo <= tp_price:
                    exit_price, exit_time, exit_reason = tp_price, bar.name, "take_profit"
                    break

        if exit_price is None:
            bar         = future.iloc[min(horizon_bars, len(future) - 1)]
            exit_price  = float(bar["close"])
            exit_time   = bar.name
            exit_reason = "horizon"

        if exit_price <= 0:
            continue

        raw_ret = exit_price / entry_price - 1
        if signal == "SELL":
            raw_ret = -raw_ret
        pnl_pct = (raw_ret - ROUND_TRIP_FEE) * 100

        trades.append({
            "signal_time":  ts.isoformat(),
            "symbol":       sym,
            "signal":       signal,
            "probability":  round(prob, 4),
            "entry_price":  round(entry_price, 8),
            "exit_price":   round(exit_price, 8),
            "pnl_percent":  round(pnl_pct, 4),
            "result":       "WIN" if pnl_pct > 0 else "LOSS",
            "exit_reason":  exit_reason,
            "entry_time":   future.index[0].isoformat(),
            "exit_time":    exit_time.isoformat() if exit_time is not None else "",
        })

    return trades


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: Path) -> Tuple[object, StandardScaler, List[str]]:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj["model"], obj["scaler"], obj["feature_cols"]


def save_model(model, scaler, feature_cols: List[str],
               path: Path, meta: dict = None) -> None:
    payload = {"model": model, "scaler": scaler,
               "feature_cols": feature_cols}
    if meta:
        payload.update(meta)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WEEKLY LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_weekly_loop(
    feature_dir: Path,
    results_dir: Path,
    year_label: str,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
    base_model_path: Path,
    use_gpu: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the weekly predict→evaluate→retrain loop for one year.

    Returns (weekly_summary_df, all_trades_df)
    """
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load starting model
    model, scaler, feature_cols = load_model(base_model_path)
    current_model_path = base_model_path

    # State
    all_trades:     List[Dict] = []
    weekly_summary: List[Dict] = []
    weekly_returns: List[float] = []
    retrain_count   = 0

    # Discover all weeks in the year
    week_starts = pd.date_range(year_start, year_end - pd.Timedelta(weeks=1),
                                freq="W-MON", tz="UTC")

    print(f"\n{'═'*65}")
    print(f"  {year_label.upper()} LOOP — {len(week_starts)} weeks")
    print(f"  {year_start.date()} → {year_end.date()}")
    print(f"  Starting model: {base_model_path.name}")
    print(f"{'═'*65}")

    for week_num, week_start in enumerate(week_starts, 1):
        week_end = week_start + pd.Timedelta(weeks=1)
        if week_end > year_end + pd.Timedelta(days=3):
            break

        t_week = time.time()
        print(f"\n  Week {week_num}/{len(week_starts)}: "
              f"{week_start.date()} → {week_end.date()}")

        # ── Predict ──────────────────────────────────────────────────────────
        week_df = load_data_for_window(
            feature_dir, week_start, week_end,
            resample_freq="4h", verbose=False
        )

        if week_df.empty:
            print(f"    [SKIP] No data for this week")
            continue

        week_df_ranked = cross_sectional_rank(week_df, feature_cols)
        signals_df     = generate_signals(model, scaler, week_df_ranked,
                                          feature_cols)

        n_signals = int((signals_df["signal"] != "HOLD").sum())
        print(f"    Signals: {n_signals} "
              f"(BUY={int((signals_df['signal']=='BUY').sum())}, "
              f"SELL={int((signals_df['signal']=='SELL').sum())})")

        # ── Simulate trades ───────────────────────────────────────────────────
        week_trades = simulate_trades(
            signals_df, feature_dir, week_start, week_end
        )

        if not week_trades:
            print(f"    [WARN] No trades executed this week")
            # Still record as zero-return week
            week_summary = {
                "year": year_label, "week_num": week_num,
                "week_start": week_start.date().isoformat(),
                "week_end": week_end.date().isoformat(),
                "week_return_pct": 0.0, "annualised_pct": 0.0,
                "sharpe": 0.0, "profit_factor": 0.0,
                "win_rate": 0.0, "n_trades": 0,
                "on_track": False, "retrained": False,
                "model": current_model_path.name,
            }
            weekly_summary.append(week_summary)
            weekly_returns.append(0.0)
            continue

        all_trades.extend(week_trades)
        trades_df = pd.DataFrame(week_trades)

        # ── Evaluate alpha ────────────────────────────────────────────────────
        btc_ret  = get_btc_weekly_return(feature_dir, week_start, week_end)
        alpha    = calculate_weekly_alpha(trades_df, btc_ret)
        weekly_returns.append(alpha["week_return_pct"] / 100.0)

        print(f"    Return: {alpha['week_return_pct']:+.2f}%  "
              f"| Annualised: {alpha['annualised_pct']:.0f}%  "
              f"| WR: {alpha['win_rate']:.0f}%  "
              f"| Trades: {alpha['n_trades']}")

        # ── Retrain decision ──────────────────────────────────────────────────
        decision = should_retrain(weekly_returns)
        retrained = False

        if decision["retrain"]:
            print(f"    RETRAIN triggered: {decision['reason']}")
            retrain_count += 1

            # Expanding window: Year 1 base + all completed weeks up to now
            train_start = year_start - pd.Timedelta(days=365)  # Year 1 start
            train_end   = week_end  # include this week in training

            print(f"    Loading expanded training data "
                  f"({train_start.date()} → {train_end.date()})...")

            train_df = load_data_for_window(
                feature_dir, train_start, train_end,
                resample_freq="4h", verbose=False
            )

            if not train_df.empty:
                train_df = build_alpha_labels(train_df)
                valid    = train_df.dropna(
                    subset=[c for c in feature_cols if c in train_df.columns]
                    + ["alpha_label"]
                )
                valid    = cross_sectional_rank(valid, feature_cols)
                avail    = [c for c in feature_cols if c in valid.columns]

                if len(valid) > 200:
                    X = valid[avail].values.astype(np.float32)
                    y = valid["alpha_label"].values.astype(int)
                    model, scaler, importance, cv_auc = train_model(
                        X, y, avail, use_gpu=use_gpu,
                        label=f"{year_label}_wk{week_num}"
                    )
                    # Save new model
                    fname = (f"model_{year_label.lower()}_"
                             f"week{week_num:03d}.pkl")
                    current_model_path = models_dir / fname
                    save_model(model, scaler, avail, current_model_path, {
                        "cv_auc":          round(cv_auc, 4),
                        "retrain_trigger": decision["reason"],
                        "week":            week_num,
                        "train_rows":      len(X),
                    })
                    # Save updated feature importance
                    imp_path = (results_dir /
                                f"feature_importance_{year_label.lower()}"
                                f"_week{week_num:03d}.csv")
                    importance.to_csv(imp_path, header=True)
                    print(f"    Retrained → {current_model_path.name}  "
                          f"AUC={cv_auc:.4f}")
                    retrained = True
                else:
                    print(f"    [WARN] Too few samples to retrain "
                          f"({len(valid)})")
            else:
                print(f"    [WARN] No training data loaded for retrain")
        else:
            print(f"    Alpha OK: {decision['reason']}")

        # ── Record weekly summary ─────────────────────────────────────────────
        week_summary = {
            "year":             year_label,
            "week_num":         week_num,
            "week_start":       week_start.date().isoformat(),
            "week_end":         week_end.date().isoformat(),
            "week_return_pct":  alpha["week_return_pct"],
            "annualised_pct":   alpha["annualised_pct"],
            "sharpe":           alpha["sharpe"],
            "profit_factor":    alpha["profit_factor"],
            "win_rate":         alpha["win_rate"],
            "n_trades":         alpha["n_trades"],
            "on_track":         alpha["on_track"],
            "rolling_annual":   decision.get("rolling_annual"),
            "retrained":        retrained,
            "retrain_count_so_far": retrain_count,
            "model":            current_model_path.name,
            "btc_week_pct":     round(btc_ret * 100, 4)
                                if btc_ret is not None else None,
            "excess_vs_btc":    alpha.get("excess_vs_btc_pct"),
        }
        weekly_summary.append(week_summary)

        elapsed_week = time.time() - t_week
        print(f"    Week done in {elapsed_week:.1f}s  "
              f"| Total retrains: {retrain_count}")

        # Free memory
        del week_df, week_df_ranked, signals_df, train_df \
            if 'train_df' in dir() else None
        gc.collect()

    # ── Save year results ─────────────────────────────────────────────────────
    weekly_df = pd.DataFrame(weekly_summary)
    trades_df = pd.DataFrame(all_trades)

    wk_path = results_dir / f"weekly_summary_{year_label.lower()}.csv"
    tr_path = results_dir / f"all_trades_{year_label.lower()}.csv"
    weekly_df.to_csv(wk_path, index=False)
    trades_df.to_csv(tr_path, index=False)

    print(f"\n  {year_label} saved:")
    print(f"    {wk_path}")
    print(f"    {tr_path}")
    print(f"    Retrain count: {retrain_count}")

    return weekly_df, trades_df


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Weekly Self-Improving Loop (Year 2 + Year 3)"
    )
    parser.add_argument("--feature-dir",  default="./feature_cache")
    parser.add_argument("--results-dir",  default="./results")
    parser.add_argument("--gpu",          action="store_true")
    parser.add_argument("--year2-only",   action="store_true",
                        help="Only run Year 2 (skip Year 3)")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("    AZALYST  —  WEEKLY SELF-IMPROVING LOOP  (NO LLM)")
    print("    Target: 1000% annual / 10x capital")
    print("    Trigger: rolling 4-week annualised < 1000% → retrain")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Load date config from training step
    date_cfg_path = results_dir / "date_config.json"
    if not date_cfg_path.exists():
        print("[ERROR] date_config.json not found. "
              "Run azalyst_train.py first.")
        return

    with open(date_cfg_path) as fh:
        dc = json.load(fh)

    year1_end  = pd.Timestamp(dc["year1_end"],  tz="UTC")
    year2_end  = pd.Timestamp(dc["year2_end"],  tz="UTC")
    year3_end  = pd.Timestamp(dc["year3_end"],  tz="UTC")
    year1_base_model = results_dir / "models" / "model_year1.pkl"

    if not year1_base_model.exists():
        print(f"[ERROR] Base model not found: {year1_base_model}")
        return

    all_weekly:  List[pd.DataFrame] = []
    all_trades_: List[pd.DataFrame] = []

    # ── Year 2 ────────────────────────────────────────────────────────────────
    wk2, tr2 = run_weekly_loop(
        feature_dir=feature_dir,
        results_dir=results_dir,
        year_label="Year2",
        year_start=year1_end,
        year_end=year2_end,
        base_model_path=year1_base_model,
        use_gpu=args.gpu,
    )
    all_weekly.append(wk2)
    all_trades_.append(tr2)

    if not wk2.empty and not tr2.empty:
        rpt2 = session_report(wk2, tr2, label="Year 2")

    # Find the last model from Year 2 to seed Year 3
    year2_models = sorted(
        (results_dir / "models").glob("model_year2_*.pkl")
    )
    year3_seed = year2_models[-1] if year2_models else year1_base_model
    print(f"\n  Year 3 seed model: {year3_seed.name}")

    # ── Year 3 ────────────────────────────────────────────────────────────────
    if not args.year2_only:
        wk3, tr3 = run_weekly_loop(
            feature_dir=feature_dir,
            results_dir=results_dir,
            year_label="Year3",
            year_start=year2_end,
            year_end=year3_end,
            base_model_path=year3_seed,
            use_gpu=args.gpu,
        )
        all_weekly.append(wk3)
        all_trades_.append(tr3)

        if not wk3.empty and not tr3.empty:
            rpt3 = session_report(wk3, tr3, label="Year 3")

    # ── Combined report ───────────────────────────────────────────────────────
    combined_weekly  = pd.concat([df for df in all_weekly  if not df.empty],
                                  ignore_index=True)
    combined_trades  = pd.concat([df for df in all_trades_ if not df.empty],
                                  ignore_index=True)

    combined_weekly.to_csv(results_dir / "weekly_summary_all.csv",  index=False)
    combined_trades.to_csv(results_dir / "all_trades_all.csv",       index=False)

    if not combined_weekly.empty and not combined_trades.empty:
        session_report(combined_weekly, combined_trades,
                       label="COMBINED (Year 2 + 3)")

    # Save alpha report for Claude review
    alpha_report = {
        "total_weeks":    len(combined_weekly),
        "total_trades":   len(combined_trades),
        "year2_annual":   float(wk2["annualised_pct"].mean())
                          if not wk2.empty else None,
        "year3_annual":   float(wk3["annualised_pct"].mean())
                          if not args.year2_only and not wk3.empty else None,
        "year2_retrain_count": int(wk2["retrained"].sum())
                               if not wk2.empty else 0,
        "year3_retrain_count": int(wk3["retrained"].sum())
                               if not args.year2_only and not wk3.empty else 0,
        "alpha_target":   "1000% annual (10x)",
        "elapsed_hours":  round((time.time() - t0) / 3600, 2),
    }
    with open(results_dir / "alpha_report.json", "w") as fh:
        json.dump(alpha_report, fh, indent=2)

    print(f"\n  All results saved to: {results_dir.resolve()}")
    print(f"  Files to send to Claude:")
    print(f"    alpha_report.json")
    print(f"    weekly_summary_all.csv")
    print(f"    all_trades_all.csv")
    print(f"    feature_importance_*.csv  (all improvement steps)")
    print(f"\n  Total time: {(time.time()-t0)/3600:.2f} hours")
    print()


if __name__ == "__main__":
    main()
