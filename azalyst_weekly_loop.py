"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  WEEKLY SELF-IMPROVING LOOP  (Year 2 + Year 3)
╠══════════════════════════════════════════════════════════════════════════════╣
║  FIX: resample param now explicit in every load_data_for_window() call.   ║
║  Previously the loop would call build_panel / load_data with no TF arg,   ║
║  defaulting to '4h' for scoring but the feature windows inside were        ║
║  computed assuming 5-min (bph=12, bpd=288) → all NaN on any other TF.    ║
║  Now both training and scoring consistently use resample='5min' which      ║
║  matches how the feature cache was built, and the 4H resampling happens   ║
║  correctly inside load_data_for_window().                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
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

from azalyst_alpha_metrics import (
    calculate_weekly_alpha, should_retrain, session_report,
    ROUND_TRIP_FEE, FEE_RATE,
)
# FIX: import load_data_for_window from the fixed azalyst_train
from azalyst_train import (
    load_data_for_window, build_alpha_labels, cross_sectional_rank,
    train_model, FEATURE_COLS, BARS_PER_DAY,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
STOP_LOSS_PCT    = -1.5
TAKE_PROFIT_PCT  =  4.0
HORIZON_BARS     = 48      # 4H in 5-min bars (used for raw OHLCV sim only)
TOP_QUANTILE     = 0.20
BOTTOM_QUANTILE  = 0.20

# FIX: training always uses 5-min feature cache resampled to 4H
TRAIN_RESAMPLE = "4h"


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL GENERATION  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(model, scaler, week_df, feature_cols, top_pct=TOP_QUANTILE,
                     bottom_pct=BOTTOM_QUANTILE):
    week_df = week_df.copy()
    week_df["prob"]   = np.nan
    week_df["signal"] = "HOLD"
    avail = [c for c in feature_cols if c in week_df.columns]

    for ts, group in week_df.groupby(level=0):
        valid = group.dropna(subset=avail)
        if len(valid) < 5:
            continue
        try:
            Xs    = scaler.transform(valid[avail].values.astype(np.float32))
            probs = model.predict_proba(Xs)[:, 1]
        except Exception:
            continue
        week_df.loc[valid.index, "prob"] = probs
        n          = len(valid)
        n_long     = max(1, int(n * top_pct))
        n_shrt     = max(1, int(n * bottom_pct))
        sorted_idx = valid.index[np.argsort(probs)]
        week_df.loc[sorted_idx[-n_long:], "signal"] = "BUY"
        week_df.loc[sorted_idx[:n_shrt],  "signal"] = "SELL"
    return week_df


# ─────────────────────────────────────────────────────────────────────────────
#  BTC BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def get_btc_weekly_return(feature_dir, week_start, week_end):
    for name in ["BTCUSDT", "BTCUSDT.parquet"]:
        for f in [feature_dir / f"{name}.parquet", feature_dir / name]:
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
#  PAPER TRADE SIMULATOR  (uses raw 5-min OHLCV regardless of signal TF)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trades(signals_df, feature_dir, week_start, week_end,
                    horizon_bars=HORIZON_BARS):
    """
    FIX: Trade simulation always reads raw 5-min OHLCV for precise SL/TP
    price checking, regardless of what timeframe the signals were generated on.
    """
    trades = []
    signal_rows = signals_df[signals_df["signal"].isin(["BUY", "SELL"])]

    for ts, row in signal_rows.iterrows():
        sym    = str(row.get("symbol", ""))
        signal = str(row["signal"])
        prob   = float(row.get("prob", 0.5))

        # Find raw OHLCV (5-min bars for execution)
        ohlcv_path = None
        for data_dir_name in ["../data", "./data", "data"]:
            p = feature_dir.parent / data_dir_name / f"{sym}.parquet"
            if p.exists(): ohlcv_path = p; break
            p2 = Path(data_dir_name) / f"{sym}.parquet"
            if p2.exists(): ohlcv_path = p2; break

        if ohlcv_path is None:
            continue
        try:
            ohlcv = pd.read_parquet(ohlcv_path)
            ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
            ohlcv.columns = [c.lower() for c in ohlcv.columns]
            ts_col = next((c for c in ohlcv.columns
                           if c in ("timestamp", "time", "open_time")), None)
            if ts_col:
                col = ohlcv[ts_col]
                ohlcv.index = pd.to_datetime(
                    col, unit="ms" if pd.api.types.is_integer_dtype(col) else None,
                    utc=True)
                ohlcv = ohlcv.drop(columns=[ts_col])
            ohlcv = ohlcv.sort_index()[["open", "high", "low", "close"]]
        except Exception:
            continue

        future = ohlcv[ohlcv.index > ts].head(horizon_bars + 10)
        if len(future) < 2:
            continue

        entry_price = float(future.iloc[0]["open"])
        if entry_price <= 0:
            continue

        sl_mult = (1 + STOP_LOSS_PCT   / 100) if signal == "BUY" else (1 - STOP_LOSS_PCT   / 100)
        tp_mult = (1 + TAKE_PROFIT_PCT / 100) if signal == "BUY" else (1 - TAKE_PROFIT_PCT / 100)
        sl_price = entry_price * sl_mult
        tp_price = entry_price * tp_mult

        exit_price = None; exit_reason = "horizon"
        for _, bar in future.iloc[1:horizon_bars + 1].iterrows():
            lo, hi = float(bar["low"]), float(bar["high"])
            if signal == "BUY":
                if lo <= sl_price: exit_price = sl_price; exit_reason = "stop_loss";   break
                if hi >= tp_price: exit_price = tp_price; exit_reason = "take_profit"; break
            else:
                if hi >= sl_price: exit_price = sl_price; exit_reason = "stop_loss";   break
                if lo <= tp_price: exit_price = tp_price; exit_reason = "take_profit"; break

        if exit_price is None:
            exit_price = float(future.iloc[min(horizon_bars, len(future)-1)]["close"])
            exit_reason = "horizon"

        if exit_price <= 0:
            continue

        raw_ret = exit_price / entry_price - 1
        if signal == "SELL":
            raw_ret = -raw_ret
        pnl_pct = (raw_ret - ROUND_TRIP_FEE) * 100

        trades.append({
            "signal_time":  ts.isoformat(), "symbol": sym, "signal": signal,
            "probability":  round(prob, 4),
            "entry_price":  round(entry_price, 8), "exit_price": round(exit_price, 8),
            "pnl_percent":  round(pnl_pct, 4),
            "result":       "WIN" if pnl_pct > 0 else "LOSS",
            "exit_reason":  exit_reason,
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj["model"], obj["scaler"], obj["feature_cols"]


def save_model(model, scaler, feature_cols, path, meta=None):
    payload = {"model": model, "scaler": scaler, "feature_cols": feature_cols}
    if meta:
        payload.update(meta)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WEEKLY LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_weekly_loop(feature_dir, results_dir, year_label, year_start, year_end,
                    base_model_path, use_gpu=False, verbose=True):
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model, scaler, feature_cols = load_model(base_model_path)
    current_model_path = base_model_path

    all_trades = []; weekly_summary = []; weekly_returns = []; retrain_count = 0

    week_starts = pd.date_range(year_start, year_end - pd.Timedelta(weeks=1),
                                freq="W-MON", tz="UTC")

    print(f"\n{'═'*65}")
    print(f"  {year_label.upper()} LOOP — {len(week_starts)} weeks")
    print(f"  {year_start.date()} → {year_end.date()}")
    print(f"  Starting model: {base_model_path.name}")
    print(f"  Feature resample: {TRAIN_RESAMPLE}  (FIX: explicit, matches training)")
    print(f"{'═'*65}")

    for week_num, week_start in enumerate(week_starts, 1):
        week_end = week_start + pd.Timedelta(weeks=1)
        if week_end > year_end + pd.Timedelta(days=3):
            break

        t_week = time.time()
        print(f"\n  Week {week_num}/{len(week_starts)}: "
              f"{week_start.date()} → {week_end.date()}")

        # ── FIX: always pass TRAIN_RESAMPLE so feature windows match ─────────
        week_df = load_data_for_window(
            feature_dir, week_start, week_end,
            resample_freq=TRAIN_RESAMPLE,   # FIX: was implicit/wrong
            verbose=False
        )

        if week_df.empty:
            print(f"    [SKIP] No data for this week")
            weekly_returns.append(0.0)
            weekly_summary.append({
                "year": year_label, "week_num": week_num,
                "week_start": week_start.date().isoformat(),
                "week_end": week_end.date().isoformat(),
                "week_return_pct": 0.0, "annualised_pct": 0.0,
                "sharpe": 0.0, "profit_factor": 0.0,
                "win_rate": 0.0, "n_trades": 0, "on_track": False,
                "retrained": False, "model": current_model_path.name,
            })
            continue

        week_df_ranked = cross_sectional_rank(week_df, feature_cols)
        signals_df     = generate_signals(model, scaler, week_df_ranked, feature_cols)

        n_buy  = int((signals_df["signal"] == "BUY").sum())
        n_sell = int((signals_df["signal"] == "SELL").sum())
        print(f"    Signals: {n_buy + n_sell} (BUY={n_buy}, SELL={n_sell})")

        # Simulate trades on raw 5-min OHLCV (execution always at 5-min precision)
        week_trades = simulate_trades(signals_df, feature_dir, week_start, week_end)

        if not week_trades:
            print(f"    [WARN] No trades executed this week")
            weekly_returns.append(0.0)
            weekly_summary.append({
                "year": year_label, "week_num": week_num,
                "week_start": week_start.date().isoformat(),
                "week_end": week_end.date().isoformat(),
                "week_return_pct": 0.0, "annualised_pct": 0.0,
                "sharpe": 0.0, "profit_factor": 0.0,
                "win_rate": 0.0, "n_trades": 0, "on_track": False,
                "retrained": False, "model": current_model_path.name,
            })
            continue

        all_trades.extend(week_trades)
        trades_df = pd.DataFrame(week_trades)

        btc_ret = get_btc_weekly_return(feature_dir, week_start, week_end)
        alpha   = calculate_weekly_alpha(trades_df, btc_ret)
        weekly_returns.append(alpha["week_return_pct"] / 100.0)

        print(f"    Return: {alpha['week_return_pct']:+.2f}%  "
              f"| Ann: {alpha['annualised_pct']:.0f}%  "
              f"| WR: {alpha['win_rate']:.0f}%  "
              f"| Trades: {alpha['n_trades']}")

        decision  = should_retrain(weekly_returns)
        retrained = False

        if decision["retrain"]:
            print(f"    RETRAIN triggered: {decision['reason']}")
            retrain_count += 1

            # FIX: expanding window retrain also uses TRAIN_RESAMPLE
            train_start = year_start - pd.Timedelta(days=365)
            train_df = load_data_for_window(
                feature_dir, train_start, week_end,
                resample_freq=TRAIN_RESAMPLE,   # FIX
                verbose=False
            )

            if not train_df.empty:
                train_df = build_alpha_labels(train_df)
                avail2   = [c for c in feature_cols if c in train_df.columns]
                valid2   = train_df.dropna(subset=avail2 + ["alpha_label"])
                valid2   = cross_sectional_rank(valid2, avail2)

                if len(valid2) > 200:
                    X2 = valid2[avail2].values.astype(np.float32)
                    y2 = valid2["alpha_label"].values.astype(int)
                    new_model, new_scaler, importance, cv_auc = train_model(
                        X2, y2, avail2, use_gpu=use_gpu,
                        label=f"{year_label}_wk{week_num}"
                    )
                    model     = new_model
                    scaler    = new_scaler
                    feature_cols = avail2
                    fname         = f"model_{year_label.lower()}_week{week_num:03d}.pkl"
                    current_model_path = models_dir / fname
                    save_model(model, scaler, feature_cols, current_model_path, {
                        "cv_auc": round(cv_auc, 4), "week": week_num,
                        "retrain_trigger": decision["reason"],
                        "resample": TRAIN_RESAMPLE,
                    })
                    importance.to_csv(
                        results_dir / f"feature_importance_{year_label.lower()}_week{week_num:03d}.csv",
                        header=True
                    )
                    print(f"    Retrained → {fname}  AUC={cv_auc:.4f}")
                    retrained = True
                else:
                    print(f"    [WARN] Too few samples ({len(valid2)})")
            else:
                print(f"    [WARN] No training data for retrain")
        else:
            print(f"    Alpha OK: {decision['reason']}")

        weekly_summary.append({
            "year":             year_label, "week_num": week_num,
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
            "btc_week_pct":     round(btc_ret * 100, 4) if btc_ret else None,
        })

        del week_df, week_df_ranked, signals_df
        gc.collect()
        print(f"    Week done in {time.time()-t_week:.1f}s | Retrains: {retrain_count}")

    weekly_df = pd.DataFrame(weekly_summary)
    trades_df = pd.DataFrame(all_trades)
    weekly_df.to_csv(results_dir / f"weekly_summary_{year_label.lower()}.csv", index=False)
    trades_df.to_csv(results_dir / f"all_trades_{year_label.lower()}.csv",     index=False)

    print(f"\n  {year_label} complete. Retrains: {retrain_count}")
    return weekly_df, trades_df


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Weekly Self-Improving Loop (Year 2 + Year 3) — FIXED"
    )
    parser.add_argument("--feature-dir",  default="./feature_cache")
    parser.add_argument("--results-dir",  default="./results")
    parser.add_argument("--gpu",          action="store_true")
    parser.add_argument("--year2-only",   action="store_true")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("    AZALYST  —  WEEKLY SELF-IMPROVING LOOP  (FIXED TF)")
    print(f"    Feature resample: {TRAIN_RESAMPLE}")
    print("╚══════════════════════════════════════════════════════════════╝")

    date_cfg_path = results_dir / "date_config.json"
    if not date_cfg_path.exists():
        print("[ERROR] date_config.json not found. Run azalyst_train.py first.")
        return

    with open(date_cfg_path) as fh:
        dc = json.load(fh)

    year1_end  = pd.Timestamp(dc["year1_end"],  tz="UTC")
    year2_end  = pd.Timestamp(dc["year2_end"],  tz="UTC")
    year3_end  = pd.Timestamp(dc["year3_end"],  tz="UTC")
    year1_base = results_dir / "models" / "model_year1.pkl"

    if not year1_base.exists():
        print(f"[ERROR] Base model not found: {year1_base}"); return

    all_weekly = []; all_trades_ = []

    wk2, tr2 = run_weekly_loop(
        feature_dir=feature_dir, results_dir=results_dir,
        year_label="Year2", year_start=year1_end, year_end=year2_end,
        base_model_path=year1_base, use_gpu=args.gpu,
    )
    all_weekly.append(wk2); all_trades_.append(tr2)
    if not wk2.empty and not tr2.empty:
        session_report(wk2, tr2, label="Year 2")

    year2_models = sorted((results_dir / "models").glob("model_year2_*.pkl"))
    year3_seed   = year2_models[-1] if year2_models else year1_base

    if not args.year2_only:
        wk3, tr3 = run_weekly_loop(
            feature_dir=feature_dir, results_dir=results_dir,
            year_label="Year3", year_start=year2_end, year_end=year3_end,
            base_model_path=year3_seed, use_gpu=args.gpu,
        )
        all_weekly.append(wk3); all_trades_.append(tr3)
        if not wk3.empty and not tr3.empty:
            session_report(wk3, tr3, label="Year 3")

    combined_wk = pd.concat([df for df in all_weekly  if not df.empty], ignore_index=True)
    combined_tr = pd.concat([df for df in all_trades_ if not df.empty], ignore_index=True)
    combined_wk.to_csv(results_dir / "weekly_summary_all.csv", index=False)
    combined_tr.to_csv(results_dir / "all_trades_all.csv",     index=False)

    if not combined_wk.empty and not combined_tr.empty:
        session_report(combined_wk, combined_tr, label="COMBINED (Year 2 + 3)")

    alpha_report = {
        "total_weeks":  len(combined_wk), "total_trades": len(combined_tr),
        "year2_annual": float(wk2["annualised_pct"].mean()) if not wk2.empty else None,
        "year3_annual": float(wk3["annualised_pct"].mean()) if not args.year2_only and not wk3.empty else None,
        "year2_retrains": int(wk2["retrained"].sum()) if not wk2.empty else 0,
        "alpha_target": "1000% annual (10x)",
        "elapsed_hours": round((time.time() - t0) / 3600, 2),
        "resample_used": TRAIN_RESAMPLE,
    }
    with open(results_dir / "alpha_report.json", "w") as fh:
        json.dump(alpha_report, fh, indent=2)

    print(f"\n  All results saved to: {results_dir.resolve()}")
    print(f"\n  Total time: {(time.time()-t0)/3600:.2f} hours")


if __name__ == "__main__":
    main()
