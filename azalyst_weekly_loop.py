"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  WEEKLY SELF-IMPROVING LOOP  v2  (Year 3)
║        Out-of-Sample Walk-Forward  |  Quarterly Retrain  |  IC Tracking    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Walk-Forward Architecture (from README)
────────────────────────────────────────
  Year 1 + Year 2 (730 days)  →  [BASE MODEL]  →  Year 3 only
  Each week:
    1. Predict  — rank symbols by outperformance probability
    2. Trade    — long top 15%, short bottom 15%
    3. Evaluate — weekly IC + return + Sharpe
    4. Retrain  — every 13 weeks (quarterly)
    5. Save     — weekly summary + all trades
  Output: weekly_summary_year3.csv, all_trades_year3.csv,
          performance_year3.json, performance_year3.png
"""

import argparse
import gc
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from azalyst_factors_v2 import FEATURE_COLS
from azalyst_alpha_metrics import session_report, should_retrain
from azalyst_train import train_model

warnings.filterwarnings("ignore")

# ── Config (matches notebook Cell 3) ─────────────────────────────────────────
RETRAIN_WEEKS    = 13        # quarterly retrain
TOP_QUANTILE     = 0.15      # top/bottom 15% for long/short
FEE_RATE         = 0.001     # 0.1% per leg
ROUND_TRIP_FEE   = FEE_RATE * 2
STOP_LOSS_PCT    = -2.0
TAKE_PROFIT_PCT  = 5.0
HORIZON_BARS     = 48        # 4H equivalent in 5-min bars
MAX_TRAIN_ROWS   = 4_000_000 # VRAM guard for RTX 2050 / T4


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS (kept from original file + restored)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_trade(ohlcv_slice, signal, entry_price, max_bars, sl_p, tp_p):
    """
    Trade simulation logic — walks forward bar by bar to find SL/TP/horizon exit.
    """
    if len(ohlcv_slice) < 2:
        return entry_price, "horizon"
    fut   = ohlcv_slice.iloc[1 : max_bars + 1]
    lows  = fut["low"].values
    highs = fut["high"].values
    if signal == "BUY":
        sl_hit = np.where(lows  <= sl_p)[0]
        tp_hit = np.where(highs >= tp_p)[0]
    else:
        sl_hit = np.where(highs >= sl_p)[0]
        tp_hit = np.where(lows  <= tp_p)[0]
    sl_bar = sl_hit[0] if len(sl_hit) else max_bars + 1
    tp_bar = tp_hit[0] if len(tp_hit) else max_bars + 1
    if sl_bar < tp_bar and sl_bar <= max_bars:
        return sl_p, "stop_loss"
    if tp_bar < sl_bar and tp_bar <= max_bars:
        return tp_p, "take_profit"
    return float(fut.iloc[min(max_bars - 1, len(fut) - 1)]["close"]), "horizon"


def cross_sectional_rank_signals(df, cols, top_q=TOP_QUANTILE):
    """
    Assigns BUY/SELL signals based on quantile ranking of probabilities.
    Called on a DataFrame indexed by (timestamp, symbol).
    """
    def assign_signals(grp):
        n_long = max(1, int(len(grp) * top_q))
        if len(grp) >= 5:
            grp = grp.sort_values("prob")
            grp.iloc[-n_long:, grp.columns.get_loc("signal")] = "BUY"
            grp.iloc[:n_long,  grp.columns.get_loc("signal")] = "SELL"
        return grp

    res = df.copy()
    res["signal"] = "HOLD"
    return res.groupby(level=0, group_keys=False).apply(assign_signals)


def compute_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank IC between predictions and actual returns."""
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return 0.0
    return float(stats.spearmanr(y_pred[mask], y_true[mask])[0])


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE STORE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_store(feature_dir: str, max_symbols: int = None) -> dict:
    """Load all symbol feature parquets from the cache into a dict."""
    files = sorted(Path(feature_dir).glob("*.parquet"))
    if max_symbols:
        files = files[:max_symbols]

    data = {}
    print(f"[WeeklyLoop] Loading {len(files)} symbols from feature store...")
    for f in files:
        try:
            df = pd.read_parquet(f)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            if len(df) > 200:
                data[f.stem] = df
        except Exception as e:
            print(f"  [WARN] {f.stem}: {e}")

    print(f"[WeeklyLoop] Loaded {len(data)} valid symbols")
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  DATE SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def get_date_splits(data: dict):
    """
    Find the Year 3 start date from the full data range.
    Year 3 = final 1/3 of the total timeline (matching notebook train/test split).
    """
    all_min, all_max = [], []
    for df in data.values():
        all_min.append(df.index.min())
        all_max.append(df.index.max())

    global_min = min(all_min)
    global_max = max(all_max)
    total_span = global_max - global_min
    year3_start = global_min + (total_span * 2 / 3)

    print(f"[WeeklyLoop] Data range : {global_min.date()} → {global_max.date()}")
    print(f"[WeeklyLoop] Year 3 from: {year3_start.date()} → {global_max.date()}")
    return year3_start, global_max


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_training_matrix(data: dict, end_date, min_rows: int = 500):
    """
    Build pooled (X, y, y_ret) from all symbols up to end_date.
    Caps at MAX_TRAIN_ROWS for VRAM safety (RTX 2050 4GB / Kaggle T4).
    """
    X_parts, y_parts, ret_parts = [], [], []

    for sym, df in data.items():
        try:
            subset = df[df.index < end_date]
            if len(subset) < min_rows:
                continue

            feat   = subset[FEATURE_COLS].values.astype(np.float32)
            labels = subset["alpha_label"].values.astype(np.float32)
            rets   = subset["future_ret_4h"].values.astype(np.float32)

            valid = np.isfinite(feat).all(axis=1) & np.isfinite(labels) & np.isfinite(rets)
            if valid.sum() < 50:
                continue

            X_parts.append(feat[valid])
            y_parts.append(labels[valid])
            ret_parts.append(rets[valid])
        except Exception:
            pass

    if not X_parts:
        return None, None, None

    X     = np.vstack(X_parts)
    y     = np.concatenate(y_parts)
    y_ret = np.concatenate(ret_parts)

    if len(X) > MAX_TRAIN_ROWS:
        idx = np.random.choice(len(X), MAX_TRAIN_ROWS, replace=False)
        X, y, y_ret = X[idx], y[idx], y_ret[idx]

    print(f"  Training matrix: {len(X):,} rows × {len(FEATURE_COLS)} features")
    return X, y, y_ret


# ─────────────────────────────────────────────────────────────────────────────
#  WEEKLY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_week(model, scaler, data: dict, week_start, week_end):
    """
    Run model inference for every symbol within the week window.
    Returns dicts of {symbol: mean_pred_prob} and {symbol: mean_actual_ret}.
    """
    predictions = {}
    actual_rets = {}

    for sym, df in data.items():
        try:
            week_data = df[(df.index >= week_start) & (df.index < week_end)]
            if len(week_data) < 3:
                continue

            feat  = week_data[FEATURE_COLS].values.astype(np.float32)
            valid = np.isfinite(feat).all(axis=1)
            if valid.sum() < 2:
                continue

            feat_scaled = scaler.transform(feat[valid])
            probs = model.predict_proba(feat_scaled)[:, 1]
            predictions[sym] = float(probs.mean())

            ret_col = week_data["future_ret_4h"].values[valid]
            finite_ret = ret_col[np.isfinite(ret_col)]
            if len(finite_ret) > 0:
                actual_rets[sym] = float(finite_ret.mean())
        except Exception:
            pass

    return predictions, actual_rets


# ─────────────────────────────────────────────────────────────────────────────
#  WEEKLY TRADE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_weekly_trades(predictions: dict, actual_rets: dict, top_q: float = TOP_QUANTILE):
    """
    Convert cross-sectional predictions to long/short trades.
    Returns list of trade records and the mean net weekly return (decimal).
    """
    if not predictions:
        return [], 0.0

    pred_series = pd.Series(predictions)
    ranked      = pred_series.rank(pct=True)
    longs       = ranked[ranked >= (1 - top_q)].index.tolist()
    shorts      = ranked[ranked <= top_q].index.tolist()

    trades = []
    for sym in longs:
        ret = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        pnl = (ret - ROUND_TRIP_FEE) * 100
        trades.append({
            "symbol":      sym,
            "signal":      "BUY",
            "pred_prob":   round(predictions[sym], 5),
            "pnl_percent": round(pnl, 4),
            "raw_ret":     round(ret * 100, 4),
        })

    for sym in shorts:
        ret = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        pnl = (-ret - ROUND_TRIP_FEE) * 100
        trades.append({
            "symbol":      sym,
            "signal":      "SELL",
            "pred_prob":   round(predictions[sym], 5),
            "pnl_percent": round(pnl, 4),
            "raw_ret":     round(ret * 100, 4),
        })

    week_ret = float(np.mean([t["pnl_percent"] for t in trades])) / 100 if trades else 0.0
    return trades, week_ret


# ─────────────────────────────────────────────────────────────────────────────
#  CHART SAVER
# ─────────────────────────────────────────────────────────────────────────────

def _save_chart(summary_df: pd.DataFrame, trades_df: pd.DataFrame, results_dir: str):
    """Save 4-panel Year 3 performance chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Azalyst v2 — Year 3 Walk-Forward Performance", fontsize=14)

        # Panel 1: Cumulative returns
        ax = axes[0][0]
        rets = summary_df["week_return_pct"].fillna(0) / 100
        cum  = (1 + rets).cumprod() - 1
        ax.plot(summary_df["week"].values, cum.values * 100, color="#1f77b4", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title("Cumulative Return (%)")
        ax.set_xlabel("Week")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3)

        # Panel 2: Weekly return distribution
        ax = axes[0][1]
        wr = summary_df["week_return_pct"].dropna()
        ax.hist(wr, bins=20, color="#ff7f0e", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(
            wr.mean(), color="red", linewidth=1.5, linestyle="--",
            label=f"Mean: {wr.mean():.2f}%",
        )
        ax.set_title("Weekly Return Distribution")
        ax.set_xlabel("Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: IC series
        ax = axes[1][0]
        ic_vals = summary_df["ic"].fillna(0)
        ax.bar(
            summary_df["week"].values,
            ic_vals.values,
            color=["#2ca02c" if v > 0 else "#d62728" for v in ic_vals],
            alpha=0.7,
        )
        ax.axhline(
            ic_vals.mean(), color="blue", linewidth=1, linestyle="--",
            label=f"Mean IC: {ic_vals.mean():.4f}",
        )
        ax.set_title("Weekly IC (Information Coefficient)")
        ax.set_xlabel("Week")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 4: Trade P&L distribution
        ax = axes[1][1]
        if len(trades_df) > 0 and "pnl_percent" in trades_df.columns:
            pnl = trades_df["pnl_percent"].dropna()
            ax.hist(pnl, bins=30, color="#9467bd", alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axvline(
                pnl.mean(), color="red", linewidth=1.5, linestyle="--",
                label=f"Mean: {pnl.mean():.3f}%",
            )
            ax.set_title("Trade P&L Distribution")
            ax.set_xlabel("P&L (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(results_dir, "performance_year3.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart   → {chart_path}")
    except Exception as e:
        print(f"[WeeklyLoop] Chart save failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WALK-FORWARD LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_weekly_loop(
    feature_dir: str,
    results_dir: str,
    gpu: bool = False,
    year2_only: bool = False,
):
    """
    Full Year 3 walk-forward loop.

    Steps:
      1. Load feature store
      2. Determine Year 3 date range
      3. Load or train base model (Year 1+2)
      4. For each week in Year 3:
           predict → trade → IC → save → retrain every 13 weeks
      5. Save all outputs + chart + performance JSON
    """
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load feature store
    data = load_feature_store(feature_dir)
    if not data:
        print("[WeeklyLoop] ERROR: No data found in feature_dir")
        return None

    # 2. Date splits
    year3_start, global_max = get_date_splits(data)

    if year2_only:
        # Shift start earlier for a shorter test run
        year3_start = year3_start - pd.Timedelta(days=180)
        print("[WeeklyLoop] Year 2 only mode — test period shifted back 180 days")

    # 3. Load or train base model
    base_model_path = os.path.join(models_dir, "model_base_y1y2.pkl")

    if os.path.exists(base_model_path):
        print(f"[WeeklyLoop] Loading base model: {base_model_path}")
        with open(base_model_path, "rb") as f:
            saved = pickle.load(f)
        model, scaler = saved["model"], saved["scaler"]
    else:
        print("[WeeklyLoop] No base model found — training on Year 1+2 data...")
        X, y, y_ret = build_training_matrix(data, year3_start)
        if X is None:
            print("[WeeklyLoop] ERROR: Could not build training matrix")
            return None
        model, scaler, importance, auc, ic, icir = train_model(
            X, y, y_ret, FEATURE_COLS, label="base_y1y2", use_gpu=gpu
        )
        with open(base_model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        importance.to_csv(os.path.join(results_dir, "feature_importance_base.csv"))
        print(f"  Base model: AUC={auc:.4f}  IC={ic:.4f}  ICIR={icir:.4f}")

    # 4. Generate weekly timestamps
    weeks = pd.date_range(start=year3_start, end=global_max, freq="W-MON")
    if len(weeks) < 2:
        print("[WeeklyLoop] ERROR: Not enough weeks in Year 3")
        return None

    print(f"\n[WeeklyLoop] Starting {len(weeks)-1} weeks of Year 3 walk-forward...\n")

    all_trades:            list = []
    weekly_summary:        list = []
    weekly_returns_history: list = []

    for week_num, (week_start, week_end) in enumerate(zip(weeks[:-1], weeks[1:]), 1):

        # Predict
        predictions, actual_rets = predict_week(model, scaler, data, week_start, week_end)

        if len(predictions) < 5:
            print(f"  Week {week_num:3d}: skipped — only {len(predictions)} symbols available")
            continue

        # Trade simulation
        trades, week_ret = simulate_weekly_trades(predictions, actual_rets)

        # IC
        if predictions and actual_rets:
            common = [s for s in predictions if s in actual_rets]
            pred_arr = np.array([predictions[s] for s in common])
            ret_arr  = np.array([actual_rets[s]  for s in common])
            week_ic  = compute_ic(pred_arr, ret_arr)
        else:
            week_ic = 0.0

        ann_proj = ((1 + week_ret) ** 52 - 1) * 100
        weekly_returns_history.append(week_ret)

        # Retrain decision (metric-based or scheduled)
        retrain_decision = should_retrain(weekly_returns_history)
        force_retrain    = (week_num % RETRAIN_WEEKS == 0)
        did_retrain      = False

        if force_retrain or retrain_decision["retrain"]:
            reason = "scheduled" if force_retrain else retrain_decision["reason"]
            print(f"  Week {week_num:3d}: RETRAINING — {reason}")
            X_new, y_new, y_ret_new = build_training_matrix(data, week_end)
            if X_new is not None:
                model, scaler, importance, auc, ic_t, icir_t = train_model(
                    X_new, y_new, y_ret_new, FEATURE_COLS,
                    label=f"y3_week{week_num:03d}", use_gpu=gpu,
                )
                pkl_path = os.path.join(models_dir, f"model_y3_week{week_num:03d}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump({"model": model, "scaler": scaler}, f)
                importance.to_csv(
                    os.path.join(results_dir, f"feature_importance_y3_week{week_num:03d}.csv")
                )
                print(f"    → AUC={auc:.4f}  IC={ic_t:.4f}  ICIR={icir_t:.4f}")
                did_retrain = True

        # Stamp trades with week metadata
        for t in trades:
            t["week"]       = week_num
            t["week_start"] = str(week_start.date())
        all_trades.extend(trades)

        on_track_thresh = (1.10) ** (1 / 52) - 1  # weekly equiv of 10x annual
        weekly_summary.append({
            "week":             week_num,
            "week_start":       str(week_start.date()),
            "week_end":         str(week_end.date()),
            "n_symbols":        len(predictions),
            "n_trades":         len(trades),
            "week_return_pct":  round(week_ret * 100, 4),
            "annualised_pct":   round(ann_proj, 2),
            "ic":               round(week_ic, 5),
            "on_track":         week_ret >= on_track_thresh,
            "retrained":        did_retrain,
        })

        # Progress log every 4 weeks
        if week_num % 4 == 0 or week_num <= 3:
            rolling_ret = np.mean(weekly_returns_history[-4:]) * 100
            print(
                f"  Week {week_num:3d} | ret={week_ret*100:+.2f}% | "
                f"IC={week_ic:+.4f} | 4w_avg={rolling_ret:+.2f}% | "
                f"n={len(predictions)}"
            )

        gc.collect()

    if not weekly_summary:
        print("[WeeklyLoop] No weekly results produced.")
        return None

    # 5. Save outputs
    summary_df = pd.DataFrame(weekly_summary)
    trades_df  = pd.DataFrame(all_trades)

    summary_path = os.path.join(results_dir, "weekly_summary_year3.csv")
    trades_path  = os.path.join(results_dir, "all_trades_year3.csv")
    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    # Performance JSON (including IC stats)
    perf = session_report(summary_df, trades_df, label="Year3 Walk-Forward")
    ic_series = summary_df["ic"].dropna()
    perf["ic_mean"]         = round(float(ic_series.mean()), 5)
    perf["ic_std"]          = round(float(ic_series.std()),  5)
    perf["icir"]            = round(float(ic_series.mean() / (ic_series.std() + 1e-8)), 4)
    perf["ic_positive_pct"] = round(float((ic_series > 0).mean() * 100), 1)

    perf_path = os.path.join(results_dir, "performance_year3.json")
    with open(perf_path, "w") as f:
        json.dump(perf, f, indent=2)

    _save_chart(summary_df, trades_df, results_dir)

    print(f"\n[WeeklyLoop] ── COMPLETE ──")
    print(f"  Summary → {summary_path}")
    print(f"  Trades  → {trades_path}")
    print(f"  Perf    → {perf_path}")
    print(f"  IC mean={perf['ic_mean']:.4f}  ICIR={perf['icir']:.4f}  "
          f"IC+%={perf['ic_positive_pct']:.1f}%")

    return perf


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Weekly Walk-Forward Loop — Year 3 out-of-sample"
    )
    parser.add_argument("--feature-dir", required=True,
                        help="Feature cache directory (built by build_feature_cache.py)")
    parser.add_argument("--results-dir", default="./results",
                        help="Output directory for CSVs, models, chart, JSON")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA GPU — auto-falls back to CPU if unavailable")
    parser.add_argument("--year2-only", action="store_true",
                        help="Shorter test window (Year 2 mode, faster)")
    args = parser.parse_args()

    run_weekly_loop(
        feature_dir = args.feature_dir,
        results_dir = args.results_dir,
        gpu         = args.gpu,
        year2_only  = args.year2_only,
    )


if __name__ == "__main__":
    main()
