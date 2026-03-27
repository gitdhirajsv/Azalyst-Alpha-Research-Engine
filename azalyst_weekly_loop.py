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

FIXES vs original:
  - future_ret_4h → future_ret  (aligns with build_feature_cache.py + notebook)
  - alpha_label now recomputed cross-sectionally AFTER pooling all symbols
    (old code read the wrong per-symbol label from cache)
  - MAX_TRAIN_ROWS: 4_000_000 → 2_000_000  (RTX 2050 4GB VRAM guard)
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
from azalyst_train import train_model, train_meta_model

warnings.filterwarnings("ignore")

# ── Config (matches notebook Cell 3) ─────────────────────────────────────────
RETRAIN_WEEKS    = 13        # quarterly retrain
TOP_QUANTILE     = 0.15      # top/bottom 15% for long/short
FEE_RATE         = 0.001     # 0.1% per leg
ROUND_TRIP_FEE   = FEE_RATE * 2
STOP_LOSS_PCT    = -2.0
TAKE_PROFIT_PCT  = 5.0
HORIZON_BARS     = 48        # 4H equivalent in 5-min bars

# FIX: was 4_000_000 — OOM on RTX 2050 4GB VRAM during quarterly retrain
# T4 (15GB) can use 4M; local RTX 2050 must stay at 2M.
MAX_TRAIN_ROWS   = 2_000_000


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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
                if pd.api.types.is_integer_dtype(df.index):
                    df.index = pd.to_datetime(df.index, unit='ms', utc=True)
                else:
                    df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            if df.index.max().year < 2018:
                continue   # 1970 timestamp — skip symbol
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
    Year 3 = final 1/3 of the total timeline.
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
    Build pooled (X, y, y_ret) from ALL symbols up to end_date.

    FIX: alpha_label is computed CROSS-SECTIONALLY after pooling all symbols.
    Old code read a per-symbol alpha_label from the cache file which was wrong —
    it just said "did price go up?" rather than "did this coin beat the median?"

    FIX: column is now 'future_ret' (not 'future_ret_4h').

    Caps at MAX_TRAIN_ROWS for VRAM safety (RTX 2050 4GB).
    """
    # ── Step 1: Pool all symbols up to end_date ───────────────────────────────
    symbol_dfs = []
    for sym, df in data.items():
        try:
            subset = df[df.index < end_date]
            if len(subset) < min_rows:
                continue

            # FIX: check for 'future_ret' (new name) not 'future_ret_4h'
            if "future_ret" not in subset.columns:
                continue

            # Fill any missing feature cols with 0
            for col in FEATURE_COLS:
                if col not in subset.columns:
                    subset = subset.copy()
                    subset[col] = 0.0

            subset = subset.copy()
            subset["_symbol"] = sym
            symbol_dfs.append(subset[FEATURE_COLS + ["future_ret", "_symbol"]])
        except Exception:
            pass

    if not symbol_dfs:
        print("  [ERROR] build_training_matrix: no valid symbol data found")
        return None, None, None

    # ── Step 2: Concatenate and sort by time ──────────────────────────────────
    pooled = pd.concat(symbol_dfs, axis=0).sort_index()
    print(f"  Pooled: {len(pooled):,} rows × {pooled['_symbol'].nunique()} symbols")

    # ── Step 3: Compute alpha_label cross-sectionally ─────────────────────────
    # FIX: this is the correct way — at each timestamp, label = 1 if this
    # coin's future_ret > median of ALL coins' future_ret at that timestamp.
    # This is what the notebook does with groupby(index).transform.
    pooled["alpha_label"] = (
        pooled.groupby(pooled.index)["future_ret"]
        .transform(lambda x: (x > x.median()).astype(float))
    )

    # ── Step 4: Extract arrays ────────────────────────────────────────────────
    feat   = pooled[FEATURE_COLS].values.astype(np.float32)
    labels = pooled["alpha_label"].values.astype(np.float32)
    rets   = pooled["future_ret"].values.astype(np.float32)

    valid  = np.isfinite(feat).all(axis=1) & np.isfinite(labels) & np.isfinite(rets)
    feat, labels, rets = feat[valid], labels[valid], rets[valid]

    if len(feat) < 50:
        print("  [ERROR] build_training_matrix: fewer than 50 valid rows after cleaning")
        return None, None, None

    # ── Step 5: VRAM cap for RTX 2050 ────────────────────────────────────────
    if len(feat) > MAX_TRAIN_ROWS:
        idx = np.random.choice(len(feat), MAX_TRAIN_ROWS, replace=False)
        idx.sort()
        feat, labels, rets = feat[idx], labels[idx], rets[idx]
        print(f"  VRAM guard: capped at {MAX_TRAIN_ROWS:,} rows")

    print(f"  Training matrix: {len(feat):,} rows × {len(FEATURE_COLS)} features | "
          f"Label balance: {labels.mean()*100:.1f}% positive")
    del pooled, symbol_dfs
    gc.collect()
    return feat, labels, rets


# ─────────────────────────────────────────────────────────────────────────────
#  WEEKLY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_week(model, scaler, data: dict, week_start, week_end,
                 meta_model=None, meta_scaler=None):
    """
    Run model inference for every symbol within the week window.
    Returns dicts of {symbol: mean_pred_prob}, {symbol: mean_actual_ret},
    and {symbol: meta_confidence} (empty dict if no meta model).

    FIX: uses 'future_ret' column (not 'future_ret_4h').
    """
    predictions = {}
    actual_rets = {}
    meta_confidences = {}

    for sym, df in data.items():
        try:
            week_data = df[(df.index >= week_start) & (df.index < week_end)]
            if len(week_data) < 3:
                continue

            # Fill missing feature cols
            for col in FEATURE_COLS:
                if col not in week_data.columns:
                    week_data = week_data.copy()
                    week_data[col] = 0.0

            feat  = week_data[FEATURE_COLS].values.astype(np.float32)
            valid = np.isfinite(feat).all(axis=1)
            if valid.sum() < 2:
                continue

            feat_scaled = scaler.transform(feat[valid])
            probs = model.predict_proba(feat_scaled)[:, 1]
            predictions[sym] = float(probs.mean())

            # Meta-labeling confidence (position sizing)
            if meta_model is not None and meta_scaler is not None:
                meta_input = np.column_stack([feat_scaled, probs.reshape(-1, 1)])
                meta_scaled = meta_scaler.transform(meta_input)
                meta_probs = meta_model.predict_proba(meta_scaled)[:, 1]
                meta_confidences[sym] = float(meta_probs.mean())

            # FIX: 'future_ret' not 'future_ret_4h'
            if "future_ret" in week_data.columns:
                ret_col     = week_data["future_ret"].values[valid]
                finite_ret  = ret_col[np.isfinite(ret_col)]
                if len(finite_ret) > 0:
                    actual_rets[sym] = float(finite_ret.mean())

        except Exception:
            pass

    return predictions, actual_rets, meta_confidences


# ─────────────────────────────────────────────────────────────────────────────
#  WEEKLY TRADE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_weekly_trades(predictions: dict, actual_rets: dict,
                           prev_longs: set = None, prev_shorts: set = None,
                           top_q: float = TOP_QUANTILE,
                           meta_confidences: dict = None):
    """
    Cross-sectional long/short simulation with position-aware fees
    and meta-labeling position sizing.

    Fees are charged only when a symbol ENTERS the portfolio (new position).
    Held positions (same side as last week) pay zero fees.

    When meta_confidences is provided, each trade's PnL is scaled by the
    meta-model's confidence (AFML Ch. 3).  High-confidence predictions
    get proportionally more weight in the portfolio return.

    Returns (trades, week_return_decimal, current_longs_set, current_shorts_set).
    """
    if prev_longs is None:
        prev_longs = set()
    if prev_shorts is None:
        prev_shorts = set()

    if not predictions:
        return [], 0.0, set(), set()

    pred_series = pd.Series(predictions)
    ranked      = pred_series.rank(pct=True)
    cur_longs   = set(ranked[ranked >= (1 - top_q)].index)
    cur_shorts  = set(ranked[ranked <= top_q].index)

    trades = []
    for sym in cur_longs:
        ret = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        fee = 0.0 if sym in prev_longs else ROUND_TRIP_FEE
        meta_size = meta_confidences.get(sym, 1.0) if meta_confidences else 1.0
        pnl = (ret - fee) * meta_size * 100
        trades.append({
            "symbol":      sym,
            "signal":      "BUY",
            "pred_prob":   round(predictions[sym], 5),
            "pnl_percent": round(pnl, 4),
            "raw_ret":     round(ret * 100, 4),
            "meta_size":   round(meta_size, 4),
        })

    for sym in cur_shorts:
        ret = actual_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        fee = 0.0 if sym in prev_shorts else ROUND_TRIP_FEE
        meta_size = meta_confidences.get(sym, 1.0) if meta_confidences else 1.0
        pnl = (-ret - fee) * meta_size * 100
        trades.append({
            "symbol":      sym,
            "signal":      "SELL",
            "pred_prob":   round(predictions[sym], 5),
            "pnl_percent": round(pnl, 4),
            "raw_ret":     round(ret * 100, 4),
            "meta_size":   round(meta_size, 4),
        })

    # Confidence-weighted weekly return
    if trades:
        sizes = np.array([t["meta_size"] for t in trades])
        pnls  = np.array([t["pnl_percent"] for t in trades])
        week_ret = float(np.average(pnls, weights=sizes)) / 100
    else:
        week_ret = 0.0

    return trades, week_ret, cur_longs, cur_shorts


# ─────────────────────────────────────────────────────────────────────────────
#  CHART SAVER
# ─────────────────────────────────────────────────────────────────────────────

def _save_chart(summary_df: pd.DataFrame, trades_df: pd.DataFrame, results_dir: str):
    """Save 4-panel Year 3 performance chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 11))
        fig.suptitle("Azalyst v2 — Year 3 Walk-Forward Performance", fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

        # Panel 1: Cumulative returns
        ax1 = fig.add_subplot(gs[0, 0])
        rets = summary_df["week_return_pct"].fillna(0) / 100
        cum  = ((1 + rets).cumprod() - 1) * 100
        ax1.plot(summary_df["week"].values, cum.values, color="#1f77b4", linewidth=2)
        ax1.fill_between(summary_df["week"].values, cum.values, alpha=0.12, color="#1f77b4")
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax1.set_title("Cumulative Return (%)", fontweight="bold")
        ax1.set_xlabel("Week #"); ax1.set_ylabel("%"); ax1.grid(True, alpha=0.25)

        # Panel 2: Weekly return distribution
        ax2 = fig.add_subplot(gs[0, 1])
        wr = summary_df["week_return_pct"].dropna()
        ax2.hist(wr, bins=min(30, max(10, len(wr)//3)), color="#ff7f0e", alpha=0.72,
                 edgecolor="black", linewidth=0.4)
        if len(wr) > 2:
            ax2.axvline(wr.mean(),   color="red",   linewidth=1.8, linestyle="--",
                        label=f"Mean {wr.mean():.2f}%")
            ax2.axvline(wr.median(), color="green", linewidth=1.2, linestyle=":",
                        label=f"Median {wr.median():.2f}%")
            ax2.legend(fontsize=9)
        ax2.set_title("Weekly Return Distribution", fontweight="bold")
        ax2.set_xlabel("Return (%)"); ax2.grid(True, alpha=0.25)

        # Panel 3: IC series
        ax3 = fig.add_subplot(gs[1, 0])
        ic_vals = summary_df["ic"].fillna(0)
        ax3.bar(summary_df["week"].values, ic_vals.values,
                color=["#2ca02c" if v > 0 else "#d62728" for v in ic_vals],
                alpha=0.75, width=0.8)
        if len(ic_vals) > 2:
            ax3.axhline(ic_vals.mean(), color="navy", linewidth=1.5, linestyle="--",
                        label=f"Mean IC {ic_vals.mean():.4f}")
            ax3.legend(fontsize=9)
        ax3.axhline(0, color="black", linewidth=0.6)
        ax3.set_title("Weekly IC (Information Coefficient)", fontweight="bold")
        ax3.set_xlabel("Week #"); ax3.grid(True, alpha=0.25)

        # Panel 4: Trade P&L distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if len(trades_df) > 0 and "pnl_percent" in trades_df.columns:
            pnl = trades_df["pnl_percent"].dropna()
            ax4.hist(pnl, bins=min(40, max(10, len(pnl)//20)), color="#9467bd",
                     alpha=0.72, edgecolor="black", linewidth=0.3)
            ax4.axvline(pnl.mean(), color="red", linewidth=1.8, linestyle="--",
                        label=f"Mean {pnl.mean():.3f}%")
            ax4.axvline(0, color="black", linewidth=0.8)
            ax4.legend(fontsize=9)
            ax4.set_title(f"Trade P&L (n={len(pnl):,})", fontweight="bold")
            ax4.set_xlabel("P&L (%)"); ax4.grid(True, alpha=0.25)

        chart_path = os.path.join(results_dir, "performance_year3.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart → {chart_path}")
    except Exception as e:
        print(f"[WeeklyLoop] Chart save failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WALK-FORWARD LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_weekly_loop(
    feature_dir:  str,
    results_dir:  str,
    gpu:          bool  = False,
    year2_only:   bool  = False,
):
    """
    Full Year 3 walk-forward loop.

    Steps:
      1. Load feature store (all symbols)
      2. Determine Year 3 date range
      3. Load or train base model (Year 1+2)
      4. For each week in Year 3:
           predict → trade → IC → save → retrain every 13 weeks
      5. Save all outputs + chart + performance JSON
    """
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load feature store (all symbols, no cap)
    data = load_feature_store(feature_dir)
    if not data:
        print("[WeeklyLoop] ERROR: No data found in feature_dir")
        return None

    # 2. Date splits
    year3_start, global_max = get_date_splits(data)

    if year2_only:
        year3_start = year3_start - pd.Timedelta(days=180)
        print("[WeeklyLoop] Year 2 only mode — test period shifted back 180 days")

    # 3. Load or train base model
    base_model_path  = os.path.join(models_dir, "model_base_y1y2.pkl")
    base_scaler_path = os.path.join(models_dir, "scaler_base_y1y2.pkl")

    if os.path.exists(base_model_path) and os.path.exists(base_scaler_path):
        print(f"[WeeklyLoop] Loading base model: {base_model_path}")
        with open(base_model_path, "rb") as f:
            model = pickle.load(f)
        with open(base_scaler_path, "rb") as f:
            scaler = pickle.load(f)
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
            pickle.dump(model, f)
        with open(base_scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        importance.to_csv(os.path.join(results_dir, "feature_importance_base.csv"))
        print(f"  Base model: AUC={auc:.4f}  IC={ic:.4f}  ICIR={icir:.4f}")
        del X, y, y_ret
        gc.collect()

    # 3b. Load or train meta-labeling model (AFML Ch. 3)
    meta_model_path  = os.path.join(models_dir, "meta_model_base.pkl")
    meta_scaler_path = os.path.join(models_dir, "meta_scaler_base.pkl")
    meta_model  = None
    meta_scaler = None

    if os.path.exists(meta_model_path) and os.path.exists(meta_scaler_path):
        print(f"[WeeklyLoop] Loading meta model: {meta_model_path}")
        with open(meta_model_path, "rb") as f:
            meta_model = pickle.load(f)
        with open(meta_scaler_path, "rb") as f:
            meta_scaler = pickle.load(f)
    else:
        print("[WeeklyLoop] Training meta-labeling model...")
        X, y, y_ret = build_training_matrix(data, year3_start)
        if X is not None:
            meta_model, meta_scaler = train_meta_model(
                model, scaler, X, y, FEATURE_COLS,
                label="meta_base", use_gpu=gpu,
            )
            if meta_model is not None:
                with open(meta_model_path, "wb") as f:
                    pickle.dump(meta_model, f)
                with open(meta_scaler_path, "wb") as f:
                    pickle.dump(meta_scaler, f)
            del X, y, y_ret
            gc.collect()

    # 4. Generate weekly timestamps
    weeks = pd.date_range(start=year3_start, end=global_max, freq="W-MON")
    if len(weeks) < 2:
        print("[WeeklyLoop] ERROR: Not enough weeks in test period")
        return None

    print(f"\n[WeeklyLoop] Starting {len(weeks)-1} weeks of walk-forward...\n")

    all_trades:             list = []
    weekly_summary:         list = []
    weekly_returns_history: list = []
    prev_longs:             set  = set()
    prev_shorts:            set  = set()

    for week_num, (week_start, week_end) in enumerate(zip(weeks[:-1], weeks[1:]), 1):

        # Predict (with optional meta-labeling confidence)
        predictions, actual_rets, meta_confs = predict_week(
            model, scaler, data, week_start, week_end,
            meta_model=meta_model, meta_scaler=meta_scaler,
        )

        if len(predictions) < 5:
            print(f"  Week {week_num:3d}: skipped — only {len(predictions)} symbols")
            continue

        # Trade simulation (position-tracked fees + meta-labeling sizing)
        trades, week_ret, cur_longs, cur_shorts = simulate_weekly_trades(
            predictions, actual_rets, prev_longs, prev_shorts,
            meta_confidences=meta_confs,
        )

        # IC
        if predictions and actual_rets:
            common   = [s for s in predictions if s in actual_rets]
            pred_arr = np.array([predictions[s] for s in common])
            ret_arr  = np.array([actual_rets[s]  for s in common])
            week_ic  = compute_ic(pred_arr, ret_arr)
        else:
            week_ic = 0.0

        ann_proj = ((1 + week_ret) ** 52 - 1) * 100
        weekly_returns_history.append(week_ret)

        # Retrain decision
        retrain_decision = should_retrain(weekly_returns_history)
        force_retrain    = (week_num % RETRAIN_WEEKS == 0)
        did_retrain      = False

        if force_retrain or retrain_decision["retrain"]:
            reason = "scheduled" if force_retrain else retrain_decision["reason"]
            print(f"  Week {week_num:3d}: RETRAINING — {reason}")
            X_new, y_new, y_ret_new = build_training_matrix(data, week_end)
            if X_new is not None:
                model_new, scaler_new, importance, auc_n, ic_n, icir_n = train_model(
                    X_new, y_new, y_ret_new, FEATURE_COLS,
                    label=f"y3_week{week_num:03d}", use_gpu=gpu,
                )
                pkl_path = os.path.join(models_dir, f"model_y3_week{week_num:03d}.pkl")
                sca_path = os.path.join(models_dir, f"scaler_y3_week{week_num:03d}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(model_new, f)
                with open(sca_path, "wb") as f:
                    pickle.dump(scaler_new, f)
                importance.to_csv(
                    os.path.join(results_dir, f"feature_importance_y3_week{week_num:03d}.csv")
                )
                model, scaler = model_new, scaler_new
                print(f"    → AUC={auc_n:.4f}  IC={ic_n:.4f}  ICIR={icir_n:.4f}")

                # Retrain meta-labeling model alongside primary
                meta_new, meta_scaler_new = train_meta_model(
                    model, scaler, X_new, y_new, FEATURE_COLS,
                    label=f"meta_week{week_num:03d}", use_gpu=gpu,
                )
                if meta_new is not None:
                    meta_model, meta_scaler = meta_new, meta_scaler_new
                    meta_pkl = os.path.join(models_dir, f"meta_model_week{week_num:03d}.pkl")
                    meta_sca = os.path.join(models_dir, f"meta_scaler_week{week_num:03d}.pkl")
                    with open(meta_pkl, "wb") as f:
                        pickle.dump(meta_model, f)
                    with open(meta_sca, "wb") as f:
                        pickle.dump(meta_scaler, f)

                did_retrain = True
                del X_new, y_new, y_ret_new
                gc.collect()

        # Tag trades with week metadata
        for t in trades:
            t["week"]       = week_num
            t["week_start"] = str(week_start.date())
        all_trades.extend(trades)

        # Turnover tracking
        n_cur = len(cur_longs) + len(cur_shorts)
        n_new = len(cur_longs - prev_longs) + len(cur_shorts - prev_shorts)
        turnover_pct = round(n_new / n_cur * 100, 1) if n_cur > 0 else 100.0
        prev_longs, prev_shorts = cur_longs, cur_shorts

        on_track_thresh = (1.10) ** (1 / 52) - 1
        weekly_summary.append({
            "week":             week_num,
            "week_start":       str(week_start.date()),
            "week_end":         str(week_end.date()),
            "n_symbols":        len(predictions),
            "n_trades":         len(trades),
            "week_return_pct":  round(week_ret * 100, 4),
            "annualised_pct":   round(ann_proj, 2),
            "ic":               round(week_ic, 5),
            "turnover_pct":     turnover_pct,
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
    trades_df.to_csv(trades_path,   index=False)

    # Performance JSON
    perf = session_report(summary_df, trades_df, label="Year3 Walk-Forward")
    ic_series = summary_df["ic"].dropna()
    perf["ic_mean"]         = round(float(ic_series.mean()), 5)
    perf["ic_std"]          = round(float(ic_series.std()),  5)
    perf["icir"]            = round(float(ic_series.mean() / (ic_series.std() + 1e-8)), 4)
    perf["ic_positive_pct"] = round(float((ic_series > 0).mean() * 100), 1)
    perf["max_train_rows"]  = MAX_TRAIN_ROWS

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
    parser.add_argument("--feature-dir",  required=True,
                        help="Feature cache directory (built by build_feature_cache.py)")
    parser.add_argument("--results-dir",  default="./results",
                        help="Output directory for CSVs, models, chart, JSON")
    parser.add_argument("--gpu",          action="store_true",
                        help="Use CUDA GPU — auto-falls back to CPU if unavailable")
    parser.add_argument("--year2-only",   action="store_true",
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
