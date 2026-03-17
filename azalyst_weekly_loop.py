"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  WEEKLY SELF-IMPROVING LOOP  v2  (Year 3 only)
╠══════════════════════════════════════════════════════════════════════════════╣
║  v2 CHANGES vs v1:                                                         ║
║   - Tests Year 3 only (Y1+Y2 already trained in azalyst_train.py)          ║
║   - Quarterly retrain (every 13 weeks) — was every week (OOM/slow)        ║
║   - XGBoost CUDA instead of LightGBM (confirmed GPU on Kaggle+RTX 2050)   ║
║   - IC (Information Coefficient) tracked every week alongside AUC         ║
║   - VRAM guard: caps retrain rows at 4M (safe for RTX 2050 / T4)          ║
║   - RobustScaler instead of StandardScaler                                 ║
║   - Purged K-Fold CV on retrains (no leakage)                              ║
║  TF FIX (v1.1): TRAIN_RESAMPLE explicit in all load_data calls            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python azalyst_weekly_loop.py --feature-dir ./feature_cache --results-dir ./results --gpu
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
from scipy import stats
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
    print("[WARN] xgboost not installed. Run: pip install xgboost")

from azalyst_alpha_metrics import (
    calculate_weekly_alpha, should_retrain, session_report,
    ROUND_TRIP_FEE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BARS_PER_DAY = 288
STOP_LOSS_PCT    = -2.0
TAKE_PROFIT_PCT  =  5.0
HORIZON_BARS     = 48      # 4H in 5-min bars
TOP_QUANTILE     = 0.20
BOTTOM_QUANTILE  = 0.20
RETRAIN_EVERY_N_WEEKS = 13  # quarterly — was weekly (too slow, OOM)
MAX_TRAIN_ROWS_GPU    = 4_000_000  # RTX 2050 / Kaggle T4 safe
TRAIN_RESAMPLE        = "4h"

FEATURE_COLS_V2 = [
    "ret_1bar", "ret_1h", "ret_4h", "ret_1d", "ret_2d", "ret_3d", "ret_1w",
    "vol_ratio", "vol_ret_1h", "vol_ret_1d", "obv_change", "vpt_change", "vol_momentum",
    "rvol_1h", "rvol_4h", "rvol_1d", "vol_ratio_1h_1d",
    "atr_norm", "parkinson_vol", "garman_klass",
    "rsi_14", "rsi_6", "macd_hist", "bb_pos", "bb_width",
    "stoch_k", "stoch_d", "cci_14", "adx_14", "dmi_diff",
    "vwap_dev", "amihud", "kyle_lambda", "spread_proxy", "body_ratio", "candle_dir",
    "wick_top", "wick_bot", "price_accel", "skew_1d", "kurt_1d", "max_ret_4h",
    "wq_alpha001", "wq_alpha012", "wq_alpha031", "wq_alpha098",
    "cs_momentum", "cs_reversal", "vol_adjusted_mom", "trend_consistency",
    "vol_regime", "trend_strength", "corr_btc_proxy", "hurst_exp", "fft_strength",
]

FEATURE_COLS_V1 = [
    "ret_1bar", "ret_1h", "ret_4h", "ret_1d",
    "vol_ratio", "vol_ret_1h", "vol_ret_1d",
    "body_ratio", "wick_top", "wick_bot", "candle_dir",
    "rvol_1h", "rvol_4h", "rvol_1d", "vol_ratio_1h_1d",
    "rsi_14", "rsi_6", "bb_pos", "bb_width",
    "vwap_dev", "ctrend_12", "ctrend_48", "price_accel",
    "skew_1d", "kurt_1d", "max_ret_4h", "amihud",
]


# ─────────────────────────────────────────────────────────────────────────────
#  XGB PARAMS  (RTX 2050 + Kaggle T4 tuned)
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_params(use_gpu: bool) -> dict:
    if use_gpu:
        return {
            "tree_method": "hist", "device": "cuda",
            "max_bin": 128, "learning_rate": 0.02,
            "max_depth": 6, "min_child_weight": 30,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "objective": "binary:logistic", "eval_metric": "auc",
            "verbosity": 0, "seed": 42,
        }
    return {
        "tree_method": "hist", "device": "cpu",
        "max_bin": 256, "learning_rate": 0.02,
        "max_depth": 6, "min_child_weight": 30,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "objective": "binary:logistic", "eval_metric": "auc",
        "verbosity": 0, "seed": 42,
        "nthread": 6,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GPU CHECK
# ─────────────────────────────────────────────────────────────────────────────

def detect_xgb_gpu() -> bool:
    if not _XGB:
        return False
    try:
        X_t = np.random.rand(500, 10).astype(np.float32)
        y_t = np.random.randint(0, 2, 500).astype(float)
        d   = xgb.DMatrix(X_t, label=y_t)
        xgb.train({"tree_method": "hist", "device": "cuda",
                   "max_bin": 64, "verbosity": 0,
                   "objective": "binary:logistic"}, d,
                  num_boost_round=5, verbose_eval=False)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  VRAM-SAFE SUBSAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def subsample_for_vram(X: np.ndarray, y: np.ndarray,
                       use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
    max_rows = MAX_TRAIN_ROWS_GPU if use_gpu else MAX_TRAIN_ROWS_GPU * 2
    n = len(X)
    if n <= max_rows:
        return X, y
    print(f"    [VRAM guard] {n:,} → {max_rows:,} rows")
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_keep  = max(10, int(max_rows * len(cls_idx) / n))
        if n_keep >= len(cls_idx):
            indices.append(cls_idx)
        else:
            step   = len(cls_idx) / n_keep
            chosen = cls_idx[np.round(np.arange(0, len(cls_idx), step))
                               .astype(int)[:n_keep]]
            indices.append(chosen)
    idx = np.sort(np.concatenate(indices))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  PURGED K-FOLD
# ─────────────────────────────────────────────────────────────────────────────

class PurgedKFold:
    def __init__(self, n_splits: int = 5, gap: int = 48):
        self.n_splits = n_splits
        self.gap      = gap

    def split(self, X: np.ndarray):
        n    = len(X)
        size = n // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            train_end  = i * size
            test_start = train_end + self.gap
            test_end   = test_start + size
            if test_end > n:
                break
            yield np.arange(0, train_end), np.arange(test_start, test_end)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data_for_window(feature_dir: Path, date_from: pd.Timestamp,
                         date_to: pd.Timestamp,
                         resample_freq: str = "4h",
                         verbose: bool = False) -> pd.DataFrame:
    frames = []
    for f in sorted(feature_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[(df.index >= date_from) & (df.index < date_to)]
            if len(df) < 10:
                continue
            avail_v2 = [c for c in FEATURE_COLS_V2 if c in df.columns]
            avail_v1 = [c for c in FEATURE_COLS_V1 if c in df.columns]
            avail = avail_v2 if len(avail_v2) > len(avail_v1) else avail_v1
            if len(avail) < 15:
                continue
            df_rs = df.resample(resample_freq).last().dropna(
                subset=avail, how="all")
            if len(df_rs) < 5:
                continue
            for col in avail:
                if col in df_rs.columns:
                    df_rs[col] = df_rs[col].astype("float32")
            df_rs["symbol"] = f.stem
            frames.append(df_rs)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-SECTIONAL RANK (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_rank(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    avail = [c for c in cols if c in df.columns]
    ranked = df.groupby(level=0, sort=False)[avail].rank(
        pct=True, na_option="keep")
    df = df.copy()
    df[avail] = ranked
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ALPHA LABEL
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "future_ret_4h" not in df.columns:
        return df
    df = df.copy()
    median_by_ts = df.groupby(level=0)["future_ret_4h"].transform("median")
    df["alpha_label"] = np.where(
        df["future_ret_4h"].notna(),
        (df["future_ret_4h"] > median_by_ts).astype("float32"),
        np.nan,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  IC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank IC between predictions and actual returns."""
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 10:
        return np.nan
    return float(stats.spearmanr(y_pred[mask], y_true[mask])[0])


# ─────────────────────────────────────────────────────────────────────────────
#  XGB TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X: np.ndarray, y: np.ndarray, feature_cols: List[str],
                  use_gpu: bool = False,
                  label: str = "") -> tuple:
    from sklearn.metrics import roc_auc_score

    scaler = RobustScaler()
    Xs     = scaler.fit_transform(X).astype(np.float32)
    Xs, ys = subsample_for_vram(Xs, y.astype(float), use_gpu)

    params = _xgb_params(use_gpu)
    cv     = PurgedKFold(n_splits=5, gap=48)
    aucs, ics = [], []

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        if len(np.unique(ys[val])) < 2:
            continue
        dtrain = xgb.DMatrix(Xs[tr], label=ys[tr])
        dval   = xgb.DMatrix(Xs[val], label=ys[val])
        try:
            m = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=[(dval, "val")],
                          early_stopping_rounds=50,
                          verbose_eval=False)
            proba = m.predict(dval)
            auc   = roc_auc_score(ys[val], proba)
            ic    = compute_ic(proba, ys[val])
            aucs.append(auc); ics.append(ic)
        except Exception as e:
            print(f"    Fold {fold} error: {e}")

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    mean_ic  = float(np.mean(ics))  if ics  else 0.0
    icir     = float(np.mean(ics) / np.std(ics)) if len(ics) > 1 and np.std(ics) > 0 else 0.0
    print(f"    CV AUC={mean_auc:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}")

    split  = int(len(Xs) * 0.9)
    dtrain = xgb.DMatrix(Xs[:split], label=ys[:split])
    dval   = xgb.DMatrix(Xs[split:], label=ys[split:])
    final  = xgb.train(params, dtrain, num_boost_round=1000,
                       evals=[(dval, "val")],
                       early_stopping_rounds=50,
                       verbose_eval=False)
    return final, scaler, mean_auc, mean_ic, icir


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(model, scaler, week_df: pd.DataFrame,
                     feature_cols: List[str]) -> pd.DataFrame:
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
            dtest = xgb.DMatrix(Xs)
            probs = model.predict(dtest)
        except Exception:
            continue
        week_df.loc[valid.index, "prob"] = probs
        n      = len(valid)
        n_long = max(1, int(n * TOP_QUANTILE))
        n_shrt = max(1, int(n * BOTTOM_QUANTILE))
        sorted_idx = valid.index[np.argsort(probs)]
        week_df.loc[sorted_idx[-n_long:], "signal"] = "BUY"
        week_df.loc[sorted_idx[:n_shrt],  "signal"] = "SELL"
    return week_df


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE SIMULATION  (always uses raw 5-min OHLCV)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trades(signals_df: pd.DataFrame, feature_dir: Path,
                    week_start: pd.Timestamp, week_end: pd.Timestamp) -> List[dict]:
    trades = []
    signal_rows = signals_df[signals_df["signal"].isin(["BUY", "SELL"])]

    for ts, row in signal_rows.iterrows():
        sym    = str(row.get("symbol", ""))
        signal = str(row["signal"])
        prob   = float(row.get("prob", 0.5))

        # Find raw OHLCV
        ohlcv_path = None
        for ddir in ["../data", "./data", "data"]:
            p = feature_dir.parent / ddir / f"{sym}.parquet"
            if p.exists(): ohlcv_path = p; break
            p2 = Path(ddir) / f"{sym}.parquet"
            if p2.exists(): ohlcv_path = p2; break

        if ohlcv_path is None:
            continue
        try:
            ohlcv = pd.read_parquet(ohlcv_path)
            ohlcv.columns = [c.lower() for c in ohlcv.columns]
            ts_col = next((c for c in ohlcv.columns
                           if c in ("timestamp", "time", "open_time")), None)
            if ts_col:
                col = ohlcv[ts_col]
                ohlcv.index = pd.to_datetime(
                    col, unit="ms" if pd.api.types.is_integer_dtype(col) else None,
                    utc=True)
                ohlcv = ohlcv.drop(columns=[ts_col])
            else:
                ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
            ohlcv = ohlcv[["open", "high", "low", "close"]].apply(
                pd.to_numeric, errors="coerce").dropna().sort_index()
        except Exception:
            continue

        future = ohlcv[ohlcv.index > ts].head(HORIZON_BARS + 10)
        if len(future) < 2:
            continue

        entry_price = float(future.iloc[0]["open"])
        if entry_price <= 0:
            continue

        if signal == "BUY":
            sl_p = entry_price * (1 + STOP_LOSS_PCT   / 100)
            tp_p = entry_price * (1 + TAKE_PROFIT_PCT / 100)
        else:
            sl_p = entry_price * (1 - STOP_LOSS_PCT   / 100)
            tp_p = entry_price * (1 - TAKE_PROFIT_PCT / 100)

        exit_price = None; exit_reason = "horizon"
        for _, bar in future.iloc[1:HORIZON_BARS + 1].iterrows():
            lo, hi = float(bar["low"]), float(bar["high"])
            if signal == "BUY":
                if lo <= sl_p: exit_price = sl_p; exit_reason = "stop_loss";   break
                if hi >= tp_p: exit_price = tp_p; exit_reason = "take_profit"; break
            else:
                if hi >= sl_p: exit_price = sl_p; exit_reason = "stop_loss";   break
                if lo <= tp_p: exit_price = tp_p; exit_reason = "take_profit"; break

        if exit_price is None:
            exit_price  = float(future.iloc[min(HORIZON_BARS, len(future)-1)]["close"])
            exit_reason = "horizon"

        raw_ret = exit_price / entry_price - 1
        if signal == "SELL": raw_ret = -raw_ret
        pnl_pct = (raw_ret - ROUND_TRIP_FEE) * 100

        trades.append({
            "signal_time": ts.isoformat(), "symbol": sym, "signal": signal,
            "probability": round(prob, 4),
            "entry_price": round(entry_price, 8),
            "exit_price":  round(exit_price,  8),
            "pnl_percent": round(pnl_pct, 4),
            "result":      "WIN" if pnl_pct > 0 else "LOSS",
            "exit_reason": exit_reason,
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_model(meta_path: Path) -> Tuple:
    """Load XGBoost model + scaler. meta_path = *_meta.pkl file."""
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    model_path = meta_path.with_suffix("").with_suffix(".json")
    if not model_path.exists():
        # Try .ubj format
        model_path = meta_path.with_suffix("").with_suffix(".ubj")
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model, meta["scaler"], meta["feature_cols"]


def save_model(model, scaler, feature_cols: List[str],
               base_path: Path, meta: dict = None) -> None:
    model.save_model(str(base_path.with_suffix(".json")))
    payload = {
        "scaler": scaler, "feature_cols": feature_cols,
        **(meta or {})
    }
    with open(str(base_path) + "_meta.pkl", "wb") as fh:
        pickle.dump(payload, fh)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_year3_loop(feature_dir: Path, results_dir: Path,
                   base_model_meta: Path, use_gpu: bool = False) -> Tuple:
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load base model trained on Y1+Y2
    model, scaler, feature_cols = load_model(base_model_meta)
    current_model_name = base_model_meta.stem

    # Load date config
    with open(results_dir / "date_config.json") as fh:
        dc = json.load(fh)
    year3_start = pd.Timestamp(dc["year12_end"],  tz="UTC")
    year3_end   = pd.Timestamp(dc["year3_end"],   tz="UTC")
    global_start = pd.Timestamp(dc["global_start"], tz="UTC")

    week_starts = pd.date_range(
        year3_start, year3_end - pd.Timedelta(weeks=1),
        freq="W-MON", tz="UTC"
    )

    print(f"\n{'═'*65}")
    print(f"  YEAR 3 WALK-FORWARD  ({len(week_starts)} weeks)")
    print(f"  {year3_start.date()} → {year3_end.date()}")
    print(f"  Retrain: every {RETRAIN_EVERY_N_WEEKS} weeks (quarterly)")
    print(f"  SL={STOP_LOSS_PCT}%  TP=+{TAKE_PROFIT_PCT}%")
    print(f"  GPU: {'YES (XGBoost CUDA)' if use_gpu else 'NO (CPU)'}")
    print(f"{'═'*65}")

    all_trades    = []
    weekly_summary = []
    weekly_returns = []
    weekly_ics     = []
    retrain_count  = 0

    for wk_num, week_start in enumerate(week_starts, 1):
        week_end = week_start + pd.Timedelta(weeks=1)
        if week_end > year3_end + pd.Timedelta(days=3):
            break

        t_wk = time.time()
        print(f"\n  Week {wk_num}/{len(week_starts)}: "
              f"{week_start.date()} → {week_end.date()}")

        # Load week data
        week_df = load_data_for_window(
            feature_dir, week_start, week_end,
            resample_freq=TRAIN_RESAMPLE, verbose=False)

        if week_df.empty:
            print(f"    [SKIP] No data")
            weekly_returns.append(0.0)
            weekly_ics.append(np.nan)
            weekly_summary.append(_empty_week_row(wk_num, week_start, week_end,
                                                  current_model_name, retrain_count))
            continue

        week_ranked = cross_sectional_rank(week_df, feature_cols)
        signals_df  = generate_signals(model, scaler, week_ranked, feature_cols)

        n_buy  = int((signals_df["signal"] == "BUY").sum())
        n_sell = int((signals_df["signal"] == "SELL").sum())
        print(f"    Signals: {n_buy + n_sell} (BUY={n_buy}, SELL={n_sell})")

        # Simulate trades
        week_trades = simulate_trades(signals_df, feature_dir, week_start, week_end)
        all_trades.extend(week_trades)

        trades_df = pd.DataFrame(week_trades) if week_trades else None
        alpha     = calculate_weekly_alpha(trades_df)
        weekly_returns.append(alpha["week_return_pct"] / 100.0)

        # IC for this week
        avail = [c for c in feature_cols if c in week_ranked.columns]
        probs_all, rets_all = [], []
        for ts, grp in signals_df.groupby(level=0):
            valid = grp.dropna(subset=avail + ["future_ret_4h"]
                               if "future_ret_4h" in grp.columns else avail)
            if len(valid) < 3 or "prob" not in valid.columns:
                continue
            probs_all.extend(valid["prob"].values)
            if "future_ret_4h" in valid.columns:
                rets_all.extend(valid["future_ret_4h"].values)
        week_ic = compute_ic(np.array(probs_all), np.array(rets_all)) \
                  if len(probs_all) > 10 and len(rets_all) == len(probs_all) else np.nan
        weekly_ics.append(week_ic)

        print(f"    Return: {alpha['week_return_pct']:+.2f}%  "
              f"IC: {week_ic:.4f}  "
              f"WR: {alpha['win_rate']:.0f}%  "
              f"Trades: {alpha['n_trades']}")

        # ── QUARTERLY RETRAIN CHECK ───────────────────────────────────────────
        retrained = False
        if wk_num % RETRAIN_EVERY_N_WEEKS == 0:
            print(f"    RETRAIN: quarterly (week {wk_num})")
            retrain_count += 1

            # Expanding window: train on all data up to now
            train_df = load_data_for_window(
                feature_dir, global_start, week_end,
                resample_freq=TRAIN_RESAMPLE, verbose=False)

            if not train_df.empty:
                train_df = build_alpha_labels(train_df)
                avail2   = [c for c in feature_cols if c in train_df.columns]
                valid2   = train_df.dropna(subset=avail2 + ["alpha_label"])
                valid2   = cross_sectional_rank(valid2, avail2)

                if len(valid2) > 500:
                    X2 = valid2[avail2].values.astype(np.float32)
                    y2 = valid2["alpha_label"].values.astype(float)
                    new_model, new_scaler, cv_auc, cv_ic, icir = train_xgboost(
                        X2, y2, avail2, use_gpu=use_gpu,
                        label=f"Y3_wk{wk_num}"
                    )
                    model     = new_model
                    scaler    = new_scaler
                    feature_cols = avail2

                    fname = f"model_y3_week{wk_num:03d}"
                    current_model_name = fname
                    save_model(model, scaler, feature_cols,
                               models_dir / fname,
                               {"cv_auc": round(cv_auc, 4),
                                "cv_ic":  round(cv_ic, 4),
                                "icir":   round(icir, 4),
                                "week":   wk_num})
                    print(f"    Retrained → {fname}  "
                          f"AUC={cv_auc:.4f}  IC={cv_ic:.4f}  ICIR={icir:.4f}")
                    retrained = True
                del train_df; gc.collect()

        weekly_summary.append({
            "week_num":         wk_num,
            "week_start":       week_start.date().isoformat(),
            "week_end":         week_end.date().isoformat(),
            "week_return_pct":  alpha["week_return_pct"],
            "annualised_pct":   alpha["annualised_pct"],
            "sharpe":           alpha["sharpe"],
            "win_rate":         alpha["win_rate"],
            "n_trades":         alpha["n_trades"],
            "weekly_ic":        round(week_ic, 5) if not np.isnan(week_ic) else None,
            "retrained":        retrained,
            "retrain_count":    retrain_count,
            "model":            current_model_name,
        })

        del week_df, week_ranked, signals_df; gc.collect()
        print(f"    Done in {time.time()-t_wk:.1f}s | Retrains: {retrain_count}")

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    wk_df = pd.DataFrame(weekly_summary)
    tr_df = pd.DataFrame(all_trades)
    wk_df.to_csv(results_dir / "weekly_summary_year3.csv", index=False)
    tr_df.to_csv(results_dir / "all_trades_year3.csv",     index=False)

    # IC series
    ic_series = pd.Series(weekly_ics, name="ic")
    icir_val  = float(ic_series.mean() / ic_series.std()) \
                if ic_series.std() > 0 else 0.0

    print(f"\n  Year 3 complete. Retrains: {retrain_count}")
    print(f"  Mean IC: {ic_series.mean():.4f}  ICIR: {icir_val:.4f}")
    return wk_df, tr_df


def _empty_week_row(wk_num, week_start, week_end, model_name, retrain_count):
    return {
        "week_num": wk_num,
        "week_start": week_start.date().isoformat(),
        "week_end":   week_end.date().isoformat(),
        "week_return_pct": 0.0, "annualised_pct": 0.0,
        "sharpe": 0.0, "win_rate": 0.0, "n_trades": 0,
        "weekly_ic": None, "retrained": False,
        "retrain_count": retrain_count, "model": model_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANCE CHART
# ─────────────────────────────────────────────────────────────────────────────

def save_performance_chart(wk_df: pd.DataFrame, tr_df: pd.DataFrame,
                            results_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Azalyst v2 — Year 3 Out-of-Sample Performance", fontsize=14)

        # 1. Cumulative return
        ax = axes[0][0]
        cum_ret = (1 + wk_df["week_return_pct"] / 100).cumprod() - 1
        ax.plot(cum_ret.values * 100, color="#21c16b", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Cumulative Return %"); ax.set_xlabel("Week"); ax.set_ylabel("%")
        ax.grid(True, alpha=0.3)

        # 2. Weekly return distribution
        ax = axes[0][1]
        ax.hist(wk_df["week_return_pct"].dropna(), bins=30,
                color="#1f77b4", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title("Weekly Return Distribution"); ax.set_xlabel("%")
        ax.grid(True, alpha=0.3)

        # 3. IC series
        ax = axes[1][0]
        ic_vals = wk_df["weekly_ic"].dropna()
        ax.bar(range(len(ic_vals)), ic_vals.values,
               color=["#21c16b" if v > 0 else "#f0626e" for v in ic_vals],
               alpha=0.7)
        ax.axhline(0, color="gray", linestyle="--")
        ax.axhline(0.02, color="orange", linestyle=":", label="IC=0.02 threshold")
        ax.set_title("Weekly IC (Information Coefficient)")
        ax.set_xlabel("Week"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Trade P&L distribution
        ax = axes[1][1]
        if not tr_df.empty:
            ax.hist(tr_df["pnl_percent"].clip(-5, 10), bins=50,
                    color="#ff7f0e", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--")
            win_rate = (tr_df["pnl_percent"] > 0).mean() * 100
            ax.set_title(f"Trade P&L Distribution (WR={win_rate:.1f}%)")
            ax.set_xlabel("P&L %")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(results_dir / "performance_year3.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved → performance_year3.png")
    except Exception as e:
        print(f"  Chart skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Year 3 Walk-Forward Loop v2"
    )
    parser.add_argument("--feature-dir",  default="./feature_cache")
    parser.add_argument("--results-dir",  default="./results")
    parser.add_argument("--gpu",          action="store_true")
    parser.add_argument("--retrain-weeks", type=int, default=RETRAIN_EVERY_N_WEEKS,
                        help=f"Retrain every N weeks (default {RETRAIN_EVERY_N_WEEKS})")
    args = parser.parse_args()

    global RETRAIN_EVERY_N_WEEKS
    RETRAIN_EVERY_N_WEEKS = args.retrain_weeks

    feature_dir = Path(args.feature_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("    AZALYST  —  WEEKLY LOOP v2  (Year 3, quarterly retrain)")
    print("╚══════════════════════════════════════════════════════════════╝")

    # GPU check
    use_gpu = False
    if args.gpu:
        print("\n[1] Checking GPU...")
        use_gpu = detect_xgb_gpu()
        print(f"    GPU: {'XGBoost CUDA ready ✓' if use_gpu else 'falling back to CPU'}")

    # Find base model
    models_dir = results_dir / "models"
    meta_path  = models_dir / "model_base_y1y2_meta.pkl"
    if not meta_path.exists():
        print(f"\n[ERROR] Base model not found: {meta_path}")
        print("  Run azalyst_train_local.py first.")
        return

    # Run Year 3
    wk_df, tr_df = run_year3_loop(
        feature_dir, results_dir, meta_path, use_gpu=use_gpu)

    # Performance summary
    if not wk_df.empty:
        session_report(wk_df, tr_df, label="Year 3 Out-of-Sample")

    # IC/ICIR summary
    ic_series = wk_df["weekly_ic"].dropna()
    icir_val  = float(ic_series.mean() / ic_series.std()) \
                if len(ic_series) > 1 and ic_series.std() > 0 else 0.0

    performance = {
        "period":           "Year 3 out-of-sample",
        "total_weeks":      len(wk_df),
        "total_trades":     len(tr_df),
        "mean_weekly_ret":  round(float(wk_df["week_return_pct"].mean()), 4),
        "mean_weekly_ic":   round(float(ic_series.mean()), 5) if len(ic_series) else None,
        "icir":             round(icir_val, 4),
        "ic_pct_positive":  round(float((ic_series > 0).mean() * 100), 1) if len(ic_series) else None,
        "overall_win_rate": round(float((tr_df["pnl_percent"] > 0).mean() * 100), 2)
                            if not tr_df.empty else 0.0,
        "total_retrains":   int(wk_df["retrained"].sum()),
        "elapsed_hours":    round((time.time() - t0) / 3600, 2),
        "gpu_used":         use_gpu,
    }
    with open(results_dir / "performance_year3.json", "w") as fh:
        json.dump(performance, fh, indent=2)

    print("\n  Performance summary:")
    for k, v in performance.items():
        print(f"    {k:<25} {v}")

    # Save chart
    save_performance_chart(wk_df, tr_df, results_dir)

    print(f"\n  Results saved to: {results_dir.resolve()}")
    print(f"  Total time: {(time.time()-t0)/3600:.2f} hours")


if __name__ == "__main__":
    main()
