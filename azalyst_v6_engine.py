"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AZALYST ALPHA RESEARCH ENGINE  v6.0  —  CONSENSUS REBUILD          ║
║                                                                            ║
║  Synthesized from 7 independent model recommendations:                     ║
║    R0 (Opus 4.6)   — Diagnostics, data-driven regime analysis              ║
║    R1 (DeepSeek)   — Beta-neutral target, rolling window                   ║
║    R2 (Gemini 3.1) — No shorts in BULL (data-proven +7.19%)               ║
║    R3 (GLM5)       — Regime-specific behavior                              ║
║    R4 (Qwen)       — force-invert = target sign problem → Elastic Net      ║
║    R5 (GPT 5.4)    — Falsification campaign, feature turnover cap          ║
║    R6 (Mistral)    — IC-gated retraining, feature leakage test             ║
║                                                                            ║
║  KEY CHANGES FROM V5:                                                      ║
║   1. Elastic Net default model (XGBoost as optional challenger)            ║
║   2. Beta-neutral target (cross-sectional demeaned returns)                ║
║   3. Rolling 26-week window (not expanding)                                ║
║   4. 10 stable features + turnover cap (max 3 changes per retrain)         ║
║   5. Regime-gated portfolio (no shorts in BULL_TREND)                      ║
║   6. Built-in falsification campaign (prove signal exists first)           ║
║   7. Long/short PnL decomposition                                         ║
║   8. Feature stability reporting (Jaccard across retrains)                 ║
║   9. 4-gate kill criteria                                                  ║
║  10. No --force-invert (broken signal = broken signal)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Import reusable infrastructure from v5 ────────────────────────────────────
from azalyst_factors_v2 import build_features, FEATURE_COLS
from azalyst_risk import RiskManager
from azalyst_db import AzalystDB
from azalyst_v5_engine import (
    LazySymbolStore,
    detect_regime,
    build_feature_store,
    inspect_feature_store,
    load_feature_store,
    get_date_splits,
    compute_drawdown,
    save_checkpoint,
    load_checkpoint,
    detect_cuda_api,
    _gpu_cleanup,
    _model_to_cpu,
    PurgedTimeSeriesCV,
)
from azalyst_train import compute_ic

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: V6 CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR    = "./data"
RESULTS_DIR = "./results_v6"
CACHE_DIR   = "./feature_cache"

# Horizons
HORIZON_BARS_1H  = 12
HORIZON_BARS     = 12

# Target — same as v5 but training uses beta-neutral version
TARGET_COL          = "future_ret_1h"
TARGET_COL_FALLBACK = "future_ret"

# Rolling window (not expanding) — consensus from 5/7 recs
ROLLING_WINDOW_WEEKS = 26

# Retraining interval
RETRAIN_WEEKS = 13

# Portfolio
FEE_RATE       = 0.001
ROUND_TRIP_FEE = FEE_RATE * 2
DEFAULT_TOP_N  = 5       # Conservative default (GPT 5.4: weak ranker → fewer picks)

# Kill switches
MAX_DRAWDOWN_KILL = -0.15

# Feature stability
MAX_FEATURE_TURNOVER = 3   # Max features to add/remove per retrain (GPT 5.4)
MIN_IC_PERIODS_TO_ADD = 2  # Positive IC in ≥2 periods to be added
MAX_TRAIN_ROWS = 2_000_000


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: V6 STABLE FEATURE SET
# ══════════════════════════════════════════════════════════════════════════════

# CORE features — NEVER dropped (persisted across all v5 retrains)
V6_CORE_FEATURES = [
    "ret_1w",              # 1-week return — momentum/reversal
    "ret_3d",              # 3-day return — short-term momentum
    "vol_regime",          # Volatility regime — state variable
]

# STABLE features — default set, economically interpretable
V6_STABLE_FEATURES = [
    "rvol_1d",             # Daily realized volatility
    "rsi_14",              # Mean reversion indicator
    "skew_1d",             # Distribution asymmetry / tail risk
    "adx_14",              # Trend strength
    "kyle_lambda",         # Price impact / liquidity
    "mean_rev_zscore_1h",  # Z-score of 1hr reversion
    "vol_ratio_1h_1d",     # Intraday vs daily vol ratio
]

# Full default set: core + stable = 10 features
V6_DEFAULT_FEATURES = V6_CORE_FEATURES + V6_STABLE_FEATURES

# CANDIDATE pool — can be added if IC is strong and stable
V6_CANDIDATE_FEATURES = [
    "ret_1d", "ret_2d", "rev_1h", "rev_1d",
    "rvol_4h", "atr_norm", "cci_14", "bb_pos",
    "vwap_dev", "amihud", "trend_strength",
    "frac_diff_close", "vol_ret_1d",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FEATURE STABILITY TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureStabilityTracker:
    """Track feature IC over time and enforce turnover cap.

    Rules (from GPT 5.4 / consensus):
    - Core features are NEVER dropped
    - Max 3 features added/removed per retrain
    - Need positive IC in ≥2 recent periods to be added
    - Need negative IC in ≥3 recent periods to be dropped
    """

    def __init__(self, core: List[str], initial_set: List[str],
                 max_turnover: int = MAX_FEATURE_TURNOVER):
        self.core = set(core)
        self.active = list(initial_set)
        self.max_turnover = max_turnover
        self.ic_history: Dict[str, List[float]] = {}
        self.retrain_feature_sets: List[Set[str]] = [set(initial_set)]

    def record_ic(self, feature_ics: Dict[str, float]):
        """Record IC measurements for all features."""
        for feat, ic_val in feature_ics.items():
            if feat not in self.ic_history:
                self.ic_history[feat] = []
            self.ic_history[feat].append(ic_val)

    def propose_update(self, candidate_pool: List[str]) -> List[str]:
        """Propose a new feature set respecting turnover cap."""
        current_set = set(self.active)
        proposed = set(self.core)  # Core always stays

        # Evaluate current non-core features for removal
        removals = []
        for feat in self.active:
            if feat in self.core:
                continue
            ics = self.ic_history.get(feat, [])
            if len(ics) >= 3:
                recent = ics[-4:]  # last 4 measurements
                neg_count = sum(1 for ic in recent if ic < 0)
                if neg_count >= 3:
                    removals.append(feat)
                else:
                    proposed.add(feat)  # Keep it
            else:
                proposed.add(feat)  # Not enough history to judge, keep

        # Evaluate candidates for addition
        additions = []
        for feat in candidate_pool:
            if feat in proposed:
                continue
            ics = self.ic_history.get(feat, [])
            if len(ics) >= 2:
                recent = ics[-4:]
                pos_count = sum(1 for ic in recent if ic > 0)
                if pos_count >= MIN_IC_PERIODS_TO_ADD:
                    avg_ic = float(np.mean(recent))
                    additions.append((feat, avg_ic))

        # Sort additions by average IC (best first)
        additions.sort(key=lambda x: -x[1])

        # Apply turnover cap
        n_removed = min(len(removals), self.max_turnover)
        actual_removals = removals[:n_removed]
        for feat in actual_removals:
            proposed.discard(feat)

        remaining_budget = self.max_turnover - n_removed
        n_added = min(len(additions), remaining_budget)
        for feat, _ in additions[:n_added]:
            proposed.add(feat)

        new_set = list(proposed)
        self.active = new_set
        self.retrain_feature_sets.append(set(new_set))
        return new_set

    def jaccard_overlap(self) -> float:
        """Compute average Jaccard overlap between consecutive feature sets."""
        if len(self.retrain_feature_sets) < 2:
            return 1.0
        overlaps = []
        for i in range(1, len(self.retrain_feature_sets)):
            a = self.retrain_feature_sets[i - 1]
            b = self.retrain_feature_sets[i]
            if len(a | b) > 0:
                overlaps.append(len(a & b) / len(a | b))
        return float(np.mean(overlaps)) if overlaps else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING MATRIX (rolling window + beta-neutral target)
# ══════════════════════════════════════════════════════════════════════════════

def build_training_matrix_v6(symbols, train_end, features: List[str],
                             rolling_weeks: int = ROLLING_WINDOW_WEEKS,
                             beta_neutral: bool = True) -> Tuple:
    """Build training matrix with ROLLING window and BETA-NEUTRAL target.

    Changes from v5:
    - Rolling window (not expanding): only uses last `rolling_weeks` of data
    - Beta-neutral: subtracts daily cross-sectional mean from targets
    - Returns (X, y_raw, y_neutral, timestamps) — timestamps for diagnostics
    """
    rolling_start = train_end - pd.Timedelta(weeks=rolling_weeks)
    safe_end = train_end - pd.Timedelta(minutes=5 * HORIZON_BARS_1H)

    print(f"  Building v6 training matrix [{rolling_start.date()} → {safe_end.date()}]"
          f" ({len(features)} features, rolling={rolling_weeks}wk)...")

    n_feat = len(features)
    initial = 500_000
    feat_arr = np.empty((initial, n_feat), dtype=np.float32)
    ret_arr = np.empty(initial, dtype=np.float32)
    ts_arr = np.empty(initial, dtype="datetime64[ns]")
    cursor = 0

    def grow(needed):
        nonlocal feat_arr, ret_arr, ts_arr
        if needed <= len(ret_arr):
            return
        new_size = max(needed, int(len(ret_arr) * 1.5))
        feat_arr = np.resize(feat_arr, (new_size, n_feat))
        ret_arr = np.resize(ret_arr, new_size)
        ts_arr = np.resize(ts_arr, new_size)

    rng = np.random.default_rng(42)
    eligible = 0

    for sym, df in symbols.items():
        tcol = TARGET_COL if TARGET_COL in df.columns else TARGET_COL_FALLBACK
        if tcol not in df.columns:
            continue

        # ROLLING WINDOW: only data within the window
        mask = (df.index >= rolling_start) & (df.index < safe_end)
        subset = df.loc[mask, features + [tcol]]
        if len(subset) < HORIZON_BARS + 50:
            continue
        eligible += 1

        f = subset[features].to_numpy(dtype=np.float32)
        r = subset[tcol].to_numpy(dtype=np.float32)
        ts = subset.index.values

        valid = np.isfinite(f).all(axis=1) & np.isfinite(r)
        keep = np.flatnonzero(valid)

        # P13: Non-overlapping return subsample (every 12th bar for 1hr target)
        if len(keep) > 0:
            keep = keep[::HORIZON_BARS_1H]
        if len(keep) == 0:
            continue

        end = cursor + len(keep)
        grow(end)
        feat_arr[cursor:end] = f[keep]
        ret_arr[cursor:end] = r[keep]
        ts_arr[cursor:end] = ts[keep]
        cursor = end

    feat_arr = feat_arr[:cursor]
    ret_arr = ret_arr[:cursor]
    ts_arr = ts_arr[:cursor]

    if len(feat_arr) < 50:
        print("  ERROR: fewer than 50 valid rows")
        return None, None, None, None

    # VRAM guard
    if len(feat_arr) > MAX_TRAIN_ROWS:
        idx = rng.choice(len(feat_arr), MAX_TRAIN_ROWS, replace=False)
        idx.sort()
        feat_arr, ret_arr, ts_arr = feat_arr[idx], ret_arr[idx], ts_arr[idx]

    # Beta-neutral target: subtract daily cross-sectional mean
    y_neutral = ret_arr.copy()
    if beta_neutral:
        # Round timestamps to day for grouping
        ts_days = pd.DatetimeIndex(ts_arr).normalize()
        unique_days = ts_days.unique()
        n_demeaned = 0
        for day in unique_days:
            day_mask = ts_days == day
            n_in_group = day_mask.sum()
            if n_in_group > 1:
                group_mean = float(ret_arr[day_mask].mean())
                y_neutral[day_mask] -= group_mean
                n_demeaned += n_in_group
        pct_demeaned = n_demeaned / len(y_neutral) * 100 if len(y_neutral) > 0 else 0
        print(f"  Beta-neutral: demeaned {pct_demeaned:.1f}% of rows "
              f"({len(unique_days)} daily groups)")

    print(f"  Training matrix: {len(feat_arr):,} rows × {n_feat} features | "
          f"{eligible} symbols | "
          f"target mean={float(np.mean(y_neutral))*100:.4f}% "
          f"std={float(np.std(y_neutral))*100:.4f}%")

    gc.collect()
    return feat_arr, ret_arr, y_neutral, ts_arr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MODEL TRAINING (Elastic Net + XGBoost challenger)
# ══════════════════════════════════════════════════════════════════════════════

def train_elastic_net(X, y, features: List[str],
                      label: str = "") -> Tuple:
    """Train Elastic Net with built-in alpha/l1_ratio CV.

    Returns: (model, scaler, importance, mean_r2, mean_ic, icir)
    """
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Elastic Net with built-in cross-validation for alpha selection
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        n_alphas=50,
        cv=5,
        max_iter=10000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xs, y)

    # Evaluate with PurgedTimeSeriesCV for honest metrics
    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    r2s, ics = [], []

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        m = ElasticNet(
            alpha=model.alpha_,
            l1_ratio=model.l1_ratio_,
            max_iter=10000,
            random_state=42,
        )
        m.fit(Xs[tr], y[tr])
        preds = m.predict(Xs[val])

        ss_res = float(np.sum((y[val] - preds) ** 2))
        ss_tot = float(np.sum((y[val] - np.mean(y[val])) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2s.append(r2)

        ic = compute_ic(preds, y[val])
        if np.isfinite(ic):
            ics.append(ic)

    mean_r2 = float(np.mean(r2s)) if r2s else 0.0
    mean_ic = float(np.mean(ics)) if ics else 0.0
    icir = float(np.mean(ics) / (np.std(ics) + 1e-8)) if len(ics) > 1 else 0.0

    # Feature importance from coefficients
    coefs = np.abs(model.coef_)
    importance = pd.Series(coefs, index=features, name="importance"
                           ).sort_values(ascending=False)

    n_nonzero = int((np.abs(model.coef_) > 1e-8).sum())
    print(f"  [{label}] ElasticNet: alpha={model.alpha_:.6f}  "
          f"l1_ratio={model.l1_ratio_:.2f}  "
          f"nonzero={n_nonzero}/{len(features)}  "
          f"R²={mean_r2:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}")

    return model, scaler, importance, mean_r2, mean_ic, icir


def train_xgb_challenger(X, y, features: List[str], cuda_api,
                         label: str = "") -> Tuple:
    """Train XGBoost as a challenger model. Must beat Elastic Net to be used."""
    import xgboost as xgb

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Conservative XGBoost params for small feature set
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,         # Shallow — only 10 features
        min_child_weight=100, # Very conservative — prevent overfitting
        subsample=0.7,
        colsample_bytree=0.8,
        reg_alpha=1.0,       # Strong L1
        reg_lambda=5.0,      # Strong L2
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
    )
    if cuda_api == "new":
        params["device"] = "cuda"
    elif cuda_api == "old":
        params["tree_method"] = "gpu_hist"

    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    r2s, ics = [], []

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        m = xgb.XGBRegressor(**params)
        try:
            m.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)
        except Exception:
            params_cpu = {k: v for k, v in params.items()
                         if k not in ("device", "tree_method")}
            m = xgb.XGBRegressor(**params_cpu)
            m.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)

        preds = m.predict(Xs[val])
        ss_res = float(np.sum((y[val] - preds) ** 2))
        ss_tot = float(np.sum((y[val] - np.mean(y[val])) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2s.append(r2)
        ic = compute_ic(preds, y[val])
        if np.isfinite(ic):
            ics.append(ic)

    mean_r2 = float(np.mean(r2s)) if r2s else 0.0
    mean_ic = float(np.mean(ics)) if ics else 0.0
    icir = float(np.mean(ics) / (np.std(ics) + 1e-8)) if len(ics) > 1 else 0.0

    # Final model on 90/10 split
    final = xgb.XGBRegressor(**params)
    split = int(len(Xs) * 0.9)
    try:
        final.fit(Xs[:split], y[:split],
                  eval_set=[(Xs[split:], y[split:])], verbose=False)
    except Exception:
        params_cpu = {k: v for k, v in params.items()
                      if k not in ("device", "tree_method")}
        final = xgb.XGBRegressor(**params_cpu)
        final.fit(Xs[:split], y[:split],
                  eval_set=[(Xs[split:], y[split:])], verbose=False)

    importance = pd.Series(final.feature_importances_, index=features,
                           name="importance").sort_values(ascending=False)
    _model_to_cpu(final)
    _gpu_cleanup()

    n_trees = final.best_ntree_limit if hasattr(final, "best_ntree_limit") else "?"
    print(f"  [{label}] XGBoost:    trees={n_trees}  "
          f"R²={mean_r2:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}")

    return final, scaler, importance, mean_r2, mean_ic, icir


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_week_v6(model, scaler, symbols, week_start, week_end,
                    features_used: List[str], is_linear: bool = True):
    """Pre-week snapshot prediction (same leakage fix as v5, no confidence model).

    Returns: (predictions, actual_close_rets)
    """
    predictions = {}
    actual_close_rets = {}

    for sym, df in symbols.items():
        try:
            pre_week = df[df.index < week_start]
            if len(pre_week) < 1:
                continue

            # Use last HORIZON_BARS_1H rows before week_start
            pre_snap = pre_week.iloc[-HORIZON_BARS_1H:]
            feat = pre_snap[features_used].values.astype(np.float32)
            valid = np.isfinite(feat).all(axis=1)
            if valid.sum() < 1:
                continue

            feat_scaled = scaler.transform(feat[valid])
            pred_rets = model.predict(feat_scaled)
            predictions[sym] = float(np.mean(pred_rets))

            # Actual weekly close-to-close return
            week_data = df[(df.index >= week_start) & (df.index < week_end)]
            if len(week_data) < 2 or "close" not in week_data.columns:
                continue
            c_start = float(week_data["close"].iloc[0])
            c_end = float(week_data["close"].iloc[-1])
            if c_start > 0 and np.isfinite(c_start) and np.isfinite(c_end):
                actual_close_rets[sym] = float(np.log(c_end / c_start))
        except Exception:
            pass

    return predictions, actual_close_rets


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: PORTFOLIO CONSTRUCTION (regime-gated, long/short decomposition)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_weekly_trades_v6(predictions, actual_close_rets,
                              prev_longs, prev_shorts,
                              regime: str,
                              leverage: float = 1.0,
                              top_n: int = DEFAULT_TOP_N):
    """Regime-gated portfolio construction with long/short PnL decomposition.

    Regime rules (consensus from all 7 recs):
    - BULL_TREND:       Long-only, half position size (NO SHORTS)
    - BEAR_TREND:       Full long-short
    - LOW_VOL_GRIND:    Full long-short
    - HIGH_VOL_LATERAL: Long-short, half position size
    """
    if not predictions:
        return [], 0.0, 0.0, 0.0, set(), set()

    pred_series = pd.Series(predictions)
    n_symbols = len(pred_series)
    n = min(top_n, n_symbols // 2)

    if n < 1:
        return [], 0.0, 0.0, 0.0, set(), set()

    sorted_syms = pred_series.sort_values(ascending=False)
    cur_longs = set(sorted_syms.head(n).index)

    # Regime gating: NO SHORTS in BULL_TREND (data-proven: removing BULL = +7.19%)
    if regime == "BULL_TREND":
        cur_shorts = set()
        position_scale = 0.5 * leverage
    elif regime == "HIGH_VOL_LATERAL":
        cur_shorts = set(sorted_syms.tail(n).index)
        position_scale = 0.5 * leverage
    else:  # BEAR_TREND, LOW_VOL_GRIND
        cur_shorts = set(sorted_syms.tail(n).index)
        position_scale = 1.0 * leverage

    trades = []
    long_pnl_sum = 0.0
    short_pnl_sum = 0.0
    n_long = 0
    n_short = 0

    for sym in cur_longs:
        ret = actual_close_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        fee = 0.0 if sym in prev_longs else ROUND_TRIP_FEE
        pnl = (ret - fee) * position_scale
        trades.append({
            "symbol": sym, "signal": "BUY",
            "pred_ret": round(predictions[sym] * 100, 5),
            "pnl_percent": round(pnl * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "position_scale": round(position_scale, 4),
        })
        long_pnl_sum += ret - fee
        n_long += 1

    for sym in cur_shorts:
        ret = actual_close_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        fee = 0.0 if sym in prev_shorts else ROUND_TRIP_FEE
        pnl = (-ret - fee) * position_scale
        trades.append({
            "symbol": sym, "signal": "SELL",
            "pred_ret": round(predictions[sym] * 100, 5),
            "pnl_percent": round(pnl * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "position_scale": round(position_scale, 4),
        })
        short_pnl_sum += -ret - fee
        n_short += 1

    # Equal-weight average returns per leg
    long_ret = (long_pnl_sum / n_long * position_scale) if n_long > 0 else 0.0
    short_ret = (short_pnl_sum / n_short * position_scale) if n_short > 0 else 0.0
    total_positions = n_long + n_short
    week_ret = (long_pnl_sum / max(n_long, 1) + short_pnl_sum / max(n_short, 1)) / (
        2.0 if n_short > 0 else 1.0) * position_scale if total_positions > 0 else 0.0

    return trades, week_ret, long_ret, short_ret, cur_longs, cur_shorts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: FALSIFICATION CAMPAIGN (GPT 5.4: "prove signal exists first")
# ══════════════════════════════════════════════════════════════════════════════

def run_falsification(symbols, test_weeks, active_features: List[str],
                      top_n: int = DEFAULT_TOP_N):
    """Test single-factor baselines against Elastic Net.

    Purpose: Determine if ML adds anything over naive cross-sectional ranking.
    If single-factor sort beats Elastic Net, the ML is adding noise.

    Baselines tested:
    1. ret_1w rank alone
    2. ret_3d rank alone
    3. vol_regime rank alone
    4. Equal-weight composite (average rank of core features)
    5. Random predictions (null hypothesis)

    Returns: dict of {baseline_name: avg_weekly_ic}
    """
    print(f"\n{'='*72}")
    print(f"  V6 FALSIFICATION CAMPAIGN — Prove Signal Exists")
    print(f"  Testing {len(test_weeks)-1} weeks with top-{top_n} per side")
    print(f"{'='*72}\n")

    baselines = {
        "ret_1w": lambda df: df["ret_1w"].iloc[-1] if "ret_1w" in df.columns else 0.0,
        "ret_3d": lambda df: df["ret_3d"].iloc[-1] if "ret_3d" in df.columns else 0.0,
        "vol_regime": lambda df: -df["vol_regime"].iloc[-1] if "vol_regime" in df.columns else 0.0,
        "random": lambda df: np.random.randn(),
    }

    results = {name: [] for name in baselines}
    results["composite"] = []

    for i in range(len(test_weeks) - 1):
        ws = test_weeks[i]
        we = test_weeks[i + 1]

        # Gather pre-week features and actual returns for all symbols
        sym_features = {}
        actual_rets = {}

        for sym, df in symbols.items():
            try:
                pre_week = df[df.index < ws]
                if len(pre_week) < 1:
                    continue

                pre_snap = pre_week.iloc[-1]
                sym_features[sym] = pre_snap

                week_data = df[(df.index >= ws) & (df.index < we)]
                if len(week_data) < 2 or "close" not in week_data.columns:
                    continue
                c_s = float(week_data["close"].iloc[0])
                c_e = float(week_data["close"].iloc[-1])
                if c_s > 0 and np.isfinite(c_s) and np.isfinite(c_e):
                    actual_rets[sym] = float(np.log(c_e / c_s))
            except Exception:
                pass

        common = set(sym_features.keys()) & set(actual_rets.keys())
        if len(common) < 10:
            continue

        ret_arr = np.array([actual_rets[s] for s in common])

        # Test each baseline
        for name, score_fn in baselines.items():
            scores = []
            for s in common:
                try:
                    scores.append(score_fn(sym_features[s]))
                except Exception:
                    scores.append(0.0)
            scores = np.array(scores)
            valid = np.isfinite(scores) & np.isfinite(ret_arr)
            if valid.sum() >= 10:
                ic = float(stats.spearmanr(scores[valid], ret_arr[valid])[0])
                results[name].append(ic)

        # Composite: average z-score of core features
        composite_scores = []
        for s in common:
            vals = []
            for feat in V6_CORE_FEATURES:
                if feat in sym_features[s].index:
                    v = sym_features[s][feat]
                    if np.isfinite(v):
                        vals.append(v)
            composite_scores.append(float(np.mean(vals)) if vals else 0.0)
        composite_scores = np.array(composite_scores)
        valid = np.isfinite(composite_scores) & np.isfinite(ret_arr)
        if valid.sum() >= 10:
            ic = float(stats.spearmanr(composite_scores[valid], ret_arr[valid])[0])
            results["composite"].append(ic)

    # Print results
    print(f"  {'Baseline':<20} {'Mean IC':>10} {'Std IC':>10} {'ICIR':>10} {'IC>0%':>10}")
    print(f"  {'─'*60}")
    summary = {}
    for name in ["ret_1w", "ret_3d", "vol_regime", "composite", "random"]:
        ics = results.get(name, [])
        if not ics:
            print(f"  {name:<20} {'N/A':>10}")
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        icir = mean_ic / (std_ic + 1e-8)
        pct_pos = float(np.mean([1 for ic in ics if ic > 0])) * 100
        print(f"  {name:<20} {mean_ic:>+10.4f} {std_ic:>10.4f} {icir:>+10.4f} {pct_pos:>9.1f}%")
        summary[name] = {"mean_ic": mean_ic, "icir": icir, "pct_pos": pct_pos}

    print()

    # Identify best baseline
    best_name = max(summary, key=lambda k: summary[k]["mean_ic"]) if summary else "none"
    best_ic = summary[best_name]["mean_ic"] if summary else 0.0
    print(f"  Best baseline: {best_name} (IC={best_ic:+.4f})")
    print(f"  ML must beat IC={best_ic:+.4f} to justify its complexity.\n")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: 4-GATE KILL CRITERIA (GPT 5.4)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_kill_criteria(weekly_summary: List[dict],
                           feature_tracker: FeatureStabilityTracker,
                           falsification_results: dict) -> dict:
    """Evaluate whether the strategy should continue.

    4 gates — ALL must pass to continue:
    1. OOS IC consistently positive (mean > 0, positive >50% of weeks)
    2. Feature set stable across retrains (Jaccard > 0.5)
    3. Returns survive regime decomposition (positive in ≥2 regimes)
    4. Simple models confirm signal (ML IC ≥ best baseline IC)
    """
    gates = {}

    # Gate 1: OOS IC consistently positive
    ics = [m["ic"] for m in weekly_summary
           if isinstance(m.get("ic"), (int, float)) and m.get("regime") != "IC_GATED"]
    if ics:
        mean_ic = float(np.mean(ics))
        pct_pos = float(np.mean([1 for ic in ics if ic > 0])) * 100
        gates["oos_ic_positive"] = mean_ic > 0 and pct_pos > 50
        gates["oos_ic_detail"] = f"mean={mean_ic:+.4f}, pos={pct_pos:.0f}%"
    else:
        gates["oos_ic_positive"] = False
        gates["oos_ic_detail"] = "no data"

    # Gate 2: Feature stability
    jaccard = feature_tracker.jaccard_overlap()
    gates["feature_stable"] = jaccard > 0.5
    gates["feature_detail"] = f"jaccard={jaccard:.3f}"

    # Gate 3: Regime survival
    regime_rets = {}
    for m in weekly_summary:
        r = m.get("regime", "UNKNOWN")
        if r in ("IC_GATED", "KILL_SWITCH"):
            continue
        if r not in regime_rets:
            regime_rets[r] = []
        regime_rets[r].append(m.get("week_return_pct", 0.0))

    positive_regimes = sum(1 for r, rets in regime_rets.items()
                           if np.mean(rets) > 0)
    gates["regime_survival"] = positive_regimes >= 2
    regime_detail = ", ".join(f"{r}={np.mean(v):+.2f}%" for r, v in regime_rets.items())
    gates["regime_detail"] = regime_detail

    # Gate 4: ML beats baseline
    if falsification_results:
        best_baseline_ic = max(v.get("mean_ic", 0) for v in falsification_results.values())
        ml_ic = float(np.mean(ics)) if ics else 0.0
        gates["ml_beats_baseline"] = ml_ic > best_baseline_ic
        gates["ml_vs_baseline"] = f"ML={ml_ic:+.4f} vs best_baseline={best_baseline_ic:+.4f}"
    else:
        gates["ml_beats_baseline"] = True  # No falsification data, pass by default
        gates["ml_vs_baseline"] = "no falsification data"

    all_pass = all(gates.get(k, False) for k in
                   ["oos_ic_positive", "feature_stable", "regime_survival", "ml_beats_baseline"])
    gates["ALL_PASS"] = all_pass

    return gates


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: FEATURE IC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_ic_v6(symbols, week_start, week_end,
                          features: List[str]) -> Dict[str, float]:
    """Compute cross-sectional IC for each feature over a week."""
    feature_ics = {}
    week_data = {}

    for sym, df in symbols.items():
        tcol = TARGET_COL if TARGET_COL in df.columns else TARGET_COL_FALLBACK
        mask = (df.index >= week_start) & (df.index < week_end)
        subset = df.loc[mask]
        if len(subset) < 3 or tcol not in subset.columns:
            continue
        week_data[sym] = (subset, tcol)

    if len(week_data) < 5:
        return {f: 0.0 for f in features}

    for feat in features:
        feat_vals = []
        ret_vals = []
        for sym, (subset, tcol) in week_data.items():
            if feat in subset.columns:
                valid = subset[[feat, tcol]].dropna()
                if len(valid) > 0:
                    feat_vals.append(float(valid[feat].mean()))
                    ret_vals.append(float(valid[tcol].mean()))
        if len(feat_vals) >= 5:
            ic, _ = stats.spearmanr(feat_vals, ret_vals)
            feature_ics[feat] = float(ic) if np.isfinite(ic) else 0.0
        else:
            feature_ics[feat] = 0.0

    return feature_ics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: CHECKPOINT (v6 paths)
# ══════════════════════════════════════════════════════════════════════════════

def _ckpt_path_v6(results_dir):
    return os.path.join(results_dir, "checkpoint_v6_latest.json")


def save_checkpoint_v6(results_dir, state):
    path = _ckpt_path_v6(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


def load_checkpoint_v6(results_dir):
    path = _ckpt_path_v6(results_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            ckpt = json.load(f)
        print(f"  [CHECKPOINT] Found v6  run_id={ckpt.get('run_id')}  "
              f"last_week={ckpt.get('last_week')}  ts={ckpt.get('ts', '?')}")
        return ckpt
    except Exception as e:
        print(f"  [CHECKPOINT] Could not load ({e}) — starting fresh")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Azalyst v6 — Consensus Rebuild")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-dd", type=float, default=MAX_DRAWDOWN_KILL)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--target", default="1h", choices=["1h", "1d", "5d"],
                        help="Forward return horizon")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Top N longs + N shorts per week (default: 5)")
    parser.add_argument("--rolling-window", type=int, default=ROLLING_WINDOW_WEEKS,
                        help="Rolling window in weeks (default: 26)")
    parser.add_argument("--no-falsify", action="store_true",
                        help="Skip falsification campaign")
    parser.add_argument("--xgb-challenger", action="store_true",
                        help="Also train XGBoost as challenger model")
    parser.add_argument("--pin-coins", type=str, default="",
                        help="Comma-separated symbols to restrict universe")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    global DATA_DIR, RESULTS_DIR, CACHE_DIR, TARGET_COL, HORIZON_BARS

    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.feature_dir:
        CACHE_DIR = args.feature_dir
    if args.out_dir:
        RESULTS_DIR = args.out_dir

    target_map = {"1h": "future_ret_1h", "1d": "future_ret_1d", "5d": "future_ret_5d"}
    target_bars = {"1h": 12, "1d": 288, "5d": 1440}
    TARGET_COL = target_map.get(args.target, "future_ret_1h")
    HORIZON_BARS = target_bars.get(args.target, 12)

    use_gpu = args.gpu and not args.no_gpu
    dd_kill = args.max_dd

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("\n" + "=" * 72)
    print("  AZALYST v6  —  Consensus Rebuild Engine")
    print("=" * 72)
    print(f"  Model        : Elastic Net (linear)"
          f"{'  +  XGBoost challenger' if args.xgb_challenger else ''}")
    print(f"  Target       : {TARGET_COL} (beta-neutral)")
    print(f"  Window       : Rolling {args.rolling_window} weeks")
    print(f"  Features     : {len(V6_DEFAULT_FEATURES)} stable"
          f" + turnover cap {MAX_FEATURE_TURNOVER}")
    print(f"  Portfolio    : top-{args.top_n} per side, regime-gated")
    print(f"  Compute      : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"  Leverage     : {args.leverage:.1f}x")
    print(f"  Kill-switch  : {dd_kill*100:.0f}% max drawdown")
    print(f"  Falsification: {'enabled' if not args.no_falsify else 'disabled'}")
    print("=" * 72 + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    _log_path = os.path.join(RESULTS_DIR, "run_log_v6.txt")
    ckpt = None if args.no_resume else load_checkpoint_v6(RESULTS_DIR)
    resuming = ckpt is not None
    _log_mode = "a" if resuming else "w"
    try:
        _log_fh = open(_log_path, _log_mode, encoding="utf-8", buffering=1)
    except Exception:
        _log_fh = None

    def _log(msg: str = "") -> None:
        print(msg)
        if _log_fh:
            try:
                _log_fh.write(msg + "\n")
                _log_fh.flush()
            except Exception:
                pass

    db = AzalystDB(f"{RESULTS_DIR}/azalyst_v6.db")

    if resuming:
        run_id = ckpt["run_id"]
        _log(f"\n  [CHECKPOINT] Resuming run_id={run_id}  "
             f"from week {ckpt['last_week'] + 1}\n")
    else:
        run_id = args.run_id or f"v6_{time.strftime('%Y%m%d_%H%M%S')}"
        db.start_run(run_id, {
            "version": "v6_consensus", "gpu": use_gpu,
            "features": len(V6_DEFAULT_FEATURES),
            "max_dd_kill": dd_kill, "retrain_weeks": RETRAIN_WEEKS,
            "rolling_window_weeks": args.rolling_window,
            "horizon_bars": HORIZON_BARS,
            "model_type": "ElasticNet",
            "top_n": args.top_n,
        })

    risk_mgr = RiskManager()
    cuda_api = detect_cuda_api() if use_gpu else None

    # ── STEP 0: Feature cache ─────────────────────────────────────────────────
    _log("STEP 0: Feature cache\n")
    data_file_count = len(list(Path(DATA_DIR).glob("*.parquet"))) if Path(DATA_DIR).exists() else 0
    if args.rebuild_cache:
        if not build_feature_store():
            _log("ERROR: Feature store build failed")
            return
    else:
        total, valid, invalid = inspect_feature_store()
        cache_incomplete = (data_file_count > 0 and valid < int(data_file_count * 0.80))
        if total == 0 or cache_incomplete:
            _log(f"  Cache incomplete: {valid}/{data_file_count} — building...")
            if not build_feature_store():
                return
        elif invalid:
            build_feature_store()
        else:
            _log(f"  Found {valid} valid cache files")

    # ── STEP 1: Load symbols ──────────────────────────────────────────────────
    _log("\nSTEP 1: Load feature cache\n")
    symbols = load_feature_store()
    if not symbols:
        _log("ERROR: No symbols loaded")
        return
    _log(f"  Loaded {len(symbols)} valid symbols")

    # ── STEP 2: Date splits ───────────────────────────────────────────────────
    _log("\nSTEP 2: Date splits\n")
    global_min, global_max, y1_end, y2_end = get_date_splits(symbols)

    os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)

    # ── Initialize feature stability tracker ──────────────────────────────────
    active_features = list(V6_DEFAULT_FEATURES)
    feature_tracker = FeatureStabilityTracker(
        core=V6_CORE_FEATURES,
        initial_set=active_features,
        max_turnover=MAX_FEATURE_TURNOVER,
    )

    # ── STEP 3: Falsification campaign ────────────────────────────────────────
    falsification_results = {}
    if not args.no_falsify and not resuming:
        _log("\nSTEP 3: Falsification campaign\n")
        # Use first 13 weeks of Y2 for falsification
        falsify_start = y1_end
        falsify_end = y1_end + pd.Timedelta(weeks=13)
        falsify_weeks = pd.date_range(start=falsify_start, end=min(falsify_end, global_max),
                                      freq="W-MON")
        if len(falsify_weeks) >= 3:
            falsification_results = run_falsification(
                symbols, falsify_weeks, active_features, top_n=args.top_n)
        else:
            _log("  Not enough weeks for falsification")
    else:
        _log("\nSTEP 3: Falsification skipped\n")

    # ── STEP 4: Initial training ──────────────────────────────────────────────
    if resuming:
        _log("\nSTEP 4: Loading model from checkpoint...\n")
        active_features = ckpt["active_features"]
        model_path = ckpt["current_model_path"]
        scaler_path = ckpt["current_scaler_path"]

        with open(model_path, "rb") as f:
            current_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            current_scaler = pickle.load(f)

        is_linear = ckpt.get("is_linear", True)
        feature_tracker = FeatureStabilityTracker(
            core=V6_CORE_FEATURES, initial_set=active_features)
        # Restore IC history
        for k, v in ckpt.get("feature_ic_history", {}).items():
            feature_tracker.ic_history[k] = list(v)
    else:
        _log("\nSTEP 4: Initial training on Y1 (rolling window)\n")

        # Use rolling window ending at y1_end
        X_train, y_raw, y_neutral, ts_train = build_training_matrix_v6(
            symbols, y1_end, active_features,
            rolling_weeks=args.rolling_window,
            beta_neutral=True,
        )
        if X_train is None:
            _log("ERROR: Could not build training matrix")
            return

        _log(f"\n  Training Elastic Net on {len(X_train):,} rows...")
        t0 = time.time()
        current_model, current_scaler, importance, mean_r2, mean_ic, icir = \
            train_elastic_net(X_train, y_neutral, active_features, label="base_y1")
        is_linear = True
        _log(f"  Time: {time.time()-t0:.1f}s")

        # Save model
        model_path = f"{RESULTS_DIR}/models/model_v6_base.pkl"
        scaler_path = f"{RESULTS_DIR}/models/scaler_v6_base.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(current_model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(current_scaler, f)
        importance.to_csv(f"{RESULTS_DIR}/feature_importance_v6_base.csv")

        # Optional XGBoost challenger
        if args.xgb_challenger:
            _log(f"\n  Training XGBoost challenger...")
            t0 = time.time()
            xgb_model, xgb_scaler, xgb_imp, xgb_r2, xgb_ic, xgb_icir = \
                train_xgb_challenger(X_train, y_neutral, active_features,
                                     cuda_api, label="xgb_y1")
            _log(f"  Time: {time.time()-t0:.1f}s")

            # Select winner
            ic_margin = 0.005  # XGBoost must beat by this margin
            if xgb_ic > mean_ic + ic_margin:
                _log(f"  >>> XGBoost WINS (IC {xgb_ic:+.4f} > {mean_ic:+.4f} + {ic_margin})")
                current_model, current_scaler = xgb_model, xgb_scaler
                importance = xgb_imp
                is_linear = False
                with open(model_path, "wb") as f:
                    pickle.dump(current_model, f)
                with open(scaler_path, "wb") as f:
                    pickle.dump(current_scaler, f)
            else:
                _log(f"  >>> Elastic Net WINS (XGB IC {xgb_ic:+.4f} "
                     f"does not beat EN IC {mean_ic:+.4f} by {ic_margin})")
            _gpu_cleanup()

        db.insert_model_artifact(run_id, "base_y1", 0, model_path, scaler_path,
                                 mean_r2, mean_ic, icir, len(active_features))

        with open(f"{RESULTS_DIR}/train_summary_v6.json", "w") as f:
            json.dump({
                "mean_r2": round(mean_r2, 5), "mean_ic": round(mean_ic, 5),
                "icir": round(icir, 5), "n_rows": int(len(X_train)),
                "n_features": len(active_features),
                "model_type": "ElasticNet" if is_linear else "XGBoost",
                "beta_neutral": True,
                "rolling_window_weeks": args.rolling_window,
            }, f, indent=2)

        del X_train, y_raw, y_neutral, ts_train
        gc.collect()

    # ── STEP 5: Walk-forward backtest ─────────────────────────────────────────
    walk_start = y1_end
    _log(f"\nSTEP 5: Walk-forward  ({walk_start.date()} → {global_max.date()})\n")

    weeks = pd.date_range(start=walk_start, end=global_max, freq="W-MON")
    if len(weeks) < 2:
        _log("  Not enough weeks")
        return

    current_model_path = model_path
    current_scaler_path = scaler_path

    if resuming:
        retrains = ckpt["retrains"]
        prev_longs = set(ckpt["prev_longs"])
        prev_shorts = set(ckpt["prev_shorts"])
        all_trades = ckpt["all_trades"]
        weekly_summary = ckpt["weekly_summary"]
        weekly_returns = ckpt["weekly_returns"]
        kill_switch_hit = ckpt.get("kill_switch_hit", False)
        ks_pause_until = ckpt.get("ks_pause_until", 0)
        resume_from_week = ckpt["last_week"]
        is_linear = ckpt.get("is_linear", True)
        _log(f"  Restored {len(weekly_returns)} weeks. "
             f"Resuming from week {resume_from_week + 1}.\n")
    else:
        retrains = 0
        prev_longs, prev_shorts = set(), set()
        all_trades = []
        weekly_summary = []
        weekly_returns = []
        kill_switch_hit = False
        ks_pause_until = 0
        resume_from_week = 0

    for week_num, (ws, we) in enumerate(zip(weeks[:-1], weeks[1:]), 1):
        if week_num <= resume_from_week:
            continue

        # Kill-switch pause
        if week_num <= ks_pause_until:
            continue

        current_dd = compute_drawdown(weekly_returns)
        if current_dd < dd_kill:
            _log(f"\n  *** KILL SWITCH *** Week {week_num}: "
                 f"DD={current_dd*100:.1f}% < {dd_kill*100:.0f}%")
            kill_switch_hit = True
            ks_pause_until = min(week_num + 3, len(weeks) - 1)
            for skip in range(4):
                if week_num + skip <= len(weeks) - 1:
                    weekly_returns.append(0.0)
                    weekly_summary.append({
                        "week": week_num + skip,
                        "week_start": str(ws.date()),
                        "week_end": str(we.date()),
                        "regime": "KILL_SWITCH",
                        "n_trades": 0, "week_return_pct": 0.0,
                        "long_return_pct": 0.0, "short_return_pct": 0.0,
                        "ic": 0.0,
                    })
            continue

        regime = detect_regime(symbols, we)

        # Feature IC computation every 2 weeks
        if week_num > 1 and week_num % 2 == 0:
            # Compute IC for ALL features (active + candidates)
            all_feats_to_check = list(set(active_features + V6_CANDIDATE_FEATURES))
            fic = compute_feature_ic_v6(symbols, ws, we, all_feats_to_check)
            feature_tracker.record_ic(fic)

        # Quarterly retrain with ROLLING WINDOW
        if week_num % RETRAIN_WEEKS == 0:
            _log(f"\n  Week {week_num:3d}: QUARTERLY RETRAIN "
                 f"(rolling {args.rolling_window}wk to {we.date()})...")

            # Feature stability: propose update with turnover cap
            new_features = feature_tracker.propose_update(V6_CANDIDATE_FEATURES)
            jaccard = feature_tracker.jaccard_overlap()
            _log(f"    Features: {len(active_features)} → {len(new_features)} "
                 f"(Jaccard={jaccard:.3f})")
            active_features = new_features

            X_rt, y_raw_rt, y_neutral_rt, ts_rt = build_training_matrix_v6(
                symbols, we, active_features,
                rolling_weeks=args.rolling_window,
                beta_neutral=True,
            )

            if X_rt is not None and len(X_rt) > 200:
                t0 = time.time()
                m_new, s_new, imp_new, r2_n, ic_n, icir_n = train_elastic_net(
                    X_rt, y_neutral_rt, active_features,
                    label=f"v6_w{week_num:03d}")

                # IC-gated retraining (Mistral): only adopt if OOS IC > 0
                if ic_n > 0:
                    current_model, current_scaler = m_new, s_new
                    is_linear = True

                    model_path_new = f"{RESULTS_DIR}/models/model_v6_week{week_num:03d}.pkl"
                    scaler_path_new = f"{RESULTS_DIR}/models/scaler_v6_week{week_num:03d}.pkl"
                    with open(model_path_new, "wb") as f:
                        pickle.dump(m_new, f)
                    with open(scaler_path_new, "wb") as f:
                        pickle.dump(s_new, f)
                    current_model_path = model_path_new
                    current_scaler_path = scaler_path_new
                    imp_new.to_csv(
                        f"{RESULTS_DIR}/feature_importance_v6_week{week_num:03d}.csv")
                    _log(f"    Adopted new model (IC={ic_n:+.4f} > 0)  "
                         f"({time.time()-t0:.1f}s)")

                    # Optional XGBoost challenger at retrain
                    if args.xgb_challenger:
                        xgb_m, xgb_s, _, _, xgb_ic, _ = train_xgb_challenger(
                            X_rt, y_neutral_rt, active_features, cuda_api,
                            label=f"xgb_w{week_num:03d}")
                        if xgb_ic > ic_n + 0.005:
                            _log(f"    >>> XGBoost beats EN at retrain "
                                 f"(IC {xgb_ic:+.4f} > {ic_n:+.4f})")
                            current_model, current_scaler = xgb_m, xgb_s
                            is_linear = False
                            with open(current_model_path, "wb") as f:
                                pickle.dump(current_model, f)
                            with open(current_scaler_path, "wb") as f:
                                pickle.dump(current_scaler, f)
                        _gpu_cleanup()
                else:
                    _log(f"    Retrain REJECTED (IC={ic_n:+.4f} ≤ 0) — "
                         f"keeping previous model  ({time.time()-t0:.1f}s)")

                retrains += 1
                del X_rt, y_raw_rt, y_neutral_rt, ts_rt
                gc.collect()

        # Prediction
        predictions, actual_close_rets = predict_week_v6(
            current_model, current_scaler, symbols, ws, we,
            active_features, is_linear=is_linear)

        # Pin-coins filter
        if args.pin_coins:
            allowed = set(s.strip().upper() for s in args.pin_coins.split(",") if s.strip())
            predictions = {k: v for k, v in predictions.items() if k in allowed}
            actual_close_rets = {k: v for k, v in actual_close_rets.items() if k in allowed}

        if len(predictions) < 5:
            weekly_returns.append(0.0)
            weekly_summary.append({
                "week": week_num, "week_start": str(ws.date()),
                "week_end": str(we.date()), "regime": regime,
                "n_trades": 0, "week_return_pct": 0.0,
                "long_return_pct": 0.0, "short_return_pct": 0.0,
                "ic": 0.0,
            })
            continue

        # Regime-gated portfolio construction
        trades, week_ret, long_ret, short_ret, cur_longs, cur_shorts = \
            simulate_weekly_trades_v6(
                predictions, actual_close_rets,
                prev_longs, prev_shorts,
                regime=regime,
                leverage=args.leverage,
                top_n=args.top_n,
            )
        weekly_returns.append(week_ret)

        # Compute weekly IC
        common = [s for s in predictions if s in actual_close_rets]
        if len(common) >= 10:
            pred_arr = np.array([predictions[s] for s in common])
            ret_arr = np.array([actual_close_rets[s] for s in common])
            week_ic = float(stats.spearmanr(pred_arr, ret_arr)[0])
        else:
            week_ic = 0.0

        # Turnover
        n_cur = len(cur_longs) + len(cur_shorts)
        n_new = len(cur_longs - prev_longs) + len(cur_shorts - prev_shorts)
        turnover = round(n_new / n_cur * 100, 1) if n_cur > 0 else 100.0
        prev_longs, prev_shorts = cur_longs, cur_shorts

        cum_ret = float(np.prod([1 + r for r in weekly_returns]) - 1)
        max_dd = compute_drawdown(weekly_returns)

        for t in trades:
            t["week"] = week_num
            t["week_start"] = str(ws.date())
        all_trades.extend(trades)

        zone = "Y2" if we <= y2_end else "Y3"

        metric = {
            "week": week_num, "week_start": str(ws.date()),
            "week_end": str(we.date()),
            "n_symbols": len(predictions), "n_trades": len(trades),
            "week_return_pct": round(week_ret * 100, 4),
            "long_return_pct": round(long_ret * 100, 4),
            "short_return_pct": round(short_ret * 100, 4),
            "ic": round(week_ic, 5),
            "turnover_pct": turnover,
            "cum_return_pct": round(cum_ret * 100, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "regime": regime,
        }
        weekly_summary.append(metric)
        db.insert_weekly_metric(run_id, metric)
        db.insert_trades(run_id, trades)

        n_short_display = len(cur_shorts)
        _log(f"  Week {week_num:3d} [{zone}]: {len(trades):3d} trades "
             f"({len(cur_longs)}L/{n_short_display}S) | "
             f"ret={week_ret*100:+.2f}% (L={long_ret*100:+.2f}% S={short_ret*100:+.2f}%) | "
             f"IC={week_ic:+.4f} | cum={cum_ret*100:+.1f}% | "
             f"DD={max_dd*100:.1f}% | {regime}")

        # Save checkpoint
        save_checkpoint_v6(RESULTS_DIR, {
            "run_id": run_id,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_week": week_num,
            "weekly_returns": weekly_returns,
            "weekly_summary": weekly_summary,
            "all_trades": all_trades,
            "prev_longs": list(prev_longs),
            "prev_shorts": list(prev_shorts),
            "retrains": retrains,
            "active_features": active_features,
            "feature_ic_history": {k: list(v)
                                   for k, v in feature_tracker.ic_history.items()},
            "kill_switch_hit": kill_switch_hit,
            "ks_pause_until": ks_pause_until,
            "current_model_path": current_model_path,
            "current_scaler_path": current_scaler_path,
            "is_linear": is_linear,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # REPORTING
    # ══════════════════════════════════════════════════════════════════════════

    trades_df = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(weekly_summary)

    if len(trades_df) > 0:
        trades_df.to_csv(f"{RESULTS_DIR}/all_trades_v6.csv", index=False)
    if len(summary_df) > 0:
        summary_df.to_csv(f"{RESULTS_DIR}/weekly_summary_v6.csv", index=False)

    n_wks = len(weekly_returns)
    cum_ret = float(np.prod([1 + r for r in weekly_returns]) - 1) if n_wks else 0.0
    ann_ret = ((1 + cum_ret) ** (52 / max(n_wks, 1)) - 1) * 100 if n_wks else 0.0
    wk_std = float(np.std(weekly_returns)) if n_wks > 1 else 0.0
    sharpe = float(np.mean(weekly_returns)) / wk_std * np.sqrt(52) if wk_std > 0 else 0.0
    max_dd = compute_drawdown(weekly_returns)

    ic_s = summary_df["ic"] if len(summary_df) > 0 else pd.Series(dtype=float)
    ic_mean = float(ic_s.mean()) if len(ic_s) > 0 else 0.0
    ic_std = float(ic_s.std()) if len(ic_s) > 1 else 0.0
    icir_val = ic_mean / (ic_std + 1e-8)

    # Long/short PnL decomposition
    long_rets = [m.get("long_return_pct", 0) for m in weekly_summary
                 if m.get("regime") not in ("KILL_SWITCH",)]
    short_rets = [m.get("short_return_pct", 0) for m in weekly_summary
                  if m.get("regime") not in ("KILL_SWITCH",)]
    total_long_pnl = sum(long_rets)
    total_short_pnl = sum(short_rets)

    # Regime decomposition
    regime_stats = {}
    for m in weekly_summary:
        r = m.get("regime", "UNKNOWN")
        if r in ("KILL_SWITCH",):
            continue
        if r not in regime_stats:
            regime_stats[r] = {"rets": [], "ics": []}
        regime_stats[r]["rets"].append(m.get("week_return_pct", 0))
        regime_stats[r]["ics"].append(m.get("ic", 0))

    # 4-Gate Kill Criteria
    _log(f"\n{'='*72}")
    _log(f"  4-GATE KILL CRITERIA EVALUATION")
    _log(f"{'='*72}\n")

    gates = evaluate_kill_criteria(weekly_summary, feature_tracker,
                                   falsification_results)
    for key in ["oos_ic_positive", "feature_stable", "regime_survival",
                "ml_beats_baseline"]:
        status = "PASS" if gates.get(key, False) else "FAIL"
        detail = gates.get(key.replace("positive", "detail")
                           .replace("stable", "detail")
                           .replace("survival", "detail")
                           .replace("baseline", "vs_baseline"), "")
        # Find the right detail key
        detail_map = {
            "oos_ic_positive": "oos_ic_detail",
            "feature_stable": "feature_detail",
            "regime_survival": "regime_detail",
            "ml_beats_baseline": "ml_vs_baseline",
        }
        detail = gates.get(detail_map.get(key, ""), "")
        _log(f"  Gate {key:<25s}: [{status}]  {detail}")

    verdict = "CONTINUE" if gates["ALL_PASS"] else "REVIEW / PIVOT"
    _log(f"\n  VERDICT: {verdict}\n")

    # Performance report
    perf = {
        "label": "v6_Consensus_WalkForward",
        "run_id": run_id,
        "total_weeks": n_wks,
        "total_trades": len(trades_df),
        "retrains": retrains,
        "total_return_pct": round(cum_ret * 100, 2),
        "annualised_pct": round(ann_ret, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "ic_mean": round(ic_mean, 5),
        "icir": round(icir_val, 4),
        "kill_switch_hit": kill_switch_hit,
        "long_total_pnl_pct": round(total_long_pnl, 2),
        "short_total_pnl_pct": round(total_short_pnl, 2),
        "model_type": "ElasticNet" if is_linear else "XGBoost_challenger",
        "features_used": len(active_features),
        "feature_jaccard": round(feature_tracker.jaccard_overlap(), 3),
        "rolling_window_weeks": args.rolling_window,
        "top_n": args.top_n,
        "beta_neutral": True,
        "regime_gated": True,
        "gates": gates,
    }

    with open(f"{RESULTS_DIR}/performance_v6.json", "w") as f:
        json.dump(perf, f, indent=2, default=str)

    db.insert_performance_summary(run_id, perf)
    db.finish_run(run_id)

    _log(f"\n{'='*72}")
    _log(f"  AZALYST v6  —  RUN COMPLETE  [{run_id}]")
    _log(f"{'='*72}")
    _log(f"  total_weeks       : {n_wks}")
    _log(f"  total_trades      : {len(trades_df)}")
    _log(f"  retrains          : {retrains}")
    _log(f"  model_type        : {'ElasticNet' if is_linear else 'XGBoost'}")
    _log(f"  features          : {len(active_features)} (Jaccard={feature_tracker.jaccard_overlap():.3f})")
    _log(f"  total_return_pct  : {cum_ret*100:+.2f}%")
    _log(f"  annualised_pct    : {ann_ret:+.2f}%")
    _log(f"  sharpe            : {sharpe:.4f}")
    _log(f"  max_drawdown_pct  : {max_dd*100:.2f}%")
    _log(f"  ic_mean           : {ic_mean:.5f}")
    _log(f"  icir              : {icir_val:.4f}")
    _log(f"  {'─'*68}")
    _log(f"  LONG  total PnL   : {total_long_pnl:+.2f}%")
    _log(f"  SHORT total PnL   : {total_short_pnl:+.2f}%")
    _log(f"  {'─'*68}")
    for r, s in regime_stats.items():
        avg_ret = float(np.mean(s["rets"]))
        avg_ic = float(np.mean(s["ics"]))
        n = len(s["rets"])
        _log(f"  {r:<20s}: ret={avg_ret:+.2f}%  IC={avg_ic:+.4f}  ({n} weeks)")
    _log(f"  {'─'*68}")
    _log(f"  KILL CRITERIA     : {'ALL PASS' if gates['ALL_PASS'] else 'FAILED — review strategy'}")
    _log(f"{'='*72}")
    _log(f"\n  Trades   → {RESULTS_DIR}/all_trades_v6.csv")
    _log(f"  Summary  → {RESULTS_DIR}/weekly_summary_v6.csv")
    _log(f"  Perf     → {RESULTS_DIR}/performance_v6.json")
    _log(f"  Database → {RESULTS_DIR}/azalyst_v6.db")

    # Clear checkpoint on completion
    db.close()
    ckpt_file = _ckpt_path_v6(RESULTS_DIR)
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
        _log("  [CHECKPOINT] Cleared — run complete")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Checkpoint preserved — run again to resume.")
        sys.exit(1)
    except Exception as _e:
        import traceback
        print(f"\n  [FATAL] {type(_e).__name__}: {_e}")
        traceback.print_exc()
        print("\n  [CHECKPOINT] Preserved — run again to resume.")
        sys.exit(1)
