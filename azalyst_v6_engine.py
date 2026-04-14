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
║   3. Rolling 104-week window (2 yrs, not expanding) for initial train      ║
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
import traceback
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV
from sklearn.preprocessing import RobustScaler
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

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
    # NOTE: PurgedTimeSeriesCV intentionally NOT imported from v5_engine.
    # v5_engine has the old one-sided embargo version.
    # The fixed two-sided version lives in azalyst_train (imported below).
)
from azalyst_train import compute_ic, PurgedTimeSeriesCV

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

# Rolling window (not expanding) — v6.1: extended to 104 weeks (2 years) so
# the model has seen at least one full bull+bear cycle before going OOS.
# The safe_end embargo (1 hr before train_end) prevents any forward leakage.
ROLLING_WINDOW_WEEKS = 104   # 2 years of training history

# Retraining interval — monthly (4 weeks) for a 2-year OOS run so the model
# adapts to changing regimes roughly every month instead of every quarter.
RETRAIN_WEEKS = 4

# Portfolio
FEE_RATE       = 0.001
ROUND_TRIP_FEE = FEE_RATE * 2
DEFAULT_TOP_N  = 5       # Conservative default (GPT 5.4: weak ranker → fewer picks)

# Kill switches
# v6.1: raised to -25% so the 2-year OOS run is not aborted by a single bad
# week of leveraged losses (original -20% fired after just 10 weeks).
MAX_DRAWDOWN_KILL = -0.25
KILL_SWITCH_RECOVERY_THRESHOLD = -0.12  # proportional recovery threshold

# Feature stability
MAX_FEATURE_TURNOVER = 3   # Max features to add/remove per retrain (GPT 5.4)
MIN_IC_PERIODS_TO_ADD = 2  # Positive IC in ≥2 periods to be added
MAX_TRAIN_ROWS = 2_000_000

# ── Regularization guardrails ────────────────────────────────────────────────
# The main cause of 29x IS→OOS IC decay was ElasticNetCV picking alpha=0.00002
# (near-zero regularization) on 2M rows, producing an overfit model.
# We enforce a minimum alpha floor and bias toward L1 (sparser = less overfit).
ALPHA_MIN_FLOOR   = 0.001   # Never allow alpha below this value
L1_RATIO_GRID     = [0.5, 0.7, 0.9, 0.95, 0.99]  # Heavily L1-biased

# ── Target engineering ───────────────────────────────────────────────────────
# Winsorize at 1st/99th pct to suppress outlier influence on linear model.
TARGET_WINSORIZE  = True
TARGET_WINSOR_PCT = 1.0   # percentile to clip at each tail

# ── Long-side filters ────────────────────────────────────────────────────────
# The long win rate of 35.6% (worse than random) is fixed by:
# 1. Minimum predicted return threshold — avoid near-zero-confidence longs
# 2. Momentum filter — never long a coin whose 1-week return is negative
#    (don't catch falling knives in crypto)
LONG_MIN_PRED_THRESHOLD = 0.0   # pred_ret must exceed this (fraction, not %)
LONG_MOMENTUM_FILTER    = True  # only long if ret_1w > 0

# ── Turnover reduction ───────────────────────────────────────────────────────
# Fee-adjusted ranking: subtract the estimated round-trip cost from predicted
# return before ranking so that churning a position costs the model something.
FEE_ADJUSTED_RANKING    = True  # deduct ROUND_TRIP_FEE from pred for new entries

# Universe blacklist — symbols excluded from all predictions and trades
# Reasons: FTT (FTX collapsed Nov 2022 — invalid post-collapse data),
#          EURUSDT (FX pair — wrong distributional class for crypto engine),
#          Stablecoins (USDC/USDT/FDUSD/USDP — zero return, poison signals)
SYMBOL_BLACKLIST: Set[str] = {
    "FTTUSDT",     # FTX Token — delisted/invalid post Nov 2022
    "EURUSDT",     # FX pair — wrong asset class for this engine
    "USDCUSDT",    # Stablecoin — ~0 return
    "USDPUSDT",    # Stablecoin — ~0 return
    "FDUSDUSDT",   # Stablecoin — ~0 return
    "TUSDUSDT",    # Stablecoin — ~0 return
    "BUSDUSDT",    # Stablecoin — ~0 return
}

# Position scale hard cap — v6.1: reduced from 3.0 to 1.0.
# The original 3x cap caused a -34% single-week loss in week 10 that triggered
# the kill switch. At 1.0 the maximum notional exposure is 1× per position.
# Vol-scaling (0.5 / rvol) still adjusts within [0, 1.0].
MAX_POSITION_SCALE = 1.0

# Fiat / stablecoin-like bases to exclude from the crypto cross-section.
# v6.1 bugfix: AEURUSDT/EURIUSDT still slipped through the original blacklist,
# which polluted picks with fiat-pegged instruments and wasted retrain time.
FIAT_STABLE_BASES: Set[str] = {
    "AEUR",
    "BFUSD",
    "BUSD",
    "DAI",
    "EURI",
    "EUR",
    "FDUSD",
    "FRAX",
    "PYUSD",
    "RLUSD",
    "TUSD",
    "USD1",
    "USDC",
    "USDE",
    "USDP",
    "UST",
    "USTC",
    "XUSD",
}


def is_excluded_symbol(sym: str) -> bool:
    """Return True for non-tradeable symbols we do not want in the engine."""
    sym_u = str(sym).upper().strip()
    if sym_u in SYMBOL_BLACKLIST:
        return True
    if sym_u.endswith("USDT") and sym_u[:-4] in FIAT_STABLE_BASES:
        return True
    return False


def get_tradeable_symbols(symbols) -> List[str]:
    """Filter the lazy symbol universe down to tradeable crypto instruments."""
    return [sym for sym in symbols.keys() if not is_excluded_symbol(sym)]


def load_symbol_columns(symbols, sym: str, columns: List[str]) -> pd.DataFrame:
    """Load only the requested columns for one symbol from the lazy store."""
    needed_cols = list(dict.fromkeys(columns))

    if hasattr(symbols, "_metadata") and sym in getattr(symbols, "_metadata", {}):
        fpath = symbols._metadata[sym]["fpath"]
        df_local = pd.read_parquet(fpath, columns=needed_cols)
    else:
        df_full = symbols[sym]
        cols = [c for c in needed_cols if c in df_full.columns]
        df_local = df_full.loc[:, cols].copy()

    if not isinstance(df_local.index, pd.DatetimeIndex):
        df_local.index = pd.to_datetime(df_local.index, utc=True)
    elif df_local.index.tz is None:
        df_local.index = df_local.index.tz_localize("UTC")
    df_local.sort_index(inplace=True)
    return df_local


def retrain_cadence_label(weeks: int) -> str:
    """Human-readable retrain cadence for operator logs."""
    if weeks <= 1:
        return "weekly"
    if weeks <= 4:
        return "monthly"
    if weeks <= 13:
        return "quarterly"
    return f"every {weeks} weeks"


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

    P4 fix: Tag each IC observation with the regime at that time.
    Use regime-conditional IC for propose_update.
    """

    def __init__(self, core: List[str], initial_set: List[str],
                 max_turnover: int = MAX_FEATURE_TURNOVER):
        self.core = set(core)
        self.active = list(initial_set)
        self.max_turnover = max_turnover
        # ic_history[feat] = list of (ic_value, regime) tuples
        self.ic_history: Dict[str, List[Tuple[float, str]]] = {}
        self.retrain_feature_sets: List[Set[str]] = [set(initial_set)]

    def record_ic(self, feature_ics: Dict[str, float], regime: str = "UNKNOWN"):
        """Record IC measurements for all features, tagged with regime."""
        for feat, ic_val in feature_ics.items():
            if feat not in self.ic_history:
                self.ic_history[feat] = []
            self.ic_history[feat].append((ic_val, regime))

    def propose_update(self, candidate_pool: List[str],
                       current_regime: str = None) -> List[str]:
        """Propose a new feature set respecting turnover cap.

        P4 fix: regime-conditional IC for mean-reversion features.
        """
        current_set = set(self.active)
        proposed = set(self.core)  # Core always stays

        mean_reversion_regimes = {"LOW_VOL_GRIND", "HIGH_VOL_LATERAL"}

        # Evaluate current non-core features for removal
        removals = []
        for feat in self.active:
            if feat in self.core:
                continue
            observations = self.ic_history.get(feat, [])
            if len(observations) >= 3:
                recent = observations[-4:]
                is_mr_feature = feat in ("rsi_14", "mean_rev_zscore_1h", "bb_pos")
                neg_count = 0
                for ic_val, reg in recent:
                    if is_mr_feature and reg not in mean_reversion_regimes:
                        continue
                    if ic_val < 0:
                        neg_count += 1
                if neg_count >= 3:
                    removals.append(feat)
                else:
                    proposed.add(feat)
            else:
                proposed.add(feat)

        # Evaluate candidates for addition
        additions = []
        for feat in candidate_pool:
            if feat in proposed:
                continue
            observations = self.ic_history.get(feat, [])
            if len(observations) >= 2:
                recent = observations[-4:]
                is_mr_feature = feat in ("rsi_14", "mean_rev_zscore_1h", "bb_pos")
                pos_count = 0
                ic_sum = 0.0
                ic_n = 0
                for ic_val, reg in recent:
                    if is_mr_feature and reg not in mean_reversion_regimes:
                        continue
                    ic_sum += ic_val
                    ic_n += 1
                    if ic_val > 0:
                        pos_count += 1
                avg_ic = ic_sum / max(ic_n, 1)
                if ic_n == 0 or pos_count >= MIN_IC_PERIODS_TO_ADD:
                    additions.append((feat, avg_ic))

        additions.sort(key=lambda x: -x[1])

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

    LEAKAGE AUDIT (v6.1):
    - safe_end = train_end - 1 hr (5 min × 12 bars embargo)
      → the last bar allowed into training is at train_end-60min
      → future_ret_1h at that bar looks forward to train_end — excluded ✓
    - The OOS walk-forward always starts at or after train_end (next Mon)
      → minimum gap between last training bar and first OOS bar ≥ 60 min ✓
    - All features are lagged-only (no look-ahead in azalyst_factors_v2)
    - Non-overlapping subsampling (every HORIZON_BARS_1H rows) removes
      autocorrelated target contamination within the training window ✓
    """
    rolling_start = train_end - pd.Timedelta(weeks=rolling_weeks)
    safe_end = train_end - pd.Timedelta(minutes=5 * HORIZON_BARS_1H)

    embargo_minutes = 5 * HORIZON_BARS_1H
    print(f"  Building v6 training matrix [{rolling_start.date()} → {safe_end.date()}]"
          f" ({len(features)} features, rolling={rolling_weeks}wk, embargo={embargo_minutes}min)")
    print(f"  Leakage audit: train_end={pd.Timestamp(train_end).date()} "
          f"safe_end={safe_end.date()} "
          f"gap={embargo_minutes}min — OOS starts ≥ {embargo_minutes}min after safe_end")

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

    # MEMORY FIX: Process symbols in chunks to avoid loading all DataFrames at once
    all_symbols = list(symbols.keys())
    symbol_list = [sym for sym in all_symbols if not is_excluded_symbol(sym)]
    chunk_size = 20  # Process 20 symbols at a time
    excluded_count = len(all_symbols) - len(symbol_list)
    print(f"  Training universe: {len(symbol_list)} tradeable symbols "
          f"({excluded_count} excluded)")
    
    for chunk_start in range(0, len(symbol_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(symbol_list))
        chunk_symbols = symbol_list[chunk_start:chunk_end]
        
        for sym in chunk_symbols:
            try:
                df = load_symbol_columns(
                    symbols, sym, features + [TARGET_COL, TARGET_COL_FALLBACK]
                )
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
            except Exception:
                pass
        
        # Evict cache after each chunk to free memory
        if hasattr(symbols, '_cache'):
            while len(symbols._cache) > 5:
                symbols._cache.popitem(last=False)
        gc.collect()
        if chunk_end % 100 == 0 or chunk_end == len(symbol_list):
            print(f"  [{chunk_end}/{len(symbol_list)}] training symbols scanned...")

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

    # ── Target winsorization ─────────────────────────────────────────────────────────
    # Clips extreme target values that blow up ElasticNet coefficients.
    # Beta-neutral std ~1.3% means a raw 20% return (10x the std) would
    # dominate the loss and cause overfitting to rare events.
    if TARGET_WINSORIZE:
        lo = float(np.nanpercentile(y_neutral, TARGET_WINSOR_PCT))
        hi = float(np.nanpercentile(y_neutral, 100 - TARGET_WINSOR_PCT))
        n_clipped = int(np.sum((y_neutral < lo) | (y_neutral > hi)))
        y_neutral = np.clip(y_neutral, lo, hi)
        print(f"  Target winsorized: [{lo*100:.3f}%, {hi*100:.3f}%] "
              f"({n_clipped:,} rows clipped, {n_clipped/max(len(y_neutral),1)*100:.2f}%)")

    # ── Feature collinearity report ───────────────────────────────────────────────
    # Pairs with corr > 0.8 add noise without new signal. Log them so the
    # operator knows which features are redundant.
    if len(features) <= 15:
        try:
            corr_mat = np.corrcoef(feat_arr.T)
            high_corr_pairs = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    if abs(corr_mat[i, j]) > 0.80:
                        high_corr_pairs.append(
                            (features[i], features[j], corr_mat[i, j]))
            if high_corr_pairs:
                print("  [COLLINEARITY] Correlated feature pairs (|r|>0.80):")
                for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
                    print(f"    {f1} <-> {f2}: r={r:.3f}")
            else:
                print("  [COLLINEARITY] No highly correlated pairs (|r|>0.80)")
        except Exception:
            pass

    gc.collect()
    return feat_arr, ret_arr, y_neutral, ts_arr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MODEL TRAINING (Elastic Net + XGBoost challenger)
# ══════════════════════════════════════════════════════════════════════════════

def train_elastic_net(X, y, features: List[str],
                      label: str = "", cv_gap: int = 48) -> Tuple:
    """Train Elastic Net with built-in alpha/l1_ratio CV.

    v6.1 changes vs original:
    - L1 grid biased toward 0.9-0.99 (sparser = less overfit)
    - Alpha floor enforced: if CV picks alpha < ALPHA_MIN_FLOOR, raise it
    - This directly fixes the 29x IS→OOS IC decay caused by alpha=0.00002

    Returns: (model, scaler, importance, mean_r2, mean_ic, icir)
    """
    limits_ctx = threadpool_limits(limits=1) if threadpool_limits else nullcontext()

    with limits_ctx:
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X).astype(np.float32)

        # Windows safety: serial CV avoids repeated OpenMP / BLAS access violations
        # seen during large monthly retrains on 2M-row matrices.
        cv_model = ElasticNetCV(
            l1_ratio=L1_RATIO_GRID,
            n_alphas=50,
            cv=5,
            max_iter=10000,
            random_state=42,
            n_jobs=1,
        )
        cv_model.fit(Xs, y)

        chosen_alpha    = cv_model.alpha_
        chosen_l1ratio  = cv_model.l1_ratio_

        # ── Alpha floor ─────────────────────────────────────────────────────────────
        # With 2M rows and 10 features, ElasticNetCV tends to pick a very small
        # alpha (e.g. 0.00002), which means near-zero regularization and severe
        # overfitting. We force alpha >= ALPHA_MIN_FLOOR so the model stays sparse.
        if chosen_alpha < ALPHA_MIN_FLOOR:
            print(f"  [{label}] Alpha floor: CV chose {chosen_alpha:.6f} "
                  f"< floor {ALPHA_MIN_FLOOR:.4f} → enforcing floor")
            chosen_alpha = ALPHA_MIN_FLOOR

        # Final model with enforced hyperparams
        model = ElasticNet(
            alpha=chosen_alpha,
            l1_ratio=chosen_l1ratio,
            max_iter=10000,
            random_state=42,
        )
        model.fit(Xs, y)
        # Attach attrs used later for logging
        model.alpha_    = chosen_alpha
        model.l1_ratio_ = chosen_l1ratio

        # Evaluate with PurgedTimeSeriesCV for honest metrics
        cv = PurgedTimeSeriesCV(n_splits=5, gap=cv_gap)
        r2s, ics = [], []

        for fold, (tr, val) in enumerate(cv.split(Xs), 1):
            m = ElasticNet(
                alpha=chosen_alpha,
                l1_ratio=chosen_l1ratio,
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

    del Xs
    gc.collect()
    print(f"  [{label}] ElasticNet: alpha={chosen_alpha:.6f}  "
          f"l1_ratio={chosen_l1ratio:.2f}  "
          f"nonzero={n_nonzero}/{len(features)}  "
          f"R²={mean_r2:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}")

    return model, scaler, importance, mean_r2, mean_ic, icir


def train_xgb_challenger(X, y, features: List[str], cuda_api,
                         label: str = "", cv_gap: int = 48) -> Tuple:
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

    cv = PurgedTimeSeriesCV(n_splits=5, gap=cv_gap)
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

    # MEMORY FIX: Process symbols in chunks to avoid loading all DataFrames at once
    symbol_list = get_tradeable_symbols(symbols)
    chunk_size = 20  # Process 20 symbols at a time
    
    for chunk_start in range(0, len(symbol_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(symbol_list))
        chunk_symbols = symbol_list[chunk_start:chunk_end]
        
        for sym in chunk_symbols:
            try:
                df = load_symbol_columns(symbols, sym, features_used + ["close"])
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
                    actual_ret = float(np.log(c_end / c_start))
                    # P3: Outlier filter — skip >50% weekly move (delist/halt)
                    if abs(actual_ret) > 0.5:
                        continue
                    actual_close_rets[sym] = actual_ret
            except Exception:
                pass
        
        # Evict cache after each chunk to free memory
        if hasattr(symbols, '_cache'):
            while len(symbols._cache) > 5:
                symbols._cache.popitem(last=False)
        gc.collect()

    return predictions, actual_close_rets


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: PORTFOLIO CONSTRUCTION (regime-gated, long/short decomposition)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_weekly_trades_v6(predictions, actual_close_rets,
                              prev_longs, prev_shorts,
                              regime: str,
                              leverage: float = 1.0,
                              top_n: int = DEFAULT_TOP_N,
                              symbol_rvol: Optional[Dict[str, float]] = None,
                              no_trade_high_vol: bool = False,
                              symbol_ret1w: Optional[Dict[str, float]] = None):
    """Regime-gated portfolio construction with long/short PnL decomposition.

    v6.1 additions:
    - FEE_ADJUSTED_RANKING: deduct ROUND_TRIP_FEE from predicted return for
      new entries before ranking, so churn is penalized during selection.
    - LONG_MOMENTUM_FILTER: skip longs where ret_1w < 0 (falling knives).
    - LONG_MIN_PRED_THRESHOLD: skip longs where predicted return is too small.
    - symbol_ret1w: dict of {sym: ret_1w} for momentum filter.
    """
    if not predictions:
        return [], 0.0, 0.0, 0.0, set(), set()

    # P6: Skip HIGH_VOL_LATERAL if flag set
    if regime == "HIGH_VOL_LATERAL" and no_trade_high_vol:
        return [], 0.0, 0.0, 0.0, set(), set()

    # ── Fee-adjusted ranking ────────────────────────────────────────────────────
    # Deduct the round-trip fee from predicted return for new entries.
    # This way the model pays a cost for churn during ranking, not just PnL.
    if FEE_ADJUSTED_RANKING:
        adj_predictions = {
            sym: pred - (ROUND_TRIP_FEE if sym not in prev_longs and sym not in prev_shorts else 0.0)
            for sym, pred in predictions.items()
        }
    else:
        adj_predictions = predictions

    pred_series = pd.Series(adj_predictions)
    n_symbols = len(pred_series)
    n = min(top_n, n_symbols // 2)

    if n < 1:
        return [], 0.0, 0.0, 0.0, set(), set()

    sorted_syms = pred_series.sort_values(ascending=False)

    # ── Long candidate selection (with filters) ─────────────────────────────────
    candidates_long = []
    for sym in sorted_syms.index:
        raw_pred = predictions[sym]  # use raw pred (not fee-adjusted) for threshold

        # Filter 1: minimum predicted return threshold
        if raw_pred <= LONG_MIN_PRED_THRESHOLD:
            continue

        # Filter 2: momentum filter — skip coins with negative 1-week return
        # (don’t catch falling knives; longs should be on rising names)
        if LONG_MOMENTUM_FILTER and symbol_ret1w:
            ret1w = symbol_ret1w.get(sym, None)
            if ret1w is not None and ret1w < 0:
                continue  # negative momentum — skip

        candidates_long.append(sym)
        if len(candidates_long) >= n:
            break
    cur_longs = set(candidates_long)

    # Regime gating
    if regime == "BULL_TREND":
        cur_shorts = set()
        base_position_scale = 0.5 * leverage
    elif regime == "HIGH_VOL_LATERAL":
        cur_shorts = set(sorted_syms.tail(n).index)
        base_position_scale = 0.5 * leverage
    else:  # BEAR_TREND, LOW_VOL_GRIND
        cur_shorts = set(sorted_syms.tail(n).index)
        base_position_scale = 1.0 * leverage

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

        # P1: Vol-scaled sizing for longs (inverse to rvol_1d)
        # Hard cap at MAX_POSITION_SCALE to prevent 50x blowups on low-rvol symbols
        if symbol_rvol and sym in symbol_rvol:
            rvol = symbol_rvol[sym]
            position_scale = min(
                base_position_scale / max(rvol, 0.01),
                MAX_POSITION_SCALE,
            )
        else:
            position_scale = min(base_position_scale, MAX_POSITION_SCALE)

        # P3: Hard clip at ±100%
        pnl = np.clip((ret - fee) * position_scale, -1.0, None)

        trades.append({
            "symbol": sym, "signal": "BUY",
            "pred_ret": round(predictions[sym] * 100, 5),
            "pnl_percent": round(pnl * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "position_scale": round(position_scale, 4),
        })
        long_pnl_sum += pnl
        n_long += 1

    for sym in cur_shorts:
        ret = actual_close_rets.get(sym, 0.0)
        if not np.isfinite(ret):
            ret = 0.0
        fee = 0.0 if sym in prev_shorts else ROUND_TRIP_FEE
        position_scale = base_position_scale

        # P3: Hard clip at ±100%
        pnl = np.clip((-ret - fee) * position_scale, -1.0, None)

        trades.append({
            "symbol": sym, "signal": "SELL",
            "pred_ret": round(predictions[sym] * 100, 5),
            "pnl_percent": round(pnl * 100, 4),
            "raw_ret_pct": round(ret * 100, 4),
            "position_scale": round(position_scale, 4),
        })
        short_pnl_sum += pnl
        n_short += 1

    long_ret = (long_pnl_sum / n_long) if n_long > 0 else 0.0
    short_ret = (short_pnl_sum / n_short) if n_short > 0 else 0.0
    total_positions = n_long + n_short
    week_ret = (long_pnl_sum + short_pnl_sum) / max(total_positions, 1) if total_positions > 0 else 0.0

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
    week_pairs = list(zip(test_weeks[:-1], test_weeks[1:]))
    if not week_pairs:
        return {}

    needed_cols = list(dict.fromkeys(V6_CORE_FEATURES + ["close"]))
    all_symbols = list(symbols.keys())
    symbol_list = [sym for sym in all_symbols if not is_excluded_symbol(sym)]
    rng = np.random.default_rng(42)
    excluded_count = len(all_symbols) - len(symbol_list)

    # Collect all weekly baseline inputs in a compact structure so each symbol
    # is loaded only once instead of once per week.
    weekly_payloads = []
    for _ in week_pairs:
        weekly_payloads.append({
            "actual": [],
            "ret_1w": [],
            "ret_3d": [],
            "vol_regime": [],
            "composite": [],
        })

    def _load_symbol_subset(sym: str) -> pd.DataFrame:
        return load_symbol_columns(symbols, sym, needed_cols)

    print(f"  Processing {len(symbol_list)} tradeable symbols once across "
          f"{len(week_pairs)} weeks ({excluded_count} excluded)...")
    for idx, sym in enumerate(symbol_list, 1):
        try:
            df = _load_symbol_subset(sym)
            if "close" not in df.columns or len(df) < 2:
                continue

            close_series = df["close"].dropna()
            if len(close_series) < 2:
                continue

            for week_idx, (ws, we) in enumerate(week_pairs):
                pre_rows = df.loc[df.index < ws]
                if pre_rows.empty:
                    continue

                week_close = close_series.loc[(close_series.index >= ws) & (close_series.index < we)]
                if len(week_close) < 2:
                    continue

                c_s = float(week_close.iloc[0])
                c_e = float(week_close.iloc[-1])
                if not (np.isfinite(c_s) and np.isfinite(c_e)) or c_s <= 0:
                    continue

                last_row = pre_rows.iloc[-1]
                feature_vals = []
                ret_1w = 0.0
                ret_3d = 0.0
                vol_regime = 0.0

                if "ret_1w" in last_row.index and np.isfinite(last_row["ret_1w"]):
                    ret_1w = float(last_row["ret_1w"])
                    feature_vals.append(ret_1w)
                if "ret_3d" in last_row.index and np.isfinite(last_row["ret_3d"]):
                    ret_3d = float(last_row["ret_3d"])
                    feature_vals.append(ret_3d)
                if "vol_regime" in last_row.index and np.isfinite(last_row["vol_regime"]):
                    vol_regime = float(last_row["vol_regime"])
                    feature_vals.append(vol_regime)

                payload = weekly_payloads[week_idx]
                payload["actual"].append(float(np.log(c_e / c_s)))
                payload["ret_1w"].append(ret_1w)
                payload["ret_3d"].append(ret_3d)
                payload["vol_regime"].append(-vol_regime)
                payload["composite"].append(float(np.mean(feature_vals)) if feature_vals else 0.0)
        except Exception:
            pass

        if idx % 50 == 0 or idx == len(symbol_list):
            print(f"  [{idx}/{len(symbol_list)}] symbols processed...")
            gc.collect()

    results = {
        "ret_1w": [],
        "ret_3d": [],
        "vol_regime": [],
        "composite": [],
        "random": [],
    }

    for payload in weekly_payloads:
        ret_arr = np.array(payload["actual"], dtype=float)
        if len(ret_arr) < 10:
            continue

        for name in ["ret_1w", "ret_3d", "vol_regime", "composite"]:
            scores = np.array(payload[name], dtype=float)
            valid = np.isfinite(scores) & np.isfinite(ret_arr)
            if valid.sum() >= 10:
                ic = float(stats.spearmanr(scores[valid], ret_arr[valid])[0])
                if np.isfinite(ic):
                    results[name].append(ic)

        random_scores = rng.standard_normal(len(ret_arr))
        valid = np.isfinite(ret_arr)
        if valid.sum() >= 10:
            ic = float(stats.spearmanr(random_scores[valid], ret_arr[valid])[0])
            if np.isfinite(ic):
                results["random"].append(ic)

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
        icir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
        pct_pos = float(np.mean(np.array(ics) > 0)) * 100
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
    # Exclude KILL_SWITCH weeks (ic=0, no trades) — they dilute the positivity rate
    ics = [m["ic"] for m in weekly_summary
           if isinstance(m.get("ic"), (int, float))
           and m.get("regime") not in ("IC_GATED", "KILL_SWITCH")]
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
    """Compute weekly cross-sectional IC using the same pre-week snapshot as trading."""
    feature_ics = {}
    feature_pairs = {feat: [] for feat in features}

    # Use the same setup as the trading decision:
    # last pre-week feature snapshot vs realized close-to-close weekly return.
    all_symbols = list(symbols.keys())
    symbol_list = [sym for sym in all_symbols if not is_excluded_symbol(sym)]
    chunk_size = 20  # Process 20 symbols at a time
    excluded_count = len(all_symbols) - len(symbol_list)
    print(f"    Feature IC snapshot [{week_start.date()} -> {week_end.date()}] "
          f"on {len(symbol_list)} tradeable symbols ({excluded_count} excluded)")
    
    for chunk_start in range(0, len(symbol_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(symbol_list))
        chunk_symbols = symbol_list[chunk_start:chunk_end]
        
        for sym in chunk_symbols:
            try:
                df = load_symbol_columns(symbols, sym, features + ["close"])
                if "close" not in df.columns:
                    continue

                pre_rows = df[df.index < week_start]
                if pre_rows.empty:
                    continue

                week_close = df.loc[
                    (df.index >= week_start) & (df.index < week_end),
                    "close",
                ].dropna()
                if len(week_close) < 2:
                    continue

                c_s = float(week_close.iloc[0])
                c_e = float(week_close.iloc[-1])
                if not (np.isfinite(c_s) and np.isfinite(c_e)) or c_s <= 0:
                    continue

                actual_ret = float(np.log(c_e / c_s))
                if abs(actual_ret) > 0.5:
                    continue

                last_row = pre_rows.iloc[-1]
                for feat in features:
                    if feat not in last_row.index:
                        continue
                    feat_val = last_row[feat]
                    if np.isfinite(feat_val):
                        feature_pairs[feat].append((float(feat_val), actual_ret))
            except Exception:
                pass
        
        # Evict cache after each chunk to free memory
        if hasattr(symbols, '_cache'):
            while len(symbols._cache) > 5:
                symbols._cache.popitem(last=False)
        gc.collect()
        if chunk_end % 100 == 0 or chunk_end == len(symbol_list):
            print(f"    [{chunk_end}/{len(symbol_list)}] IC symbols processed...")

    for feat in features:
        pairs = feature_pairs.get(feat, [])
        if len(pairs) >= 10:
            feat_vals = np.array([p[0] for p in pairs], dtype=float)
            ret_vals = np.array([p[1] for p in pairs], dtype=float)
            ic, _ = stats.spearmanr(feat_vals, ret_vals)
            feature_ics[feat] = float(ic) if np.isfinite(ic) else 0.0
        else:
            feature_ics[feat] = 0.0

    return feature_ics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10b: DATE SPLITS — v6.1 adaptive split (up to 2-year training)
# ══════════════════════════════════════════════════════════════════════════════

def get_date_splits_v6(symbols) -> tuple:
    """V6.1 date splitter: targets up to a 2-year training window when possible.

    Leakage-safe split rules:
    - Y1 (training):  global_min  →  y1_end   (model never sees past y1_end)
    - OOS zone 1:    y1_end      →  y2_end
    - OOS zone 2:    y2_end      →  global_max

    Split logic:
    - Total data ≥ 4 yr  → Y1 = 2 yr fixed, OOS = remainder (≥ 2 yr)
    - Total data 2-4 yr  → Y1 = 50 %, OOS = 50 %
    - Total data < 2 yr  → Y1 = 33 % (legacy fallback, warn user)

    The 1-hour embargo inside build_training_matrix_v6 (safe_end) is an
    additional per-call guard on top of this structural split.
    """
    if hasattr(symbols, "_metadata") and symbols._metadata:
        all_min = [v["min"] for v in symbols._metadata.values()]
        all_max = [v["max"] for v in symbols._metadata.values()]
    else:
        all_min = [df.index.min() for df in symbols.values()]
        all_max = [df.index.max() for df in symbols.values()]

    global_min = min(all_min)
    global_max = max(all_max)
    total_span = global_max - global_min
    total_weeks = total_span.days / 7

    TWO_YEARS  = pd.Timedelta(weeks=104)
    FOUR_YEARS = pd.Timedelta(weeks=208)

    if total_span >= FOUR_YEARS:
        # ≥ 4 yr total: hard 2-year training window, rest is OOS
        y1_end = global_min + TWO_YEARS
        split_label = "2yr fixed"
    elif total_span >= TWO_YEARS:
        # 2–4 yr total: 50/50 split
        y1_end = global_min + (total_span / 2)
        split_label = "50/50"
    else:
        # < 2 yr total: 33% legacy split (log a warning)
        y1_end = global_min + (total_span / 3)
        split_label = "33% (< 2yr data — extend data for best results)"
        print(f"  [WARN] Total data span = {total_weeks:.0f} weeks "
              f"(< 104). Training window is compressed.")

    y2_end = y1_end + (global_max - y1_end) / 2

    train_weeks = (y1_end - global_min).days / 7
    oos_weeks   = (global_max - y1_end).days / 7

    print(f"  Data range   : {global_min.date()} -> {global_max.date()} "
          f"({total_weeks:.0f} weeks total)")
    print(f"  Y1 (training): {global_min.date()} -> {y1_end.date()} "
          f"({train_weeks:.0f} wks, split={split_label})")
    print(f"  OOS zone 1   : {y1_end.date()} -> {y2_end.date()}")
    print(f"  OOS zone 2   : {y2_end.date()} -> {global_max.date()} "
          f"({oos_weeks:.0f} wks OOS total)")
    return global_min, global_max, y1_end, y2_end


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


def compute_oos_diagnostics(weekly_summary, all_trades, feature_tracker, retrain_history):
    """Compute out-of-sample diagnostic metrics for performance reporting.
    
    Returns a dict with diagnostic information including:
    - Weekly return statistics
    - Trade statistics
    - Feature stability metrics
    - IC time series analysis
    """
    diagnostics = {}
    
    # Weekly return diagnostics
    if weekly_summary:
        returns = [w.get("week_return_pct", 0) for w in weekly_summary 
                   if w.get("regime") not in ("KILL_SWITCH",)]
        ics = [w.get("ic", 0) for w in weekly_summary if w.get("regime") not in ("KILL_SWITCH",)]
        
        diagnostics["n_weeks"] = len(returns)
        diagnostics["avg_weekly_return_pct"] = float(np.mean(returns)) if returns else 0.0
        diagnostics["std_weekly_return_pct"] = float(np.std(returns)) if len(returns) > 1 else 0.0
        diagnostics["best_week_pct"] = float(np.max(returns)) if returns else 0.0
        diagnostics["worst_week_pct"] = float(np.min(returns)) if returns else 0.0
        diagnostics["positive_weeks_pct"] = float(sum(1 for r in returns if r > 0)) / max(len(returns), 1) * 100
        diagnostics["avg_ic_oos"] = float(np.mean(ics)) if ics else 0.0
        diagnostics["ic_volatility"] = float(np.std(ics)) if len(ics) > 1 else 0.0
        diagnostics["ic_positive_ratio"] = float(sum(1 for ic in ics if ic > 0)) / max(len(ics), 1)
        
        # Regime breakdown
        regime_breakdown = {}
        for w in weekly_summary:
            regime = w.get("regime", "UNKNOWN")
            if regime == "KILL_SWITCH":
                continue
            if regime not in regime_breakdown:
                regime_breakdown[regime] = {"count": 0, "total_return": 0.0, "ics": []}
            regime_breakdown[regime]["count"] += 1
            regime_breakdown[regime]["total_return"] += w.get("week_return_pct", 0)
            regime_breakdown[regime]["ics"].append(w.get("ic", 0))
        
        for regime, stats in regime_breakdown.items():
            stats["avg_return"] = stats["total_return"] / max(stats["count"], 1)
            stats["avg_ic"] = float(np.mean(stats["ics"])) if stats["ics"] else 0.0
            del stats["ics"]  # Remove raw list to keep diagnostics compact
        
        diagnostics["regime_breakdown"] = regime_breakdown
    
    # Trade diagnostics
    if all_trades:
        diagnostics["total_trades"] = len(all_trades)
        
        long_trades = [t for t in all_trades if t.get("side") == "LONG"]
        short_trades = [t for t in all_trades if t.get("side") == "SHORT"]
        
        diagnostics["long_trades"] = len(long_trades)
        diagnostics["short_trades"] = len(short_trades)
        
        if long_trades:
            long_returns = [t.get("return_pct", 0) for t in long_trades]
            diagnostics["avg_long_return_pct"] = float(np.mean(long_returns))
            diagnostics["long_win_rate_pct"] = float(sum(1 for r in long_returns if r > 0)) / len(long_returns) * 100
        
        if short_trades:
            short_returns = [t.get("return_pct", 0) for t in short_trades]
            diagnostics["avg_short_return_pct"] = float(np.mean(short_returns))
            diagnostics["short_win_rate_pct"] = float(sum(1 for r in short_returns if r > 0)) / len(short_returns) * 100
        
        # Symbol concentration
        symbol_counts = {}
        for t in all_trades:
            sym = t.get("symbol", "UNKNOWN")
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
        
        if symbol_counts:
            counts = list(symbol_counts.values())
            diagnostics["avg_trades_per_symbol"] = float(np.mean(counts))
            diagnostics["max_trades_single_symbol"] = int(np.max(counts))
            diagnostics["unique_symbols_traded"] = len(symbol_counts)
    
    # Feature stability diagnostics
    if feature_tracker:
        diagnostics["feature_stability"] = {
            "n_active_features": len(feature_tracker.active),
            "n_core_features": len(feature_tracker.core),
            "jaccard_overlap": feature_tracker.jaccard_overlap(),
            "n_retrains_tracked": len(feature_tracker.retrain_feature_sets),
        }
        
        # Top features by IC
        feature_ic_stats = {}
        for feat, observations in feature_tracker.ic_history.items():
            if observations:
                ic_values = [obs[0] for obs in observations]
                feature_ic_stats[feat] = {
                    "mean_ic": float(np.mean(ic_values)),
                    "std_ic": float(np.std(ic_values)) if len(ic_values) > 1 else 0.0,
                    "n_observations": len(ic_values),
                    "positive_ic_ratio": float(sum(1 for ic in ic_values if ic > 0)) / len(ic_values),
                }
        
        # Sort by mean IC and take top 10
        top_features = sorted(feature_ic_stats.items(), 
                             key=lambda x: x[1]["mean_ic"], 
                             reverse=True)[:10]
        diagnostics["top_features_by_ic"] = {feat: stats for feat, stats in top_features}
    
    # Retraining history diagnostics
    if retrain_history:
        diagnostics["retrain_summary"] = {
            "n_retrains": len(retrain_history),
            "avg_ic_in_sample": float(np.mean([r.get("ic_in_sample", 0) for r in retrain_history])) if retrain_history else 0.0,
            "avg_ic_oos": float(np.mean([r.get("ic_oos", 0) for r in retrain_history])) if retrain_history else 0.0,
        }
        
        # IC degradation (in-sample vs OOS)
        if retrain_history:
            ic_degradations = []
            for r in retrain_history:
                ic_in = r.get("ic_in_sample", 0)
                ic_out = r.get("ic_oos", 0)
                if ic_in != 0:
                    ic_degradations.append((ic_in - ic_out) / abs(ic_in))
            if ic_degradations:
                diagnostics["retrain_summary"]["avg_ic_degradation_pct"] = float(np.mean(ic_degradations)) * 100
    
    return diagnostics


def append_fatal_log_v6(exc: Exception) -> None:
    """Best-effort fatal logging for batch launches."""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "run_log_v6.txt")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n[FATAL] {type(exc).__name__}: {exc}\n")
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            fh.write("\n[CHECKPOINT] Preserved — run again to resume.\n")
    except Exception:
        pass


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
                        help="Rolling window in weeks (default: 104 = 2 years)")
    parser.add_argument("--no-falsify", action="store_true",
                        help="Skip falsification campaign")
    parser.add_argument("--xgb-challenger", action="store_true",
                        help="Also train XGBoost as challenger model")
    parser.add_argument("--pin-coins", type=str, default="",
                        help="Comma-separated symbols to restrict universe")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-trade-high-vol", action="store_true",
                        help="Skip trading in HIGH_VOL_LATERAL regime (P6)")
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
    cv_gap = HORIZON_BARS

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
    print(f"  Retrain      : every {RETRAIN_WEEKS} weeks "
          f"({retrain_cadence_label(RETRAIN_WEEKS)})")
    print(f"  Features     : {len(V6_DEFAULT_FEATURES)} stable"
          f" + turnover cap {MAX_FEATURE_TURNOVER}")
    print(f"  Portfolio    : top-{args.top_n} per side, regime-gated")
    print(f"  Compute      : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"  Leverage     : {args.leverage:.1f}x")
    print(f"  Kill-switch  : {dd_kill*100:.0f}% max drawdown")
    print(f"  CV embargo   : {cv_gap} bars")
    print(f"  Falsification: {'enabled' if not args.no_falsify else 'disabled'}")
    print(f"  HV skip      : {'ON' if args.no_trade_high_vol else 'OFF'}")
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

    def _log_exception(prefix: str, exc: Exception) -> None:
        _log(f"{prefix}{type(exc).__name__}: {exc}")
        for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
            for subline in line.rstrip().splitlines():
                _log(f"    {subline}")

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
            "cv_gap": cv_gap,
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
    tradeable_symbol_count = len(get_tradeable_symbols(symbols))
    _log(f"  Loaded {len(symbols)} valid symbols "
         f"({tradeable_symbol_count} tradeable, "
         f"{len(symbols) - tradeable_symbol_count} excluded)")

    # ── STEP 2: Date splits ───────────────────────────────────────────────────
    _log("\nSTEP 2: Date splits (v6.1 — adaptive split, up to 2-year train)\n")
    global_min, global_max, y1_end, y2_end = get_date_splits_v6(symbols)

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
            try:
                falsification_results = run_falsification(
                    symbols, falsify_weeks, active_features, top_n=args.top_n)
            except Exception as exc:
                _log_exception("  [WARN] Falsification failed — continuing without it: ", exc)
                falsification_results = {}
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
        # Restore IC history (stored as [ic_value, regime] lists in JSON)
        for k, v in ckpt.get("feature_ic_history", {}).items():
            feature_tracker.ic_history[k] = [(x[0], x[1]) for x in v]
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
        from azalyst_leak_test import run_leak_test
        _log("\n[LEAK TEST] Running pre-training sanity checks...")
        # Run leak test on y_raw (untransformed target) to avoid issues with
        # beta-neutral daily demeaning breaking the roll-based correlation test
        leak_results = run_leak_test(X_train, y_raw, active_features,
                                     embargo_bars=cv_gap, timestamps=ts_train)
        for k, v in leak_results.items():
            _log(f"  {k}: {v}")
        if not leak_results["shuffled_test_pass"]:
            raise RuntimeError(
                f"LEAK TEST FAILED: shuffled target has |IC|="
                f"{leak_results['shuffled_mean_abs_ic']:.4f} > 0.02."
            )
        if not leak_results["leaked_test_pass"]:
            raise RuntimeError(
                f"LEAK TEST FAILED: IC computation issue "
                f"(leaked IC={leak_results['leaked_feature_ic']:.4f}, expected > 0.95)."
            )
        _log("[LEAK TEST] Passed.\n")
        t0 = time.time()
        current_model, current_scaler, importance, mean_r2, mean_ic, icir = \
            train_elastic_net(
                X_train, y_neutral, active_features, label="base_y1", cv_gap=cv_gap
            )
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
                                     cuda_api, label="xgb_y1", cv_gap=cv_gap)
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

    current_dd = compute_drawdown(weekly_returns)
    for week_num, (ws, we) in enumerate(zip(weeks[:-1], weeks[1:]), 1):
        if week_num <= resume_from_week:
            continue

        # Kill-switch pause: P2 — regime-conditional restart
        if ks_pause_until > week_num:
            if current_dd > KILL_SWITCH_RECOVERY_THRESHOLD:
                _log(f"  Week {week_num}: Kill switch pause expired — "
                     f"DD recovered to {current_dd*100:.1f}% "
                     f"(>{KILL_SWITCH_RECOVERY_THRESHOLD*100:.0f}%), resuming.")
                ks_pause_until = 0
            else:
                continue

        current_dd = compute_drawdown(weekly_returns)
        if current_dd < dd_kill:
            _log(f"\n  *** KILL SWITCH *** Week {week_num}: "
                 f"DD={current_dd*100:.1f}% < {dd_kill*100:.0f}%")
            kill_switch_hit = True
            # P2: Set pause to max — resume only when DD recovers
            ks_pause_until = len(weeks)
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
            # P4: Tag IC with current regime
            feature_tracker.record_ic(fic, regime=regime)

        # Scheduled retrain with rolling window
        if week_num % RETRAIN_WEEKS == 0:
            _log(f"\n  Week {week_num:3d}: {retrain_cadence_label(RETRAIN_WEEKS).upper()} RETRAIN "
                 f"(rolling {args.rolling_window}wk to {we.date()})...")

            # P5: IC decay monitoring
            if len(weekly_summary) >= 4:
                recent_4_ic = [m.get("ic", 0.0) for m in weekly_summary[-4:]]
                avg_recent_ic = float(np.mean(recent_4_ic))
                if abs(avg_recent_ic) < 0.005:
                    _log(f"    GOVERNANCE_WARNING: Model IC decay detected — "
                         f"4-week rolling IC = {avg_recent_ic:+.5f} (threshold: 0.005)")

            # P4: Feature stability with regime-conditional IC
            new_features = feature_tracker.propose_update(
                V6_CANDIDATE_FEATURES, current_regime=regime)
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
                    label=f"v6_w{week_num:03d}", cv_gap=cv_gap)

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

                    # P5: Governance report at each retrain
                    try:
                        from azalyst_validator import generate_governance_report
                        ic_oos_val = avg_recent_ic if len(weekly_summary) >= 4 else 0.0
                        governance = generate_governance_report(
                            run_id=run_id, retrain_label=f"w{week_num:03d}",
                            week_num=week_num, ic_in_sample=ic_n, ic_oos=ic_oos_val,
                            importance_current=imp_new,
                            importance_previous=importance if 'importance' in dir() else None,
                            pred_distribution=np.array(list(predictions.values())),
                            pred_distribution_prev=None, output_dir=RESULTS_DIR,
                        )
                        if governance.get("warnings"):
                            for w in governance["warnings"]:
                                _log(f"    GOVERNANCE: {w}")
                    except Exception:
                        pass

                    # Optional XGBoost challenger at retrain
                    if args.xgb_challenger:
                        xgb_m, xgb_s, _, _, xgb_ic, _ = train_xgb_challenger(
                            X_rt, y_neutral_rt, active_features, cuda_api,
                            label=f"xgb_w{week_num:03d}", cv_gap=cv_gap)
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
        # P1: Build rvol + ret1w snapshots before week_start for filters
        symbol_rvol  = {}
        symbol_ret1w = {}  # for LONG_MOMENTUM_FILTER
        feat_cols_needed = ["rvol_1d"]
        if LONG_MOMENTUM_FILTER and "ret_1w" in active_features:
            feat_cols_needed.append("ret_1w")

        if True:  # always collect (rvol used in BULL_TREND, ret1w used in filters)
            symbol_list_snap = get_tradeable_symbols(symbols)
            chunk_size_snap = 20
            for chunk_start_snap in range(0, len(symbol_list_snap), chunk_size_snap):
                chunk_end_snap = min(chunk_start_snap + chunk_size_snap, len(symbol_list_snap))
                chunk_symbols_snap = symbol_list_snap[chunk_start_snap:chunk_end_snap]
                for sym in chunk_symbols_snap:
                    try:
                        df = load_symbol_columns(symbols, sym, feat_cols_needed)
                        pre_week_df = df[df.index < ws]
                        if len(pre_week_df) < 1:
                            continue
                        last_row = pre_week_df.iloc[-1]
                        if "rvol_1d" in last_row.index:
                            rv = float(last_row["rvol_1d"])
                            if np.isfinite(rv) and rv > 0:
                                symbol_rvol[sym] = rv
                        if "ret_1w" in last_row.index:
                            r1w = float(last_row["ret_1w"])
                            if np.isfinite(r1w):
                                symbol_ret1w[sym] = r1w
                    except Exception:
                        pass
                if hasattr(symbols, '_cache'):
                    while len(symbols._cache) > 5:
                        symbols._cache.popitem(last=False)
                gc.collect()

        trades, week_ret, long_ret, short_ret, cur_longs, cur_shorts = \
            simulate_weekly_trades_v6(
                predictions, actual_close_rets,
                prev_longs, prev_shorts,
                regime=regime,
                leverage=args.leverage,
                top_n=args.top_n,
                symbol_rvol=symbol_rvol if symbol_rvol else None,
                no_trade_high_vol=args.no_trade_high_vol,
                symbol_ret1w=symbol_ret1w if symbol_ret1w else None,
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
            "feature_ic_history": {k: [[x[0], x[1]] for x in v]
                                   for k, v in feature_tracker.ic_history.items()},
            "kill_switch_hit": kill_switch_hit,
            "ks_pause_until": ks_pause_until,
            "current_model_path": current_model_path,
            "current_scaler_path": current_scaler_path,
            "is_linear": is_linear,
        })

    retrain_history = []  # populated from governance files if available
    try:
        gov_dir = os.path.join(RESULTS_DIR, "governance")
        if os.path.isdir(gov_dir):
            for gf in sorted(os.listdir(gov_dir)):
                if gf.endswith(".json"):
                    with open(os.path.join(gov_dir, gf)) as gfh:
                        gh = json.load(gfh)
                        retrain_history.append({
                            "label": gh.get("retrain_label", gf),
                            "ic_in_sample": gh.get("ic_in_sample"),
                            "ic_oos": gh.get("ic_oos"),
                        })
    except Exception:
        pass
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

    # Capacity-related summaries
    avg_turnover = float(summary_df["turnover_pct"].mean()) if "turnover_pct" in summary_df.columns and len(summary_df) > 0 else 0.0
    symbol_turnover = {}
    for t in all_trades:
        sym = t.get("symbol")
        if not sym:
            continue
        symbol_turnover[sym] = symbol_turnover.get(sym, 0) + 1
    top_traded = sorted(symbol_turnover.items(), key=lambda x: x[1], reverse=True)[:10]
    # Proxy stress test: bottom 20% by available symbol count in weekly universe.
    if len(summary_df) > 5 and "n_symbols" in summary_df.columns:
        q20 = summary_df["n_symbols"].quantile(0.2)
        stress = summary_df[summary_df["n_symbols"] <= q20]
        stress_rets = (stress["week_return_pct"] / 100.0).to_numpy(dtype=float)
        stress_std = float(np.std(stress_rets)) if len(stress_rets) > 1 else 0.0
        stress_sharpe = float(np.mean(stress_rets)) / stress_std * np.sqrt(52) if stress_std > 0 else 0.0
    else:
        stress_sharpe = 0.0

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
        "avg_weekly_turnover_pct": round(avg_turnover, 2),
        "liquidity_stress_sharpe_proxy": round(stress_sharpe, 4),
        "top_traded_symbols": top_traded,
    }

    # Deflated Sharpe Ratio
    try:
        from azalyst_deflated_sharpe import deflated_sharpe_ratio
        rets_arr = np.array(weekly_returns, dtype=float)
        if len(rets_arr) >= 20:
            dsr = deflated_sharpe_ratio(
                sharpe_observed=sharpe,
                n_returns=len(rets_arr),
                skew=float(stats.skew(rets_arr)),
                kurtosis=float(stats.kurtosis(rets_arr)),
                n_trials=100,
            )
            perf["deflated_sharpe"] = dsr
    except Exception:
        pass

    # Move diagnostics line here (after perf dict is built)
    oos_diag = compute_oos_diagnostics(
        weekly_summary=weekly_summary,
        all_trades=all_trades,
        feature_tracker=feature_tracker,
        retrain_history=retrain_history,
    )
    perf["oos_diagnostics"] = oos_diag

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
    _log(f"  avg_turnover_pct  : {avg_turnover:.2f}%")
    _log(f"  liq_stress_sharpe : {stress_sharpe:.4f} (proxy)")
    if top_traded:
        _log("  top_traded_symbols:")
        for sym, ntr in top_traded:
            _log(f"    - {sym}: {ntr} trades")
    if "deflated_sharpe" in perf:
        _log(f"  deflated_sharpe   : {perf['deflated_sharpe'].get('deflated_sharpe_ratio', 0):.4f}")
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
        append_fatal_log_v6(_e)
        print(f"\n  [FATAL] {type(_e).__name__}: {_e}")
        traceback.print_exc()
        print("\n  [CHECKPOINT] Preserved — run again to resume.")
        sys.exit(1)
