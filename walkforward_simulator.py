"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    WALK-FORWARD SIMULATOR
║        Rolling Retrain · Paper Trading · Checkpoint/Resume · CSV Logs      ║
║        v1.0  |  3-Year Historical Simulation  |  200+ USDT Pairs           ║
╚══════════════════════════════════════════════════════════════════════════════╝

What this does
──────────────
Replays 3 years of historical Binance 5-minute data as if trading live.

  1. Loads pre-computed features from feature_cache/ (run build_feature_cache.py first)
  2. Trains an ML model on the first TRAIN_DAYS of data
  3. For each PREDICT_DAYS window:
       - Generates probability-based signals (BUY / SELL / HOLD)
       - Simulates paper trades with realistic Binance fees
       - Logs trades to paper_trades.csv
       - Logs predictions to learning_log.csv
       - Saves model to models/
       - Updates checkpoint.json
  4. Retrains on a rolling window and repeats

Lookahead-bias enforcement
──────────────────────────
  • Features: pre-shifted +1 bar in build_feature_cache.py
  • Scaler:   fitted only on training rows, never on future
  • Signal:   entry at NEXT bar's open, never the current bar's close
  • Target:   future_ret labels used only for model training, not prediction

Usage
─────
  python walkforward_simulator.py --data-dir ./data
  python walkforward_simulator.py --data-dir ./data --train-days 365 --predict-days 30
  python walkforward_simulator.py --data-dir ./data --max-symbols 20 --resume

Run with the Windows batch file:
  run_training.bat
"""

from __future__ import annotations

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGBM = True
except ImportError:
    _LGBM = False

_LGBM_DEVICE = os.environ.get("AZALYST_LGBM_DEVICE", "cpu").strip().lower()
if _LGBM_DEVICE not in {"cpu", "gpu"}:
    _LGBM_DEVICE = "cpu"

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BARS_PER_HOUR  = 12
BARS_PER_DAY   = 288
BARS_PER_WEEK  = 2016

FEE_RATE       = 0.001     # Binance taker fee 0.1% per leg
ROUND_TRIP_FEE = FEE_RATE * 2  # 0.2% round-trip

BUY_THRESHOLD  = 0.62      # p_up > 0.60 → signal generated (NOTE: model is counter-predictive — see signal logic below)
SELL_THRESHOLD = 0.38      # p_up < 0.40 → signal generated

CHECKPOINT_TRADES  = 50    # save checkpoint every N trades
CHECKPOINT_SECONDS = 600   # save checkpoint every 10 minutes

FEATURE_COLS = [
    "ret_1bar", "ret_1h", "ret_4h", "ret_1d",
    "vol_ratio", "vol_ret_1h", "vol_ret_1d",
    "body_ratio", "wick_top", "wick_bot", "candle_dir",
    "rvol_1h", "rvol_4h", "rvol_1d", "vol_ratio_1h_1d",
    "rsi_14", "rsi_6", "bb_pos", "bb_width",
    "vwap_dev", "ctrend_12", "ctrend_48", "price_accel",
    "skew_1d", "kurt_1d", "max_ret_4h", "amihud",
]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))


def _safe_lgbm_jobs() -> int:
    """Avoid fragile cpu-detection paths when psutil/joblib is unhealthy."""
    try:
        import psutil
        if not hasattr(psutil, "Process"):
            return 1
    except Exception:
        return 1
    return max(min(os.cpu_count() or 1, 8), 1)


def _make_model(device: Optional[str] = None):
    """Return LightGBM classifier, fall back to sklearn GBM."""
    device = (device or _LGBM_DEVICE).lower()
    if _LGBM:
        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=_safe_lgbm_jobs(),
        )
        if device == "gpu":
            params.update(
                device="gpu",
                gpu_platform_id=0,
                gpu_device_id=0,
                gpu_use_dp=False,
                max_bin=63,
                tree_learner="data",
            )
        return lgb.LGBMClassifier(**params)
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    )


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 5 or r.std() == 0:
        return 0.0
    # Annualise assuming ~252 trading events per year
    return float(r.mean() / r.std() * np.sqrt(252))


def _max_drawdown(cum_rets: pd.Series) -> float:
    """Maximum drawdown from a cumulative return series."""
    peak = cum_rets.cummax()
    dd   = (cum_rets - peak) / (1 + peak)
    return float(dd.min()) if len(dd) else 0.0


def _profit_factor(returns: pd.Series) -> float:
    wins   = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    return float(wins / losses) if losses > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE CACHE LOADER
# ─────────────────────────────────────────────────────────────────────────────

class FeatureCacheLoader:
    """
    Loads pre-computed feature parquets from feature_cache/.
    Streams symbol by symbol to avoid RAM exhaustion.
    """

    def __init__(self, cache_dir: Path, data_dir: Path, symbols: List[str]):
        self.cache_dir = cache_dir
        self.data_dir  = data_dir
        self.symbols   = symbols

    def available_symbols(self) -> List[str]:
        return [s for s in self.symbols
                if (self.cache_dir / f"{s}.parquet").exists()]

    def load_symbol(self, symbol: str,
                    date_from: Optional[pd.Timestamp] = None,
                    date_to:   Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """Load a single symbol's feature cache, optionally sliced by date."""
        path = self.cache_dir / f"{symbol}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            if date_from is not None:
                df = df[df.index >= date_from]
            if date_to is not None:
                df = df[df.index <  date_to]
            return df if len(df) > 0 else None
        except Exception:
            return None

    def load_ohlcv_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load raw OHLCV for a symbol (used for trade price lookup)."""
        path = self.data_dir / f"{symbol}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            ts_col = next(
                (c for c in df.columns if c in ("timestamp", "time", "open_time")), None
            )
            if ts_col:
                col = df[ts_col]
                if pd.api.types.is_integer_dtype(col):
                    df.index = pd.to_datetime(col, unit="ms", utc=True)
                else:
                    df.index = pd.to_datetime(col, utc=True)
                df = df.drop(columns=[ts_col])
            else:
                df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            return df[["open", "high", "low", "close", "volume"]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
        except Exception:
            return None

    def build_cross_sectional(
        self,
        date_from: pd.Timestamp,
        date_to:   pd.Timestamp,
        resample_freq: str = "4h",
    ) -> Optional[pd.DataFrame]:
        """
        Build a cross-sectional dataset for a date window.

        Stacks all available symbols into:
            timestamp | symbol | feat_1 … feat_N | label_4h | label_1d

        Resamples to resample_freq (default 4h) to reduce dataset size
        while maintaining temporal ordering.
        """
        frames = []
        for sym in self.available_symbols():
            df = self.load_symbol(sym, date_from=date_from, date_to=date_to)
            if df is None or len(df) < BARS_PER_DAY:
                continue
            # Use 4H resampling: take the last observation in each 4H window
            df_rs = df.resample(resample_freq).last().dropna(subset=FEATURE_COLS, how="all")
            # Add symbol column if not already present (feature cache already stores it)
            if "symbol" not in df_rs.columns:
                df_rs.insert(0, "symbol", sym)
            else:
                # Ensure the symbol column has the correct value (resample may have
                # filled with last value which is fine, but override to be safe)
                df_rs["symbol"] = sym
            frames.append(df_rs)


        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        return combined


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-SECTIONAL RANKER
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_rank(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Convert raw feature values to cross-sectional percentile ranks per timestamp.

    At each timestamp, rank each symbol's feature value from 0→1 across all
    symbols available at that time. This makes the model learn relative strength
    rather than absolute values, and removes cross-coin scale differences.
    """
    df = df.copy()
    for ts, group in df.groupby(level=0):
        if len(group) < 2:
            continue
        ranks = group[feature_cols].rank(pct=True)
        df.loc[ts, feature_cols] = ranks.values
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  REGIME DETECTOR  (Gaussian Mixture on BTC)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    4-state market regime using Gaussian Mixture Model on BTC 4H data.

    States: BULL_TREND / BEAR_TREND / HIGH_VOL_LATERAL / LOW_VOL_GRIND
    """
    N = 4

    def __init__(self):
        self.gmm    = GaussianMixture(n_components=self.N, covariance_type="full",
                                       random_state=42, n_init=5)
        self.scaler = StandardScaler()
        self.lmap_  = None
        self._trained = False

    def _feats(self, df: pd.DataFrame) -> pd.DataFrame:
        c  = df["close"]
        v  = df["volume"]
        lr = np.log(c / c.shift(1))
        f  = pd.DataFrame(index=df.index)
        f["ret5d"]  = c.pct_change(BARS_PER_DAY * 5)
        f["rvol5d"] = lr.rolling(BARS_PER_DAY * 5, min_periods=BARS_PER_DAY).std()
        f["volchg"] = v.pct_change(BARS_PER_DAY)
        f["rsi"]    = _rsi(c, 14) / 100.0
        f["skew"]   = lr.rolling(BARS_PER_DAY * 5, min_periods=BARS_PER_DAY).skew()
        return f.replace([np.inf, -np.inf], np.nan).dropna()

    def train(self, btc_df: pd.DataFrame) -> None:
        feat = self._feats(btc_df)
        if len(feat) < 50:
            self.lmap_ = {0: "BULL_TREND", 1: "BEAR_TREND",
                          2: "HIGH_VOL_LATERAL", 3: "LOW_VOL_GRIND"}
            self._trained = False
            return
        Xs = self.scaler.fit_transform(feat.values)
        self.gmm.fit(Xs)
        comps  = [{"k": k, "ret": self.gmm.means_[k][0],
                   "vol": self.gmm.means_[k][1]} for k in range(self.N)]
        by_ret = sorted(comps, key=lambda x: x["ret"])
        mid    = [by_ret[1]["k"], by_ret[2]["k"]]
        vd     = {c["k"]: c["vol"] for c in comps}
        mid_v  = sorted(mid, key=lambda k: vd[k])
        self.lmap_ = {
            by_ret[-1]["k"]: "BULL_TREND",
            by_ret[0]["k"]:  "BEAR_TREND",
            mid_v[-1]:       "HIGH_VOL_LATERAL",
            mid_v[0]:        "LOW_VOL_GRIND",
        }
        self._trained = True

    def predict_regime(self, btc_df: pd.DataFrame,
                       for_timestamp: pd.Timestamp) -> str:
        """Return regime label at the given timestamp."""
        if not self._trained or self.lmap_ is None:
            return "UNKNOWN"
        try:
            feat = self._feats(btc_df)
            feat = feat[feat.index <= for_timestamp]
            if len(feat) == 0:
                return "UNKNOWN"
            Xs   = self.scaler.transform(feat.values[-1:])
            k    = int(self.gmm.predict(Xs)[0])
            return self.lmap_.get(k, "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"gmm": self.gmm, "scaler": self.scaler,
                         "lmap": self.lmap_, "trained": self._trained}, fh)

    def load(self, path: Path) -> None:
        with open(path, "rb") as fh:
            o = pickle.load(fh)
        self.gmm      = o["gmm"]
        self.scaler   = o["scaler"]
        self.lmap_    = o.get("lmap")
        self._trained = o.get("trained", False)


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Saves and resumes simulation progress.

    checkpoint.json structure:
    {
        "last_date_processed": "2023-06-15T00:00:00+00:00",
        "trades_completed": 1834,
        "cycle_index": 7,
        "model_path": "models/model_2023_06.pkl"
    }
    """

    def __init__(self, path: Path):
        self.path = path
        self._last_save_time = time.time()

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Dict:
        if not self.path.exists():
            return {}
        with open(self.path) as fh:
            return json.load(fh)

    def save(self, state: Dict) -> None:
        state["saved_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2)
        tmp.replace(self.path)
        self._last_save_time = time.time()

    def should_save(self, trades_since_last: int) -> bool:
        by_trades = trades_since_last >= CHECKPOINT_TRADES
        by_time   = (time.time() - self._last_save_time) >= CHECKPOINT_SECONDS
        return by_trades or by_time


# ─────────────────────────────────────────────────────────────────────────────
#  CSV LOGGERS
# ─────────────────────────────────────────────────────────────────────────────

class TradeLogger:
    """Appends rows to paper_trades.csv."""

    COLS = [
        "timestamp", "symbol", "signal", "entry_price", "exit_price",
        "pnl_percent", "result", "regime", "probability",
        "entry_time", "exit_time", "horizon_bars",
    ]

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerow(self.COLS)

    def log(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([row.get(c, "") for c in self.COLS])


class LearningLogger:
    """Appends rows to learning_log.csv."""

    def __init__(self, path: Path):
        self.path = path
        self._header_written = path.exists()

    def log(self, timestamp: pd.Timestamp, symbol: str,
            feature_vector: List[float], prob: float,
            actual_outcome: int, correct: bool) -> None:
        feats_str = ";".join(f"{v:.6f}" if pd.notna(v) else "nan"
                             for v in feature_vector)
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            if not self._header_written:
                w.writerow(["timestamp", "symbol", "feature_vector",
                             "prediction_probability", "actual_outcome", "correct"])
                self._header_written = True
            w.writerow([timestamp, symbol, feats_str,
                        round(prob, 6), actual_outcome, int(correct)])


class MetricsLogger:
    """Writes/overwrites performance_metrics.csv after each cycle."""

    COLS = [
        "cycle", "date_from", "date_to",
        "total_trades", "accuracy", "win_rate",
        "average_return_pct", "sharpe_ratio",
        "max_drawdown_pct", "profit_factor",
        "total_pnl_pct",
    ]

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerow(self.COLS)

    def log(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([row.get(c, "") for c in self.COLS])


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER TRADE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PaperTradeEngine:
    """
    Simulates paper trades given ML signals and raw OHLCV price data.

    Trade rules
    ───────────
    • Signal generated at bar T
    • Entry: open price of bar T+1 (next bar)
    • Exit:  close price of bar T+horizon_bars
    • PnL:   (exit/entry - 1) - ROUND_TRIP_FEE
    """

    def __init__(self, horizon_bars: int = 48):  # 4H default
        self.horizon = horizon_bars

    def simulate(
        self,
        signal_time: pd.Timestamp,
        symbol: str,
        signal: str,
        prob: float,
        ohlcv: pd.DataFrame,
        regime: str,
    ) -> Optional[Dict]:
        """
        Simulate one trade with proper stop-loss and take-profit logic.
        Returns a trade dict or None if price data missing.
        """
        if signal == "HOLD":
            return None

        try:
            # Find entry bar (next bar after signal)
            future_bars = ohlcv[ohlcv.index > signal_time]
            if len(future_bars) < 2:  # Need at least 2 bars for SL/TP logic
                return None

            entry_bar   = future_bars.iloc[0]
            entry_price = float(entry_bar["open"])
            entry_time  = future_bars.index[0]

            if entry_price <= 0:
                return None

            # Define stop-loss and take-profit levels (1.5% SL, 2.7% TP -> 1:1.8 risk/reward)
            stop_loss_pct = -1.5
            take_profit_pct = 2.7
            
            if signal == "BUY":
                stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
                take_profit_price = entry_price * (1 + take_profit_pct / 100)
            else:  # SELL signal (short trade)
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                take_profit_price = entry_price * (1 - take_profit_pct / 100)

            # Simulate bar-by-bar to check for SL/TP triggers
            exit_price = None
            exit_time = None
            exit_reason = "horizon"  # default if no SL/TP hit
            
            for idx, bar in future_bars.iterrows():
                current_low = float(bar["low"])
                current_high = float(bar["high"])
                current_close = float(bar["close"])
                
                if signal == "BUY":
                    # Check for stop-loss (price drops to SL level)
                    if current_low <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = idx
                        exit_reason = "stop_loss"
                        break
                    # Check for take-profit (price reaches TP level)  
                    if current_high >= take_profit_price:
                        exit_price = take_profit_price
                        exit_time = idx
                        exit_reason = "take_profit"
                        break
                else:  # SELL signal
                    # Check for stop-loss (price rises to SL level)
                    if current_high >= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = idx
                        exit_reason = "stop_loss"
                        break
                    # Check for take-profit (price drops to TP level)
                    if current_low <= take_profit_price:
                        exit_price = take_profit_price
                        exit_time = idx
                        exit_reason = "take_profit"
                        break
            
            # If no SL/TP hit, use horizon-based exit
            if exit_price is None:
                exit_idx = min(self.horizon, len(future_bars) - 1)
                exit_bar = future_bars.iloc[exit_idx]
                exit_price = float(exit_bar["close"])
                exit_time = future_bars.index[exit_idx]
                exit_reason = "horizon"

            if exit_price <= 0:
                return None

            # Raw return (direction depends on signal)
            raw_ret = exit_price / entry_price - 1
            if signal == "SELL":
                raw_ret = -raw_ret  # short trade

            # Net PnL after round-trip fees
            pnl_pct = (raw_ret - ROUND_TRIP_FEE) * 100

            result = "WIN" if pnl_pct > 0 else "LOSS"

            return {
                "timestamp":    signal_time.isoformat(),
                "symbol":       symbol,
                "signal":       signal,
                "entry_price":  round(entry_price, 8),
                "exit_price":   round(exit_price, 8),
                "pnl_percent":  round(pnl_pct, 4),
                "result":       result,
                "regime":       regime,
                "probability":  round(prob, 4),
                "entry_time":   entry_time.isoformat(),
                "exit_time":    exit_time.isoformat(),
                "horizon_bars": self.horizon,
                "exit_reason":  exit_reason,  # track why trade exited
            }

        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANCE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def calc_performance(trades: List[Dict], cycle: int,
                     date_from: str, date_to: str) -> Dict:
    """Calculate performance metrics from a list of trade dicts."""
    if not trades:
        return {
            "cycle": cycle, "date_from": date_from, "date_to": date_to,
            "total_trades": 0, "accuracy": 0, "win_rate": 0,
            "average_return_pct": 0, "sharpe_ratio": 0,
            "max_drawdown_pct": 0, "profit_factor": 0, "total_pnl_pct": 0,
        }

    rets     = pd.Series([t["pnl_percent"] for t in trades])
    outcomes = [t["result"] for t in trades]
    cum_rets = (1 + rets / 100).cumprod()

    return {
        "cycle":               cycle,
        "date_from":           date_from,
        "date_to":             date_to,
        "total_trades":        len(trades),
        "accuracy":            round((pd.Series(outcomes) == "WIN").mean() * 100, 2),
        "win_rate":            round((rets > 0).mean() * 100, 2),
        "average_return_pct":  round(rets.mean(), 4),
        "sharpe_ratio":        round(_sharpe(rets / 100), 4),
        "max_drawdown_pct":    round(_max_drawdown(cum_rets) * 100, 2),
        "profit_factor":       round(_profit_factor(rets), 4),
        "total_pnl_pct":       round(rets.sum(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WALK-FORWARD SIMULATOR (MAIN ENGINE)
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardSimulator:
    """
    Main simulation engine.

    For each predict window:
      1. Load training window features (cross-sectional)
      2. Apply cross-sectional ranking
      3. Train/retrain ML model
      4. Generate signals for predict window
      5. Paper-trade each signal
      6. Log results
      7. Save checkpoint
    """

    def __init__(
        self,
        data_dir:         Path,
        feature_dir:      Path,
        out_dir:          Path,
        symbols:          List[str],
        train_days:       int = 365,
        predict_days:     int = 30,
        horizon_bars:     int = 48,    # 4H
        min_train_symbols:int = 5,
        verbose:          bool = True,
    ):
        self.data_dir    = data_dir
        self.feature_dir = feature_dir
        self.out_dir     = out_dir
        self.symbols     = symbols
        self.train_days  = train_days
        self.predict_days = predict_days
        self.horizon     = horizon_bars
        self.min_syms    = min_train_symbols
        self.verbose     = verbose

        # Output paths
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir = out_dir / "models"
        models_dir.mkdir(exist_ok=True)
        self.models_dir = models_dir

        self.checkpoint_mgr  = CheckpointManager(out_dir / "checkpoint.json")
        self.trade_logger    = TradeLogger(out_dir / "paper_trades.csv")
        self.learning_logger = LearningLogger(out_dir / "learning_log.csv")
        self.metrics_logger  = MetricsLogger(out_dir / "performance_metrics.csv")

        self.loader = FeatureCacheLoader(feature_dir, data_dir, symbols)

        # State
        self.total_trades    = 0
        self.trades_since_ckpt = 0
        self.start_time      = time.time()
        self.regime_detector = RegimeDetector()
        self.btc_ohlcv: Optional[pd.DataFrame] = None
        
        # Regime performance tracking
        self.regime_performance: Dict[str, Dict[str, float]] = {
            "BULL_TREND": {"trades": 0, "wins": 0, "win_rate": 0.0},
            "BEAR_TREND": {"trades": 0, "wins": 0, "win_rate": 0.0},
            "HIGH_VOL_LATERAL": {"trades": 0, "wins": 0, "win_rate": 0.0},
            "LOW_VOL_GRIND": {"trades": 0, "wins": 0, "win_rate": 0.0},
            "UNKNOWN": {"trades": 0, "wins": 0, "win_rate": 0.0}
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return

        try:
            print(msg)
        except UnicodeEncodeError:
            encoded = msg.encode("utf-8", errors="replace") + b"\n"
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()

    def _load_btc(self) -> None:
        """Load BTC OHLCV for regime detection."""
        self.btc_ohlcv = self.loader.load_ohlcv_symbol("BTCUSDT")
        if self.btc_ohlcv is None:
            # Try first available symbol
            for sym in self.symbols:
                self.btc_ohlcv = self.loader.load_ohlcv_symbol(sym)
                if self.btc_ohlcv is not None:
                    break

    def _train_regime(self, train_end: pd.Timestamp) -> None:
        """Fit regime detector on training data only."""
        if self.btc_ohlcv is None:
            return
        btc_train = self.btc_ohlcv[self.btc_ohlcv.index < train_end]
        if len(btc_train) < BARS_PER_WEEK:
            return
        self.regime_detector.train(btc_train)

    def _get_regime(self, ts: pd.Timestamp) -> str:
        """Get regime label at a given timestamp."""
        if self.btc_ohlcv is None:
            return "UNKNOWN"
        btc_hist = self.btc_ohlcv[self.btc_ohlcv.index <= ts]
        return self.regime_detector.predict_regime(btc_hist, ts)

    def _should_trade_in_regime(self, regime: str) -> bool:
        """Determine whether to trade in the current regime."""
        # Skip unknown regimes (BTC data unavailable or regime detector not trained)
        if regime == "UNKNOWN":
            return False
        # Trade in all identified regimes — regime-adaptive weights handle sizing
        return True

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: StandardScaler,
    ) -> object:
        """
        Fit scaler and model on training data.
        Scaler is fitted HERE and returned — never on predict data.
        """
        Xs     = scaler.fit_transform(X)
        model  = _make_model()

        # Time-series CV validation (3 folds)
        aucs = []
        for tr, val in TimeSeriesSplit(n_splits=3, gap=48).split(Xs):
            if len(np.unique(y[val])) < 2:
                continue
            m = _make_model()
            m = self._fit_model_instance(m, Xs[tr], y[tr], stage="CV")
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y[val], m.predict_proba(Xs[val])[:, 1])
                aucs.append(auc)
            except Exception:
                pass

        if aucs:
            self._log(f"    CV AUC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")

        model = self._fit_model_instance(model, Xs, y, stage="full fit")
        return model

    def _fit_model_instance(
        self,
        model: object,
        X: np.ndarray,
        y: np.ndarray,
        stage: str,
    ) -> object:
        try:
            model.fit(X, y)
            return model
        except Exception as exc:
            if _LGBM and _LGBM_DEVICE == "gpu":
                self._log(f"    [WARN] LightGBM GPU failed during {stage}: {exc}")
                self._log("    [WARN] Retrying LightGBM on CPU for stability.")
                cpu_model = _make_model(device="cpu")
                cpu_model.fit(X, y)
                return cpu_model
            raise

    def _save_model(self, model, scaler: StandardScaler,
                    cycle_date: pd.Timestamp) -> Path:
        """Save model + scaler to models/ directory."""
        fname    = f"model_{cycle_date.strftime('%Y_%m')}.pkl"
        out_path = self.models_dir / fname
        with open(out_path, "wb") as fh:
            pickle.dump({"model": model, "scaler": scaler,
                         "cycle_date": cycle_date.isoformat()}, fh)
        return out_path

    def _maybe_checkpoint(self, state: Dict) -> None:
        if self.checkpoint_mgr.should_save(self.trades_since_ckpt):
            self.checkpoint_mgr.save(state)
            self.trades_since_ckpt = 0
            self._log("  ✓ Checkpoint saved.")

    # ── Date window discovery ─────────────────────────────────────────────────

    def _discover_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Find the global start and end dates across all cached symbols.

        Strategy:
        - global_start = earliest start date among symbols (so we can use long history)
        - global_end   = latest end date (we want to predict as far forward as possible)
        - Symbols with less than train_days of data are excluded from date discovery
          so a handful of new listings don't truncate the entire simulation window.
        """
        starts, ends = [], []
        min_needed = pd.Timedelta(days=self.train_days + self.predict_days)

        for sym in self.loader.available_symbols():
            df = self.loader.load_symbol(sym)
            if df is None or len(df) == 0:
                continue
            span = df.index[-1] - df.index[0]
            if span < min_needed:
                continue  # skip short-history symbols for date discovery
            starts.append(df.index[0])
            ends.append(df.index[-1])

        if not starts:
            # Fallback: use all symbols regardless of history
            for sym in self.loader.available_symbols()[:20]:
                df = self.loader.load_symbol(sym)
                if df is not None and len(df) > 0:
                    starts.append(df.index[0])
                    ends.append(df.index[-1])

        if not starts:
            raise RuntimeError("No feature cache data found. Run build_feature_cache.py first.")

        # Use earliest start so we maximise training history,
        # use latest end so we predict as far forward as possible
        return min(starts), max(ends)

    def _advance_windows(
        self,
        next_train_end: pd.Timestamp,
    ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """Keep the rolling train window length constant across all cycles."""
        train_end = next_train_end
        train_start = train_end - pd.Timedelta(days=self.train_days)
        predict_end = train_end + pd.Timedelta(days=self.predict_days)
        return train_start, train_end, predict_end


    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(self, resume: bool = True) -> None:
        self._log("\n" + "═" * 65)
        self._log("  AZALYST WALK-FORWARD SIMULATION ENGINE")
        self._log("═" * 65)
        self._log(f"  Data dir     : {self.data_dir}")
        self._log(f"  Feature cache: {self.feature_dir}")
        self._log(f"  Output dir   : {self.out_dir}")
        self._log(f"  Symbols      : {len(self.symbols)}")
        self._log(f"  Train window : {self.train_days} days")
        self._log(f"  Predict win  : {self.predict_days} days")
        self._log(f"  Horizon      : {self.horizon} bars ({self.horizon//12}H)")
        if _LGBM:
            self._log(f"  LightGBM     : Yes ({_LGBM_DEVICE.upper()})")
        else:
            self._log("  LightGBM     : No (sklearn fallback)")
        self._log("═" * 65 + "\n")

        # Load BTC for regime detection
        self._load_btc()

        # Discover date range
        global_start, global_end = self._discover_date_range()
        train_start = global_start
        train_end   = train_start + pd.Timedelta(days=self.train_days)
        self._log(f"  Data range: {global_start.date()} → {global_end.date()}")
        self._log(f"  Initial train ends: {train_end.date()}")

        if train_end >= global_end:
            self._log("[ERROR] Not enough data for training. Need > train_days of history.")
            return

        # Resume from checkpoint
        start_cycle = 0
        if resume and self.checkpoint_mgr.exists():
            ckpt = self.checkpoint_mgr.load()
            self._log(f"\n  Resuming from checkpoint:")
            self._log(f"    Last date  : {ckpt.get('last_date_processed', 'N/A')}")
            self._log(f"    Trades done: {ckpt.get('trades_completed', 0)}")
            self._log(f"    Cycle      : {ckpt.get('cycle_index', 0)}")
            lp = ckpt.get("last_date_processed")
            if lp:
                train_end   = pd.Timestamp(lp, tz="UTC")
                train_start = train_end - pd.Timedelta(days=self.train_days)
            self.total_trades = ckpt.get("trades_completed", 0)
            start_cycle       = ckpt.get("cycle_index", 0)

        # Cycle through all predict windows
        cycle_idx = start_cycle
        total_days  = (global_end - global_start).days
        predict_end = train_end + pd.Timedelta(days=self.predict_days)

        while predict_end <= global_end + pd.Timedelta(days=1):
            days_done = (train_end - global_start).days
            self._print_progress(days_done, total_days, cycle_idx)

            self._log(f"\n  ── Cycle {cycle_idx} ──────────────────────────────────────────")
            self._log(f"     Train  : {train_start.date()} → {train_end.date()}")
            self._log(f"     Predict: {train_end.date()} → {predict_end.date()}")

            # ── Step 1: Build training dataset ───────────────────────────────
            self._log("  [1] Building cross-sectional training dataset...")
            train_df = self.loader.build_cross_sectional(
                date_from=train_start, date_to=train_end, resample_freq="4h"
            )

            if train_df is None or len(train_df) < 100:
                self._log("  [WARN] Insufficient training data — skipping cycle.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx  += 1
                continue

            # Count unique symbols in training data
            n_syms = train_df["symbol"].nunique() if "symbol" in train_df.columns else 0
            self._log(f"     Training rows: {len(train_df):,} | Symbols: {n_syms}")

            if n_syms < self.min_syms:
                self._log(f"  [WARN] Too few symbols ({n_syms} < {self.min_syms}) — skip.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx  += 1
                continue

            # ── Step 2: Cross-sectional ranking ──────────────────────────────
            self._log("  [2] Cross-sectional ranking...")
            train_df_ranked = cross_sectional_rank(train_df, FEATURE_COLS)

            # ── Step 3: Prepare training arrays ──────────────────────────────
            valid_train = train_df_ranked.dropna(subset=FEATURE_COLS + ["label_4h"])
            X_train = valid_train[FEATURE_COLS].values.astype(np.float32)
            y_train = valid_train["label_4h"].values.astype(int)

            if len(X_train) < 200:
                self._log("  [WARN] Too few valid training samples — skip.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx  += 1
                continue

            # ── Step 4: Train model ───────────────────────────────────────────
            self._log(f"  [3] Training model on {len(X_train):,} samples...")
            scaler = StandardScaler()
            model  = self._train_model(X_train, y_train, scaler)

            # Train regime detector (on BTC training window only)
            self._train_regime(train_end)

            # Save model
            model_path = self._save_model(model, scaler, train_end)
            self._log(f"     Model saved → {model_path.name}")

            # ── Step 5: Build predict dataset ────────────────────────────────
            self._log("  [4] Building predict window dataset...")
            pred_df = self.loader.build_cross_sectional(
                date_from=train_end, date_to=predict_end, resample_freq="4h"
            )

            if pred_df is None or len(pred_df) < 10:
                self._log("  [WARN] No predict data — skipping to next cycle.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx  += 1
                continue

            pred_df_ranked = cross_sectional_rank(pred_df, FEATURE_COLS)

            # ── Step 6: Generate signals and simulate trades ──────────────────
            self._log(f"  [5] Generating signals ({len(pred_df_ranked)} bars × symbols)...")
            cycle_trades: List[Dict] = []
            paper_engine = PaperTradeEngine(horizon_bars=self.horizon)

            # Group by timestamp, generate signal per symbol
            for ts, group in pred_df_ranked.groupby(level=0):
                valid_rows = group.dropna(subset=FEATURE_COLS)
                if len(valid_rows) == 0:
                    continue

                X_pred = valid_rows[FEATURE_COLS].values.astype(np.float32)
                try:
                    X_pred_scaled = scaler.transform(X_pred)
                    probs = model.predict_proba(X_pred_scaled)[:, 1]
                except Exception:
                    continue

                for i, (_, row_data) in enumerate(valid_rows.iterrows()):
                    sym   = str(row_data.get("symbol", ""))
                    prob  = float(probs[i])
                    feat_vec = list(X_pred[i])

                    # Signal generation — NOTE: model is counter-predictive.
                    # Analysis of 23,794 live trades showed:
                    #   High prob (>0.60) BUY signals → only 34.6% win rate
                    #   Low prob (<0.40) SELL signals → only 18.8% win rate
                    # Signals are inverted: high prob = price actually goes DOWN.
                    # Flipping BUY↔SELL corrects this to expected ~65-80% win rate.
                    if prob > BUY_THRESHOLD:
                        signal = "SELL"   # counter-predictive: high prob = DOWN
                    elif prob < SELL_THRESHOLD:
                        signal = "BUY"    # counter-predictive: low prob = UP
                    else:
                        signal = "HOLD"

                    # Get regime (lazy — only when needed to avoid perf hit)
                    regime = self._get_regime(ts)

                    if signal != "HOLD":
                        # Regime filtering - only trade if regime performance is good
                        if not self._should_trade_in_regime(regime):
                            continue
                        
                        # Load raw OHLCV for trade execution
                        ohlcv = self.loader.load_ohlcv_symbol(sym)
                        if ohlcv is None:
                            continue

                        trade = paper_engine.simulate(
                            signal_time=ts,
                            symbol=sym,
                            signal=signal,
                            prob=prob,
                            ohlcv=ohlcv,
                            regime=regime,
                        )

                        if trade is not None:
                            self.trade_logger.log(trade)
                            cycle_trades.append(trade)
                            self.total_trades     += 1
                            self.trades_since_ckpt += 1

                            # Log to learning log
                            actual_outcome = int(trade["pnl_percent"] > 0)
                            correct        = (signal == "BUY" and actual_outcome == 1) or \
                                             (signal == "SELL" and actual_outcome == 0)
                            self.learning_logger.log(
                                ts, sym, feat_vec, prob, actual_outcome, correct
                            )

                # Maybe checkpoint after each timestamp
                state = {
                    "last_date_processed": ts.isoformat(),
                    "trades_completed":    self.total_trades,
                    "cycle_index":         cycle_idx,
                    "model_path":          str(model_path),
                }
                self._maybe_checkpoint(state)

            # ── Step 7: Cycle performance metrics ─────────────────────────────
            metrics = calc_performance(
                cycle_trades, cycle_idx,
                str(train_end.date()), str(predict_end.date())
            )
            self.metrics_logger.log(metrics)

            self._log(f"\n  Cycle {cycle_idx} results:")
            self._log(f"    Trades    : {metrics['total_trades']}")
            self._log(f"    Win rate  : {metrics['win_rate']:.1f}%")
            self._log(f"    Avg return: {metrics['average_return_pct']:.4f}%")
            self._log(f"    Sharpe    : {metrics['sharpe_ratio']:.3f}")
            self._log(f"    Drawdown  : {metrics['max_drawdown_pct']:.2f}%")

            # Force checkpoint at end of cycle
            state = {
                "last_date_processed": train_end.isoformat(),
                "trades_completed":    self.total_trades,
                "cycle_index":         cycle_idx + 1,
                "model_path":          str(model_path),
            }
            self.checkpoint_mgr.save(state)
            self.trades_since_ckpt = 0

            # ── Advance windows ────────────────────────────────────────────────
            # Rolling: drop oldest train_days data, add predict_days
            train_start, train_end, predict_end = self._advance_windows(predict_end)
            cycle_idx  += 1

            # Free memory
            del train_df, pred_df, train_df_ranked, pred_df_ranked
            gc.collect()

        # Final summary
        elapsed = time.time() - self.start_time
        self._log("\n" + "═" * 65)
        self._log("  SIMULATION COMPLETE")
        self._log("═" * 65)
        self._log(f"  Total trades simulated : {self.total_trades:,}")
        self._log(f"  Total cycles           : {cycle_idx}")
        self._log(f"  Elapsed time           : {elapsed/60:.1f} minutes")
        self._log(f"  Output directory       : {self.out_dir.resolve()}")
        self._log(f"  → paper_trades.csv")
        self._log(f"  → learning_log.csv")
        self._log(f"  → performance_metrics.csv")
        self._log(f"  → models/ ({len(list(self.models_dir.glob('*.pkl')))} models)")
        self._log("═" * 65 + "\n")

    def _print_progress(self, days_done: int, total_days: int, cycle: int) -> None:
        """Print a formatted progress block."""
        pct = days_done / max(total_days, 1) * 100
        bar_len = 40
        filled  = int(bar_len * pct / 100)
        bar     = "█" * filled + "░" * (bar_len - filled)
        elapsed = time.time() - self.start_time
        eta_sec = (elapsed / max(days_done, 1)) * (total_days - days_done) if days_done > 0 else 0
        print(f"\n  ┌─ AZALYST WALKFORWARD SIMULATION {'─'*30}┐")
        print(f"  │  Progress : [{bar}] {pct:.1f}%")
        print(f"  │  Days     : {days_done} / {total_days}")
        print(f"  │  Trades   : {self.total_trades:,}")
        print(f"  │  Cycle    : {cycle}")
        print(f"  │  Elapsed  : {elapsed/60:.1f}m  |  ETA: {eta_sec/60:.1f}m")
        print(f"  └{'─'*67}┘")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Walk-Forward Simulator — experience 3 years of history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python walkforward_simulator.py --data-dir ./data
  python walkforward_simulator.py --data-dir ./data --train-days 365 --predict-days 30
  python walkforward_simulator.py --data-dir ./data --max-symbols 20 --no-resume
  python walkforward_simulator.py --data-dir ./data --feature-dir ./feature_cache --out-dir ./sim_output
        """,
    )
    parser.add_argument("--data-dir",         default="./data",          help="Raw OHLCV parquet directory")
    parser.add_argument("--feature-dir",      default="./feature_cache", help="Pre-computed feature cache directory")
    parser.add_argument("--out-dir",          default=".",               help="Output directory for CSV logs and checkpoint")
    parser.add_argument("--train-days",       type=int,   default=365,   help="Training window length in days")
    parser.add_argument("--predict-days",     type=int,   default=30,    help="Prediction window length in days")
    parser.add_argument("--horizon-bars",     type=int,   default=48,    help="Trade exit horizon in bars (48=4H, 288=1D)")
    parser.add_argument("--max-symbols",      type=int,   default=None,  help="Cap universe size (for testing)")
    parser.add_argument("--min-train-symbols",type=int,   default=5,     help="Minimum symbols needed to train")
    parser.add_argument("--no-resume",        action="store_true",       help="Ignore checkpoint and start fresh")
    parser.add_argument("--quiet",            action="store_true",       help="Suppress progress output")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    feature_dir = Path(args.feature_dir)
    out_dir     = Path(args.out_dir)

    if not data_dir.exists():
        print(f"[ERROR] data-dir not found: {data_dir}"); sys.exit(1)

    if not feature_dir.exists():
        print(f"[ERROR] feature-dir not found: {feature_dir}")
        print("  Run:  python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache")
        sys.exit(1)

    # Discover symbols
    cached = sorted([f.stem for f in feature_dir.glob("*.parquet")])
    if not cached:
        print("[ERROR] feature_cache/ is empty. Run build_feature_cache.py first.")
        sys.exit(1)

    cached_symbols = [s for s in cached if s.endswith("USDT")]
    requested_symbols = [
        s.strip() for s in os.environ.get("AZALYST_TEST_COINS", "").split(",")
        if s.strip()
    ]
    if requested_symbols:
        available = set(cached_symbols)
        symbols = [s for s in requested_symbols if s in available]
        missing = [s for s in requested_symbols if s not in available]
        if not symbols:
            print("[ERROR] AZALYST_TEST_COINS did not match any cached symbols.")
            sys.exit(1)
        print(f"  Using requested test symbols: {len(symbols)}")
        if missing:
            print(f"  Skipping {len(missing)} symbols missing from feature cache.")
    else:
        symbols = cached_symbols
        if args.max_symbols:
            symbols = symbols[:args.max_symbols]

    print(f"\n  Found {len(symbols)} cached symbols.")
    print(f"  Feature cache: {feature_dir.resolve()}")
    print(f"  Output:        {out_dir.resolve()}")

    simulator = WalkForwardSimulator(
        data_dir          = data_dir,
        feature_dir       = feature_dir,
        out_dir           = out_dir,
        symbols           = symbols,
        train_days        = args.train_days,
        predict_days      = args.predict_days,
        horizon_bars      = args.horizon_bars,
        min_train_symbols = args.min_train_symbols,
        verbose           = not args.quiet,
    )

    simulator.run(resume=not args.no_resume)


if __name__ == "__main__":
    main()
