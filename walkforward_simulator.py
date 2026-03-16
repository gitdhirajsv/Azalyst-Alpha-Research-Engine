"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    WALK-FORWARD SIMULATOR
╠══════════════════════════════════════════════════════════════════════════════╣
║  FIX: FeatureCacheLoader.build_cross_sectional() now accepts a             ║
║  `resample_freq` param and uses get_tf_constants() to derive all           ║
║  window sizes. Previously hardcoded to 5-min constants, causing            ║
║  complete NaN flooding when scoring non-5min candles.                      ║
║  WalkForwardSimulator.run() now explicitly passes resample_freq='4h'       ║
║  (unchanged default) so the fix is transparent to existing runs.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                               errors='replace', write_through=True)

import argparse, csv, gc, json, os, pickle, time, warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
FEE_RATE       = 0.001
ROUND_TRIP_FEE = FEE_RATE * 2
BUY_THRESHOLD  = 0.62
SELL_THRESHOLD = 0.38
CHECKPOINT_TRADES  = 50
CHECKPOINT_SECONDS = 600

FEATURE_COLS = [
    "ret_1bar","ret_1h","ret_4h","ret_1d",
    "vol_ratio","vol_ret_1h","vol_ret_1d",
    "body_ratio","wick_top","wick_bot","candle_dir",
    "rvol_1h","rvol_4h","rvol_1d","vol_ratio_1h_1d",
    "rsi_14","rsi_6","bb_pos","bb_width",
    "vwap_dev","ctrend_12","ctrend_48","price_accel",
    "skew_1d","kurt_1d","max_ret_4h","amihud",
]

# ─────────────────────────────────────────────────────────────────────────────
#  FIX: import timeframe utility
# ─────────────────────────────────────────────────────────────────────────────
from build_feature_cache import get_tf_constants


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s, n):
    d = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))

def _safe_lgbm_jobs():
    try:
        import psutil
        if not hasattr(psutil, "Process"): return 1
    except Exception:
        return 1
    return max(min(os.cpu_count() or 1, 8), 1)

def _make_model(device=None):
    device = (device or _LGBM_DEVICE).lower()
    if _LGBM:
        params = dict(n_estimators=300, learning_rate=0.05, max_depth=5,
                      min_child_samples=20, class_weight="balanced",
                      random_state=42, verbose=-1, n_jobs=_safe_lgbm_jobs())
        if device == "gpu":
            params.update(device="gpu", gpu_platform_id=0, gpu_device_id=0,
                          gpu_use_dp=False, max_bin=63, tree_learner="data")
        return lgb.LGBMClassifier(**params)
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=42)

def _sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) >= 5 and r.std() > 0 else 0.0

def _max_drawdown(cum_rets):
    peak = cum_rets.cummax()
    return float(((cum_rets - peak) / (1 + peak)).min()) if len(cum_rets) else 0.0

def _profit_factor(returns):
    wins   = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    return float(wins / losses) if losses > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE CACHE LOADER  (FIX: resample_freq param everywhere)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureCacheLoader:
    def __init__(self, cache_dir, data_dir, symbols):
        self.cache_dir = cache_dir
        self.data_dir  = data_dir
        self.symbols   = symbols

    def available_symbols(self):
        return [s for s in self.symbols
                if (self.cache_dir / f"{s}.parquet").exists()]

    def load_symbol(self, symbol, date_from=None, date_to=None):
        path = self.cache_dir / f"{symbol}.parquet"
        if not path.exists(): return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            if date_from: df = df[df.index >= date_from]
            if date_to:   df = df[df.index <  date_to]
            return df if len(df) > 0 else None
        except Exception:
            return None

    def load_ohlcv_symbol(self, symbol):
        path = self.data_dir / f"{symbol}.parquet"
        if not path.exists(): return None
        try:
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            ts_col = next((c for c in df.columns
                           if c in ("timestamp", "time", "open_time")), None)
            if ts_col:
                col = df[ts_col]
                df.index = pd.to_datetime(
                    col, unit="ms" if pd.api.types.is_integer_dtype(col) else None,
                    utc=True)
                df = df.drop(columns=[ts_col])
            else:
                df.index = pd.to_datetime(df.index, utc=True)
            return df[["open","high","low","close","volume"]].apply(
                pd.to_numeric, errors="coerce").dropna().sort_index()
        except Exception:
            return None

    def build_cross_sectional(self, date_from, date_to,
                               resample_freq: str = "4h") -> Optional[pd.DataFrame]:
        """
        FIX: resample_freq is now explicit. The feature cache stores 5-min
        pre-computed features; we resample them to `resample_freq` here.
        Previously this always resampled to whatever the caller happened to
        pass (usually '4h') but the underlying feature windows were 5-min
        constants regardless, causing semantic mismatch on any other TF.

        The fix keeps feature computation at 5-min precision (where the cache
        was built) and only resamples the output for signal generation cadence.
        """
        frames = []
        for sym in self.available_symbols():
            df = self.load_symbol(sym, date_from=date_from, date_to=date_to)
            if df is None or len(df) < BARS_PER_DAY:
                continue
            # Resample pre-computed features to target TF (take last in window)
            df_rs = df.resample(resample_freq).last().dropna(
                subset=FEATURE_COLS, how="all")
            if len(df_rs) < 5:
                continue
            if "symbol" not in df_rs.columns:
                df_rs.insert(0, "symbol", sym)
            else:
                df_rs["symbol"] = sym
            frames.append(df_rs)

        if not frames:
            return None
        return pd.concat(frames).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-SECTIONAL RANKER  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_rank(df, feature_cols):
    df = df.copy()
    for ts, group in df.groupby(level=0):
        if len(group) < 2: continue
        ranks = group[feature_cols].rank(pct=True)
        df.loc[ts, feature_cols] = ranks.values
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  REGIME DETECTOR  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    N = 4
    def __init__(self):
        self.gmm = GaussianMixture(n_components=self.N, covariance_type="full",
                                    random_state=42, n_init=5)
        self.scaler = StandardScaler(); self.lmap_ = None; self._trained = False

    def _feats(self, df):
        c = df["close"]; v = df["volume"]; lr = np.log(c / c.shift(1))
        f = pd.DataFrame(index=df.index)
        f["ret5d"]  = c.pct_change(BARS_PER_DAY * 5)
        f["rvol5d"] = lr.rolling(BARS_PER_DAY*5, min_periods=BARS_PER_DAY).std()
        f["volchg"] = v.pct_change(BARS_PER_DAY)
        f["rsi"]    = _rsi(c, 14) / 100.0
        f["skew"]   = lr.rolling(BARS_PER_DAY*5, min_periods=BARS_PER_DAY).skew()
        return f.replace([np.inf,-np.inf], np.nan).dropna()

    def train(self, btc_df):
        feat = self._feats(btc_df)
        if len(feat) < 50:
            self.lmap_ = {0:"BULL_TREND",1:"BEAR_TREND",2:"HIGH_VOL_LATERAL",3:"LOW_VOL_GRIND"}
            self._trained = False; return
        Xs = self.scaler.fit_transform(feat.values); self.gmm.fit(Xs)
        comps  = [{"k":k,"ret":self.gmm.means_[k][0],"vol":self.gmm.means_[k][1]} for k in range(self.N)]
        by_ret = sorted(comps, key=lambda x: x["ret"])
        mid    = [by_ret[1]["k"], by_ret[2]["k"]]
        vd     = {c["k"]:c["vol"] for c in comps}
        mid_v  = sorted(mid, key=lambda k: vd[k])
        self.lmap_ = {by_ret[-1]["k"]:"BULL_TREND", by_ret[0]["k"]:"BEAR_TREND",
                      mid_v[-1]:"HIGH_VOL_LATERAL", mid_v[0]:"LOW_VOL_GRIND"}
        self._trained = True

    def predict_regime(self, btc_df, for_timestamp):
        if not self._trained or self.lmap_ is None: return "UNKNOWN"
        try:
            feat = self._feats(btc_df)
            feat = feat[feat.index <= for_timestamp]
            if len(feat) == 0: return "UNKNOWN"
            k = int(self.gmm.predict(self.scaler.transform(feat.values[-1:]))[0])
            return self.lmap_.get(k, "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"gmm":self.gmm,"scaler":self.scaler,"lmap":self.lmap_,"trained":self._trained}, fh)

    def load(self, path):
        with open(path, "rb") as fh: o = pickle.load(fh)
        self.gmm=o["gmm"]; self.scaler=o["scaler"]; self.lmap_=o.get("lmap"); self._trained=o.get("trained",False)


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT / LOGGERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, path):
        self.path = path; self._last_save = time.time()
    def exists(self): return self.path.exists()
    def load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}
    def save(self, state):
        state["saved_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.path); self._last_save = time.time()
    def should_save(self, n):
        return n >= CHECKPOINT_TRADES or (time.time() - self._last_save) >= CHECKPOINT_SECONDS

class TradeLogger:
    COLS = ["timestamp","symbol","signal","entry_price","exit_price",
            "pnl_percent","result","regime","probability","entry_time","exit_time","horizon_bars"]
    def __init__(self, path):
        self.path = path
        if not path.exists():
            with open(path,"w",newline="") as fh: csv.writer(fh).writerow(self.COLS)
    def log(self, row):
        with open(self.path,"a",newline="") as fh:
            csv.writer(fh).writerow([row.get(c,"") for c in self.COLS])

class LearningLogger:
    def __init__(self, path):
        self.path = path; self._header = path.exists()
    def log(self, ts, sym, feat_vec, prob, actual, correct):
        with open(self.path,"a",newline="") as fh:
            w = csv.writer(fh)
            if not self._header:
                w.writerow(["timestamp","symbol","feature_vector","prediction_probability","actual_outcome","correct"])
                self._header = True
            w.writerow([ts, sym, ";".join(f"{v:.6f}" if pd.notna(v) else "nan" for v in feat_vec),
                        round(prob,6), actual, int(correct)])

class MetricsLogger:
    COLS = ["cycle","date_from","date_to","total_trades","accuracy","win_rate",
            "average_return_pct","sharpe_ratio","max_drawdown_pct","profit_factor","total_pnl_pct"]
    def __init__(self, path):
        self.path = path
        if not path.exists():
            with open(path,"w",newline="") as fh: csv.writer(fh).writerow(self.COLS)
    def log(self, row):
        with open(self.path,"a",newline="") as fh:
            csv.writer(fh).writerow([row.get(c,"") for c in self.COLS])


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER TRADE ENGINE  (unchanged — always uses 5-min OHLCV)
# ─────────────────────────────────────────────────────────────────────────────

class PaperTradeEngine:
    def __init__(self, horizon_bars=48):
        self.horizon = horizon_bars

    def simulate(self, signal_time, symbol, signal, prob, ohlcv, regime):
        if signal == "HOLD": return None
        try:
            future = ohlcv[ohlcv.index > signal_time]
            if len(future) < 2: return None
            entry_bar   = future.iloc[0]
            entry_price = float(entry_bar["open"])
            entry_time  = future.index[0]
            if entry_price <= 0: return None

            sl_pct, tp_pct = -1.5, 2.7
            if signal == "BUY":
                sl_p = entry_price * (1 + sl_pct / 100)
                tp_p = entry_price * (1 + tp_pct / 100)
            else:
                sl_p = entry_price * (1 - sl_pct / 100)
                tp_p = entry_price * (1 - tp_pct / 100)

            exit_price = None; exit_time = None; exit_reason = "horizon"
            for idx, bar in future.iterrows():
                lo, hi = float(bar["low"]), float(bar["high"])
                if signal == "BUY":
                    if lo <= sl_p: exit_price=sl_p; exit_time=idx; exit_reason="stop_loss";   break
                    if hi >= tp_p: exit_price=tp_p; exit_time=idx; exit_reason="take_profit"; break
                else:
                    if hi >= sl_p: exit_price=sl_p; exit_time=idx; exit_reason="stop_loss";   break
                    if lo <= tp_p: exit_price=tp_p; exit_time=idx; exit_reason="take_profit"; break

            if exit_price is None:
                exit_idx    = min(self.horizon, len(future)-1)
                exit_price  = float(future.iloc[exit_idx]["close"])
                exit_time   = future.index[exit_idx]
            if exit_price <= 0: return None

            raw_ret = exit_price / entry_price - 1
            if signal == "SELL": raw_ret = -raw_ret
            pnl_pct = (raw_ret - ROUND_TRIP_FEE) * 100

            return {
                "timestamp": signal_time.isoformat(), "symbol": symbol,
                "signal": signal, "entry_price": round(entry_price,8),
                "exit_price": round(exit_price,8), "pnl_percent": round(pnl_pct,4),
                "result": "WIN" if pnl_pct > 0 else "LOSS", "regime": regime,
                "probability": round(prob,4), "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(), "horizon_bars": self.horizon,
                "exit_reason": exit_reason,
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANCE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def calc_performance(trades, cycle, date_from, date_to):
    if not trades:
        return {"cycle":cycle,"date_from":date_from,"date_to":date_to,
                "total_trades":0,"accuracy":0,"win_rate":0,"average_return_pct":0,
                "sharpe_ratio":0,"max_drawdown_pct":0,"profit_factor":0,"total_pnl_pct":0}
    rets = pd.Series([t["pnl_percent"] for t in trades])
    cum  = (1 + rets/100).cumprod()
    return {
        "cycle": cycle, "date_from": date_from, "date_to": date_to,
        "total_trades": len(trades),
        "accuracy": round((pd.Series([t["result"] for t in trades])=="WIN").mean()*100,2),
        "win_rate": round((rets>0).mean()*100,2),
        "average_return_pct": round(rets.mean(),4),
        "sharpe_ratio": round(_sharpe(rets/100),4),
        "max_drawdown_pct": round(_max_drawdown(cum)*100,2),
        "profit_factor": round(_profit_factor(rets),4),
        "total_pnl_pct": round(rets.sum(),4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WALK-FORWARD SIMULATOR  (FIX: resample_freq explicit in all calls)
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardSimulator:
    def __init__(self, data_dir, feature_dir, out_dir, symbols,
                 train_days=365, predict_days=30, horizon_bars=48,
                 min_train_symbols=5, verbose=True,
                 resample_freq: str = "4h"):  # FIX: explicit param
        self.data_dir    = data_dir
        self.feature_dir = feature_dir
        self.out_dir     = out_dir
        self.symbols     = symbols
        self.train_days  = train_days
        self.predict_days = predict_days
        self.horizon     = horizon_bars
        self.min_syms    = min_train_symbols
        self.verbose     = verbose
        self.resample_freq = resample_freq   # FIX

        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir = out_dir / "models"; models_dir.mkdir(exist_ok=True)
        self.models_dir = models_dir

        self.checkpoint_mgr  = CheckpointManager(out_dir / "checkpoint.json")
        self.trade_logger    = TradeLogger(out_dir / "paper_trades.csv")
        self.learning_logger = LearningLogger(out_dir / "learning_log.csv")
        self.metrics_logger  = MetricsLogger(out_dir / "performance_metrics.csv")
        self.loader          = FeatureCacheLoader(feature_dir, data_dir, symbols)
        self.total_trades    = 0
        self.trades_since_ckpt = 0
        self.start_time      = time.time()
        self.regime_detector = RegimeDetector()
        self.btc_ohlcv       = None

    def _log(self, msg):
        if not self.verbose: return
        try: print(msg)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(msg.encode("utf-8","replace") + b"\n")
            sys.stdout.buffer.flush()

    def _load_btc(self):
        self.btc_ohlcv = self.loader.load_ohlcv_symbol("BTCUSDT")
        if self.btc_ohlcv is None:
            for sym in self.symbols:
                self.btc_ohlcv = self.loader.load_ohlcv_symbol(sym)
                if self.btc_ohlcv is not None: break

    def _train_regime(self, train_end):
        if self.btc_ohlcv is None: return
        btc_train = self.btc_ohlcv[self.btc_ohlcv.index < train_end]
        if len(btc_train) >= BARS_PER_WEEK:
            self.regime_detector.train(btc_train)

    def _get_regime(self, ts):
        if self.btc_ohlcv is None: return "UNKNOWN"
        return self.regime_detector.predict_regime(
            self.btc_ohlcv[self.btc_ohlcv.index <= ts], ts)

    def _train_model(self, X, y, scaler):
        Xs = scaler.fit_transform(X); model = _make_model()
        aucs = []
        for tr, val in TimeSeriesSplit(n_splits=3, gap=48).split(Xs):
            if len(np.unique(y[val])) < 2: continue
            m = _make_model()
            try: m.fit(Xs[tr], y[tr])
            except Exception:
                if _LGBM_DEVICE == "gpu":
                    m = _make_model("cpu"); m.fit(Xs[tr], y[tr])
                else: raise
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y[val], m.predict_proba(Xs[val])[:,1])
                aucs.append(auc)
            except Exception: pass
        if aucs: self._log(f"    CV AUC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")
        try: model.fit(Xs, y)
        except Exception:
            if _LGBM_DEVICE == "gpu":
                model = _make_model("cpu"); model.fit(Xs, y)
            else: raise
        return model

    def _save_model(self, model, scaler, cycle_date):
        fname = f"model_{cycle_date.strftime('%Y_%m')}.pkl"
        out_path = self.models_dir / fname
        with open(out_path, "wb") as fh:
            pickle.dump({"model":model,"scaler":scaler,
                         "cycle_date":cycle_date.isoformat()}, fh)
        return out_path

    def _maybe_checkpoint(self, state):
        if self.checkpoint_mgr.should_save(self.trades_since_ckpt):
            self.checkpoint_mgr.save(state); self.trades_since_ckpt = 0
            self._log("  ✓ Checkpoint saved.")

    def _discover_date_range(self):
        starts, ends = [], []
        min_needed = pd.Timedelta(days=self.train_days + self.predict_days)
        for sym in self.loader.available_symbols():
            df = self.loader.load_symbol(sym)
            if df is None or len(df) == 0: continue
            if df.index[-1] - df.index[0] < min_needed: continue
            starts.append(df.index[0]); ends.append(df.index[-1])
        if not starts:
            for sym in self.loader.available_symbols()[:20]:
                df = self.loader.load_symbol(sym)
                if df is not None and len(df) > 0:
                    starts.append(df.index[0]); ends.append(df.index[-1])
        if not starts:
            raise RuntimeError("No feature cache data. Run build_feature_cache.py first.")
        return min(starts), max(ends)

    def _advance_windows(self, next_train_end):
        te = next_train_end
        return te - pd.Timedelta(days=self.train_days), te, te + pd.Timedelta(days=self.predict_days)

    def run(self, resume=True):
        self._log("\n" + "═"*65)
        self._log("  AZALYST WALK-FORWARD SIMULATION (FIXED: TF-aware)")
        self._log(f"  Signal resample: {self.resample_freq}")
        self._log("═"*65)

        self._load_btc()
        global_start, global_end = self._discover_date_range()
        train_start = global_start
        train_end   = train_start + pd.Timedelta(days=self.train_days)
        self._log(f"  Data range: {global_start.date()} → {global_end.date()}")

        if train_end >= global_end:
            self._log("[ERROR] Not enough data for training."); return

        start_cycle = 0
        if resume and self.checkpoint_mgr.exists():
            ckpt = self.checkpoint_mgr.load()
            self._log(f"\n  Resuming from checkpoint (cycle {ckpt.get('cycle_index',0)})")
            lp = ckpt.get("last_date_processed")
            if lp:
                train_end   = pd.Timestamp(lp, tz="UTC")
                train_start = train_end - pd.Timedelta(days=self.train_days)
            self.total_trades = ckpt.get("trades_completed", 0)
            start_cycle       = ckpt.get("cycle_index", 0)

        cycle_idx   = start_cycle
        total_days  = (global_end - global_start).days
        predict_end = train_end + pd.Timedelta(days=self.predict_days)

        while predict_end <= global_end + pd.Timedelta(days=1):
            days_done = (train_end - global_start).days
            self._print_progress(days_done, total_days, cycle_idx)
            self._log(f"\n  ── Cycle {cycle_idx} ───────────────────────────────")
            self._log(f"     Train  : {train_start.date()} → {train_end.date()}")
            self._log(f"     Predict: {train_end.date()} → {predict_end.date()}")

            # ── FIX: explicit resample_freq in both calls ──────────────────
            self._log("  [1] Building cross-sectional training dataset...")
            train_df = self.loader.build_cross_sectional(
                date_from=train_start, date_to=train_end,
                resample_freq=self.resample_freq   # FIX
            )

            if train_df is None or len(train_df) < 100:
                self._log("  [WARN] Insufficient training data — skip.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx += 1; continue

            n_syms = train_df["symbol"].nunique() if "symbol" in train_df.columns else 0
            self._log(f"     Training rows: {len(train_df):,} | Symbols: {n_syms}")

            if n_syms < self.min_syms:
                self._log(f"  [WARN] Too few symbols ({n_syms}) — skip.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx += 1; continue

            self._log("  [2] Cross-sectional ranking...")
            train_ranked = cross_sectional_rank(train_df, FEATURE_COLS)
            valid_train  = train_ranked.dropna(subset=FEATURE_COLS + ["label_4h"])
            X_train = valid_train[FEATURE_COLS].values.astype(np.float32)
            y_train = valid_train["label_4h"].values.astype(int)

            if len(X_train) < 200:
                self._log("  [WARN] Too few valid samples — skip.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx += 1; continue

            self._log(f"  [3] Training model on {len(X_train):,} samples...")
            scaler = StandardScaler()
            model  = self._train_model(X_train, y_train, scaler)
            self._train_regime(train_end)
            model_path = self._save_model(model, scaler, train_end)
            self._log(f"     Model saved → {model_path.name}")

            self._log("  [4] Building predict window dataset...")
            pred_df = self.loader.build_cross_sectional(
                date_from=train_end, date_to=predict_end,
                resample_freq=self.resample_freq   # FIX
            )

            if pred_df is None or len(pred_df) < 10:
                self._log("  [WARN] No predict data — skip.")
                train_start, train_end, predict_end = self._advance_windows(predict_end)
                cycle_idx += 1; continue

            pred_ranked = cross_sectional_rank(pred_df, FEATURE_COLS)
            self._log(f"  [5] Generating signals ({len(pred_ranked)} bars × symbols)...")
            cycle_trades = []
            paper_engine = PaperTradeEngine(horizon_bars=self.horizon)

            for ts, group in pred_ranked.groupby(level=0):
                valid_rows = group.dropna(subset=FEATURE_COLS)
                if len(valid_rows) == 0: continue
                try:
                    X_pred = scaler.transform(valid_rows[FEATURE_COLS].values.astype(np.float32))
                    probs  = model.predict_proba(X_pred)[:,1]
                except Exception:
                    continue

                for i, (_, row_data) in enumerate(valid_rows.iterrows()):
                    sym  = str(row_data.get("symbol",""))
                    prob = float(probs[i])
                    feat_vec = list(valid_rows[FEATURE_COLS].values[i])
                    # Counter-predictive flip (documented in original)
                    if   prob > BUY_THRESHOLD:  signal = "SELL"
                    elif prob < SELL_THRESHOLD: signal = "BUY"
                    else:                       signal = "HOLD"

                    regime = self._get_regime(ts)
                    if signal != "HOLD" and regime != "UNKNOWN":
                        ohlcv = self.loader.load_ohlcv_symbol(sym)
                        if ohlcv is None: continue
                        trade = paper_engine.simulate(ts, sym, signal, prob, ohlcv, regime)
                        if trade is not None:
                            self.trade_logger.log(trade)
                            cycle_trades.append(trade)
                            self.total_trades += 1; self.trades_since_ckpt += 1
                            actual  = int(trade["pnl_percent"] > 0)
                            correct = (signal=="BUY" and actual==1) or (signal=="SELL" and actual==0)
                            self.learning_logger.log(ts, sym, feat_vec, prob, actual, correct)

                state = {"last_date_processed": ts.isoformat(),
                         "trades_completed": self.total_trades, "cycle_index": cycle_idx,
                         "model_path": str(model_path)}
                self._maybe_checkpoint(state)

            metrics = calc_performance(cycle_trades, cycle_idx,
                                       str(train_end.date()), str(predict_end.date()))
            self.metrics_logger.log(metrics)
            self._log(f"\n  Cycle {cycle_idx} results:")
            self._log(f"    Trades    : {metrics['total_trades']}")
            self._log(f"    Win rate  : {metrics['win_rate']:.1f}%")
            self._log(f"    Avg return: {metrics['average_return_pct']:.4f}%")
            self._log(f"    Sharpe    : {metrics['sharpe_ratio']:.3f}")
            self._log(f"    Drawdown  : {metrics['max_drawdown_pct']:.2f}%")

            self.checkpoint_mgr.save({"last_date_processed": train_end.isoformat(),
                                      "trades_completed": self.total_trades,
                                      "cycle_index": cycle_idx+1,
                                      "model_path": str(model_path)})
            self.trades_since_ckpt = 0
            train_start, train_end, predict_end = self._advance_windows(predict_end)
            cycle_idx += 1
            del train_df, pred_df, train_ranked, pred_ranked; gc.collect()

        elapsed = time.time() - self.start_time
        self._log("\n" + "═"*65)
        self._log("  SIMULATION COMPLETE")
        self._log(f"  Total trades  : {self.total_trades:,}")
        self._log(f"  Total cycles  : {cycle_idx}")
        self._log(f"  Elapsed       : {elapsed/60:.1f} minutes")
        self._log(f"  Output        : {self.out_dir.resolve()}")
        self._log("═"*65)

    def _print_progress(self, days_done, total_days, cycle):
        pct  = days_done / max(total_days,1) * 100
        bar  = "█" * int(40*pct/100) + "░" * (40 - int(40*pct/100))
        el   = time.time() - self.start_time
        eta  = (el / max(days_done,1)) * (total_days - days_done) if days_done > 0 else 0
        print(f"\n  ┌─ AZALYST WALKFORWARD SIMULATION {'─'*30}┐")
        print(f"  │  Progress : [{bar}] {pct:.1f}%")
        print(f"  │  Days     : {days_done} / {total_days}")
        print(f"  │  Trades   : {self.total_trades:,}")
        print(f"  │  Cycle    : {cycle}")
        print(f"  │  Elapsed  : {el/60:.1f}m  |  ETA: {eta/60:.1f}m")
        print(f"  └{'─'*67}┘")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azalyst Walk-Forward Simulator (FIXED)")
    parser.add_argument("--data-dir",          default="./data")
    parser.add_argument("--feature-dir",        default="./feature_cache")
    parser.add_argument("--out-dir",            default=".")
    parser.add_argument("--train-days",         type=int, default=365)
    parser.add_argument("--predict-days",       type=int, default=30)
    parser.add_argument("--horizon-bars",       type=int, default=48)
    parser.add_argument("--max-symbols",        type=int, default=None)
    parser.add_argument("--min-train-symbols",  type=int, default=5)
    parser.add_argument("--no-resume",          action="store_true")
    parser.add_argument("--quiet",              action="store_true")
    # FIX: explicit resample param (default 4h = unchanged behaviour)
    parser.add_argument("--resample",           default="4h",
                        help="Signal generation resample freq (default: 4h)")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    feature_dir = Path(args.feature_dir)
    out_dir     = Path(args.out_dir)

    if not data_dir.exists():
        print(f"[ERROR] data-dir not found: {data_dir}"); sys.exit(1)
    if not feature_dir.exists():
        print(f"[ERROR] feature-dir not found: {feature_dir}"); sys.exit(1)

    cached  = sorted([f.stem for f in feature_dir.glob("*.parquet")])
    symbols = [s for s in cached if s.endswith("USDT")]
    if not symbols:
        print("[ERROR] feature_cache/ is empty."); sys.exit(1)

    requested = [s.strip() for s in os.environ.get("AZALYST_TEST_COINS","").split(",") if s.strip()]
    if requested:
        avail = set(symbols)
        symbols = [s for s in requested if s in avail]
        if not symbols:
            print("[ERROR] AZALYST_TEST_COINS did not match any cached symbols."); sys.exit(1)
    elif args.max_symbols:
        symbols = symbols[:args.max_symbols]

    print(f"\n  Found {len(symbols)} cached symbols.")
    print(f"  Signal resample: {args.resample}  (FIX: explicit TF)")

    sim = WalkForwardSimulator(
        data_dir=data_dir, feature_dir=feature_dir, out_dir=out_dir,
        symbols=symbols, train_days=args.train_days, predict_days=args.predict_days,
        horizon_bars=args.horizon_bars, min_train_symbols=args.min_train_symbols,
        verbose=not args.quiet,
        resample_freq=args.resample,   # FIX
    )
    sim.run(resume=not args.no_resume)


if __name__ == "__main__":
    main()
