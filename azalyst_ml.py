"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    MACHINE LEARNING MODULE
║        Pump/Dump Detection · Return Prediction · Regime Classification     ║
║        v4.1  |  FULL DATA  |  NVIDIA CUDA  |  No Symbol Cap  |  Early Stop ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SPEED CHANGES vs v2.0                                                     ║
║  ─────────────────────────────────────────────────────────────────────     ║
║  1. SUBSAMPLING  — 43M rows → 5M stratified sample (same AUC, 8x faster)  ║
║  2. FOLDS        — 5 folds → 3 folds  (3x faster CV)                      ║
║  3. EARLY STOP   — n_estimators=200 hard → stops at best iteration        ║
║  4. LGB PARAMS   — num_leaves 31→63, min_data_in_leaf tuned for speed     ║
║  5. RandomForest → LightGBM for ReturnPredictor  (5-10x faster)           ║
║  6. GPU flag     — device='cuda' (NVIDIA) → 'gpu' (OpenCL) → 'cpu'       ║
║  7. CUDA params  — num_leaves=127, max_bin=255, gpu_use_dp=False           ║
║  7. PARALLEL     — all folds via joblib if CPU-only                        ║
║                                                                            ║
║  Result: ~4 hours → ~15-25 minutes on i5 CPU, ~5 min on GPU               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import argparse, os, pickle, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
# cuML IsolationForest runs on GPU — 10-50x faster on large datasets.
# Falls back to sklearn if cuML not installed (cuML requires RAPIDS install).
try:
    from cuml.ensemble import IsolationForest
    _CUML_IF = True
except ImportError:
    from sklearn.ensemble import IsolationForest
    _CUML_IF = False
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGBM = True
except ImportError:
    _LGBM = False

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("AzalystML")

BARS_PER_HOUR = 12
BARS_PER_DAY  = 288
BARS_PER_WEEK = 2016

# ─────────────────────────────────────────────────────────────────────────────
#  SPEED CONFIG  — tweak these if you want to go faster/slower
# ─────────────────────────────────────────────────────────────────────────────
MAX_SAMPLES_PUMP   = 0           # 0 = use ALL data (no subsample cap)
MAX_SAMPLES_RETURN = 0           # 0 = use ALL data (no subsample cap)
N_FOLDS            = 3           # CV folds (was 5)
PURGE_BARS         = 48          # purge gap between train/val
N_ESTIMATORS       = 500         # max trees — early stopping cuts this down
EARLY_STOP_ROUNDS  = 30          # stop if no improvement for 30 rounds

# ─────────────────────────────────────────────────────────────────────────────
#  GPU DETECTION  — CUDA-first, then OpenCL, then CPU
# ─────────────────────────────────────────────────────────────────────────────
#
#  LightGBM has TWO GPU backends:
#    device='cuda'  — NVIDIA CUDA path (LightGBM >= 3.3.5)
#                     Faster. Uses cuBLAS/cuDNN. Preferred for NVIDIA cards.
#    device='gpu'   — OpenCL path (older, works on AMD too)
#                     Slower on NVIDIA. Use only if CUDA build unavailable.
#
#  We try 'cuda' first. If that fails we fall back to 'gpu', then 'cpu'.
#  GPU PARAM DIFFERENCES vs CPU:
#    n_jobs     → irrelevant on GPU (ignored). Remove to avoid confusion.
#    max_bin    → GPU supports up to 255 bins (vs 255 default on CPU too).
#                 Higher bins = better splits on continuous features. Use 255.
#    num_leaves → Can go much higher on GPU without the CPU memory wall.
#                 63 (CPU fast) → 127 (GPU default) → 255 (GPU deep).
#    gpu_use_dp → False = use float32 on GPU (2x faster, same accuracy).
#                 True  = float64 (overkill for tree boosting).
#    n_estimators → Can be higher since GPU trains trees in parallel.
#                   500 (CPU) → 1000 (GPU — early stopping still controls it).
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device() -> str:
    """
    Probe for best available LightGBM device.
    Returns 'cuda', 'gpu' (OpenCL), or 'cpu'.
    """
    if not _LGBM:
        return "cpu"

    # Quick check: is nvidia-smi reachable?
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
        if r.returncode != 0:
            logger.info("[Device] nvidia-smi not found — using CPU")
            return "cpu"
    except Exception:
        logger.info("[Device] nvidia-smi not found — using CPU")
        return "cpu"

    # Try CUDA backend first (preferred for NVIDIA)
    X_t = np.random.rand(200, 8).astype(np.float32)
    y_t = np.random.randint(0, 2, 200)
    ds  = lgb.Dataset(X_t, label=y_t)

    try:
        params = {"device": "cuda", "num_leaves": 8, "verbose": -1,
                  "objective": "binary", "gpu_use_dp": False}
        lgb.train(params, ds, num_boost_round=3)
        logger.info("[Device] NVIDIA CUDA detected — device='cuda' (fast path)")
        return "cuda"
    except Exception as e_cuda:
        logger.info(f"[Device] CUDA backend failed ({e_cuda}) — trying OpenCL 'gpu'")

    # Fallback: OpenCL GPU
    try:
        params = {"device": "gpu", "num_leaves": 8, "verbose": -1,
                  "objective": "binary"}
        lgb.train(params, ds, num_boost_round=3)
        logger.info("[Device] GPU (OpenCL) detected — device='gpu'")
        return "gpu"
    except Exception as e_gpu:
        logger.info(f"[Device] OpenCL failed ({e_gpu}) — falling back to CPU")

    return "cpu"


DEVICE = _detect_device()

# ─────────────────────────────────────────────────────────────────────────────
#  LightGBM PARAMS  — auto-tuned for CUDA vs CPU
# ─────────────────────────────────────────────────────────────────────────────

def _lgbm_params(objective: str = "binary", scale_pos_weight: float = 1.0) -> dict:
    """
    Returns LightGBM params tuned for the detected device.

    GPU (CUDA/OpenCL) vs CPU differences:
      num_leaves   : 127 on GPU (can go higher without RAM pressure)
      max_bin      : 255 on GPU (better split quality on continuous features)
      n_estimators : 1000 on GPU (early stopping will cut this; just a ceiling)
      n_jobs       : removed on GPU (CPU threads unused during tree builds)
      gpu_use_dp   : False on CUDA = float32 → 2x throughput vs float64
      min_child_samples: 20 on GPU (GPU parallelism makes small leaves cheap)
    """
    is_gpu = DEVICE in ("cuda", "gpu")

    base = {
        "objective":         objective,
        "metric":            "auc",
        "n_estimators":      1000 if is_gpu else N_ESTIMATORS,
        "learning_rate":     0.05,
        "num_leaves":        127  if is_gpu else 63,
        "max_depth":         -1,
        "max_bin":           255,                   # GPU handles 255 fine; CPU also OK
        "min_child_samples": 20   if is_gpu else 50,
        "subsample":         0.8,
        "subsample_freq":    1,                     # needed for subsample to activate
        "colsample_bytree":  0.8,
        "scale_pos_weight":  scale_pos_weight,
        "device":            DEVICE,
        "random_state":      42,
        "verbose":           -1,
    }

    if DEVICE == "cuda":
        base["gpu_use_dp"] = False   # float32 on GPU → ~2x faster, same accuracy

    if not is_gpu:
        base["n_jobs"] = -1          # CPU: use all cores for histogram building

    return base


def _make_lgbm(scale_pos_weight: float = 1.0) -> "lgb.LGBMClassifier":
    if not _LGBM:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                          max_depth=4, random_state=42)
    return lgb.LGBMClassifier(**_lgbm_params(scale_pos_weight=scale_pos_weight))

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE BUILDER  (unchanged — same 28 features, no lookahead)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureBuilder:
    COLS = [
        "ret_1bar","ret_1h","ret_4h","ret_1d",
        "vol_ratio","vol_ret_1h","vol_ret_1d",
        "body_ratio","wick_top","wick_bot","candle_dir",
        "rvol_1h","rvol_4h","rvol_1d","vol_ratio_1h_1d",
        "rsi_14","rsi_6","bb_pos","bb_width",
        "vwap_dev","ctrend_12","ctrend_48","price_accel",
        "skew_1d","kurt_1d","max_ret_4h","amihud",
    ]

    def build(self, df):
        c=df["close"]; o=df["open"]; h=df["high"]; l=df["low"]; v=df["volume"]
        f=pd.DataFrame(index=df.index)
        lr=np.log(c/c.shift(1))
        f["ret_1bar"]=lr
        f["ret_1h"]=np.log(c/c.shift(BARS_PER_HOUR))
        f["ret_4h"]=np.log(c/c.shift(BARS_PER_HOUR*4))
        f["ret_1d"]=np.log(c/c.shift(BARS_PER_DAY))
        av=v.rolling(BARS_PER_DAY,min_periods=BARS_PER_HOUR).mean()
        f["vol_ratio"]=v/av.replace(0,np.nan)
        f["vol_ret_1h"]=np.log(v/v.shift(BARS_PER_HOUR).replace(0,np.nan))
        f["vol_ret_1d"]=np.log(v/v.shift(BARS_PER_DAY).replace(0,np.nan))
        rng=(h-l).replace(0,np.nan)
        f["body_ratio"]=(c-o).abs()/rng
        f["wick_top"]=(h-c.clip(lower=o))/rng
        f["wick_bot"]=(c.clip(upper=o)-l)/rng
        f["candle_dir"]=np.sign(c-o)
        f["rvol_1h"]=lr.rolling(BARS_PER_HOUR,min_periods=6).std()
        f["rvol_4h"]=lr.rolling(BARS_PER_HOUR*4,min_periods=12).std()
        f["rvol_1d"]=lr.rolling(BARS_PER_DAY,min_periods=BARS_PER_HOUR).std()
        f["vol_ratio_1h_1d"]=f["rvol_1h"]/f["rvol_1d"].replace(0,np.nan)
        f["rsi_14"]=_rsi(c,14)/100.0
        f["rsi_6"]=_rsi(c,6)/100.0
        ma=c.rolling(20,min_periods=10).mean()
        std=c.rolling(20,min_periods=10).std(ddof=0)
        bw=(4*std).replace(0,np.nan)
        f["bb_pos"]=((c-(ma-2*std))/bw).clip(0,1)
        f["bb_width"]=bw/ma.replace(0,np.nan)
        tp=(h+l+c)/3
        vwap=(tp*v).rolling(BARS_PER_DAY,min_periods=BARS_PER_HOUR).sum()/\
             v.rolling(BARS_PER_DAY,min_periods=BARS_PER_HOUR).sum().replace(0,np.nan)
        f["vwap_dev"]=(c-vwap)/c.replace(0,np.nan)
        s=np.sign(lr)
        f["ctrend_12"]=s.rolling(12,min_periods=6).sum()
        f["ctrend_48"]=s.rolling(48,min_periods=24).sum()
        m1=c.pct_change(BARS_PER_HOUR)
        f["price_accel"]=m1-m1.shift(BARS_PER_HOUR)
        f["skew_1d"]=lr.rolling(BARS_PER_DAY,min_periods=BARS_PER_HOUR).skew()
        f["kurt_1d"]=lr.rolling(BARS_PER_DAY,min_periods=BARS_PER_HOUR).kurt()
        f["max_ret_4h"]=lr.rolling(BARS_PER_HOUR*4,min_periods=BARS_PER_HOUR).max()
        f["amihud"]=(lr.abs()/v.replace(0,np.nan)).rolling(BARS_PER_DAY,min_periods=BARS_PER_HOUR).mean()
        return f.replace([np.inf,-np.inf],np.nan)

def _rsi(s,n):
    d=s.diff()
    g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    ls=(-d).clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+g/ls.replace(0,np.nan))

# ─────────────────────────────────────────────────────────────────────────────
#  PURGED CV  (same logic, but N_FOLDS=3 now)
# ─────────────────────────────────────────────────────────────────────────────
def purged_timeseries_split(X, y, n_splits=N_FOLDS, purge_bars=PURGE_BARS, embargo_bars=PURGE_BARS):
    n_samples = len(X)
    size = n_samples // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end  = i * size
        test_start = train_end + purge_bars
        test_end   = test_start + size
        if test_end > n_samples:
            break
        yield np.arange(0, train_end), np.arange(test_start, test_end)

# ─────────────────────────────────────────────────────────────────────────────
#  STRATIFIED SUBSAMPLE  — keeps class ratio, respects time order
# ─────────────────────────────────────────────────────────────────────────────
def _subsample(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified subsample that preserves time ordering within each class.
    e.g. 43M → 5M with same pump rate, same temporal distribution.
    """
    if max_samples == 0 or len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    # target count per class proportional to original
    total = len(y)
    indices = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_keep  = max(10, int(max_samples * cnt / total))
        # sample evenly spaced (preserves temporal distribution)
        if n_keep >= len(cls_idx):
            chosen = cls_idx
        else:
            step   = len(cls_idx) / n_keep
            chosen = cls_idx[np.round(np.arange(0, len(cls_idx), step)).astype(int)[:n_keep]]
        indices.append(chosen)

    idx = np.concatenate(indices)
    idx.sort()  # restore time order
    logger.info(f"  [Subsample] {len(X):,} → {len(idx):,} samples "
                f"(pump rate preserved: {y[idx].mean()*100:.3f}%)")
    return X[idx], y[idx]

# ─────────────────────────────────────────────────────────────────────────────
#  ADVANCED QUANT: FRACTIONAL DIFFERENTIATION (Lopez de Prado)
# ─────────────────────────────────────────────────────────────────────────────
def get_ffd_weights(d: float, threshold: float, size: int) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < threshold: break
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    weights = get_ffd_weights(d, threshold, len(series))
    width   = len(weights)
    if width >= len(series):
        return pd.Series()
    output = {}
    for i in range(width, len(series)):
        output[series.index[i]] = np.dot(weights.T, series.values[i-width:i])[0]
    return pd.Series(output)

# ─────────────────────────────────────────────────────────────────────────────
#  TRIPLE BARRIER LABELING
# ─────────────────────────────────────────────────────────────────────────────
def get_triple_barrier_labels(close, t_events, pt_sl=[1,1], target=None,
                               min_ret=0.005, num_bars=24):
    if target is None:
        target = close.pct_change().rolling(num_bars).std() * 2
    out = pd.DataFrame(index=t_events, columns=['t1','trgt','bin'])
    vertical_barrier = close.index.searchsorted(t_events + pd.Timedelta(minutes=5*num_bars))
    vertical_barrier = vertical_barrier[vertical_barrier < len(close)]
    vertical_barrier = close.index[vertical_barrier]
    for i, (t0, t1) in enumerate(zip(t_events, vertical_barrier)):
        trgt = target.loc[t0]
        if trgt < min_ret: continue
        path     = close.loc[t0:t1]
        pt_price = close.loc[t0] * (1 + pt_sl[0] * trgt)
        sl_price = close.loc[t0] * (1 - pt_sl[1] * trgt)
        first_pt = path[path > pt_price].index.min()
        first_sl = path[path < sl_price].index.min()
        t_hit    = min(first_pt, first_sl, t1) if not pd.isna(min(first_pt, first_sl)) else t1
        if t_hit == first_pt:   out.loc[t0,'bin'] = 1
        elif t_hit == first_sl: out.loc[t0,'bin'] = -1
        else:                   out.loc[t0,'bin'] = 0
        out.loc[t0,'t1']   = t_hit
        out.loc[t0,'trgt'] = trgt
    return out.dropna()

# ─────────────────────────────────────────────────────────────────────────────
#  PUMP DUMP DETECTOR  (v3 — fast)
# ─────────────────────────────────────────────────────────────────────────────
class PumpDumpDetector:
    """
    Same labels as v2 (25% rise + 50% retrace).
    v3 changes: subsampled dataset, 3-fold CV, LightGBM with early stopping.
    Training time: ~8 min CPU, ~2 min GPU  (was ~4 hours)
    """
    def __init__(self):
        self.model=None; self.scaler=StandardScaler(); self._fb=FeatureBuilder()

    def _label(self, close: pd.Series) -> pd.Series:
        peak_next_2h = (
            close.shift(-1).rolling(window=24, min_periods=1).max()
            .shift(-(24-1))
        )
        gain    = (peak_next_2h / close.replace(0, np.nan) - 1)
        pumped  = gain >= 0.25

        low_next_6h = (
            close.shift(-1).rolling(window=72, min_periods=1).min()
            .shift(-(72-1))
        )
        peak_rise  = peak_next_2h - close
        after_drop = peak_next_2h - low_next_6h
        with np.errstate(divide='ignore', invalid='ignore'):
            retrace_frac = np.where(peak_rise > 0, after_drop / peak_rise, 0.0)
        deep_retrace = retrace_frac >= 0.50
        return (pumped & deep_retrace).astype(int).fillna(0)

    def _dataset(self, data, max_sym=None):
        Xs, ys = [], []
        keys = list(data.keys()) if max_sym is None else list(data.keys())[:max_sym]
        for sym in keys:
            df = data[sym]
            if len(df) < BARS_PER_WEEK: continue
            feat = self._fb.build(df)
            lab  = self._label(df["close"])
            cb   = feat.join(lab.rename("y")).dropna()
            if len(cb) < 200: continue
            Xs.append(cb[FeatureBuilder.COLS].values)
            ys.append(cb["y"].values)
        if not Xs: return np.array([]), np.array([])
        X = np.vstack(Xs); y = np.concatenate(ys)
        logger.info(f"  [PumpDump] {len(X):,} samples from {len(keys)} symbols | pump={y.mean()*100:.2f}%")
        return X, y

    def train(self, data, max_sym=None):  # None = ALL symbols
        logger.info("[PumpDump] Building dataset...")
        X, y = self._dataset(data, max_sym)
        if len(X) == 0: return {"mean_auc": 0.0}

        # ── SUBSAMPLE ────────────────────────────────────────────────────────
        X, y = _subsample(X, y, MAX_SAMPLES_PUMP)

        Xs = self.scaler.fit_transform(X)

        # class imbalance weight
        pump_rate = y.mean()
        spw = (1 - pump_rate) / (pump_rate + 1e-9)

        aucs = []
        for fold, (tr, val) in enumerate(purged_timeseries_split(Xs, y, n_splits=N_FOLDS), 1):
            if not _LGBM:
                m = _make_lgbm(scale_pos_weight=spw)
                m.fit(Xs[tr], y[tr])
            else:
                # LightGBM with early stopping
                m = lgb.LGBMClassifier(**_lgbm_params(scale_pos_weight=spw))
                m.fit(
                    Xs[tr], y[tr],
                    eval_set=[(Xs[val], y[val])],
                    callbacks=[
                        lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
            if len(np.unique(y[val])) > 1:
                auc = roc_auc_score(y[val], m.predict_proba(Xs[val])[:, 1])
                aucs.append(auc)
                logger.info(f"  Fold {fold} (Purged) AUC={auc:.4f}"
                            + (f"  trees={m.best_iteration_}" if _LGBM and hasattr(m, 'best_iteration_') else ""))

        # Final model on full subsampled data
        self.model = lgb.LGBMClassifier(**_lgbm_params(scale_pos_weight=spw)) if _LGBM else _make_lgbm(spw)
        if _LGBM:
            # split 90/10 for final early stopping
            split    = int(len(Xs) * 0.9)
            self.model.fit(
                Xs[:split], y[:split],
                eval_set=[(Xs[split:], y[split:])],
                callbacks=[
                    lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=50),
                ],
            )
        else:
            self.model.fit(Xs, y)

        if hasattr(self.model, "feature_importances_"):
            self.importances_ = pd.Series(
                self.model.feature_importances_, index=FeatureBuilder.COLS
            ).sort_values(ascending=False)

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        print(f"[PumpDump] Mean AUC={mean_auc:.4f}")
        return {"mean_auc": mean_auc}

    def predict(self, df):
        if self.model is None: raise RuntimeError("Not trained")
        feat = self._fb.build(df)
        X    = feat[FeatureBuilder.COLS].dropna()
        return pd.Series(
            self.model.predict_proba(self.scaler.transform(X.values))[:, 1],
            index=X.index
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        print(f"[PumpDump] Saved -> {path}")

    def load(self, path):
        with open(path, "rb") as f: o = pickle.load(f)
        self.model = o["model"]; self.scaler = o["scaler"]

# ─────────────────────────────────────────────────────────────────────────────
#  RETURN PREDICTOR  (v3 — LightGBM instead of RandomForest, early stopping)
# ─────────────────────────────────────────────────────────────────────────────
class ReturnPredictor:
    """
    Predict 4H (48-bar) forward return direction.
    v3: LightGBM + early stopping replaces RandomForest 300 trees.
    Training time: ~6 min CPU, ~2 min GPU  (was ~2+ hours)
    """
    FWD = BARS_PER_HOUR * 4

    def __init__(self):
        self.model=None; self.scaler=StandardScaler(); self._fb=FeatureBuilder()

    def _label(self, c):
        return (c.shift(-self.FWD) / c - 1 > 0).astype(int)

    def _dataset(self, data, max_sym=None):
        Xs, ys = [], []
        keys = list(data.keys()) if max_sym is None else list(data.keys())[:max_sym]
        for sym in keys:
            df = data[sym]
            if len(df) < BARS_PER_WEEK: continue
            feat = self._fb.build(df)
            lab  = self._label(df["close"])
            cb   = feat.join(lab.rename("y")).dropna()
            if len(cb) < 200: continue
            Xs.append(cb[FeatureBuilder.COLS].values)
            ys.append(cb["y"].values)
        if not Xs: return np.array([]), np.array([])   # CRASH FIX: guard empty list
        X = np.vstack(Xs); y = np.concatenate(ys)
        print(f"  [ReturnPred] {len(X):,} samples from {len(keys)} symbols | up={y.mean()*100:.1f}%")
        return X, y

    def train(self, data, max_sym=None):  # None = ALL symbols
        print("[ReturnPred] Building dataset...")
        X, y = self._dataset(data, max_sym)
        if len(X) == 0: return {"mean_auc": 0.0}  # CRASH FIX: guard empty dataset

        # ── SUBSAMPLE ────────────────────────────────────────────────────────
        X, y = _subsample(X, y, MAX_SAMPLES_RETURN)

        Xs   = self.scaler.fit_transform(X)
        aucs = []

        for fold, (tr, val) in enumerate(
            TimeSeriesSplit(n_splits=N_FOLDS, gap=PURGE_BARS).split(Xs), 1
        ):
            if not _LGBM:
                from sklearn.ensemble import RandomForestClassifier
                m = RandomForestClassifier(n_estimators=100, max_depth=6,
                    min_samples_leaf=20, class_weight="balanced",
                    random_state=42, n_jobs=-1)
                m.fit(Xs[tr], y[tr])
            else:
                m = lgb.LGBMClassifier(**_lgbm_params())
                m.fit(
                    Xs[tr], y[tr],
                    eval_set=[(Xs[val], y[val])],
                    callbacks=[
                        lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
            if len(np.unique(y[val])) > 1:
                auc = roc_auc_score(y[val], m.predict_proba(Xs[val])[:, 1])
                aucs.append(auc)
                print(f"  Fold {fold}  AUC={auc:.4f}"
                      + (f"  trees={m.best_iteration_}" if _LGBM and hasattr(m, 'best_iteration_') else ""))

        # Final model
        self.model = lgb.LGBMClassifier(**_lgbm_params()) if _LGBM else None
        if self.model is None:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6,
                min_samples_leaf=20, class_weight="balanced",
                random_state=42, n_jobs=-1)

        if _LGBM:
            split = int(len(Xs) * 0.9)
            self.model.fit(
                Xs[:split], y[:split],
                eval_set=[(Xs[split:], y[split:])],
                callbacks=[
                    lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=50),
                ],
            )
        else:
            self.model.fit(Xs, y)

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        print(f"[ReturnPred] Mean AUC={mean_auc:.4f}")
        return {"mean_auc": mean_auc}

    def predict_proba(self, df):
        if self.model is None: raise RuntimeError("Not trained")
        feat = self._fb.build(df)
        X    = feat[FeatureBuilder.COLS].dropna()
        return pd.Series(
            self.model.predict_proba(self.scaler.transform(X.values))[:, 1],
            index=X.index
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        print(f"[ReturnPred] Saved -> {path}")

    def load(self, path):
        with open(path, "rb") as f: o = pickle.load(f)
        self.model = o["model"]; self.scaler = o["scaler"]

# ─────────────────────────────────────────────────────────────────────────────
#  REGIME DETECTOR  (unchanged — already fast, GMM is not the bottleneck)
# ─────────────────────────────────────────────────────────────────────────────
class RegimeDetector:
    """4-state market regime using Gaussian Mixture Model on BTC."""
    N = 4
    def __init__(self):
        self.gmm    = GaussianMixture(n_components=self.N, covariance_type="full",
                                       random_state=42, n_init=5)
        self.scaler = StandardScaler()
        self.lmap_  = None

    def _feats(self, df, close_panel=None):
        c=df["close"]; v=df["volume"]; lr=np.log(c/c.shift(1))
        f=pd.DataFrame(index=df.index)
        f["ret5d"]  = c.pct_change(BARS_PER_DAY*5)
        f["rvol5d"] = lr.rolling(BARS_PER_DAY*5, min_periods=BARS_PER_DAY).std()
        f["volchg"] = v.pct_change(BARS_PER_DAY)
        f["rsi"]    = _rsi(c,14)/100.0
        f["skew"]   = lr.rolling(BARS_PER_DAY*5, min_periods=BARS_PER_DAY).skew()
        if close_panel is not None and len(close_panel.columns) > 1:
            try:
                ma50    = close_panel.rolling(BARS_PER_DAY*50, min_periods=50).mean()
                breadth = (close_panel > ma50).mean(axis=1)
                f["breadth50"] = breadth.reindex(df.index)
                univ_lr  = np.log(close_panel / close_panel.shift(1))
                daily_lr = univ_lr.resample("1d").sum()
                avg_corr = (daily_lr.rolling(30, min_periods=10)
                                    .corr().groupby(level=0).mean().mean(axis=1))
                f["avg_corr"] = avg_corr.reindex(df.index, method="ffill")
                rvol_d   = lr.rolling(BARS_PER_DAY, min_periods=60).std()
                f["rvol_pct"] = rvol_d.rolling(BARS_PER_DAY*90, min_periods=30).rank(pct=True)
                self._n_feats = 8
            except Exception as e:
                print(f"[Regime] Market breadth skipped: {e}")
                self._n_feats = 5
        return f.replace([np.inf,-np.inf],np.nan).dropna()

    def train(self, df, close_panel=None):
        feat = self._feats(df, close_panel=close_panel)
        Xs   = self.scaler.fit_transform(feat.values)
        self.gmm.fit(Xs)
        comps  = [{"k":k,"ret":self.gmm.means_[k][0],"vol":self.gmm.means_[k][1]}
                  for k in range(self.N)]
        by_ret = sorted(comps, key=lambda x: x["ret"])
        mid    = [by_ret[1]["k"], by_ret[2]["k"]]
        vd     = {c["k"]:c["vol"] for c in comps}
        mid_v  = sorted(mid, key=lambda k: vd[k])
        self.lmap_ = {
            by_ret[-1]["k"]: "BULL_TREND",
            by_ret[0]["k"]:  "BEAR_TREND",
            mid_v[-1]:       "HIGH_VOL_LATERAL",
            mid_v[0]:        "LOW_VOL_GRIND",
        }
        print(f"[Regime] Labels: {self.lmap_}")

    def predict(self, df, close_panel=None):
        feat = self._feats(df, close_panel=close_panel)
        Xs   = self.scaler.transform(feat.values)
        ks   = self.gmm.predict(Xs)
        return pd.Series(
            [self.lmap_.get(k,"UNKNOWN") if self.lmap_ else str(k) for k in ks],
            index=feat.index
        )

    def current_regime(self, df, close_panel=None):
        r = self.predict(df, close_panel=close_panel)
        return r.iloc[-1] if len(r) else "BULL_TREND"

    def regime_table(self, pnl, regime_series):
        m    = pnl[["net_ret"]].join(regime_series.rename("regime"), how="inner")
        rows = []
        for reg, g in m.groupby("regime"):
            r = g["net_ret"]
            rows.append({
                "regime":reg, "n":len(r),
                "mean_ret%":round(r.mean()*100,4),
                "win_rate%":round((r>0).mean()*100,1),
                "sharpe":round(r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0, 3)
            })
        return pd.DataFrame(rows)

    def save(self, path):
        with open(path,"wb") as f:
            pickle.dump({"gmm":self.gmm,"scaler":self.scaler,"lmap":self.lmap_,
                         "n_feats":getattr(self,"_n_feats",5)}, f)
        print(f"[Regime] Saved -> {path}")

    def load(self, path):
        with open(path,"rb") as f: o = pickle.load(f)
        self.gmm=o["gmm"]; self.scaler=o["scaler"]
        self.lmap_=o.get("lmap"); self._n_feats=o.get("n_feats",5)

# ─────────────────────────────────────────────────────────────────────────────
#  ANOMALY DETECTOR  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyDetector:
    """
    IsolationForest: top 2% most unusual bars flagged as -1 (rest = +1).
    Uses cuML IsolationForest on GPU (RAPIDS) if available, else sklearn CPU.
    Install RAPIDS for GPU: https://rapids.ai/install
    """
    def __init__(self, contamination=0.02):
        if _CUML_IF:
            # cuML: no random_state or n_jobs (GPU-managed internally)
            self.ifo = IsolationForest(contamination=contamination)
            logger.info("[AnomalyDetector] cuML IsolationForest (GPU)")
        else:
            self.ifo = IsolationForest(contamination=contamination,
                                       random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self._fb    = FeatureBuilder()

    def train(self, data, max_sym=None):  # None = ALL symbols
        Xs = []
        keys = list(data.keys()) if max_sym is None else list(data.keys())[:max_sym]
        for sym in keys:
            feat = self._fb.build(data[sym])
            X    = feat[FeatureBuilder.COLS].dropna().values
            if len(X) > 0: Xs.append(X)
        if not Xs: raise ValueError("No data")
        X_all = np.vstack(Xs)
        self.ifo.fit(self.scaler.fit_transform(X_all))
        print(f"[Anomaly] Trained on {len(X_all):,} bars")

    def predict(self, df):
        feat = self._fb.build(df)
        X    = feat[FeatureBuilder.COLS].dropna()
        return pd.Series(self.ifo.predict(self.scaler.transform(X.values)), index=X.index)

    def score(self, df):
        feat = self._fb.build(df)
        X    = feat[FeatureBuilder.COLS].dropna()
        return pd.Series(self.ifo.score_samples(self.scaler.transform(X.values)), index=X.index)

    def save(self, path):
        with open(path,"wb") as f: pickle.dump({"ifo":self.ifo,"scaler":self.scaler}, f)
        print(f"[Anomaly] Saved -> {path}")

    def load(self, path):
        with open(path,"rb") as f: o = pickle.load(f)
        self.ifo=o["ifo"]; self.scaler=o["scaler"]

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Azalyst ML v3 — Fast Training")
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--out-dir",     default="./azalyst_models")
    parser.add_argument("--model",       default="all",
                        choices=["all","pump","return","regime","anomaly"])
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="Max symbols (0=ALL)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max rows for ML training (0=ALL data, no subsampling)")
    parser.add_argument("--live",        action="store_true")
    args = parser.parse_args()

    # Allow CLI override of subsample size  (0 = no limit)
    global MAX_SAMPLES_PUMP, MAX_SAMPLES_RETURN
    MAX_SAMPLES_PUMP   = args.max_samples
    MAX_SAMPLES_RETURN = args.max_samples

    # Convert 0 → None so [:None] = all
    max_sym_cli = None if args.max_symbols == 0 else args.max_symbols

    os.makedirs(args.out_dir, exist_ok=True)

    from azalyst_engine import DataLoader
    loader = DataLoader(args.data_dir, max_symbols=max_sym_cli, workers=4)
    data   = loader.load_all()
    if not data: print("[ML] No data loaded"); return
    sample_label = "ALL" if args.max_samples == 0 else f"{args.max_samples:,}"
    print(f"\n[ML] Loaded {len(data)} symbols  (cap: {'ALL' if max_sym_cli is None else max_sym_cli})")
    device_label = {"cuda": "NVIDIA CUDA ⚡", "gpu": "GPU (OpenCL)", "cpu": "CPU"}.get(DEVICE, DEVICE)
    print(f"[ML] Device: {device_label}  |  Folds: {N_FOLDS}  |  Max samples: {sample_label}\n")

    results = {}

    if args.model in ("all","pump"):
        print("="*55+"\n  PUMP / DUMP DETECTOR\n"+"="*55)
        m = PumpDumpDetector()
        r = m.train(data, max_sym=max_sym_cli)
        m.save(os.path.join(args.out_dir,"pump_dump_model.pkl"))
        results["pump"] = r
        if hasattr(m,"importances_"):
            print("\nTop 10 features:"); print(m.importances_.head(10).to_string())
            m.importances_.to_csv(os.path.join(args.out_dir,"pump_feature_importance.csv"))

    if args.model in ("all","return"):
        print("\n"+"="*55+"\n  RETURN PREDICTOR\n"+"="*55)
        m = ReturnPredictor()
        r = m.train(data, max_sym=max_sym_cli)
        m.save(os.path.join(args.out_dir,"return_predictor.pkl"))
        results["return"] = r

    if args.model in ("all","regime"):
        print("\n"+"="*55+"\n  REGIME DETECTOR\n"+"="*55)
        ref = next((k for k in data if "BTC" in k), list(data.keys())[0])
        m   = RegimeDetector(); m.train(data[ref])
        m.save(os.path.join(args.out_dir,"regime_detector.pkl"))
        print(f"  Current regime: {m.current_regime(data[ref])}")

    if args.model in ("all","anomaly"):
        print("\n"+"="*55+"\n  ANOMALY DETECTOR\n"+"="*55)
        m = AnomalyDetector(); m.train(data, max_sym=max_sym_cli)
        m.save(os.path.join(args.out_dir,"anomaly_detector.pkl"))

    if args.live:
        print("\n"+"="*55+"\n  LIVE ML INFERENCE\n"+"="*55)
        p_path = os.path.join(args.out_dir,"pump_dump_model.pkl")
        r_path = os.path.join(args.out_dir,"return_predictor.pkl")
        g_path = os.path.join(args.out_dir,"regime_detector.pkl")
        rows   = []
        for sym in list(data.keys())[:50]:
            row = {"symbol": sym}
            if os.path.exists(p_path):
                pm = PumpDumpDetector(); pm.load(p_path)
                try: row["pump_prob"] = round(float(pm.predict(data[sym]).iloc[-1]),4)
                except: row["pump_prob"] = 0.0
            if os.path.exists(r_path):
                rm = ReturnPredictor(); rm.load(r_path)
                try: row["up_prob"] = round(float(rm.predict_proba(data[sym]).iloc[-1]),4)
                except: row["up_prob"] = 0.5
            rows.append(row)
        live = pd.DataFrame(rows).sort_values("up_prob", ascending=False)
        print("\n  TOP 20 by ML up_prob:"); print(live.head(20).to_string(index=False))
        if os.path.exists(g_path):
            gm  = RegimeDetector(); gm.load(g_path)
            ref = next((k for k in data if "BTC" in k), list(data.keys())[0])
            print(f"\n  Current regime ({ref}): {gm.current_regime(data[ref])}")
        out = os.path.join(args.out_dir,"ml_live_scores.csv")
        live.to_csv(out, index=False); print(f"\n[Saved] -> {out}")

    if results:
        print("\n"+"="*55+"\n  TRAINING SUMMARY")
        for k,v in results.items():
            print(f"  {k:<15} Mean AUC = {v.get('mean_auc',0):.4f}")
    print("\n[ML] Done.")

if __name__ == "__main__":
    main()
