"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  YEAR 1+2 TRAINING  (RTX 2050 local GPU version)
╠══════════════════════════════════════════════════════════════════════════════╣
║  Changes for local RTX 2050 (4GB VRAM):                                   ║
║   - XGBoost instead of LightGBM (confirmed CUDA working)                  ║
║   - VRAM guard: caps training rows at 4M (safe for 4GB)                   ║
║   - max_bin=128 (LightGBM uses 255 which is too heavy for 2050)            ║
║   - Stratified time-series subsample (preserves temporal distribution)     ║
║   - Batch feature loading (avoids 16GB RAM spike)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python azalyst_train.py --feature-dir ./feature_cache --out-dir ./results --gpu
    python azalyst_train.py --feature-dir ./feature_cache --out-dir ./results  (CPU)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
    print("[WARN] xgboost not installed. Run: pip install xgboost")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BARS_PER_DAY = 288

# RTX 2050 4GB VRAM safe limit
# 4M rows × 65 features × 4 bytes = 1.04GB + XGBoost histogram overhead (~2GB)
# Total ~3.5GB — leaves 0.5GB headroom
MAX_TRAIN_ROWS_GPU = 4_000_000
MAX_TRAIN_ROWS_CPU = 8_000_000   # CPU uses RAM, not VRAM — can go higher

FEATURE_COLS = [
    # Returns (7)
    "ret_1bar", "ret_1h", "ret_4h", "ret_1d", "ret_2d", "ret_3d", "ret_1w",
    # Volume (6)
    "vol_ratio", "vol_ret_1h", "vol_ret_1d", "obv_change", "vpt_change", "vol_momentum",
    # Volatility (7)
    "rvol_1h", "rvol_4h", "rvol_1d", "vol_ratio_1h_1d",
    "atr_norm", "parkinson_vol", "garman_klass",
    # Technical (10)
    "rsi_14", "rsi_6", "macd_hist", "bb_pos", "bb_width",
    "stoch_k", "stoch_d", "cci_14", "adx_14", "dmi_diff",
    # Microstructure (6)
    "vwap_dev", "amihud", "kyle_lambda", "spread_proxy", "body_ratio", "candle_dir",
    # Price structure (6)
    "wick_top", "wick_bot", "price_accel", "skew_1d", "kurt_1d", "max_ret_4h",
    # WorldQuant alphas (8)
    "wq_alpha001", "wq_alpha012", "wq_alpha031", "wq_alpha098",
    "cs_momentum", "cs_reversal", "vol_adjusted_mom", "trend_consistency",
    # Regime (5)
    "vol_regime", "trend_strength", "corr_btc_proxy", "hurst_exp", "fft_strength",
]

# Fall back to v1 features if v2 cache not built yet
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
#  GPU DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_xgb_gpu() -> bool:
    """Test XGBoost CUDA on RTX 2050."""
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
        print("  XGBoost CUDA: RTX 2050 ready ✓")
        return True
    except Exception as e:
        print(f"  XGBoost CUDA failed: {e} — using CPU")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  VRAM-SAFE SUBSAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def subsample_for_vram(X: np.ndarray, y: np.ndarray,
                       use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cap rows to fit RTX 2050 4GB VRAM.
    Preserves temporal ordering and class balance.
    """
    max_rows = MAX_TRAIN_ROWS_GPU if use_gpu else MAX_TRAIN_ROWS_CPU
    n = len(X)
    if n <= max_rows:
        return X, y

    print(f"  [VRAM guard] {n:,} rows → {max_rows:,} rows")

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
#  DATE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_date_range(feature_dir: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts, ends = [], []
    for f in sorted(feature_dir.glob("*.parquet"))[:80]:
        try:
            idx = pd.to_datetime(pd.read_parquet(f, columns=[]).index, utc=True)
            if len(idx) > BARS_PER_DAY * 30:
                starts.append(idx.min()); ends.append(idx.max())
        except Exception:
            pass
    if not starts:
        raise RuntimeError(f"No valid parquet files in {feature_dir}")
    return min(starts), max(ends)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING  (memory-efficient for 16GB RAM)
# ─────────────────────────────────────────────────────────────────────────────

def load_data_for_window(feature_dir: Path, date_from: pd.Timestamp,
                         date_to: pd.Timestamp, resample_freq: str = "4h",
                         verbose: bool = True) -> pd.DataFrame:
    """
    Load feature cache for date window.
    Uses float32 throughout to halve RAM usage (important with 16GB).
    """
    frames = []
    files  = sorted(feature_dir.glob("*.parquet"))

    for i, f in enumerate(files, 1):
        try:
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[(df.index >= date_from) & (df.index < date_to)]
            if len(df) < 10:
                continue

            # Detect available features (v2 or v1 fallback)
            avail_v2 = [c for c in FEATURE_COLS if c in df.columns]
            avail_v1 = [c for c in FEATURE_COLS_V1 if c in df.columns]
            avail = avail_v2 if len(avail_v2) > len(avail_v1) else avail_v1

            if len(avail) < 15:
                continue

            df_rs = df.resample(resample_freq).last().dropna(
                subset=avail, how="all"
            )
            if len(df_rs) < 5:
                continue

            # Cast to float32 immediately — halves RAM (critical on 16GB)
            for col in avail:
                if col in df_rs.columns:
                    df_rs[col] = df_rs[col].astype("float32")

            df_rs["symbol"] = f.stem
            frames.append(df_rs)
        except Exception:
            pass

        if verbose and i % 30 == 0:
            print(f"    {i}/{len(files)} scanned, {len(frames)} valid")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    if verbose:
        mb = combined.memory_usage(deep=True).sum() / 1e6
        print(f"    Panel: {len(combined):,} rows, "
              f"{combined['symbol'].nunique()} symbols  ({mb:.0f}MB RAM)")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-SECTIONAL LABEL + RANKING
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional alpha label: 1 = outperforms universe median."""
    if "future_ret_4h" not in df.columns:
        raise ValueError("'future_ret_4h' missing. Run build_feature_cache.py first.")
    df = df.copy()
    df["alpha_label"] = np.nan
    median_by_ts = df.groupby(level=0)["future_ret_4h"].transform("median")
    df["alpha_label"] = np.where(
        df["future_ret_4h"].notna(),
        (df["future_ret_4h"] > median_by_ts).astype("float32"),
        np.nan,
    )
    return df


def cross_sectional_rank(df: pd.DataFrame,
                          cols: List[str] = None) -> pd.DataFrame:
    """Vectorized cross-sectional percentile rank — no Python loops."""
    cols  = cols or FEATURE_COLS_V1
    avail = [c for c in cols if c in df.columns]
    ranked = df.groupby(level=0, sort=False)[avail].rank(
        pct=True, na_option="keep")
    df = df.copy()
    df[avail] = ranked
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  PURGED K-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────

class PurgedKFold:
    """
    Purged time-series cross-validation with embargo.
    gap=48 bars prevents the model from seeing data adjacent to validation.
    Prevents leakage from overlapping return windows.
    """
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
#  XGBOOST TRAINING  (RTX 2050 tuned)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X: np.ndarray, y: np.ndarray, feature_cols: List[str],
                  use_gpu: bool = False, label: str = "") -> tuple:
    """
    Train XGBoost with Purged K-Fold CV.
    Tuned specifically for RTX 2050 4GB VRAM.
    """
    from sklearn.metrics import roc_auc_score
    from scipy import stats

    scaler = RobustScaler()   # better than StandardScaler for fat-tailed crypto
    Xs     = scaler.fit_transform(X).astype(np.float32)

    # VRAM guard
    Xs_sub, y_sub = subsample_for_vram(Xs, y, use_gpu)
    print(f"  Training [{label}]: {len(Xs_sub):,} samples, "
          f"{X.shape[1]} features, {'GPU (RTX 2050)' if use_gpu else 'CPU (i5-11260H)'}")

    # XGBoost params
    if use_gpu:
        params = {
            "tree_method":      "hist",
            "device":           "cuda",
            "max_bin":          128,       # 4GB VRAM safe
            "learning_rate":    0.02,
            "max_depth":        6,
            "min_child_weight": 30,
            "subsample":        0.8,
            "colsample_bytree": 0.7,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "verbosity":        0,
            "seed":             42,
        }
    else:
        params = {
            "tree_method":      "hist",
            "device":           "cpu",
            "max_bin":          256,
            "learning_rate":    0.02,
            "max_depth":        6,
            "min_child_weight": 30,
            "subsample":        0.8,
            "colsample_bytree": 0.7,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "verbosity":        0,
            "seed":             42,
            "nthread":          6,         # i5-11260H: 6 cores
        }

    cv   = PurgedKFold(n_splits=5, gap=48)
    aucs = []
    ics  = []

    for fold, (tr, val) in enumerate(cv.split(Xs_sub), 1):
        if len(np.unique(y_sub[val])) < 2:
            continue
        dtrain = xgb.DMatrix(Xs_sub[tr], label=y_sub[tr])
        dval   = xgb.DMatrix(Xs_sub[val], label=y_sub[val])
        try:
            model = xgb.train(
                params, dtrain,
                num_boost_round=1000,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            proba = model.predict(dval)
            auc   = roc_auc_score(y_sub[val], proba)
            ic    = float(stats.spearmanr(proba, y_sub[val])[0])
            aucs.append(auc); ics.append(ic)
            print(f"    Fold {fold}: AUC={auc:.4f}  IC={ic:.4f}  "
                  f"trees={model.best_iteration}")
        except Exception as e:
            print(f"    Fold {fold}: error — {e}")

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    mean_ic  = float(np.mean(ics))  if ics  else 0.0
    icir     = float(np.mean(ics) / np.std(ics)) if len(ics) > 1 and np.std(ics) > 0 else 0.0
    print(f"  CV  AUC={mean_auc:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}")

    # Final model on full subsampled data
    split  = int(len(Xs_sub) * 0.9)
    dtrain = xgb.DMatrix(Xs_sub[:split], label=y_sub[:split])
    dval   = xgb.DMatrix(Xs_sub[split:], label=y_sub[split:])
    final  = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Feature importance
    importance = pd.Series(
        final.get_score(importance_type="gain"),
        name="importance"
    ).reindex(feature_cols, fill_value=0).sort_values(ascending=False)

    return final, scaler, importance, mean_auc, mean_ic, icir


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Year 1+2 Training — RTX 2050 local GPU"
    )
    parser.add_argument("--feature-dir",  default="./feature_cache")
    parser.add_argument("--out-dir",      default="./results")
    parser.add_argument("--gpu",          action="store_true",
                        help="Use RTX 2050 CUDA")
    parser.add_argument("--year12-days",  type=int, default=730,
                        help="Days of data for Year 1+2 training (default 730 = 2 years)")
    parser.add_argument("--resample",     default="4h")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir     = Path(args.out_dir)
    models_dir  = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("    AZALYST — YEAR 1+2 TRAINING  (RTX 2050 4GB VRAM)")
    print("╚══════════════════════════════════════════════════════════════╝")

    # GPU check
    use_gpu = False
    if args.gpu:
        print("\n[1/6] Detecting RTX 2050 CUDA...")
        use_gpu = detect_xgb_gpu()
    else:
        print("\n[1/6] CPU mode (i5-11260H 6 cores)")

    print(f"\n[2/6] Discovering date range...")
    global_start, global_end = discover_date_range(feature_dir)
    year12_end = global_start + pd.Timedelta(days=args.year12_days)
    year3_end  = global_end
    print(f"  Full range  : {global_start.date()} → {global_end.date()}")
    print(f"  Train (Y1+2): {global_start.date()} → {year12_end.date()}")
    print(f"  Test  (Y3)  : {year12_end.date()} → {year3_end.date()}")
    print(f"  Resample    : {args.resample}")

    date_config = {
        "global_start": global_start.isoformat(),
        "global_end":   global_end.isoformat(),
        "year12_end":   year12_end.isoformat(),
        "year3_end":    year3_end.isoformat(),
    }
    with open(out_dir / "date_config.json", "w") as fh:
        json.dump(date_config, fh, indent=2)

    print(f"\n[3/6] Loading Year 1+2 data...")
    df = load_data_for_window(feature_dir, global_start, year12_end,
                              resample_freq=args.resample)
    if df.empty:
        print("[ERROR] No data loaded."); return

    print(f"\n[4/6] Building cross-sectional alpha labels...")
    df    = build_alpha_labels(df)
    avail = [c for c in FEATURE_COLS if c in df.columns]
    if len(avail) < 20:
        avail = [c for c in FEATURE_COLS_V1 if c in df.columns]
        print(f"  Using v1 features ({len(avail)} cols) — rebuild cache for v2")
    else:
        print(f"  Using v2 features ({len(avail)} cols)")

    valid = df.dropna(subset=avail + ["alpha_label"])
    print(f"  Valid labelled rows: {len(valid):,}")
    print(f"  Alpha rate         : {valid['alpha_label'].mean()*100:.1f}%")

    print(f"\n[5/6] Cross-sectional ranking + training...")
    valid = cross_sectional_rank(valid, avail)
    X     = valid[avail].values.astype(np.float32)
    y     = valid["alpha_label"].values.astype(float)

    model, scaler, importance, cv_auc, cv_ic, icir = train_xgboost(
        X, y, avail, use_gpu=use_gpu, label="Year1+2"
    )

    print(f"\n[6/6] Saving artefacts...")
    model_path = models_dir / "model_base_y1y2.json"
    model.save_model(str(model_path))

    # Save scaler + meta together
    meta_path = models_dir / "model_base_y1y2_meta.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump({
            "scaler":        scaler,
            "feature_cols":  avail,
            "year12_end":    year12_end.isoformat(),
            "n_train_rows":  int(len(X)),
            "cv_auc":        round(cv_auc, 4),
            "cv_ic":         round(cv_ic, 4),
            "icir":          round(icir, 4),
            "resample":      args.resample,
            "gpu_used":      use_gpu,
        }, fh)

    importance.to_csv(out_dir / "feature_importance_base.csv", header=True)
    print(f"  Model → {model_path}")
    print(f"  Meta  → {meta_path}")

    print("\n  Top 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:<30}  {imp:>8.1f}")

    summary = {
        "year12_end":       year12_end.isoformat(),
        "year3_start":      year12_end.isoformat(),
        "year3_end":        year3_end.isoformat(),
        "n_symbols":        int(df["symbol"].nunique()),
        "n_train_rows":     int(len(X)),
        "n_features":       len(avail),
        "alpha_rate_pct":   round(float(y.mean()) * 100, 2),
        "cv_auc":           round(cv_auc, 4),
        "cv_ic":            round(cv_ic, 4),
        "icir":             round(icir, 4),
        "elapsed_min":      round((time.time() - t0) / 60, 2),
        "resample":         args.resample,
        "gpu_used":         use_gpu,
        "hardware":         "RTX 2050 4GB" if use_gpu else "i5-11260H CPU",
    }
    with open(out_dir / "train_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Training complete in {(time.time()-t0)/60:.1f} min")
    print(f"\n  Next step:")
    print(f"    python azalyst_weekly_loop.py \\")
    print(f"        --feature-dir {feature_dir} \\")
    print(f"        --results-dir {out_dir} {'--gpu' if use_gpu else ''}")


if __name__ == "__main__":
    main()
