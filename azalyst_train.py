"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  YEAR 1 INITIAL TRAINING  (no LLM, pure quant)
╠══════════════════════════════════════════════════════════════════════════════╣
║  FIX: resample param now forwarded to load_data_for_window() and           ║
║  compute_features() so bar-count windows always match the candle TF.      ║
║  Training still uses 5-min data resampled to 4H (unchanged behaviour).    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python azalyst_train.py --feature-dir ./feature_cache --out-dir ./results
    python azalyst_train.py --feature-dir ./feature_cache --out-dir ./results --gpu
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGBM = True
except ImportError:
    _LGBM = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  (5-min defaults — training always uses 5-min data)
# ─────────────────────────────────────────────────────────────────────────────
BARS_PER_DAY = 288

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
#  IMPORT FIXED HELPERS FROM build_feature_cache
# ─────────────────────────────────────────────────────────────────────────────
from build_feature_cache import get_tf_constants, compute_features, compute_targets


# ─────────────────────────────────────────────────────────────────────────────
#  DATE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_date_range(feature_dir: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts, ends = [], []
    for f in sorted(feature_dir.glob("*USDT.parquet"))[:60]:
        try:
            idx = pd.to_datetime(pd.read_parquet(f, columns=[]).index, utc=True)
            if len(idx) > BARS_PER_DAY * 30:
                starts.append(idx.min()); ends.append(idx.max())
        except Exception:
            pass
    if not starts:
        for f in sorted(feature_dir.glob("*.parquet"))[:60]:
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
#  DATA LOADING  (FIX: resample_freq passed through)
# ─────────────────────────────────────────────────────────────────────────────

def load_data_for_window(
    feature_dir: Path,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    resample_freq: str = "4h",   # FIX: was implicitly always 4h
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all symbol parquets for a date window.

    FIX: resample_freq is now explicit. The feature cache stores pre-computed
    5-min features; we resample the *already-correct* features to resample_freq
    (defaulting to 4H as before).  If the cache was built with a different TF,
    pass that TF here so windows match.
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
            avail_feats = [c for c in FEATURE_COLS if c in df.columns]
            if len(avail_feats) < len(FEATURE_COLS) - 3:
                continue
            df_rs = df.resample(resample_freq).last().dropna(
                subset=avail_feats, how="all"
            )
            if len(df_rs) < 5:
                continue
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
        print(f"    Panel: {len(combined):,} rows, {combined['symbol'].nunique()} symbols")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-SECTIONAL LABEL + RANKING  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "future_ret_4h" not in df.columns:
        raise ValueError("'future_ret_4h' column missing. Run build_feature_cache.py first.")
    df = df.copy()
    df["alpha_label"] = np.nan
    for ts, group in df.groupby(level=0):
        fwd = group["future_ret_4h"].dropna()
        if len(fwd) < 3:
            continue
        median = fwd.median()
        df.loc[group.index, "alpha_label"] = np.where(
            group["future_ret_4h"].notna(),
            (group["future_ret_4h"] > median).astype(float),
            np.nan,
        )
    return df


def cross_sectional_rank(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    cols  = cols or FEATURE_COLS
    df    = df.copy()
    avail = [c for c in cols if c in df.columns]
    for ts, group in df.groupby(level=0):
        if len(group) < 2:
            continue
        df.loc[group.index, avail] = group[avail].rank(pct=True).values
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _lgbm_params(use_gpu: bool = False) -> dict:
    p = {
        "objective": "binary", "metric": "auc",
        "n_estimators": 1000, "learning_rate": 0.03,
        "num_leaves": 127 if use_gpu else 63,
        "max_bin": 255, "min_child_samples": 20,
        "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "class_weight": "balanced",
        "random_state": 42, "verbose": -1,
    }
    if use_gpu:
        p["device"] = "cuda"; p["gpu_use_dp"] = False; p["n_jobs"] = 1
    else:
        p["n_jobs"] = -1
    return p


def train_model(X, y, feature_cols, use_gpu=False, n_cv_folds=3, purge_bars=48, label=""):
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    print(f"  Training {label}: {len(X):,} samples, {X.shape[1]} features, "
          f"{'GPU' if use_gpu else 'CPU'}")
    aucs   = []
    splits = TimeSeriesSplit(n_splits=n_cv_folds, gap=purge_bars)
    for fold, (tr, val) in enumerate(splits.split(Xs), 1):
        if len(np.unique(y[val])) < 2:
            continue
        if _LGBM:
            m = lgb.LGBMClassifier(**_lgbm_params(use_gpu))
            m.fit(Xs[tr], y[tr],
                  eval_set=[(Xs[val], y[val])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=4, random_state=42)
            m.fit(Xs[tr], y[tr])
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y[val], m.predict_proba(Xs[val])[:, 1])
            aucs.append(auc)
            trees = (f"  trees={m.best_iteration_}"
                     if _LGBM and hasattr(m, "best_iteration_") else "")
            print(f"    Fold {fold}: AUC = {auc:.4f}{trees}")
        except Exception:
            pass

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    print(f"  CV Mean AUC: {mean_auc:.4f}  "
          f"[{'SIGNAL ✓' if mean_auc > 0.53 else 'WEAK'}]")

    if _LGBM:
        final = lgb.LGBMClassifier(**_lgbm_params(use_gpu))
        split = int(len(Xs) * 0.9)
        final.fit(Xs[:split], y[:split],
                  eval_set=[(Xs[split:], y[split:])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        final = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=4, random_state=42)
        final.fit(Xs, y)

    importance = pd.Series(
        getattr(final, "feature_importances_", np.zeros(len(feature_cols))),
        index=feature_cols, name="importance",
    ).sort_values(ascending=False)
    return final, scaler, importance, mean_auc


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azalyst Year 1 Training")
    parser.add_argument("--feature-dir", default="./feature_cache")
    parser.add_argument("--out-dir",     default="./results")
    parser.add_argument("--gpu",         action="store_true")
    parser.add_argument("--year1-days",  type=int, default=365)
    # FIX: explicit resample param (default unchanged = 4h)
    parser.add_argument("--resample",    default="4h",
                        help="Resample freq for training panel (default: 4h)")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir     = Path(args.out_dir)
    models_dir  = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("    AZALYST  —  YEAR 1 TRAINING  (FIXED: timeframe-aware)")
    print("╚══════════════════════════════════════════════════════════════╝")

    print("\n[1/6] Discovering date range...")
    global_start, global_end = discover_date_range(feature_dir)
    year1_end = global_start + pd.Timedelta(days=args.year1_days)
    year2_end = year1_end    + pd.Timedelta(days=365)
    year3_end = global_end
    print(f"  Data range : {global_start.date()} → {global_end.date()}")
    print(f"  Year 1     : {global_start.date()} → {year1_end.date()}")
    print(f"  Resample   : {args.resample}")

    date_config = {
        "global_start": global_start.isoformat(), "global_end": global_end.isoformat(),
        "year1_start":  global_start.isoformat(), "year1_end":  year1_end.isoformat(),
        "year2_end":    year2_end.isoformat(),     "year3_end":  year3_end.isoformat(),
    }
    with open(out_dir / "date_config.json", "w") as fh:
        json.dump(date_config, fh, indent=2)

    print(f"\n[2/6] Loading Year 1 data (resample={args.resample})...")
    df = load_data_for_window(feature_dir, global_start, year1_end,
                              resample_freq=args.resample)   # FIX: explicit
    if df.empty:
        print("[ERROR] No data loaded."); return

    print("\n[3/6] Building cross-sectional alpha labels...")
    df = build_alpha_labels(df)
    valid = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns] + ["alpha_label"])
    print(f"  Valid labelled rows : {len(valid):,}")
    print(f"  Alpha rate          : {valid['alpha_label'].mean()*100:.1f}%")

    print("\n[4/6] Cross-sectional feature ranking...")
    valid     = cross_sectional_rank(valid)
    avail     = [c for c in FEATURE_COLS if c in valid.columns]
    X = valid[avail].values.astype(np.float32)
    y = valid["alpha_label"].values.astype(int)

    print("\n[5/6] Training LightGBM...")
    model, scaler, importance, cv_auc = train_model(X, y, avail, use_gpu=args.gpu,
                                                    label="Year1")

    print("\n[6/6] Saving artefacts...")
    model_path = models_dir / "model_year1.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({
            "model": model, "scaler": scaler, "feature_cols": avail,
            "year1_end": year1_end.isoformat(), "n_train_rows": int(len(X)),
            "cv_auc": round(cv_auc, 4), "label": "year1_initial",
            "resample": args.resample,   # FIX: store resample so loop can match
        }, fh)
    print(f"  Model → {model_path}")

    importance.to_csv(out_dir / "feature_importance_year1.csv", header=True)
    print("\n  Top 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:<25}  {imp:>8.1f}")

    summary = {
        "year1_end": year1_end.isoformat(),
        "n_symbols": int(df["symbol"].nunique() if "symbol" in df.columns else 0),
        "n_train_rows": int(len(X)), "n_features": len(avail),
        "alpha_rate_pct": round(float(y.mean()) * 100, 2),
        "cv_auc": round(cv_auc, 4),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "resample": args.resample,
        "gpu": args.gpu,
    }
    with open(out_dir / "train_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Training complete in {(time.time()-t0)/60:.1f} min.")
    print(f"\n  Next step:")
    print(f"    python azalyst_weekly_loop.py --feature-dir {feature_dir} --results-dir {out_dir}")


if __name__ == "__main__":
    main()
