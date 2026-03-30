"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    TRAINING MODULE v5
║        XGBoost Regression  |  Purged K-Fold  |  Weighted R² + IC            ║
║        v5.0  |  CUDA GPU  |  Short-Horizon Forecasting (15min/1hr)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Key changes from v4 (lessons learned from audit):
  1. REGRESSION not classification — predict continuous returns
  2. Weighted R² metric (Jane Street competition winner technique)
  3. IC + ICIR as primary validation metrics
  4. Pump-dump aware: option to exclude pump-dump bars from training
  5. Per-bar prediction — no week-averaging
  6. Confidence model predicts P(direction correct) for sizing
"""

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from scipy import stats


# ── Purged Time-Series CV ────────────────────────────────────────────────────

class PurgedTimeSeriesCV:
    """
    Lopez de Prado (2018) purged walk-forward CV.
    Prevents look-ahead bias from autocorrelated features.
    gap = embargo period in bars (default 48 = 4 hours at 5-min frequency).
    """
    def __init__(self, n_splits=5, gap=48):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            val_start = train_end + self.gap
            val_end = val_start + fold_size
            if val_end > n:
                break
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_ic(y_pred, y_true):
    """Spearman rank IC between prediction and actual."""
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 10:
        return 0.0
    return float(stats.spearmanr(y_pred[mask], y_true[mask])[0])


def weighted_r2_score(y_true, y_pred, weights=None):
    """
    Weighted R² metric (Jane Street competition metric).
    R² = 1 - Σw(y-ŷ)² / Σw(y-ȳ)²
    """
    if weights is None:
        weights = np.ones_like(y_true)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weights)
    if mask.sum() < 10:
        return 0.0
    y_t, y_p, w = y_true[mask], y_pred[mask], weights[mask]
    y_bar = np.average(y_t, weights=w)
    ss_res = np.sum(w * (y_t - y_p) ** 2)
    ss_tot = np.sum(w * (y_t - y_bar) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ── GPU Detection ────────────────────────────────────────────────────────────

def _probe_gpu():
    """Check whether XGBoost CUDA is actually available."""
    try:
        _x = np.random.rand(20, 5).astype(np.float32)
        _y = np.array([0] * 10 + [1] * 10)
        xgb.XGBClassifier(
            device="cuda:0", n_estimators=2, verbosity=0
        ).fit(_x, _y)
        return True
    except Exception:
        return False


def _probe_gpu_regression():
    """Check whether XGBoost CUDA regression works."""
    try:
        _x = np.random.rand(20, 5).astype(np.float32)
        _y = np.random.randn(20).astype(np.float32)
        xgb.XGBRegressor(
            device="cuda:0", n_estimators=2, verbosity=0
        ).fit(_x, _y)
        return True
    except Exception:
        return False


# ── Model Factories ──────────────────────────────────────────────────────────

def make_xgb_regressor(use_gpu=True, n_estimators=1000, max_depth=6,
                       min_child_weight=30):
    """
    v5 XGBRegressor for continuous return prediction.
    Key difference from v4: regression (squarederror) instead of classification.
    """
    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.02,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbosity=0,
        random_state=42,
    )
    if use_gpu:
        params["device"] = "cuda:0"
    return xgb.XGBRegressor(**params)


def make_xgb_model(use_gpu=True):
    """Legacy v4 classifier factory — kept for backward compat."""
    params = dict(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
        random_state=42,
    )
    if use_gpu:
        params["device"] = "cuda:0"
    return xgb.XGBClassifier(**params)


# ── v5 Regression Training Pipeline ─────────────────────────────────────────

def train_regression_model(X, y_ret, feature_cols, label="", use_gpu=True):
    """
    v5 regression training pipeline.

    Target: continuous forward return (NOT binary label).
    Metrics: Weighted R² + IC + ICIR (NOT AUC).
    Model: XGBRegressor (NOT XGBClassifier).

    Returns: (model, scaler, importance, mean_r2, mean_ic, icir)
    """
    if use_gpu:
        if _probe_gpu_regression():
            print(f"[{label}] GPU: CUDA confirmed — using cuda:0 (regression)")
        else:
            use_gpu = False
            print(f"[{label}] GPU: CUDA unavailable — falling back to CPU")

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    r2s, ics = [], []

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        m = make_xgb_regressor(use_gpu)
        m.fit(Xs[tr], y_ret[tr], eval_set=[(Xs[val], y_ret[val])], verbose=False)

        preds = m.predict(Xs[val])
        r2 = weighted_r2_score(y_ret[val], preds)
        r2s.append(r2)

        ic = compute_ic(preds, y_ret[val])
        if np.isfinite(ic):
            ics.append(ic)

    mean_r2 = float(np.mean(r2s)) if r2s else 0.0
    mean_ic = float(np.mean(ics)) if ics else 0.0
    icir = float(np.mean(ics) / (np.std(ics) + 1e-8)) if len(ics) > 1 else 0.0

    # Final model on 90/10 split
    final = make_xgb_regressor(use_gpu)
    split = int(len(Xs) * 0.9)
    final.fit(
        Xs[:split], y_ret[:split],
        eval_set=[(Xs[split:], y_ret[split:])],
        verbose=100,
    )

    importance = pd.Series(
        final.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    return final, scaler, importance, mean_r2, mean_ic, icir


# ── v5 Confidence Model — replaces meta-labeling ────────────────────────────

def train_confidence_model(base_model, base_scaler, X, y_ret, feature_cols,
                           label="confidence", use_gpu=True):
    """
    v5 confidence model (replaces meta-labeling from AFML Ch. 3):
    - Predicts P(base model direction is correct)
    - Uses OOS regression predictions from base model as additional feature
    - Output scales position size: high confidence → full size

    Returns: (confidence_model, confidence_scaler) or (None, None)
    """
    if use_gpu:
        if _probe_gpu():
            print(f"[{label}] GPU: CUDA confirmed")
        else:
            use_gpu = False
            print(f"[{label}] GPU: falling back to CPU")

    Xs = base_scaler.transform(X).astype(np.float32)

    # Collect OOS predictions from purged CV
    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    oos_preds = np.full(len(y_ret), np.nan, dtype=np.float32)

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        m_temp = make_xgb_regressor(use_gpu, n_estimators=500, max_depth=4)
        m_temp.fit(Xs[tr], y_ret[tr], eval_set=[(Xs[val], y_ret[val])],
                   verbose=False)
        oos_preds[val] = m_temp.predict(Xs[val])

    valid = np.isfinite(oos_preds)
    if valid.sum() < 200:
        print(f"  [{label}] Insufficient OOS data ({valid.sum()} rows) — skipping")
        return None, None

    # Target: was the base model's direction correct?
    direction_correct = (
        np.sign(oos_preds[valid]) == np.sign(y_ret[valid])
    ).astype(np.float32)

    # Augmented features: scaled X + base OOS prediction
    X_conf = np.column_stack([Xs[valid], oos_preds[valid]])
    conf_scaler = RobustScaler()
    X_conf_s = conf_scaler.fit_transform(X_conf).astype(np.float32)

    conf = make_xgb_model(use_gpu)
    conf.set_params(n_estimators=500, max_depth=4, min_child_weight=50)

    split = int(len(X_conf_s) * 0.9)
    conf.fit(
        X_conf_s[:split], direction_correct[:split],
        eval_set=[(X_conf_s[split:], direction_correct[split:])],
        verbose=False,
    )

    val_acc = float(
        (conf.predict(X_conf_s[split:]) == direction_correct[split:]).mean()
    )
    print(f"  [{label}] Direction accuracy: {val_acc*100:.1f}% on "
          f"{valid.sum():,} OOS rows")
    return conf, conf_scaler


# ── Legacy v4 Classification Training (backward compat) ─────────────────────

def train_model(X, y, y_ret, feature_cols, label="", use_gpu=True):
    """
    Legacy v4 binary classification training — kept for backward compat.
    The v5 engine uses train_regression_model() instead.
    """
    if use_gpu:
        if _probe_gpu():
            print(f"[{label}] GPU: CUDA confirmed — using cuda:0")
        else:
            use_gpu = False
            print(f"[{label}] GPU: CUDA unavailable — falling back to CPU")

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    aucs, ics = [], []

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        if len(np.unique(y[val])) < 2:
            continue
        m = make_xgb_model(use_gpu)
        m.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)

        probs = m.predict_proba(Xs[val])[:, 1]
        try:
            auc = roc_auc_score(y[val], probs)
            aucs.append(auc)
        except Exception:
            pass

        if y_ret is not None:
            ic = compute_ic(probs, y_ret[val])
            ics.append(ic)

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    mean_ic = float(np.mean(ics)) if ics else 0.0
    icir = float(np.mean(ics) / (np.std(ics) + 1e-8)) if ics else 0.0

    final = make_xgb_model(use_gpu)
    split = int(len(Xs) * 0.9)
    final.fit(
        Xs[:split], y[:split],
        eval_set=[(Xs[split:], y[split:])],
        verbose=False,
    )

    importance = pd.Series(
        final.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    return final, scaler, importance, mean_auc, mean_ic, icir


def train_meta_model(primary_model, primary_scaler, X, y, feature_cols,
                     label="meta", use_gpu=True):
    """Legacy v4 meta-labeling — kept for backward compat."""
    if use_gpu:
        if _probe_gpu():
            print(f"[{label}] GPU: CUDA confirmed")
        else:
            use_gpu = False
            print(f"[{label}] GPU: falling back to CPU")

    Xs = primary_scaler.transform(X)

    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
    oos_preds = np.full(len(y), np.nan, dtype=np.float32)

    for fold, (tr, val) in enumerate(cv.split(Xs), 1):
        if len(np.unique(y[val])) < 2:
            continue
        m_temp = make_xgb_model(use_gpu)
        m_temp.fit(Xs[tr], y[tr], eval_set=[(Xs[val], y[val])], verbose=False)
        oos_preds[val] = m_temp.predict_proba(Xs[val])[:, 1]

    valid = np.isfinite(oos_preds)
    if valid.sum() < 200:
        print(f"  [{label}] Insufficient OOS data ({valid.sum()} rows) — skipping meta")
        return None, None

    meta_y = ((oos_preds[valid] >= 0.5).astype(float) == y[valid]).astype(float)

    X_meta = np.column_stack([Xs[valid], oos_preds[valid]])
    meta_scaler = RobustScaler()
    X_meta_s = meta_scaler.fit_transform(X_meta)

    meta = make_xgb_model(use_gpu)
    meta.set_params(n_estimators=500, max_depth=4, min_child_weight=50)

    split = int(len(X_meta_s) * 0.9)
    meta.fit(
        X_meta_s[:split], meta_y[:split],
        eval_set=[(X_meta_s[split:], meta_y[split:])],
        verbose=False,
    )

    val_acc = float((meta.predict(X_meta_s[split:]) == meta_y[split:]).mean())
    print(f"  [{label}] Meta-model: accuracy={val_acc*100:.1f}% on {valid.sum():,} OOS rows")
    return meta, meta_scaler
