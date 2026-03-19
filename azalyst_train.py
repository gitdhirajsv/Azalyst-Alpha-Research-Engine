"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    TRAINING MODULE v2
║        XGBoost GPU (cuda:0)  |  Purged K-Fold CV                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from scipy import stats


def compute_ic(y_pred, y_true):
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 10:
        return 0.0
    return float(stats.spearmanr(y_pred[mask], y_true[mask])[0])


class PurgedTimeSeriesCV:
    """
    Purged K-Fold as defined in the final v2 snippet.
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


def _probe_gpu() -> bool:
    """
    Check whether XGBoost CUDA is actually available on this machine.
    Returns True if cuda:0 works, False otherwise.
    This prevents crashes on CPU-only environments or when GPU is not
    assigned even though --gpu flag was passed.
    """
    try:
        _x = np.random.rand(20, 5).astype(np.float32)
        _y = np.array([0] * 10 + [1] * 10)
        xgb.XGBClassifier(
            device="cuda:0", n_estimators=2, verbosity=0
        ).fit(_x, _y)
        return True
    except Exception:
        return False


def make_xgb_model(use_gpu=True):
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


def train_model(X, y, y_ret, feature_cols, label="", use_gpu=True):
    """
    Final v2 training function.

    GPU is probed at the start of every call — if cuda:0 is unavailable
    it silently falls back to CPU rather than crashing the whole pipeline.
    """
    # FIX: always verify GPU before using it — don't trust the flag blindly.
    # Gemini's version hard-coded cuda:0 and never checked, crashing on CPU.
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


# ─────────────────────────────────────────────────────────────────────────────
#  META-LABELING  (Lopez de Prado, AFML Ch. 3)
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_model(primary_model, primary_scaler, X, y, feature_cols,
                     label="meta", use_gpu=True):
    """
    Train a second-stage model that predicts P(primary model is correct).

    Uses purged CV to generate honest out-of-sample predictions from
    *temporary* primary models (one per fold), avoiding information leakage.
    The meta-model learns when the primary signal is trustworthy.
    Output is used for position sizing in the weekly loop.
    """
    if use_gpu:
        if _probe_gpu():
            print(f"[{label}] GPU: CUDA confirmed")
        else:
            use_gpu = False
            print(f"[{label}] GPU: falling back to CPU")

    Xs = primary_scaler.transform(X)

    # Collect OOS predictions from purged CV (same splits as primary)
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

    # Meta-label: 1 if CV-predicted direction matches actual alpha_label
    meta_y = ((oos_preds[valid] >= 0.5).astype(float) == y[valid]).astype(float)

    # Augmented features: scaled X + primary OOS probability
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
