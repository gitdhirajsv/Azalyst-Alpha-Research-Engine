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
    if mask.sum() < 10: return 0.0
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
            train_end   = (i + 1) * fold_size
            val_start   = train_end + self.gap      
            val_end     = val_start + fold_size
            if val_end > n:
                break
            train_idx = np.arange(0, train_end)
            val_idx   = np.arange(val_start, val_end)
            yield train_idx, val_idx

def make_xgb_model(use_gpu=True):
    params = dict(
        n_estimators    = 1000,
        learning_rate   = 0.02,
        max_depth       = 6,
        min_child_weight= 30,
        subsample       = 0.8,
        colsample_bytree= 0.7,
        colsample_bylevel=0.7,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        eval_metric     = 'auc',
        early_stopping_rounds = 50,
        verbosity       = 0,
        random_state    = 42,
    )
    if use_gpu:
        params['device'] = 'cuda:0'
    return xgb.XGBClassifier(**params)

def train_model(X, y, y_ret, feature_cols, label='', use_gpu=True):
    """
    Final v2 training function.
    """
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
        except: pass

        if y_ret is not None:
            ic = compute_ic(probs, y_ret[val])
            ics.append(ic)

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    mean_ic  = float(np.mean(ics))  if ics  else 0.0
    icir     = float(np.mean(ics) / (np.std(ics) + 1e-8)) if ics else 0.0
    
    final = make_xgb_model(use_gpu)
    split = int(len(Xs) * 0.9)
    final.fit(Xs[:split], y[:split], eval_set=[(Xs[split:], y[split:])], verbose=False)

    importance = pd.Series(
        final.feature_importances_,
        index=feature_cols, name='importance'
    ).sort_values(ascending=False)

    return final, scaler, importance, mean_auc, mean_ic, icir
