"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    MACHINE LEARNING MODULE v2          
║        Return Prediction · Pump Detection · Regime Classification            ║
║        v2.0  |  XGBoost CUDA  |  Purged K-Fold  |  RobustScaler              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os, pickle, warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

# FIX: was "from azalyst_factors_v2 import compute_v2_features, FEATURE_COLS"
# compute_v2_features does not exist — the correct function is build_features
from azalyst_factors_v2 import build_features, FEATURE_COLS

warnings.filterwarnings("ignore")


def _detect_gpu() -> str:
    """
    Auto-detect CUDA availability and return the correct device string.
    Returns 'cuda:0' if CUDA works, 'cpu' otherwise.
    This prevents hard crashes on CPU-only Kaggle sessions.
    """
    if not _XGB:
        return "cpu"
    try:
        _x = np.random.rand(20, 5).astype(np.float32)
        _y = np.array([0] * 10 + [1] * 10)
        xgb.XGBClassifier(
            device="cuda:0", n_estimators=2, verbosity=0
        ).fit(_x, _y)
        return "cuda:0"
    except Exception:
        return "cpu"


class ReturnPredictorV2:
    """
    v2 Return Predictor using XGBoost and 65 features.
    GPU is auto-detected — won't crash on CPU-only environments.
    """

    def __init__(self, device: str = "cuda:0"):
        # Always validate device at construction time
        self.device = _detect_gpu() if device == "cuda:0" else device
        self.model = None
        self.scaler = RobustScaler()

    def train(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        params = {
            "tree_method": "hist",
            "device": self.device,
            "max_bin": 128,
            "learning_rate": 0.02,
            "max_depth": 6,
            "min_child_weight": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(Xs, label=y)
        self.model = xgb.train(params, dtrain, num_boost_round=1000)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        return self.model.predict(xgb.DMatrix(Xs))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "model_binary": self.model.save_raw()}, f
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.model = xgb.Booster()
            self.model.load_model(data["model_binary"])
