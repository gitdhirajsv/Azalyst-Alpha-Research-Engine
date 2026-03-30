"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    MACHINE LEARNING MODULE v5          
║        Return Prediction · Pump Detection · Regime Classification            ║
║        v5.0  |  XGBoost Regression  |  Purged K-Fold  |  RobustScaler        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os, pickle, warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler
from scipy import stats

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

from azalyst_factors_v2 import build_features, FEATURE_COLS

warnings.filterwarnings("ignore")


def _detect_gpu() -> str:
    """
    Auto-detect CUDA availability for regression mode.
    Returns 'cuda:0' if CUDA works, 'cpu' otherwise.
    """
    if not _XGB:
        return "cpu"
    try:
        _x = np.random.rand(20, 5).astype(np.float32)
        _y = np.random.randn(20).astype(np.float32)
        xgb.XGBRegressor(
            device="cuda:0", n_estimators=2, verbosity=0
        ).fit(_x, _y)
        return "cuda:0"
    except Exception:
        return "cpu"


class ReturnPredictorV2:
    """
    v5 Return Predictor using XGBoost Regression.
    Predicts continuous forward returns (not binary classification).
    GPU is auto-detected — won't crash on CPU-only environments.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = _detect_gpu() if device == "cuda:0" else device
        self.model = None
        self.scaler = RobustScaler()

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train regression model. y should be continuous forward returns."""
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
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(Xs, label=y)
        self.model = xgb.train(params, dtrain, num_boost_round=1000)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous returns."""
        Xs = self.scaler.transform(X).astype(np.float32)
        return self.model.predict(xgb.DMatrix(Xs))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Legacy alias for predict() — returns continuous predictions."""
        return self.predict(X)

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
