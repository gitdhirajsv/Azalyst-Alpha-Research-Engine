"""
IC-based feature filtering utilities for the current v6 training flow.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_feature_ic(x_col: np.ndarray, y_ret: np.ndarray) -> float:
    mask = np.isfinite(x_col) & np.isfinite(y_ret)
    if mask.sum() < 50:
        return 0.0
    ic, _ = spearmanr(x_col[mask], y_ret[mask])
    return float(ic) if np.isfinite(ic) else 0.0


def compute_icir(
    X: np.ndarray,
    y_ret: np.ndarray,
    feature_names: List[str],
    n_windows: int = 10,
    min_window_size: int = 500,
) -> pd.Series:
    """
    Compute ICIR = mean(IC_t) / std(IC_t) across rolling windows.
    """
    n_samples, n_features = X.shape
    window_size = max(min_window_size, n_samples // n_windows)
    if window_size * 2 > n_samples:
        return pd.Series(0.0, index=feature_names, name="ICIR")

    ic_matrix = np.zeros((n_features, n_windows))
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n_samples)
        if end - start < 50:
            continue
        for j in range(n_features):
            ic_matrix[j, w] = compute_feature_ic(X[start:end, j], y_ret[start:end])

    ic_mean = ic_matrix.mean(axis=1)
    ic_std = ic_matrix.std(axis=1)
    icir = np.where(ic_std > 1e-10, ic_mean / ic_std, 0.0)
    return pd.Series(icir, index=feature_names, name="ICIR")


def filter_features_by_ic(
    X: np.ndarray,
    y_ret: np.ndarray,
    feature_names: List[str],
    ic_threshold: float = 0.005,
    min_features: int = 20,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], dict]:
    ic_vals = [compute_feature_ic(X[:, j], y_ret) for j in range(X.shape[1])]
    ic_series = pd.Series(ic_vals, index=feature_names, name="IC")
    icir_series = compute_icir(X, y_ret, feature_names)

    selected_mask = ic_series.abs() >= float(ic_threshold)
    selected_features = ic_series[selected_mask].sort_values(key=abs, ascending=False)

    if selected_mask.sum() < min_features:
        selected_features = ic_series.abs().nlargest(min_features)
        selected_features = ic_series.loc[selected_features.index].sort_values(
            key=abs,
            ascending=False,
        )
        selected_mask = pd.Series(False, index=ic_series.index)
        selected_mask[selected_features.index] = True

    selected_names = selected_features.index.tolist()
    selected_idx = np.array([feature_names.index(name) for name in selected_names], dtype=int)
    X_filtered = X[:, selected_idx]

    info = {
        "n_total": int(len(feature_names)),
        "n_selected": int(len(selected_names)),
        "mean_abs_ic_selected": float(ic_series[selected_names].abs().mean()) if selected_names else 0.0,
        "mean_icir_selected": float(icir_series[selected_names].mean()) if selected_names else 0.0,
        "selected_features": selected_names,
    }
    if verbose:
        print(f"  [IC FILTER] selected {info['n_selected']}/{info['n_total']} features")
    return X_filtered, selected_names, info


def rank_features_by_ic(
    X: np.ndarray,
    y_ret: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    ic_vals = [compute_feature_ic(X[:, j], y_ret) for j in range(X.shape[1])]
    ic_series = pd.Series(ic_vals, index=feature_names, name="IC")
    icir_series = compute_icir(X, y_ret, feature_names)
    out = pd.concat([ic_series, icir_series], axis=1)
    out["abs_ic"] = out["IC"].abs()
    return out.sort_values("abs_ic", ascending=False)
