"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    IC-BASED FEATURE FILTER v1
║        Feature Importance via IC  |  Remove Noise  |  Improve Signal        ║
║        v1.0  |  Preprocessing module  |  Pre-training optimization          ║
╚══════════════════════════════════════════════════════════════════════════════╝

IC-Based Feature Filtering:
  This module filters features before training based on their individual IC
  (Information Coefficient) correlation with targets.
  
  Benefits:
  - Removes noisy features that reduce signal strength  
  - Speeds up training with fewer features
  - Improves generalization by reducing overfitting
  - More robust predictions
  
  Strategy:
  - Compute IC for each feature against target returns
  - Keep features with |IC| > ic_threshold (default 0.005)
  - Compute ICIR (IC / IC_std) for feature ranking
  - Optional: weight features by ICIR * cross-sectional rank correlation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Optional


def compute_feature_ic(feature: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Spearman rank IC (Information Coefficient) between a feature and target.
    IC = correlation(rank(feature), rank(target))
    
    Args:
        feature: Feature values (1D array)
        target: Target values (1D array, typically returns)
        
    Returns:
        IC value in [-1, 1]
    """
    mask = np.isfinite(feature) & np.isfinite(target)
    if mask.sum() < 10:
        return 0.0
    
    try:
        ic, _ = stats.spearmanr(feature[mask], target[mask])
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        return 0.0


def compute_feature_ic_series(X: np.ndarray, y_ret: np.ndarray, 
                               feature_names: List[str],
                               rolling_window: Optional[int] = None) -> pd.Series:
    """
    Compute IC for each feature across entire dataset or rolling windows.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y_ret: Target returns (n_samples,)
        feature_names: List of feature names
        rolling_window: If provided, compute time-varying IC and take median
        
    Returns:
        Series of IC values indexed by feature name
    """
    n_samples, n_features = X.shape
    ic_values = np.zeros(n_features)
    
    if rolling_window is None:
        # Single IC calculation for entire dataset
        for j in range(n_features):
            ic_values[j] = compute_feature_ic(X[:, j], y_ret)
    else:
        # Rolling IC calculation (more stable estimate)
        rolling_ics = []
        for start in range(0, n_samples - rolling_window + 1, rolling_window // 2):
            end = start + rolling_window
            for j in range(n_features):
                ic = compute_feature_ic(X[start:end, j], y_ret[start:end])
                rolling_ics.append((j, ic))
        
        # Aggregate rolling ICs (median is robust to outliers)
        for j in range(n_features):
            ics_for_j = [ic for feat_idx, ic in rolling_ics if feat_idx == j]
            if ics_for_j:
                ic_values[j] = float(np.median(ics_for_j))
    
    return pd.Series(ic_values, index=feature_names, name="IC")


def compute_icir(ic_series: pd.Series, rolling_window: Optional[int] = None) -> pd.Series:
    """
    Compute IC Information Ratio (ICIR) = IC / std(IC).
    Ranks features by stability of their predictive power.
    
    Args:
        ic_series: Series of IC values
        rolling_window: If provided, compute time-varying ICIR
        
    Returns:
        Series of ICIR values (higher = more stable predictor)
    """
    ic_std = ic_series.std()
    if ic_std < 1e-10:
        return pd.Series(0.0, index=ic_series.index, name="ICIR")
    
    return ic_series / ic_std


def filter_features_by_ic(X: np.ndarray, y_ret: np.ndarray,
                          feature_names: List[str],
                          ic_threshold: float = 0.005,
                          min_features: int = 20,
                          verbose: bool = True) -> Tuple[np.ndarray, List[str], pd.Series]:
    """
    Filter features based on IC threshold.
    
    Strategy:
    1. Compute IC for each feature
    2. Keep features with |IC| >= ic_threshold
    3. Ensure at least min_features are retained (take top by ICIR if needed)
    4. Return filtered feature matrix and indices
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y_ret: Target returns (n_samples,)
        feature_names: List of feature names
        ic_threshold: Minimum absolute IC to keep feature (default 0.5%)
        min_features: Minimum features to retain (fallback)
        verbose: Print filtering statistics
        
    Returns:
        (X_filtered, selected_feature_names, ic_series)
    """
    # Compute IC for each feature
    ic_series = compute_feature_ic_series(X, y_ret, feature_names)
    icir_series = compute_icir(ic_series)
    
    # Features that meet IC threshold
    selected_mask = ic_series.abs() >= ic_threshold
    selected_features = ic_series[selected_mask].sort_values(key=abs, ascending=False)
    
    # Fallback: if too few features pass threshold, take top by ICIR
    if selected_mask.sum() < min_features:
        selected_features = ice_series.abs().nlargest(min_features)
        selected_mask = pd.Series(False, index=ic_series.index)
        selected_mask[selected_features.index] = True
    
    selected_names = selected_features.index.tolist()
    selected_indices = np.array([feature_names.index(name) for name in selected_names])
    X_filtered = X[:, selected_indices]
    
    if verbose:
        print(f"\n{'─' * 80}")
        print(f"  IC-BASED FEATURE FILTERING")
        print(f"{'─' * 80}")
        print(f"  Original features      : {len(feature_names)}")
        print(f"  Features passing IC    : {selected_mask.sum()} (threshold={ic_threshold:.4f})")
        print(f"  Filtered features      : {len(selected_names)}")
        print(f"  Compression ratio      : {len(selected_names)/len(feature_names)*100:.1f}%")
        print(f"\n  Top 10 features by |IC|:")
        for name, ic_val in selected_features.head(10).items():
            icir = icir_series[name]
            print(f"    {name:25s}  IC={ic_val:+.6f}  ICIR={icir:+.3f}")
        
        bottom = selected_features.tail(5)
        print(f"\n  Bottom 5 retained features by IC:")
        for name, ic_val in bottom.items():
            icir = icir_series[name]
            print(f"    {name:25s}  IC={ic_val:+.6f}  ICIR={icir:+.3f}")
        print(f"{'─' * 80}\n")
    
    return X_filtered, selected_names, ic_series


def rank_features_by_ic(X: np.ndarray, y_ret: np.ndarray,
                        feature_names: List[str],
                        top_n: int = 20,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Rank all features by their IC and ICIR.
    Useful for understanding which features drive predictions.
    
    Args:
        X: Feature matrix
        y_ret: Target returns
        feature_names: Feature names
        top_n: Number of top/bottom features to display
        verbose: Print results
        
    Returns:
        DataFrame with IC, ICIR, and abs(IC) rankings
    """
    ic_series = compute_feature_ic_series(X, y_ret, feature_names)
    icir_series = compute_icir(ic_series)
    
    ranking = pd.DataFrame({
        "IC": ic_series,
        "ICIR": icir_series,
        "abs_IC": ic_series.abs(),
    }).sort_values("abs_IC", ascending=False)
    
    if verbose:
        print(f"\n{'─' * 80}")
        print(f"  FEATURE IC RANKING (Top {top_n})")
        print(f"{'─' * 80}")
        for idx, (name, row) in enumerate(ranking.head(top_n).iterrows(), 1):
            print(f"  {idx:2d}. {name:25s}  IC={row['IC']:+.6f}  ICIR={row['ICIR']:+.3f}")
        
        print(f"\n  FEATURE IC RANKING (Bottom {top_n})")
        for idx, (name, row) in enumerate(ranking.tail(top_n).iterrows(), 1):
            print(f"  {len(ranking)-top_n+idx:2d}. {name:25s}  IC={row['IC']:+.6f}  ICIR={row['ICIR']:+.3f}")
        print(f"{'─' * 80}\n")
    
    return ranking


def get_feature_weights_by_ic(ic_series: pd.Series) -> pd.Series:
    """
    Convert IC values to feature weights for weighted training.
    Positive IC features get upweighted, negative IC features downweighted.
    
    Weight formula: w_i = (1 + sign(IC_i) * |IC_i|)^2
    This ensures:
    - Features with |IC| = 0 get weight 1.0
    - Features with IC = 0.01 get weight ≈ 1.0004
    - Features with IC = 0.1 get weight ≈ 1.04
    - Features with IC = -0.1 get weight ≈ 0.96
    
    Args:
        ic_series: Series of IC values
        
    Returns:
        Series of feature weights
    """
    # Clamp IC to [-0.5, 0.5] to avoid extreme weights
    ic_clipped = ic_series.clip(-0.5, 0.5)
    
    # Weight formula
    weights = (1.0 + np.sign(ic_clipped) * ic_clipped.abs()) ** 2
    
    # Normalize to mean 1.0 (so total sample weight unchanged)
    weights = weights / weights.mean()
    
    return weights
