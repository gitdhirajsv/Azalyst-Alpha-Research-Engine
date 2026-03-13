"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    RISK MANAGEMENT MODULE
║        Portfolio Optimization · Risk Constraints · VaR/CVaR Calculation    ║
║        v1.0  |  SciPy + NumPy  |  Institutional Grade Risk Controls        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Optional, Tuple, Union

class RiskManager:
    """
    Handles portfolio-level risk, optimization, and metric calculation.
    """

    def __init__(self, entry_fee: float = 0.001, exit_fee: float = 0.001):
        self.entry_fee = entry_fee
        self.exit_fee = exit_fee

    # ── Portfolio Optimization ────────────────────────────────────────────────

    def compute_mvo_weights(self, 
                            returns_df: pd.DataFrame, 
                            target_return: Optional[float] = None,
                            risk_free_rate: float = 0.0) -> pd.Series:
        """
        Mean-Variance Optimization (Sharpe Maximization).
        """
        mu = returns_df.mean()
        sigma = returns_df.cov()
        n = len(mu)
        
        if n == 0:
            return pd.Series()

        def objective(w):
            port_ret = np.sum(mu * w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            if port_vol == 0: return 0
            return -(port_ret - risk_free_rate) / port_vol

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 0.25) for _ in range(n)]  # Max 25% per asset constraint
        
        init_w = np.array([1.0/n] * n)
        res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not res.success:
            # Fallback to equal weight if optimization fails
            return pd.Series([1.0/n]*n, index=returns_df.columns)
            
        return pd.Series(res.x, index=returns_df.columns)

    def compute_hrp_weights(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Hierarchical Risk Parity (HRP) using cluster linkage.
        Lopez de Prado (2016)
        """
        corr = returns_df.corr().fillna(0)
        cov = returns_df.cov().fillna(0)
        
        # 1. Quasi-Diagonalization
        # Distance matrix: dist = sqrt(0.5*(1-corr))
        d_mat = np.sqrt(0.5 * (1 - corr))
        link = linkage(squareform(d_mat), method='single')
        sort_ix = leaves_list(link)
        sorted_symbols = returns_df.columns[sort_ix].tolist()
        
        # 2. Recursive Bisection
        weights = pd.Series(1.0, index=sorted_symbols)
        
        def get_cluster_var(cov, cluster_items):
            cluster_cov = cov.loc[cluster_items, cluster_items]
            ivp = 1.0 / np.diag(cluster_cov)
            ivp /= ivp.sum()
            return np.dot(ivp.T, np.dot(cluster_cov, ivp))

        def bisection(items):
            if len(items) <= 1: return
            
            mid = len(items) // 2
            items_l = items[:mid]
            items_r = items[mid:]
            
            var_l = get_cluster_var(cov, items_l)
            var_r = get_cluster_var(cov, items_r)
            
            alpha = 1 - var_l / (var_l + var_r)
            weights[items_l] *= alpha
            weights[items_r] *= (1 - alpha)
            
            bisection(items_l)
            bisection(items_r)

        bisection(sorted_symbols)
        return weights.reindex(returns_df.columns)

    # ── Institutional Framework: Black-Litterman ──────────────────────────────

    def black_litterman(self, 
                        mu_prior: pd.Series, 
                        cov: pd.DataFrame, 
                        P: np.ndarray, 
                        Q: np.ndarray, 
                        tau: float = 0.05, 
                        Omega: Optional[np.ndarray] = None) -> pd.Series:
        """
        Black-Litterman model to combine market priors with investor views.
        mu_prior: Equilibrium returns (prior)
        P: Picker matrix (K views x N assets)
        Q: Vector of views (K x 1)
        tau: Scalar indicating uncertainty of prior
        Omega: Uncertainty of views matrix (K x K)
        """
        N = len(mu_prior)
        if Omega is None:
            # Idzorek's method or simplified: diag(P * (tau * cov) * P.T)
            Omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov, P.T))))
            
        # BL Master Formula for expected returns
        term1 = np.linalg.inv(np.linalg.inv(tau * cov) + np.dot(P.T, np.dot(np.linalg.inv(Omega), P)))
        term2 = np.dot(np.linalg.inv(tau * cov), mu_prior) + np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        
        mu_bl = np.dot(term1, term2)
        return pd.Series(mu_bl, index=mu_prior.index)

    # ── Risk Metrics ──────────────────────────────────────────────────────────

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Value at Risk (VaR)."""
        if returns.empty: return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Expected Shortfall (Conditional VaR)."""
        if returns.empty: return 0.0
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    # ── Constraint Engine ─────────────────────────────────────────────────────

    def apply_constraints(self, weights: pd.Series, 
                         max_weight: float = 0.2, 
                         min_weight: float = 0.0) -> pd.Series:
        """Enforces concentration limits."""
        w = weights.clip(min_weight, max_weight)
        return w / w.sum()

if __name__ == "__main__":
    # Quick sanity check
    rm = RiskManager()
    data = pd.DataFrame(np.random.normal(0.001, 0.02, (1000, 10)), columns=[f"asset_{i}" for i in range(10)])
    mvo = rm.compute_mvo_weights(data)
    hrp = rm.compute_hrp_weights(data)
    print("MVO Weights Sum:", mvo.sum())
    print("HRP Weights Sum:", hrp.sum())
    print("VaR (95%):", rm.calculate_var(data.iloc[:, 0]))
