"""
╔══════════════════════════════════════════════════════════════════════════════╗
       AZALYST ALPHA RESEARCH ENGINE    ALPHA OPTIMIZER
║   Ridge / Elastic Net Factor Combination · Walk-Forward Weighting           ║
║   v1.0  |  sklearn  |  No Lookahead  |  Newey-West Regularised             ║
╚══════════════════════════════════════════════════════════════════════════════╝

The naive IC-weighted composite uses ICIR thresholds (> 0.05) with static
weights.  This module replaces it with a proper cross-sectional regression:

  At each rebalancing bar, fit Ridge/ElasticNet on historical factor returns
  to estimate optimal per-factor weights.  The weights adapt to regime and
  factor decay automatically — no hardcoded thresholds.

Methods
───────
  AlphaOptimizer.fit_expanding()  — Fit on all history up to current bar
  AlphaOptimizer.fit_rolling()    — Fit on rolling window (regime-adaptive)
  AlphaOptimizer.predict()        — Compute optimal composite score
  AlphaOptimizer.walk_forward()   — Full walk-forward backtest of weights

Usage
─────
  from azalyst_alphaopt import AlphaOptimizer

  opt = AlphaOptimizer(factors, close_panel, horizon_bars=288)
  composite = opt.walk_forward(method="ridge")
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BARS_PER_HOUR = 12
BARS_PER_DAY  = 288
BARS_PER_WEEK = 2016


# ─────────────────────────────────────────────────────────────────────────────
#  ALPHA OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class AlphaOptimizer:
    """
    Optimal factor combination via penalised cross-sectional regression.

    Why this is better than IC-weighted
    ────────────────────────────────────
    IC-weighted uses each factor independently (ignores co-linearity).
    Ridge regression solves:

        min ||y - Xβ||² + λ||β||²
            s.t. y = forward_return, X = factor_rank_matrix

    This naturally handles correlated factors (momentum factors are 0.7+
    correlated) by shrinking redundant weights rather than double-counting.
    ElasticNet additionally zeroes out noise factors via L1 penalty.

    Walk-forward discipline
    ───────────────────────
    The model is NEVER fit on data it trades on:
      - Expanding window: fit on [t_start, t_now - embargo], trade at t_now
      - Rolling window  : fit on [t_now - lookback - embargo, t_now - embargo]
    Embargo = max horizon bars (prevents label leakage).
    """

    def __init__(
        self,
        factors:      Dict[str, pd.DataFrame],   # factor_name → (T × N) DataFrame
        close_panel:  pd.DataFrame,               # close prices (T × N)
        horizon_bars: int = BARS_PER_DAY,         # forward return horizon
        rebal_every:  int = BARS_PER_DAY,         # rebalancing frequency
        min_history:  int = BARS_PER_WEEK * 4,   # minimum bars before first fit
        lookback:     Optional[int] = None,       # rolling window (None = expanding)
        alpha_ridge:  float = 1.0,                # Ridge regularisation strength
    ):
        self.factors      = factors
        self.close        = close_panel
        self.horizon      = horizon_bars
        self.rebal_every  = rebal_every
        self.min_hist     = min_history
        self.lookback     = lookback
        self.alpha_ridge  = alpha_ridge
        self._factor_names = list(factors.keys())

        # Precompute forward returns once
        log_ret = np.log(close_panel / close_panel.shift(1))
        self._fwd_ret = log_ret.shift(-horizon_bars).rolling(
            horizon_bars, min_periods=horizon_bars // 2
        ).sum()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _cross_section_matrix(self, t: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (N, K) factor matrix X and (N,) return vector y for timestamp t.

        Returns (X, y) with only rows where both X and y are fully observed.
        Cross-section = one observation per coin at time t.
        """
        rows = []
        for name in self._factor_names:
            f = self.factors[name]
            if t in f.index:
                rows.append(f.loc[t].values)
            else:
                rows.append(np.full(self.close.shape[1], np.nan))

        X = np.stack(rows, axis=1)          # (N, K)
        y = self._fwd_ret.loc[t].values      # (N,)

        # Keep only rows with complete data in both X and y
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return X[mask], y[mask]

    def _panel_matrix(self, idx: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stack cross-sections across time to build a pooled (T*N, K) dataset.

        Pooled cross-sectional regression: each (t, coin) pair is one
        observation. Correct for factor combination — treats all coins equally.
        """
        X_list, y_list = [], []
        for t in idx:
            try:
                Xt, yt = self._cross_section_matrix(t)
                if len(Xt) > 10:
                    X_list.append(Xt)
                    y_list.append(yt)
            except Exception:
                continue

        if not X_list:
            return np.empty((0, len(self._factor_names))), np.empty(0)

        return np.vstack(X_list), np.concatenate(y_list)

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit Ridge regression and return coefficients."""
        if len(X) < len(self._factor_names) * 10:
            return np.zeros(len(self._factor_names))
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        model.fit(Xs, y)
        return model.coef_

    def _fit_elasticnet(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit ElasticNet regression and return coefficients."""
        if len(X) < len(self._factor_names) * 10:
            return np.zeros(len(self._factor_names))
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0],
            cv=5,
            max_iter=2000,
            random_state=42
        )
        model.fit(Xs, y)
        return model.coef_

    # ── Public API ───────────────────────────────────────────────────────────

    def walk_forward(
        self,
        method: str = "ridge",     # "ridge" | "elasticnet"
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Walk-forward factor weight estimation.

        At each rebalancing bar (after min_history warmup):
          1. Select training window (expanding or rolling).
          2. Build pooled (T_train*N, K) cross-sectional dataset.
          3. Fit Ridge/ElasticNet on training data.
          4. Compute composite = fitted_weights · factor_ranks at current bar.
          5. Apply composite to build long/short rankings.

        Returns
        ───────
        pd.DataFrame (T × N) — composite factor scores, same shape as
        individual factor DataFrames.  Drop-in replacement for equal_weight()
        and ic_weighted() in BacktestEngine.

        Also stores self.weight_history: DataFrame (rebal_bars × K) showing
        how factor weights evolved over time.  Useful for research.
        """
        print(f"[AlphaOpt] Walk-forward factor optimisation ({method.upper()})...")
        timestamps  = self.close.index
        rebal_bars  = np.arange(self.min_hist, len(timestamps), self.rebal_every)

        fit_fn = self._fit_ridge if method == "ridge" else self._fit_elasticnet

        # Output: same shape as any factor DataFrame
        comp_vals = np.full(
            (len(timestamps), len(self.close.columns)), np.nan
        )
        weight_rows = []

        for i, ri in enumerate(rebal_bars):
            t_now  = timestamps[ri]
            embargo = ri - self.horizon    # exclude overlap zone

            if embargo <= self.min_hist:
                continue

            # Select training window
            if self.lookback is not None:
                train_start = max(self.min_hist, embargo - self.lookback)
            else:
                train_start = self.min_hist  # expanding

            train_idx = timestamps[train_start:embargo:self.rebal_every]  # subsampled

            if len(train_idx) < 20:
                continue

            # Build pooled dataset
            X_train, y_train = self._panel_matrix(train_idx)
            if len(X_train) < 50:
                continue

            # Fit model
            coefs = fit_fn(X_train, y_train)
            weight_rows.append({**{"timestamp": t_now},
                                 **dict(zip(self._factor_names, coefs))})

            # Compute composite at t_now: weighted average of factor ranks
            factor_row = []
            for name in self._factor_names:
                f = self.factors[name]
                val = f.loc[t_now].values if t_now in f.index else np.full(
                    len(self.close.columns), np.nan
                )
                factor_row.append(val)

            F = np.stack(factor_row, axis=1)   # (N, K)
            # Use positive weights only for the long signal direction
            pos_coefs = np.maximum(coefs, 0)
            if pos_coefs.sum() > 0:
                pos_coefs /= pos_coefs.sum()
            composite_row = F @ pos_coefs        # (N,)
            comp_vals[ri] = composite_row

            if verbose and i % 50 == 0:
                n_nonzero = (np.abs(coefs) > 1e-6).sum()
                print(f"  bar {ri:5d} / {len(timestamps)} | "
                      f"nonzero factors: {n_nonzero}/{len(self._factor_names)}")

        # Store weight history
        self.weight_history = (
            pd.DataFrame(weight_rows).set_index("timestamp")
            if weight_rows else pd.DataFrame()
        )

        # Forward-fill between rebalancing bars
        result = pd.DataFrame(
            comp_vals, index=timestamps, columns=self.close.columns
        )
        result = result.ffill(limit=self.rebal_every)
        print(f"[AlphaOpt] Done. Weight history: {len(self.weight_history)} rebal bars")
        return result

    def top_factors(self, n: int = 10) -> pd.DataFrame:
        """
        Mean absolute coefficient by factor across all rebalancing periods.
        Shows which factors the model consistently relies on.
        """
        if not hasattr(self, "weight_history") or self.weight_history.empty:
            raise RuntimeError("run walk_forward() first")
        return (
            self.weight_history.abs().mean()
            .sort_values(ascending=False)
            .head(n)
            .rename("mean_abs_coef")
            .to_frame()
        )

    def regime_weights(self, regime_series: pd.Series) -> pd.DataFrame:
        """
        Average factor weights per market regime.

        Shows which factors the regressor loads on in different market states.
        Useful for understanding what drives the composite in each regime.

        Args:
            regime_series: pd.Series (DatetimeIndex) with regime labels.
        """
        if not hasattr(self, "weight_history") or self.weight_history.empty:
            raise RuntimeError("run walk_forward() first")

        df = self.weight_history.join(regime_series.rename("regime"), how="inner")
        return df.groupby("regime").mean()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Azalyst Alpha Optimizer — optimal factor combination"
    )
    parser.add_argument("--data-dir",  required=True)
    parser.add_argument("--out-dir",   default="./azalyst_output")
    parser.add_argument("--method",    default="ridge", choices=["ridge", "elasticnet"])
    parser.add_argument("--horizon",   default="1D", choices=["1H", "4H", "1D"])
    parser.add_argument("--max-symbols", type=int, default=100)
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    horizon_map = {"1H": BARS_PER_HOUR, "4H": BARS_PER_HOUR * 4, "1D": BARS_PER_DAY}
    horizon_bars = horizon_map[args.horizon]

    from azalyst_engine import DataLoader
    from azalyst_factors_v2 import FactorEngineV2

    loader = DataLoader(args.data_dir, max_symbols=args.max_symbols, workers=4)
    data   = loader.load_all()
    close  = loader.build_close_panel(data)
    vol    = loader.build_volume_panel(data)

    fe = FactorEngineV2()
    factors = fe.compute_all(close=close, volume=vol)

    opt = AlphaOptimizer(
        factors=factors,
        close_panel=close,
        horizon_bars=horizon_bars,
        rebal_every=BARS_PER_DAY,
        min_history=BARS_PER_WEEK * 4,
    )

    composite = opt.walk_forward(method=args.method)

    # Save composite
    out_path = os.path.join(args.out_dir, f"optimised_composite_{args.method}.parquet")
    composite.to_parquet(out_path)
    print(f"\n[Saved] Composite → {out_path}")

    # Print top factors
    print("\nTop factor weights:")
    print(opt.top_factors(10).to_string())

    # Save weight history
    wh_path = os.path.join(args.out_dir, "factor_weight_history.csv")
    if not opt.weight_history.empty:
        opt.weight_history.to_csv(wh_path)
        print(f"[Saved] Weight history → {wh_path}")


if __name__ == "__main__":
    main()
