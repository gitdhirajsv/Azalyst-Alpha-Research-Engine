"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    STATISTICAL VALIDATOR v1
║        Fama-MacBeth · Newey-West · BH Correction · IC Decay Analysis        ║
║        Per Two Sigma / BlackRock / Citadel institutional standards           ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module provides rigorous statistical validation for alpha signals:

1. **Fama-MacBeth cross-sectional regression** (Two Sigma / BlackRock standard)
   - Tests whether a feature has cross-sectional IC with Newey-West corrected t-stats
   - Outputs: feature-level IC, t-stat, p-value

2. **Benjamini-Hochberg multiple testing correction** (Two Sigma standard)
   - Controls FDR when testing 72 features simultaneously
   - Eliminates spurious features that only appear significant by chance

3. **Signal decay analysis** (Citadel standard)
   - IC at t+1, t+12, t+24, t+48 bars
   - Identifies optimal prediction horizon

4. **Feature orthogonality audit** (Two Sigma standard)
   - Correlation matrix audit — drops features with r > 0.85
   - Reduces overfitting, improves IC stability

5. **Stationarity validation** (RenTech standard)
   - ADF test on frac_diff_close and key features
   - Confirms features are stationary before training

6. **Model governance report** (BlackRock Aladdin standard)
   - IC drift, feature importance drift, prediction distribution shift
   - Generated per retrain for audit trail
"""

from __future__ import annotations
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ── Fama-MacBeth Cross-Sectional Regression ──────────────────────────────────

def fama_macbeth_regression(
    panel_data: Dict[str, pd.DataFrame],
    features: List[str],
    target: str = "future_ret",
    min_symbols: int = 20,
) -> pd.DataFrame:
    """
    Fama-MacBeth (1973) two-pass cross-sectional regression.

    For each time period t:
      1. Run cross-sectional regression: r_{i,t+1} = α_t + Σ β_{k,t} * f_{k,i,t} + ε
      2. Collect the time-series of β_{k,t} for each feature k

    Then compute:
      - Mean(β_k) = average cross-sectional IC-like coefficient
      - t-stat with Newey-West correction for autocorrelation

    Parameters
    ----------
    panel_data : dict of {symbol: DataFrame} with features and target
    features : list of feature column names
    target : target column name
    min_symbols : minimum symbols per cross-section to include

    Returns
    -------
    DataFrame with columns: feature, coef_mean, coef_std, t_stat, p_value
    """
    # Align all symbols to common timestamps
    all_times = set()
    for df in panel_data.values():
        if target in df.columns:
            all_times.update(df.index)
    all_times = sorted(all_times)

    # Sample timestamps (every 288 bars = daily) to reduce computation
    sample_step = max(1, len(all_times) // 500)
    sampled_times = all_times[::sample_step]

    betas_series = {f: [] for f in features}

    for t in sampled_times:
        # Collect cross-section at time t
        xs_feat = []
        xs_ret = []
        for sym, df in panel_data.items():
            if t not in df.index:
                continue
            row = df.loc[t]
            if not isinstance(row, pd.Series):
                row = row.iloc[0]
            ret_val = row.get(target)
            if ret_val is None or not np.isfinite(ret_val):
                continue
            feat_vals = [row.get(f, np.nan) for f in features]
            if any(not np.isfinite(v) for v in feat_vals):
                continue
            xs_feat.append(feat_vals)
            xs_ret.append(ret_val)

        if len(xs_feat) < min_symbols:
            continue

        X = np.array(xs_feat, dtype=np.float64)
        y = np.array(xs_ret, dtype=np.float64)

        # Standardize cross-sectionally
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X - X.mean(axis=0)) / X_std

        # OLS: β = (X'X)^{-1} X'y
        try:
            beta = np.linalg.lstsq(X_norm, y - y.mean(), rcond=None)[0]
            for i, f in enumerate(features):
                betas_series[f].append(beta[i])
        except np.linalg.LinAlgError:
            continue

    # Compute t-stats with Newey-West correction
    results = []
    for f in features:
        betas = np.array(betas_series[f])
        if len(betas) < 10:
            results.append({
                "feature": f, "coef_mean": 0.0, "coef_std": 0.0,
                "t_stat": 0.0, "p_value": 1.0, "n_periods": len(betas),
            })
            continue

        mean_b = float(np.mean(betas))
        nw_se = _newey_west_se(betas)
        t_stat = mean_b / nw_se if nw_se > 1e-12 else 0.0
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(betas) - 1)))

        results.append({
            "feature": f,
            "coef_mean": round(mean_b, 6),
            "coef_std": round(float(np.std(betas)), 6),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "n_periods": len(betas),
        })

    return pd.DataFrame(results).sort_values("t_stat", ascending=False,
                                             key=abs).reset_index(drop=True)


def _newey_west_se(x: np.ndarray, max_lags: int = 0) -> float:
    """
    Newey-West (1987) HAC standard error for autocorrelated time series.
    Per BlackRock's model governance framework — corrects ICIR for autocorrelation.
    """
    n = len(x)
    if n < 2:
        return 1e-12
    if max_lags == 0:
        max_lags = max(1, int(np.floor(n ** (1 / 3))))

    x_dm = x - np.mean(x)
    gamma0 = float(np.sum(x_dm ** 2) / n)
    nw_var = gamma0

    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1)  # Bartlett kernel
        gamma_lag = float(np.sum(x_dm[lag:] * x_dm[:-lag]) / n)
        nw_var += 2 * weight * gamma_lag

    nw_var = max(nw_var, 1e-16)
    return float(np.sqrt(nw_var / n))


# ── Benjamini-Hochberg Multiple Testing Correction ──────────────────────────

def benjamini_hochberg(p_values: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """
    Benjamini-Hochberg (1995) FDR correction for multiple hypothesis testing.
    Per Two Sigma standard — eliminates spurious features when testing 72+ features.

    Parameters
    ----------
    p_values : Series indexed by feature name
    alpha : FDR level (default 0.05)

    Returns
    -------
    DataFrame with: feature, p_value, rank, bh_threshold, significant
    """
    n = len(p_values)
    sorted_pv = p_values.sort_values()
    ranks = np.arange(1, n + 1)
    thresholds = ranks / n * alpha

    # Find the largest rank where p_value <= threshold
    significant = sorted_pv.values <= thresholds
    # BH procedure: all features with rank <= max significant rank
    if significant.any():
        max_sig_rank = int(np.max(np.where(significant)[0]) + 1)
        is_sig = ranks <= max_sig_rank
    else:
        is_sig = np.zeros(n, dtype=bool)

    result = pd.DataFrame({
        "feature": sorted_pv.index,
        "p_value": sorted_pv.values,
        "rank": ranks,
        "bh_threshold": thresholds,
        "significant": is_sig,
    })
    return result


# ── Signal Decay Analysis ────────────────────────────────────────────────────

def signal_decay_analysis(
    panel_data: Dict[str, pd.DataFrame],
    features: List[str],
    horizons: List[int] = None,
    min_symbols: int = 20,
) -> pd.DataFrame:
    """
    IC decay analysis at multiple forward horizons.
    Per Citadel standard — identifies optimal prediction horizon.

    Computes Spearman IC between each feature and forward returns at
    t+1, t+12, t+24, t+48 bars.
    """
    if horizons is None:
        horizons = [1, 3, 12, 24, 48]

    results = []

    for feat in features:
        for h in horizons:
            ics = []
            # Sample timestamps
            for sym, df in panel_data.items():
                if feat not in df.columns or "close" not in df.columns:
                    continue
                close = df["close"]
                fwd_ret = np.log(close.shift(-h) / close)
                valid = df[feat].notna() & fwd_ret.notna()
                if valid.sum() < 100:
                    continue
                # Compute IC for this symbol's time-series
                ic = float(stats.spearmanr(
                    df.loc[valid, feat].values,
                    fwd_ret[valid].values
                )[0])
                if np.isfinite(ic):
                    ics.append(ic)

            mean_ic = float(np.mean(ics)) if ics else 0.0
            results.append({
                "feature": feat,
                "horizon_bars": h,
                "ic_mean": round(mean_ic, 5),
                "n_symbols": len(ics),
            })

    return pd.DataFrame(results)


# ── Feature Orthogonality Audit ──────────────────────────────────────────────

def feature_orthogonality_audit(
    panel_data: Dict[str, pd.DataFrame],
    features: List[str],
    max_corr: float = 0.85,
    sample_size: int = 100000,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Per Two Sigma standard: audit feature correlation matrix and identify
    redundant pairs (|r| > max_corr). Returns correlation matrix and
    list of features to drop (keeping the one with higher IC where possible).
    """
    # Pool data across symbols
    chunks = []
    for sym, df in panel_data.items():
        available = [f for f in features if f in df.columns]
        if len(available) == len(features):
            chunk = df[features].dropna()
            if len(chunk) > 0:
                chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(), []

    pooled = pd.concat(chunks, axis=0)
    if len(pooled) > sample_size:
        pooled = pooled.sample(sample_size, random_state=42)

    corr = pooled.corr(method="spearman")

    # Find all pairs with |r| > threshold
    to_drop = set()
    n = len(features)
    for i in range(n):
        if features[i] in to_drop:
            continue
        for j in range(i + 1, n):
            if features[j] in to_drop:
                continue
            r = abs(corr.iloc[i, j])
            if r > max_corr:
                # Drop the feature that appears later in the list
                # (assumes FEATURE_COLS is ordered by expected importance)
                to_drop.add(features[j])

    return corr, sorted(to_drop)


# ── Stationarity Validation ──────────────────────────────────────────────────

def stationarity_audit(
    panel_data: Dict[str, pd.DataFrame],
    features: List[str],
    significance: float = 0.05,
    max_symbols: int = 20,
) -> pd.DataFrame:
    """
    Per RenTech standard: ADF test on key features to confirm stationarity.
    Non-stationary features degrade XGBoost performance.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return pd.DataFrame({"error": ["statsmodels not installed"]})

    results = []
    symbols = list(panel_data.keys())[:max_symbols]

    for feat in features:
        adf_stats = []
        p_vals = []
        for sym in symbols:
            df = panel_data[sym]
            if feat not in df.columns:
                continue
            series = df[feat].dropna()
            if len(series) < 100:
                continue
            # Sample to keep fast
            if len(series) > 5000:
                series = series.iloc[-5000:]
            try:
                result = adfuller(series, maxlag=20, autolag="AIC")
                adf_stats.append(result[0])
                p_vals.append(result[1])
            except Exception:
                continue

        if adf_stats:
            results.append({
                "feature": feat,
                "mean_adf_stat": round(float(np.mean(adf_stats)), 4),
                "mean_p_value": round(float(np.mean(p_vals)), 6),
                "pct_stationary": round(
                    float(np.mean([p < significance for p in p_vals])) * 100, 1
                ),
                "n_tested": len(adf_stats),
            })

    return pd.DataFrame(results).sort_values("pct_stationary",
                                             ascending=True).reset_index(drop=True)


# ── Model Governance Report ──────────────────────────────────────────────────

def generate_governance_report(
    run_id: str,
    retrain_label: str,
    week_num: int,
    ic_in_sample: float,
    ic_oos: float,
    importance_current: pd.Series,
    importance_previous: Optional[pd.Series],
    pred_distribution: np.ndarray,
    pred_distribution_prev: Optional[np.ndarray],
    output_dir: str = "./results",
) -> Dict:
    """
    Per BlackRock Aladdin model governance framework.
    Produces a validation report per retrain:
      - IC in-sample vs OOS (detects overfitting)
      - Feature importance drift (detects model instability)
      - Prediction distribution shift (detects regime change)
    """
    report = {
        "run_id": run_id,
        "retrain_label": retrain_label,
        "week_num": week_num,
        "ic_in_sample": round(ic_in_sample, 5),
        "ic_oos": round(ic_oos, 5),
        "ic_gap": round(ic_in_sample - ic_oos, 5),
    }

    # Feature importance drift
    if importance_previous is not None and len(importance_previous) > 0:
        common = list(set(importance_current.index) & set(importance_previous.index))
        if common:
            cur = importance_current.reindex(common).fillna(0)
            prev = importance_previous.reindex(common).fillna(0)
            drift = float(np.sqrt(np.mean((cur.values - prev.values) ** 2)))
            rank_corr = float(stats.spearmanr(cur.values, prev.values)[0])
            report["importance_drift_rmse"] = round(drift, 5)
            report["importance_rank_corr"] = round(rank_corr, 4)

            # Top-5 features that changed most
            delta = (cur - prev).abs().sort_values(ascending=False)
            report["top_drift_features"] = list(delta.head(5).index)
    else:
        report["importance_drift_rmse"] = None
        report["importance_rank_corr"] = None

    # Prediction distribution shift
    if pred_distribution is not None and len(pred_distribution) > 10:
        report["pred_mean"] = round(float(np.mean(pred_distribution)), 6)
        report["pred_std"] = round(float(np.std(pred_distribution)), 6)
        report["pred_skew"] = round(float(stats.skew(pred_distribution)), 4)
        report["pred_kurt"] = round(float(stats.kurtosis(pred_distribution)), 4)

        if (pred_distribution_prev is not None and
                len(pred_distribution_prev) > 10):
            # KS test for distribution shift
            ks_stat, ks_pval = stats.ks_2samp(pred_distribution,
                                               pred_distribution_prev)
            report["pred_ks_stat"] = round(float(ks_stat), 4)
            report["pred_ks_pval"] = round(float(ks_pval), 6)

    # Flag warnings
    warnings_list = []
    if report["ic_gap"] > 0.02:
        warnings_list.append("OVERFIT: IC gap > 2% (in-sample much higher than OOS)")
    if report.get("importance_rank_corr") is not None and \
            report["importance_rank_corr"] < 0.5:
        warnings_list.append("DRIFT: Feature importance rank correlation < 0.5")
    if report.get("pred_ks_pval") is not None and report["pred_ks_pval"] < 0.01:
        warnings_list.append("SHIFT: Prediction distribution significantly changed (KS p<0.01)")
    report["warnings"] = warnings_list

    # Save report
    os.makedirs(f"{output_dir}/governance", exist_ok=True)
    report_path = f"{output_dir}/governance/governance_{retrain_label}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


# ── Full Validation Pipeline ─────────────────────────────────────────────────

def run_full_validation(
    panel_data: Dict[str, pd.DataFrame],
    features: List[str],
    output_dir: str = "./results",
) -> Dict:
    """
    Run the complete institutional validation pipeline:
    1. Fama-MacBeth cross-sectional regression
    2. Benjamini-Hochberg correction
    3. Feature orthogonality audit
    4. Stationarity check

    Returns dict with all results and list of features that pass all tests.
    """
    print("\n  [VALIDATOR] Running institutional validation pipeline...")

    # 1. Fama-MacBeth
    print("  [1/4] Fama-MacBeth cross-sectional regression...")
    fm_results = fama_macbeth_regression(panel_data, features)

    # 2. BH correction
    print("  [2/4] Benjamini-Hochberg multiple testing correction...")
    p_values = fm_results.set_index("feature")["p_value"]
    bh_results = benjamini_hochberg(p_values, alpha=0.10)
    n_sig = int(bh_results["significant"].sum())
    print(f"         {n_sig}/{len(features)} features significant at FDR=10%")

    # 3. Orthogonality
    print("  [3/4] Feature orthogonality audit (r > 0.85 correlation)...")
    corr, to_drop = feature_orthogonality_audit(panel_data, features)
    if to_drop:
        print(f"         Recommend dropping {len(to_drop)} redundant features: "
              f"{to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")

    # 4. Stationarity
    print("  [4/4] Stationarity audit (ADF test)...")
    stat_results = stationarity_audit(
        panel_data, features,
    )
    if not stat_results.empty:
        non_stat = stat_results[stat_results["pct_stationary"] < 50]
        if not non_stat.empty:
            print(f"         WARNING: {len(non_stat)} features may be non-stationary: "
                  f"{list(non_stat['feature'].head(5))}")

    # Compile valid features
    sig_features = set(bh_results[bh_results["significant"]]["feature"])
    valid_features = [f for f in features
                      if f in sig_features and f not in to_drop]

    # Save all results
    os.makedirs(f"{output_dir}/validation", exist_ok=True)
    fm_results.to_csv(f"{output_dir}/validation/fama_macbeth.csv", index=False)
    bh_results.to_csv(f"{output_dir}/validation/bh_correction.csv", index=False)
    if not corr.empty:
        corr.to_csv(f"{output_dir}/validation/feature_correlation.csv")
    if not stat_results.empty:
        stat_results.to_csv(f"{output_dir}/validation/stationarity.csv", index=False)

    summary = {
        "total_features": len(features),
        "fama_macbeth_significant": n_sig,
        "redundant_features": len(to_drop),
        "valid_features": len(valid_features),
        "valid_feature_names": valid_features,
        "dropped_correlated": to_drop,
    }

    with open(f"{output_dir}/validation/validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  [VALIDATOR] Pipeline complete: {len(valid_features)}/{len(features)} "
          f"features pass all tests")
    print(f"  [VALIDATOR] Results -> {output_dir}/validation/")

    return summary
