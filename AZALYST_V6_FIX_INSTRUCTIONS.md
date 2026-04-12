# AZALYST V6 — FIX INSTRUCTIONS FOR SONNET

**Audience:** Claude Sonnet running in an agentic coding environment (Cursor, Claude Code, Windsurf, or similar) with file-edit and shell access to the Azalyst Alpha Research Engine repository.

**Author context:** These instructions were produced by Claude Opus after a code review of the supporting modules (`azalyst_ic_filter.py`, `azalyst_ml.py`, `azalyst_pump_dump.py`, `azalyst_risk.py`, `azalyst_signal_combiner.py`, `azalyst_tf_utils.py`, `azalyst_train.py`, `build_feature_cache.py`). Opus did **not** see `azalyst_v6_engine.py`, `azalyst_factors_v2.py`, `azalyst_alphaopt.py`, `azalyst_validator.py`, or `azalyst_db.py`. You must read those yourself before editing them.

**User's goal:** Run v6 cleanly on 446 coins and produce an honest backtest before any paper/live trading. The user is **not** looking for inflated Sharpe numbers. Fixes here will generally make backtest numbers *worse*, and that is the correct and desired outcome. Do not "optimize for performance."

**Critical rule:** If any instruction below conflicts with what you actually see in the code, STOP and report the conflict rather than forcing the change. The repo state may have drifted from what Opus reviewed.

---

## PHASE 0 — REPO DISCOVERY (do this first, do not skip)

Before touching any file, produce a discovery report by doing the following:

### 0.1 List the actual files present
```bash
ls -la *.py *.bat *.md *.ipynb 2>/dev/null
ls -la results_v6/ 2>/dev/null
ls -la feature_cache/ 2>/dev/null
ls -la data/ 2>/dev/null | head -20
```

Report:
- Which of these files exist: `azalyst_v6_engine.py`, `azalyst_v5_engine.py`, `azalyst_factors_v2.py`, `azalyst_alphaopt.py`, `azalyst_validator.py`, `azalyst_db.py`, `azalyst_train.py`, `azalyst_ic_filter.py`, `azalyst_ml.py`, `azalyst_pump_dump.py`, `azalyst_risk.py`, `azalyst_signal_combiner.py`, `azalyst_tf_utils.py`, `build_feature_cache.py`
- Count of `.parquet` files in `data/`
- Count of `.parquet` files in `feature_cache/` (if it exists)
- Whether `results_v6/weekly_summary_v6.csv` exists and how many rows it has

### 0.2 Read v6 entry points and report structure
Read these files fully (`azalyst_v6_engine.py` is the main one — read every line):

- `azalyst_v6_engine.py` — the main backtest loop
- `azalyst_factors_v2.py` — feature construction
- `azalyst_alphaopt.py` — Elastic Net factor combiner (if it exists)
- `azalyst_validator.py` — significance testing
- `azalyst_db.py` — data access layer

For each file produce a 5-10 line summary of: what it does, its public entry points, and any obvious red flags (eval on training data, missing `.shift()` calls on features, references to `future_*` columns without shifting, hardcoded paths).

### 0.3 Report findings
Do not proceed to Phase 1 until you have printed the discovery report and the user has confirmed it matches their repo. If running in fully autonomous mode, proceed but make the discovery report the first section of your final output.

---

## PHASE 1 — CRITICAL BUG FIXES IN REVIEWED MODULES

These are bugs Opus found by reading the code directly. Each fix is small, surgical, and the correct version is specified precisely.

### 1.1 `azalyst_ic_filter.py` — FIX TYPO THAT WILL CRASH FALLBACK PATH

**File:** `azalyst_ic_filter.py`
**Function:** `filter_features_by_ic`
**Bug:** Variable name typo `ice_series` instead of `ic_series` in the fallback branch. Silent until triggered, then crashes with `NameError`.

Find this block:
```python
    # Fallback: if too few features pass threshold, take top by ICIR
    if selected_mask.sum() < min_features:
        selected_features = ice_series.abs().nlargest(min_features)
        selected_mask = pd.Series(False, index=ic_series.index)
        selected_mask[selected_features.index] = True
```

Replace with:
```python
    # Fallback: if too few features pass threshold, take top by |IC|
    if selected_mask.sum() < min_features:
        selected_features = ic_series.abs().nlargest(min_features)
        # Re-fetch with sign preserved, sorted by |IC| desc
        selected_features = ic_series.loc[selected_features.index].sort_values(
            key=abs, ascending=False
        )
        selected_mask = pd.Series(False, index=ic_series.index)
        selected_mask[selected_features.index] = True
```

### 1.2 `azalyst_ic_filter.py` — FIX BROKEN ICIR FORMULA

**File:** `azalyst_ic_filter.py`
**Function:** `compute_icir`
**Bug:** Current implementation divides each feature's IC by the cross-sectional std of ICs across features. This is not ICIR. ICIR is `mean(IC_t) / std(IC_t)` where `t` indexes time — it measures stability of a single feature's IC across rolling time windows.

Replace the entire `compute_icir` function with:
```python
def compute_icir(X: np.ndarray, y_ret: np.ndarray,
                 feature_names: List[str],
                 n_windows: int = 10,
                 min_window_size: int = 500) -> pd.Series:
    """
    Compute true ICIR = mean(IC_t) / std(IC_t) across time windows.
    For each feature, splits the data into n_windows chronological chunks,
    computes IC in each chunk, and returns the ratio of mean to std.
    Higher ICIR = more stable predictive power over time.

    Args:
        X: Feature matrix (n_samples, n_features)
        y_ret: Target returns (n_samples,)
        feature_names: List of feature names
        n_windows: Number of chronological windows to split into
        min_window_size: Minimum samples per window

    Returns:
        Series of ICIR values indexed by feature name
    """
    n_samples, n_features = X.shape
    window_size = max(min_window_size, n_samples // n_windows)
    if window_size * 2 > n_samples:
        # Not enough data for multi-window ICIR — return zeros with warning
        return pd.Series(0.0, index=feature_names, name="ICIR")

    # Collect IC per feature per window
    ic_matrix = np.zeros((n_features, n_windows))
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n_samples)
        if end - start < 50:
            continue
        for j in range(n_features):
            ic_matrix[j, w] = compute_feature_ic(X[start:end, j], y_ret[start:end])

    # Per-feature mean and std across time windows
    ic_mean = ic_matrix.mean(axis=1)
    ic_std = ic_matrix.std(axis=1)
    icir = np.where(ic_std > 1e-10, ic_mean / ic_std, 0.0)

    return pd.Series(icir, index=feature_names, name="ICIR")
```

Then update the two call sites in the same file:
- In `filter_features_by_ic`, change `icir_series = compute_icir(ic_series)` to `icir_series = compute_icir(X, y_ret, feature_names)`
- In `rank_features_by_ic`, change `icir_series = compute_icir(ic_series)` to `icir_series = compute_icir(X, y_ret, feature_names)`

### 1.3 `azalyst_train.py` — FIX FEATURE LEAKAGE IN IC FILTERING

**File:** `azalyst_train.py`
**Function:** `train_regression_model`
**Bug:** IC filtering is done on the full `X, y_ret` before the purged CV split, meaning validation folds inform feature selection. This inflates reported CV metrics.

Find this block inside `train_regression_model`:
```python
    # IC-based feature filtering (NEW v5.1)
    if use_ic_filtering and len(feature_cols) > 30:
        from azalyst_ic_filter import filter_features_by_ic
        X_filtered, selected_features, ic_info = filter_features_by_ic(
            X, y_ret, feature_cols,
            ic_threshold=ic_threshold,
            min_features=20,
            verbose=True
        )
        X = X_filtered
        feature_cols = selected_features
```

Replace with:
```python
    # IC-based feature filtering (v5.1) — LEAK-FREE VERSION
    # Only use the first 70% of data (chronologically) to select features.
    # The remaining 30% includes what the purged CV will use for validation.
    # This prevents validation information from contaminating feature selection.
    if use_ic_filtering and len(feature_cols) > 30:
        from azalyst_ic_filter import filter_features_by_ic
        cutoff = int(len(X) * 0.70)
        X_filter_set = X[:cutoff]
        y_filter_set = y_ret[:cutoff]
        _, selected_features, ic_info = filter_features_by_ic(
            X_filter_set, y_filter_set, feature_cols,
            ic_threshold=ic_threshold,
            min_features=20,
            verbose=True
        )
        # Apply the selection to the full dataset
        selected_indices = np.array([feature_cols.index(n) for n in selected_features])
        X = X[:, selected_indices]
        feature_cols = selected_features
        print(f"  [LEAK-FREE] Feature selection used first {cutoff:,} rows only")
```

### 1.4 `azalyst_train.py` — ADD TWO-SIDED EMBARGO TO PurgedTimeSeriesCV

**File:** `azalyst_train.py`
**Class:** `PurgedTimeSeriesCV`
**Bug:** Current implementation embargoes between train and validation only in one direction (gap before val). For overlapping labels (e.g., 5-day forward returns), you must also embargo *after* validation before training resumes in the next fold. This is the López de Prado "purging and embargo" fix.

Replace the entire class with:
```python
class PurgedTimeSeriesCV:
    """
    Lopez de Prado (AFML Ch. 7) purged walk-forward CV with two-sided embargo.

    Prevents look-ahead bias from autocorrelated features and overlapping labels.
    For a forward-return label with horizon h, gap must be >= h to prevent leakage.

    Args:
        n_splits: Number of folds
        gap: Embargo period in bars (default 48 = 4 hours at 5-min frequency).
             MUST be >= label horizon. For 5-day labels on 5-min bars, use 1440.
    """
    def __init__(self, n_splits=5, gap=48):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X):
        n = len(X)
        # Reserve space for n_splits validation folds plus embargo on each side
        total_embargo = self.gap * 2 * self.n_splits
        usable = n - total_embargo
        if usable <= 0:
            raise ValueError(
                f"Dataset too small ({n} rows) for {self.n_splits} splits "
                f"with gap={self.gap}. Need > {total_embargo} rows."
            )
        fold_size = usable // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            val_start = train_end + self.gap  # embargo before val
            val_end = val_start + fold_size
            if val_end > n:
                break
            # Train is everything before (train_end), which already has
            # an implicit gap before val_start. For the next iteration,
            # the next fold's train includes data after val_end + gap,
            # but since we're using expanding train only up to train_end
            # of the CURRENT fold, the two-sided protection is realized
            # by the fold layout itself.
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx
```

Additionally, add this warning at the top of `train_regression_model` right after the GPU probe:
```python
    # Warn if embargo is smaller than likely label horizon
    _label_horizon_5d = 1440  # 5-day horizon at 5-min bars
    if hasattr(cv := PurgedTimeSeriesCV(), 'gap') and cv.gap < 288:
        print(f"[WARN] CV gap=48 bars (4h). If targets are >4h horizon, "
              f"increase gap in PurgedTimeSeriesCV to match or exceed horizon.")
```

Actually don't add that warning — it's confusing. Instead, whenever `train_regression_model` is called from `azalyst_v6_engine.py`, the caller must pass a gap that matches the target horizon. Add this parameter to the function signature:

Change:
```python
def train_regression_model(X, y_ret, feature_cols, label="", use_gpu=True,
                           use_ic_filtering=True, ic_threshold=0.005):
```

To:
```python
def train_regression_model(X, y_ret, feature_cols, label="", use_gpu=True,
                           use_ic_filtering=True, ic_threshold=0.005,
                           cv_gap=48):
```

And change the CV instantiation inside the function from:
```python
    cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
```

To:
```python
    cv = PurgedTimeSeriesCV(n_splits=5, gap=cv_gap)
    print(f"  [CV] Purged K-Fold with gap={cv_gap} bars "
          f"(must be >= label horizon)")
```

Do the same for `train_confidence_model` and `train_model` (the legacy v4 function): add `cv_gap=48` parameter, pass it to the `PurgedTimeSeriesCV` constructor.

### 1.5 `azalyst_train.py` — FIX WEIGHTED R² DOCSTRING LIE

**File:** `azalyst_train.py`
**Function:** `weighted_r2_score`
**Bug:** Docstring claims this is the Jane Street weighted R² metric, but it's called everywhere with `weights=None`, making it plain unweighted R². Either pass real weights or rename.

Replace the function with:
```python
def weighted_r2_score(y_true, y_pred, weights=None):
    """
    R² metric with optional sample weights.

    If weights is None, computes standard R² = 1 - SS_res/SS_tot.
    If weights is provided, uses the weighted form (Jane Street comp metric).
    Note: to actually get the "Jane Street" behavior, you must pass
    economically-meaningful weights (e.g., inverse vol, liquidity, or
    position size). Passing None gives plain R².
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
```

### 1.6 `azalyst_pump_dump.py` — DELETE DEAD LOOK-AHEAD CODE

**File:** `azalyst_pump_dump.py`
**Function:** `compute_pump_dump_scores`
**Bug:** `max_up` and `max_down` variables compute forward-inclusive rolling stats that look ahead. They are currently unused, but leaving them in invites a future bug.

Find and delete these three lines:
```python
    # Forward-looking reversal (for labeling, NOT features — shifted later)
    # Check if a spike reverses within the window
    max_up = ret_1h.rolling(rev_window, min_periods=1).max()
    max_down = ret_1h.rolling(rev_window, min_periods=1).min()
```

Keep everything below that uses `trailing_max` and `trailing_min` (those are correctly `.shift(1)`-ed).

### 1.7 `azalyst_risk.py` — WIRE UP FEES THAT ARE STORED BUT NEVER USED

**File:** `azalyst_risk.py`
**Class:** `RiskManager`
**Bug:** `entry_fee` and `exit_fee` are stored in `__init__` but never referenced anywhere in the class. Depending on how v6 computes PnL, this may or may not be a real bug — but at minimum the dead storage is misleading.

After reading `azalyst_v6_engine.py` in Phase 0, determine whether the engine applies fees itself or delegates to `RiskManager`. If the engine applies fees, add a comment in `RiskManager.__init__`:
```python
    def __init__(self, entry_fee: float = 0.001, exit_fee: float = 0.001):
        # NOTE: These values are stored for reference only. Fees are applied
        # in azalyst_v6_engine.py at position entry/exit time, not here.
        # See `apply_transaction_costs` in v6 engine.
        self.entry_fee = entry_fee
        self.exit_fee = exit_fee
```

If the engine does NOT apply fees anywhere, this is a real bug — stop and report it to the user, because their entire backtest is gross-of-fees.

### 1.8 `azalyst_risk.py` — ADD COVARIANCE SHRINKAGE TO MVO

**File:** `azalyst_risk.py`
**Method:** `compute_mvo_weights`
**Bug:** Plain sample covariance at N~50 with weekly crypto data is extremely ill-conditioned. MVO will concentrate deterministically in 2-3 assets.

Find:
```python
    def compute_mvo_weights(self,
                            returns_df: pd.DataFrame,
                            target_return: Optional[float] = None,
                            risk_free_rate: float = 0.0) -> pd.Series:
        """
        Mean-Variance Optimization (Sharpe Maximization).
        """
        mu = returns_df.mean()
        sigma = returns_df.cov()
```

Replace with:
```python
    def compute_mvo_weights(self,
                            returns_df: pd.DataFrame,
                            target_return: Optional[float] = None,
                            risk_free_rate: float = 0.0,
                            use_shrinkage: bool = True) -> pd.Series:
        """
        Mean-Variance Optimization (Sharpe Maximization).
        Uses Ledoit-Wolf covariance shrinkage by default — sample covariance
        is unusable at N~50 with short crypto histories.
        """
        mu = returns_df.mean()
        if use_shrinkage:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(returns_df.fillna(0).values)
                sigma = pd.DataFrame(
                    lw.covariance_,
                    index=returns_df.columns,
                    columns=returns_df.columns,
                )
            except Exception:
                sigma = returns_df.cov()
        else:
            sigma = returns_df.cov()
```

### 1.9 `build_feature_cache.py` — FIX HORIZON MATH FOR NON-5MIN TIMEFRAMES

**File:** `build_feature_cache.py`
**Function:** `_process_symbol`
**Bug:** Horizon-in-bars calculations use inline dict lookups that give wrong answers for timeframes above 1h. For `resample='4h'`, `60 // 240 = 0` then `max(1, 0) = 1` — so "1-hour horizon" becomes "1 bar = 4 hours." Silent.

Find this block:
```python
        horizon_15m = max(1, 15 // max(1, {'1min':1,'3min':3,'5min':5,'15min':15,
                          '30min':30,'1h':60,'4h':240,'1d':1440}.get(resample, 5)))
        horizon_1h = max(1, 60 // max(1, {'1min':1,'3min':3,'5min':5,'15min':15,
                         '30min':30,'1h':60,'4h':240,'1d':1440}.get(resample, 5)))
```

And the similar blocks for `horizon_1d` and `horizon_5d`. Replace all of them with a single block that uses the existing `azalyst_tf_utils`:
```python
        # Horizon-in-bars for each forward-return target.
        # Use tf_utils to handle all timeframes correctly.
        from azalyst_tf_utils import get_tf_constants
        _bph, _bpd, _ = get_tf_constants(resample)
        # bars per minute = bph / 60
        bars_per_min = _bph / 60.0
        horizon_15m = max(1, int(round(15 * bars_per_min)))
        horizon_1h  = max(1, int(round(60 * bars_per_min)))
        horizon_1d  = max(1, _bpd)
        horizon_5d  = max(1, _bpd * 5)
```

Then the subsequent `feats["future_ret_15m"] = ...` lines stay the same — they just use the correct horizons now.

### 1.10 `build_feature_cache.py` — TIGHTEN NAN DROP THRESHOLD

**File:** `build_feature_cache.py`
**Bug:** `dropna(subset=FEATURE_COLS, how="all")` only drops rows where *every* feature is NaN. Rows with 60/65 NaNs pass through, relying on XGBoost's missing-value handling in ways that are hard to audit.

Find:
```python
        feats = feats.dropna(subset=FEATURE_COLS, how="all").astype(np.float32)
```

Replace with:
```python
        # Require at least 80% of features to be non-NaN per row.
        # Rows with massive NaN counts (warmup periods) get dropped.
        min_non_nan = int(0.80 * len(FEATURE_COLS))
        feats = feats.dropna(subset=FEATURE_COLS, thresh=min_non_nan).astype(np.float32)
```

---

## PHASE 2 — INVESTIGATION TASKS FOR FILES OPUS DID NOT SEE

For each file below, READ IT FULLY first, then apply the check. Report findings before making any changes.

### 2.1 `azalyst_v6_engine.py` — full audit

Read the entire file and report on each of the following:

**A. Regime gating severity**
Search for `IC_GATING_THRESHOLD`, `kill_switch`, `skip_week`, or similar. Determine:
- What fraction of weeks are gated out in a typical run? (The user's `validate_startup.py` warns of ~67% at threshold=-0.03.)
- Is the gate using forward-looking information? (e.g., computing regime from data that includes the trade week itself)

If gating > 40% of weeks, flag this loudly. A strategy that trades 33% of the time has 1/3 the effective sample size and the Sharpe confidence interval triples.

**B. Beta neutralization**
Search for `beta`, `neutraliz`, `residualize`, `BTC`, `hedge`. Determine:
- Is beta computed in-sample or rolling? Rolling window length?
- Is it beta vs BTC only, or a multi-asset factor basket?
- Is the beta used to construct residual returns *before* ranking, or to adjust weights *after* ranking?
- Is there look-ahead? (Using the same week's BTC return to compute beta for that week is a leak.)

Report what you find. The correct version: rolling beta from a strictly prior window (e.g., 60-day lookback ending at t-1), residualize all coin returns against BTC return at time t, then rank on residuals.

**C. Transaction cost model**
Search for `fee`, `cost`, `slippage`, `0.001`, `0.0004`, `0.0010`. Determine:
- Are fees charged on new entries only (correct) or every bar (wrong)?
- Is slippage modeled, or only exchange fees?
- What is the total round-trip cost assumed?
- Is there a turnover breakdown in the output?

For crypto on Binance spot, realistic total cost per round-trip is 8-15 bps including slippage on top-500 liquidity names. If the engine assumes less than 5 bps, Sharpe is inflated.

**D. Train/test split integrity**
Search for `train_end`, `test_start`, `split_date`, `Y1`, `Y2`, `Y3`. Determine:
- Is Year 3 (or whatever the OOS period is) *ever* touched during training, feature construction, or hyperparameter selection?
- Are rolling statistics (means, stds, quantiles) computed on the full dataset or only on training data?

Cross-sectional feature normalization computed on the full pool is a subtle leak. Report the normalization logic explicitly.

**E. Forward-return label alignment**
For each `future_ret_*` column, verify that at time t the label uses only data from [t, t+h], and that no feature at time t has been computed using data from [t, t+h]. This is the single most common bug source.

### 2.2 `azalyst_factors_v2.py` — full audit

Read the entire file and report on each of the following:

**A. Fractional differentiation**
Search for `frac_diff`, `fracdiff`, `FFD`. The README says d=0.4 with the Fixed-Width Window method (López de Prado AFML Ch. 5).
- Is the weight vector computed correctly? (Recursion: `w[k] = -w[k-1] * (d - k + 1) / k`)
- Is the convolution using *trailing* data only? (The whole point of FFD is stationarity with no look-ahead.)
- Is the threshold (for truncating weights) correctly applied?

If the convolution accidentally uses `mode='same'` with centered alignment instead of trailing, every feature derived from frac_diff leaks the future.

**B. Cross-sectional features**
Search for `rank`, `pct`, `cs_`, `groupby.*transform`. Determine:
- Are cross-sectional ranks computed per-timestamp (correct) or per-symbol across time (wrong)?
- If per-timestamp, what happens when some symbols have NaN? Are they included in the rank denominator, biasing ranks?

**C. Hurst exponent and FFT features**
- Is the Hurst computation using a rolling window of past data only?
- Does the FFT use a windowed subset that ends at time t?

**D. Feature list export**
Report the actual contents of `FEATURE_COLS`. The README claims 56 features, the notebook header claims 65. Get the ground truth.

### 2.3 `azalyst_alphaopt.py` — Elastic Net factor combiner

If this file exists, read it and report:

**A. Training window**
- Is the ElasticNet fit on the same window the factor scores come from? (If yes, this is overfitting the combiner to the training period.)
- Is there a rolling refit, and if so what window?

**B. Regularization strength**
- What are the `alpha` and `l1_ratio` values? Are they tuned via CV or hardcoded?
- If tuned, is the tuning itself walk-forward?

**C. Factor orthogonalization**
- Are factors decorrelated before ElasticNet? (If not, coefficient interpretation is unreliable but the combined score may still be fine.)

### 2.4 `azalyst_validator.py` — statistical significance testing

If this file exists, read it and report:

**A. Multiple testing correction**
- Is there Bonferroni, Benjamini-Hochberg, or other FDR correction applied to per-feature p-values?
- How many features are being tested? (The README says 56; correcting for 56 tests lowers each individual significance threshold substantially.)

**B. Newey-West standard errors**
- If Newey-West is used, what is the lag truncation?
- Is it applied to factor premia or to the strategy PnL series?

**C. Deflated Sharpe ratio**
- Is the López de Prado deflated Sharpe ratio computed? (Accounts for sample length, skew, kurtosis, and number of strategies tried.)
- If not, add a task to compute it after the backtest completes.

### 2.5 `azalyst_db.py` — data access layer

Read and report:
- What format is the underlying data? (Parquet, DuckDB, SQLite?)
- Is there any caching that could serve stale data?
- Are timestamps stored as UTC? Are they strictly monotonic per symbol?

---

## PHASE 3 — SAFETY RAILS TO ADD

These are small additions that catch bugs going forward. Add them only after Phases 1 and 2 are complete.

### 3.1 Add a leakage smoke test

Create a new file `azalyst_leak_test.py`:

```python
"""
Leakage smoke test — runs before every v6 training session.
Builds a synthetic feature that is a shifted copy of the target.
If the trained model's IC on this feature is not near 1.0 (when properly
aligned) or near 0.0 (when misaligned by more than the embargo), there
is a leak or an alignment bug.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def run_leak_test(X: np.ndarray, y_ret: np.ndarray, feature_names: list,
                  embargo_bars: int = 48) -> dict:
    """
    Three checks:
    1. Shuffle target within fold — IC should collapse to ~0.
    2. Add a feature = y_ret shifted by -1 (future leak) — IC should be ~1.0.
       If training code catches this, the shifted feature is correctly rejected.
    3. Add a feature = y_ret shifted by +embargo_bars (past only) — IC is legit.

    Returns a dict of test results. Call this before train_regression_model.
    """
    results = {}

    # Test 1: shuffle y — IC should be ~0
    y_shuffled = np.random.permutation(y_ret)
    ics = []
    for j in range(min(20, X.shape[1])):
        mask = np.isfinite(X[:, j]) & np.isfinite(y_shuffled)
        if mask.sum() > 100:
            ic, _ = spearmanr(X[mask, j], y_shuffled[mask])
            if np.isfinite(ic):
                ics.append(abs(ic))
    results['shuffled_mean_abs_ic'] = float(np.mean(ics)) if ics else 0.0
    results['shuffled_test_pass'] = results['shuffled_mean_abs_ic'] < 0.02

    # Test 2: leaked feature — IC should be ~1 (if this is NOT ~1, something
    # is wrong with the IC computation itself)
    leaked = np.roll(y_ret, -1)
    leaked[-1] = np.nan
    mask = np.isfinite(leaked) & np.isfinite(y_ret)
    ic_leaked, _ = spearmanr(leaked[mask], y_ret[mask])
    results['leaked_feature_ic'] = float(ic_leaked)
    results['leaked_test_pass'] = ic_leaked > 0.95

    # Test 3: past-only shifted feature
    past = np.roll(y_ret, embargo_bars + 1)
    past[:embargo_bars + 1] = np.nan
    mask = np.isfinite(past) & np.isfinite(y_ret)
    ic_past, _ = spearmanr(past[mask], y_ret[mask])
    results['past_feature_ic'] = float(ic_past)

    return results


if __name__ == "__main__":
    # Self-test
    np.random.seed(42)
    n = 10000
    X = np.random.randn(n, 10)
    y = np.random.randn(n) * 0.01
    print(run_leak_test(X, y, [f"f{i}" for i in range(10)]))
```

Then in `azalyst_v6_engine.py`, right before the first call to `train_regression_model`, add:

```python
    from azalyst_leak_test import run_leak_test
    print("\n[LEAK TEST] Running pre-training sanity checks...")
    leak_results = run_leak_test(X_train, y_train, feature_cols)
    for k, v in leak_results.items():
        print(f"  {k}: {v}")
    if not leak_results['shuffled_test_pass']:
        raise RuntimeError(
            f"LEAK TEST FAILED: shuffled target has |IC|="
            f"{leak_results['shuffled_mean_abs_ic']:.4f} > 0.02. "
            f"Something is wrong with your features or IC computation."
        )
    if not leak_results['leaked_test_pass']:
        raise RuntimeError(
            f"LEAK TEST FAILED: IC computation is broken — a perfect leak "
            f"gave IC={leak_results['leaked_feature_ic']:.4f}, expected > 0.95."
        )
    print("[LEAK TEST] Passed.\n")
```

### 3.2 Add deflated Sharpe computation

Create `azalyst_deflated_sharpe.py`:

```python
"""
Deflated Sharpe Ratio — López de Prado (2014).
Adjusts observed Sharpe for:
  - Finite sample length
  - Non-normal returns (skew and kurtosis)
  - Multiple testing (number of strategies tried)

Use: after a backtest, plug in the observed Sharpe and the number of
distinct strategy configurations that were tested to get here.
"""
import numpy as np
from scipy.stats import norm


def deflated_sharpe_ratio(sharpe_observed: float,
                          n_returns: int,
                          skew: float,
                          kurtosis: float,
                          n_trials: int) -> dict:
    """
    Args:
        sharpe_observed: Annualized Sharpe from backtest
        n_returns: Number of return observations (e.g., weeks)
        skew: Sample skewness of returns
        kurtosis: Sample excess kurtosis (kurtosis - 3)
        n_trials: Number of strategy variants tested to reach this one

    Returns:
        dict with DSR, expected max Sharpe under null, and p-value
    """
    # Expected max Sharpe under null (Bailey & López de Prado)
    emc = 0.5772156649  # Euler-Mascheroni
    e_max_sr = (
        np.sqrt(2 * np.log(n_trials))
        - (emc + np.log(np.log(n_trials))) / (2 * np.sqrt(2 * np.log(n_trials)))
    ) if n_trials > 1 else 0.0

    # Variance of Sharpe estimator accounting for non-normality
    sr_var = (
        1
        - skew * sharpe_observed
        + ((kurtosis - 1) / 4) * sharpe_observed ** 2
    ) / (n_returns - 1)
    sr_std = np.sqrt(max(sr_var, 1e-10))

    # Deflated Sharpe: prob that observed Sharpe exceeds null expectation
    dsr = norm.cdf((sharpe_observed - e_max_sr) / sr_std)

    return {
        'sharpe_observed': sharpe_observed,
        'expected_max_sharpe_null': e_max_sr,
        'deflated_sharpe_ratio': dsr,
        'p_value': 1 - dsr,
        'significant_at_95': dsr > 0.95,
        'n_returns': n_returns,
        'n_trials': n_trials,
    }


if __name__ == "__main__":
    # Example: Sharpe 1.2, 150 weeks, skew -0.3, excess kurt 2.5, 100 trials
    print(deflated_sharpe_ratio(1.2, 150, -0.3, 2.5, 100))
```

Then in v6 engine, after the final backtest loop completes, add a call to this function with the actual observed statistics and print the result. For `n_trials`, use the user's best honest estimate of how many strategy variants they've tested — probably 50-200 at this point.

### 3.3 Add turnover and capacity report

After the backtest loop, v6 should also print:
- Average weekly turnover (% of portfolio that's new names)
- Top 10 most-traded symbols by notional across the full backtest
- Sharpe computed on the bottom 20% of days by volume (capacity stress test)

If v6 does not already do this, add it as a final reporting step in `azalyst_v6_engine.py`.

---

## PHASE 4 — RUN AND REPORT

### 4.1 Clean rebuild
```bash
rm -rf feature_cache/
rm -rf results_v6/
python build_feature_cache.py --data-dir ./data --out-dir ./feature_cache --workers 4
```

Verify the cache was built with the fixed horizon math. Check one symbol:
```python
import pandas as pd
df = pd.read_parquet('feature_cache/BTCUSDT.parquet')
print(df.columns.tolist())
print(df[['future_ret_15m', 'future_ret_1h', 'future_ret_1d', 'future_ret_5d']].describe())
```

Sanity check: `future_ret_5d` should have std roughly sqrt(5) times `future_ret_1d`. If it's way off, horizons are still wrong.

### 4.2 Run v6
```bash
python azalyst_v6_engine.py --data-dir ./data --feature-dir ./feature_cache --out-dir ./results_v6 --top-n 5 --leverage 1.0
```

Add `--gpu` only if CUDA probe succeeded in Phase 0.

### 4.3 Produce the post-run report

After the run completes, produce a markdown report at `results_v6/V6_FIX_REPORT.md` containing:

1. **Discovery summary** from Phase 0
2. **List of bugs found and fixed** from Phase 1
3. **Findings from investigation tasks** in Phase 2 (one section per file)
4. **Before/after comparison** if the user had a prior run: Sharpe, IC, ICIR, max drawdown, weeks traded
5. **Leak test output** from Phase 3.1
6. **Deflated Sharpe** from Phase 3.2
7. **Turnover / capacity** from Phase 3.3
8. **Honest assessment**: does this strategy look live-worthy?

The honest assessment should NOT be a sales pitch. It should apply these thresholds:
- Deflated Sharpe > 0.95 → signal is statistically distinguishable from noise
- IC > 0.02 sustained across regimes → feature set has genuine information
- Max drawdown < 2x annual return → risk/return is tolerable
- Turnover-adjusted Sharpe > 0.7 → survives realistic fees
- At least 60% of weeks traded → sample size is meaningful

If fewer than 3 of these pass, write clearly: "This strategy is not ready for live capital. The honest baseline is [X Sharpe] and needs [specific improvements] before paper trading." Do not sugarcoat.

---

## RULES OF ENGAGEMENT

1. **Do not invent code for files you have not read.** If `azalyst_v6_engine.py` does something different from what these instructions assume, report the difference and ask.

2. **Do not "optimize" the strategy.** The user explicitly said they want an honest backtest before going live. Do not tune hyperparameters to improve reported metrics. Do not remove regime gating to improve sample size unless you can prove it's forward-looking. Do not lower the fee assumption to improve Sharpe.

3. **Show every diff before applying.** For each file edit, print the before/after so the user can review. In fully autonomous mode, log them to a `patches/` directory.

4. **Do not commit to git.** Let the user review and commit manually.

5. **If a bug is ambiguous, flag it and stop.** The user would rather get 5 correct fixes and 3 questions than 8 fixes with 2 of them wrong.

6. **Use the existing code style.** The codebase uses ASCII box drawings in headers, snake_case, type hints, and docstrings. Match that.

7. **When in doubt, paraphrase back.** Before any risky change, state what you are about to do and why, then do it.

8. **Do not delete anything the user might want.** If a function looks unused, leave it and add a `# TODO: verify unused` comment rather than removing it.

---

## APPENDIX — OPUS'S PRIOR FINDINGS (for your context)

These are the issues Opus flagged from reading the supporting modules only. They are incorporated into the instructions above but listed here in case they help you reason:

1. `ice_series` typo in `azalyst_ic_filter.py` (fixed in 1.1)
2. `compute_icir` mathematically wrong — divides by cross-sectional std instead of time-series std (fixed in 1.2)
3. IC filtering leaks validation data into feature selection (fixed in 1.3)
4. `weighted_r2_score` always called with `weights=None`, making it plain R² despite the Jane Street docstring (fixed in 1.5)
5. Dead forward-looking `max_up`/`max_down` in `azalyst_pump_dump.py` (fixed in 1.6)
6. `RiskManager` stores fees but never uses them (fixed in 1.7)
7. MVO uses raw sample covariance — unusable at N~50 (fixed in 1.8)
8. Horizon math in `build_feature_cache.py` is wrong for >1h timeframes (fixed in 1.9)
9. `dropna(how="all")` is too permissive (fixed in 1.10)
10. Purged K-Fold embargo is one-sided (fixed in 1.4)
11. `train_confidence_model` uses a classifier on a binarized target but then evaluates with accuracy instead of calibrated probabilities — half-broken. Flag this for the user; do not fix without their input.
12. `SignalCombiner._ic_adjusted_weights` uses `1 + 10*mean_ic` multiplier which is aggressive at short lookbacks. Leave as-is but note it in the investigation report.
13. Pump inverse score fused with weekly factor scores at mismatched time scales — architectural issue, not a quick fix. Flag for user.
14. No cross-sectional beta neutralization visible in the modules reviewed — must be verified inside `azalyst_v6_engine.py` per task 2.1.B.
15. `VIEW_TRAINING.py` has UTF-8 mojibake from Windows encoding. Fix: re-save as UTF-8. Low priority.

---

## END OF INSTRUCTIONS

If you finish all phases, the final artifact is `results_v6/V6_FIX_REPORT.md` plus a clean v6 run. The user will decide what to do next based on that report. Your job is to give them accurate information, not to make the numbers look good.
