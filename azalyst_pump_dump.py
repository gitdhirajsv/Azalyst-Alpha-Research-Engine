"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    PUMP-DUMP DETECTOR v1
║        Real-Time Anomaly Detection  |  Volume-Price Divergence              ║
║        v1.0  |  Multi-Signal Scoring  |  Configurable Thresholds            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Detects pump-and-dump schemes using multiple concurrent signals:
  1. Price spike: outsized return relative to trailing volatility
  2. Volume spike: volume z-score above threshold
  3. Spread expansion: abnormal high-low range
  4. Pattern: sharp up followed by sharp down (or vice versa)
  5. Composite score: weighted combination of all signals

Returns per-bar scores in [0, 1] where:
  0.0 = normal market activity
  1.0 = high confidence pump-dump detected
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# ── Configuration ─────────────────────────────────────────────────────────────

PUMP_DUMP_CONFIG = {
    "ret_zscore_thresh": 3.0,       # return z-score to flag
    "vol_zscore_thresh": 2.5,       # volume z-score to flag
    "range_zscore_thresh": 2.5,     # range z-score to flag
    "reversal_window_bars": 12,     # 1hr window to detect reversal pattern
    "reversal_thresh": 0.6,         # fraction of pump that reverses
    "lookback_bars_vol": 288,       # 1 day for vol baseline
    "lookback_bars_ret": 288,       # 1 day for return baseline
    "min_bars_warmup": 288,         # minimum bars before scoring
    "pump_weight": 0.3,             # weight for price spike component
    "vol_weight": 0.25,             # weight for volume spike component
    "range_weight": 0.2,            # weight for range expansion component
    "reversal_weight": 0.25,        # weight for reversal pattern component
}


# ── Core Detection Functions ──────────────────────────────────────────────────

def compute_pump_dump_scores(df: pd.DataFrame,
                             config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compute per-bar pump-dump scores from OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns [open, high, low, close, volume] and DatetimeIndex
    config : optional override for PUMP_DUMP_CONFIG

    Returns
    -------
    DataFrame with columns:
        pump_dump_score : float [0, 1] composite score
        is_pump         : bool - detected pump event
        is_dump         : bool - detected dump event
        price_spike_z   : float - return z-score
        volume_spike_z  : float - volume z-score
        range_spike_z   : float - range z-score
        reversal_score  : float [0, 1] - reversal pattern strength
    """
    cfg = {**PUMP_DUMP_CONFIG, **(config or {})}

    c = df['close'].astype(np.float64)
    o = df['open'].astype(np.float64)
    h = df['high'].astype(np.float64)
    l = df['low'].astype(np.float64)
    v = df['volume'].astype(np.float64)

    lr = np.log(c / c.shift(1))
    n = len(df)
    lb_vol = cfg['lookback_bars_vol']
    lb_ret = cfg['lookback_bars_ret']
    min_warmup = cfg['min_bars_warmup']

    result = pd.DataFrame(index=df.index)

    # --- 1. Price Spike Z-Score ---
    ret_1h = np.log(c / c.shift(12))  # 1-hour return
    ret_mean = ret_1h.rolling(lb_ret, min_periods=lb_ret // 2).mean()
    ret_std = ret_1h.rolling(lb_ret, min_periods=lb_ret // 2).std()
    price_z = (ret_1h - ret_mean) / ret_std.replace(0, np.nan)
    result['price_spike_z'] = price_z.astype(np.float32)

    # --- 2. Volume Spike Z-Score ---
    vol_mean = v.rolling(lb_vol, min_periods=lb_vol // 2).mean()
    vol_std = v.rolling(lb_vol, min_periods=lb_vol // 2).std()
    vol_z = (v - vol_mean) / vol_std.replace(0, np.nan)
    result['volume_spike_z'] = vol_z.astype(np.float32)

    # --- 3. Range Spike Z-Score ---
    bar_range = h - l
    range_mean = bar_range.rolling(lb_vol, min_periods=lb_vol // 2).mean()
    range_std = bar_range.rolling(lb_vol, min_periods=lb_vol // 2).std()
    range_z = (bar_range - range_mean) / range_std.replace(0, np.nan)
    result['range_spike_z'] = range_z.astype(np.float32)

    # --- 4. Reversal Pattern Detection ---
    rev_window = cfg['reversal_window_bars']
    rev_thresh = cfg['reversal_thresh']

    # Reversal: price went up sharply then came back down (or vice versa)
    # Use trailing data only (no look-ahead)
    trailing_max = ret_1h.shift(1).rolling(rev_window, min_periods=1).max()
    trailing_min = ret_1h.shift(1).rolling(rev_window, min_periods=1).min()

    # Up-then-down: previous max was high, current is low
    up_then_down = (trailing_max > 0) & (ret_1h < -trailing_max * rev_thresh)
    # Down-then-up: previous min was low, current is high
    down_then_up = (trailing_min < 0) & (ret_1h > -trailing_min * rev_thresh)

    reversal_raw = up_then_down.astype(float) + down_then_up.astype(float)
    result['reversal_score'] = reversal_raw.clip(0, 1).astype(np.float32)

    # --- 5. Composite Score ---
    w_pump = cfg['pump_weight']
    w_vol = cfg['vol_weight']
    w_range = cfg['range_weight']
    w_rev = cfg['reversal_weight']

    # Normalize z-scores to [0, 1] using sigmoid-like transform
    def _zscore_to_prob(z, thresh):
        """Convert z-score to probability-like score in [0, 1]."""
        # Centered at threshold, steep sigmoid
        return 1.0 / (1.0 + np.exp(-(z.abs() - thresh) * 2.0))

    price_prob = _zscore_to_prob(price_z, cfg['ret_zscore_thresh'])
    vol_prob = _zscore_to_prob(vol_z, cfg['vol_zscore_thresh'])
    range_prob = _zscore_to_prob(range_z, cfg['range_zscore_thresh'])

    composite = (w_pump * price_prob +
                 w_vol * vol_prob +
                 w_range * range_prob +
                 w_rev * result['reversal_score'])

    # Zero out during warmup period
    composite.iloc[:min_warmup] = 0.0
    result['pump_dump_score'] = composite.clip(0, 1).astype(np.float32)

    # --- 6. Binary Flags ---
    thresh_ret = cfg['ret_zscore_thresh']
    thresh_vol = cfg['vol_zscore_thresh']

    result['is_pump'] = (
        (price_z > thresh_ret) &
        (vol_z > thresh_vol * 0.5) &
        (result['pump_dump_score'] > 0.5)
    )
    result['is_dump'] = (
        (price_z < -thresh_ret) &
        (vol_z > thresh_vol * 0.5) &
        (result['pump_dump_score'] > 0.5)
    )

    return result


def classify_pump_dump_regime(scores: pd.DataFrame,
                              window: int = 12) -> pd.Series:
    """
    Classify each bar into a pump-dump regime.

    Returns
    -------
    Series with values: 'NORMAL', 'PUMP_ACTIVE', 'DUMP_ACTIVE', 'POST_PUMP_DUMP'
    """
    regime = pd.Series('NORMAL', index=scores.index, dtype='object')

    if 'is_pump' in scores.columns:
        pump_active = scores['is_pump'].rolling(window, min_periods=1).sum() > 0
        dump_active = scores['is_dump'].rolling(window, min_periods=1).sum() > 0

        # Post-pump-dump: recent pump followed by dump (or vice versa)
        recent_pump = scores['is_pump'].rolling(window * 3, min_periods=1).sum() > 0
        recent_dump = scores['is_dump'].rolling(window * 3, min_periods=1).sum() > 0
        post_pd = recent_pump & recent_dump & ~pump_active & ~dump_active

        regime[pump_active] = 'PUMP_ACTIVE'
        regime[dump_active] = 'DUMP_ACTIVE'
        regime[post_pd] = 'POST_PUMP_DUMP'

    return regime


def filter_pump_dump_symbols(symbols_scores: Dict[str, pd.DataFrame],
                             timestamp: pd.Timestamp,
                             threshold: float = 0.6) -> set:
    """
    Return set of symbols currently in pump-dump state that should be avoided.

    Parameters
    ----------
    symbols_scores : dict of {symbol: pump_dump_scores DataFrame}
    timestamp : current time
    threshold : pump_dump_score above this → avoid

    Returns
    -------
    Set of symbol names to exclude from trading
    """
    avoid = set()
    for sym, scores in symbols_scores.items():
        if timestamp in scores.index:
            row = scores.loc[timestamp]
        else:
            # Find nearest bar
            mask = scores.index <= timestamp
            if not mask.any():
                continue
            row = scores.iloc[mask.values.nonzero()[0][-1]]

        if row.get('pump_dump_score', 0) > threshold:
            avoid.add(sym)
        elif row.get('is_pump', False) or row.get('is_dump', False):
            avoid.add(sym)

    return avoid


def compute_pump_dump_summary(scores: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for a symbol's pump-dump scores.
    """
    if len(scores) == 0:
        return {"n_pumps": 0, "n_dumps": 0, "mean_score": 0.0, "max_score": 0.0}

    return {
        "n_pumps": int(scores.get('is_pump', pd.Series(dtype=bool)).sum()),
        "n_dumps": int(scores.get('is_dump', pd.Series(dtype=bool)).sum()),
        "mean_score": float(scores['pump_dump_score'].mean()),
        "max_score": float(scores['pump_dump_score'].max()),
        "pct_above_05": float((scores['pump_dump_score'] > 0.5).mean() * 100),
        "pct_above_07": float((scores['pump_dump_score'] > 0.7).mean() * 100),
    }
