"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  WEEKLY SELF-IMPROVING LOOP  v2  (Year 3)
║        Out-of-Sample Walk-Forward  |  Feature Store Engine                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from azalyst_factors_v2 import FEATURE_COLS

def _simulate_trade(ohlcv_slice, signal, entry_price, max_bars, sl_p, tp_p):
    """
    Trade simulation logic as defined in the final v2 snippet.
    """
    if len(ohlcv_slice) < 2:
        return entry_price, 'horizon'
    fut   = ohlcv_slice.iloc[1:max_bars+1]
    lows  = fut['low'].values
    highs = fut['high'].values
    if signal == 'BUY':
        sl_hit = np.where(lows  <= sl_p)[0]
        tp_hit = np.where(highs >= tp_p)[0]
    else:
        sl_hit = np.where(highs >= sl_p)[0]
        tp_hit = np.where(lows  <= tp_p)[0]
    sl_bar = sl_hit[0] if len(sl_hit) else max_bars + 1
    tp_bar = tp_hit[0] if len(tp_hit) else max_bars + 1
    if sl_bar < tp_bar and sl_bar <= max_bars: return sl_p,  'stop_loss'
    if tp_bar < sl_bar and tp_bar <= max_bars: return tp_p,  'take_profit'
    return float(fut.iloc[min(max_bars-1, len(fut)-1)]['close']), 'horizon'

def cross_sectional_rank_signals(df, cols, top_q=0.15):
    """
    Assigns BUY/SELL signals based on quantile ranking of probabilities.
    """
    def assign_signals(grp):
        n_long = max(1, int(len(grp) * top_q))
        if len(grp) >= 5:
            grp = grp.sort_values('prob')
            grp.iloc[-n_long:, grp.columns.get_loc('signal')] = 'BUY'
            grp.iloc[:n_long,  grp.columns.get_loc('signal')] = 'SELL'
        return grp
    
    res = df.copy()
    res['signal'] = 'HOLD'
    return res.groupby(level=0, group_keys=False).apply(assign_signals)
