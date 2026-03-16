"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    TIMEFRAME UTILITIES
║  FIX: All bar-count constants are now derived from the actual candle TF.   ║
║  Previously hardcoded to 5-min (BARS_PER_HOUR=12, BARS_PER_DAY=288)       ║
║  causing total NaN floods when scoring weekly/daily candles.               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import re


def get_tf_constants(resample_str: str) -> tuple[int, int, int]:
    """
    Convert a pandas resample string to (bars_per_hour, bars_per_day, horizon_bars).

    horizon_bars = "4-hour equivalent" number of bars (floored at 1).

    Examples
    --------
    get_tf_constants('5min')  → (12, 288, 48)
    get_tf_constants('4h')    → (1,    6,  1)
    get_tf_constants('1D')    → (1,    1,  1)
    get_tf_constants('1W')    → (1,    1,  1)
    """
    s = resample_str.lower().strip()

    _map = {
        '1min': 1,   '1t': 1,
        '3min': 3,   '3t': 3,
        '5min': 5,   '5t': 5,
        '15min': 15, '15t': 15,
        '30min': 30, '30t': 30,
        '1h': 60,    '60min': 60, '60t': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,  '1b': 1440,  'd': 1440,
        '1w': 10080, 'w': 10080,  'w-mon': 10080, '1w-mon': 10080,
    }
    mins = _map.get(s)

    if mins is None:
        m = re.match(r'^(\d+)([a-z]+)', s)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            unit_mins = {'min': 1, 't': 1, 'h': 60, 'd': 1440, 'w': 10080}.get(unit, 1)
            mins = n * unit_mins
        else:
            mins = 5  # safe default — treat as 5-min

    bph = max(1, 60   // mins)   # bars per hour
    bpd = max(1, 1440 // mins)   # bars per day
    hor = max(1, 240  // mins)   # 4h-equivalent horizon
    return bph, bpd, hor


def is_5min(resample_str: str) -> bool:
    """True if the resample string resolves to 5-minute bars."""
    return resample_str.lower().strip() in ('5min', '5t', '5m')
