"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FACTOR ENGINEERING v2
║        65 Quantitative Factors  |  Vectorized & Optimized                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from typing import Optional

BARS_PER_HOUR = 12
BARS_PER_DAY  = 288

FEATURE_COLS = [
    'ret_1bar','ret_1h','ret_4h','ret_1d','ret_2d','ret_3d','ret_1w',
    'vol_ratio','vol_ret_1h','vol_ret_1d','obv_change','vpt_change','vol_momentum',
    'rvol_1h','rvol_4h','rvol_1d','vol_ratio_1h_1d','atr_norm','parkinson_vol','garman_klass',
    'rsi_14','rsi_6','macd_hist','bb_pos','bb_width','stoch_k','stoch_d','cci_14','adx_14','dmi_diff',
    'vwap_dev','amihud','kyle_lambda','spread_proxy','body_ratio','candle_dir',
    'wick_top','wick_bot','price_accel','skew_1d','kurt_1d','max_ret_4h',
    'wq_alpha001','wq_alpha012','wq_alpha031','wq_alpha098',
    'cs_momentum','cs_reversal','vol_adjusted_mom','trend_consistency',
    'vol_regime','trend_strength','corr_btc_proxy','hurst_exp','fft_strength',
    'frac_diff_close',
]

def _rsi(s, n):
    d  = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def frac_diff_ffd(series: pd.Series, d: float = 0.4,
                  threshold: float = 1e-5) -> pd.Series:
    """
    Fixed-Width Window Fractional Differentiation (AFML Ch. 5).

    d ∈ (0, 1): d=0 is raw price (non-stationary, max memory),
    d=1.0 is standard returns (stationary, zero memory).
    d≈0.4 balances stationarity with memory retention.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1], dtype=np.float64)   # oldest weight first
    width = len(w)

    arr = series.values.astype(np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(width - 1, len(arr)):
        chunk = arr[i - width + 1 : i + 1]
        if np.isnan(chunk).any():
            continue
        out[i] = np.dot(w, chunk)
    return pd.Series(out, index=series.index, dtype=np.float32)

def build_features(df, timeframe='5min'):
    """
    Standard v2 feature builder reconciled with latest snippet.
    """
    bph, bpd = BARS_PER_HOUR, BARS_PER_DAY
    if timeframe != '5min':
        mins = {'1h':60,'4h':240,'1d':1440}.get(timeframe, 5)
        bph  = max(1, 60 // mins)
        bpd  = max(1, 1440 // mins)

    c, o, h, l, v = df['close'].astype(np.float32), df['open'].astype(np.float32), df['high'].astype(np.float32), df['low'].astype(np.float32), df['volume'].astype(np.float32)
    f = pd.DataFrame(index=df.index, dtype=np.float32)

    lr = np.log(c / c.shift(1))
    f['ret_1bar'] = lr
    f['ret_1h']   = np.log(c / c.shift(bph))
    f['ret_4h']   = np.log(c / c.shift(bph * 4))
    f['ret_1d']   = np.log(c / c.shift(bpd))
    f['ret_2d']   = np.log(c / c.shift(bpd * 2))
    f['ret_3d']   = np.log(c / c.shift(bpd * 3))
    f['ret_1w']   = np.log(c / c.shift(bpd * 5))

    av = v.rolling(bpd, min_periods=bph).mean()
    f['vol_ratio']   = v / av.replace(0, np.nan)
    f['vol_ret_1h']  = np.log(v / v.shift(bph).replace(0, np.nan))
    f['vol_ret_1d']  = np.log(v / v.shift(bpd).replace(0, np.nan))
    obv = (np.sign(lr) * v).cumsum()
    f['obv_change']  = obv.diff(bph) / (obv.abs().rolling(bpd, min_periods=bph).mean() + 1e-8)
    vpt = (lr * v).cumsum()
    f['vpt_change']  = vpt.diff(bph)
    f['vol_momentum'] = v.rolling(bph, min_periods=2).mean() / v.rolling(bpd, min_periods=bph).mean()

    f['rvol_1h']  = lr.rolling(bph, min_periods=max(2, bph//2)).std()
    f['rvol_4h']  = lr.rolling(bph * 4, min_periods=bph).std()
    f['rvol_1d']  = lr.rolling(bpd, min_periods=bph).std()
    f['vol_ratio_1h_1d'] = f['rvol_1h'] / f['rvol_1d'].replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    f['atr_norm'] = atr14 / c.replace(0, np.nan)
    f['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * np.log(h/l.replace(0,np.nan))**2).rolling(bpd, min_periods=bph).mean()
    gk = (0.5 * np.log(h/l.replace(0,np.nan))**2 - (2*np.log(2)-1) * np.log(c/o.replace(0,np.nan))**2)
    f['garman_klass'] = gk.rolling(bpd, min_periods=bph).mean()

    f['rsi_14'] = _rsi(c, 14) / 100.0
    f['rsi_6']  = _rsi(c,  6) / 100.0
    macd_line   = _ema(c, 12) - _ema(c, 26)
    signal_line = _ema(macd_line, 9)
    f['macd_hist'] = (macd_line - signal_line) / c.replace(0, np.nan)
    ma20  = c.rolling(20, min_periods=10).mean()
    std20 = c.rolling(20, min_periods=10).std(ddof=0)
    bw    = (4 * std20).replace(0, np.nan)
    f['bb_pos']   = ((c - (ma20 - 2*std20)) / bw).clip(0, 1)
    f['bb_width'] = bw / ma20.replace(0, np.nan)
    low14  = l.rolling(14, min_periods=7).min()
    high14 = h.rolling(14, min_periods=7).max()
    k = ((c - low14) / (high14 - low14).replace(0, np.nan) * 100).clip(0, 100)
    f['stoch_k'] = k / 100.0
    f['stoch_d'] = k.rolling(3).mean() / 100.0
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(14, min_periods=7).mean()
    tp_mad = (tp - tp_ma).abs().rolling(14, min_periods=7).mean()
    f['cci_14'] = ((tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))).clip(-3, 3) / 3
    plus_dm  = pd.Series(np.where((h.diff()>0) & (h.diff()>(-l.diff())), h.diff(), 0), index=df.index)
    minus_dm = pd.Series(np.where((-l.diff()>0) & (-l.diff()>h.diff()), -l.diff(), 0), index=df.index)
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    f['adx_14']  = dx.ewm(span=14, adjust=False).mean() / 100.0
    f['dmi_diff'] = (plus_di - minus_di) / 100.0

    vwap = ((tp * v).rolling(bpd, min_periods=bph).sum() / v.rolling(bpd, min_periods=bph).sum().replace(0, np.nan))
    f['vwap_dev']     = (c - vwap) / c.replace(0, np.nan)
    f['amihud']       = (lr.abs() / v.replace(0, np.nan)).rolling(bpd, min_periods=bph).mean()
    f['kyle_lambda']  = (lr.abs() / (v * c).replace(0, np.nan)).rolling(bpd, min_periods=bph).mean()
    f['spread_proxy'] = (h - l) / c.replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    f['body_ratio'] = (c - o).abs() / rng
    f['candle_dir'] = np.sign(c - o)

    f['wick_top']   = (h - c.clip(lower=o)) / rng
    f['wick_bot']   = (c.clip(upper=o) - l) / rng
    m1 = c.pct_change(bph)
    f['price_accel'] = m1 - m1.shift(bph)
    f['skew_1d']    = lr.rolling(bpd, min_periods=max(4, bph)).skew()
    f['kurt_1d']    = lr.rolling(bpd, min_periods=max(4, bph)).kurt()
    f['max_ret_4h'] = lr.rolling(max(2, bph*4), min_periods=bph).max()

    f['wq_alpha001'] = np.sign(f['ret_1d']) * f['rvol_1d']
    f['wq_alpha012'] = np.sign(v.diff()) * (-lr)
    _c_rank = c.rank()
    pv_corr = _c_rank.rolling(bpd, min_periods=bph).corr(v)
    f['wq_alpha031'] = -pv_corr
    ret_5d = np.log(c / c.shift(bpd))
    f['wq_alpha098'] = ret_5d / (f['rvol_1d'].replace(0, np.nan))
    f['cs_momentum']     = f['ret_4h']           
    f['cs_reversal']     = -f['ret_1d']          
    f['vol_adjusted_mom'] = f['ret_4h'] / f['rvol_4h'].replace(0, np.nan)
    s = np.sign(lr)
    f['trend_consistency'] = s.rolling(min(48, bpd//2), min_periods=max(2,bph)).sum()

    f['vol_regime'] = f['rvol_1d'].rolling(bpd*5, min_periods=bpd).rank(pct=True)
    f['trend_strength'] = f['adx_14']
    f['corr_btc_proxy'] = lr.rolling(bpd, min_periods=bph).corr(lr.shift(1))
    
    rs_vals = []
    close_arr = c.values
    for i in range(len(close_arr)):
        if i < bpd:
            rs_vals.append(np.nan)
            continue
        seg = close_arr[i-bpd:i]
        mean_seg = np.mean(seg)
        deviations = np.cumsum(seg - mean_seg)
        r = np.max(deviations) - np.min(deviations)
        s_val = np.std(seg)
        rs_vals.append(r / s_val if s_val > 0 else np.nan)
    f['hurst_exp'] = pd.Series(rs_vals, index=df.index, dtype=np.float32)
    
    fft_strength = pd.Series(index=df.index, dtype=np.float32)
    win = min(bpd, 288)
    for i in range(len(close_arr)):
        if i < win:
            fft_strength.iloc[i] = np.nan
            continue
        seg = close_arr[i-win:i]
        seg_ret = np.diff(np.log(seg + 1e-10))
        fft_mag = np.abs(np.fft.rfft(seg_ret))
        fft_strength.iloc[i] = float(np.max(fft_mag[1:]) / (np.mean(fft_mag[1:]) + 1e-10))
    f['fft_strength'] = fft_strength.astype(np.float32)

    # Fractional differentiation of log-price (AFML Ch. 5)
    # Preserves price level memory while achieving stationarity
    f['frac_diff_close'] = frac_diff_ffd(np.log(c.clip(lower=1e-10)), d=0.4)

    return f.replace([np.inf, -np.inf], np.nan).shift(1).astype(np.float32)
