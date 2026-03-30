"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FACTOR ENGINEERING v5
║        Short-Horizon Regression  |  Reversal-Dominated  |  Pump-Dump        ║
║        v5.0  |  15min/1hr Forecasting  |  Jane Street Inspired              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Design principles (learned from v4 audit):
  1. REVERSAL > MOMENTUM in crypto — proven IC: REV_1H=+0.066, RVOL_1D=+0.132
  2. Regression targets, NOT binary classification
  3. Quantile-rank normalization (Jane Street technique)
  4. Lagged cross-asset returns as features
  5. Pump-dump detection indicators
  6. Per-bar prediction — no week-averaging
"""

import numpy as np
import pandas as pd
from typing import Optional

BARS_PER_HOUR = 12
BARS_PER_DAY  = 288

# ── v5 Feature Set: Reversal-Dominated, Short-Horizon ────────────────────────
# 72 features across 11 categories — heavy on reversal, volatility, microstructure
# Momentum features KEPT but down-weighted via IC-gating at runtime

FEATURE_COLS = [
    # --- Returns (7) --- kept for completeness, IC-gating handles momentum
    'ret_1bar', 'ret_1h', 'ret_4h', 'ret_1d', 'ret_2d', 'ret_3d', 'ret_1w',

    # --- Reversal signals (8) --- PRIMARY alpha source in crypto
    'rev_1h', 'rev_4h', 'rev_1d', 'rev_2d',
    'mean_rev_zscore_1h', 'mean_rev_zscore_4h',
    'overbought_rev', 'oversold_rev',

    # --- Volume (6) ---
    'vol_ratio', 'vol_ret_1h', 'vol_ret_1d', 'obv_change', 'vpt_change', 'vol_momentum',

    # --- Realized Volatility (7) --- second strongest alpha source
    'rvol_1h', 'rvol_4h', 'rvol_1d', 'vol_ratio_1h_1d',
    'atr_norm', 'parkinson_vol', 'garman_klass',

    # --- Technical (10) ---
    'rsi_14', 'rsi_6', 'macd_hist', 'bb_pos', 'bb_width',
    'stoch_k', 'stoch_d', 'cci_14', 'adx_14', 'dmi_diff',

    # --- Microstructure (6) ---
    'vwap_dev', 'amihud', 'kyle_lambda', 'spread_proxy', 'body_ratio', 'candle_dir',

    # --- Price Structure (6) ---
    'wick_top', 'wick_bot', 'price_accel', 'skew_1d', 'kurt_1d', 'max_ret_4h',

    # --- WorldQuant + Cross-Sectional (6) ---
    'wq_alpha001', 'wq_alpha012', 'wq_alpha031', 'wq_alpha098',
    'vol_adjusted_mom', 'trend_consistency',

    # --- Regime (5) ---
    'vol_regime', 'trend_strength', 'corr_btc_proxy', 'hurst_exp', 'fft_strength',

    # --- Memory-Preserving (1) ---
    'frac_diff_close',

    # --- Pump-Dump Indicators (6) --- detect abnormal moves
    'pump_score', 'dump_score', 'vol_spike_zscore',
    'ret_vol_ratio_1h', 'tail_risk_1h', 'abnormal_range',

    # --- Quantile-Ranked Features (4) --- Jane Street technique
    'qrank_ret_1h', 'qrank_rvol_1d', 'qrank_rev_1h', 'qrank_vol_ratio',
]

# Legacy alias for backward compat
LEGACY_FEATURE_COLS = [
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
    v5 feature builder — reversal-dominated, pump-dump aware, quantile-ranked.
    72 features across 11 categories.

    Key changes from v4:
    - 8 reversal features (was 2) — primary alpha source
    - 6 pump-dump indicators — detect abnormal moves
    - 4 quantile-ranked features — Jane Street technique
    - Removed cs_momentum/cs_reversal aliases (now explicit)
    """
    bph, bpd = BARS_PER_HOUR, BARS_PER_DAY
    if timeframe != '5min':
        mins = {'1h':60,'4h':240,'1d':1440}.get(timeframe, 5)
        bph  = max(1, 60 // mins)
        bpd  = max(1, 1440 // mins)

    c = df['close'].astype(np.float32)
    o = df['open'].astype(np.float32)
    h = df['high'].astype(np.float32)
    l = df['low'].astype(np.float32)
    v = df['volume'].astype(np.float32)
    f = pd.DataFrame(index=df.index, dtype=np.float32)

    # ── Returns (7) ───────────────────────────────────────────────────────────
    lr = np.log(c / c.shift(1))
    f['ret_1bar'] = lr
    f['ret_1h']   = np.log(c / c.shift(bph))
    f['ret_4h']   = np.log(c / c.shift(bph * 4))
    f['ret_1d']   = np.log(c / c.shift(bpd))
    f['ret_2d']   = np.log(c / c.shift(bpd * 2))
    f['ret_3d']   = np.log(c / c.shift(bpd * 3))
    f['ret_1w']   = np.log(c / c.shift(bpd * 5))

    # ── Reversal Signals (8) — PRIMARY alpha source in crypto ─────────────────
    f['rev_1h'] = -f['ret_1h']   # negative of momentum = reversal
    f['rev_4h'] = -f['ret_4h']
    f['rev_1d'] = -f['ret_1d']
    f['rev_2d'] = -f['ret_2d']

    # Z-score of price deviation from rolling mean (mean-reversion strength)
    ma_1h = c.rolling(bph, min_periods=max(2, bph // 2)).mean()
    std_1h = c.rolling(bph, min_periods=max(2, bph // 2)).std()
    f['mean_rev_zscore_1h'] = -(c - ma_1h) / std_1h.replace(0, np.nan)

    ma_4h = c.rolling(bph * 4, min_periods=bph).mean()
    std_4h = c.rolling(bph * 4, min_periods=bph).std()
    f['mean_rev_zscore_4h'] = -(c - ma_4h) / std_4h.replace(0, np.nan)

    # RSI-based reversal: overbought (>0.7) → expect down, oversold (<0.3) → expect up
    rsi14 = _rsi(c, 14) / 100.0
    f['overbought_rev'] = -(rsi14 - 0.5).clip(lower=0) * 2  # negative when overbought
    f['oversold_rev']   =  (0.5 - rsi14).clip(lower=0) * 2  # positive when oversold

    # ── Volume (6) ────────────────────────────────────────────────────────────
    av = v.rolling(bpd, min_periods=bph).mean()
    f['vol_ratio']   = v / av.replace(0, np.nan)
    f['vol_ret_1h']  = np.log(v / v.shift(bph).replace(0, np.nan))
    f['vol_ret_1d']  = np.log(v / v.shift(bpd).replace(0, np.nan))
    obv = (np.sign(lr) * v).cumsum()
    f['obv_change']  = obv.diff(bph) / (obv.abs().rolling(bpd, min_periods=bph).mean() + 1e-8)
    vpt = (lr * v).cumsum()
    f['vpt_change']  = vpt.diff(bph)
    f['vol_momentum'] = v.rolling(bph, min_periods=2).mean() / v.rolling(bpd, min_periods=bph).mean()

    # ── Realized Volatility (7) ───────────────────────────────────────────────
    f['rvol_1h']  = lr.rolling(bph, min_periods=max(2, bph//2)).std()
    f['rvol_4h']  = lr.rolling(bph * 4, min_periods=bph).std()
    f['rvol_1d']  = lr.rolling(bpd, min_periods=bph).std()
    f['vol_ratio_1h_1d'] = f['rvol_1h'] / f['rvol_1d'].replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    f['atr_norm'] = atr14 / c.replace(0, np.nan)
    f['parkinson_vol'] = np.sqrt(
        1/(4*np.log(2)) * np.log(h/l.replace(0, np.nan))**2
    ).rolling(bpd, min_periods=bph).mean()
    gk = (0.5 * np.log(h/l.replace(0, np.nan))**2
          - (2*np.log(2)-1) * np.log(c/o.replace(0, np.nan))**2)
    f['garman_klass'] = gk.rolling(bpd, min_periods=bph).mean()

    # ── Technical (10) ────────────────────────────────────────────────────────
    f['rsi_14'] = rsi14
    f['rsi_6']  = _rsi(c, 6) / 100.0
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

    # ── Microstructure (6) ────────────────────────────────────────────────────
    vwap = ((tp * v).rolling(bpd, min_periods=bph).sum()
            / v.rolling(bpd, min_periods=bph).sum().replace(0, np.nan))
    f['vwap_dev']     = (c - vwap) / c.replace(0, np.nan)
    f['amihud']       = (lr.abs() / v.replace(0, np.nan)).rolling(bpd, min_periods=bph).mean()
    f['kyle_lambda']  = (lr.abs() / (v * c).replace(0, np.nan)).rolling(bpd, min_periods=bph).mean()
    f['spread_proxy'] = (h - l) / c.replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    f['body_ratio'] = (c - o).abs() / rng
    f['candle_dir'] = np.sign(c - o)

    # ── Price Structure (6) ───────────────────────────────────────────────────
    f['wick_top']   = (h - c.clip(lower=o)) / rng
    f['wick_bot']   = (c.clip(upper=o) - l) / rng
    m1 = c.pct_change(bph)
    f['price_accel'] = m1 - m1.shift(bph)
    f['skew_1d']    = lr.rolling(bpd, min_periods=max(4, bph)).skew()
    f['kurt_1d']    = lr.rolling(bpd, min_periods=max(4, bph)).kurt()
    f['max_ret_4h'] = lr.rolling(max(2, bph*4), min_periods=bph).max()

    # ── WorldQuant + Cross-Sectional (6) ──────────────────────────────────────
    f['wq_alpha001'] = np.sign(f['ret_1d']) * f['rvol_1d']
    f['wq_alpha012'] = np.sign(v.diff()) * (-lr)
    _c_rank = c.rank()
    pv_corr = _c_rank.rolling(bpd, min_periods=bph).corr(v)
    f['wq_alpha031'] = -pv_corr
    ret_5d = np.log(c / c.shift(bpd))
    f['wq_alpha098'] = ret_5d / (f['rvol_1d'].replace(0, np.nan))
    f['vol_adjusted_mom'] = f['ret_4h'] / f['rvol_4h'].replace(0, np.nan)
    s = np.sign(lr)
    f['trend_consistency'] = s.rolling(min(48, bpd//2), min_periods=max(2, bph)).sum()

    # ── Regime (5) ────────────────────────────────────────────────────────────
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

    # ── Memory-Preserving (1) — Fractional Differentiation (AFML Ch. 5) ──────
    f['frac_diff_close'] = frac_diff_ffd(np.log(c.clip(lower=1e-10)), d=0.4)

    # ── Pump-Dump Indicators (6) — detect abnormal moves ─────────────────────
    # Pump score: large positive return with volume spike
    vol_ma = v.rolling(bpd, min_periods=bph).mean()
    vol_std = v.rolling(bpd, min_periods=bph).std()
    vol_z = (v - vol_ma) / vol_std.replace(0, np.nan)

    ret_1h_abs = f['ret_1h'].abs()
    ret_std_1d = lr.rolling(bpd, min_periods=bph).std()

    # Pump: big up move + volume spike
    f['pump_score'] = (f['ret_1h'].clip(lower=0) / ret_std_1d.replace(0, np.nan)) * vol_z.clip(lower=0)
    # Dump: big down move + volume spike
    f['dump_score'] = ((-f['ret_1h']).clip(lower=0) / ret_std_1d.replace(0, np.nan)) * vol_z.clip(lower=0)
    # Volume spike z-score
    f['vol_spike_zscore'] = vol_z
    # Return/volatility ratio (detects outsized moves)
    f['ret_vol_ratio_1h'] = f['ret_1h'] / f['rvol_1h'].replace(0, np.nan)
    # Tail risk: max absolute return in last hour / daily vol
    f['tail_risk_1h'] = lr.abs().rolling(bph, min_periods=2).max() / f['rvol_1d'].replace(0, np.nan)
    # Abnormal range: (high-low)/typical range
    typical_range = (h - l).rolling(bpd, min_periods=bph).mean()
    f['abnormal_range'] = (h - l) / typical_range.replace(0, np.nan)

    # ── Quantile-Ranked Features (4) — Jane Street technique ─────────────────
    # Rolling quantile rank normalizes features to [0, 1] — robust to outliers
    qrank_window = bpd * 5  # 5-day rolling window
    f['qrank_ret_1h']    = f['ret_1h'].rolling(qrank_window, min_periods=bpd).rank(pct=True)
    f['qrank_rvol_1d']   = f['rvol_1d'].rolling(qrank_window, min_periods=bpd).rank(pct=True)
    f['qrank_rev_1h']    = f['rev_1h'].rolling(qrank_window, min_periods=bpd).rank(pct=True)
    f['qrank_vol_ratio'] = f['vol_ratio'].rolling(qrank_window, min_periods=bpd).rank(pct=True)

    return f.replace([np.inf, -np.inf], np.nan).shift(1).astype(np.float32)
