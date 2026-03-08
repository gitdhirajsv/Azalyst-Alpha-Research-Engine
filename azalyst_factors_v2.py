"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    EXTENDED FACTOR LIBRARY v2          
║        35 Quantitative Factors  |  Crypto-Native Alpha  |  Fully Vectorized ║
╚══════════════════════════════════════════════════════════════════════════════╝

Factor Universe (35 factors across 7 categories)
─────────────────────────────────────────────────
Category      Factors
────────────  ─────────────────────────────────────────────────────────────
Momentum      MOM_1H  MOM_4H  MOM_1D  MOM_3D  MOM_1W  MOM_2W  MOM_30D
              OVERNIGHT  CLOSE_TO_OPEN
Reversal      REV_1H  REV_4H  REV_1D
Volatility    RVOL_1D  RVOL_1W  VOL_OF_VOL  DOWNVOL_1W
Liquidity     AMIHUD  CORWIN_SCHULTZ  TURNOVER  VOL_RATIO  VOL_MOM_1D
Microstructure MAX_RET  SKEW_1W  KURT_1W  PRICE_ACCEL  VOLUME_SURPRISE
               VWAP_DEV  BTC_BETA  IDIO_MOM
Technical     TREND_48  BB_POS  RSI_RANK  MA_SLOPE  WEEK52_HIGH  WEEK52_LOW

Research basis
──────────────
Liu & Tsyvinski (2021)   — crypto momentum
Fang & Li (2024)         — CTREND t-stat 4.22
Amihud (2002)            — illiquidity premium
Corwin & Schultz (2012)  — bid-ask spread from OHLC
Bali et al. (2011)       — MAX effect (lottery coins)
Harvey & Siddique (2000) — skewness premium
Hong & Stein (1999)      — information diffusion (MOM_30D skip bias)
Adrian et al. (2019)     — downside vol premium
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BARS_PER_HOUR = 12
BARS_PER_DAY  = 288
BARS_PER_WEEK = 2016

FACTOR_NAMES_V2 = [
    # Momentum (9)
    "MOM_1H", "MOM_4H", "MOM_1D", "MOM_3D", "MOM_1W", "MOM_2W", "MOM_30D",
    "OVERNIGHT", "CLOSE_TO_OPEN",
    # Reversal (3)
    "REV_1H", "REV_4H", "REV_1D",
    # Volatility (4)
    "RVOL_1D", "RVOL_1W", "VOL_OF_VOL", "DOWNVOL_1W",
    # Liquidity (5)
    "AMIHUD", "CORWIN_SCHULTZ", "TURNOVER", "VOL_RATIO", "VOL_MOM_1D",
    # Microstructure (8)
    "MAX_RET", "SKEW_1W", "KURT_1W", "PRICE_ACCEL",
    "VOLUME_SURPRISE", "VWAP_DEV", "BTC_BETA", "IDIO_MOM",
    # Technical (6)
    "TREND_48", "BB_POS", "RSI_RANK", "MA_SLOPE", "WEEK52_HIGH", "WEEK52_LOW",
]


# ─────────────────────────────────────────────────────────────────────────────
#  EXTENDED FACTOR ENGINE v2
# ─────────────────────────────────────────────────────────────────────────────

class FactorEngineV2:
    """
    35-factor quantitative library.

    All methods accept pd.DataFrame panels (T × N) and return the same shape
    with cross-sectional percentile ranks (0→1) applied at each timestamp.

    New factors vs v1
    ─────────────────
    MOM_30D         : 30-day momentum skipping last 1 day (skip-1-month standard)
    OVERNIGHT       : Close-to-open return (captures news/flow gaps)
    CLOSE_TO_OPEN   : Open-to-close intraday return (clean of gap bias)
    REV_1D          : 1-day reversal (liquidity provider alpha)
    DOWNVOL_1W      : Downside semi-deviation (Adrian et al. 2019)
    CORWIN_SCHULTZ  : Bid-ask spread proxy from daily High/Low (microstructure)
    TURNOVER        : Volume / 30D avg volume (relative activity)
    KURT_1W         : Excess kurtosis (fat-tail coins underperform)
    VOLUME_SURPRISE : Residual volume vs AR(1) model (unexpected activity)
    VWAP_DEV        : Price distance from rolling VWAP (mean reversion signal)
    BTC_BETA        : Rolling beta to BTC (crypto systematic risk)
    IDIO_MOM        : Momentum orthogonal to BTC (pure idiosyncratic alpha)
    MA_SLOPE        : Slope of 48-bar EMA (trend strength)
    WEEK52_HIGH     : Distance from 52-week high (near-ATH effect)
    WEEK52_LOW      : Distance from 52-week low (recovery rebound)
    """

    def __init__(self, bph: int = BARS_PER_HOUR):
        self.bph = bph

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ret(self, close: pd.DataFrame, n: int) -> pd.DataFrame:
        return close.pct_change(n)

    def _log_ret(self, close: pd.DataFrame) -> pd.DataFrame:
        return np.log(close / close.shift(1))

    def _ema(self, df: pd.DataFrame, span: int) -> pd.DataFrame:
        return df.ewm(span=span, adjust=False).mean()

    def _rolling_std(self, df: pd.DataFrame, w: int) -> pd.DataFrame:
        return df.rolling(w, min_periods=w // 2).std()

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank at each timestamp (0→1)."""
        return df.rank(axis=1, pct=True)

    def _rsi(self, close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ── Momentum factors ──────────────────────────────────────────────────────

    def mom_1h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(self._ret(close, self.bph))

    def mom_4h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(self._ret(close, self.bph * 4))

    def mom_1d(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(self._ret(close, BARS_PER_DAY))

    def mom_3d(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(self._ret(close, BARS_PER_DAY * 3))

    def mom_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(self._ret(close, BARS_PER_WEEK))

    def mom_2w(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(self._ret(close, BARS_PER_WEEK * 2))

    def mom_30d(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        30-day momentum with 1-day skip (standard academic implementation).
        Skip-1: avoids short-term reversal contaminating the medium-term signal.
        Return window: [t-30d, t-1d] (skip last day).
        """
        ret_30d = close.pct_change(BARS_PER_DAY * 30)
        ret_1d  = close.pct_change(BARS_PER_DAY)
        # Compound: strip out last day's return
        skip_ret = (1 + ret_30d) / (1 + ret_1d).replace(0, np.nan) - 1
        return self._rank(skip_ret)

    def overnight(self, open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
        """
        Close-to-open return: captures gap-up/down from news & order flow overnight.
        Formula: open[t] / close[t-1] - 1
        Cross-sectional rank: long gap-up, short gap-down.
        """
        prev_close = close.shift(1)
        gap = open_ / prev_close.replace(0, np.nan) - 1
        return self._rank(gap)

    def close_to_open(self, open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
        """
        Open-to-close intraday return. Orthogonal to overnight gap.
        Captures intraday momentum / institutional flow.
        """
        intraday = close / open_.replace(0, np.nan) - 1
        return self._rank(intraday)

    # ── Reversal factors ──────────────────────────────────────────────────────

    def rev_1h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(-self._ret(close, self.bph))

    def rev_4h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(-self._ret(close, self.bph * 4))

    def rev_1d(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        1-day reversal. Captures liquidity provider alpha:
        extreme 1D losers tend to bounce next day (and vice versa).
        Short-lived but robust at 5m resolution.
        """
        return self._rank(-self._ret(close, BARS_PER_DAY))

    # ── Volatility factors ────────────────────────────────────────────────────

    def rvol_1d(self, close: pd.DataFrame) -> pd.DataFrame:
        lr = self._log_ret(close)
        rv = self._rolling_std(lr, BARS_PER_DAY)
        return self._rank(-rv)  # Short high-vol (volatility risk premium)

    def rvol_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        lr = self._log_ret(close)
        rv = self._rolling_std(lr, BARS_PER_WEEK)
        return self._rank(-rv)

    def vol_of_vol(self, close: pd.DataFrame) -> pd.DataFrame:
        lr  = self._log_ret(close)
        rv  = self._rolling_std(lr, BARS_PER_DAY)
        vov = self._rolling_std(rv, BARS_PER_WEEK)
        return self._rank(-vov)

    def downvol_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Downside semi-deviation over 1 week (Adrian et al. 2019).
        Only counts negative returns in the volatility estimate.
        Stronger risk premium than symmetric vol — investors hate left tails.
        Short high-downvol coins.
        """
        lr     = self._log_ret(close)
        neg_lr = lr.where(lr < 0, 0)
        # Sqrt of mean squared negative returns (semideviation)
        downvol = (neg_lr ** 2).rolling(BARS_PER_WEEK, min_periods=BARS_PER_DAY).mean().pow(0.5)
        return self._rank(-downvol)

    # ── Liquidity factors ─────────────────────────────────────────────────────

    def amihud(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """Amihud (2002) illiquidity: |return| / volume. Long illiquid."""
        lr    = self._log_ret(close).abs()
        illiq = lr / volume.replace(0, np.nan)
        illiq = illiq.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).mean()
        return self._rank(illiq)

    def corwin_schultz(self, high: pd.DataFrame,
                       low: pd.DataFrame) -> pd.DataFrame:
        """
        Corwin & Schultz (2012) bid-ask spread estimator from H/L prices.
        Spread ≈ (2*(exp(α)-1)) / (1+exp(α))  where α derived from log(H/L).

        Crypto application: proxy for taker pressure / market impact cost.
        High spread → illiquid → avoid (or exploit illiquidity premium).
        """
        ln_hl   = np.log(high / low.replace(0, np.nan))
        # 2-period sum of squared log(H/L)
        beta    = (ln_hl ** 2 + ln_hl.shift(1) ** 2).rolling(2, min_periods=2).mean()
        gamma   = np.log(high.rolling(2).max() / low.rolling(2).min().replace(0, np.nan)) ** 2
        alpha   = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        alpha   = alpha.clip(lower=0)
        spread  = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        rolling_spread = spread.rolling(BARS_PER_DAY, min_periods=self.bph).mean()
        # Long low-spread (liquid) coins
        return self._rank(-rolling_spread)

    def turnover(self, volume: pd.DataFrame) -> pd.DataFrame:
        """
        Relative turnover: volume / rolling 30D average volume.
        High relative turnover = unusual activity = attention effect.
        Short high-turnover (attention-driven overpricing).
        """
        avg_30d = volume.rolling(BARS_PER_DAY * 30, min_periods=BARS_PER_DAY).mean()
        turn    = volume / avg_30d.replace(0, np.nan)
        # Short lottery / attention coins
        return self._rank(-turn)

    def vol_ratio(self, volume: pd.DataFrame) -> pd.DataFrame:
        avg = volume.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).mean()
        return self._rank(volume / avg.replace(0, np.nan))

    def vol_mom_1d(self, volume: pd.DataFrame) -> pd.DataFrame:
        return self._rank(volume.pct_change(BARS_PER_DAY))

    # ── Microstructure factors ────────────────────────────────────────────────

    def max_ret(self, close: pd.DataFrame) -> pd.DataFrame:
        """MAX: maximum return in lookback. Short lottery (MAX effect, -ve)."""
        lr    = self._log_ret(close)
        max_r = lr.rolling(self.bph * 4, min_periods=self.bph).max()
        return self._rank(-max_r)

    def skew_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        """Return skewness over 1 week. Short positive skew (Harvey & Siddique)."""
        lr = self._log_ret(close)
        sk = lr.rolling(BARS_PER_WEEK, min_periods=BARS_PER_DAY).skew()
        return self._rank(-sk)

    def kurt_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Excess kurtosis over 1 week. Fat-tail coins have higher crash risk.
        Short high-kurtosis (leptokurtic) coins.
        """
        lr   = self._log_ret(close)
        kurt = lr.rolling(BARS_PER_WEEK, min_periods=BARS_PER_DAY).kurt()
        return self._rank(-kurt)

    def price_accel(self, close: pd.DataFrame) -> pd.DataFrame:
        """Momentum acceleration: 2nd derivative of price. Long accelerating."""
        mom_short = self._ret(close, self.bph)
        mom_prev  = mom_short.shift(self.bph)
        return self._rank(mom_short - mom_prev)

    def volume_surprise(self, volume: pd.DataFrame) -> pd.DataFrame:
        """
        Volume surprise: residual volume after removing AR(1) trend.
        Captures truly unexpected volume spikes (pre-pump signal).
        Formula: log(vol_t) - μ - φ × log(vol_{t-1})
        AR(1) fit via rolling OLS approximation.
        """
        lv      = np.log(volume.replace(0, np.nan) + 1)
        lv_lag  = lv.shift(1)
        # Rolling autocorrelation proxy for phi
        roll_cov = lv.rolling(BARS_PER_DAY).cov(lv_lag)
        roll_var = lv_lag.rolling(BARS_PER_DAY).var()
        phi      = roll_cov / roll_var.replace(0, np.nan)
        phi      = phi.clip(-0.99, 0.99)
        mu       = lv.rolling(BARS_PER_DAY, min_periods=self.bph).mean()
        residual = lv - mu - phi * lv_lag
        return self._rank(residual)

    def vwap_dev(self, close: pd.DataFrame,
                 high: pd.DataFrame,
                 low: pd.DataFrame,
                 volume: pd.DataFrame) -> pd.DataFrame:
        """
        VWAP deviation: (close - VWAP) / close.
        Uses 288-bar (1-day) rolling VWAP — stationary.
        Negative dev = below VWAP = mean-reversion long signal.
        """
        tp      = (high + low + close) / 3
        tpv_sum = (tp * volume).rolling(BARS_PER_DAY, min_periods=self.bph).sum()
        v_sum   = volume.rolling(BARS_PER_DAY, min_periods=self.bph).sum().replace(0, np.nan)
        vwap    = tpv_sum / v_sum
        dev     = (close - vwap) / close.replace(0, np.nan)
        # Reversal: long below VWAP (negative dev = potential reversion up)
        return self._rank(-dev)

    def btc_beta(self, close: pd.DataFrame,
                 btc_symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Rolling 1-week beta to BTC.
        High beta = more market risk. Use for risk-adjusted sizing.
        Coins with low/negative beta are diversifiers.
        Factor: long low-beta (defensive) = beta anomaly analog.
        """
        if btc_symbol not in close.columns:
            # Fallback: first column is treated as market
            btc_ret = self._log_ret(close.iloc[:, :1]).iloc[:, 0]
        else:
            btc_ret = self._log_ret(close[[btc_symbol]]).iloc[:, 0]

        coin_ret = self._log_ret(close)
        roll_cov = coin_ret.rolling(BARS_PER_WEEK, min_periods=BARS_PER_DAY).apply(
            lambda x: np.cov(x, btc_ret.loc[x.index[-len(x):].intersection(btc_ret.index)][:len(x)])[0, 1]
            if len(x) > 1 else np.nan, raw=False
        )
        roll_var = btc_ret.rolling(BARS_PER_WEEK, min_periods=BARS_PER_DAY).var()
        beta_df  = roll_cov.div(roll_var.replace(0, np.nan), axis=0)
        # Long low-beta (beta anomaly — lower risk-adjusted returns for high beta)
        return self._rank(-beta_df)

    def idio_mom(self, close: pd.DataFrame,
                 btc_symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Idiosyncratic momentum: coin momentum orthogonal to BTC momentum.
        Formula: MOM_1W_coin - beta_1W × MOM_1W_btc
        Pure alpha, market-beta neutral. More predictive than raw momentum
        in regime studies (Gutierrez & Kelley 2008 crypto analogue).
        """
        mom_all = self._ret(close, BARS_PER_WEEK)
        if btc_symbol not in close.columns:
            btc_mom = mom_all.iloc[:, 0]
        else:
            btc_mom = mom_all[btc_symbol]

        # Rolling cross-sectional beta to BTC momentum
        roll_cov = mom_all.rolling(BARS_PER_DAY * 30, min_periods=BARS_PER_WEEK).cov(btc_mom)
        roll_var = btc_mom.rolling(BARS_PER_DAY * 30, min_periods=BARS_PER_WEEK).var()
        beta_mom = roll_cov.div(roll_var.replace(0, np.nan), axis=0)
        idio     = mom_all.sub(beta_mom.mul(btc_mom, axis=0))
        return self._rank(idio)

    # ── Technical factors ─────────────────────────────────────────────────────

    def trend_48(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Cambridge CTREND (Fang & Li 2024): sum of sign(returns) over 48 bars.
        Aggregates weak individual signals into reliable trend measure.
        t-stat 4.22, 2.62% weekly alpha (strongest published 5m factor).
        """
        lr    = self._log_ret(close)
        trend = np.sign(lr).rolling(48, min_periods=24).sum()
        return self._rank(trend)

    def bb_pos(self, close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Bollinger Band position [0, 1]. 0=at lower band, 1=at upper."""
        ma    = close.rolling(window, min_periods=window // 2).mean()
        std   = close.rolling(window, min_periods=window // 2).std(ddof=0)
        upper = ma + 2 * std
        lower = ma - 2 * std
        pos   = (close - lower) / (upper - lower).replace(0, np.nan)
        return self._rank(pos.clip(0, 1))

    def rsi_rank(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._rank(self._rsi(close, 14))

    def ma_slope(self, close: pd.DataFrame,
                 fast: int = 12, slow: int = 48) -> pd.DataFrame:
        """
        EMA slope: (EMA_fast - EMA_slow) / EMA_slow. Trend strength proxy.
        Positive = uptrend, Negative = downtrend.
        Long high-slope (trending up).
        """
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        slope    = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
        return self._rank(slope)

    def week52_high(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Distance from 52-week high: (close - rolling_max_52w) / rolling_max_52w.
        Near-ATH coins are momentum leaders. Long near-52w-high.
        George & Hwang (2004) crypto analogue.
        """
        high_52w = close.rolling(BARS_PER_WEEK * 52, min_periods=BARS_PER_WEEK).max()
        dist     = (close - high_52w) / high_52w.replace(0, np.nan)
        # dist is ≤ 0; less negative = closer to high = momentum signal
        return self._rank(dist)

    def week52_low(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Rebound from 52-week low: (close - rolling_min_52w) / rolling_min_52w.
        Coins bouncing off yearly lows can signal capitulation exhaustion.
        """
        low_52w = close.rolling(BARS_PER_WEEK * 52, min_periods=BARS_PER_WEEK).min()
        dist    = (close - low_52w) / low_52w.replace(0, np.nan)
        # Positive and large = far above yearly low = long signal
        return self._rank(dist)

    # ── Batch compute ─────────────────────────────────────────────────────────

    def compute_all(self,
                    close:  pd.DataFrame,
                    volume: pd.DataFrame,
                    high:   Optional[pd.DataFrame] = None,
                    low:    Optional[pd.DataFrame]  = None,
                    open_:  Optional[pd.DataFrame]  = None,
                    btc_symbol: str = "BTCUSDT") -> Dict[str, pd.DataFrame]:
        """
        Compute all 35 factors. Returns dict: factor_name → ranked DataFrame.

        Args:
            close:      Close price panel (T × N)
            volume:     Volume panel (T × N)
            high:       High panel — optional; needed for Corwin-Schultz, VWAP
            low:        Low panel — optional
            open_:      Open panel — optional; needed for overnight/close_to_open
            btc_symbol: BTC column name for beta/idio momentum
        """
        print(f"[FactorEngineV2] Computing {len(FACTOR_NAMES_V2)} factors...")
        factors: Dict[str, pd.DataFrame] = {}

        _have_hl = (high is not None) and (low is not None)
        _have_open = open_ is not None

        def _compute(name: str, fn):
            try:
                result = fn()
                factors[name] = result
                print(f"   {name}")
            except Exception as e:
                print(f"   {name}: {e}")

        # Momentum
        _compute("MOM_1H",       lambda: self.mom_1h(close))
        _compute("MOM_4H",       lambda: self.mom_4h(close))
        _compute("MOM_1D",       lambda: self.mom_1d(close))
        _compute("MOM_3D",       lambda: self.mom_3d(close))
        _compute("MOM_1W",       lambda: self.mom_1w(close))
        _compute("MOM_2W",       lambda: self.mom_2w(close))
        _compute("MOM_30D",      lambda: self.mom_30d(close))
        if _have_open:
            _compute("OVERNIGHT",    lambda: self.overnight(open_, close))
            _compute("CLOSE_TO_OPEN",lambda: self.close_to_open(open_, close))

        # Reversal
        _compute("REV_1H",  lambda: self.rev_1h(close))
        _compute("REV_4H",  lambda: self.rev_4h(close))
        _compute("REV_1D",  lambda: self.rev_1d(close))

        # Volatility
        _compute("RVOL_1D",   lambda: self.rvol_1d(close))
        _compute("RVOL_1W",   lambda: self.rvol_1w(close))
        _compute("VOL_OF_VOL",lambda: self.vol_of_vol(close))
        _compute("DOWNVOL_1W",lambda: self.downvol_1w(close))

        # Liquidity
        _compute("AMIHUD",    lambda: self.amihud(close, volume))
        if _have_hl:
            _compute("CORWIN_SCHULTZ", lambda: self.corwin_schultz(high, low))
        _compute("TURNOVER",  lambda: self.turnover(volume))
        _compute("VOL_RATIO", lambda: self.vol_ratio(volume))
        _compute("VOL_MOM_1D",lambda: self.vol_mom_1d(volume))

        # Microstructure
        _compute("MAX_RET",       lambda: self.max_ret(close))
        _compute("SKEW_1W",       lambda: self.skew_1w(close))
        _compute("KURT_1W",       lambda: self.kurt_1w(close))
        _compute("PRICE_ACCEL",   lambda: self.price_accel(close))
        _compute("VOLUME_SURPRISE",lambda: self.volume_surprise(volume))
        if _have_hl:
            _compute("VWAP_DEV",  lambda: self.vwap_dev(close, high, low, volume))
        _compute("BTC_BETA",      lambda: self.btc_beta(close, btc_symbol))
        _compute("IDIO_MOM",      lambda: self.idio_mom(close, btc_symbol))

        # Technical
        _compute("TREND_48",  lambda: self.trend_48(close))
        _compute("BB_POS",    lambda: self.bb_pos(close))
        _compute("RSI_RANK",  lambda: self.rsi_rank(close))
        _compute("MA_SLOPE",  lambda: self.ma_slope(close))
        _compute("WEEK52_HIGH",lambda: self.week52_high(close))
        _compute("WEEK52_LOW", lambda: self.week52_low(close))

        print(f"[FactorEngineV2] Done: {len(factors)} factors computed")
        return factors

    # ── Composite builders ────────────────────────────────────────────────────

    def _eq_weight(self, factors: Dict[str, pd.DataFrame],
                   keys: List[str]) -> pd.DataFrame:
        """Equal-weight composite of selected factors (safe 3D stack)."""
        stack = [factors[k] for k in keys if k in factors]
        if not stack:
            raise ValueError(f"None of {keys} available in factors")
        ref = stack[0]
        arr = np.nanmean(
            np.stack([f.reindex(index=ref.index, columns=ref.columns).values
                      for f in stack], axis=0),
            axis=0)
        return pd.DataFrame(arr, index=ref.index, columns=ref.columns)

    def momentum_composite(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        keys = ["MOM_1H","MOM_4H","MOM_1D","MOM_1W","MOM_2W","TREND_48","IDIO_MOM"]
        return self._eq_weight(factors, keys)

    def reversal_composite(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        keys = ["REV_1H","REV_4H","REV_1D","BB_POS","VWAP_DEV"]
        return self._eq_weight(factors, keys)

    def quality_composite(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Filter: avoid illiquid, high-vol, lottery coins."""
        keys = ["AMIHUD","CORWIN_SCHULTZ","RVOL_1D","DOWNVOL_1W","SKEW_1W","MAX_RET","KURT_1W"]
        return self._eq_weight(factors, keys)

    def ic_weighted_composite(self,
                               factors:  Dict[str, pd.DataFrame],
                               ic_table: pd.DataFrame,
                               horizon:  str = "1D",
                               min_icir: float = 0.1) -> pd.DataFrame:
        """
        IC-weighted composite. Factors with ICIR < min_icir are excluded.
        Weights are proportional to ICIR at the chosen horizon.
        """
        ic_row = ic_table[ic_table["horizon"] == horizon].set_index("factor")
        weighted = []
        for name, df in factors.items():
            if name not in ic_row.index:
                continue
            icir = ic_row.loc[name, "ICIR"]
            if icir > min_icir:
                weighted.append((name, icir, df))

        if not weighted:
            print("[WARN] No factors passed ICIR threshold — equal weight fallback")
            return self._eq_weight(factors, list(factors.keys()))

        total_w = sum(w for _, w, _ in weighted)
        ref = weighted[0][2]
        arr = sum(
            (w / total_w) * f.reindex(index=ref.index, columns=ref.columns).values
            for _, w, f in weighted
        )
        return pd.DataFrame(arr, index=ref.index, columns=ref.columns)

    def all_factor_composite(self,
                              factors:  Dict[str, pd.DataFrame],
                              ic_table: Optional[pd.DataFrame] = None,
                              horizon:  str = "1D") -> pd.DataFrame:
        """Master composite: IC-weighted if ic_table provided, else equal weight."""
        if ic_table is not None and not ic_table.empty:
            return self.ic_weighted_composite(factors, ic_table, horizon)
        return self._eq_weight(factors, list(factors.keys()))


# ─────────────────────────────────────────────────────────────────────────────
#  FACTOR METADATA REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_META = {
    # name: (category, direction, needs_open, needs_hl, description)
    "MOM_1H":        ("momentum",      +1, False, False, "1H momentum"),
    "MOM_4H":        ("momentum",      +1, False, False, "4H momentum"),
    "MOM_1D":        ("momentum",      +1, False, False, "1D momentum"),
    "MOM_3D":        ("momentum",      +1, False, False, "3D momentum"),
    "MOM_1W":        ("momentum",      +1, False, False, "1W momentum"),
    "MOM_2W":        ("momentum",      +1, False, False, "2W momentum"),
    "MOM_30D":       ("momentum",      +1, False, False, "30D momentum skip-1D"),
    "OVERNIGHT":     ("momentum",      +1, True,  False, "Close-to-open gap"),
    "CLOSE_TO_OPEN": ("momentum",      +1, True,  False, "Open-to-close intraday"),
    "REV_1H":        ("reversal",      -1, False, False, "1H reversal"),
    "REV_4H":        ("reversal",      -1, False, False, "4H reversal"),
    "REV_1D":        ("reversal",      -1, False, False, "1D reversal (liquidity)"),
    "RVOL_1D":       ("volatility",    -1, False, False, "1D realized vol (short)"),
    "RVOL_1W":       ("volatility",    -1, False, False, "1W realized vol (short)"),
    "VOL_OF_VOL":    ("volatility",    -1, False, False, "Vol of vol (short)"),
    "DOWNVOL_1W":    ("volatility",    -1, False, False, "1W downside semideviation"),
    "AMIHUD":        ("liquidity",     +1, False, False, "Amihud illiquidity"),
    "CORWIN_SCHULTZ":("liquidity",     +1, False, True,  "Bid-ask spread proxy"),
    "TURNOVER":      ("liquidity",     -1, False, False, "Relative turnover"),
    "VOL_RATIO":     ("liquidity",     +1, False, False, "Volume vs 1D avg"),
    "VOL_MOM_1D":    ("liquidity",     +1, False, False, "1D volume momentum"),
    "MAX_RET":       ("microstructure",-1, False, False, "MAX return (lottery short)"),
    "SKEW_1W":       ("microstructure",-1, False, False, "1W return skewness"),
    "KURT_1W":       ("microstructure",-1, False, False, "1W return kurtosis"),
    "PRICE_ACCEL":   ("microstructure",+1, False, False, "Price acceleration"),
    "VOLUME_SURPRISE":("microstructure",+1,False, False, "Unexpected volume spike"),
    "VWAP_DEV":      ("microstructure",-1, False, True,  "VWAP deviation (reversion)"),
    "BTC_BETA":      ("microstructure",-1, False, False, "Rolling BTC beta (low = long)"),
    "IDIO_MOM":      ("microstructure",+1, False, False, "Idiosyncratic momentum"),
    "TREND_48":      ("technical",     +1, False, False, "CTREND 48-bar sign sum"),
    "BB_POS":        ("technical",     +1, False, False, "Bollinger Band position"),
    "RSI_RANK":      ("technical",     +1, False, False, "RSI cross-sectional rank"),
    "MA_SLOPE":      ("technical",     +1, False, False, "EMA fast/slow slope"),
    "WEEK52_HIGH":   ("technical",     +1, False, False, "Distance from 52W high"),
    "WEEK52_LOW":    ("technical",     +1, False, False, "Recovery from 52W low"),
}
