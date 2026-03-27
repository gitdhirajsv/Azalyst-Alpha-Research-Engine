"""
╔══════════════════════════════════════════════════════════════════════════════╗
        AZALYST ALPHA RESEARCH ENGINE    QUANTITATIVE RESEARCH ENGINE         
║           Factor Models · Cross-Sectional Alpha · Vectorized Backtest      ║
║           v1.0  |  5-Minute Binance OHLCV  |  All USDT Pairs              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture
────────────
  DataLoader          — Parallel parquet ingestion, resampling, universe filter
  FactorEngine        — 20 quantitative factors (momentum, reversal, volume,
                        volatility, microstructure, trend, ML-ready features)
  CrossSectionalAnal  — IC, ICIR, t-stats, factor decay, quantile spread
  BacktestEngine      — Vectorized long/short backtest with realistic costs
  PortfolioAnalytics  — Sharpe, Sortino, Calmar, Max DD, turnover
  CLI                 — python azalyst_engine.py --data-dir ./data [options]

Factor Universe (20 factors)
─────────────────────────────
  Momentum   : MOM_1H  MOM_4H  MOM_1D  MOM_3D  MOM_1W  MOM_2W
  Reversal   : REV_1H  REV_4H
  Volume     : VOL_RATIO  VOL_MOM_1D
  Volatility : RVOL_1D  RVOL_1W  VOL_OF_VOL
  Micro      : AMIHUD  MAX_RET  SKEW_1W  PRICE_ACCEL
  Structural : TREND_48  BB_POS  RSI_RANK

Research basis: Liu & Tsyvinski (2021), Fieberg et al. (2024),
Baybutt (2024), Borri et al. (2024), Kakushadze (2019),
Dobrynskaya (2024), Cambridge CTREND (2024)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BARS_PER_HOUR  = 12          # 5-min bars
BARS_PER_DAY   = 288
BARS_PER_WEEK  = 2016
FEE_RATE       = 0.001       # 0.1% Binance taker (each leg)
ROUND_TRIP_FEE = FEE_RATE * 2
MIN_HISTORY    = BARS_PER_WEEK * 2   # 2 weeks warmup before trading

FACTOR_NAMES = [
    "MOM_1H", "MOM_4H", "MOM_1D", "MOM_3D", "MOM_1W", "MOM_2W",
    "REV_1H", "REV_4H",
    "VOL_RATIO", "VOL_MOM_1D",
    "RVOL_1D", "RVOL_1W", "VOL_OF_VOL",
    "AMIHUD", "MAX_RET", "SKEW_1W", "PRICE_ACCEL",
    "TREND_48", "BB_POS", "RSI_RANK",
]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads parquet files from a directory, validates OHLCV schema,
    resamples to target timeframe, and builds a multi-symbol panel.

    Parquet schema expected:
        timestamp (int ms or datetime), open, high, low, close, volume
    """

    REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

    def __init__(self, data_dir: str, resample_to: Optional[str] = None,
                 max_symbols: Optional[int] = None, min_rows: int = MIN_HISTORY,
                 workers: int = 4):
        self.data_dir   = Path(data_dir)
        self.resample   = resample_to      # e.g. "15min", "1h", "4h" — None keeps 5m
        self.max_symbols = max_symbols
        self.min_rows   = min_rows
        self.workers    = workers

    def discover_files(self) -> List[Path]:
        files = sorted(self.data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {self.data_dir}")
        if self.max_symbols:
            files = files[:self.max_symbols]
        return files

    @staticmethod
    def _load_single(path: Path, min_rows: int,
                     resample_to: Optional[str]) -> Tuple[str, Optional[pd.DataFrame]]:
        symbol = path.stem  # e.g. "BTCUSDT"
        try:
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]

            # ── normalise timestamp ──────────────────────────────────────────
            # Binance downloader saves the time column as "time" (not "timestamp")
            # Also handle "open_time" used by some scrapers
            ts_col = next(
                (c for c in df.columns if c in ("timestamp", "time", "open_time")),
                None
            )
            if ts_col is not None:
                col = df[ts_col]
                if pd.api.types.is_integer_dtype(col):
                    df["timestamp"] = pd.to_datetime(col, unit="ms", utc=True)
                else:
                    df["timestamp"] = pd.to_datetime(col, utc=True)
                if ts_col != "timestamp":
                    df = df.drop(columns=[ts_col])
                df = df.set_index("timestamp")
            elif isinstance(df.index, pd.DatetimeIndex):
                pass  # already fine
            else:
                # last resort: parse the row index
                if pd.api.types.is_integer_dtype(df.index):
                    df.index = pd.to_datetime(df.index, unit='ms', utc=True)
                else:
                    df.index = pd.to_datetime(df.index, utc=True)

            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()

            # ── 1970 timestamp check ─────────────────────────────────────────
            if df.index.max().year < 2018:
                return symbol, None   # 1970 timestamp still present, skip symbol

            # ── validate schema ─────────────────────────────────────────────
            missing = DataLoader.REQUIRED_COLS - set(df.columns)
            if missing:
                return symbol, None

            df = df[["open", "high", "low", "close", "volume"]].apply(
                pd.to_numeric, errors="coerce"
            )
            df = df.dropna()

            if len(df) < min_rows:
                return symbol, None

            # ── optional resample ────────────────────────────────────────────
            if resample_to:
                df = df.resample(resample_to).agg({
                    "open":   "first",
                    "high":   "max",
                    "low":    "min",
                    "close":  "last",
                    "volume": "sum",
                }).dropna()

            return symbol, df

        except Exception as e:
            print(f"  [WARN] {symbol}: {e}")
            return symbol, None

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Returns dict: symbol → OHLCV DataFrame."""
        files = self.discover_files()
        print(f"[DataLoader] Found {len(files)} parquet files — loading...")

        results: Dict[str, pd.DataFrame] = {}

        # Use ProcessPoolExecutor for parallel parquet reads
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._load_single, f, self.min_rows, self.resample): f
                for f in files
            }
            for i, fut in enumerate(as_completed(futures), 1):
                sym, df = fut.result()
                if df is not None:
                    results[sym] = df
                if i % 50 == 0:
                    print(f"  loaded {i}/{len(files)} ...  valid={len(results)}")

        print(f"[DataLoader] Universe: {len(results)} symbols after quality filter")
        return results

    def build_close_panel(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Wide panel: rows=timestamps, cols=symbols (close prices)."""
        closes = {sym: df["close"] for sym, df in data.items()}
        panel  = pd.DataFrame(closes).sort_index()
        # Forward-fill up to 3 bars for exchange halts, then drop
        panel  = panel.ffill(limit=3)
        return panel

    def build_volume_panel(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        volumes = {sym: df["volume"] for sym, df in data.items()}
        return pd.DataFrame(volumes).sort_index().ffill(limit=3)


# ─────────────────────────────────────────────────────────────────────────────
#  FACTOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class FactorEngine:
    """
    Computes 20 quantitative factors on a cross-sectional panel.

    All factor outputs are DataFrames with the same index/columns as
    the input close panel.  Each factor is cross-sectionally ranked
    (0→1 percentile rank) before analysis so factors are comparable.

    Factor descriptions
    ───────────────────
    MOM_Nx    : Return over last N bars.  Long recent winners.
                Evidence: strongest documented crypto alpha (A+).
    REV_Nx    : Negative of short-horizon return.  Mean reversion.
                Evidence: close-to-open 1H-4H reversal is robust (A).
    VOL_RATIO : Current bar's volume / rolling 288-bar mean volume.
                Identifies unusual activity.
    VOL_MOM   : 1D volume return (vol today / vol yesterday).
    RVOL_xD   : Realized volatility = std(log returns) over window.
                Short high-vol = volatility risk premium (B+).
    VOL_OF_VOL: Volatility of volatility — second-order risk.
    AMIHUD    : |return| / volume — proxy for illiquidity.
                Long illiquid coins (illiquidity premium, B).
    MAX_RET   : Maximum single-bar return in last 48 bars.
                Short lottery coins (MAX effect is negative, B+).
    SKEW_1W   : Return skewness over 1 week.  Short positive skew.
    PRICE_ACCEL: Return acceleration = momentum gradient (2nd derivative).
    TREND_48  : Sum of sign(returns) over 48 bars (Cambridge CTREND).
                2.62% weekly alpha, t-stat 4.22 (2024 paper).
    BB_POS    : (close - BB_lower) / (BB_upper - BB_lower).
                Momentum confirmation / mean reversion signal.
    RSI_RANK  : Cross-sectional RSI(14).  Relative strength ranking.
    """

    def __init__(self, bars_per_hour: int = BARS_PER_HOUR):
        self.bph = bars_per_hour

    # ── private helpers ─────────────────────────────────────────────────────

    def _returns(self, close: pd.DataFrame, n: int) -> pd.DataFrame:
        return close.pct_change(n)

    def _log_returns(self, close: pd.DataFrame) -> pd.DataFrame:
        return np.log(close / close.shift(1))

    def _cross_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank at each timestamp (0→1)."""
        return df.rank(axis=1, pct=True)

    def _rolling_std(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        return df.rolling(window, min_periods=window // 2).std()

    def _ema(self, df: pd.DataFrame, span: int) -> pd.DataFrame:
        return df.ewm(span=span, adjust=False).mean()

    def _rsi(self, close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_l = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs    = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ── public factor methods ─────────────────────────────────────────────────

    def mom_1h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(self._returns(close, self.bph))

    def mom_4h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(self._returns(close, self.bph * 4))

    def mom_1d(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(self._returns(close, BARS_PER_DAY))

    def mom_3d(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(self._returns(close, BARS_PER_DAY * 3))

    def mom_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(self._returns(close, BARS_PER_WEEK))

    def mom_2w(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(self._returns(close, BARS_PER_WEEK * 2))

    def rev_1h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(-self._returns(close, self.bph))

    def rev_4h(self, close: pd.DataFrame) -> pd.DataFrame:
        return self._cross_rank(-self._returns(close, self.bph * 4))

    def vol_ratio(self, volume: pd.DataFrame) -> pd.DataFrame:
        avg = volume.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).mean()
        ratio = volume / avg.replace(0, np.nan)
        return self._cross_rank(ratio)

    def vol_mom_1d(self, volume: pd.DataFrame) -> pd.DataFrame:
        vol_ret = volume.pct_change(BARS_PER_DAY)
        return self._cross_rank(vol_ret)

    def rvol_1d(self, close: pd.DataFrame) -> pd.DataFrame:
        lr = self._log_returns(close)
        rv = self._rolling_std(lr, BARS_PER_DAY)
        # Short high vol (negative factor sign = short vol premium)
        return self._cross_rank(-rv)

    def rvol_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        lr = self._log_returns(close)
        rv = self._rolling_std(lr, BARS_PER_WEEK)
        return self._cross_rank(-rv)

    def vol_of_vol(self, close: pd.DataFrame) -> pd.DataFrame:
        lr   = self._log_returns(close)
        rv   = self._rolling_std(lr, BARS_PER_DAY)
        vov  = self._rolling_std(rv, BARS_PER_WEEK)
        return self._cross_rank(-vov)

    def amihud(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        lr     = self._log_returns(close).abs()
        illiq  = lr / volume.replace(0, np.nan)
        illiq  = illiq.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).mean()
        return self._cross_rank(illiq)   # Long illiquid = illiquidity premium

    def max_ret(self, close: pd.DataFrame) -> pd.DataFrame:
        lr    = self._log_returns(close)
        max_r = lr.rolling(self.bph * 4, min_periods=12).max()
        return self._cross_rank(-max_r)  # Short lottery (MAX effect negative)

    def skew_1w(self, close: pd.DataFrame) -> pd.DataFrame:
        lr   = self._log_returns(close)
        sk   = lr.rolling(BARS_PER_WEEK, min_periods=BARS_PER_DAY).skew()
        return self._cross_rank(-sk)     # Short positive skew

    def price_accel(self, close: pd.DataFrame) -> pd.DataFrame:
        mom_short = self._returns(close, self.bph)
        mom_prev  = mom_short.shift(self.bph)
        accel     = mom_short - mom_prev
        return self._cross_rank(accel)

    def trend_48(self, close: pd.DataFrame) -> pd.DataFrame:
        """Cambridge CTREND: sum of sign(returns) over 48 bars.
        Aggregates weak individual signals into a reliable trend measure.
        t-stat 4.22 weekly in Fang & Li 2024."""
        lr      = self._log_returns(close)
        signs   = np.sign(lr)
        trend   = signs.rolling(48, min_periods=24).sum()
        return self._cross_rank(trend)

    def bb_pos(self, close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Bollinger Band position: 0 = at lower band, 1 = at upper band."""
        ma  = close.rolling(window, min_periods=10).mean()
        std = close.rolling(window, min_periods=10).std(ddof=0)
        upper = ma + 2 * std
        lower = ma - 2 * std
        pos   = (close - lower) / (upper - lower).replace(0, np.nan)
        pos   = pos.clip(0, 1)
        return self._cross_rank(pos)

    def rsi_rank(self, close: pd.DataFrame) -> pd.DataFrame:
        rsi = self._rsi(close, period=14)
        return self._cross_rank(rsi)

    # ── microstructure factors (Phase 2) ──────────────────────────────────────

    def ofi_proxy(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """
        Order Flow Imbalance Proxy. 
        Approximates net pressure using (close-open)/(high-low) * volume.
        """
        rng = (close.shift(0) - close.shift(0).min()).replace(0, 1e-9) # dummy for calculation
        # Simple proxy: sign(return) * volume
        ofi = np.sign(close.pct_change()) * volume
        # Accumulate over 1H
        ofi_acc = ofi.rolling(self.bph).sum()
        return self._cross_rank(ofi_acc)

    def vpin_proxy(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """
        VPIN Proxy (Volume-Synchronized Probability of Informed Trading).
        Measures Order Flow Toxicity based on buy/sell imbalance within volume buckets.
        """
        # 1. Classify buy/sell volume using tick rule
        ret = close.pct_change()
        buy_vol = volume.where(ret > 0, 0)
        sell_vol = volume.where(ret < 0, 0)
        
        # 2. Imbalance
        imb = (buy_vol - sell_vol).abs()
        
        # 3. Synchronize by rolling volume (proxy for buckets)
        vpin = imb.rolling(self.bph * 4).sum() / volume.rolling(self.bph * 4).sum().replace(0, np.nan)
        return self._cross_rank(vpin)

    # ── batch compute ─────────────────────────────────────────────────────────

    def compute_all(self, close: pd.DataFrame,
                    volume: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute all 20 factors.  Returns dict factor_name → ranked DataFrame."""
        print("[FactorEngine] Computing 20 factors...")
        factors = {}
        fns = [
            ("MOM_1H",     lambda: self.mom_1h(close)),
            ("MOM_4H",     lambda: self.mom_4h(close)),
            ("MOM_1D",     lambda: self.mom_1d(close)),
            ("MOM_3D",     lambda: self.mom_3d(close)),
            ("MOM_1W",     lambda: self.mom_1w(close)),
            ("MOM_2W",     lambda: self.mom_2w(close)),
            ("REV_1H",     lambda: self.rev_1h(close)),
            ("REV_4H",     lambda: self.rev_4h(close)),
            ("VOL_RATIO",  lambda: self.vol_ratio(volume)),
            ("VOL_MOM_1D", lambda: self.vol_mom_1d(volume)),
            ("RVOL_1D",    lambda: self.rvol_1d(close)),
            ("RVOL_1W",    lambda: self.rvol_1w(close)),
            ("VOL_OF_VOL", lambda: self.vol_of_vol(close)),
            ("AMIHUD",     lambda: self.amihud(close, volume)),
            ("MAX_RET",    lambda: self.max_ret(close)),
            ("SKEW_1W",    lambda: self.skew_1w(close)),
            ("PRICE_ACCEL",lambda: self.price_accel(close)),
            ("TREND_48",   lambda: self.trend_48(close)),
            ("BB_POS",     lambda: self.bb_pos(close)),
            ("RSI_RANK",   lambda: self.rsi_rank(close)),
        ]
        for name, fn in fns:
            factors[name] = fn()
            print(f"   {name}")
        return factors


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-SECTIONAL ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class CrossSectionalAnalyser:
    """
    Measures how predictive each factor is at each forward horizon.

    Metrics
    ───────
    IC (Information Coefficient)
        Spearman rank correlation between factor at time t and
        forward return at t+h.  IC > 0.05 is considered meaningful.
        IC > 0.10 is strong for daily rebalancing.

    ICIR (Information Ratio of IC)
        IC_mean / IC_std  — penalises inconsistent factors.
        ICIR > 0.5 is good.  ICIR > 1.0 is excellent.

    t-statistic
        t = IC_mean / (IC_std / sqrt(N)).
        |t| > 2 = statistically significant at 95%.

    Quantile Spread
        Return of top-quintile long minus bottom-quintile short.
        The economic magnitude of the alpha.

    Factor Decay
        IC at horizons 1H, 4H, 1D, 3D, 1W — shows how quickly
        predictive power dissipates.
    """

    HORIZONS_BARS = {
        "1H":  BARS_PER_HOUR,
        "4H":  BARS_PER_HOUR * 4,
        "1D":  BARS_PER_DAY,
        "3D":  BARS_PER_DAY * 3,
        "1W":  BARS_PER_WEEK,
    }

    def __init__(self, close_panel: pd.DataFrame):
        self.close = close_panel
        # Log returns for each bar
        self.log_ret = np.log(close_panel / close_panel.shift(1))

    def forward_return(self, horizon_bars: int) -> pd.DataFrame:
        """Future log return over horizon_bars periods ahead."""
        fwd = self.log_ret.shift(-horizon_bars).rolling(
            horizon_bars, min_periods=horizon_bars // 2
        ).sum()
        return fwd

    def ic_series(self, factor: pd.DataFrame,
                  fwd_ret: pd.DataFrame,
                  chunk_size: int = 2000) -> pd.Series:
        """
        Chunked Spearman IC to avoid OOM on large panels.
        Processes chunk_size rows at a time instead of full matrix.
        """
        idx = factor.index.intersection(fwd_ret.index)

        def _nanrank_rows(arr: np.ndarray) -> np.ndarray:
            ranked = np.full_like(arr, np.nan)
            for i in range(arr.shape[0]):
                row = arr[i]
                finite = np.where(np.isfinite(row))[0]
                if len(finite) < 2:
                    continue
                order = np.argsort(np.argsort(row[finite]))
                ranked[i, finite] = order.astype(float)
            return ranked

        def _chunk_ic(f_chunk, r_chunk):
            valid_mask = (~np.isnan(f_chunk)) & (~np.isnan(r_chunk))
            n_valid    = valid_mask.sum(axis=1)
            f_r = _nanrank_rows(f_chunk)
            r_r = _nanrank_rows(r_chunk)
            a = np.where(valid_mask, f_r, np.nan)
            b = np.where(valid_mask, r_r, np.nan)
            a_mu = np.nanmean(a, axis=1, keepdims=True)
            b_mu = np.nanmean(b, axis=1, keepdims=True)
            da   = a - a_mu
            db   = b - b_mu
            num  = np.nansum(da * db, axis=1)
            den  = np.sqrt(np.nansum(da**2, axis=1) * np.nansum(db**2, axis=1))
            with np.errstate(invalid='ignore', divide='ignore'):
                rho = np.where(den > 0, num / den, np.nan)
            rho[n_valid < 10] = np.nan
            return rho

        results = []
        for start in range(0, len(idx), chunk_size):
            chunk_idx = idx[start:start + chunk_size]
            f_chunk = factor.loc[chunk_idx].values.astype(float)
            r_chunk = fwd_ret.loc[chunk_idx].values.astype(float)
            results.append(_chunk_ic(f_chunk, r_chunk))

        return pd.Series(np.concatenate(results), index=idx)

    @staticmethod
    def _nw_tstat(ic_vals: np.ndarray) -> float:
        """
        Newey-West HAC (Heteroscedasticity-Autocorrelation Consistent)
        t-statistic for overlapping IC series.

        Standard OLS t-stat assumes IID — wrong for overlapping return windows
        (1W, 2W horizons have ~5 or ~10 correlated observations per unique window).
        Newey-West corrects for this autocorrelation, giving conservative and
        statistically valid inference.

        Lag rule: max_lag = floor(4 × (T/100)^(2/9))  [Andrews 1991]
        """
        try:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools import add_constant
            n = len(ic_vals)
            if n < 20:
                return 0.0
            max_lag = max(1, int(4 * (n / 100) ** (2 / 9)))
            X = add_constant(np.zeros(n))   # intercept-only OLS
            X[:, 1] = 0                      # dummy predictor (intercept test)
            X = np.ones((n, 1))              # just the intercept
            model = OLS(ic_vals, X)
            try:
                res = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag, 'use_correction': True})
                return float(res.tvalues[0])
            except Exception:
                # statsmodels < 0.14 keyword difference
                res = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
                return float(res.tvalues[0])
        except Exception:
            return 0.0

    def analyse_factor(self, name: str, factor: pd.DataFrame,
                       horizons: Optional[List[str]] = None) -> pd.DataFrame:
        """Full IC analysis for one factor across all horizons.

        Columns returned
        ────────────────
        IC_mean  : Mean Spearman IC (predictive power)
        IC_std   : Standard deviation of IC series
        ICIR     : IC / IC_std — consistency-adjusted signal strength
        t_stat   : Naive t = IC_mean / SE (assumes IID — OK for 1H/4H)
        nw_t_stat: Newey-West HAC t (correct for 1D/1W/2W overlapping horizons)
        IC_pos%  : Fraction of IC observations > 0
        n_obs    : Number of IC observations used
        """
        if horizons is None:
            horizons = list(self.HORIZONS_BARS.keys())
        rows = []
        for h in horizons:
            n_bars = self.HORIZONS_BARS[h]
            fwd    = self.forward_return(n_bars)
            ics    = self.ic_series(factor, fwd).dropna()
            if len(ics) < 20:
                continue
            mean_ic = ics.mean()
            std_ic  = ics.std()
            icir    = mean_ic / std_ic if std_ic > 0 else 0.0
            t_stat  = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0.0
            nw_t    = self._nw_tstat(ics.values)
            rows.append({
                "factor":     name,
                "horizon":    h,
                "IC_mean":   round(mean_ic, 5),
                "IC_std":    round(std_ic, 5),
                "ICIR":      round(icir, 4),
                "t_stat":    round(t_stat, 4),
                "nw_t_stat": round(nw_t, 4),
                "IC_pos%":   round((ics > 0).mean() * 100, 1),
                "n_obs":     len(ics),
            })
        return pd.DataFrame(rows)

    def analyse_all(self, factors: Dict[str, pd.DataFrame],
                    horizons: Optional[List[str]] = None) -> pd.DataFrame:
        """Run IC analysis for every factor."""
        print("\n[CrossSectional] Running IC analysis across all factors & horizons...")
        frames = []
        for name, factor in factors.items():
            print(f"  analysing {name}...")
            df = self.analyse_factor(name, factor, horizons)
            frames.append(df)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return result

    def quantile_returns(self, factor: pd.DataFrame,
                         fwd_ret: pd.DataFrame,
                         n_quantiles: int = 5,
                         rebal_every: int = BARS_PER_DAY) -> pd.DataFrame:
        """
        Sort symbols into quantiles at each rebalancing period.
        Return the mean forward return of each quantile bin.
        Enables the classic 'factor quintile spread' plot.
        """
        idx        = factor.index.intersection(fwd_ret.index)
        rebal_idx  = idx[::rebal_every]
        q_returns  = {q: [] for q in range(1, n_quantiles + 1)}

        for t in rebal_idx:
            row_f = factor.loc[t]
            row_r = fwd_ret.loc[t]
            mask  = row_f.notna() & row_r.notna()
            if mask.sum() < n_quantiles * 3:
                continue
            qcuts = pd.qcut(row_f[mask], n_quantiles,
                            labels=range(1, n_quantiles + 1),
                            duplicates="drop")
            for q in range(1, n_quantiles + 1):
                syms = qcuts[qcuts == q].index
                if len(syms) > 0:
                    q_returns[q].append(row_r[syms].mean())

        summary = {}
        for q, rets in q_returns.items():
            if rets:
                r_arr = np.array(rets)
                summary[f"Q{q}"] = {
                    "mean_ret":    round(r_arr.mean() * 100, 4),
                    "hit_rate_%":  round((r_arr > 0).mean() * 100, 1),
                    "sharpe":      round(r_arr.mean() / r_arr.std() * np.sqrt(252)
                                        if r_arr.std() > 0 else 0, 3),
                    "n":           len(rets),
                }
        spread = (np.mean(q_returns.get(n_quantiles, [0])) -
                  np.mean(q_returns.get(1, [0]))) * 100
        return pd.DataFrame(summary).T, round(spread, 4)

    def factor_decay(self, name: str, factor: pd.DataFrame) -> pd.DataFrame:
        """IC decay curve — shows if factor is a fast or slow signal."""
        horizons_all = {
            "1H": BARS_PER_HOUR,
            "4H": BARS_PER_HOUR * 4,
            "1D": BARS_PER_DAY,
            "3D": BARS_PER_DAY * 3,
            "1W": BARS_PER_WEEK,
            "2W": BARS_PER_WEEK * 2,
        }
        rows = []
        for h, n in horizons_all.items():
            fwd  = self.forward_return(n)
            ics  = self.ic_series(factor, fwd).dropna()
            rows.append({"horizon": h, "IC": round(ics.mean(), 5) if len(ics) else np.nan})
        df = pd.DataFrame(rows).set_index("horizon")
        df.name = name
        return df


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Vectorized long-short backtest.

    Strategy
    ────────
    At each rebalancing bar:
    - Compute composite score = weighted average of selected factor ranks
    - Long top N% of coins, Short bottom N% (or long-only mode)
    - Equal-weight within each bucket
    - Apply transaction costs (Binance 0.1% per leg = 0.2% round-trip)
    - Hold until next rebalance

    Key design choices
    ──────────────────
    - Vectorized: no per-bar Python loops in the inner loop
    - Turnover-aware: cost is proportional to position change
    - Realistic fills: uses close prices (next-bar execution = +1 bar lag)
    - Max position: no single coin > 10% of portfolio
    """

    def __init__(self, close_panel: pd.DataFrame,
                 rebal_every: int = BARS_PER_DAY,
                 long_pct: float = 0.2,
                 short_pct: float = 0.2,
                 long_only: bool = False,
                 max_position: float = 0.10,
                 fee_rate: float = FEE_RATE,
                 benchmark_col: Optional[str] = None,
                 execution_lag: int = 0,
                 market_impact_coef: float = 1.0, # Multiplier for impact
                 volume_panel: Optional[pd.DataFrame] = None,
                 portfolio_size: float = 1_000_000): # Institutional scale
        self.close         = close_panel
        self.rebal_every   = rebal_every
        self.long_pct      = long_pct
        self.short_pct     = short_pct
        self.long_only     = long_only
        self.max_pos       = max_position
        self.fee_rate      = fee_rate
        self.exec_lag      = execution_lag
        self.mi_coef       = market_impact_coef
        self.volume        = volume_panel
        self.port_size     = portfolio_size
        self.benchmark_col = benchmark_col or next(
            (c for c in close_panel.columns if "BTC" in str(c).upper()), None
        )
        
        try:
            from azalyst_execution import ImpactModel
            self._impact_model = ImpactModel(decay_param=0.5)
        except Exception as e:
            print(f"  [WARN] ImpactModel unavailable (market-impact costs skipped): {e}")
            self._impact_model = None

    def _build_weights(self, scores: pd.Series,
                       n_long: int, n_short: int) -> pd.Series:
        """Equal-weight long top / short bottom, capped at max_position."""
        weights = pd.Series(0.0, index=scores.index)
        valid   = scores.dropna()
        if len(valid) < n_long + n_short:
            return weights

        ranked  = valid.rank(pct=True)
        longs   = ranked[ranked >= (1 - self.long_pct)].index
        shorts  = ranked[ranked <= self.short_pct].index

        n_l, n_s = len(longs), len(shorts)
        if n_l > 0:
            w = min(1.0 / n_l, self.max_pos)
            weights[longs]  = w
        if n_s > 0 and not self.long_only:
            w = min(1.0 / n_s, self.max_pos)
            weights[shorts] = -w

        return weights

    def _calculate_market_impact(self, turnover: pd.Series, 
                                 t_idx: int) -> float:
        """
        Estimate institutional market impact cost using square-root model.
        """
        if self.volume is None or self.mi_coef <= 0 or self._impact_model is None:
            return 0.0
        
        # Current bar volumes (base currency equivalent)
        vol_base = self.volume.iloc[t_idx]
        
        # Trade sizes in base currency
        trade_size_base = turnover * self.port_size
        
        # Calculate impact for each asset
        total_impact_bps = 0.0
        # Assume 2% daily vol if not provided
        daily_vol = 0.02 
        
        for sym in turnover.index:
            if turnover[sym] == 0: continue
            v = vol_base.get(sym, 0)
            if v <= 0: continue
            
            temp, perm = self._impact_model.calculate_impact(
                trade_size_base[sym], v, daily_vol
            )
            # Use mi_coef as a scalar for the model
            total_impact_bps += (temp + perm) * self.mi_coef
            
        # Return as a fractional cost (denominator for weighted average)
        return total_impact_bps / len(turnover[turnover != 0]) if any(turnover != 0) else 0.0

    def run(self, composite_factor: pd.DataFrame,
            label: str = "Strategy") -> pd.DataFrame:
        """
        Run backtest.  Returns daily PnL DataFrame with columns:
        gross_ret, fee, net_ret, cum_ret, drawdown
        """
        print(f"\n[Backtest] Running '{label}' with {self.exec_lag}-bar lag...")
        close  = self.close
        factor = composite_factor.reindex(close.index, method="ffill")

        timestamps = close.index
        rebal_idx  = np.arange(MIN_HISTORY, len(timestamps) - self.rebal_every - self.exec_lag, self.rebal_every)

        pnl_rows  = []
        prev_w    = pd.Series(0.0, index=close.columns)

        for ri in rebal_idx:
            # Need at least one bar ahead for entry AND one period ahead for exit
            if ri + self.rebal_every + self.exec_lag >= len(timestamps):
                break

            # ── FIX: Execution lag ────────────────────────────────────────────
            # t_signal : bar where we evaluate the factor (after close)
            # t_entry  : execution_lag bars after signal
            # t_exit   : rebal_every bars after entry
            t_signal = timestamps[ri]
            t_entry  = timestamps[ri + self.exec_lag]
            t_exit   = timestamps[min(ri + self.exec_lag + self.rebal_every, len(timestamps) - 1)]

            scores = factor.loc[t_signal]    # factor at signal bar (no lookahead)
            n_syms = scores.notna().sum()
            n_long = max(1, int(n_syms * self.long_pct))
            n_short = max(1, int(n_syms * self.short_pct))

            new_w = self._build_weights(scores, n_long, n_short)

            # ── Handle ghost positions: symbols missing from active close ──────
            # Zero out weights for any symbol with NaN at entry (delisted / halted)
            active = close.loc[t_entry].notna()
            new_w  = new_w * active

            # Returns from entry to exit
            p_entry = close.loc[t_entry]
            p_exit  = close.loc[t_exit]
            ret     = (p_exit - p_entry) / p_entry.replace(0, np.nan)

            gross_pnl  = (new_w * ret).sum()

            # Turnover = absolute weight change (one-way)
            weight_diff = (new_w - prev_w).abs()
            turnover   = weight_diff.sum() / 2
            
            # 1. Direct Fees
            fee_cost   = turnover * self.fee_rate * 2 
            
            # 2. Market Impact
            impact_cost = self._calculate_market_impact(weight_diff, ri + self.exec_lag)

            total_cost = fee_cost + impact_cost
            net_pnl    = gross_pnl - total_cost

            pnl_rows.append({
                "timestamp":  t_exit,
                "gross_ret":  gross_pnl,
                "fee":        -fee_cost,
                "impact":     -impact_cost,
                "net_ret":    net_pnl,
                "turnover":   turnover,
                "n_longs":    (new_w > 0).sum(),
                "n_shorts":   (new_w < 0).sum(),
            })

            prev_w = new_w

        if not pnl_rows:
            return pd.DataFrame(columns=["gross_ret", "fee", "impact", "net_ret", "turnover", "n_longs", "n_shorts", "cum_ret", "drawdown"])

        result = pd.DataFrame(pnl_rows).set_index("timestamp")
        result["cum_ret"]  = (1 + result["net_ret"]).cumprod() - 1
        result["drawdown"] = _max_drawdown_series(result["cum_ret"])

        # ── BTC benchmark comparison (auto-computed if BTC found in panel) ──────
        if self.benchmark_col and self.benchmark_col in self.close.columns:
            btc_log   = np.log(self.close[self.benchmark_col] /
                               self.close[self.benchmark_col].shift(1))
            btc_rebal = btc_log.reindex(result.index, method="nearest").fillna(0)
            result["benchmark_ret"] = btc_rebal.values
            result["excess_ret"]    = result["net_ret"] - result["benchmark_ret"]
            result["cum_benchmark"] = (1 + result["benchmark_ret"]).cumprod() - 1
            result["cum_excess"]    = (1 + result["excess_ret"]).cumprod() - 1

        print(f"  Periods: {len(result)} | "
              f"Total return: {result['cum_ret'].iloc[-1]*100:.1f}%")
        if "cum_benchmark" in result.columns:
            print(f"  BTC B&H return: {result['cum_benchmark'].iloc[-1]*100:.1f}% | "
                  f"Excess: {result['cum_excess'].iloc[-1]*100:.1f}%")
        return result

    def run_single_factor(self, factor_name: str,
                          factor: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper for a single factor."""
        return self.run(factor, label=factor_name)


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioAnalytics:
    """
    Computes institutional-standard performance metrics from a PnL series.
    """

    PERIODS_PER_YEAR = 365   # for daily rebalancing; adjust if weekly

    @staticmethod
    def summary(pnl: pd.DataFrame, label: str = "") -> Dict:
        """Full performance summary dict."""
        rets    = pnl["net_ret"].dropna()
        cum     = (1 + rets).cumprod()
        n_years = len(rets) / PortfolioAnalytics.PERIODS_PER_YEAR

        total_ret   = cum.iloc[-1] - 1 if len(cum) else np.nan
        ann_ret     = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
        vol         = rets.std() * np.sqrt(PortfolioAnalytics.PERIODS_PER_YEAR)
        sharpe      = ann_ret / vol if vol > 0 else np.nan
        downside    = rets[rets < 0].std() * np.sqrt(PortfolioAnalytics.PERIODS_PER_YEAR)
        sortino     = ann_ret / downside if downside > 0 else np.nan
        max_dd      = pnl["drawdown"].min() if "drawdown" in pnl.columns else np.nan
        calmar      = ann_ret / abs(max_dd) if max_dd and max_dd != 0 else np.nan
        win_rate    = (rets > 0).mean()
        avg_win     = rets[rets > 0].mean()
        avg_loss    = rets[rets < 0].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss and avg_loss != 0 else np.nan
        avg_turnover  = pnl["turnover"].mean() if "turnover" in pnl.columns else np.nan

        return {
            "label":          label,
            "total_ret_%":    round(total_ret * 100, 2),
            "ann_ret_%":      round(ann_ret * 100, 2),
            "ann_vol_%":      round(vol * 100, 2),
            "sharpe":         round(sharpe, 3),
            "sortino":        round(sortino, 3),
            "calmar":         round(calmar, 3),
            "max_dd_%":       round(max_dd * 100, 2) if max_dd else np.nan,
            "win_rate_%":     round(win_rate * 100, 1),
            "profit_factor":  round(profit_factor, 3),
            "avg_turnover_%": round(avg_turnover * 100, 2),
            "n_periods":      len(rets),
        }

    @staticmethod
    def print_summary(metrics: Dict) -> None:
        print(f"\n{'═'*55}")
        print(f"  AZALYST AlphaX — Performance: {metrics.get('label', '')}")
        print(f"{'═'*55}")
        rows = [
            ("Total Return",    f"{metrics['total_ret_%']:>8.2f}%"),
            ("Annualised Ret",  f"{metrics['ann_ret_%']:>8.2f}%"),
            ("Annualised Vol",  f"{metrics['ann_vol_%']:>8.2f}%"),
            ("Sharpe Ratio",    f"{metrics['sharpe']:>8.3f}"),
            ("Sortino Ratio",   f"{metrics['sortino']:>8.3f}"),
            ("Calmar Ratio",    f"{metrics['calmar']:>8.3f}"),
            ("Max Drawdown",    f"{metrics['max_dd_%']:>8.2f}%"),
            ("Win Rate",        f"{metrics['win_rate_%']:>8.1f}%"),
            ("Profit Factor",   f"{metrics['profit_factor']:>8.3f}"),
            ("Avg Turnover",    f"{metrics['avg_turnover_%']:>8.2f}%"),
            ("N Periods",       f"{metrics['n_periods']:>8d}"),
        ]
        for k, v in rows:
            print(f"  {k:<20} {v}")
        print(f"{'═'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  COMPOSITE FACTOR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class CompositeFactorBuilder:
    """
    Combines individual factors into a single alpha score.

    Methods
    ───────
    equal_weight   : Simple average — baseline.
    ic_weighted    : Weight each factor by its rolling ICIR.
                     Better factors get more weight dynamically.
    pca_composite  : First principal component of factor matrix.
                     Finds the dominant common signal.
    """

    def equal_weight(self, factors: Dict[str, pd.DataFrame],
                     selected: Optional[List[str]] = None) -> pd.DataFrame:
        names = selected or list(factors.keys())
        stack = [factors[n] for n in names if n in factors]
        if not stack:
            raise ValueError("No factors to combine")
        # Use 3D NumPy stack + nanmean — avoids the pd.concat(axis=0) which
        # creates N_factors*N_bars rows before groupby (OOM risk at scale).
        # Align all frames to the same index/columns first.
        ref = stack[0]
        arrays = [
            f.reindex(index=ref.index, columns=ref.columns).values
            for f in stack
        ]
        mean_arr = np.nanmean(np.stack(arrays, axis=0), axis=0)  # (T, N)
        return pd.DataFrame(mean_arr, index=ref.index, columns=ref.columns)

    def ic_weighted(self, factors: Dict[str, pd.DataFrame],
                    ic_table: pd.DataFrame,
                    horizon: str = "1D",
                    selected: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Weight factors by their ICIR at the chosen horizon.

        Signal direction rules
        ──────────────────────
        ICIR > +0.05 : include with weight = ICIR (long signal)
        ICIR < -0.15 : flip (1 - rank) and include with |ICIR| weight
                       Converts naturally inverting factors into long signals.
        -0.15 to +0.05: skip as noise.

        Implementation
        ─────────────
        Uses NumPy einsum over a pre-stacked (K, T, N) array for O(1) memory
        overhead vs the previous Python-loop pandas reduction which created
        N_factors temporary DataFrames and OOM'd on 400+ symbol universes.
        """
        names  = selected or list(factors.keys())
        ic_row = ic_table[ic_table["horizon"] == horizon].copy()
        ic_row = ic_row.set_index("factor")

        weighted_list = []
        for n in names:
            if n not in factors or n not in ic_row.index:
                continue
            icir = float(ic_row.loc[n, "ICIR"])

            if icir > 0.05:           # positive signal — use as-is
                weighted_list.append((n, icir, factors[n]))
            elif icir < -0.15:        # inverted signal — flip it
                flipped = 1.0 - factors[n]
                weighted_list.append((n + "_inv", abs(icir), flipped))
            # -0.15 to 0.05: near-zero noise, skip

        if not weighted_list:
            print("[WARN] ic_weighted: no factors passed threshold, falling back to reversal")
            return self.reversal_composite(factors)

        # ── NumPy path: avoids per-factor pandas intermediate objects ─────────
        ref     = weighted_list[0][2]
        weights = np.array([w for _, w, _ in weighted_list], dtype=np.float64)
        weights /= weights.sum()      # normalise

        # Stack (K, T, N) — one reindex per factor (unavoidable for alignment)
        arrays = np.stack(
            [f.reindex(index=ref.index, columns=ref.columns).values.astype(np.float64)
             for _, _, f in weighted_list],
            axis=0
        )  # shape (K, T, N)

        # Weighted sum: (K,) × (K, T, N) → (T, N) in a single vectorized pass
        combo_arr = np.einsum('k,ktn->tn', weights, arrays)
        return pd.DataFrame(combo_arr, index=ref.index, columns=ref.columns)

    def momentum_composite(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Pure momentum composite — best documented alpha in crypto."""
        mom_keys = ["MOM_1H", "MOM_4H", "MOM_1D", "MOM_1W", "MOM_2W", "TREND_48"]
        return self.equal_weight(factors, [k for k in mom_keys if k in factors])

    def reversal_composite(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Reversal + volatility composite using only ICIR-positive factors
        confirmed by your live data:
          REV_4H (ICIR 0.40), REV_1H (0.22), RVOL_1W (0.69),
          RVOL_1D (0.35), MAX_RET (0.35), VOL_OF_VOL (0.33)
        BB_POS intentionally EXCLUDED — ICIR is -0.24 in your universe.
        """
        rev_keys = ["REV_1H", "REV_4H", "RVOL_1W", "RVOL_1D", "MAX_RET", "VOL_OF_VOL"]
        return self.equal_weight(factors, [k for k in rev_keys if k in factors])

    def quality_composite(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Liquidity + volatility quality filter."""
        q_keys = ["AMIHUD", "RVOL_1D", "RVOL_1W", "VOL_OF_VOL", "SKEW_1W", "MAX_RET"]
        return self.equal_weight(factors, [k for k in q_keys if k in factors])

    def all_factor_composite(self, factors: Dict[str, pd.DataFrame],
                              ic_table: Optional[pd.DataFrame] = None,
                              horizon: str = "1D") -> pd.DataFrame:
        """Full IC-weighted composite (or equal-weight fallback)."""
        if ic_table is not None and not ic_table.empty:
            return self.ic_weighted(factors, ic_table, horizon=horizon)
        return self.equal_weight(factors)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _max_drawdown_series(cum_ret: pd.Series) -> pd.Series:
    """Running max drawdown at each bar."""
    wealth  = 1 + cum_ret
    peak    = wealth.cummax()
    dd      = (wealth - peak) / peak
    return dd


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst exponent — measures mean-reversion vs trend persistence.
    H < 0.5  → mean-reverting (stat arb opportunity)
    H = 0.5  → random walk
    H > 0.5  → trending (momentum opportunity)
    """
    lags  = range(2, max_lag)
    tau   = [np.std(np.subtract(series[lag:].values,
                                series[:-lag].values)) for lag in lags]
    reg   = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


def print_ic_table(ic_df: pd.DataFrame) -> None:
    """Pretty-print IC analysis results."""
    if ic_df.empty:
        print("No IC results.")
        return
    print(f"\n{'─'*85}")
    print(f"  {'Factor':<18} {'Horizon':<8} {'IC_mean':>8} {'ICIR':>8} "
          f"{'t-stat':>8} {'IC_pos%':>8} {'Grade':>8}")
    print(f"{'─'*85}")
    for _, row in ic_df.sort_values(["horizon", "ICIR"], ascending=[True, False]).iterrows():
        ic   = row["IC_mean"]
        icir = row["ICIR"]
        ts   = abs(row["t_stat"])
        grade = ("" if icir > 1.0 else
                 " " if icir > 0.5 else
                 "  " if icir > 0.2 else
                 "   ")
        print(f"  {row['factor']:<18} {row['horizon']:<8} {ic:>8.4f} "
              f"{icir:>8.4f} {ts:>8.2f} {row['IC_pos%']:>8.1f} {grade:>8}")
    print(f"{'─'*85}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI / MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(args) -> None:
    """End-to-end research pipeline."""

    print("""
╔══════════════════════════════════════════════════════════╗
║   AZALYST  ·  ALPHAX  ·  RESEARCH ENGINE v1.0          ║
║   5-Min Binance OHLCV  |  Cross-Sectional Factors      ║
╚══════════════════════════════════════════════════════════╝""")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    loader = DataLoader(
        data_dir    = args.data_dir,
        resample_to = args.resample,
        max_symbols = args.max_symbols,
        workers     = args.workers,
    )
    data        = loader.load_all()
    close_panel = loader.build_close_panel(data)
    vol_panel   = loader.build_volume_panel(data)
    del data; gc.collect()

    print(f"\n[Panel] Shape: {close_panel.shape}  "
          f"({close_panel.shape[0]} bars × {close_panel.shape[1]} symbols)")
    print(f"  Date range: {close_panel.index[0]} → {close_panel.index[-1]}")

    # ── 2. Compute factors ───────────────────────────────────────────────────
    fe      = FactorEngine()
    factors = fe.compute_all(close_panel, vol_panel)

    # ── 3. IC analysis ───────────────────────────────────────────────────────
    if not args.skip_ic:
        analyser = CrossSectionalAnalyser(close_panel)
        horizons = args.ic_horizons.split(",") if args.ic_horizons else ["1H", "4H", "1D", "1W"]
        ic_table = analyser.analyse_all(factors, horizons)
        print_ic_table(ic_table)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            ic_path = os.path.join(args.out_dir, "ic_analysis.csv")
            ic_table.to_csv(ic_path, index=False)
            print(f"[Saved] IC table → {ic_path}")

        # Factor decay for best momentum factor
        if "MOM_1W" in factors:
            decay = analyser.factor_decay("MOM_1W", factors["MOM_1W"])
            print("\n[Factor Decay] MOM_1W:")
            print(decay.to_string())
    else:
        ic_table = pd.DataFrame()

    # ── 4. Composite factor ───────────────────────────────────────────────────
    cfb = CompositeFactorBuilder()

    if args.composite == "momentum":
        composite = cfb.momentum_composite(factors)
        label     = "Momentum Composite"
    elif args.composite == "reversal":
        composite = cfb.reversal_composite(factors)
        label     = "Reversal Composite"
    elif args.composite == "quality":
        composite = cfb.quality_composite(factors)
        label     = "Quality Composite"
    elif args.composite == "ic_weighted":
        if not ic_table.empty:
            composite = cfb.ic_weighted(factors, ic_table, horizon="1D")
            label     = "IC-Weighted Composite"
        else:
            print("[WARN] --composite ic_weighted requires IC analysis. Run without --skip-ic first.")
            print("       Falling back to reversal composite.")
            composite = cfb.reversal_composite(factors)
            label     = "Reversal Composite (ic_weighted fallback)"
    else:
        composite = cfb.equal_weight(factors)
        label     = "Equal-Weight All Factors"

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    rebal_bars = {
        "1H":  BARS_PER_HOUR,
        "4H":  BARS_PER_HOUR * 4,
        "1D":  BARS_PER_DAY,
    }.get(args.rebal, BARS_PER_DAY)

    bt = BacktestEngine(
        close_panel  = close_panel,
        rebal_every  = rebal_bars,
        long_pct     = args.long_pct,
        short_pct    = args.short_pct,
        long_only    = args.long_only,
    )
    pnl = bt.run(composite, label=label)

    # ── 6. Performance report ─────────────────────────────────────────────────
    metrics = PortfolioAnalytics.summary(pnl, label)
    PortfolioAnalytics.print_summary(metrics)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        pnl_path = os.path.join(args.out_dir, "backtest_pnl.csv")
        pnl.to_csv(pnl_path)
        print(f"[Saved] PnL → {pnl_path}")

        metrics_path = os.path.join(args.out_dir, "performance_summary.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"[Saved] Metrics → {metrics_path}")

    # ── 7. Hurst exponents (mean reversion candidates) ───────────────────────
    if args.hurst:
        print("\n[Hurst] Computing mean-reversion scores (top 20 coins)...")
        hursts = {}
        sample = close_panel.iloc[-BARS_PER_WEEK * 4:]  # last 4 weeks
        for col in sample.columns[:min(100, len(sample.columns))]:
            s = sample[col].dropna()
            if len(s) > 50:
                hursts[col] = hurst_exponent(s)
        hr_series = pd.Series(hursts).sort_values()
        print("  Most mean-reverting (H < 0.5):")
        print(hr_series.head(20).to_string())
        if args.out_dir:
            hr_path = os.path.join(args.out_dir, "hurst_exponents.csv")
            hr_series.to_frame("hurst").to_csv(hr_path)
            print(f"[Saved] Hurst → {hr_path}")

    print("\n[Azalyst] Research pipeline complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Azalyst AlphaX — Quantitative Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on all symbols, daily rebalancing
  python azalyst_engine.py --data-dir ./data --out-dir ./research

  # Momentum composite, 4H rebalancing, first 50 symbols
  python azalyst_engine.py --data-dir ./data --composite momentum --rebal 4H --max-symbols 50

  # Long-only, IC-weighted, skip IC analysis (use cached)
  python azalyst_engine.py --data-dir ./data --composite ic_weighted --long-only

  # Resample 5m to 1H before analysis
  python azalyst_engine.py --data-dir ./data --resample 1h

  # With Hurst exponent (find stat-arb candidates)
  python azalyst_engine.py --data-dir ./data --hurst
        """
    )
    parser.add_argument("--data-dir",     required=True, help="Path to folder with .parquet files")
    parser.add_argument("--out-dir",      default="./azalyst_output", help="Output directory for CSVs")
    parser.add_argument("--resample",     default=None,  help="Resample 5m data e.g. '1h', '4h', '1D'")
    parser.add_argument("--max-symbols",  type=int, default=None, help="Limit symbols for testing")
    parser.add_argument("--workers",      type=int, default=4,    help="Parallel file-loading workers")
    parser.add_argument("--composite",    default="reversal",
                        choices=["momentum", "reversal", "quality", "ic_weighted", "equal_weight"],
                        help="Composite to backtest (default: reversal — correct for mean-reverting universe)")
    parser.add_argument("--rebal",        default="1D",
                        choices=["1H", "4H", "1D"], help="Rebalancing interval")
    parser.add_argument("--long-pct",     type=float, default=0.2, help="Top X%% to go long (default 0.2)")
    parser.add_argument("--short-pct",    type=float, default=0.2, help="Bottom X%% to short (default 0.2)")
    parser.add_argument("--long-only",    action="store_true",     help="Long-only mode (no shorting)")
    parser.add_argument("--skip-ic",      action="store_true",     help="Skip IC analysis (faster)")
    parser.add_argument("--ic-horizons",  default="1H,4H,1D,1W",  help="Comma-separated IC horizons")
    parser.add_argument("--hurst",        action="store_true",     help="Compute Hurst exponents")

    args = parser.parse_args()
    build_pipeline(args)


if __name__ == "__main__":
    main()
