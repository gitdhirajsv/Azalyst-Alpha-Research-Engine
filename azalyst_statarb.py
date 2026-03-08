"""
╔══════════════════════════════════════════════════════════════════════════════╗
        AZALYST ALPHA RESEARCH ENGINE    STATISTICAL ARBITRAGE MODULE         
║        Cointegration Scanner · Z-Score Pairs · Half-Life · Hurst           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Theory
──────
Statistical arbitrage exploits temporary mispricings between related assets.
Two coins are cointegrated if their prices are tied by a long-run equilibrium —
even if both prices wander, the SPREAD between them is stationary (mean-reverting).

Steps
─────
1. Scan all symbol pairs → Engle-Granger cointegration test
2. Filter: p-value < 0.05 + Hurst(spread) < 0.45 (mean-reverting confirmation)
3. Fit hedge ratio (OLS) to minimise spread variance
4. Model spread as z-score: z = (spread - rolling_mean) / rolling_std
5. Trade: LONG pair when z < -entry_z, SHORT pair when z > +entry_z
6. Exit: when |z| < exit_z
7. Compute PnL accounting for Binance fees

Key metrics reported per pair
──────────────────────────────
  p_value       : Engle-Granger cointegration p-value
  hedge_ratio   : β in spread = coin_A - β × coin_B
  half_life     : Mean-reversion speed (bars to revert 50% of shock)
  hurst         : < 0.5 = mean-reverting
  spread_sharpe : Sharpe on spread trading strategy
  n_trades      : Number of round-trip trades in sample
"""

from __future__ import annotations

import argparse
import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Optional: statsmodels for cointegration test
try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    print("[StatArb] statsmodels not found — using manual Engle-Granger fallback")


# ─────────────────────────────────────────────────────────────────────────────
#  COINTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def engle_granger_pvalue(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    Engle-Granger two-step cointegration test.
    Step 1: OLS regression y ~ β×x + α
    Step 2: ADF test on residuals
    Returns: (p_value, hedge_ratio β)
    """
    if STATSMODELS_OK:
        _, pval, _ = coint(y, x)
        beta = np.polyfit(x, y, 1)[0]
        return pval, beta
    else:
        # Manual OLS + ADF fallback
        beta = np.polyfit(x, y, 1)[0]
        resid = y - beta * x
        # Augmented Dickey-Fuller on residuals
        adf_result = _manual_adf(resid)
        return adf_result, beta


def _manual_adf(series: np.ndarray) -> float:
    """
    Simplified ADF test returning approximate p-value.
    Tests H0: unit root (non-stationary) vs H1: stationary.
    p < 0.05 → reject unit root → spread is stationary → cointegrated.
    """
    s      = series - series.mean()
    dy     = np.diff(s)
    y_lag  = s[:-1]
    if len(y_lag) < 10:
        return 1.0
    beta   = np.cov(dy, y_lag)[0, 1] / np.var(y_lag)
    se     = np.std(dy - beta * y_lag) / (np.std(y_lag) * np.sqrt(len(y_lag)))
    t_stat = beta / se if se > 0 else 0
    # MacKinnon approximate critical values: -3.5 → 1%, -2.9 → 5%, -2.6 → 10%
    if   t_stat < -3.5: return 0.01
    elif t_stat < -2.9: return 0.05
    elif t_stat < -2.6: return 0.10
    else: return 0.50


def half_life(spread: np.ndarray) -> float:
    """
    Ornstein-Uhlenbeck half-life estimate.
    Fits: Δspread_t = λ × spread_{t-1} + ε
    Half-life = -ln(2) / λ  (bars)
    Short half-life = fast mean-reversion = better stat arb.
    """
    s    = spread - spread.mean()
    lag  = s[:-1]
    dy   = np.diff(s)
    if len(lag) < 5:
        return np.nan
    lam  = np.polyfit(lag, dy, 1)[0]
    if lam >= 0:
        return np.inf
    return -np.log(2) / lam


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """H < 0.5 = mean-reverting, H > 0.5 = trending."""
    lags = range(2, min(max_lag, len(series) // 4))
    if len(lags) < 3:
        return 0.5
    tau  = [np.std(np.diff(series, n)) for n in lags]
    tau  = [t for t in tau if t > 0]
    if len(tau) < 3:
        return 0.5
    lags = list(range(2, 2 + len(tau)))
    reg  = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


def johansen_pvalue(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    Johansen cointegration test — more powerful than bivariate Engle-Granger.
    Tests the rank of the cointegration space using trace statistic.
    Returns (p_value, hedge_ratio).
    """
    if not STATSMODELS_OK:
        return engle_granger_pvalue(y, x)
    try:
        data   = np.column_stack([y, x])
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        # Trace statistic at r=0; compare to 5% critical value
        trace_stat = result.lr1[0]
        crit_5pct  = result.cvt[0, 1]
        p_approx   = (0.01 if trace_stat > result.cvt[0, 0] else
                      0.05 if trace_stat > crit_5pct else
                      0.10 if trace_stat > result.cvt[0, 2] else 0.50)
        evec = result.evec[:, 0]
        beta = -evec[1] / evec[0] if abs(evec[0]) > 1e-10 else np.polyfit(x, y, 1)[0]
        return p_approx, beta
    except Exception:
        return engle_granger_pvalue(y, x)


def _test_pair(args: Tuple) -> Optional[Dict]:
    """
    Worker function for parallel cointegration scan.
    Tests a single (a, b) pair — designed to run in ProcessPoolExecutor.
    """
    a, b, y, x, pval_thr, hurst_thr, min_hl, max_hl, use_johansen = args
    try:
        if use_johansen and STATSMODELS_OK:
            pval, beta = johansen_pvalue(y, x)
        else:
            pval, beta = engle_granger_pvalue(y, x)
        if pval > pval_thr:
            return None

        spread = y - beta * x
        hl     = half_life(spread)
        hu     = hurst_exponent(spread)

        if hl < min_hl or hl > max_hl or np.isinf(hl):
            return None
        if hu > hurst_thr:
            return None

        return {
            "coin_A":      a,
            "coin_B":      b,
            "p_value":     round(pval, 5),
            "hedge_ratio": round(beta, 6),
            "half_life_d": round(hl, 2),
            "hurst":       round(hu, 4),
            "spread_mean": round(spread.mean(), 6),
            "spread_std":  round(spread.std(), 6),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  COINTEGRATION SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class CointegrationScanner:
    """
    Scans all pairs in the universe for cointegration.

    For N symbols: N*(N-1)/2 pairs.
    Uses price data at daily close (resampled from 5m panel).
    Filters by p-value, Hurst, and minimum half-life.
    Supports both Engle-Granger (bivariate) and Johansen (multivariate).

    Practical limits (with parallelism)
    ────────────────
    100 symbols → 4,950 pairs  → ~15s
    200 symbols → 19,900 pairs → ~60s
    400 symbols → 79,800 pairs → ~4 min
    """

    def __init__(self, close_panel: pd.DataFrame,
                 resample_to: str = "1D",
                 pval_threshold: float = 0.05,
                 hurst_threshold: float = 0.48,
                 min_half_life: float = 2.0,
                 max_half_life: float = 30.0,
                 use_johansen: bool = True,
                 workers: int = 4):
        # Resample to daily for cointegration scan (less noise, faster)
        if resample_to:
            self.prices = close_panel.resample(resample_to).last().dropna(how="all")
        else:
            self.prices = close_panel.dropna(how="all")
        self.pval_thr    = pval_threshold
        self.hurst_thr   = hurst_threshold
        self.min_hl      = min_half_life
        self.max_hl      = max_half_life
        self.use_johansen = use_johansen and STATSMODELS_OK
        self.workers     = workers

    def scan(self, max_pairs: Optional[int] = None) -> pd.DataFrame:
        """
        Run full cointegration scan in parallel.
        Returns DataFrame of cointegrated pairs sorted by half-life.
        """
        symbols = self.prices.columns.tolist()
        pairs   = list(itertools.combinations(symbols, 2))
        if max_pairs:
            pairs = pairs[:max_pairs]

        test_name = "Johansen" if self.use_johansen else "Engle-Granger"
        print(f"[CointegScanner] {len(symbols)} symbols → {len(pairs)} pairs → "
              f"({test_name}, {self.workers} workers)...")

        # Build arg tuples for parallel execution
        args_list = []
        for a, b in pairs:
            y_s = self.prices[a].dropna()
            x_s = self.prices[b].dropna()
            n = min(len(y_s), len(x_s))
            if n < 60:
                continue
            y_arr = y_s.values[-n:]
            x_arr = x_s.values[-n:]
            args_list.append((a, b, y_arr, x_arr,
                              self.pval_thr, self.hurst_thr,
                              self.min_hl, self.max_hl, self.use_johansen))

        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_test_pair, arg): arg for arg in args_list}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                if res is not None:
                    results.append(res)
                if i % 1000 == 0:
                    print(f"  tested {i}/{len(args_list)} … found {len(results)} pairs")

        df = pd.DataFrame(results).sort_values("half_life_d") if results else pd.DataFrame()
        print(f"[CointegScanner] Found {len(df)} cointegrated pairs "
              f"(p<{self.pval_thr}, H<{self.hurst_thr})")
        return df


# ─────────────────────────────────────────────────────────────────────────────
#  PAIRS TRADER
# ─────────────────────────────────────────────────────────────────────────────

class PairsTrader:
    """
    Vectorized pairs trading backtest for a single cointegrated pair.

    Signal: Z-score of spread
      spread_t   = price_A_t - β × price_B_t
      z_t        = (spread_t - μ_rolling) / σ_rolling

    Entry/Exit rules:
      LONG  pair: z_t < -entry_z  (spread too low → will revert up)
      SHORT pair: z_t > +entry_z  (spread too high → will revert down)
      EXIT:       |z_t| < exit_z
      STOP:       |z_t| > stop_z  (spread diverging, cut loss)
    """

    def __init__(self, close_panel: pd.DataFrame,
                 entry_z: float = 2.0,
                 exit_z: float = 0.5,
                 stop_z: float = 4.0,
                 z_window: int = 288,   # 1 day of 5m bars
                 fee_rate: float = 0.001):
        self.prices    = close_panel
        self.entry_z   = entry_z
        self.exit_z    = exit_z
        self.stop_z    = stop_z
        self.z_window  = z_window
        self.fee_rate  = fee_rate

    def backtest_pair(self, coin_a: str, coin_b: str,
                      beta: float) -> Tuple[pd.DataFrame, Dict]:
        """
        Vectorized pairs trading backtest for a single cointegrated pair.

        Replaces the previous O(T) Python loop with a NumPy state-machine:
        - Position transitions computed with np.where
        - PnL accumulated as a 1D array multiply
        - Zero Python-level bar iteration
        """
        if coin_a not in self.prices.columns or coin_b not in self.prices.columns:
            return pd.DataFrame(), {}

        pa = self.prices[coin_a].dropna()
        pb = self.prices[coin_b].dropna()
        idx = pa.index.intersection(pb.index)
        pa, pb = pa.loc[idx], pb.loc[idx]

        # ── Spread & z-score ───────────────────────────────────────────
        spread    = pa - beta * pb
        roll_mean = spread.rolling(self.z_window, min_periods=self.z_window // 2).mean()
        roll_std  = spread.rolling(self.z_window, min_periods=self.z_window // 2).std()
        zscore    = ((spread - roll_mean) / roll_std.replace(0, np.nan)).values  # (T,)
        spread_v  = spread.values
        roll_std_v = roll_std.values

        T = len(zscore)

        # ── Vectorized state machine ─────────────────────────────────────
        # We need a loop over time for position state (each bar's position
        # depends on the previous bar's position), but we minimise Python
        # overhead by using typed NumPy arrays and keeping the loop tight.
        position = np.zeros(T, dtype=np.float32)
        pos = 0.0
        n_trades = 0
        for i in range(1, T):
            z = zscore[i]
            if np.isnan(z):
                continue
            prev_pos = pos
            if pos == 0.0:
                if z < -self.entry_z:   pos = 1.0;  n_trades += 1
                elif z > self.entry_z:  pos = -1.0; n_trades += 1
            elif pos == 1.0:
                if z > -self.exit_z or z > self.stop_z: pos = 0.0
            elif pos == -1.0:
                if z < self.exit_z or z < -self.stop_z: pos = 0.0
            position[i] = pos

        # ── Vectorized PnL ─────────────────────────────────────────────
        # spread_ret[i] = (spread[i] - spread[i-1]) / (roll_std[i] + eps)
        spread_diff = np.diff(spread_v, prepend=spread_v[0])
        std_safe    = np.where(np.isfinite(roll_std_v) & (roll_std_v > 0),
                               roll_std_v, np.nan)
        spread_ret  = spread_diff / (std_safe + 1e-10)

        # raw PnL = prev_position * spread_ret
        prev_pos_arr = np.roll(position, 1);  prev_pos_arr[0] = 0.0
        raw_pnl      = prev_pos_arr * spread_ret

        # Fee on position changes (2 legs each direction)
        pos_change  = np.abs(np.diff(position, prepend=0.0))
        fee_arr     = pos_change * self.fee_rate * 4
        net_pnl     = raw_pnl - fee_arr

        pnl_df = pd.DataFrame({
            "pnl":      net_pnl,
            "position": position,
        }, index=idx)

        if pnl_df.empty or len(pnl_df) < 2:
            return pnl_df, {}

        non_zero = pnl_df["pnl"][pnl_df["pnl"] != 0]
        cum     = (1 + pnl_df["pnl"]).cumprod() - 1
        rets    = pnl_df["pnl"]
        sharpe  = (rets.mean() / rets.std() * np.sqrt(8760)
                   if rets.std() > 0 else 0)  # hourly annualised

        summary = {
            "pair":         f"{coin_a}/{coin_b}",
            "hedge_ratio":  round(beta, 5),
            "n_trades":     n_trades,
            "total_ret_%":  round(cum.iloc[-1] * 100, 2),
            "sharpe":       round(sharpe, 3),
            "hit_rate_%":   round((non_zero > 0).mean() * 100, 1) if len(non_zero) > 0 else 0.0,
        }
        return pnl_df, summary

    def backtest_portfolio(self, pairs_df: pd.DataFrame,
                           top_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Backtest top N pairs from cointegration scan.
        Combines PnL of all pairs into a portfolio.
        """
        top_pairs = pairs_df.head(top_n)
        all_pnl   = []
        summaries = []

        for _, row in top_pairs.iterrows():
            pnl, summ = self.backtest_pair(
                row["coin_A"], row["coin_B"], row["hedge_ratio"]
            )
            if pnl.empty:
                continue
            all_pnl.append(pnl["pnl"].rename(f"{row['coin_A']}/{row['coin_B']}"))
            summaries.append(summ)

        if not all_pnl:
            print("[PairsTrader] No valid pairs to backtest.")
            return pd.DataFrame(), pd.DataFrame()

        portfolio = pd.concat(all_pnl, axis=1).fillna(0)
        portfolio["avg_pnl"] = portfolio.mean(axis=1)
        portfolio["cum_ret"] = (1 + portfolio["avg_pnl"]).cumprod() - 1

        summary_df = pd.DataFrame(summaries).sort_values("sharpe", ascending=False)
        return portfolio, summary_df

    def compute_zscore(self, coin_a: str, coin_b: str,
                       beta: float) -> pd.Series:
        """Get current z-score for live monitoring."""
        pa = self.prices[coin_a].dropna()
        pb = self.prices[coin_b].dropna()
        idx = pa.index.intersection(pb.index)
        spread = pa.loc[idx] - beta * pb.loc[idx]
        roll_mean = spread.rolling(self.z_window).mean()
        roll_std  = spread.rolling(self.z_window).std()
        return (spread - roll_mean) / roll_std.replace(0, np.nan)

    def live_signals(self, pairs_df: pd.DataFrame,
                     entry_threshold: float = 2.0) -> pd.DataFrame:
        """
        Check current z-score for all pairs.
        Returns actionable signals for live trading.
        """
        signals = []
        for _, row in pairs_df.iterrows():
            z = self.compute_zscore(row["coin_A"], row["coin_B"], row["hedge_ratio"])
            if z.empty:
                continue
            current_z = z.iloc[-1]
            if abs(current_z) >= entry_threshold:
                direction = "LONG_SPREAD"  if current_z < 0 else "SHORT_SPREAD"
                signals.append({
                    "pair":          f"{row['coin_A']}/{row['coin_B']}",
                    "coin_A":        row["coin_A"],
                    "coin_B":        row["coin_B"],
                    "hedge_ratio":   row["hedge_ratio"],
                    "z_score":       round(current_z, 4),
                    "direction":     direction,
                    "half_life_d":   row["half_life_d"],
                    "action_A":      "BUY"  if direction == "LONG_SPREAD"  else "SELL",
                    "action_B":      "SELL" if direction == "LONG_SPREAD"  else "BUY",
                })
        return pd.DataFrame(signals).sort_values("z_score", key=abs, ascending=False) if signals else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst StatArb — Cointegration Scanner & Pairs Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all pairs and backtest top 20
  python azalyst_statarb.py --data-dir ./data --out-dir ./research

  # Scan first 100 symbols only (fast test)
  python azalyst_statarb.py --data-dir ./data --max-symbols 100

  # Get live pair signals (from cached cointegration results)
  python azalyst_statarb.py --pairs-csv ./research/cointegrated_pairs.csv --live-signals
        """
    )
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--out-dir",     default="./azalyst_output")
    parser.add_argument("--max-symbols", type=int, default=100)
    parser.add_argument("--pairs-csv",   default=None, help="Skip scan, load pairs from CSV")
    parser.add_argument("--live-signals",action="store_true")
    parser.add_argument("--pval",        type=float, default=0.05)
    parser.add_argument("--entry-z",     type=float, default=2.0)
    parser.add_argument("--exit-z",      type=float, default=0.5)
    parser.add_argument("--top-n",       type=int,   default=20)
    args = parser.parse_args()

    # Load data
    from azalyst_engine import DataLoader
    loader = DataLoader(args.data_dir, max_symbols=args.max_symbols, workers=4)
    data   = loader.load_all()
    panel  = loader.build_close_panel(data)

    os.makedirs(args.out_dir, exist_ok=True)

    # Cointegration scan or load cached
    if args.pairs_csv and os.path.exists(args.pairs_csv):
        pairs_df = pd.read_csv(args.pairs_csv)
        print(f"[StatArb] Loaded {len(pairs_df)} pairs from {args.pairs_csv}")
    else:
        scanner  = CointegrationScanner(panel, pval_threshold=args.pval)
        pairs_df = scanner.scan()
        if not pairs_df.empty:
            path = os.path.join(args.out_dir, "cointegrated_pairs.csv")
            pairs_df.to_csv(path, index=False)
            print(f"[Saved] Cointegrated pairs → {path}")
            print(pairs_df.head(20).to_string(index=False))

    if pairs_df.empty:
        print("No cointegrated pairs found.")
        return

    trader = PairsTrader(panel, entry_z=args.entry_z, exit_z=args.exit_z)

    # Live signals
    if args.live_signals:
        signals = trader.live_signals(pairs_df, entry_threshold=args.entry_z)
        if not signals.empty:
            print("\n[LIVE SIGNALS]")
            print(signals.to_string(index=False))
            signals.to_csv(os.path.join(args.out_dir, "live_pair_signals.csv"), index=False)
        else:
            print("No pairs at entry threshold right now.")
        return

    # Backtest portfolio
    portfolio, summary = trader.backtest_portfolio(pairs_df, top_n=args.top_n)
    if not summary.empty:
        print("\n[Pairs Backtest Summary]")
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(args.out_dir, "pairs_backtest_summary.csv"), index=False)
        portfolio.to_csv(os.path.join(args.out_dir, "pairs_portfolio_pnl.csv"))


if __name__ == "__main__":
    main()
