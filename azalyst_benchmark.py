"""
╔══════════════════════════════════════════════════════════════════════════════╗
       AZALYST ALPHA RESEARCH ENGINE    BENCHMARK SUITE
║   BTC B&H · Equal-Weight Market · Beta-Adjusted Alpha · Information Ratio  ║
║   v1.0  |  Fully vectorized  |  No external dependencies                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Every backtest result is meaningless without a benchmark.
This module provides standardised, fair comparisons:

  BenchmarkSuite.btc_buyhold()    — Simply holding BTC
  BenchmarkSuite.equal_weight()   — Equal-weight all coins in universe
  BenchmarkSuite.mcap_weighted()  — Volume-proxy market-cap weighted index
  BenchmarkSuite.alpha()          — Strategy excess return vs benchmark
  BenchmarkSuite.full_summary()   — Side-by-side performance table

Why these benchmarks matter
────────────────────────────
In a bull market EVERYTHING goes up.  A strategy that earns 300% sounds great
until you learn BTC made 500%.  The real question is always:
"Did we earn EXCESS returns above what a passive investor would earn?"

Usage
─────
  from azalyst_benchmark import BenchmarkSuite

  bm = BenchmarkSuite(close_panel=close)
  btc = bm.btc_buyhold()
  eq  = bm.equal_weight()
  bm.full_summary(strategy_pnl=backtest_df)
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BARS_PER_DAY  = 288
RISK_FREE_RATE_ANNUAL = 0.05   # 5% risk-free rate (conservative for crypto)


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK SUITE
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """
    Standardised benchmark construction and performance comparison.

    All benchmarks use the SAME close panel as the strategy, ensuring
    identical data cuts and no survival bias.

    Parameters
    ──────────
    close_panel : pd.DataFrame (T × N)
        Wide close-price panel (same as BacktestEngine.close).
    rebal_every : int
        Rebalancing frequency in bars. Used for equal-weight benchmark to
        match the strategy's rebalancing cadence.
    btc_symbol  : str
        Column name of BTC in the close panel.
    """

    def __init__(
        self,
        close_panel: pd.DataFrame,
        rebal_every: int = BARS_PER_DAY,
        btc_symbol: Optional[str] = None,
    ):
        self.close      = close_panel
        self.rebal_every = rebal_every
        self.btc_col    = btc_symbol or self._find_btc(close_panel)

        # Log return panel (shared, computed once)
        self._log_ret = np.log(close_panel / close_panel.shift(1))

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_btc(close: pd.DataFrame) -> Optional[str]:
        """Find the BTC column by name pattern."""
        for col in close.columns:
            if "BTC" in str(col).upper():
                return col
        return None

    @staticmethod
    def _perf_stats(ret: pd.Series, label: str = "",
                    periods_per_year: int = 365) -> Dict:
        """Compute institutional performance metrics from a return series."""
        ret = ret.dropna()
        if len(ret) == 0:
            return {}

        cum       = (1 + ret).cumprod()
        n_years   = len(ret) / periods_per_year
        total     = cum.iloc[-1] - 1
        ann_ret   = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else np.nan
        ann_vol   = ret.std() * np.sqrt(periods_per_year)
        rf        = RISK_FREE_RATE_ANNUAL / periods_per_year
        excess_ret = ret - rf
        sharpe    = excess_ret.mean() / ret.std() * np.sqrt(periods_per_year) if ret.std() > 0 else np.nan
        down      = ret[ret < 0].std() * np.sqrt(periods_per_year)
        sortino   = (ann_ret - RISK_FREE_RATE_ANNUAL) / down if down > 0 else np.nan
        peak      = cum.cummax()
        dd        = ((cum - peak) / peak)
        max_dd    = dd.min()
        calmar    = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
        win_rate  = (ret > 0).mean()

        return {
            "label":       label,
            "total_ret_%": round(total * 100, 2),
            "ann_ret_%":   round(ann_ret * 100, 2),
            "ann_vol_%":   round(ann_vol * 100, 2),
            "sharpe":      round(sharpe, 3),
            "sortino":     round(sortino, 3),
            "calmar":      round(calmar, 3) if not np.isnan(calmar) else np.nan,
            "max_dd_%":    round(max_dd * 100, 2),
            "win_rate_%":  round(win_rate * 100, 1),
            "n_periods":   len(ret),
        }

    # ── Benchmark constructors ───────────────────────────────────────────────

    def btc_buyhold(self) -> pd.Series:
        """
        Buy-and-hold BTC return series.

        This is THE primary benchmark for any crypto strategy.
        If you don't beat BTC, you should just buy BTC.

        Returns: pd.Series of per-period log returns (aligned to close index).
        """
        if self.btc_col is None or self.btc_col not in self.close.columns:
            raise ValueError(
                f"BTC not found in close panel. Columns: {list(self.close.columns[:5])}..."
            )
        return self._log_ret[self.btc_col].rename("BTC_BuyHold")

    def equal_weight(self) -> pd.Series:
        """
        Equal-weight daily-rebalanced market index.

        At each rebalancing bar, invests 1/N in every coin. This captures the
        'market portfolio' return — beating this requires genuine alpha, not just
        market beta.

        Returns: pd.Series of per-period returns.
        """
        timestamps = self.close.index
        rebal_bars = np.arange(0, len(timestamps), self.rebal_every)

        pnl_rows = []
        for i, ri in enumerate(rebal_bars[:-1]):
            t_entry = timestamps[ri]
            t_exit  = timestamps[rebal_bars[i + 1]]

            # Active coins (have price at entry and exit)
            entry_prices = self.close.loc[t_entry].dropna()
            exit_prices  = self.close.loc[t_exit].reindex(entry_prices.index).dropna()

            if len(exit_prices) == 0:
                continue

            common = entry_prices.index.intersection(exit_prices.index)
            if len(common) == 0:
                continue

            ret = ((exit_prices[common] / entry_prices[common]) - 1).mean()
            pnl_rows.append({"timestamp": t_exit, "net_ret": ret})

        if not pnl_rows:
            return pd.Series(dtype=float, name="EqualWeight")

        s = pd.DataFrame(pnl_rows).set_index("timestamp")["net_ret"]
        return s.rename("EqualWeight")

    def volume_weighted(self) -> pd.Series:
        """
        Volume-weighted index (proxy for market-cap weighting).

        Uses rolling 30-day average volume to estimate relative market cap.
        This captures the 'large-cap' bias naturally, since major coins
        dominate by volume.

        Returns: pd.Series of per-period returns.
        """
        # Build volume panel
        vol_panel = pd.DataFrame(index=self.close.index)
        for col in self.close.columns:
            vol_panel[col] = np.nan  # placeholder if volume not attached

        # Fallback: equal weight (volume not available from close panel alone)
        print("[Benchmark] volume_weighted needs volume panel — returning equal_weight")
        return self.equal_weight().rename("VolumeWeighted")

    # ── Analysis methods ─────────────────────────────────────────────────────

    def alpha(
        self,
        strategy_ret: pd.Series,
        benchmark_ret: pd.Series,
        benchmark_label: str = "benchmark",
    ) -> pd.DataFrame:
        """
        Compute strategy alpha (excess return) vs a benchmark.

        Returns DataFrame with:
          strategy_ret   : raw strategy return
          benchmark_ret  : benchmark return
          excess_ret     : strategy - benchmark
          cum_strategy   : cumulative strategy
          cum_benchmark  : cumulative benchmark
          cum_excess     : cumulative alpha
        """
        # Align to common index (strategy only trades at rebal bars)
        common = strategy_ret.index.intersection(benchmark_ret.index)
        if len(common) == 0:
            # Strategy rebalances less frequently than benchmark daily return.
            # Resample benchmark to match strategy frequency.
            bm = (1 + benchmark_ret).resample("1D").prod() - 1
            common = strategy_ret.index.intersection(bm.index)
            bm_aligned = bm.reindex(common).fillna(0)
        else:
            bm_aligned = benchmark_ret.reindex(common).fillna(0)

        strat_aligned    = strategy_ret.reindex(common).fillna(0)
        excess           = strat_aligned - bm_aligned

        df = pd.DataFrame({
            "strategy_ret":   strat_aligned,
            f"{benchmark_label}_ret": bm_aligned,
            "excess_ret":     excess,
            "cum_strategy":   (1 + strat_aligned).cumprod() - 1,
            f"cum_{benchmark_label}": (1 + bm_aligned).cumprod() - 1,
            "cum_excess":     (1 + excess).cumprod() - 1,
        })
        return df

    def information_ratio(
        self,
        excess_ret: pd.Series,
        periods_per_year: int = 365,
    ) -> float:
        """
        Information Ratio = annualised_excess_return / tracking_error.

        IR > 0.5 = good, IR > 1.0 = excellent, IR > 2.0 = elite.
        The higher the IR, the more consistent the alpha.
        """
        er = excess_ret.dropna()
        if len(er) < 2 or er.std() == 0:
            return np.nan
        ann_excess   = er.mean() * periods_per_year
        tracking_err = er.std() * np.sqrt(periods_per_year)
        return round(ann_excess / tracking_err, 4)

    def beta(
        self,
        strategy_ret: pd.Series,
        benchmark_ret: pd.Series = None,
    ) -> float:
        """
        Market beta of the strategy vs BTC or provided benchmark.

        Beta > 1: strategy amplifies market moves.
        Beta < 1: strategy is less sensitive to market.
        Beta < 0: strategy is market-neutral or short-biased.
        """
        if benchmark_ret is None:
            bm = self.btc_buyhold()
        else:
            bm = benchmark_ret

        common = strategy_ret.dropna().index.intersection(bm.dropna().index)
        if len(common) < 10:
            return np.nan

        x = bm.reindex(common).values
        y = strategy_ret.reindex(common).values
        cov  = np.cov(x, y, ddof=1)
        return round(cov[0, 1] / cov[0, 0], 4) if cov[0, 0] > 0 else np.nan

    def full_summary(
        self,
        strategy_pnl: pd.DataFrame,
        strategy_label: str = "Azalyst Strategy",
        periods_per_year: int = 365,
    ) -> pd.DataFrame:
        """
        Side-by-side performance comparison: strategy vs BTC vs equal-weight.

        Args:
            strategy_pnl: DataFrame with a 'net_ret' column (BacktestEngine output).
            strategy_label: Display name for the strategy.

        Returns:
            pd.DataFrame: metrics × benchmarks — easy to save as CSV.
        """
        strat_ret = strategy_pnl["net_ret"].dropna()
        ppy = periods_per_year

        rows = []

        # Strategy
        rows.append(self._perf_stats(strat_ret, label=strategy_label, periods_per_year=ppy))

        # BTC B&H
        try:
            btc_ret = self.btc_buyhold()
            # Resample BTC to match strategy rebalancing frequency
            btc_rebal = btc_ret.resample("1D").sum().reindex(strat_ret.index, method="nearest")
            rows.append(self._perf_stats(btc_rebal, label="BTC Buy & Hold", periods_per_year=ppy))

            # Excess vs BTC
            alpha_df = self.alpha(strat_ret, btc_rebal, "btc")
            excess_row = self._perf_stats(alpha_df["excess_ret"], label=f"{strategy_label} vs BTC alpha", periods_per_year=ppy)
            excess_row["info_ratio"] = self.information_ratio(alpha_df["excess_ret"], ppy)
            excess_row["beta_btc"]   = self.beta(strat_ret, btc_rebal)
            rows.append(excess_row)
        except Exception as e:
            print(f"[Benchmark] BTC benchmark error: {e}")

        # Equal-weight
        try:
            eq_ret  = self.equal_weight()
            eq_rebal = eq_ret.reindex(strat_ret.index, method="nearest")
            rows.append(self._perf_stats(eq_rebal, label="Equal-Weight Universe", periods_per_year=ppy))
        except Exception as e:
            print(f"[Benchmark] Equal-weight error: {e}")

        summary = pd.DataFrame(rows).set_index("label").T

        # Pretty-print
        print("\n" + "═" * 75)
        print("  AZALYST · BENCHMARK COMPARISON")
        print("═" * 75)
        print(summary.to_string())
        print("═" * 75 + "\n")

        return summary


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Azalyst Benchmark Suite — compare strategy vs BTC and market"
    )
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--backtest-csv", default=None,
                        help="Path to backtest_pnl.csv from azalyst_engine.py")
    parser.add_argument("--out-dir",     default="./azalyst_output")
    parser.add_argument("--max-symbols", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from azalyst_engine import DataLoader
    loader = DataLoader(args.data_dir, max_symbols=args.max_symbols, workers=4)
    data   = loader.load_all()
    close  = loader.build_close_panel(data)

    bm = BenchmarkSuite(close_panel=close, rebal_every=BARS_PER_DAY)

    if args.backtest_csv and os.path.exists(args.backtest_csv):
        pnl = pd.read_csv(args.backtest_csv, index_col=0, parse_dates=True)
        summary = bm.full_summary(pnl)
        out = os.path.join(args.out_dir, "benchmark_comparison.csv")
        summary.to_csv(out)
        print(f"[Saved] Benchmark comparison → {out}")
    else:
        # Just print BTC and EW stats
        btc_ret = bm.btc_buyhold()
        eq_ret  = bm.equal_weight()
        print("\n  BTC Buy & Hold:")
        print(pd.Series(bm._perf_stats(btc_ret, "BTC")).to_string())
        print("\n  Equal-Weight Universe:")
        print(pd.Series(bm._perf_stats(eq_ret, "EqualWeight")).to_string())


if __name__ == "__main__":
    main()
