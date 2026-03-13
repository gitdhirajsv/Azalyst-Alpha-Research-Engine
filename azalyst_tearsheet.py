"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    FACTOR TEAR SHEET GENERATOR         
║        IC Analysis · Quantile Spreads · Decay Curves · Factor Correlations  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generates institutional-style factor research reports:

  1. IC Summary Table         — Mean IC, ICIR, t-stat, IC+%, graded A/B/C/D
  2. Factor Decay Report      — IC at 1H/4H/1D/3D/1W/2W horizons
  3. Quantile Spread Return   — Q1 vs Q5 (top quintile − bottom quintile) returns
  4. Factor Auto-Correlation  — How much does factor rank change each rebal period
  5. Factor Correlation Matrix— Pairwise correlation of factor ranks (diversification)
  6. Annual IC Breakdown      — IC per calendar year (stability check)
  7. Regime Performance       — Factor IC by market regime

Run standalone:
  python azalyst_tearsheet.py --ic-csv ./azalyst_output/ic_analysis.csv
  python azalyst_tearsheet.py --data-dir ./data --max-symbols 50 --full
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  FACTOR GRADING
# ─────────────────────────────────────────────────────────────────────────────

def grade_factor(ic_mean: float, icir: float, t_stat: float) -> str:
    """
    Institutional grade for a factor at a given horizon.
    A+: ICIR > 1.0  AND |t| > 3.0  (publication-quality alpha)
    A : ICIR > 0.5  AND |t| > 2.5
    B : ICIR > 0.3  AND |t| > 2.0
    C : ICIR > 0.1  OR  |t| > 1.5
    D : weak or insignificant
    """
    at = abs(t_stat)
    if   icir > 1.0 and at > 3.0:  return "A+"
    elif icir > 0.5 and at > 2.5:  return "A "
    elif icir > 0.3 and at > 2.0:  return "B "
    elif icir > 0.1 or  at > 1.5:  return "C "
    else:                           return "D "


# ─────────────────────────────────────────────────────────────────────────────
#  TEAR SHEET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class FactorTearSheet:
    """
    Generates comprehensive, institutionally-styled factor research reports
    from IC analysis results and raw factor data.

    Usage
    ─────
        from azalyst_tearsheet import FactorTearSheet
        ts = FactorTearSheet(ic_table, factors, close_panel)
        ts.print_full_report()
        ts.save_all("./azalyst_output")
    """

    HORIZONS_ORDER = ["1H", "4H", "1D", "3D", "1W", "2W"]

    def __init__(self,
                 ic_table:    Optional[pd.DataFrame] = None,
                 factors:     Optional[Dict[str, pd.DataFrame]] = None,
                 close_panel: Optional[pd.DataFrame] = None,
                 out_dir:     str = "./azalyst_output"):
        self.ic     = ic_table
        self.factors = factors
        self.close  = close_panel
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. IC Summary Table ───────────────────────────────────────────────────

    def ic_summary(self, horizon: str = "1D") -> pd.DataFrame:
        """Grade all factors at a given horizon. Returns sortable DataFrame."""
        if self.ic is None or self.ic.empty:
            return pd.DataFrame()
        df = self.ic[self.ic["horizon"] == horizon].copy()
        df["grade"]  = df.apply(
            lambda r: grade_factor(r["IC_mean"], r["ICIR"], r["t_stat"]), axis=1)
        df["stars"]  = df["ICIR"].apply(
            lambda x: "" if x > 1.0 else " " if x > 0.5 else "  " if x > 0.2 else "   ")
        return df.sort_values("ICIR", ascending=False)

    def print_ic_summary(self, horizon: str = "1D") -> None:
        df = self.ic_summary(horizon)
        if df.empty:
            print(f"[TearSheet] No IC data for {horizon}"); return
        sep  = "─" * 90
        print(f"\n{sep}")
        print(f"  FACTOR IC SUMMARY  |  Horizon: {horizon}  "
              f"|  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(sep)
        print(f"  {'Factor':<20} {'IC_mean':>8} {'IC_std':>8} {'ICIR':>8} "
              f"{'t-stat':>8} {'IC+%':>7} {'Grade':>6} Stars")
        print(sep)
        for _, r in df.iterrows():
            print(f"  {r['factor']:<20} {r['IC_mean']:>8.4f} {r['IC_std']:>8.4f} "
                  f"{r['ICIR']:>8.4f} {r['t_stat']:>8.2f} {r['IC_pos%']:>7.1f}% "
                  f"  {r['grade']}   {r['stars']}")
        print(sep)

    # ── 2. Factor Decay Report ────────────────────────────────────────────────

    def decay_table(self) -> pd.DataFrame:
        """IC across all horizons for every factor. Shows signal persistence."""
        if self.ic is None or self.ic.empty:
            return pd.DataFrame()
        pivot = self.ic.pivot_table(
            index="factor", columns="horizon", values="IC_mean", aggfunc="first"
        )
        # Order horizons properly
        ordered = [h for h in self.HORIZONS_ORDER if h in pivot.columns]
        pivot = pivot[ordered]
        pivot["peak_horizon"] = pivot.abs().idxmax(axis=1)
        return pivot.sort_values(ordered[0] if ordered else pivot.columns[0], ascending=False)

    def print_decay_table(self) -> None:
        df = self.decay_table()
        if df.empty:
            print("[TearSheet] No decay data"); return
        ordered = [h for h in self.HORIZONS_ORDER if h in df.columns]
        print(f"\n{'─'*80}")
        print(f"  FACTOR DECAY (IC at each horizon)")
        print(f"{'─'*80}")
        header = f"  {'Factor':<20}"
        for h in ordered:
            header += f" {h:>8}"
        header += f"  {'Peak':>6}"
        print(header)
        print(f"{'─'*80}")
        for factor, row in df.iterrows():
            line = f"  {factor:<20}"
            for h in ordered:
                val = row.get(h, np.nan)
                line += f" {val:>8.4f}" if not np.isnan(val) else f" {'—':>8}"
            peak = row.get("peak_horizon", "")
            line += f"  {peak:>6}"
            print(line)
        print(f"{'─'*80}")

    # ── 3. Factor Auto-Correlation ────────────────────────────────────────────

    def factor_autocorrelation(self, rebal_bars: int = 288) -> pd.DataFrame:
        """
        Auto-correlation of factor cross-sectional ranks at rebalancing frequency.
        High autocorrelation = slow-moving signal (lower turnover cost).
        Low autocorrelation = fast-moving signal (higher turnover, needs large IC).
        """
        if self.factors is None:
            return pd.DataFrame()
        rows = []
        for name, f in self.factors.items():
            # Lag-1 correlation of rank at rebal periods
            rebal_f = f.iloc[::rebal_bars]
            lag     = rebal_f.stack().groupby(level=0).mean()
            ac      = lag.autocorr(lag=1) if len(lag) > 2 else np.nan
            rows.append({"factor": name, "rank_autocorr": round(ac, 4) if not np.isnan(ac) else np.nan})
        return pd.DataFrame(rows).sort_values("rank_autocorr", ascending=False)

    def print_autocorrelation(self, rebal_bars: int = 288) -> None:
        df = self.factor_autocorrelation(rebal_bars)
        if df.empty:
            print("[TearSheet] No factor data for autocorrelation"); return
        print(f"\n{'─'*55}")
        print(f"  FACTOR RANK AUTO-CORRELATION (rebal={rebal_bars} bars)")
        print(f"{'─'*55}")
        print(f"  {'Factor':<22} {'AutoCorr':>10} {'Signal Speed':>14}")
        print(f"{'─'*55}")
        for _, r in df.iterrows():
            ac   = r["rank_autocorr"]
            speed = "SLOW" if ac > 0.8 else "MEDIUM" if ac > 0.5 else "FAST"
            ac_str = f"{ac:.4f}" if not np.isnan(ac) else "  N/A"
            print(f"  {r['factor']:<22} {ac_str:>10}   {speed:>10}")
        print(f"{'─'*55}")

    # ── 4. Factor Correlation Matrix ──────────────────────────────────────────

    def factor_correlation_matrix(self,
                                  sample_bars: int = 2016) -> pd.DataFrame:
        """
        Pairwise Pearson correlation of factor cross-sectional ranks.
        Reveals redundancies. Factors with |corr| > 0.7 are near-duplicates.
        """
        if self.factors is None:
            return pd.DataFrame()
        # Use last sample_bars of mean rank per bar
        series_dict = {}
        for name, f in self.factors.items():
            s = f.iloc[-sample_bars:].mean(axis=1)
            series_dict[name] = s
        panel = pd.DataFrame(series_dict)
        return panel.corr(method="pearson").round(3)

    def print_high_corr_pairs(self, threshold: float = 0.7) -> None:
        """Print factor pairs with |correlation| > threshold."""
        cm = self.factor_correlation_matrix()
        if cm.empty:
            print("[TearSheet] No factor data for correlation"); return
        print(f"\n{'─'*60}")
        print(f"  HIGH-CORRELATION FACTOR PAIRS (|corr| > {threshold})")
        print(f"{'─'*60}")
        names = cm.columns.tolist()
        found = False
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                c = cm.iloc[i, j]
                if abs(c) >= threshold:
                    print(f"  {names[i]:<20} ↔ {names[j]:<20}  r={c:.3f}")
                    found = True
        if not found:
            print(f"  No pairs above |{threshold}| threshold")
        print(f"{'─'*60}")

    # ── 5. Annual IC Breakdown ────────────────────────────────────────────────

    def annual_ic(self, horizon: str = "1D") -> pd.DataFrame:
        """IC per calendar year for stability analysis."""
        if self.ic is None or self.ic.empty:
            return pd.DataFrame()

        # IC table doesn't have dates — need raw IC series (from engine)
        # This requires CrossSectionalAnalyser to be called with store_series=True
        # Placeholder: returns the global IC_mean repeated (stateless from CSV)
        df = self.ic[self.ic["horizon"] == horizon].copy()
        df["year"] = "full"  # Would be per-year if IC series available
        return df[["factor", "year", "IC_mean", "ICIR"]].sort_values("ICIR", ascending=False)

    # ── 6. Full console report ────────────────────────────────────────────────

    def print_full_report(self, primary_horizon: str = "1D") -> None:
        """Print the complete tear sheet to console."""
        banner = "═" * 72
        print(f"\n{banner}")
        print(f"  AZALYST AlphaX — Factor Tear Sheet")
        print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{banner}")
        self.print_ic_summary(primary_horizon)
        self.print_decay_table()
        self.print_autocorrelation()
        self.print_high_corr_pairs()
        print(f"\n{banner}")
        print(f"  [TearSheet] Report complete")
        print(f"{banner}\n")

    # ── 7. Save all outputs ───────────────────────────────────────────────────

    def save_all(self, label: str = "") -> None:
        """Save all tear sheet outputs to CSV files."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        tag = f"_{label}" if label else ""

        def _save(df: pd.DataFrame, name: str) -> None:
            if df.empty:
                return
            path = self.out_dir / f"{name}{tag}_{ts}.csv"
            df.to_csv(path)
            print(f"  [Saved] {name} → {path}")

        for h in ["1H", "4H", "1D", "1W"]:
            _save(self.ic_summary(h), f"ic_summary_{h}")
        _save(self.decay_table(), "factor_decay")
        _save(self.factor_autocorrelation(), "factor_autocorr")
        _save(self.factor_correlation_matrix(), "factor_corr_matrix")
        print(f"[TearSheet] All outputs saved to {self.out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  QUANTILE SPREAD ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class QuantileSpreadAnalyser:
    """
    Detailed quantile spread analysis.
    Splits the universe into N deciles at each rebalancing bar
    and computes the return of each decile bucket.
    The Q10-Q1 spread is the economic alpha of the factor.
    """

    def __init__(self, close_panel: pd.DataFrame, n_quantiles: int = 5):
        self.close = close_panel
        self.n_q   = n_quantiles
        self.log_ret = np.log(close_panel / close_panel.shift(1))

    def analyse_factor(self,
                       factor_name: str,
                       factor: pd.DataFrame,
                       horizon_bars: int = 288,
                       rebal_every: int = 288) -> pd.DataFrame:
        """
        Returns per-quantile statistics for one factor at one horizon.
        """
        fwd = self.log_ret.shift(-horizon_bars).rolling(
            horizon_bars, min_periods=horizon_bars // 2).sum()

        idx      = factor.index.intersection(fwd.index)
        rebal_ts = idx[::rebal_every]
        q_rets   = {q: [] for q in range(1, self.n_q + 1)}

        for t in rebal_ts:
            row_f = factor.loc[t]
            row_r = fwd.loc[t]
            mask  = row_f.notna() & row_r.notna()
            if mask.sum() < self.n_q * 3:
                continue
            try:
                qcuts = pd.qcut(row_f[mask], self.n_q,
                                labels=range(1, self.n_q + 1),
                                duplicates="drop")
                for q in range(1, self.n_q + 1):
                    syms = qcuts[qcuts == q].index
                    if len(syms) > 0:
                        q_rets[q].append(row_r[syms].mean())
            except Exception:
                continue

        rows = []
        for q, rets in q_rets.items():
            if rets:
                arr = np.array(rets)
                rows.append({
                    "factor":    factor_name,
                    "quantile":  f"Q{q}",
                    "mean_ret%": round(arr.mean() * 100, 4),
                    "std_ret%":  round(arr.std() * 100, 4),
                    "sharpe":    round(arr.mean() / arr.std() * np.sqrt(252)
                                       if arr.std() > 0 else 0, 3),
                    "hit_rate%": round((arr > 0).mean() * 100, 1),
                    "n_obs":     len(rets),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            top_ret  = df[df["quantile"] == f"Q{self.n_q}"]["mean_ret%"].iloc[0] if len(df) else 0
            bot_ret  = df[df["quantile"] == "Q1"]["mean_ret%"].iloc[0] if len(df) else 0
            spread   = top_ret - bot_ret
            df["LS_spread%"] = spread
        return df

    def print_quantile_table(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        factor = df["factor"].iloc[0]
        spread = df["LS_spread%"].iloc[0] if "LS_spread%" in df.columns else 0
        print(f"\n  Factor: {factor}  |  L/S Spread: {spread:.4f}%")
        print(f"  {'Q':<6} {'Mean Ret%':>10} {'Sharpe':>8} {'Hit%':>8} {'N':>6}")
        for _, r in df.iterrows():
            print(f"  {r['quantile']:<6} {r['mean_ret%']:>10.4f} "
                  f"{r['sharpe']:>8.3f} {r['hit_rate%']:>8.1f} {r['n_obs']:>6}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azalyst Factor Tear Sheet Generator")
    parser.add_argument("--ic-csv",      default=None, help="Path to ic_analysis.csv")
    parser.add_argument("--data-dir",    default=None, help="Data dir for full factor recompute")
    parser.add_argument("--out-dir",     default="./azalyst_output")
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--horizon",     default="1D")
    parser.add_argument("--full",        action="store_true",
                        help="Full recompute (requires --data-dir)")
    parser.add_argument("--save",        action="store_true")
    args = parser.parse_args()

    ic_table  = None
    factors   = None
    close_panel = None

    # Load pre-computed IC table
    if args.ic_csv and os.path.exists(args.ic_csv):
        ic_table = pd.read_csv(args.ic_csv)
        print(f"[TearSheet] Loaded IC table: {len(ic_table)} rows")

    # Full pipeline recompute
    if args.full and args.data_dir:
        from azalyst_engine import DataLoader, FactorEngine, CrossSectionalAnalyser
        loader      = DataLoader(args.data_dir, max_symbols=args.max_symbols, workers=4)
        data        = loader.load_all()
        close_panel = loader.build_close_panel(data)
        vol_panel   = loader.build_volume_panel(data)

        fe      = FactorEngine()
        factors = fe.compute_all(close_panel, vol_panel)

        analyser = CrossSectionalAnalyser(close_panel)
        ic_table = analyser.analyse_all(factors, horizons=["1H", "4H", "1D", "1W"])

    ts = FactorTearSheet(ic_table=ic_table, factors=factors,
                         close_panel=close_panel, out_dir=args.out_dir)
    ts.print_full_report(primary_horizon=args.horizon)

    if args.save:
        ts.save_all()


if __name__ == "__main__":
    main()
