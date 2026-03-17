"""
╔══════════════════════════════════════════════════════════════════════════════╗
       AZALYST ALPHA RESEARCH ENGINE    RESEARCH REPORT & LIVE SCANNER        
║        Factor Signals · ML Scores · Regime · Pair Signals                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BARS_PER_HOUR = 12
BARS_PER_DAY  = 288


# ─────────────────────────────────────────────────────────────────────────────
#  RESEARCH REPORT
# ─────────────────────────────────────────────────────────────────────────────

class ResearchReport:
    """Generates a full research report from existing output files."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def generate(self) -> None:
        print("\n" + "═"*65)
        print("  AZALYST  ·  ALPHAX  ·  RESEARCH REPORT")
        print("  " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        print("═"*65)

        ic_path = os.path.join(self.out_dir, "ic_analysis.csv")
        if os.path.exists(ic_path):
            ic_df = pd.read_csv(ic_path)
            print("\n┌─ TOP FACTORS BY ICIR (1D horizon) ─────────────────────────┐")
            top = ic_df[ic_df["horizon"] == "1D"].sort_values("ICIR", ascending=False).head(10)
            for _, row in top.iterrows():
                stars = "" if row["ICIR"] > 1.0 else " " if row["ICIR"] > 0.5 else "  "
                print(f"  {stars} {row['factor']:<18}  IC={row['IC_mean']:>7.4f}  "
                      f"ICIR={row['ICIR']:>6.3f}  t={row['t_stat']:>5.1f}  "
                      f"IC+%={row['IC_pos%']:.0f}%")
            print("└────────────────────────────────────────────────────────────┘")

        bt_path = os.path.join(self.out_dir, "performance_summary.csv")
        if os.path.exists(bt_path):
            bt_df = pd.read_csv(bt_path)
            print("\n┌─ BACKTEST PERFORMANCE ──────────────────────────────────────┐")
            row = bt_df.iloc[0]
            metrics = [
                ("Strategy",       str(row.get("label", "N/A"))),
                ("Total Return",   f"{row.get('total_ret_%', 0):.2f}%"),
                ("Annualised Ret", f"{row.get('ann_ret_%', 0):.2f}%"),
                ("Sharpe",         f"{row.get('sharpe', 0):.3f}"),
                ("Sortino",        f"{row.get('sortino', 0):.3f}"),
                ("Max Drawdown",   f"{row.get('max_dd_%', 0):.2f}%"),
                ("Win Rate",       f"{row.get('win_rate_%', 0):.1f}%"),
                ("Avg Turnover",   f"{row.get('avg_turnover_%', 0):.2f}%"),
            ]
            for k, v in metrics:
                print(f"  {k:<20} {v}")
            print("└────────────────────────────────────────────────────────────┘")

        pairs_path = os.path.join(self.out_dir, "cointegrated_pairs.csv")
        if os.path.exists(pairs_path):
            pairs = pd.read_csv(pairs_path)
            print(f"\n┌─ TOP 10 COINTEGRATED PAIRS ──────────────────────────────────┐")
            for _, row in pairs.head(10).iterrows():
                print(f"  {row['coin_A']}/{row['coin_B']:<20}  "
                      f"p={row['p_value']:.4f}  "
                      f"HL={row['half_life_d']:.1f}d  "
                      f"H={row['hurst']:.3f}")
            print("└────────────────────────────────────────────────────────────┘")

        pairs_bt_path = os.path.join(self.out_dir, "pairs_backtest_summary.csv")
        if os.path.exists(pairs_bt_path):
            pairs_bt = pd.read_csv(pairs_bt_path)
            print(f"\n┌─ PAIRS BACKTEST (TOP 5) ─────────────────────────────────────┐")
            for _, row in pairs_bt.head(5).iterrows():
                print(f"  {str(row.get('pair','')):<25}  "
                      f"Sharpe={row.get('sharpe',0):.3f}  "
                      f"Return={row.get('total_ret_%',0):.1f}%  "
                      f"Trades={int(row.get('n_trades',0))}")
            print("└────────────────────────────────────────────────────────────┘")

        print("\n" + "═"*65)
        print("  [Azalyst] Report complete")
        print("═"*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE ALPHA SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class LiveAlphaScanner:
    """
    Runs all alpha engines on the most recent data and produces
    a unified signal table ranked by composite alpha score.
    """

    REGIME_WEIGHTS = {
        "BULL_TREND":        {"factor": 0.5, "momentum": 0.5, "reversal": 0.0, "statarb": 0.0},
        "BEAR_TREND":        {"factor": 0.3, "momentum": 0.0, "reversal": 0.4, "statarb": 0.3},
        "HIGH_VOL_LATERAL":  {"factor": 0.2, "momentum": 0.0, "reversal": 0.5, "statarb": 0.3},
        "LOW_VOL_GRIND":     {"factor": 0.2, "momentum": 0.2, "reversal": 0.2, "statarb": 0.4},
    }

    def __init__(self, data: Dict[str, pd.DataFrame],
                 models_dir: Optional[str] = None,
                 pairs_csv: Optional[str] = None,
                 composite: str = "reversal",
                 ic_csv: Optional[str] = None):
        self.data         = data
        self.models_dir   = models_dir
        self.composite    = composite
        self.pairs_df     = pd.read_csv(pairs_csv) if pairs_csv and os.path.exists(pairs_csv) else None
        self._ic_table    = pd.read_csv(ic_csv) if ic_csv and os.path.exists(ic_csv) else None

        self.return_model = None
        if models_dir and os.path.isdir(models_dir):
            self._load_models(models_dir)

    def _load_models(self, models_dir: str) -> None:
        return_path = os.path.join(models_dir, "return_predictor.pkl")
        if os.path.exists(return_path):
            from azalyst_ml import ReturnPredictorV2
            self.return_model = ReturnPredictorV2()
            self.return_model.load(return_path)

    def _get_regime(self) -> str:
        btc_key = next((k for k in self.data if "BTC" in k), None)
        if btc_key:
            btc = self.data[btc_key]["close"]
            ret_4w = btc.pct_change(BARS_PER_DAY * 28).iloc[-1]
            if ret_4w > 0.05:
                return "BULL_TREND"
            elif ret_4w < -0.05:
                return "BEAR_TREND"
        return "BULL_TREND"

    def _factor_scores_latest(self) -> pd.Series:
        """
        Compute composite score on latest available bar.

        FIX: Gemini introduced FactorEngineV2 (which doesn't exist) with
        methods like ic_weighted_composite, reversal_composite, quality_composite
        (which also don't exist). The correct classes are FactorEngine from
        azalyst_engine.py and CompositeFactorBuilder also from azalyst_engine.py.
        """
        # FIX: FactorEngine lives in azalyst_engine, not azalyst_factors_v2
        from azalyst_engine import FactorEngine, CompositeFactorBuilder

        close_panel = pd.DataFrame(
            {sym: df["close"] for sym, df in self.data.items()}
        ).sort_index().ffill(limit=3)

        volume_panel = pd.DataFrame(
            {sym: df["volume"] for sym, df in self.data.items()}
        ).sort_index().ffill(limit=3)

        fe = FactorEngine()
        factors = fe.compute_all(close_panel, volume_panel)

        # FIX: CompositeFactorBuilder has the actual composite methods
        cfb = CompositeFactorBuilder()

        if self.composite == "ic_weighted" and self._ic_table is not None:
            comp = cfb.ic_weighted(factors, self._ic_table, horizon="1D")
        elif self.composite == "quality":
            comp = cfb.quality_composite(factors)
        elif self.composite == "momentum":
            comp = cfb.momentum_composite(factors)
        else:
            # Default: reversal — confirmed dominant alpha in this universe
            comp = cfb.reversal_composite(factors)

        return comp.iloc[-1].dropna()

    def scan(self) -> pd.DataFrame:
        """Run full live scan. Returns sorted signal table."""
        print("\n[LiveAlpha] Running live signal scan...")

        regime = self._get_regime()
        print(f"  Market Regime: {regime}")

        try:
            factor_scores = self._factor_scores_latest()
        except Exception as e:
            print(f"  [WARN] Factor engine failed: {e}")
            factor_scores = pd.Series(dtype=float)

        signals = []
        for sym, df in self.data.items():
            if len(df) < BARS_PER_DAY:
                continue

            row = {"symbol": sym, "regime": regime}
            row["factor_score"] = round(factor_scores.get(sym, 0.5), 4)

            if self.return_model:
                try:
                    from azalyst_factors_v2 import build_features, FEATURE_COLS
                    feats = build_features(df)
                    latest = feats[FEATURE_COLS].dropna(how="all").iloc[-1:]
                    if not latest.empty:
                        proba = self.return_model.predict_proba(
                            latest.values.astype(np.float32)
                        )
                        row["up_prob"] = round(float(proba[0]), 4)
                    else:
                        row["up_prob"] = 0.5
                except Exception:
                    row["up_prob"] = 0.5
            else:
                row["up_prob"] = 0.5

            row["pump_prob"] = 0.0

            close = df["close"]
            row["ret_1h"]  = round(close.pct_change(BARS_PER_HOUR).iloc[-1] * 100, 3)
            row["ret_4h"]  = round(close.pct_change(BARS_PER_HOUR * 4).iloc[-1] * 100, 3)
            row["ret_1d"]  = round(close.pct_change(BARS_PER_DAY).iloc[-1] * 100, 3)
            row["price"]   = round(close.iloc[-1], 6)

            composite = row["factor_score"] * 0.5 + row["up_prob"] * 0.5
            row["alpha_score"] = round(composite, 4)
            row["signal"] = (
                "STRONG BUY" if composite > 0.75 else
                "BUY"        if composite > 0.60 else
                "SELL"       if composite < 0.30 else
                "SHORT"      if composite < 0.20 else
                "NEUTRAL"
            )
            signals.append(row)

        result = pd.DataFrame(signals).sort_values("alpha_score", ascending=False)
        return result

    def print_top(self, signals: pd.DataFrame, top_n: int = 20) -> None:
        buys  = signals[signals["signal"].isin(["STRONG BUY", "BUY"])].head(top_n)
        sells = signals[signals["signal"].isin(["SELL", "SHORT"])].tail(top_n)

        print(f"\n{'═'*75}")
        print(f"  AZALYST AlphaX — Live Signals  |  Regime: {signals['regime'].iloc[0]}")
        print(f"{'═'*75}")
        print(f"\n  TOP LONG CANDIDATES:")
        print(f"  {'Symbol':<15} {'Score':>7} {'1H%':>7} {'4H%':>7} {'1D%':>7} "
              f"{'UpProb':>7} {'PumpP':>7} {'Signal'}")
        print(f"  {'-'*70}")
        for _, r in buys.iterrows():
            print(f"  {r['symbol']:<15} {r['alpha_score']:>7.4f} "
                  f"{r['ret_1h']:>7.2f} {r['ret_4h']:>7.2f} {r['ret_1d']:>7.2f} "
                  f"{r['up_prob']:>7.3f} {r['pump_prob']:>7.3f}  {r['signal']}")

        if not sells.empty:
            print(f"\n  TOP SHORT CANDIDATES:")
            print(f"  {'Symbol':<15} {'Score':>7} {'1H%':>7} {'4H%':>7} {'1D%':>7} "
                  f"{'UpProb':>7} {'PumpP':>7} {'Signal'}")
            print(f"  {'-'*70}")
            for _, r in sells.sort_values("alpha_score").iterrows():
                print(f"  {r['symbol']:<15} {r['alpha_score']:>7.4f} "
                      f"{r['ret_1h']:>7.2f} {r['ret_4h']:>7.2f} {r['ret_1d']:>7.2f} "
                      f"{r['up_prob']:>7.3f} {r['pump_prob']:>7.3f}  {r['signal']}")
        print(f"{'═'*75}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azalyst Report — Research Reports & Live Signal Scanner"
    )
    parser.add_argument("--data-dir",    default=None)
    parser.add_argument("--out-dir",     default="./azalyst_output")
    parser.add_argument("--models-dir",  default="./azalyst_models")
    parser.add_argument("--pairs-csv",   default=None)
    parser.add_argument("--ic-csv",      default=None)
    parser.add_argument("--composite",   default="reversal",
                        choices=["reversal", "quality", "ic_weighted", "momentum"])
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--live",        action="store_true")
    parser.add_argument("--save",        action="store_true")
    args = parser.parse_args()

    report = ResearchReport(args.out_dir)
    report.generate()

    if args.report_only or not args.data_dir:
        return

    if args.live:
        from azalyst_engine import DataLoader
        loader  = DataLoader(args.data_dir, max_symbols=args.max_symbols, workers=4)
        data    = loader.load_all()
        scanner = LiveAlphaScanner(
            data,
            models_dir = args.models_dir if os.path.isdir(args.models_dir) else None,
            pairs_csv  = args.pairs_csv,
            composite  = args.composite,
            ic_csv     = args.ic_csv or os.path.join(args.out_dir, "ic_analysis.csv"),
        )
        signals = scanner.scan()
        scanner.print_top(signals)

        if args.save:
            os.makedirs(args.out_dir, exist_ok=True)
            ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = os.path.join(args.out_dir, f"live_signals_{ts}.csv")
            signals.to_csv(path, index=False)
            print(f"[Saved] Signals → {path}")


if __name__ == "__main__":
    main()
