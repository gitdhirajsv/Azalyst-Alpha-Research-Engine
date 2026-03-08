"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    MASTER ORCHESTRATOR
║        Data → Factors → IC → Regime → ML → Signals → Report               ║
║        v1.0  |  Single entry point for the full research pipeline          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline Stages
───────────────
  Stage 1  DATA LOADING
           DataLoader (azalyst_engine) loads all parquet files from --data-dir,
           builds close + volume panels.

  Stage 2  FACTOR COMPUTATION
           FactorEngineV2 (azalyst_factors_v2) computes all 35 cross-sectional
           factors on the full panel.

  Stage 3  IC RESEARCH
           CrossSectionalAnalyser (azalyst_engine) computes IC, ICIR, t-stats,
           factor decay, quantile spreads.  Saves factor_ic_results.csv.

  Stage 4  REGIME DETECTION
           RegimeDetector (azalyst_ml) classifies the current market state.
           Uses BTC data + market breadth.

  Stage 5  ML SCORING
           PumpDumpDetector + ReturnPredictor score every symbol on latest bar.
           Models loaded from --model-dir if they exist, else trained inline.

  Stage 6  STATARB SIGNALS
           Reads statarb_pairs.csv (produced by azalyst_statarb.py) and
           computes current z-scores for all active pairs.

  Stage 7  SIGNAL COMBINATION
           SignalCombiner fuses factor + ML + statarb scores using
           regime-adaptive weights into a ranked composite signal table.

  Stage 8  REPORTING
           Saves signals.csv, prints top-N table, runs tearsheet + report.

Usage
─────
  python azalyst_orchestrator.py --data-dir ./data
  python azalyst_orchestrator.py --data-dir ./data --out-dir ./azalyst_output --top 30
  python azalyst_orchestrator.py --data-dir ./data --max-symbols 50  # quick test
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Orchestrator")

# ── Constants ─────────────────────────────────────────────────────────────────
BARS_PER_HOUR = 12
BARS_PER_DAY  = 288
BARS_PER_WEEK = 2016
STATARB_Z_WINDOW = BARS_PER_DAY * 30  # rolling z-score window for pairs


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: compute live z-scores from saved statarb pairs
# ─────────────────────────────────────────────────────────────────────────────

def _compute_statarb_zscores(
    pairs_csv: str,
    data: Dict[str, pd.DataFrame],
    window: int = STATARB_Z_WINDOW,
) -> pd.Series:
    """
    Reads azalyst_statarb output CSV, recomputes current z-scores for each
    pair, and returns a Series indexed by the LONG symbol of each pair.

    Only uses data already loaded — no additional I/O.
    """
    if not os.path.exists(pairs_csv):
        logger.warning(f"[StatArb] Pairs file not found: {pairs_csv} — skipping statarb signals")
        return pd.Series(dtype=float)

    try:
        pairs = pd.read_csv(pairs_csv)
    except Exception as e:
        logger.warning(f"[StatArb] Could not read pairs CSV: {e}")
        return pd.Series(dtype=float)

    required_cols = {"symbol_a", "symbol_b", "hedge_ratio"}
    if not required_cols.issubset(pairs.columns):
        logger.warning(f"[StatArb] Pairs CSV missing columns: {required_cols - set(pairs.columns)}")
        return pd.Series(dtype=float)

    zscores: dict = {}
    for _, row in pairs.iterrows():
        sym_a = row["symbol_a"]
        sym_b = row["symbol_b"]
        beta  = float(row["hedge_ratio"])

        if sym_a not in data or sym_b not in data:
            continue

        close_a = data[sym_a]["close"]
        close_b = data[sym_b]["close"]
        aligned = pd.concat([close_a, close_b], axis=1, join="inner").dropna()
        if len(aligned) < window // 4:
            continue

        aligned.columns = ["a", "b"]
        spread = aligned["a"] - beta * aligned["b"]

        rolling_mean = spread.rolling(window, min_periods=window // 4).mean()
        rolling_std  = spread.rolling(window, min_periods=window // 4).std()

        latest_z = (spread.iloc[-1] - rolling_mean.iloc[-1]) / (rolling_std.iloc[-1] + 1e-12)

        # z < 0 → sym_a is cheap relative to sym_b → long sym_a
        # z > 0 → sym_a is expensive → short sym_a
        zscores[sym_a] = float(latest_z)
        # Symmetric: sym_b gets the opposite z
        zscores[sym_b] = float(-latest_z)

    return pd.Series(zscores)


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class AzalystOrchestrator:
    """
    Single entry point that chains every Azalyst module into one pipeline.

    Designed to be run after walkforward_simulator.py and azalyst_statarb.py
    have already been executed (their outputs are consumed here).
    Can also be run standalone — missing inputs are skipped gracefully.
    """

    def __init__(
        self,
        data_dir:    str,
        out_dir:     str  = "./azalyst_output",
        model_dir:   str  = "./models",
        resample:    str  = "1H",
        max_symbols: Optional[int] = None,
        workers:     int  = 4,
        ic_horizons: list = None,
        top_n:       int  = 30,
    ):
        self.data_dir    = data_dir
        self.out_dir     = out_dir
        self.model_dir   = model_dir
        self.resample    = resample
        self.max_symbols = max_symbols
        self.workers     = workers
        self.ic_horizons = ic_horizons or [BARS_PER_HOUR, BARS_PER_DAY, BARS_PER_DAY * 3]
        self.top_n       = top_n

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Runtime state populated by each stage
        self.data:        Dict[str, pd.DataFrame] = {}
        self.close_panel: pd.DataFrame = pd.DataFrame()
        self.vol_panel:   pd.DataFrame = pd.DataFrame()
        self.factors:     Dict[str, pd.DataFrame] = {}
        self.ic_results:  pd.DataFrame = pd.DataFrame()
        self.regime:      str = "BULL_TREND"
        self.factor_scores_latest: pd.Series = pd.Series(dtype=float)
        self.return_proba_latest:  pd.Series = pd.Series(dtype=float)
        self.pump_proba_latest:    pd.Series = pd.Series(dtype=float)
        self.statarb_z:            pd.Series = pd.Series(dtype=float)
        self.signals:     pd.DataFrame = pd.DataFrame()

    # ── Stage 1: Data ─────────────────────────────────────────────────────────

    def stage_data(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 1  DATA LOADING")
        logger.info("=" * 60)

        from azalyst_engine import DataLoader

        loader = DataLoader(
            data_dir    = self.data_dir,
            resample_to = self.resample,
            max_symbols = self.max_symbols,
            workers     = self.workers,
        )
        self.data        = loader.load_all()
        self.close_panel = loader.build_close_panel(self.data)
        self.vol_panel   = loader.build_volume_panel(self.data)

        logger.info(f"Universe: {len(self.data)} symbols  |  "
                    f"Panel shape: {self.close_panel.shape}")

    # ── Stage 2: Factors ──────────────────────────────────────────────────────

    def stage_factors(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 2  FACTOR COMPUTATION (35 factors)")
        logger.info("=" * 60)

        from azalyst_factors_v2 import FactorEngineV2

        fe = FactorEngineV2()

        # Build OHLC panels needed by v2 factors
        open_panel  = pd.DataFrame({s: d["open"]  for s, d in self.data.items()}).sort_index().ffill(limit=3)
        high_panel  = pd.DataFrame({s: d["high"]  for s, d in self.data.items()}).sort_index().ffill(limit=3)
        low_panel   = pd.DataFrame({s: d["low"]   for s, d in self.data.items()}).sort_index().ffill(limit=3)

        btc_key = next((k for k in self.data if "BTC" in k and "USDT" in k), None)

        # Compute each factor group
        logger.info("  Computing momentum factors ...")
        self.factors["MOM_1H"]       = fe.mom_1h(self.close_panel)
        self.factors["MOM_4H"]       = fe.mom_4h(self.close_panel)
        self.factors["MOM_1D"]       = fe.mom_1d(self.close_panel)
        self.factors["MOM_3D"]       = fe.mom_3d(self.close_panel)
        self.factors["MOM_1W"]       = fe.mom_1w(self.close_panel)
        self.factors["MOM_2W"]       = fe.mom_2w(self.close_panel)
        self.factors["MOM_30D"]      = fe.mom_30d(self.close_panel)
        self.factors["OVERNIGHT"]    = fe.overnight(open_panel, self.close_panel)
        self.factors["CLOSE_TO_OPEN"]= fe.close_to_open(open_panel, self.close_panel)

        logger.info("  Computing reversal factors ...")
        self.factors["REV_1H"]  = fe.rev_1h(self.close_panel)
        self.factors["REV_4H"]  = fe.rev_4h(self.close_panel)
        self.factors["REV_1D"]  = fe.rev_1d(self.close_panel)

        logger.info("  Computing volatility factors ...")
        self.factors["RVOL_1D"]    = fe.rvol_1d(self.close_panel)
        self.factors["RVOL_1W"]    = fe.rvol_1w(self.close_panel)
        self.factors["VOL_OF_VOL"] = fe.vol_of_vol(self.close_panel)
        self.factors["DOWNVOL_1W"] = fe.downvol_1w(self.close_panel)

        logger.info("  Computing liquidity factors ...")
        self.factors["AMIHUD"]       = fe.amihud(self.close_panel, self.vol_panel)
        self.factors["CORWIN_SCHULTZ"]= fe.corwin_schultz(high_panel, low_panel)
        self.factors["TURNOVER"]     = fe.turnover(self.vol_panel)
        self.factors["VOL_RATIO"]    = fe.vol_ratio(self.vol_panel)
        self.factors["VOL_MOM_1D"]   = fe.vol_mom_1d(self.vol_panel)

        logger.info("  Computing microstructure factors ...")
        self.factors["MAX_RET"]         = fe.max_ret(self.close_panel)
        self.factors["SKEW_1W"]         = fe.skew_1w(self.close_panel)
        self.factors["KURT_1W"]         = fe.kurt_1w(self.close_panel)
        self.factors["PRICE_ACCEL"]     = fe.price_accel(self.close_panel)
        self.factors["VOLUME_SURPRISE"] = fe.volume_surprise(self.vol_panel)
        self.factors["VWAP_DEV"]        = fe.vwap_dev(self.close_panel, self.vol_panel)

        if btc_key:
            btc_close = self.close_panel[[btc_key]] if btc_key in self.close_panel.columns else None
            if btc_close is not None:
                self.factors["BTC_BETA"]  = fe.btc_beta(self.close_panel, btc_close)
                self.factors["IDIO_MOM"]  = fe.idio_mom(self.close_panel, btc_close)

        logger.info("  Computing technical factors ...")
        self.factors["TREND_48"]   = fe.trend_48(self.close_panel)
        self.factors["BB_POS"]     = fe.bb_pos(self.close_panel)
        self.factors["RSI_RANK"]   = fe.rsi_rank(self.close_panel)
        self.factors["MA_SLOPE"]   = fe.ma_slope(self.close_panel)
        self.factors["WEEK52_HIGH"]= fe.week52_high(self.close_panel)
        self.factors["WEEK52_LOW"] = fe.week52_low(self.close_panel)

        # Build composite factor score (average of all factor ranks)
        # Used as the "factor_scores" input to SignalCombiner
        factor_stack = pd.concat(
            [f.iloc[-1] for f in self.factors.values() if not f.empty],
            axis=1
        )
        self.factor_scores_latest = factor_stack.mean(axis=1).dropna()
        logger.info(f"  {len(self.factors)} factors computed.  "
                    f"Composite scores for {len(self.factor_scores_latest)} symbols.")

    # ── Stage 3: IC Research ──────────────────────────────────────────────────

    def stage_ic_research(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 3  IC / ICIR RESEARCH")
        logger.info("=" * 60)

        from azalyst_engine import CrossSectionalAnalyser

        csa = CrossSectionalAnalyser(self.close_panel)
        rows = []

        for name, factor in self.factors.items():
            for horizon in self.ic_horizons:
                try:
                    result = csa.analyse_factor(
                        name    = name,
                        factor  = factor,
                        horizon = horizon,
                    )
                    result["horizon_bars"] = horizon
                    rows.append(result)
                except Exception as e:
                    logger.warning(f"  IC failed for {name} @ {horizon}: {e}")

        if rows:
            self.ic_results = pd.DataFrame(rows)
            ic_path = os.path.join(self.out_dir, "factor_ic_results.csv")
            self.ic_results.to_csv(ic_path, index=False)
            logger.info(f"  IC results saved → {ic_path}")

            # Print top 10 factors by ICIR at 1D horizon
            day_ic = self.ic_results[self.ic_results["horizon_bars"] == BARS_PER_DAY]
            if not day_ic.empty:
                top = day_ic.nlargest(10, "icir")[["factor", "ic_mean", "icir", "t_stat", "grade"]]
                logger.info(f"\n  Top 10 factors by ICIR (1D horizon):\n{top.to_string(index=False)}\n")
        else:
            logger.warning("  No IC results computed.")

    # ── Stage 4: Regime Detection ─────────────────────────────────────────────

    def stage_regime(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 4  REGIME DETECTION")
        logger.info("=" * 60)

        regime_path = os.path.join(self.model_dir, "regime_detector.pkl")

        try:
            from azalyst_ml import RegimeDetector

            btc_key = next((k for k in self.data if "BTC" in k and "USDT" in k), None)
            if btc_key is None:
                logger.warning("  BTC not found in universe — defaulting to BULL_TREND")
                self.regime = "BULL_TREND"
                return

            rd = RegimeDetector()

            if os.path.exists(regime_path):
                rd.load(regime_path)
                logger.info(f"  Loaded regime model from {regime_path}")
            else:
                logger.info("  Training regime model on BTC data ...")
                rd.train(self.data[btc_key], close_panel=self.close_panel)
                rd.save(regime_path)

            self.regime = rd.current_regime(
                self.data[btc_key],
                close_panel=self.close_panel
            )
            logger.info(f"  Current regime: {self.regime}")

        except Exception as e:
            logger.warning(f"  Regime detection failed: {e}  — using BULL_TREND")
            self.regime = "BULL_TREND"

    # ── Stage 5: ML Scoring ───────────────────────────────────────────────────

    def stage_ml_scoring(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 5  ML SCORING (pump/dump + return predictor)")
        logger.info("=" * 60)

        pump_path   = os.path.join(self.model_dir, "pump_dump_model.pkl")
        return_path = os.path.join(self.model_dir, "return_predictor.pkl")

        try:
            from azalyst_ml import PumpDumpDetector, ReturnPredictor

            # ── Pump/Dump ───────────────────────────────────────────────────
            pump_model = PumpDumpDetector()
            if os.path.exists(pump_path):
                pump_model.load(pump_path)
                logger.info(f"  Loaded pump model from {pump_path}")
            else:
                logger.info("  Training pump/dump model ...")
                pump_model.train(self.data)
                pump_model.save(pump_path)

            pump_scores = {}
            for sym, df in self.data.items():
                try:
                    proba = pump_model.predict(df)
                    if len(proba) > 0:
                        pump_scores[sym] = float(proba.iloc[-1])
                except Exception:
                    pass
            self.pump_proba_latest = pd.Series(pump_scores)
            logger.info(f"  Pump scores computed for {len(pump_scores)} symbols.")

            # ── Return Predictor ─────────────────────────────────────────────
            ret_model = ReturnPredictor()
            if os.path.exists(return_path):
                ret_model.load(return_path)
                logger.info(f"  Loaded return model from {return_path}")
            else:
                logger.info("  Training return predictor ...")
                ret_model.train(self.data)
                ret_model.save(return_path)

            return_scores = {}
            for sym, df in self.data.items():
                try:
                    proba = ret_model.predict_proba(df)
                    if len(proba) > 0:
                        return_scores[sym] = float(proba.iloc[-1])
                except Exception:
                    pass
            self.return_proba_latest = pd.Series(return_scores)
            logger.info(f"  Return scores computed for {len(return_scores)} symbols.")

        except Exception as e:
            logger.warning(f"  ML scoring failed: {e}")

    # ── Stage 6: StatArb Z-Scores ─────────────────────────────────────────────

    def stage_statarb(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 6  STATARB Z-SCORES")
        logger.info("=" * 60)

        pairs_csv = os.path.join(self.out_dir, "statarb_pairs.csv")
        self.statarb_z = _compute_statarb_zscores(pairs_csv, self.data)
        logger.info(f"  Z-scores computed for {len(self.statarb_z)} symbols from pairs.")

    # ── Stage 7: Signal Combination ───────────────────────────────────────────

    def stage_signals(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 7  SIGNAL COMBINATION")
        logger.info("=" * 60)

        from azalyst_signal_combiner import SignalCombiner

        combiner = SignalCombiner(regime=self.regime)
        self.signals = combiner.combine(
            factor_scores  = self.factor_scores_latest  if not self.factor_scores_latest.empty  else None,
            return_proba   = self.return_proba_latest   if not self.return_proba_latest.empty   else None,
            pump_proba     = self.pump_proba_latest      if not self.pump_proba_latest.empty     else None,
            statarb_zscore = self.statarb_z              if not self.statarb_z.empty             else None,
        )

        combiner.print_signals(self.signals, top_n=self.top_n)
        signal_path = combiner.save(self.signals, out_dir=self.out_dir)

        # Grade distribution
        if not self.signals.empty:
            dist = self.signals["grade"].value_counts()
            logger.info(f"  Signal grade distribution:\n{dist.to_string()}")

    # ── Stage 8: Reporting ────────────────────────────────────────────────────

    def stage_report(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 8  TEARSHEET + REPORTS")
        logger.info("=" * 60)

        # Factor tearsheet
        try:
            from azalyst_tearsheet import FactorTearSheet
            if not self.ic_results.empty:
                ts = FactorTearSheet(
                    ic_results  = self.ic_results,
                    close_panel = self.close_panel,
                    factors     = self.factors,
                    out_dir     = self.out_dir,
                )
                ts.print_full_report()
                ts.save_all(label="orchestrator")
                logger.info("  Tearsheet saved.")
        except Exception as e:
            logger.warning(f"  Tearsheet failed: {e}")

        # Research report
        try:
            from azalyst_report import ResearchReport
            rr = ResearchReport(out_dir=self.out_dir)
            rr.generate()
            logger.info("  Research report generated.")
        except Exception as e:
            logger.warning(f"  Research report failed: {e}")

    # ── Run full pipeline ─────────────────────────────────────────────────────

    def run(self, stages: Optional[list] = None) -> None:
        """
        Run the full pipeline or a subset of stages.

        stages : list of stage names to run, e.g. ["data","factors","signals"]
                 None runs all stages in order.
        """
        all_stages = [
            ("data",     self.stage_data),
            ("factors",  self.stage_factors),
            ("ic",       self.stage_ic_research),
            ("regime",   self.stage_regime),
            ("ml",       self.stage_ml_scoring),
            ("statarb",  self.stage_statarb),
            ("signals",  self.stage_signals),
            ("report",   self.stage_report),
        ]

        run_set = set(stages) if stages else {name for name, _ in all_stages}

        t_start = datetime.now(timezone.utc)
        print(f"\n{'═'*64}")
        print(f"  AZALYST ORCHESTRATOR  |  {t_start.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Data dir : {self.data_dir}")
        print(f"  Out dir  : {self.out_dir}")
        print(f"  Symbols  : {'ALL' if not self.max_symbols else self.max_symbols}")
        print(f"  Resample : {self.resample}")
        print(f"{'═'*64}\n")

        for name, fn in all_stages:
            if name not in run_set:
                logger.info(f"SKIPPING stage: {name}")
                continue
            try:
                fn()
                gc.collect()
            except Exception as e:
                logger.error(f"STAGE {name.upper()} FAILED: {e}", exc_info=True)
                raise

        elapsed = (datetime.now(timezone.utc) - t_start).total_seconds()
        print(f"\n{'═'*64}")
        print(f"  PIPELINE COMPLETE  |  Elapsed: {elapsed/60:.1f} min")
        print(f"  Outputs saved to:  {self.out_dir}")
        print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Azalyst Orchestrator — full research pipeline"
    )
    parser.add_argument("--data-dir",    default="./data",            help="Parquet data directory")
    parser.add_argument("--out-dir",     default="./azalyst_output",  help="Output directory")
    parser.add_argument("--model-dir",   default="./models",          help="ML model directory")
    parser.add_argument("--resample",    default="1H",                help="OHLCV resample (5min/15min/1H/4H)")
    parser.add_argument("--max-symbols", type=int, default=None,      help="Limit symbols for testing")
    parser.add_argument("--workers",     type=int, default=4,         help="Parallel workers")
    parser.add_argument("--top",         type=int, default=30,        help="Top N signals to display")
    parser.add_argument(
        "--stages", nargs="+",
        choices=["data","factors","ic","regime","ml","statarb","signals","report"],
        default=None,
        help="Run only specific stages (default: all)"
    )
    args = parser.parse_args()

    orch = AzalystOrchestrator(
        data_dir    = args.data_dir,
        out_dir     = args.out_dir,
        model_dir   = args.model_dir,
        resample    = args.resample,
        max_symbols = args.max_symbols,
        workers     = args.workers,
        top_n       = args.top,
    )
    orch.run(stages=args.stages)


if __name__ == "__main__":
    main()
