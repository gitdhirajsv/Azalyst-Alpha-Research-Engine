"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    MASTER ORCHESTRATOR
║        Data → Factors → IC → Regime → ML → Signals → Report               ║
║        v1.0  |  Single entry point for the full research pipeline          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline Stages
───────────────
  Stage 1  DATA LOADING
  Stage 2  FACTOR COMPUTATION
  Stage 3  IC RESEARCH
  Stage 4  REGIME DETECTION
  Stage 5  ML SCORING
  Stage 6  STATARB Z-SCORES
  Stage 7  SIGNAL COMBINATION
  Stage 8  REPORTING
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

BARS_PER_HOUR = 12
BARS_PER_DAY  = 288
BARS_PER_WEEK = 2016
STATARB_Z_WINDOW = BARS_PER_DAY * 30


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: compute live z-scores from saved statarb pairs
# ─────────────────────────────────────────────────────────────────────────────

def _compute_statarb_zscores(
    pairs_csv: str,
    data: Dict[str, pd.DataFrame],
    window: int = STATARB_Z_WINDOW,
) -> pd.Series:
    if not os.path.exists(pairs_csv):
        logger.warning(f"[StatArb] Pairs file not found: {pairs_csv} — skipping")
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

        zscores[sym_a] = float(latest_z)
        zscores[sym_b] = float(-latest_z)

    return pd.Series(zscores)


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class AzalystOrchestrator:

    def __init__(
        self,
        data_dir:    str,
        out_dir:     str  = "./azalyst_output",
        model_dir:   str  = "./models",
        resample:    str  = "1h",
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
        # Use string horizon names that CrossSectionalAnalyser actually accepts
        self.ic_horizons = ic_horizons or ["1H", "4H", "1D"]
        self.top_n       = top_n

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

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

        logger.info(f"Universe: {len(self.data)} symbols  |  Panel shape: {self.close_panel.shape}")

    # ── Stage 2: Factors ──────────────────────────────────────────────────────

    def stage_factors(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 2  FACTOR COMPUTATION")
        logger.info("=" * 60)

        # FIX: FactorEngineV2 does not exist in azalyst_factors_v2.py —
        # that file only has a standalone build_features() function.
        # FactorEngine (with compute_all and individual factor methods) lives
        # in azalyst_engine.py.
        from azalyst_engine import FactorEngine, CompositeFactorBuilder

        fe = FactorEngine()

        logger.info("  Computing 20 factors via FactorEngine.compute_all()...")
        self.factors = fe.compute_all(self.close_panel, self.vol_panel)

        # Build composite factor score from the latest bar of all factors
        cfb = CompositeFactorBuilder()
        composite = cfb.equal_weight(self.factors)
        if not composite.empty:
            self.factor_scores_latest = composite.iloc[-1].dropna()

        logger.info(
            f"  {len(self.factors)} factors computed. "
            f"Composite scores for {len(self.factor_scores_latest)} symbols."
        )

    # ── Stage 3: IC Research ──────────────────────────────────────────────────

    def stage_ic_research(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 3  IC / ICIR RESEARCH")
        logger.info("=" * 60)

        # FIX: the old code called csa.analyse_factor(name, factor, horizon=<int>)
        # but the method signature is analyse_factor(name, factor, horizons=List[str]).
        # The correct approach is to call analyse_all() which handles everything.
        from azalyst_engine import CrossSectionalAnalyser

        csa = CrossSectionalAnalyser(self.close_panel)

        logger.info(f"  Running IC analysis at horizons: {self.ic_horizons}")
        self.ic_results = csa.analyse_all(
            self.factors,
            horizons=self.ic_horizons,
        )

        if not self.ic_results.empty:
            ic_path = os.path.join(self.out_dir, "ic_analysis.csv")
            self.ic_results.to_csv(ic_path, index=False)
            logger.info(f"  IC results saved → {ic_path}")

            day_ic = self.ic_results[self.ic_results["horizon"] == "1D"]
            if not day_ic.empty:
                top = day_ic.nlargest(10, "ICIR")[["factor", "IC_mean", "ICIR", "t_stat"]]
                logger.info(f"\n  Top 10 factors (1D horizon):\n{top.to_string(index=False)}\n")
        else:
            logger.warning("  No IC results computed.")

    # ── Stage 2.5: Statistical Validation ────────────────────────────────────

    def stage_validation(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 2.5  STATISTICAL VALIDATION")
        logger.info("=" * 60)

        try:
            from azalyst_validator import FactorValidator
            if not self.factors or self.ic_results.empty:
                logger.warning("  Validation skipped — no factors or IC results yet.")
                return
            validator = FactorValidator(
                factors      = self.factors,
                close_panel  = self.close_panel,
                volume_panel = self.vol_panel,
                ic_table     = self.ic_results,
                out_dir      = self.out_dir,
                fm_horizon   = BARS_PER_DAY,
                alpha        = 0.05,
            )
            validator.run(run_fm=True, run_neutral=True)
        except Exception as e:
            logger.warning(f"  Validation stage failed (non-fatal): {e}")

    # ── Stage 4: Regime Detection ─────────────────────────────────────────────

    def stage_regime(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 4  REGIME DETECTION")
        logger.info("=" * 60)

        regime_path = os.path.join(self.model_dir, "regime_detector.pkl")

        try:
            from azalyst_ml import ReturnPredictorV2

            btc_key = next((k for k in self.data if "BTC" in k and "USDT" in k), None)
            if btc_key is None:
                logger.warning("  BTC not found — defaulting to BULL_TREND")
                self.regime = "BULL_TREND"
                return

            # Simple regime via BTC momentum
            btc_close = self.data[btc_key]["close"]
            ret_4w = btc_close.pct_change(BARS_PER_WEEK * 4).iloc[-1]
            rvol   = np.log(btc_close / btc_close.shift(1)).rolling(BARS_PER_DAY).std().iloc[-1]
            avg_rvol = np.log(btc_close / btc_close.shift(1)).rolling(BARS_PER_DAY * 30).std().iloc[-1]

            if ret_4w > 0.05 and rvol < avg_rvol * 1.5:
                self.regime = "BULL_TREND"
            elif ret_4w < -0.05:
                self.regime = "BEAR_TREND"
            elif rvol > avg_rvol * 1.5:
                self.regime = "HIGH_VOL_LATERAL"
            else:
                self.regime = "LOW_VOL_GRIND"

            logger.info(f"  Current regime: {self.regime}")

        except Exception as e:
            logger.warning(f"  Regime detection failed: {e}  — using BULL_TREND")
            self.regime = "BULL_TREND"

    # ── Stage 5: ML Scoring ───────────────────────────────────────────────────

    def stage_ml_scoring(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 5  ML SCORING")
        logger.info("=" * 60)

        return_path = os.path.join(self.model_dir, "return_predictor.pkl")

        try:
            from azalyst_ml import ReturnPredictorV2
            from azalyst_factors_v2 import build_features, FEATURE_COLS

            ret_model = ReturnPredictorV2()
            if os.path.exists(return_path):
                ret_model.load(return_path)
                logger.info(f"  Loaded return model from {return_path}")

            return_scores = {}
            for sym, df in self.data.items():
                try:
                    feats = build_features(df)
                    latest = feats[FEATURE_COLS].dropna(how="all").iloc[-1:]
                    if latest.empty or latest.isnull().all(axis=1).any():
                        continue
                    proba = ret_model.predict_proba(latest.values.astype(np.float32))
                    return_scores[sym] = float(proba[0])
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
        combiner.save(self.signals, out_dir=self.out_dir)

        if not self.signals.empty:
            dist = self.signals["grade"].value_counts()
            logger.info(f"  Signal grade distribution:\n{dist.to_string()}")

    # ── Stage 8: Reporting ────────────────────────────────────────────────────

    def stage_report(self) -> None:
        logger.info("=" * 60)
        logger.info("STAGE 8  TEARSHEET + REPORTS")
        logger.info("=" * 60)

        try:
            from azalyst_tearsheet import FactorTearSheet
            if not self.ic_results.empty:
                ts = FactorTearSheet(
                    ic_table    = self.ic_results,
                    close_panel = self.close_panel,
                    factors     = self.factors,
                    out_dir     = self.out_dir,
                )
                ts.print_full_report()
                ts.save_all(label="orchestrator")
                logger.info("  Tearsheet saved.")
        except Exception as e:
            logger.warning(f"  Tearsheet failed: {e}")

        try:
            from azalyst_report import ResearchReport
            rr = ResearchReport(out_dir=self.out_dir)
            rr.generate()
            logger.info("  Research report generated.")
        except Exception as e:
            logger.warning(f"  Research report failed: {e}")

    # ── Run full pipeline ─────────────────────────────────────────────────────

    def run(self, stages: Optional[list] = None) -> None:
        all_stages = [
            ("data",       self.stage_data),
            ("factors",    self.stage_factors),
            ("ic",         self.stage_ic_research),
            ("validation", self.stage_validation),
            ("regime",     self.stage_regime),
            ("ml",         self.stage_ml_scoring),
            ("statarb",    self.stage_statarb),
            ("signals",    self.stage_signals),
            ("report",     self.stage_report),
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
    parser.add_argument("--data-dir",    default="./data")
    parser.add_argument("--out-dir",     default="./azalyst_output")
    parser.add_argument("--model-dir",   default="./models")
    parser.add_argument("--resample",    default="1h")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--top",         type=int, default=30)
    parser.add_argument(
        "--stages", nargs="+",
        choices=["data","factors","ic","validation","regime","ml","statarb","signals","report"],
        default=None,
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
