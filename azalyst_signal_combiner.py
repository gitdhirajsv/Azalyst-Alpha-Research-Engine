"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    UNIFIED SIGNAL COMBINER
║        Factor Quantiles · Regime Weights · ML Probabilities → Final Score  ║
║        v1.0  |  Regime-Adaptive Signal Fusion  |  Institutional Grade      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Signal Architecture
───────────────────
  Four signal sources are fused into one composite score (0 → 1 per symbol):

  [1] FACTOR SCORE (0→1)
      Cross-sectional percentile rank of the composite factor (FactorEngineV2).
      Momentum + reversal + quality sub-composites averaged.

  [2] ML RETURN SCORE (0→1)
      ReturnPredictor.predict_proba() — probability the coin goes UP next 4H.

  [3] ML PUMP RISK (0→1, inverted)
      PumpDumpDetector probability.  High pump score → PENALISE the signal.
      Contribution = 1 - pump_prob, so clean coins score higher.

  [4] STATARB SIGNAL (0→1)
      Z-score of spread converted to a directional score:
        z < -entry_z  →  score 1.0 (long the underperformer)
        z > +entry_z  →  score 0.0 (short the overperformer)
        |z| < exit_z  →  score 0.5 (neutral)

  Weights are selected from a REGIME_WEIGHT_TABLE keyed on the 4-state
  RegimeDetector output (BULL_TREND / BEAR_TREND / HIGH_VOL_LATERAL /
  LOW_VOL_GRIND).  This means the same factor can be up/downweighted
  depending on market environment.

Final Signal Grades
───────────────────
  STRONG BUY   composite ≥ 0.75
  BUY          composite ≥ 0.60
  HOLD         0.40 < composite < 0.60
  SELL         composite ≤ 0.40
  STRONG SELL  composite ≤ 0.25
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SignalCombiner")

# ─────────────────────────────────────────────────────────────────────────────
#  REGIME-ADAPTIVE WEIGHT TABLE
#  Weights must sum to 1.0 per regime.
# ─────────────────────────────────────────────────────────────────────────────

REGIME_WEIGHT_TABLE: Dict[str, Dict[str, float]] = {
    # Trending bull: ride momentum, factor composite dominant
    "BULL_TREND": {
        "factor":   0.45,
        "ml_return": 0.35,
        "pump_inv":  0.10,
        "statarb":  0.10,
    },
    # Bear market: mean-reversion and statarb more valuable
    "BEAR_TREND": {
        "factor":   0.25,
        "ml_return": 0.20,
        "pump_inv":  0.20,
        "statarb":  0.35,
    },
    # High vol choppy: avoid momentum, lean on statarb and pump filter
    "HIGH_VOL_LATERAL": {
        "factor":   0.15,
        "ml_return": 0.15,
        "pump_inv":  0.35,
        "statarb":  0.35,
    },
    # Low vol grind: balanced, all signals informative
    "LOW_VOL_GRIND": {
        "factor":   0.30,
        "ml_return": 0.30,
        "pump_inv":  0.15,
        "statarb":  0.25,
    },
}

SIGNAL_GRADES = [
    (0.75, "STRONG BUY"),
    (0.60, "BUY"),
    (0.40, "HOLD"),
    (0.25, "SELL"),
    (0.00, "STRONG SELL"),
]


def _grade(score: float) -> str:
    for threshold, label in SIGNAL_GRADES:
        if score >= threshold:
            return label
    return "STRONG SELL"


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL COMBINER
# ─────────────────────────────────────────────────────────────────────────────

class SignalCombiner:
    """
    Merges all alpha sources into a single ranked signal table.

    Parameters
    ----------
    regime : str
        Current market regime from RegimeDetector.current_regime().
        Defaults to BULL_TREND if not provided.
    statarb_entry_z : float
        Z-score threshold to consider a statarb signal active.
    """

    def __init__(
        self,
        regime: str = "BULL_TREND",
        statarb_entry_z: float = 2.0,
        statarb_exit_z: float = 0.5,
    ):
        self.regime = regime if regime in REGIME_WEIGHT_TABLE else "BULL_TREND"
        self.entry_z = statarb_entry_z
        self.exit_z  = statarb_exit_z
        self.weights = REGIME_WEIGHT_TABLE[self.regime]
        self.ic_history: Dict[str, List[float]] = {}   # per-source rolling IC
        logger.info(f"[SignalCombiner] Regime={self.regime}  Weights={self.weights}")

    # ── IC tracking (Grinold & Kahn dynamic reweighting) ─────────────────────

    def update_ic(self, source: str, ic_value: float) -> None:
        """Record a weekly IC observation for a signal source."""
        if source not in self.ic_history:
            self.ic_history[source] = []
        self.ic_history[source].append(float(ic_value))

    def _ic_adjusted_weights(self, lookback: int = 13) -> Dict[str, float]:
        """
        Adjust base regime weights by rolling IC quality.

        Sources with higher recent IC get proportionally more weight.
        Requires >= 4 weeks of IC history per source to activate.
        Falls back to base regime weights if insufficient history.
        """
        multipliers = {}
        for source in self.weights:
            ics = self.ic_history.get(source, [])
            if len(ics) >= 4:
                recent = ics[-lookback:]
                mean_ic = float(np.mean(recent))
                # Positive IC → boost weight, negative → shrink
                # Clamp multiplier to [0.1, 3.0] for stability
                multipliers[source] = max(0.1, min(3.0, 1.0 + 10.0 * mean_ic))
            else:
                multipliers[source] = 1.0

        adjusted = {k: self.weights[k] * multipliers[k] for k in self.weights}
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    # ── Signal source processors ──────────────────────────────────────────────

    def _process_factor_scores(
        self, factor_scores: pd.Series
    ) -> pd.Series:
        """
        Cross-sectional percentile rank of raw composite factor scores.
        Input: pd.Series indexed by symbol, raw composite values.
        Output: pd.Series 0→1 percentile rank.
        """
        if factor_scores.empty:
            return factor_scores
        ranked = factor_scores.rank(pct=True)
        return ranked.clip(0.0, 1.0)

    def _process_ml_return(
        self, return_proba: pd.Series
    ) -> pd.Series:
        """
        ML up-probability already in [0,1].  Just clip for safety.
        """
        return return_proba.clip(0.0, 1.0)

    def _process_pump_inv(
        self, pump_proba: pd.Series
    ) -> pd.Series:
        """
        Inverts pump probability so high-pump coins score LOW.
        1 - pump_prob → clean coins score high.
        """
        return (1.0 - pump_proba.clip(0.0, 1.0))

    def _process_statarb(
        self, zscore: pd.Series
    ) -> pd.Series:
        """
        Converts a z-score series into a directional 0→1 signal.
        z < -entry_z   → 1.0  (coin is cheap vs pair → long signal)
        z >  entry_z   → 0.0  (coin is expensive → short / avoid)
        |z| < exit_z   → 0.5  (no clear edge)
        Linear interpolation in between.
        """
        score = pd.Series(0.5, index=zscore.index)
        # Long signal zone: z strongly negative
        score[zscore <= -self.entry_z] = 1.0
        # Short signal zone: z strongly positive
        score[zscore >= self.entry_z]  = 0.0
        # Linear fade between exit_z and entry_z
        long_fade  = (zscore > -self.entry_z) & (zscore < -self.exit_z)
        short_fade = (zscore <  self.entry_z) & (zscore >  self.exit_z)
        if long_fade.any():
            score[long_fade] = 0.5 + 0.5 * (
                (-zscore[long_fade] - self.exit_z) / (self.entry_z - self.exit_z)
            )
        if short_fade.any():
            score[short_fade] = 0.5 - 0.5 * (
                (zscore[short_fade] - self.exit_z) / (self.entry_z - self.exit_z)
            )
        return score.clip(0.0, 1.0)

    # ── Main combine method ───────────────────────────────────────────────────

    def combine(
        self,
        factor_scores:  Optional[pd.Series] = None,
        return_proba:   Optional[pd.Series] = None,
        pump_proba:     Optional[pd.Series] = None,
        statarb_zscore: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Fuses all available signal sources into a composite score table.

        All inputs are pd.Series indexed by symbol name.
        Missing sources receive a neutral score of 0.5 with zero weight
        redistributed proportionally to present sources.

        Returns
        -------
        pd.DataFrame with columns:
            symbol, factor_score, ml_return, pump_inv, statarb,
            composite, grade, regime
        Sorted descending by composite score.
        """
        # Collect all symbols from available inputs
        all_symbols: set = set()
        for s in [factor_scores, return_proba, pump_proba, statarb_zscore]:
            if s is not None and not s.empty:
                all_symbols.update(s.index)

        if not all_symbols:
            logger.warning("[SignalCombiner] No signal inputs provided.")
            return pd.DataFrame()

        symbols = sorted(all_symbols)

        # Build component score series (default neutral 0.5 if missing)
        def _fill(s: Optional[pd.Series], neutral: float = 0.5) -> pd.Series:
            if s is None or s.empty:
                return pd.Series(neutral, index=symbols)
            return s.reindex(symbols).fillna(neutral)

        f_score = self._process_factor_scores(_fill(factor_scores))
        r_score = self._process_ml_return(_fill(return_proba))
        p_score = self._process_pump_inv(_fill(pump_proba))
        s_score = (
            self._process_statarb(statarb_zscore.reindex(symbols).fillna(0.0))
            if statarb_zscore is not None and not statarb_zscore.empty
            else pd.Series(0.5, index=symbols)
        )

        # Adjust weights: zero-out sources that were not provided, renorm
        present = {
            "factor":    factor_scores  is not None and not factor_scores.empty,
            "ml_return": return_proba   is not None and not return_proba.empty,
            "pump_inv":  pump_proba     is not None and not pump_proba.empty,
            "statarb":   statarb_zscore is not None and not statarb_zscore.empty,
        }
        w = {k: v if present[k] else 0.0
             for k, v in (self._ic_adjusted_weights()
                          if self.ic_history else self.weights).items()}
        total_w = sum(w.values())
        if total_w == 0:
            total_w = 1.0
        w = {k: v / total_w for k, v in w.items()}

        composite = (
            f_score * w["factor"]   +
            r_score * w["ml_return"] +
            p_score * w["pump_inv"] +
            s_score * w["statarb"]
        )

        result = pd.DataFrame({
            "symbol":       symbols,
            "factor_score": f_score.values.round(4),
            "ml_return":    r_score.values.round(4),
            "pump_inv":     p_score.values.round(4),
            "statarb":      s_score.values.round(4),
            "composite":    composite.values.round(4),
        })
        result["grade"]  = result["composite"].apply(_grade)
        result["regime"] = self.regime
        result = result.sort_values("composite", ascending=False).reset_index(drop=True)

        logger.info(
            f"[SignalCombiner] {len(result)} symbols scored  "
            f"| Top: {result.iloc[0]['symbol']} ({result.iloc[0]['composite']:.3f})"
            if len(result) > 0 else "[SignalCombiner] Empty result"
        )
        return result

    # ── Convenience: print ranked table ──────────────────────────────────────

    def print_signals(self, signals: pd.DataFrame, top_n: int = 20) -> None:
        if signals.empty:
            print("No signals.")
            return
        print(f"\n{'═'*72}")
        print(f"  AZALYST SIGNAL TABLE  |  Regime: {self.regime}  |  Top {top_n}")
        print(f"{'═'*72}")
        print(f"  {'#':<4} {'Symbol':<14} {'Factor':>7} {'ML_Ret':>7} {'PumpInv':>8} "
              f"{'StatArb':>8} {'Score':>7}  Grade")
        print(f"  {'─'*68}")
        for i, row in signals.head(top_n).iterrows():
            print(f"  {i+1:<4} {row['symbol']:<14} "
                  f"{row['factor_score']:>7.3f} "
                  f"{row['ml_return']:>7.3f} "
                  f"{row['pump_inv']:>8.3f} "
                  f"{row['statarb']:>8.3f} "
                  f"{row['composite']:>7.3f}  {row['grade']}")
        print(f"{'═'*72}\n")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, signals: pd.DataFrame, out_dir: str = ".", label: str = "") -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        suffix = f"_{label}" if label else ""
        path = os.path.join(out_dir, f"signals{suffix}.csv")
        signals.to_csv(path, index=False)
        logger.info(f"[SignalCombiner] Saved → {path}")
        return path
