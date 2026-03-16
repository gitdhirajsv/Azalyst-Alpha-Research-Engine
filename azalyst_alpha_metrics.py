"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  ALPHA METRICS  (Target: 1000% annual / 10x capital)
║  NO LLM. Pure quant evaluation.                                             ║
║  Retrain trigger: rolling 4-week annualised return < 1000%                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  TARGETS
# ─────────────────────────────────────────────────────────────────────────────
ANNUAL_RETURN_TARGET = 10.0                                     # 1000% = 10x
WEEKLY_TARGET        = (1 + ANNUAL_RETURN_TARGET) ** (1 / 52) - 1   # ~4.65%
ROLLING_WINDOW_WEEKS = 4
MIN_WEEKS_TO_JUDGE   = 2
FEE_RATE             = 0.001      # 0.1% per leg
ROUND_TRIP_FEE       = FEE_RATE * 2


def annualised_return(weekly_returns: pd.Series) -> float:
    r = weekly_returns.dropna()
    if len(r) == 0:
        return 0.0
    compounded = float((1 + r).prod())
    return compounded ** (52.0 / len(r)) - 1


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 52) -> float:
    r = returns.dropna()
    if len(r) < 3 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def information_ratio(strategy_returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
    """IR = annualised excess return / tracking error. IR > 0.5 = alpha."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() * 52) / (excess.std() * np.sqrt(52)))


def max_drawdown(cum_returns: pd.Series) -> float:
    cr = cum_returns.dropna()
    if len(cr) == 0:
        return 0.0
    peak = cr.cummax()
    return float(((cr - peak) / (1 + peak)).min())


def profit_factor(trade_pnls: pd.Series) -> float:
    wins   = trade_pnls[trade_pnls > 0].sum()
    losses = trade_pnls[trade_pnls < 0].abs().sum()
    return float(wins / losses) if losses > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
#  WEEKLY ALPHA REPORT
# ─────────────────────────────────────────────────────────────────────────────

def calculate_weekly_alpha(
    trades_df: pd.DataFrame,
    btc_weekly_return: Optional[float] = None,
) -> Dict:
    """
    Alpha report for ONE week of trades.
    trades_df must have column: pnl_percent (net PnL per trade after fees, in %)
    """
    if trades_df is None or len(trades_df) == 0:
        return {
            "week_return_pct": 0.0, "annualised_pct": 0.0,
            "sharpe": 0.0, "excess_vs_btc_pct": None,
            "profit_factor": 0.0, "win_rate": 0.0,
            "n_trades": 0, "on_track": False,
        }

    pnls        = trades_df["pnl_percent"].dropna() / 100.0
    week_return = float(pnls.mean()) if len(pnls) > 0 else 0.0
    ann         = (1 + week_return) ** 52 - 1
    excess      = (week_return - btc_weekly_return) if btc_weekly_return is not None else None

    return {
        "week_return_pct":   round(week_return * 100, 4),
        "annualised_pct":    round(ann * 100, 2),
        "sharpe":            round(sharpe_ratio(pnls, 52), 4),
        "excess_vs_btc_pct": round(excess * 100, 4) if excess is not None else None,
        "profit_factor":     round(profit_factor(pnls), 4),
        "win_rate":          round((pnls > 0).mean() * 100, 2),
        "n_trades":          len(pnls),
        "on_track":          week_return >= WEEKLY_TARGET,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RETRAIN DECISION
# ─────────────────────────────────────────────────────────────────────────────

def should_retrain(weekly_returns_history: List[float]) -> Dict:
    """
    Core loop trigger. Returns retrain=True when not on track for 1000% annual.

    Logic:
      1. Need MIN_WEEKS_TO_JUDGE weeks before judging
      2. If last week was catastrophic (< -15%) → retrain immediately
      3. Rolling ROLLING_WINDOW_WEEKS annualised < 1000% → retrain
    """
    n = len(weekly_returns_history)

    if n < MIN_WEEKS_TO_JUDGE:
        return {"retrain": False,
                "reason": f"Not enough weeks yet ({n} < {MIN_WEEKS_TO_JUDGE})",
                "rolling_annual": None, "weeks_evaluated": n}

    series    = pd.Series(weekly_returns_history)
    last_week = series.iloc[-1]

    if last_week < -0.15:
        return {"retrain": True,
                "reason": f"Catastrophic week: {last_week*100:.1f}% (< -15%)",
                "rolling_annual": None, "weeks_evaluated": n}

    window      = series.iloc[-ROLLING_WINDOW_WEEKS:]
    rolling_ann = annualised_return(window)

    if rolling_ann < ANNUAL_RETURN_TARGET:
        return {"retrain": True,
                "reason": (f"Rolling {len(window)}-week annualised "
                           f"{rolling_ann*100:.1f}% < target "
                           f"{ANNUAL_RETURN_TARGET*100:.0f}%"),
                "rolling_annual": round(rolling_ann * 100, 2),
                "weeks_evaluated": n}

    return {"retrain": False,
            "reason": (f"On track: {rolling_ann*100:.1f}% >= "
                       f"{ANNUAL_RETURN_TARGET*100:.0f}%"),
            "rolling_annual": round(rolling_ann * 100, 2),
            "weeks_evaluated": n}


# ─────────────────────────────────────────────────────────────────────────────
#  FULL SESSION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def session_report(
    weekly_summary_df: pd.DataFrame,
    all_trades_df: pd.DataFrame,
    label: str = "Session",
) -> Dict:
    """Full-session performance report. Printed + returned as dict."""
    w_rets = weekly_summary_df["week_return_pct"].dropna() / 100.0
    pnls   = all_trades_df["pnl_percent"].dropna() / 100.0

    cum       = (1 + w_rets).cumprod()
    total_ret = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0
    ann       = annualised_return(w_rets)

    print(f"\n{'═'*65}")
    print(f"  {label.upper()} — PERFORMANCE REPORT")
    print(f"{'═'*65}")
    print(f"  Target          : {ANNUAL_RETURN_TARGET*100:.0f}% annual (1000%)")
    print(f"  Total Return    : {total_ret*100:.2f}%")
    print(f"  Annualised      : {ann*100:.2f}%")
    print(f"  Sharpe (weekly) : {sharpe_ratio(w_rets, 52):.4f}")
    print(f"  Max Drawdown    : {max_drawdown(cum)*100:.2f}%")
    print(f"  Win Rate        : {(pnls > 0).mean()*100:.2f}%")
    print(f"  Total Trades    : {len(pnls):,}")
    weeks_on_track = int((w_rets >= WEEKLY_TARGET).sum())
    print(f"  Weeks on Track  : {weeks_on_track} / {len(w_rets)}")
    print(f"  Alpha Achieved  : {'YES ✓' if ann >= ANNUAL_RETURN_TARGET else 'NO ✗'}")
    print(f"{'═'*65}\n")

    return {
        "label":              label,
        "total_return_pct":   round(total_ret * 100, 2),
        "annualised_pct":     round(ann * 100, 2),
        "sharpe":             round(sharpe_ratio(w_rets, 52), 4),
        "max_drawdown_pct":   round(max_drawdown(cum) * 100, 2),
        "profit_factor":      round(profit_factor(pnls), 4),
        "overall_win_rate":   round((pnls > 0).mean() * 100, 2),
        "total_trades":       len(pnls),
        "weeks_with_alpha":   weeks_on_track,
        "alpha_achieved":     ann >= ANNUAL_RETURN_TARGET,
    }
