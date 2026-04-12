"""
Azalyst v6 Results Overview for Spyder.

Run this file with F5 in Spyder to:
  1. Print a compact summary of the latest files in results_v6/
  2. Open a 2x2 matplotlib overview of returns, trades, and features
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_v6"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_float(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return numeric


def _pick_feature_file(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("feature_importance_v6*.csv"))
    if not files:
        return None
    week_files = [p for p in files if "week" in p.stem.lower()]
    return week_files[-1] if week_files else files[-1]


def resolve_overall_return_pct(performance: dict, weekly: pd.DataFrame) -> float | None:
    for key in ("overall_return_pct", "total_return_pct", "overall_return", "total_return"):
        value = _to_float(performance.get(key))
        if value is not None:
            return value

    if weekly.empty:
        return None

    if "cum_return_pct" in weekly.columns:
        cumulative = pd.to_numeric(weekly["cum_return_pct"], errors="coerce").dropna()
        if not cumulative.empty:
            return float(cumulative.iloc[-1])

    if "week_return_pct" in weekly.columns:
        weekly_returns = pd.to_numeric(weekly["week_return_pct"], errors="coerce").dropna()
        if not weekly_returns.empty:
            compounded = np.prod(1 + weekly_returns.to_numpy(dtype=float) / 100.0) - 1
            return float(compounded * 100.0)

    return None


def load_results(results_dir: Path = RESULTS_DIR) -> dict:
    feature_file = _pick_feature_file(results_dir)
    feature_df = _read_csv(feature_file) if feature_file else pd.DataFrame()

    weekly = _read_csv(results_dir / "weekly_summary_v6.csv")
    trades = _read_csv(results_dir / "all_trades_v6.csv")
    performance = _read_json(results_dir / "performance_v6.json")
    train_summary = _read_json(results_dir / "train_summary_v6.json")

    if not weekly.empty and "week_start" in weekly.columns:
        weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
    if not trades.empty and "week_start" in trades.columns:
        trades["week_start"] = pd.to_datetime(trades["week_start"], errors="coerce")

    return {
        "weekly": weekly,
        "trades": trades,
        "performance": performance,
        "train_summary": train_summary,
        "feature_df": feature_df,
        "feature_file": feature_file,
    }


def build_flags(performance: dict, weekly: pd.DataFrame, trades: pd.DataFrame) -> list[str]:
    flags: list[str] = []

    if performance.get("kill_switch_hit"):
        flags.append("Kill-switch was hit during the run")
    if performance.get("max_drawdown_pct", 0) < -20:
        flags.append(f"Max drawdown is very deep: {performance.get('max_drawdown_pct')}%")
    if performance.get("sharpe", 0) > 10:
        flags.append(f"Sharpe looks unusually high: {performance.get('sharpe')}")

    if not weekly.empty and "week_return_pct" in weekly.columns:
        max_abs_week = float(weekly["week_return_pct"].abs().max())
        if max_abs_week > 100:
            flags.append(f"Weekly return spike detected: {max_abs_week:.2f}%")

    if not trades.empty:
        if "position_scale" in trades.columns:
            max_scale = float(trades["position_scale"].max())
            if max_scale > 5:
                flags.append(f"Large position scale detected: {max_scale:.2f}x")
        if "pnl_percent" in trades.columns:
            min_trade = float(trades["pnl_percent"].min())
            if min_trade <= -100:
                flags.append(f"Trade loss clipped to {min_trade:.2f}%")

    return flags


def print_overview(data: dict, results_dir: Path = RESULTS_DIR) -> None:
    weekly = data["weekly"]
    trades = data["trades"]
    performance = data["performance"]
    train_summary = data["train_summary"]
    feature_df = data["feature_df"]
    feature_file = data["feature_file"]
    overall_return_pct = resolve_overall_return_pct(performance, weekly)

    print("\n" + "=" * 78)
    print("AZALYST v6 RESULTS OVERVIEW")
    print("=" * 78)
    print(f"Results folder : {results_dir}")

    if performance:
        print(f"Run ID         : {performance.get('run_id', 'N/A')}")
        print(f"Model          : {performance.get('model_type', 'N/A')}")
        print(f"Sharpe         : {performance.get('sharpe', 'N/A')}")
        if overall_return_pct is None:
            print("Overall Return % : N/A")
        else:
            print(f"Overall Return % : {overall_return_pct:.2f}")
        print(f"Max DD %       : {performance.get('max_drawdown_pct', 'N/A')}")
        print(f"IC Mean        : {performance.get('ic_mean', 'N/A')}")
        print(f"Kill Switch    : {performance.get('kill_switch_hit', 'N/A')}")
        print(f"Top-N          : {performance.get('top_n', 'N/A')}")
        print(f"Avg Turnover % : {performance.get('avg_weekly_turnover_pct', 'N/A')}")

    if train_summary:
        print("-" * 78)
        print(f"Train Rows     : {train_summary.get('n_rows', 'N/A')}")
        print(f"Train Features : {train_summary.get('n_features', 'N/A')}")
        print(f"Train IC       : {train_summary.get('mean_ic', 'N/A')}")
        print(f"Train R2       : {train_summary.get('mean_r2', 'N/A')}")

    if not weekly.empty:
        print("-" * 78)
        print(f"Weeks          : {len(weekly)}")
        print(f"Date Range     : {weekly['week_start'].min()} -> {weekly['week_start'].max()}")
        print(f"Best Week %    : {weekly['week_return_pct'].max():.2f}")
        print(f"Worst Week %   : {weekly['week_return_pct'].min():.2f}")
        if "regime" in weekly.columns:
            print("Regimes        :")
            for regime, count in weekly["regime"].value_counts().items():
                print(f"  {regime:<18} {count}")

    if not trades.empty:
        print("-" * 78)
        print(f"Trades         : {len(trades)}")
        if "signal" in trades.columns:
            signal_counts = trades["signal"].value_counts()
            for signal, count in signal_counts.items():
                print(f"  {signal:<6} {count}")
        if "pnl_percent" in trades.columns:
            win_rate = float((trades["pnl_percent"] > 0).mean() * 100)
            print(f"Win Rate %     : {win_rate:.2f}")
            print(f"Avg Trade %    : {trades['pnl_percent'].mean():.2f}")
        print("Top Symbols    :")
        for symbol, count in trades["symbol"].value_counts().head(10).items():
            print(f"  {symbol:<12} {count}")

    if not feature_df.empty:
        print("-" * 78)
        print(f"Feature File   : {feature_file.name if feature_file else 'N/A'}")
        feature_col = feature_df.columns[0]
        if feature_col == "Unnamed: 0":
            feature_df = feature_df.rename(columns={feature_col: "feature"})
            feature_col = "feature"
        top_features = feature_df.sort_values("importance", ascending=False).head(10)
        print("Top Features   :")
        for _, row in top_features.iterrows():
            print(f"  {str(row[feature_col]):<24} {float(row['importance']):.6f}")

    flags = build_flags(performance, weekly, trades)
    if flags:
        print("-" * 78)
        print("FLAGS")
        for flag in flags:
            print(f"  - {flag}")

    print("=" * 78 + "\n")


def plot_overview(data: dict) -> None:
    weekly = data["weekly"].copy()
    trades = data["trades"].copy()
    performance = data["performance"]
    feature_df = data["feature_df"].copy()
    overall_return_pct = resolve_overall_return_pct(performance, weekly)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.canvas.manager.set_window_title("Azalyst v6 Results Overview")

    # Panel 1: cumulative return + drawdown
    ax = axes[0, 0]
    if not weekly.empty:
        x = weekly["week"] if "week" in weekly.columns else np.arange(1, len(weekly) + 1)
        ax.plot(x, weekly["cum_return_pct"], color="#1f77b4", linewidth=2.2, label="Cum Return %")
        ax.set_title("Cumulative Return")
        ax.set_xlabel("Week")
        ax.set_ylabel("Cum Return %")
        ax.grid(True, alpha=0.3)
        if overall_return_pct is not None:
            sign = "+" if overall_return_pct >= 0 else ""
            ax.text(
                0.02,
                0.98,
                f"Overall: {sign}{overall_return_pct:.2f}%",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
            )
        ax2 = ax.twinx()
        ax2.plot(x, weekly["max_drawdown_pct"], color="#d62728", linewidth=1.8, label="Drawdown %")
        ax2.set_ylabel("Drawdown %")
    else:
        ax.text(0.5, 0.5, "weekly_summary_v6.csv not found", ha="center", va="center")
        ax.set_axis_off()

    # Panel 2: weekly returns
    ax = axes[0, 1]
    if not weekly.empty:
        x = weekly["week"] if "week" in weekly.columns else np.arange(1, len(weekly) + 1)
        colors = np.where(weekly["week_return_pct"] >= 0, "#2ca02c", "#d62728")
        ax.bar(x, weekly["week_return_pct"], color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Weekly Return %")
        ax.set_xlabel("Week")
        ax.set_ylabel("Return %")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No weekly data", ha="center", va="center")
        ax.set_axis_off()

    # Panel 3: top feature importance
    ax = axes[1, 0]
    if not feature_df.empty and "importance" in feature_df.columns:
        first_col = feature_df.columns[0]
        if first_col == "Unnamed: 0":
            feature_df = feature_df.rename(columns={first_col: "feature"})
            first_col = "feature"
        top_features = feature_df.sort_values("importance", ascending=True).tail(10)
        ax.barh(top_features[first_col].astype(str), top_features["importance"], color="#9467bd")
        ax.set_title("Top Feature Importance")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No feature importance file found", ha="center", va="center")
        ax.set_axis_off()

    # Panel 4: trade counts + flags
    ax = axes[1, 1]
    if not trades.empty:
        top_symbols = trades["symbol"].value_counts().head(10).sort_values(ascending=True)
        ax.barh(top_symbols.index.astype(str), top_symbols.values, color="#ff7f0e")
        ax.set_title("Top Traded Symbols")
        ax.set_xlabel("Trade Count")
        ax.grid(True, axis="x", alpha=0.3)
    else:
        flags = build_flags(performance, weekly, trades)
        text = "\n".join(flags) if flags else "No trades or flags available"
        ax.text(0.03, 0.95, text, va="top", ha="left", family="monospace")
        ax.set_title("Flags")
        ax.set_axis_off()

    title_parts = ["Azalyst v6 Results Overview"]
    if performance:
        title_parts.append(f"run_id={performance.get('run_id', 'N/A')}")
        if overall_return_pct is not None:
            title_parts.append(f"return={overall_return_pct:.2f}%")
        title_parts.append(f"sharpe={performance.get('sharpe', 'N/A')}")
        title_parts.append(f"dd={performance.get('max_drawdown_pct', 'N/A')}%")
    fig.suptitle("  |  ".join(title_parts), fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def main() -> None:
    data = load_results(RESULTS_DIR)
    print_overview(data, RESULTS_DIR)
    plot_overview(data)


if __name__ == "__main__":
    main()
