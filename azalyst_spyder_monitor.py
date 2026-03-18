"""
Azalyst — Spyder Live Monitor
Run in Spyder (F5) while RUN_AZALYST.bat is running.
Supports both outputs:
1) azalyst_engine.py -> backtest_pnl.csv / ic_analysis.csv / performance_summary.csv
2) azalyst_local_gpu.py -> weekly_summary_year3.csv / all_trades_year3.csv / performance_year3.json
"""

import json
import os
import time
import warnings

import matplotlib
import pandas as pd

# Force external plot window in Spyder.
try:
    from IPython import get_ipython

    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "qt5")
except Exception:
    pass

for _backend in ("Qt5Agg", "Qt6Agg", "TkAgg", "WXAgg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        pass

import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
REFRESH_SECS = 10

# Engine output files.
E_PNL = os.path.join(RESULTS_DIR, "backtest_pnl.csv")
E_IC = os.path.join(RESULTS_DIR, "ic_analysis.csv")
E_METRICS = os.path.join(RESULTS_DIR, "performance_summary.csv")

# Local GPU runner output files.
L_WEEKLY = os.path.join(RESULTS_DIR, "weekly_summary_year3.csv")
L_TRADES = os.path.join(RESULTS_DIR, "all_trades_year3.csv")
L_PERF = os.path.join(RESULTS_DIR, "performance_year3.json")


def detect_mode() -> str:
    if os.path.exists(E_PNL) or os.path.exists(E_METRICS) or os.path.exists(E_IC):
        return "engine"
    if os.path.exists(L_WEEKLY) or os.path.exists(L_PERF) or os.path.exists(L_TRADES):
        return "local_gpu"
    return "waiting"


def _safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_engine_data():
    pnl = _safe_read_csv(E_PNL, index_col=0, parse_dates=True)
    ic = _safe_read_csv(E_IC)
    metrics = {}
    mdf = _safe_read_csv(E_METRICS)
    if not mdf.empty:
        metrics = mdf.iloc[0].to_dict()
    return pnl, ic, metrics


def load_local_gpu_data():
    weekly = _safe_read_csv(L_WEEKLY)
    trades = _safe_read_csv(L_TRADES)
    perf = _safe_read_json(L_PERF)
    return weekly, trades, perf


def _waiting(ax, msg: str):
    ax.clear()
    ax.text(
        0.5,
        0.5,
        msg,
        ha="center",
        va="center",
        color="gray",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def draw_engine(fig, axes):
    pnl, ic, m = load_engine_data()

    for row in axes:
        for ax in row:
            ax.clear()

    ax1 = axes[0][0]
    if not pnl.empty and "cum_ret" in pnl.columns:
        cum = pnl["cum_ret"] * 100
        ax1.plot(pnl.index, cum, "#1f77b4", linewidth=1.8, label="Strategy")
        ax1.fill_between(pnl.index, cum, alpha=0.1, color="#1f77b4")
        ax1.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        if "cum_benchmark" in pnl.columns:
            ax1.plot(
                pnl.index,
                pnl["cum_benchmark"] * 100,
                "#ff7f0e",
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                label="BTC B&H",
            )
        total = m.get("total_ret_%", float(cum.iloc[-1]) if len(cum) else 0.0)
        ann = m.get("ann_ret_%", 0.0)
        ax1.set_title(
            f"Cumulative Return | Total={total:+.1f}% Ann={ann:+.1f}%",
            fontweight="bold",
            fontsize=9,
        )
        ax1.set_ylabel("%")
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="x", rotation=20)
    else:
        _waiting(ax1, "Waiting for backtest_pnl.csv")

    ax2 = axes[0][1]
    if not pnl.empty and "drawdown" in pnl.columns:
        dd = pnl["drawdown"] * 100
        ax2.fill_between(pnl.index, dd, 0, color="#d62728", alpha=0.55)
        ax2.plot(pnl.index, dd, "#d62728", linewidth=0.8)
        ax2.axhline(0, color="black", linewidth=0.5)
        max_dd = m.get("max_dd_%", float(dd.min()) if len(dd) else 0.0)
        ax2.set_title(f"Drawdown | Max DD={max_dd:.1f}%", fontweight="bold", fontsize=9)
        ax2.set_ylabel("%")
        ax2.grid(True, alpha=0.25)
        ax2.tick_params(axis="x", rotation=20)
    else:
        _waiting(ax2, "Waiting for drawdown")

    ax3 = axes[1][0]
    if not ic.empty and "IC_mean" in ic.columns and "factor" in ic.columns:
        grp = (
            ic.groupby("factor")
            .agg(IC_mean=("IC_mean", "mean"), ICIR=("ICIR", "mean"))
            .reset_index()
        )
        grp = grp.reindex(grp["ICIR"].abs().nlargest(15).index)
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in grp["IC_mean"]]
        ax3.barh(grp["factor"], grp["IC_mean"], color=colors, alpha=0.78)
        ax3.axvline(0, color="black", linewidth=0.7)
        ax3.set_xlabel("Mean IC")
        ax3.set_title("Top Factors by |ICIR|", fontweight="bold", fontsize=9)
        ax3.grid(True, axis="x", alpha=0.25)
    else:
        _waiting(ax3, "Waiting for ic_analysis.csv")

    ax4 = axes[1][1]
    ax4.axis("off")
    if m:
        lines = [
            f"Label         : {m.get('label', 'Azalyst')}",
            "",
            f"Total Return  : {m.get('total_ret_%', 0):>+8.2f}%",
            f"Ann Return    : {m.get('ann_ret_%', 0):>+8.2f}%",
            f"Ann Vol       : {m.get('ann_vol_%', 0):>8.2f}%",
            "",
            f"Sharpe        : {m.get('sharpe', 0):>8.3f}",
            f"Sortino       : {m.get('sortino', 0):>8.3f}",
            f"Calmar        : {m.get('calmar', 0):>8.3f}",
            f"Max Drawdown  : {m.get('max_dd_%', 0):>8.2f}%",
            f"Win Rate      : {m.get('win_rate_%', 0):>8.1f}%",
            f"N Periods     : {int(m.get('n_periods', 0)):>8d}",
        ]
    else:
        lines = [
            "Engine mode detected.",
            "",
            f"[{'DONE' if os.path.exists(E_PNL) else '    '}] backtest_pnl.csv",
            f"[{'DONE' if os.path.exists(E_IC) else '    '}] ic_analysis.csv",
            f"[{'DONE' if os.path.exists(E_METRICS) else '    '}] performance_summary.csv",
            "",
            f"Updated: {time.strftime('%H:%M:%S')}",
        ]
    ax4.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax4.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", alpha=0.85),
    )

    fig.suptitle(
        f"Azalyst Monitor | Engine Mode | {time.strftime('%H:%M:%S')}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def draw_local_gpu(fig, axes):
    w, t, p = load_local_gpu_data()

    for row in axes:
        for ax in row:
            ax.clear()

    nw = len(w)

    ax1 = axes[0][0]
    if nw > 0 and "week_return_pct" in w.columns and "week" in w.columns:
        rets = w["week_return_pct"].fillna(0) / 100.0
        cum = ((1 + rets).cumprod() - 1) * 100
        ax1.plot(w["week"], cum, "#1f77b4", linewidth=2)
        ax1.fill_between(w["week"], cum, alpha=0.12, color="#1f77b4")
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        total = p.get("total_return_pct", float(cum.iloc[-1]) if len(cum) else 0.0)
        ann = p.get("annualised_pct", 0.0)
        ax1.set_title(
            f"Cumulative Return | Total={total:.1f}% Ann={ann:.1f}%",
            fontweight="bold",
            fontsize=9,
        )
        ax1.set_xlabel("Week")
        ax1.set_ylabel("%")
        ax1.grid(True, alpha=0.25)
    else:
        _waiting(ax1, "Waiting for weekly_summary_year3.csv")

    ax2 = axes[0][1]
    if nw > 0 and "ic" in w.columns and "week" in w.columns:
        ic = w["ic"].fillna(0)
        ax2.bar(
            w["week"],
            ic,
            color=["#2ca02c" if v > 0 else "#d62728" for v in ic],
            alpha=0.75,
            width=0.8,
        )
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_title(f"Weekly IC | ICIR={p.get('icir', 0):.4f}", fontweight="bold", fontsize=9)
        ax2.set_xlabel("Week")
        ax2.set_ylabel("IC")
        ax2.grid(True, alpha=0.25)
    else:
        _waiting(ax2, "Waiting for IC series")

    ax3 = axes[1][0]
    if nw > 2 and "week_return_pct" in w.columns:
        wr = w["week_return_pct"].dropna()
        bins = min(25, max(8, len(wr) // 3))
        ax3.hist(wr, bins=bins, color="#ff7f0e", alpha=0.72, edgecolor="black", linewidth=0.4)
        ax3.axvline(wr.mean(), color="red", linewidth=1.8, linestyle="--", label=f"Mean {wr.mean():.2f}%")
        ax3.axvline(wr.median(), color="green", linewidth=1.2, linestyle=":", label=f"Median {wr.median():.2f}%")
        ax3.set_title(f"Weekly Returns | Sharpe={p.get('sharpe', 0):.3f}", fontweight="bold", fontsize=9)
        ax3.set_xlabel("Return (%)")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.25)
    else:
        _waiting(ax3, "Need >2 weeks for histogram")

    ax4 = axes[1][1]
    ax4.axis("off")
    if p:
        lines = [
            f"GPU           : {p.get('gpu', 'RTX 2050')}",
            f"Weeks run     : {p.get('total_weeks', nw)}",
            f"Total trades  : {p.get('total_trades', len(t)):,}",
            f"Retrains done : {p.get('retrains', 0)}",
            "",
            f"Total Return  : {p.get('total_return_pct', 0):+.2f}%",
            f"Annualised    : {p.get('annualised_pct', 0):+.2f}%",
            f"Sharpe        : {p.get('sharpe', 0):.4f}",
            f"IC Mean       : {p.get('ic_mean', 0):.5f}",
            f"ICIR          : {p.get('icir', 0):.4f}",
            f"IC Positive % : {p.get('ic_positive_pct', 0):.1f}%",
        ]
    else:
        lines = [
            "Local GPU mode detected.",
            "",
            f"[{'DONE' if os.path.exists(L_WEEKLY) else '    '}] weekly_summary_year3.csv",
            f"[{'DONE' if os.path.exists(L_TRADES) else '    '}] all_trades_year3.csv",
            f"[{'DONE' if os.path.exists(L_PERF) else '    '}] performance_year3.json",
            "",
            f"Updated: {time.strftime('%H:%M:%S')}",
        ]
    ax4.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax4.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", alpha=0.85),
    )

    fig.suptitle(
        f"Azalyst Monitor | Local GPU Mode | {time.strftime('%H:%M:%S')}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def draw_waiting(fig, axes):
    for row in axes:
        for ax in row:
            _waiting(ax, "Waiting for first output files...")
    fig.suptitle(
        f"Azalyst Monitor | Waiting | {time.strftime('%H:%M:%S')}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])


print("=" * 58)
print("  Azalyst Spyder Live Monitor")
print(f"  Results dir : {RESULTS_DIR}")
print(f"  Refresh     : every {REFRESH_SECS}s")
print("  Press Stop in Spyder toolbar to exit")
print("=" * 58)
