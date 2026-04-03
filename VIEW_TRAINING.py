"""
Azalyst Alpha Research Engine  ΓÇö  Spyder Monitor (Live)

Auto-launched by RUN_AZALYST.bat when you choose Terminal + Spyder mode.
Can also be run manually:  python VIEW_TRAINING.py  or F5 in Spyder.

Reads results_top6/checkpoint_v4_latest.json and results_top6/run_log.txt every 5 s
and renders a 4-panel live dashboard:
  ΓÇó Training Quality by Week  (win rate + rolling Sharpe)
  ΓÇó PnL and Drawdown          (cumulative return + max drawdown curve)
  ΓÇó Current Status            (week, trades, win rate, Sharpe, DD, PF)
  ΓÇó Recent Log Tail           (last N lines from the engine log file)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")          # works as a standalone window on Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ΓöÇΓöÇ Paths ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results_top6"
CKPT = RES  / "checkpoint_v4_latest.json"
LOG  = RES  / "run_log.txt"

REFRESH = 5   # seconds between refreshes
LOG_TAIL = 18  # log lines to show

# ΓöÇΓöÇ Theme (dark navy bg + white card panels ΓÇö matches SVG design) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
BG    = "#0d1422"   # dark navy figure background
PANEL = "#f5f7fb"   # white card panels
ACC1  = "#ff7f0e"   # orange (matplotlib default)
ACC2  = "#1f77b4"   # blue  (matplotlib default)
ACC3  = "#27a060"   # green
ACC4  = "#d94040"   # red
TXT   = "#111827"   # dark text on panels
MUTED = "#6b7a99"
GRID  = "#e2e8f0"
TITLE_FG = "#e7f5ee"  # light text for suptitle on dark bg


# ΓöÇΓöÇ Data helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def _load_ckpt() -> dict:
    try:
        return json.loads(CKPT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_log(n: int = LOG_TAIL) -> list[str]:
    try:
        with LOG.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [ln.rstrip() for ln in lines[-n:]]
    except Exception:
        return [f"Waiting for {LOG.name} ΓÇª  (pipeline not started yet)"]


def _win_rates_by_week(all_trades: list) -> list[float]:
    """Return per-week win-rate % from the trades list in the checkpoint."""
    if not all_trades:
        return []
    df = pd.DataFrame(all_trades)
    if "week" not in df.columns or "pnl_percent" not in df.columns:
        return []
    result = []
    for w in sorted(df["week"].unique()):
        wk = df[df["week"] == w]
        result.append(float((wk["pnl_percent"] > 0).mean() * 100))
    return result


def _rolling_sharpe(weekly_returns: list, window: int = 4) -> list[float]:
    if len(weekly_returns) < 2:
        return [0.0] * len(weekly_returns)
    out = []
    for i in range(len(weekly_returns)):
        sub = weekly_returns[max(0, i - window + 1): i + 1]
        mu  = float(np.mean(sub))
        std = float(np.std(sub))
        out.append(mu / std * np.sqrt(52) if std > 0 else 0.0)
    return out


# ΓöÇΓöÇ Style ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    "#cbd5e1",
        "axes.labelcolor":   TXT,
        "axes.titlecolor":   TXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        GRID,
        "grid.linewidth":    0.5,
        "text.color":        TXT,
        "font.family":       ["Segoe UI", "Arial", "sans-serif"],
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
        "legend.facecolor":  PANEL,
    })


# ΓöÇΓöÇ Render one frame ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def _render(fig: plt.Figure, axes: list, ckpt: dict, log_lines: list[str]) -> None:
    ax_qual, ax_pnl, ax_status, ax_log = axes

    weekly_summary = ckpt.get("weekly_summary", [])
    weekly_returns = ckpt.get("weekly_returns", [])
    all_trades     = ckpt.get("all_trades", [])
    last_week      = ckpt.get("last_week", 0)
    retrains       = ckpt.get("retrains", 0)
    kill_sw        = ckpt.get("kill_switch_hit", False)
    ts             = ckpt.get("ts", "ΓÇö")
    run_id         = ckpt.get("run_id", "ΓÇö")

    weeks    = [m["week"]               for m in weekly_summary]
    cum_ret  = [m.get("cum_return_pct",  0) for m in weekly_summary]
    max_dd   = [m.get("max_drawdown_pct", 0) for m in weekly_summary]

    win_rates = _win_rates_by_week(all_trades)
    sharpes   = _rolling_sharpe(weekly_returns)

    # ΓöÇΓöÇ Panel 1: Training Quality by Cycle ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    ax_qual.clear()
    ax_qual.set_facecolor(PANEL)
    ax_qual.grid(True, alpha=0.4, color=GRID)
    if win_rates:
        ax_qual.plot(range(1, len(win_rates) + 1), win_rates,
                     color=ACC2, linewidth=2.2, label="Win rate")
    if sharpes:
        ax_qual.plot(range(1, len(sharpes) + 1), [s * 10 for s in sharpes],
                     color=ACC1, linewidth=2.2, label="Sharpe")
    ax_qual.set_title("Training Quality by Cycle")
    if win_rates or sharpes:
        ax_qual.legend(fontsize=9, loc="upper right",
                       labelcolor=[ACC2, ACC1][:len([x for x in [win_rates, sharpes] if x])])
    ax_qual.tick_params(colors=MUTED)

    # ΓöÇΓöÇ Panel 2: PnL and Drawdown ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    ax_pnl.clear()
    ax_pnl.set_facecolor(PANEL)
    ax_pnl.grid(True, alpha=0.4, color=GRID)
    if weeks:
        ax_pnl.plot(weeks, cum_ret,
                    color=ACC2, linewidth=2.2, label="Total PnL %")
        ax_pnl.plot(weeks, max_dd,
                    color=ACC1, linewidth=2.2, label="Drawdown %")
    ax_pnl.set_title("PnL and Drawdown")
    if weeks:
        ax_pnl.legend(fontsize=9, loc="upper right",
                      labelcolor=[ACC2, ACC1])
    ax_pnl.tick_params(colors=MUTED)

    # ΓöÇΓöÇ Panel 3: Current Status ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    ax_status.clear()
    ax_status.set_facecolor(PANEL)
    ax_status.set_axis_off()
    for sp in ax_status.spines.values():
        sp.set_edgecolor("#dde3ed")
        sp.set_linewidth(0.8)

    n_trades = len(all_trades)
    if all_trades:
        tdf      = pd.DataFrame(all_trades)
        wr       = float((tdf["pnl_percent"] > 0).mean() * 100)
        wins_sum = tdf[tdf["pnl_percent"] > 0]["pnl_percent"].sum()
        loss_sum = abs(tdf[tdf["pnl_percent"] <= 0]["pnl_percent"].sum())
        pf       = wins_sum / loss_sum if loss_sum > 0 else 0.0
    else:
        wr = pf = 0.0

    cur_dd  = weekly_summary[-1].get("max_drawdown_pct", 0) if weekly_summary else 0.0
    cur_ret = weekly_summary[-1].get("cum_return_pct",   0) if weekly_summary else 0.0
    cur_shp = sharpes[-1] if sharpes else 0.0

    is_run  = bool(weekly_summary) and not kill_sw
    st_lbl  = "RUNNING" if is_run else ("KILL-SWITCH" if kill_sw else "IDLE")
    st_col  = ACC3 if is_run else (ACC4 if kill_sw else MUTED)

    # monospace padded rows matching SVG layout
    status_rows = [
        ("Run state     ", st_lbl,            st_col),
        ("Run ID        ", str(run_id)[:20],  TXT),
        ("Last ckpt     ", ts,                TXT),
        ("Current week  ", str(last_week),    TXT),
        ("Retrains      ", str(retrains),     TXT),
        (None, None, None),  # spacer
        ("Latest cycle  ", str(last_week),    TXT),
        ("Trades        ", f"{n_trades:,}",   TXT),
        ("Win rate      ", f"{wr:.3f}%",      ACC3 if wr >= 50 else TXT),
        ("Sharpe        ", f"{cur_shp:.3f}",  TXT),
        ("Drawdown      ", f"{cur_dd:.2f}%",  TXT),
        ("Cum return    ", f"{cur_ret:.2f}%", TXT),
        ("Profit factor ", f"{pf:.3f}",       TXT),
    ]

    MONO = "Consolas"
    y_pos, step = 0.96, 0.071
    for label, val, color in status_rows:
        if label is None:
            y_pos -= step * 0.5
            continue
        # label + colon
        ax_status.text(0.04, y_pos, f"{label}: ",
                       transform=ax_status.transAxes,
                       fontsize=8.8, color=TXT, va="top",
                       fontfamily=MONO)
        # value (colored)
        ax_status.text(0.60, y_pos, val,
                       transform=ax_status.transAxes,
                       fontsize=8.8, color=color, va="top",
                       fontfamily=MONO)
        y_pos -= step

    ax_status.set_title("Current Status")

    # ΓöÇΓöÇ Panel 4: Recent Log Tail ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    ax_log.clear()
    ax_log.set_facecolor(PANEL)
    ax_log.set_axis_off()
    ax_log.text(
        0.015, 0.975,
        "\n".join(log_lines),
        transform=ax_log.transAxes,
        fontsize=8, color=TXT, va="top", ha="left",
        fontfamily="Consolas",
        linespacing=1.5,
    )
    ax_log.set_title("Recent Log Tail")

    fig.suptitle(
        "Azalyst Alpha Research Engine  -  Spyder Monitor",
        fontsize=15, fontweight="bold", color=TITLE_FG, y=0.984,
        fontfamily="Segoe UI",
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ΓöÇΓöÇ Main loop ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def run_dashboard(refresh: int = REFRESH) -> None:
    _apply_style()

    fig = plt.figure("Azalyst Monitor", figsize=(15, 9), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.04, right=0.97,
        top=0.93,  bottom=0.05,
        hspace=0.48, wspace=0.32,
    )
    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    plt.ion()
    plt.show(block=False)

    print(f"[Azalyst Monitor] Live dashboard started ΓÇö refreshes every {refresh}s")
    print(f"[Azalyst Monitor] Checkpoint : {CKPT}")
    print(f"[Azalyst Monitor] Log file   : {LOG}")
    print("[Azalyst Monitor] Close the window or Ctrl+C to exit.\n")

    try:
        while plt.fignum_exists(fig.number):
            ckpt      = _load_ckpt()
            log_lines = _tail_log()
            _render(fig, axes, ckpt, log_lines)
            plt.pause(refresh)
    except KeyboardInterrupt:
        pass

    print("[Azalyst Monitor] Exited.")


if __name__ == "__main__":
    run_dashboard()
