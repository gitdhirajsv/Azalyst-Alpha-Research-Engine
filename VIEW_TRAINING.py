"""
Azalyst Alpha Research Engine  —  Spyder Monitor (Live)

Auto-launched by RUN_AZALYST.bat when you choose Terminal + Spyder mode.
Can also be run manually:  python VIEW_TRAINING.py  or F5 in Spyder.

Reads results/checkpoint_v4_latest.json and results/run_log.txt every 5 s
and renders a 4-panel live dashboard:
  • Training Quality by Week  (win rate + rolling Sharpe)
  • PnL and Drawdown          (cumulative return + max drawdown curve)
  • Current Status            (week, trades, win rate, Sharpe, DD, PF)
  • Recent Log Tail           (last N lines from the engine log file)
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results"
CKPT = RES  / "checkpoint_v4_latest.json"
LOG  = RES  / "run_log.txt"

REFRESH = 5   # seconds between refreshes
LOG_TAIL = 18  # log lines to show

# ── Theme (dark — matches RUN_AZALYST terminal palette) ──────────────────────
BG    = "#0b1220"
PANEL = "#111b2e"
ACC1  = "#f07f2a"   # orange
ACC2  = "#4fa8d5"   # blue
ACC3  = "#21c16b"   # green
ACC4  = "#e05252"   # red
TXT   = "#d9f5e2"
MUTED = "#8fb7a0"
GRID  = "#1a2a3a"


# ── Data helpers ──────────────────────────────────────────────────────────────
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
        return [f"Waiting for {LOG.name} …  (pipeline not started yet)"]


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


# ── Style ─────────────────────────────────────────────────────────────────────
def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TXT,
        "axes.titlecolor":   TXT,
        "xtick.color":       TXT,
        "ytick.color":       TXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.5,
        "text.color":        TXT,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
    })


# ── Render one frame ──────────────────────────────────────────────────────────
def _render(fig: plt.Figure, axes: list, ckpt: dict, log_lines: list[str]) -> None:
    ax_qual, ax_pnl, ax_status, ax_log = axes

    weekly_summary = ckpt.get("weekly_summary", [])
    weekly_returns = ckpt.get("weekly_returns", [])
    all_trades     = ckpt.get("all_trades", [])
    last_week      = ckpt.get("last_week", 0)
    retrains       = ckpt.get("retrains", 0)
    kill_sw        = ckpt.get("kill_switch_hit", False)
    ts             = ckpt.get("ts", "—")
    run_id         = ckpt.get("run_id", "—")

    weeks    = [m["week"]               for m in weekly_summary]
    cum_ret  = [m.get("cum_return_pct",  0) for m in weekly_summary]
    max_dd   = [m.get("max_drawdown_pct", 0) for m in weekly_summary]

    win_rates = _win_rates_by_week(all_trades)
    sharpes   = _rolling_sharpe(weekly_returns)

    # ── Panel 1: Training Quality by Week ────────────────────────────────────
    ax_qual.clear()
    ax_qual.set_facecolor(PANEL)
    ax_qual.grid(True, alpha=0.3, color=GRID)
    if win_rates:
        ax_qual.plot(range(1, len(win_rates) + 1), win_rates,
                     color=ACC1, linewidth=1.8, label="Win rate %")
    if sharpes:
        ax_qual.plot(range(1, len(sharpes) + 1), [s * 10 for s in sharpes],
                     color=ACC2, linewidth=1.8, label="Sharpe ×10")
    ax_qual.set_title("Training Quality by Week")
    ax_qual.legend(fontsize=8, framealpha=0.15, labelcolor=TXT, loc="upper left")
    ax_qual.tick_params(colors=TXT)

    # ── Panel 2: PnL and Drawdown ─────────────────────────────────────────────
    ax_pnl.clear()
    ax_pnl.set_facecolor(PANEL)
    ax_pnl.grid(True, alpha=0.3, color=GRID)
    if weeks:
        ax_pnl.plot(weeks, cum_ret,
                    color=ACC2, linewidth=1.8, label="Total PnL %")
        ax_pnl.plot(weeks, max_dd,
                    color=ACC1, linewidth=1.8, label="Drawdown %")
    ax_pnl.set_title("PnL and Drawdown")
    ax_pnl.legend(fontsize=8, framealpha=0.15, labelcolor=TXT, loc="upper left")
    ax_pnl.tick_params(colors=TXT)

    # ── Panel 3: Current Status ───────────────────────────────────────────────
    ax_status.clear()
    ax_status.set_facecolor(PANEL)
    ax_status.set_axis_off()
    for sp in ax_status.spines.values():
        sp.set_edgecolor(ACC1)
        sp.set_linewidth(1.2)

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

    status_rows = [
        ("Current Status",  "",                 None),
        ("Run state",       st_lbl,             st_col),
        ("Run ID",          str(run_id)[:22],   TXT),
        ("Last checkpoint", ts,                 TXT),
        ("Current week",    str(last_week),      TXT),
        ("Retrains",        str(retrains),       TXT),
        ("",                "",                 None),
        ("Latest metrics",  "",                 None),
        ("Trades",          f"{n_trades:,}",    TXT),
        ("Win rate",        f"{wr:.3f}%",        ACC3 if wr >= 50 else ACC1),
        ("Sharpe",          f"{cur_shp:.3f}",   ACC3 if cur_shp >= 1 else (ACC4 if cur_shp < 0 else ACC1)),
        ("Drawdown",        f"{cur_dd:.2f}%",   ACC4 if abs(cur_dd) > 20 else ACC1),
        ("Cum return",      f"{cur_ret:.2f}%",  ACC3 if cur_ret >= 0 else ACC4),
        ("Profit factor",   f"{pf:.3f}",        ACC3 if pf >= 1.5 else ACC1),
    ]

    y_pos, step = 0.97, 0.072
    for label, val, color in status_rows:
        if color is None:
            ax_status.text(0.04, y_pos, label,
                           transform=ax_status.transAxes,
                           fontsize=9, color=ACC1, fontweight="bold", va="top")
        else:
            ax_status.text(0.04, y_pos, label,
                           transform=ax_status.transAxes,
                           fontsize=8.5, color=MUTED, va="top")
            ax_status.text(0.96, y_pos, val,
                           transform=ax_status.transAxes,
                           fontsize=8.5, color=color, va="top", ha="right")
        y_pos -= step

    ax_status.set_title("Current Status")

    # ── Panel 4: Recent Log Tail ──────────────────────────────────────────────
    ax_log.clear()
    ax_log.set_facecolor("#060d18")
    ax_log.set_axis_off()
    ax_log.text(
        0.01, 0.99,
        "\n".join(log_lines),
        transform=ax_log.transAxes,
        fontsize=7.5, color="#7dff9f", va="top", ha="left",
        fontfamily="monospace",
    )
    ax_log.set_title("Recent Log Tail")

    fig.suptitle(
        "Azalyst Alpha Research Engine  —  Spyder Monitor",
        fontsize=13, fontweight="bold", color=ACC1, y=0.99,
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_dashboard(refresh: int = REFRESH) -> None:
    _apply_style()

    fig = plt.figure("Azalyst Monitor", figsize=(14, 9), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.05, right=0.97,
        top=0.94,  bottom=0.05,
        hspace=0.42, wspace=0.35,
    )
    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    plt.ion()
    plt.show(block=False)

    print(f"[Azalyst Monitor] Live dashboard started — refreshes every {refresh}s")
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
