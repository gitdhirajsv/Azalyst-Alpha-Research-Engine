"""
Spyder-friendly live monitor.

Run this file from Spyder while `RUN_SPYDER_SHIFT_MONITOR.bat` keeps the
autonomous workflow alive in the background. It reads local artifacts only,
so it does not interfere with the Ollama / team logic.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import monitor_dashboard as md


ROOT = Path(__file__).resolve().parent
METRICS_FILE = ROOT / "performance_metrics.csv"
REFRESH_SECONDS = 5


def _read_metrics_rows() -> list[dict]:
    if not METRICS_FILE.exists():
        return []

    try:
        with METRICS_FILE.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _to_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except Exception:
        return 0.0


def _latest_cycle_rows(rows: list[dict]) -> list[dict]:
    latest_by_cycle: dict[str, dict] = {}
    order: list[str] = []

    for row in rows:
        cycle = str(row.get("cycle", "")).strip()
        if not cycle:
            continue
        if cycle not in latest_by_cycle:
            order.append(cycle)
        latest_by_cycle[cycle] = row

    return [latest_by_cycle[cycle] for cycle in order]


def _draw_figure(fig, axes, status: dict, rows: list[dict]) -> None:
    rows = _latest_cycle_rows(rows)
    cycles = [_to_float(row, "cycle") for row in rows]
    win_rates = [_to_float(row, "win_rate") for row in rows]
    sharpes = [_to_float(row, "sharpe_ratio") for row in rows]
    drawdowns = [_to_float(row, "max_drawdown_pct") for row in rows]
    pnls = [_to_float(row, "total_pnl_pct") for row in rows]

    ax = axes[0][0]
    ax.clear()
    if cycles:
        ax.plot(cycles, win_rates, marker="o", linewidth=2, label="Win rate")
        ax.plot(cycles, sharpes, marker="s", linewidth=2, label="Sharpe")
        ax.set_title("Training Quality by Cycle")
        ax.set_xlabel("Cycle")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No metrics yet", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[0][1]
    ax.clear()
    if cycles:
        ax.plot(cycles, pnls, marker="o", linewidth=2, label="Total PnL %")
        ax.plot(cycles, drawdowns, marker="s", linewidth=2, label="Drawdown %")
        ax.set_title("PnL and Drawdown")
        ax.set_xlabel("Cycle")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Waiting for simulation output", ha="center", va="center")
        ax.set_axis_off()

    latest = status["last_metrics"]
    ax = axes[1][0]
    ax.clear()
    ax.axis("off")
    process_text = "\n".join(
        [
            f"Run state      : {'RUNNING' if status['running'] else 'STOPPED'}",
            f"Team PID       : {status['team_pid'] or 'n/a'}",
            f"Simulator PID  : {status['sim_pid'] or 'n/a'}",
            f"Cycle index    : {status['cycle_index']}",
            f"Current step   : {status['current_step']}",
            f"Last checkpoint: {status['last_checkpoint']}",
            "",
            f"Latest cycle   : {latest['cycle']}",
            f"Trades         : {latest['trades']}",
            f"Win rate       : {latest['win_rate']}%",
            f"Sharpe         : {latest['sharpe']}",
            f"Drawdown       : {latest['drawdown']}%",
            f"Profit factor  : {latest['profit_factor']}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        process_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    ax.set_title("Current Status", loc="left")

    ax = axes[1][1]
    ax.clear()
    ax.axis("off")
    log_lines = status["recent_log_lines"][-18:] or ["No log output yet."]
    ax.text(
        0.02,
        0.98,
        "\n".join(log_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    ax.set_title("Recent Log Tail", loc="left")

    fig.suptitle("Azalyst Alpha Research Engine - Spyder Monitor", fontsize=14)
    fig.tight_layout()


def render_monitor() -> None:
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.canvas.manager.set_window_title("Azalyst Spyder Monitor")

    monitor = md.PipelineMonitor()

    while plt.fignum_exists(fig.number):
        status = monitor.get_status()
        rows = _read_metrics_rows()
        _draw_figure(fig, axes, status, rows)
        plt.pause(REFRESH_SECONDS)


def save_snapshot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    _draw_figure(fig, axes, md.PipelineMonitor().get_status(), _read_metrics_rows())
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azalyst Spyder monitor helper")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Save a static monitor snapshot instead of opening the live view.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.snapshot:
        save_snapshot(args.snapshot)
    else:
        render_monitor()
