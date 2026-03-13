"""
Azalyst live monitor dashboard.

Opens a tiny local web app that shows:
- whether the team/simulator processes are alive
- the latest checkpoint + metrics snapshot
- a tail of the team log so you can watch training progress live
"""

from __future__ import annotations

import csv
import ctypes
import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK_DIR = ROOT / ".azalyst_locks"
TEAM_LOCK = LOCK_DIR / "autonomous_team.lock"
SIM_LOCK = LOCK_DIR / "walkforward_simulator.lock"
CHECKPOINT_FILE = ROOT / "checkpoint.json"
METRICS_FILE = ROOT / "performance_metrics.csv"
TEAM_LOG_FILE = ROOT / "team_log.txt"


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if hasattr(ctypes, "windll"):
        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if process:
            ctypes.windll.kernel32.CloseHandle(process)
            return True
        return False
    return False


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_last_metric_row(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rows[-1] if rows else {}
    except Exception:
        return {}


def _tail_lines(path: Path, max_lines: int = 60) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [line.rstrip("\n") for line in lines[-max_lines:]]
    except Exception:
        return []


def _to_float(row: dict, key: str) -> str:
    try:
        return f"{float(row.get(key, 0)):.3f}"
    except Exception:
        return "n/a"


def _to_int(row: dict, key: str) -> str:
    try:
        return f"{int(float(row.get(key, 0))):,}"
    except Exception:
        return "n/a"


class PipelineMonitor:
    def get_status(self) -> dict:
        team_lock = _read_json(TEAM_LOCK)
        sim_lock = _read_json(SIM_LOCK)
        checkpoint = _read_json(CHECKPOINT_FILE)
        metric_row = _read_last_metric_row(METRICS_FILE)

        team_pid = int(team_lock.get("pid", 0) or 0)
        sim_pid = int(sim_lock.get("pid", 0) or 0)
        team_running = _pid_exists(team_pid)
        sim_running = _pid_exists(sim_pid)

        return {
            "running": team_running or sim_running,
            "team_pid": team_pid if team_running else None,
            "sim_pid": sim_pid if sim_running else None,
            "team_started": team_lock.get("created_at", "n/a"),
            "sim_started": sim_lock.get("created_at", "n/a"),
            "current_step": checkpoint.get("last_step", "n/a"),
            "cycle_index": checkpoint.get("cycle_index", "n/a"),
            "last_checkpoint": checkpoint.get("timestamp", "n/a"),
            "last_metrics": {
                "cycle": metric_row.get("cycle", "n/a"),
                "trades": _to_int(metric_row, "total_trades"),
                "win_rate": _to_float(metric_row, "win_rate"),
                "sharpe": _to_float(metric_row, "sharpe_ratio"),
                "drawdown": _to_float(metric_row, "max_drawdown_pct"),
                "profit_factor": _to_float(metric_row, "profit_factor"),
            },
            "recent_log_lines": _tail_lines(TEAM_LOG_FILE, max_lines=60),
        }


def build_dashboard_html(status: dict) -> str:
    metrics = status["last_metrics"]
    log_text = html.escape("\n".join(status["recent_log_lines"]) or "No log output yet.")
    run_label = "RUNNING" if status["running"] else "STOPPED"
    run_class = "running" if status["running"] else "stopped"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Azalyst Live Monitor</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #111b2e;
      --panel-2: #16243d;
      --text: #d9f5e2;
      --muted: #8fb7a0;
      --accent: #21c16b;
      --warn: #e0b34a;
      --danger: #f0626e;
      --border: rgba(143, 183, 160, 0.22);
      --shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Consolas, "Courier New", monospace;
      background:
        radial-gradient(circle at top left, rgba(33, 193, 107, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(224, 179, 74, 0.10), transparent 24%),
        var(--bg);
      color: var(--text);
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0.04em;
    }}
    .hero p {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    .badge {{
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 700;
      border: 1px solid var(--border);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .badge.running {{ color: var(--accent); }}
    .badge.stopped {{ color: var(--danger); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent), var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .card h2 {{
      margin: 0 0 14px;
      font-size: 14px;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .status {{ grid-column: span 4; }}
    .metrics {{ grid-column: span 4; }}
    .pids {{ grid-column: span 4; }}
    .logs {{ grid-column: span 12; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }}
    .stat {{
      background: var(--panel-2);
      border-radius: 12px;
      padding: 12px;
      border: 1px solid rgba(143, 183, 160, 0.12);
    }}
    .label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 20px;
      font-weight: 700;
    }}
    .meta {{
      display: grid;
      gap: 10px;
    }}
    .meta-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid rgba(143, 183, 160, 0.08);
      padding-bottom: 8px;
    }}
    .meta-row:last-child {{
      border-bottom: 0;
      padding-bottom: 0;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #08111d;
      border: 1px solid rgba(143, 183, 160, 0.08);
      border-radius: 12px;
      padding: 14px;
      max-height: 560px;
      overflow: auto;
      color: #7dff9f;
      line-height: 1.45;
    }}
    @media (max-width: 900px) {{
      .status, .metrics, .pids, .logs {{ grid-column: span 12; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div>
        <h1>Azalyst Live Monitor</h1>
        <p>Auto-refresh every 5 seconds. Open this while the autonomous team is running.</p>
      </div>
      <div class="badge {run_class}">{run_label}</div>
    </div>

    <div class="grid">
      <section class="card status">
        <h2>Session</h2>
        <div class="meta">
          <div class="meta-row"><span>Current step</span><strong>{html.escape(str(status["current_step"]))}</strong></div>
          <div class="meta-row"><span>Cycle index</span><strong>{html.escape(str(status["cycle_index"]))}</strong></div>
          <div class="meta-row"><span>Last checkpoint</span><strong>{html.escape(str(status["last_checkpoint"]))}</strong></div>
        </div>
      </section>

      <section class="card metrics">
        <h2>Latest Metrics</h2>
        <div class="stats">
          <div class="stat"><span class="label">Cycle</span><span class="value">{html.escape(str(metrics["cycle"]))}</span></div>
          <div class="stat"><span class="label">Trades</span><span class="value">{metrics["trades"]}</span></div>
          <div class="stat"><span class="label">Win rate</span><span class="value">{metrics["win_rate"]}%</span></div>
          <div class="stat"><span class="label">Sharpe</span><span class="value">{metrics["sharpe"]}</span></div>
          <div class="stat"><span class="label">Drawdown</span><span class="value">{metrics["drawdown"]}%</span></div>
          <div class="stat"><span class="label">Profit factor</span><span class="value">{metrics["profit_factor"]}</span></div>
        </div>
      </section>

      <section class="card pids">
        <h2>Processes</h2>
        <div class="meta">
          <div class="meta-row"><span>Team PID</span><strong>{status["team_pid"] or "not running"}</strong></div>
          <div class="meta-row"><span>Team started</span><strong>{html.escape(str(status["team_started"]))}</strong></div>
          <div class="meta-row"><span>Simulator PID</span><strong>{status["sim_pid"] or "not running"}</strong></div>
          <div class="meta-row"><span>Simulator started</span><strong>{html.escape(str(status["sim_started"]))}</strong></div>
        </div>
      </section>

      <section class="card logs">
        <h2>Recent Log</h2>
        <pre>{log_text}</pre>
      </section>
    </div>
  </div>
</body>
</html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    monitor = PipelineMonitor()

    def do_GET(self) -> None:
        if self.path == "/status":
            payload = json.dumps(self.monitor.get_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(payload)
            return

        status = self.monitor.get_status()
        body = build_dashboard_html(status).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return

def start_monitor_server(port: int = 8080) -> None:
    server = HTTPServer(("127.0.0.1", port), MonitorHandler)
    print(f"Azalyst Live Monitor running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop the monitor.")
    server.serve_forever()


if __name__ == "__main__":
    start_monitor_server()
