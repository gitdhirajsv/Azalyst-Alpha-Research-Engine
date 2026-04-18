"""
Azalyst Paper Trader — Flask web app.

Endpoints:
  GET  /          → live dashboard (auto-refreshes every 60s)
  GET  /api/stats → JSON stats
  GET  /ping      → keep-alive (point UptimeRobot here)
  POST /run-now?key=ADMIN_KEY  → manual trigger for testing
"""
import os
import json
from datetime import datetime, timezone

from flask import Flask, jsonify, request, Response
from apscheduler.schedulers.background import BackgroundScheduler

from paper_trader import PaperTrader

# ── Init ──────────────────────────────────────────────────────────────────────
app     = Flask(__name__)
DB_PATH = os.environ.get("DB_PATH", "/tmp/azalyst_paper.db")

trader = PaperTrader(
    db_path         = DB_PATH,
    initial_balance = float(os.environ.get("INITIAL_BALANCE", "1000.0")),
    top_n           = int(os.environ.get("TOP_N", "5")),
    n_symbols       = int(os.environ.get("N_SYMBOLS", "50")),
)

# ── Background scheduler ──────────────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    trader.run_weekly_cycle,
    trigger    = "cron",
    day_of_week = "mon",
    hour        = 0,
    minute      = 15,
    id          = "weekly_cycle",
    misfire_grace_time = 3600,
)
scheduler.start()


# ── Dashboard HTML ────────────────────────────────────────────────────────────

def _regime_badge(regime: str) -> str:
    colors = {
        "BULL_TREND":       "#27a060",
        "BEAR_TREND":       "#d94040",
        "LOW_VOL_GRIND":    "#1f77b4",
        "HIGH_VOL_LATERAL": "#ff7f0e",
        "KILL_SWITCH":      "#888780",
    }
    c = colors.get(regime, "#888780")
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:10px;font-size:11px">{regime}</span>'


def build_dashboard(stats: dict) -> str:
    bal         = stats["balance"]
    init_bal    = stats["initial_balance"]
    cum_ret     = stats["cum_return_pct"]
    max_dd      = stats["max_drawdown_pct"]
    cycles      = stats["cycle_count"]
    win_rate    = stats["win_rate_pct"]
    open_pos    = stats["open_positions"]
    recent_c    = stats["recent_cycles"]
    recent_t    = stats["recent_trades"]
    ts          = stats["timestamp"][:16].replace("T", " ") + " UTC"

    ret_color  = "#27a060" if cum_ret >= 0 else "#d94040"
    dd_color   = "#d94040" if max_dd < -5 else "#e8a020"
    ret_sign   = "+" if cum_ret >= 0 else ""

    # Equity curve sparkline data (last 12 cycles)
    curve_pts = [c.get("cum_return_pct") for c in reversed(recent_c) if c.get("cum_return_pct") is not None]
    spark_vals = json.dumps(curve_pts) if curve_pts else "[]"

    # Open positions table
    pos_rows = ""
    for p in open_pos:
        sym   = p["symbol"]
        side  = p["side"]
        entry = p.get("entry_price", 0)
        size  = p.get("size_usd", 0)
        sc    = "#27a060" if side == "LONG" else "#d94040"
        pos_rows += f"""
        <tr>
          <td><strong>{sym}</strong></td>
          <td style="color:{sc}">{side}</td>
          <td>${entry:.4f}</td>
          <td>${size:.2f}</td>
        </tr>"""

    if not pos_rows:
        pos_rows = '<tr><td colspan="4" style="color:#6b7a99">No open positions</td></tr>'

    # Recent cycles table
    cycle_rows = ""
    for c in recent_c[:8]:
        ret  = c.get("week_return_pct") or 0
        ic   = c.get("ic") or 0
        rg   = c.get("regime", "—")
        bal_ = c.get("balance_after") or "—"
        rc   = "#27a060" if ret >= 0 else "#d94040"
        ic_c = "#27a060" if ic >= 0 else "#d94040"
        rs   = f"${bal_:.2f}" if isinstance(bal_, float) else bal_
        cycle_rows += f"""
        <tr>
          <td>{c.get("started_at","")[:10]}</td>
          <td style="color:{rc}">{ret:+.2f}%</td>
          <td style="color:{ic_c}">{ic:+.4f}</td>
          <td>{_regime_badge(rg)}</td>
          <td>{rs}</td>
        </tr>"""

    if not cycle_rows:
        cycle_rows = '<tr><td colspan="5" style="color:#6b7a99">No cycles yet — first cycle runs Monday 00:15 UTC</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Azalyst Paper Trader</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:monospace;background:#ffffff;color:#2d3f55;padding:20px;font-size:13px}}
h1{{color:#ff7f0e;font-size:18px;margin-bottom:4px}}
.sub{{color:#888;font-size:11px;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#ffffff;border:1px solid #e0e0e0;border-radius:8px;padding:14px;box-shadow:0 2px 4px rgba(0,0,0,0.05)}}
.card h2{{color:#ff7f0e;font-size:11px;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}}
.card .val{{font-size:22px;font-weight:600}}
table{{width:100%;border-collapse:collapse;margin-top:8px}}
th{{color:#888;text-align:left;padding:6px 8px;border-bottom:1px solid #eee;font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
td{{padding:5px 8px;border-bottom:1px solid #f5f5f5;font-size:12px}}
tr:hover td{{background:#fafafa}}
canvas{{width:100%;height:80px}}
</style>
<script>setTimeout(()=>location.reload(),60000)</script>
</head>
<body>
<h1>🔬 Azalyst Paper Trader — 30-Day Live Test</h1>
<div class="sub">Last updated: {ts} &nbsp;·&nbsp; {cycles} weekly cycles completed &nbsp;·&nbsp; {len(open_pos)} open positions</div>

<div class="grid">
  <div class="card">
    <h2>Balance</h2>
    <div class="val">${bal:.2f}</div>
    <div style="color:#6b7a99;font-size:11px;margin-top:4px">started ${init_bal:.0f}</div>
  </div>
  <div class="card">
    <h2>Total return</h2>
    <div class="val" style="color:{ret_color}">{ret_sign}{cum_ret:.2f}%</div>
  </div>
  <div class="card">
    <h2>Max drawdown</h2>
    <div class="val" style="color:{dd_color}">{max_dd:.2f}%</div>
  </div>
  <div class="card">
    <h2>Win rate</h2>
    <div class="val">{win_rate:.1f}%</div>
    <div style="color:#6b7a99;font-size:11px;margin-top:4px">{stats["total_closed_trades"]} closed trades</div>
  </div>
</div>

<div class="card" style="margin-bottom:16px">
  <h2>Equity curve</h2>
  <canvas id="sparkline"></canvas>
</div>

<div class="card" style="margin-bottom:16px">
  <h2>Open positions</h2>
  <table>
    <tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Size USD</th></tr>
    {pos_rows}
  </table>
</div>

<div class="card" style="margin-bottom:16px">
  <h2>Weekly cycles</h2>
  <table>
    <tr><th>Week start</th><th>Return</th><th>IC</th><th>Regime</th><th>Balance</th></tr>
    {cycle_rows}
  </table>
</div>

<script>
const vals = {spark_vals};
const canvas = document.getElementById('sparkline');
if(canvas && vals.length > 1){{
  canvas.width  = canvas.offsetWidth * devicePixelRatio;
  canvas.height = 80 * devicePixelRatio;
  canvas.style.height = '80px';
  const ctx = canvas.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const W = canvas.offsetWidth, H = 80;
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const range = mx - mn || 1;
  const x = i => (i / (vals.length - 1)) * W;
  const y = v => H - 10 - ((v - mn) / range) * (H - 20);
  ctx.strokeStyle = '#ff7f0e';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  vals.forEach((v,i) => i===0 ? ctx.moveTo(x(i),y(v)) : ctx.lineTo(x(i),y(v)));
  ctx.stroke();
  ctx.strokeStyle = '#2d3f55';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([3,3]);
  const yz = y(0);
  ctx.beginPath(); ctx.moveTo(0,yz); ctx.lineTo(W,yz); ctx.stroke();
}}
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    stats = trader.get_dashboard_stats()
    return Response(build_dashboard(stats), mimetype="text/html")


@app.route("/api/stats")
def api_stats():
    return jsonify(trader.get_dashboard_stats())


@app.route("/ping")
def ping():
    """Keep-alive endpoint — point UptimeRobot here (every 5 min)."""
    return "pong", 200


@app.route("/run-now", methods=["POST"])
def run_now():
    """Manual trigger. Requires ?key=ADMIN_KEY query param."""
    key = request.args.get("key", "")
    if key != os.environ.get("ADMIN_KEY", ""):
        return jsonify({"error": "unauthorized"}), 401
    result = trader.run_weekly_cycle()
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
