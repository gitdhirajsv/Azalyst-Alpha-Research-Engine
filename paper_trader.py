"""
Azalyst Paper Trader — weekly signal generation + position tracking.

Features computed inline (no heavy azalyst_factors_v2 import) to stay
within Render free-tier 512 MB RAM limit.

Proven features from 77-week OOS backtest (IC-weighted composite):
  kyle_lambda  IC=+0.112  weight=0.35
  amihud       IC=+0.109  weight=0.34
  ret_3d       IC=+0.013  weight=0.12
  vol_regime   IC=+0.028  weight=0.10
  rsi_14       IC=+0.018  weight=0.09

Regime rules (same as backtest):
  BULL_TREND          → long-only  0.5× per position
  BEAR_TREND          → short-only 1.0× per position
  LOW_VOL_GRIND       → long+short 1.0× per position
  HIGH_VOL_LATERAL    → short-only 0.5× per position
"""

import gc
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from live_data import get_top_symbols, fetch_universe_data, get_current_prices

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
BARS_PER_DAY = 288          # 5-min bars in 24 hours
ROUND_TRIP_FEE = 0.002      # 0.2% entry + exit combined
MAX_DRAWDOWN_KILL = -0.20   # halt trading at -20% from peak

FEATURE_WEIGHTS = {
    "kyle_lambda": 0.35,
    "amihud":      0.34,
    "ret_3d":      0.12,
    "vol_regime":  0.10,
    "rsi_14":      0.09,
}

# ── INLINE FEATURE COMPUTATION ────────────────────────────────────────────────

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-diff).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def compute_proven_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the 5 proven features from OHLCV DataFrame.
    Returns a dict {feature_name: last_valid_value}.
    Requires at least BARS_PER_DAY * 5 = 1440 bars for vol_regime.
    """
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    bpd = BARS_PER_DAY

    lr = np.log(c / c.shift(1))

    # kyle_lambda: |return| / (price * volume), rolling mean
    kyle = (lr.abs() / (c * v).replace(0, np.nan)).rolling(bpd, min_periods=bpd // 2).mean()

    # amihud: |return| / volume, rolling mean
    amihud = (lr.abs() / v.replace(0, np.nan)).rolling(bpd, min_periods=bpd // 2).mean()

    # ret_3d: 3-day log return
    ret_3d = np.log(c / c.shift(bpd * 3))

    # vol_regime: percentile rank of daily realised vol over last 5 days
    rvol = lr.rolling(bpd, min_periods=bpd // 2).std()
    vol_regime = rvol.rolling(bpd * 5, min_periods=bpd).rank(pct=True)

    # rsi_14 (0–1 normalised)
    rsi = _rsi(c, 14) / 100.0

    result = {}
    for name, series in [
        ("kyle_lambda", kyle),
        ("amihud", amihud),
        ("ret_3d", ret_3d),
        ("vol_regime", vol_regime),
        ("rsi_14", rsi),
    ]:
        vals = series.dropna()
        result[name] = float(vals.iloc[-1]) if len(vals) > 0 else np.nan

    return result


def _detect_regime(symbol_data: Dict[str, pd.DataFrame]) -> str:
    """Regime detection using BTC (or best proxy) 5-day window."""
    proxy = (
        symbol_data.get("BTCUSDT")
        or symbol_data.get("ETHUSDT")
        or (max(symbol_data.values(), key=len) if symbol_data else None)
    )
    if proxy is None:
        return "LOW_VOL_GRIND"

    close = proxy["close"].astype(float)
    lr    = np.log(close / close.shift(1)).dropna()

    idx_5d  = max(0, len(close) - BARS_PER_DAY * 5)
    btc_ret = float(np.log(close.iloc[-1] / close.iloc[idx_5d])) if close.iloc[idx_5d] > 0 else 0.0

    avg_vol    = float(lr.std())
    recent_vol = float(lr.tail(BARS_PER_DAY).std()) if len(lr) >= BARS_PER_DAY else avg_vol

    high_vol = recent_vol > avg_vol * 1.3

    if btc_ret > 0.03 and not high_vol:
        return "BULL_TREND"
    if btc_ret < -0.03:
        return "BEAR_TREND"
    if high_vol:
        return "HIGH_VOL_LATERAL"
    return "LOW_VOL_GRIND"


# ── PAPER TRADER ──────────────────────────────────────────────────────────────

class PaperTrader:

    def __init__(
        self,
        db_path:         str   = "/tmp/azalyst_paper.db",
        initial_balance: float = 1000.0,
        top_n:           int   = 5,
        n_symbols:       int   = 50,
    ):
        self.db_path         = db_path
        self.initial_balance = initial_balance
        self.top_n           = top_n
        self.n_symbols       = n_symbols
        self._init_db()
        self._ensure_state()

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cycles (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at      TEXT NOT NULL,
                    closed_at       TEXT,
                    regime          TEXT,
                    n_longs         INTEGER DEFAULT 0,
                    n_shorts        INTEGER DEFAULT 0,
                    closed_pnl_usd  REAL    DEFAULT 0,
                    week_return_pct REAL    DEFAULT 0,
                    balance_after   REAL,
                    cum_return_pct  REAL,
                    ic              REAL    DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id      INTEGER REFERENCES cycles(id),
                    symbol        TEXT NOT NULL,
                    side          TEXT NOT NULL,   -- 'LONG' or 'SHORT'
                    entry_price   REAL NOT NULL,
                    exit_price    REAL,
                    size_usd      REAL NOT NULL,
                    pnl_usd       REAL,
                    pnl_pct       REAL,
                    feature_score REAL,
                    status        TEXT DEFAULT 'open',
                    opened_at     TEXT NOT NULL,
                    closed_at     TEXT
                );
            """)
            db.commit()

    def _get(self, key: str, default=None):
        with self._conn() as db:
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set(self, key: str, value):
        with self._conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            db.commit()

    def _ensure_state(self):
        if self._get("balance") is None:
            self._set("balance", self.initial_balance)
        if self._get("cycle_count") is None:
            self._set("cycle_count", 0)
        if self._get("open_positions") is None:
            self._set("open_positions", [])

    # ── Core logic ────────────────────────────────────────────────────────────

    def _compute_signals(
        self, symbol_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Cross-sectional IC-weighted feature ranking for all symbols."""
        records = []
        for sym, df in symbol_data.items():
            if len(df) < BARS_PER_DAY * 5:
                continue
            try:
                feats = compute_proven_features(df)
                feats["symbol"] = sym
                records.append(feats)
            except Exception:
                pass
            gc.collect()

        if not records:
            return pd.DataFrame()

        sig = pd.DataFrame(records).set_index("symbol")

        # Cross-sectional rank-normalise then weight
        composite = np.zeros(len(sig))
        for feat, weight in FEATURE_WEIGHTS.items():
            if feat not in sig.columns:
                continue
            vals  = sig[feat].values.astype(float)
            valid = np.isfinite(vals)
            if valid.sum() < 5:
                continue
            ranks = np.full(len(vals), 0.5)
            ranks[valid] = stats.rankdata(vals[valid]) / (valid.sum() + 1)
            composite += weight * ranks

        sig["score"] = composite
        return sig.reset_index()

    def _compute_drawdown(self) -> float:
        with self._conn() as db:
            rows = db.execute(
                "SELECT balance_after FROM cycles WHERE balance_after IS NOT NULL ORDER BY id"
            ).fetchall()
        if not rows:
            return 0.0
        curve = np.array([self.initial_balance] + [r[0] for r in rows])
        peak  = np.maximum.accumulate(curve)
        dd    = (curve - peak) / peak
        return float(dd.min())

    def _compute_ic(
        self,
        predictions: Dict[str, float],
        actual_rets: Dict[str, float],
    ) -> float:
        common = [s for s in predictions if s in actual_rets]
        if len(common) < 10:
            return 0.0
        pred = np.array([predictions[s] for s in common])
        ret  = np.array([actual_rets[s] for s in common])
        ic, _ = stats.spearmanr(pred, ret)
        return float(ic) if np.isfinite(ic) else 0.0

    # ── Weekly cycle ──────────────────────────────────────────────────────────

    def run_weekly_cycle(self) -> dict:
        """
        Full weekly cycle:
          1. Close open positions at current prices
          2. Fetch fresh universe data
          3. Compute signals
          4. Detect regime, select top-N longs/shorts
          5. Open new positions
          6. Record everything
        """
        now = datetime.now(timezone.utc)
        print(f"\n[{now.isoformat()}] === Weekly cycle started ===")

        balance     = self._get("balance")
        cycle_count = self._get("cycle_count")

        # Kill-switch check
        dd = self._compute_drawdown()
        if dd < MAX_DRAWDOWN_KILL:
            msg = f"KILL SWITCH: drawdown {dd*100:.1f}% < {MAX_DRAWDOWN_KILL*100:.0f}%"
            print(f"  {msg}")
            return {"status": "kill_switch", "drawdown_pct": dd * 100}

        # ── Step 1: close open positions ──────────────────────────────────────
        open_positions = self._get("open_positions") or []
        closed_pnl     = 0.0
        actual_rets_for_ic: Dict[str, float] = {}

        if open_positions:
            print(f"  Closing {len(open_positions)} open positions...")
            symbols_to_price = list({p["symbol"] for p in open_positions})
            current_prices   = get_current_prices(symbols_to_price)

            with self._conn() as db:
                for pos in open_positions:
                    sym   = pos["symbol"]
                    price = current_prices.get(sym)
                    if price is None:
                        continue
                    entry  = pos["entry_price"]
                    size   = pos["size_usd"]
                    ret_frac = (price - entry) / entry
                    if pos["side"] == "SHORT":
                        ret_frac = -ret_frac
                    pnl = size * ret_frac
                    closed_pnl += pnl
                    actual_rets_for_ic[sym] = ret_frac

                    db.execute(
                        "UPDATE trades SET exit_price=?, pnl_usd=?, pnl_pct=?, "
                        "status='closed', closed_at=? WHERE id=?",
                        (price, round(pnl, 4), round(ret_frac * 100, 4), now.isoformat(), pos["id"]),
                    )
                db.commit()

            balance += closed_pnl
            self._set("balance", balance)

            # Update cycle record with closing PnL
            with self._conn() as db:
                db.execute(
                    "UPDATE cycles SET closed_at=?, closed_pnl_usd=?, "
                    "week_return_pct=?, balance_after=?, cum_return_pct=? "
                    "WHERE id=(SELECT MAX(id) FROM cycles)",
                    (
                        now.isoformat(),
                        round(closed_pnl, 4),
                        round(closed_pnl / max(balance - closed_pnl, 1) * 100, 4),
                        round(balance, 4),
                        round((balance / self.initial_balance - 1) * 100, 4),
                    ),
                )
                db.commit()

        self._set("open_positions", [])

        # ── Step 2: fetch fresh data ──────────────────────────────────────────
        print(f"  Fetching top {self.n_symbols} symbols...")
        symbols     = get_top_symbols(self.n_symbols)
        print(f"  Fetching 5-min data ({len(symbols)} symbols)...")
        symbol_data = fetch_universe_data(symbols, limit=1500)
        print(f"  Got data for {len(symbol_data)} symbols")

        if len(symbol_data) < 10:
            return {"status": "error", "message": "insufficient symbol data"}

        # ── Step 3: signals ───────────────────────────────────────────────────
        print("  Computing cross-sectional signals...")
        sig_df = self._compute_signals(symbol_data)
        if sig_df.empty or len(sig_df) < 10:
            return {"status": "error", "message": "signal computation failed"}

        # ── Step 4: regime + selection ────────────────────────────────────────
        regime  = _detect_regime(symbol_data)
        sig_df  = sig_df.sort_values("score", ascending=False).reset_index(drop=True)
        pred_by_symbol = dict(zip(sig_df["symbol"], sig_df["score"]))

        n = min(self.top_n, len(sig_df) // 2)

        if regime == "BEAR_TREND":
            longs, shorts = [], list(sig_df.tail(n)["symbol"])
            pos_scale     = 1.0
        elif regime == "BULL_TREND":
            longs, shorts = list(sig_df.head(n)["symbol"]), []
            pos_scale     = 0.5
        elif regime == "HIGH_VOL_LATERAL":
            longs, shorts = [], list(sig_df.tail(n)["symbol"])
            pos_scale     = 0.5
        else:  # LOW_VOL_GRIND
            longs  = list(sig_df.head(n)["symbol"])
            shorts = list(sig_df.tail(n)["symbol"])
            pos_scale = 1.0

        n_pos = len(longs) + len(shorts)
        if n_pos == 0:
            return {"status": "no_positions", "regime": regime}

        size_each = (balance / n_pos) * pos_scale  # USD notional per position

        print(f"  Regime={regime} | {len(longs)}L/{len(shorts)}S | ${size_each:.2f}/position")

        # ── Step 5: open new positions + record cycle ─────────────────────────
        new_positions = []

        with self._conn() as db:
            cycle_id = db.execute(
                "INSERT INTO cycles (started_at, regime, n_longs, n_shorts) VALUES (?,?,?,?)",
                (now.isoformat(), regime, len(longs), len(shorts)),
            ).lastrowid
            db.commit()

            for sym, side in [(s, "LONG") for s in longs] + [(s, "SHORT") for s in shorts]:
                price = float(symbol_data[sym]["close"].iloc[-1]) if sym in symbol_data else None
                if price is None:
                    continue
                score    = pred_by_symbol.get(sym, 0.5)
                net_size = size_each * (1 - ROUND_TRIP_FEE / 2)  # deduct half fee upfront

                tid = db.execute(
                    "INSERT INTO trades "
                    "(cycle_id, symbol, side, entry_price, size_usd, status, opened_at, feature_score)"
                    " VALUES (?,?,?,?,?,'open',?,?)",
                    (cycle_id, sym, side, price, round(net_size, 4), now.isoformat(), round(score, 6)),
                ).lastrowid
                new_positions.append({
                    "id":          tid,
                    "symbol":      sym,
                    "side":        side,
                    "entry_price": price,
                    "size_usd":    net_size,
                })

            db.commit()

        self._set("open_positions", new_positions)
        self._set("cycle_count", cycle_count + 1)

        ic = self._compute_ic(pred_by_symbol, actual_rets_for_ic)
        with self._conn() as db:
            db.execute("UPDATE cycles SET ic=? WHERE id=?", (round(ic, 5), cycle_id))
            db.commit()

        cum_ret = (balance / self.initial_balance - 1) * 100
        print(
            f"  Cycle {cycle_count+1} complete | "
            f"balance=${balance:.2f} | cum={cum_ret:+.1f}% | IC={ic:+.4f}"
        )

        return {
            "status":          "ok",
            "cycle":           cycle_count + 1,
            "regime":          regime,
            "longs":           longs,
            "shorts":          shorts,
            "balance":         round(balance, 2),
            "cum_return_pct":  round(cum_ret, 2),
            "ic":              round(ic, 4),
        }

    # ── Dashboard data ────────────────────────────────────────────────────────

    def get_dashboard_stats(self) -> dict:
        balance        = self._get("balance") or self.initial_balance
        cycle_count    = self._get("cycle_count") or 0
        open_positions = self._get("open_positions") or []
        
        # --- Fetch live prices for real-time tracking ---
        live_equity = balance
        updated_positions = []
        if open_positions:
            symbols = [p["symbol"] for p in open_positions]
            try:
                current_prices = get_current_prices(symbols)
                for pos in open_positions:
                    p = dict(pos)
                    curr = current_prices.get(p["symbol"])
                    if curr:
                        p["current_price"] = curr
                        entry = p["entry_price"]
                        ret = (curr - entry) / entry
                        if p["side"] == "SHORT":
                            ret = -ret
                        p["unrealized_pnl_usd"] = p["size_usd"] * ret
                        p["unrealized_pnl_pct"] = ret * 100
                        live_equity += p["unrealized_pnl_usd"]
                    updated_positions.append(p)
            except Exception as e:
                print(f"  Live tracking error: {e}")
                updated_positions = open_positions
        else:
            updated_positions = []

        with self._conn() as db:
            cycles = [
                dict(r) for r in db.execute(
                    "SELECT * FROM cycles ORDER BY id DESC LIMIT 12"
                ).fetchall()
            ]
            trades = [
                dict(r) for r in db.execute(
                    "SELECT * FROM trades ORDER BY id DESC LIMIT 40"
                ).fetchall()
            ]
            total_closed_trades = db.execute(
                "SELECT COUNT(*) FROM trades WHERE status='closed'"
            ).fetchone()[0]
            win_trades = db.execute(
                "SELECT COUNT(*) FROM trades WHERE status='closed' AND pnl_pct>0"
            ).fetchone()[0]

        cum_return    = (live_equity / self.initial_balance - 1) * 100
        max_dd        = self._compute_drawdown()
        win_rate      = (win_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0.0

        return {
            "balance":              round(balance, 2),
            "live_equity":          round(live_equity, 2),
            "initial_balance":      self.initial_balance,
            "cum_return_pct":       round(cum_return, 2),
            "max_drawdown_pct":     round(max_dd * 100, 2),
            "cycle_count":          cycle_count,
            "win_rate_pct":         round(win_rate, 1),
            "total_closed_trades":  total_closed_trades,
            "open_positions":       updated_positions,
            "recent_cycles":        cycles,
            "recent_trades":        trades,
            "timestamp":            datetime.now(timezone.utc).isoformat(),
        }

