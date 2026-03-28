"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AZALYST ALPHA RESEARCH ENGINE — SQLite Persistence Layer           ║
║        Zero-server DB  |  Trades · Metrics · Models · SHAP · Signals       ║
║        v4.0  |  Local-only  |  No external dependencies                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_DB_PATH_DEFAULT = "results/azalyst.db"
_thread_local = threading.local()


class AzalystDB:
    """
    SQLite-backed persistence for all Azalyst pipeline outputs.

    Usage:
        db = AzalystDB("results/azalyst.db")
        db.insert_trade(...)
        db.insert_weekly_metric(...)
        df = db.get_trades(run_id="v4_20260328")
    """

    def __init__(self, db_path: str = _DB_PATH_DEFAULT):
        self.db_path = str(Path(db_path).resolve())
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        """Thread-safe connection context manager."""
        conn = getattr(_thread_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            _thread_local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT PRIMARY KEY,
                    started_at   TEXT NOT NULL,
                    finished_at  TEXT,
                    config_json  TEXT,
                    status       TEXT DEFAULT 'running'
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    week         INTEGER NOT NULL,
                    week_start   TEXT,
                    symbol       TEXT NOT NULL,
                    signal       TEXT NOT NULL,
                    pred_prob    REAL,
                    pnl_pct      REAL,
                    raw_ret_pct  REAL,
                    meta_size    REAL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS weekly_metrics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT NOT NULL,
                    week            INTEGER NOT NULL,
                    week_start      TEXT,
                    week_end        TEXT,
                    n_symbols       INTEGER,
                    n_trades        INTEGER,
                    week_return_pct REAL,
                    annualised_pct  REAL,
                    ic              REAL,
                    turnover_pct    REAL,
                    on_track        INTEGER,
                    cum_return_pct  REAL,
                    max_drawdown_pct REAL,
                    regime          TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    label        TEXT NOT NULL,
                    week         INTEGER,
                    model_path   TEXT,
                    scaler_path  TEXT,
                    auc          REAL,
                    ic           REAL,
                    icir         REAL,
                    n_features   INTEGER,
                    created_at   TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS shap_values (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    label        TEXT NOT NULL,
                    feature      TEXT NOT NULL,
                    mean_abs_shap REAL,
                    rank         INTEGER,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS feature_ic (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    week         INTEGER NOT NULL,
                    feature      TEXT NOT NULL,
                    ic           REAL,
                    selected     INTEGER DEFAULT 1,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS performance_summary (
                    run_id              TEXT PRIMARY KEY,
                    total_weeks         INTEGER,
                    total_trades        INTEGER,
                    retrains            INTEGER,
                    total_return_pct    REAL,
                    annualised_pct      REAL,
                    sharpe              REAL,
                    max_drawdown_pct    REAL,
                    ic_mean             REAL,
                    icir                REAL,
                    ic_positive_pct     REAL,
                    var_95              REAL,
                    cvar_95             REAL,
                    kill_switch_hit     INTEGER DEFAULT 0,
                    config_json         TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
                CREATE INDEX IF NOT EXISTS idx_trades_week ON trades(run_id, week);
                CREATE INDEX IF NOT EXISTS idx_weekly_run ON weekly_metrics(run_id);
                CREATE INDEX IF NOT EXISTS idx_shap_run ON shap_values(run_id);
                CREATE INDEX IF NOT EXISTS idx_fic_run ON feature_ic(run_id, week);
            """)

    # ── Run management ────────────────────────────────────────────────────────

    def start_run(self, run_id: str, config: Optional[Dict] = None) -> str:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, started_at, config_json, status) "
                "VALUES (?, ?, ?, 'running')",
                (run_id, datetime.utcnow().isoformat(), json.dumps(config or {}))
            )
        return run_id

    def finish_run(self, run_id: str, status: str = "completed"):
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET finished_at=?, status=? WHERE run_id=?",
                (datetime.utcnow().isoformat(), status, run_id)
            )

    # ── Trades ────────────────────────────────────────────────────────────────

    def insert_trades(self, run_id: str, trades: List[Dict]):
        if not trades:
            return
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO trades (run_id, week, week_start, symbol, signal, "
                "pred_prob, pnl_pct, raw_ret_pct, meta_size) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, t.get("week", 0), t.get("week_start", ""),
                  t["symbol"], t["signal"], t.get("pred_prob"),
                  t.get("pnl_percent"), t.get("raw_ret_pct"),
                  t.get("meta_size")) for t in trades]
            )

    def get_trades(self, run_id: str, week: Optional[int] = None) -> pd.DataFrame:
        q = "SELECT * FROM trades WHERE run_id=?"
        params: list = [run_id]
        if week is not None:
            q += " AND week=?"
            params.append(week)
        with self._conn() as conn:
            return pd.read_sql_query(q, conn, params=params)

    # ── Weekly metrics ────────────────────────────────────────────────────────

    def insert_weekly_metric(self, run_id: str, metric: Dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO weekly_metrics (run_id, week, week_start, week_end, "
                "n_symbols, n_trades, week_return_pct, annualised_pct, ic, "
                "turnover_pct, on_track, cum_return_pct, max_drawdown_pct, regime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, metric.get("week"), metric.get("week_start"),
                 metric.get("week_end"), metric.get("n_symbols"),
                 metric.get("n_trades"), metric.get("week_return_pct"),
                 metric.get("annualised_pct"), metric.get("ic"),
                 metric.get("turnover_pct"), int(metric.get("on_track", False)),
                 metric.get("cum_return_pct"), metric.get("max_drawdown_pct"),
                 metric.get("regime"))
            )

    def get_weekly_metrics(self, run_id: str) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM weekly_metrics WHERE run_id=? ORDER BY week",
                conn, params=[run_id]
            )

    # ── Model artifacts ───────────────────────────────────────────────────────

    def insert_model_artifact(self, run_id: str, label: str, week: int = 0,
                              model_path: str = "", scaler_path: str = "",
                              auc: float = 0.0, ic: float = 0.0,
                              icir: float = 0.0, n_features: int = 56):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO model_artifacts (run_id, label, week, model_path, "
                "scaler_path, auc, ic, icir, n_features, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, label, week, model_path, scaler_path,
                 auc, ic, icir, n_features, datetime.utcnow().isoformat())
            )

    # ── SHAP values ───────────────────────────────────────────────────────────

    def insert_shap_values(self, run_id: str, label: str,
                           shap_importance: Dict[str, float]):
        ranked = sorted(shap_importance.items(), key=lambda x: -abs(x[1]))
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO shap_values (run_id, label, feature, mean_abs_shap, rank) "
                "VALUES (?, ?, ?, ?, ?)",
                [(run_id, label, feat, val, i + 1)
                 for i, (feat, val) in enumerate(ranked)]
            )

    def get_shap_values(self, run_id: str, label: str = "") -> pd.DataFrame:
        q = "SELECT * FROM shap_values WHERE run_id=?"
        params: list = [run_id]
        if label:
            q += " AND label=?"
            params.append(label)
        q += " ORDER BY rank"
        with self._conn() as conn:
            return pd.read_sql_query(q, conn, params=params)

    # ── Feature IC tracking ───────────────────────────────────────────────────

    def insert_feature_ic(self, run_id: str, week: int,
                          feature_ics: Dict[str, float],
                          selected_features: Optional[List[str]] = None):
        selected = set(selected_features or feature_ics.keys())
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO feature_ic (run_id, week, feature, ic, selected) "
                "VALUES (?, ?, ?, ?, ?)",
                [(run_id, week, feat, ic_val, int(feat in selected))
                 for feat, ic_val in feature_ics.items()]
            )

    def get_feature_ic(self, run_id: str,
                       last_n_weeks: Optional[int] = None) -> pd.DataFrame:
        q = "SELECT * FROM feature_ic WHERE run_id=?"
        params: list = [run_id]
        if last_n_weeks is not None:
            q += " AND week >= (SELECT MAX(week) FROM feature_ic WHERE run_id=?) - ?"
            params.extend([run_id, last_n_weeks])
        with self._conn() as conn:
            return pd.read_sql_query(q, conn, params=params)

    # ── Performance summary ───────────────────────────────────────────────────

    def insert_performance_summary(self, run_id: str, perf: Dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO performance_summary "
                "(run_id, total_weeks, total_trades, retrains, total_return_pct, "
                "annualised_pct, sharpe, max_drawdown_pct, ic_mean, icir, "
                "ic_positive_pct, var_95, cvar_95, kill_switch_hit, config_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, perf.get("total_weeks"), perf.get("total_trades"),
                 perf.get("retrains"), perf.get("total_return_pct"),
                 perf.get("annualised_pct"), perf.get("sharpe"),
                 perf.get("max_drawdown_pct"), perf.get("ic_mean"),
                 perf.get("icir"), perf.get("ic_positive_pct"),
                 perf.get("var_95"), perf.get("cvar_95"),
                 int(perf.get("kill_switch_hit", False)),
                 json.dumps(perf))
            )

    def get_performance_summary(self, run_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM performance_summary WHERE run_id=?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_runs(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT r.*, p.total_return_pct, p.sharpe, p.max_drawdown_pct "
                "FROM runs r LEFT JOIN performance_summary p ON r.run_id=p.run_id "
                "ORDER BY r.started_at DESC", conn
            )

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        placeholders = ",".join(["?"] * len(run_ids))
        with self._conn() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM performance_summary WHERE run_id IN ({placeholders})",
                conn, params=run_ids
            )

    def close(self):
        conn = getattr(_thread_local, "conn", None)
        if conn is not None:
            conn.close()
            _thread_local.conn = None


if __name__ == "__main__":
    db = AzalystDB("results/azalyst_test.db")
    rid = db.start_run("test_run_001", {"features": 56, "gpu": True})
    db.insert_trades(rid, [
        {"week": 1, "symbol": "BTCUSDT", "signal": "BUY",
         "pred_prob": 0.65, "pnl_percent": 1.23, "meta_size": 0.8},
    ])
    db.insert_weekly_metric(rid, {
        "week": 1, "week_start": "2025-03-10", "week_end": "2025-03-17",
        "n_symbols": 300, "n_trades": 90, "week_return_pct": 0.45,
        "ic": 0.03, "cum_return_pct": 0.45, "max_drawdown_pct": -0.1,
    })
    db.insert_shap_values(rid, "base", {"ret_4h": 0.05, "rsi_14": 0.03})
    db.finish_run(rid)
    print("DB test OK:", db.list_runs())
    db.close()
    os.remove("results/azalyst_test.db")
    print("Test passed — DB created and cleaned up.")
