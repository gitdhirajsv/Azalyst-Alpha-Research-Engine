"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AZALYST v7 — PAPER TRADE RUNNER (live Binance data, no real orders) ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  What this does:                                                             ║
║    1. Fetches live 5-min klines for the training universe via Binance REST  ║
║    2. Rebuilds the v7 features (same code path as backtest)                 ║
║    3. Loads the latest trained XGBoost model + scaler                        ║
║    4. Detects regime, predicts, applies regime-gated top-N portfolio         ║
║    5. Simulates fills at last close + slippage, tracks positions and PnL     ║
║    6. Marks-to-market every run, rebalances weekly (default Mon 00:00 UTC)   ║
║                                                                              ║
║  NO REAL ORDERS ARE PLACED. Paper mode only. Use as the validation stage    ║
║  before wiring the same signals into a live Binance futures/spot bot.       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:
    python azalyst_paper_trade.py --once
    python azalyst_paper_trade.py --loop 3600    # refresh every 1 hour
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Reuse the exact factor + regime code from the backtest engine
from azalyst_factors_v2 import build_features
from azalyst_v5_engine import detect_regime

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results_v7"
MODELS_DIR = RESULTS_DIR / "models"
STATE_DIR = ROOT / "paper_trade_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

POSITIONS_FILE = STATE_DIR / "positions.json"
TRADE_LOG_FILE = STATE_DIR / "trade_log.csv"
EQUITY_CURVE_FILE = STATE_DIR / "equity_curve.csv"
RUN_LOG_FILE = STATE_DIR / "run_log.txt"

BINANCE_BASE = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
TICKER_PATH = "/api/v3/ticker/price"
EXCHINFO_PATH = "/api/v3/exchangeInfo"

# Klines: 1000 bars per call. 5-min × 2000 bars ≈ 6.9 days — enough for weekly features.
BARS_PER_FETCH = 1000
BARS_TO_PULL = 2000       # two REST pages → ~7 days
INTERVAL = "5m"

REBALANCE_INTERVAL_DAYS = 7          # match backtest cadence
DEFAULT_TOP_N = 5
DEFAULT_LEVERAGE = 0.5
STARTING_EQUITY = 10_000.0
ROUND_TRIP_FEE = 0.002               # 0.2% round-trip, matches backtest
SLIPPAGE_PCT = 0.0005                # 5 bps each side (conservative)
MAX_DRAWDOWN_KILL = -0.20
KILL_RECOVERY = -0.12

REQUEST_TIMEOUT = 15
RATE_LIMIT_SLEEP = 0.1               # 10 req/s — well under Binance weight limits

BLACKLIST = {
    "FTTUSDT", "EURUSDT", "USDCUSDT", "USDPUSDT",
    "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT",
}
FIAT_BASES = {
    "AEUR", "BFUSD", "BUSD", "DAI", "EURI", "EUR", "FDUSD",
    "FRAX", "PYUSD", "RLUSD", "TUSD", "USD1", "USDC", "USDE",
    "USDP", "UST", "USTC", "XUSD",
}


def log(msg: str):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def is_excluded(sym: str) -> bool:
    s = sym.upper().strip()
    if s in BLACKLIST:
        return True
    if s.endswith("USDT") and s[:-4] in FIAT_BASES:
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# BINANCE REST HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get(path: str, params: Optional[dict] = None) -> dict | list:
    for attempt in range(4):
        try:
            r = requests.get(BINANCE_BASE + path, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429 or r.status_code == 418:
                wait = 2 ** attempt
                log(f"  rate-limited ({r.status_code}) — sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {path}")


def fetch_exchange_symbols() -> set:
    """Return all TRADING USDT symbols currently live on Binance spot."""
    info = _get(EXCHINFO_PATH)
    syms = set()
    for s in info.get("symbols", []):
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
            syms.add(s["symbol"])
    return syms


def fetch_klines(symbol: str, bars: int = BARS_TO_PULL) -> Optional[pd.DataFrame]:
    """Fetch N bars of 5-min klines. Returns OHLCV DataFrame indexed UTC."""
    frames = []
    remaining = bars
    end_time = None
    while remaining > 0:
        chunk = min(remaining, BARS_PER_FETCH)
        params = {"symbol": symbol, "interval": INTERVAL, "limit": chunk}
        if end_time is not None:
            params["endTime"] = end_time
        try:
            data = _get(KLINES_PATH, params)
        except Exception as e:
            log(f"    klines fail {symbol}: {e}")
            return None
        if not data:
            break
        frames.append(data)
        # Walk backwards: set next endTime to just before first bar we got
        end_time = data[0][0] - 1
        remaining -= len(data)
        time.sleep(RATE_LIMIT_SLEEP)
        if len(data) < chunk:
            break

    if not frames:
        return None

    rows = []
    for chunk in frames:
        rows.extend(chunk)
    rows.sort(key=lambda r: r[0])

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_all_prices() -> Dict[str, float]:
    """One REST call → dict of symbol → last price."""
    data = _get(TICKER_PATH)
    return {row["symbol"]: float(row["price"]) for row in data}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_model() -> Tuple[Path, Path, List[str]]:
    """Pick the most recent weekly model + its feature list."""
    model_files = sorted(MODELS_DIR.glob("model_v7_week*.pkl"))
    if not model_files:
        base = MODELS_DIR / "model_v7_base.pkl"
        scaler = MODELS_DIR / "scaler_v7_base.pkl"
        if not base.exists():
            raise FileNotFoundError(f"No model found in {MODELS_DIR}")
        # Base model — use feature_importance_v7_base.csv
        imp_csv = RESULTS_DIR / "feature_importance_v7_base.csv"
    else:
        model_path = model_files[-1]
        week_tag = model_path.stem.split("_")[-1]  # "week072"
        scaler = MODELS_DIR / f"scaler_v7_{week_tag}.pkl"
        imp_csv = RESULTS_DIR / f"feature_importance_v7_{week_tag}.csv"
        base = model_path

    if not scaler.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler}")
    if not imp_csv.exists():
        raise FileNotFoundError(f"Feature list not found: {imp_csv}")

    imp = pd.read_csv(imp_csv, index_col=0)
    features = imp.index.tolist()
    return base, scaler, features


def load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "equity": STARTING_EQUITY,
        "peak_equity": STARTING_EQUITY,
        "paused": False,
        "last_rebalance": None,
        "positions": {},   # symbol → {side, entry_price, size_usd, scale, opened_at}
        "history": [],
    }


def save_state(state: dict):
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def append_equity_row(row: dict):
    exists = EQUITY_CURVE_FILE.exists()
    df = pd.DataFrame([row])
    df.to_csv(EQUITY_CURVE_FILE, mode="a", header=not exists, index=False)


def append_trade_rows(rows: List[dict]):
    if not rows:
        return
    exists = TRADE_LOG_FILE.exists()
    pd.DataFrame(rows).to_csv(TRADE_LOG_FILE, mode="a",
                              header=not exists, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# CORE CYCLE
# ──────────────────────────────────────────────────────────────────────────────

def get_universe() -> List[str]:
    """Universe = training parquets ∩ Binance live trading symbols."""
    parquets = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    training_universe = [Path(p).stem for p in parquets]
    live = fetch_exchange_symbols()
    universe = [s for s in training_universe
                if s in live and not is_excluded(s)]
    return universe


def build_live_features(universe: List[str],
                        features_used: List[str]) -> Tuple[Dict[str, pd.DataFrame],
                                                           Dict[str, pd.DataFrame]]:
    """Fetch klines + compute features for every symbol. Returns (features_df, ohlc)."""
    feat_store: Dict[str, pd.DataFrame] = {}
    ohlc_store: Dict[str, pd.DataFrame] = {}
    n = len(universe)
    for i, sym in enumerate(universe, 1):
        if i % 25 == 0:
            log(f"  fetched {i}/{n} symbols")
        df = fetch_klines(sym)
        if df is None or len(df) < 500:
            continue
        try:
            feats = build_features(df, timeframe="5min")
        except Exception as e:
            log(f"    build_features fail {sym}: {e}")
            continue
        # Scaler expects exactly features_used in order. Skip symbols missing any.
        missing = [c for c in features_used if c not in feats.columns]
        if missing:
            log(f"    skip {sym}: missing features {missing}")
            continue
        merged = feats[features_used].copy()
        merged["close"] = df["close"]
        # rvol_1d used for position sizing — include if present
        if "rvol_1d" in feats.columns and "rvol_1d" not in merged.columns:
            merged["rvol_1d"] = feats["rvol_1d"]
        feat_store[sym] = merged.dropna(how="all")
        ohlc_store[sym] = df
    return feat_store, ohlc_store


def predict_universe(model, scaler, feat_store: Dict[str, pd.DataFrame],
                     features_used: List[str]) -> Dict[str, float]:
    """Score each symbol with the latest snapshot (last 12 bars = 1hr)."""
    preds: Dict[str, float] = {}
    for sym, df in feat_store.items():
        snap = df[features_used].tail(12)
        if len(snap) < 1:
            continue
        X = snap.values.astype(np.float32)
        valid = np.isfinite(X).all(axis=1)
        if valid.sum() < 1:
            continue
        X_scaled = scaler.transform(X[valid])
        try:
            y = model.predict(X_scaled)
        except Exception as e:
            log(f"    predict fail {sym}: {e}")
            continue
        preds[sym] = float(np.mean(y))
    return preds


def select_portfolio(preds: Dict[str, float], regime: str,
                     top_n: int, leverage: float,
                     symbol_rvol: Dict[str, float]) -> Tuple[List[Tuple[str, str, float]], float]:
    """Regime-gated top-N per side. Returns (list of (sym, side, position_scale), base_scale)."""
    if not preds:
        return [], 0.0
    pred_s = pd.Series(preds).sort_values(ascending=False)
    n = min(top_n, len(pred_s) // 2)
    if n < 1:
        return [], 0.0

    if regime == "BEAR_TREND":
        longs, shorts = [], list(pred_s.tail(n).index)
        base_scale = 1.0 * leverage
    elif regime == "BULL_TREND":
        longs = [s for s in pred_s.index if pred_s[s] > 0][:n]
        shorts = []
        base_scale = 0.5 * leverage
    elif regime == "HIGH_VOL_LATERAL":
        longs, shorts = [], list(pred_s.tail(n).index)
        base_scale = 0.5 * leverage
    else:  # LOW_VOL_GRIND
        longs = [s for s in pred_s.index if pred_s[s] > 0][:n]
        shorts = list(pred_s.tail(n).index)
        base_scale = 1.0 * leverage

    out = []
    for sym in longs:
        rvol = symbol_rvol.get(sym, 1.0) or 1.0
        scale = min(base_scale / max(rvol, 0.01), 1.0)
        out.append((sym, "LONG", scale))
    for sym in shorts:
        out.append((sym, "SHORT", min(base_scale, 1.0)))
    return out, base_scale


def mark_to_market(state: dict, prices: Dict[str, float]) -> float:
    """Update unrealised PnL of open positions. Returns total unrealised $."""
    unrealised = 0.0
    for sym, pos in state["positions"].items():
        mark = prices.get(sym)
        if mark is None or mark <= 0:
            continue
        entry = pos["entry_price"]
        notional = pos["size_usd"]
        if pos["side"] == "LONG":
            pnl = notional * (mark / entry - 1.0)
        else:
            pnl = notional * (1.0 - mark / entry)
        pos["mark_price"] = mark
        pos["unrealised_pnl"] = round(pnl, 2)
        unrealised += pnl
    return unrealised


def close_all(state: dict, prices: Dict[str, float], reason: str,
              ts: str) -> List[dict]:
    """Close all positions, realise PnL, log trades. Returns trade rows."""
    rows = []
    for sym, pos in list(state["positions"].items()):
        mark = prices.get(sym)
        if mark is None or mark <= 0:
            # Can't price — close at entry, zero PnL (conservative)
            mark = pos["entry_price"]
        entry = pos["entry_price"]
        notional = pos["size_usd"]
        if pos["side"] == "LONG":
            gross = notional * (mark / entry - 1.0)
        else:
            gross = notional * (1.0 - mark / entry)
        fee = notional * (ROUND_TRIP_FEE / 2)   # exit half of round-trip
        slip = notional * SLIPPAGE_PCT
        net = gross - fee - slip
        state["equity"] += net
        rows.append({
            "timestamp": ts, "symbol": sym, "action": "CLOSE", "side": pos["side"],
            "entry_price": entry, "exit_price": mark, "size_usd": notional,
            "scale": pos["scale"], "gross_pnl": round(gross, 2),
            "fees_slippage": round(fee + slip, 2), "net_pnl": round(net, 2),
            "reason": reason, "equity_after": round(state["equity"], 2),
        })
        del state["positions"][sym]
    return rows


def open_positions(state: dict, picks: List[Tuple[str, str, float]],
                   prices: Dict[str, float], ts: str) -> List[dict]:
    """Enter new positions. Equal-weight allocation within long + short sleeves."""
    rows = []
    if not picks:
        return rows
    # Allocate equity equally across picks (scaled by position_scale)
    total_scale = sum(p[2] for p in picks)
    if total_scale <= 0:
        return rows
    for sym, side, scale in picks:
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        alloc_usd = state["equity"] * (scale / max(total_scale, 1.0))
        if alloc_usd <= 1.0:
            continue
        fee = alloc_usd * (ROUND_TRIP_FEE / 2)
        slip = alloc_usd * SLIPPAGE_PCT
        state["equity"] -= (fee + slip)
        state["positions"][sym] = {
            "side": side,
            "entry_price": price,
            "size_usd": round(alloc_usd, 2),
            "scale": round(scale, 4),
            "opened_at": ts,
        }
        rows.append({
            "timestamp": ts, "symbol": sym, "action": "OPEN", "side": side,
            "entry_price": price, "exit_price": "", "size_usd": round(alloc_usd, 2),
            "scale": round(scale, 4), "gross_pnl": 0.0,
            "fees_slippage": round(fee + slip, 2), "net_pnl": 0.0,
            "reason": "rebalance", "equity_after": round(state["equity"], 2),
        })
    return rows


def compute_symbol_rvol(feat_store: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    out = {}
    for sym, df in feat_store.items():
        if "rvol_1d" in df.columns:
            v = df["rvol_1d"].dropna()
            if len(v) > 0:
                out[sym] = float(v.iloc[-1])
    return out


def regime_proxy_store(ohlc_store: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """detect_regime expects a symbols dict with DataFrames that have 'close'."""
    return {s: df for s, df in ohlc_store.items() if s in ("BTCUSDT", "ETHUSDT")}


# ──────────────────────────────────────────────────────────────────────────────
# ONE CYCLE
# ──────────────────────────────────────────────────────────────────────────────

def run_cycle(top_n: int, leverage: float, force_rebalance: bool):
    ts = datetime.now(timezone.utc).isoformat()
    log("=" * 78)
    log(f"PAPER TRADE CYCLE  ts={ts}  top_n={top_n}  lev={leverage}")
    log("=" * 78)

    state = load_state()

    # 1. Fast path: mark-to-market only if not time to rebalance
    prices = fetch_all_prices()
    log(f"  fetched {len(prices)} ticker prices")

    need_rebalance = force_rebalance
    if state["last_rebalance"]:
        last = datetime.fromisoformat(state["last_rebalance"])
        age = datetime.now(timezone.utc) - last
        if age >= timedelta(days=REBALANCE_INTERVAL_DAYS):
            need_rebalance = True
            log(f"  last rebalance {age} ago → rebalance")
        else:
            log(f"  last rebalance {age} ago → mark-to-market only")
    else:
        need_rebalance = True
        log(f"  no prior rebalance → initial rebalance")

    unreal = mark_to_market(state, prices)
    total_equity = state["equity"] + unreal
    drawdown = (total_equity / state["peak_equity"] - 1.0) if state["peak_equity"] > 0 else 0.0
    state["peak_equity"] = max(state["peak_equity"], total_equity)

    log(f"  realised equity=${state['equity']:.2f}  "
        f"unrealised=${unreal:.2f}  total=${total_equity:.2f}  dd={drawdown:+.2%}")

    # 2. Kill switch
    if drawdown <= MAX_DRAWDOWN_KILL and not state["paused"]:
        log(f"  !! KILL SWITCH TRIPPED — drawdown {drawdown:.2%} ≤ {MAX_DRAWDOWN_KILL:.0%}")
        state["paused"] = True
        need_rebalance = False
    elif state["paused"] and drawdown >= KILL_RECOVERY:
        log(f"  kill switch released — drawdown recovered to {drawdown:.2%}")
        state["paused"] = False

    if state["paused"]:
        need_rebalance = False
        log("  state=PAUSED — no new trades this cycle")

    # 3. Rebalance pipeline
    if need_rebalance:
        try:
            model_path, scaler_path, features_used = find_latest_model()
            log(f"  model: {model_path.name}  scaler: {scaler_path.name}  "
                f"features={len(features_used)}")
            model = load_pickle(model_path)
            scaler = load_pickle(scaler_path)

            log("  building live universe...")
            universe = get_universe()
            log(f"  universe: {len(universe)} symbols (training ∩ live Binance)")

            log("  fetching klines + building features...")
            feat_store, ohlc_store = build_live_features(universe, features_used)
            log(f"  features built for {len(feat_store)} symbols")

            preds = predict_universe(model, scaler, feat_store, features_used)
            log(f"  predictions: {len(preds)} symbols")

            regime_store = regime_proxy_store(ohlc_store)
            if regime_store:
                week_end = datetime.now(timezone.utc)
                regime = detect_regime(regime_store, week_end)
            else:
                regime = "LOW_VOL_GRIND"
            log(f"  regime: {regime}")

            rvol = compute_symbol_rvol(feat_store)
            picks, base_scale = select_portfolio(preds, regime, top_n, leverage, rvol)
            log(f"  picks: {len(picks)} ({sum(1 for p in picks if p[1]=='LONG')}L / "
                f"{sum(1 for p in picks if p[1]=='SHORT')}S)  base_scale={base_scale:.3f}")

            rows = close_all(state, prices, "rebalance", ts)
            append_trade_rows(rows)
            rows = open_positions(state, picks, prices, ts)
            append_trade_rows(rows)

            state["last_rebalance"] = ts
            state["last_regime"] = regime
            state["last_pick_count"] = len(picks)
        except Exception as e:
            log(f"  REBALANCE FAILED: {e}")
            log(traceback.format_exc())

    # 4. Persist and log equity row
    unreal = mark_to_market(state, prices)
    total_equity = state["equity"] + unreal
    append_equity_row({
        "timestamp": ts,
        "realised_equity": round(state["equity"], 2),
        "unrealised_pnl": round(unreal, 2),
        "total_equity": round(total_equity, 2),
        "drawdown_pct": round(drawdown * 100, 3),
        "open_positions": len(state["positions"]),
        "paused": state["paused"],
        "regime": state.get("last_regime", ""),
    })
    state["last_cycle"] = ts
    save_state(state)

    log(f"  SAVED state=({state['equity']:.2f} real, {unreal:.2f} unreal) "
        f"positions={len(state['positions'])}")
    log("=" * 78)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Azalyst v7 paper-trade runner")
    ap.add_argument("--once", action="store_true", help="run a single cycle then exit")
    ap.add_argument("--loop", type=int, default=0,
                    help="run continuously, sleeping N seconds between cycles")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--leverage", type=float, default=DEFAULT_LEVERAGE)
    ap.add_argument("--force-rebalance", action="store_true",
                    help="force a rebalance even if interval not reached")
    args = ap.parse_args()

    if not args.once and args.loop <= 0:
        args.once = True

    if args.once:
        run_cycle(args.top_n, args.leverage, args.force_rebalance)
        return

    log(f"LOOP MODE: every {args.loop}s")
    while True:
        try:
            run_cycle(args.top_n, args.leverage, args.force_rebalance)
        except KeyboardInterrupt:
            log("interrupted — exiting loop")
            break
        except Exception as e:
            log(f"cycle crashed: {e}")
            log(traceback.format_exc())
        args.force_rebalance = False  # only honour flag on first cycle
        log(f"sleeping {args.loop}s...")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
