"""
Binance live data fetcher — 5-min OHLCV for Azalyst paper trader.
No API key required for public endpoints.
"""
import time
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

BINANCE_BASE = "https://api.binance.com"
REQUEST_SLEEP = 0.25   # seconds between requests (well under rate limit)
MAX_RETRIES   = 3

# Stablecoins and non-crypto to exclude from universe
STABLE_BASES = {
    "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI", "FRAX",
    "EUR", "AEUR", "EURI", "UST", "USTC", "XUSD", "USD1",
    "USDE", "RLUSD", "BFUSD", "PYUSD",
}


def _get(endpoint: str, params: dict = None) -> list | dict:
    url = f"{BINANCE_BASE}{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                print("  Rate limited — sleeping 60s")
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def get_top_symbols(n: int = 55) -> List[str]:
    """Top N USDT pairs by 24h quote volume, excluding stablecoins."""
    data = _get("/api/v3/ticker/24hr")
    pairs = [
        x for x in data
        if isinstance(x, dict)
        and x["symbol"].endswith("USDT")
        and x["symbol"][:-4] not in STABLE_BASES
        and float(x.get("quoteVolume", 0)) > 5_000_000
        and float(x.get("lastPrice", 0)) > 0
    ]
    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [x["symbol"] for x in pairs[:n]]


def get_klines(symbol: str, interval: str = "5m", limit: int = 1500) -> pd.DataFrame:
    """
    Fetch up to `limit` 5-min OHLCV bars for symbol.
    Binance max per request = 1000; fetches in two batches if needed.
    """
    all_rows = []
    end_time  = None
    remaining = limit

    while remaining > 0:
        batch  = min(remaining, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if end_time is not None:
            params["endTime"] = end_time

        data = _get("/api/v3/klines", params)
        if not data:
            break

        all_rows = list(data) + all_rows   # prepend (older data)
        remaining -= len(data)

        if len(data) < batch:
            break                          # Binance returned fewer bars than asked — no more history
        end_time = int(data[0][0]) - 1     # move window back
        time.sleep(REQUEST_SLEEP)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df.sort_index()


def get_current_price(symbol: str) -> Optional[float]:
    """Latest trade price for a symbol."""
    try:
        data = _get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])
    except Exception:
        return None


def fetch_universe_data(symbols: List[str], limit: int = 1500) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for all symbols. Returns {symbol: DataFrame}.
    Skips symbols with < 500 bars (not enough for feature computation).
    """
    result = {}
    for i, sym in enumerate(symbols):
        try:
            df = get_klines(sym, limit=limit)
            if len(df) >= 500:
                result[sym] = df
            time.sleep(REQUEST_SLEEP)
        except Exception as e:
            print(f"  WARN [{sym}]: {e}")
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(symbols)}] symbols fetched ({len(result)} ok)")
    return result


def get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Batch fetch current prices for a list of symbols."""
    prices = {}
    for sym in symbols:
        p = get_current_price(sym)
        if p is not None:
            prices[sym] = p
        time.sleep(REQUEST_SLEEP)
    return prices
