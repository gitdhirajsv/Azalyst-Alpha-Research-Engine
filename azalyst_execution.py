"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    EXECUTION & MARKET INFRASTRUCTURE
║        Event-driven LOB · Smart Order Router · VWAP/TWAP/Iceberg           ║
║        v4.0  |  Incremental L2 Order Book  |  Time-Decay Impact Model      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  v4.0 ARCHITECTURE CHANGES (vs v3.0)                                       ║
║  ────────────────────────────────────────────────────────────────────────  ║
║  OrderBook (NEW)                                                           ║
║    • Persistent sorted data structure. In production C++, this would be    ║
║      std::map<Price, Level, std::greater<>> for bids with atomic access    ║
║      per level for lock-free reads from a separate strategy thread.        ║
║    • apply_l2_delta() — O(log n) incremental update, not O(n) rebuild     ║
║    • Correct architecture for tick-by-tick simulation                      ║
║                                                                            ║
║  ImpactModel (UPGRADED)                                                    ║
║    • Time-based exponential decay: impact(t) = I₀ × exp(-ln2 × dt/t½)    ║
║    • Half-life parameterized in seconds (default 300s = 5 min)             ║
║    • Tracks running impact state for a sequence of trades                  ║
║                                                                            ║
║  LOBSimulator (UNCHANGED — snapshot mode for backtesting)                  ║
║  SmartOrderRouter (UNCHANGED)                                              ║
║  ExecutionAlgos (UNCHANGED)                                                ║
║                                                                            ║
║  DESIGN NOTE FOR RENAISSANCE INTERVIEW:                                    ║
║  In production, the matching engine and strategy logic run in separate     ║
║  threads. The LOB update stream is fed via a SPSC (Single-Producer         ║
║  Single-Consumer) lock-free ring buffer. The strategy thread consumes      ║
║  from the ring buffer and issues child orders via a second channel.        ║
║  See EXECUTION_DESIGN.md for the full architectural design document.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# sortedcontainers gives O(log n) sorted dict — correct data structure for LOB
try:
    from sortedcontainers import SortedDict
    _HAS_SORTED = True
except ImportError:
    _HAS_SORTED = False


# ─────────────────────────────────────────────────────────────────────────────
#  EVENT-DRIVEN ORDER BOOK  (v4 — correct architecture)
#
#  Key insight: a real LOB is NOT reconstructed each bar from scratch.
#  It is a persistent data structure that receives incremental L2 updates:
#    - price level added   → insert(price, size)
#    - price level updated → update(price, size)
#    - price level deleted → delete(price)
#
#  In C++ production: bids = std::map<Price, Level, std::greater<Price>>
#                      asks = std::map<Price, Level>
#  Both give O(log n) access. The top of book is O(1): bids.rbegin(), asks.begin()
#
#  In Python we use sortedcontainers.SortedDict with a custom key for bids.
# ─────────────────────────────────────────────────────────────────────────────

class OrderBook:
    """
    Persistent, incrementally-updated Limit Order Book.

    Architecture
    ────────────
    Bids: SortedDict with negated key → highest price first (max at index 0).
    Asks: SortedDict with natural key  → lowest price first (min at index 0).

    Each entry: price (float) → size (float, base currency).

    In production C++ this would be:
        std::map<double, double, std::greater<double>> bids;
        std::map<double, double> asks;
    Both store (price → aggregated_size_at_price).
    All operations O(log n). Top of book O(1).

    Parameters
    ----------
    symbol : str
        Instrument identifier (for logging/tracking).
    """

    def __init__(self, symbol: str = ""):
        self.symbol   = symbol
        self._n_updates = 0

        if _HAS_SORTED:
            # Bids: negate key so highest price sorts first
            self.bids: SortedDict = SortedDict(lambda k: -k)
            self.asks: SortedDict = SortedDict()
        else:
            # Fallback: plain dicts — O(n) for top-of-book but functionally correct
            self.bids: dict = {}
            self.asks: dict = {}

    def apply_l2_delta(self, side: str, price: float, size: float) -> None:
        """
        Apply one L2 (price-level) order book update.

        Parameters
        ----------
        side  : 'bid' or 'ask'
        price : Price level
        size  : New total size at this level. 0 = remove level.

        This is O(log n) for SortedDict, O(1) for plain dict.
        In production this is called millions of times per second.
        """
        book = self.bids if side == "bid" else self.asks

        if size == 0.0:
            # Level cleared — remove
            if _HAS_SORTED:
                book.pop(price, None)
            else:
                book.pop(price, None)
        else:
            book[price] = size

        self._n_updates += 1

    def reset(self, snapshot_bids: Dict[float, float],
              snapshot_asks: Dict[float, float]) -> None:
        """
        Initialize from a full L2 snapshot (used at session start).
        Subsequent updates arrive via apply_l2_delta.
        """
        if _HAS_SORTED:
            self.bids = SortedDict(lambda k: -k, snapshot_bids)
            self.asks = SortedDict(snapshot_asks)
        else:
            self.bids = dict(snapshot_bids)
            self.asks = dict(snapshot_asks)

    def best_bid(self) -> Optional[float]:
        """Best (highest) bid price. O(1)."""
        if not self.bids:
            return None
        if _HAS_SORTED:
            return self.bids.keys()[0]
        return max(self.bids.keys())

    def best_ask(self) -> Optional[float]:
        """Best (lowest) ask price. O(1)."""
        if not self.asks:
            return None
        if _HAS_SORTED:
            return self.asks.keys()[0]
        return min(self.asks.keys())

    def mid_price(self) -> Optional[float]:
        """Mid-price = (best_bid + best_ask) / 2."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread_bps(self) -> Optional[float]:
        """Bid-ask spread in basis points."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None or bb <= 0:
            return None
        return (ba - bb) / bb * 10_000

    def depth(self, n_levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """
        Return top-N price levels for each side.
        Returns {'bids': [(price, size), ...], 'asks': [(price, size), ...]}.
        Useful for visualisation and market impact estimation.
        """
        if _HAS_SORTED:
            bid_levels = list(zip(self.bids.keys()[:n_levels],
                                   self.bids.values()[:n_levels]))
            ask_levels = list(zip(self.asks.keys()[:n_levels],
                                   self.asks.values()[:n_levels]))
        else:
            bid_levels = sorted(self.bids.items(), key=lambda x: -x[0])[:n_levels]
            ask_levels = sorted(self.asks.items())[:n_levels]

        return {"bids": bid_levels, "asks": ask_levels}

    def execute_market_buy(
        self, amount_quote: float
    ) -> Tuple[float, float, int]:
        """
        Simulate market buy by walking up the ask side.

        This is called SWEEP — walks up the ask queue from best ask
        until the order is filled or the book is exhausted.

        Parameters
        ----------
        amount_quote : float
            Amount of quote currency to spend (e.g. USDT).

        Returns
        -------
        (avg_fill_price, filled_quote, levels_consumed)
        """
        if not self.asks:
            return np.nan, 0.0, 0

        if _HAS_SORTED:
            ask_items = list(zip(self.asks.keys(), self.asks.values()))
        else:
            ask_items = sorted(self.asks.items())

        total_base   = 0.0
        total_quote  = 0.0
        remaining    = amount_quote
        levels_hit   = 0

        for price, size in ask_items:
            if remaining <= 0:
                break
            avail_quote  = price * size
            fill_quote   = min(remaining, avail_quote)
            fill_base    = fill_quote / price
            total_base  += fill_base
            total_quote += fill_quote
            remaining   -= fill_quote
            levels_hit  += 1

        if total_base == 0:
            return ask_items[0][0] if ask_items else np.nan, 0.0, 0

        avg_price = total_quote / total_base
        return avg_price, total_quote, levels_hit

    def execute_market_sell(
        self, amount_base: float
    ) -> Tuple[float, float, int]:
        """
        Simulate market sell by walking down the bid side.

        Returns
        -------
        (avg_fill_price, proceeds_quote, levels_consumed)
        """
        if not self.bids:
            return np.nan, 0.0, 0

        if _HAS_SORTED:
            bid_items = list(zip(self.bids.keys(), self.bids.values()))
        else:
            bid_items = sorted(self.bids.items(), key=lambda x: -x[0])

        total_proceeds = 0.0
        remaining_base = amount_base
        levels_hit     = 0

        for price, size in bid_items:
            if remaining_base <= 0:
                break
            fill_base      = min(remaining_base, size)
            total_proceeds += fill_base * price
            remaining_base -= fill_base
            levels_hit     += 1

        if amount_base <= remaining_base:
            return bid_items[0][0] if bid_items else np.nan, 0.0, 0

        filled_base = amount_base - remaining_base
        avg_price   = total_proceeds / filled_base if filled_base > 0 else np.nan
        return avg_price, total_proceeds, levels_hit

    def __repr__(self) -> str:
        bb = self.best_bid()
        ba = self.best_ask()
        sp = self.spread_bps()
        return (f"OrderBook({self.symbol} | "
                f"bid={bb:.4f} ask={ba:.4f} spread={sp:.1f}bps | "
                f"depth_bids={len(self.bids)} depth_asks={len(self.asks)})") \
            if bb and ba else f"OrderBook({self.symbol} | EMPTY)"


# ─────────────────────────────────────────────────────────────────────────────
#  IMPACT MODEL  (v4 — time-based exponential decay)
#
#  The key insight missing from v3: impact is NOT a function of bar number.
#  It's a function of ELAPSED TIME. A trade at 09:00 still affects the
#  market at 09:02 regardless of how many 5-minute bars have elapsed.
#
#  Model: I(t) = I₀ × exp( -ln(2) × (t - t₀) / t_half )
#  where t_half is the half-life in seconds.
#
#  Typical half-lives observed in crypto:
#    Temp impact: 60-600 seconds (depends on liquidity)
#    Perm impact: ∞ (information is incorporated permanently)
# ─────────────────────────────────────────────────────────────────────────────

class ImpactModel:
    """
    Market impact model with time-based exponential decay.

    Tracks a running impact state for a sequence of trades.
    Each trade adds temporary impact; time decays it.

    Parameters
    ----------
    temp_half_life_seconds : float
        Half-life of temporary impact. Default 300s (5 min).
    perm_fraction : float
        Fraction of total impact that is permanent. Default 0.2.
    impact_multiplier : float
        Overall impact scale factor. Tune to your universe.
    """

    def __init__(
        self,
        temp_half_life_seconds: float = 300.0,
        perm_fraction:          float = 0.2,
        impact_multiplier:      float = 1.0,
        # v3 compat: decay_param kept but now used as fallback
        decay_param:            float = 0.5,
    ):
        self.t_half      = temp_half_life_seconds
        self.perm_frac   = perm_fraction
        self.multiplier  = impact_multiplier
        self._decay_fallback = decay_param  # backwards compat

        # Running state
        self._shocks: List[Tuple[float, float]] = []  # (timestamp_sec, shock_bps)

    def calculate_impact(
        self,
        trade_size_base:  float,
        volume_avg_base:  float,
        daily_vol:        float,
    ) -> Tuple[float, float]:
        """
        Square-root impact model (Almgren et al. / Grinold & Kahn).

        Returns (temp_impact_bps, perm_impact_bps).

        The square-root model is the market standard:
            I = σ × √(Q / V)
        where σ = daily volatility, Q = trade size, V = daily volume.
        """
        if volume_avg_base <= 0:
            return 0.0, 0.0

        participation = max(trade_size_base / volume_avg_base, 0.0)
        total_impact  = self.multiplier * daily_vol * np.sqrt(participation)

        temp = total_impact * (1 - self.perm_frac)
        perm = total_impact * self.perm_frac
        return temp, perm

    def record_trade(self, trade_timestamp_sec: float, impact_bps: float) -> None:
        """
        Record a new impact shock at a given timestamp (Unix seconds).
        Impact will decay over time according to the half-life.
        """
        self._shocks.append((trade_timestamp_sec, impact_bps))

    def current_impact(self, current_timestamp_sec: float) -> float:
        """
        Compute total residual impact at current_timestamp_sec.
        Expired shocks (reduced to < 1% of original) are pruned.

        I_total(t) = Σ I_i × exp(-ln2 × (t - t_i) / t_half)
        """
        if not self._shocks:
            return 0.0

        total = 0.0
        active_shocks = []

        for t_trade, shock in self._shocks:
            elapsed = max(current_timestamp_sec - t_trade, 0.0)
            decay   = np.exp(-np.log(2) * elapsed / self.t_half)
            residual = shock * decay
            if residual > shock * 0.01:  # prune when < 1% of original
                active_shocks.append((t_trade, shock))
            total += residual

        self._shocks = active_shocks
        return total

    def apply_decay(self, current_shock: float) -> float:
        """
        Legacy method for v3 compatibility.
        Decays a single shock by the bar-based decay parameter.
        Prefer current_impact() for new code.
        """
        return current_shock * self._decay_fallback

    def impact_with_decay(
        self,
        trade_time:        float,
        current_time:      float,
        initial_shock:     float,
        half_life_seconds: float = 300.0,
    ) -> float:
        """
        Compute the residual impact of a single trade.
        Standalone function, doesn't use internal state.
        """
        elapsed = max(current_time - trade_time, 0.0)
        return initial_shock * np.exp(-np.log(2) * elapsed / half_life_seconds)

    def reset(self) -> None:
        """Clear all recorded shocks."""
        self._shocks = []


# ─────────────────────────────────────────────────────────────────────────────
#  LOB SIMULATOR  (v3 — snapshot mode for backtesting, unchanged from v3)
#  Used when we don't have L2 data and need to SYNTHESIZE a book.
#  The OrderBook class above is for when we HAVE L2 data.
# ─────────────────────────────────────────────────────────────────────────────

class LOBSimulator:
    """
    Synthesizes a Limit Order Book snapshot from OHLCV data.

    Use when L2 data is unavailable. Uses exponential liquidity decay
    to model depth — a reasonable approximation for backtesting.

    For production simulation with actual L2 data, use OrderBook.
    """

    def __init__(self, depth_levels: int = 10, spread_bps: float = 2.0):
        self.depth_levels = depth_levels
        self.spread_bps   = spread_bps / 10_000.0

    def get_simulated_book(
        self, mid_price: float, volume_avg: float
    ) -> Dict[str, np.ndarray]:
        """
        Generates a synthetic LOB from mid-price and average bar volume.
        Liquidity decays exponentially away from mid (realistic approximation).
        """
        levels    = np.arange(1, self.depth_levels + 1)
        ask_prices = mid_price * (1 + self.spread_bps / 2 + levels * 0.0005)
        bid_prices = mid_price * (1 - self.spread_bps / 2 - levels * 0.0005)

        base_size = volume_avg * 0.1
        sizes     = base_size * np.exp(-0.2 * (levels - 1))

        return {
            "asks": np.column_stack((ask_prices, sizes)),
            "bids": np.column_stack((bid_prices, sizes)),
        }

    def execute_buy(
        self, amount_quote: float, book: Dict[str, np.ndarray]
    ) -> Tuple[float, float]:
        """Execute a buy order against the asks. Returns (avg_price, remaining_quote)."""
        asks              = book["asks"]
        total_filled_base = 0.0
        total_spent_quote = 0.0
        remaining         = amount_quote

        for price, size in asks:
            if remaining <= 0:
                break
            max_fill_quote = price * size
            fill_quote     = min(remaining, max_fill_quote)
            fill_base      = fill_quote / price
            total_filled_base += fill_base
            total_spent_quote += fill_quote
            remaining         -= fill_quote

        avg_price = (total_spent_quote / total_filled_base
                     if total_filled_base > 0 else asks[0, 0])
        return avg_price, remaining


# ─────────────────────────────────────────────────────────────────────────────
#  SMART ORDER ROUTER  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

class SmartOrderRouter:
    """
    Splits large orders across multiple venues to minimize market impact.
    In production, venue weights would be dynamic based on real-time liquidity.
    """

    def __init__(self, venues: List[str] = ["Binance", "Coinbase", "Kraken"]):
        self.venues = venues
        self.sim    = LOBSimulator()
        # Static liquidity weights (production: dynamic from real L2)
        self._weights = {"Binance": 0.60, "Coinbase": 0.30, "Kraken": 0.10}

    def route_buy(
        self, total_quote: float, mid_price: float, vol_avg: float
    ) -> Dict[str, Any]:
        """Route buy order across venues proportional to liquidity."""
        results          = {}
        total_filled_base = 0.0

        for venue in self.venues:
            w           = self._weights.get(venue, 0.0)
            venue_quote = total_quote * w
            book        = self.sim.get_simulated_book(mid_price, vol_avg)
            avg_p, rem  = self.sim.execute_buy(venue_quote, book)
            results[venue] = {"avg_price": avg_p, "filled_quote": venue_quote - rem}
            total_filled_base += (venue_quote - rem) / avg_p

        overall_avg = (total_quote / total_filled_base
                       if total_filled_base > 0 else mid_price)
        return {"avg_price": overall_avg, "venue_details": results}


# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTION ALGORITHMS  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionAlgos:
    """
    Institutional execution strategies: VWAP, TWAP, Iceberg.

    VWAP: Distributes order proportional to historical volume profile.
          Minimizes market timing risk. Standard for large orders.
    TWAP: Evenly splits over time. Simpler, predictable.
    Iceberg: Hides order size. Shows only a small visible fraction.
    """

    @staticmethod
    def get_vwap_schedule(
        total_amount: float, volume_profile: np.ndarray
    ) -> np.ndarray:
        """Calculates trades per bar based on historical volume profile."""
        weights = volume_profile / volume_profile.sum()
        return total_amount * weights

    @staticmethod
    def get_twap_schedule(total_amount: float, num_bars: int) -> np.ndarray:
        """Evenly distributes trade amount over a fixed number of bars."""
        return np.full(num_bars, total_amount / num_bars)

    @staticmethod
    def iceberg_execution(
        bar_amount: float, display_frac: float = 0.1
    ) -> Tuple[float, float]:
        """
        Splits a single bar execution into displayed and hidden portions.
        Returns (visible_amount, hidden_amount).
        """
        visible = bar_amount * display_frac
        hidden  = bar_amount * (1 - display_frac)
        return visible, hidden

    @staticmethod
    def implementation_shortfall(
        decision_price: float,
        avg_fill_price: float,
        total_quantity: float,
        benchmark_price: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Implementation Shortfall = decision price - execution price.
        The industry-standard TCA (Transaction Cost Analysis) metric.

        Components:
        - Delay cost: benchmark_price - decision_price (opportunity cost)
        - Market impact: avg_fill_price - benchmark_price
        - Total IS: avg_fill_price - decision_price
        """
        total_is = (avg_fill_price - decision_price) * total_quantity
        is_bps   = (avg_fill_price / decision_price - 1) * 10_000

        result = {
            "total_is_$":   total_is,
            "is_bps":        is_bps,
            "decision_px":   decision_price,
            "avg_fill_px":   avg_fill_price,
        }

        if benchmark_price is not None:
            delay_cost  = (benchmark_price - decision_price) * total_quantity
            impact_cost = (avg_fill_price  - benchmark_price) * total_quantity
            result["delay_cost_$"]  = delay_cost
            result["impact_cost_$"] = impact_cost

        return result
