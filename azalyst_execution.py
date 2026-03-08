"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    EXECUTION & MARKET INFRASTRUCTURE
║        LOB Simulator · Smart Order Router · Institutional Algos (VWAP/TWAP)║
║        v3.0  |  Hedge Fund Grade Execution Layer  |  Market Impact Pro      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class LOBSimulator:
    """
    Simulates Limit Order Book (LOB) depth and liquidity.
    Used for high-fidelity backtesting of large institutional orders.
    """
    def __init__(self, depth_levels: int = 10, spread_bps: float = 2.0):
        self.depth_levels = depth_levels
        self.spread_bps = spread_bps / 10000.0

    def get_simulated_book(self, mid_price: float, volume_avg: float) -> Dict[str, np.ndarray]:
        """
        Generates a synthetic LOB based on mid-price and average volume per bar.
        Liquidity typically decays exponentially as we move away from mid.
        """
        levels = np.arange(1, self.depth_levels + 1)
        # Prices
        ask_prices = mid_price * (1 + self.spread_bps/2 + levels * 0.0005)
        bid_prices = mid_price * (1 - self.spread_bps/2 - levels * 0.0005)
        
        # Simulated Sizes (Exponential decay of liquidity)
        base_size = volume_avg * 0.1
        sizes = base_size * np.exp(-0.2 * (levels - 1))
        
        return {
            "asks": np.column_stack((ask_prices, sizes)),
            "bids": np.column_stack((bid_prices, sizes))
        }

    def execute_buy(self, amount_quote: float, book: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Executes a buy order against the asks and returns (avg_price, remaining_quote)."""
        asks = book["asks"]
        total_filled_base = 0
        total_spent_quote = 0
        remaining = amount_quote
        
        for price, size in asks:
            if remaining <= 0: break
            max_fill_quote = price * size
            fill_quote = min(remaining, max_fill_quote)
            fill_base = fill_quote / price
            
            total_filled_base += fill_base
            total_spent_quote += fill_quote
            remaining -= fill_quote
            
        avg_price = total_spent_quote / total_filled_base if total_filled_base > 0 else asks[0,0]
        return avg_price, remaining

class SmartOrderRouter:
    """
    Simulates splitting large orders across multiple venues to minimize impact.
    """
    def __init__(self, venues: List[str] = ["Binance", "Coinbase", "Kraken"]):
        self.venues = venues
        self.sim = LOBSimulator()

    def route_buy(self, total_quote: float, mid_price: float, vol_avg: float) -> Dict[str, Any]:
        """
        Routes quote amount across venues. 
        In this simulation, we allocate proportional to venue liquidity.
        """
        weights = {"Binance": 0.6, "Coinbase": 0.3, "Kraken": 0.1}
        results = {}
        total_filled_base = 0
        
        for venue in self.venues:
            venue_quote = total_quote * weights.get(venue, 0)
            book = self.sim.get_simulated_book(mid_price, vol_avg)
            avg_p, rem = self.sim.execute_buy(venue_quote, book)
            results[venue] = {"avg_price": avg_p, "filled_quote": venue_quote - rem}
            total_filled_base += (venue_quote - rem) / avg_p
            
        overall_avg = total_quote / total_filled_base if total_filled_base > 0 else mid_price
        return {"avg_price": overall_avg, "venue_details": results}

class ExecutionAlgos:
    """
    Hedge Fund execution strategies: VWAP, TWAP, Iceberg.
    """
    @staticmethod
    def get_vwap_schedule(total_amount: float, volume_profile: np.ndarray) -> np.ndarray:
        """Calculates trades per bar based on a volume profile."""
        weights = volume_profile / volume_profile.sum()
        return total_amount * weights

    @staticmethod
    def get_twap_schedule(total_amount: float, num_bars: int) -> np.ndarray:
        """Evenly distributes trade amount over a fixed number of bars."""
        return np.full(num_bars, total_amount / num_bars)

    @staticmethod
    def iceberg_execution(bar_amount: float, display_frac: float = 0.1) -> Tuple[float, float]:
        """Splits a single bar execution into 'displayed' and 'hidden' (iceberg)."""
        visible = bar_amount * display_frac
        hidden = bar_amount * (1 - display_frac)
        return visible, hidden

class ImpactModel:
    """
    Simulates temporary vs. permanent market impact and decay.
    Temporary impact decays over time (mean reversion).
    Permanent impact represents information flow.
    """
    def __init__(self, decay_param: float = 0.5):
        self.decay_param = decay_param # Half-life style decay

    def calculate_impact(self, 
                         trade_size_base: float, 
                         volume_avg_base: float, 
                         daily_vol: float) -> Tuple[float, float]:
        """
        Square-root impact model (Grinold & Kahn).
        Returns (temp_impact_bps, perm_impact_bps).
        """
        participation = trade_size_base / volume_avg_base
        # 1.0 is a typical multiplier for impact
        total_impact = 1.0 * daily_vol * np.sqrt(participation)
        
        # Split into 80% temporary (decayable) and 20% permanent
        temp = total_impact * 0.8
        perm = total_impact * 0.2
        return temp, perm

    def apply_decay(self, current_shock: float) -> float:
        """Simulates the decay of a temporary liquidity shock."""
        return current_shock * self.decay_param
