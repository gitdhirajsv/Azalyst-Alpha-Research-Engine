"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST ALPHA RESEARCH ENGINE    TRADER AUDITOR & SIGNAL EXTRACTOR
║        Binance Lead Trader Benchmarking · Strategy Reversing · Copy Audit  ║
║        v4.0  |  Alpha Hunter Module  |  Institutional vs. Social Quant    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import requests
import logging
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AzalystAuditor")

class TraderAuditor:
    """
    Audits and reverse-engineers successful traders from social/copy platforms.
    """
    def __init__(self):
        self.session = requests.Session()
        # Mocking headers to look like a browser for Binance BAPI
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Referer": "https://www.binance.com/en/copy-trading"
        })

    def fetch_trader_stats(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Fetches core profile details.
        """
        url = "https://www.binance.com/bapi/futures/v1/friendly/future/copy-trade/lead-portfolio/detail"
        try:
            resp = self.session.get(f"{url}?portfolioId={portfolio_id}", timeout=10)
            data = resp.json()
            if data.get("success"):
                return data["data"]
            return {}
        except Exception as e:
            logger.error(f"Error fetching trader stats: {e}")
            return {}

    def fetch_performance_stats(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Fetches ROI, MDD, and Sharpe from the dedicated performance endpoint.
        """
        url = "https://www.binance.com/bapi/futures/v1/friendly/future/copy-trade/lead-portfolio/performance"
        # BAPI often requires a POST or specific params for performance
        try:
            resp = self.session.get(f"{url}?portfolioId={portfolio_id}", timeout=10)
            data = resp.json()
            if data.get("success"):
                return data["data"]
            return {}
        except Exception as e:
            logger.error(f"Error fetching performance stats: {e}")
            return {}

    def fetch_position_history(self, portfolio_id: str) -> pd.DataFrame:
        """
        Extracts recent closed positions to identify the trader's strategy.
        """
        url = "https://www.binance.com/bapi/futures/v1/friendly/future/copy-trade/lead-portfolio/position-history"
        payload = {"portfolioId": portfolio_id, "pageNumber": 1, "pageSize": 50}
        try:
            resp = self.session.post(url, json=payload, timeout=10)
            data = resp.json()
            if data.get("success") and "list" in data["data"]:
                df = pd.DataFrame(data["data"]["list"])
                if not df.empty:
                    # Binance uses 'closed' for closing timestamp
                    if 'closed' in df.columns:
                        df["closedTime"] = pd.to_datetime(df["closed"], unit='ms')
                    elif 'updateTime' in df.columns:
                        df["closedTime"] = pd.to_datetime(df["updateTime"], unit='ms')
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching position history: {e}")
            return pd.DataFrame()

    def classify_strategy(self, history: pd.DataFrame) -> str:
        """
        Reverse-engineers the trader's style from their trade logs.
        """
        if history.empty: return "UNKNOWN"
        
        # 1. Check for 'Grid' or 'Martingale'
        # Convert closedPnl to numeric
        pnl = pd.to_numeric(history["closingPnl"], errors='coerce').fillna(0)
        win_rate = (pnl > 0).mean()
        
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        
        avg_rr = 1.0
        if not losers.empty and not winners.empty:
            avg_rr = winners.mean() / abs(losers.mean())
        
        # 2. Timing
        avg_trade_time = 0
        if "closedTime" in history.columns:
            avg_trade_time = (history["closedTime"].diff().dt.total_seconds().abs().mean()) / 60

        if win_rate > 0.80 and avg_rr < 0.6:
            return "GRID / MARTINGALE (High Win Rate, Skewed Risk)"
        elif avg_trade_time > 0 and avg_trade_time < 60:
            return "HFT / SCALPER"
        elif avg_rr > 1.3:
            return "TREND FOLLOWER"
        else:
            return "DISCRETIONARY / MEAN REVERSION"

    def institutional_audit(self, stats: Dict[str, Any], perf: Dict[str, Any]) -> Dict[str, Any]:
        """
        Benchmarks against Azalyst institutional standards.
        """
        # Cross-reference keys: mdd, roi, sharpRatio
        mdd = float(perf.get("mdd", 0)) / 100.0 # BAPI returns % as 0-100
        roi = float(perf.get("roi", 0))
        sharpe = float(perf.get("sharpRatio", stats.get("sharpRatio", 0)))
        
        status = "REJECTED (Alpha Deficiency)"
        if sharpe > 2.5 and mdd < 0.20:
            status = "INSTITUTIONAL GRADE (Azalyst Certified)"
        elif sharpe > 1.5:
            status = "PROFESSIONAL ALPHA (Monitoring)"
            
        return {
            "status": status,
            "metrics": {"ROI": f"{roi:.2f}%", "MDD": f"{mdd*100:.2f}%", "Sharpe": f"{sharpe:.2f}"}
        }

if __name__ == "__main__":
    import json
    import os
    
    auditor = TraderAuditor()
    json_path = "tmp_trader_audit.json"
    
    if os.path.exists(json_path):
        logger.info(f"Local Data Found. Starting Institutional Audit of Top 3 Traders...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for pid, content in data.items():
            stats = content['detail'].get('data', {})
            perf = content['performance'].get('data', {})
            history_list = content['history'].get('data', {}).get('list', [])
            hist = pd.DataFrame(history_list)
            
            print(f"\n{'='*50}")
            print(f"TRADER: {stats.get('nickname')} ({pid})")
            print(f"{'='*50}")
            
            audit = auditor.institutional_audit(stats, perf)
            print(f"Status:   {audit['status']}")
            print(f"Metrics:  {audit['metrics']}")
            
            if not hist.empty:
                # Ensure closingPnl is present
                if 'closingPnl' not in hist.columns:
                    hist['closingPnl'] = 0
                style = auditor.classify_strategy(hist)
                print(f"Style:    {style}")
                print(f"Wins:     {perf.get('winOrders')} / {perf.get('totalOrder')}")
            else:
                print("Style:    NO PUBLIC HISTORY AVAILABLE")
    else:
        logger.warning("No local data file found. Try fetching live (may be rate-limited).")
        # Fallback to single ID if file missing (already implemented above in old script)
