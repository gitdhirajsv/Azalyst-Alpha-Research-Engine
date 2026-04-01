#!/usr/bin/env python3
"""
Quick optimization tests without modifying main engine.
Tests hypothesis: Kill-switch too aggressive, causing -67% of weeks with 0% return.

Test 1: Raise kill-switch threshold from -0.03 to -0.10 (more permissive)
Test 2: Try short-bias (average -0.2886% longs vs +0.0289% shorts)
Test 3: Invert IC when negative (use opposite direction for anticorrelated periods)
"""

import sqlite3
import numpy as np
import pandas as pd
import json

def analyze_optimization_impact():
    """Estimate impact of optimizations based on diagnostic data"""
    
    # Load from CSV instead of database (analysis from latest run)
    import os
    weekly_csv = 'results/weekly_summary_v4.csv'
    all_trades_csv = 'results/all_trades_v4.csv'
    
    df_weekly = pd.read_csv(weekly_csv)
    df_trades = pd.read_csv(all_trades_csv) if os.path.exists(all_trades_csv) else None
    
    # Ensure column names are clean
    df_weekly.columns = df_weekly.columns.str.strip()
    print(f"Weekly columns: {df_weekly.columns.tolist()}")
    print(f"First rows:\n{df_weekly.head()}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION IMPACT ANALYSIS")
    print("="*80)
    
    # Rename columns for consistency
    df = df_weekly.copy()
    if 'week_return_pct' in df.columns:
        df['return_pct'] = df['week_return_pct']
    if 'rolling_ic' in df.columns:
        df['ic'] = df['rolling_ic']
        
    # Current performance
    total_return = df['return_pct'].sum()
    weeks_trading = (df['n_trades'] > 0).sum() if 'n_trades' in df.columns else len(df)
    weeks_gated = (df['regime'] == 'KILL_SWITCH').sum() if 'regime' in df.columns else 0
    
    print(f"\nCURRENT BASELINE:")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Weeks Trading: {weeks_trading}/{len(df)} ({weeks_trading/len(df)*100:.1f}%)")
    print(f"  Weeks Gated (Kill-Switch): {weeks_gated}/{len(df)} ({weeks_gated/len(df)*100:.1f}%)")
    
    # Estimate optimization 1: Disable kill-switch
    # Assumption: If kill-switch gated weeks had same median return as trading weeks
    trading_weeks = df[df['n_trades'] > 0] if 'n_trades' in df.columns else df[df['regime'] != 'KILL_SWITCH']
    median_trading_return = trading_weeks['return_pct'].median() if 'return_pct' in trading_weeks.columns else 0.5
    
    gated_weeks = df[df['regime'] == 'KILL_SWITCH'] if 'regime' in df.columns else pd.DataFrame()
    estimated_gated_return = gated_weeks.shape[0] * median_trading_return if gated_weeks.shape[0] > 0 else 0
    
    potential_gain_disable_ks = estimated_gated_return
    new_total_disable_ks = total_return + potential_gain_disable_ks
    
    print(f"\nOPTIMIZATION 1: DISABLE KILL-SWITCH")
    print(f"  Gated weeks: {gated_weeks.shape[0]}")
    print(f"  Median trading week return: {median_trading_return:.2f}%")
    print(f"  Estimated gain if gated weeks traded: +{estimated_gated_return:.2f}%")
    print(f"  Estimated new total: {new_total_disable_ks:.2f}%")
    print(f"  Status: {'✓ POSITIVE' if new_total_disable_ks > 0 else '✗ Still negative'}")
    
    # Optimization 2: Short-bias only
    # Get actual long vs short performance from trades table
    if df_trades is not None:
        long_trades = df_trades[df_trades['signal'] == 'BUY']
        short_trades = df_trades[df_trades['signal'] == 'SELL']
        
        long_avg_pnl = long_trades['pnl_percent'].mean() if len(long_trades) > 0 else 0
        short_avg_pnl = short_trades['pnl_percent'].mean() if len(short_trades) > 0 else 0
        
        print(f"\nOPTIMIZATION 2: SHORT-BIAS ONLY")
        print(f"  Long trades: {len(long_trades)}, avg PnL: {long_avg_pnl:.4f}%")
        print(f"  Short trades: {len(short_trades)}, avg PnL: {short_avg_pnl:.4f}%")
        print(f"  Difference: {(short_avg_pnl - long_avg_pnl):.4f}%")
        
        # Rough estimate: if we only take shorts instead of both
        if len(long_trades) > 0 and len(short_trades) > 0:
            current_avg_per_trade = (long_avg_pnl + short_avg_pnl) / 2
            short_only_avg = short_avg_pnl
            
            potential_gain_short = (short_only_avg - current_avg_per_trade) * (len(long_trades) + len(short_trades)) / 100
            print(f"  Estimated gain from short-only: +{potential_gain_short:.2f}%")
        else:
            potential_gain_short = 0
    else:
        potential_gain_short = 0
    
    # Optimization 3: Invert negative IC
    bear_trend_weeks = df[df['regime'] == 'BEAR_TREND'] if 'regime' in df.columns else pd.DataFrame()
    if len(bear_trend_weeks) > 0:
        bear_median = bear_trend_weeks['return_pct'].median()
        bear_investment = bear_trend_weeks['return_pct'].sum()
        inverted_estimate = -bear_investment  # If we inverted positions, flip sign
        
        print(f"\nOPTIMIZATION 3: INVERT ON NEGATIVE IC")
        print(f"  BEAR_TREND weeks: {len(bear_trend_weeks)}")
        print(f"  Current return: {bear_investment:.2f}%")
        print(f"  If inverted: {inverted_estimate:.2f}%")
        print(f"  Potential gain: +{(inverted_estimate - bear_investment):.2f}%")
    
    # Combined estimate
    print(f"\nCOMBINED OPTIMIZATION ESTIMATE:")
    combined_potential = potential_gain_disable_ks
    combined_potential += potential_gain_short * 0.5  # Conservative estimate
    
    print(f"  Kill-switch removal: +{potential_gain_disable_ks:.2f}%")
    print(f"  Short-bias impact: +{potential_gain_short * 0.5:.2f}% (conservative)")
    print(f"  Estimated combined gain: +{combined_potential:.2f}%")
    print(f"  New estimated total: {total_return + combined_potential:.2f}%")
    
    print("\n" + "="*80)
    print("RECOMMENDATION: Test kill-switch removal first (highest impact)")
    print("="*80)

if __name__ == "__main__":
    analyze_optimization_impact()
