"""
Single week test on top trending coin - analyze long/short behavior
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path

# Load results from latest run
results_dir = Path("test_50_results_ic_filter")
if not results_dir.exists():
    results_dir = Path("test_50_results")

# Load trades
trades_file = results_dir / "all_trades_v4.csv"
if trades_file.exists():
    trades_df = pd.read_csv(trades_file)
    
    # Get unique weeks and pick one randomly
    trades_df['week'] = pd.to_datetime(trades_df['week'])
    unique_weeks = trades_df['week'].unique()
    
    if len(unique_weeks) > 0:
        random_week = pd.Timestamp(random.choice(unique_weeks))
        week_trades = trades_df[trades_df['week'] == random_week]
        
        print("\n" + "="*90)
        print(f"  SINGLE WEEK ANALYSIS: {random_week.strftime('%Y-%m-%d')}")
        print("="*90)
        
        if len(week_trades) > 0:
            # Find top trending coin (most trades or biggest return)
            coin_returns = week_trades.groupby('symbol')['pnl_pct'].sum().sort_values(ascending=False)
            top_coin = coin_returns.index[0]
            
            # Get trades for top coin in this week
            top_coin_trades = week_trades[week_trades['symbol'] == top_coin]
            
            print(f"\n  TOP TRENDING COIN: {top_coin}")
            print(f"  Total Trades This Week: {len(week_trades)}")
            print(f"  Trades on {top_coin}: {len(top_coin_trades)}")
            print(f"\n  {'Date':<12} {'Type':<8} {'Entry':<12} {'Exit':<12} {'Return %':<12} {'PnL $':<12}")
            print("  " + "-"*78)
            
            longs = top_coin_trades[top_coin_trades['direction'] == 'LONG']
            shorts = top_coin_trades[top_coin_trades['direction'] == 'SHORT']
            
            for idx, row in top_coin_trades.iterrows():
                date_str = str(row['date'])[:10] if 'date' in row else str(row['week'])[:10]
                print(f"  {date_str:<12} {row['direction']:<8} {row['entry_price']:>11.2f} {row['exit_price']:>11.2f} {row['pnl_pct']:>+10.2f}% {row['pnl']:>+10.2f}")
            
            total_pnl = top_coin_trades['pnl_pct'].sum()
            avg_return = top_coin_trades['pnl_pct'].mean()
            
            print("  " + "-"*78)
            print(f"\n  SUMMARY FOR {top_coin}:")
            print(f"    Long Trades:   {len(longs):>3d}  ({longs['pnl_pct'].sum():>+7.2f}% total)")
            print(f"    Short Trades:  {len(shorts):>3d}  ({shorts['pnl_pct'].sum():>+7.2f}% total)")
            print(f"    Total Return:  {total_pnl:>+7.2f}%")
            print(f"    Avg Trade:     {avg_return:>+7.2f}%")
            print(f"    Win Rate:      {(top_coin_trades['pnl_pct'] > 0).sum()}/{len(top_coin_trades)} ({(top_coin_trades['pnl_pct'] > 0).sum()/len(top_coin_trades)*100:.1f}%)")
            
            # Direction bias
            if len(longs) > 0 and len(shorts) == 0:
                print(f"    Bias:          LONG ONLY (no shorts)")
            elif len(shorts) > 0 and len(longs) == 0:
                print(f"    Bias:          SHORT ONLY (no longs)")
            elif len(longs) > len(shorts):
                print(f"    Bias:          LONG BIAS ({len(longs)} longs vs {len(shorts)} shorts)")
            elif len(shorts) > len(longs):
                print(f"    Bias:          SHORT BIAS ({len(shorts)} shorts vs {len(longs)} longs)")
            else:
                print(f"    Bias:          BALANCED")
            
        else:
            print(f"\n  No trades found for week {random_week}")
        
        print("\n" + "="*90 + "\n")
    else:
        print("No weeks found in trades data")
else:
    print(f"Trades file not found: {trades_file}")
