"""
Analyze 10 random weeks - top trending coins with long/short breakdown
"""

import pandas as pd
import random

# Read trades data
df = pd.read_csv("test_50_results_ic_filter/all_trades_v4.csv")

# Get unique weeks
all_weeks = sorted(df['week'].unique())
print(f"\nTotal weeks available: {len(all_weeks)}")

# Select 10 random weeks
random_weeks = sorted(random.sample(list(all_weeks), min(10, len(all_weeks))))

print("\n" + "="*110)
print("  10-WEEK RANDOM SAMPLE - TOP TRENDING COINS ANALYSIS (LONG vs SHORT)")
print("="*110)

week_results = []

for idx, week_num in enumerate(random_weeks, 1):
    week_data = df[df['week'] == week_num].copy()
    week_start = week_data['week_start'].iloc[0]
    
    # Find top coin by absolute PnL
    top_coin_idx = week_data['pnl_percent'].abs().idxmax()
    top_coin_data = week_data.loc[[top_coin_idx]]
    top_coin = top_coin_data['symbol'].values[0]
    top_coin_ret = top_coin_data['pnl_percent'].values[0]
    top_coin_signal = top_coin_data['signal'].values[0]
    
    # Get all coins for this week
    longs = week_data[week_data['signal'] == 'BUY']
    shorts = week_data[week_data['signal'] == 'SELL']
    
    long_ret = longs['pnl_percent'].sum()
    short_ret = shorts['pnl_percent'].sum()
    week_ret = long_ret + short_ret
    
    long_winners = (longs['pnl_percent'] > 0).sum()
    short_winners = (shorts['pnl_percent'] > 0).sum()
    
    week_results.append({
        'week': week_num,
        'start': week_start,
        'top_coin': top_coin,
        'top_return': top_coin_ret,
        'top_signal': top_coin_signal,
        'longs': len(longs),
        'long_ret': long_ret,
        'long_wins': long_winners,
        'shorts': len(shorts),
        'short_ret': short_ret,
        'short_wins': short_winners,
        'total_trades': len(week_data),
        'week_ret': week_ret,
    })

# Display results
print(f"\n{'Week':<6} {'Date':<12} {'Top Coin':<15} {'Ret %':<10} {'L/S':<6} {'Longs':<15} {'Shorts':<15} {'Week %':<10}")
print("-"*110)

for r in week_results:
    tops = r['top_signal'][:1]  # B or S
    longs_str = f"{r['longs']} ({r['long_wins']}W) {r['long_ret']:+.2f}%"
    shorts_str = f"{r['shorts']} ({r['short_wins']}W) {r['short_ret']:+.2f}%"
    print(f"{r['week']:<6} {r['start']:<12} {r['top_coin']:<15} {r['top_return']:>+8.2f}%  {tops:<6} {longs_str:<15} {shorts_str:<15} {r['week_ret']:>+8.2f}%")

# Summary statistics
print("\n" + "="*110)
print("  SUMMARY STATISTICS")
print("="*110)

total_longs = sum(r['longs'] for r in week_results)
total_long_wins = sum(r['long_wins'] for r in week_results)
total_long_ret = sum(r['long_ret'] for r in week_results)

total_shorts = sum(r['shorts'] for r in week_results)
total_short_wins = sum(r['short_wins'] for r in week_results)
total_short_ret = sum(r['short_ret'] for r in week_results)

total_trades = total_longs + total_shorts
total_weeks_ret = sum(r['week_ret'] for r in week_results)

print(f"\nTotal Weeks Analyzed:     {len(week_results)}")
print(f"\n  LONG POSITIONS:")
print(f"    Total Trades:         {total_longs}")
print(f"    Winners:              {total_long_wins}/{total_longs} ({total_long_wins/total_longs*100:.1f}% win rate)")
print(f"    Total Return:         {total_long_ret:>+7.2f}%")
print(f"    Avg Return/Trade:     {total_long_ret/total_longs:>+7.2f}%")

print(f"\n  SHORT POSITIONS:")
print(f"    Total Trades:         {total_shorts}")
print(f"    Winners:              {total_short_wins}/{total_shorts} ({total_short_wins/total_shorts*100:.1f}% win rate)")
print(f"    Total Return:         {total_short_ret:>+7.2f}%")
print(f"    Avg Return/Trade:     {total_short_ret/total_shorts:>+7.2f}%")

print(f"\n  OVERALL:")
print(f"    Total Trades:         {total_trades}")
print(f"    Total Winners:        {total_long_wins + total_short_wins}/{total_trades} ({(total_long_wins + total_short_wins)/total_trades*100:.1f}%)")
print(f"    Total Return:         {total_weeks_ret:>+7.2f}%")

# Direction bias
if total_long_ret > abs(total_short_ret):
    print(f"\n  ✓ Direction Bias:        LONG FAVORED (Longs {total_long_ret:+.2f}% vs Shorts {total_short_ret:+.2f}%)")
elif abs(total_short_ret) > total_long_ret:
    print(f"\n  ✓ Direction Bias:        SHORT FAVORED (Shorts {total_short_ret:+.2f}% vs Longs {total_long_ret:+.2f}%)")
else:
    print(f"\n  ✓ Direction Bias:        FAIRLY BALANCED")

# Best and worst week
best_week = max(week_results, key=lambda x: x['week_ret'])
worst_week = min(week_results, key=lambda x: x['week_ret'])

print(f"\n  Best Week:    Week {best_week['week']:2d} ({best_week['start']}) - {best_week['top_coin']:<10} {best_week['week_ret']:>+6.2f}%")
print(f"  Worst Week:   Week {worst_week['week']:2d} ({worst_week['start']}) - {worst_week['top_coin']:<10} {worst_week['week_ret']:>+6.2f}%")

print("\n" + "="*110 + "\n")
