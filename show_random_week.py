import pandas as pd
import random

# Read trades
df = pd.read_csv("test_50_results_ic_filter/all_trades_v4.csv")

# Get unique weeks
unique_weeks = sorted(df['week'].unique())
print(f"Total weeks: {len(unique_weeks)}")
print(f"Weeks: {unique_weeks}")

# Pick random week
random_week = random.choice(unique_weeks)
week_data = df[df['week'] == random_week].copy()

# Map signal to direction
week_data['direction'] = week_data['signal'].apply(lambda x: 'LONG' if x == 'BUY' else 'SHORT')

print(f"\n{'='*95}")
print(f"  WEEK {random_week} ANALYSIS - {week_data['week_start'].iloc[0]}")
print(f"{'='*95}\n")

# Get top coin by absolute PnL
top_coin = week_data.loc[week_data['pnl_percent'].abs().idxmax(), 'symbol']
top_coin_data = week_data[week_data['symbol'] == top_coin]

print(f"  TOP TRENDING COIN: {top_coin}")
print(f"  Week Trades: {len(week_data)} | {top_coin} Trades: {len(top_coin_data)}\n")

# Show trades
print(f"  {'Symbol':<12} {'Type':<6} {'Signal':<8} {'Prediction':<12} {'Return %':<12} {'Meta Size':<10}")
print(f"  {'-'*82}")

for _, row in top_coin_data.iterrows():
    direction = "LONG" if row['signal'] == 'BUY' else "SHORT"
    print(f"  {row['symbol']:<12} {direction:<6} {row['signal']:<8} {row['pred_ret']:>+10.5f}  {row['pnl_percent']:>+10.2f}%  {row['meta_size']:>8.4f}")

# Summary stats
longs = top_coin_data[top_coin_data['signal'] == 'BUY']
shorts = top_coin_data[top_coin_data['signal'] == 'SELL']

print(f"\n  {'-'*82}")
print(f"\n  SUMMARY:")
print(f"    Long Trades:   {len(longs):2d}  Return: {longs['pnl_percent'].sum():>+7.2f}%  Avg: {longs['pnl_percent'].mean():>+6.2f}%")
print(f"    Short Trades:  {len(shorts):2d}  Return: {shorts['pnl_percent'].sum():>+7.2f}%  Avg: {shorts['pnl_percent'].mean():>+6.2f}%")
print(f"    Total:         {len(top_coin_data):2d}  Return: {top_coin_data['pnl_percent'].sum():>+7.2f}%  Avg: {top_coin_data['pnl_percent'].mean():>+6.2f}%")
print(f"\n    Direction Bias: ", end="")

if len(longs) > 0 and len(shorts) == 0:
    print(f"LONG ONLY ({len(longs)} trades)")
elif len(shorts) > 0 and len(longs) == 0:
    print(f"SHORT ONLY ({len(shorts)} trades)")
elif len(longs) > len(shorts):
    ratio = len(longs) / (len(shorts) + 0.001)
    print(f"LONG BIAS ({len(longs)} longs vs {len(shorts)} shorts, {ratio:.1f}x)")
elif len(shorts) > len(longs):
    ratio = len(shorts) / (len(longs) + 0.001)
    print(f"SHORT BIAS ({len(shorts)} shorts vs {len(longs)} longs, {ratio:.1f}x)")
else:
    print(f"BALANCED ({len(longs)} longs, {len(shorts)} shorts)")

print(f"\n  Win Rate: {(top_coin_data['pnl_percent'] > 0).sum()}/{len(top_coin_data)} ({(top_coin_data['pnl_percent'] > 0).sum()/len(top_coin_data)*100:.1f}%)")
print(f"{'='*95}\n")
