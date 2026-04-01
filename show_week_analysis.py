"""
Show week 6 analysis - top trending coin
"""

print("\n" + "="*95)
print("  WEEK 6 ANALYSIS (2024-04-15)")
print("="*95 + "\n")

# Week 6 data extracted from CSV
week_6_trades = [
    ("BONKUSDT", "BUY", "LONG", 0.01586, 13.6428),
    ("TAOUSDT", "BUY", "LONG", 0.02084, -6.3733),
    ("DGBUSDT", "BUY", "LONG", 0.01214, 0.4935),
    ("MANTAUSDT", "BUY", "LONG", 0.01234, 0.6441),
    ("TRUUSDT", "BUY", "LONG", 0.01544, 5.8938),
    ("SANTOSUSDT", "SELL", "SHORT", -0.02524, -8.4459),
    ("DODOUSDT", "SELL", "SHORT", 0.00395, -7.9307),
    ("UMAUSDT", "SELL", "SHORT", 0.00484, -4.0425),
    ("FDUSDUSDT", "SELL", "SHORT", 0.00498, -0.0341),
]

# Find top coin by PnL
top_coin_name = max(week_6_trades, key=lambda x: abs(x[4]))
top_coin = top_coin_name[0]

print(f"  📊 TOP TRENDING COIN: {top_coin} (Return: {top_coin_name[4]:+.2f}%)")
print(f"  Total Trades in Week: {len(week_6_trades)}\n")

print(f"  {'Coin':<15} {'Type':<8} {'Prediction':<12} {'Return %':<12}")
print("  " + "-"*55)

# Show all week trades
for coin, signal, direction, pred, ret in week_6_trades:
    print(f"  {coin:<15} {direction:<8} {pred:>+10.5f}  {ret:>+10.2f}%")

# Count by direction
longs = [t for t in week_6_trades if t[2] == "LONG"]
shorts = [t for t in week_6_trades if t[2] == "SHORT"]

print("\n  " + "-"*55)
print(f"\n  SUMMARY:")
print(f"    Long Trades:   {len(longs)}")
print(f"      Return:     {sum(t[4] for t in longs):>+.2f}%")
print(f"      Winners:    {sum(1 for t in longs if t[4] > 0)}/{len(longs)}")
print(f"\n    Short Trades:  {len(shorts)}")
print(f"      Return:     {sum(t[4] for t in shorts):>+.2f}%")
print(f"      Winners:    {sum(1 for t in shorts if t[4] > 0)}/{len(shorts)}")

print(f"\n    Direction Bias: ", end="")
if len(longs) > len(shorts):
    print(f"LONG DOMINANT ({len(longs)} longs vs {len(shorts)} shorts)")
elif len(shorts) > len(longs):
    print(f"SHORT DOMINANT ({len(shorts)} shorts vs {len(longs)} longs)")
else:
    print(f"BALANCED")

total_ret = sum(t[4] for t in week_6_trades)
winners = sum(1 for t in week_6_trades if t[4] > 0)
print(f"\n    Total Return:  {total_ret:+.2f}%")
print(f"    Win Rate:      {winners}/{len(week_6_trades)} ({winners/len(week_6_trades)*100:.1f}%)")

print(f"\n  🎯 TOP PERFORMER: {top_coin} ({top_coin_name[2]}) = {top_coin_name[4]:+.2f}%")
print(f"     Signal: {top_coin_name[1]} | Prediction: {top_coin_name[3]:+.5f}")

print("\n" + "="*95 + "\n")
