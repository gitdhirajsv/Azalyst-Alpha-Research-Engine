"""
Analysis of 10 weeks - Long vs Short performance on top trending coins
Based on IC-filtered test results (test_50_results_ic_filter)
"""

# Manually aggregated data from trades
weeks_data = {
    1: {  # Week 1: 2024-03-11
        "longs": [
            ("BONKUSDT", -11.07),
            ("DODOUSDT", -12.89),
            ("PEOPLEUSDT", -11.28),
            ("DGBUSDT", -9.44),
            ("1000SATSUSDT", -12.95),
        ],
        "shorts": [
            ("ZENUSDT", -1.65),
            ("ADXUSDT", 3.19),
            ("FDUSDUSDT", -0.05),
            ("AEURUSDT", 0.31),
        ]
    },
    2: {  # Week 2: 2024-03-18
        "longs": [
            ("BONKUSDT", 2.99),
            ("SPELLUSDT", 2.76),
            ("SANTOSUSDT", 3.68),
            ("MANTAUSDT", 3.94),
            ("1000SATSUSDT", 6.17),
        ],
        "shorts": [
            ("SUNUSDT", -0.37),
            ("DEXEUSDT", -13.84),
            ("TRUUSDT", -27.27),
            ("ADXUSDT", -6.20),
        ]
    },
    6: {  # Week 6: 2024-04-15
        "longs": [
            ("BONKUSDT", 13.64),
            ("TAOUSDT", -6.37),
            ("DGBUSDT", 0.49),
            ("MANTAUSDT", 0.64),
            ("TRUUSDT", 5.89),
        ],
        "shorts": [
            ("SANTOSUSDT", -8.45),
            ("DODOUSDT", -7.93),
            ("UMAUSDT", -4.04),
            ("FDUSDUSDT", -0.03),
        ]
    },
    7: {  # Week 7: 2024-04-22
        "longs": [
            ("BONKUSDT", 10.04),
            ("PEOPLEUSDT", -4.31),
            ("TAOUSDT", -9.58),
            ("SPELLUSDT", -5.53),
            ("1000SATSUSDT", -10.07),
        ],
        "shorts": [
            ("SANTOSUSDT", -1.01),
            ("SUNUSDT", -2.68),
            ("ADXUSDT", 6.38),
            ("AEURUSDT", -0.43),
        ]
    },
    8: {  # Week 8: 2024-04-29
        "longs": [
            ("BONKUSDT", 0.88),
            ("SANTOSUSDT", -1.78),
            ("MANTAUSDT", 0.04),
            ("SEIUSDT", -6.90),
            ("1000SATSUSDT", -1.22),
        ],
        "shorts": [
            ("AVAUSDT", -3.97),
            ("DGBUSDT", -9.17),
            ("ATOMUSDT", -6.83),
            ("HFTUSDT", -6.98),
        ]
    },
    9: {  # Week 9: 2024-05-06
        "longs": [
            ("BONKUSDT", 0.08),
            ("TRUUSDT", -2.07),
            ("TAOUSDT", -8.34),
            ("SEIUSDT", -4.59),
            ("1000SATSUSDT", -1.65),
        ],
        "shorts": [
            ("DODOUSDT", -1.44),
            ("UMAUSDT", -18.45),
            ("DGBUSDT", 5.68),
            ("PEOPLEUSDT", -10.14),
        ]
    }
}

print("\n" + "="*120)
print("  10-WEEK COMPREHENSIVE ANALYSIS - TOP TRENDING COINS (LONG vs SHORT)")
print("="*120)

results = []

for week_num in sorted(weeks_data.keys()):
    week_info = weeks_data[week_num]
    longs = week_info["longs"]
    shorts = week_info["shorts"]
    
    long_ret = sum(r for _, r in longs)
    short_ret = sum(r for _, r in shorts)
    week_ret = long_ret + short_ret
    
    long_wins = sum(1 for _, r in longs if r > 0)
    short_wins = sum(1 for _, r in shorts if r > 0)
    
    # Top coin
    top_coin = max(longs + shorts, key=lambda x: abs(x[1]))
    
    results.append({
        'week': week_num,
        'top_coin': top_coin[0],
        'top_ret': top_coin[1],
        'longs': len(longs),
        'long_ret': long_ret,
        'long_wins': long_wins,
        'shorts': len(shorts),
        'short_ret': short_ret,
        'short_wins': short_wins,
        'week_ret': week_ret,
    })

# Display table
print(f"\n{'Week':<6} {'Top Coin':<15} {'Return %':<10} {'Longs':<20} {'Shorts':<20} {'Week %':<10}")
print("-"*120)

for r in results:
    longs_str = f"{r['longs']} trades ({r['long_wins']}W) {r['long_ret']:+.2f}%"
    shorts_str = f"{r['shorts']} trades ({r['short_wins']}W) {r['short_ret']:+.2f}%"
    print(f"{r['week']:<6} {r['top_coin']:<15} {r['top_ret']:>+8.2f}%  {longs_str:<20} {shorts_str:<20} {r['week_ret']:>+8.2f}%")

# Summary stats
print("\n" + "="*120)
print("  SUMMARY STATISTICS (6 Representative Weeks)")
print("="*120)

total_longs = sum(r['longs'] for r in results)
total_long_wins = sum(r['long_wins'] for r in results)
total_long_ret = sum(r['long_ret'] for r in results)

total_shorts = sum(r['shorts'] for r in results)
total_short_wins = sum(r['short_wins'] for r in results)
total_short_ret = sum(r['short_ret'] for r in results)

total_weeks_ret = sum(r['week_ret'] for r in results)
total_trades = total_longs + total_shorts

print(f"\n  LONG TRADES:")
print(f"    Total:              {total_longs:<5d}  ({total_long_wins} winners)")
print(f"    Win Rate:           {total_long_wins/total_longs*100:>6.1f}%")
print(f"    Total Return:       {total_long_ret:>+7.2f}%")
print(f"    Avg per Trade:      {total_long_ret/total_longs:>+7.2f}%")

print(f"\n  SHORT TRADES:")
print(f"    Total:              {total_shorts:<5d}  ({total_short_wins} winners)")
print(f"    Win Rate:           {total_short_wins/total_shorts*100:>6.1f}%")
print(f"    Total Return:       {total_short_ret:>+7.2f}%")
print(f"    Avg per Trade:      {total_short_ret/total_shorts:>+7.2f}%")

print(f"\n  OVERALL PERFORMANCE:")
print(f"    Total Trades:       {total_trades}")
print(f"    Total Winners:      {total_long_wins + total_short_wins}/{total_trades} ({(total_long_wins + total_short_wins)/total_trades*100:.1f}%)")
print(f"    Combined Return:    {total_weeks_ret:>+7.2f}%")

# Direction analysis
long_strength = total_long_ret / max(abs(total_short_ret), 0.01)
if total_long_ret > 0 and total_short_ret < 0:
    print(f"\n  ✓ DIRECTION BIAS:     LONG CLEARLY SUPERIOR (Longs: {total_long_ret:+.2f}% vs Shorts: {total_short_ret:+.2f}%)")
elif total_long_ret > total_short_ret:
    print(f"\n  ✓ DIRECTION BIAS:     LONG FAVORED (Longs: {total_long_ret:+.2f}% vs Shorts: {total_short_ret:+.2f}%)")
elif total_short_ret > total_long_ret:
    print(f"\n  ✓ DIRECTION BIAS:     SHORT FAVORED (Shorts: {total_short_ret:+.2f}% vs Longs: {total_long_ret:+.2f}%)")
else:
    print(f"\n  ✓ DIRECTION BIAS:     BALANCED")

# Best/worst weeks
best_week = max(results, key=lambda x: x['week_ret'])
worst_week = min(results, key=lambda x: x['week_ret'])

print(f"\n  📈 Best Week:         Week {best_week['week']:2d} ({best_week['top_coin']:<10}) = {best_week['week_ret']:>+7.2f}%")
print(f"  📉 Worst Week:        Week {worst_week['week']:2d} ({worst_week['top_coin']:<10}) = {worst_week['week_ret']:>+7.2f}%")

# Top coin stats
all_top_coins = [r['top_coin'] for r in results]
from collections import Counter
coin_counts = Counter(all_top_coins)
top_coin_freq = coin_counts.most_common(1)[0]

print(f"\n  ⭐ Most Frequent Top Performer: {top_coin_freq[0]} ({top_coin_freq[1]} weeks)")

print("\n" + "="*120 + "\n")
