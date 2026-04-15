import json
import csv
from collections import defaultdict

# Load performance JSON
with open(r"D:\Azalyst Alpha Research Engine\results_v7\performance_v7.json") as f:
    perf = json.load(f)

# Load weekly CSV
weeks = []
with open(r"D:\Azalyst Alpha Research Engine\results_v7\weekly_summary_v7.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        weeks.append(row)

# Filter out KILL_SWITCH weeks
active_weeks = [w for w in weeks if w['regime'] != 'KILL_SWITCH' and float(w['n_trades']) > 0]

print("=" * 80)
print("1. REGIME-SPECIFIC PERFORMANCE BREAKDOWN")
print("=" * 80)

regimes = defaultdict(list)
for w in active_weeks:
    regimes[w['regime']].append(w)

for regime_name in ['BULL_TREND', 'BEAR_TREND', 'LOW_VOL_GRIND', 'HIGH_VOL_LATERAL']:
    ws = regimes.get(regime_name, [])
    if not ws:
        print(f"\n{regime_name}: No active weeks")
        continue
    n_weeks = len(ws)
    avg_ret = sum(float(w['week_return_pct']) for w in ws) / n_weeks
    avg_ic = sum(float(w['ic']) for w in ws) / n_weeks
    long_pnl = sum(float(w['long_return_pct']) for w in ws)
    short_pnl = sum(float(w['short_return_pct']) for w in ws)
    win_weeks = sum(1 for w in ws if float(w['week_return_pct']) > 0)
    win_rate = win_weeks / n_weeks * 100

    print(f"\n--- {regime_name} ---")
    print(f"  Number of weeks:    {n_weeks}")
    print(f"  Avg weekly return:  {avg_ret:.4f}%")
    print(f"  Total return:       {sum(float(w['week_return_pct']) for w in ws):.4f}%")
    print(f"  Avg IC:             {avg_ic:.6f}")
    print(f"  Long PnL (sum):     {long_pnl:.4f}%")
    print(f"  Short PnL (sum):    {short_pnl:.4f}%")
    print(f"  Win rate:           {win_rate:.1f}% ({win_weeks}/{n_weeks})")
    print(f"  Std weekly return:  {(sum((float(w['week_return_pct'])-avg_ret)**2 for w in ws)/n_weeks)**0.5:.4f}%")

print("\n" + "=" * 80)
print("2. FEATURE IC BREAKDOWN ANALYSIS")
print("=" * 80)

features = perf['oos_diagnostics']['top_features_by_ic']
# Sort by mean_ic
sorted_feats = sorted(features.items(), key=lambda x: x[1]['mean_ic'], reverse=True)

print(f"\n{'Feature':<20} {'Mean IC':>10} {'Std IC':>10} {'N Obs':>6} {'Pos Ratio':>10} {'Assessment':<20}")
print("-" * 80)
for fname, fdata in sorted_feats:
    mean_ic = fdata['mean_ic']
    std_ic = fdata['std_ic']
    n_obs = fdata['n_observations']
    pos_ratio = fdata['positive_ic_ratio']
    icir = mean_ic / std_ic if std_ic > 0 else 0

    if mean_ic > 0.05 and pos_ratio >= 0.7:
        assessment = "STRONG_POSITIVE"
    elif mean_ic > 0 and pos_ratio >= 0.5:
        assessment = "MODERATE_POSITIVE"
    elif mean_ic > 0:
        assessment = "WEAK_POSITIVE"
    else:
        assessment = "NEGATIVE/NOISY"

    print(f"{fname:<20} {mean_ic:>10.6f} {std_ic:>10.6f} {n_obs:>6} {pos_ratio:>10.4f} {assessment:<20}")

print("\n--- Consistent Positive IC Features (mean_ic > 0, pos_ratio >= 0.7) ---")
consistent = [(f, d) for f, d in sorted_feats if d['mean_ic'] > 0 and d['positive_ic_ratio'] >= 0.7]
for f, d in consistent:
    print(f"  {f}: mean_ic={d['mean_ic']:.6f}, pos_ratio={d['positive_ic_ratio']:.4f}, n_obs={d['n_observations']}")

print("\n--- Noisy/Weak Features (pos_ratio < 0.6 or low n_obs) ---")
noisy = [(f, d) for f, d in sorted_feats if d['positive_ic_ratio'] < 0.6 or d['n_observations'] < 10]
for f, d in noisy:
    print(f"  {f}: mean_ic={d['mean_ic']:.6f}, pos_ratio={d['positive_ic_ratio']:.4f}, n_obs={d['n_observations']}")

print("\n" + "=" * 80)
print("3. WORST LONG WEEKS ANALYSIS")
print("=" * 80)

# Sort by worst long return
sorted_by_long = sorted(active_weeks, key=lambda w: float(w['long_return_pct']))
worst_long_weeks = sorted_by_long[:5]

print("\nTop 5 worst long-return weeks:")
for w in worst_long_weeks:
    print(f"\n  Week {w['week']} ({w['week_start']}): regime={w['regime']}")
    print(f"    Long return:  {w['long_return_pct']}%")
    print(f"    Short return: {w['short_return_pct']}%")
    print(f"    Total return: {w['week_return_pct']}%")
    print(f"    IC: {w['ic']}")
    print(f"    Turnover: {w['turnover_pct']}%")
    print(f"    n_symbols: {w['n_symbols']}")

print("\n" + "=" * 80)
print("4. SCENARIO: BULL_TREND ONLY vs SHORTS ONLY")
print("=" * 80)

# BULL_TREND only
bull_weeks = regimes.get('BULL_TREND', [])
bull_total = sum(float(w['week_return_pct']) for w in bull_weeks)
bull_cum = 1.0
for w in bull_weeks:
    bull_cum *= (1 + float(w['week_return_pct']) / 100)
bull_cum_ret = (bull_cum - 1) * 100

print(f"\n--- ONLY Trade in BULL_TREND ---")
print(f"  Weeks traded: {len(bull_weeks)}")
print(f"  Simple sum of returns: {bull_total:.4f}%")
print(f"  Compounded return: {bull_cum_ret:.4f}%")
for w in bull_weeks:
    print(f"    Week {w['week']}: {w['week_return_pct']}% (long={w['long_return_pct']}%, short={w['short_return_pct']}%)")

# SHORTS ONLY (zero out longs)
short_only_cum = 1.0
for w in active_weeks:
    short_ret = float(w['short_return_pct']) / 100
    short_only_cum *= (1 + short_ret)
short_only_ret = (short_only_cum - 1) * 100

# Also compute shorts-only sum
short_sum = sum(float(w['short_return_pct']) for w in active_weeks)

print(f"\n--- ONLY Go SHORT (no longs) ---")
print(f"  Compounded return: {short_only_ret:.4f}%")
print(f"  Simple sum of short returns: {short_sum:.4f}%")

# Compare to actual
actual_cum = 1.0
for w in active_weeks:
    actual_cum *= (1 + float(w['week_return_pct']) / 100)
actual_ret = (actual_cum - 1) * 100
print(f"\n  Actual total return (long+short): {actual_ret:.4f}%")
print(f"  Shorts-only improvement: {short_only_ret - actual_ret:.4f}pp")

print("\n" + "=" * 80)
print("5. SCENARIO: TOP 3 FEATURES ONLY (kyle_lambda, amihud, ret_3d)")
print("=" * 80)

print("""
NOTE: This scenario cannot be computed exactly from the available data.
The weekly returns and PnL already incorporate all 13 features in the model.
We do not have per-feature contribution to returns.

However, we can note:
- These 3 features have the highest mean IC:
  - kyle_lambda: mean_ic=0.1583, pos_ratio=0.857, n_obs=14
  - amihud:      mean_ic=0.1065, pos_ratio=0.714, n_obs=14
  - ret_3d:      mean_ic=0.0724, pos_ratio=0.714, n_obs=14

- Combined they represent the most predictive signals
- A model with only these 3 would likely have:
  - Less overfitting (fewer parameters)
  - More stable feature selection
  - Potentially lower OOS IC degradation

The current model shows severe IC degradation: avg IC drop from +0.103 (IS) to -0.039 (OOS)
= 138% degradation. With only top 3 stable features, this could plausibly be reduced.

Estimated (speculative): If the model used only top 3 features,
the OOS IC might be ~50-70% of the current (since these 3 dominate the signal),
but with much less degradation. Rough estimate: OOS IC of +0.02 to +0.05
instead of -0.035, which could translate to annualized returns of 5-15%
vs the current 6.04% (which is dragged down by BEAR_TREND losses).
""")

# Additional useful computation: what if we skip BEAR_TREND weeks?
print("\n" + "=" * 80)
print("BONUS: Skip BEAR_TREND weeks (only trade non-BEAR)")
print("=" * 80)
non_bear_cum = 1.0
for w in active_weeks:
    if w['regime'] != 'BEAR_TREND':
        non_bear_cum *= (1 + float(w['week_return_pct']) / 100)
non_bear_ret = (non_bear_cum - 1) * 100
non_bear_weeks = [w for w in active_weeks if w['regime'] != 'BEAR_TREND']
print(f"  Weeks traded: {len(non_bear_weeks)}")
print(f"  Compounded return: {non_bear_ret:.4f}%")
print(f"  BEAR weeks skipped: {len(regimes.get('BEAR_TREND', []))}")
print(f"  BEAR weeks total loss: {sum(float(w['week_return_pct']) for w in regimes.get('BEAR_TREND', [])):.4f}%")

# Skip LOW_VOL_GRIND with negative IC
print("\n" + "=" * 80)
print("BONUS: Only trade when IC > 0 (perfect foresight)")
print("=" * 80)
perfect_cum = 1.0
perfect_weeks = 0
for w in active_weeks:
    if float(w['ic']) > 0:
        perfect_cum *= (1 + float(w['week_return_pct']) / 100)
        perfect_weeks += 1
perfect_ret = (perfect_cum - 1) * 100
print(f"  Weeks with positive IC: {perfect_weeks}/{len(active_weeks)}")
print(f"  Compounded return: {perfect_ret:.4f}%")
