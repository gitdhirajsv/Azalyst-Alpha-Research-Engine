import pandas as pd
import json
import numpy as np

# Load final results
perf = json.load(open('test_50_results_final/performance_v4.json'))
weekly = pd.read_csv('test_50_results_final/weekly_summary_v4.csv')
trades = pd.read_csv('test_50_results_final/all_trades_v4.csv')

print('=== PERFORMANCE DIAGNOSIS ===')
print(f'Total Return: {perf["total_return_pct"]:.2f}%')
print(f'IC Mean: {perf["ic_mean"]:.4f}')
print(f'IC Positive: {perf["ic_positive_pct"]:.1f}%')
print(f'Sharpe: {perf["sharpe"]:.4f}')
print(f'Max DD: {perf["max_drawdown_pct"]:.2f}%')
print()

# Analyze by regime
print('=== BY REGIME ===')
for regime in sorted(weekly['regime'].unique()):
    regime_data = weekly[weekly['regime'] == regime]
    print(f'{regime}: {len(regime_data)} weeks, avg ret={regime_data["week_return_pct"].mean():.2f}%, avg IC={regime_data["ic"].mean():.4f}')
print()

# Win rate analysis
print('=== WIN RATE ===')
positive_weeks = (weekly['week_return_pct'] > 0).sum()
total_weeks = len(weekly[weekly['regime'] != 'KILL_SWITCH'])
print(f'Positive weeks: {positive_weeks}/{total_weeks} ({100*positive_weeks/total_weeks:.1f}%)')
if len(weekly[weekly['week_return_pct'] > 0]) > 0:
    print(f'Avg win: {weekly[weekly["week_return_pct"]>0]["week_return_pct"].mean():.2f}%')
if len(weekly[weekly['week_return_pct'] < 0]) > 0:
    print(f'Avg loss: {weekly[weekly["week_return_pct"]<0]["week_return_pct"].mean():.2f}%')
print()

# IC-PnL alignment
print('=== IC vs PnL ALIGNMENT ===')
non_ks = weekly[weekly['regime'] != 'KILL_SWITCH']
ic_corr = non_ks['ic'].corr(non_ks['week_return_pct'])
print(f'IC-PnL correlation: {ic_corr:.4f}')
print()

# Feature stability
imp = pd.read_csv('results/feature_importance_v4_base.csv')
imp.rename(columns={'Unnamed: 0': 'feature'}, inplace=True)
print('=== TOP 10 FEATURES (Base Model) ===')
print(imp.nlargest(10, 'importance')[['feature', 'importance']])
print()

# Trade analysis
print('=== TRADE ANALYSIS ===')
print(f'Total trades: {len(trades)}')
print(f'Long trades: {len(trades[trades["signal"] == "BUY"])}')
print(f'Short trades: {len(trades[trades["signal"] == "SELL"])}')
trades['pnl'] = trades['pnl_percent']
print(f'Avg long pnl: {trades[trades["signal"] == "BUY"]["pnl"].mean():.4f}%')
print(f'Avg short pnl: {trades[trades["signal"] == "SELL"]["pnl"].mean():.4f}%')
