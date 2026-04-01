#!/usr/bin/env python3
"""
Test optimizations for Azalyst v5:
1. Disable kill-switch (allow trading even at low IC)
2. Short-bias positions only (filter out long trades)
3. Invert predictions when IC negative (turn anticorrelated signal into correlated)
"""

import sqlite3
import numpy as np
import pandas as pd
from azalyst_v5_engine import predict_week, simulate_weekly_trades

def test_optimization(name, run_config):
    """Run optimization test and report results"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Config: {run_config}")
    print(f"{'='*60}")
    
    conn = sqlite3.connect('azalyst.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM runs ORDER BY id DESC LIMIT 1')
    run = c.fetchone()
    run_id = run[0]
    
    # Get all weekly results
    c.execute('''
    SELECT week_num, rolling_ic_mean, total_return, positive_weeks 
    FROM weekly_metrics WHERE run_id = ? ORDER BY week_num
    ''', (run_id,))
    weeks_data = c.fetchall()
    
    total_return = sum([w[2] for w in weeks_data])
    weeks_positive = sum([w[3] for w in weeks_data])
    avg_ic = np.mean([w[1] for w in weeks_data])
    
    print(f"Total Return: {total_return:.2f}%")
    print(f"Positive weeks: {weeks_positive}/{len(weeks_data)} ({weeks_positive/len(weeks_data)*100:.1f}%)")
    print(f"Avg IC: {avg_ic:.4f}")
    print(f"Max DD: {min([w[2] for w in weeks_data]):.2f}%")
    
    conn.close()
    return {
        'name': name,
        'total_return': total_return,
        'positive_weeks': weeks_positive,
        'total_weeks': len(weeks_data),
        'avg_ic': avg_ic
    }

# Test 1: Current baseline (for reference)
print("\n" + "="*60)
print("BASELINE (Current Kill-Switch Enabled)")
print("="*60)
results = []

baseline = test_optimization(
    "Baseline (Kill-Switch ON)",
    "IC threshold: -0.03, Long+Short, Position: rank-based"
)
results.append(baseline)

print("\n\nTo test optimization scenarios, we need to modify azalyst_v5_engine.py:")
print("1. Add parameter: disable_kill_switch=False")
print("2. Add parameter: short_bias_only=False")
print("3. Add parameter: invert_on_negative_ic=False")
print("\nThen run full backtest with each configuration.")

# Print summary
print("\n" + "="*60)
print("OPTIMIZATION TARGETS")
print("="*60)
print("✓ Baseline: -8.79% return, 22/44 positive weeks (50%)")
print("✓ Kill-Switch blocking: 59 weeks (67% of time)")
print("✓ Long bias: avg -0.2886% vs Short: avg +0.0289%")
print("\nTest sequence:")
print("1. Raise kill-switch threshold from -0.03 to 0.00")
print("2. Try short-bias only (filter long trades)")
print("3. Invert predictions when rolling IC < 0")
print("4. Combine: short-bias + inverted IC")
print("5. Disable kill-switch entirely (if safe)")
