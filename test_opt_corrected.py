#!/usr/bin/env python3
"""
Corrected optimization: Raise MAX_DRAWDOWN_KILL threshold
Baseline: -6.23% on Y2 (with DD kill at -15%)
Expected: Better returns with DD kill at -30%
This allows strategy to recover from temporary drawdowns instead of halting
"""

import subprocess
import sys
import os
import pandas as pd

print("\n" + "="*80)
print("CORRECTED OPTIMIZATION: Raise MAX_DRAWDOWN_KILL from -15% to -30%")
print("="*80)
print("Baseline (Y2): -6.23% (triggered kill-switch 59 times)")
print("Expected: Better returns (allow temporary DD recovery)")
print("="*80 + "\n")

print("Running corrected optimization test...\n")

result = subprocess.run([
    sys.executable, "azalyst_v5_engine.py",
    "--data-dir", "test_50_data",
    "--feature-dir", "test_50_cache",
    "--out-dir", "test_50_results_opt_corrected",
    "--no-shap",
    "--no-gpu",
    "--max-dd", "-0.30"  # Raise threshold from -0.15 to -0.30
], capture_output=False)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✓ CORRECTED OPTIMIZATION TEST COMPLETED")
    print("="*80)
    
    # Compare results
    baseline_file = "test_50_results_final/weekly_summary_v4.csv"
    optimized_file = "test_50_results_opt_corrected/weekly_summary_v4.csv"
    
    if os.path.exists(baseline_file) and os.path.exists(optimized_file):
        baseline_df = pd.read_csv(baseline_file)
        optimized_df = pd.read_csv(optimized_file)
        
        baseline_return = baseline_df['week_return_pct'].sum()
        optimized_return = optimized_df['week_return_pct'].sum()
        improvement = optimized_return - baseline_return
        
        # Focus on Y2 which is comparable
        print(f"\nTOTAL (all weeks):")
        print(f"  BASELINE:     {baseline_return:.2f}%")
        print(f"  OPTIMIZED:    {optimized_return:.2f}%")
        print(f"  IMPROVEMENT:  {improvement:+.2f}%")
        
        # Check kill-switch regime
        baseline_ks = (baseline_df['regime'] == 'KILL_SWITCH').sum()
        optimized_ks = (optimized_df['regime'] == 'KILL_SWITCH').sum()
        
        print(f"\nKill-switch triggered weeks:")
        print(f"  Baseline: {baseline_ks}/{len(baseline_df)}")
        print(f"  Optimized: {optimized_ks}/{len(optimized_df)}")
        print(f"  Reduction: {baseline_ks - optimized_ks} weeks (more trading)")
        
        if improvement > 0:
            print(f"\n✓ SUCCESS: Raising DD threshold improved returns by {improvement:.2f}%")
        elif improvement < 0:
            print(f"\n⚠ Raising DD threshold decreased returns by {abs(improvement):.2f}%")
            print("  The original -15% threshold may have been optimal")
        else:
            print(f"\n→ No change in returns")
    else:
        print("✗ Could not find result files to compare")
else:
    print(f"\n✗ Engine failed with return code {result.returncode}")
    sys.exit(1)
