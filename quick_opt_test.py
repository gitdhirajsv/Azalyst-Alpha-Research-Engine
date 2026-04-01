#!/usr/bin/env python3
"""
Quick optimization test on 50-symbol dataset (pre-cached)
Baseline: -8.79% (from test_50_results_final/)
Expected with kill-switch disabled: +4-7% improvement
Runtime: ~5 minutes instead of ~1 hour
"""

import subprocess
import sys
import os
import shutil
import pandas as pd
import json

print("\n" + "="*80)
print("QUICK OPTIMIZATION TEST: Disabled Kill-Switch (50-Symbol Dataset)")
print("="*80)
print("Baseline: -8.79% (from test_50_results_final/)")
print("Expected: +4-7% improvement (disable IC_GATING_THRESHOLD)")
print("="*80 + "\n")

# Use the 50-symbol test cache for speed
if os.path.exists("test_50_cache"):
    print("✓ Using pre-built 50-symbol test cache (faster)")
else:
    print("✗ 50-symbol cache not found, creating...")
    if os.path.exists("feature_cache"):
        shutil.copytree("feature_cache", "test_50_cache")

print("\nRunning optimization test (this will take ~5 minutes)...\n")

result = subprocess.run([
    sys.executable, "azalyst_v5_engine.py",
    "--data-dir", "test_50_data",
    "--feature-dir", "test_50_cache",
    "--out-dir", "test_50_results_optimized",
    "--no-shap",
    "--no-gpu"  # CPU only for stability
], capture_output=False)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✓ OPTIMIZATION TEST COMPLETED")
    print("="*80)
    
    # Compare results
    baseline_file = "test_50_results_final/weekly_summary_v4.csv"
    optimized_file = "test_50_results_optimized/weekly_summary_v4.csv"
    
    if os.path.exists(baseline_file) and os.path.exists(optimized_file):
        baseline_df = pd.read_csv(baseline_file)
        optimized_df = pd.read_csv(optimized_file)
        
        baseline_return = baseline_df['week_return_pct'].sum()
        optimized_return = optimized_df['week_return_pct'].sum()
        improvement = optimized_return - baseline_return
        
        print(f"\nBASELINE (kill-switch ON):     {baseline_return:.2f}%")
        print(f"OPTIMIZED (kill-switch OFF):   {optimized_return:.2f}%")
        print(f"IMPROVEMENT:                   {improvement:+.2f}%")
        print(f"GAIN %:                        {(improvement/abs(baseline_return)*100) if baseline_return != 0 else 'N/A':.1f}%")
        
        # Check kill-switch regime
        baseline_ks = (baseline_df['regime'] == 'KILL_SWITCH').sum()
        optimized_ks = (optimized_df['regime'] == 'KILL_SWITCH').sum()
        
        print(f"\nKill-switch triggered weeks:")
        print(f"  Baseline: {baseline_ks}/{len(baseline_df)}")
        print(f"  Optimized: {optimized_ks}/{len(optimized_df)}")
        
        if improvement > 0:
            print(f"\n✓ SUCCESS: Kill-switch removal improved returns by {improvement:.2f}%")
        else:
            print(f"\n⚠ Kill-switch removal had minimal impact ({improvement:+.2f}%)")
            print("  Consider testing Option 2: Short-bias only")
    else:
        print("✗ Could not find result files to compare")
else:
    print(f"\n✗ Engine failed with return code {result.returncode}")
    sys.exit(1)
