#!/usr/bin/env python3
"""
Test optimization: Disabled kill-switch
Expected result: + 4-7% gain (59 gated weeks now trade at ~0.5% median = +2.95%)
Run on recent data (weeks 81-103, ~6 months) for quick feedback

Current: -0.95% on weeks 81-103
Expected with kill-switch removed: ~+2-4% (if the median holds)
"""

import subprocess
import sys

print("\n" + "="*80)
print("TESTING OPTIMIZATION: Disabled Kill-Switch")
print("="*80)
print("Baseline on weeks 81-103: -0.95%")
print("Expected with kill-switch disabled: ~+2-4% (median trading week * gated weeks)")
print("Mechanism: IC_GATING_THRESHOLD changed from -0.03 to -1.00")
print("="*80 + "\n")

print("Running engine with disabled kill-switch...")
print("(This will take ~2-5 minutes on GPU)...\n")

result = subprocess.run([
    sys.executable, "azalyst_v5_engine.py",
    "--gpu"
    # Could add --no-shap to speed up
], capture_output=False)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✓ Engine completed successfully")
    print("="*80)
    print("\nTo analyze results:")
    print("  python diagnose.py")
    print("  python estimate_optimization_gain.py")
    print("\nTo compare against baseline:")
    print("  git diff results/weekly_summary_v4.csv")
else:
    print(f"\n✗ Engine failed with return code {result.returncode}")
    sys.exit(1)
