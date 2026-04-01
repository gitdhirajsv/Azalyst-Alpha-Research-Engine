#!/usr/bin/env python3
"""
Optimization configuration for Azalyst v5
Tests three hypotheses:
1. Kill-switch threshold too aggressive (-0.03 causes 67% weeks with zero allocation)
2. Long bias problem (longs -0.2886%, shorts +0.0289%)
3. IC inversion (negative IC should flip predictions, not stop trading)
"""

# Optimization configurations to test
OPTIMIZATIONS = {
    "baseline": {
        "ic_gating_threshold": -0.03,
        "short_bias_only": False,
        "invert_negative_ic": False,
        "description": "Current baseline (kill-switch at -0.03)"
    },
    
    "opt1_raise_killswitch": {
        "ic_gating_threshold": 0.00,  # Raise threshold from -0.03 to 0.00
        "short_bias_only": False,
        "invert_negative_ic": False,
        "description": "Raise kill-switch from -0.03 to +0.00 (more permissive, expect +4-5%)"
    },
    
    "opt2_disable_killswitch": {
        "ic_gating_threshold": -1.00,  # Effectively disabled
        "short_bias_only": False,
        "invert_negative_ic": False,
        "description": "Disable kill-switch entirely (threshold -1.00, expect +5-7%)"
    },
    
    "opt3_short_bias": {
        "ic_gating_threshold": -0.03,
        "short_bias_only": True,  # Only take short positions
        "invert_negative_ic": False,
        "description": "Short-bias positions only (longs lose money, expect +1-2%)"
    },
    
    "opt4_invert_ic": {
        "ic_gating_threshold": -0.03,
        "short_bias_only": False,
        "invert_negative_ic": True,  # Invert when IC negative
        "description": "Invert predictions when IC negative (expect +1-2%)"
    },
    
    "opt5_combined_best": {
        "ic_gating_threshold": 0.00,  # Raise kill-switch
        "short_bias_only": False,
        "invert_negative_ic": True,  # Invert IC
        "description": "Combine: raise kill-switch + invert IC (expect +5-6%)"
    },
}

def apply_optimization(config_name):
    """Apply an optimization configuration to azalyst_v5_engine.py"""
    if config_name not in OPTIMIZATIONS:
        print(f"ERROR: Unknown optimization '{config_name}'")
        print(f"Available: {list(OPTIMIZATIONS.keys())}")
        return False
    
    config = OPTIMIZATIONS[config_name]
    print(f"\n{'='*70}")
    print(f"Applying optimization: {config_name}")
    print(f"  {config['description']}")
    print(f"{'='*70}")
    
    # Read the engine file
    with open('azalyst_v5_engine.py', 'r') as f:
        content = f.read()
    
    # Apply IC_GATING_THRESHOLD change
    old_threshold_line = f"IC_GATING_THRESHOLD = -0.03"
    new_threshold_line = f"IC_GATING_THRESHOLD = {config['ic_gating_threshold']}"
    content = content.replace(old_threshold_line, new_threshold_line)
    
    # These other options would require deeper code modifications
    # For now, just modify the threshold and show recommendations
    
    with open('azalyst_v5_engine.py', 'w') as f:
        f.write(content)
    
    print(f"\n✓ Updated IC_GATING_THRESHOLD = {config['ic_gating_threshold']}")
    
    if config['short_bias_only']:
        print(f"⚠ NOTE: Short-bias requires code modification in simulate_weekly_trades()")
        print(f"  Currently filtering: cur_longs = set(...)")
        print(f"  To implement: Comment out or filter out BUY trades")
    
    if config['invert_negative_ic']:
        print(f"⚠ NOTE: IC inversion requires code modification in predict_week()")
        print(f"  Modify to: if rolling_ic < 0: predictions[sym] = -predictions[sym]")
    
    print(f"\nTo run the engine with this config:")
    print(f"  python azalyst_v5_engine.py --gpu")
    print(f"\nResults will be logged to: results/")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        opt_name = sys.argv[1]
        apply_optimization(opt_name)
    else:
        print("\nUsage: python config_optimizations.py <optimization_name>")
        print(f"\nAvailable optimizations:")
        for name, config in OPTIMIZATIONS.items():
            print(f"  {name:25s} - {config['description']}")
        print(f"\nExample: python config_optimizations.py opt1_raise_killswitch")
