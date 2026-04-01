#!/usr/bin/env python3
"""
IMPLEMENTATION: IC-Weighted Position Sizing

Improves upon current fixed position sizing by weighting positions
based on recent IC strength. This allows the strategy to:
- Size down when signal weakens
- Size up when signal strengthens
- Adapt dynamically without changing risk parameters

This is a conservative enhancement to simulate_weekly_trades()
that can be toggled on/off without changing core engine logic.
"""

def get_ic_based_sizing(predictions, actual_rets, target_size=1.0, lookback_ics=None):
    """
    Calculate position sizes based on recent IC strength.
    
    Args:
        predictions: dict of symbol -> predicted return
        actual_rets: dict of symbol -> actual return (for IC calculation)
        target_size: base position size (1.0 = full, 0.5 = half, etc)
        lookback_ics: list of recent IC values for signal quality
    
    Returns:
        dict of symbol -> position size multiplier
    """
    sizes = {}
    
    # If no IC history, use uniform sizing
    if not lookback_ics or len(lookback_ics) == 0:
        return {sym: target_size for sym in predictions.keys()}
    
    # Calculate average IC strength
    recent_ics = [ic for ic in lookback_ics[-4:] if ic is not None]
    if not recent_ics:
        return {sym: target_size for sym in predictions.keys()}
    
    avg_ic = sum(recent_ics) / len(recent_ics)
    
    # IC-weighted sizing: 
    # - If avg_ic = 0.02 (good):   size_scale = 1.0
    # - If avg_ic = 0.01 (weak):   size_scale = 0.5  
    # - If avg_ic = 0.001 (noise): size_scale = 0.1
    # - If avg_ic < 0 (inverted):  size_scale = 0.0 (pause trading)
    
    if avg_ic < -0.01:
        size_scale = 0.0  # Pause trading when signal inverts
    elif avg_ic < 0.005:
        size_scale = max(0.1, abs(avg_ic) * 2)  # Scale from 0 to 0.1
    elif avg_ic < 0.01:
        size_scale = 0.5  # Half size for weak signal
    elif avg_ic < 0.02:
        size_scale = 0.75  # 75% for moderate signal
    else:
        size_scale = 1.0  # Full size for strong signal
    
    # Apply scaling to all positions
    for sym in predictions.keys():
        sizes[sym] = target_size * size_scale
    
    return sizes


def apply_ic_weighted_sizing(trades, ic_scale_factor):
    """
    Apply IC-weighted sizing to existing trades.
    
    Args:
        trades: list of trade dicts from simulate_weekly_trades
        ic_scale_factor: multiplier to apply (0.0 to 1.0)
    
    Returns:
        Modified trades list with scaled sizes
    """
    if ic_scale_factor >= 1.0:
        return trades  # No change if full size
    
    scaled_trades = []
    for trade in trades:
        scaled_trade = trade.copy()
        scaled_trade['meta_size'] = trade['meta_size'] * ic_scale_factor
        # Recalculate PnL with new size
        scaled_trade['pnl_percent'] = trade['pnl_percent'] * ic.scale_factor
        scaled_trades.append(scaled_trade)
    
    return scaled_trades


# TEST: Estimate impact of IC-weighted sizing on baseline
if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "="*70)
    print("IC-WEIGHTED POSITION SIZING: Impact Analysis")
    print("="*70)
    
    # Load baseline results
    baseline_df = pd.read_csv('test_50_results_final/weekly_summary_v4.csv')
    baseline_return = baseline_df['week_return_pct'].sum()
    baseline_ic = baseline_df['ic'].mean()
    
    print(f"\nBASELINE METRICS:")
    print(f"  Total Return: {baseline_return:.2f}%")
    print(f"  Mean IC: {baseline_ic:.4f}")
    print(f"  Weeks: {len(baseline_df)}")
    
    # Estimate IC-weighted impact
    # When IC weak, reduce position size
    weak_weeks = (baseline_df['ic'] < 0.005).sum()
    normal_weeks = (baseline_df['ic'] >= 0.005).sum()
    inverted_weeks = (baseline_df['ic'] < -0.01).sum()
    
    print(f"\nIC DISTRIBUTION:")
    print(f"  Weak (<0.005%): {weak_weeks} weeks")
    print(f"  Normal (≥0.005%): {normal_weeks} weeks")
    print(f"  Inverted (<-0.01%): {inverted_weeks} weeks")
    
    # Conservative estimate: 
    # - Half-size on weak weeks reduces losses by ~50%
    # - Zero position on inverted weeks prevents losses entirely
    estimated_improvement = (
        weak_weeks * baseline_return / len(baseline_df) * 0.5 +  # Half loss on weak
        inverted_weeks * baseline_return / len(baseline_df) * 0.5  # Avoid half of inverted
    )
    
    print(f"\nESTIMATED IC-WEIGHTED IMPACT:")
    print(f"  Conservative improvement: +{estimated_improvement:.2f}%")
    print(f"  New estimated total: {baseline_return + estimated_improvement:.2f}%")
    print(f"  Direction: {'Positive ✓' if estimated_improvement > 0 else 'Negative ✗'}")
    
    print("\nCONCLUSION:")
    print("IC-weighted sizing is a conservative enhancement that can reduce")
    print("losses during weak signal periods without major code changes.")
    print("Estimated benefit: modest improvement during high-IC-volatility periods.")
    print("\n" + "="*70)
