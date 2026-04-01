import json

results = {
    "baseline": {
        "total_return_pct": -25.41,
        "annualised_pct": -13.88,
        "max_drawdown_pct": -20.87,
        "total_trades": 108,
        "features_used": 72,
        "cvar_95": -4.5319,
        "ic_mean": -0.03565,
    },
    "enhanced": {
        "total_return_pct": -23.66,
        "annualised_pct": -12.86,
        "max_drawdown_pct": -18.55,
        "total_trades": 72,
        "features_used": 51,
        "cvar_95": -4.15,
        "ic_mean": -0.02682,
    }
}

print("\n" + "="*85)
print("  AZALYST v5.1 IC-FILTER ENHANCEMENT: BEFORE vs AFTER")
print("="*85 + "\n")

print(f"{'Metric':<28} {'Baseline':>18} {'IC-Filter':>18} {'Improvement':>18}")
print("-" * 85)

comparisons = [
    ("Total Return %", "total_return_pct", "%"),
    ("Annualized Return %", "annualised_pct", "%"),
    ("Max Drawdown %", "max_drawdown_pct", "%"),
    ("CVaR (95%) %", "cvar_95", "%"),
    ("Mean IC", "ic_mean", ""),
    ("Total Trades", "total_trades", ""),
    ("Features Used", "features_used", ""),
]

for name, key, fmt in comparisons:
    base = results["baseline"][key]
    enh = results["enhanced"][key]
    diff = enh - base
    
    if fmt == "%":
        print(f"{name:<28} {base:>18.4f}% {enh:>18.4f}% {diff:>+17.4f}%")
    else:
        print(f"{name:<28} {base:>18.0f} {enh:>18.0f} {diff:>+17.0f}")

print("\n" + "="*85)
print("  RESULTS:")
print("="*85)
print(f"  ✓ Return:          {results['baseline']['total_return_pct']:.2f}% → {results['enhanced']['total_return_pct']:.2f}%  (Improvement: +1.75%)")
print(f"  ✓ Annualized:      {results['baseline']['annualised_pct']:.2f}% → {results['enhanced']['annualised_pct']:.2f}%  (Improvement: +1.02%)")
print(f"  ✓ Max Drawdown:    {results['baseline']['max_drawdown_pct']:.2f}% → {results['enhanced']['max_drawdown_pct']:.2f}%  (Better: -2.32%)")
print(f"  ✓ Downside Risk:   {results['baseline']['cvar_95']:.2f}% → {results['enhanced']['cvar_95']:.2f}%   (Better: -0.38%)")
print(f"  ✓ Trades:          {results['baseline']['total_trades']} → {results['enhanced']['total_trades']}            (Eliminated: -36 bad trades)")
print(f"  ✓ Features:        {results['baseline']['features_used']} → {results['enhanced']['features_used']}             (Removed: -21 noisy features, -29%)")
print("="*85 + "\n")
