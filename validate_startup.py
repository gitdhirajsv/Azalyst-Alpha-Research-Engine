#!/usr/bin/env python3
"""
Azalyst Startup Validation - Check all critical systems before running
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

def check_directories():
    """Check if required directories exist"""
    required_dirs = ["data", "feature_cache", "results"]
    print("\n" + "="*70)
    print("CHECKING DIRECTORIES")
    print("="*70)
    
    all_ok = True
    for d in required_dirs:
        if os.path.exists(d):
            file_count = len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
            print(f"  ✓ {d}/ exists ({file_count} files)")
        else:
            print(f"  ✗ {d}/ NOT FOUND")
            os.makedirs(d, exist_ok=True)
            print(f"    Created.")
            all_ok = False
    
    return all_ok

def check_imports():
    """Check if all required Python modules import correctly"""
    print("\n" + "="*70)
    print("CHECKING PYTHON MODULES")
    print("="*70)
    
    modules = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("xgboost", "xgb"),
        ("scipy", "scipy"),
    ]
    
    all_ok = True
    for module_name, alias in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False
    
    return all_ok

def check_local_modules():
    """Check if all local Azalyst modules exist and can be imported"""
    print("\n" + "="*70)
    print("CHECKING LOCAL MODULES")
    print("="*70)
    
    required_modules = [
        "azalyst_db.py",
        "azalyst_factors_v2.py",
        "azalyst_risk.py",
        "azalyst_pump_dump.py",
        "azalyst_train.py",
        "azalyst_v5_engine.py",
    ]
    
    all_ok = True
    for module_file in required_modules:
        if os.path.exists(module_file):
            print(f"  ✓ {module_file}")
        else:
            print(f"  ✗ {module_file} NOT FOUND")
            all_ok = False
    
    # Try importing key modules
    print("\n  Testing imports...")
    try:
        from azalyst_factors_v2 import build_features
        print(f"    ✓ azalyst_factors_v2.build_features()")
    except Exception as e:
        print(f"    ✗ azalyst_factors_v2: {e}")
        all_ok = False
    
    try:
        from azalyst_db import AzalystDB
        print(f"    ✓ azalyst_db.AzalystDB()")
    except Exception as e:
        print(f"    ✗ azalyst_db: {e}")
        all_ok = False
    
    try:
        from azalyst_risk import RiskManager
        print(f"    ✓ azalyst_risk.RiskManager()")
    except Exception as e:
        print(f"    ✗ azalyst_risk: {e}")
        all_ok = False
    
    try:
        from azalyst_pump_dump import compute_pump_dump_scores
        print(f"    ✓ azalyst_pump_dump.compute_pump_dump_scores()")
    except Exception as e:
        print(f"    ✗ azalyst_pump_dump: {e}")
        all_ok = False
    
    return all_ok

def check_config():
    """Check IC_GATING_THRESHOLD is properly configured"""
    print("\n" + "="*70)
    print("CHECKING ENGINE CONFIGURATION")
    print("="*70)
    
    try:
        with open("azalyst_v5_engine.py", "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        # Check for the optimization
        if "IC_GATING_THRESHOLD = -1.00" in content:
            print(f"  ✓ Kill-switch DISABLED (IC_GATING_THRESHOLD = -1.00)")
            return True
        elif "IC_GATING_THRESHOLD = -0.03" in content:
            print(f"  ⚠ Kill-switch ENABLED (IC_GATING_THRESHOLD = -0.03)")
            print(f"    This will block ~67% of weeks. Consider disabling in config.")
            return False
        else:
            print(f"  ⚠ Could not parse IC_GATING_THRESHOLD")
            return False
    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        return False

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  AZALYST ALPHA RESEARCH ENGINE - STARTUP VALIDATION".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    checks = [
        ("Directories", check_directories),
        ("Python Modules", check_imports),
        ("Local Modules", check_local_modules),
        ("Engine Configuration", check_config),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"    ✗ Exception in {name}: {e}")
            results[name] = False
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(results.values())
    
    print("="*70)
    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Engine is ready to run!")
        print("\n  To start the engine:")
        print("    python azalyst_v5_engine.py --gpu")
        print("    (or use RUN_AZALYST.bat on Windows)")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Fix issues above before running")
        return 1

if __name__ == "__main__":
    sys.exit(main())
