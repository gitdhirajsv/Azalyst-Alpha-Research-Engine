#!/usr/bin/env python3
"""
Azalyst V7 startup validation.

Checks the directories, Python dependencies, local modules, and the current
V7 runtime configuration before launching the engine.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_directories() -> bool:
    _print_header("CHECKING DIRECTORIES")
    required_dirs = ["data", "feature_cache", "results_v7"]
    all_ok = True

    for name in required_dirs:
        path = ROOT / name
        if path.exists():
            file_count = sum(1 for child in path.iterdir() if child.is_file())
            print(f"  OK  {name}/ exists ({file_count} files)")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  WARN {name}/ missing -> created")
            all_ok = False

    parquet_count = len(list((ROOT / "data").glob("*.parquet")))
    if parquet_count:
        print(f"  OK  data/ contains {parquet_count} parquet files")
    else:
        print("  WARN data/ has no parquet files yet")
        all_ok = False

    return all_ok


def check_imports() -> bool:
    _print_header("CHECKING PYTHON MODULES")
    modules = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "pyarrow",
        "xgboost",
    ]
    all_ok = True

    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  OK  {module_name}")
        except Exception as exc:
            print(f"  FAIL {module_name}: {exc}")
            all_ok = False

    return all_ok


def check_local_modules() -> bool:
    _print_header("CHECKING LOCAL MODULES")
    required_files = [
        "azalyst_v7_engine.py",
        "azalyst_v5_engine.py",
        "azalyst_factors_v2.py",
        "azalyst_train.py",
        "azalyst_risk.py",
        "azalyst_db.py",
        "azalyst_ic_filter.py",
        "azalyst_tf_utils.py",
        "build_feature_cache.py",
        "VIEW_TRAINING.py",
        "RUN_AZALYST.bat",
    ]
    all_ok = True

    for filename in required_files:
        if (ROOT / filename).exists():
            print(f"  OK  {filename}")
        else:
            print(f"  FAIL {filename} missing")
            all_ok = False

    print("\n  Testing imports...")
    import_checks = [
        ("azalyst_factors_v2", "build_features"),
        ("azalyst_db", "AzalystDB"),
        ("azalyst_risk", "RiskManager"),
        ("azalyst_train", "compute_ic"),
    ]
    for module_name, attr_name in import_checks:
        try:
            module = importlib.import_module(module_name)
            getattr(module, attr_name)
            print(f"    OK  {module_name}.{attr_name}")
        except Exception as exc:
            print(f"    FAIL {module_name}.{attr_name}: {exc}")
            all_ok = False

    return all_ok


def check_config() -> bool:
    _print_header("CHECKING V7 CONFIGURATION")
    engine_path = ROOT / "azalyst_v7_engine.py"
    if not engine_path.exists():
        print("  FAIL azalyst_v7_engine.py missing")
        return False

    content = engine_path.read_text(encoding="utf-8", errors="replace")
    checks = {
        'RESULTS_DIR = "./results_v7"': "results directory is results_v7",
        'CACHE_DIR   = "./feature_cache"': "feature cache directory is feature_cache",
        "ROLLING_WINDOW_WEEKS = 13": "rolling window is 13 weeks",
        "RETRAIN_WEEKS = 13": "retrain cadence is 13 weeks",
        "DEFAULT_TOP_N  = 5": "default portfolio is top-5 per side",
    }
    all_ok = True

    for needle, description in checks.items():
        if needle in content:
            print(f"  OK  {description}")
        else:
            print(f"  WARN could not confirm {description}")
            all_ok = False

    print("  INFO Existing feature_cache/ can be reused unless factors, timeframe, or raw data changed.")
    return all_ok


def main() -> int:
    print("\n" + "=" * 70)
    print("AZALYST ALPHA RESEARCH ENGINE - V7 STARTUP VALIDATION")
    print("=" * 70)

    checks = [
        ("Directories", check_directories),
        ("Python Modules", check_imports),
        ("Local Modules", check_local_modules),
        ("V7 Configuration", check_config),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            results[name] = False

    _print_header("SUMMARY")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4} {name}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll checks passed. Suggested launch:")
        print("  python azalyst_v7_engine.py --no-gpu --top-n 5")
        return 0

    print("\nSome checks need attention before launch.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
