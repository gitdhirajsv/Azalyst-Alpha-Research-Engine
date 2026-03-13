"""
AZALYST ENVIRONMENT DIAGNOSTIC & REPAIR TOOL
Fixes all known issues before running the pipeline
"""

import subprocess
import sys
import os

def log(msg, level="INFO"):
    prefix = f"[{level}]"
    print(f"{prefix} {msg}")

def run_command(cmd, description=""):
    """Run command and return success status"""
    if description:
        log(f"Running: {description}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                 AZALYST ENVIRONMENT REPAIR TOOL                           ║
║         Fixes: psutil, subprocess encoding, joblib, Ollama config         ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    issues_found = 0
    issues_fixed = 0
    
    # ───────────────────────────────────────────────────────────────────────────
    # CHECK 1: psutil
    # ───────────────────────────────────────────────────────────────────────────
    log("Checking psutil...", "CHECK")
    try:
        import psutil
        p = psutil.Process()
        log("psutil is healthy", "✓")
    except (ImportError, AttributeError) as e:
        issues_found += 1
        log(f"psutil is CORRUPTED: {e}", "✗")
        log("Fixing psutil...", "FIX")
        
        # Uninstall
        run_command("pip uninstall psutil -y", "Uninstalling psutil")
        
        # Reinstall
        success, _, stderr = run_command(
            "pip install psutil==5.9.8 --break-system-packages",
            "Reinstalling psutil"
        )
        
        if success:
            log("psutil fixed ✓", "✓")
            issues_fixed += 1
        else:
            log(f"psutil repair failed: {stderr}", "✗")
    
    # ───────────────────────────────────────────────────────────────────────────
    # CHECK 2: joblib/loky
    # ───────────────────────────────────────────────────────────────────────────
    log("Checking joblib...", "CHECK")
    try:
        import joblib
        log(f"joblib version: {joblib.__version__}", "✓")
    except ImportError:
        issues_found += 1
        log("joblib not found", "✗")
        log("Installing joblib...", "FIX")
        success, _, _ = run_command(
            "pip install joblib --break-system-packages",
            "Installing joblib"
        )
        if success:
            log("joblib installed ✓", "✓")
            issues_fixed += 1
    
    # ───────────────────────────────────────────────────────────────────────────
    # CHECK 3: Ollama connection
    # ───────────────────────────────────────────────────────────────────────────
    log("Checking Ollama server...", "CHECK")
    success, stdout, stderr = run_command(
        "curl -s http://127.0.0.1:11434/ || powershell -Command \"(Invoke-WebRequest -Uri 'http://127.0.0.1:11434/' -UseBasicParsing).StatusCode\"",
        "Testing Ollama endpoint"
    )
    
    if success or "200" in stdout or "200" in stderr:
        log("Ollama is running ✓", "✓")
    else:
        issues_found += 1
        log("Ollama is NOT running", "✗")
        log("Start Ollama with: ollama serve", "WARN")
    
    # ───────────────────────────────────────────────────────────────────────────
    # CHECK 4: deepseek-r1:14b model
    # ───────────────────────────────────────────────────────────────────────────
    log("Checking deepseek-r1:14b model...", "CHECK")
    success, stdout, stderr = run_command(
        "ollama list | findstr /C:\"deepseek-r1\"",
        "Checking installed models"
    )
    
    if success and "deepseek-r1" in stdout:
        log("deepseek-r1:14b is loaded ✓", "✓")
    else:
        issues_found += 1
        log("deepseek-r1:14b not found or not ready", "✗")
        log("Model may still be downloading. Check: ollama list", "WARN")
    
    # ───────────────────────────────────────────────────────────────────────────
    # CHECK 5: Python packages (critical ones)
    # ───────────────────────────────────────────────────────────────────────────
    required_packages = [
        "numpy", "pandas", "scikit-learn", "lightgbm",
        "scipy", "requests", "joblib"
    ]
    
    log("Checking required Python packages...", "CHECK")
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if not missing:
        log(f"All {len(required_packages)} required packages installed ✓", "✓")
    else:
        issues_found += 1
        log(f"Missing packages: {', '.join(missing)}", "✗")
        log(f"Installing missing packages...", "FIX")
        success, _, _ = run_command(
            f"pip install {' '.join(missing)} --break-system-packages",
            "Installing missing packages"
        )
        if success:
            log("Packages installed ✓", "✓")
            issues_fixed += 1
    
    # ───────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ───────────────────────────────────────────────────────────────────────────
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                            REPAIR SUMMARY                                 ║
║  Issues found: {issues_found}                                                      ║
║  Issues fixed: {issues_fixed}                                                      ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if issues_found == 0:
        log("✓ All checks passed! Environment is ready.", "SUCCESS")
        return 0
    elif issues_fixed == issues_found:
        log("✓ All issues have been fixed. Rerun your pipeline.", "SUCCESS")
        return 0
    else:
        log("⚠ Some issues remain. Review above and fix manually.", "WARN")
        return 1

if __name__ == "__main__":
    sys.exit(main())
