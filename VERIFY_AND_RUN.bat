@echo off
title AZALYST FIX VERIFICATION & RUN
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   AZALYST FIX VERIFICATION                              ║
echo  ║   Check all patches are in place before running team    ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ═════════════════════════════════════════════════════════════════════════════
::  STEP 1: Verify psutil
:: ═════════════════════════════════════════════════════════════════════════════
echo  [STEP 1/5] Checking psutil...
python -c "import psutil; p = psutil.Process(); print('  ✓ psutil is healthy')" 2>nul
if errorlevel 1 (
    echo  ✗ psutil is BROKEN
    echo  FIX: Run repair_environment.py
    pause
    exit /b 1
)

:: ═════════════════════════════════════════════════════════════════════════════
::  STEP 2: Verify Ollama is running
:: ═════════════════════════════════════════════════════════════════════════════
echo  [STEP 2/5] Checking Ollama server...
curl -s http://127.0.0.1:11434/ >nul 2>nul
if errorlevel 1 (
    echo  ✗ Ollama is NOT running
    echo  FIX: Start Ollama in another CMD: ollama serve
    pause
    exit /b 1
) else (
    echo  ✓ Ollama is running
)

:: ═════════════════════════════════════════════════════════════════════════════
::  STEP 3: Verify deepseek-r1:14b is loaded
:: ═════════════════════════════════════════════════════════════════════════════
echo  [STEP 3/5] Checking deepseek-r1:14b model...
ollama list 2>nul | findstr /C:"deepseek-r1" >nul
if errorlevel 1 (
    echo  ✗ deepseek-r1:14b is NOT loaded
    echo  FIX: Run in another CMD: ollama pull deepseek-r1:14b
    echo.
    echo  Note: This is ~8.3 GB, may take 10-30 minutes
    pause
    exit /b 1
) else (
    echo  ✓ deepseek-r1:14b is loaded
)

:: ═════════════════════════════════════════════════════════════════════════════
::  STEP 4: Verify UTF-8 patches are in place
:: ═════════════════════════════════════════════════════════════════════════════
echo  [STEP 4/5] Checking UTF-8 patches...

REM Check azalyst_autonomous_team.py for encoding='utf-8'
findstr /C:"encoding='utf-8'" azalyst_autonomous_team.py >nul 2>nul
if errorlevel 1 (
    echo  ✗ azalyst_autonomous_team.py is NOT patched
    echo  FIX: Replace with azalyst_autonomous_team_PATCHED.py
    pause
    exit /b 1
) else (
    echo  ✓ azalyst_autonomous_team.py has UTF-8 fix
)

REM Check walkforward_simulator.py for TextIOWrapper
findstr /C:"TextIOWrapper" walkforward_simulator.py >nul 2>nul
if errorlevel 1 (
    echo  ✗ walkforward_simulator.py is NOT patched
    echo  FIX: Replace with the patched version
    pause
    exit /b 1
) else (
    echo  ✓ walkforward_simulator.py has UTF-8 fix
)

:: ═════════════════════════════════════════════════════════════════════════════
::  STEP 5: Ready to run!
:: ═════════════════════════════════════════════════════════════════════════════
echo  [STEP 5/5] All checks passed!
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   READY TO RUN AUTONOMOUS TEAM                          ║
echo  ║   Starting: azalyst_autonomous_team.py                  ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  Expected:
echo    • Progress bar with UTF-8 characters (█ ░ ═)
echo    • Walkforward simulator runs
echo    • Olivia analyzes results
echo    • Marcus applies fixes
echo    • Loop continues for up to 10 iterations
echo.
echo  Output files:
echo    • paper_trades.csv           — Trades
echo    • performance_metrics.csv    — Results
echo    • team_log.txt               — Full transcript
echo.
timeout /t 3 /nobreak

python azalyst_autonomous_team.py

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   TEAM SESSION COMPLETE                                 ║
echo  ║   Check team_log.txt for full transcript               ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
pause
