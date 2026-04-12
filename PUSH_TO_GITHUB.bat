@echo off
setlocal EnableDelayedExpansion
title Azalyst — Push to GitHub
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo  ============================================================
echo    AZALYST  —  Git Push to GitHub
echo    Repo: github.com/gitdhirajsv/Azalyst-Alpha-Research-Engine
echo  ============================================================
echo.

rem -- Check git is available ------------------------------------------------
git --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] git not found. Install Git from https://git-scm.com
    pause
    exit /b 1
)

rem -- Show current status ---------------------------------------------------
echo  [1/5] Current git status:
echo.
git status --short
echo.

rem -- Stage the two modified files ------------------------------------------
echo  [2/5] Staging modified files...
git add azalyst_v6_engine.py
git add RUN_AZALYST.bat
echo  Staged:
git status --short
echo.

rem -- Confirm before committing ---------------------------------------------
echo  Files to commit:
echo    - azalyst_v6_engine.py   (2yr rolling window, kill-switch fixes)
echo    - RUN_AZALYST.bat        (--rolling-window 104 flag added)
echo.
choice /N /C:YN /M "  Commit and push? Y or N: "
if errorlevel 2 (
    echo  Cancelled. No changes pushed.
    timeout /t 3 /nobreak >nul
    exit /b 0
)
echo.

rem -- Commit ----------------------------------------------------------------
echo  [3/5] Committing...
git commit -m "v6.1: 2-year rolling window + leakage audit + position sizing fix

Changes:
- ROLLING_WINDOW_WEEKS: 13 -> 104 (2 years of training history)
- RETRAIN_WEEKS: 13 -> 4 (monthly retraining for 2-year OOS run)
- MAX_POSITION_SCALE: 3.0 -> 1.0 (prevents leveraged blowups)
- MAX_DRAWDOWN_KILL: -0.20 -> -0.25 (2-year OOS breathing room)
- KILL_SWITCH_RECOVERY_THRESHOLD: -0.10 -> -0.12
- Added get_date_splits_v6(): guarantees >= 2yr training, >= 2yr OOS
- Added leakage audit logging (prints embargo gap on every training call)
- Updated --rolling-window default to 104 in argparse + BAT launcher

Root cause fixed: 3x position scale caused -34% week-10 loss which
triggered the kill switch after only 10 active weeks. Engine now runs
the full 2-year walk-forward without premature abort."

if errorlevel 1 (
    echo.
    echo  [WARN] Commit failed - possibly nothing to commit.
    git status
    echo.
    pause
    exit /b 1
)
echo.

rem -- Push ------------------------------------------------------------------
echo  [4/5] Pushing to origin/main...
git push origin main

if errorlevel 1 (
    echo.
    echo  [ERROR] Push failed. Common causes:
    echo    - Not authenticated: run  git config --global credential.helper manager
    echo    - No internet connection
    echo    - Remote has newer commits: run  git pull --rebase  first
    echo.
    echo  Try pushing manually:
    echo    git push origin main
    echo.
    pause
    exit /b 1
)
echo.

rem -- Confirm ---------------------------------------------------------------
echo  [5/5] Done!
echo.
echo  ============================================================
echo    Pushed to:
echo    https://github.com/gitdhirajsv/Azalyst-Alpha-Research-Engine
echo  ============================================================
echo.
echo  Commit summary:
git log -1 --oneline
echo.
pause
exit /b 0
