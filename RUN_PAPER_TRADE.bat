@echo off
setlocal EnableDelayedExpansion
title Azalyst v7 - Paper Trade (Live Binance Data, No Real Orders)
color 0B
chcp 65001 >nul 2>&1
cd /d "%~dp0"

rem -- Add common Python locations to PATH ------------------------------------
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310"
    "C:\Python313" "C:\Python312" "C:\Python311" "C:\Python310"
    "%ProgramFiles%\Python313" "%ProgramFiles%\Python312"
    "%ProgramFiles%\Python311" "%ProgramFiles%\Python310"
    "%ProgramData%\Anaconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
) do if exist "%%~d\python.exe" set "PATH=%%~d;%%~d\Scripts;!PATH!"

where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] python.exe not found on PATH.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo    AZALYST v7 - PAPER TRADE (live data, simulated fills)
echo  ============================================================
echo    NO REAL ORDERS WILL BE PLACED.
echo    Data source: Binance public REST API (no API key needed).
echo    State dir  : paper_trade_state\
echo  ============================================================
echo.

rem -- Mode selection ---------------------------------------------------------
echo  1. Single cycle (recommended for first run)
echo  2. Loop every 1 hour   (interactive)
echo  3. Loop every 4 hours  (interactive)
echo  4. Force rebalance NOW (ignores weekly interval)
echo.
set /p MODE="Select mode [1-4, default=1]: "
if "%MODE%"=="" set MODE=1

rem -- Top-N and leverage -----------------------------------------------------
set /p TOPN="Top-N per side [default=5]: "
if "%TOPN%"=="" set TOPN=5

set /p LEV="Leverage [default=0.5]: "
if "%LEV%"=="" set LEV=0.5

echo.
echo  Running with --top-n %TOPN% --leverage %LEV%
echo.

if "%MODE%"=="1" (
    python azalyst_paper_trade.py --once --top-n %TOPN% --leverage %LEV%
) else if "%MODE%"=="2" (
    python azalyst_paper_trade.py --loop 3600 --top-n %TOPN% --leverage %LEV%
) else if "%MODE%"=="3" (
    python azalyst_paper_trade.py --loop 14400 --top-n %TOPN% --leverage %LEV%
) else if "%MODE%"=="4" (
    python azalyst_paper_trade.py --once --force-rebalance --top-n %TOPN% --leverage %LEV%
) else (
    echo  Invalid mode.
)

echo.
echo  ============================================================
echo    Done. Check paper_trade_state\ for state + logs:
echo      - positions.json     current open positions + equity
echo      - trade_log.csv      every OPEN / CLOSE event
echo      - equity_curve.csv   equity + drawdown over time
echo      - run_log.txt        full runner log
echo  ============================================================
pause
