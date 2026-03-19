@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo  ============================================================
echo    AZALYST ALPHA RESEARCH ENGINE  v3.1
echo    XGBoost  ^|  20 Factors  ^|  Binance OHLCV 5m
echo  ============================================================
echo.
echo  System scan in progress...
echo.

:: ── Step 1: Python check ─────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [FAIL] Python not found in PATH.
    echo         Install Python 3.10+ from https://python.org and check "Add to PATH".
    echo.
    goto :DONE_FAIL
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK]   Python %PY_VER% detected

:: ── Step 2: GPU auto-detection (no prompts) ───────────────────────────────────
set GPU_FOUND=0
set GPU_NAME=None
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        set GPU_NAME=%%g
        set GPU_FOUND=1
    )
)
if "!GPU_FOUND!"=="1" (
    echo  [OK]   GPU detected: !GPU_NAME! - GPU mode will be used
) else (
    echo  [INFO] No NVIDIA GPU detected - running in CPU mode
)

:: ── Step 3: Install all packages from requirements.txt ───────────────────────
echo.
echo  [Setup] Installing / verifying packages from requirements.txt...
if not exist "%~dp0requirements.txt" (
    echo  [FAIL] requirements.txt not found next to this bat file.
    echo         Expected: %~dp0requirements.txt
    echo.
    goto :DONE_FAIL
)
pip install -r "%~dp0requirements.txt" --upgrade -q
if errorlevel 1 (
    echo  [FAIL] Package installation failed. Check your internet connection.
    echo         You can also run manually:  pip install -r requirements.txt
    echo.
    goto :DONE_FAIL
)
echo  [OK]   All packages installed / up-to-date

:: ── Step 4: Create data\ and results\ folders if missing ─────────────────────
echo.
if not exist "%~dp0data\" (
    echo  [WARN] data\ folder not found - creating it now.
    mkdir "%~dp0data"
    echo  [INFO] Place your Binance 5-min OHLCV .parquet files in: %~dp0data\
)

if not exist "%~dp0results\" (
    echo  [INFO] results\ folder not found - creating it now.
    mkdir "%~dp0results"
)
echo  [OK]   Folders ready: data\  results\

:: ── Step 5: Check for .parquet files ─────────────────────────────────────────
set PARQUET_COUNT=0
for %%f in ("%~dp0data\*.parquet") do set /a PARQUET_COUNT+=1
if "!PARQUET_COUNT!"=="0" (
    echo.
    echo  [FAIL] No .parquet files found in %~dp0data\
    echo         Add your Binance 5-min OHLCV .parquet files to the data\ folder and re-run.
    echo.
    goto :DONE_FAIL
)
echo  [OK]   Data: !PARQUET_COUNT! .parquet file(s) found in data\

:: ── Step 6: Apply GPU environment if detected ────────────────────────────────
if "!GPU_FOUND!"=="1" (
    set CUDA_VISIBLE_DEVICES=0
    set CUDA_DEVICE_ORDER=PCI_E_BUS_ID
)

:: ── Step 7: Run the engine ────────────────────────────────────────────────────
echo.
echo  ============================================================
echo    RUNNING AZALYST PIPELINE
echo    Started : %date% %time%
echo    Data    : %~dp0data\
echo    Results : %~dp0results\
echo  ============================================================
echo.

python azalyst_engine.py --data-dir "%~dp0data" --out-dir "%~dp0results"
set EXIT_CODE=!errorlevel!

echo.
echo  ============================================================
echo  Finished: %date% %time%
echo  ============================================================

if "!EXIT_CODE!"=="0" (
    color 0A
    echo.
    echo  [OK]   Pipeline completed successfully!
    echo         Output files saved to %~dp0results\
    echo           ic_analysis.csv         - Factor IC / ICIR scores
    echo           backtest_pnl.csv        - Daily PnL with fees
    echo           performance_summary.csv - Sharpe, Sortino, Calmar
    echo.
    goto :DONE_OK
)

color 0C
echo.
echo  [FAIL] Pipeline exited with code !EXIT_CODE!
echo.
echo  Common fixes:
echo    Missing packages?  pip install -r requirements.txt
echo    No .parquet files? Add them to %~dp0data\
echo    Python not found?  Reinstall Python 3.10+ with "Add to PATH" checked
echo.

:DONE_FAIL
echo  Press any key to close...
pause >nul
exit /b 1

:DONE_OK
echo  Press any key to close...
pause >nul
exit /b 0
