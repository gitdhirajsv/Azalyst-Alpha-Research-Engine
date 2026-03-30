@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: ── Boost PATH with common Python install locations ──────────────────────────
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
) do if exist "%%~d\python.exe" (
    set "PATH=%%~d;%%~d\Scripts;!PATH!"
)

echo.
echo  ============================================================
echo    AZALYST ALPHA RESEARCH ENGINE  v4.0
echo    XGBoost  ^|  56 Factors  ^|  Binance OHLCV 5m
echo  ============================================================
echo.
echo  System scan in progress...
echo.

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 1: Find Python
:: ─────────────────────────────────────────────────────────────────────────────
set "PYTHON_EXE="

:: Try plain 'python' first
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto :PY_FOUND
)

:: Try 'py -3' launcher
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py -3"
    goto :PY_FOUND
)

:: Search well-known absolute paths
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
    "%ProgramData%\Anaconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
) do (
    if not defined PYTHON_EXE if exist "%%~p" (
        set "PYTHON_EXE=%%~p"
    )
)

if not defined PYTHON_EXE (
    echo.
    echo  [ERROR] Python not found.
    echo  Install Python 3.10+ from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:PY_FOUND
echo  [OK] Python: !PYTHON_EXE!

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 2: GPU detection
:: ─────────────────────────────────────────────────────────────────────────────
set "GPU_FOUND=0"
set "GPU_NAME=None"
set "CUDA_READY=0"

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "GPU_FOUND=1"
    for /f "tokens=1,* delims=:" %%a in ('nvidia-smi -L 2^>nul') do (
        if "!GPU_NAME!"=="None" (
            set "GPU_NAME=%%b"
            for /f "tokens=1 delims=(" %%n in ("!GPU_NAME!") do set "GPU_NAME=%%n"
        )
    )
    echo  [OK] GPU detected:!GPU_NAME!
) else (
    echo  [INFO] No NVIDIA GPU detected - CPU mode available
)

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 3: Package check + install
:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo  [Setup] Checking core packages...

:: Core check - does NOT include alphalens or binance (they have unstable import names)
!PYTHON_EXE! -c "import xgboost, lightgbm, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, psutil, statsmodels, polars, duckdb, requests, websockets, pytz, dotenv, sortedcontainers, shap" >nul 2>&1
if not errorlevel 1 (
    echo  [OK] Core packages present
    goto :PKGS_OPTIONAL
)

:: Install from requirements.txt if present
if exist "%~dp0requirements.txt" (
    echo  [Setup] Installing requirements.txt ^(one-time, ~3 min^)...
    !PYTHON_EXE! -m pip install --disable-pip-version-check -r "%~dp0requirements.txt" -q
    if errorlevel 1 (
        echo.
        echo  [ERROR] requirements.txt install failed.
        echo  Try running manually:
        echo    !PYTHON_EXE! -m pip install -r "%~dp0requirements.txt"
        echo.
        pause
        exit /b 1
    )
    echo  [OK] requirements.txt installed
    goto :PKGS_OPTIONAL
)

:: Fallback: install individual packages (no alphalens - handled separately)
echo  [Setup] Installing missing packages ^(one-time, ~3 min^)...
!PYTHON_EXE! -m pip install --disable-pip-version-check xgboost lightgbm numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels polars duckdb requests websockets pytz python-dotenv sortedcontainers shap -q
if errorlevel 1 (
    echo.
    echo  [ERROR] Package install failed. Check your internet connection.
    echo.
    pause
    exit /b 1
)
echo  [OK] Core packages installed

:PKGS_OPTIONAL
:: Try alphalens-reloaded separately (non-fatal if it fails)
!PYTHON_EXE! -c "import alphalens" >nul 2>&1
if errorlevel 1 (
    echo  [Setup] Installing alphalens-reloaded...
    !PYTHON_EXE! -m pip install alphalens-reloaded -q >nul 2>&1
)

:: Try python-binance separately (non-fatal)
!PYTHON_EXE! -c "import binance" >nul 2>&1
if errorlevel 1 (
    !PYTHON_EXE! -m pip install python-binance -q >nul 2>&1
)

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 4: XGBoost CUDA probe (only if GPU found)
:: ─────────────────────────────────────────────────────────────────────────────
if "!GPU_FOUND!"=="1" (
    echo  [Setup] Testing XGBoost CUDA...
    !PYTHON_EXE! -c "import numpy as np, xgboost as xgb; X=np.random.rand(256,8).astype('float32'); y=np.array([0,1]*128); xgb.XGBClassifier(device='cuda',n_estimators=2,max_depth=2,verbosity=0).fit(X,y); print('CUDA_OK')" >nul 2>&1
    if not errorlevel 1 (
        set "CUDA_READY=1"
        echo  [OK] XGBoost CUDA ready
    ) else (
        echo  [WARN] GPU found but CUDA probe failed - will use CPU
    )
)

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 5: Data directory check
:: ─────────────────────────────────────────────────────────────────────────────
if not exist "%~dp0data\" (
    echo.
    echo  [ERROR] data\ folder not found at: %~dp0data
    echo  Create it and add your Binance 5-min OHLCV .parquet files.
    echo.
    pause
    exit /b 1
)

:: Count parquets without relying on delayed expansion inside for-loop
set "PARQUET_FOUND=0"
for %%f in ("%~dp0data\*.parquet") do set "PARQUET_FOUND=1"
if "!PARQUET_FOUND!"=="0" (
    echo.
    echo  [ERROR] No .parquet files found in %~dp0data\
    echo  Add your Binance 5-min OHLCV .parquet files to data\
    echo.
    pause
    exit /b 1
)
echo  [OK] Data folder: .parquet files found

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 6: Local module check
:: ─────────────────────────────────────────────────────────────────────────────
echo  [Setup] Checking local Azalyst modules...
set "MISSING_MODULES="
if not exist "%~dp0azalyst_v4_engine.py"   set "MISSING_MODULES=!MISSING_MODULES! azalyst_v4_engine.py"
if not exist "%~dp0azalyst_factors_v2.py"  set "MISSING_MODULES=!MISSING_MODULES! azalyst_factors_v2.py"
if not exist "%~dp0azalyst_risk.py"        set "MISSING_MODULES=!MISSING_MODULES! azalyst_risk.py"
if not exist "%~dp0azalyst_db.py"          set "MISSING_MODULES=!MISSING_MODULES! azalyst_db.py"

if not "!MISSING_MODULES!"=="" (
    echo.
    echo  [ERROR] Missing required Python files:!MISSING_MODULES!
    echo  Make sure you are running this BAT from the Azalyst project root folder
    echo  ^(the folder that contains azalyst_v4_engine.py^).
    echo.
    pause
    exit /b 1
)
echo  [OK] All local modules present

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 7: Configuration prompts
:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo ================================================================
echo   CONFIGURATION
echo ================================================================
echo.

:: Q1: GPU or CPU
set "COMPUTE_CHOICE=cpu"
if "!GPU_FOUND!"=="0" (
    echo  [1/1] Compute: CPU only ^(no GPU detected^)
    goto :CONFIRM
)

:Q1_LOOP
echo  [1/1] Select compute device:
echo.
if "!CUDA_READY!"=="1" (
    echo        [1] GPU  -!GPU_NAME! ^(CUDA ready, ~4x faster^)
) else (
    echo        [1] GPU  -!GPU_NAME! ^(CUDA probe failed - may fall back to CPU^)
)
echo        [2] CPU  - All CPU cores
echo.
set /p "Q1=  Your choice (1/2): "
if "!Q1!"=="1" ( set "COMPUTE_CHOICE=gpu" & echo  [OK] GPU selected & goto :CONFIRM )
if "!Q1!"=="2" ( set "COMPUTE_CHOICE=cpu" & echo  [OK] CPU selected & goto :CONFIRM )
echo  Enter 1 or 2.
echo.
goto :Q1_LOOP

:CONFIRM
echo.
echo ================================================================
echo   LAUNCH SUMMARY
echo ================================================================
echo.
echo   Compute  : !COMPUTE_CHOICE!
if "!COMPUTE_CHOICE!"=="gpu" echo   GPU      :!GPU_NAME!
echo   Script   : azalyst_v4_engine.py
echo   Data     : %~dp0data\
echo   Features : %~dp0feature_cache\
echo   Results  : %~dp0results\
echo.
echo ================================================================
echo.
set /p "CONFIRM_RUN=  Start? (Y/N): "
if /i not "!CONFIRM_RUN!"=="Y" (
    echo  Cancelled.
    timeout /t 2 /nobreak >nul
    exit /b 0
)
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 8: Set GPU env and run
:: ─────────────────────────────────────────────────────────────────────────────
if "!COMPUTE_CHOICE!"=="gpu" (
    set "CUDA_VISIBLE_DEVICES=0"
    set "CUDA_DEVICE_ORDER=PCI_E_BUS_ID"
)

:: Boost CPU performance (non-critical, ignore failure)
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

echo ================================================================
echo   RUNNING AZALYST v4 PIPELINE
echo ================================================================
echo.
echo   Started: %date% %time%
echo.
echo ----------------------------------------------------------------
echo.

if "!COMPUTE_CHOICE!"=="gpu" (
    !PYTHON_EXE! -u "%~dp0azalyst_v4_engine.py" --gpu ^
        --data-dir "%~dp0data" ^
        --feature-dir "%~dp0feature_cache" ^
        --out-dir "%~dp0results"
) else (
    !PYTHON_EXE! -u "%~dp0azalyst_v4_engine.py" ^
        --data-dir "%~dp0data" ^
        --feature-dir "%~dp0feature_cache" ^
        --out-dir "%~dp0results"
)

set "EXIT_CODE=!errorlevel!"

:: Restore power plan
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e >nul 2>&1

echo.
echo ----------------------------------------------------------------
echo.

if "!EXIT_CODE!"=="0" (
    color 0A
    echo  Pipeline completed successfully!
    echo.
    echo  Output files:
    echo    results\weekly_summary_v4.csv    - Week-by-week returns and IC
    echo    results\all_trades_v4.csv        - Every simulated trade
    echo    results\performance_v4.json      - Sharpe, IC, ICIR summary
    echo    results\azalyst.db               - SQLite full history
    echo.
    if exist "%~dp0results\checkpoint_v4_latest.json" (
        del /f /q "%~dp0results\checkpoint_v4_latest.json" >nul 2>&1
    )
) else (
    color 0C
    echo  [ERROR] Pipeline exited with code !EXIT_CODE!
    echo.
    echo  Common causes:
    echo    - Missing .parquet data files in data\
    echo    - Corrupt feature cache  ^(delete feature_cache\ and retry^)
    echo    - GPU out of memory       ^(re-run and choose CPU^)
    echo    - Python import error     ^(check output above for the exact error^)
    echo.
    if exist "%~dp0results\checkpoint_v4_latest.json" (
        echo  A checkpoint was saved. Run this again to resume from where it stopped.
        echo  To force a fresh start: delete results\checkpoint_v4_latest.json first.
        echo.
    )
)

echo   Finished: %date% %time%
echo.
echo   Press any key to close...
pause >nul
exit /b !EXIT_CODE!
