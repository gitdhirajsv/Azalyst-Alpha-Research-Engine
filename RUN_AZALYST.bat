@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

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

echo.
echo  ============================================================
echo    AZALYST ALPHA RESEARCH ENGINE  v4.0
echo  ============================================================
echo.

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: ── Find Python ──────────────────────────────────────────────────────────────
set "PYTHON_EXE="

python --version >nul 2>&1
if not errorlevel 1 ( set "PYTHON_EXE=python" & goto :PY_FOUND )

py -3 --version >nul 2>&1
if not errorlevel 1 ( set "PYTHON_EXE=py -3" & goto :PY_FOUND )

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
    "%ProgramData%\Anaconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
) do if not defined PYTHON_EXE if exist "%%~p" set "PYTHON_EXE=%%~p"

if not defined PYTHON_EXE (
    echo  [ERROR] Python not found. Install from https://python.org
    echo  Check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:PY_FOUND
echo  [OK] Python: !PYTHON_EXE!

:: ── GPU detection ─────────────────────────────────────────────────────────────
set "GPU_FOUND=0"
set "GPU_NAME=None"
set "CUDA_READY=0"

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "GPU_FOUND=1"
    for /f "usebackq delims=" %%a in (`nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul`) do (
        if "!GPU_NAME!"=="None" set "GPU_NAME=%%a"
    )
    if "!GPU_NAME!"=="None" set "GPU_NAME=NVIDIA GPU"
    echo  [OK] GPU: !GPU_NAME!
) else (
    echo  [INFO] No NVIDIA GPU detected - CPU mode only
)

:: ── Package check ─────────────────────────────────────────────────────────────
echo.
echo  [Setup] Checking packages...

!PYTHON_EXE! -c "import xgboost, lightgbm, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, psutil, statsmodels, polars, duckdb, requests, websockets, pytz, dotenv, sortedcontainers, shap" >nul 2>&1
if not errorlevel 1 (
    echo  [OK] All core packages present
    goto :PKGS_OPTIONAL
)

if exist "%~dp0requirements.txt" (
    echo  [Setup] Installing requirements.txt...
    !PYTHON_EXE! -m pip install --disable-pip-version-check -r "%~dp0requirements.txt" -q
    if errorlevel 1 (
        echo  [ERROR] Install failed. Try:
        echo    !PYTHON_EXE! -m pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo  [OK] requirements.txt installed
    goto :PKGS_OPTIONAL
)

echo  [Setup] Installing core packages...
!PYTHON_EXE! -m pip install --disable-pip-version-check xgboost lightgbm numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels polars duckdb requests websockets pytz python-dotenv sortedcontainers shap -q
if errorlevel 1 ( echo  [ERROR] Package install failed. & pause & exit /b 1 )

:PKGS_OPTIONAL
!PYTHON_EXE! -c "import alphalens" >nul 2>&1
if errorlevel 1 !PYTHON_EXE! -m pip install alphalens-reloaded -q >nul 2>&1
!PYTHON_EXE! -c "import binance" >nul 2>&1
if errorlevel 1 !PYTHON_EXE! -m pip install python-binance -q >nul 2>&1

:: ── CUDA probe ────────────────────────────────────────────────────────────────
if "!GPU_FOUND!"=="1" (
    echo  [Setup] Testing XGBoost CUDA...
    !PYTHON_EXE! -c "import numpy as np,xgboost as xgb;X=np.random.rand(256,8).astype('float32');y=np.array([0,1]*128);xgb.XGBClassifier(device='cuda',n_estimators=2,max_depth=2,verbosity=0).fit(X,y)" >nul 2>&1
    if not errorlevel 1 ( set "CUDA_READY=1" & echo  [OK] XGBoost CUDA ready ) else echo  [WARN] CUDA probe failed - will fall back to CPU
)

:: ── Local module check ────────────────────────────────────────────────────────
echo  [Setup] Checking local modules...
set "MOD_OK=1"
if not exist "%~dp0azalyst_v4_engine.py"  ( echo  [ERROR] Missing: azalyst_v4_engine.py  & set "MOD_OK=0" )
if not exist "%~dp0azalyst_factors_v2.py" ( echo  [ERROR] Missing: azalyst_factors_v2.py & set "MOD_OK=0" )
if not exist "%~dp0azalyst_risk.py"       ( echo  [ERROR] Missing: azalyst_risk.py       & set "MOD_OK=0" )
if not exist "%~dp0azalyst_db.py"         ( echo  [ERROR] Missing: azalyst_db.py         & set "MOD_OK=0" )
if "!MOD_OK!"=="0" (
    echo.
    echo  Run this BAT from the Azalyst project root folder.
    pause
    exit /b 1
)
echo  [OK] Local modules found

:: ── Data folder check ─────────────────────────────────────────────────────────
if not exist "%~dp0data\" (
    echo  [ERROR] data\ folder not found.
    echo  Create it and add your Binance 5-min OHLCV .parquet files.
    pause
    exit /b 1
)
set "PARQUET_FOUND=0"
for %%f in ("%~dp0data\*.parquet") do set "PARQUET_FOUND=1"
if "!PARQUET_FOUND!"=="0" (
    echo  [ERROR] No .parquet files in data\
    pause
    exit /b 1
)
echo  [OK] Data files found

:: ── Feature cache info ────────────────────────────────────────────────────────
set "CACHE_COUNT=0"
if exist "%~dp0feature_cache\" (
    for %%f in ("%~dp0feature_cache\*.parquet") do set "CACHE_COUNT=1"
)
if "!CACHE_COUNT!"=="0" (
    echo.
    echo  [INFO] Feature cache is empty.
    echo  The engine will build it from data\ before training.
    echo  First run can take 30-90 min depending on symbol count.
    echo  Do NOT close this window during the build.
    echo.
)

:: ── Config ────────────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   CONFIGURATION
echo  ============================================================
echo.

set "COMPUTE_CHOICE=cpu"
if "!GPU_FOUND!"=="0" (
    echo  Compute: CPU only (no GPU found)
    goto :CONFIRM
)

:Q1_LOOP
echo  Select compute device:
echo    [1] GPU - !GPU_NAME! (CUDA=%CUDA_READY%)
echo    [2] CPU - all cores
echo.
set /p "Q1=  Choice (1/2): "
if "!Q1!"=="1" ( set "COMPUTE_CHOICE=gpu" & echo  [OK] GPU selected & goto :CONFIRM )
if "!Q1!"=="2" ( set "COMPUTE_CHOICE=cpu" & echo  [OK] CPU selected & goto :CONFIRM )
echo  Enter 1 or 2.
echo.
goto :Q1_LOOP

:CONFIRM
echo.
echo  ============================================================
echo   READY
echo    Compute : !COMPUTE_CHOICE!
echo    Data    : %~dp0data\
echo    Cache   : %~dp0feature_cache\
echo    Results : %~dp0results\
echo  ============================================================
echo.
set /p "GO=  Start? (Y/N): "
if /i not "!GO!"=="Y" ( echo  Cancelled. & timeout /t 2 /nobreak >nul & exit /b 0 )
echo.

:: ── Run ───────────────────────────────────────────────────────────────────────
if "!COMPUTE_CHOICE!"=="gpu" (
    set "CUDA_VISIBLE_DEVICES=0"
    set "CUDA_DEVICE_ORDER=PCI_E_BUS_ID"
)

powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

echo  ============================================================
echo   RUNNING
echo    Started: %date% %time%
echo  ============================================================
echo.

if "!COMPUTE_CHOICE!"=="gpu" (
    !PYTHON_EXE! -u "%~dp0azalyst_v4_engine.py" --gpu --data-dir "%~dp0data" --feature-dir "%~dp0feature_cache" --out-dir "%~dp0results"
) else (
    !PYTHON_EXE! -u "%~dp0azalyst_v4_engine.py" --data-dir "%~dp0data" --feature-dir "%~dp0feature_cache" --out-dir "%~dp0results"
)

set "EXIT_CODE=!errorlevel!"

powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e >nul 2>&1

echo.
echo  ============================================================
echo    Finished: %date% %time%
echo  ============================================================
echo.

if "!EXIT_CODE!"=="0" (
    if not exist "%~dp0results\weekly_summary_v4.csv" (
        color 0E
        echo  [WARN] Pipeline exited cleanly but produced no results.
        echo.
        echo  This usually means the feature cache built successfully but
        echo  found 0 valid symbols. Check your parquet file format:
        echo.
        echo    !PYTHON_EXE! -c "import pandas as pd; df=pd.read_parquet('data/BTCUSDT.parquet'); print(df.head(3)); print(df.dtypes)"
        echo.
        echo  Files need columns: open, high, low, close, volume
        echo  Timestamps must be after 2018 (not Unix 1970 epoch).
    ) else (
        color 0A
        echo  Pipeline completed successfully.
        echo.
        echo    results\weekly_summary_v4.csv
        echo    results\all_trades_v4.csv
        echo    results\performance_v4.json
        echo    results\azalyst.db
    )
) else (
    color 0C
    echo  [ERROR] Exit code !EXIT_CODE!
    echo.
    echo  Common fixes:
    echo    GPU OOM  - re-run and choose CPU
    echo    Bad data - delete feature_cache\ and retry
    echo    Import   - check error above for exact message
    echo.
    if exist "%~dp0results\checkpoint_v4_latest.json" (
        echo  Checkpoint saved. Run again to resume from where it stopped.
    )
)

echo.
echo  Press any key to close...
pause >nul
exit /b !EXIT_CODE!
