@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: ── FIX: Admin elevation (required for power plan changes) ───────────────────
net session >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Requesting administrator privileges for power plan...
    powershell -Command "Start-Process '%~f0' -Verb RunAs" >nul 2>&1
    if errorlevel 1 (
        echo  [WARN] Could not elevate. Power plan will not be set. Continuing anyway.
    ) else (
        exit /b
    )
)

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
if errorlevel 1 goto :NO_GPU
set "GPU_FOUND=1"
for /f "usebackq delims=" %%a in (`nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul`) do (
    if "!GPU_NAME!"=="None" set "GPU_NAME=%%a"
)
if "!GPU_NAME!"=="None" set "GPU_NAME=NVIDIA GPU"
echo  [OK] GPU: !GPU_NAME!
goto :GPU_DONE
:NO_GPU
echo  [INFO] No NVIDIA GPU detected - CPU mode only
:GPU_DONE

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
if not exist "%~dp0azalyst_v5_engine.py"  ( echo  [ERROR] Missing: azalyst_v5_engine.py  & set "MOD_OK=0" )
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
set "PARQUET_FOUND=0"
if exist "%~dp0data\" (
    for %%f in ("%~dp0data\*.parquet") do set "PARQUET_FOUND=1"
)
if exist "%~dp0data_top6\" (
    for %%f in ("%~dp0data_top6\*.parquet") do set "PARQUET_FOUND=1"
)
if "!PARQUET_FOUND!"=="0" (
    echo  [ERROR] No .parquet files found in data\ or data_top6\
    echo  Create data\ and add Binance 5-min OHLCV .parquet files,
    echo  or add Top-6 coin parquet files to data_top6\
    pause
    exit /b 1
)
echo  [OK] Data files found

:: ── Feature cache info ────────────────────────────────────────────────────────
set "CACHE_COUNT=0"
if exist "%~dp0feature_cache\" (
    for %%f in ("%~dp0feature_cache\*.parquet") do set "CACHE_COUNT=1"
)
if exist "%~dp0cache_top6\" (
    for %%f in ("%~dp0cache_top6\*.parquet") do set "CACHE_COUNT=1"
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
set "SKIP_SHAP=0"

:: Jump to GPU question if GPU is available; otherwise CPU-only path
if "!GPU_FOUND!"=="1" goto :ASK_COMPUTE
echo  Compute: CPU only (no GPU found)
goto :Q_MODE

:ASK_COMPUTE
echo  Select compute device:
echo    [1] GPU  -  !GPU_NAME!  ~4x faster
echo    [2] CPU  -  all cores
echo.
choice /N /C:12 /M "  Choice (1/2): "
if errorlevel 2 goto :SET_CPU
goto :SET_GPU
:SET_CPU
set "COMPUTE_CHOICE=cpu"
set "SKIP_SHAP=0"
echo  [OK] CPU selected
goto :Q_MODE
:SET_GPU
set "COMPUTE_CHOICE=gpu"
set "SKIP_SHAP=1"
echo  [OK] GPU selected

:: ── Run mode: Terminal only vs Terminal + Spyder monitor ─────────────────────
:Q_MODE
echo.
set "LAUNCH_MONITOR=0"
echo  Run mode:
echo    [1] Terminal only
echo    [2] Terminal + Spyder  (live monitor opens in a second window;
echo                            closing it will NOT stop the engine)
echo.
choice /N /C:12 /M "  Choice (1/2): "
if errorlevel 2 goto :SET_SPYDER
goto :SET_TERMINAL
:SET_SPYDER
set "LAUNCH_MONITOR=1"
echo  [OK] Terminal + Spyder
goto :CONFIRM
:SET_TERMINAL
set "LAUNCH_MONITOR=0"
echo  [OK] Terminal only

:: ── Universe mode ───────────────────────────────────────────────────────────
:Q_UNIVERSE
echo.
echo  Universe:
echo    [1] TOP-6 Persistent Coins  -  data_top6\  (curated  winning config)
echo        1000SATSUSDT, BONKUSDT, ADXUSDT, FDUSDUSDT, WINUSDT, AEURUSDT
echo    [2] Custom / Full Universe  -  data\        (all coins in data\)
echo.
choice /N /C:12 /M "  Choice (1/2): "
if errorlevel 2 goto :SET_FULL_UNIVERSE
goto :SET_TOP6_UNIVERSE

:SET_TOP6_UNIVERSE
set "UNIVERSE_MODE=top6"
set "DATA_DIR_ARG=%~dp0data_top6"
set "CACHE_DIR_ARG=%~dp0cache_top6"
set "OUT_DIR_ARG=%~dp0results_top6"
set "PIN_COINS_ARG=1000SATSUSDT,BONKUSDT,ADXUSDT,FDUSDUSDT,WINUSDT,AEURUSDT"
echo  [OK] Top-6 persistent coins selected  (5d horizon, force-invert, 3x leverage)
goto :CONFIRM

:SET_FULL_UNIVERSE
set "UNIVERSE_MODE=full"
set "DATA_DIR_ARG=%~dp0data"
set "CACHE_DIR_ARG=%~dp0feature_cache"
set "OUT_DIR_ARG=%~dp0results"
set "PIN_COINS_ARG="
echo  [OK] Full universe selected

:CONFIRM
echo.
echo  ============================================================
echo   READY
echo    Compute  : !COMPUTE_CHOICE!
echo    Universe : !UNIVERSE_MODE!
echo    Monitor  : !LAUNCH_MONITOR! (0=terminal only, 1=terminal+spyder)
echo    Data     : !DATA_DIR_ARG!
echo    Cache    : !CACHE_DIR_ARG!
echo    Results  : !OUT_DIR_ARG!
echo  ============================================================
echo.
choice /N /C:YN /M "  Start? (Y/N): "
if errorlevel 2 ( echo  Cancelled. & timeout /t 2 /nobreak >nul & exit /b 0 )
echo.

:: ── Run ───────────────────────────────────────────────────────────────────────
:: FIX: was PCI_E_BUS_ID — correct value is PCI_BUS_ID
if "!COMPUTE_CHOICE!"=="gpu" (
    set "CUDA_VISIBLE_DEVICES=0"
    set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
)

powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

:: ── Launch Spyder monitor in its own window (independent of engine) ──────────
if "!LAUNCH_MONITOR!"=="1" (
    if exist "%~dp0VIEW_TRAINING.py" (
        start "Azalyst Monitor" !PYTHON_EXE! "%~dp0VIEW_TRAINING.py"
        echo  [Monitor] Spyder monitor launched in separate window.
        echo  [Monitor] You can close it at any time — the engine keeps running.
        echo.
    ) else (
        echo  [Monitor] VIEW_TRAINING.py not found — running terminal only.
        echo.
    )
)

echo  ============================================================
echo   RUNNING
echo    Started: %date% %time%
echo  ============================================================
echo.

:: ── Build the python command based on choices ─────────────────────────────────
set "PY_ARGS=--data-dir "!DATA_DIR_ARG!" --feature-dir "!CACHE_DIR_ARG!" --out-dir "!OUT_DIR_ARG!""

if "!COMPUTE_CHOICE!"=="gpu" (
    set "PY_ARGS=--gpu !PY_ARGS!"
)

:: GPU: skip SHAP by default to stay within 4GB VRAM; CPU: SHAP always on
if "!SKIP_SHAP!"=="1" (
    set "PY_ARGS=!PY_ARGS! --no-shap"
)

:: Top-6 mode: apply winning config + pin to persistent coin universe
if "!UNIVERSE_MODE!"=="top6" (
    set "PY_ARGS=!PY_ARGS! --target 5d --force-invert --leverage 3 --ic-gating-threshold -1.0 --max-dd -1.0 --no-resume --pin-coins "!PIN_COINS_ARG!""
)

!PYTHON_EXE! -u "%~dp0azalyst_v5_engine.py" !PY_ARGS!

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
        echo  Most likely causes on your laptop:
        echo.
        echo  1. Timestamp issue in your parquet files. Run this to check:
        echo     !PYTHON_EXE! -c "import pandas as pd; df=pd.read_parquet('data/BTCUSDT.parquet'); idx=df.index if hasattr(df.index,'year') else pd.to_datetime(df.index,unit='ms',utc=True); print('Max year:',idx.max().year,'Min year:',idx.min().year); print(df.head(3)); print(df.dtypes)"
        echo.
        echo  2. Feature cache built OK but 0 symbols passed the year^>=2018 check.
        echo     Delete feature_cache\ folder and re-run.
        echo.
        echo  3. Columns missing: parquet needs open,high,low,close,volume columns.
        echo.
        echo  4. Armoury Crate not in Performance mode — GPU throttled silently.
        echo     Set Armoury Crate to Performance mode and re-run with GPU.
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
    echo    GPU OOM       - re-run, choose CPU or choose Skip SHAP=Yes
    echo    Bad timestamp - delete feature_cache\ and retry
    echo    Import error  - check error message above for exact cause
    echo.
    if exist "%~dp0results\checkpoint_v4_latest.json" (
        echo  Checkpoint saved. Run again to resume from where it stopped.
    )
)

echo.
echo  Press any key to close...
pause >nul
exit /b !EXIT_CODE!
