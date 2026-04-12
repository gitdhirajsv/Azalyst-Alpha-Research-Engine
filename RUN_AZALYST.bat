@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

rem -- Admin elevation (needed for power plan changes) ------------------------
net session >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Requesting administrator privileges for power plan...
    powershell -NoProfile -Command "Start-Process '%~f0' -Verb RunAs" >nul 2>&1
    if errorlevel 1 (
        echo  [WARN] Could not elevate. Power plan will not be set. Continuing anyway.
    ) else (
        exit /b
    )
)

rem -- Boost PATH with common Python locations --------------------------------
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
echo    AZALYST ALPHA RESEARCH ENGINE  v6.0
echo  ============================================================
echo.

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem -- Find Python ------------------------------------------------------------
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "PYTHON_LABEL="

call :TRY_PYTHON python
if defined PYTHON_EXE goto :PY_FOUND
call :TRY_PYTHON py -3
if defined PYTHON_EXE goto :PY_FOUND

for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    "%USERPROFILE%\.local\bin\python3.14.exe"
    "%USERPROFILE%\.local\bin\python3.13.exe"
    "%USERPROFILE%\.local\bin\python3.12.exe"
    "%USERPROFILE%\.local\bin\python3.11.exe"
    "%USERPROFILE%\.local\bin\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "%ProgramData%\Anaconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
) do (
    if not defined PYTHON_EXE if exist "%%~fp" (
        set "PYTHON_EXE=%%~fp"
        set "PYTHON_ARGS="
    )
)

if not defined PYTHON_EXE (
    echo  [ERROR] Python not found. Install from https://python.org
    echo  Check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:PY_FOUND
if defined PYTHON_ARGS (
    set "PYTHON_LABEL=!PYTHON_EXE! !PYTHON_ARGS!"
) else (
    set "PYTHON_LABEL=!PYTHON_EXE!"
)
echo  [OK] Python: !PYTHON_LABEL!

rem -- GPU detection ----------------------------------------------------------
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
rem -- Package check ----------------------------------------------------------
echo.
echo  [Setup] Checking packages...

call :RUN_PYTHON -c "import xgboost, lightgbm, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, psutil, statsmodels, polars, duckdb, requests, websockets, pytz, dotenv, sortedcontainers, shap" >nul 2>&1
if not errorlevel 1 (
    echo  [OK] All core packages present
    goto :PKGS_OPTIONAL
)

if exist "%~dp0requirements.txt" (
    echo  [Setup] Installing requirements.txt...
    call :RUN_PYTHON -m pip install --disable-pip-version-check -r "%~dp0requirements.txt" -q
    if errorlevel 1 (
        echo  [ERROR] Install failed. Try:
        echo    !PYTHON_LABEL! -m pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo  [OK] requirements.txt installed
    goto :PKGS_OPTIONAL
)

echo  [Setup] Installing core packages...
call :RUN_PYTHON -m pip install --disable-pip-version-check xgboost lightgbm numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels polars duckdb requests websockets pytz python-dotenv sortedcontainers shap -q
if errorlevel 1 (
    echo  [ERROR] Package install failed.
    pause
    exit /b 1
)

:PKGS_OPTIONAL
call :RUN_PYTHON -c "import alphalens" >nul 2>&1
if errorlevel 1 call :RUN_PYTHON -m pip install alphalens-reloaded -q >nul 2>&1
call :RUN_PYTHON -c "import binance" >nul 2>&1
if errorlevel 1 call :RUN_PYTHON -m pip install python-binance -q >nul 2>&1

rem -- CUDA probe -------------------------------------------------------------
if "!GPU_FOUND!"=="1" (
    echo  [Setup] Testing XGBoost CUDA...
    call :RUN_PYTHON -c "import numpy as np,xgboost as xgb;X=np.random.rand(256,8).astype('float32');y=np.array([0,1]*128);xgb.XGBClassifier(device='cuda',n_estimators=2,max_depth=2,verbosity=0).fit(X,y)" >nul 2>&1
    if not errorlevel 1 (
        set "CUDA_READY=1"
        echo  [OK] XGBoost CUDA ready
    ) else (
        echo  [WARN] CUDA probe failed - will fall back to CPU
    )
)

rem -- Local module check -----------------------------------------------------
echo  [Setup] Checking local modules...
set "MOD_OK=1"
if not exist "%~dp0azalyst_v6_engine.py"  ( echo  [ERROR] Missing: azalyst_v6_engine.py & set "MOD_OK=0" )
if not exist "%~dp0azalyst_factors_v2.py" ( echo  [ERROR] Missing: azalyst_factors_v2.py & set "MOD_OK=0" )
if not exist "%~dp0azalyst_risk.py"       ( echo  [ERROR] Missing: azalyst_risk.py & set "MOD_OK=0" )
if not exist "%~dp0azalyst_db.py"         ( echo  [ERROR] Missing: azalyst_db.py & set "MOD_OK=0" )
if "!MOD_OK!"=="0" (
    echo.
    echo  Run this BAT from the Azalyst project root folder.
    pause
    exit /b 1
)
echo  [OK] Local modules found

rem -- Data folder check ------------------------------------------------------
set "PARQUET_FOUND=0"
if exist "%~dp0data\" (
    for %%f in ("%~dp0data\*.parquet") do set "PARQUET_FOUND=1"
)
if "!PARQUET_FOUND!"=="0" (
    echo  [ERROR] No .parquet files found in data\
    echo  Add Binance 5-min OHLCV .parquet files to data\
    pause
    exit /b 1
)
echo  [OK] Data files found

rem -- Feature cache info -----------------------------------------------------
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

rem -- Config -----------------------------------------------------------------
echo.
echo  ============================================================
echo   CONFIGURATION
echo  ============================================================
echo.

set "COMPUTE_CHOICE=cpu"
set "SKIP_SHAP=0"
set "UNIVERSE_MODE=full"
set "DATA_DIR_ARG=%~dp0data"
set "CACHE_DIR_ARG=%~dp0feature_cache"
set "OUT_DIR_ARG=%~dp0results"
set "PIN_COINS_ARG="
set "TOP_N_ARG="
set "NO_RESUME_FLAG="
set "NO_FALSIFY_FLAG="

if "!GPU_FOUND!"=="1" goto :ASK_COMPUTE
echo  Compute: CPU only (no GPU found)
goto :Q_MODE

:ASK_COMPUTE
echo  Select compute device:
echo    [1] GPU  -  !GPU_NAME!  ~4x faster
echo    [2] CPU  -  all cores
echo.
choice /N /C:12 /M "  Choice 1 or 2: "
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

rem -- Run mode ---------------------------------------------------------------
:Q_MODE
echo.
set "LAUNCH_MONITOR=0"
echo  Run mode:
echo    [1] Terminal only
echo    [2] Terminal + Spyder  (live monitor opens in a second window)
echo.
choice /N /C:12 /M "  Choice 1 or 2: "
if errorlevel 2 goto :SET_SPYDER
goto :SET_TERMINAL

:SET_SPYDER
set "LAUNCH_MONITOR=1"
echo  [OK] Terminal + Spyder
goto :Q_UNIVERSE

:SET_TERMINAL
set "LAUNCH_MONITOR=0"
echo  [OK] Terminal only

rem -- Universe mode ----------------------------------------------------------
:Q_UNIVERSE
set "UNIVERSE_MODE=v6"
set "DATA_DIR_ARG=%~dp0data"
set "CACHE_DIR_ARG=%~dp0feature_cache"
set "OUT_DIR_ARG=%~dp0results_v6"
set "TOP_N_ARG=5"
echo  [OK] V6 Consensus Rebuild  (Elastic Net, beta-neutral, regime-gated, top-5)
goto :Q_RESUME

rem -- Fresh run vs resume ----------------------------------------------------
:Q_RESUME
echo.
if exist "%~dp0results_v6\checkpoint_v6_latest.json" (
    echo  Checkpoint found in results_v6\
    echo    [1] Fresh run   - ignore saved checkpoint
    echo    [2] Resume run  - continue from saved checkpoint
    echo.
    choice /N /C:12 /M "  Choice 1 or 2: "
    if errorlevel 2 goto :SET_RESUME
    goto :SET_FRESH
) else (
    set "NO_RESUME_FLAG=--no-resume"
    echo  [OK] No checkpoint found - starting fresh
)
goto :Q_FALSIFY

:SET_FRESH
set "NO_RESUME_FLAG=--no-resume"
echo  [OK] Fresh run selected
goto :Q_FALSIFY

:SET_RESUME
set "NO_RESUME_FLAG="
echo  [OK] Resume selected
goto :Q_FALSIFY

rem -- Falsification toggle ---------------------------------------------------
:Q_FALSIFY
echo.
echo  Falsification campaign:
echo    [1] Enabled  - baseline sanity check ^(slower^)
echo    [2] Skip     - faster launch / debugging
echo.
choice /N /C:12 /M "  Choice 1 or 2: "
if errorlevel 2 goto :SET_NO_FALSIFY
goto :SET_FALSIFY

:SET_FALSIFY
set "NO_FALSIFY_FLAG="
echo  [OK] Falsification enabled
goto :CONFIRM

:SET_NO_FALSIFY
set "NO_FALSIFY_FLAG=--no-falsify"
echo  [OK] Falsification skipped
goto :CONFIRM

:CONFIRM
echo.
echo  ============================================================
echo   READY
echo    Compute  : !COMPUTE_CHOICE!
echo    Universe : !UNIVERSE_MODE!
echo    Top-N    : !TOP_N_ARG! per side
echo    Monitor  : !LAUNCH_MONITOR!  (0=terminal only, 1=terminal+spyder)
echo    Data     : !DATA_DIR_ARG!
echo    Cache    : !CACHE_DIR_ARG!
echo    Results  : !OUT_DIR_ARG!
if defined NO_RESUME_FLAG (echo    Resume   : fresh run) else (echo    Resume   : checkpoint resume)
if defined NO_FALSIFY_FLAG (echo    Falsify  : skipped) else (echo    Falsify  : enabled)
echo  ============================================================
echo.
choice /N /C:YN /M "  Start? Y or N: "
if errorlevel 2 (
    echo  Cancelled.
    timeout /t 2 /nobreak >nul
    exit /b 0
)
echo.

rem -- Run --------------------------------------------------------------------
rem FIX: was PCI_E_BUS_ID - correct value is PCI_BUS_ID
if "!COMPUTE_CHOICE!"=="gpu" (
    set "CUDA_VISIBLE_DEVICES=0"
    set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
)

powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

rem -- Launch monitor in its own window ---------------------------------------
if "!LAUNCH_MONITOR!"=="1" (
    if exist "%~dp0VIEW_TRAINING.py" (
        if defined PYTHON_ARGS (
            powershell -NoProfile -Command "Start-Process -FilePath '%PYTHON_EXE%' -ArgumentList @('%PYTHON_ARGS%','%~dp0VIEW_TRAINING.py') -WorkingDirectory '%~dp0'" >nul 2>&1
        ) else (
            powershell -NoProfile -Command "Start-Process -FilePath '%PYTHON_EXE%' -ArgumentList @('%~dp0VIEW_TRAINING.py') -WorkingDirectory '%~dp0'" >nul 2>&1
        )
        echo  [Monitor] Separate monitor launched.
        echo.
    ) else (
        echo  [Monitor] VIEW_TRAINING.py not found - running terminal only.
        echo.
    )
)

echo  ============================================================
echo   RUNNING
echo    Started: %date% %time%
echo  ============================================================
echo.

set "GPU_FLAG="
if "!COMPUTE_CHOICE!"=="gpu" set "GPU_FLAG=--gpu"

set "SHAP_FLAG="
if "!SKIP_SHAP!"=="1" set "SHAP_FLAG=--no-shap"

call :RUN_PYTHON -u "%~dp0azalyst_v6_engine.py" --data-dir "!DATA_DIR_ARG!" --feature-dir "!CACHE_DIR_ARG!" --out-dir "!OUT_DIR_ARG!" --top-n !TOP_N_ARG! --rolling-window 104 --leverage 1.0 !GPU_FLAG! !SHAP_FLAG! !NO_RESUME_FLAG! !NO_FALSIFY_FLAG!

:POST_RUN
set "EXIT_CODE=!errorlevel!"

powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e >nul 2>&1

echo.
echo  ============================================================
echo    Finished: %date% %time%
echo  ============================================================
echo.

if "!EXIT_CODE!"=="0" (
    if not exist "%~dp0results_v6\weekly_summary_v6.csv" (
        color 0E
        echo  [WARN] Pipeline exited cleanly but produced no results.
        echo.
        echo  Most likely causes:
        echo  1. Timestamp issue in your parquet files.
        echo  2. Feature cache built OK but no symbols passed the year check.
        echo  3. Required parquet columns are missing.
        echo  4. GPU throttling or fallback occurred.
        echo.
        echo  Try deleting feature_cache\ and running again.
    ) else (
        color 0A
        echo  Pipeline completed successfully.
        echo.
        echo    results_v6\weekly_summary_v6.csv
        echo    results_v6\all_trades_v6.csv
        echo    results_v6\performance_v6.json
        echo    results_v6\azalyst_v6.db
    )
) else (
    color 0C
    echo  [ERROR] Exit code !EXIT_CODE!
    echo.
    echo  Common fixes:
    echo    GPU OOM       - re-run and choose CPU
    echo    Bad timestamp - delete feature_cache\ and retry
    echo    Import error  - check results_v6\run_log_v6.txt
    echo.
    if exist "%~dp0results_v6\checkpoint_v6_latest.json" (
        echo  Checkpoint saved ^(results_v6^). Run again to resume from where it stopped.
    )
)

echo.
echo  Press any key to close...
pause >nul
exit /b !EXIT_CODE!

:RUN_PYTHON
if not "!PYTHON_EXE:\=!"=="!PYTHON_EXE!" (
    if defined PYTHON_ARGS (
        call "!PYTHON_EXE!" !PYTHON_ARGS! %*
    ) else (
        call "!PYTHON_EXE!" %*
    )
) else (
    if defined PYTHON_ARGS (
        call !PYTHON_EXE! !PYTHON_ARGS! %*
    ) else (
        call !PYTHON_EXE! %*
    )
)
exit /b %errorlevel%

:TRY_PYTHON
set "TRY_PYTHON_EXE=%~1"
if not "!TRY_PYTHON_EXE:\=!"=="!TRY_PYTHON_EXE!" (
    if "%~2"=="" (
        call "!TRY_PYTHON_EXE!" --version >nul 2>&1
    ) else (
        call "!TRY_PYTHON_EXE!" %~2 --version >nul 2>&1
    )
) else (
    if "%~2"=="" (
        call !TRY_PYTHON_EXE! --version >nul 2>&1
    ) else (
        call !TRY_PYTHON_EXE! %~2 --version >nul 2>&1
    )
)
if not errorlevel 1 (
    set "PYTHON_EXE=%~1"
    set "PYTHON_ARGS=%~2"
)
exit /b 0
