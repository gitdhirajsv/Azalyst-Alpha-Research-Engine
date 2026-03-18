@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo  ==========================================================
echo    AZALYST ALPHA RESEARCH ENGINE  v3.1
echo    XGBoost  ^|  20 Factors  ^|  Binance OHLCV 5m
echo  ==========================================================
echo.
echo  System scan in progress...
echo.

:: -- Step 1: Python check -------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found in PATH.
    echo  Install Python 3.10+ from https://python.org  (check Add to PATH)
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Python %PY_VER%

:: -- Step 2: GPU detection ------------------------------------------------------
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
    echo  [OK] GPU detected: !GPU_NAME!
) else (
    echo  [INFO] No NVIDIA GPU - CPU mode
)

:: -- Step 3: Spyder detection ---------------------------------------------------
set SPYDER_FOUND=0
set SPYDER_CMD=spyder
where spyder >nul 2>&1
if not errorlevel 1 ( set SPYDER_FOUND=1 & echo  [OK] Spyder found & goto :SPYDER_DONE )
python -c "import spyder" >nul 2>&1
if not errorlevel 1 ( set SPYDER_FOUND=1 & set SPYDER_CMD=python -m spyder & echo  [OK] Spyder found (module) & goto :SPYDER_DONE )
for %%p in (
    "%LOCALAPPDATA%\Programs\Spyder\spyder.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\Scripts\spyder.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\Scripts\spyder.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\Scripts\spyder.exe"
    "C:\ProgramData\Anaconda3\Scripts\spyder.exe"
    "C:\ProgramData\miniconda3\Scripts\spyder.exe"
    "%USERPROFILE%\anaconda3\Scripts\spyder.exe"
    "%USERPROFILE%\miniconda3\Scripts\spyder.exe"
    "C:\Program Files\Spyder\spyder.exe"
) do (
    if exist %%p ( set SPYDER_FOUND=1 & set SPYDER_CMD=%%p & echo  [OK] Spyder: %%p & goto :SPYDER_DONE )
)
echo  [INFO] Spyder not found
:SPYDER_DONE
echo.

:: -- Step 4: Install packages ---------------------------------------------------
echo  [Setup] Checking packages...
python -c "import xgboost, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, psutil, statsmodels" 2>nul
if errorlevel 1 (
    echo  [Setup] Installing missing packages (one-time, ~2 min)...
    pip install "xgboost>=2.0.3" numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels --upgrade -q
    if errorlevel 1 (
        echo  [ERROR] Package install failed. Check internet connection.
        pause & exit /b 1
    )
    echo  [OK] Packages installed
) else (
    echo  [OK] All packages present
)
echo.

:: ================================================================
echo ================================================================
echo   CONFIGURATION
echo ================================================================
echo.

:: -- Q1: GPU or CPU ------------------------------------------------------------
set COMPUTE_CHOICE=cpu
set COMPUTE_LABEL=CPU

if "!GPU_FOUND!"=="1" (
    :Q1_LOOP
    echo  [1/2] Select compute device:
    echo.
    echo        [1] GPU  - !GPU_NAME!  (faster ~4x)
    echo        [2] CPU  - All cores
    echo.
    set /p Q1="  Your choice (1/2): "
    if "!Q1!"=="1" ( set COMPUTE_CHOICE=gpu & set COMPUTE_LABEL=GPU (!GPU_NAME!) & echo  [OK] GPU mode & goto :Q2 )
    if "!Q1!"=="2" ( set COMPUTE_CHOICE=cpu & set COMPUTE_LABEL=CPU & echo  [OK] CPU mode & goto :Q2 )
    echo  [!] Enter 1 or 2.
    echo.
    goto :Q1_LOOP
) else (
    echo  [1/2] Compute: CPU only (no GPU detected)
)

:Q2
echo.

:: -- Q2: Spyder ----------------------------------------------------------------
set USE_SPYDER=0

if "!SPYDER_FOUND!"=="1" (
    :Q2_LOOP
    echo  [2/2] Output mode:
    echo.
    echo        [1] Terminal only
    echo        [2] Terminal + Spyder  (closing Spyder will NOT stop the pipeline)
    echo.
    set /p Q2="  Your choice (1/2): "
    if "!Q2!"=="1" ( set USE_SPYDER=0 & echo  [OK] Terminal only & goto :Q2_DONE )
    if "!Q2!"=="2" ( set USE_SPYDER=1 & echo  [OK] Terminal + Spyder & goto :Q2_DONE )
    echo  [!] Enter 1 or 2.
    echo.
    goto :Q2_LOOP
) else (
    echo  [2/2] Output: Terminal only (Spyder not installed)
    echo         To install Spyder:  pip install spyder
)

:Q2_DONE
echo.

:: ================================================================
echo ================================================================
echo   LAUNCH SUMMARY
echo ================================================================
echo.
echo   Compute  : %COMPUTE_LABEL%
if "!USE_SPYDER!"=="1" (
    echo   Output   : Terminal + Spyder (live charts)
) else (
    echo   Output   : Terminal only
)
echo   Data dir : %~dp0data\
echo   Results  : %~dp0results\
echo.
echo ================================================================
echo.
set /p CONFIRM="  Start? (Y/N): "
if /i not "!CONFIRM!"=="Y" ( echo  Cancelled. & timeout /t 2 /nobreak >nul & exit /b 0 )
echo.

:: -- Apply GPU env --------------------------------------------------------------
if "!COMPUTE_CHOICE!"=="gpu" (
    set CUDA_VISIBLE_DEVICES=0
    set CUDA_DEVICE_ORDER=PCI_E_BUS_ID
    echo  [Setup] GPU mode: CUDA_VISIBLE_DEVICES=0
)

:: -- Power plan (non-critical) --------------------------------------------------
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

:: -- Launch Spyder (detached - closing it will NOT stop pipeline) ---------------
if "!USE_SPYDER!"=="1" (
    echo.
    echo ================================================================
    echo   LAUNCHING SPYDER
    echo ================================================================
    echo.
    echo  Spyder opens in background. Once loaded:
    echo    - Open azalyst_spyder_monitor.py and press F5 for live charts
    echo    - Closing Spyder will NOT stop the pipeline
    echo.
    if "!SPYDER_CMD!"=="spyder" (
        start "" /B spyder --new-instance --workdir="%~dp0" 2>nul
    ) else if "!SPYDER_CMD!"=="python -m spyder" (
        start "" /B python -m spyder --new-instance --workdir="%~dp0" 2>nul
    ) else (
        start "" !SPYDER_CMD! --new-instance --workdir="%~dp0" 2>nul
    )
    echo  [OK] Spyder launching... waiting 5s
    timeout /t 5 /nobreak >nul
    echo.
)

:: ================================================================
echo ================================================================
echo   RUNNING AZALYST PIPELINE
echo ================================================================
echo.
echo  Compute  : %COMPUTE_LABEL%
echo  Started  : %date% %time%
echo  Data     : %~dp0data\
echo  Results  : %~dp0results\
echo.
echo ----------------------------------------------------------------
echo.

:: -- Pre-flight: data directory check ------------------------------------------
if not exist "%~dp0data\" (
    echo  [ERROR] Data folder not found: %~dp0data
    echo  Create a 'data' subfolder next to this .bat file and place your .parquet files there.
    echo.
    pause
    exit /b 1
)
set PARQUET_COUNT=0
for %%f in ("%~dp0data\*.parquet") do if exist "%%f" set /a PARQUET_COUNT+=1
if "!PARQUET_COUNT!"=="0" (
    echo  [ERROR] No .parquet files found in: %~dp0data
    echo  Place your Binance OHLCV 5m .parquet files in the 'data' folder next to this .bat file.
    echo.
    pause
    exit /b 1
)
echo  [OK] Data folder found with !PARQUET_COUNT! .parquet file(s)
echo.

python azalyst_engine.py --data-dir "%~dp0data" --out-dir "%~dp0results"

set EXIT_CODE=%errorlevel%

:: -- Restore power plan --------------------------------------------------------
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e >nul 2>&1

echo.
echo ----------------------------------------------------------------

if "!EXIT_CODE!"=="0" (
    color 0A
    echo.
    echo  Pipeline completed successfully!
    echo.
    echo  Output files saved to %~dp0results\
    echo    ic_analysis.csv          - Factor IC / ICIR scores
    echo    backtest_pnl.csv         - Daily PnL with fees
    echo    performance_summary.csv  - Sharpe, Sortino, Calmar
    echo.
) else (
    color 0C
    echo.
    echo  [ERROR] Pipeline failed (exit code !EXIT_CODE!)
    echo.
    echo  Common fixes:
    echo    No .parquet files?   Add them to %~dp0data\
    echo    Missing packages?    pip install xgboost numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels
    echo    Python not found?    Reinstall Python 3.10+ with "Add to PATH" checked
    echo.
)

echo  Finished: %date% %time%
echo.
echo  Press any key to close...
pause >nul
exit /b !EXIT_CODE!
