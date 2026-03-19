@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine
color 0A
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: -- Boost PATH with common Python install locations -------------------------
:: Add base Python dirs AND their Scripts subfolders so spyder.exe is found
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
echo    AZALYST ALPHA RESEARCH ENGINE  v3.1
echo    XGBoost  ^|  20 Factors  ^|  Binance OHLCV 5m
echo  ============================================================
echo.
echo  System scan in progress...
echo.

:: -- Step 1: UTF-8 + Python check --------------------------------------------
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHON_CMD="
python --version >nul 2>&1
if not errorlevel 1 ( set "PYTHON_CMD=python" & goto :PY_FOUND )
py -3 --version >nul 2>&1
if not errorlevel 1 ( set "PYTHON_CMD=py -3" & goto :PY_FOUND )
echo  [ERROR] Python not found in PATH.
echo  Install Python 3.10+ from https://python.org  (check Add to PATH)
echo.
pause
exit /b 1

:PY_FOUND
for /f "tokens=2" %%v in ('!PYTHON_CMD! --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Python !PY_VER! (!PYTHON_CMD!)

:: -- .venv fallback for running the pipeline (prefer global Python for installs)
set "VENV_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
:: RUN_PYTHON = what actually executes the pipeline scripts
:: PYTHON_CMD = used for pip installs (global Python preferred)
set "RUN_PYTHON=!PYTHON_CMD!"
if "!VENV_PYTHON!" neq "" set "RUN_PYTHON=!VENV_PYTHON!"

:: -- Step 2: GPU detection ---------------------------------------------------
set GPU_FOUND=0
set GPU_NAME=None
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=1-4 delims=:" %%a in ('nvidia-smi -L 2^>nul') do (
        if "!GPU_FOUND!"=="0" (
            set "GPU_NAME=%%b"
            set "GPU_NAME=!GPU_NAME:~1!"
            for /f "tokens=1 delims=(" %%n in ("!GPU_NAME!") do set "GPU_NAME=%%n"
            set GPU_FOUND=1
        )
    )
)
if "!GPU_FOUND!"=="1" (
    echo  [OK] GPU detected: !GPU_NAME!
) else (
    echo  [INFO] No NVIDIA GPU - CPU mode
)

:: -- Step 3: Spyder detection ------------------------------------------------
set SPYDER_FOUND=0
set SPYDER_CMD=spyder
set SPYDER_MODE=PATH

:: 3a. Direct command in PATH (Scripts folder now boosted above)
where spyder >nul 2>&1
if not errorlevel 1 ( set SPYDER_FOUND=1 & set SPYDER_MODE=PATH & echo  [OK] Spyder found in PATH & goto :SPYDER_DONE )

:: 3b. Importable as module in run-Python (quoted for paths with spaces)
"!RUN_PYTHON!" -c "import spyder" >nul 2>&1
if not errorlevel 1 ( set SPYDER_FOUND=1 & set SPYDER_MODE=MODULE & echo  [OK] Spyder found (module) & goto :SPYDER_DONE )

:: 3c. Common explicit paths
for %%p in (
    "%~dp0.venv\Scripts\spyder.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\Scripts\spyder.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\Scripts\spyder.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\Scripts\spyder.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\Scripts\spyder.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313\Scripts\spyder.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\Scripts\spyder.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\Scripts\spyder.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\Scripts\spyder.exe"
    "%LOCALAPPDATA%\Programs\Spyder\spyder.exe"
    "C:\ProgramData\Anaconda3\Scripts\spyder.exe"
    "C:\ProgramData\miniconda3\Scripts\spyder.exe"
    "%USERPROFILE%\anaconda3\Scripts\spyder.exe"
    "%USERPROFILE%\miniconda3\Scripts\spyder.exe"
    "C:\Program Files\Spyder\spyder.exe"
) do (
    if exist %%p ( set SPYDER_FOUND=1 & set SPYDER_MODE=EXE & set "SPYDER_CMD=%%~p" & echo  [OK] Spyder: %%p & goto :SPYDER_DONE )
)

:: 3d. Not found - auto-install into global Python (user preference)
echo  [Setup] Spyder not found - installing now (one-time, ~3 min)...
!PYTHON_CMD! -m pip install spyder -q
if not errorlevel 1 (
    echo  [OK] Spyder installed successfully
    set SPYDER_FOUND=1
    set SPYDER_MODE=MODULE
    :: add the fresh Scripts dir to PATH so 'spyder' works next time
    for /f "tokens=*" %%s in ('!PYTHON_CMD! -c "import sysconfig; print(sysconfig.get_path(chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(115)))" 2^>nul') do set "PATH=%%s;!PATH!"
    goto :SPYDER_DONE
)
echo  [WARN] Spyder install failed - continuing without Spyder
:SPYDER_DONE
echo.

:: -- Step 4: Package check ---------------------------------------------------
echo  [Setup] Checking packages in run environment...
"!RUN_PYTHON!" -c "import xgboost, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, psutil, statsmodels" >nul 2>&1
if not errorlevel 1 goto :PKGS_OK
echo  [Setup] Installing missing packages into global Python (one-time, ~2 min)...
:: Note: no --upgrade to avoid RECORD-file errors on packages installed outside pip
!PYTHON_CMD! -m pip install xgboost numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels -q
if errorlevel 1 (
    echo  [WARN] Some packages failed via global pip - trying run-env pip...
    "!RUN_PYTHON!" -m pip install xgboost numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels -q
    if errorlevel 1 (
        echo  [ERROR] Package install failed in both environments.
        echo         Check internet connection, then run:
        echo           pip install xgboost numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels
        echo.
        pause
        exit /b 1
    )
)
echo  [OK] Packages installed
goto :PKGS_DONE
:PKGS_OK
echo  [OK] All packages present
:PKGS_DONE
echo.

:: -- Pre-flight: data directory check ----------------------------------------
if not exist "%~dp0data\" (
    echo  [ERROR] Data folder not found: %~dp0data
    echo  Create a 'data' subfolder next to this .bat file and add your .parquet files.
    echo.
    pause
    exit /b 1
)
set PARQUET_COUNT=0
for %%f in ("%~dp0data\*.parquet") do if exist "%%f" set /a PARQUET_COUNT+=1
if "!PARQUET_COUNT!"=="0" (
    echo  [ERROR] No .parquet files found in %~dp0data
    echo  Add your Binance 5-min OHLCV .parquet files to the data\ folder.
    echo.
    pause
    exit /b 1
)
echo  [OK] Data: !PARQUET_COUNT! .parquet file(s) found
echo.

:: ================================================================
echo ================================================================
echo   CONFIGURATION
echo ================================================================
echo.

:: -- Q1: GPU or CPU ----------------------------------------------------------
set COMPUTE_CHOICE=cpu
set COMPUTE_LABEL=CPU
if "!GPU_FOUND!"=="0" goto :Q1_CPU_ONLY

:Q1_LOOP
echo  [1/2] Select compute device:
echo.
echo        [1] GPU  - !GPU_NAME!  (faster ~4x)
echo        [2] CPU  - All cores
echo.
set /p Q1="  Your choice (1/2): "
if "!Q1!"=="1" ( set COMPUTE_CHOICE=gpu & set COMPUTE_LABEL=GPU & echo  [OK] GPU mode & goto :Q2 )
if "!Q1!"=="2" ( set COMPUTE_CHOICE=cpu & set COMPUTE_LABEL=CPU & echo  [OK] CPU mode & goto :Q2 )
echo  [!] Enter 1 or 2.
echo.
goto :Q1_LOOP

:Q1_CPU_ONLY
echo  [1/2] Compute: CPU only (no GPU detected)

:Q2
echo.

:: -- Q2: Spyder --------------------------------------------------------------
set USE_SPYDER=0
if "!SPYDER_FOUND!"=="0" goto :Q2_NO_SPYDER

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

:Q2_NO_SPYDER
echo  [2/2] Output: Terminal only (Spyder install was skipped or failed)

:Q2_DONE
echo.

:: ================================================================
echo ================================================================
echo   LAUNCH SUMMARY
echo ================================================================
echo.
echo   Compute  : !COMPUTE_LABEL!
if "!COMPUTE_CHOICE!"=="gpu" echo   GPU      : !GPU_NAME!
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

:: -- Apply GPU env -----------------------------------------------------------
if "!COMPUTE_CHOICE!"=="gpu" (
    set CUDA_VISIBLE_DEVICES=0
    set CUDA_DEVICE_ORDER=PCI_E_BUS_ID
    echo  [Setup] GPU mode: CUDA_VISIBLE_DEVICES=0
)

:: -- Power plan (non-critical) -----------------------------------------------
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

:: -- Launch Spyder (detached - closing it will NOT stop pipeline) ------------
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
    if "!SPYDER_MODE!"=="PATH" (
        start "" /B spyder --new-instance --workdir="%~dp0" 2>nul
    ) else if "!SPYDER_MODE!"=="MODULE" (
        start "" /B "!RUN_PYTHON!" -m spyder --new-instance --workdir="%~dp0" 2>nul
    ) else (
        start "" "!SPYDER_CMD!" --new-instance --workdir="%~dp0" 2>nul
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
echo  Compute  : !COMPUTE_LABEL!
if "!COMPUTE_CHOICE!"=="gpu" echo  GPU      : !GPU_NAME!
echo  Started  : %date% %time%
echo  Data     : %~dp0data\
echo  Results  : %~dp0results\
echo.
echo ----------------------------------------------------------------
echo.

if "!COMPUTE_CHOICE!"=="gpu" (
    "!RUN_PYTHON!" azalyst_local_gpu.py --gpu --data-dir "%~dp0data" --out-dir "%~dp0results"
) else (
    "!RUN_PYTHON!" azalyst_engine.py --data-dir "%~dp0data" --out-dir "%~dp0results"
)

set EXIT_CODE=!errorlevel!

:: -- Restore power plan ------------------------------------------------------
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
