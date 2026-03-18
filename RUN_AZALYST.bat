@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine — Launcher
color 0A
chcp 65001 >nul 2>&1

:: ── Set working directory to wherever this .bat file lives ──────────────────
cd /d "%~dp0"

:: ═══════════════════════════════════════════════════════════════════════════
::   AZALYST LAUNCHER  v3.1
::   Runs: azalyst_engine.py --data-dir ./data --out-dir ./results
:: ═══════════════════════════════════════════════════════════════════════════

call :PRINT_HEADER

:: ── Step 1: Check Python ────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found in PATH.
    echo.
    echo  Please install Python 3.10+ from https://python.org
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Python %PY_VER% detected
echo.

:: ── Step 2: GPU Detection ───────────────────────────────────────────────────
set GPU_FOUND=0
set GPU_NAME=None
set CUDA_WORKS=0

nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        set GPU_NAME=%%g
        set GPU_FOUND=1
    )
)

if "%GPU_FOUND%"=="1" (
    echo  [OK] GPU detected: %GPU_NAME%
    python -c "import xgboost as xgb; import numpy as np; xgb.XGBClassifier(device='cuda',n_estimators=2,verbosity=0).fit(np.random.rand(20,5),np.array([0]*10+[1]*10)); print('CUDA_OK')" 2>nul | find "CUDA_OK" >nul
    if not errorlevel 1 (
        set CUDA_WORKS=1
        echo  [OK] CUDA / XGBoost GPU confirmed working
    ) else (
        echo  [WARN] GPU found but CUDA test failed - GPU option will attempt anyway
        set CUDA_WORKS=0
    )
) else (
    echo  [INFO] No NVIDIA GPU detected - CPU mode only
)
echo.

:: ── Step 3: RAM Detection (info only) ───────────────────────────────────────
for /f "skip=1 tokens=2" %%r in ('wmic OS get TotalVisibleMemorySize 2^>nul') do (
    set /a RAM_MB=%%r/1024
    goto :RAM_DONE
)
:RAM_DONE
if not defined RAM_MB set RAM_MB=8192
echo  [OK] RAM: %RAM_MB% MB

for /f "skip=1 tokens=2" %%p in ('wmic OS get SizeStoredInPagingFiles 2^>nul') do (
    set /a VMEM_MB=%%p/1024
    goto :VMEM_DONE
)
:VMEM_DONE
if not defined VMEM_MB set VMEM_MB=0
if "%VMEM_MB%"=="0" (
    echo  [INFO] Virtual memory page file: not configured (recommended: 3x RAM)
) else (
    echo  [OK] Virtual memory: %VMEM_MB% MB page file configured
)
echo.

:: ── Step 4: Spyder Detection ────────────────────────────────────────────────
set SPYDER_FOUND=0
set SPYDER_CMD=

where spyder >nul 2>&1
if not errorlevel 1 ( set SPYDER_FOUND=1 & set SPYDER_CMD=spyder & echo  [OK] Spyder found in PATH & goto :SPYDER_DONE )

python -c "import spyder" >nul 2>&1
if not errorlevel 1 ( set SPYDER_FOUND=1 & set SPYDER_CMD=python -m spyder & echo  [OK] Spyder found as module & goto :SPYDER_DONE )

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
    if exist %%p ( set SPYDER_FOUND=1 & set SPYDER_CMD=%%p & echo  [OK] Spyder found at %%p & goto :SPYDER_DONE )
)
echo  [INFO] Spyder not installed - terminal-only mode available
:SPYDER_DONE
echo.

:: ── Step 5: Install missing packages ────────────────────────────────────────
echo  [Setup] Checking required packages...
python -c "import xgboost, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, psutil" 2>nul
if errorlevel 1 (
    echo  [Setup] Installing missing packages (one-time setup, takes ~2 min)...
    pip install "xgboost>=2.0.3" numpy pandas scikit-learn scipy matplotlib pyarrow psutil statsmodels --upgrade -q
    if errorlevel 1 (
        echo.
        echo  [ERROR] Package installation failed. Check your internet connection.
        pause & exit /b 1
    )
    echo  [OK] Packages installed
) else (
    echo  [OK] All required packages present
)
echo.

:: ═══════════════════════════════════════════════════════════════════════════
::   INTERACTIVE CONFIGURATION  (2 questions)
:: ═══════════════════════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════════════
echo   CONFIGURATION
echo ════════════════════════════════════════════════════════════════
echo.

:: ── Q1: CPU or GPU ──────────────────────────────────────────────────────────
if "%GPU_FOUND%"=="0" (
    echo  [1/2] Compute: CPU only (no NVIDIA GPU detected)
    set COMPUTE_CHOICE=cpu
    set COMPUTE_LABEL=CPU
    goto :Q2
)

:Q1_LOOP
echo  [1/2] Select compute device:
echo.
echo        [1] GPU  - %GPU_NAME%  (recommended, ~4x faster)
echo        [2] CPU  - All cores
echo        [3] Auto - Try GPU first, fall back to CPU
echo.
set /p Q1_CHOICE="  Your choice (1/2/3): "

if "%Q1_CHOICE%"=="1" ( set COMPUTE_CHOICE=gpu  & set COMPUTE_LABEL=%GPU_NAME% & echo  [OK] GPU mode & goto :Q2 )
if "%Q1_CHOICE%"=="2" ( set COMPUTE_CHOICE=cpu  & set COMPUTE_LABEL=CPU        & echo  [OK] CPU mode & goto :Q2 )
if "%Q1_CHOICE%"=="3" ( set COMPUTE_CHOICE=auto & set COMPUTE_LABEL=Auto       & echo  [OK] Auto mode & goto :Q2 )
echo  [!] Enter 1, 2, or 3.
echo.
goto :Q1_LOOP

:Q2
echo.

:: ── Q2: Terminal only or Terminal + Spyder ─────────────────────────────────
if "%SPYDER_FOUND%"=="0" (
    echo  [2/2] Output mode: Terminal only (Spyder not installed)
    echo        Install Spyder:  pip install spyder
    set USE_SPYDER=0
    set OUTPUT_LABEL=Terminal only
    goto :SUMMARY
)

:Q2_LOOP
echo  [2/2] Output mode:
echo.
echo        [1] Terminal only
echo        [2] Terminal + Spyder - charts open in Spyder IDE
echo            (closing Spyder will NOT stop the pipeline)
echo.
set /p Q2_CHOICE="  Your choice (1/2): "

if "!Q2_CHOICE!"=="1" ( set USE_SPYDER=0 & set OUTPUT_LABEL=Terminal only        & echo  [OK] Terminal only & goto :SUMMARY )
if "!Q2_CHOICE!"=="2" ( set USE_SPYDER=1 & set OUTPUT_LABEL=Terminal + Spyder    & echo  [OK] Terminal + Spyder & goto :SUMMARY )
echo  [!] Enter 1 or 2.
echo.
goto :Q2_LOOP

:SUMMARY
echo.

:: ═══════════════════════════════════════════════════════════════════════════
::   SUMMARY
:: ═══════════════════════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════════════
echo   READY TO RUN
echo ════════════════════════════════════════════════════════════════
echo.
echo   Compute  : %COMPUTE_LABEL%
echo   Output   : %OUTPUT_LABEL%
echo   Data dir : .\data\
echo   Results  : .\results\
echo.
echo ════════════════════════════════════════════════════════════════
echo.
set /p CONFIRM="  Start? (Y/N): "
if /i not "%CONFIRM%"=="Y" ( echo  Cancelled. & timeout /t 3 /nobreak >nul & exit /b 0 )
echo.

:: ═══════════════════════════════════════════════════════════════════════════
::   APPLY SETTINGS
:: ═══════════════════════════════════════════════════════════════════════════

:: ── GPU environment ────────────────────────────────────────────────────────
if "%COMPUTE_CHOICE%"=="gpu" (
    set CUDA_VISIBLE_DEVICES=0
    set CUDA_DEVICE_ORDER=PCI_E_BUS_ID
)

:: ── Power plan ────────────────────────────────────────────────────────────
echo  [Setup] Setting High Performance power plan...
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1
if not errorlevel 1 ( echo  [OK] High Performance power plan active
) else ( echo  [INFO] Could not set power plan (non-critical) )

echo.

:: ═══════════════════════════════════════════════════════════════════════════
::   SPYDER LAUNCH
:: ═══════════════════════════════════════════════════════════════════════════

if "%USE_SPYDER%"=="1" (
    echo ════════════════════════════════════════════════════════════════
    echo   LAUNCHING SPYDER
    echo ════════════════════════════════════════════════════════════════
    echo.
    echo  Spyder will open in background. Once it loads:
    echo    1. File ^> Open ^> azalyst_spyder_monitor.py
    echo    2. Press F5 to start the live chart (refreshes every 5s)
    echo    3. You can close Spyder at any time — pipeline keeps running
    echo.

    if "%SPYDER_CMD%"=="spyder" (
        start "" /B spyder --new-instance --workdir="%~dp0" 2>nul
    ) else if "%SPYDER_CMD%"=="python -m spyder" (
        start "" /B python -m spyder --new-instance --workdir="%~dp0" 2>nul
    ) else (
        start "" /B %SPYDER_CMD% --new-instance --workdir="%~dp0" 2>nul
    )

    echo  [OK] Spyder launch sent. Waiting 8s for it to initialize...
    timeout /t 8 /nobreak >nul
    echo.
)

:: ═══════════════════════════════════════════════════════════════════════════
::   RUN THE PIPELINE
:: ═══════════════════════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════════════
echo   RUNNING AZALYST PIPELINE
echo ════════════════════════════════════════════════════════════════
echo.
echo  Compute  : %COMPUTE_LABEL%
echo  Output   : %OUTPUT_LABEL%
echo  Data dir : .\data\
echo  Results  : .\results\
echo  Started  : %date% %time%
echo.
echo ────────────────────────────────────────────────────────────────
echo.

python azalyst_engine.py --data-dir ./data --out-dir ./results --skip-ic

set EXIT_CODE=%errorlevel%

:: ═══════════════════════════════════════════════════════════════════════════
::   DONE
:: ═══════════════════════════════════════════════════════════════════════════

echo.
echo ────────────────────────────────────────────────────────────────

if "%EXIT_CODE%"=="0" (
    color 0A
    echo.
    echo  Pipeline complete! Results saved to .\results\
    echo.
    echo  Key output files:
    echo    .\results\ic_analysis.csv
    echo    .\results\backtest_pnl.csv
    echo    .\results\performance_summary.csv
    echo.
) else (
    color 0C
    echo.
    echo  [ERROR] Pipeline exited with code %EXIT_CODE%
    echo.
    echo  Common fixes:
    echo    No data?        Add .parquet files to .\data\
    echo    GPU error?      Re-run, select CPU or Auto mode
    echo    XGBoost error?  pip install xgboost --upgrade
    echo.
)

echo  Finished: %date% %time%
echo.

:: Restore balanced power plan
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e >nul 2>&1

echo  Press any key to close...
pause >nul
exit /b %EXIT_CODE%

:: ═══════════════════════════════════════════════════════════════════════════
:PRINT_HEADER
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║          AZALYST ALPHA RESEARCH ENGINE  v3.1                ║
echo  ║   20 Cross-Sectional Factors  ^|  IC Analysis  ^|  Backtest   ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
echo  System scan in progress...
echo.
goto :EOF
