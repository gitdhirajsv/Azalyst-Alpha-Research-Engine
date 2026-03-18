@echo off
setlocal EnableDelayedExpansion
title Azalyst Alpha Research Engine — Launcher
color 0A
chcp 65001 >nul 2>&1

:: ── Set working directory to wherever this .bat file lives ──────────────────
cd /d "%~dp0"

:: ═══════════════════════════════════════════════════════════════════════════
::   AZALYST LAUNCHER  v3.0
::   Requires: azalyst_local_gpu.py  (argparse: --gpu / --no-gpu / --year2-only)
::             azalyst_weekly_loop.py (argparse: --gpu / --year2-only)
::             build_feature_cache.py (argparse: --data-dir / --out-dir / --workers)
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

:: ── Step 3: RAM / Virtual RAM Detection ─────────────────────────────────────
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
    echo         To set it: System Properties ^> Advanced ^> Performance ^> Virtual Memory
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
    pip install "xgboost>=2.0.3" numpy pandas scikit-learn scipy matplotlib pyarrow psutil --upgrade -q
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
::   INTERACTIVE CONFIGURATION  (4 questions)
:: ═══════════════════════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════════════
echo   CONFIGURATION
echo ════════════════════════════════════════════════════════════════
echo.

:: ── Q1: CPU or GPU ──────────────────────────────────────────────────────────
if "%GPU_FOUND%"=="0" (
    echo  [1/4] Compute: CPU only (no NVIDIA GPU detected)
    set COMPUTE_CHOICE=cpu
    set COMPUTE_LABEL=CPU
    goto :Q2
)

:Q1_LOOP
echo  [1/4] Select compute device:
echo.
echo        [1] GPU  - %GPU_NAME%  (recommended, ~4x faster)
echo        [2] CPU  - All cores   (slower but always works)
echo        [3] Auto - Try GPU first, fall back to CPU if CUDA fails
echo.
set /p Q1_CHOICE="  Your choice (1/2/3): "

if "%Q1_CHOICE%"=="1" ( set COMPUTE_CHOICE=gpu  & set COMPUTE_LABEL=GPU (RTX 2050) & echo  [OK] GPU mode & goto :Q2 )
if "%Q1_CHOICE%"=="2" ( set COMPUTE_CHOICE=cpu  & set COMPUTE_LABEL=CPU            & echo  [OK] CPU mode & goto :Q2 )
if "%Q1_CHOICE%"=="3" ( set COMPUTE_CHOICE=auto & set COMPUTE_LABEL=Auto           & echo  [OK] Auto mode & goto :Q2 )
echo  [!] Enter 1, 2, or 3.
echo.
goto :Q1_LOOP

:Q2
echo.

:: ── Q2: Terminal only or Terminal + Spyder ─────────────────────────────────
:Q2_LOOP
echo  [2/4] Output mode:
echo.
if "%SPYDER_FOUND%"=="1" (
    echo        [1] Terminal only     - See progress in this window
    echo        [2] Terminal + Spyder - Live 4-panel ML chart in Spyder IDE
    echo            (closing Spyder later will NOT stop the pipeline)
    echo.
    set /p Q2_CHOICE="  Your choice (1/2): "
    if "!Q2_CHOICE!"=="1" ( set USE_SPYDER=0 & echo  [OK] Terminal only & goto :Q3 )
    if "!Q2_CHOICE!"=="2" ( set USE_SPYDER=1 & echo  [OK] Terminal + Spyder & goto :Q3 )
    echo  [!] Enter 1 or 2.
    echo.
    goto :Q2_LOOP
) else (
    echo        [1] Terminal only  (Spyder not installed)
    echo        Install Spyder:  pip install spyder
    echo.
    set USE_SPYDER=0
    set /p Q2_DUMMY="  Press Enter to continue..."
    goto :Q3
)

:Q3
echo.

:: ── Q3: Virtual RAM (page file) optimization ─────────────────────────────────
:Q3_LOOP
echo  [3/4] Memory settings:
echo.
echo        Your RAM: %RAM_MB% MB
if not "%VMEM_MB%"=="0" echo        Virtual memory page file: %VMEM_MB% MB
echo.
echo        [1] Optimized   - Tells Python libs to spill to disk (uses virtual RAM)
echo        [2] Conservative - Safer for 8GB RAM systems
echo        [3] Default      - No changes (fine for 16GB+ RAM)
echo.
set /p Q3_CHOICE="  Your choice (1/2/3): "

if "%Q3_CHOICE%"=="1" ( set MEM_MODE=optimized    & echo  [OK] Optimized  & goto :Q4 )
if "%Q3_CHOICE%"=="2" ( set MEM_MODE=conservative & echo  [OK] Conservative & goto :Q4 )
if "%Q3_CHOICE%"=="3" ( set MEM_MODE=default      & echo  [OK] Default & goto :Q4 )
echo  [!] Enter 1, 2, or 3.
echo.
goto :Q3_LOOP

:Q4
echo.

:: ── Q4: What to run ──────────────────────────────────────────────────────────
:Q4_LOOP
echo  [4/4] Pipeline scope:
echo.
echo        [1] Full pipeline   - Year 1+2 training + Year 3 walk-forward (3-5 hrs)
echo        [2] Quick test      - Year 2 only, shorter test window (~1-2 hrs)
echo        [3] GPU test only   - Verify GPU + speed benchmark (2 min)
echo        [4] Resume          - Skip base model training if already cached
echo.
set /p Q4_CHOICE="  Your choice (1/2/3/4): "

if "%Q4_CHOICE%"=="1" ( set RUN_MODE=full    & set YEAR2_ONLY_FLAG=   & echo  [OK] Full pipeline & goto :Q4_DONE )
if "%Q4_CHOICE%"=="2" ( set RUN_MODE=quick   & set YEAR2_ONLY_FLAG=--year2-only & echo  [OK] Quick test & goto :Q4_DONE )
if "%Q4_CHOICE%"=="3" ( set RUN_MODE=gpucheck & echo  [OK] GPU diagnostic only & goto :Q4_DONE )
if "%Q4_CHOICE%"=="4" ( set RUN_MODE=resume  & set YEAR2_ONLY_FLAG=   & echo  [OK] Resume mode & goto :Q4_DONE )
echo  [!] Enter 1, 2, 3, or 4.
echo.
goto :Q4_LOOP

:Q4_DONE
echo.

:: ═══════════════════════════════════════════════════════════════════════════
::   SUMMARY
:: ═══════════════════════════════════════════════════════════════════════════

echo.
echo ════════════════════════════════════════════════════════════════
echo   LAUNCH SUMMARY
echo ════════════════════════════════════════════════════════════════
echo.
echo   Compute       : %COMPUTE_LABEL%
if "%USE_SPYDER%"=="1" (
    echo   Output        : Terminal + Spyder (live charts)
) else (
    echo   Output        : Terminal only
)
echo   Memory mode   : %MEM_MODE%
echo   Pipeline      : %RUN_MODE%
echo   Working dir   : %~dp0
echo.
echo ════════════════════════════════════════════════════════════════
echo.
set /p CONFIRM="  Start? (Y/N): "
if /i not "%CONFIRM%"=="Y" ( echo  Cancelled. & timeout /t 3 /nobreak >nul & exit /b 0 )
echo.

:: ═══════════════════════════════════════════════════════════════════════════
::   APPLY SETTINGS
:: ═══════════════════════════════════════════════════════════════════════════

:: ── Memory environment variables ─────────────────────────────────────────────
if "%MEM_MODE%"=="optimized" (
    set MMAP_DIR=%TEMP%\azalyst_mmap
    if not exist "%MMAP_DIR%" mkdir "%MMAP_DIR%"
    :: Tell joblib/sklearn to spill large arrays to disk (uses Windows page file)
    set JOBLIB_TEMP_FOLDER=%MMAP_DIR%
    :: Tell numpy to use disk-backed memmap for temp arrays
    set NPY_DISTUTILS_APPEND_FLAGS=1
    echo  [Setup] Memory spill-to-disk: ON (using %MMAP_DIR%)
)
if "%MEM_MODE%"=="conservative" (
    set JOBLIB_TEMP_FOLDER=%TEMP%
    echo  [Setup] Conservative memory mode: ON
)

:: ── GPU environment ────────────────────────────────────────────────────────
if not "%COMPUTE_CHOICE%"=="cpu" (
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
::   BUILD GPU FLAG  (wired to argparse in azalyst_local_gpu.py)
:: ═══════════════════════════════════════════════════════════════════════════

set GPU_FLAG=
if "%COMPUTE_CHOICE%"=="gpu"  set GPU_FLAG=--gpu
if "%COMPUTE_CHOICE%"=="auto" set GPU_FLAG=
if "%COMPUTE_CHOICE%"=="cpu"  set GPU_FLAG=--no-gpu

:: ═══════════════════════════════════════════════════════════════════════════
::   GPU DIAGNOSTIC MODE
:: ═══════════════════════════════════════════════════════════════════════════

if "%RUN_MODE%"=="gpucheck" (
    echo ════════════════════════════════════════════════════════════════
    echo   GPU DIAGNOSTIC
    echo ════════════════════════════════════════════════════════════════
    echo.
    nvidia-smi 2>nul || echo  nvidia-smi not found
    echo.
    python -c "
import xgboost as xgb, numpy as np, time, platform
print(f'  Platform: {platform.processor()}')
print(f'  XGBoost : {xgb.__version__}')
X = np.random.rand(50000, 65).astype('float32')
y = np.random.randint(0, 2, 50000)
for label, kw in [('GPU (cuda)', {'device':'cuda'}), ('CPU (all cores)', {'nthread':-1})]:
    try:
        t0 = time.time()
        xgb.XGBClassifier(**kw, n_estimators=100, verbosity=0).fit(X, y)
        print(f'  {label:<20}: {time.time()-t0:.1f}s')
    except Exception as e:
        print(f'  {label:<20}: FAILED ({e})')
"
    echo.
    pause
    exit /b 0
)

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
        start "" %SPYDER_CMD% --new-instance --workdir="%~dp0" 2>nul
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
echo  Pipeline : %RUN_MODE%
echo  Started  : %date% %time%
echo.
echo  Live progress prints every 4 weeks.
echo  Results auto-save to .\results\
if "%USE_SPYDER%"=="1" (
    echo  Open azalyst_spyder_monitor.py in Spyder and press F5 for charts.
)
echo.
echo ────────────────────────────────────────────────────────────────
echo.

:: Run azalyst_local_gpu.py with the correct argparse flags
:: --gpu / --no-gpu  →  wired to the GPU choice above
:: --year2-only      →  wired to pipeline scope Q4
python azalyst_local_gpu.py %GPU_FLAG% %YEAR2_ONLY_FLAG%

set EXIT_CODE=%errorlevel%

:: ═══════════════════════════════════════════════════════════════════════════
::   DONE
:: ═══════════════════════════════════════════════════════════════════════════

echo.
echo ────────────────────────────────────────────────────────────────

if "%EXIT_CODE%"=="0" (
    color 0A
    echo.
    echo  Pipeline completed successfully!
    echo.
    echo  Key output files:
    echo    .\results\performance_year3.json      ^<-- send to Claude for grading
    echo    .\results\weekly_summary_year3.csv
    echo    .\results\all_trades_year3.csv
    echo    .\results\performance_year3.png
    echo.

    if exist ".\results\performance_year3.png" (
        set /p OPEN_CHART="  Open results chart now? (Y/N): "
        if /i "!OPEN_CHART!"=="Y" start "" ".\results\performance_year3.png"
    )
) else (
    color 0C
    echo.
    echo  [ERROR] Pipeline exited with code %EXIT_CODE%
    echo.
    echo  Common fixes:
    echo    No data?        Add SYMBOL.parquet files to .\data\
    echo    GPU OOM?        Re-run, select CPU or Conservative RAM
    echo    XGBoost error?  pip install xgboost --upgrade
    echo    CUDA error?     Re-run, select Auto or CPU mode
    echo.
    echo  GPU OOM specifically: in azalyst_local_gpu.py change:
    echo    MAX_TRAIN_ROWS = 1_000_000
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
echo  ║          AZALYST ALPHA RESEARCH ENGINE  v2.0                ║
echo  ║   XGBoost CUDA  ^|  65 Features  ^|  Purged K-Fold            ║
echo  ║   i5-11260H + RTX 2050  ^|  Walk-Forward Research            ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
echo  System scan in progress...
echo.
goto :EOF
