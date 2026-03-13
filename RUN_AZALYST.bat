@echo off
setlocal EnableDelayedExpansion

title Azalyst Master Runner

echo.
echo  ============================================================
echo       AZALYST ALPHA RESEARCH ENGINE  --  MASTER RUNNER v3.0
echo       Stop anytime - Restart to auto-resume from checkpoint
echo  ============================================================
echo.

:: ── CONFIGURATION
set DATA_DIR=.\data
set FEATURE_DIR=.\feature_cache
set OUTPUT_DIR=.\azalyst_output
set MODEL_DIR=.\models
set CKPT_DIR=.\pipeline_checkpoints

set TRAIN_DAYS=365
set PREDICT_DAYS=30
set HORIZON_BARS=48
set WORKERS=4
set MAX_SYMBOLS=0
set RESAMPLE=1h

:: ── Create essential folders
if not exist "%OUTPUT_DIR%"  mkdir "%OUTPUT_DIR%"
if not exist "%MODEL_DIR%"   mkdir "%MODEL_DIR%"
if not exist "%CKPT_DIR%"    mkdir "%CKPT_DIR%"

:: ── SELF-HEALING ENVIRONMENT DETECTION
echo  [SYSTEM] Checking environment...

:: 1. Detect Python
set PYTHON_EXE=python
where %PYTHON_EXE% >nul 2>nul
if errorlevel 1 (
    set PYTHON_EXE=py
    where !PYTHON_EXE! >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found! Please install Python 3.10+ and add it to PATH.
        pause
        exit /b 1
    )
)

:: 2. Handle Virtual Environment
set VENV_DIR=.venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo  [INFO] Creating virtual environment in %VENV_DIR%...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. 
        pause
        exit /b 1
    )
    set INITIAL_INSTALL=1
)

:: 3. Activate Venv
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 4. Auto-install dependencies
if "%INITIAL_INSTALL%"=="1" (
    echo  [INFO] Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo  [SUCCESS] Environment ready.
)

:: ── Read batch checkpoint if exists
:: LAST_DONE = the last fully completed step number (0 = nothing done yet)
set BATCH_CKPT_FILE=batch_checkpoint.json
set LAST_DONE=0
if exist "!BATCH_CKPT_FILE!" (
    echo  [INFO] Found batch checkpoint !BATCH_CKPT_FILE! - reading progress...
    for /f "tokens=2 delims=:," %%a in ('findstr /i "batch_step" !BATCH_CKPT_FILE!') do (
        set LAST_DONE=%%a
        set LAST_DONE=!LAST_DONE: =!
    )
    echo  [INFO] Last completed batch step: !LAST_DONE!
)

echo.
echo  Data folder : %DATA_DIR%
echo  Features    : %FEATURE_DIR%
echo  Output      : %OUTPUT_DIR%
echo  Workers     : %WORKERS%
echo  Resample    : %RESAMPLE%
echo  Last done   : Step !LAST_DONE!
echo.

:: ── Helper macro explanation:
::    Each step checks: if LAST_DONE GTR <step_number> → already done, skip
::    If LAST_DONE EQU <step_number> → this step was interrupted, re-run it
::    If LAST_DONE LSS <step_number> → not yet started, run it

:: ============================================================
::  STEP 1 -- BUILD FEATURE CACHE
:: ============================================================
if !LAST_DONE! GTR 1 (
    echo  [STEP 1/8]  Feature cache  -- ALREADY DONE, skipping
) else (
    echo  [STEP 1/8]  Building ML feature cache ...
    if "%MAX_SYMBOLS%"=="0" (
        python build_feature_cache.py --data-dir %DATA_DIR% --out-dir %FEATURE_DIR% --workers %WORKERS%
    ) else (
        python build_feature_cache.py --data-dir %DATA_DIR% --out-dir %FEATURE_DIR% --workers %WORKERS% --max-symbols %MAX_SYMBOLS%
    )
    if errorlevel 1 ( echo [ERROR] Step 1 failed & pause & exit /b 1 )
    echo {"batch_step": 1, "last_step": "step1_features", "timestamp": "%date% %time%"} > batch_checkpoint.json
    set LAST_DONE=1
    echo  [STEP 1/8]  Feature cache complete.
)

:: ============================================================
::  STEP 2 -- WALK-FORWARD ML SIMULATOR
:: ============================================================
if !LAST_DONE! GTR 2 (
    echo  [STEP 2/8]  Walk-forward simulator  -- ALREADY DONE, skipping
) else (
    echo  [STEP 2/8]  Walk-forward ML simulator ...
    if "%MAX_SYMBOLS%"=="0" (
        python walkforward_simulator.py --data-dir %DATA_DIR% --feature-dir %FEATURE_DIR% --train-days %TRAIN_DAYS% --predict-days %PREDICT_DAYS% --horizon-bars %HORIZON_BARS%
    ) else (
        python walkforward_simulator.py --data-dir %DATA_DIR% --feature-dir %FEATURE_DIR% --train-days %TRAIN_DAYS% --predict-days %PREDICT_DAYS% --horizon-bars %HORIZON_BARS% --max-symbols %MAX_SYMBOLS%
    )
    if errorlevel 1 ( echo [ERROR] Step 2 failed & pause & exit /b 1 )
    echo {"batch_step": 2, "last_step": "step2_walkforward", "timestamp": "%date% %time%"} > batch_checkpoint.json
    set LAST_DONE=2
    echo  [STEP 2/8]  Walk-forward simulation complete.
)

:: ============================================================
::  STEP 3 -- CROSS-SECTIONAL FACTOR ENGINE
:: ============================================================
if !LAST_DONE! GTR 3 (
    echo  [STEP 3/8]  Factor engine  -- ALREADY DONE, skipping
) else (
    echo  [STEP 3/8]  Cross-sectional factor engine ...
    if "%MAX_SYMBOLS%"=="0" (
        python azalyst_engine.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR% --resample %RESAMPLE% --workers %WORKERS%
    ) else (
        python azalyst_engine.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR% --resample %RESAMPLE% --workers %WORKERS% --max-symbols %MAX_SYMBOLS%
    )
    if errorlevel 1 ( echo [ERROR] Step 3 failed & pause & exit /b 1 )
    echo {"batch_step": 3, "last_step": "step3_factors", "timestamp": "%date% %time%"} > batch_checkpoint.json
    set LAST_DONE=3
    echo  [STEP 3/8]  Factor engine complete.
)

:: ============================================================
::  STEP 3.5 -- STATISTICAL VALIDATION (Fama-MacBeth + Style Neutral)
:: ============================================================
echo  [STEP 3.5]  Statistical validation (Fama-MacBeth, style neutralization) ...
if "%MAX_SYMBOLS%"=="0" (
    python azalyst_validator.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR% --ic-csv %OUTPUT_DIR%\ic_analysis.csv
) else (
    python azalyst_validator.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR% --ic-csv %OUTPUT_DIR%\ic_analysis.csv --max-symbols %MAX_SYMBOLS%
)
if errorlevel 1 ( echo [WARN] Validation had an issue - continuing... )

:: ============================================================
::  STEP 4 -- STATISTICAL ARBITRAGE SCANNER
:: ============================================================
if !LAST_DONE! GTR 4 (
    echo  [STEP 4/8]  StatArb scanner  -- ALREADY DONE, skipping
) else (
    echo  [STEP 4/8]  Statistical arbitrage scanner ...
    if "%MAX_SYMBOLS%"=="0" (
        python azalyst_statarb.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR%
    ) else (
        python azalyst_statarb.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR% --max-symbols %MAX_SYMBOLS%
    )
    if errorlevel 1 ( echo [ERROR] Step 4 failed & pause & exit /b 1 )
    echo {"batch_step": 4, "last_step": "step4_statarb", "timestamp": "%date% %time%"} > batch_checkpoint.json
    set LAST_DONE=4
    echo  [STEP 4/8]  StatArb scan complete.
)

:: ============================================================
::  STEP 5 -- ML MODEL TRAINING
:: ============================================================
if !LAST_DONE! GTR 5 (
    echo  [STEP 5/8]  ML training  -- ALREADY DONE, skipping
) else (
    echo  [STEP 5/8]  ML model training ...
    python azalyst_ml.py --data-dir %DATA_DIR% --out-dir %MODEL_DIR% --model all --max-symbols 0 --max-samples 0
    if errorlevel 1 ( echo [ERROR] Step 5 failed & pause & exit /b 1 )
    echo {"batch_step": 5, "last_step": "step5_ml", "timestamp": "%date% %time%"} > batch_checkpoint.json
    set LAST_DONE=5
    echo  [STEP 5/8]  ML training complete.
)

:: ============================================================
::  STEP 6 -- BENCHMARK ANALYSIS
:: ============================================================
if !LAST_DONE! GTR 6 (
    echo  [STEP 6/8]  Benchmark  -- ALREADY DONE, skipping
) else (
    echo  [STEP 6/8]  Benchmark analysis ...
    python azalyst_benchmark.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR%
    if errorlevel 1 ( echo [ERROR] Step 6 failed & pause & exit /b 1 )
    echo {"batch_step": 6, "last_step": "step6_benchmark", "timestamp": "%date% %time%"} > batch_checkpoint.json
    set LAST_DONE=6
    echo  [STEP 6/8]  Benchmark complete.
)

:: ============================================================
::  STEP 7 -- TEARSHEET + REPORTS
:: ============================================================
if !LAST_DONE! GTR 7 (
    echo  [STEP 7/8]  Reports  -- ALREADY DONE, skipping
) else (
    echo  [STEP 7/8]  Generating tearsheet and research reports ...
    python azalyst_tearsheet.py --trades paper_trades.csv --out-dir %OUTPUT_DIR%
    if errorlevel 1 ( echo [WARN] Tearsheet had an issue - continuing... )
    python azalyst_report.py --out-dir %OUTPUT_DIR%
    if errorlevel 1 ( echo [ERROR] Step 7 failed & pause & exit /b 1 )
    echo {"batch_step": 7, "last_step": "step7_reports", "timestamp": "%date% %time%"} > batch_checkpoint.json
    set LAST_DONE=7
    echo  [STEP 7/8]  Reports complete.
)

:: ============================================================
::  STEP 8 -- ORCHESTRATOR
:: ============================================================
if !LAST_DONE! GTR 8 (
    echo  [STEP 8/8]  Orchestrator  -- ALREADY DONE, skipping
) else (
    echo  [STEP 8/8]  Orchestrator: fusing all signals ...
    if "%MAX_SYMBOLS%"=="0" (
        python azalyst_orchestrator.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR% --model-dir %MODEL_DIR% --resample %RESAMPLE% --workers %WORKERS%
    ) else (
        python azalyst_orchestrator.py --data-dir %DATA_DIR% --out-dir %OUTPUT_DIR% --model-dir %MODEL_DIR% --resample %RESAMPLE% --workers %WORKERS% --max-symbols %MAX_SYMBOLS%
    )
    if errorlevel 1 (
        echo [WARN] Orchestrator had an issue. Other steps still completed.
    ) else (
        echo {"batch_step": 8, "last_step": "step8_orchestrator", "timestamp": "%date% %time%"} > batch_checkpoint.json
        set LAST_DONE=8
        echo  [STEP 8/8]  Done. Final signals saved to %OUTPUT_DIR%\signals.csv
    )
)

:: ============================================================
::  DONE
:: ============================================================
echo.
echo  ============================================================
echo                    PIPELINE COMPLETE
echo  ============================================================
echo.
echo  Outputs in: %OUTPUT_DIR%\
echo    signals.csv              -- FINAL ranked signal table
echo    factor_ic_results.csv    -- IC / ICIR / alpha analysis
echo    statarb_pairs.csv        -- cointegrated pairs found
echo    paper_trades.csv         -- simulated trades
echo    performance_metrics.csv  -- per-cycle performance
echo    models\                  -- trained ML snapshots
echo.
echo  To force re-run a step: set checkpoint.json cycle_index to (step - 1)
echo.
pause
