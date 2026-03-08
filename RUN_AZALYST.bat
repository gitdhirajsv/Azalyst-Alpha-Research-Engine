@echo off
setlocal EnableDelayedExpansion
:: ═══════════════════════════════════════════════════════════════════════════
::    AZALYST ALPHA RESEARCH ENGINE  —  MASTER RUNNER  v2.0
::
::    Double-click to run the full pipeline on your existing data folder.
::    SAFE TO STOP AND RESTART AT ANY TIME.
::    Each step saves a .done checkpoint — on re-run it skips finished steps
::    and continues exactly where it left off.
::
::    Pipeline:
::      Step 1 — Build ML feature cache          (build_feature_cache.py)
::      Step 2 — Walk-forward ML simulator        (walkforward_simulator.py)
::      Step 3 — Cross-sectional factor engine    (azalyst_engine.py)
::      Step 4 — Statistical arbitrage scanner    (azalyst_statarb.py)
::      Step 5 — ML model training                (azalyst_ml.py)
::      Step 6 — Benchmark analysis               (azalyst_benchmark.py)
::      Step 7 — Tearsheet + reports              (azalyst_tearsheet.py / azalyst_report.py)
::
::    To FORCE a step to re-run, delete its .done file inside:
::      pipeline_checkpoints\
::
:: ═══════════════════════════════════════════════════════════════════════════

title Azalyst Master Runner

echo.
echo  ╔═══════════════════════════════════════════════════════════════════╗
echo  ║         AZALYST ALPHA RESEARCH ENGINE  —  MASTER RUNNER  v2.0    ║
echo  ║    Stop anytime  ^|  Restart to auto-resume from last checkpoint   ║
echo  ╚═══════════════════════════════════════════════════════════════════╝
echo.

:: ── CONFIGURATION ───────────────────────────────────────────────────────────
set DATA_DIR=.\data
set FEATURE_DIR=.\feature_cache
set OUTPUT_DIR=.\azalyst_output
set MODEL_DIR=.\models
set CKPT_DIR=.\pipeline_checkpoints

set TRAIN_DAYS=365
set PREDICT_DAYS=30
set HORIZON_BARS=48
set WORKERS=4

:: Set MAX_SYMBOLS=0 to use ALL symbols, or e.g. 30 for a quick test run
set MAX_SYMBOLS=0

:: Timeframe for factor engine (options: 5min  15min  1H  4H)
set RESAMPLE=1H

:: ── Create folders ───────────────────────────────────────────────────────────
if not exist "%OUTPUT_DIR%"  mkdir "%OUTPUT_DIR%"
if not exist "%MODEL_DIR%"   mkdir "%MODEL_DIR%"
if not exist "%CKPT_DIR%"    mkdir "%CKPT_DIR%"

:: ── Activate virtual environment if one exists ───────────────────────────────
if exist ".\venv\Scripts\activate.bat" (
    echo  [Setup] Activating venv...
    call .\venv\Scripts\activate.bat
) else if exist ".\.venv\Scripts\activate.bat" (
    echo  [Setup] Activating .venv...
    call .\.venv\Scripts\activate.bat
)

echo.
echo  Data folder : %DATA_DIR%
echo  Features    : %FEATURE_DIR%
echo  Output      : %OUTPUT_DIR%
echo  Workers     : %WORKERS%
echo  Resample    : %RESAMPLE%
echo.
echo  ───────────────────────────────────────────────────────────────────
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 1 — BUILD FEATURE CACHE
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step1_features.done" (
    echo  [STEP 1/7]  Feature cache  ^|  ALREADY DONE — skipping
    echo              Delete  %CKPT_DIR%\step1_features.done  to re-run
) else (
    echo  [STEP 1/7]  Building ML feature cache ...
    echo  ───────────────────────────────────────────────────────────────────

    if "%MAX_SYMBOLS%"=="0" (
        python build_feature_cache.py ^
            --data-dir  %DATA_DIR% ^
            --out-dir   %FEATURE_DIR% ^
            --workers   %WORKERS%
    ) else (
        python build_feature_cache.py ^
            --data-dir    %DATA_DIR% ^
            --out-dir     %FEATURE_DIR% ^
            --workers     %WORKERS% ^
            --max-symbols %MAX_SYMBOLS%
    )

    if errorlevel 1 (
        echo.
        echo  [ERROR] Step 1 failed. Fix the error above then re-run this file.
        pause
        exit /b 1
    )
    echo. > "%CKPT_DIR%\step1_features.done"
    echo  [STEP 1/7]  Feature cache complete.
)
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 2 — WALK-FORWARD ML SIMULATOR
::  (has its own internal checkpoint.json — resumable mid-simulation too)
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step2_walkforward.done" (
    echo  [STEP 2/7]  Walk-forward simulator  ^|  ALREADY DONE — skipping
) else (
    echo  [STEP 2/7]  Walk-forward ML simulator ...
    echo             ^(Also auto-resumes from checkpoint.json if interrupted mid-run^)
    echo  ───────────────────────────────────────────────────────────────────

    if "%MAX_SYMBOLS%"=="0" (
        python walkforward_simulator.py ^
            --data-dir      %DATA_DIR% ^
            --feature-dir   %FEATURE_DIR% ^
            --train-days    %TRAIN_DAYS% ^
            --predict-days  %PREDICT_DAYS% ^
            --horizon-bars  %HORIZON_BARS%
    ) else (
        python walkforward_simulator.py ^
            --data-dir      %DATA_DIR% ^
            --feature-dir   %FEATURE_DIR% ^
            --train-days    %TRAIN_DAYS% ^
            --predict-days  %PREDICT_DAYS% ^
            --horizon-bars  %HORIZON_BARS% ^
            --max-symbols   %MAX_SYMBOLS%
    )

    if errorlevel 1 (
        echo.
        echo  [ERROR] Step 2 failed or was interrupted.
        echo          Re-run this file — it resumes from checkpoint.json automatically.
        pause
        exit /b 1
    )
    echo. > "%CKPT_DIR%\step2_walkforward.done"
    echo  [STEP 2/7]  Walk-forward simulation complete.
)
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 3 — CROSS-SECTIONAL FACTOR ENGINE
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step3_factors.done" (
    echo  [STEP 3/7]  Factor engine  ^|  ALREADY DONE — skipping
) else (
    echo  [STEP 3/7]  Cross-sectional factor engine ...
    echo  ───────────────────────────────────────────────────────────────────

    if "%MAX_SYMBOLS%"=="0" (
        python azalyst_engine.py ^
            --data-dir  %DATA_DIR% ^
            --out-dir   %OUTPUT_DIR% ^
            --resample  %RESAMPLE% ^
            --workers   %WORKERS%
    ) else (
        python azalyst_engine.py ^
            --data-dir    %DATA_DIR% ^
            --out-dir     %OUTPUT_DIR% ^
            --resample    %RESAMPLE% ^
            --workers     %WORKERS% ^
            --max-symbols %MAX_SYMBOLS%
    )

    if errorlevel 1 (
        echo.
        echo  [ERROR] Step 3 failed. Re-run this file to retry.
        pause
        exit /b 1
    )
    echo. > "%CKPT_DIR%\step3_factors.done"
    echo  [STEP 3/7]  Factor engine complete.
)
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 4 — STATISTICAL ARBITRAGE SCANNER
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step4_statarb.done" (
    echo  [STEP 4/7]  StatArb scanner  ^|  ALREADY DONE — skipping
) else (
    echo  [STEP 4/7]  Statistical arbitrage scanner ...
    echo  ───────────────────────────────────────────────────────────────────

    if "%MAX_SYMBOLS%"=="0" (
        python azalyst_statarb.py ^
            --data-dir  %DATA_DIR% ^
            --out-dir   %OUTPUT_DIR% ^
            --workers   %WORKERS%
    ) else (
        python azalyst_statarb.py ^
            --data-dir    %DATA_DIR% ^
            --out-dir     %OUTPUT_DIR% ^
            --workers     %WORKERS% ^
            --max-symbols %MAX_SYMBOLS%
    )

    if errorlevel 1 (
        echo.
        echo  [ERROR] Step 4 failed. Re-run this file to retry.
        pause
        exit /b 1
    )
    echo. > "%CKPT_DIR%\step4_statarb.done"
    echo  [STEP 4/7]  StatArb scan complete.
)
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 5 — ML MODEL TRAINING
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step5_ml.done" (
    echo  [STEP 5/7]  ML training  ^|  ALREADY DONE — skipping
) else (
    echo  [STEP 5/7]  ML model training (pump/dump + regime + return predictor) ...
    echo  ───────────────────────────────────────────────────────────────────

    python azalyst_ml.py ^
        --data-dir  %DATA_DIR% ^
        --out-dir   %MODEL_DIR% ^
        --model     all

    if errorlevel 1 (
        echo.
        echo  [ERROR] Step 5 failed. Re-run this file to retry.
        pause
        exit /b 1
    )
    echo. > "%CKPT_DIR%\step5_ml.done"
    echo  [STEP 5/7]  ML training complete.
)
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 6 — BENCHMARK ANALYSIS
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step6_benchmark.done" (
    echo  [STEP 6/7]  Benchmark  ^|  ALREADY DONE — skipping
) else (
    echo  [STEP 6/7]  Benchmark analysis ...
    echo  ───────────────────────────────────────────────────────────────────

    python azalyst_benchmark.py ^
        --data-dir  %DATA_DIR% ^
        --out-dir   %OUTPUT_DIR%

    if errorlevel 1 (
        echo.
        echo  [ERROR] Step 6 failed. Re-run this file to retry.
        pause
        exit /b 1
    )
    echo. > "%CKPT_DIR%\step6_benchmark.done"
    echo  [STEP 6/7]  Benchmark complete.
)
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 7 — TEARSHEET + REPORTS
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step7_reports.done" (
    echo  [STEP 7/7]  Reports  ^|  ALREADY DONE — skipping
) else (
    echo  [STEP 7/7]  Generating tearsheet and research reports ...
    echo  ───────────────────────────────────────────────────────────────────

    python azalyst_tearsheet.py ^
        --trades   paper_trades.csv ^
        --out-dir  %OUTPUT_DIR%

    if errorlevel 1 (
        echo  [WARN] Tearsheet had an issue — check output above, continuing...
    )

    python azalyst_report.py ^
        --out-dir  %OUTPUT_DIR%

    if errorlevel 1 (
        echo.
        echo  [ERROR] Step 7 failed. Re-run this file to retry.
        pause
        exit /b 1
    )
    echo. > "%CKPT_DIR%\step7_reports.done"
    echo  [STEP 7/7]  Reports complete.
)

:: ════════════════════════════════════════════════════════════════════════════
::  STEP 8 — ORCHESTRATOR  (fuses everything into final ranked signal table)
:: ════════════════════════════════════════════════════════════════════════════
if exist "%CKPT_DIR%\step8_orchestrator.done" (
    echo  [STEP 8/8]  Orchestrator  ^|  ALREADY DONE — skipping
) else (
    echo  [STEP 8/8]  Orchestrator: fusing factor + ML + statarb into signals ...
    echo  ───────────────────────────────────────────────────────────────────

    if "%MAX_SYMBOLS%"=="0" (
        python azalyst_orchestrator.py ^
            --data-dir   %DATA_DIR% ^
            --out-dir    %OUTPUT_DIR% ^
            --model-dir  %MODEL_DIR% ^
            --resample   %RESAMPLE% ^
            --workers    %WORKERS%
    ) else (
        python azalyst_orchestrator.py ^
            --data-dir    %DATA_DIR% ^
            --out-dir     %OUTPUT_DIR% ^
            --model-dir   %MODEL_DIR% ^
            --resample    %RESAMPLE% ^
            --workers     %WORKERS% ^
            --max-symbols %MAX_SYMBOLS%
    )

    if errorlevel 1 (
        echo  [WARN] Orchestrator had an issue. Other steps still completed.
    ) else (
        echo. > "%CKPT_DIR%\step8_orchestrator.done"
        echo  [STEP 8/8]  Done. Final signals saved to %OUTPUT_DIR%\signals.csv
    )
)
echo.

:: ════════════════════════════════════════════════════════════════════════════
::  DONE
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo  ╔═══════════════════════════════════════════════════════════════════╗
echo  ║                    PIPELINE COMPLETE                              ║
echo  ╚═══════════════════════════════════════════════════════════════════╝
echo.
echo  All outputs saved to:  %OUTPUT_DIR%\
echo.
echo    signals.csv               — FINAL ranked signal table  ^<^<^< START HERE
echo    factor_ic_results.csv     — IC / ICIR / alpha analysis
echo    statarb_pairs.csv         — cointegrated pairs found
echo    paper_trades.csv          — simulated trades
echo    learning_log.csv          — ML prediction accuracy
echo    performance_metrics.csv   — per-cycle performance
echo    tearsheet.*               — portfolio tearsheet
echo    models\                   — trained ML snapshots
echo.
echo  ─────────────────────────────────────────────────────────────────────
echo  To re-run any step: delete its .done file from  %CKPT_DIR%\
echo  Then double-click this file again.
echo  ─────────────────────────────────────────────────────────────────────
echo.
pause
