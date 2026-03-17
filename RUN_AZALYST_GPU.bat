@echo off
title Azalyst — RTX 2050 GPU Runner
color 0A

echo.
echo ================================================================
echo   AZALYST ALPHA RESEARCH ENGINE
echo   GPU: NVIDIA RTX 2050   CPU: i5-11260H
echo   One-click launcher
echo ================================================================
echo.

:: ── Set working directory to wherever this .bat file lives ───────────────────
cd /d "%~dp0"

:: ── Step 1: Check Python ──────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: ── Step 2: Check / install dependencies ─────────────────────────────────────
echo [Setup] Checking dependencies...
python -c "import xgboost, numpy, pandas, sklearn, scipy, matplotlib, pyarrow" 2>nul
if errorlevel 1 (
    echo [Setup] Installing missing packages...
    pip install xgboost>=2.0.3 numpy pandas scikit-learn scipy matplotlib pyarrow --upgrade -q
    if errorlevel 1 (
        echo [ERROR] pip install failed. Check internet connection.
        pause
        exit /b 1
    )
    echo [Setup] Packages installed.
)

:: ── Step 3: Force NVIDIA GPU — disable Intel UHD for CUDA ────────────────────
:: CUDA_VISIBLE_DEVICES=0 pins to RTX 2050 (device 0)
:: Intel UHD is not a CUDA device so it never interferes, but we set this
:: explicitly to be absolutely safe.
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_E_BUS_ID

:: ── Step 4: Windows power mode — prevent throttling ──────────────────────────
echo [Setup] Setting high-performance power mode...
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

:: ── Step 5: Launch Spyder in background (if installed) ───────────────────────
python -c "import spyder" >nul 2>&1
if not errorlevel 1 (
    echo [Spyder] Launching Spyder IDE in background...
    start "" /B python -m spyder --new-instance --workdir="%~dp0" --window-title="Azalyst RTX2050" 2>nul
    timeout /t 4 /nobreak >nul
    echo [Spyder] Spyder started. Open azalyst_spyder_monitor.py inside it.
) else (
    echo [Spyder] Spyder not found — skipping. Install with: pip install spyder
)

:: ── Step 6: Run the GPU pipeline ─────────────────────────────────────────────
echo.
echo [Run] Starting GPU pipeline...
echo       You can watch live progress in Spyder console OR this window.
echo.

python azalyst_local_gpu.py

:: ── Step 7: Done ─────────────────────────────────────────────────────────────
echo.
echo ================================================================
echo   Run complete. Results saved to .\results\
echo   Open performance_year3.png to see the chart.
echo ================================================================
echo.
pause
