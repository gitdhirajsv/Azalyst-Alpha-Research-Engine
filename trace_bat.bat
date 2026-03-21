@echo on
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo STEP1: Starting Python check
python --version >nul 2>&1
echo STEP2: After python check, errorlevel=%errorlevel%
if errorlevel 1 (
    echo STEP3: ERROR - Python not found
    echo END_TRACE
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo STEP4: Python %PY_VER%

echo STEP5: GPU check
set GPU_FOUND=0
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
echo STEP6: nvidia-smi errorlevel=%errorlevel%
if not errorlevel 1 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        set GPU_NAME=%%g
        set GPU_FOUND=1
    )
)
echo STEP7: GPU_FOUND=!GPU_FOUND!

echo STEP8: Spyder check
set SPYDER_FOUND=0
where spyder >nul 2>&1
echo STEP9: where spyder errorlevel=%errorlevel%
if not errorlevel 1 ( set SPYDER_FOUND=1 & echo SPYDER via where & goto :SPYDER_DONE )
python -c "import spyder" >nul 2>&1
echo STEP10: python -m spyder errorlevel=%errorlevel%
if not errorlevel 1 ( set SPYDER_FOUND=1 & goto :SPYDER_DONE )
echo STEP11: Spyder not found via common methods

:SPYDER_DONE
echo STEP12: SPYDER_FOUND=!SPYDER_FOUND!

echo STEP13: Package check
python -c "import xgboost, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, psutil, statsmodels" 2>nul
echo STEP14: packages errorlevel=%errorlevel%

echo STEP15: Data dir check
if not exist "%~dp0data\" (
    echo STEP16: ERROR - data dir not found
    exit /b 1
)
echo STEP17: Data dir OK

set PARQUET_COUNT=0
for %%f in ("%~dp0data\*.parquet") do if exist "%%f" set /a PARQUET_COUNT+=1
echo STEP18: PARQUET_COUNT=!PARQUET_COUNT!

echo TRACE_COMPLETE
exit /b 0
