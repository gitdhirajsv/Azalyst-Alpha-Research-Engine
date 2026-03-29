@echo off
title Azalyst — Training Dashboard
color 0A
cd /d "%~dp0"

:: Find Python
set "PYTHON_EXE="
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%ProgramData%\Anaconda3"
    "C:\Python313" "C:\Python312" "C:\Python311" "C:\Python310"
) do (
    if exist "%%~d\python.exe" (
        set "PYTHON_EXE=%%~d\python.exe"
        goto :found
    )
)

:: fallback — hope python is on PATH
set "PYTHON_EXE=python"
:found

echo.
echo  ============================================================
echo    AZALYST  —  ML TRAINING RESULTS DASHBOARD
echo    Loading plots from results/ ...
echo  ============================================================
echo.

"%PYTHON_EXE%" VIEW_TRAINING.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Dashboard failed to open.
    echo  Make sure matplotlib and pandas are installed:
    echo    python -m pip install matplotlib pandas
    pause
)
