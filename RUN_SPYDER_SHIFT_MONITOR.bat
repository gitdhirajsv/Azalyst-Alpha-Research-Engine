@echo off
setlocal

title AZALYST SPYDER SHIFT LAUNCHER

cd /d "%~dp0"

set "DRY_RUN=%AZALYST_DRY_RUN%"
set "AZALYST_SKIP_BROWSER_MONITOR=1"
set "AZALYST_SKIP_NOTEBOOK_MONITOR=1"
set "PYTHON_EXE=python"
set "SPYDER_EXE="
set "SPYDER_PROFILE=%~dp0.spyder_azalyst"

where %PYTHON_EXE% >nul 2>nul
if errorlevel 1 (
    set "PYTHON_EXE=py"
)

if exist "C:\ProgramData\spyder-6\envs\spyder-runtime\Scripts\spyder.exe" (
    set "SPYDER_EXE=C:\ProgramData\spyder-6\envs\spyder-runtime\Scripts\spyder.exe"
)

if not defined SPYDER_EXE (
    for /f "delims=" %%I in ('where spyder 2^>nul') do (
        if not defined SPYDER_EXE set "SPYDER_EXE=%%I"
    )
)

echo.
echo  ============================================================
echo      AZALYST SPYDER SHIFT LAUNCHER
echo      Spyder-first monitor + autonomous monitor workflow
echo  ============================================================
echo.

if defined SPYDER_EXE (
    echo  [spyder] Opening Spyder workspace...
    if "%DRY_RUN%"=="1" (
        echo    DRY RUN: %PYTHON_EXE% prepare_spyder_profile.py
    ) else (
        %PYTHON_EXE% prepare_spyder_profile.py >nul
    )
    if "%DRY_RUN%"=="1" (
        echo    DRY RUN: "%SPYDER_EXE%" --new-instance --conf-dir "%SPYDER_PROFILE%" -w "%~dp0" --window-title "Azalyst Spyder Monitor" "%~dp0spyder_live_monitor.py" "%~dp0walkforward_simulator.py" "%~dp0azalyst_autonomous_team.py"
    ) else (
        start "Spyder 6" /D "%~dp0" "%SPYDER_EXE%" --new-instance --conf-dir "%SPYDER_PROFILE%" -w "%~dp0" --window-title "Azalyst Spyder Monitor" "%~dp0spyder_live_monitor.py" "%~dp0walkforward_simulator.py" "%~dp0azalyst_autonomous_team.py"
    )
) else (
    echo  [spyder] Spyder not found. Continuing without IDE.
)

echo  [launcher] Browser and Jupyter monitors are disabled for this path.
echo  [launcher] Spyder will auto-run spyder_live_monitor.py in its console.
echo  [launcher] Handing off to RUN_SHIFT_MONITOR.bat...
call "%~dp0RUN_SHIFT_MONITOR.bat"
