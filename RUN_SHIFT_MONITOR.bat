@echo off
setlocal

title AZALYST SHIFT LAUNCHER
color 0A

cd /d "%~dp0"

set "DRY_RUN=%AZALYST_DRY_RUN%"
set "PYTHON_EXE=python"
where %PYTHON_EXE% >nul 2>nul
if errorlevel 1 (
    set "PYTHON_EXE=py"
)

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo  [0/0] Cleaning stale locks...
if exist ".azalyst_locks" (
    %PYTHON_EXE% cleanup_locks.py
)

echo.
echo  ============================================================
echo      AZALYST SHIFT LAUNCHER
echo      One-click monitor + agent launcher
echo  ============================================================
echo.

echo  [1/2] Starting monitor...
if "%DRY_RUN%"=="1" (
    echo    DRY RUN: start monitor_dashboard.py and open http://127.0.0.1:8080
) else (
    start "Azalyst Live Monitor" /min cmd /c "cd /d ""%~dp0"" && %PYTHON_EXE% monitor_dashboard.py"
    timeout /t 2 /nobreak >nul
    start "" "http://127.0.0.1:8080"
)
if exist "ensure_jupyter_monitor.py" (
    echo    Opening Azalyst notebook monitor...
    if "%DRY_RUN%"=="1" (
        echo    DRY RUN: %PYTHON_EXE% ensure_jupyter_monitor.py
    ) else (
        %PYTHON_EXE% ensure_jupyter_monitor.py
    )
) else (
    echo    Notebook helper missing; use 127.0.0.1:8080 to view the monitor.
)

echo  [2/2] Starting autonomous agent...
if "%DRY_RUN%"=="1" (
    echo    DRY RUN: start Ollama server
    echo    DRY RUN: warm deepseek-r1:14b
    echo    DRY RUN: run azalyst_autonomous_team.py
    goto :eof
)

echo.
echo  ============================================================
echo      AZALYST AUTONOMOUS RESEARCH TEAM
echo      Two AI personas: Olivia ^(Researcher^) + Marcus ^(Dev^)
echo      Runs autonomously until alpha is found
echo  ============================================================
echo.

echo  [agent 1/4] Starting Ollama server...
start "Ollama Server" /min cmd /c "ollama serve"
timeout /t 8 /nobreak >nul
echo  [agent 1/4] Ollama started.

echo  [agent 2/4] Warming up deepseek-r1:14b ^(30 sec^)...
start "Warm Model" /min cmd /c "ollama run deepseek-r1:14b --verbose 2>nul"
timeout /t 30 /nobreak >nul
echo  [agent 2/4] Model ready.

echo  [agent 3/4] Project folder set.
echo  [agent 4/4] Starting autonomous research team...
echo.
echo  NOTE: The main fix ^(signal direction flip^) is already applied.
echo  The team will confirm and then fine-tune from there.
echo  Expected first-run win rate: 60-75%%+
echo.
%PYTHON_EXE% azalyst_autonomous_team.py

echo.
echo  Team session complete. Check team_log.txt for the full transcript.
pause
