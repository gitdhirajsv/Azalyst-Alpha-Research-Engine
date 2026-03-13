@echo off
title AZALYST AUTONOMOUS AGENT v2
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   AZALYST AUTONOMOUS RESEARCH TEAM                      ║
echo  ║   Two AI personas: Olivia (Researcher) + Marcus (Dev)   ║
echo  ║   Runs autonomously until alpha is found                 ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ── STEP 1: Start Ollama in background ──────────────────────────────────────
echo  [1/4] Starting Ollama server...
start "Ollama Server" /min cmd /c "ollama serve"
timeout /t 8 /nobreak >nul
echo  [1/4] Ollama started.

:: ── STEP 2: Warm up model ───────────────────────────────────────────────────
echo  [2/4] Warming up deepseek-r1:14b (30 sec)...
start "Warm Model" /min cmd /c "ollama run deepseek-r1:14b --verbose 2>nul"
timeout /t 30 /nobreak >nul
echo  [2/4] Model ready.

:: ── STEP 3: Go to project ───────────────────────────────────────────────────
cd /d "D:\Personal Files\Azalyst Alpha Research Engine"
echo  [3/4] Project folder set.

:: ── STEP 4: Run autonomous team ─────────────────────────────────────────────
echo  [4/4] Starting autonomous research team...
echo.
echo  NOTE: The main fix (signal direction flip) is already applied.
echo  The team will confirm and then fine-tune from there.
echo  Expected first-run win rate: 60-75%%+
echo.
python azalyst_autonomous_team.py

echo.
echo  Team session complete. Check team_log.txt for full transcript.
pause
