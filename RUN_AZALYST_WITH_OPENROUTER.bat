@echo off
REM Azalyst with OpenRouter Configuration
REM Run this batch file to launch Azalyst with OpenRouter API credentials

echo Setting OpenRouter environment variables...
set OPENAI_API_KEY=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
set OPENAI_BASE_URL=https://openrouter.ai/api/v1
set OPENAI_MODEL=qwen/qwen3-coder:free

echo.
echo ✓ OpenRouter environment variables configured
echo   API Key: %OPENAI_API_KEY:~0,20%...
echo   Base URL: %OPENAI_BASE_URL%
echo   Model: %OPENAI_MODEL%
echo.

echo Starting Azalyst...
call RUN_AZALYST.bat
pause
