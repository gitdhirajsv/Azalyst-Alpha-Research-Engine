# Azalyst with OpenRouter Configuration
# Run this PowerShell script to launch Azalyst with OpenRouter API credentials

# Set OpenRouter environment variables
$env:OPENAI_API_KEY = "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"
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
$env:OPENAI_MODEL = "qwen/qwen3-coder:free"

Write-Host "✓ OpenRouter environment variables configured" -ForegroundColor Green
Write-Host "  API Key: $($env:OPENAI_API_KEY.Substring(0, 20))..." -ForegroundColor Cyan
Write-Host "  Base URL: $env:OPENAI_BASE_URL" -ForegroundColor Cyan
Write-Host "  Model: $env:OPENAI_MODEL" -ForegroundColor Cyan
Write-Host ""

# Run Azalyst
Write-Host "Starting Azalyst..." -ForegroundColor Yellow
python RUN_AZALYST.bat
