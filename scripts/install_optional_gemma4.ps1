$ErrorActionPreference = "Stop"

$models = @(
    "gemma4:e4b",
    "gemma4"
)

Write-Host "Installing optional Gemma 4 comparison models for LocalPilot..." -ForegroundColor Cyan

foreach ($model in $models) {
    Write-Host "Pulling $model" -ForegroundColor Yellow
    ollama pull $model
}

Write-Host ""
Write-Host "Optional Gemma 4 install complete." -ForegroundColor Green
Write-Host "You can now run: model compare gemma4" -ForegroundColor Green
