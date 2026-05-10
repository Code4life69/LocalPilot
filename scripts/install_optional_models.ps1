$ErrorActionPreference = "Stop"

$models = @(
    "deepseek-r1:14b-qwen-distill-q4_K_M",
    "llama3.1:8b"
)

Write-Host "Installing optional LocalPilot models..." -ForegroundColor Cyan

foreach ($model in $models) {
    Write-Host ""
    Write-Host "Pulling $model" -ForegroundColor Yellow
    ollama pull $model
}

Write-Host ""
Write-Host "Optional model install complete." -ForegroundColor Green
Write-Host "qwen3:30b is not included here because it is already the optional quality_slow role." -ForegroundColor DarkYellow
