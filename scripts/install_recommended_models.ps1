$ErrorActionPreference = "Stop"

$models = @(
    "gemma4:31b",
    "qwen2.5-coder:14b-instruct-q3_K_M",
    "qwen2.5-coder:7b",
    "qwen2.5vl:7b",
    "granite3.3:2b",
    "nomic-embed-text"
)

Write-Host "Installing recommended LocalPilot models..." -ForegroundColor Cyan
Write-Host "Using exact configured tags from LocalPilot role config." -ForegroundColor DarkCyan

foreach ($model in $models) {
    Write-Host ""
    Write-Host "Pulling $model" -ForegroundColor Yellow
    ollama pull $model
}

Write-Host ""
Write-Host "Recommended model install complete." -ForegroundColor Green
Write-Host "Optional slow quality mode is not included here: qwen3:30b" -ForegroundColor DarkYellow
