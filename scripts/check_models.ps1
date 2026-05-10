$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

Write-Host "Ollama installed models:" -ForegroundColor Cyan
ollama list

Write-Host ""
Write-Host "LocalPilot model status:" -ForegroundColor Cyan
& $pythonExe (Join-Path $projectRoot "localpilot.py") --model-status
