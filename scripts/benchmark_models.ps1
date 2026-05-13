$ErrorActionPreference = "Stop"

$hostUrl = "http://127.0.0.1:11434"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

function Test-OllamaReachable {
    try {
        Invoke-RestMethod -Uri "$hostUrl/api/tags" -Method Get -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Get-TokensPerSecond($evalCount, $evalDuration) {
    if (-not $evalCount -or -not $evalDuration -or $evalDuration -le 0) {
        return $null
    }
    return [Math]::Round(($evalCount / ($evalDuration / 1000000000.0)), 2)
}

function Invoke-TextBenchmark {
    param(
        [string]$Model,
        [string]$Prompt,
        [int]$NumCtx = 4096,
        [double]$Temperature = 0.2
    )

    $body = @{
        model = $Model
        prompt = $Prompt
        stream = $false
        options = @{
            num_ctx = $NumCtx
            temperature = $Temperature
        }
    } | ConvertTo-Json -Depth 6

    try {
        $response = Invoke-RestMethod -Uri "$hostUrl/api/generate" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 180
        [PSCustomObject]@{
            ok = $true
            model = $Model
            eval_count = [int]($response.eval_count | ForEach-Object { $_ } | Select-Object -First 1)
            eval_duration = [long]($response.eval_duration | ForEach-Object { $_ } | Select-Object -First 1)
            load_duration = [long]($response.load_duration | ForEach-Object { $_ } | Select-Object -First 1)
            tokens_per_second = Get-TokensPerSecond -evalCount $response.eval_count -evalDuration $response.eval_duration
        }
    } catch {
        [PSCustomObject]@{
            ok = $false
            model = $Model
            error = $_.Exception.Message
        }
    }
}

function Show-BenchmarkResult {
    param(
        [pscustomobject]$Result,
        [string]$Label
    )

    if (-not $Result.ok) {
        Write-Host ("- {0}: warning -> {1}" -f $Label, $Result.error) -ForegroundColor Yellow
        return
    }

    $loadSeconds = [Math]::Round(($Result.load_duration / 1000000000.0), 2)
    $tps = if ($null -ne $Result.tokens_per_second) { $Result.tokens_per_second } else { "n/a" }
    Write-Host ("- {0}: model={1}, tps={2}, load={3}s, eval_tokens={4}" -f $Label, $Result.model, $tps, $loadSeconds, $Result.eval_count) -ForegroundColor Green
}

if (-not (Test-OllamaReachable)) {
    Write-Host "Ollama is not reachable at $hostUrl. Start Ollama and try again." -ForegroundColor Red
    exit 1
}

Write-Host "Benchmarking recommended LocalPilot models..." -ForegroundColor Cyan

$textBenchmarks = @(
    @{ Label = "main"; Model = "gemma4:31b"; Prompt = "Say one short sentence about local AI."; NumCtx = 4096; Temperature = 0.35 },
    @{ Label = "coder"; Model = "qwen2.5-coder:14b-instruct-q3_K_M"; Prompt = "Write a tiny Python function that adds two numbers."; NumCtx = 4096; Temperature = 0.2 },
    @{ Label = "coder_fallback"; Model = "qwen2.5-coder:7b"; Prompt = "Write a tiny Python function that adds two numbers."; NumCtx = 4096; Temperature = 0.2 },
    @{ Label = "router"; Model = "granite3.3:2b"; Prompt = "Classify this request as chat, code, research, desktop, or memory: show notes"; NumCtx = 2048; Temperature = 0.0 }
)

foreach ($entry in $textBenchmarks) {
    $result = Invoke-TextBenchmark -Model $entry.Model -Prompt $entry.Prompt -NumCtx $entry.NumCtx -Temperature $entry.Temperature
    Show-BenchmarkResult -Result $result -Label $entry.Label
}

Write-Host "Vision benchmark (best effort)..." -ForegroundColor Cyan
& $pythonExe (Join-Path $projectRoot "localpilot.py") --vision-test

Write-Host ""
Write-Host "ollama ps" -ForegroundColor Cyan
ollama ps
