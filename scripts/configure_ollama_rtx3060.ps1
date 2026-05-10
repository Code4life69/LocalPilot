$ErrorActionPreference = "Stop"

$settings = @{
    OLLAMA_KEEP_ALIVE     = "2m"
    OLLAMA_FLASH_ATTENTION = "1"
    OLLAMA_KV_CACHE_TYPE  = "q8_0"
    OLLAMA_NUM_PARALLEL   = "1"
    OLLAMA_MAX_LOADED_MODELS = "1"
    OLLAMA_CONTEXT_LENGTH = "4096"
}

Write-Host "Configuring Ollama for RTX 3060 / 32GB RAM / i5-12400F..." -ForegroundColor Cyan

foreach ($entry in $settings.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "User")
    Write-Host ("Set {0}={1}" -f $entry.Key, $entry.Value) -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done. Fully quit Ollama and start it again for these settings to apply." -ForegroundColor Green
Write-Host "This includes quitting any background Ollama tray or server process." -ForegroundColor DarkYellow
