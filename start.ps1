param(
    [switch]$OpenBrowser,
    [switch]$Setup
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$frontendDir = Join-Path $root 'frontend'
$frontendUrl = 'http://127.0.0.1:5173'
$stopScript = Join-Path $root 'scripts\stop_servers.ps1'

if (-not (Test-Path $venvPython)) {
    python -m venv (Join-Path $root '.venv')
}

if ($Setup -or -not (Test-Path (Join-Path $root 'models\sentence_reader.pt'))) {
    Write-Host 'Setting up sentence model...'
    & (Join-Path $root 'setup.ps1')
}

Write-Host 'Stopping any old backend/frontend on ports 8000 and 5173...'
& $stopScript

$backendCommand = "Set-Location '$root'; `$env:PYTHONPATH='$root'; & '$venvPython' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $backendCommand) | Out-Null

$frontendNodeModules = Join-Path $frontendDir 'node_modules'
if (-not (Test-Path $frontendNodeModules)) {
    Push-Location $frontendDir
    try { npm install } finally { Pop-Location }
}

$frontendCommand = "Set-Location '$frontendDir'; npm run dev -- --host 0.0.0.0"
Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $frontendCommand) | Out-Null

Start-Sleep -Seconds 3
try {
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5
    if ($health.model_type -ne 'sentence_ctc') {
        Write-Host 'WARNING: Backend is not using the sentence model. Run .\scripts\stop_servers.ps1 then .\start.ps1 again.' -ForegroundColor Yellow
    } else {
        Write-Host "Backend OK: sentence model at $($health.model_path)" -ForegroundColor Green
    }
} catch {
    Write-Host 'WARNING: Backend did not respond on port 8000 yet.' -ForegroundColor Yellow
}

if ($OpenBrowser) {
    Start-Sleep -Milliseconds 800
    Start-Process $frontendUrl
}

Write-Host 'LipRead Studio (sentence model) is starting.'
Write-Host 'Frontend: http://127.0.0.1:5173'
Write-Host 'Backend:  http://127.0.0.1:8000/health'
Write-Host 'Check health shows: "model_type": "sentence_ctc" (not word_classifier.pt)'
