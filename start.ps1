param(
    [switch]$OpenBrowser
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$frontendDir = Join-Path $root 'frontend'
$backendUrl = 'http://127.0.0.1:8000'
$frontendUrl = 'http://127.0.0.1:5173'

if (-not (Test-Path $venvPython)) {
    throw "Python virtual environment not found at $venvPython. Create it first with: python -m venv .venv"
}

if (-not (Test-Path $frontendDir)) {
    throw "Frontend directory not found at $frontendDir"
}

$backendCommand = "Set-Location '$root'; & '$venvPython' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $backendCommand) | Out-Null

$frontendNodeModules = Join-Path $frontendDir 'node_modules'
if (-not (Test-Path $frontendNodeModules)) {
    Push-Location $frontendDir
    try {
        npm install
    }
    finally {
        Pop-Location
    }
}

$frontendCommand = "Set-Location '$frontendDir'; npm run dev -- --host 0.0.0.0"
Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $frontendCommand) | Out-Null

if ($OpenBrowser) {
    Start-Sleep -Milliseconds 500
    Start-Process $frontendUrl
}