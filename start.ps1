param(
    [switch]$OpenBrowser,
    [switch]$Setup
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$frontendDir = Join-Path $root 'frontend'
$frontendUrl = 'http://127.0.0.1:5173'

if (-not (Test-Path $venvPython)) {
    python -m venv (Join-Path $root '.venv')
}

if ($Setup -or -not (Test-Path (Join-Path $root 'models\word_classifier.pt'))) {
    Write-Host 'Installing Python dependencies...'
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $root 'requirements.txt')
    Write-Host 'Bootstrapping demo training data...'
    $env:PYTHONPATH = $root
    & $venvPython (Join-Path $root 'scripts\bootstrap_demo.py')
    Write-Host 'Training word classifier...'
    & $venvPython (Join-Path $root 'scripts\train_words.py') --epochs 12
}

$backendCommand = "Set-Location '$root'; `$env:PYTHONPATH='$root'; & '$venvPython' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
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
    Start-Sleep -Milliseconds 800
    Start-Process $frontendUrl
}

Write-Host 'LipRead Studio is starting.'
Write-Host 'Frontend: http://127.0.0.1:5173'
Write-Host 'Backend:  http://127.0.0.1:8000/health'
