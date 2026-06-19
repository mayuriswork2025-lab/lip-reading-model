param(
    [switch]$Retrain
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$frontendDir = Join-Path $root 'frontend'

Write-Host '=== LipRead Studio Setup ===' -ForegroundColor Cyan
Write-Host "Project: $root"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python not found. Install Python 3.10+ from https://www.python.org/downloads/'
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw 'Node.js/npm not found. Install from https://nodejs.org/'
}

if (-not (Test-Path $venvPython)) {
    Write-Host 'Creating Python virtual environment...'
    python -m venv (Join-Path $root '.venv')
}

Write-Host 'Installing Python packages...'
& $venvPython -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
& $venvPython -m pip install -r (Join-Path $root 'requirements.txt') --trusted-host pypi.org --trusted-host files.pythonhosted.org

Write-Host 'Installing frontend packages...'
Push-Location $frontendDir
try {
    npm install
}
finally {
    Pop-Location
}

$env:PYTHONPATH = $root
$modelPath = Join-Path $root 'models\word_classifier.pt'

if ($Retrain -or -not (Test-Path $modelPath)) {
    Write-Host 'Building training data...'
    & $venvPython (Join-Path $root 'scripts\bootstrap_demo.py')
    Write-Host 'Training model (CPU — may take several minutes)...'
    & $venvPython (Join-Path $root 'scripts\train_words.py') --epochs 8 --batch-size 4
}
else {
    Write-Host "Trained model found at $modelPath"
}

Write-Host ''
Write-Host 'Running quick test...'
& $venvPython (Join-Path $root 'scripts\test_demo.py')

Write-Host ''
Write-Host 'Setup complete!' -ForegroundColor Green
Write-Host 'Start demo:  .\start.ps1 -OpenBrowser'
Write-Host 'Web UI:      http://127.0.0.1:5173'
