$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$zipPath = Join-Path (Split-Path $root -Parent) "lip-reading-model-team.zip"
$staging = Join-Path $env:TEMP "lip-reading-model-staging"

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Write-Host "Staging clean copy..."
robocopy $root $staging /E /XD .venv .git node_modules data\jobs __pycache__ .vscode /XF overlapped-weights368.h5 unseen-weights178.h5 guide_extracted.txt *.log /NFL /NDL /NJH /NJS /NC /NS | Out-Null

Write-Host "Creating zip: $zipPath"
Compress-Archive -Path "$staging\*" -DestinationPath $zipPath -Force
Remove-Item $staging -Recurse -Force

$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "Done: $zipPath ($sizeMb MB)"
Write-Host "Teammates: unzip, run setup.ps1, then start.ps1 -OpenBrowser"
