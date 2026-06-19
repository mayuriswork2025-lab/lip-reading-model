param(
  [string]$url,
  [string]$out
)
if (-not $url) { Write-Host "Usage: .\download_weights.ps1 -url <url> [-out <filename>]"; exit 1 }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$models = Join-Path $root "evaluation\models"
if (-not (Test-Path $models)) { New-Item -ItemType Directory -Path $models | Out-Null }
$fname = if ($out) { $out } else { Split-Path $url -Leaf }
$dest = Join-Path $models $fname
Invoke-WebRequest -Uri $url -OutFile $dest
Write-Host "Downloaded to $dest"
