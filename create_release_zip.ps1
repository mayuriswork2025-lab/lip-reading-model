param(
  [string]$OutName = "LipRead-Studio-v1.zip"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Creating release ZIP: $OutName"

# Collect files while excluding large model weights, virtualenv and node_modules
$files = Get-ChildItem -Path $root -Recurse -File | Where-Object {
    $_.FullName -notmatch '\\evaluation\\models\\.*\.h5$' -and
    $_.FullName -notmatch '\\frontend\\node_modules\\' -and
    $_.FullName -notmatch '\\.venv\\' -and
    $_.Name -notlike '*.zip'
}

$tempList = Join-Path $env:TEMP "lipread_files.txt"
$files.FullName | Out-File -FilePath $tempList -Encoding utf8

$dest = Join-Path $root "..\$OutName"
Compress-Archive -Path (Get-Content $tempList) -DestinationPath $dest -Force
Remove-Item $tempList

Write-Host "Created $dest"
