param(
    [int[]]$Ports = @(8000, 5173)
)

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        $processId = $connection.OwningProcess
        if ($processId) {
            Write-Host "Stopping PID $processId on port $port"
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host 'Servers stopped.'
