$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Get-CimInstance Win32_Process | Where-Object {
    ($_.CommandLine -like "*$root*") -and ($_.Name -match "python.exe|node.exe")
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host "PS-9 services stopped." -ForegroundColor Green
