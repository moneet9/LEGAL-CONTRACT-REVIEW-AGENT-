$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Starting PS-9 Legal Contract Review Agent..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11+ and run this script again."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js/npm was not found. Install Node.js 20+ and run this script again."
}

$venv = Join-Path $root ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creating Python environment..." -ForegroundColor Yellow
    python -m venv $venv
}

Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
& $python -m pip install --upgrade pip --quiet
& $python -m pip install -r (Join-Path $root "backend\requirements.txt") --quiet

if (-not (Test-Path (Join-Path $root ".env"))) {
    Copy-Item (Join-Path $root ".env.example") (Join-Path $root ".env")
    Write-Host "Created .env from .env.example. Add GEMINI_API_KEY for Gemini analysis." -ForegroundColor Yellow
}

$frontend = Join-Path $root "frontend"
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    npm install --prefix $frontend
}

$backendProcess = Start-Process -FilePath $python `
    -ArgumentList "-m","uvicorn","backend.app.main:app","--reload","--port","8000" `
    -WorkingDirectory $root -PassThru

$frontendProcess = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run","dev","--","--host","0.0.0.0" `
    -WorkingDirectory $frontend -PassThru

Write-Host "" 
Write-Host "PS-9 is running." -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173"
Write-Host "Backend:  http://localhost:8000"
Write-Host "API docs: http://localhost:8000/docs"
Write-Host "Processes: backend $($backendProcess.Id), frontend $($frontendProcess.Id)"
Write-Host "Close the two service windows or run stop.ps1 to stop PS-9."
