@echo off
setlocal
cd /d "%~dp0"
echo Starting PS-9 Legal Contract Review Agent...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
  echo.
  echo PS-9 failed to start. Review the message above.
  pause
  exit /b 1
)
pause
