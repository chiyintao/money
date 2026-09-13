@echo off
setlocal
set "PORT=8101"
set "FOUND=0"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  set "FOUND=1"
  echo Stopping paper dashboard process %%p...
  taskkill /PID %%p /F >nul 2>nul
)
if "%FOUND%"=="0" echo No process is listening on port %PORT%.
endlocal
