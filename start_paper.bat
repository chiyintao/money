@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem Paper-trading dashboard launcher.
rem
rem Two things here are load-bearing, and both were wrong before.
rem
rem   * The wait budget. Startup is slow: the venue's exchangeInfo is a 1.1 MB
rem     document that takes ~15s to fetch from this network, and the dashboard is
rem     only published after the contract specs and maintenance brackets have been
rem     requested. The old budget was 20s, so it expired first, the script printed
rem     "did not start", and it never reached the line that opens the browser --
rem     while the service itself came up fine moments later.
rem
rem   * What "ready" means. A listening socket is not a working dashboard. A
rem     refactor once left the static root pointing at a directory that publishes
rem     nothing: /health answered 200 while / and every asset returned 404. A
rem     netstat check calls that ready and opens a blank page. Probe the page the
rem     user actually opens.
rem ---------------------------------------------------------------------------

set "URL=http://127.0.0.1:8101"
set "PORT=8101"
set "WAIT_SECONDS=120"

if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo Created .env from .env.example
)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Python 3.10 or newer.
  pause
  exit /b 1
)

python -c "import aiohttp, dotenv" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
  )
)

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /C:":%PORT%" ^| findstr /C:"LISTENING"') do (
  echo Stopping old process on port %PORT%: %%p
  taskkill /PID %%p /F >nul 2>nul
)

if not exist logs mkdir logs
echo Starting Binance Futures paper dashboard...
rem No arguments on purpose: app.main parses none, and the mode comes from
rem TRADING_MODE in .env. It used to be launched with "--mode paper", which
rem nothing read.
start "Binance Paper Dashboard" /D "%~dp0" python -u -m app.main

echo Waiting for %URL% to answer (up to %WAIT_SECONDS%s)...
rem One PowerShell process rather than a batch loop: 'timeout /t 1' aborts when
rem stdin is redirected, which turns a wait loop into a spin.
powershell -NoProfile -Command "$deadline=(Get-Date).AddSeconds(%WAIT_SECONDS%); while((Get-Date) -lt $deadline){ try{ $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 '%URL%/'; if($r.StatusCode -eq 200 -and $r.Content.Length -gt 200){ exit 0 } }catch{}; Start-Sleep -Milliseconds 800 }; exit 1"
if errorlevel 1 goto failed

echo Dashboard is ready: %URL%
start "" "%URL%"
endlocal
exit /b 0

:failed
echo [ERROR] The dashboard did not answer on %URL%/ within %WAIT_SECONDS% seconds.
echo Check the "Binance Paper Dashboard" window for the error message.
pause
endlocal
exit /b 1
