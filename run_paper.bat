@echo off
chcp 65001 >nul
setlocal

rem Carpeta del proyecto = carpeta donde esta este .bat (funciona en C:, D: o cualquier ruta)
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$project = $env:PROJECT_DIR;" ^
  "$python = Join-Path $project '.venv\Scripts\python.exe';" ^
  "$main = Join-Path $project 'main.py';" ^
  "$logsDir = Join-Path $project 'logs';" ^
  "$summaryLog = Join-Path $logsDir 'paper_runner.log';" ^
  "$goldLog = Join-Path $logsDir 'paper_gold_runner.log';" ^
  "$eurusdLog = Join-Path $logsDir 'paper_eurusd_runner.log';" ^
  "$autoLog = Join-Path $logsDir 'mt5_demo_auto_runner.log';" ^
  "$lock = Join-Path $project 'paper\paper.lock';" ^
  "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false;" ^
  "$OutputEncoding = [Console]::OutputEncoding;" ^
  "Set-Location $project;" ^
  "New-Item -ItemType Directory -Force -Path $logsDir | Out-Null;" ^
  "New-Item -ItemType Directory -Force -Path (Split-Path $lock) | Out-Null;" ^
  ". (Join-Path $project 'scripts\paper_lock.ps1');" ^
  "function Rotate-Log($path) {" ^
  "  if ((Test-Path $path) -and ((Get-Item $path).Length -gt 20MB)) {" ^
  "    $stampRotate = Get-Date -Format 'yyyyMMdd_HHmmss';" ^
  "    Rename-Item -Path $path -NewName ((Split-Path $path -Leaf) + '.' + $stampRotate + '.bak');" ^
  "  }" ^
  "}" ^
  "function Write-Log($path, $message) {" ^
  "  $message | Out-File -FilePath $path -Encoding utf8 -Append;" ^
  "}" ^
  "function Invoke-PaperCommand($name, $arguments, $logPath) {" ^
  "  Rotate-Log $logPath;" ^
  "  $start = Get-Date;" ^
  "  Write-Log $logPath '================================================';" ^
  "  Write-Log $logPath ($name + ' START: ' + $start.ToString('yyyy-MM-dd HH:mm:ss'));" ^
  "  Write-Log $logPath ('COMMAND: ' + $python + ' ' + $main + ' ' + ($arguments -join ' '));" ^
  "  Write-Log $logPath '------------------------------------------------';" ^
  "  $exitCode = 0;" ^
  "  try {" ^
  "    $oldErrorActionPreference = $ErrorActionPreference;" ^
  "    $ErrorActionPreference = 'Continue';" ^
  "    & $python $main @arguments *>&1 | Out-File -FilePath $logPath -Encoding utf8 -Append;" ^
  "    $ErrorActionPreference = $oldErrorActionPreference;" ^
  "    $exitCode = $LASTEXITCODE;" ^
  "    if ($null -eq $exitCode) { $exitCode = 0 }" ^
  "  } catch {" ^
  "    if ($null -ne $oldErrorActionPreference) { $ErrorActionPreference = $oldErrorActionPreference }" ^
  "    Write-Log $logPath ('ERROR RUNNER ' + $name + ': ' + $_.Exception.Message);" ^
  "    $exitCode = 1;" ^
  "  }" ^
  "  $end = Get-Date;" ^
  "  Write-Log $logPath '------------------------------------------------';" ^
  "  Write-Log $logPath ($name + ' END: ' + $end.ToString('yyyy-MM-dd HH:mm:ss'));" ^
  "  Write-Log $logPath ($name + ' EXIT CODE: ' + $exitCode);" ^
  "  Write-Log $logPath '';" ^
  "  return [int]$exitCode;" ^
  "}" ^
  "Rotate-Log $summaryLog;" ^
  "$runStart = Get-Date;" ^
  "Write-Log $summaryLog '================================================';" ^
  "Write-Log $summaryLog ('PAPER RUN START: ' + $runStart.ToString('yyyy-MM-dd HH:mm:ss'));" ^
  "Write-Log $summaryLog '================================================';" ^
  "$currentPid = [string]$PID;" ^
  "$ownLockStream = New-PaperLock $lock $currentPid;" ^
  "if ($null -eq $ownLockStream) { exit 0 }" ^
  "$goldExit = 1;" ^
  "$eurusdExit = 1;" ^
  "$autoExit = 1;" ^
  "try {" ^
  "  if (-not (Test-Path $python)) { throw ('No existe Python del entorno virtual: ' + $python) }" ^
  "  Write-Log $summaryLog 'Ejecutando PAPER GOLD...';" ^
  "  $goldExit = Invoke-PaperCommand 'GOLD PAPER' @('PAPER') $goldLog;" ^
  "  Write-Log $summaryLog ('GOLD EXIT CODE: ' + $goldExit);" ^
  "  Write-Log $summaryLog 'Ejecutando PAPER EURUSD...';" ^
  "  $eurusdExit = Invoke-PaperCommand 'EURUSD PAPER' @('PAPER', 'EURUSD') $eurusdLog;" ^
  "  Write-Log $summaryLog ('EURUSD EXIT CODE: ' + $eurusdExit);" ^
  "  Write-Log $summaryLog 'Ejecutando MT5 DEMO AUTO RUN...';" ^
  "  $autoExit = Invoke-PaperCommand 'MT5 DEMO AUTO RUN' @('MT5', 'DEMO', 'AUTO', 'RUN') $autoLog;" ^
  "  Write-Log $summaryLog ('MT5 DEMO AUTO RUN EXIT CODE: ' + $autoExit);" ^
  "} catch {" ^
  "  Write-Log $summaryLog ('ERROR RUNNER: ' + $_.Exception.Message);" ^
  "} finally {" ^
  "  Clear-OwnPaperLock $lock $currentPid $ownLockStream;" ^
  "  $overallExit = 0;" ^
  "  if (($goldExit -ne 0) -or ($eurusdExit -ne 0) -or ($autoExit -ne 0)) { $overallExit = 1 }" ^
  "  $runEnd = Get-Date;" ^
  "  Write-Log $summaryLog ('PAPER RUN END: ' + $runEnd.ToString('yyyy-MM-dd HH:mm:ss'));" ^
  "  Write-Log $summaryLog ('GOLD EXIT CODE: ' + $goldExit);" ^
  "  Write-Log $summaryLog ('EURUSD EXIT CODE: ' + $eurusdExit);" ^
  "  Write-Log $summaryLog ('MT5 DEMO AUTO RUN EXIT CODE: ' + $autoExit);" ^
  "  Write-Log $summaryLog ('RUNNER EXIT CODE: ' + $overallExit);" ^
  "  Write-Log $summaryLog '';" ^
  "}" ^
  "exit $overallExit"

exit /b %ERRORLEVEL%
