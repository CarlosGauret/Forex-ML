@echo off
chcp 65001 >nul
setlocal

rem Carpeta del proyecto = carpeta donde esta este .bat
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
cd /d "%PROJECT_DIR%"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

set "BACKUP_DIR=%PROJECT_DIR%\backups\%STAMP%"
mkdir "%BACKUP_DIR%" >nul 2>nul

copy /Y "%PROJECT_DIR%\paper\start.json" "%BACKUP_DIR%\start.json" >nul
copy /Y "%PROJECT_DIR%\paper\state.json" "%BACKUP_DIR%\state.json" >nul
copy /Y "%PROJECT_DIR%\paper\trades.csv" "%BACKUP_DIR%\trades.csv" >nul
copy /Y "%PROJECT_DIR%\paper\signals.csv" "%BACKUP_DIR%\signals.csv" >nul
copy /Y "%PROJECT_DIR%\paper\equity.csv" "%BACKUP_DIR%\equity.csv" >nul

echo Backup PAPER creado:
echo %BACKUP_DIR%
echo.
echo No se copio .env.

exit /b 0
