@echo off
REM ── Self-elevate to admin ──
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs -WorkingDirectory '%~dp0'"
    exit /b
)

title STTTS GUI
cd /d "%~dp0"
echo Running STTTS GUI (as admin)...
set LOGFILE=%TEMP%\sttts_gui_error.log
python gui.py 2>"%LOGFILE%"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python script crashed with code %errorlevel%
    echo Check %LOGFILE% for details.
    type "%LOGFILE%"
) else (
    echo [OK] Exited normally
)
