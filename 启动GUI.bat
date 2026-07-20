@echo off
title STTTS GUI
cd /d "%~dp0"
echo Running STTTS GUI...
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
