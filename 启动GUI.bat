@echo off
title STTTS GUI
cd /d "%~dp0"
echo Running STTTS GUI...
"C:\ProgramData\miniconda3\python.exe" gui.py 2>gui_error.log
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python script crashed with code %errorlevel%
    echo Check gui_error.log for details.
    type gui_error.log
) else (
    echo [OK] Exited normally
)
pause
