@echo off
title STTTS GUI
cd /d "%~dp0"
python gui.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python script crashed with code %errorlevel%
)
pause
