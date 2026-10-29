@echo off
REM ============================================================
REM Smart Attendance Predictor - Windows One-Click Launcher
REM Double-click this file to launch the entire system.
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo   Smart Attendance Prediction System (Windows Launcher)
echo ============================================================

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python was not found in your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv" (
    echo First-time setup: Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing required packages (one-time setup, please wait)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

python run.py
pause

