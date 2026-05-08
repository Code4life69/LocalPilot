@echo off
setlocal
cd /d "%~dp0"

set "BOOTSTRAP_PYTHON="
set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "BOOTSTRAP_PYTHON=py"
    ) else (
        where python >nul 2>nul
        if %errorlevel%==0 (
            set "BOOTSTRAP_PYTHON=python"
        )
    )
)

if not exist "%PYTHON_EXE%" if not defined BOOTSTRAP_PYTHON (
    echo Could not find Python.
    echo Install Python or create the virtual environment first.
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo Creating LocalPilot virtual environment...
    %BOOTSTRAP_PYTHON% -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv
        pause
        exit /b 1
    )
)

echo Checking required packages...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install required packages.
    pause
    exit /b 1
)

echo Starting LocalPilot...
"%PYTHON_EXE%" localpilot.py
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo LocalPilot exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
