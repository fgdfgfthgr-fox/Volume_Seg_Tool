@echo off
setlocal enabledelayedexpansion

:: Set python executable
set python_cmd=python

:: Set venv directory without trailing slash
set venv_dir=venv

:: Set WebUI script
set WEBUI_SCRIPT=WebUI.py

:: Get the script directory
for %%i in (%0) do set SCRIPT_DIR=%%~dpi

if not defined install_dir set install_dir=!SCRIPT_DIR!

:: Check if virtual environment is already activated
if not defined VIRTUAL_ENV (
    echo Activating python venv...
    cd /d "!install_dir!" || (
        echo ERROR: Can't cd to !install_dir!, aborting...
        exit /b 1
    )
    call "!venv_dir!\Scripts\activate.bat" || (
        echo ERROR: Cannot activate python venv, aborting...
        exit /b 1
    )
) else (
    echo Python venv already activated: !VIRTUAL_ENV!
)

:: Start WebUI script
set exit_code=!errorlevel!
echo.
if !exit_code! equ 0 (
    echo Workflow completed successfully (exit code 0).
) else (
    echo Workflow failed with exit code !exit_code!.
)
echo Press any key to exit...
pause >nul
exit /b !exit_code!