@echo off
setlocal enabledelayedexpansion

set python_cmd=python
set venv_dir=venv
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

echo Starting WebUI script...
!python_cmd! -u "!WEBUI_SCRIPT!" %*

:end
