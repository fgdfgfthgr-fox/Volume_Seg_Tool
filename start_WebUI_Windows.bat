@echo off
setlocal enabledelayedexpansion

set python_cmd=python
set venv_dir=venv
set WEBUI_SCRIPT=WebUI.py

:: Get the script directory
for %%i in (%0) do set SCRIPT_DIR=%%~dpi

if not defined install_dir set install_dir=!SCRIPT_DIR!

echo.
echo This script will test whether VST is compatible with your system by running a short training and prediction workflow using example data.
echo If successful, a model named 'example_name.ckpt' will be saved in the 'trained_model' folder. You will then see a message telling you that it succeeded.
echo If it fails for any reason, please open a GitHub issue and tell me what happened!
echo If you have deleted the included example data and replaced it with your own, this script may fail, which is expected.
echo Press any key to continue or Ctrl+C to abort...
pause >nul


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
