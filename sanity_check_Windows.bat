@echo off
setlocal enabledelayedexpansion

:: Set python executable
set python_cmd=python

:: Set venv directory without trailing slash
set venv_dir=venv

set workflow_script="workflow.py"

:: Get the script directory
set SCRIPT_DIR=%~dp0

if not defined install_dir set install_dir=!SCRIPT_DIR!

:: Introductory message (identical to Linux version)
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
    if exist "!venv_dir!\Scripts\activate.bat" (
        call "!venv_dir!\Scripts\activate.bat"
    ) else (
        echo ERROR: Cannot activate python venv, aborting...
        exit /b 1
    )
) else (
    echo Python venv already activated: !VIRTUAL_ENV!
)

:: Start script
echo Starting the coverage test...
!python_cmd! -u !workflow_script! %*
:: set exit_code=%errorlevel%

:: Report the result
:: echo.
:: if !exit_code! equ 0 (
::    echo Workflow completed successfully (exit code 0).
::) else (
::    echo Workflow failed with exit code !exit_code!.
::)

echo Press any key to exit...
pause >nul
exit /b 0