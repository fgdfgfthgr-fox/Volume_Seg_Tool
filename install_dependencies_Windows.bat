@echo off
setlocal enabledelayedexpansion

:: Get the script directory
for %%i in ("%~dp0") do set "SCRIPT_DIR=%%~fi"

if not defined install_dir set install_dir=!SCRIPT_DIR!

:: Set python executable
set python_cmd=python

:: Set git executable
set GIT=git

:: Set python venv directory without trailing slash
set venv_dir=venv

:: Set environment script
set ENV_SCRIPT=prepare_environment.py

:: Disable sentry logging
set ERROR_REPORTING=FALSE

:: Do not reinstall existing pip packages on Windows
set PIP_IGNORE_INSTALLED=0

:: Check GPU info, if neither AMD or NVIDIA gpu found, install CPU only version of PyTorch
set gpu_info=
for /f "tokens=*" %%i in ('"wmic path win32_videocontroller get caption"') do set gpu_info=!gpu_info! %%i

echo !gpu_info! | findstr /i "NVIDIA" >nul
if %ERRORLEVEL% neq 0 (
    echo !gpu_info! | findstr /i "AMD" >nul
    if %ERRORLEVEL% equ 0 (
	echo.
        echo Warning: AMD GPU on Windows is not yet supported by PyTorch and will resort to CPU-only installation.
	echo.
        if not defined TORCH_COMMAND set TORCH_COMMAND=pip install torch torchvision
    ) else (
	echo.
        rem Neither NVIDIA nor AMD GPUs are found, proceed with CPU-only installation
	echo.
        if not defined TORCH_COMMAND set TORCH_COMMAND=pip install torch torchvision
    )
)


echo.
echo Installing dependencies for Volume Seg Tool...
echo.

:: Check if virtual environment is already activated
if not defined VIRTUAL_ENV (
    echo Creating and activating python venv...
    cd /d "!install_dir!" || (
        echo ERROR: Can't cd to !install_dir!, aborting...
        exit /b 1
    )
    if not exist "!venv_dir!" (
        !python_cmd! -m venv "!venv_dir!"
    )
    call !venv_dir!\Scripts\activate || (
        echo ERROR: Cannot activate python venv, aborting...
        exit /b 1
    )
) else (
    echo Python venv already activated: !VIRTUAL_ENV!
)

:: Run the environment script
!python_cmd! -u "!ENV_SCRIPT!" %*

:end