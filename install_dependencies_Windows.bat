@echo off
setlocal enabledelayedexpansion

:: Function to pause and wait for user input before exiting
goto :main

:pause_and_exit
echo.
echo Press any key to exit...
pause >nul
exit /b %1

:main
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

:: Initialize GPU detection variables
set cuda_version=
set torch_command_set=0
set gpu_failure_reason=

:: Check for NVIDIA GPU via wmic
set gpu_list=
for /f "skip=1 tokens=*" %%i in ('"wmic path win32_videocontroller get caption"') do (
    set "line=%%i"
    if not "!line!"=="" set gpu_list=!gpu_list! !line!
)

:: Check if any NVIDIA GPU is present
echo !gpu_list! | findstr /i "NVIDIA" >nul
if !errorlevel! equ 0 (
    :: NVIDIA GPU detected – try to get CUDA version
    where nvidia-smi >nul 2>nul
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('nvidia-smi ^| findstr /c:"CUDA Version"') do (
            set "line=%%i"
            set "line=!line:*CUDA Version:=!"
            for /f "tokens=1" %%v in ("!line!") do set cuda_version=%%v
        )
    )

    if defined cuda_version (
        :: Parse major.minor
        for /f "tokens=1,2 delims=." %%a in ("!cuda_version!") do (
            set cuda_major=%%a
            set cuda_minor=%%b
        )

        :: Check supported versions
        if !cuda_major! equ 12 (
            if !cuda_minor! geq 8 (
                set TORCH_COMMAND=pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
                set torch_command_set=1
            ) else if !cuda_minor! geq 6 (
                set TORCH_COMMAND=pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu126
                set torch_command_set=1
            )
        ) else if !cuda_major! equ 11 if !cuda_minor! geq 8 (
            set TORCH_COMMAND=pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu118
            set torch_command_set=1
        )

        if !torch_command_set! equ 0 (
            set gpu_failure_reason=Unsupported CUDA version !cuda_version! (requires >=11.8)
        )
    ) else (
        :: NVIDIA GPU present but nvidia-smi failed or CUDA not found
        set gpu_failure_reason=NVIDIA GPU detected but CUDA not found
    )
) else (
    :: No NVIDIA – check for AMD
    echo !gpu_list! | findstr /i "AMD" >nul
    if !errorlevel! equ 0 (
        set gpu_failure_reason=AMD GPU detected but Windows ROCm is not supported
    ) else (
        set gpu_failure_reason=No compatible GPU detected (NVIDIA required)
    )
)

:: If no compatible GPU setup found, halt with error
if !torch_command_set! equ 0 (
    echo.
    echo ERROR: !gpu_failure_reason!
    echo This tool requires a CUDA-capable NVIDIA GPU with a supported driver.
    echo Installation aborted.
    call :pause_and_exit 1
)

:: Python version check (must be >=3.9)
%python_cmd% -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.9 or higher is required.
    call :pause_and_exit 1
)

:: Check for venv module
%python_cmd% -c "import venv" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python venv module is not available. Please ensure Python venv is installed.
    call :pause_and_exit 1
)

echo.
echo Installing dependencies for Volume Seg Tool...
echo.

:: Create and activate virtual environment if needed
if not defined VIRTUAL_ENV (
    echo Creating and activating python venv...
    cd /d "!install_dir!" || (
        echo ERROR: Can't cd to !install_dir!, aborting...
        call :pause_and_exit 1
    )
    if not exist "!venv_dir!" (
        !python_cmd! -m venv "!venv_dir!"
    )
    call !venv_dir!\Scripts\activate || (
        echo ERROR: Cannot activate python venv, aborting...
        call :pause_and_exit 1
    )
) else (
    echo Python venv already activated: !VIRTUAL_ENV!
)

:: Run the environment script
!python_cmd! -u "!ENV_SCRIPT!" %*

:: If we get here, installation completed successfully
echo.
echo Installation completed successfully!
call :pause_and_exit 0