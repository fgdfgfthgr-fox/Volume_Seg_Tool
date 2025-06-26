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

:: GPU detection and setup
set gpu_info=
set cuda_version=
set torch_command_set=0

:: Check for NVIDIA GPU and CUDA version using nvidia-smi
where nvidia-smi >nul 2>nul
if !errorlevel! equ 0 (
    set cuda_version=
    for /f "tokens=*" %%i in ('nvidia-smi ^| findstr /c:"CUDA Version"') do (
        set "line=%%i"
        set "line=!line:*CUDA Version:=!"
        for /f "tokens=1" %%v in ("!line!") do set cuda_version=%%v
    )
)

:: Check GPU info
for /f "tokens=*" %%i in ('"wmic path win32_videocontroller get caption"') do set gpu_info=!gpu_info! %%i

echo !gpu_info! | findstr /i "NVIDIA" >nul
if !errorlevel! equ 0 (
    if defined cuda_version (
        :: Convert version to comparable format (major.minor)
        for /f "tokens=1,2 delims=." %%a in ("!cuda_version!") do (
            set cuda_major=%%a
            set cuda_minor=%%b
        )

        :: Check if CUDA version is >= 11.8
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
            echo.
            echo WARNING: Detected CUDA version !cuda_version! which is not supported by PyTorch 2.7.1
            echo.
        )
    ) else (
        echo.
        echo WARNING: NVIDIA GPU detected but CUDA installation not found: !cuda_version!
        echo.
    )
) else (
    echo !gpu_info! | findstr /i "AMD" >nul
    if !errorlevel! equ 0 (
        echo.
        echo WARNING: AMD GPU detected but Windows ROCm support is not available
        echo.
    )
)

:: Set CPU install if no compatible GPU setup found
if !torch_command_set! equ 0 (
    echo.
    echo Installing CPU-only version of PyTorch
    echo.
    set TORCH_COMMAND=pip install torch torchvision
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