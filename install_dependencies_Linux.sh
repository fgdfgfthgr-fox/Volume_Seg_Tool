#!/bin/bash

# Function to pause and wait for user input before exiting
pause() {
    echo
    echo "Press any key to exit..."
    read -n 1 -s
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ -z "${install_dir}" ]]
then
    install_dir="$SCRIPT_DIR"
fi

# python3 executable
python_cmd="python3"

# git executable
export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/venv)
venv_dir="venv"

ENV_SCRIPT="prepare_environment.py"

# Disable sentry logging
export ERROR_REPORTING=FALSE

# Do not reinstall existing pip packages on Debian/Ubuntu
export PIP_IGNORE_INSTALLED=0

# GPU detection
torch_command_set=0

# Check for NVIDIA driver and CUDA version
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi --query 2>/dev/null | grep "CUDA Version" | awk '{print $NF}')
    # Fallback to regular nvidia-smi output if --query fails
    if [ -z "$cuda_version" ]; then
        cuda_version=$(nvidia-smi 2>/dev/null | grep -i "cuda version" | awk -F': ' '{print $2}' | awk '{print $1}')
    fi
fi

# Check for ROCm version - try multiple detection methods in priority order
rocm_version=""

# Method 1: amd-smi version (preferred when available)
if command -v amd-smi &> /dev/null; then
    rocm_version=$(amd-smi version 2>/dev/null \
        | sed -nE 's/.*ROCm version:[[:space:]]*([0-9]+(\.[0-9]+)*).*/\1/p' \
        | head -n1)
fi

# Method 2: rocminfo (some ROCm installs don't ship amd-smi)
if [[ -z "$rocm_version" ]] && command -v rocminfo &> /dev/null; then
    rocm_version=$(rocminfo 2>/dev/null \
        | grep -i "ROCm Runtime Version" \
        | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' \
        | head -n1)
fi

# Method 3: version file (respect ROCM_PATH if set)
if [[ -z "$rocm_version" ]]; then
    rocm_path="${ROCM_PATH:-/opt/rocm}"
    if [[ -r "${rocm_path}/.info/version" ]]; then
        rocm_version=$(head -n1 "${rocm_path}/.info/version" 2>/dev/null)
    elif compgen -G "${rocm_path}/.info/version-*" > /dev/null 2>&1; then
        rocm_version=$(cat "${rocm_path}/.info/version-"* 2>/dev/null | head -n1)
    fi
fi

# Normalize to major.minor (strip pre-release suffixes, whitespace, etc.)
if [[ -n "$rocm_version" ]]; then
    rocm_version=$(echo "$rocm_version" | grep -oE '[0-9]+\.[0-9]+' | head -n1)
fi

# Get GPU info - try multiple detection methods in priority order
gpu_info=""

# Method 1: lspci (most descriptive)
if command -v lspci &> /dev/null; then
    gpu_info=$(lspci 2>/dev/null | grep -iE "VGA|3D|Display")
fi

# Method 2: /sys/class/drm with vendor ID -> name mapping
if [[ -z "$gpu_info" && -d /sys/class/drm ]]; then
    for vendor_file in /sys/class/drm/*/device/vendor; do
        [[ -r "$vendor_file" ]] || continue
        vendor_id=$(tr -d '[:space:]' < "$vendor_file" 2>/dev/null)
        case "$vendor_id" in
            0x10de) gpu_info+=" NVIDIA" ;;
            0x1002) gpu_info+=" AMD" ;;
            0x8086) gpu_info+=" Intel" ;;
            0x13b5) gpu_info+=" ARM" ;;
        esac
    done
fi

# Method 3: glxinfo (actual rendering device)
if [[ -z "$gpu_info" ]] && command -v glxinfo &> /dev/null; then
    gpu_info=$(glxinfo -B 2>/dev/null | grep -iE "OpenGL renderer|Device:" | head -n 5)
fi

# Method 4: presence of vendor-specific tools as a last resort
if [[ -z "$gpu_info" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        gpu_info="NVIDIA"
    elif command -v rocm-smi &> /dev/null || command -v amd-smi &> /dev/null; then
        gpu_info="AMD"
    fi
fi

# NVIDIA GPU detection
if [[ -n "$gpu_info" && "$gpu_info" == *"NVIDIA"* ]] || [[ -n "$cuda_version" ]]; then
    if [ -n "$cuda_version" ]; then
        IFS='.' read -ra cuda_parts <<< "$cuda_version"
        cuda_major=${cuda_parts[0]}
        cuda_minor=${cuda_parts[1]}

        if [ "$cuda_major" -eq 13 ] && [ "$cuda_minor" -ge 0 ]; then
            export TORCH_COMMAND="pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130"
            torch_command_set=1
        elif [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 8 ]; then
            export TORCH_COMMAND="pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128"
            torch_command_set=1
        elif [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 6 ]; then
            export TORCH_COMMAND="pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu126"
            torch_command_set=1
        fi

        if [ $torch_command_set -eq 0 ]; then
            printf "\e[1m\e[33mWARNING: Detected CUDA version %s which is not supported by PyTorch 2.11.0\e[0m\n" "$cuda_version"
        fi
    else
        printf "\e[1m\e[33mWARNING: NVIDIA GPU detected but CUDA installation not found.
        Please install CUDA according to https://developer.nvidia.com/cuda/toolkit \e[0m\n"
    fi

# AMD GPU detection
elif [[ -n "$gpu_info" && "$gpu_info" == *"AMD"* ]] || [[ -n "$rocm_version" ]]; then
    if [ -n "$rocm_version" ]; then
        IFS='.' read -ra rocm_parts <<< "$rocm_version"
        rocm_major=${rocm_parts[0]}
        rocm_minor=${rocm_parts[1]}

        # For ROCm, any version >= 6.3 is fine
        if [ "$rocm_major" -gt 6 ] || { [ "$rocm_major" -eq 6 ] && [ "$rocm_minor" -ge 3 ]; }; then
            export TORCH_COMMAND="pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/rocm6.3"
            torch_command_set=1
        fi

        if [ $torch_command_set -eq 0 ]; then
            printf "\e[1m\e[33mWARNING: Detected ROCm version %s which is not supported by PyTorch 2.7.1\e[0m\n" "$rocm_version"
        fi
    else
        printf "\e[1m\e[33mWARNING: AMD GPU detected but ROCm installation not found.
        Please install ROCm according to https://rocm.docs.amd.com/en/latest/install/rocm.html\e[0m\n"
    fi
fi

# If no compatible GPU setup found
if [ $torch_command_set -eq 0 ]; then
    printf "\e[1m\e[31mERROR: No compatible GPU and drivers found.\n"
    printf "This tool requires a GPU with CUDA or ROCm support. Aborting.\e[0m\n"
    pause
    exit 1
fi

# Check for venv module
if ! "${python_cmd}" -c "import venv" &>/dev/null
then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    pause
    exit 1
fi

printf "\e[1m\e[32mInstalling dependencies for Volume Seg Tool...\n"

if [[ -z "${VIRTUAL_ENV}" ]];
then
    printf "Creating and activating python venv\n"
    cd "${install_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/, aborting...\e[0m" "${install_dir}"; pause; exit 1; }
    if [[ ! -d "${venv_dir}" ]]
    then
        "${python_cmd}" -m venv "${venv_dir}"
    fi
    # shellcheck source=/dev/null
    if [[ -f "${venv_dir}"/bin/activate ]]
    then
        source "${venv_dir}"/bin/activate
    else
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
        pause
        exit 1
    fi
else
    printf "python venv already activate: ${VIRTUAL_ENV}"
fi

"${python_cmd}" -u "${ENV_SCRIPT}" "$@"

# Add pause at the end if the script was run successfully
echo
echo "Installation completed successfully!"
pause