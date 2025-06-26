SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ -z "${install_dir}" ]]
then
    install_dir="$SCRIPT_DIR"
fi

# python3 executable
python_cmd="python3"

# git executable
export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
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

# Check for ROCm version
rocm_version=""
if [ -f /opt/rocm/.info/version ]; then
    rocm_version=$(cat /opt/rocm/.info/version | cut -d. -f1-2)
elif [ -f /opt/rocm/.info/version-* ]; then
    rocm_version=$(cat /opt/rocm/.info/version-* | cut -d. -f1-2)
fi

# Get GPU info
gpu_info=""
if command -v lspci &> /dev/null; then
    gpu_info=$(lspci 2>/dev/null | grep -E "VGA|Display")
else
    # Fallback to other GPU detection methods if lspci is missing
    if [ -d /sys/class/drm ]; then
        gpu_info=$(grep -i "vendor" /sys/class/drm/*/device/vendor 2>/dev/null | cut -d':' -f2 | uniq)
    fi
fi

# NVIDIA GPU detection
if [[ -n "$gpu_info" && "$gpu_info" == *"NVIDIA"* ]] || [[ -n "$cuda_version" ]]; then
    if [ -n "$cuda_version" ]; then
        IFS='.' read -ra cuda_parts <<< "$cuda_version"
        cuda_major=${cuda_parts[0]}
        cuda_minor=${cuda_parts[1]}

        if [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 8 ]; then
            export TORCH_COMMAND="pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128"
            torch_command_set=1
        elif [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 6 ]; then
            export TORCH_COMMAND="pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu126"
            torch_command_set=1
        elif [ "$cuda_major" -eq 11 ] && [ "$cuda_minor" -ge 8 ]; then
            export TORCH_COMMAND="pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu118"
            torch_command_set=1
        fi

        if [ $torch_command_set -eq 0 ]; then
            printf "\e[1m\e[33mWARNING: Detected CUDA version %s which is not supported by PyTorch 2.7.1\e[0m\n" "$cuda_version"
        fi
    else
        printf "\e[1m\e[33mWARNING: NVIDIA GPU detected but CUDA installation not found\e[0m\n"
    fi
# AMD GPU detection
elif [[ -n "$gpu_info" && "$gpu_info" == *"AMD"* ]] || [[ -n "$rocm_version" ]]; then
    if [ -n "$rocm_version" ]; then
        IFS='.' read -ra rocm_parts <<< "$rocm_version"
        rocm_major=${rocm_parts[0]}
        rocm_minor=${rocm_parts[1]}

        # For ROCm, any version >= 6.3 is fine
        if [ "$rocm_major" -eq 6 ] && [ "$rocm_minor" -ge 3 ]; then
            export TORCH_COMMAND="pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/rocm6.3"
            torch_command_set=1
        fi

        if [ $torch_command_set -eq 0 ]; then
            printf "\e[1m\e[33mWARNING: Detected ROCm version %s which is not supported by PyTorch 2.7.1\e[0m\n" "$rocm_version"
        fi
    else
        printf "\e[1m\e[33mWARNING: AMD GPU detected but ROCm installation not found\e[0m\n"
    fi
fi

# Set CPU install if no compatible GPU setup found
if [ $torch_command_set -eq 0 ]; then
    printf "\e[1m\e[33mInstalling CPU-only version of PyTorch\e[0m\n"
    export TORCH_COMMAND="pip install torch torchvision"
fi

# Check for venv module
if ! "${python_cmd}" -c "import venv" &>/dev/null
then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

printf "\e[1m\e[32mInstalling dependencies for Volume Seg Tool...\n"

if [[ -z "${VIRTUAL_ENV}" ]];
then
    printf "Creating and activating python venv\n"
    cd "${install_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/, aborting...\e[0m" "${install_dir}"; exit 1; }
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
        exit 1
    fi
else
    printf "python venv already activate: ${VIRTUAL_ENV}"
fi

"${python_cmd}" -u "${ENV_SCRIPT}" "$@"