#!/bin/bash

# Get the script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ -z "${install_dir}" ]]; then
    install_dir="$SCRIPT_DIR"
fi

# Python executable
python_cmd="python3"

# Git executable
export GIT="git"

# Python venv without trailing slash (defaults to ${install_dir}/${venv_dir})
venv_dir="venv"

ENV_SCRIPT="prepare_environment.py"

# Disable sentry logging
export ERROR_REPORTING=FALSE

# Do not reinstall existing pip packages on macOS
export PIP_IGNORE_INSTALLED=0

# Use the torch version that work for Mac
export TORCH_COMMAND="pip install torch torchvision"

# Check if python3-venv is installed
if ! "${python_cmd}" -c "import venv" &>/dev/null; then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

printf "\e[1m\e[32mInstalling dependencies for Volume Seg Tool...\n"

# Check if virtual environment is already activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    printf "Creating and activating python venv"
    cd "${install_dir}" || { printf "\e[1m\e[31mERROR: Can't cd to %s/, aborting...\e[0m" "${install_dir}"; exit 1; }
    if [[ ! -d "${venv_dir}" ]]; then
        "${python_cmd}" -m venv "${venv_dir}"
    fi
    source "${venv_dir}"/bin/activate || { printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"; exit 1; }
else
    printf "python venv already activated: ${VIRTUAL_ENV}"
fi

"${python_cmd}" -u "${ENV_SCRIPT}" "$@"
