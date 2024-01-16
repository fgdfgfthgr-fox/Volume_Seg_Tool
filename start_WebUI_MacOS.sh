#!/bin/bash

# Set python executable
python_cmd="python3"

# Set venv directory without trailing slash
venv_dir="venv"

# Set WebUI script
WEBUI_SCRIPT="WebUI.py"

# Get the script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ -z "${install_dir}" ]]; then
    install_dir="$SCRIPT_DIR"
fi

# Check if virtual environment is already activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    printf "Activating python venv...\n"
    cd "${install_dir}" || { printf "\e[1m\e[31mERROR: Can't cd to %s/, aborting...\e[0m" "${install_dir}"; exit 1; }
    if [[ -f "${venv_dir}"/bin/activate ]]; then
        source "${venv_dir}"/bin/activate
    else
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
        exit 1
    fi
else
    printf "python venv already activated: ${VIRTUAL_ENV}"
fi

printf "Starting WebUI script...\n"
"${python_cmd}" -u "${WEBUI_SCRIPT}" "$@"
