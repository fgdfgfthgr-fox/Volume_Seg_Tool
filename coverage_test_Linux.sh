#!/bin/bash

python_cmd="python3"
venv_dir="venv"
workflow_script="workflow.py"

# Get the script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ -z "${install_dir}" ]]; then
    install_dir="$SCRIPT_DIR"
fi

# Introductory message
echo ""
echo "This script will test whether VST is compatible with your system by running a short training and prediction workflow using example data."
echo "If successful, a model named 'example_name.ckpt' will be saved in the 'trained_model' folder. You will then see a message telling you that it succeeded."
echo "If it fails for any reason, please open a GitHub issue and tell me what happened!"
echo "If you have deleted the included example data and replaced it with your own, this script may fail, which is expected."
read -n 1 -s -r -p "Press any key to continue or Ctrl+C to abort..." 
echo   # Move to a new line after key press

# Activate virtual environment if not already activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Activating python venv..."
    cd "${install_dir}" || { echo -e "\e[1m\e[31mERROR: Can't cd to ${install_dir}, aborting...\e[0m"; exit 1; }
    if [[ -f "${venv_dir}/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "${venv_dir}/bin/activate"
    else
        echo -e "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
        exit 1
    fi
else
    echo "Python venv already activated: ${VIRTUAL_ENV}"
fi

# Run the workflow script
echo "Starting the coverage test..."
"${python_cmd}" -u "${workflow_script}" "$@"
exit_code=$?

# Report the result
echo ""
if [[ ${exit_code} -eq 0 ]]; then
    echo "Workflow completed successfully (exit code 0)."
else
    echo "Workflow failed with exit code ${exit_code}."
fi

read -n 1 -s -r -p "Press any key to exit..."
echo   # Final newline for clean exit
exit ${exit_code}