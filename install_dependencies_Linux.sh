
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

gpu_info=$(lspci 2>/dev/null | grep -E "VGA|Display")
if ! echo "$gpu_info" | grep -q "NVIDIA";
then
    if echo "$gpu_info" | grep -q "AMD" && [[ -z "${TORCH_COMMAND}" ]]
    then
        export TORCH_COMMAND="pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6"
    fi
fi

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